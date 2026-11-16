"""
Synthetic data generator for the corporate churn early-warning project.

Produces two tables that mirror the *shape* of a corporate bulk-payments
platform, without containing any real client data:

    data/raw/clients.parquet           one row per corporate client
    data/raw/payments_weekly.parquet   weekly outbound / inbound payment activity

Behaviour that is deliberately simulated so the modelling problem is realistic:
  * churn time is drawn from a proportional-hazards process driven by
    segment, tenure, product holding, RM coverage and payment-failure propensity
  * churners *fade* (activity ramps down over 4-16 weeks) before going silent
  * some healthy clients have temporary dips (seasonal / project-driven)
  * some clients stop paying out but keep receiving funds -> NOT churn under the
    "authentic churn" definition (both directions must stop)

Usage:
    python data/generate_synthetic.py --clients 3000 --weeks 78 --seed 42
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SEGMENTS = ["LARGE_CORPORATE", "COMMERCIAL", "SME", "PUBLIC_SECTOR"]
INDUSTRIES = ["MANUFACTURING", "OIL_GAS", "TRADE", "SERVICES", "AGRIC", "TELECOMS", "EDUCATION"]


def generate(n_clients: int = 3000, n_weeks: int = 78, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    ids = np.array([f"C{100000 + i}" for i in range(n_clients)])

    segment = rng.choice(SEGMENTS, n_clients, p=[0.15, 0.30, 0.45, 0.10])
    industry = rng.choice(INDUSTRIES, n_clients)
    tenure_months = rng.gamma(2.2, 22, n_clients).clip(3, 240).round().astype(int)
    products_held = rng.poisson(np.where(segment == "LARGE_CORPORATE", 4.0, 2.2)).clip(1, 9)
    has_rm = (rng.random(n_clients) < np.where(segment == "SME", 0.35, 0.85)).astype(int)
    fail_propensity = rng.beta(1.2, 30, n_clients)  # share of payments that fail

    # ---------------- churn hazard (weekly) ----------------
    seg_effect = pd.Series({"LARGE_CORPORATE": -0.6, "COMMERCIAL": 0.0, "SME": 0.45, "PUBLIC_SECTOR": -0.3})
    lin = (
        seg_effect[segment].to_numpy()
        - 0.35 * np.log1p(tenure_months / 12)
        - 0.18 * (products_held - 2)
        - 0.50 * has_rm
        + 9.0 * fail_propensity
        + rng.normal(0, 0.35, n_clients)          # unobserved heterogeneity
    )
    weekly_hazard = 0.012 * np.exp(lin)
    churn_week = rng.geometric(np.clip(weekly_hazard, 1e-4, 0.5)) + 6  # nobody churns in first 6 weeks
    churn_week = np.where(churn_week >= n_weeks, 10_000, churn_week)  # 10_000 = never within horizon
    # ~45% of churners leave abruptly (no visible fade); the rest fade gently
    # over 6-26 weeks, only dropping to 30-70% of normal before going silent
    abrupt = rng.random(n_clients) < 0.45
    ramp = np.where(abrupt, 1, rng.integers(6, 27, n_clients))
    fade_floor = np.where(abrupt, 1.0, rng.uniform(0.3, 0.7, n_clients))

    # Clients that stop paying out but keep receiving (not authentic churn)
    outbound_only_stop = (rng.random(n_clients) < 0.05) & (churn_week == 10_000)
    outbound_stop_week = np.where(outbound_only_stop, rng.integers(20, n_weeks, n_clients), 10_000)

    # Temporary dips for healthy clients
    has_dip = rng.random(n_clients) < 0.30
    dip_start = rng.integers(10, n_weeks - 8, n_clients)
    dip_len = rng.integers(3, 12, n_clients)

    # Baseline activity
    size_mult = pd.Series({"LARGE_CORPORATE": 6.0, "COMMERCIAL": 2.5, "SME": 1.0, "PUBLIC_SECTOR": 3.0})[segment].to_numpy()
    lam_out = rng.gamma(2.0, 6.0, n_clients) * size_mult
    lam_in = rng.gamma(2.0, 4.0, n_clients) * size_mult
    ticket_out = rng.lognormal(12.5, 0.8, n_clients) * size_mult
    ticket_in = rng.lognormal(12.8, 0.8, n_clients) * size_mult

    weeks = np.arange(n_weeks)
    W = weeks[None, :]
    season = 1 + 0.15 * np.sin(2 * np.pi * W / 52) + np.where(W % 52 >= 49, 0.25, 0)  # year-end spike

    # activity multiplier: fade before churn, zero after
    start_fade = (churn_week - ramp)[:, None]
    fade = np.clip((churn_week[:, None] - W) / ramp[:, None], 0, 1)
    mult = np.where(W < start_fade, 1.0, fade_floor[:, None] + (1 - fade_floor[:, None]) * fade)
    mult = np.where(W >= churn_week[:, None], 0.0, mult)
    dip = has_dip[:, None] & (W >= dip_start[:, None]) & (W < (dip_start + dip_len)[:, None])
    mult = np.where(dip, mult * 0.3, mult)

    mult_out = np.where(W >= outbound_stop_week[:, None], 0.0, mult)
    mult_in = mult

    out_cnt = rng.poisson(lam_out[:, None] * season * mult_out)
    in_cnt = rng.poisson(lam_in[:, None] * season * mult_in)
    out_amt = out_cnt * ticket_out[:, None] * rng.lognormal(0, 0.45, out_cnt.shape)
    in_amt = in_cnt * ticket_in[:, None] * rng.lognormal(0, 0.25, in_cnt.shape)
    failed = rng.binomial(out_cnt, np.clip(fail_propensity[:, None] * (1 + 0.5 * (1 - mult_out)), 0, 1))  # failures rise a little while fading

    week_start = pd.Timestamp("2025-01-06") + pd.to_timedelta(weeks * 7, unit="D")

    def long(direction, cnt, amt, fail=None):
        df = pd.DataFrame({
            "client_id": np.repeat(ids, n_weeks),
            "week_idx": np.tile(weeks, n_clients),
            "week_start": np.tile(week_start, n_clients),
            "direction": direction,
            "txn_count": cnt.ravel(),
            "txn_amount": amt.ravel().round(2),
            "failed_count": (fail.ravel() if fail is not None else 0),
        })
        return df[df.txn_count > 0]

    payments = pd.concat([long("OUT", out_cnt, out_amt, failed), long("IN", in_cnt, in_amt)], ignore_index=True)
    payments = payments.sort_values(["client_id", "week_idx", "direction"]).reset_index(drop=True)

    clients = pd.DataFrame({
        "client_id": ids,
        "segment": segment,
        "industry": industry,
        "tenure_months_at_start": tenure_months,
        "products_held": products_held,
        "has_rm": has_rm,
    })
    return clients, payments


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", type=int, default=3000)
    ap.add_argument("--weeks", type=int, default=78)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "raw")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    clients, payments = generate(a.clients, a.weeks, a.seed)
    clients.to_parquet(a.out / "clients.parquet", index=False)
    payments.to_parquet(a.out / "payments_weekly.parquet", index=False)
    print(f"clients: {len(clients):,} rows  |  payments_weekly: {len(payments):,} rows  ->  {a.out}")


if __name__ == "__main__":
    main()
