"""
End-to-end pipeline:

    1. Generate synthetic data (if missing)
    2. Build SQL features for a training snapshot and an out-of-time snapshot
    3. VIF feature reduction
    4. XGBoost classifier (who will churn) + Cox PH survival model (when)
    5. SHAP explanations + two-tier risk segmentation
    6. Write metrics, figures and scored output to reports/

Usage:
    python run_pipeline.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lifelines import KaplanMeierFitter
from sklearn.metrics import roc_auc_score, roc_curve

from src.features import build_model_base, connect, to_matrix
from src.models import assign_tier, gini, tier_report, train_cox, train_xgb
from src.select_features import vif_reduce

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
FIG = REPORTS / "figures"
TRAIN_SNAPSHOT, OOT_SNAPSHOT = 52, 65
HIGH, MEDIUM = 0.55, 0.35

plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})


def main() -> dict:
    FIG.mkdir(parents=True, exist_ok=True)
    if not (ROOT / "data/raw/payments_weekly.parquet").exists():
        subprocess.run([sys.executable, str(ROOT / "data/generate_synthetic.py")], check=True)

    con = connect()
    train = build_model_base(con, TRAIN_SNAPSHOT)
    oot = build_model_base(con, OOT_SNAPSHOT)
    print(f"train snapshot: {len(train):,} clients, churn rate {train.churn_flag.mean():.1%}")
    print(f"OOT snapshot:   {len(oot):,} clients, churn rate {oot.churn_flag.mean():.1%}")

    X_all = to_matrix(train)
    features, dropped = vif_reduce(X_all, threshold=5.0)
    print(f"VIF reduction: {X_all.shape[1]} -> {len(features)} features; dropped {[d[0] for d in dropped]}")

    model, auc_val = train_xgb(X_all[features], train.churn_flag)
    X_oot = to_matrix(oot).reindex(columns=features, fill_value=0.0)
    p_oot = model.predict_proba(X_oot)[:, 1]
    auc_oot = roc_auc_score(oot.churn_flag, p_oot)
    print(f"ROC-AUC validation {auc_val:.3f} | out-of-time {auc_oot:.3f}")

    # ---- ROC
    fig, ax = plt.subplots(figsize=(5.5, 5))
    fpr, tpr, _ = roc_curve(oot.churn_flag, p_oot)
    ax.plot(fpr, tpr, color="#E8A54B", lw=2, label=f"Out-of-time (AUC {auc_oot:.3f})")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.legend(loc="lower right")
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1)
    ax.set_title("XGBoost churn model: out-of-time ROC")
    fig.tight_layout(); fig.savefig(FIG / "roc_oot.png"); plt.close(fig)

    # ---- SHAP
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_oot)
    shap.summary_plot(sv, X_oot, show=False, max_display=12)
    plt.title("What drives churn risk (SHAP, OOT snapshot)")
    plt.tight_layout(); plt.savefig(FIG / "shap_summary.png", bbox_inches="tight"); plt.close()
    top_reasons = (
        pd.DataFrame(np.abs(sv), columns=features).mean().sort_values(ascending=False).head(5).round(4).to_dict()
    )

    # ---- per-client reason codes (top 3 positive SHAP contributors)
    contrib = pd.DataFrame(sv, columns=features)
    reasons = contrib.apply(lambda r: ", ".join(r[r > 0].sort_values(ascending=False).head(3).index), axis=1)

    # ---- tiers
    tiers = tier_report(oot.churn_flag, p_oot, HIGH, MEDIUM)
    print(tiers)

    # ---- Cox PH
    cph, prepare = train_cox(X_all[features], train.surv_duration_weeks, train.surv_event)
    c_index_oot = cph.score(
        prepare(X_oot, oot.surv_duration_weeks, oot.surv_event), scoring_method="concordance_index"
    )
    hr = cph.summary[["exp(coef)", "p"]].rename(columns={"exp(coef)": "hazard_ratio_per_sd"}).sort_values(
        "hazard_ratio_per_sd", ascending=False
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    cph.plot(ax=ax)
    ax.set_title("Cox PH: log hazard ratio per 1 SD")
    fig.tight_layout(); fig.savefig(FIG / "cox_hazard_ratios.png"); plt.close(fig)

    # ---- Kaplan-Meier by risk tier: tiers should separate *time to churn* too
    oot_tier = assign_tier(p_oot, HIGH, MEDIUM)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = {"HIGH": "#D9534F", "MEDIUM": "#E8A54B", "LOW": "#5B8DEF"}
    for t in ["HIGH", "MEDIUM", "LOW"]:
        m = oot_tier == t
        if m.sum() == 0:
            continue
        KaplanMeierFitter().fit(oot.surv_duration_weeks[m], oot.surv_event[m], label=f"{t} (n={m.sum()})").plot_survival_function(
            ax=ax, color=colors[t]
        )
    ax.set_xlabel("weeks after snapshot"); ax.set_ylabel("share of clients still active")
    ax.set_title("Retention curves by risk tier (OOT)")
    fig.tight_layout(); fig.savefig(FIG / "km_by_tier.png"); plt.close(fig)

    # ---- outputs
    scored = oot[["client_id", "segment"]].assign(risk_score=p_oot.round(4), risk_tier=oot_tier, top_reasons=reasons.values)
    scored.sort_values("risk_score", ascending=False).to_csv(REPORTS / "scored_clients_oot.csv", index=False)

    metrics = {
        "train_snapshot_week": TRAIN_SNAPSHOT,
        "oot_snapshot_week": OOT_SNAPSHOT,
        "train_clients": int(len(train)),
        "oot_clients": int(len(oot)),
        "train_churn_rate": round(float(train.churn_flag.mean()), 4),
        "oot_churn_rate": round(float(oot.churn_flag.mean()), 4),
        "candidate_features": int(X_all.shape[1]),
        "final_features": features,
        "n_final_features": len(features),
        "vif_dropped": [d[0] for d in dropped],
        "roc_auc_validation": round(float(auc_val), 4),
        "roc_auc_oot": round(float(auc_oot), 4),
        "gini_oot": round(float(gini(auc_oot)), 4),
        "cox_c_index_oot": round(float(c_index_oot), 4),
        "tier_thresholds": {"HIGH": HIGH, "MEDIUM": MEDIUM},
        "tier_report_oot": tiers.reset_index().to_dict(orient="records"),
        "top_shap_drivers": top_reasons,
    }
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    hr.round(3).to_csv(REPORTS / "cox_hazard_ratios.csv")
    print(f"Cox C-index (OOT): {c_index_oot:.3f}")
    print(f"Outputs written to {REPORTS}")
    return metrics


if __name__ == "__main__":
    main()
