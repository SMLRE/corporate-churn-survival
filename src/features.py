"""Run the SQL feature pipeline in DuckDB for a given snapshot week."""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SQL = ROOT / "sql"
RAW = ROOT / "data" / "raw"


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"CREATE VIEW clients AS SELECT * FROM read_parquet('{RAW / 'clients.parquet'}')")
    con.execute(f"CREATE VIEW payments_weekly AS SELECT * FROM read_parquet('{RAW / 'payments_weekly.parquet'}')")
    return con


def _sql(name: str, **params) -> str:
    return (SQL / name).read_text().format(**params)


def build_model_base(con: duckdb.DuckDBPyConnection, snapshot: int, obs: int = 13, outcome: int = 13) -> pd.DataFrame:
    max_week = con.execute("SELECT MAX(week_idx) FROM payments_weekly").fetchone()[0]
    if snapshot + outcome - 1 > max_week:
        raise ValueError(f"snapshot {snapshot} + outcome window exceeds data (max week {max_week})")
    p = dict(snapshot=snapshot, obs=obs, outcome=outcome, max_week=max_week)
    con.execute("CREATE OR REPLACE TEMP TABLE labels AS " + _sql("01_population_and_label.sql", **p))
    con.execute("CREATE OR REPLACE TEMP TABLE rfm    AS " + _sql("02_rfm_features.sql", **p))
    con.execute("CREATE OR REPLACE TEMP TABLE trend  AS " + _sql("03_trend_features.sql", **p))
    return con.execute(_sql("04_model_base.sql", **p)).df()


CATEGORICAL = ["segment"]
ID_AND_TARGETS = ["client_id", "snapshot_week", "churn_flag", "surv_event", "surv_duration_weeks", "industry"]


def to_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categoricals and drop ids / targets."""
    X = df.drop(columns=ID_AND_TARGETS)
    X = pd.get_dummies(X, columns=CATEGORICAL, prefix="seg", drop_first=True, dtype=float)
    return X.astype(float)
