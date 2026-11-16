"""Model training: XGBoost classifier (who churns) + Cox PH (when)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from lifelines import CoxPHFitter
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

XGB_PARAMS = dict(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_lambda=2.0,
    eval_metric="auc",
    random_state=7,
    n_jobs=1,
)


def train_xgb(X: pd.DataFrame, y: pd.Series, seed: int = 7):
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
    # Churn is rare, so classes are re-weighted. Scores are then risk scores for
    # ranking and tiering, not calibrated probabilities.
    spw = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    model = xgb.XGBClassifier(**XGB_PARAMS, scale_pos_weight=spw, early_stopping_rounds=40)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    auc_val = roc_auc_score(y_va, model.predict_proba(X_va)[:, 1])
    return model, auc_val


def train_cox(X: pd.DataFrame, duration: pd.Series, event: pd.Series, penalizer: float = 0.05):
    """Fit a Cox PH model on standardised features (hazard ratios are per 1 SD).

    Returns the fitted model and a `prepare` function that applies the same
    scaling to any other snapshot before scoring.
    """
    mu, sd = X.mean(), X.std().replace(0, 1)

    def prepare(Xn: pd.DataFrame, dur: pd.Series, ev: pd.Series) -> pd.DataFrame:
        return ((Xn[X.columns] - mu) / sd).assign(
            surv_duration_weeks=dur.to_numpy(), surv_event=ev.to_numpy()
        )

    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(prepare(X, duration, event), duration_col="surv_duration_weeks", event_col="surv_event")
    return cph, prepare


def gini(auc: float) -> float:
    return 2 * auc - 1


def assign_tier(p: np.ndarray, high: float = 0.55, medium: float = 0.35) -> np.ndarray:
    return np.where(p >= high, "HIGH", np.where(p >= medium, "MEDIUM", "LOW"))


def tier_report(y: pd.Series, p: np.ndarray, high: float = 0.55, medium: float = 0.35) -> pd.DataFrame:
    tiers = assign_tier(p, high, medium)
    df = pd.DataFrame({"tier": tiers, "y": y.to_numpy()})
    base = df.y.mean()
    rep = df.groupby("tier").agg(clients=("y", "size"), churners=("y", "sum"), churn_rate=("y", "mean"))
    rep["share_of_churners"] = rep.churners / df.y.sum()
    rep["lift_vs_base"] = rep.churn_rate / base
    return rep.reindex(["HIGH", "MEDIUM", "LOW"]).round(3)
