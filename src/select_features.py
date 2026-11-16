"""Iterative VIF reduction: drop the most collinear feature until all VIF < threshold."""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


def vif_table(X: pd.DataFrame) -> pd.DataFrame:
    Xc = X.assign(_const=1.0).to_numpy()
    vifs = [variance_inflation_factor(Xc, i) for i in range(X.shape[1])]
    return pd.DataFrame({"feature": X.columns, "vif": vifs}).sort_values("vif", ascending=False)


def vif_reduce(X: pd.DataFrame, threshold: float = 5.0, keep: tuple[str, ...] = ()) -> tuple[list[str], list[tuple[str, float]]]:
    cols = list(X.columns)
    dropped: list[tuple[str, float]] = []
    while True:
        t = vif_table(X[cols])
        t = t[~t.feature.isin(keep)]
        worst = t.iloc[0]
        if worst.vif < threshold or not np.isfinite(worst.vif) and len(cols) <= 2:
            break
        cols.remove(worst.feature)
        dropped.append((worst.feature, float(worst.vif)))
    return cols, dropped
