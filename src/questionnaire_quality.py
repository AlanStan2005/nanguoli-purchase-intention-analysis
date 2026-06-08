from __future__ import annotations

import numpy as np
import pandas as pd


def correlation_matrix(items: pd.DataFrame) -> pd.DataFrame:
    return items.astype(float).corr()


def kmo_measure(items: pd.DataFrame) -> float:
    corr = correlation_matrix(items).to_numpy()
    inv_corr = np.linalg.pinv(corr)
    partial = -inv_corr / np.sqrt(np.outer(np.diag(inv_corr), np.diag(inv_corr)))
    np.fill_diagonal(partial, 0)
    corr_sq = corr**2
    partial_sq = partial**2
    np.fill_diagonal(corr_sq, 0)
    return float(corr_sq.sum() / (corr_sq.sum() + partial_sq.sum()))


def bartlett_statistic(items: pd.DataFrame) -> dict:
    corr = correlation_matrix(items).to_numpy()
    n, p = items.shape
    det_corr = max(np.linalg.det(corr), 1e-12)
    chi_square = -(n - 1 - (2 * p + 5) / 6) * np.log(det_corr)
    dof = p * (p - 1) / 2
    return {"chi_square": float(chi_square), "dof": float(dof)}


def scale_score(items: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.Series:
    if weights is None:
        return items.astype(float).mean(axis=1)
    aligned = np.array([weights.get(c, 0.0) for c in items.columns])
    aligned = aligned / aligned.sum()
    return pd.Series(items.to_numpy(dtype=float).dot(aligned), index=items.index)
