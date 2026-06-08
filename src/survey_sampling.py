import numpy as np
import pandas as pd


def pps_sample(frame: pd.DataFrame, size_col: str, n: int, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    prob = frame[size_col] / frame[size_col].sum()
    idx = rng.choice(frame.index, size=n, replace=False, p=prob)
    return frame.loc[idx].copy()


def cronbach_alpha(items: pd.DataFrame) -> float:
    item_scores = items.to_numpy(dtype=float)
    item_vars = item_scores.var(axis=0, ddof=1)
    total_var = item_scores.sum(axis=1).var(ddof=1)
    k = item_scores.shape[1]
    return k / (k - 1) * (1 - item_vars.sum() / total_var)
