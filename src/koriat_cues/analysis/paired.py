"""Paired primed−baseline deltas for headline and pre-registration checks."""
from __future__ import annotations

from math import erfc, sqrt

import numpy as np
import pandas as pd


def paired_delta_vs_baseline(
    df: pd.DataFrame,
    cond: str,
    col: str,
    *,
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[float, float, float, float, int]:
    """Return (mean Δ, low-CI, high-CI, p, n) for paired primed−baseline on one column.

    `col == "correct"` uses the 0/1 accuracy column.
    """
    a = df[df.condition == "baseline"].drop_duplicates("item_id").set_index("item_id")
    b = df[df.condition == cond].drop_duplicates("item_id").set_index("item_id")
    m = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
    if col == "correct":
        d = m["correct_b"].astype(int) - m["correct_a"].astype(int)
    else:
        if f"{col}_b" not in m.columns or f"{col}_a" not in m.columns:
            return float("nan"), float("nan"), float("nan"), float("nan"), 0
        d = m[f"{col}_b"] - m[f"{col}_a"]
    d = d.dropna().to_numpy()
    n = len(d)
    if n < 2:
        return float("nan"), float("nan"), float("nan"), float("nan"), n
    rng = np.random.default_rng(seed)
    boots = np.array([d[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    lo, hi = np.quantile(boots, [0.025, 0.975])
    se = float(d.std(ddof=1) / np.sqrt(n))
    t = float(d.mean()) / se if se > 0 else float("nan")
    try:
        from scipy import stats as _st

        p = float(_st.t.sf(abs(t), n - 1) * 2) if np.isfinite(t) else float("nan")
    except ImportError:
        p = erfc(abs(t) / sqrt(2)) if np.isfinite(t) else float("nan")
    return float(d.mean()), float(lo), float(hi), p, n
