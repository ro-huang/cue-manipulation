"""Compute per-item confidence and accuracy shifts relative to baseline."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def compute_shifts(
    df: pd.DataFrame,
    measure_cols: Iterable[str],
    baseline_cond: str = "baseline",
) -> pd.DataFrame:
    """For each (model, item, condition != baseline), compute confidence and accuracy shifts.

    accuracy_shift: 1 if the item is correct in this condition but wrong at baseline;
                    -1 if correct at baseline but wrong here; 0 otherwise.
    confidence_shift_<m>: condition_confidence_<m> - baseline_confidence_<m>.

    Returns a long-format DataFrame with one row per (model, item, condition != baseline).
    """
    measure_cols = list(measure_cols)
    base = df[df["condition"] == baseline_cond].set_index(["model_name", "item_id"])
    other = df[df["condition"] != baseline_cond]

    rows: list[dict] = []
    for _, r in other.iterrows():
        key = (r["model_name"], r["item_id"])
        if key not in base.index:
            continue
        b = base.loc[key]
        if isinstance(b, pd.DataFrame):
            b = b.iloc[0]
        corr_b = int(bool(b["correct"]))
        corr_c = int(bool(r["correct"]))
        acc_shift = 0
        if corr_c == 1 and corr_b == 0:
            acc_shift = 1
        elif corr_c == 0 and corr_b == 1:
            acc_shift = -1
        row = {
            "model_name": r["model_name"],
            "item_id": r["item_id"],
            "condition": r["condition"],
            "accuracy_shift": acc_shift,
            "baseline_correct": corr_b,
            "condition_correct": corr_c,
        }
        for m in measure_cols:
            b_val = float(b[m]) if pd.notna(b[m]) else np.nan
            c_val = float(r[m]) if pd.notna(r[m]) else np.nan
            row[f"confidence_shift_{m}"] = c_val - b_val
        rows.append(row)

    return pd.DataFrame(rows)


def per_condition_summary(shifts: pd.DataFrame, measure_cols: Iterable[str]) -> pd.DataFrame:
    """Aggregate shifts by (model, condition)."""
    measure_cols = list(measure_cols)
    agg = {f"confidence_shift_{m}": ["mean", "std", "count"] for m in measure_cols}
    agg["accuracy_shift"] = ["mean", "std", "count"]
    grouped = shifts.groupby(["model_name", "condition"]).agg(agg)
    grouped.columns = ["__".join(c) for c in grouped.columns]
    return grouped.reset_index()
