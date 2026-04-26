"""Dissociation metrics.

The classical dissociation index (SPEC §Core analysis) is `DI = mean(Δconf) / mean(Δacc)`.
It is theoretically illuminating but undefined when `mean(Δacc) ≈ 0` — which is exactly
the central Koriat case (cue-familiarity inflation of confidence without any accuracy gain).

This module reports four complementary metrics, so the NaN of the ratio never buries the
substantive signal:

  1. `mean_confidence_shift_<m>` and `mean_accuracy_shift`  — raw means.
  2. `diff_<m>` = mean(Δconf) − mean(Δacc)                  — always defined; positive
      values mean the model's confidence moves more than its accuracy.
  3. `DI_ratio_<m>`                                         — the SPEC definition, NaN
      when accuracy shift is too small to stabilize.
  4. `DI_ratio_ci_lo_<m>`, `DI_ratio_ci_hi_<m>`              — percentile bootstrap CI on
      the ratio (1000 resamples of items), reported where the ratio is defined.
  5. `item_corr_<m>`                                        — per-item Pearson correlation
      between Δconf and Δacc within the condition. Values near 0 indicate dissociation
      (confidence moves independently of accuracy); near 1 indicates tracking.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


EPS = 1e-3
_BOOT_REPS = 1000


def _bootstrap_ratio_ci(
    conf: np.ndarray,
    acc: np.ndarray,
    n_reps: int = _BOOT_REPS,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for mean(conf)/mean(acc) over items.

    Returns (NaN, NaN) if too few valid bootstrap replicates produced a defined ratio
    (acc mean too often hits zero).
    """
    if len(conf) < 3:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = len(conf)
    ratios: list[float] = []
    for _ in range(n_reps):
        idx = rng.integers(0, n, size=n)
        a = acc[idx].mean()
        if abs(a) < EPS:
            continue
        ratios.append(float(conf[idx].mean() / a))
    if len(ratios) < 10:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(ratios, [2.5, 97.5])
    return (float(lo), float(hi))


def _item_correlation(conf: np.ndarray, acc: np.ndarray) -> float:
    """Pearson r between item-level Δconf and Δacc. NaN if variance is zero."""
    if len(conf) < 3:
        return float("nan")
    if np.std(conf) < 1e-9 or np.std(acc) < 1e-9:
        return float("nan")
    return float(np.corrcoef(conf, acc)[0, 1])


def dissociation_index(
    shifts: pd.DataFrame,
    measure_cols: Iterable[str],
    condition: str = "cue_familiarity_priming",
    seed: int = 0,
) -> pd.DataFrame:
    """Return a per-(model, measure) table of dissociation metrics for one condition."""
    sub = shifts[shifts["condition"] == condition]
    out: list[dict] = []
    for mdl, g in sub.groupby("model_name"):
        acc = g["accuracy_shift"].to_numpy(dtype=float)
        row: dict = {
            "model_name": mdl,
            "condition": condition,
            "n": int(len(g)),
            "mean_accuracy_shift": float(acc.mean()),
        }
        for m in measure_cols:
            col = f"confidence_shift_{m}"
            if col not in g.columns:
                continue
            conf = g[col].to_numpy(dtype=float)
            valid = ~np.isnan(conf) & ~np.isnan(acc)
            conf_v, acc_v = conf[valid], acc[valid]
            if len(conf_v) == 0:
                continue
            row[f"mean_confidence_shift_{m}"] = float(conf_v.mean())
            row[f"diff_{m}"] = float(conf_v.mean() - acc_v.mean())
            row[f"item_corr_{m}"] = _item_correlation(conf_v, acc_v)
            if abs(acc_v.mean()) > EPS:
                row[f"DI_ratio_{m}"] = float(conf_v.mean() / acc_v.mean())
                lo, hi = _bootstrap_ratio_ci(conf_v, acc_v, seed=seed)
                row[f"DI_ratio_ci_lo_{m}"] = lo
                row[f"DI_ratio_ci_hi_{m}"] = hi
            else:
                row[f"DI_ratio_{m}"] = np.nan
                row[f"DI_ratio_ci_lo_{m}"] = np.nan
                row[f"DI_ratio_ci_hi_{m}"] = np.nan
        out.append(row)
    return pd.DataFrame(out)


def compare_measures(
    shifts: pd.DataFrame,
    measure_cols: Iterable[str],
    conditions: Iterable[str] = (
        "cue_familiarity_priming",
        "partial_accessibility",
        "illusory_tot",
    ),
) -> pd.DataFrame:
    """Per (model, measure, condition), report the robust dissociation metrics."""
    out: list[dict] = []
    for (mdl, cond), g in shifts.groupby(["model_name", "condition"]):
        if cond not in conditions:
            continue
        acc = g["accuracy_shift"].to_numpy(dtype=float)
        for m in measure_cols:
            col = f"confidence_shift_{m}"
            if col not in g.columns:
                continue
            conf = g[col].to_numpy(dtype=float)
            valid = ~np.isnan(conf) & ~np.isnan(acc)
            conf_v, acc_v = conf[valid], acc[valid]
            if len(conf_v) == 0:
                continue
            entry = {
                "model_name": mdl,
                "condition": cond,
                "measure": m,
                "n": int(len(conf_v)),
                "mean_confidence_shift": float(conf_v.mean()),
                "mean_accuracy_shift": float(acc_v.mean()),
                "diff": float(conf_v.mean() - acc_v.mean()),
                "item_corr": _item_correlation(conf_v, acc_v),
            }
            if abs(acc_v.mean()) > EPS:
                entry["ratio"] = float(conf_v.mean() / acc_v.mean())
            else:
                entry["ratio"] = np.nan
            out.append(entry)
    df = pd.DataFrame(out)
    return df.sort_values("diff", ascending=False).reset_index(drop=True)
