"""Print pre-registered C1/C2/C3 pass-fail per model from trials.parquet.

Criteria mirror `.context/preregistration.md` §3.1 (paired Δ vs baseline per item).
Measures are min-max standardized within model before deltas, matching 05_analyze /
10_headline_figure.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from koriat_cues.confidence.measures import standardize_per_model

PRIMARY_MEASURES = ["log_prob", "verbal_cat", "p_true"]


def _paired_delta(df: pd.DataFrame, cond: str, col: str, n_boot: int = 1000, seed: int = 0):
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
        from math import erfc, sqrt as _sq

        p = erfc(abs(t) / _sq(2)) if np.isfinite(t) else float("nan")
    return float(d.mean()), float(lo), float(hi), p, n


def _one_sided_gt0(mean: float, p_two: float) -> bool:
    """Primary tests expect positive shift; use one-sided p < 0.05 from two-sided t p."""
    if not np.isfinite(mean) or not np.isfinite(p_two):
        return False
    if mean <= 0:
        return False
    return p_two / 2.0 < 0.05


def _one_sided_lt0(mean: float, p_two: float) -> bool:
    if not np.isfinite(mean) or not np.isfinite(p_two):
        return False
    if mean >= 0:
        return False
    return p_two / 2.0 < 0.05


def _ns_or_small(mean: float, p_two: float, abs_thresh: float) -> bool:
    """Not significant (two-sided p > .05) OR small |mean|."""
    if abs(mean) < abs_thresh:
        return True
    return np.isfinite(p_two) and p_two > 0.05


def verdict_for_model(df: pd.DataFrame, model_name: str) -> dict[str, bool | str]:
    sub = df[df.model_name == model_name].copy()
    if sub.empty:
        return {"model": model_name, "error": "no rows"}

    present = [m for m in PRIMARY_MEASURES if m in sub.columns and sub[m].notna().any()]
    sub = standardize_per_model(sub, present)

    # C1: cue_familiarity_priming
    m_lp, _, _, p_lp, _ = _paired_delta(sub, "cue_familiarity_priming", "log_prob")
    m_vc, _, _, p_vc, _ = _paired_delta(sub, "cue_familiarity_priming", "verbal_cat")
    m_pt, _, _, p_pt, _ = _paired_delta(sub, "cue_familiarity_priming", "p_true")
    m_ac, _, _, p_ac, _ = _paired_delta(sub, "cue_familiarity_priming", "correct")
    c1 = (
        _one_sided_gt0(m_lp, p_lp)
        and _one_sided_gt0(m_vc, p_vc)
        and _ns_or_small(m_pt, p_pt, 0.02)
        and _ns_or_small(m_ac, p_ac, 0.03)
    )

    # C2: partial_accessibility
    m_lp2, _, _, p_lp2, _ = _paired_delta(sub, "partial_accessibility", "log_prob")
    m_vc2, _, _, p_vc2, _ = _paired_delta(sub, "partial_accessibility", "verbal_cat")
    m_pt2, _, _, p_pt2, _ = _paired_delta(sub, "partial_accessibility", "p_true")
    m_ac2, _, _, p_ac2, _ = _paired_delta(sub, "partial_accessibility", "correct")
    ptrue_ok = (np.isfinite(p_pt2) and p_pt2 > 0.05) or (
        np.isfinite(m_pt2) and np.isfinite(m_lp2) and abs(m_pt2) < abs(m_lp2)
    )
    c2 = (
        _one_sided_gt0(m_lp2, p_lp2)
        and _one_sided_gt0(m_vc2, p_vc2)
        and ptrue_ok
        and _one_sided_lt0(m_ac2, p_ac2)
    )

    # C3: target_priming accuracy up
    m_ac3, _, _, p_ac3, _ = _paired_delta(sub, "target_priming", "correct")
    c3 = _one_sided_gt0(m_ac3, p_ac3)

    lines = [
        f"  C1 cue_familiarity: log_prob Δ={m_lp:.4f} p={p_lp:.4g} → {'PASS' if _one_sided_gt0(m_lp, p_lp) else 'FAIL'}",
        f"                      verbal_cat Δ={m_vc:.4f} p={p_vc:.4g} → {'PASS' if _one_sided_gt0(m_vc, p_vc) else 'FAIL'}",
        f"                      p_true Δ={m_pt:.4f} p={p_pt:.4g} → {'PASS' if _ns_or_small(m_pt, p_pt, 0.02) else 'FAIL'} (ns or |Δ|<0.02)",
        f"                      accuracy Δ={m_ac:.4f} p={p_ac:.4g} → {'PASS' if _ns_or_small(m_ac, p_ac, 0.03) else 'FAIL'} (ns or |Δ|<0.03)",
        f"  C1 overall: {'PASS' if c1 else 'FAIL'}",
        f"  C2 partial_acc: log_prob Δ={m_lp2:.4f} p={p_lp2:.4g} → {'PASS' if _one_sided_gt0(m_lp2, p_lp2) else 'FAIL'}",
        f"                  verbal_cat Δ={m_vc2:.4f} p={p_vc2:.4g} → {'PASS' if _one_sided_gt0(m_vc2, p_vc2) else 'FAIL'}",
        f"                  p_true flat OR |Δ|<|Δ_logprob|: {'PASS' if ptrue_ok else 'FAIL'}",
        f"                  accuracy Δ={m_ac2:.4f} p={p_ac2:.4g} → {'PASS' if _one_sided_lt0(m_ac2, p_ac2) else 'FAIL'}",
        f"  C2 overall: {'PASS' if c2 else 'FAIL'}",
        f"  C3 target_priming accuracy Δ={m_ac3:.4f} p={p_ac3:.4g} → {'PASS' if c3 else 'FAIL'}",
        f"  C1∧C2∧C3: {'PASS' if (c1 and c2 and c3) else 'FAIL'}",
    ]
    return {
        "model": model_name,
        "C1": c1,
        "C2": c2,
        "C3": c3,
        "all": c1 and c2 and c3,
        "detail": "\n".join(lines),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--trials",
        type=Path,
        required=True,
        help="Path to trials.parquet (may contain multiple model_name values).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to write the same report as markdown.",
    )
    args = ap.parse_args()

    df = pd.read_parquet(args.trials)
    df = df.drop_duplicates(subset=["model_name", "item_id", "condition"], keep="first")
    models = sorted(df["model_name"].unique())

    blocks: list[str] = []
    for m in models:
        v = verdict_for_model(df, m)
        if "error" in v:
            blocks.append(f"## {m}\n{v['error']}\n")
            continue
        blocks.append(f"## {v['model']}\n{v['detail']}\n")

    text = "# Pre-registration verdict (§3.1)\n\n" + "\n".join(blocks)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
