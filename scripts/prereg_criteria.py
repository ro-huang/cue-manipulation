"""Print C1/C2/C3 pre-registration verdicts per model from trials.parquet.

Uses the same paired Δ vs baseline and bootstrap CIs as
`scripts/10_headline_figure.py` (per-model min-max standardization of
log_prob, verbal_cat, p_true). See `.context/preregistration.md` §3.1.

Usage:
    PYTHONPATH=src python scripts/prereg_criteria.py runs/replication_v2/trials.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from koriat_cues.analysis.paired import paired_delta_vs_baseline
from koriat_cues.confidence.measures import standardize_per_model


def _fmt(mean: float, p: float) -> str:
    if not (pd.notna(mean) and pd.notna(p)):
        return "n/a"
    return f"Δ={mean:+.4f}, p={p:.4g}"


def _c1_pass(mean: float, p: float, *, want_pos: bool, flat_p: float, flat_abs: float) -> bool:
    if not (pd.notna(mean) and pd.notna(p)):
        return False
    if want_pos:
        return mean > 0 and p < 0.05
    return p > flat_p or abs(mean) < flat_abs


def _c2_p_true_vs_log(mean_lp: float, mean_pt: float, p_pt: float) -> bool:
    """p_true flat (n.s.) OR smaller |Δ| than log_prob."""
    if not (pd.notna(mean_lp) and pd.notna(mean_pt) and pd.notna(p_pt)):
        return False
    if p_pt > 0.05:
        return True
    return abs(mean_pt) < abs(mean_lp)


def verdicts_for_model(df: pd.DataFrame) -> dict[str, bool | str]:
    measures = ["log_prob", "verbal_cat", "p_true"]
    present = [m for m in measures if m in df.columns and df[m].notna().any()]
    work = standardize_per_model(df.copy(), present)

    def cell(cond: str, col: str):
        mean, _lo, _hi, p, n = paired_delta_vs_baseline(work, cond, col)
        return mean, p, n

    # C1: cue_familiarity_priming
    m_lp, p_lp, n1 = cell("cue_familiarity_priming", "log_prob")
    m_vc, p_vc, _ = cell("cue_familiarity_priming", "verbal_cat")
    m_pt, p_pt, _ = cell("cue_familiarity_priming", "p_true")
    m_acc, p_acc, _ = cell("cue_familiarity_priming", "correct")
    c1 = (
        _c1_pass(m_lp, p_lp, want_pos=True, flat_p=0.05, flat_abs=0.0)
        and _c1_pass(m_vc, p_vc, want_pos=True, flat_p=0.05, flat_abs=0.0)
        and _c1_pass(m_pt, p_pt, want_pos=False, flat_p=0.05, flat_abs=0.02)
        and _c1_pass(m_acc, p_acc, want_pos=False, flat_p=0.05, flat_abs=0.03)
    )

    # C2: partial_accessibility
    m_lp2, p_lp2, _ = cell("partial_accessibility", "log_prob")
    m_vc2, p_vc2, _ = cell("partial_accessibility", "verbal_cat")
    m_pt2, p_pt2, _ = cell("partial_accessibility", "p_true")
    m_acc2, p_acc2, _ = cell("partial_accessibility", "correct")
    c2 = (
        pd.notna(m_lp2)
        and pd.notna(p_lp2)
        and m_lp2 > 0
        and p_lp2 < 0.05
        and pd.notna(m_vc2)
        and pd.notna(p_vc2)
        and m_vc2 > 0
        and p_vc2 < 0.05
        and _c2_p_true_vs_log(m_lp2, m_pt2, p_pt2)
        and pd.notna(m_acc2)
        and pd.notna(p_acc2)
        and m_acc2 < 0
        and p_acc2 < 0.05
    )

    # C3: target_priming accuracy up
    m_acc3, p_acc3, _ = cell("target_priming", "correct")
    c3 = pd.notna(m_acc3) and pd.notna(p_acc3) and m_acc3 > 0 and p_acc3 < 0.05

    lines = [
        f"  n paired (C1 cue_fam): {n1}",
        f"  C1 cue_familiarity_priming: log_prob {_fmt(m_lp, p_lp)} → "
        f"{'PASS' if _c1_pass(m_lp, p_lp, want_pos=True, flat_p=0.05, flat_abs=0.0) else 'FAIL'}",
        f"                              verbal_cat {_fmt(m_vc, p_vc)} → "
        f"{'PASS' if _c1_pass(m_vc, p_vc, want_pos=True, flat_p=0.05, flat_abs=0.0) else 'FAIL'}",
        f"                              p_true {_fmt(m_pt, p_pt)} → "
        f"{'PASS' if _c1_pass(m_pt, p_pt, want_pos=False, flat_p=0.05, flat_abs=0.02) else 'FAIL'}",
        f"                              accuracy {_fmt(m_acc, p_acc)} → "
        f"{'PASS' if _c1_pass(m_acc, p_acc, want_pos=False, flat_p=0.05, flat_abs=0.03) else 'FAIL'}",
        f"  C1 overall: {'PASS' if c1 else 'FAIL'}",
        f"  C2 partial_accessibility: log_prob {_fmt(m_lp2, p_lp2)}, verbal_cat {_fmt(m_vc2, p_vc2)}, "
        f"p_true {_fmt(m_pt2, p_pt2)}, accuracy {_fmt(m_acc2, p_acc2)}",
        f"  C2 p_true rule (flat or |Δ_pt| < |Δ_lp|): "
        f"{'PASS' if _c2_p_true_vs_log(m_lp2, m_pt2, p_pt2) else 'FAIL'}",
        f"  C2 overall: {'PASS' if c2 else 'FAIL'}",
        f"  C3 target_priming accuracy: {_fmt(m_acc3, p_acc3)} → {'PASS' if c3 else 'FAIL'}",
    ]
    headline = "PASS" if (c1 and c2 and c3) else "FAIL"
    lines.append(f"  Headline (C1∧C2∧C3): {headline}")
    return {
        "C1": bool(c1),
        "C2": bool(c2),
        "C3": bool(c3),
        "headline": headline,
        "detail": "\n".join(lines),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "parquet",
        type=Path,
        help="Path to trials.parquet (e.g. runs/replication_v2/trials.parquet)",
    )
    args = ap.parse_args()
    df = pd.read_parquet(args.parquet)
    df = df.drop_duplicates(subset=["model_name", "item_id", "condition"], keep="first").reset_index(
        drop=True
    )
    for model_name, sub in df.groupby("model_name"):
        print(f"\n=== {model_name} ===")
        r = verdicts_for_model(sub)
        print(r["detail"])
        print(f"  C1={r['C1']}  C2={r['C2']}  C3={r['C3']}  headline={r['headline']}")


if __name__ == "__main__":
    main()
