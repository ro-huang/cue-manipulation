"""Pre-registered pass/fail criteria (.context/preregistration.md §3.1).

Uses paired Δ vs baseline per item, paired one-sample t-tests, and the same
per-model min–max standardization as ``scripts/10_headline_figure.py`` /
``scripts/05_analyze.py`` for confidence measures. Accuracy (``correct``) is 0/1
and is not rescaled.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import scipy.stats as st

from koriat_cues.confidence.measures import standardize_per_model

PRIMARY_MEASURES = ["log_prob", "verbal_cat", "p_true"]


def paired_delta_vector(
    df: pd.DataFrame,
    cond: str,
    col: str,
    ref_cond: str = "baseline",
) -> pd.Series:
    """Primed − ref per ``item_id`` for one column (or ``correct`` if col == \"correct\")."""
    a = df[df.condition == ref_cond].drop_duplicates("item_id").set_index("item_id")
    b = df[df.condition == cond].drop_duplicates("item_id").set_index("item_id")
    m = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
    if col == "correct":
        d = m["correct_b"].astype(int) - m["correct_a"].astype(int)
    else:
        d = m[f"{col}_b"] - m[f"{col}_a"]
    return d.dropna()


def paired_stats(d: pd.Series) -> tuple[float, float, int]:
    """Mean Δ, two-sided p vs 0, n (or nan, nan, 0 if too few rows)."""
    arr = d.to_numpy(dtype=float)
    n = int(arr.size)
    if n < 2:
        return float("nan"), float("nan"), n
    mu = float(np.mean(arr))
    if float(np.std(arr, ddof=1)) == 0.0:
        return mu, 1.0 if mu == 0.0 else 0.0, n
    p = float(st.ttest_1samp(arr, 0.0).pvalue)
    return mu, p, n


@dataclass(frozen=True)
class CriterionResult:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class ModelPreregVerdict:
    model_name: str
    c1: CriterionResult
    c2: CriterionResult
    c3: CriterionResult

    @property
    def headline_pass(self) -> bool:
        return self.c1.passed and self.c2.passed and self.c3.passed


def _fmt(mu: float, p: float, n: int) -> str:
    if not np.isfinite(mu) or not np.isfinite(p):
        return f"n={n} (insufficient data)"
    return f"Δ={mu:+.4f}, p={p:.4g}, n={n}"


def evaluate_model_prereg(df_model: pd.DataFrame) -> ModelPreregVerdict:
    """Evaluate C1–C3 for one model's trials (all conditions present in ``df_model``)."""
    mname = str(df_model["model_name"].iloc[0])
    present = [
        x for x in PRIMARY_MEASURES
        if x in df_model.columns and df_model[x].notna().any()
    ]
    work = df_model.copy()
    work = standardize_per_model(work, present)

    # --- C1: cue_familiarity_priming ---
    c1_parts: list[str] = []
    c1_ok = True
    cond_c1 = "cue_familiarity_priming"
    for col in ("log_prob", "verbal_cat"):
        if col not in present:
            c1_parts.append(f"{col}: MISSING → fail")
            c1_ok = False
            continue
        d = paired_delta_vector(work, cond_c1, col)
        mu, p, n = paired_stats(d)
        ok = np.isfinite(mu) and mu > 0 and np.isfinite(p) and p < 0.05
        c1_parts.append(f"{col}: {_fmt(mu, p, n)} → {'pass' if ok else 'fail'}")
        c1_ok = c1_ok and ok

    if "p_true" in present:
        d = paired_delta_vector(work, cond_c1, "p_true")
        mu, p, n = paired_stats(d)
        ok = (np.isfinite(p) and p > 0.05) or (np.isfinite(mu) and abs(mu) < 0.02)
        c1_parts.append(f"p_true: {_fmt(mu, p, n)} → {'pass' if ok else 'fail'}")
        c1_ok = c1_ok and ok
    else:
        c1_parts.append("p_true: MISSING → fail")
        c1_ok = False

    d_acc = paired_delta_vector(work, cond_c1, "correct")
    mu, p, n = paired_stats(d_acc)
    ok = (np.isfinite(p) and p > 0.05) or (np.isfinite(mu) and abs(mu) < 0.03)
    c1_parts.append(f"accuracy: {_fmt(mu, p, n)} → {'pass' if ok else 'fail'}")
    c1_ok = c1_ok and ok

    c1 = CriterionResult("C1 cue_familiarity_priming", c1_ok, "; ".join(c1_parts))

    # --- C2: partial_accessibility ---
    c2_parts: list[str] = []
    c2_ok = True
    cond_c2 = "partial_accessibility"
    for col in ("log_prob", "verbal_cat"):
        if col not in present:
            c2_parts.append(f"{col}: MISSING → fail")
            c2_ok = False
            continue
        d = paired_delta_vector(work, cond_c2, col)
        mu, p, n = paired_stats(d)
        ok = np.isfinite(mu) and mu > 0 and np.isfinite(p) and p < 0.05
        c2_parts.append(f"{col}: {_fmt(mu, p, n)} → {'pass' if ok else 'fail'}")
        c2_ok = c2_ok and ok

    if "p_true" in present:
        d_lp = paired_delta_vector(work, cond_c2, "log_prob")
        d_pt = paired_delta_vector(work, cond_c2, "p_true")
        mu_lp, _, n_lp = paired_stats(d_lp)
        mu_pt, p_pt, n_pt = paired_stats(d_pt)
        flat = np.isfinite(p_pt) and p_pt > 0.05
        smaller_mag = np.isfinite(mu_pt) and np.isfinite(mu_lp) and abs(mu_pt) < abs(mu_lp)
        ok = flat or smaller_mag
        c2_parts.append(
            f"p_true: Δ_lp={mu_lp:+.4f}, Δ_pt={mu_pt:+.4f}, p_pt={p_pt:.4g}, n={n_pt} "
            f"(flat={flat}, |Δ_pt|<|Δ_lp|={smaller_mag}) → {'pass' if ok else 'fail'}"
        )
        c2_ok = c2_ok and ok
    else:
        c2_parts.append("p_true: MISSING → fail")
        c2_ok = False

    d_acc = paired_delta_vector(work, cond_c2, "correct")
    mu, p, n = paired_stats(d_acc)
    ok = np.isfinite(mu) and mu < 0 and np.isfinite(p) and p < 0.05
    c2_parts.append(f"accuracy: {_fmt(mu, p, n)} → {'pass' if ok else 'fail'}")
    c2_ok = c2_ok and ok

    c2 = CriterionResult("C2 partial_accessibility", c2_ok, "; ".join(c2_parts))

    # --- C3: target_priming accuracy up ---
    cond_c3 = "target_priming"
    d_acc = paired_delta_vector(work, cond_c3, "correct")
    mu, p, n = paired_stats(d_acc)
    ok = np.isfinite(mu) and mu > 0 and np.isfinite(p) and p < 0.05
    c3 = CriterionResult(
        "C3 target_priming (accuracy)",
        ok,
        f"accuracy: {_fmt(mu, p, n)} → {'pass' if ok else 'fail'}",
    )

    return ModelPreregVerdict(model_name=mname, c1=c1, c2=c2, c3=c3)


def evaluate_run_parquet(
    df: pd.DataFrame,
    model_names: Iterable[str] | None = None,
) -> list[ModelPreregVerdict]:
    """Deduplicate like the headline script, optionally filter models, return verdicts."""
    df = df.drop_duplicates(subset=["model_name", "item_id", "condition"], keep="first").reset_index(
        drop=True
    )
    names = df["model_name"].unique()
    if model_names is not None:
        want = set(model_names)
        names = [n for n in names if n in want]
    return [evaluate_model_prereg(df[df.model_name == n]) for n in names]


def format_verdict_lines(verdicts: list[ModelPreregVerdict]) -> str:
    lines: list[str] = []
    for v in verdicts:
        lines.append(f"=== {v.model_name} ===")
        for c in (v.c1, v.c2, v.c3):
            mark = "PASS" if c.passed else "FAIL"
            lines.append(f"  [{mark}] {c.name}")
            lines.append(f"       {c.detail}")
        lines.append(f"  Headline (C1∧C2∧C3): {'PASS' if v.headline_pass else 'FAIL'}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
