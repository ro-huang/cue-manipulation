"""Stage 6: debugging plots.

These plots are deliberately designed to *surface problems* with the results
and the experimental setup, not to confirm them. Read them with a sceptic's eye:

  A. Levels per condition + 95% CI       — does the headline survive error bars?
  B. Per-measure distribution per cond   — saturation, bimodality, ceiling effects
  C. Δacc vs Δconfidence per item        — is the effect driven by outliers?
  D. Calibration curves                   — do confidence buckets predict accuracy?
  E. Stratification by baseline accuracy — does the priming effect hold across difficulty?
  F. Prime length / lexical overlap      — is cue_familiarity actually about familiarity?
  G. Cross-measure correlation per cond   — how independent are the four channels?
  H. Trial-order sanity                   — drift, reordering, or warm-up artifacts?
  I. Top predictions per condition        — is the model parroting / mode-collapsing?

Output: runs/<run>/plots/*.png
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st


_ALL_MEASURES = ["log_prob", "verbal_cat", "verbal_num", "caa_proj", "p_true"]
# `MEASURES` is set per-run from columns actually present in the parquet.
MEASURES: list[str] = list(_ALL_MEASURES)
NICE = {
    "log_prob": "log P(first answer token)",
    "verbal_cat": "verbalized categorical",
    "verbal_num": "verbalized numeric (0-100)",
    "caa_proj": "CAA projection",
    "p_true": "P(True) self-elicitation",
}


def _grid_2col(n: int) -> tuple[int, int]:
    """Pick (rows, cols) for `n` panels using 2 cols (or 1 if n==1)."""
    if n <= 1:
        return 1, 1
    return (n + 1) // 2, 2


def _hide_unused(axes_flat, n_used: int) -> None:
    for ax in list(axes_flat)[n_used:]:
        ax.set_visible(False)
COND_ORDER = [
    "baseline", "whitespace", "random_paragraph",
    "cue_familiarity_priming", "illusory_tot", "partial_accessibility",
    "fluency_degradation", "target_priming",
]
COND_COLOR = {
    "baseline": "#4f4f4f",
    "whitespace": "#9467bd",         # content-free filler
    "random_paragraph": "#1f77b4",   # the yoked content control
    "cue_familiarity_priming": "#d62728",
    "illusory_tot": "#ff7f0e",
    "partial_accessibility": "#8c564b",
    "fluency_degradation": "#bcbd22",
    "target_priming": "#2ca02c",
}


def _ci(x: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """t-based 95% CI on the mean."""
    if len(x) < 2:
        return (np.nan, np.nan)
    se = float(np.std(x, ddof=1) / np.sqrt(len(x)))
    h = se * st.t.ppf(1 - alpha / 2, len(x) - 1)
    m = float(np.mean(x))
    return (m - h, m + h)


def plot_a_levels(df: pd.DataFrame, out: Path) -> None:
    """A. Bar plot of mean ± 95% CI per condition for accuracy + each measure."""
    fields = [("correct", "accuracy")] + [(m, NICE[m]) for m in MEASURES]
    fig, axes = plt.subplots(1, len(fields), figsize=(3.2 * len(fields), 4), sharey=False)
    if len(fields) == 1:
        axes = [axes]
    for ax, (col, name) in zip(axes, fields):
        means, errs, colors = [], [], []
        for cond in COND_ORDER:
            sub = df[df.condition == cond][col].dropna().to_numpy(dtype=float)
            m = float(np.mean(sub))
            lo, hi = _ci(sub)
            means.append(m)
            errs.append((m - lo, hi - m))
            colors.append(COND_COLOR[cond])
        errs = np.array(errs).T
        ax.bar(COND_ORDER, means, yerr=errs, color=colors, edgecolor="black", capsize=4)
        ax.set_title(name)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("A. Per-condition levels (mean ± 95% CI). Watch for: tiny bars relative to CIs (effect is noise)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "A_levels.png", dpi=130)
    plt.close(fig)


def plot_b_distributions(df: pd.DataFrame, out: Path) -> None:
    """B. Distribution of each measure under each condition (overlaid).

    Reveals: ceiling/saturation (verbal_cat clipping at 1.0), bimodality, very-narrow
    distributions that mean differences hide behind, outlier-driven means.
    """
    nrows, ncols = _grid_2col(len(MEASURES))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 4.5 * nrows))
    axes_flat = np.atleast_1d(axes).flatten()
    for ax, m in zip(axes_flat, MEASURES):
        for cond in COND_ORDER:
            data = df[df.condition == cond][m].dropna().to_numpy(dtype=float)
            ax.hist(
                data, bins=30, alpha=0.4, label=cond, color=COND_COLOR[cond], density=True,
            )
        ax.set_xlabel(NICE[m])
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    _hide_unused(axes_flat, len(MEASURES))
    fig.suptitle(
        "B. Distributions per condition. Watch for: ceiling on verbal_cat, "
        "outlier tails on log_prob, narrow CAA blob",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out / "B_distributions.png", dpi=130)
    plt.close(fig)


def _paired_deltas(df: pd.DataFrame, cond: str, col: str) -> pd.DataFrame:
    """Per-item Δ(condition − baseline) for one measure. Returns DataFrame with item_id, dacc, dval."""
    base = df[df.condition == "baseline"].drop_duplicates("item_id").set_index("item_id")
    other = df[df.condition == cond].drop_duplicates("item_id").set_index("item_id")
    m = base.join(other, lsuffix="_b", rsuffix="_c", how="inner")
    out = pd.DataFrame({
        "item_id": list(m.index),
        "dacc": (m["correct_c"].astype(int) - m["correct_b"].astype(int)).values,
        "dval": (m[f"{col}_c"] - m[f"{col}_b"]).values,
    })
    return out.dropna().reset_index(drop=True)


def plot_c_paired_scatter(df: pd.DataFrame, out: Path) -> None:
    """C. Per-item paired Δ-accuracy vs Δ-confidence. Each dot = one item.

    Reveals whether the headline mean Δ is driven by (a) most items moving similarly,
    (b) bimodal split, or (c) a few outliers. Also reveals the per-item correlation
    structure that the regression pools.
    """
    cond_list = [c for c in COND_ORDER if c != "baseline"]
    fig, axes = plt.subplots(
        len(cond_list), len(MEASURES),
        figsize=(4 * len(MEASURES), 4 * len(cond_list)), sharey=False,
    )
    axes = np.atleast_2d(axes)
    if axes.shape == (len(MEASURES),) or axes.shape == (1,):
        axes = axes.reshape(len(cond_list), len(MEASURES))
    for j, m in enumerate(MEASURES):
        for i, cond in enumerate(cond_list):
            ax = axes[i, j]
            d = _paired_deltas(df, cond, m)
            # Add tiny jitter to discrete dacc (∈ {-1, 0, 1}) so it doesn't all stack.
            jitter = np.random.default_rng(0).normal(0, 0.02, size=len(d))
            ax.scatter(d["dacc"] + jitter, d["dval"], alpha=0.3, s=14, color=COND_COLOR[cond])
            ax.axhline(0, color="grey", linewidth=0.5)
            ax.axvline(0, color="grey", linewidth=0.5)
            ax.axhline(d["dval"].mean(), color="black", linewidth=1.5, linestyle="--",
                       label=f"mean Δ={d['dval'].mean():+.3f}")
            r, p = st.pearsonr(d["dacc"], d["dval"])
            ax.set_title(f"{cond}\n{NICE[m]}\nr={r:+.2f}, p={p:.2g}, n={len(d)}", fontsize=9)
            ax.set_xlabel("Δaccuracy (item)")
            ax.set_ylabel(f"Δ{m}")
            ax.legend(fontsize=8, loc="upper left")
            ax.grid(alpha=0.3)
    fig.suptitle(
        "C. Per-item Δ scatter. Watch for: clouds far from origin (effect real), "
        "single-side asymmetry (effect = floor/ceiling), strong r (confidence tracks accuracy)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out / "C_paired_scatter.png", dpi=130)
    plt.close(fig)


def plot_d_calibration(df: pd.DataFrame, out: Path) -> None:
    """D. Calibration curves: bin items by stated confidence, plot accuracy in each bin.

    Reveals: under perfect calibration the line is y=x. Big departures show the model
    over- or under-confident. We plot per condition so we can see whether priming
    *changes* the calibration relationship (the substantive claim) vs just the level.
    """
    fig, axes = plt.subplots(1, len(MEASURES), figsize=(4 * len(MEASURES), 4))
    if len(MEASURES) == 1:
        axes = [axes]
    bins = np.linspace(0, 1, 9)
    for ax, m in zip(axes, MEASURES):
        # Standardize per-trial measures to [0,1] within the whole df for plotting.
        x = df[m].astype(float)
        if m in ("log_prob", "caa_proj") or (
            np.isfinite(x).any() and (float(np.nanmin(x)) < 0 or float(np.nanmax(x)) > 1)
        ):
            lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
            x = (x - lo) / (hi - lo + 1e-9)
        for cond in COND_ORDER:
            mask = df.condition == cond
            xc = x[mask].to_numpy()
            yc = df.loc[mask, "correct"].astype(int).to_numpy()
            valid = ~np.isnan(xc)
            xc, yc = xc[valid], yc[valid]
            if len(xc) < 5:
                continue
            digitized = np.digitize(xc, bins) - 1
            digitized = np.clip(digitized, 0, len(bins) - 2)
            xs, ys = [], []
            for k in range(len(bins) - 1):
                inb = digitized == k
                if inb.sum() < 5:
                    continue
                xs.append((bins[k] + bins[k + 1]) / 2)
                ys.append(yc[inb].mean())
            ax.plot(xs, ys, "-o", color=COND_COLOR[cond], label=cond, alpha=0.85)
        ax.plot([0, 1], [0, 1], "k:", alpha=0.5, label="perfect calibration")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(f"{NICE[m]} (rescaled)")
        ax.set_ylabel("accuracy in bin")
        ax.set_title(NICE[m])
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle(
        "D. Calibration: accuracy vs reported confidence per bin. "
        "Watch for: target_priming line shifted up vs baseline (more correct at same conf level)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out / "D_calibration.png", dpi=130)
    plt.close(fig)


def plot_e_baseline_strat(df: pd.DataFrame, items_path: Path, out: Path) -> None:
    """E. Stratify Δ-confidence by baseline accuracy decile.

    The pre-registered band was [0.2, 0.8]. Items at the easy end (0.7-0.8) should
    have less room to move; items at the hard end (0.2-0.3) might respond differently.
    Big effect heterogeneity across deciles would suggest the effect is bound to a
    specific difficulty regime.
    """
    items = []
    with open(items_path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    base_acc = {it["id"]: it["baseline_accuracy"] for it in items}
    df = df.copy()
    df["base_acc"] = df["item_id"].map(base_acc)
    bins = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    df["bin"] = pd.cut(df["base_acc"], bins=bins, include_lowest=True)

    cond_list = [c for c in COND_ORDER if c != "baseline"]
    fig, axes = plt.subplots(
        len(cond_list), len(MEASURES),
        figsize=(4 * len(MEASURES), 4 * len(cond_list)),
    )
    axes = np.atleast_2d(axes)
    for j, m in enumerate(MEASURES):
        for i, cond in enumerate(cond_list):
            ax = axes[i, j]
            d = _paired_deltas(df, cond, m).merge(
                df[["item_id", "bin"]].drop_duplicates(),
                on="item_id", how="left",
            )
            grouped = d.groupby("bin", observed=True)["dval"].agg(["mean", "count", "sem"])
            xs = range(len(grouped))
            ax.bar(xs, grouped["mean"], yerr=1.96 * grouped["sem"],
                   color=COND_COLOR[cond], edgecolor="black", capsize=3)
            ax.set_xticks(xs)
            ax.set_xticklabels([str(b) for b in grouped.index], rotation=15, fontsize=8)
            ax.set_title(f"{cond}\nΔ{m}", fontsize=9)
            ax.set_xlabel("baseline-accuracy bin")
            ax.set_ylabel(f"mean Δ{m}")
            ax.axhline(0, color="grey", linewidth=0.5)
            ax.grid(axis="y", alpha=0.3)
            for x, n in enumerate(grouped["count"]):
                ax.text(x, ax.get_ylim()[1] * 0.95, f"n={n}",
                        ha="center", va="top", fontsize=7)
    fig.suptitle(
        "E. Δ stratified by baseline difficulty. Watch for: effect concentrated in 1-2 bins (regime-bound)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out / "E_baseline_strat.png", dpi=130)
    plt.close(fig)


def _word_set(s: str) -> set[str]:
    return set(re.findall(r"[a-z]+", (s or "").lower()))


def plot_f_prime_diagnostics(df: pd.DataFrame, primes_path: Path, out: Path) -> None:
    """F. Prime-quality diagnostics:
       (1) length distribution per condition,
       (2) lexical overlap (Jaccard) of prime with question, per condition,
       (3) Δverbal_cat vs lexical overlap for cue_familiarity_priming.

    If our cue_familiarity primes are systematically longer or share more words with
    the question than target primes, that's a confound. If the effect SCALES with
    overlap, that supports the cue-familiarity mechanism — but it also means we're
    measuring something fragile.
    """
    primes = []
    with open(primes_path) as f:
        for line in f:
            line = line.strip()
            if line:
                primes.append(json.loads(line))
    rows = []
    for ps in primes:
        q = ps.get("question", "")
        for c, prime in ps["primes"].items():
            text = prime.get("text") or ""
            if not text:
                continue
            qw, pw = _word_set(q), _word_set(text)
            j = len(qw & pw) / max(1, len(qw | pw))
            rows.append({
                "item_id": ps["item_id"],
                "condition": c,
                "n_words": len(text.split()),
                "jaccard": j,
            })
    pdf = pd.DataFrame(rows)

    fig = plt.figure(figsize=(14, 9))
    # (1) length per condition
    ax1 = fig.add_subplot(2, 2, 1)
    for cond in [c for c in COND_ORDER if c != "baseline"]:
        sub = pdf[pdf.condition == cond]["n_words"]
        ax1.hist(sub, bins=30, alpha=0.5, label=cond, color=COND_COLOR[cond])
    ax1.set_xlabel("prime length (words)")
    ax1.set_ylabel("count")
    ax1.set_title("F1. prime length distribution\n(should overlap; Δlength = confound)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # (2) Jaccard with question per condition
    ax2 = fig.add_subplot(2, 2, 2)
    for cond in [c for c in COND_ORDER if c != "baseline"]:
        sub = pdf[pdf.condition == cond]["jaccard"]
        ax2.hist(sub, bins=20, alpha=0.5, label=cond, color=COND_COLOR[cond])
    ax2.set_xlabel("Jaccard(prime, question) — surface lexical overlap")
    ax2.set_ylabel("count")
    ax2.set_title(
        "F2. lexical overlap with question\n"
        "(cue_familiarity should be HIGH; target should be LOWER)"
    )
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # (3) Δverbal_cat vs Jaccard for cue_familiarity_priming
    ax3 = fig.add_subplot(2, 1, 2)
    for cond in ["cue_familiarity_priming", "target_priming"]:
        d = _paired_deltas(df, cond, "verbal_cat").merge(
            pdf[pdf.condition == cond][["item_id", "jaccard", "n_words"]],
            on="item_id", how="inner",
        )
        if len(d) == 0:
            continue
        ax3.scatter(d["jaccard"], d["dval"], alpha=0.4, s=18,
                    color=COND_COLOR[cond], label=f"{cond} (n={len(d)})")
        if len(d) >= 5:
            r, p = st.pearsonr(d["jaccard"], d["dval"])
            slope, intercept, *_ = st.linregress(d["jaccard"], d["dval"])
            xs = np.linspace(d["jaccard"].min(), d["jaccard"].max(), 50)
            ax3.plot(xs, slope * xs + intercept, color=COND_COLOR[cond], linewidth=1.5,
                     label=f"  fit r={r:+.2f}, p={p:.2g}")
    ax3.axhline(0, color="grey", linewidth=0.5)
    ax3.set_xlabel("Jaccard(prime, question)")
    ax3.set_ylabel("Δverbal_cat (per item)")
    ax3.set_title(
        "F3. Does Δverbal_cat scale with lexical overlap? "
        "Positive r in cue_familiarity = cue-familiarity mechanism confirmed"
    )
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "F_prime_diagnostics.png", dpi=130)
    plt.close(fig)


def plot_g_correlation(df: pd.DataFrame, out: Path) -> None:
    """G. Cross-measure correlation matrix per condition.

    All 4 measures should be positively correlated within an item if they're tapping
    the same construct. If verbal_cat and CAA aren't correlated, they're measuring
    different things — which is the substantive claim, but should be visible here.
    """
    fig, axes = plt.subplots(1, len(COND_ORDER), figsize=(13, 4.5))
    for ax, cond in zip(axes, COND_ORDER):
        sub = df[df.condition == cond][MEASURES].astype(float)
        c = sub.corr()
        im = ax.imshow(c, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(MEASURES)))
        ax.set_yticks(range(len(MEASURES)))
        ax.set_xticklabels(MEASURES, rotation=30, fontsize=8)
        ax.set_yticklabels(MEASURES, fontsize=8)
        for i in range(len(MEASURES)):
            for j in range(len(MEASURES)):
                ax.text(j, i, f"{c.iloc[i,j]:.2f}",
                        ha="center", va="center", fontsize=8,
                        color="white" if abs(c.iloc[i, j]) > 0.5 else "black")
        ax.set_title(cond, fontsize=10)
    fig.suptitle(
        "G. Within-condition correlation between measures. "
        "Watch for: verbal_* and caa_proj diverging (different constructs)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out / "G_correlation.png", dpi=130)
    plt.close(fig)


def plot_h_trial_order(df: pd.DataFrame, out: Path) -> None:
    """H. Trial-order plot. Each row in the parquet has an implicit trial index from
    when it was generated. We plot the rolling mean of each measure by index, per
    condition. If there's any drift / warm-up / mid-run change, it shows here.
    """
    df = df.copy()
    df["row"] = np.arange(len(df))
    nrows, ncols = _grid_2col(len(MEASURES))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows))
    axes_flat = np.atleast_1d(axes).flatten()
    window = max(20, len(df) // 50)
    for ax, m in zip(axes_flat, MEASURES):
        for cond in COND_ORDER:
            sub = df[df.condition == cond].sort_values("row")
            x = sub[m].astype(float).rolling(window, min_periods=window // 2).mean()
            ax.plot(sub["row"], x, color=COND_COLOR[cond], alpha=0.85, label=cond)
        ax.set_xlabel("row index in parquet")
        ax.set_ylabel(f"rolling mean {m} (window={window})")
        ax.set_title(NICE[m])
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    _hide_unused(axes_flat, len(MEASURES))
    fig.suptitle(
        "H. Rolling mean by trial order. Watch for: drift across rows = state bug or schedule artifact",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out / "H_trial_order.png", dpi=130)
    plt.close(fig)


def plot_i_top_predictions(df: pd.DataFrame, out: Path) -> None:
    """I. Top-N predictions per condition. If the model collapses to a few high-frequency
    answers under priming, we'd see it here. Real Koriat-style enhancement should
    distribute across many distinct answers.
    """
    fig, axes = plt.subplots(1, len(COND_ORDER), figsize=(15, 6))
    for ax, cond in zip(axes, COND_ORDER):
        sub = df[df.condition == cond]
        c = Counter(sub["prediction"].astype(str).str.lower().str.strip()).most_common(15)
        labels = [f"{k!r}" for k, _ in c]
        counts = [v for _, v in c]
        ax.barh(range(len(c))[::-1], counts, color=COND_COLOR[cond], edgecolor="black")
        ax.set_yticks(range(len(c))[::-1])
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("count")
        ax.set_title(cond, fontsize=10)
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle(
        "I. Top-15 predictions by condition. Watch for: collapse to a few answers in any condition",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out / "I_top_predictions.png", dpi=130)
    plt.close(fig)


def plot_m_against_yoked(df: pd.DataFrame, out: Path) -> None:
    """M. Both controls together: levels per condition, plus paired Δ of each priming
    condition against BOTH random_paragraph and whitespace, item-matched.

    Two control comparisons disambiguate:
      - vs random_paragraph: does priming beat irrelevant content?
      - vs whitespace: does priming beat any-prefix-at-all?
    """
    fields = [("correct", "accuracy")] + [(m, NICE[m]) for m in MEASURES]
    fig, axes = plt.subplots(3, len(fields), figsize=(3.5 * len(fields), 11))
    axes = np.atleast_2d(axes)

    # Row 0: bars per condition
    for ax, (col, name) in zip(axes[0], fields):
        means, errs, colors = [], [], []
        for cond in COND_ORDER:
            sub = df[df.condition == cond][col].dropna().to_numpy(dtype=float)
            m = float(np.mean(sub))
            lo, hi = _ci(sub)
            means.append(m)
            errs.append((m - lo, hi - m))
            colors.append(COND_COLOR[cond])
        errs = np.array(errs).T
        ax.bar(COND_ORDER, means, yerr=errs, color=colors, edgecolor="black", capsize=4)
        ax.set_title(name, fontsize=10)
        ax.tick_params(axis="x", rotation=25, labelsize=7)
        ax.grid(axis="y", alpha=0.3)

    def _paired(ax, col, ref_cond, ref_short):
        means, errs, colors, sigs, ns = [], [], [], [], []
        labels = []
        for cond in ["cue_familiarity_priming", "target_priming"]:
            a = df[df.condition == ref_cond].drop_duplicates("item_id").set_index("item_id")
            b = df[df.condition == cond].drop_duplicates("item_id").set_index("item_id")
            m_ = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
            if col == "correct":
                d = (m_["correct_b"].astype(int) - m_["correct_a"].astype(int)).dropna()
            else:
                d = (m_[f"{col}_b"] - m_[f"{col}_a"]).dropna()
            mn = float(d.mean())
            lo, hi = _ci(d.values)
            t = st.ttest_1samp(d, 0)
            means.append(mn); errs.append([(mn - lo), (hi - mn)]); colors.append(COND_COLOR[cond])
            sigs.append("***" if t.pvalue < 1e-3 else "**" if t.pvalue < 1e-2 else "*" if t.pvalue < 5e-2 else "n.s.")
            ns.append(len(d))
            labels.append({"cue_familiarity_priming": "cue_fam", "target_priming": "target"}[cond])
        errs = np.array(errs).T
        ax.bar(labels, means, yerr=errs, color=colors, edgecolor="black", capsize=4)
        for x, mv, sg in zip(range(len(means)), means, sigs):
            offset = 0.005 if mv >= 0 else -0.005
            ax.text(x, mv + offset, sg, ha="center",
                    va="bottom" if mv >= 0 else "top", fontsize=8)
        ax.axhline(0, color="grey", linewidth=0.5)
        ax.grid(axis="y", alpha=0.3)
        ax.set_title(f"Δ vs {ref_short}", fontsize=9)
        ax.tick_params(labelsize=8)

    for ax, (col, name) in zip(axes[1], fields):
        _paired(ax, col, "random_paragraph", "rand_para")
    for ax, (col, name) in zip(axes[2], fields):
        _paired(ax, col, "whitespace", "whitespace")

    fig.suptitle(
        "M. Two yoked controls side-by-side.\n"
        "Top: levels per condition. Middle: priming Δ vs random_paragraph. Bottom: priming Δ vs whitespace.",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out / "M_vs_controls.png", dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()

    rd = Path(args.run_dir)
    plots = rd / "plots"
    plots.mkdir(exist_ok=True)

    df = pd.read_parquet(rd / "trials.parquet")
    n_before = len(df)
    df = df.drop_duplicates(subset=["model_name", "item_id", "condition"], keep="first").reset_index(drop=True)
    print(f"trials: {n_before} → {len(df)} after dedup")

    # Restrict measures to those actually populated in this run.
    global MEASURES
    MEASURES = [m for m in _ALL_MEASURES if m in df.columns and df[m].notna().any()]
    print(f"plotting measures: {MEASURES}")

    print("[A] levels per condition")
    plot_a_levels(df, plots)
    print("[B] distribution per measure")
    plot_b_distributions(df, plots)
    print("[C] paired-Δ scatter")
    plot_c_paired_scatter(df, plots)
    print("[D] calibration curves")
    plot_d_calibration(df, plots)
    print("[E] baseline-difficulty stratification")
    plot_e_baseline_strat(df, rd / "items_filtered.jsonl", plots)
    print("[F] prime quality diagnostics")
    plot_f_prime_diagnostics(df, rd / "primes.jsonl", plots)
    print("[G] cross-measure correlation")
    plot_g_correlation(df, plots)
    print("[H] trial-order rolling mean")
    plot_h_trial_order(df, plots)
    print("[I] top predictions per condition")
    plot_i_top_predictions(df, plots)
    if "random_paragraph" in df["condition"].unique():
        print("[M] random_paragraph yoked control")
        plot_m_against_yoked(df, plots)
    print(f"wrote plots to {plots}/")


if __name__ == "__main__":
    main()
