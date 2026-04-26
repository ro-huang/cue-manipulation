"""Headline figure for the extended Koriat condition matrix.

Shows the inflation gap (Δconf − Δacc) by condition × channel, plus the per-channel
shifts vs the baseline reference. Designed to surface the manipulation-specific
dissociation pattern: surface-channel inflation under cue_familiarity / illusory_tot,
uniform-channel inflation under partial_accessibility.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st


_ALL_MEASURES = ["log_prob", "verbal_cat", "verbal_num", "caa_proj", "p_true"]
NICE = {
    "log_prob": "log P(answer)",
    "verbal_cat": "verbal cat",
    "verbal_num": "verbal num",
    "caa_proj": "CAA proj",
    "p_true": "P(True)",
}
CHANNEL_FAMILY = {
    "log_prob": "surface",
    "verbal_cat": "surface",
    "verbal_num": "surface",
    "caa_proj": "coherence",
    "p_true": "coherence",
}
COND_ORDER = [
    "cue_familiarity_priming",
    "illusory_tot",
    "partial_accessibility",
    "fluency_degradation",
    "target_priming",
    "random_paragraph",
    "whitespace",
]
COND_SHORT = {
    "cue_familiarity_priming": "cue_familiarity",
    "illusory_tot": "illusory_tot",
    "partial_accessibility": "partial_access",
    "fluency_degradation": "fluency_degrad",
    "target_priming": "target",
    "random_paragraph": "rand_para",
    "whitespace": "whitespace",
}


def _paired_delta(df, ref_cond, cond, col):
    a = df[df.condition == ref_cond].drop_duplicates("item_id").set_index("item_id")
    b = df[df.condition == cond].drop_duplicates("item_id").set_index("item_id")
    m = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
    if col == "correct":
        d = (m["correct_b"].astype(int) - m["correct_a"].astype(int))
    else:
        d = (m[f"{col}_b"] - m[f"{col}_a"])
    d = d.dropna()
    if len(d) < 2:
        return float("nan"), float("nan"), float("nan"), 0
    se = float(np.std(d, ddof=1) / np.sqrt(len(d)))
    h = se * st.t.ppf(0.975, len(d) - 1)
    p = float(st.ttest_1samp(d, 0).pvalue) if d.std() > 0 else 1.0
    return float(d.mean()), h, p, len(d)


def _star(p):
    if p < 1e-3: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return ""


def _standardize(df, measures):
    """Min-max scale per (model, measure) to [0,1] so channels are comparable."""
    df = df.copy()
    for m in measures:
        for mdl, sub in df.groupby("model_name"):
            lo, hi = sub[m].min(), sub[m].max()
            if hi - lo > 1e-9:
                df.loc[sub.index, m] = (sub[m] - lo) / (hi - lo)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--ref", default="baseline", help="reference condition (baseline or whitespace)")
    args = ap.parse_args()
    rd = Path(args.run_dir)

    df = pd.read_parquet(rd / "trials.parquet")
    df = df.drop_duplicates(subset=["model_name", "item_id", "condition"], keep="first").reset_index(drop=True)
    measures = [m for m in _ALL_MEASURES if m in df.columns and df[m].notna().any()]
    df = _standardize(df, measures)

    conds = [c for c in COND_ORDER if c in df["condition"].unique()]

    # Compute inflation gap (Δconf − Δacc) and absolute shifts per (cond, channel)
    gap_matrix = np.full((len(conds), len(measures)), np.nan)
    conf_shift = np.full((len(conds), len(measures)), np.nan)
    conf_err = np.full((len(conds), len(measures)), np.nan)
    conf_p = np.full((len(conds), len(measures)), np.nan)
    acc_shift = np.zeros(len(conds))
    acc_err = np.zeros(len(conds))
    n_per_cond = np.zeros(len(conds), dtype=int)
    for i, cond in enumerate(conds):
        a_mean, a_err, _, n = _paired_delta(df, args.ref, cond, "correct")
        acc_shift[i] = a_mean
        acc_err[i] = a_err
        n_per_cond[i] = n
        for j, m in enumerate(measures):
            c_mean, c_err, p, _ = _paired_delta(df, args.ref, cond, m)
            conf_shift[i, j] = c_mean
            conf_err[i, j] = c_err
            conf_p[i, j] = p
            gap_matrix[i, j] = c_mean - a_mean

    fig = plt.figure(figsize=(15, 11), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.1, 1.0])

    # Top panel: per-condition grouped bar with all channels + accuracy reference line
    ax = fig.add_subplot(gs[0])
    n_chan = len(measures)
    bar_w = 0.13
    centers = np.arange(len(conds))
    palette = {
        "log_prob": "#d62728", "verbal_cat": "#e07b7b", "verbal_num": "#f4a6a6",
        "caa_proj": "#1f77b4", "p_true": "#7eb8e8",
    }
    for j, m in enumerate(measures):
        offset = (j - (n_chan - 1) / 2) * bar_w
        xs = centers + offset
        ys = conf_shift[:, j]
        es = conf_err[:, j]
        ax.bar(xs, ys, width=bar_w, yerr=es, color=palette.get(m, "#888"),
               edgecolor="black", linewidth=0.5, capsize=2,
               label=f"{NICE[m]} ({CHANNEL_FAMILY[m]})")
        for x, y, p in zip(xs, ys, conf_p[:, j]):
            s = _star(p)
            if s:
                va = "bottom" if y >= 0 else "top"
                yoff = 0.005 if y >= 0 else -0.005
                ax.text(x, y + yoff, s, ha="center", va=va, fontsize=6)
    # Plot accuracy shift as a black diamond marker overlaid at each condition's center
    ax.errorbar(centers, acc_shift, yerr=acc_err, fmt="D", color="black",
                markersize=8, capsize=4, label="accuracy Δ", zorder=10)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(centers)
    ax.set_xticklabels([COND_SHORT[c] for c in conds], rotation=15)
    ax.set_ylabel(f"paired Δ vs {args.ref}")
    ax.set_title(
        f"Per-condition shifts (paired Δ vs {args.ref}, n≈{int(np.median(n_per_cond))} items). "
        "Black diamonds = accuracy shift; bars = confidence shifts by channel."
    )
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)

    # Bottom panel: heatmap of inflation gap (Δconf − Δacc)
    ax2 = fig.add_subplot(gs[1])
    vmax = max(0.5, float(np.nanmax(np.abs(gap_matrix))))
    im = ax2.imshow(gap_matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax2.set_xticks(range(len(measures)))
    ax2.set_xticklabels([f"{NICE[m]}\n({CHANNEL_FAMILY[m]})" for m in measures])
    ax2.set_yticks(range(len(conds)))
    ax2.set_yticklabels([COND_SHORT[c] for c in conds])
    for i in range(len(conds)):
        for j in range(len(measures)):
            v = gap_matrix[i, j]
            txt = f"{v:+.2f}"
            color = "white" if abs(v) > vmax * 0.6 else "black"
            ax2.text(j, i, txt, ha="center", va="center", color=color, fontsize=10)
    cbar = fig.colorbar(im, ax=ax2, fraction=0.025, pad=0.02)
    cbar.set_label("Δconf − Δacc  (positive = overconfidence)")
    ax2.set_title(
        "Inflation gap: how much confidence inflates above what accuracy says.\n"
        "Surface channels (red family) inflate under cue_familiarity & illusory_tot; "
        "all channels inflate under partial_accessibility."
    )

    fig.suptitle(
        "Llama-3.1-8B · 461 items · extended Koriat condition matrix (8 conditions, 5 channels)",
        fontsize=13,
    )
    out = rd / "plots" / "extended_summary.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
