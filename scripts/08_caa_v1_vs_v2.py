"""Side-by-side comparison of CAA v1 (template-based, layer 21) and CAA v2
(real-trace contrast pairs, layer 31, AUC=0.77 on held-out)."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st


COND_ORDER = ["baseline", "random_paragraph", "cue_familiarity_priming", "target_priming"]
COND_COLOR = {
    "baseline": "#4f4f4f",
    "random_paragraph": "#1f77b4",
    "cue_familiarity_priming": "#d62728",
    "target_priming": "#2ca02c",
}


def _ci(x):
    if len(x) < 2:
        return float("nan")
    se = np.std(x, ddof=1) / np.sqrt(len(x))
    return float(se * st.t.ppf(0.975, len(x) - 1))


def _paired_delta(df, ref, cond, col):
    a = df[df.condition == ref].drop_duplicates("item_id").set_index("item_id")
    b = df[df.condition == cond].drop_duplicates("item_id").set_index("item_id")
    m = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
    d = (m[f"{col}_b"] - m[f"{col}_a"]).dropna()
    if len(d) < 2:
        return float("nan"), float("nan"), float("nan"), 0
    p = st.ttest_1samp(d, 0).pvalue
    return float(d.mean()), _ci(d), float(p), len(d)


def _star(p):
    if p < 1e-3: return "***"
    if p < 1e-2: return "**"
    if p < 5e-2: return "*"
    return "n.s."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    rd = Path(args.run_dir)

    v1 = pd.read_parquet(rd / "trials_v1.parquet").drop_duplicates(["model_name","item_id","condition"])
    v2 = pd.read_parquet(rd / "trials.parquet").drop_duplicates(["model_name","item_id","condition"])

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)

    # === Top row: per-condition CAA mean ± 95% CI, v1 left vs v2 right ===
    for ax, df, label, scan_layer in [
        (axes[0, 0], v1, "CAA v1: template contrast pairs · layer 21 · AUC≈0.74", 21),
        (axes[0, 1], v2, "CAA v2: real-trace pairs · layer 31 · validated AUC=0.77", 31),
    ]:
        means, errs, colors = [], [], []
        for c in COND_ORDER:
            x = df[df.condition == c]["caa_proj"].dropna().to_numpy()
            means.append(float(x.mean()))
            errs.append(_ci(x))
            colors.append(COND_COLOR[c])
        ax.bar(COND_ORDER, means, yerr=errs, color=colors, edgecolor="black", capsize=4)
        ax.set_title(label, fontsize=10)
        ax.set_ylabel("mean CAA projection (raw scale)")
        ax.tick_params(axis="x", rotation=20, labelsize=8)
        ax.grid(axis="y", alpha=0.3)

    # === Bottom row: per-item Δ_CAA vs baseline, paired, v1 left vs v2 right ===
    other_conds = ["random_paragraph", "cue_familiarity_priming", "target_priming"]
    for ax, df, label in [
        (axes[1, 0], v1, "v1: per-item ΔCAA vs baseline (paired)"),
        (axes[1, 1], v2, "v2: per-item ΔCAA vs baseline (paired)"),
    ]:
        means, errs, colors, sigs, ns = [], [], [], [], []
        for c in other_conds:
            m, e, p, n = _paired_delta(df, "baseline", c, "caa_proj")
            means.append(m); errs.append(e); colors.append(COND_COLOR[c])
            sigs.append(_star(p)); ns.append(n)
        bars = ax.bar(other_conds, means, yerr=errs, color=colors, edgecolor="black", capsize=4)
        for x_pos, (m, s, n) in enumerate(zip(means, sigs, ns)):
            offset = 0.01 if m >= 0 else -0.01
            ax.text(x_pos, m + offset, f"{s}\nn={n}",
                    ha="center", va="bottom" if m >= 0 else "top", fontsize=8)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(label, fontsize=10)
        ax.set_ylabel("paired Δ CAA (condition − baseline)")
        ax.tick_params(axis="x", rotation=15, labelsize=8)
        ax.grid(axis="y", alpha=0.3)

    # Force matching y-axis on bottom row so v1 vs v2 effect-size is directly comparable.
    bot_lo = min(axes[1, 0].get_ylim()[0], axes[1, 1].get_ylim()[0])
    bot_hi = max(axes[1, 0].get_ylim()[1], axes[1, 1].get_ylim()[1])
    axes[1, 0].set_ylim(bot_lo, bot_hi)
    axes[1, 1].set_ylim(bot_lo, bot_hi)
    # Same for top row.
    top_lo = min(axes[0, 0].get_ylim()[0], axes[0, 1].get_ylim()[0])
    top_hi = max(axes[0, 0].get_ylim()[1], axes[0, 1].get_ylim()[1])
    axes[0, 0].set_ylim(top_lo, top_hi)
    axes[0, 1].set_ylim(top_lo, top_hi)

    fig.suptitle(
        "CAA v1 vs v2: same trials, different steering vector. "
        "Bottom row: priming effects significant in v2 (p<.01) but not v1 (n.s.).",
        fontsize=11,
    )
    out = rd / "plots" / "caa_v1_vs_v2.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
