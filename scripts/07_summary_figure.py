"""Single clean figure summarizing the headline result with the v2 CAA vector."""
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
    "verbal_cat": "verbal categorical",
    "verbal_num": "verbal numeric",
    "caa_proj": "CAA projection (v2)",
    "p_true": "P(True) self-elicit",
}
COND_COLOR = {
    "whitespace": "#9467bd",
    "random_paragraph": "#1f77b4",
    "cue_familiarity_priming": "#d62728",
    "target_priming": "#2ca02c",
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
    return float(d.mean()), h, float(st.ttest_1samp(d, 0).pvalue), len(d)


def _star(p):
    if p < 1e-3: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return "n.s."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    rd = Path(args.run_dir)

    df = pd.read_parquet(rd / "trials.parquet")
    df = df.drop_duplicates(subset=["model_name", "item_id", "condition"], keep="first").reset_index(drop=True)
    measures = [m for m in _ALL_MEASURES if m in df.columns and df[m].notna().any()]
    n_fields = 1 + len(measures)  # accuracy + measures

    fig = plt.figure(figsize=(13, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1])

    cond_order = ["whitespace", "random_paragraph", "cue_familiarity_priming", "target_priming"]
    cond_short = {"whitespace": "whitespace",
                  "random_paragraph": "rand_para",
                  "cue_familiarity_priming": "cue_familiarity",
                  "target_priming": "target"}

    for row, (ref, ref_label) in enumerate([
        ("baseline", "Δ vs unmodified baseline (n=461)"),
        ("whitespace", "Δ vs whitespace control (the cleanest neutral; n=461)"),
    ]):
        ax = fig.add_subplot(gs[row])
        x_positions = []
        x_labels = []
        means, errs, colors, sigs = [], [], [], []
        cond_loop = ["cue_familiarity_priming", "target_priming"] if ref == "random_paragraph" else cond_order
        bar_w = min(0.18, 0.85 / max(n_fields, 1))
        group_centers = list(range(len(cond_loop)))
        center_idx = (n_fields - 1) / 2.0
        for gi, cond in enumerate(cond_loop):
            for j, (col, label) in enumerate([("correct", "accuracy")] + [(m, NICE[m]) for m in measures]):
                d_mean, d_err, p, n = _paired_delta(df, ref, cond, col)
                offset = (j - center_idx) * bar_w
                means.append(d_mean)
                errs.append(d_err)
                colors.append(COND_COLOR[cond])
                sigs.append(_star(p))
                x_positions.append(gi + offset)
                if gi == 0:
                    x_labels.append(label)
        ax.bar(x_positions, means, width=bar_w, yerr=errs,
               color=colors, edgecolor="black", capsize=3)
        for x, m, s in zip(x_positions, means, sigs):
            ax.text(x, m + (0.01 if m >= 0 else -0.01),
                    s, ha="center", va="bottom" if m >= 0 else "top", fontsize=7)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(group_centers)
        ax.set_xticklabels([cond_short[c] for c in cond_loop])
        ax.set_ylabel("paired Δ")
        ax.set_title(ref_label, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        # Manual legend
        if row == 0:
            from matplotlib.patches import Patch
            order_label = "(left→right): accuracy, " + ", ".join(measures)
            handles = [
                Patch(facecolor="#888", edgecolor="black", label="accuracy"),
                Patch(facecolor="#bbb", edgecolor="black", label=order_label),
            ]
            ax.legend(handles=handles, loc="upper left", fontsize=8)

    fig.suptitle(
        "Llama-3.1-8B · 461 items · v2 validated CAA (layer 31, AUC 0.77 on held-out)\n"
        "Top: Δ vs unmodified baseline. Bottom: Δ vs random_paragraph yoked control (the proper test).",
        fontsize=12,
    )
    out = rd / "plots" / "summary_headline.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
