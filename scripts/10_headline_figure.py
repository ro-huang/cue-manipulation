"""Headline dissociation figure for the pre-registered behavioral claim.

Reads one or more `runs/<run_name>/trials.parquet` and produces a single figure
showing the locked-design contrast: stated-confidence channels (log_prob,
verbal_cat) inflate under FOK manipulations while the grounded-confidence
channel (p_true) does not, and accuracy drops only under partial_accessibility.

One row per --run-dir. Within a row: paired Δ from baseline for each primary
condition × {log_prob, verbal_cat, p_true, accuracy}, with bootstrap 95% CIs.

Usage:
    python scripts/10_headline_figure.py \\
        --run-dir runs/mve_llama8b \\
        --run-dir runs/replication_qwen7b \\
        --out runs/replication_qwen7b/plots/headline.png

Pre-reg: .context/preregistration.md.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from koriat_cues.confidence.measures import standardize_per_model

PRIMARY_MEASURES = ["log_prob", "verbal_cat", "p_true"]
PRIMARY_CONDITIONS = [
    "cue_familiarity_priming",
    "partial_accessibility",
    "target_priming",
    "random_paragraph",
    "whitespace",
]
COND_SHORT = {
    "cue_familiarity_priming": "cue_fam",
    "partial_accessibility": "partial_acc",
    "target_priming": "target",
    "random_paragraph": "rand_para",
    "whitespace": "whitespace",
}
# Stated channels = warm; grounded = cool; accuracy = neutral.
COLORS = {
    "log_prob": "#d62728",     # red
    "verbal_cat": "#ff7f0e",   # orange
    "p_true": "#1f77b4",       # blue
    "correct": "#7f7f7f",      # grey
}
NICE = {
    "log_prob": "log P(ans)",
    "verbal_cat": "verbal_cat",
    "p_true": "P(True)",
    "correct": "accuracy",
}
BAR_ORDER = ["log_prob", "verbal_cat", "p_true", "correct"]


def _paired_delta(df: pd.DataFrame, cond: str, col: str, n_boot: int = 1000, seed: int = 0):
    """Return (mean Δ, low-CI, high-CI, p, n) for paired primed−baseline on one column.

    `col == "correct"` is treated as a 0/1 accuracy column.
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
    # Two-sided paired t-test against 0.
    se = float(d.std(ddof=1) / np.sqrt(n))
    t = float(d.mean()) / se if se > 0 else float("nan")
    # Use scipy if available; else asymptotic normal.
    try:
        from scipy import stats as _st
        p = float(_st.t.sf(abs(t), n - 1) * 2) if np.isfinite(t) else float("nan")
    except ImportError:
        from math import erfc, sqrt as _sq
        p = erfc(abs(t) / _sq(2)) if np.isfinite(t) else float("nan")
    return float(d.mean()), float(lo), float(hi), p, n


def _star(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 1e-3:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _row(ax: plt.Axes, df: pd.DataFrame, title: str) -> None:
    """Draw one model's row of grouped bars on `ax`."""
    n_groups = len(PRIMARY_CONDITIONS)
    n_bars = len(BAR_ORDER)
    bar_w = 0.8 / n_bars
    x_centers = np.arange(n_groups, dtype=float)

    for j, col in enumerate(BAR_ORDER):
        means, lows, highs, stars = [], [], [], []
        for cond in PRIMARY_CONDITIONS:
            mean, lo, hi, p, _n = _paired_delta(df, cond, col)
            means.append(mean)
            lows.append(mean - lo if np.isfinite(lo) else 0.0)
            highs.append(hi - mean if np.isfinite(hi) else 0.0)
            stars.append(_star(p))
        offset = (j - (n_bars - 1) / 2) * bar_w
        xs = x_centers + offset
        ax.bar(
            xs,
            means,
            width=bar_w,
            yerr=[lows, highs],
            color=COLORS[col],
            edgecolor="black",
            linewidth=0.5,
            capsize=2,
            label=NICE[col],
        )
        for x, m, s in zip(xs, means, stars):
            if not s:
                continue
            ax.text(
                x,
                m + (0.005 if m >= 0 else -0.005),
                s,
                ha="center",
                va="bottom" if m >= 0 else "top",
                fontsize=7,
            )

    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x_centers)
    ax.set_xticklabels([COND_SHORT[c] for c in PRIMARY_CONDITIONS], fontsize=9)
    ax.set_ylabel("paired Δ vs baseline")
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", alpha=0.3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", action="append", required=True, dest="run_dirs",
                    help="One or more run directories; pass multiple times for multi-model figure.")
    ap.add_argument("--out", required=True, help="Output figure path.")
    ap.add_argument("--title-suffix", default="", help="Optional suffix appended to figure title.")
    args = ap.parse_args()

    run_dirs = [Path(d) for d in args.run_dirs]
    rows: list[tuple[str, pd.DataFrame]] = []
    for rd in run_dirs:
        df = pd.read_parquet(rd / "trials.parquet")
        df = df.drop_duplicates(subset=["model_name", "item_id", "condition"], keep="first").reset_index(drop=True)
        # Min-max standardize each confidence measure to [0,1] within model, mirroring
        # the pipeline in 05_analyze.py so cross-measure bar heights are comparable.
        # Accuracy ("correct") is already 0/1 and is left untouched.
        present = [m for m in PRIMARY_MEASURES if m in df.columns and df[m].notna().any()]
        df = standardize_per_model(df, present)
        # Sub-row per (run_dir, model_name) so a single multi-model parquet expands cleanly.
        for model_name, sub in df.groupby("model_name"):
            label = f"{rd.name} · {model_name} (n={sub.item_id.nunique()})"
            rows.append((label, sub))

    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 1, figsize=(11, 3.2 * n_rows), squeeze=False)
    for (title, sub), ax in zip(rows, axes[:, 0]):
        _row(ax, sub, title)

    # Single legend at top.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(BAR_ORDER),
        bbox_to_anchor=(0.5, 1.0 - 0.02 / n_rows),
        frameon=False,
        fontsize=9,
    )
    suptitle = "Stated vs grounded confidence under cue manipulations · paired Δ from baseline"
    if args.title_suffix:
        suptitle += f" · {args.title_suffix}"
    fig.suptitle(suptitle, fontsize=12, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
