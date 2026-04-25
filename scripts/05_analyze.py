"""Stage 5: analyze trials.parquet and write summary tables."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from koriat_cues.analysis import (
    compare_measures,
    compute_shifts,
    dissociation_index,
    per_condition_summary,
    run_mixed_regression,
    run_regression,
)
from koriat_cues.analysis.regression import regression_summary
from koriat_cues.config import load_config
from koriat_cues.confidence.measures import standardize_per_model


MEASURE_COLS = ["log_prob", "verbal_cat", "verbal_num", "caa_proj"]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    df = pd.read_parquet(cfg.run_dir / "trials.parquet")
    enabled = [m for m in MEASURE_COLS if m in cfg.confidence_measures]

    df = standardize_per_model(df, enabled)
    shifts = compute_shifts(df, enabled)
    summary = per_condition_summary(shifts, enabled)
    summary.to_csv(cfg.run_dir / "summary_per_condition.csv", index=False)

    di = dissociation_index(shifts, enabled)
    di.to_csv(cfg.run_dir / "dissociation_index.csv", index=False)

    cmp = compare_measures(shifts, enabled)
    cmp.to_csv(cfg.run_dir / "compare_measures.csv", index=False)

    # Cluster-robust OLS — our primary fit, robust to few items.
    regs = run_regression(shifts, enabled)
    with open(cfg.run_dir / "regression_clustered.txt", "w") as f:
        f.write("# OLS with cluster-robust SEs on item_id.\n\n")
        for mdl, by_measure in regs.items():
            for measure, fit in by_measure.items():
                f.write(f"=== {mdl} / {measure} ===\n")
                f.write(str(regression_summary(fit)))
                f.write("\n\n")

    # MixedLM with item random intercept — richer model, reported as supplement.
    mixed = run_mixed_regression(shifts, enabled)
    with open(cfg.run_dir / "regression_mixed.txt", "w") as f:
        f.write("# MixedLM with random item intercept. "
                "Fits that failed to converge are reported as NO_FIT.\n\n")
        for mdl, by_measure in mixed.items():
            for measure, fit in by_measure.items():
                f.write(f"=== {mdl} / {measure} ===\n")
                if fit is None:
                    f.write("NO_FIT\n\n")
                    continue
                f.write(str(regression_summary(fit)))
                f.write(f"\n(random-effect variance: {float(fit.cov_re.iloc[0,0]):.4g})\n\n")

    print("[05] per-condition summary:")
    print(summary.to_string(index=False))
    print("\n[05] dissociation metrics (cue_familiarity_priming):")
    print(di.to_string(index=False))
    print(f"\n[05] wrote tables to {cfg.run_dir}")


if __name__ == "__main__":
    main()
