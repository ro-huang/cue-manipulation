import pandas as pd

from koriat_cues.analysis import (
    compute_shifts,
    dissociation_index,
    per_condition_summary,
)
from koriat_cues.confidence.measures import standardize_per_model


def _toy_trials() -> pd.DataFrame:
    # 3 items × 2 conditions (baseline + cue_familiarity_priming).
    rows = []
    for i, (bl_corr, cue_corr, bl_conf, cue_conf) in enumerate(
        [
            # baseline_correct, cue_correct, baseline_conf, cue_conf
            (False, False, 0.4, 0.8),
            (False, True,  0.3, 0.7),
            (True,  False, 0.8, 0.9),
        ]
    ):
        rows.append(
            dict(
                model_name="m", item_id=f"i{i}", condition="baseline",
                correct=bl_corr, verbal_cat=bl_conf, caa_proj=bl_conf,
            )
        )
        rows.append(
            dict(
                model_name="m", item_id=f"i{i}", condition="cue_familiarity_priming",
                correct=cue_corr, verbal_cat=cue_conf, caa_proj=cue_conf,
            )
        )
    return pd.DataFrame(rows)


def test_shift_has_one_row_per_non_baseline_trial():
    df = _toy_trials()
    shifts = compute_shifts(df, ["verbal_cat", "caa_proj"])
    assert len(shifts) == 3
    assert set(shifts["condition"]) == {"cue_familiarity_priming"}


def test_accuracy_shift_signs():
    df = _toy_trials()
    shifts = compute_shifts(df, ["verbal_cat"])
    # Item 0: both wrong → 0. Item 1: wrong→right → +1. Item 2: right→wrong → -1.
    assert sorted(shifts["accuracy_shift"].tolist()) == [-1, 0, 1]


def test_dissociation_metrics_reflect_confidence_gain():
    df = _toy_trials()
    shifts = compute_shifts(df, ["verbal_cat"])
    di = dissociation_index(shifts, ["verbal_cat"])
    import numpy as np
    row = di.iloc[0]
    # Mean accuracy shift is 0 → the ratio is NaN by definition.
    assert np.isnan(row["DI_ratio_verbal_cat"])
    assert np.isnan(row["DI_ratio_ci_lo_verbal_cat"])
    # But the DIFFERENCE is always defined and captures the Koriat effect.
    assert row["diff_verbal_cat"] > 0.25
    # Mean confidence shift unchanged.
    assert row["mean_confidence_shift_verbal_cat"] > 0.25


def test_dissociation_ratio_when_accuracy_moves():
    # Construct a dataset where acc shifts are non-zero and check the ratio is
    # populated with a sensible bootstrap CI.
    import pandas as pd
    rows = []
    # All 10 items: +0.3 confidence, +0.1 accuracy → ratio ≈ 3.
    for i in range(10):
        rows.append(dict(model_name="m", item_id=f"i{i}", condition="baseline",
                         correct=False, verbal_cat=0.4))
        rows.append(dict(model_name="m", item_id=f"i{i}", condition="cue_familiarity_priming",
                         correct=(i < 1),  # 1/10 items flip correct
                         verbal_cat=0.7))
    df = pd.DataFrame(rows)
    shifts = compute_shifts(df, ["verbal_cat"])
    di = dissociation_index(shifts, ["verbal_cat"])
    row = di.iloc[0]
    assert abs(row["DI_ratio_verbal_cat"] - 3.0) < 0.5  # 0.3 / 0.1 = 3
    assert row["DI_ratio_ci_lo_verbal_cat"] < row["DI_ratio_ci_hi_verbal_cat"]
    # diff is 0.3 - 0.1 = 0.2.
    assert abs(row["diff_verbal_cat"] - 0.2) < 1e-6


def test_standardize_per_model_bounds():
    df = _toy_trials()
    out = standardize_per_model(df, ["caa_proj"])
    assert out["caa_proj"].min() >= 0.0
    assert out["caa_proj"].max() <= 1.0


def test_per_condition_summary_shapes():
    df = _toy_trials()
    shifts = compute_shifts(df, ["verbal_cat"])
    summary = per_condition_summary(shifts, ["verbal_cat"])
    assert len(summary) == 1
    assert "confidence_shift_verbal_cat__mean" in summary.columns
