"""Tests for pre-registered headline criteria (koriat_cues.analysis.prereg)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from koriat_cues.analysis.prereg import evaluate_model_prereg, paired_stats


def _synth_item(
    item_id: str,
    bl_lp: float,
    bl_vc: float,
    bl_pt: float,
    bl_ok: bool,
    cue_lp: float,
    cue_vc: float,
    cue_pt: float,
    cue_ok: bool,
    part_lp: float,
    part_vc: float,
    part_pt: float,
    part_ok: bool,
    tgt_ok: bool,
) -> list[dict]:
    """One item: baseline + cue_fam + partial + target (+ whitespace minimal)."""
    rows = [
        dict(
            model_name="synth",
            item_id=item_id,
            condition="baseline",
            log_prob=bl_lp,
            verbal_cat=bl_vc,
            p_true=bl_pt,
            correct=bl_ok,
        ),
        dict(
            model_name="synth",
            item_id=item_id,
            condition="cue_familiarity_priming",
            log_prob=cue_lp,
            verbal_cat=cue_vc,
            p_true=cue_pt,
            correct=cue_ok,
        ),
        dict(
            model_name="synth",
            item_id=item_id,
            condition="partial_accessibility",
            log_prob=part_lp,
            verbal_cat=part_vc,
            p_true=part_pt,
            correct=part_ok,
        ),
        dict(
            model_name="synth",
            item_id=item_id,
            condition="target_priming",
            log_prob=bl_lp + 0.01,
            verbal_cat=bl_vc + 0.01,
            p_true=bl_pt + 0.01,
            correct=tgt_ok,
        ),
    ]
    return rows


def test_paired_stats_two_point():
    d = pd.Series([1.0, -1.0])
    mu, p, n = paired_stats(d)
    assert n == 2
    assert abs(mu) < 1e-9
    assert p > 0.05


def test_synthetic_passes_headline_criteria():
    rng = np.random.default_rng(0)
    rows: list[dict] = []
    for i in range(40):
        bl_lp = float(rng.uniform(0.2, 0.5))
        bl_vc = float(rng.uniform(0.2, 0.5))
        bl_pt = float(rng.uniform(0.35, 0.55))
        # Half the items: baseline correct → partial wrong (C2 acc ↓); half: baseline
        # wrong → target right (C3 acc ↑). Same item cannot do both on accuracy.
        bl_ok = i < 20
        rows.extend(
            _synth_item(
                f"i{i}",
                bl_lp,
                bl_vc,
                bl_pt,
                bl_ok,
                cue_lp=bl_lp + 0.2,
                cue_vc=bl_vc + 0.2,
                cue_pt=bl_pt,
                cue_ok=bl_ok,
                part_lp=bl_lp + 0.25,
                part_vc=bl_vc + 0.22,
                part_pt=bl_pt,
                part_ok=False if bl_ok else bl_ok,
                tgt_ok=True if not bl_ok else bl_ok,
            )
        )
    df = pd.DataFrame(rows)
    v = evaluate_model_prereg(df)
    assert v.c1.passed, v.c1.detail
    assert v.c2.passed, v.c2.detail
    assert v.c3.passed, v.c3.detail
    assert v.headline_pass


def test_c1_fails_when_log_prob_drops():
    rng = np.random.default_rng(1)
    rows: list[dict] = []
    for i in range(40):
        bl_lp = float(rng.uniform(0.3, 0.6))
        bl_vc = float(rng.uniform(0.3, 0.6))
        bl_pt = 0.4
        bl_ok = True
        rows.extend(
            _synth_item(
                f"i{i}",
                bl_lp,
                bl_vc,
                bl_pt,
                bl_ok,
                cue_lp=bl_lp - 0.1,
                cue_vc=bl_vc + 0.2,
                cue_pt=bl_pt,
                cue_ok=bl_ok,
                part_lp=bl_lp + 0.2,
                part_vc=bl_vc + 0.2,
                part_pt=bl_pt,
                part_ok=False,
                tgt_ok=True,
            )
        )
    df = pd.DataFrame(rows)
    v = evaluate_model_prereg(df)
    assert not v.c1.passed
