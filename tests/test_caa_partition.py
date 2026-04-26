"""Tests for the partition logic in scripts/01_build_caa_vector.py.

We import `evaluate_layers` directly and feed it synthetic activations + labels.
The fixture is deterministic: confident-class items get a +signal along axis 0,
unsure-class items get a -signal, and there's i.i.d. noise on the rest.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "01_build_caa_vector.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("build_caa_vector", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("build_caa_vector", mod)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def synthetic_activations():
    """Two classes that differ along axis 0 only, plus a separate val split.

    By construction, the mean-difference vector should align with axis 0.
    """
    rng = np.random.default_rng(0)
    n_train, n_val, hidden = 80, 60, 16
    n_layers = 3

    # Train: half "confident" (class label True / +signal), half "unsure" (False / -signal).
    H_train = rng.normal(0, 1, size=(n_train, n_layers, hidden)).astype(np.float32)
    correct_train = np.zeros(n_train, dtype=bool)
    correct_train[:n_train // 2] = True
    # Confident items get +2 on axis 0 at layer 1 (the "best" layer).
    H_train[correct_train, 1, 0] += 2.0
    H_train[~correct_train, 1, 0] -= 2.0
    # log-prob is correlated but noisy.
    lp_train = np.where(correct_train, 0.0, -2.0) + rng.normal(0, 1.0, size=n_train).astype(np.float32)

    # Validation: same generative process, independent draws.
    H_val = rng.normal(0, 1, size=(n_val, n_layers, hidden)).astype(np.float32)
    correct_val = np.zeros(n_val, dtype=bool)
    correct_val[:n_val // 2] = True
    H_val[correct_val, 1, 0] += 2.0
    H_val[~correct_val, 1, 0] -= 2.0
    lp_val = np.where(correct_val, 0.0, -2.0) + rng.normal(0, 1.0, size=n_val).astype(np.float32)

    return dict(
        H_train=H_train, lp_train=lp_train, correct_train=correct_train,
        H_val=H_val, lp_val=lp_val, correct_val=correct_val,
    )


def test_correctness_partition_picks_best_layer(synthetic_activations):
    mod = _load_script()
    df, vectors = mod.evaluate_layers(
        synthetic_activations["H_train"],
        synthetic_activations["lp_train"],
        synthetic_activations["correct_train"],
        synthetic_activations["H_val"],
        synthetic_activations["lp_val"],
        synthetic_activations["correct_val"],
        partition="correctness", top_q=0.25, bot_q=0.25,
    )
    best = df.sort_values("auc_correct", ascending=False).iloc[0]
    # Layer 1 carries the synthetic signal; AUC should be high there.
    assert int(best["layer"]) == 1
    assert best["auc_correct"] > 0.85
    # Vector at layer 1 should have its largest component on axis 0.
    v = vectors[1]
    assert int(np.argmax(np.abs(v))) == 0


def test_logprob_partition_still_works(synthetic_activations):
    """Sanity: the legacy logprob partition runs and produces a reasonable AUC."""
    mod = _load_script()
    df, _ = mod.evaluate_layers(
        synthetic_activations["H_train"],
        synthetic_activations["lp_train"],
        synthetic_activations["correct_train"],
        synthetic_activations["H_val"],
        synthetic_activations["lp_val"],
        synthetic_activations["correct_val"],
        partition="logprob", top_q=0.25, bot_q=0.25,
    )
    best = df.sort_values("auc_correct", ascending=False).iloc[0]
    # log-prob and correctness are correlated in the fixture, so layer 1 should still win.
    assert int(best["layer"]) == 1
    assert best["auc_correct"] > 0.7


def test_unknown_partition_raises(synthetic_activations):
    mod = _load_script()
    with pytest.raises(ValueError):
        mod.evaluate_layers(
            synthetic_activations["H_train"],
            synthetic_activations["lp_train"],
            synthetic_activations["correct_train"],
            synthetic_activations["H_val"],
            synthetic_activations["lp_val"],
            synthetic_activations["correct_val"],
            partition="something_else", top_q=0.25, bot_q=0.25,
        )
