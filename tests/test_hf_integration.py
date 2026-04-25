"""End-to-end integration test with a tiny HF model.

Skipped by default — set KORIAT_RUN_HF_TEST=1 to run. Downloads ~5MB.
Catches schema bugs in the full trial → parquet → analysis path that synthetic
tests miss (e.g., the empty-struct parquet bug hit during the pilot).
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("KORIAT_RUN_HF_TEST") != "1",
    reason="set KORIAT_RUN_HF_TEST=1 to run HF-backed integration tests",
)


def test_small_model_generation_and_parquet_roundtrip():
    """Uses sshleifer/tiny-gpt2 (no chat template) as a minimal stand-in."""
    import pandas as pd

    from koriat_cues.config import ModelConfig
    from koriat_cues.models.hf_model import HFModel
    from koriat_cues.confidence.measures import run_trial
    from koriat_cues.experiment.runner import _to_row

    cfg = ModelConfig(
        name="tiny",
        hf_id="sshleifer/tiny-gpt2",
        dtype="float32",
        device_map="cpu",
        max_new_tokens=4,
        caa_layer=-1,
    )
    model = HFModel(cfg)

    # tiny-gpt2 has no chat template; assemble a plain prompt manually.
    messages = [{"role": "user", "content": "Question: What is 2+2?\nAnswer:"}]
    # For this tiny model we can't realistically format via chat template, so call
    # `generate` on the raw text — that still exercises the hidden-state path.
    gen = model.generate("Question: What is 2+2?\nAnswer:", max_new_tokens=4, capture_post_newline=True)
    assert isinstance(gen.text, str)
    assert isinstance(gen.first_token_logprob, float)

    # Build a fake TrialOutputs and check it round-trips through parquet.
    from koriat_cues.confidence.measures import TrialOutputs

    t = TrialOutputs(
        model_name="tiny",
        item_id="x",
        condition="baseline",
        question="Q?",
        prime=None,
        gold_answers=["4"],
        prediction="4",
        correct=True,
        log_prob=-0.5,
        verbal_cat=0.75,
        verbal_cat_raw="high",
        verbal_num=0.8,
        verbal_num_raw="80",
        caa_proj=0.1,
    )
    row = _to_row(t)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "trials.parquet"
        pd.DataFrame([row]).to_parquet(path, index=False)
        back = pd.read_parquet(path)
    assert len(back) == 1
    assert back.iloc[0]["correct"] == True
    assert back.iloc[0]["extra"] == "{}"  # JSON-serialized empty dict
    # gold_answers round-tripped as a list.
    assert list(back.iloc[0]["gold_answers"]) == ["4"]
