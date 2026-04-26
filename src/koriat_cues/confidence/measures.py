"""Collect all four confidence measures for a single trial.

A `trial` is one (model, item, condition) combination. We run three model calls:
  1. Answer the question (captures text, log-prob of first answer token,
     post-answer-newline hidden state for CAA projection).
  2. Elicit a verbalized categorical rating ("very low" ... "very high").
  3. Elicit a verbalized numeric rating (0-100).

Standardization happens at analysis time: per-model min-max rescale each
measure to [0, 1] so the four are on comparable axes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, Optional

import numpy as np
import pandas as pd

from ..caa.vector import project

if TYPE_CHECKING:  # pragma: no cover
    from ..models.hf_model import HFModel


VERBAL_CAT_SCALE: dict[str, float] = {
    "very low": 0.0,
    "low": 0.25,
    "medium": 0.5,
    "high": 0.75,
    "very high": 1.0,
}

# Fallback mapping for hedge-words the model uses when it ignores the 5-point rubric.
# Lowercase, matched as whole words. Order doesn't matter; we pick the longest match.
_HEDGE_FALLBACK: dict[str, float] = {
    "certain": 1.0, "definitely": 1.0, "sure": 1.0, "confident": 0.75,
    "pretty confident": 0.75, "fairly confident": 0.75, "fairly sure": 0.75,
    "mostly": 0.75, "probably": 0.5, "likely": 0.5, "somewhat": 0.5,
    "moderate": 0.5, "moderately": 0.5, "unsure": 0.25, "uncertain": 0.25,
    "doubtful": 0.25, "guess": 0.25, "guessing": 0.25,
    "no idea": 0.0, "not a clue": 0.0, "not confident": 0.0,
}


@dataclass
class TrialOutputs:
    model_name: str
    item_id: str
    condition: str
    question: str
    prime: Optional[str]
    gold_answers: list[str]
    prediction: str = ""
    correct: Optional[bool] = None
    log_prob: Optional[float] = None
    verbal_cat: Optional[float] = None
    verbal_cat_raw: Optional[str] = None
    verbal_num: Optional[float] = None
    verbal_num_raw: Optional[str] = None
    caa_proj: Optional[float] = None
    p_true: Optional[float] = None
    p_true_raw: Optional[str] = None
    extra: dict = field(default_factory=dict)


def _parse_categorical(text: str) -> tuple[Optional[float], Optional[str]]:
    """Parse a categorical rating from free-form text.

    Primary: exact scale labels ("very low" ... "very high"). Fallback: hedge words.
    """
    norm = text.lower().strip()
    for key in sorted(VERBAL_CAT_SCALE.keys(), key=len, reverse=True):
        if re.search(rf"\b{re.escape(key)}\b", norm):
            return VERBAL_CAT_SCALE[key], key
    for key in sorted(_HEDGE_FALLBACK.keys(), key=len, reverse=True):
        if re.search(rf"\b{re.escape(key)}\b", norm):
            return _HEDGE_FALLBACK[key], key
    return None, None


def _parse_numeric(text: str) -> tuple[Optional[float], Optional[str]]:
    m = re.search(r"(-?\d{1,3}(?:\.\d+)?)", text)
    if not m:
        return None, None
    val = float(m.group(1))
    val = max(0.0, min(100.0, val))
    return val / 100.0, m.group(1)


def _eval_correct(prediction: str, gold: Iterable[str]) -> bool:
    from ..eval.grader import grade_prediction
    return grade_prediction(prediction, list(gold))


def run_trial(
    model: "HFModel",
    messages: list[dict],
    item_id: str,
    condition: str,
    gold_answers: list[str],
    question: str,
    prime: Optional[str],
    steering_vector: Optional[np.ndarray],
    measures: Iterable[str],
    max_new_tokens: int = 32,
) -> TrialOutputs:
    """Run the full trial: answer + (optional) verbal cat + (optional) verbal num.

    `steering_vector` is required if "caa_proj" is in `measures`.
    """
    out = TrialOutputs(
        model_name=model.cfg.name,
        item_id=item_id,
        condition=condition,
        question=question,
        prime=prime,
        gold_answers=list(gold_answers),
    )

    measures = set(measures)
    need_hs = "caa_proj" in measures
    prompt = model.format_chat(messages)
    gen = model.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        capture_post_newline=need_hs,
    )

    # Strip "Answer:" artifacts and take the first line.
    pred = gen.text.split("\n")[0].strip()
    pred = re.sub(r"^answer\s*[:\-]\s*", "", pred, flags=re.IGNORECASE).strip()
    out.prediction = pred
    out.correct = _eval_correct(pred, gold_answers)

    if "log_prob" in measures:
        out.log_prob = gen.first_token_logprob

    if need_hs and gen.post_newline_hidden is not None and steering_vector is not None:
        out.caa_proj = project(gen.post_newline_hidden.numpy(), steering_vector)

    if "verbal_cat" in measures:
        out.verbal_cat, out.verbal_cat_raw = _verbalize(
            model, messages, pred, categorical=True, max_new=24
        )
    if "verbal_num" in measures:
        out.verbal_num, out.verbal_num_raw = _verbalize(
            model, messages, pred, categorical=False, max_new=16
        )
    if "p_true" in measures:
        out.p_true, out.p_true_raw = _elicit_p_true(model, messages, pred)

    return out


_FEWSHOT_CAT = (
    "Examples of valid responses:\n"
    "- If you are very confident: very high\n"
    "- If you are fairly confident: high\n"
    "- If unsure: medium\n"
    "- If you're guessing: low\n"
    "- If you have no idea: very low\n"
)

_FEWSHOT_NUM = (
    "Examples:\n"
    "- If you are very confident: 95\n"
    "- If fairly confident: 75\n"
    "- If unsure: 50\n"
    "- If guessing: 25\n"
    "- If no idea: 5\n"
)


def _verbalize(
    model: "HFModel",
    trial_messages: list[dict],
    prediction: str,
    categorical: bool,
    max_new: int,
) -> tuple[Optional[float], Optional[str]]:
    """Elicit a confidence rating by continuing the trial chat.

    Returns (parsed_value, raw_model_output). The raw output is always captured so
    that parse failures are diagnosable. Parsed value is None when both the primary
    and fallback parsers fail.
    """
    if categorical:
        rubric = (
            "Rate your confidence that your answer is correct on this five-point scale: "
            "very low, low, medium, high, very high.\n"
            f"{_FEWSHOT_CAT}"
            "Reply with exactly one rating from the scale, on a single line. "
            "Do not add any other words."
        )
    else:
        rubric = (
            "Rate your confidence that your answer is correct as an integer from 0 to 100, "
            "where 0 means no confidence and 100 means complete certainty.\n"
            f"{_FEWSHOT_NUM}"
            "Reply with exactly one integer, on a single line. Do not add any other words."
        )
    follow = trial_messages + [
        {"role": "assistant", "content": prediction},
        {"role": "user", "content": rubric},
    ]
    prompt = model.format_chat(follow)
    gen = model.generate(prompt, max_new_tokens=max_new, do_sample=False, capture_post_newline=False)
    raw = gen.text.strip()
    if categorical:
        parsed, _ = _parse_categorical(raw)
    else:
        parsed, _ = _parse_numeric(raw)
    # Always return the raw model output, not the matched phrase, so failures can be audited.
    return parsed, raw


_P_TRUE_RUBRIC = (
    "Is the answer above correct? "
    "Reply with exactly one word: True or False. "
    "Do not add any other words."
)


def _p_true_from_logprobs(lp_true: float, lp_false: float) -> float:
    """Normalize the two raw log-probs into P(True) ∈ [0, 1] via softmax over {True, False}."""
    if not np.isfinite(lp_true) and not np.isfinite(lp_false):
        return float("nan")
    if not np.isfinite(lp_true):
        return 0.0
    if not np.isfinite(lp_false):
        return 1.0
    m = max(lp_true, lp_false)
    e_t = float(np.exp(lp_true - m))
    e_f = float(np.exp(lp_false - m))
    return e_t / (e_t + e_f)


def _elicit_p_true(
    model: "HFModel",
    trial_messages: list[dict],
    prediction: str,
) -> tuple[Optional[float], Optional[str]]:
    """Kadavath-style P(True) elicitation.

    Asks the model whether its own answer is correct as a single-word True/False
    question and reads the log-probs of those two tokens directly. Returns
    (P(True) ∈ [0, 1], raw_string_with_log_probs_for_diagnostics).

    This is a *second-order* confidence signal — distinct from the answer's own
    first-token log-prob (`log_prob`) and from the freely-generated verbal rating.
    """
    follow = trial_messages + [
        {"role": "assistant", "content": prediction},
        {"role": "user", "content": _P_TRUE_RUBRIC},
    ]
    prompt = model.format_chat(follow)
    scores = model.score_next_token_words(prompt, ["True", "False"])
    lp_true, lp_false = scores.get("True", float("nan")), scores.get("False", float("nan"))
    if not np.isfinite(lp_true) and not np.isfinite(lp_false):
        return None, "tokenization_failed"
    p = _p_true_from_logprobs(lp_true, lp_false)
    raw = f"lp_true={lp_true:.3f} lp_false={lp_false:.3f}"
    return p, raw


def standardize_per_model(df: pd.DataFrame, measure_cols: Iterable[str]) -> pd.DataFrame:
    """Min-max standardize each measure to [0, 1] within each model.

    Missing values are preserved. Columns that are already in [0, 1] (verbal_*) are
    unchanged.
    """
    df = df.copy()
    for col in measure_cols:
        if col not in df.columns:
            continue
        for name, sub in df.groupby("model_name"):
            vals = sub[col].astype(float)
            if vals.notna().sum() < 2:
                continue
            lo, hi = float(vals.min()), float(vals.max())
            if hi - lo < 1e-9:
                df.loc[sub.index, col] = 0.5
            else:
                df.loc[sub.index, col] = (vals - lo) / (hi - lo)
    return df
