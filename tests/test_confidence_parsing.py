import math

from koriat_cues.confidence.measures import (
    _parse_categorical,
    _parse_numeric,
    _p_true_from_logprobs,
    VERBAL_CAT_SCALE,
)


def test_parse_categorical_prefers_longest_match():
    val, raw = _parse_categorical("I'd say very high.")
    assert raw == "very high"
    assert val == VERBAL_CAT_SCALE["very high"]


def test_parse_categorical_medium():
    val, _ = _parse_categorical("Medium")
    assert val == VERBAL_CAT_SCALE["medium"]


def test_parse_categorical_low_not_very_low():
    val, raw = _parse_categorical("Low")
    assert raw == "low"
    assert val == VERBAL_CAT_SCALE["low"]


def test_parse_numeric_clamps():
    val, _ = _parse_numeric("My confidence is 350%")
    assert val == 1.0


def test_parse_numeric_extracts_integer():
    val, raw = _parse_numeric("about 72 out of 100")
    assert raw == "72"
    assert abs(val - 0.72) < 1e-9


def test_parse_numeric_rejects_when_no_digits():
    val, raw = _parse_numeric("unsure")
    assert val is None
    assert raw is None


def test_hedge_fallback_when_scale_label_missing():
    # Model ignored the rubric and hedged instead — fallback should still return a bucket.
    val, raw = _parse_categorical("I'm pretty confident about this.")
    assert val is not None
    assert raw == "pretty confident"


def test_hedge_fallback_no_idea():
    val, _ = _parse_categorical("no idea, really")
    assert val == 0.0


def test_hedge_fallback_prefers_primary_scale():
    # Scale label present → primary parser wins over hedge fallback.
    val, raw = _parse_categorical("I'd say very high — very confident actually.")
    assert raw == "very high"


def test_p_true_softmax_equal_logprobs_gives_half():
    p = _p_true_from_logprobs(-1.5, -1.5)
    assert abs(p - 0.5) < 1e-9


def test_p_true_softmax_higher_true_logprob_gives_above_half():
    # lp_true = -0.1, lp_false = -2.3 → softmax favors True.
    p = _p_true_from_logprobs(-0.1, -2.3)
    assert 0.5 < p < 1.0


def test_p_true_softmax_matches_manual_softmax():
    lp_t, lp_f = -0.5, -1.8
    expected = math.exp(lp_t) / (math.exp(lp_t) + math.exp(lp_f))
    p = _p_true_from_logprobs(lp_t, lp_f)
    assert abs(p - expected) < 1e-6


def test_p_true_softmax_clamps_when_one_logprob_is_inf():
    # If False is unreachable, P(True) should pin to 1.0; vice versa to 0.0.
    assert _p_true_from_logprobs(-0.5, float("-inf")) == 1.0
    assert _p_true_from_logprobs(float("-inf"), -0.5) == 0.0
