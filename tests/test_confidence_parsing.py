from koriat_cues.confidence.measures import _parse_categorical, _parse_numeric, VERBAL_CAT_SCALE


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
