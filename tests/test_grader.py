from koriat_cues.eval.grader import grade_prediction, normalize, normalized_match


def test_normalize_strips_articles_and_case():
    assert normalize("The JANE Austen!") == "jane austen"


def test_grade_exact_match():
    assert grade_prediction("Jane Austen", ["Jane Austen"]) is True


def test_grade_uses_aliases():
    assert grade_prediction("Laennec", ["René Laennec", "Laennec"]) is True


def test_grade_substring_only_for_multi_token_gold():
    assert grade_prediction("Sir Isaac Newton", ["Isaac Newton"]) is True
    # Single-token gold should not match a substring inside a longer prediction.
    assert grade_prediction("snowball", ["no"]) is False


def test_grade_picks_first_line():
    pred = "Jane Austen\nActually wait, maybe someone else"
    assert grade_prediction(pred, ["Jane Austen"]) is True


def test_normalized_match_handles_trailing_punctuation():
    # SQuAD-style normalization maps punctuation to spaces; trailing period is harmless.
    assert normalized_match("Jane Austen.", "Jane Austen") is True
