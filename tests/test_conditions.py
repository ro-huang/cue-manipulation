from koriat_cues.primes.conditions import CONDITION_SPECS, assemble_prompt


def test_all_expected_conditions_present():
    expected = {
        "baseline",
        "cue_familiarity_priming",
        "target_priming",
        "partial_accessibility",
        "illusory_tot",
        "fluency_degradation",
        "random_paragraph",
    }
    assert set(CONDITION_SPECS.keys()) == expected


def test_baseline_omits_prime_from_prompt():
    msgs = assemble_prompt("Who invented the stethoscope?", prime=None, condition="baseline")
    user = msgs[-1]["content"]
    assert "Context" not in user
    assert "stethoscope" in user


def test_prime_goes_before_question_by_default():
    msgs = assemble_prompt(
        "Who invented the stethoscope?",
        prime="Medical instruments are old.",
        condition="cue_familiarity_priming",
    )
    user = msgs[-1]["content"]
    # Context comes first, question second.
    assert user.index("Context:") < user.index("Question:")


def test_counterbalance_order_flips_prime_position():
    msgs = assemble_prompt(
        "Who invented the stethoscope?",
        prime="Medical instruments are old.",
        condition="cue_familiarity_priming",
        order="question_then_prime",
    )
    user = msgs[-1]["content"]
    assert user.index("Question:") < user.index("Context:")


def test_fluency_degradation_uses_prime_as_question():
    msgs = assemble_prompt(
        "Who invented the stethoscope?",
        prime="By whom was the auscultatory device conceived?",
        condition="fluency_degradation",
    )
    user = msgs[-1]["content"]
    assert "auscultatory" in user
    assert "stethoscope" not in user
