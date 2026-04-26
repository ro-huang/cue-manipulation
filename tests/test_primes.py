from koriat_cues.primes.generator import Prime, PrimeSet, yoke_random_paragraphs


def test_yoke_random_paragraphs_matches_length():
    ps = PrimeSet(item_id="x", question="q", gold_answer="a")
    ps.primes["cue_familiarity_priming"] = Prime(
        condition="cue_familiarity_priming",
        text="one two three four five six seven eight nine ten eleven twelve",
    )
    ps.primes["random_paragraph"] = Prime(condition="random_paragraph", text=None)
    yoke_random_paragraphs([ps], seed=0)
    assert ps.primes["random_paragraph"].text is not None
    # Yoked length matches (±0) the longest other prime.
    target = len(ps.primes["cue_familiarity_priming"].text.split())
    yoked = len(ps.primes["random_paragraph"].text.split())
    assert yoked == target


def test_primeset_round_trip():
    ps = PrimeSet(item_id="x", question="q", gold_answer="a")
    ps.primes["baseline"] = Prime(condition="baseline", text=None)
    ps.primes["cue_familiarity_priming"] = Prime(condition="cue_familiarity_priming", text="hello")
    d = ps.to_dict()
    back = PrimeSet.from_dict(d)
    assert back.item_id == "x"
    assert back.primes["cue_familiarity_priming"].text == "hello"
    assert back.primes["baseline"].text is None
