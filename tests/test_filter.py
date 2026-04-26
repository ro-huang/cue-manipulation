from koriat_cues.data.filter import _looks_single_entity
from koriat_cues.data.loader import Item
from koriat_cues.data.filter import filter_single_entity


def test_keeps_canonical_single_entities():
    assert _looks_single_entity("Jane Austen")
    assert _looks_single_entity("Samoa")
    assert _looks_single_entity("The Beatles")
    assert _looks_single_entity("Washington, D.C.")  # comma in abbreviation, not list
    assert _looks_single_entity("New York City")


def test_rejects_list_markers():
    assert not _looks_single_entity("Crick and Watson")
    assert not _looks_single_entity("apples or oranges")
    assert not _looks_single_entity("Samoa & Tonga")
    assert not _looks_single_entity("1970 and 1971")
    assert not _looks_single_entity("London; Paris")


def test_accepts_comma_abbreviations():
    # Commas alone are permitted because they occur in legitimate single-entity
    # abbreviations like "Washington, D.C." and "St. Petersburg, Russia" (rare).
    assert _looks_single_entity("Washington, D.C.")


def test_rejects_overly_long_answers():
    assert not _looks_single_entity("the thirty-fifth president of the United States")


def test_rejects_empty():
    assert not _looks_single_entity("")
    assert not _looks_single_entity("   ")


def test_filter_single_entity_on_items():
    items = [
        Item(id="1", question="q1", answers=["Samoa"]),
        Item(id="2", question="q2", answers=["Crick and Watson"]),
        Item(id="3", question="q3", answers=["Washington, D.C."]),
        Item(id="4", question="q4", answers=["apples or oranges"]),
    ]
    kept = filter_single_entity(items)
    assert [it.id for it in kept] == ["1", "3"]
