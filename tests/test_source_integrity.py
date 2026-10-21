"""The English a unit is scored against belongs to the harness, not to the agent."""
from mathula_tv.dub_submission import _source_text, validate_submission

SEGMENTS = [
    {"segment_id": "s1", "start_ms": 0, "end_ms": 4000,
     "restored_text": "The state paid him about 7,000 rand. Victor Majola denies the charge."},
    {"segment_id": "s2", "start_ms": 4000, "end_ms": 8000, "restored_text": "He appears again on Friday."},
]


def _item(item_id, start, end, segs, zulu="Isihumusho.", **extra):
    return {"item_id": item_id, "start_ms": start, "end_ms": end, "speaker_id": "SPEAKER_00", "zulu_text": zulu,
            "source_segment_ids": segs, **extra}


def _errors(items):
    return validate_submission({"items": items}, SEGMENTS)[0]


def test_an_agent_supplied_source_text_is_ignored_for_a_unit_that_owns_its_segments():
    item = _item("a", 0, 4000, ["s1"], source_text="A woman is awaiting trial.")   # a convenient shorter "source"
    assert _source_text(item, {s["segment_id"]: s for s in SEGMENTS}, shared=set()) == SEGMENTS[0]["restored_text"]


def test_a_shared_segment_unit_may_carry_an_exact_excerpt():
    items = [_item("a", 0, 2000, ["s1"], source_text="The state paid him about 7,000 rand."),
             _item("b", 2000, 4000, ["s1"], source_text="Victor Majola denies the charge."),
             _item("c", 4000, 8000, ["s2"])]
    assert _errors(items) == []
    assert _source_text(items[0], {s["segment_id"]: s for s in SEGMENTS}, shared={"s1"}) == items[0]["source_text"]


def test_a_paraphrased_excerpt_is_rejected():
    items = [_item("a", 0, 2000, ["s1"], source_text="He got some money."),
             _item("b", 2000, 4000, ["s1"], source_text="Victor Majola denies the charge."),
             _item("c", 4000, 8000, ["s2"])]
    assert any("copied exactly" in e for e in _errors(items))


def test_units_sharing_a_segment_must_together_carry_all_of_its_english():
    items = [_item("a", 0, 2000, ["s1"], source_text="The state paid him"),          # drops "about 7,000 rand" and the rest
             _item("b", 2000, 4000, ["s1"], source_text="Victor Majola denies the charge."),
             _item("c", 4000, 8000, ["s2"])]
    errors = _errors(items)
    assert any("together carry all" in e and "7" in e for e in errors)


def test_a_shared_unit_without_an_excerpt_is_still_an_error():
    items = [_item("a", 0, 2000, ["s1"]), _item("b", 2000, 4000, ["s1"], source_text="Victor Majola denies the charge."),
             _item("c", 4000, 8000, ["s2"])]
    assert any("explicit source_text" in e for e in _errors(items))
