from pathlib import Path

import pytest

from mathula_tv.pronunciation_ledger import Ledger, problems_of, request_text


def test_a_word_shown_with_something_named_wrong_and_never_touched_is_unexamined_until_tried_used_or_explained():
    ledger = Ledger()
    ledger.shown("MaFighter", "saved reading, in the reminder", closeness=0.94, problems=["the voice's word ends without the final s"])
    ledger.shown("Creek", "saved reading, in the reminder", closeness=1.0, problems=[])                      # nothing named wrong: nothing to ask
    assert [item["word"] for item in ledger.unexamined()] == ["MaFighter"]
    assert "MaFighter" in request_text(ledger.unexamined()) and "final s" in request_text(ledger.unexamined())
    ledger.note("ne-MaFighter", "kept_as_is", "the final s is soft in the speaker too")                       # a prefix does not make it another word
    assert ledger.unexamined() == []
    other = Ledger()
    other.shown("South", "measured as written, in the reminder", problems=["the voice adds ə at the end"])
    other.tried("South", ["Saut", "Sauth"])
    assert other.unexamined() == [] and other.summary()["tried"] == 1


def test_a_note_needs_a_known_decision_and_a_reason_and_the_ledger_serialises():
    ledger = Ledger()
    with pytest.raises(ValueError, match="decision"):
        ledger.note("Creek", "maybe", "because it sounded fine")
    with pytest.raises(ValueError, match="why"):
        ledger.note("Creek", "kept_as_is", "ok")
    assert ledger.note("Creek", "changed", "Kreet matched every sound")["ok"] is True
    ledger.used("Creek", "Kreet", 1.0)
    data = ledger.to_dict()
    assert data["words"][0]["used"] == {"spelling": "Kreet", "closeness": 1.0} and data["words"][0]["note"]["decision"] == "changed" and data["unexamined"] == []


def test_problems_come_from_differences_the_ending_and_what_the_sentence_loses():
    row = {"differences": ["speaker ts, voice dʒ (consonant)"], "ending": "the voice ends with a vowel", "lost_in_its_sentence": ["t", "s"]}
    assert problems_of(row) == ["speaker ts, voice dʒ (consonant)", "ending: the voice ends with a vowel", "in its sentence the voice loses t s"]
    assert problems_of({"differences": [], "ending": None}) == []
