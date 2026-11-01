from pathlib import Path

import pytest

from mathula_tv.pronunciation_ledger import Ledger, far_request_text, problems_of, request_text


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
    saved = {"differences": [], "spelling": "Eyai", "english_hears": "Ed.", "english_listener_understands": False}
    assert problems_of(saved) == ["an English listener writes 'Ed.' when the voice says the saved 'Eyai', not the word"]      # a saved reading the listener does not write back
    assert problems_of({**saved, "spelling": None}) == [] and problems_of({**saved, "english_listener_understands": True}) == []     # a word as written is not judged this way
    assert problems_of({**saved, "letters_heard": True}) == []                                    # the phone recogniser decoded the letters: the speech-to-text's miss is no problem
    assert len(problems_of({**saved, "letters_heard": False})) == 1


def test_a_word_far_from_the_speaker_is_not_cleared_by_a_reason_alone_only_by_hearing_a_spelling():
    ledger = Ledger()
    ledger.shown("High", "measured as written, in the word list", closeness=0.47, problems=["speaker aɪ, voice i (vowel)"])
    ledger.shown("OK", "measured as written, in the word list", closeness=0.63, problems=["speaker eː, voice o (vowel)"])
    ledger.shown("Zuma", "measured as written, in the word list", closeness=1.0, problems=[])
    ledger.shown("Raila", "listed by list_foreign_words")                                                 # never measured: nothing to say
    ledger.note("High", "kept_as_is", "part of the official Western Cape High Court title")                # a reason about the kind of word does not count
    assert [(i["word"], i["closeness"]) for i in ledger.untried_far()] == [("High", 0.47), ("OK", 0.63)]   # weakest first; the close and the unmeasured are left out
    ledger.tried("High", ["Hai", "Hay"])
    ledger.used("OK", "Okey", 0.9)
    assert ledger.untried_far() == [] and ledger.to_dict()["untried_far"] == []
    ledger = Ledger()
    ledger.shown("AI", "saved reading, in the reminder", closeness=0.85, problems=problems_of({"differences": [], "spelling": "Eyai", "english_hears": "Ed.", "english_listener_understands": False}))
    ledger.shown("Heidi", "saved reading, in the reminder", closeness=0.9, problems=[])
    ledger.note("AI", "kept_as_is", "understandable from the context")                               # a reason is not enough for this
    asked_far = ledger.untried_far()
    assert [(i["word"], i.get("listener_writes")) for i in asked_far] == [("AI", "'Ed.'")]               # a closeness of 0.85 would not have listed it
    assert "AI (an English listener writes 'Ed.' for the reading saved for it)" in far_request_text(asked_far)
    ledger.tried("AI", ["Ey Aayi"])
    assert ledger.untried_far() == []                                                                    # hearing a spelling answers it
    many = [{"word": f"Word{i}", "problems": [f"problem {i}"]} for i in range(20)]
    asked = request_text(many)
    assert "Word0 (" in asked and "Word19 (" in asked                           # all twenty are asked about, not the first eight
    text = far_request_text([{"word": "High", "closeness": 0.47}])
    assert "High (closeness 0.47)" in text and "not a reason" in text and "try_pronunciation" in text

