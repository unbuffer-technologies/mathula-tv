import pytest

from mathula_tv.ear_lessons import HARD_TEXT
from mathula_tv.pronunciation import ITEM_CLASSES, PronunciationDictionary, item_class_table, with_default_organisation_initialisms
from mathula_tv.say_it import say_it

# one line per complaint (tests/fixtures/hard_text.txt, grown by ear_lessons.add_hard_text): after the rules, no digit, abbreviation or capital run is left for the voice to guess at
CLOSED_WORLD = [line for line in (HARD_TEXT.read_text(encoding="utf-8").splitlines() if HARD_TEXT.is_file() else []) if line.strip()]


def _dictionary():
    return with_default_organisation_initialisms(PronunciationDictionary("v1", language="zu-ZA", job_id="job-1"))


def test_every_item_class_is_documented_and_runs_on_empty_text():
    names = [c.name for c in ITEM_CLASSES]
    assert names == ["date", "currency", "figure", "reference_number", "legal_number", "phone_number", "ordinal", "title", "initialism"]
    for row in item_class_table():
        assert row["example"] and row["ear_override"]
    d = _dictionary()
    assert d.apply("Sawubona").tts_text == "Sawubona"                          # nothing to find: nothing changes
    assert d.apply("Sawubona", kinds={"date"}).tts_text == "Sawubona"


def test_only_the_asked_for_kinds_run():
    d = _dictionary()
    assert d.apply("eSection 79", kinds={"date"}).tts_text == "eSection 79"                 # the figure class answers "number", not "date"
    assert d.apply("eSection 79", kinds={"number"}).tts_text == "eSection seventy-nayn"


@pytest.mark.parametrize("line", CLOSED_WORLD)
def test_nothing_is_left_as_written_for_the_voice_to_guess_at(line):
    out = say_it(_dictionary(), line)
    assert out["still_as_written"] == [], (line, out["spoken"], out["still_as_written"])


def test_the_hard_text_fixture_exists_and_holds_this_sessions_complaints():
    assert len(CLOSED_WORLD) >= 12
    joined = " ".join(CLOSED_WORLD)
    assert "sinoDkt Arthur Shopola" in joined and "eSection 79" in joined and "owezi-2019" in joined and "kuka-31%" in joined
