import json

from mathula_tv.dub_repair import remember_verified_fixes
from mathula_tv.pronunciation_memory import (
    MEMORY_SOURCE, apply_to_lexicon, entries_for, load, record_outcomes, remember,
)

THEMBA = "zu-ZA-ThembaNeural"
THANDO = "zu-ZA-ThandoNeural"


def test_the_same_reading_confirmed_by_another_job_is_counted_not_duplicated(tmp_path):
    path = tmp_path / "memory.json"
    remember(path, "hlu", "hluuu", source="repair", voice=THEMBA, job_id="job-a")
    remember(path, "hlu", "hluuu", source="repair", voice=THEMBA, job_id="job-b")
    entries = list(load(path)["entries"].values())
    assert len(entries) == 1 and entries[0]["jobs"] == ["job-a", "job-b"]


def test_a_persons_reading_is_not_overwritten_by_a_repair(tmp_path):
    path = tmp_path / "memory.json"
    remember(path, "Jacob Zuma", "Jeykhob Zuma", source="person", voice=THEMBA, job_id="manual", kind="personal_name")
    remember(path, "Jacob Zuma", "Jakab Zuma", source="repair", voice=THEMBA, job_id="job-x")
    assert entries_for(path, THEMBA)[0]["tts_text"] == "Jeykhob Zuma"


def test_an_entry_applies_only_to_the_voice_it_was_verified_on(tmp_path):
    path = tmp_path / "memory.json"
    remember(path, "hlu", "hluuu", source="repair", voice=THEMBA, job_id="job-a")
    assert [e["display_text"] for e in entries_for(path, THEMBA)] == ["hlu"] and entries_for(path, THANDO) == []


def test_memory_entries_replace_the_automatic_reading_of_the_same_text_in_a_jobs_lexicon(tmp_path):
    memory, lexicon = tmp_path / "memory.json", tmp_path / "pronunciation_names.json"
    remember(memory, "Floyd Shivambu", "Floyd Shivambu", source="person", voice=THEMBA, job_id="manual", kind="personal_name")
    lexicon.write_text(json.dumps({"voice": THEMBA, "entries": [
        {"display_text": "Floyd Shivambu", "tts_text": "Floyid Shivambu", "kind": "personal_name"},
        {"display_text": "Nomvalo", "tts_text": "Nomvalo", "kind": "personal_name"}]}))
    assert apply_to_lexicon(lexicon, memory, THEMBA) == ["Floyd Shivambu"]
    entries = {e["display_text"]: e for e in json.loads(lexicon.read_text())["entries"]}
    assert entries["Floyd Shivambu"]["tts_text"] == "Floyd Shivambu" and entries["Floyd Shivambu"]["source"] == MEMORY_SOURCE
    assert "Nomvalo" in entries


def test_a_reading_that_fails_again_in_two_jobs_is_switched_off_but_a_persons_never_is(tmp_path):
    memory, lexicon = tmp_path / "memory.json", tmp_path / "pronunciation_names.json"
    remember(memory, "hlu", "hluuu", source="repair", voice=THEMBA, job_id="job-a")
    remember(memory, "Jacob", "Jeykhob", source="person", voice=THEMBA, job_id="manual")
    lexicon.write_text(json.dumps({"voice": THEMBA, "entries": []}))
    apply_to_lexicon(lexicon, memory, THEMBA)
    swallowed = [{"words": ["hlu."]}, {"words": ["Jeykhob"]}]
    assert record_outcomes(lexicon, memory, THEMBA, swallowed) == []
    assert record_outcomes(lexicon, memory, THEMBA, swallowed) == ["hlu"]
    assert [e["display_text"] for e in entries_for(memory, THEMBA)] == ["Jacob"]


def test_a_repair_is_remembered_only_after_the_real_render_no_longer_drops_the_word(tmp_path):
    work = tmp_path
    (work / "jobs" / "job-a" / "native_dub").mkdir(parents=True)
    fixes = [{"word": "hlu.", "fixed": True, "respelling": "hluuu", "voice": THEMBA, "unit": "u4"}]
    still_dropped = {"units": [{"unit": "u4", "dropped": [{"words": ["hlu."]}]}]}
    assert remember_verified_fixes(work, "job-a", fixes, still_dropped) == []
    assert remember_verified_fixes(work, "job-a", fixes, {"units": []}) == ["hlu"]
