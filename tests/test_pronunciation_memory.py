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


def test_a_verified_repair_is_not_overwritten_by_an_agents_choice(tmp_path):
    path = tmp_path / "memory.json"
    remember(path, "Jacob Zuma", "Jeykhob Zuma", source="repair", voice=THEMBA, job_id="job-x", kind="personal_name")
    remember(path, "Jacob Zuma", "Jakab Zuma", source="agent", voice=THEMBA, job_id="job-y")
    assert entries_for(path, THEMBA)[0]["tts_text"] == "Jeykhob Zuma"


def test_no_person_can_put_a_reading_in_the_memory(tmp_path):
    import pytest

    from mathula_tv.pronunciation_memory import main

    path = tmp_path / "memory.json"
    with pytest.raises(ValueError):
        remember(path, "Science", "Sayense", source="person", voice=THEMBA, job_id="manual")           # no such source any more
    assert load(path)["entries"] == {}
    with pytest.raises(SystemExit):
        main(["remember", "Science", "Sayense"])                                                       # and no command line to do it with


def test_an_entry_applies_only_to_the_voice_it_was_verified_on(tmp_path):
    path = tmp_path / "memory.json"
    remember(path, "hlu", "hluuu", source="repair", voice=THEMBA, job_id="job-a")
    assert [e["display_text"] for e in entries_for(path, THEMBA)] == ["hlu"] and entries_for(path, THANDO) == []


def test_memory_entries_replace_the_automatic_reading_of_the_same_text_in_a_jobs_lexicon(tmp_path):
    memory, lexicon = tmp_path / "memory.json", tmp_path / "pronunciation_names.json"
    remember(memory, "Floyd Shivambu", "Floyd Shivambu", source="repair", voice=THEMBA, job_id="job-a", kind="personal_name")
    lexicon.write_text(json.dumps({"voice": THEMBA, "entries": [
        {"display_text": "Floyd Shivambu", "tts_text": "Floyid Shivambu", "kind": "personal_name"},
        {"display_text": "Nomvalo", "tts_text": "Nomvalo", "kind": "personal_name"}]}))
    assert apply_to_lexicon(lexicon, memory, THEMBA) == ["Floyd Shivambu"]
    entries = {e["display_text"]: e for e in json.loads(lexicon.read_text())["entries"]}
    assert entries["Floyd Shivambu"]["tts_text"] == "Floyd Shivambu" and entries["Floyd Shivambu"]["source"] == MEMORY_SOURCE
    assert "Nomvalo" in entries


def test_the_memory_never_respells_an_ordinary_zulu_word_but_keeps_its_entry(tmp_path):
    memory, lexicon = tmp_path / "memory.json", tmp_path / "pronunciation_names.json"
    remember(memory, "eNkandla", "eNggangla", source="agent", voice=THEMBA, job_id="job-a", kind="english_code_switch")
    remember(memory, "Heidi", "Haidi", source="agent", voice=THEMBA, job_id="job-a", kind="english_code_switch")
    lexicon.write_text(json.dumps({"voice": THEMBA, "entries": []}))
    assert apply_to_lexicon(lexicon, memory, THEMBA, skip=lambda text: text == "eNkandla") == ["Heidi"]
    assert {e["display_text"] for e in json.loads(lexicon.read_text())["entries"]} == {"Heidi"}
    from mathula_tv.pronunciation_memory import entries_for

    assert {e["display_text"] for e in entries_for(memory, THEMBA)} == {"eNkandla", "Heidi"}               # the entry is still there, only not applied


def test_a_closeness_measured_with_another_ruler_does_not_keep_a_saved_reading(tmp_path):
    from mathula_tv.pronunciation_memory import beaten_by_saved, load, remember, saved
    from mathula_tv.phone_features import RULER_VERSION

    memory = tmp_path / "memory.json"
    remember(memory, "Heidi", "Haidi", source="agent", voice=THEMBA, job_id="a", closeness=0.95)
    assert load(memory)["entries"][f"heidi|{THEMBA}"]["ruler"] == RULER_VERSION
    assert beaten_by_saved(saved(memory, "Heidi", THEMBA), "Haidii", 0.5)                                  # measured with this ruler: it stays
    data = load(memory)
    del data["entries"][f"heidi|{THEMBA}"]["ruler"]                                                        # an entry from before rulers were recorded
    memory.write_text(json.dumps(data))
    assert not beaten_by_saved(saved(memory, "Heidi", THEMBA), "Haidii", 0.5)


def test_a_reading_that_fails_again_in_two_jobs_is_switched_off_whoever_chose_it(tmp_path):
    memory, lexicon = tmp_path / "memory.json", tmp_path / "pronunciation_names.json"
    remember(memory, "hlu", "hluuu", source="repair", voice=THEMBA, job_id="job-a")
    remember(memory, "Jacob", "Jeykhob", source="agent", voice=THEMBA, job_id="job-a")
    lexicon.write_text(json.dumps({"voice": THEMBA, "entries": []}))
    apply_to_lexicon(lexicon, memory, THEMBA)
    swallowed = [{"words": ["hlu."]}, {"words": ["Jeykhob"]}]
    assert record_outcomes(lexicon, memory, THEMBA, swallowed) == []
    assert record_outcomes(lexicon, memory, THEMBA, swallowed) == ["hlu", "Jacob"]
    assert entries_for(memory, THEMBA) == []


def test_a_repair_is_remembered_only_after_the_real_render_no_longer_drops_the_word(tmp_path):
    work = tmp_path
    (work / "jobs" / "job-a" / "native_dub").mkdir(parents=True)
    fixes = [{"word": "hlu.", "fixed": True, "respelling": "hluuu", "voice": THEMBA, "unit": "u4"}]
    still_dropped = {"units": [{"unit": "u4", "dropped": [{"words": ["hlu."]}]}]}
    assert remember_verified_fixes(work, "job-a", fixes, still_dropped) == []
    assert remember_verified_fixes(work, "job-a", fixes, {"units": []}) == ["hlu"]


def test_a_jobs_lexicon_stops_carrying_an_entry_that_has_left_the_memory(tmp_path):
    import json

    from mathula_tv.pronunciation_memory import apply_to_lexicon, memory_path, remember

    memory_file = memory_path(tmp_path)
    remember(memory_file, "Science", "Sayense", source="agent", voice="v", job_id="j1", kind="english_code_switch")
    remember(memory_file, "fund", "fund", source="agent", voice="v", job_id="j1", kind="english_code_switch")
    lexicon = tmp_path / "pronunciation_names.json"
    lexicon.write_text(json.dumps({"entries": [{"display_text": "Zuma", "tts_text": "Zuuma", "source": "web"}]}), encoding="utf-8")
    assert sorted(apply_to_lexicon(lexicon, memory_file, "v")) == ["Science", "fund"]
    data = json.loads(memory_file.read_text(encoding="utf-8"))
    del data["entries"]["science|v"]                                                               # the entry is cleared from the memory
    memory_file.write_text(json.dumps(data), encoding="utf-8")
    assert apply_to_lexicon(lexicon, memory_file, "v") == ["fund"]
    texts = {e["display_text"] for e in json.loads(lexicon.read_text(encoding="utf-8"))["entries"]}
    assert texts == {"Zuma", "fund"}                                                               # Science is gone from the job; others stay


def test_a_respelling_typed_into_a_units_text_is_put_back_as_the_word():
    from mathula_tv.pronunciation_memory import restore_written_words

    entries = [
        {"display_text": "maFighter", "tts_text": "Fayithaz", "kind": "english_code_switch"},
        {"display_text": "Technology", "tts_text": "Tekhnoloji", "kind": "english_code_switch"},
        {"display_text": "science", "tts_text": "sayensi", "kind": "english_code_switch"},
        {"display_text": "sick", "tts_text": "sik", "kind": "english_code_switch"},                         # too short to look for
        {"display_text": "fund", "tts_text": "fund", "kind": "english_code_switch"},                        # no respelling at all
        {"display_text": "benefit", "tts_text": "bhenefit", "kind": "english_code_switch", "disabled": True},
        {"display_text": "hlu", "tts_text": "hluuu", "kind": "other"},
    ]
    items = [{"item_id": "u02", "zulu_text": "~pause800 Fayithaz, thina siyinhlangano. Phela maFayithaz kakhulu."},
             {"item_id": "u20", "zulu_text": "ungqongqoshe weSayensi and Tekhnoloji? Phela Sayensi iyisifundo. Bhenefit kanye hluuu."}]
    out, notes = restore_written_words(items, entries)
    assert out[0]["zulu_text"] == "~pause800 MaFighter, thina siyinhlangano. Phela maFighter kakhulu."          # a capital stays a capital; the glued prefix is not doubled
    assert out[1]["zulu_text"] == "ungqongqoshe weScience and Technology? Phela Science iyisifundo. Bhenefit kanye hluuu."
    assert items[0]["zulu_text"].startswith("~pause800 Fayithaz")                                               # the caller's units are untouched
    assert len(notes) == 3 and "'Fayithaz'" in notes[0] and "keeps the word 'maFighter'" in notes[0]
    assert restore_written_words(items, [])[1] == []


def test_an_entry_saved_without_the_zulu_prefix_of_its_word_gets_the_prefix_back_when_applied(tmp_path):
    import json

    from mathula_tv.pronunciation_memory import apply_to_lexicon, remember

    memory, lexicon = tmp_path / "memory.json", tmp_path / "pronunciation_names.json"
    remember(memory, "weScience", "saynsi", source="agent", voice=THEMBA, job_id="a", kind="english_code_switch")
    remember(memory, "MaFighter", "MaFayitas", source="agent", voice=THEMBA, job_id="a", kind="english_code_switch")        # already carries it
    remember(memory, "Technology", "Teknoloji", source="agent", voice=THEMBA, job_id="a", kind="english_code_switch")        # no prefix on the word
    remember(memory, "hlu", "hluuu", source="repair", voice=THEMBA, job_id="a", kind="other")
    lexicon.write_text(json.dumps({"voice": THEMBA, "entries": []}))
    apply_to_lexicon(lexicon, memory, THEMBA)
    spoken = {e["display_text"]: e["tts_text"] for e in json.loads(lexicon.read_text())["entries"]}
    assert spoken == {"weScience": "wesaynsi", "MaFighter": "MaFayitas", "Technology": "Teknoloji", "hlu": "hluuu"}


def test_a_saved_reading_that_measured_closer_is_not_replaced_by_one_that_measured_less_close(tmp_path):
    from mathula_tv.pronunciation_memory import beaten_by_saved, entries_for, remember, saved

    path = tmp_path / "experience" / "pronunciation_memory.json"
    voice = "zu-ZA-ThembaNeural"
    remember(path, "MaFighter", "MaFighters", source="agent", voice=voice, job_id="a", closeness=1.0)
    kept = remember(path, "MaFighter", "MaFighter", source="agent", voice=voice, job_id="b", closeness=0.94)         # a run out of turns, a worse pick
    assert kept["tts_text"] == "MaFighters" and entries_for(path, voice)[0]["tts_text"] == "MaFighters"
    assert beaten_by_saved(saved(path, "MaFighter", voice), "MaFighter", 0.94) and not beaten_by_saved(saved(path, "MaFighter", voice), "MaFighter", 0.99)
    remember(path, "MaFighter", "Fayitas", source="agent", voice=voice, job_id="c", closeness=0.99)                  # within the margin: the newer one stands
    assert entries_for(path, voice)[0]["tts_text"] == "Fayitas" and entries_for(path, voice)[0]["closeness"] == 0.99
    remember(path, "Plain", "Plane", source="agent", voice=voice, job_id="a")                                         # no closeness recorded: replaced as before
    remember(path, "Plain", "Plein", source="agent", voice=voice, job_id="b", closeness=0.5)
    assert [e["tts_text"] for e in entries_for(path, voice) if e["display_text"] == "Plain"] == ["Plein"]


def test_a_reading_the_agent_chose_earlier_does_not_stay_in_a_lexicon_once_the_memory_switches_it_off(tmp_path):
    import json

    from mathula_tv import pronunciation_memory as pm

    memory = tmp_path / "memory.json"
    pm.remember(memory, "Heidi", "Dyi", source="agent", voice="v", job_id="j", kind="english_code_switch")
    pm.remember(memory, "Party", "Phaathi2", source="agent", voice="v", job_id="j", kind="english_code_switch")
    lexicon = tmp_path / "pronunciation_names.json"
    lexicon.write_text(json.dumps({"entries": [
        {"display_text": "WhatsApp", "tts_text": "Whatsep", "kind": "organisation_name"},
        {"display_text": "Heidi", "tts_text": "Dyi", "source": "agent-pronunciation"},
        {"display_text": "Western", "tts_text": "Issen", "source": "agent-pronunciation"}]}), encoding="utf-8")
    data = pm.load(memory)
    data["entries"][pm._key("Heidi", "v")]["disabled"] = True
    pm._save(memory, data)
    pm.apply_to_lexicon(lexicon, memory, "v")
    shown = {e["display_text"]: e["tts_text"] for e in json.loads(lexicon.read_text(encoding="utf-8"))["entries"]}
    assert shown == {"WhatsApp": "Whatsep", "Party": "Phaathi2"}          # the names stage's reading stays, a switched-off or unbacked agent reading goes
