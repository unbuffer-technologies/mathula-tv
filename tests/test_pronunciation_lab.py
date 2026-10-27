import pytest

from mathula_tv import pronunciation_lab as lab
from mathula_tv.pronunciation_lab import PronunciationBudgetExhausted, PronunciationLab, align, candidate_words
from mathula_tv.pronunciation_search import looks_isizulu


def test_identical_sounds_are_fully_close():
    out = align(["f", "aɪ", "t", "ə", "z"], ["f", "aɪ", "t", "ə", "z"])
    assert out["closeness"] == 1.0 and out["differences"] == []


def test_a_wrong_vowel_and_a_dropped_consonant_are_named_in_plain_words():
    out = align(["f", "l", "ɔɪ", "d"], ["f", "l", "o", "i", "d"])
    assert any("ɔɪ" in d and "vowel" in d for d in out["differences"]) and out["closeness"] < 1.0
    dropped = align(["m", "ɛ", "d", "ɪ", "k", "l"], ["m", "ɛ", "ɪ", "k", "l"])
    assert any("leaves out d" in d for d in dropped["differences"])


def test_a_closer_reading_scores_closer_than_a_worse_one():
    source = ["f", "aɪ", "t", "ə", "z"]
    close = align(source, ["f", "ai", "t", "e", "z"])                 # near vowels
    far = align(source, ["m", "a", "f", "e", "k", "t", "o", "r"])     # the "mafector" reading
    assert close["closeness"] > far["closeness"] + 0.3 and far["closeness"] < 0.5
    assert any("adds m" in d for d in far["differences"]) and len(far["differences"]) >= 3


def test_noisy_edges_of_the_speakers_window_barely_count():
    plain = align(["f", "aɪ", "t", "ə", "z"], ["f", "aɪ", "t", "ə", "z"])
    edges = align(["d", "f", "aɪ", "t", "ə", "z", "b"], ["f", "aɪ", "t", "ə", "z"])
    assert edges["closeness"] >= 0.85 and plain["closeness"] == 1.0


WORDS = [{"text": "Fighters,", "start": 313.8}, {"text": "medical", "start": 180.0}, {"text": "aid", "start": 180.6}, {"text": "the", "start": 1.0}]


def test_english_words_the_speaker_says_are_found_even_with_a_zulu_prefix_glued_on():
    found = candidate_words(["MaFighter, sifuna ukuthi ogogo", "Kufanele ibe ne-medical aid. Kunamathenda."], WORDS, looks_isizulu=looks_isizulu)
    names = {f["word"] for f in found}
    assert "MaFighter" in names and "medical" in names and "aid" not in names          # "aid" is under 4 letters
    fighter = next(f for f in found if f["word"] == "MaFighter")
    assert fighter["speaker_says"].startswith("Fighters") and "prefix" in fighter["note"]
    assert candidate_words(["MaFighter"], WORDS, looks_isizulu=looks_isizulu, exclude={"mafighter"}) == []


def test_a_prefix_glued_on_without_a_hyphen_is_still_found():
    words = [{"text": "science"}, {"text": "technology"}, {"text": "Fighters"}]
    texts = ["Kungani sidinga ungqongqoshe weScience and Technology? Phela iScience iyisifundo.", "NeTechnology iyisifundo."]
    found = candidate_words(texts, words, looks_isizulu=looks_isizulu)
    assert len(found) == 2                                                                # one entry per English word
    science = next(f for f in found if f["speaker_says"] == "science")
    technology = next(f for f in found if f["speaker_says"] == "technology")
    assert {science["word"], *science["also_written"]} == {"weScience", "iScience"} and science["occurrences_in_dub"] == 2
    assert {technology["word"], *technology["also_written"]} == {"Technology", "NeTechnology"}
    assert "prefix" in science["note"] and "prefix" in technology["note"]
    # a Zulu-shaped word with no glued prefix stays filtered out, even if the English transcript happens to contain it
    assert candidate_words(["Sinemali eningi"], [{"text": "emali"}, {"text": "malis"}], looks_isizulu=looks_isizulu) == []
    assert candidate_words(["imali"], [{"text": "imali"}], looks_isizulu=looks_isizulu) == []


class FakeLab(PronunciationLab):
    """The lab with the audio parts replaced: the speaker's sounds and the voice's sounds are fixed lists."""

    def __init__(self, voice_sounds):
        super().__init__("/nonexistent", voice="zu-ZA-ThembaNeural", key="k", region="r", work_dir="/tmp/none")
        self._voice = voice_sounds

    def source(self, word):
        return {"found": word == "Fighters", "phones": ["f", "aɪ", "t", "ə", "z"], "occurrences": 2, "each_occurrence": []}

    def voice_reading(self, spelling):
        return self._voice[spelling]


def test_trying_spellings_reports_sounds_differences_and_the_closest_so_far():
    fake = FakeLab({"MaFighter": ["m", "a", "f", "e", "k", "t", "o", "r"], "Fayitas": ["f", "a", "i", "t", "a", "s"]})
    out = fake.try_spellings("Fighters", ["MaFighter", "Fayitas"])
    assert out["speaker_sounds"] == "f aɪ t ə z" and out["closest_so_far"]["spelling"] == "Fayitas"
    assert [r["spelling"] for r in out["results"]] == ["MaFighter", "Fayitas"] and out["tries_left_for_word"] == lab.MAX_TRIES_PER_WORD - 2
    assert fake.tried["Fighters"]["Fayitas"]["closeness"] > fake.tried["Fighters"]["MaFighter"]["closeness"]
    assert "never says" in fake.try_spellings("Unspoken", ["x"])["error"]


def test_the_number_of_tries_is_capped_per_word_and_per_clip():
    fake = FakeLab({f"s{i}": ["f", "aɪ", "t", "ə", "z"] for i in range(10)})
    fake.try_spellings("Fighters", ["s0", "s1", "s2"])
    fake.try_spellings("Fighters", ["s3", "s4"])
    with pytest.raises(PronunciationBudgetExhausted):
        fake.try_spellings("Fighters", ["s5"])
    assert fake.total_tries == lab.MAX_TRIES_PER_WORD


# ---- the tools on a real workbench (the audio parts of the lab are faked) -------------------------------------------------

def make_workbench(tmp_path, voice_sounds):
    import json

    from mathula_tv.dub_workbench import Workbench
    from mathula_tv.experience_store import ExperienceStore

    store = ExperienceStore(tmp_path)               # as in production: the work dir; the store makes its own experience folder
    store.corpus_path.parent.mkdir(parents=True, exist_ok=True)
    store.corpus_path.write_text(json.dumps({"schema_version": "x", "clips": {"clipa": {"job_id": "job-1", "duration_s": 1, "segment_count": 1}}}))
    (store.clip_dir("clipa") / "native_dub").mkdir(parents=True)
    run_dir = store.repeat_dir("run-1", "clipa", 1)
    run_dir.mkdir(parents=True)
    (run_dir / "submission.json").write_text(json.dumps({"items": [{"item_id": "u1", "zulu_text": "MaFighter, sifuna ukuthi ogogo bangakhokhi."}]}))
    workbench = Workbench(store, lambda: None, voices={"SPEAKER_00": {"selected_voice": "zu-ZA-ThembaNeural"}})
    workbench.episode = {"clip_id": "clipa", "dir": str(run_dir), "tts_budget": 10, "tts_used": 0, "submissions": 1}
    fake = FakeLab(voice_sounds)
    fake._transcript = WORDS
    workbench._pronunciation_lab = fake
    return workbench, store


def test_the_agent_has_the_three_pronunciation_tools_and_they_are_dispatched():
    from types import SimpleNamespace

    from mathula_tv import dub_agent

    names = {t["function"]["name"] for t in dub_agent.TOOL_SPECS}
    assert {"list_foreign_words", "try_pronunciation", "use_pronunciation"} <= names
    workbench = SimpleNamespace(**{n: (lambda *a, **k: None) for n in ("open_clip", "recall_lessons", "measure_tts", "measure_tts_batch", "check_text",
                                                                      "preview_turn_padding", "submit_dub", "get_submission", "list_foreign_words",
                                                                      "try_pronunciation", "use_pronunciation")})
    dispatch = dub_agent.build_dispatch(workbench, clip_id="clipa", run_id="r", tts_budget=5)
    assert {"list_foreign_words", "try_pronunciation", "use_pronunciation"} <= set(dispatch)
    assert "list_foreign_words" in dub_agent.SYSTEM_PROMPT and "Never glue a Zulu prefix" in dub_agent.SYSTEM_PROMPT


def test_the_agent_is_pointed_at_the_english_words_its_dub_keeps(tmp_path):
    workbench, _ = make_workbench(tmp_path, {})
    found = workbench.list_foreign_words()
    assert [w["word"] for w in found["words"]] == ["MaFighter"] and found["tries_left_in_clip"] == lab.MAX_TRIES_PER_CLIP


def test_a_spelling_must_be_tried_before_it_is_used_and_using_it_reaches_the_lexicon_and_the_shared_memory(tmp_path):
    import json

    from mathula_tv.pronunciation_memory import entries_for, memory_path

    workbench, store = make_workbench(tmp_path, {"Fighters": ["f", "aɪ", "t", "ə", "z"]})
    with pytest.raises(ValueError):
        workbench.use_pronunciation("Fighters", "Fighters")                  # never tried
    workbench.try_pronunciation("Fighters", ["Fighters"])
    result = workbench.use_pronunciation("Fighters", "Fighters")
    assert result["ok"] and result["closeness"] == 1.0
    lexicon = json.loads((store.clip_dir("clipa") / "native_dub" / "pronunciation_names.json").read_text())
    assert [(e["display_text"], e["tts_text"], e["source"]) for e in lexicon["entries"]] == [("Fighters", "Fighters", "agent-pronunciation")]
    remembered = entries_for(memory_path(tmp_path), "zu-ZA-ThembaNeural")
    assert [(e["display_text"], e["source"]) for e in remembered] == [("Fighters", "agent")]
    assert workbench.pronunciation_choices[0]["spelling"] == "Fighters"


def test_a_person_approved_reading_is_not_replaced_by_the_agents_choice(tmp_path):
    from mathula_tv.pronunciation_memory import entries_for, memory_path, remember

    workbench, _ = make_workbench(tmp_path, {"Fayitas": ["f", "aɪ", "t", "ə", "z"]})
    remember(memory_path(tmp_path), "Fighters", "Fayitha", source="person", voice="zu-ZA-ThembaNeural", job_id="manual")
    workbench.try_pronunciation("Fighters", ["Fayitas"])
    workbench.use_pronunciation("Fighters", "Fayitas")
    assert entries_for(memory_path(tmp_path), "zu-ZA-ThembaNeural")[0]["tts_text"] == "Fayitha"


def test_the_models_reading_of_the_zulu_finds_what_the_rules_cannot():
    words = [{"text": "science"}, {"text": "Fighters"}]
    named = [{"written": "iScience", "english": "science"}, {"written": "iSayansi", "english": "sayansi"}]    # the second is not said by the speaker
    found = candidate_words(["Phela iScience iyisifundo. iSayansi."], words, looks_isizulu=looks_isizulu, named=named)
    assert [f["word"] for f in found] == ["iScience"]
    assert found[0]["speaker_says"] == "science" and "prefix" in found[0]["note"] and found[0]["occurrences_in_dub"] == 1
    # a word already given a reading, or that the rules found, is not listed twice
    assert candidate_words(["iScience"], words, looks_isizulu=looks_isizulu, named=named, exclude={"science"}) == []
    both = candidate_words(["weScience", "iScience"], words, looks_isizulu=looks_isizulu, named=named)
    assert len(both) == 1 and both[0]["occurrences_in_dub"] == 2 and both[0]["speaker_says"] == "science"
