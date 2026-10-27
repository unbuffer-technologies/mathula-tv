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
    # a name the model marks as a name is not offered to the lab, even though the speaker says it
    names = candidate_words(["eNkomazi", "iScience"], [{"text": "Nkomazi"}, {"text": "science"}], looks_isizulu=looks_isizulu,
                            named=[{"written": "eNkomazi", "english": "Nkomazi", "kind": "name"}, {"written": "iScience", "english": "science", "kind": "word"}])
    assert [f["word"] for f in names] == ["iScience"]
    both = candidate_words(["weScience", "iScience"], words, looks_isizulu=looks_isizulu, named=named)
    assert len(both) == 1 and both[0]["occurrences_in_dub"] == 2 and both[0]["speaker_says"] == "science"


def test_try_spellings_flags_same_sounds_and_says_what_to_change_next():
    lab_ = FakeLab({"correctional": ["k", "o", "r", "i", "tʃ", "i", "n", "a", "l"], "koreshinal": ["k", "o", "r", "i", "tʃ", "i", "n", "a", "l"],
                    "koreksinal": ["k", "o", "r", "e", "k", "s", "i", "n", "a", "l"]})
    lab_.source = lambda word: {"found": True, "phones": ["k", "o", "r", "e", "k", "s", "n", "a", "l"], "occurrences": 3, "each_occurrence": []}
    first = lab_.try_spellings("correctional", ["correctional", "koreshinal"])
    assert "same_sounds_as" not in first["results"][0] and first["results"][1]["same_sounds_as"] == "correctional"
    assert "gave only 1 different sound" in first["advice"] and "tries left" in first["advice"] and "Still different" in first["advice"]
    second = lab_.try_spellings("correctional", ["koreksinal"])
    assert second["closest_so_far"]["spelling"] == "koreksinal"
    assert second["closest_so_far"]["closeness"] > first["closest_so_far"]["closeness"] and second["advice"].endswith("use it.")      # now close enough


def test_advice_is_short_when_the_closest_spelling_is_close_enough():
    lab_ = FakeLab({"Fighters": ["f", "aɪ", "t", "ə", "z"]})
    out = lab_.try_spellings("Fighters", ["Fighters"])
    assert out["advice"].endswith("use it.")


class ListenerLab(FakeLab):
    """FakeLab where an English listener's answers are fixed."""

    def __init__(self, voice_sounds, heard):
        super().__init__(voice_sounds)
        self._heard = heard

    def source(self, word):
        return {"found": True, "phones": ["s", "aɪ", "ə", "n", "s"], "occurrences": 2, "spoken_form": "science", "each_occurrence": []}

    def english_hears(self, spelling):
        return self._heard.get(spelling)


def test_the_english_listener_leads_the_ranking_and_the_advice():
    lab_ = ListenerLab({"Sains": ["s", "aɪ", "n", "s"], "Sayinsi": ["s", "aɪ", "iː", "n", "s"]}, {"Sains": "Scenes.", "Sayinsi": "Science."})
    out = lab_.try_spellings("science", ["Sains", "Sayinsi"])
    by_spelling = {r["spelling"]: r for r in out["results"]}
    assert by_spelling["Sains"]["english_hears"] == "Scenes." and by_spelling["Sains"]["english_listener_understands"] is False
    assert by_spelling["Sayinsi"]["english_listener_understands"] is True
    assert out["closest_so_far"]["spelling"] == "Sayinsi"
    assert "writes the word for 'Sayinsi'" in out["advice"] and "Scenes" in out["advice"]


def test_nobody_understood_says_what_was_heard_and_a_listener_outage_changes_nothing():
    none = ListenerLab({"Science": ["s", "k", "iː", "n", "ts"]}, {"Science": "Skins."}).try_spellings("science", ["Science"])
    assert "No spelling is understood yet" in none["advice"] and "Skins" in none["advice"]
    down = ListenerLab({"Sains": ["s", "aɪ", "n", "s"]}, {}).try_spellings("science", ["Sains"])          # english_hears -> None: could not ask
    assert "english_hears" not in down["results"][0] and "closeness" in down["results"][0]


def test_understood_as_ignores_spacing_case_punctuation_and_a_plural_s():
    from mathula_tv.pronunciation_lab import understood_as

    assert understood_as("Correctional.", "correctional") and understood_as("Fighter", "Fighters") and understood_as("Science", "science")
    assert not understood_as("Scenes.", "science") and not understood_as("", "science")


def test_saved_agent_readings_an_english_listener_rejects_go_back_on_the_list():
    from mathula_tv.pronunciation_lab import failing_readings

    class Checker:
        def reading_understood(self, word, spelling):
            return {("sick", "sick"): False, ("fund", "fund"): True, ("deputy", "dephuthi"): None}[(word, spelling)]

    entries = [
        {"display_text": "sick", "tts_text": "sick", "source": "agent", "kind": "english_code_switch"},
        {"display_text": "fund", "tts_text": "fund", "source": "agent", "kind": "english_code_switch"},
        {"display_text": "deputy", "tts_text": "dephuthi", "source": "agent", "kind": "english_code_switch"},          # could not be asked: fine
        {"display_text": "Fighters", "tts_text": "Fayitas", "source": "person", "kind": "english_code_switch"},        # approved by ear: never re-checked
        {"display_text": "hlu", "tts_text": "hluuu", "source": "repair", "kind": "other"},                              # not an English word
        {"display_text": "old", "tts_text": "old", "source": "agent", "kind": "english_code_switch", "disabled": True},
    ]
    assert failing_readings(Checker(), entries) == {"sick"}


def test_reading_understood_asks_once_and_spends_no_try():
    class Lab(ListenerLab):
        asked = 0

        def _audio_for(self, spelling):
            return None

        def english_hears(self, spelling):
            Lab.asked += 1
            return "Sick."

    lab_ = Lab({}, {})
    assert lab_.reading_understood("sick", "sik") is True and lab_.reading_understood("sick", "sik") is True
    assert Lab.asked == 1 and lab_.total_tries == 0
