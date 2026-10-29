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


def test_a_verified_repair_is_not_replaced_by_the_agents_choice(tmp_path):
    from mathula_tv.pronunciation_memory import entries_for, memory_path, remember

    workbench, _ = make_workbench(tmp_path, {"Fayitas": ["f", "aɪ", "t", "ə", "z"]})
    remember(memory_path(tmp_path), "Fighters", "Fayitha", source="repair", voice="zu-ZA-ThembaNeural", job_id="job-a")
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


def test_the_english_listener_and_the_sounds_agree_and_the_advice_says_so():
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
            return {("sick", "sick"): False, ("fund", "fund"): True, ("deputy", "dephuthi"): None, ("Fighters", "Fayitas"): False}[(word, spelling)]

    entries = [
        {"display_text": "sick", "tts_text": "sick", "source": "agent", "kind": "english_code_switch"},
        {"display_text": "fund", "tts_text": "fund", "source": "agent", "kind": "english_code_switch"},
        {"display_text": "deputy", "tts_text": "dephuthi", "source": "agent", "kind": "english_code_switch"},          # could not be asked: fine
        {"display_text": "Fighters", "tts_text": "Fayitas", "source": "repair", "kind": "english_code_switch"},        # whoever chose it: checked like any other
        {"display_text": "hlu", "tts_text": "hluuu", "source": "repair", "kind": "other"},                              # not an English word
        {"display_text": "old", "tts_text": "old", "source": "agent", "kind": "english_code_switch", "disabled": True},
    ]
    assert failing_readings(Checker(), entries) == {"sick", "fighters"}


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


def test_measured_sound_differences_are_shown_with_each_try_and_in_the_advice():
    class Measured(ListenerLab):
        def sound_differences(self, source, spelling):
            return ["the voice's stop number 1 (b), against 2 of the speaker's occurrence(s): voicing starts 52 ms after the release against the speaker's 26"]                 if spelling == "Sains" else []

    out = Measured({"Sains": ["s", "aɪ", "n", "s"]}, {"Sains": "Scenes."}).try_spellings("science", ["Sains"])
    assert out["results"][0]["how_the_sounds_are_made_differently"][0].startswith("the voice's stop number 1")
    assert "Measured in the closest ('Sains')" in out["advice"]
    quiet = Measured({"Sayinsi": ["s", "aɪ", "iː", "n", "s"]}, {"Sayinsi": "Science."}).try_spellings("science", ["Sayinsi"])
    assert "how_the_sounds_are_made_differently" not in quiet["results"][0]


def test_a_sound_the_voice_makes_alone_and_loses_in_its_sentence_is_named():
    from mathula_tv.pronunciation_lab import lost_in_the_sentence

    speaker = ["t", "e", "k", "n", "o", "l", "o", "dʒ"]
    alone = ["t", "e", "k", "n", "o", "l", "o", "dʒ", "i"]
    assert lost_in_the_sentence(speaker, alone, ["t", "e", "k", "n", "o", "l", "o"]) == ["dʒ"]             # the ending is gone
    assert lost_in_the_sentence(speaker, alone, ["t", "e", "k", "n", "o", "l", "o", "ɡ"]) == ["dʒ"]        # a hard g instead of j
    assert lost_in_the_sentence(speaker, alone, ["a", "t", "e", "k", "n", "o", "l", "o", "dʒ", "i", "p", "e"]) == []     # neighbours' sounds do not count
    # a sound the voice never makes for the spelling is not blamed on the sentence; nor one the speaker does not make
    assert lost_in_the_sentence(["f", "a", "n", "d"], ["f", "a", "n", "t"], ["f", "a", "n", "t"]) == []
    assert lost_in_the_sentence(["s", "a"], ["s", "a", "k"], ["s", "a"]) == []


def test_a_word_is_found_in_its_sentence_with_the_other_readings_applied():
    fake = FakeLab({})
    fake.sentences = ["Asibadingi abaningi. Kungani udinga ungqongqoshe weScience and Technology? Phela isayensi iyisifundo kwezemfundo, ifundiswa khona."]
    fake.readings = {"Science": "Sayense", "Technology": "Technology"}
    context = fake._context("Technology")
    assert context.before == "Kungani udinga ungqongqoshe weSayense and " and context.after.startswith("? Phela isayensi")
    glued = fake._context("weScience")                                            # the prefix stays on, only the English word is respelled
    assert glued.before == "Kungani udinga ungqongqoshe we" and "and Technology?" in glued.after
    assert fake._context("Gauteng") is None and fake.sentence_shown("Gauteng") is None
    assert fake.sentence_shown("Technology").startswith("Kungani udinga ungqongqoshe weSayense and [the spelling]? Phela")
    assert PronunciationLab("/x", voice="v", key="k", region="r", work_dir="/tmp/none").sounds_in_sentence("Technology", "Technology") is None


def test_a_spelling_that_loses_a_sound_in_its_sentence_is_not_the_answer():
    class Heard(FakeLab):
        def source(self, word):
            return {"found": True, "phones": ["t", "e", "k", "n", "o", "l", "o", "dʒ"], "occurrences": 2, "each_occurrence": []}

        def readings_in_sentence(self, word, spelling):
            return [{"Technology": ["t", "e", "k", "n", "o", "l", "o", "ɡ"], "Technolodgy": ["t", "e", "k", "n", "o", "l", "o", "dʒ", "i"]}[spelling]]

    heard = Heard({"Technology": ["t", "e", "k", "n", "o", "l", "o", "dʒ", "i"], "Technolodgy": ["t", "e", "k", "n", "o", "l", "o", "dʒ", "i"]})
    heard.sentences = ["Kungani udinga weScience and Technology? Phela."]
    out = heard.try_spellings("Technology", ["Technology", "Technolodgy"])
    plain, respelled = out["results"]
    assert plain["lost_in_its_sentence"] == ["dʒ"] and "lost_in_its_sentence" not in respelled and plain["said_in_its_sentence"].endswith("ɡ")
    assert out["closest_so_far"]["spelling"] == "Technolodgy" and "'Technology' loses dʒ" in out["advice"]
    assert out["also_heard_in_this_sentence"].endswith("[the spelling]? Phela.")


def test_a_saved_reading_that_loses_a_sound_in_its_sentence_goes_back_on_the_list():
    from mathula_tv.pronunciation_lab import failing_readings

    class Heard(FakeLab):
        def source(self, word):
            return {"found": True, "phones": ["t", "e", "k", "n", "o", "l", "o", "dʒ"], "occurrences": 2, "each_occurrence": []}

        def reading_understood(self, word, spelling):
            return True

        def voice_reading(self, spelling):
            return ["t", "e", "k", "n", "o", "l", "o", "dʒ", "i"]

        def readings_in_sentence(self, word, spelling):
            return [["t", "e", "k", "n", "o", "l", "o", "ɡ"] if spelling == "Technology" else ["t", "e", "k", "n", "o", "l", "o", "dʒ", "i"]]

    heard = Heard({})
    heard.sentences = ["Kungani udinga weScience and Technology? Phela."]
    entries = [{"display_text": "Technology", "tts_text": "Technology", "source": "agent", "kind": "english_code_switch"}]
    assert failing_readings(heard, entries) == {"technology"}                           # understood, but said with a g where the speaker said j
    entries[0]["tts_text"] = "Technolodgy"
    assert failing_readings(heard, entries) == set()
    heard.sentences = []                                                                # in no sentence: nothing to hear it in, so no complaint
    entries[0]["tts_text"] = "Technology"
    assert failing_readings(heard, entries) == set()


def test_the_lab_shows_which_spelling_keeps_each_sound_so_the_letters_can_be_put_together():
    class Heard(FakeLab):
        def source(self, word):
            return {"found": True, "phones": ["t", "e", "k", "n", "o", "l", "o", "dʒ"], "occurrences": 2, "each_occurrence": []}

        def readings_in_sentence(self, word, spelling):
            return [{"Technology": ["t", "e", "k", "n", "o", "l", "o"], "Teknolodji": ["t", "e", "n", "o", "l", "o", "dʒ"]}[spelling]]

    heard = Heard({"Technology": ["t", "e", "k", "n", "o", "l", "o", "dʒ", "i"], "Teknolodji": ["t", "e", "k", "n", "o", "l", "o", "tʃ", "i"]})
    heard.sentences = ["Kungani udinga weScience and Technology? Phela."]
    advice = heard.try_spellings("Technology", ["Technology", "Teknolodji"])["advice"]
    assert "k: kept by 'Technology'; lost by 'Teknolodji'" in advice and "dʒ: kept by 'Teknolodji'; lost by 'Technology'" in advice
    assert "Put the letters that keep each sound together" in advice


def test_choosing_a_spelling_that_loses_a_sound_in_its_sentence_is_answered_once_before_it_is_accepted(tmp_path):
    class Heard(FakeLab):
        def source(self, word):
            return {"found": True, "phones": ["f", "aɪ", "t", "ə", "z"], "occurrences": 2, "each_occurrence": []}

        def readings_in_sentence(self, word, spelling):
            return [["f", "aɪ"]]

    workbench, _ = make_workbench(tmp_path, {})
    heard = Heard({"Fayitas": ["f", "aɪ", "t", "ə", "z"]})
    heard.sentences = ["MaFighter, sifuna ukuthi ogogo bangakhokhi."]
    workbench._pronunciation_lab = heard
    heard.tried["Fighters"] = {"Fayitas": {"spelling": "Fayitas", "closeness": 0.9, "voice_sounds": "f aɪ t ə z", "lost_in_its_sentence": ["t", "z"],
                                           "said_in_its_sentence": "f aɪ"}}
    with pytest.raises(ValueError, match="loses t z"):
        workbench.use_pronunciation("Fighters", "Fayitas")
    assert workbench.use_pronunciation("Fighters", "Fayitas")["ok"] is True                 # asked again: the agent's decision


def test_a_sound_the_recogniser_labels_differently_from_one_cut_to_the_next_is_not_called_lost():
    from mathula_tv.pronunciation_lab import kept_in_the_sentence, lost_in_the_sentence

    speaker = ["t", "e", "k", "n", "o", "l", "o", "dʒ"]
    alone = ["t", "e", "k", "n", "o", "l", "o", "dʒ", "i"]
    half_voiced = [["t", "e", "k", "n", "o", "l", "o", "dʒ"], ["t", "e", "k", "n", "o", "l", "o", "tʃ"], ["t", "e", "k", "n", "o", "l", "o", "dʒ"], ["t", "e", "k", "n", "o", "l", "o", "dʒ", "p", "e"], ["t", "e", "k", "n", "o", "l", "o", "dʒ"]]
    hard_g = [["t", "e", "k", "n", "o", "l", "o"], ["t", "e", "k", "n", "o", "l", "o", "ɡ"], ["t", "e", "k", "n", "o", "l", "o", "ɡ"], ["t", "e", "k", "n", "o", "l", "o", "dʒ", "p", "e"], ["t", "e", "k", "n", "o", "l", "o", "k"]]
    assert lost_in_the_sentence(speaker, alone, half_voiced) == [] and "dʒ" in kept_in_the_sentence(speaker, half_voiced)       # 4 of 5 have it
    assert lost_in_the_sentence(speaker, alone, hard_g) == ["dʒ"] and "dʒ" not in kept_in_the_sentence(speaker, hard_g)         # 1 of 5 has it
    assert lost_in_the_sentence(speaker, alone, hard_g[0]) == ["dʒ"]                                                            # one decode still works as before


def test_the_same_synthesised_sentence_is_cut_with_several_margins(tmp_path):
    import json
    import wave

    from mathula_tv import pronunciation_search as ps

    def synthesize(marked, pitch, wav):
        with wave.open(str(wav), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(1000)
            w.writeframes(b"\x00\x00" * 5000)                                                 # 5 s at 1 kHz: one sample per millisecond
        return {"n0": 2000.0, "n1": 2500.0}

    windows = ps.NameWindows(synthesize, lambda wav: (), tmp_path)
    cuts = windows.audio_with_margins(ps.Context(before="a ", after=".", pitch="+0%"), ps.Candidate("word"), ((40, 120), (1000, 1000)))
    sizes = []
    for cut in cuts:
        with wave.open(str(cut), "rb") as w:
            sizes.append(w.getnframes())
    assert sizes == [500 + 40 + 120, 500 + 2000] and windows.syntheses == 1                          # one synthesis, two cuts


def test_voiced_and_voiceless_partners_count_as_one_sound_when_the_recogniser_labels_are_voted_on():
    from mathula_tv.pronunciation_lab import kept_in_the_sentence, lost_in_the_sentence

    speaker = ["m", "e", "d", "i", "k", "a", "l"]
    alone = ["m", "e", "d", "i", "k", "a", "l"]
    flips = [["m", "e", "t", "i", "k", "a", "l"], ["m", "e", "t", "i", "ɡ", "a", "l"], ["m", "e", "d", "i", "k", "a", "l"]]
    assert lost_in_the_sentence(speaker, alone, flips) == [] and {"d", "k"} <= set(kept_in_the_sentence(speaker, flips))        # d/t and k/ɡ are voicing only
    assert lost_in_the_sentence(speaker, alone, [["m", "e", "i", "a", "l"]]) == ["d", "k"]                                        # a sound that is gone is gone


def test_the_zulu_prefix_glued_on_an_english_word_is_not_what_is_judged():
    from mathula_tv.pronunciation_lab import without_prefix, written_prefix

    assert written_prefix("MaFighter") == "Ma" and written_prefix("weScience") == "we" and written_prefix("ITechnology") == "I"
    assert written_prefix("Technology") == "" and written_prefix("ne-medical") == ""
    assert without_prefix("maFayitha", "Ma") == "Fayitha" and without_prefix("Fayitas", "Ma") == "Fayitas"       # a spelling without it is judged as it is
    assert without_prefix("weSayensi", "we") == "Sayensi" and without_prefix("Ma", "Ma") == "Ma"

    class Heard(FakeLab):
        def source(self, word):
            return {"found": True, "phones": ["f", "aɪ", "t", "a", "s"], "occurrences": 2, "each_occurrence": []}

        def english_hears(self, spelling):
            return {"Fayitas": "Fighters.", "maFayitas": "My fighters."}[spelling]

    heard = Heard({"Fayitas": ["f", "aɪ", "t", "a", "s"]})
    out = heard.try_spellings("MaFighter", ["maFayitas"])
    record = out["results"][0]
    assert record["spelling"] == "maFayitas" and record["voice_sounds"] == "f aɪ t a s" and record["english_hears"] == "Fighters."
    assert record["closeness"] == 1.0                                              # the "ma" is not counted as a sound the voice adds


def test_the_speakers_word_is_found_under_any_glued_prefix_and_pause_markers_are_not_spoken():
    fake = FakeLab({})
    fake._transcript = [{"text": "science"}, {"text": "Fighters,"}]
    fake._source_cache.clear()
    assert fake._apply_readings("~pause800 Ma", "x") == "Ma" and fake._apply_readings("Leta ~en 5 amadoda", "x") == "Leta 5 amadoda"


def test_matching_sounds_with_a_listener_that_wrote_another_word_are_put_to_the_agent_as_two_kinds_of_evidence():
    class Listener(FakeLab):
        def source(self, word):
            return {"found": True, "phones": ["f", "aɪ", "t", "a", "s"], "occurrences": 2, "each_occurrence": []}

        def english_hears(self, spelling):
            return "Fight us."

    out = Listener({"Fayitas": ["f", "aɪ", "t", "a", "s"]}).try_spellings("Fighters", ["Fayitas"])
    assert "match the speaker's (closeness 1.0)" in out["advice"] and "weigh the two" in out["advice"] and "Change the sounds" not in out["advice"]
    far = Listener({"Fayitas": ["m", "a", "f", "e", "k", "t", "o", "r"]}).try_spellings("Fighters", ["Fayitas"])
    assert "No spelling is understood yet" in far["advice"]


def test_the_word_as_the_zulu_writes_it_is_measured_without_spending_a_try():
    class Heard(FakeLab):
        def source(self, word):
            return {"found": True, "phones": ["f", "aɪ", "t", "a", "s"], "occurrences": 2, "each_occurrence": [], "spoken_form": "Fighters"}

        def english_hears(self, spelling):
            return {"Fighter": "Theta."}[spelling]

    heard = Heard({"Fighter": ["f", "eɪ", "t", "ə"]})
    row = heard.as_written("MaFighter")                                              # the prefix "Ma" is not judged
    assert row["voice_sounds"] == "f eɪ t ə" and row["speaker_sounds"] == "f aɪ t a s" and row["english_hears"] == "Theta."
    assert row["closeness"] < 0.8 and any("vowel" in d for d in row["differences"]) and heard.total_tries == 0
    assert FakeLab({}).as_written("Unspoken") is None


def test_the_acoustic_differences_in_the_sentence_are_shown_with_each_try_and_in_the_advice():
    class Measured(FakeLab):
        def source(self, word):
            return {"found": True, "phones": ["f", "aɪ", "t", "a", "s"], "occurrences": 2, "each_occurrence": []}

        def acoustics_in_sentence(self, word, spelling, source):
            return ["the voice's stop number 7 (tʃ), against 2 of the speaker's occurrence(s): the vocal folds vibrate through 0% of the closure against the speaker's 100%"]

        def readings_in_sentence(self, word, spelling):
            return [["f", "aɪ", "t", "a", "s"]]

    out = Measured({"Fayitas": ["f", "aɪ", "t", "a", "s"]}).try_spellings("Fighters", ["Fayitas"])
    assert out["results"][0]["how_the_sounds_are_made_in_its_sentence"][0].startswith("the voice's stop number 7 (tʃ)")
    assert "Measured in its sentence ('Fayitas'; the median over several cuts of the same audio)" in out["advice"]


def test_a_saved_reading_is_measured_in_its_sentence_without_spending_a_try(tmp_path):
    from mathula_tv.pronunciation_memory import memory_path, remember

    class Saved(FakeLab):
        def source(self, word):
            return {"found": True, "phones": ["s", "aɪ", "n", "ts", "ə"], "occurrences": 2, "each_occurrence": []}

        def readings_in_sentence(self, word, spelling):
            return None

        def english_hears(self, spelling):
            return "Science."

    workbench, _ = make_workbench(tmp_path, {})
    saved = Saved({"Sayensi": ["s", "aɪ", "j", "ɪ", "n"]})
    workbench._pronunciation_lab = saved
    run_dir = tmp_path / "x"
    import json
    from pathlib import Path

    Path(workbench._episode()["dir"], "submission.json").write_text(json.dumps({"items": [{"item_id": "u1", "zulu_text": "Phela weScience iyisifundo kakhulu."}]}))
    remember(memory_path(tmp_path), "Science", "Sayensi", source="agent", voice="zu-ZA-ThembaNeural", job_id="j", kind="english_code_switch")
    remember(memory_path(tmp_path), "Gauteng", "Gautheng", source="agent", voice="zu-ZA-ThembaNeural", job_id="j", kind="english_code_switch")    # not in this text
    remember(memory_path(tmp_path), "hlu", "hluuu", source="repair", voice="zu-ZA-ThembaNeural", job_id="j", kind="other")                        # not an English word
    rows = workbench.saved_readings_evidence()
    assert [(r["word"], r["spelling"]) for r in rows] == [("Science", "Sayensi")]
    assert rows[0]["voice_sounds"] == "s aɪ j ɪ n" and rows[0]["closeness"] < 0.8 and saved.total_tries == 0


def test_when_the_listener_and_the_sounds_point_at_different_spellings_both_are_shown_and_neither_leads():
    class Disagree(ListenerLab):
        def source(self, word):
            return {"found": True, "phones": ["f", "aɪ", "t", "a", "s"], "occurrences": 2, "spoken_form": "Fighters", "each_occurrence": []}

    lab_ = Disagree({"Fayitas": ["f", "aɪ", "t", "a", "s"], "Fayithaz": ["f", "a", "j", "iː", "t", "a", "z", "z"]},
                    {"Fayitas": "Fight us.", "Fayithaz": "Fighters."})
    out = lab_.try_spellings("Fighters", ["Fayitas", "Fayithaz"])
    assert out["closest_so_far"]["spelling"] == "Fayitas" and out["understood_by_the_listener"] == ["Fayithaz"]          # the listener does not outrank the sounds
    advice = out["advice"]
    assert "point at different spellings" in advice and "'Fight us.'" in advice and "Each check is blind somewhere" in advice
    assert "Use one of those" not in advice


def test_the_lab_says_which_units_speak_a_word():
    fake = FakeLab({})
    fake.sentences = ["Phela weScience iyisifundo.", "Sifuna ukwazi.", "Ngithi iScience nayo."]
    assert fake.units_with("Science", ["u1", "u2", "u3"]) == ["u1", "u3"] and fake.units_with("Gauteng", ["u1", "u2", "u3"]) == []


def test_a_changed_reading_starts_a_new_folder_for_measured_speech(tmp_path):
    workbench, _ = make_workbench(tmp_path, {"Fayitas": ["f", "aɪ", "t", "ə", "z"], "Fayitha": ["f", "aɪ", "t", "ə", "z"]})
    first = workbench._measure_dir()
    workbench.try_pronunciation("Fighters", ["Fayitas", "Fayitha"])
    workbench.use_pronunciation("Fighters", "Fayitas")
    second = workbench._measure_dir()
    workbench.use_pronunciation("Fighters", "Fayitha")
    third = workbench._measure_dir()
    assert first.name == "measure" and second == first / "readings-1" and third == first / "readings-2"


def test_suggested_spellings_are_spoken_measured_and_ready_to_use_and_cost_one_try():
    from mathula_tv.pronunciation_lab import MAX_TRIES_PER_WORD

    table = {"graphemes": {"f": ["f"], "ai": ["aɪ"], "ay": ["aɪ"], "t": ["t"], "a": ["a"], "s": ["s"], "z": ["z"], "i": ["i"]}}

    class Suggest(FakeLab):
        def source(self, word):
            return {"found": True, "phones": ["f", "aɪ", "t", "a", "s"], "occurrences": 2, "spoken_form": "Fighters", "each_occurrence": []}

        def _letter_table(self):
            return table

        def voice_reading(self, spelling):                      # a toy voice: aɪ for ai/ay, one sound per other letter
            sounds, text = [], spelling.lower()
            while text:
                for letters in ("ai", "ay"):
                    if text.startswith(letters):
                        sounds.append("aɪ")
                        text = text[2:]
                        break
                else:
                    sounds.append(text[0])
                    text = text[1:]
            return sounds

        def english_hears(self, spelling):
            return "Fighters."

    lab_ = Suggest({})
    out = lab_.suggest_spellings("MaFighter")
    assert out["results"] and len(out["results"]) <= 5 and "different sound sequences" in out["how_these_were_found"] and "varied by small edits" in out["how_these_were_found"]
    first = out["results"][0]
    assert first["spelling"].startswith("Ma") and first["closeness"] == 1.0 and first["english_hears"] == "Fighters."          # the Zulu prefix is kept on the spelling
    assert len({r["voice_sounds"] for r in out["results"]}) == len(out["results"])                                        # different sounds, not the same idea twice
    assert lab_.tries_per_word["MaFighter"] == 1 and out["tries_left_for_word"] == MAX_TRIES_PER_WORD - 1
    assert first["spelling"] in lab_.tried["MaFighter"]                                                                   # so use_pronunciation accepts it


def test_suggestions_include_the_spellings_that_open_the_mouth_like_the_speaker_not_only_the_closest_in_sounds():
    table = {"graphemes": {"s": ["s"], "n": ["n"], "ts": ["ts"], "a": ["a"], "ai": ["aɪ"], "ei": ["eɪ"], "ay": ["aɪ"]}}

    class Opening(FakeLab):
        openings = {"ay": 960.0, "ai": 560.0, "ei": 380.0}

        def source(self, word):
            return {"found": True, "phones": ["s", "aɪ", "n", "ts", "a"], "occurrences": 1, "spoken_form": "science", "each_occurrence": []}

        def _letter_table(self):
            return table

        def voice_reading(self, spelling):
            text, out = spelling.lower(), []
            for letters, sound in (("ts", "ts"), ("ai", "aɪ"), ("ay", "aɪ"), ("ei", "eɪ"), ("aa", "a")):
                text = text.replace(letters, "|" + sound + "|")
            for part in text.split("|"):
                out += [part] if part in ("ts", "aɪ", "eɪ", "a") else list(part)
            return out

        def _speaker_opening(self, source):
            return 950.0

        def first_vowel_opening(self, spelling):
            return next((value for letters, value in self.openings.items() if letters in spelling.lower()), 700.0)

    lab_ = Opening({})
    shown = [r["spelling"] for r in lab_.suggest_spellings("science")["results"]]
    assert len(shown) <= 5 and any("ay" in spelling.lower() for spelling in shown)                    # a spelling that opens the mouth like the speaker is measured


def test_a_spelling_is_heard_followed_by_a_word_and_the_following_words_sounds_are_cut_off():
    from pathlib import Path

    class Fake(PronunciationLab):
        gop = type("Gop", (), {"decode_timed": staticmethod(lambda wav: [("s", 0.0, 0.02), ("aɪ", 0.08, 0.1), ("n", 0.2, 0.22), ("ts", 0.3, 0.32),
                                                                         ("a", 0.4, 0.42), ("k", 0.62, 0.64), ("a", 0.7, 0.72)])})()

        def _audio_for(self, spelling):
            self._word_end["x.wav"] = 0.5                    # the spelling's text ends 0.5 s into the clip
            return "x.wav"

    lab_ = Fake("/nonexistent", voice="v", key="k", region="r", work_dir="/tmp/none")
    assert lab_.voice_reading("Saensa") == ["s", "aɪ", "n", "ts", "a"]                  # the final vowel is kept, the next word's k and a are not
    from mathula_tv import pronunciation_lab as module

    assert module.STANDALONE_AFTER.startswith(" ") and module.STANDALONE_AFTER.strip(". ") != ""     # a word follows: the utterance does not end on the spelling


def test_a_word_listed_for_checking_carries_the_everyday_zulu_word_for_it():
    words = [{"text": "science"}, {"text": "technology"}]
    named = [{"written": "weScience", "english": "science", "kind": "word", "zulu": "isayensi"},
             {"written": "Technology", "english": "technology", "kind": "word", "zulu": "ubuchwepheshe"}]
    found = candidate_words(["Kungani udinga weScience and Technology?"], words, looks_isizulu=looks_isizulu, named=named)
    assert {f["speaker_says"]: f.get("zulu_word") for f in found} == {"science": "isayensi", "technology": "ubuchwepheshe"}
    none = candidate_words(["weScience"], words, looks_isizulu=looks_isizulu, named=[{"written": "weScience", "english": "science", "kind": "word", "zulu": ""}])
    assert "zulu_word" not in none[0]

def test_loanwords_listed_are_only_english_nouns_the_speaker_says_one_per_word(tmp_path):
    import json
    from pathlib import Path

    workbench, _ = make_workbench(tmp_path, {})
    Path(workbench._episode()["dir"], "submission.json").write_text(json.dumps({"items": [{"item_id": "u1", "zulu_text": "weScience ne-medical ama-benefit ma-benefit aid and Malema"}]}))

    class Judge:
        def foreign_words(self, texts):
            return [{"written": "weScience", "english": "Science", "kind": "word", "zulu": "isayensi"},
                    {"written": "ne-medical", "english": "medical", "kind": "word", "zulu": ""},
                    {"written": "ama-benefit", "english": "benefit", "kind": "word", "zulu": "inzuzo"},
                    {"written": "ma-benefit", "english": "benefit", "kind": "word", "zulu": "inzuzo"},
                    {"written": "aid", "english": "aid", "kind": "word", "zulu": "usizo"},
                    {"written": "Malema", "english": "Malema", "kind": "name", "zulu": ""}]

    workbench.judge = Judge()
    workbench._pronunciation_lab._transcript = [{"text": "science"}, {"text": "medical"}, {"text": "benefits"}, {"text": "aid"}, {"text": "Malema"}]
    found = workbench.loanwords()
    assert [(item["english"], item["zulu"]) for item in found] == [("Science", "isayensi"), ("medical", ""), ("benefit", "inzuzo")]       # "medical" has no Zulu word: listed so one can be created; not "aid" (too short)
    workbench.judge = None
    assert workbench.loanwords() == []


def test_a_final_vowel_the_recogniser_missed_is_put_back_in_the_voices_sounds_and_a_mismatched_ending_is_named():
    class Ends(FakeLab):
        def source(self, word):
            return {"found": True, "phones": ["f", "aɪ", "t", "a", "s"], "occurrences": 2, "each_occurrence": []}

        def voice_reading(self, spelling):
            heard = ["f", "aɪ", "t", "a", "s"]
            ending = self._ending.get(spelling)
            return heard + ["ə"] if ending and ending["present"] and not heard[-1] in ("a", "ə") else heard

    lab_ = Ends({})
    lab_._ending["Faitasi"] = {"present": True, "ms": 70}
    lab_._ending["Faitas"] = {"present": False, "ms": 0}
    record = lab_._measure("Fighters", "Faitasi", lab_.source("Fighters"))
    assert record["voice_sounds"].endswith("s ə") and "ends the word with a vowel of about 70 ms" in record["ending"] and "consonant s" in record["ending"]
    assert "ending" not in lab_._measure("Fighters", "Faitas", lab_.source("Fighters"))                 # the same ending as the speaker's: nothing to say

    class Science(FakeLab):
        def source(self, word):
            return {"found": True, "phones": ["s", "aɪ", "n", "ts", "ə"], "occurrences": 2, "each_occurrence": []}

    s_ = Science({})
    s_._ending["Seins"] = {"present": False, "ms": 0}
    assert "ends at the last s with no vowel after it" in s_.ending_note("Science", "Seins", s_.source("Science"))
    s_._ending["Seins"] = None
    assert s_.ending_note("Science", "Seins", s_.source("Science")) is None                              # a word with no hiss to anchor on: no claim


def test_a_reading_chosen_for_a_word_with_a_zulu_prefix_keeps_the_prefix(tmp_path):
    from mathula_tv.pronunciation_memory import entries_for, memory_path

    class Heard(FakeLab):
        def source(self, word):
            return {"found": True, "phones": ["s", "aɪ", "n", "ts", "ə"], "occurrences": 2, "each_occurrence": []}

    workbench, _ = make_workbench(tmp_path, {})
    workbench._pronunciation_lab = Heard({"saynsi": ["s", "aɪ", "n", "ts", "ɪ"], "Saynsi": ["s", "aɪ", "n", "ts", "ɪ"]})
    workbench.try_pronunciation("weScience", ["saynsi", "weSaynsi"])
    out = workbench.use_pronunciation("weScience", "saynsi")
    assert out["spelling"] == "wesaynsi" and "prefix 'we' stays" in out["note"]                                      # only the English part was tried: "we" is kept
    assert [e["tts_text"] for e in entries_for(memory_path(tmp_path), "zu-ZA-ThembaNeural") if e["display_text"] == "weScience"] == ["wesaynsi"]
    assert workbench.use_pronunciation("weScience", "weSaynsi")["spelling"] == "weSaynsi"                         # it already carries the prefix: as chosen


def test_a_zulu_word_is_created_for_an_english_word_from_the_speakers_sounds_spoken_and_remembered(tmp_path):
    from mathula_tv.pronunciation_lab import MAX_TRIES_PER_WORD

    class Loan(FakeLab):
        def source(self, word):
            return {"found": True, "phones": ["b", "ʌ", "s"], "occurrences": 2, "each_occurrence": []}

        def voice_reading(self, spelling):
            return ["b", "a", "s", "i"] if "bh" in spelling else ["p", "a", "s", "i"]

        def zulu_hears(self, spelling):
            return "Bhasi." if spelling == "bhasi" else ""

    lab_ = Loan({})
    out = lab_.create_loanword("bus")
    first = out["zulu_words"][0]
    assert first["zulu_word"] == "bhasi" and first["with_i"] == "ibhasi"
    assert first["a_zulu_listener_writes"] == "Bhasi." and first["zulu_listener_similarity"] == 1.0 and out["zulu_words"][1]["zulu_listener_similarity"] == 0.0
    assert len({row["voice_sounds"] for row in out["zulu_words"]}) == len(out["zulu_words"])         # different sounds, not the same idea twice
    assert lab_.tries_per_word["bus"] == 1 and out["tries_left_for_word"] == MAX_TRIES_PER_WORD - 1 and "use_loanword" in out["next"]
    assert "bhasi" in lab_._created["bus"]
    assert "never says" in FakeLab({}).create_loanword("Unspoken")["error"]


def test_a_chosen_loanword_is_remembered_for_later_clips_but_only_if_it_was_created(tmp_path):
    from mathula_tv.created_loanwords import path_for, zulu_for

    workbench, _ = make_workbench(tmp_path, {})

    class Loan(FakeLab):
        def source(self, word):
            return {"found": True, "phones": ["b", "ʌ", "s"], "occurrences": 2, "each_occurrence": []}

        def voice_reading(self, spelling):
            return ["b", "a", "s", "i"]

    workbench._pronunciation_lab = Loan({})
    with pytest.raises(ValueError, match="create 'bhasi'"):
        workbench.use_loanword("bus", "bhasi")
    workbench.create_loanword("bus")
    prefixed = workbench.use_loanword("bus", "ibhasi")                                   # the form with its noun prefix is the word made
    assert prefixed["zulu_word"] == "bhasi"
    out = workbench.use_loanword("bus", "bhasi")
    assert out["ok"] and "needs no respelling" in out["note"]
    assert zulu_for(path_for(tmp_path), "Bus") == "bhasi" and zulu_for(path_for(tmp_path), "train") is None


def test_the_recognisers_words_take_the_restorers_word_for_word_corrections_and_keep_the_timing():
    heard = [{"text": t, "start": float(i)} for i, t in enumerate(
        ["Must", "have", "provident", "fund", "Must", "have", "patient", "fund.", "near", "the", "Leopard", "creed", "today"])]
    out = lab.restore_words(heard, [
        {"original_text": "Must have patient fund.", "corrected_text": "Must have pension fund."},
        {"original_text": "near the Leopard creed today", "corrected_text": "near the Leopard Creek today"},
        {"original_text": "Must have provident", "corrected_text": "Must really have provident"}])           # a different number of words: left as heard
    assert [w["text"] for w in out] == ["Must", "have", "provident", "fund", "Must", "have", "pension", "fund.", "near", "the", "Leopard", "Creek", "today"]
    assert out[6]["heard_as"] == "patient" and out[6]["start"] == 6.0 and "heard_as" not in out[0]
    assert lab.restore_words(heard, [{"original_text": "never said this", "corrected_text": "x"}]) == heard


def test_the_length_guide_gives_each_segment_the_zulu_length_that_fills_its_window_for_a_voice_it_knows():
    from mathula_tv.dub_workbench import length_guide

    segments = [{"segment_id": "s1", "speaker_id": "A", "start_ms": 0, "end_ms": 10000},
                {"segment_id": "s2", "speaker_id": "B", "start_ms": 10000, "end_ms": 12000},
                {"segment_id": "s3", "speaker_id": "A", "start_ms": 12000, "end_ms": 12000}]
    voices = {"A": {"selected_voice": "zu-ZA-ThembaNeural"}, "B": {"selected_voice": "zu-ZA-SomeOtherNeural"}}
    assert length_guide(segments, voices) == {"s1": 125}                         # 10 s x 12.4 = 124 -> 125; unknown voice and empty window: no guide
