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
            return {("sick", "sick"): False, ("fund", "fund"): True, ("deputy", "dephuthi"): None, ("Fighters", "Fayitas"): False}[(word, spelling)]

    entries = [
        {"display_text": "sick", "tts_text": "sick", "source": "agent", "kind": "english_code_switch"},
        {"display_text": "fund", "tts_text": "fund", "source": "agent", "kind": "english_code_switch"},
        {"display_text": "deputy", "tts_text": "dephuthi", "source": "agent", "kind": "english_code_switch"},          # could not be asked: fine
        {"display_text": "Fighters", "tts_text": "Fayitas", "source": "person", "kind": "english_code_switch"},        # chosen by ear: checked like any other
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
