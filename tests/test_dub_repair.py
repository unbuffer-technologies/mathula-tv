import json

from mathula_tv.dub_repair import REPAIR_SOURCE, add_to_lexicon, candidates, find_respelling, with_candidate

SPOKEN = "Bethola ukuthi sebesusiwe kulolo hlu. Kukhona abazokwenqatshwa yimiphakathi."
SSML = '<speak><voice name="zu-ZA-ThembaNeural"><prosody pitch="-6%">Bethola ukuthi sebesusiwe kulolo hlu. Kukhona abazokwenqatshwa yimiphakathi.</prosody></voice></speak>'
ITEM = {"words": ["hlu."], "after": "kulolo", "before": "Kukhona"}


def words(*items):
    return [{"text": t, "start": s, "end": e} for t, s, e in items]


def test_candidates_give_the_word_more_voiced_material_and_never_return_the_word_itself():
    assert candidates("hlu.") == ["hluu", "hluuu", "hhlu"]
    assert "hlu" not in candidates("hlu")


def test_only_the_dropped_occurrence_is_replaced():
    ssml = SSML.replace("yimiphakathi.", "yimiphakathi hlu.")
    assert "kulolo hluu. Kukhona" in with_candidate(ssml, "kulolo", "hlu.", "hluu") and ssml.count("hlu") == 2
    assert with_candidate(ssml, "kulolo", "hlu.", "hluu").count("hluu") == 1
    assert with_candidate("<speak>nothing</speak>", "kulolo", "hlu.", "hluu") is None


def test_the_first_candidate_that_brings_the_word_back_is_kept_and_the_tests_are_bounded():
    gap = words(("Bethola", 0, .4), ("ukuthi", .4, .8), ("sebesusiwe", .8, 1.6), ("kulolo", 1.6, 2.1), ("Kukhona", 3.2, 3.7),
                ("abazokwenqatshwa", 3.7, 4.6), ("yimiphakathi", 4.6, 5.4))
    said = words(("Bethola", 0, .4), ("ukuthi", .4, .8), ("sebesusiwe", .8, 1.6), ("kulolo", 1.6, 2.1), ("hluu", 2.1, 2.6),
                 ("Kukhona", 2.8, 3.3), ("abazokwenqatshwa", 3.3, 4.2), ("yimiphakathi", 4.2, 5.0))
    trials = []

    def synthesize(text, path):
        trials.append(text)
        return path

    transcripts = iter([gap, said])
    result = find_respelling(ITEM, SPOKEN, SSML, lambda wav: next(transcripts), synthesize, __import__("pathlib").Path("."))
    assert result["fixed"] and result["respelling"] == "hluuu" and len(trials) == 2
    assert [t["still_dropped"] for t in result["tried"]] == [True, False]


def test_a_word_no_candidate_fixes_is_reported_not_retried_forever():
    gap = words(("Bethola", 0, .4), ("ukuthi", .4, .8), ("sebesusiwe", .8, 1.6), ("kulolo", 1.6, 2.1), ("Kukhona", 3.2, 3.7),
                ("abazokwenqatshwa", 3.7, 4.6), ("yimiphakathi", 4.6, 5.4))
    calls = []
    result = find_respelling(ITEM, SPOKEN, SSML, lambda wav: gap, lambda text, path: calls.append(text) or path, __import__("pathlib").Path("."))
    assert not result["fixed"] and len(calls) == 3


def test_repair_entries_are_added_to_the_lexicon_without_touching_the_names_stage_entries(tmp_path):
    path = tmp_path / "pronunciation_names.json"
    path.write_text(json.dumps({"schema": "x", "entries": [{"display_text": "Floyd Shivambu", "tts_text": "Floyd Shivambu", "kind": "personal_name"}]}))
    add_to_lexicon(path, [{"word": "hlu.", "respelling": "hluu"}])
    entries = json.loads(path.read_text())["entries"]
    assert [(e["display_text"], e["tts_text"]) for e in entries] == [("Floyd Shivambu", "Floyd Shivambu"), ("hlu", "hluu")]
    assert entries[1]["source"] == REPAIR_SOURCE
