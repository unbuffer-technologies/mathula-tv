from mathula_tv import speaker_fillers as sf
from mathula_tv.dub_agent import FOCUS_NOTE, focus_budget_note


def _segments():
    return [
        {"speaker_id": "SPEAKER_02", "restored_text": "You know, we did this. Honestly, it worked. In fact, it worked well."},
        {"speaker_id": "SPEAKER_02", "restored_text": "You know, the city is ours. Honestly, we are ready. Of course we are."},
        {"speaker_id": "SPEAKER_02", "restored_text": "Fortunately enough, we won. Fortunately, nobody left."},
        {"speaker_id": "SPEAKER_01", "restored_text": "Thank you. Where do we start? Tell me about it."},
    ]


def test_the_markers_a_speaker_uses_are_counted_from_their_own_english_and_nobody_elses():
    found = sf.fillers_by_speaker(_segments())
    mine = {x["phrase"]: x["count"] for x in found["SPEAKER_02"]}
    assert mine["you know"] == 2 and mine["honestly"] == 2
    assert "in fact" not in mine and "very" not in mine                                             # said once by a speaker of many sentences: a habit needs two uses
    assert found["SPEAKER_01"] == []                                                                 # a speaker who says none gets none, not a default list
    assert all(x["example"] for x in found["SPEAKER_02"])


def test_a_longer_phrase_is_not_counted_again_as_its_shorter_one():
    counts = {x["phrase"]: x["count"] for x in sf.fillers_by_speaker(_segments(), min_count=1)["SPEAKER_02"]}
    assert counts.get("fortunately enough") == 1 and counts.get("fortunately") == 1                  # one of each, not two "fortunately"


def test_the_budget_says_how_many_letters_close_each_gap_and_names_the_speakers_own_padding():
    items = [{"item_id": "u-04", "speaker_id": "SPEAKER_02", "start_ms": 0, "end_ms": 24000, "zulu_text": "a" * 200},
             {"item_id": "u-05", "speaker_id": "SPEAKER_01", "start_ms": 24000, "end_ms": 36000, "zulu_text": "b" * 100},
             {"item_id": "u-06", "speaker_id": "SPEAKER_01", "start_ms": 36000, "end_ms": 40000, "zulu_text": "c" * 50}]
    estimate = lambda text, speaker, start, end: {"u": None}.get("u") or (len(text) * 100 if text[0] == "a" else len(text) * 125 if text[0] == "b" else 4100)         # 20 s, 12.5 s, 4.1 s
    rows = sf.budget_rows(items, ["u-04", "u-05", "u-06"], estimate)
    by = {r["item_id"]: r for r in rows}
    assert by["u-04"]["gap_s"] == 4.0 and by["u-04"]["letters_to_change"] == 40                       # 10 letters a second: 4 s is 40 letters
    assert by["u-05"]["gap_s"] == -0.5 and by["u-05"]["letters_to_change"] < 0                        # over its window: cut
    note = sf.budget_note(rows, sf.fillers_by_speaker(_segments()))
    assert "u-04" in note and "ADD about 40 letters" in note and "CUT about" in note and "'you know' x2" in note
    assert "u-06" in note and "leave it" not in note.split("u-04")[0]                                  # the unit at its edge is told to be left
    assert "ONE check_texts" in note and "do not invent filler" in note


def test_the_task_message_carries_the_budget_and_a_failure_to_build_it_leaves_the_message_alone(capsys):
    class Broken:
        segments = None

        def _text_ms(self, *a):
            raise RuntimeError("no voice")

    assert focus_budget_note(Broken(), [{"item_id": "u-01", "speaker_id": "S", "start_ms": 0, "end_ms": 1000, "zulu_text": "x"}], ["u-01"]) == ""
    assert "could not be built" in capsys.readouterr().out                                          # and it says so
    assert "never over" in FOCUS_NOTE and "half a second" in FOCUS_NOTE


def test_the_conversational_fallback_is_read_from_the_two_corpora_and_leaves_out_english_names_and_fragments(tmp_path):
    import json

    (tmp_path / "code_switching" / "soap_opera").mkdir(parents=True)
    (tmp_path / "spoken").mkdir()
    rows = []
    for _ in range(30):
        rows.append({"segments": [{"lang": "zul", "text": "kodwa ke manje sesihlukene"}]})                  # "ke": said by speakers in conversation
        rows.append({"segments": [{"lang": "zul", "text": "uyazi ucaleb"}, {"lang": "eng", "text": "okay sure"}]})
        rows.append({"segments": [{"lang": "zul", "text": "angikho e_ mood _ini"}]})                      # "e" and "ini" are bound pieces of a word
    (tmp_path / "code_switching" / "soap_opera" / "balanced_engzul.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    structured = [{"transcript": "ukuthi izinkomo " * 500 + "ke uyazi okay e ini"}] * 100                                # the structured speech: "ke" and "uyazi" are rare there
    structured += [{"transcript": "ke uyazi okay e ini"}] * 45
    (tmp_path / "spoken" / "anv_zul.jsonl").write_text("\n".join(json.dumps(r) for r in structured), encoding="utf-8")
    found = {x["word"]: x for x in sf.conversational_fillers(tmp_path)}
    assert "ke" in found and "uyazi" in found                                                       # said in conversation, rare in the talks
    assert "okay" not in found                                                                      # tagged English
    assert "ucaleb" not in found                                                                    # a name: not common in the structured speech
    assert "e" not in found and "ini" not in found                                                  # fragments
    assert found["ke"]["example"].startswith("kodwa ke manje")
    assert sf.conversational_fillers(tmp_path / "missing") == []                                    # no corpus: no fallback, never an error


def test_a_speaker_with_few_filling_words_of_their_own_is_shown_what_conversation_says():
    rows = [{"item_id": "u-05", "speaker_id": "SPEAKER_01", "window_s": 10.0, "estimated_s": 8.0, "gap_s": 2.0, "letters_now": 80, "letters_to_change": 20, "letters_within": 5}]
    conversation = [{"word": "ke", "count": 161, "ratio": 58.2, "example": "kodwa ke manje sesihlukene"}]
    note = sf.budget_note(rows, {"SPEAKER_01": []}, conversation)
    assert "no filling words of their own" in note and "'ke' x161" in note and "kodwa ke manje" in note
    plenty = {"SPEAKER_01": [{"phrase": p, "count": 3, "example": "x"} for p in ("you know", "honestly", "obviously")]}
    assert "conversation say" not in sf.budget_note(rows, plenty, conversation)                     # three of their own: their own are used
