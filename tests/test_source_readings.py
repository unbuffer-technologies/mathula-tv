from pathlib import Path

from mathula_tv import source_readings

SEGMENTS = [
    {"segment_id": "seg-1", "speaker_id": "A", "start_ms": 0, "end_ms": 5000, "source_text": "Good morning to you.", "restored_text": "Good morning to you."},
    {"segment_id": "seg-2", "speaker_id": "B", "start_ms": 6000, "end_ms": 12000, "source_text": "He would not be staying in jail for eight years.",
     "restored_text": "He would not be staying in jail for eight years.", "changed": False},
    {"segment_id": "seg-3", "speaker_id": "A", "start_ms": 13000, "end_ms": 18000, "source_text": "Him? Thanks, Pundo.", "restored_text": "Um, thanks, Pundo.", "changed": True, "reason_code": "asr_substitution"},
]
WORDS = [{"text": w, "start": 6.0 + i * 0.5, "end": 6.4 + i * 0.5, "confidence": c} for i, (w, c) in enumerate(
    [("He", 0.95), ("would", 0.9), ("not", 0.55), ("be", 0.92), ("staying", 0.88), ("in", 0.9), ("jail", 0.7), ("for", 0.9), ("eight", 0.93), ("years", 0.95)])]
CAPTIONS = [(5.8, 9.0, "he wouldn't be staying in jail"), (9.0, 12.0, "for eight years")]


def test_the_readings_of_a_segment_name_the_unsure_words_the_caption_and_where_they_differ():
    result = source_readings.build(SEGMENTS, words=WORDS, captions=CAPTIONS, track="en-orig", segment_id="seg-2")
    seg = result["segments"][0]
    assert seg["segment_id"] == "seg-2" and seg["restoration_changed_it"] is False and "recogniser_wrote" not in seg
    assert [w["word"] for w in seg["words_the_recogniser_was_unsure_of"]] == ["not", "jail"] and seg["words_the_recogniser_was_unsure_of"][0]["confidence"] == 0.55
    assert seg["youtube_caption (en-orig)"] == "he wouldn't be staying in jail for eight years"
    assert "not" in " ".join(seg["where_the_two_readings_differ"]["only_in_the_recogniser"]) or "would not" in " ".join(seg["where_the_two_readings_differ"]["only_in_the_recogniser"])
    assert seg["sentence_before"] == "Good morning to you." and seg["sentence_after"] == "Um, thanks, Pundo." and seg["mean_word_confidence"] == 0.86
    assert "cannot play the audio" in result["note"]


def test_a_changed_segment_shows_what_the_recogniser_wrote_and_why_it_was_changed():
    seg = source_readings.build(SEGMENTS, words=[], captions=[], track="", segment_id="seg-3")["segments"][0]
    assert seg["recogniser_wrote"] == "Him? Thanks, Pundo." and seg["restoration_reason"] == "asr_substitution" and seg["english_you_have"] == "Um, thanks, Pundo."
    assert seg["youtube_caption"] == "none for this moment" and seg["mean_word_confidence"] is None


def test_a_phrase_finds_the_segment_ignoring_case_and_punctuation_and_a_miss_says_how_to_ask():
    found = source_readings.build(SEGMENTS, words=WORDS, captions=CAPTIONS, track="en-orig", phrase="would NOT be staying, in jail")
    assert [s["segment_id"] for s in found["segments"]] == ["seg-2"]
    miss = source_readings.build(SEGMENTS, words=WORDS, captions=CAPTIONS, track="en-orig", phrase="nothing like it")
    assert miss["error"] == "no such segment" and "segment_id" in miss["hint"]
    assert source_readings.build(SEGMENTS, words=[], captions=[], track="", segment_id="seg-9")["error"] == "no such segment"


def test_the_workbench_reads_the_job_files_for_the_tool(tmp_path):
    import json

    from mathula_tv.dub_workbench import Workbench

    job = tmp_path / "work" / "jobs" / "job1"
    (job / "analysis").mkdir(parents=True)
    (job / "input" / "youtube_captions").mkdir(parents=True)
    (job / "analysis" / "transcript_en_raw.json").write_text(json.dumps({"words": WORDS}), encoding="utf-8")
    events = [{"tStartMs": 5800, "dDurationMs": 3200, "segs": [{"utf8": "he wouldn't be staying in jail"}]}]
    (job / "input" / "youtube_captions" / "captions-x.en-orig.json3").write_text(json.dumps({"events": events}), encoding="utf-8")

    class Store:
        root = tmp_path / "work" / "experience"

        def load_corpus(self):
            return {"clips": {"c1": {"job_id": "job1"}}}

    class Bench:
        store = Store()
        segments = SEGMENTS
        source_readings = Workbench.source_readings

        def _episode(self):
            return {"clip_id": "c1"}

    result = Bench().source_readings(phrase="staying in jail")
    seg = result["segments"][0]
    assert seg["segment_id"] == "seg-2" and "youtube_caption (en-orig)" in seg and seg["words_the_recogniser_was_unsure_of"]


def test_the_agent_is_given_the_source_readings_tool_and_no_pause_imitation_rules():
    from mathula_tv.dub_agent import SYSTEM_PROMPT, TOOL_SPECS

    assert any(spec["function"]["name"] == "source_readings" for spec in TOOL_SPECS)
    flat = " ".join(SYSTEM_PROMPT.split())
    assert "call source_readings before you decide what it meant" in flat
    assert "~pause" not in flat and "held pause" not in flat and "pauses_inside" not in flat          # the voice clones the speaker: the agent is not asked to imitate pauses


def test_a_phrase_narrows_the_answer_to_a_window_round_it_in_the_english_the_confidences_and_the_caption():
    long_text = " ".join(f"w{i}" for i in range(60)) + " he would not be staying in jail for eight years " + " ".join(f"z{i}" for i in range(60))
    words = [{"text": f"w{i}", "start": float(i), "end": i + 0.4, "confidence": 0.95} for i in range(60)]
    base = 60.0
    for n, (w, c) in enumerate([("he", 0.9), ("would", 0.9), ("not", 0.5), ("be", 0.9), ("staying", 0.9), ("in", 0.9), ("jail", 0.9), ("for", 0.9), ("eight", 0.9), ("years", 0.9)]):
        words.append({"text": w, "start": base + n * 0.5, "end": base + n * 0.5 + 0.4, "confidence": c})
    words += [{"text": f"z{i}", "start": base + 6 + i, "end": base + 6.4 + i, "confidence": 0.3} for i in range(60)]          # unsure words far away must not appear
    segment = {"segment_id": "seg-9", "speaker_id": "A", "start_ms": 0, "end_ms": 130000, "source_text": long_text, "restored_text": long_text}
    captions = [(2.0, 4.0, "far away at the start"), (59.0, 66.0, "he wouldn't be staying in jail for eight years")]
    seg = source_readings.build([segment], words=words, captions=captions, track="en-orig", phrase="would not be staying in jail")["segments"][0]
    assert seg["english_you_have"].startswith("… w") and seg["english_you_have"].endswith("…") and len(seg["english_you_have"].split()) < 60 and "would not be staying in jail" in seg["english_you_have"]
    assert [w["word"] for w in seg["words_the_recogniser_was_unsure_of"]] == ["not"] and "window_from_s" in seg
    assert seg["youtube_caption (en-orig)"] == "he wouldn't be staying in jail for eight years"            # the caption from the start of the segment is left out
    assert "sentence_before" not in seg
    whole = source_readings.build([segment], words=words, captions=captions, track="en-orig", segment_id="seg-9")["segments"][0]
    assert len(whole["english_you_have"]) <= source_readings.FALLBACK_CHARS + 2 and len(whole["words_the_recogniser_was_unsure_of"]) == 12          # asked by id: the whole segment, cut and capped


def test_the_differences_are_only_those_between_the_first_and_last_place_the_two_readings_agree():
    diff = source_readings._disagreement("so then he said that the president was not going to stay in jail for eight years and then",
                                         "he said the president was not going to stay in jail for eight years")
    assert diff["only_in_the_recogniser"] == ["that"] and diff["only_in_the_caption"] == []                  # the ends ("so then", "and then") are the excerpt running past the caption, not differences
    assert source_readings._disagreement("a b c d e", "a b x d e") == {"only_in_the_recogniser": ["c"], "only_in_the_caption": ["x"]}
