from mathula_tv.transcript_merge import reconcile
from mathula_tv.diarization import select_authoritative


def test_overlap_unknown_and_pause_boundaries():
    words = [{"text":"hello","start":0,"end":.5,"confidence":.9},{"text":"both","start":1,"end":1.2,"confidence":.8},{"text":"lost","start":3,"end":3.2,"confidence":.7}]
    turns = [{"speaker":"A","start":0,"end":1.2},{"speaker":"B","start":.9,"end":1.4}]
    result = reconcile(words, turns, tolerance=.1)
    assert result["words"][1]["overlap"] is True
    assert result["words"][2]["speaker"] == "UNKNOWN"
    assert len(result["segments"]) == 3


def test_pyannote_fallback_is_truthful():
    turns, source, warnings = select_authoritative(None, {"turns":[{"speaker":"AZURE_1","start":0,"end":1}]}, "GPU disconnected")
    assert source == "azure" and "GPU disconnected" in warnings[0]



def _said(text_times):
    return [{"text": t, "start": s, "end": e, "confidence": .6, "word_id": f"word_{i:06d}", "source_phrase_id": f"phrase_{i // 5:05d}"} for i, (t, s, e) in enumerate(text_times)]


NAMES = [("namely", 1.0, 1.3), ("Dr.", 1.3, 1.5), ("Ivan", 1.5, 1.8), ("Khoza", 1.8, 2.2), ("and", 2.2, 2.4), ("Kaiza", 2.4, 2.8), ("Motaung.", 2.8, 3.3)]
LOOP = [(w, 3.4 + .2 * i, 3.6 + .2 * i) for i, w in enumerate(["Ralphs,", "Ralphs,", "and", "Ralphs.", "Ralphs,", "Ralphs,", "and", "Ralphs.", "Ralphs,"])]


def test_a_word_the_recogniser_repeats_over_nothing_is_not_speech_and_is_dropped_by_shape_alone():
    from mathula_tv.transcript_merge import drop_recogniser_loops, recogniser_loops

    words = _said(NAMES + LOOP + [("Later", 30.0, 30.4)])
    kept, loops = drop_recogniser_loops(words)
    assert [w["text"] for w in kept] == [w[0] for w in NAMES] + ["Later"]
    assert len(loops) == 1 and loops[0]["words"] == len(LOOP) and loops[0]["text"].startswith("Ralphs, Ralphs, and Ralphs.") and (loops[0]["start"], loops[0]["end"]) == (3.4, 5.2)
    five = _said([("no,", 0, .2), ("no,", .2, .4), ("no,", .4, .6), ("no,", .6, .8), ("no.", .8, 1.0), ("Sir", 1.1, 1.3)])
    assert recogniser_loops(five) == [] and drop_recogniser_loops(five)[1] == []          # five is a person insisting; six is a loop
    assert recogniser_loops(_said([("and", 0, 1), ("and", 1, 2), ("and", 2, 3), ("and", 3, 4), ("and", 4, 5), ("and", 5, 6), ("and", 6, 7)])) == []          # "and" alone is never the loop
    mixed = _said([("Ralphs", 0, 1)] * 3 + [("Hello", 1, 2)] + [("Ralphs", 2, 3)] * 3)
    assert recogniser_loops(mixed) == []          # a different word between breaks it


def test_reconcile_keeps_the_loop_out_of_the_transcript_and_says_so():
    words = [{"text": w, "start": s, "end": e, "confidence": .6} for w, s, e in NAMES + LOOP + [("Later", 30.0, 30.4)]]
    result = reconcile(words, [{"speaker": "A", "start": 0, "end": 40}])
    assert "Ralphs" not in result["segments"][0]["source_text"] and not any("Ralphs" in w["text"] for w in result["words"])
    assert result["recogniser_loops_dropped"][0]["words"] == len(LOOP) and any("recogniser loop" in w for w in result["warnings"])
    clean = reconcile(words[:len(NAMES)], [{"speaker": "A", "start": 0, "end": 40}])
    assert "recogniser_loops_dropped" not in clean


def test_a_transcript_already_built_is_repaired_segment_by_segment(tmp_path):
    import json

    from mathula_tv.transcript_merge import drop_loops_from_transcript, drop_loops_in_job

    words = _said(NAMES + LOOP + [("Later", 30.0, 30.4)])
    first = words[:len(NAMES) + 4]          # the loop begins inside the real segment (as in job 1abf3388) and goes on into a segment of its own
    rest = words[len(NAMES) + 4:len(NAMES) + len(LOOP)]
    last = words[-1:]
    def seg(i, run):
        return {"segment_id": f"seg-{i:05d}", "speaker": "SPEAKER_00", "start": run[0]["start"], "end": run[-1]["end"], "source_text": " ".join(w["text"] for w in run),
                "source_word_ids": [w["word_id"] for w in run], "source_phrase_ids": list(dict.fromkeys(w["source_phrase_id"] for w in run)), "recognition_confidence": .6}
    transcript = {"words": words, "segments": [seg(1, first), seg(2, rest), seg(3, last)], "warnings": [], "complete_text": " ".join(w["text"] for w in words)}
    job = tmp_path / "job"
    (job / "analysis").mkdir(parents=True)
    for name in ("transcript_en_raw.json", "transcript_en.json"):
        (job / "analysis" / name).write_text(json.dumps(transcript), encoding="utf-8")
    report = drop_loops_in_job(job)
    assert set(report) == {"transcript_en_raw.json", "transcript_en.json"} and report["transcript_en_raw.json"][0]["words"] == len(LOOP)
    fixed = json.loads((job / "analysis" / "transcript_en_raw.json").read_text(encoding="utf-8"))
    assert [s["segment_id"] for s in fixed["segments"]] == ["seg-00001", "seg-00003"]          # the segment that was all loop is gone
    head = fixed["segments"][0]
    assert head["source_text"] == "namely Dr. Ivan Khoza and Kaiza Motaung." and head["end"] == 3.3 and head["source_word_ids"] == [w["word_id"] for w in NAMES_WITH_IDS(words)]
    assert "Ralphs" not in fixed["complete_text"] and fixed["complete_text"].endswith("Later") and any("recogniser loop" in w for w in fixed["warnings"])
    assert (job / "analysis" / "transcript_en_raw.before_loop_drop.json").is_file()
    assert drop_loops_in_job(job) == {}          # nothing left to drop the second time, and the backup is not overwritten
    assert json.loads((job / "analysis" / "transcript_en_raw.before_loop_drop.json").read_text(encoding="utf-8"))["words"][len(NAMES)]["text"] == "Ralphs,"


def NAMES_WITH_IDS(words):
    return words[:len(NAMES)]
