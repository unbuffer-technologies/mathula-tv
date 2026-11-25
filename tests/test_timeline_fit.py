from types import SimpleNamespace

import pytest

from mathula_tv.dub_workbench import Workbench


def _bench(lengths):
    """A clip of five segments: three by SPEAKER_00 (0-4 s, 4.5-8 s, 8.5-12 s), then SPEAKER_01 (12.2-15 s) and SPEAKER_00 again (15.5-18 s).  `lengths` maps a text to its estimated ms."""
    bench = object.__new__(Workbench)
    bench.segments = [
        {"segment_id": "s1", "speaker_id": "SPEAKER_00", "start_ms": 0, "end_ms": 4000},
        {"segment_id": "s2", "speaker_id": "SPEAKER_00", "start_ms": 4500, "end_ms": 8000},
        {"segment_id": "s3", "speaker_id": "SPEAKER_00", "start_ms": 8500, "end_ms": 12000},
        {"segment_id": "s4", "speaker_id": "SPEAKER_01", "start_ms": 12200, "end_ms": 15000},
        {"segment_id": "s5", "speaker_id": "SPEAKER_00", "start_ms": 15500, "end_ms": 18000},
    ]
    bench.voices = {}
    bench.OMNIVOICE_BAND_PCT = 10.0
    bench._text_ms = lambda text, speaker, start, end: (lengths[text], 10.0)           # the estimate is not under test here, the layout is
    return bench


def test_a_draft_that_fits_every_window_starts_each_sentence_on_the_speakers_onset():
    bench = _bench({"a": 3800, "b": 3400, "c": 3500})
    out = bench.timeline_fit([{"segment_id": "s1", "text": "a"}, {"segment_id": "s2", "text": "b"}, {"segment_id": "s3", "text": "c"}])
    assert [r["late_by_ms"] for r in out["sentences"]] == [0, 0, 0] and out["late_sentences"] == [] and out["where_to_look"] == [] and out["lateness_at_end_ms"] == 0


def test_lateness_is_inherited_and_the_hint_points_back_to_the_sentence_that_lost_the_time():
    bench = _bench({"a": 5200, "b": 3300, "c": 3300})                              # the first sentence runs 1.2 s past its window; the next two would fit alone
    out = bench.timeline_fit([{"segment_id": "s3", "text": "c"}, {"segment_id": "s1", "text": "a"}, {"segment_id": "s2", "text": "b"}])          # any order in
    rows = {r["segment_id"]: r for r in out["sentences"]}
    assert [r["segment_id"] for r in out["sentences"]] == ["s1", "s2", "s3"]
    assert rows["s1"]["late_by_ms"] == 0 and rows["s1"]["over_window_by_ms"] == 1200
    assert rows["s2"]["starts_ms"] == 5200 + 300 and rows["s2"]["late_by_ms"] == 1000          # it would fit its own window, but it inherits the delay
    assert out["late_sentences"] == ["s2"] or "s2" in out["late_sentences"]
    hint = out["where_to_look"][0]
    assert hint["late_from"] == "s2" and hint["shorten_these"][0] == {"segment_id": "s1", "over_window_by_ms": 1200}      # go back to s1, not s2


def test_a_sentence_that_runs_into_the_next_speaker_is_named_and_unknown_segments_are_refused():
    bench = _bench({"c": 4300})                                                    # s3 ends at 8.5 s + 4.3 s = 12.8 s, the next speaker starts at 12.2 s
    out = bench.timeline_fit([{"segment_id": "s3", "text": "c"}])
    assert out["runs_into_another_speaker"] == ["s3"] and out["sentences"][0]["runs_into_next_speaker_ms"] == 600
    with pytest.raises(ValueError, match="unknown segment_id"):
        bench.timeline_fit([{"segment_id": "zz", "text": "c"}])
    with pytest.raises(ValueError, match="between 1 and"):
        bench.timeline_fit([])


def test_the_tool_is_offered_to_the_agent_with_its_guidance_and_a_dispatch_entry():
    from mathula_tv import dub_agent

    assert "timeline_fit" in {t["function"]["name"] for t in dub_agent.TOOL_SPECS}
    assert "where lateness BEGINS" in dub_agent.SYSTEM_PROMPT and "go back to the sentence that lost the time" in dub_agent.SYSTEM_PROMPT
    dispatch = dub_agent.build_dispatch(SimpleNamespace(**{n: (lambda *a, **k: None) for n in ("open_clip", "recall_lessons", "measure_tts", "measure_tts_batch", "check_text", "preview_turn_padding",
                                                           "submit_dub", "get_submission", "estimate_fit", "timeline_fit")}), clip_id="c", run_id="r", tts_budget=5)
    assert "timeline_fit" in dispatch
