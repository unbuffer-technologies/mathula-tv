import json

from mathula_tv.draft_ledger import LEDGER_NOTE_PREFIX, DraftLedger

SEGMENTS = [{"segment_id": "s1", "speaker_id": "SPEAKER_00", "start_ms": 0, "end_ms": 4000}, {"segment_id": "s2", "speaker_id": "SPEAKER_00", "start_ms": 4500, "end_ms": 8000}]


def _measure(text, start, end, measured, window=None, fits=False):
    return (json.dumps({"text": text, "start_ms": start, "end_ms": end, "speaker_id": "SPEAKER_00"}),
            json.dumps({"measured_ms": measured, "window_ms": window or end - start, "fits_naturally": fits, "direction": None if fits else "compress"}))


def test_the_ledger_is_built_from_the_agents_own_tool_calls_and_keeps_the_latest_text_per_window():
    ledger = DraftLedger(SEGMENTS)
    ledger.observe("measure_tts", *_measure("first try", 0, 4000, 4600))
    ledger.observe("measure_tts", *_measure("second try", 0, 4000, 3950, fits=True))
    ledger.observe("measure_tts_batch", json.dumps({"requests": [{"text": "b one", "start_ms": 4500, "end_ms": 8000}, {"text": "b two", "start_ms": 8500, "end_ms": 12000}]}),
                   json.dumps({"results": [{"measured_ms": 3300, "window_ms": 3500, "fits_naturally": True}, {"measured_ms": 3900, "window_ms": 3500, "fits_naturally": False}]}))
    shown = ledger.render()
    assert "first try" not in shown and "second try" in shown and "b one" in shown and "b two" in shown
    assert "measured 4.0s in a 4.0s window (fits)" in shown and "over by 11%" in shown
    assert shown.index("second try") < shown.index("b one") < shown.index("b two")          # in time order


def test_estimates_and_timeline_items_are_recorded_and_a_measurement_does_not_lose_its_text_to_an_estimate():
    ledger = DraftLedger()
    ledger.set_segments(SEGMENTS)
    ledger.observe("timeline_fit", json.dumps({"items": [{"segment_id": "s2", "text": "timeline draft"}]}), json.dumps({"sentences": [{"segment_id": "s2", "estimated_ms": 3600}]}))
    assert "only estimated: 3.6s in a 3.5s window" in ledger.render() and "timeline draft" in ledger.render()
    ledger.observe("estimate_fit", json.dumps({"requests": [{"text": "timeline draft", "start_ms": 4500, "end_ms": 8000}]}), json.dumps({"results": [{"estimated_ms": 3550}]}))
    ledger.observe("measure_tts", *_measure("timeline draft", 4500, 8000, 3480, fits=True))
    assert ledger.render().count("timeline draft") == 1 and "measured 3.5s in a 3.5s window (fits)" in ledger.render()


def test_a_full_submission_becomes_the_record_and_a_partial_one_only_replaces_what_it_names():
    ledger = DraftLedger(SEGMENTS)
    ledger.observe("measure_tts", *_measure("loose draft", 0, 4000, 4000, fits=True))
    items = [{"item_id": "u1", "start_ms": 0, "end_ms": 4000, "speaker_id": "SPEAKER_00", "zulu_text": "unit one"}, {"item_id": "u2", "start_ms": 4500, "end_ms": 8000, "speaker_id": "SPEAKER_00", "zulu_text": "unit two"}]
    ledger.observe("submit_dub", json.dumps({"items": items}), json.dumps({"accepted": True}))
    assert ledger.submitted and "loose draft" not in ledger.render() and "unit one" in ledger.render() and "submitted" in ledger.render()
    ledger.observe("submit_dub", json.dumps({"items": [{"item_id": "u2", "start_ms": 4500, "end_ms": 8000, "zulu_text": "unit two, shorter"}], "partial": True}), json.dumps({"accepted": True}))
    assert "unit one" in ledger.render() and "unit two, shorter" in ledger.render() and "unit two," in ledger.render() and ledger.render().count("unit two") == 1


def test_a_saved_conversation_can_rebuild_the_ledger_for_a_checkpoint_from_before_it_existed():
    args, result = _measure("old draft", 0, 4000, 4100)
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "task"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "measure_tts", "arguments": args}}]},
                {"role": "tool", "tool_call_id": "c1", "content": result},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "c2", "type": "function", "function": {"name": "measure_tts", "arguments": json.dumps({"text": "newer draft", "start_ms": 0, "end_ms": 4000})}}]},
                {"role": "tool", "tool_call_id": "c2", "content": json.dumps({"error": "budget_exhausted"})}]
    ledger = DraftLedger()
    ledger.replay(messages)
    assert "newer draft" in ledger.render() and "old draft" not in ledger.render()                 # replayed in order: the later draft is the latest for its window


def test_a_draft_carried_by_a_failed_call_is_still_the_agents_latest_text():
    """Found by the loop test: a measurement refused for budget carried a draft the agent wrote; compaction must not lose it."""
    ledger = DraftLedger()
    ledger.observe("measure_tts", json.dumps({"text": "refused draft", "start_ms": 0, "end_ms": 4000}), json.dumps({"error": "budget_exhausted"}))
    ledger.observe("measure_tts_batch", json.dumps({"requests": [{"text": "batch draft", "start_ms": 5000, "end_ms": 8000}]}), json.dumps({"error": "budget_exhausted"}))
    shown = ledger.render()
    assert "refused draft" in shown and "batch draft" in shown and shown.count("not measured") == 2


def test_a_call_the_ledger_cannot_read_is_missed_never_a_failure_and_the_checkpoint_round_trips():
    ledger = DraftLedger()
    ledger.observe("measure_tts", "not json", "{}")
    ledger.observe("measure_tts", json.dumps({"text": "x"}), json.dumps({"measured_ms": 5}))              # no window in the call
    ledger.observe("no_such_tool", "{}", "{}")
    assert ledger.entries == {}
    ledger.observe("measure_tts", *_measure("kept", 0, 4000, 4100))
    again = DraftLedger()
    again.load(json.loads(json.dumps(ledger.to_dict())))
    assert again.render() == ledger.render() and LEDGER_NOTE_PREFIX.startswith("[Working drafts")
