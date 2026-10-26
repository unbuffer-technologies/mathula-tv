import json
from types import SimpleNamespace

from mathula_tv import code_switch as cs


def test_disagreement_is_zero_for_the_same_words_and_high_for_few_shared():
    assert cs.disagreement("The ANC has no money", "the anc has no money!") == 0
    assert cs.disagreement("Siyipatala through tax", "Lethu siyayibadala thru taxt") > 0.5
    assert cs.disagreement("", "anything") == 0


def test_the_most_different_readings_are_chosen_capped_and_returned_in_time_order():
    rows = [{"id": i, "start_s": float(i), "end_s": i + 1.0, "english": e, "isizulu": z} for i, (e, z) in enumerate([
        ("what is our job", "what is our job"),                           # the same: left alone
        ("Siyipatala through tax", "lethu siyayibadala thru taxt"),
        ("ANC Aina Mali", "enzi annamali"),
        ("Let us go", "let us go now"),
    ])]
    assert [r["id"] for r in cs.select_stretches(rows)] == [1, 2]
    assert len(cs.select_stretches(rows, cap=1)) == 1
    assert all("disagreement" in r for r in cs.select_stretches(rows))


def test_phones_are_grouped_into_words_where_the_speaker_pauses_and_segments_found_by_overlap():
    timed = [("n", 0.0, 0.05), ("a", 0.05, 0.1), ("m", 0.4, 0.45), ("a", 0.45, 0.5)]
    assert cs.group_phones(timed) == "na | ma"
    segments = [{"segment_id": "a", "start_ms": 0, "end_ms": 5000}, {"segment_id": "b", "start_ms": 5000, "end_ms": 9000}]
    assert cs.segment_for(segments, 4.0, 7.0) == "b"                           # 2 s in b against 1 s in a
    assert cs.segment_for(segments, 20.0, 22.0) is None


def test_build_collects_evidence_for_the_hard_stretches_only_and_the_agent_view_is_scoped(tmp_path):
    native, analysis = tmp_path / "native_dub", tmp_path / "analysis"
    native.mkdir()
    analysis.mkdir()
    (native / "pass1_restored_transcript.json").write_text(json.dumps({"segments": [
        {"segment_id": "seg-1", "start_ms": 0, "end_ms": 6000}, {"segment_id": "seg-2", "start_ms": 6000, "end_ms": 12000}]}), encoding="utf-8")
    (native / "context_ledger.json").write_text(json.dumps({"context_ledger": {"summary": "A rally in Mpumalanga."}}), encoding="utf-8")
    (analysis / "azure_diarization.json").write_text(json.dumps({"phrases": [
        {"start": 1.0, "end": 4.0, "text": "What is our job"}, {"start": 7.0, "end": 10.0, "text": "Siyipatala through tax"}]}), encoding="utf-8")
    readings = {1.0: "what is our job", 7.0: "lethu siyayibadala thru taxt"}
    asked = {}

    class Provider:
        def complete_json(self, **kwargs):
            asked.update(kwargs)
            ids = [s["id"] for s in kwargs["payload"]["stretches"]]
            return SimpleNamespace(data={"stretches": [{"id": i, "reconstruction": [{"text": "siyibhadala", "language": "isizulu"}, {"text": "through tax", "language": "english"}],
                                                        "english_meaning": "we pay for it through tax", "confidence": "high", "evidence": "phones se pa ta la"} for i in ids]})

    path = cs.build(tmp_path, Provider(), zulu_reading=lambda a, b: readings[a], afrikaans_reading=lambda a, b: "sy baddala through text",
                    phones=lambda a, b: [("s", 0.0, 0.1), ("e", 0.1, 0.2)], progress=lambda m: None)
    doc = cs.load(path)
    assert len(doc["stretches"]) == 1 and doc["stretches_examined"] == 2          # only the stretch the recognisers disagree on
    stretch = doc["stretches"][0]
    assert stretch["segment_id"] == "seg-2" and stretch["confidence"] == "high"
    assert asked["payload"]["rally"] == "A rally in Mpumalanga."
    assert asked["payload"]["stretches"][0]["phones_grouped_by_pauses"] == "se"
    # cached by the transcript's checksum: a second build does not call the model again
    asked.clear()
    assert cs.build(tmp_path, Provider(), zulu_reading=lambda a, b: "", afrikaans_reading=lambda a, b: "", phones=lambda a, b: [], progress=lambda m: None) == path
    assert not asked
    view = cs.for_agent(doc, ["seg-2"])
    assert view["stretches"][0]["he_said"] == "siyibhadala [isizulu] through tax [english]"
    assert "never translate the garbled English" in view["note"]
    assert cs.for_agent(doc, ["seg-1"]) is None
