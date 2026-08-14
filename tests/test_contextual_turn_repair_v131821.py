from __future__ import annotations

from types import SimpleNamespace

from mathula_tv.cli import _reset_for_contextual_speaker_repair, parser
from mathula_tv.transcript_merge import (
    CONTEXTUAL_SPEAKER_TURN_REPAIR_VERSION,
    _map_to_canonical_speaker_id,
    reconcile,
)


def _word(
    word_id: int,
    text: str,
    start: float,
    end: float,
    phrase_id: str,
) -> dict[str, object]:
    return {
        "word_id": f"word_{word_id:04d}",
        "text": text,
        "start": start,
        "end": end,
        "confidence": 0.99,
        "source_phrase_id": phrase_id,
    }


def test_chair_vocative_replies_are_reassigned_and_split_before_translation() -> None:
    if hasattr(_map_to_canonical_speaker_id, "_mapping"):
        del _map_to_canonical_speaker_id._mapping

    words = [
        _word(1, "What", 0.0, 0.2, "p1"),
        _word(2, "now?", 0.2, 0.5, "p1"),
        _word(3, "Well,", 1.0, 1.2, "p2"),
        _word(4, "Chair,", 1.2, 1.5, "p2"),
        _word(5, "we", 1.5, 1.7, "p2"),
        _word(6, "have", 1.7, 1.9, "p2"),
        _word(7, "it.", 1.9, 2.1, "p2"),
        _word(8, "The", 3.0, 3.2, "p3"),
        _word(9, "hearing", 3.2, 3.6, "p3"),
        _word(10, "is", 3.6, 3.8, "p3"),
        _word(11, "postponed.", 3.8, 4.3, "p3"),
        # Azure incorrectly absorbs this whole reply into the chair's turn.
        _word(12, "19", 5.0, 5.4, "p4"),
        _word(13, "August,", 7.1, 7.5, "p4"),
        _word(14, "Chair.", 7.5, 7.8, "p4"),
        _word(15, "You", 9.0, 9.2, "p5"),
        _word(16, "are", 9.2, 9.4, "p5"),
        _word(17, "excused.", 9.4, 9.8, "p5"),
        _word(18, "Chair,", 10.5, 10.8, "p6"),
        _word(19, "may", 10.8, 11.0, "p6"),
        _word(20, "I?", 11.0, 11.2, "p6"),
        _word(21, "You", 11.3, 11.5, "p7"),
        _word(22, "may.", 11.5, 11.8, "p7"),
        # Azure also merges this embedded reply with the chair on both sides.
        _word(23, "Thank", 12.2, 12.4, "p8"),
        _word(24, "you,", 12.4, 12.6, "p8"),
        _word(25, "Chair.", 12.6, 12.9, "p8"),
        _word(26, "Let's", 13.2, 13.5, "p9"),
        _word(27, "adjourn.", 13.5, 13.9, "p9"),
    ]
    turns = [
        {"speaker": "AZURE_CHAIR", "start": 0.0, "end": 1.0},
        {"speaker": "AZURE_ADVOCATE", "start": 1.0, "end": 2.1},
        {"speaker": "AZURE_CHAIR", "start": 3.0, "end": 10.5},
        {"speaker": "AZURE_ADVOCATE", "start": 10.5, "end": 11.2},
        {"speaker": "AZURE_CHAIR", "start": 11.3, "end": 13.9},
    ]

    transcript = reconcile(words, turns, source="azure")

    assert [repair["text"] for repair in transcript["speaker_turn_repairs"]] == [
        "19 August, Chair.",
        "Thank you, Chair.",
    ]
    assert all(
        repair["repair_version"] == CONTEXTUAL_SPEAKER_TURN_REPAIR_VERSION
        for repair in transcript["speaker_turn_repairs"]
    )
    repaired = {
        segment["source_text"]: segment
        for segment in transcript["segments"]
        if segment["speaker_assignment_method"] == "contextual_vocative_repair"
    }
    assert repaired["19 August, Chair."]["speaker_id"] == "SPEAKER_01"
    assert repaired["19 August, Chair."]["raw_speaker"] == "AZURE_CHAIR"
    assert repaired["Thank you, Chair."]["speaker_id"] == "SPEAKER_01"
    assert all(
        segment["speaker_id"] == "SPEAKER_00"
        for segment in transcript["segments"]
        if segment["source_text"] in {"You may.", "Let's adjourn."}
    )


def test_one_isolated_chair_vocative_never_overrides_diarization() -> None:
    if hasattr(_map_to_canonical_speaker_id, "_mapping"):
        del _map_to_canonical_speaker_id._mapping

    transcript = reconcile(
        [
            _word(1, "Thank", 0.0, 0.2, "p1"),
            _word(2, "you,", 0.2, 0.4, "p1"),
            _word(3, "Chair.", 0.4, 0.7, "p1"),
            _word(4, "Proceed.", 0.8, 1.1, "p2"),
        ],
        [
            {"speaker": "AZURE_1", "start": 0.0, "end": 0.7},
            {"speaker": "AZURE_2", "start": 0.8, "end": 1.1},
        ],
    )

    assert transcript["speaker_turn_repairs"] == []
    assert transcript["segments"][0]["raw_speaker"] == "AZURE_1"


def test_explicit_broadcast_handoff_splits_one_faulty_raw_speaker() -> None:
    words = [
        _word(1, "Let's", 0.0, 0.2, "p1"),
        _word(2, "take", 0.2, 0.4, "p1"),
        _word(3, "you", 0.4, 0.6, "p1"),
        _word(4, "to", 0.6, 0.8, "p1"),
        _word(5, "our", 0.8, 1.0, "p1"),
        _word(6, "SABC", 1.0, 1.2, "p1"),
        _word(7, "News", 1.2, 1.4, "p1"),
        _word(8, "anchor", 1.4, 1.7, "p1"),
        _word(9, "Thembekile", 1.7, 2.1, "p1"),
        _word(10, "Mrototo,", 2.1, 2.5, "p1"),
        _word(11, "joining", 2.5, 2.8, "p1"),
        _word(12, "us", 2.8, 3.0, "p1"),
        _word(13, "live.", 3.0, 3.3, "p1"),
        _word(14, "How", 3.3, 3.5, "p1"),
        _word(15, "crucial", 3.5, 3.8, "p1"),
        _word(16, "is", 3.8, 4.0, "p1"),
        _word(17, "this?", 4.0, 4.3, "p1"),
        _word(18, "Well,", 8.4, 8.7, "p2"),
        _word(19, "Nathi,", 8.7, 9.0, "p2"),
        _word(20, "good", 9.0, 9.2, "p2"),
        _word(21, "afternoon.", 9.2, 9.6, "p2"),
        _word(22, "The", 9.8, 10.0, "p3"),
        _word(23, "story", 10.0, 10.2, "p3"),
        _word(24, "continues.", 10.2, 10.6, "p3"),
        _word(25, "A", 11.0, 11.2, "p4"),
        _word(26, "guest", 11.2, 11.5, "p4"),
        _word(27, "speaks.", 11.5, 11.8, "p4"),
        _word(28, "Back", 12.0, 12.3, "p5"),
        _word(29, "to", 12.3, 12.5, "p5"),
        _word(30, "the", 12.5, 12.7, "p5"),
        _word(31, "report.", 12.7, 13.0, "p5"),
    ]
    transcript = reconcile(
        words,
        [
            # Azure incorrectly gives the presenter and remote anchor one ID.
            {"speaker": "AZURE_1", "start": 0.0, "end": 10.6},
            {"speaker": "AZURE_2", "start": 11.0, "end": 11.8},
            {"speaker": "AZURE_1", "start": 12.0, "end": 13.0},
        ],
    )

    assert len(transcript["speaker_turn_repairs"]) == 1
    repair = transcript["speaker_turn_repairs"][0]
    assert repair["reason"] == "explicit_broadcast_handoff_merged_by_diarization"
    assert repair["original_speaker"] == "AZURE_1"
    assert repair["repaired_speaker"] == "AZURE_1__CONTEXT_HANDOFF_01"
    assert repair["repair_version"] == CONTEXTUAL_SPEAKER_TURN_REPAIR_VERSION

    by_text = {segment["source_text"]: segment for segment in transcript["segments"]}
    assert by_text[
        "Let's take you to our SABC News anchor Thembekile Mrototo, joining us live. How crucial is this?"
    ]["speaker_id"] == "SPEAKER_00"
    reply = by_text["Well, Nathi, good afternoon. The story continues."]
    assert reply["speaker_id"] == "SPEAKER_01"
    assert by_text["Back to the report."]["speaker_id"] == "SPEAKER_01"
    assert reply["raw_speaker"] == "AZURE_1"
    assert reply["speaker_assignment_method"] == (
        "contextual_broadcast_handoff_repair"
    )
    assert by_text["A guest speaks."]["speaker_id"] == "SPEAKER_02"


def test_repair_command_rebuilds_analysis_without_retranscription() -> None:
    args = parser().parse_args(
        ["repair-speaker-turns", "job-123", "--live-operation"]
    )
    assert args.command == "repair-speaker-turns"
    assert args.job_id == "job-123"

    job = SimpleNamespace(
        state="review_ready",
        completed_stages=["source_derivatives", "classification", "translation", "render"],
        review_readiness="ready",
        last_error={"message": "old"},
        attempt_history=[],
    )
    previous = _reset_for_contextual_speaker_repair(job)

    assert previous == "review_ready"
    assert job.state == "analysis_running"
    assert job.completed_stages == ["source_derivatives"]
    assert job.last_error is None
    assert job.attempt_history[-1]["stage"] == "contextual_speaker_turn_repair"
