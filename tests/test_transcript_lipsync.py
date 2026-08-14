from __future__ import annotations

from dataclasses import replace

from mathula_tv.direct_azure_dub import (
    DirectAzureDubRenderer,
    SpeechBlock,
    attach_source_pause_anchors,
    attach_transcript_lipsync_plan,
    build_source_word_speaker_anchors,
)
from mathula_tv.transcript_lipsync import build_transcript_lipsync_plan


def _block(
    block_id: str,
    speaker_id: str,
    start_ms: int,
    end_ms: int,
) -> SpeechBlock:
    return SpeechBlock(
        block_id=block_id,
        speaker_id=speaker_id,
        start_ms=start_ms,
        end_ms=end_ms,
        segment_ids=(block_id,),
        source_text=f"source {block_id}",
        translated_text=f"translated {block_id}",
        tts_text=f"translated {block_id}",
    )


def _word(
    word_id: str,
    speaker_id: str,
    start_ms: int,
    end_ms: int,
    text: str = "word",
) -> dict[str, object]:
    return {
        "word_id": word_id,
        "speaker_id": speaker_id,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "text": text,
    }


def test_long_source_pause_creates_two_speech_islands() -> None:
    plan = build_transcript_lipsync_plan(
        [_block("block_0001", "SPEAKER_00", 0, 2500)],
        [
            _word("w1", "SPEAKER_00", 100, 400, "slow"),
            _word("w2", "SPEAKER_00", 1400, 1700, "answer"),
        ],
    )

    record = plan["blocks"][0]
    assert record["timing_evidence"] == "azure_word_timestamps"
    assert len(record["speech_islands"]) == 2
    assert record["source_pauses"] == [
        {
            "start_ms": 400,
            "end_ms": 1400,
            "duration_ms": 1000,
            "left_word_id": "w1",
            "right_word_id": "w2",
            "kind": "island_boundary",
        }
    ]


def test_word_intersection_authorizes_cross_speaker_overlap() -> None:
    plan = build_transcript_lipsync_plan(
        [
            _block("block_0001", "SPEAKER_00", 0, 1200),
            _block("block_0002", "SPEAKER_01", 700, 1600),
        ],
        [
            _word("a", "SPEAKER_00", 100, 1000),
            _word("b", "SPEAKER_01", 800, 1200),
        ],
    )

    assert plan["interaction_count"] == 1
    event = plan["interactions"][0]
    assert event["kind"] == "source_overlap"
    assert event["overlap_ms"] == 200
    assert event["source_authorized"] is True


def test_pause_between_words_does_not_create_false_overlap() -> None:
    plan = build_transcript_lipsync_plan(
        [
            _block("block_0001", "SPEAKER_00", 0, 1400),
            _block("block_0002", "SPEAKER_01", 500, 900),
        ],
        [
            _word("a1", "SPEAKER_00", 0, 400),
            _word("b", "SPEAKER_01", 500, 700),
            _word("a2", "SPEAKER_00", 750, 1200),
        ],
    )

    assert plan["interaction_count"] == 0


def test_short_a_b_a_turn_is_marked_as_interjection() -> None:
    plan = build_transcript_lipsync_plan(
        [
            _block("block_0001", "SPEAKER_00", 0, 800),
            _block("block_0002", "SPEAKER_01", 700, 1200),
            _block("block_0003", "SPEAKER_00", 1100, 1900),
        ],
        [
            _word("a1", "SPEAKER_00", 100, 780),
            _word("b1", "SPEAKER_01", 720, 1150),
            _word("a2", "SPEAKER_00", 1100, 1750),
        ],
    )

    events = [event for event in plan["interactions"] if event["kind"] == "sandwiched_interjection"]
    assert len(events) == 1
    assert events[0]["interjector_block_id"] == "block_0002"
    block = next(value for value in plan["blocks"] if value["block_id"] == "block_0002")
    assert any(value["role"] == "interjector" for value in block["interaction_roles"])


def test_missing_words_falls_back_without_changing_legacy_block() -> None:
    original = _block("block_0001", "SPEAKER_00", 100, 900)
    adjusted, plan = attach_transcript_lipsync_plan(
        [original],
        {"segments": [], "words": []},
    )

    assert adjusted[0].start_ms == original.start_ms
    assert adjusted[0].end_ms == original.end_ms
    assert adjusted[0].translated_text == original.translated_text
    assert adjusted[0].source_timing_evidence == "block_boundary_fallback"
    assert plan["fallback_block_count"] == 1
    assert plan["compatibility"]["legacy_cli_flags_unchanged"] is True


def test_source_overlap_is_not_collapsed_to_compact_handoff() -> None:
    blocks = [
        _block("block_0001", "SPEAKER_00", 0, 1100),
        _block("block_0002", "SPEAKER_01", 800, 1600),
    ]
    transcript = {
        "segments": [],
        "words": [
            _word("a", "SPEAKER_00", 100, 1000),
            _word("b", "SPEAKER_01", 800, 1200),
        ],
    }

    anchors = build_source_word_speaker_anchors(blocks, transcript)

    assert anchors[0]["source_authorized_overlap"] is True
    assert anchors[0]["source_overlap_ms"] == 200
    assert anchors[0]["compact_handoff"] is False


def test_word_timed_semantic_fit_uses_the_visible_speech_endpoint() -> None:
    block = replace(
        _block("block_0001", "SPEAKER_00", 1000, 5000),
        source_speech_start_ms=1000,
        source_speech_end_ms=3200,
        source_timing_evidence="azure_word_timestamps",
    )

    assert DirectAzureDubRenderer._semantic_fit_target_band(block, 250) == (
        1950,
        2200,
        2450,
    )


def test_word_timed_block_cannot_borrow_the_following_silence() -> None:
    timed = replace(
        _block("block_0001", "SPEAKER_00", 0, 2000),
        source_speech_end_ms=1800,
        source_timing_evidence="azure_word_timestamps",
    )
    following = _block("block_0002", "SPEAKER_00", 4000, 5000)
    legacy = replace(timed, source_timing_evidence="block_boundary_fallback")

    assert DirectAzureDubRenderer._same_speaker_transition_borrow_limit(
        [timed, following], 0
    ) == 0
    assert DirectAzureDubRenderer._same_speaker_transition_borrow_limit(
        [legacy, following], 0
    ) == 1000


def test_long_word_gap_is_retained_as_locked_island_pause() -> None:
    block = _block("block_0001", "SPEAKER_00", 0, 3000)
    transcript = {
        "segments": [],
        "words": [
            _word("w1", "SPEAKER_00", 100, 400, "slow"),
            _word("w2", "SPEAKER_00", 2400, 2700, "answer"),
        ],
    }

    adjusted, artifact = attach_source_pause_anchors([block], transcript)

    assert artifact[0]["anchors"][0]["source_gap_ms"] == 2000
    assert adjusted[0].source_pause_anchors[0]["requested_pause_ms"] == 1900
    assert adjusted[0].source_pause_anchors[0]["pause_kind"] == "island_boundary"
