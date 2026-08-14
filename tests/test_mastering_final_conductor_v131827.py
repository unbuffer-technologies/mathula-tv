from __future__ import annotations

from mathula_tv.direct_azure_dub import (
    DirectAzureDubRenderer,
    SpeechBlock,
    enforce_source_word_speaker_alignment,
    rebalance_five_block_timeline,
    rebalance_sandwiched_interjection_timeline,
)


def _block(block_id: str, speaker: str, start: int, end: int) -> SpeechBlock:
    return SpeechBlock(
        block_id=block_id,
        speaker_id=speaker,
        start_ms=start,
        end_ms=end,
        segment_ids=(block_id,),
        source_text="source",
        translated_text="inkulumo",
        tts_text="inkulumo",
    )


def test_final_conductor_accepts_audited_delivery_compression() -> None:
    block = _block("block_0060", "SPEAKER_00", 0, 4930)
    candidate = {
        "candidate_id": "balanced",
        "compression_strategy": "Merge redundant wording while retaining both facts",
        "unit": {
            "unit_id": "block_0060",
            "spoken_text": "Asikhefuze. Sizozwa uma usukulungele.",
            "tts_text": "Asikhefuze. Sizozwa uma usukulungele.",
            "numbers_preserved": True,
            "dates_preserved": True,
            "negation_preserved": True,
            "omitted_or_compressed_detail": [
                "Merged the redundant literal stand-down wording"
            ],
        },
    }

    spoken, tts, compressions = (
        DirectAzureDubRenderer._validate_finalization_candidate(block, candidate)
    )

    assert spoken == tts == "Asikhefuze. Sizozwa uma usukulungele."
    assert compressions == (
        "Merged the redundant literal stand-down wording",
    )


def test_final_conductor_still_rejects_hard_fact_certification_failure() -> None:
    block = _block("block_0060", "SPEAKER_00", 0, 4930)
    candidate = {
        "unit": {
            "unit_id": "block_0060",
            "spoken_text": "Asikhefuze.",
            "tts_text": "Asikhefuze.",
            "numbers_preserved": True,
            "dates_preserved": True,
            "negation_preserved": False,
            "omitted_or_compressed_detail": [],
        }
    }

    try:
        DirectAzureDubRenderer._validate_finalization_candidate(block, candidate)
    except ValueError as exc:
        assert "numbers, dates, and negation" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("hard-fact certification failure was accepted")


def test_measured_non_overlapping_capacity_is_not_undone_by_word_clamp() -> None:
    source = [
        _block("block_0060", "A", 798270, 803200),
        _block("block_0061", "B", 803200, 804120),
        _block("block_0062", "A", 804180, 805320),
    ]
    fitted = [
        _block("block_0060", "A", 796522, 802102),
        _block("block_0061", "B", 802102, 803932),
        _block("block_0062", "A", 803932, 805320),
    ]
    anchors = [
        {
            "boundary_index": 0,
            "compact_handoff": True,
            "handoff_min_ms": 803140,
            "handoff_max_ms": 803200,
            "left_word": {"text": "ready", "end_ms": 802950},
            "right_word": {"text": "Thank", "start_ms": 803390},
            "tolerance_ms": 250,
        },
        {
            "boundary_index": 1,
            "compact_handoff": False,
            "left_end_min_ms": 803620,
            "left_end_max_ms": 804120,
            "right_start_min_ms": 804180,
            "right_start_max_ms": 804680,
            "left_word": {"text": "Chair", "end_ms": 803870},
            "right_word": {"text": "Let's", "start_ms": 804430},
            "tolerance_ms": 250,
        },
    ]

    adjusted, corrections = enforce_source_word_speaker_alignment(
        fitted,
        source,
        anchors,
        required_windows_ms={0: 5580, 1: 1830, 2: 1388},
    )

    assert adjusted == fitted
    assert all(
        value["reason"] == "measured_capacity_fit_takes_precedence"
        for value in corrections
    )


def test_overlap_solver_does_not_replace_an_already_resolved_target() -> None:
    blocks = [
        _block("previous", "A", 0, 2000),
        _block("interjection", "B", 2000, 2900),
        _block("following", "A", 2900, 5000),
    ]

    adjusted, audits = rebalance_sandwiched_interjection_timeline(
        blocks,
        {0: 2300, 1: 1700, 2: 2300},
        target_indices=set(),
        max_total_overlap_ms=2000,
        max_boundary_overlap_ms=1000,
    )

    assert adjusted == blocks
    assert audits == []


def test_five_block_solver_resolves_adjacent_overflows_together() -> None:
    blocks = [
        _block("outer_left", "A", 0, 2200),
        _block("large_surplus", "A", 2200, 14450),
        _block("first_overflow", "A", 14450, 19380),
        _block("short_interjection", "B", 19380, 20300),
        _block("outer_right", "A", 20360, 21500),
    ]
    required = {
        0: 1987,
        1: 2910,
        2: 5322,
        3: 1745,
        4: 1131,
    }

    adjusted, audits = rebalance_five_block_timeline(
        blocks,
        required,
        {2, 3},
    )

    assert audits[0]["applied"] is True
    assert audits[0]["neighbouring_deficits_ms"] == {
        "first_overflow": 392,
        "short_interjection": 825,
    }
    assert adjusted[2].duration_ms >= required[2]
    assert adjusted[3].duration_ms >= required[3]
    assert all(
        adjusted[index].end_ms <= adjusted[index + 1].start_ms
        for index in range(4)
    )
