from __future__ import annotations

from mathula_tv.direct_azure_dub import (
    SpeechBlock,
    extend_final_speech_block_into_source_tail,
    rebalance_sandwiched_interjection_timeline,
)
from mathula_tv.dub_mastering import (
    _has_production_pause_blocker,
    _safe_overrides_for_policy_migration,
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


def test_unused_source_tail_becomes_bounded_final_timing_capacity() -> None:
    blocks = [
        _block("block_0061", "B", 803_200, 804_180),
        _block("block_0062", "A", 804_180, 805_320),
    ]

    adjusted, audit = extend_final_speech_block_into_source_tail(
        blocks,
        source_duration_ms=807_427,
    )

    assert adjusted[0] == blocks[0]
    assert adjusted[1].end_ms == 807_427
    assert audit["applied"] is True
    assert audit["borrowed_tail_ms"] == 2_107
    assert audit["source_video_extended"] is False


def test_final_triplet_uses_source_tail_without_speaker_overlap() -> None:
    blocks = [
        _block("block_0060", "A", 797_810, 803_390),
        _block("block_0061", "B", 803_390, 804_370),
        _block("block_0062", "A", 804_430, 805_320),
    ]
    blocks, _ = extend_final_speech_block_into_source_tail(
        blocks,
        source_duration_ms=807_427,
    )

    adjusted, audits = rebalance_sandwiched_interjection_timeline(
        blocks,
        {0: 5_580, 1: 1_830, 2: 1_471},
    )

    assert [(item.start_ms, item.end_ms) for item in adjusted] == [
        (797_810, 803_390),
        (803_390, 805_220),
        (805_220, 807_427),
    ]
    assert audits[0]["cross_speaker_overlap"] is False
    assert audits[0]["total_overlap_ms"] == 0
    assert audits[0]["safe_slack_ms"] == 736


def test_small_clause_and_date_pause_overruns_remain_warnings_only() -> None:
    assert not _has_production_pause_blocker(
        [
            {"duration_ms": 1_520, "allowed_duration_ms": 1_300},
            {"duration_ms": 1_360, "allowed_duration_ms": 900},
        ]
    )


def test_genuine_two_second_unplanned_pause_still_blocks_production() -> None:
    assert _has_production_pause_blocker(
        [{"duration_ms": 2_000, "allowed_duration_ms": 900}]
    )


def test_v5_prepared_variant_override_survives_v6_policy_migration() -> None:
    payload = {
        "overrides": [
            {
                "block_id": "block_0059",
                "strategy": "select_variant",
                "variant_id": "concise",
            },
            {
                "block_id": "block_0060",
                "strategy": "select_variant",
                "variant_id": "ai_balanced_r1",
            },
        ]
    }

    retained = _safe_overrides_for_policy_migration(
        payload,
        installed_policy="mathula-production-dub-score-v5",
    )

    assert retained == [
        {
            "block_id": "block_0059",
            "strategy": "select_variant",
            "variant_id": "concise",
        }
    ]
