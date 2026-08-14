from __future__ import annotations

import pytest

from mathula_tv.direct_azure_dub import (
    DirectDubOptions,
    PreparedTranslationVariant,
    SpeechBlock,
    TimingOverflowError,
    _bounded_boundary_time_stretch_plan,
)


def _block(block_id: str, start_ms: int, end_ms: int) -> SpeechBlock:
    variant = PreparedTranslationVariant(
        variant_id="compact",
        spoken_text=block_id,
        tts_text=block_id,
    )
    return SpeechBlock(
        block_id=block_id,
        speaker_id="SPEAKER_00",
        start_ms=start_ms,
        end_ms=end_ms,
        segment_ids=(block_id,),
        source_text=block_id,
        translated_text=block_id,
        tts_text=block_id,
        variants=(variant,),
        selected_variant_id="compact",
    )


def _overflow(measured_ms: int, target_ms: int) -> TimingOverflowError:
    return TimingOverflowError(
        "measured overflow",
        measured_duration_ms=measured_ms,
        target_duration_ms=target_ms,
        stretch_ratio=measured_ms / target_ms,
    )


def test_supplied_final_block_shortfall_is_automatically_eligible() -> None:
    blocks = [
        _block("block_0014_variant_compact", 486_350, 530_690),
        _block("block_0015_variant_compact", 530_750, 553_880),
    ]

    plan = _bounded_boundary_time_stretch_plan(
        blocks,
        1,
        _overflow(24_654, 23_130),
        maximum_ratio=1.08,
    )

    assert plan is not None
    assert plan["boundary"] == "end"
    assert plan["required_time_stretch_ratio"] == pytest.approx(1.065888)
    assert plan["authoritative_translation_changed"] is False
    assert plan["speech_truncated"] is False
    assert plan["speaker_overlap_added"] is False
    assert plan["human_review_required"] is False


def test_same_ratio_is_not_auto_approved_inside_the_timeline() -> None:
    blocks = [
        _block("block_0001", 0, 1000),
        _block("block_0002", 1000, 2000),
        _block("block_0003", 2000, 3000),
    ]

    assert (
        _bounded_boundary_time_stretch_plan(
            blocks,
            1,
            _overflow(1066, 1000),
            maximum_ratio=1.08,
        )
        is None
    )


def test_excessive_boundary_compression_still_requires_review() -> None:
    blocks = [_block("block_0001", 0, 1000)]

    assert (
        _bounded_boundary_time_stretch_plan(
            blocks,
            0,
            _overflow(1081, 1000),
            maximum_ratio=1.08,
        )
        is None
    )


def test_boundary_limit_is_tightly_validated() -> None:
    assert DirectDubOptions().boundary_max_time_stretch_ratio == 1.08

    class Backend:
        class Bounds:
            rate_min_percent = -50
            rate_max_percent = 50

        ssml_bounds = Bounds()

    DirectDubOptions(boundary_max_time_stretch_ratio=1.08).validate(Backend())
    with pytest.raises(ValueError, match="boundary time-stretch ratio"):
        DirectDubOptions(boundary_max_time_stretch_ratio=1.16).validate(Backend())
