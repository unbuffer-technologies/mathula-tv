from __future__ import annotations

from types import MethodType

from mathula_tv.direct_azure_dub import (
    DirectAzureDubRenderer,
    DirectDubOptions,
    PreparedTranslationVariant,
    SpeechBlock,
)


def _block() -> SpeechBlock:
    return SpeechBlock(
        block_id="block_0010",
        speaker_id="SPEAKER_01",
        start_ms=145_670,
        end_ms=150_670,
        segment_ids=("seg-10",),
        source_text="Following sentence",
        translated_text="natural",
        tts_text="natural",
        variants=(
            PreparedTranslationVariant(
                "natural", "natural", "natural", 4_750, 4_700
            ),
            PreparedTranslationVariant(
                "concise", "concise", "concise", 4_100, 4_000
            ),
            PreparedTranslationVariant(
                "compact", "compact", "compact", 3_300, 3_200
            ),
        ),
        selected_variant_id="natural",
    )


def test_following_variant_is_azure_measured_until_enough_capacity() -> None:
    renderer = object.__new__(DirectAzureDubRenderer)
    measured = {"concise": 4_300, "compact": 3_200}
    calls: list[str] = []

    def fake_synthesize(self, block, artifacts, **kwargs):
        calls.append(block.selected_variant_id)
        duration = measured[block.selected_variant_id]
        return {
            "selected_azure_rate_percent": 0,
            "azure_attempts": [{"rate_percent": 0, "duration_ms": duration}],
            "duration_ms": duration,
            "output_path": "/tmp/fake.wav",
            "output_sha256": "a" * 64,
            "cadence_qc": {"passed": True},
        }

    renderer._synthesize_block = MethodType(fake_synthesize, renderer)
    source = _block()
    selected = source.with_variant(source.variants[0])

    result = renderer._shorten_following_block_for_timeline_balance(
        selected,
        source,
        object(),
        voice="zu-ZA-ThembaNeural",
        base_rate_percent=0,
        pitch_percent=0,
        volume_percent=0,
        options=DirectDubOptions(collision_max_time_stretch_ratio=1.03),
        fixed_capacity_ms=100,
        required_capacity_ms=1_200,
    )

    assert result is not None
    block, fitted, attempts = result
    assert calls == ["concise", "compact"]
    assert block.block_id == "block_0010_variant_compact"
    assert block.selected_variant_id == "compact"
    assert fitted["translation_variant"]["selection_reason"] == (
        "following_variant_timeline_balance_azure_verified"
    )
    assert attempts[-1]["total_capacity_ms"] >= 1_200


def test_following_variant_keeps_richest_measured_candidate_that_is_enough() -> None:
    renderer = object.__new__(DirectAzureDubRenderer)
    calls: list[str] = []

    def fake_synthesize(self, block, artifacts, **kwargs):
        calls.append(block.selected_variant_id)
        duration = 4_000 if block.selected_variant_id == "concise" else 3_000
        return {
            "selected_azure_rate_percent": 0,
            "azure_attempts": [{"rate_percent": 0, "duration_ms": duration}],
            "duration_ms": duration,
            "output_path": "/tmp/fake.wav",
            "output_sha256": "b" * 64,
            "cadence_qc": {"passed": True},
        }

    renderer._synthesize_block = MethodType(fake_synthesize, renderer)
    source = _block()
    selected = source.with_variant(source.variants[0])

    result = renderer._shorten_following_block_for_timeline_balance(
        selected,
        source,
        object(),
        voice="zu-ZA-ThembaNeural",
        base_rate_percent=0,
        pitch_percent=0,
        volume_percent=0,
        options=DirectDubOptions(collision_max_time_stretch_ratio=1.03),
        fixed_capacity_ms=250,
        required_capacity_ms=1_300,
    )

    assert result is not None
    block, _, _ = result
    assert calls == ["concise"]
    assert block.selected_variant_id == "concise"


def test_global_fit_accepts_measured_two_and_half_second_middle_turn() -> None:
    from mathula_tv.direct_azure_dub import (
        rebalance_sandwiched_interjection_timeline,
    )

    blocks = [
        SpeechBlock(
            block_id="block_0008_variant_natural",
            speaker_id="SPEAKER_01",
            start_ms=140_000,
            end_ms=143_150,
            segment_ids=("seg-8",),
            source_text="before",
            translated_text="before",
            tts_text="before",
        ),
        SpeechBlock(
            block_id="block_0009_variant_compact",
            speaker_id="SPEAKER_02",
            start_ms=143_150,
            end_ms=145_670,
            segment_ids=("seg-9",),
            source_text="middle",
            translated_text="middle",
            tts_text="middle",
        ),
        SpeechBlock(
            block_id="block_0010_variant_compact",
            speaker_id="SPEAKER_01",
            start_ms=145_670,
            end_ms=150_670,
            segment_ids=("seg-10",),
            source_text="after",
            translated_text="after",
            tts_text="after",
        ),
    ]

    adjusted, audits = rebalance_sandwiched_interjection_timeline(
        blocks,
        {0: 3_150, 1: 3_258, 2: 3_260},
    )

    assert audits[0]["applied"] is True
    assert adjusted[1].duration_ms >= 3_258
    assert adjusted[2].duration_ms >= 3_260
    assert adjusted[0].end_ms <= adjusted[1].start_ms
    assert adjusted[1].end_ms <= adjusted[2].start_ms
