from __future__ import annotations

from types import MethodType

from mathula_tv.direct_azure_dub import (
    DirectAzureDubRenderer,
    DirectDubOptions,
    PreparedTranslationVariant,
    SpeechBlock,
    choose_three_block_variant_balance,
)


def _option(variant_id: str, steps: int, released: int) -> dict[str, object]:
    return {
        "variant_id": variant_id,
        "compression_steps": steps,
        "capacity_release_ms": released,
    }


def _block(block_id: str, start_ms: int, end_ms: int) -> SpeechBlock:
    return SpeechBlock(
        block_id=block_id,
        speaker_id="SPEAKER_01",
        start_ms=start_ms,
        end_ms=end_ms,
        segment_ids=(f"seg-{block_id}",),
        source_text="Source sentence",
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


def test_three_block_choice_spreads_reduction_before_forcing_compact() -> None:
    previous = [
        _option("natural", 0, 0),
        _option("concise", 1, 400),
        _option("compact", 2, 800),
    ]
    following = [
        _option("natural", 0, 0),
        _option("concise", 1, 400),
        _option("compact", 2, 800),
    ]

    selected = choose_three_block_variant_balance(
        previous,
        following,
        base_capacity_ms=0,
        required_capacity_ms=700,
    )

    assert selected is not None
    selected_previous, selected_following = selected
    assert selected_previous["variant_id"] == "concise"
    assert selected_following["variant_id"] == "concise"


def test_three_block_choice_changes_only_one_neighbour_when_enough() -> None:
    previous = [
        _option("natural", 0, 0),
        _option("concise", 1, 800),
    ]
    following = [
        _option("natural", 0, 0),
        _option("concise", 1, 350),
        _option("compact", 2, 900),
    ]

    selected = choose_three_block_variant_balance(
        previous,
        following,
        base_capacity_ms=0,
        required_capacity_ms=700,
    )

    assert selected is not None
    selected_previous, selected_following = selected
    assert selected_previous["variant_id"] == "concise"
    assert selected_following["variant_id"] == "natural"


def test_neighbour_variants_are_azure_measured_not_estimated() -> None:
    renderer = object.__new__(DirectAzureDubRenderer)
    measured = {"concise": 4_200, "compact": 3_100}
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
    source = _block("block_0008", 140_000, 145_000)
    selected = source.with_variant(source.variants[0])

    options, attempts = (
        renderer._measure_shorter_neighbour_variants_for_timeline_balance(
            selected,
            source,
            object(),
            role="previous",
            voice="zu-ZA-ThembaNeural",
            base_rate_percent=0,
            pitch_percent=0,
            volume_percent=0,
            options=DirectDubOptions(collision_max_time_stretch_ratio=1.03),
            current_required_window_ms=4_800,
        )
    )

    assert calls == ["concise", "compact"]
    assert [item["variant_id"] for item in options] == ["concise", "compact"]
    assert options[0]["capacity_release_ms"] == 4_800 - 4_078
    assert options[1]["capacity_release_ms"] == 4_800 - 3_010
    assert attempts[-1]["role"] == "previous"
    assert options[-1]["fitted"]["translation_variant"]["selection_reason"] == (
        "three_block_timeline_balance_azure_verified"
    )
