from __future__ import annotations

from pathlib import Path

import pytest

from mathula_tv.direct_azure_dub import (
    DirectDubArtifacts,
    DirectDubOptions,
    PreparedTranslationVariant,
    PreparedVariantsExhausted,
    SpeechBlock,
    TimingOverflowError,
    load_approved_segments,
)
from mathula_tv.direct_azure_dub import DirectAzureDubRenderer


def _translation() -> dict:
    return {
        "schema_version": "mathula.translation.multivariant.installed.v1",
        "units": [
            {
                "unit_id": "seg-1",
                "speaker_id": "SPEAKER_00",
                "start_ms": 0,
                "end_ms": 2000,
                "source_text": "Hello EFF",
                "protected_spans": ["EFF"],
                "variants": [
                    {
                        "variant_id": "natural",
                        "spoken_text": "Sawubona kwi-EFF namhlanje",
                        "target_duration_ms": 1900,
                        "estimated_duration_ms": 2600,
                        "meaning_preserved": True,
                        "omitted_optional_details": [],
                    },
                    {
                        "variant_id": "concise",
                        "spoken_text": "Sawubona kwi-EFF",
                        "target_duration_ms": 1640,
                        "estimated_duration_ms": 1700,
                        "meaning_preserved": True,
                        "omitted_optional_details": [],
                    },
                    {
                        "variant_id": "compact",
                        "spoken_text": "I-EFF, sawubona",
                        "target_duration_ms": 1360,
                        "estimated_duration_ms": 1200,
                        "meaning_preserved": True,
                        "omitted_optional_details": [],
                    },
                ],
            }
        ],
    }


def test_loader_preserves_three_prepared_variants() -> None:
    segment = load_approved_segments(
        _translation(),
        source_duration_ms=2000,
    )[0]
    assert [item.variant_id for item in segment.variants] == [
        "natural",
        "concise",
        "compact",
    ]
    assert segment.translated_text == "Sawubona kwi-EFF namhlanje"


def test_duration_ranking_prefers_longest_predicted_fit() -> None:
    block = SpeechBlock(
        block_id="block_0001",
        speaker_id="SPEAKER_00",
        start_ms=0,
        end_ms=2000,
        segment_ids=("seg-1",),
        source_text="Hello",
        translated_text="natural",
        tts_text="natural",
        variants=(
            PreparedTranslationVariant("natural", "natural", "natural", 1900, 2600),
            PreparedTranslationVariant("concise", "concise", "concise", 1640, 1700),
            PreparedTranslationVariant("compact", "compact", "compact", 1360, 1200),
        ),
    )
    candidates = DirectAzureDubRenderer._variant_candidates(block)
    assert [item.variant_id for item in candidates] == [
        "concise",
        "compact",
        "natural",
    ]


def test_timing_is_resolved_only_from_prepared_variants() -> None:
    options = DirectDubOptions()
    assert not hasattr(options, "timing_repair_backend")


def test_exhausted_variants_keep_shortest_azure_measurement(tmp_path: Path) -> None:
    block = SpeechBlock(
        block_id="block_0015",
        speaker_id="SPEAKER_00",
        start_ms=439_630,
        end_ms=445_057,
        segment_ids=("unit_0074",),
        source_text="Thank you for joining us.",
        translated_text="natural",
        tts_text="natural",
        variants=(
            PreparedTranslationVariant("natural", "natural", "natural", 5_000, 7_000),
            PreparedTranslationVariant("concise", "concise", "concise", 4_500, 6_500),
            PreparedTranslationVariant("compact", "compact", "compact", 4_000, 6_000),
        ),
    )
    measured = {"natural": 7_789, "concise": 7_205, "compact": 6_289}
    renderer = object.__new__(DirectAzureDubRenderer)

    def synthesize(candidate, *_args, **_kwargs):
        duration = measured[candidate.selected_variant_id]
        raise TimingOverflowError(
            "overflow",
            measured_duration_ms=duration,
            target_duration_ms=candidate.duration_ms,
            stretch_ratio=duration / candidate.duration_ms,
        )

    renderer._synthesize_block = synthesize

    class Progress:
        def emit(self, *_args, **_kwargs):
            return None

    with pytest.raises(PreparedVariantsExhausted) as caught:
        renderer._synthesize_prepared_variants(
            block,
            DirectDubArtifacts(tmp_path),
            voice="zu-ZA-ThembaNeural",
            base_rate_percent=0,
            pitch_percent=0,
            volume_percent=0,
            options=DirectDubOptions(),
            force=False,
            progress=Progress(),
            block_number=15,
            block_total=15,
        )

    assert caught.value.block.selected_variant_id == "compact"
    assert caught.value.overflow.measured_duration_ms == 6_289


def test_loader_accepts_dubbing_unit_style_hard_timing_boundary() -> None:
    translation = _translation()
    unit = translation["units"][0]
    unit.pop("end_ms")
    unit["preferred_end_ms"] = 1_700
    unit["hard_end_ms"] = 2_000
    unit["preferred_duration_ms"] = 1_700
    unit["maximum_duration_ms"] = 2_000

    segment = load_approved_segments(
        translation,
        source_duration_ms=2_000,
    )[0]

    assert segment.start_ms == 0
    assert segment.end_ms == 2_000
