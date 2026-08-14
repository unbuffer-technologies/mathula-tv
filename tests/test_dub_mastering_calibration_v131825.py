from __future__ import annotations

from pathlib import Path

from mathula_tv.direct_azure_dub import (
    DUB_MASTERING_OVERRIDE_SCHEMA_VERSION,
    PreparedTranslationVariant,
    SpeechBlock,
    apply_dub_mastering_overrides,
)
from mathula_tv.dub_mastering import analyze_mastered_dub


def _word(text: str, start: float, end: float) -> dict:
    return {"text": text, "start": start, "end": end, "confidence": 0.98}


def _block(
    block_id: str,
    *,
    start_ms: int,
    end_ms: int,
    spoken_text: str,
    tts_text: str | None = None,
    protected_entities: list[str] | None = None,
) -> dict:
    tts_text = tts_text or spoken_text
    return {
        "block_id": block_id,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "speaker_id": "SPEAKER_01",
        "translated_text": spoken_text,
        "tts_text": tts_text,
        "protected_entities": protected_entities or [],
        "translation_variant": {
            "selected_variant_id": "natural",
            "spoken_text": spoken_text,
            "tts_text": tts_text,
        },
        "variants": [
            {
                "variant_id": "natural",
                "spoken_text": spoken_text,
                "tts_text": tts_text,
                "meaning_preserved": True,
            }
        ],
        "base_prosody": {"rate_percent": 0},
        "selected_azure_rate_percent": 0,
        "post_synthesis_time_stretch_ratio": 1.0,
    }


def test_hidden_tts_reference_does_not_create_false_word_errors(tmp_path: Path) -> None:
    audio = tmp_path / "mix.wav"
    audio.write_bytes(b"audio")
    report = {
        "blocks": [
            _block(
                "block_0001_variant_natural",
                start_ms=0,
                end_ms=3000,
                spoken_text="Sikhulume noMnu. Adams",
                tts_text="Sikhulume noMnumzane Adams",
                protected_entities=["Fadiel Adams"],
            )
        ]
    }
    normalized = {
        "provider": "azure",
        "phrases": [],
        "words": [
            _word("Sikhulume", 0.1, 0.5),
            _word("noMnumzane", 0.6, 1.1),
            _word("Adams", 1.2, 1.6),
        ],
    }

    audit = analyze_mastered_dub(
        job_id="job",
        direct_dub_report=report,
        normalized_stt=normalized,
        audio_path=audio,
        round_number=0,
    )

    block = audit["blocks"][0]
    assert block["quality_reference_used"] == "tts_text"
    assert block["word_error_rate"] == 0
    assert block["protected_entities_missing"] == []
    assert block["protected_entities_assessed"][0]["expected_surface_tokens"] == [
        "adams"
    ]


def test_translated_generic_body_is_not_a_false_missing_english_identity(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "mix.wav"
    audio.write_bytes(b"audio")
    report = {
        "blocks": [
            _block(
                "block_0001_variant_natural",
                start_ms=0,
                end_ms=2000,
                spoken_text="iKhomishini ilalele",
                protected_entities=["Madlanga Commission of Inquiry"],
            )
        ]
    }
    normalized = {
        "provider": "azure",
        "phrases": [],
        "words": [
            _word("ikhomishini", 0.1, 0.7),
            _word("ilalele", 0.8, 1.2),
        ],
    }

    audit = analyze_mastered_dub(
        job_id="job",
        direct_dub_report=report,
        normalized_stt=normalized,
        audio_path=audio,
        round_number=0,
    )

    block = audit["blocks"][0]
    assert block["protected_entities_missing"] == []
    assert block["protected_entities_not_assessable"] == [
        "Madlanga Commission of Inquiry"
    ]
    assert audit["missing_protected_entity_count"] == 0


def test_global_word_alignment_recovers_words_across_rigid_block_boundary(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "mix.wav"
    audio.write_bytes(b"audio")
    report = {
        "blocks": [
            _block(
                "block_0001_variant_natural",
                start_ms=0,
                end_ms=1000,
                spoken_text="Yebo Sihlalo",
            ),
            _block(
                "block_0002_variant_natural",
                start_ms=1000,
                end_ms=2200,
                spoken_text="Ngiyabonga kakhulu",
            ),
        ]
    }
    normalized = {
        "provider": "azure",
        "phrases": [],
        "words": [
            _word("Yebo", 0.1, 0.4),
            # Azure midpoint is beyond block 1's rigid end.
            _word("Sihlalo", 0.9, 1.2),
            _word("Ngiyabonga", 1.25, 1.6),
            _word("kakhulu", 1.7, 2.0),
        ],
    }

    audit = analyze_mastered_dub(
        job_id="job",
        direct_dub_report=report,
        normalized_stt=normalized,
        audio_path=audio,
        round_number=0,
    )

    assert audit["blocks"][0]["word_error_rate"] == 0
    assert audit["blocks"][0]["recognized_word_count"] == 2
    assert audit["blocks"][0]["timestamp_window_word_count"] == 1
    assert audit["word_alignment"]["method"].startswith("global_monotonic")


def test_mastering_override_can_be_reversed_without_duplicate_variant_suffix() -> None:
    variants = tuple(
        PreparedTranslationVariant(
            variant_id=value,
            spoken_text=f"{value} text",
            tts_text=f"{value} text",
        )
        for value in ("natural", "concise", "compact")
    )
    original = SpeechBlock(
        block_id="block_0001",
        speaker_id="SPEAKER_01",
        start_ms=0,
        end_ms=1000,
        segment_ids=("unit_0001",),
        source_text="source",
        translated_text="natural text",
        tts_text="natural text",
        variants=variants,
    )

    def payload(variant_id: str) -> dict:
        return {
            "schema_version": DUB_MASTERING_OVERRIDE_SCHEMA_VERSION,
            "revision": 1,
            "overrides": [
                {
                    "block_id": "block_0001",
                    "strategy": "select_variant",
                    "variant_id": variant_id,
                    "reason_codes": ["excessive_time_compression"],
                }
            ],
        }

    concise, _ = apply_dub_mastering_overrides([original], payload("concise"))
    natural, _ = apply_dub_mastering_overrides(concise, payload("natural"))

    assert len(concise[0].variants) == 3
    assert natural[0].block_id == "block_0001_variant_natural"
    assert natural[0].translated_text == "natural text"
