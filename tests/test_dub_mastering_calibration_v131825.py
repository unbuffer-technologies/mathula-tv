from __future__ import annotations

from array import array
from pathlib import Path
import wave

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


def test_protected_entity_accepts_spoken_number_and_phonetic_stt_spelling(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "mix.wav"
    audio.write_bytes(b"audio")
    report = {
        "blocks": [
            _block(
                "block_0001_variant_natural",
                start_ms=0,
                end_ms=3000,
                spoken_text="The Big Five cartel",
                tts_text="The Big Faiv cartel",
                protected_entities=["The Big Five cartel"],
            )
        ]
    }
    normalized = {
        "provider": "azure",
        "phrases": [{"text": "the big 5 curtle"}],
        "words": [
            _word("the", 0.1, 0.3),
            _word("big", 0.4, 0.7),
            _word("5", 0.8, 1.0),
            _word("curtle", 1.1, 1.5),
        ],
    }

    audit = analyze_mastered_dub(
        job_id="job-big-five",
        direct_dub_report=report,
        normalized_stt=normalized,
        audio_path=audio,
        round_number=0,
    )

    block = audit["blocks"][0]
    assert block["protected_entities_missing"] == []
    assert block["protected_entities_assessed"][0]["recognized"] is True


def test_protected_entity_uses_reviewed_tts_acronym_surface(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "mix.wav"
    audio.write_bytes(b"audio")
    report = {
        "blocks": [
            _block(
                "block_0001_variant_natural",
                start_ms=0,
                end_ms=3000,
                spoken_text=(
                    "Icala lizocutshungulwa yi-Independent Police "
                    "Investigative Directorate."
                ),
                tts_text="Icala lizocutshungulwa yi-Ai-pid.",
                protected_entities=[
                    "Independent Police Investigative Directorate"
                ],
            )
        ]
    }
    normalized = {
        "provider": "azure",
        "phrases": [{"text": "icala lizocutshungulwa yi-ipid"}],
        "words": [
            _word("icala", 0.1, 0.4),
            _word("lizocutshungulwa", 0.5, 1.2),
            _word("yi-ipid", 1.3, 1.8),
        ],
    }

    audit = analyze_mastered_dub(
        job_id="job-ipid",
        direct_dub_report=report,
        normalized_stt=normalized,
        audio_path=audio,
        round_number=0,
    )

    assessment = audit["blocks"][0]["protected_entities_assessed"][0]
    assert assessment["expected_surface_tokens"] == ["yiaipid"]
    assert assessment["recognized"] is True
    assert audit["blocks"][0]["production_blockers"] == []


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


def test_identity_matching_accepts_isizulu_affixes_and_joined_stt_tokens(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "mix.wav"
    audio.write_bytes(b"audio")
    report = {
        "blocks": [
            _block(
                "block_0001_variant_natural",
                start_ms=0,
                end_ms=4000,
                spoken_text=(
                    "uVusimuzi Matlala usehola i-Directorate for Priority "
                    "Crime Investigation"
                ),
                protected_entities=[
                    "Vusimuzi Matlala",
                    "Directorate for Priority Crime Investigation",
                ],
            )
        ]
    }
    normalized = {
        "provider": "azure",
        "phrases": [],
        "words": [
            _word("uvusimuzi", 0.1, 0.5),
            _word("matlala", 0.6, 0.9),
            _word("usehola", 1.0, 1.3),
            _word("id-directorate", 1.4, 1.8),
            _word("farpriority", 1.9, 2.4),
            _word("cream", 2.5, 2.8),
            _word("investigation", 2.9, 3.5),
        ],
    }

    audit = analyze_mastered_dub(
        job_id="job",
        direct_dub_report=report,
        normalized_stt=normalized,
        audio_path=audio,
        round_number=0,
    )

    assert audit["blocks"][0]["protected_entities_missing"] == []


def test_identity_matching_accepts_measured_code_switch_spellings(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "mix.wav"
    audio.write_bytes(b"audio")
    text = (
        "The Big Five cartel SAPS Crime Intelligence "
        "Vusimuzi Cat Matlala"
    )
    report = {
        "blocks": [
            _block(
                "block_0001_variant_natural",
                start_ms=0,
                end_ms=6000,
                spoken_text=text,
                protected_entities=[
                    "The Big Five cartel",
                    "SAPS Crime Intelligence",
                    "Vusimuzi Cat Matlala",
                ],
            )
        ]
    }
    observed = (
        "the big fever curtel ne-sebscreamer intelligence "
        "vusimuzi kathy matlala"
    ).split()
    normalized = {
        "provider": "azure",
        "phrases": [],
        "words": [
            _word(value, index * 0.5, index * 0.5 + 0.35)
            for index, value in enumerate(observed)
        ],
    }

    audit = analyze_mastered_dub(
        job_id="job",
        direct_dub_report=report,
        normalized_stt=normalized,
        audio_path=audio,
        round_number=0,
    )

    assert audit["blocks"][0]["protected_entities_missing"] == []


def test_stt_word_gap_with_dialogue_audio_is_not_counted_as_silence(
    tmp_path: Path,
) -> None:
    final_mix = tmp_path / "final_mix.wav"
    final_mix.write_bytes(b"mix")
    dialogue = tmp_path / "dialogue.wav"
    sample_rate = 8000
    samples = array("h", [0] * (sample_rate * 4))
    for index in range(round(0.35 * sample_rate), round(2.25 * sample_rate)):
        samples[index] = 5000 if index % 16 < 8 else -5000
    with wave.open(str(dialogue), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(samples.tobytes())
    text = "Ngakho ubufakazi be WhatsApp buyathakazelisa impela namuhla Adams uyavuma"
    report = {
        "blocks": [
            _block(
                "block_0001_variant_natural",
                start_ms=0,
                end_ms=4000,
                spoken_text=text,
            )
        ]
    }
    normalized = {
        "provider": "azure",
        "phrases": [],
        "words": [
            _word("Ngakho", 0.1, 0.4),
            _word("buyathakazelisa", 2.2, 2.5),
            _word("impela", 2.6, 2.9),
            _word("namuhla", 3.0, 3.2),
            _word("Adams", 3.25, 3.45),
            _word("uyavuma", 3.5, 3.7),
        ],
    }

    audit = analyze_mastered_dub(
        job_id="job",
        direct_dub_report=report,
        normalized_stt=normalized,
        audio_path=final_mix,
        round_number=0,
    )

    block = audit["blocks"][0]
    assert block["word_error_rate"] <= 0.35
    assert block["internal_pauses"] == []
    assert "unexpected_internal_pause" not in block["reason_codes"]


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
