from __future__ import annotations

from mathula_tv.direct_azure_dub import load_approved_segments
from mathula_tv.multivariant_translation import build_translation_request


def _variants(text: str) -> list[dict[str, object]]:
    return [
        {
            "variant_id": variant_id,
            "spoken_text": text,
            "estimated_duration_ms": duration,
            "meaning_preserved": True,
            "omitted_optional_details": [],
        }
        for variant_id, duration in (
            ("natural", 1_000),
            ("concise", 900),
            ("compact", 800),
        )
    ]


def test_installed_punctuation_only_unit_is_not_sent_to_tts() -> None:
    segments = load_approved_segments(
        {
            "units": [
                {
                    "unit_id": "unit_0100",
                    "speaker_id": "SPEAKER_01",
                    "start_ms": 566_260,
                    "end_ms": 573_770,
                    "source_text": ".......",
                    "variants": _variants("..."),
                },
                {
                    "unit_id": "unit_0101",
                    "speaker_id": "SPEAKER_01",
                    "start_ms": 573_770,
                    "end_ms": 575_770,
                    "source_text": "The witness continued.",
                    "variants": _variants("Ufakazi waqhubeka."),
                },
            ]
        },
        source_duration_ms=600_000,
    )

    assert [item.segment_id for item in segments] == ["unit_0101"]
    assert segments[0].tts_text == "Ufakazi waqhubeka."


def test_punctuation_only_source_is_excluded_before_ai_translation() -> None:
    request = build_translation_request(
        transcript={
            "language": "en-ZA",
            "segments": [
                {
                    "segment_id": "unit_0100",
                    "speaker": "SPEAKER_01",
                    "start": 566.260,
                    "end": 573.770,
                    "source_text": ".......",
                },
                {
                    "segment_id": "unit_0101",
                    "speaker": "SPEAKER_01",
                    "start": 573.770,
                    "end": 575.770,
                    "source_text": "The witness continued.",
                },
            ],
        },
        target_locale="zu-ZA",
        job_id="punctuation-only-filter",
    )

    assert [item["unit_id"] for item in request["units"]] == ["unit_0101"]


def test_lexical_text_with_ellipses_remains_speakable() -> None:
    segments = load_approved_segments(
        {
            "units": [
                {
                    "unit_id": "unit_0001",
                    "speaker_id": "SPEAKER_01",
                    "start_ms": 0,
                    "end_ms": 2_000,
                    "source_text": "And... I continued.",
                    "variants": _variants("Futhi... ngaqhubeka."),
                }
            ]
        },
        source_duration_ms=2_000,
    )

    assert len(segments) == 1
    assert segments[0].tts_text == "Futhi, ngaqhubeka."


def test_symbol_only_source_is_not_silently_classified_as_punctuation() -> None:
    segments = load_approved_segments(
        {
            "units": [
                {
                    "unit_id": "unit_0002",
                    "speaker_id": "SPEAKER_01",
                    "start_ms": 0,
                    "end_ms": 2_000,
                    "source_text": "$",
                    "variants": _variants("amadola"),
                }
            ]
        },
        source_duration_ms=2_000,
    )

    assert len(segments) == 1
    assert segments[0].tts_text == "amadola"
