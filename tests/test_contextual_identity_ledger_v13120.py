from __future__ import annotations

import pytest

from mathula_tv.multivariant_translation import (
    MultivariantTranslationError,
    build_translation_request,
    normalize_multivariant_response,
    validate_multivariant_response,
)


def _request(span: str, source_text: str) -> dict:
    return build_translation_request(
        transcript={
            "language": "en-ZA",
            "segments": [
                {
                    "segment_id": "unit_0197",
                    "speaker": "SPEAKER_01",
                    "start": 0,
                    "end": 5,
                    "source_text": source_text,
                    "protected_spans": [span],
                }
            ],
        },
        target_locale="zu-ZA",
        job_id="job-contextual-identity",
    )


def _response(texts: tuple[str, str, str]) -> dict:
    return {
        "units": [
            {
                "unit_id": "unit_0197",
                "required_facts": [],
                "optional_details": [],
                "variants": [
                    {
                        "variant_id": variant_id,
                        "spoken_text": text,
                        "estimated_duration_ms": duration,
                        "preserved_english_spans": [],
                        "omitted_optional_details": [],
                        "meaning_preserved": True,
                    }
                    for variant_id, text, duration in zip(
                        ("natural", "concise", "compact"),
                        texts,
                        (3000, 2500, 1900),
                        strict=True,
                    )
                ],
            }
        ],
        "global_terminology": [],
        "warnings": [],
    }


def test_contextual_institution_does_not_mutate_or_reject_spoken_text() -> None:
    request = _request(
        "Madlanga Commission of Inquiry",
        "The Madlanga Commission of Inquiry heard the evidence.",
    )
    original = (
        "Ubufakazi obugcwele buzwakele.",
        "Ubufakazi buzwakele.",
        "Buzwakele.",
    )
    normalized = normalize_multivariant_response(_response(original), request=request)
    validation = validate_multivariant_response(normalized, request=request)

    assert tuple(
        item["spoken_text"] for item in normalized["units"][0]["variants"]
    ) == original
    assert validation["contextual_identity_review_count"] == 3
    assert all(
        item["policy"] == "contextual_identity"
        for variant in validation["units"][0]["variants"]
        for item in variant["protected_span_evidence"]
    )


def test_person_identity_remains_strict_per_variant() -> None:
    request = _request(
        "Advocate Sandile Khumalo SC",
        "Advocate Sandile Khumalo SC addressed the Commission.",
    )
    normalized = normalize_multivariant_response(
        _response(("Ukhulumile.", "Ukhulume.", "Wakhuluma.")),
        request=request,
    )

    with pytest.raises(MultivariantTranslationError, match="Advocate Sandile Khumalo SC"):
        validate_multivariant_response(normalized, request=request)
