from __future__ import annotations

from mathula_tv.multivariant_translation import (
    build_translation_request,
    normalize_multivariant_response,
    validate_multivariant_response,
)


def _request(source_text: str) -> dict:
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
                    "protected_spans": ["Madlanga Commission of Inquiry"],
                }
            ],
        },
        target_locale="zu-ZA",
        job_id="job-identity-prevalidation",
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


def test_expands_generic_commission_before_strict_validation() -> None:
    request = _request(
        "The Madlanga Commission of Inquiry heard the evidence."
    )
    normalized = normalize_multivariant_response(
        _response(
            (
                "IKhomishini izwe ubufakazi obugcwele.",
                "IKhomishini izwe ubufakazi.",
                "IKhomishini izwe.",
            )
        ),
        request=request,
    )

    validate_multivariant_response(normalized, request=request)

    assert all(
        "Madlanga" in variant["spoken_text"]
        for variant in normalized["units"][0]["variants"]
    )


def test_injects_reviewed_identity_when_provider_omits_commission_noun() -> None:
    request = _request(
        "The Madlanga Commission of Inquiry heard the evidence."
    )
    normalized = normalize_multivariant_response(
        _response(
            (
                "Ubufakazi obugcwele buzwakele.",
                "Ubufakazi buzwakele.",
                "Buzwakele.",
            )
        ),
        request=request,
    )

    validate_multivariant_response(normalized, request=request)

    assert all(
        "iKhomishini kaMadlanga" in variant["spoken_text"]
        for variant in normalized["units"][0]["variants"]
    )


def test_generic_source_alias_remains_generic() -> None:
    request = _request("The Commission heard the evidence.")
    normalized = normalize_multivariant_response(
        _response(
            (
                "IKhomishini izwe ubufakazi obugcwele.",
                "IKhomishini izwe ubufakazi.",
                "IKhomishini izwe.",
            )
        ),
        request=request,
    )

    validate_multivariant_response(normalized, request=request)

    assert all(
        "Madlanga" not in variant["spoken_text"]
        for variant in normalized["units"][0]["variants"]
    )
