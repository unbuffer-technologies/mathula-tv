from __future__ import annotations

from mathula_tv.multivariant_translation import (
    build_translation_request,
    normalize_multivariant_response,
    validate_multivariant_response,
)


def test_named_madlanga_commission_is_restored_from_generic_zulu_noun():
    request = build_translation_request(
        transcript={
            "language": "en-ZA",
            "segments": [
                {
                    "segment_id": "unit_0197",
                    "speaker": "SPEAKER_01",
                    "start": 0,
                    "end": 5,
                    "source_text": (
                        "The Madlanga Commission of Inquiry heard the evidence."
                    ),
                    "protected_spans": ["Madlanga Commission of Inquiry"],
                }
            ],
        },
        target_locale="zu-ZA",
        job_id="job-identity",
    )
    raw = {
        "units": [
            {
                "unit_id": "unit_0197",
                "required_facts": ["the commission heard evidence"],
                "optional_details": [],
                "variants": [
                    {
                        "variant_id": "natural",
                        "spoken_text": "IKhomishini izwe ubufakazi.",
                        "estimated_duration_ms": 2500,
                        "preserved_english_spans": [],
                        "omitted_optional_details": [],
                        "meaning_preserved": True,
                    },
                    {
                        "variant_id": "concise",
                        "spoken_text": "IKhomishini izwe ubufakazi.",
                        "estimated_duration_ms": 2200,
                        "preserved_english_spans": [],
                        "omitted_optional_details": [],
                        "meaning_preserved": True,
                    },
                    {
                        "variant_id": "compact",
                        "spoken_text": "IKhomishini izwe.",
                        "estimated_duration_ms": 1800,
                        "preserved_english_spans": [],
                        "omitted_optional_details": [],
                        "meaning_preserved": True,
                    },
                ],
            }
        ],
        "global_terminology": [],
        "warnings": [],
    }

    normalized = normalize_multivariant_response(raw, request=request)
    validation = validate_multivariant_response(normalized, request=request)

    unit = normalized["units"][0]
    assert all(
        "kaMadlanga" in variant["spoken_text"]
        for variant in unit["variants"]
    )
    repair_codes = {
        warning.get("code")
        for warning in unit["warnings"]
        if isinstance(warning, dict)
    }
    assert "source_specific_identity_expanded_from_generic_target" in repair_codes
    assert validation["units"][0]["unit_id"] == "unit_0197"
