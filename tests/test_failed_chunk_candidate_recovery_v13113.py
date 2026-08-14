from __future__ import annotations

from mathula_tv.atomic_io import atomic_write_json
from mathula_tv.multivariant_translation import build_translation_request
from mathula_tv.one_call_translation import _recover_failed_candidate


def test_saved_failed_candidate_is_recovered_without_provider_call(tmp_path) -> None:
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
        job_id="job-failed-candidate-recovery",
    )
    request["request_sha256"] = "request-0197"
    candidate = {
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
                        (
                            "IKhomishini izwe ubufakazi obugcwele.",
                            "IKhomishini izwe ubufakazi.",
                            "IKhomishini izwe.",
                        ),
                        (3000, 2500, 1900),
                        strict=True,
                    )
                ],
            }
        ],
        "global_terminology": [],
        "warnings": [],
    }
    atomic_write_json(
        tmp_path / "failure.json",
        {
            "request_sha256": "request-0197",
            "error": {
                "details": {
                    "provider": "azure-foundry-claude",
                    "model_returned": "claude-opus-4-8",
                    "prompt_version": "test",
                    "provider_attempts": 1,
                    "token_usage": {"input_tokens": 100, "output_tokens": 200},
                    "candidate_output": candidate,
                }
            },
        },
    )

    recovered = _recover_failed_candidate(
        chunk_dir=tmp_path,
        request=request,
    )

    assert recovered is not None
    response, metadata, validation = recovered
    assert metadata["recovered_from_failed_candidate"] is True
    assert validation["units"][0]["unit_id"] == "unit_0197"
    assert all(
        "Madlanga" in variant["spoken_text"]
        for variant in response["units"][0]["variants"]
    )
