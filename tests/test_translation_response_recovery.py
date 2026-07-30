from __future__ import annotations

import json
from pathlib import Path

import pytest

from mathula_tv.ai_provider import (
    AnthropicClaudeProvider,
    AnthropicConfig,
    StructuredAIRequest,
)
from mathula_tv.atomic_io import atomic_write_json, read_json
from mathula_tv.config import Settings
from mathula_tv.errors import ClaudeInvalidStructuredOutput
from mathula_tv.media import checksum
from mathula_tv.multivariant_translation import (
    MultivariantTranslationError,
    build_job_translation_request,
    normalize_multivariant_response,
    validate_multivariant_response,
)
from mathula_tv.production_pipeline import ProductionDubbingPipeline


class _Response:
    status_code = 200
    headers: dict[str, str] = {}

    def __init__(self, body: dict) -> None:
        self._body = body

    def json(self) -> dict:
        return self._body


class _Session:
    def post(self, _url: str, **_kwargs):
        return _Response(
            {
                "model": "claude-opus-4-8",
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps({"answer": "preserve me"}),
                    }
                ],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 3, "output_tokens": 4},
            }
        )


def test_provider_validation_failure_preserves_raw_and_parsed_candidate() -> None:
    provider = AnthropicClaudeProvider(
        AnthropicConfig(api_key="test", max_retries=1, max_repairs=0),
        session=_Session(),
        sleep=lambda _delay: None,
    )
    request = StructuredAIRequest(
        operation="preservation-test",
        payload={"source": "data"},
        output_schema={
            "type": "object",
            "required": ["answer"],
            "properties": {"answer": {"type": "string"}},
            "additionalProperties": False,
        },
        prompt_version="test-v1",
        system_prompt="Return JSON.",
        response_schema_version="test-response-v1",
        max_repairs=0,
        validator=lambda _value: (_ for _ in ()).throw(ValueError("local rejection")),
    )

    with pytest.raises(ClaudeInvalidStructuredOutput) as raised:
        provider.complete_structured(request)

    details = raised.value.to_dict()["details"]
    assert details["validation_stage"] == "validator"
    assert details["raw_output"] == '{"answer": "preserve me"}'
    assert details["candidate_output"] == {"answer": "preserve me"}
    assert details["candidate_recoverable_without_provider_call"] is True
    assert len(details["raw_output_sha256"]) == 64
    assert len(details["candidate_output_sha256"]) == 64


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        tmp_path,
        "bucket",
        "mathula-tv",
        tmp_path / "case",
        tmp_path / "politics",
        "",
        "eastus",
        "en-ZA",
        "2025-10-15",
        10,
        600,
        "",
        "",
        "preview",
        "model",
    )


def _recovery_job(
    tmp_path: Path,
    *,
    unit_id: str = "unit_0001",
    source_text: str = "She was appointed as the head of Crime Intelligence in 2025.",
    protected_spans: list[str] | None = None,
):
    protected_spans = list(protected_spans or [])
    pipeline = ProductionDubbingPipeline(_settings(tmp_path))
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    job = pipeline.jobs.create(
        source_filename="source.mp4",
        local_source_path=str(source),
        source_checksum=checksum(source),
        source_duration=7.0,
        target_language="zu-ZA",
    )
    job.state = "translation_running"
    pipeline.jobs.save(job)
    root = pipeline.jobs.job_dir(job.job_id)
    transcript = {
        "schema_version": "transcript-v1",
        "language": "en-ZA",
        "autocorrect": {"rendered": True},
        "segments": [
            {
                "segment_id": unit_id,
                "speaker": "SPEAKER_00",
                "start": 0.0,
                "end": 7.0,
                "source_text": source_text,
                "protected_spans": protected_spans,
            }
        ],
    }
    transcript_path = root / "analysis" / "transcript_en.json"
    atomic_write_json(transcript_path, transcript)
    atomic_write_json(
        root / "analysis" / "autocorrection_state.json",
        {
            "ready_for_language_ai": True,
            "blocking_review_count": 0,
            "authoritative_transcript_sha256": checksum(transcript_path),
        },
    )
    units = {
        "schema_version": "dubbing-units-v1",
        "units": [
            {
                "unit_id": unit_id,
                "speaker_id": "SPEAKER_00",
                "start_ms": 0,
                "end_ms": 7000,
                "available_duration_ms": 7000,
                "source_text": source_text,
                "protected_spans": protected_spans,
            }
        ],
    }
    atomic_write_json(root / "dubbing" / "dubbing_units.json", units)
    atomic_write_json(
        root / "analysis" / "entity_bindings.json",
        {
            "source_transcript_sha256": checksum(transcript_path),
            "source_dubbing_units_sha256": checksum(
                root / "dubbing" / "dubbing_units.json"
            ),
            "registry_sha256": "test",
            "per_unit_bindings": {},
        },
    )
    pipeline.bind_entities = lambda _job, force=False: read_json(  # type: ignore[method-assign]
        root / "analysis" / "entity_bindings.json"
    )
    pipeline.build_units = lambda _job, force=False: read_json(  # type: ignore[method-assign]
        root / "dubbing" / "dubbing_units.json"
    )
    request, _paths = build_job_translation_request(
        job_root=root,
        job_id=job.job_id,
        target_locale="zu-ZA",
    )
    source_unit = request["units"][0]
    targets = source_unit["duration_targets"]
    candidate = {
        "schema_version": "mathula.translation.multivariant.v1",
        "source_language": "en-ZA",
        "target_language": "isiZulu",
        "target_locale": "zu-ZA",
        "translation_strategy": "contextual_code_switching",
        "units": [
            {
                "unit_id": unit_id,
                "speaker_id": source_unit["speaker_id"],
                "start_ms": source_unit["start_ms"],
                "end_ms": source_unit["end_ms"],
                "available_duration_ms": source_unit["available_duration_ms"],
                "overlap_group": source_unit.get("overlap_group"),
                "source_text": source_unit["source_text"],
                "protected_spans": protected_spans,
                "required_facts": [],
                "optional_details": [],
                "variants": [
                    {
                        "variant_id": "natural",
                        "compression_level": "natural",
                        "target_duration_ms": targets["natural_target_duration_ms"],
                        "spoken_text": (
                            "Waqokwa njengenhloko ye-Crime Intelligence ngo-2025."
                        ),
                        "estimated_duration_ms": 4400,
                        "preserved_english_spans": [],
                        "omitted_optional_details": [],
                        "meaning_preserved": True,
                    },
                    {
                        "variant_id": "concise",
                        "compression_level": "concise",
                        "target_duration_ms": targets["concise_target_duration_ms"],
                        "spoken_text": "Waba yinhloko ye-Crime Intelligence ngo-2025.",
                        "estimated_duration_ms": 3600,
                        "preserved_english_spans": [],
                        "omitted_optional_details": [],
                        "meaning_preserved": True,
                    },
                    {
                        "variant_id": "compact",
                        "compression_level": "compact",
                        "target_duration_ms": targets["compact_target_duration_ms"],
                        "spoken_text": "Waqokwa ngo-2025.",
                        "estimated_duration_ms": 1800,
                        "preserved_english_spans": [],
                        "omitted_optional_details": [
                            "Crime Intelligence Head role"
                        ],
                        "meaning_preserved": True,
                    },
                ],
                "selection_status": "unselected",
                "selected_variant_id": None,
                "warnings": [],
            }
        ],
        "global_terminology": [],
        "warnings": [],
    }
    return pipeline, job, root, request, candidate


def test_semantic_role_label_is_promoted_as_source_grounded_metadata(tmp_path: Path) -> None:
    _pipeline, _job, _root, request, candidate = _recovery_job(tmp_path)

    normalized = normalize_multivariant_response(candidate, request=request)
    validate_multivariant_response(normalized, request=request)

    assert normalized["units"][0]["optional_details"] == [
        "Crime Intelligence Head role"
    ]


def test_failed_cloud_candidate_is_installed_without_provider_call(tmp_path: Path) -> None:
    pipeline, job, root, request, candidate = _recovery_job(tmp_path)
    run_dir = root / "translation" / "cloud_runs" / "20260728T142749.665381Z"
    atomic_write_json(run_dir / "translation_request.json", request)
    atomic_write_json(
        run_dir / "failure.json",
        {
            "error": {
                "details": {
                    "provider": "azure-foundry-claude",
                    "model_requested": "claude-opus-4-8",
                    "model_returned": "claude-opus-4-8",
                    "prompt_version": request["prompt_version"],
                    "candidate_output": candidate,
                    "candidate_output_sha256": "a" * 64,
                }
            }
        },
    )
    atomic_write_json(run_dir / "status.json", {"status": "failed"})
    job.media["translation_run_directory"] = str(run_dir)
    pipeline.jobs.save(job)

    result = pipeline.recover_translation_response(job)

    assert result["provider_http_requests_started"] == 0
    assert result["status"] == "installed"
    assert (run_dir / "translation_response.recovered.json").is_file()
    assert (run_dir / "recovery_manifest.json").is_file()
    stored = pipeline.jobs.load(job.job_id)
    assert stored.state == "azure_tts_queued"
    assert stored.last_error is None
    assert pipeline.paths(job.job_id).raw_multivariant_translation.is_file()
    assert pipeline.paths(job.job_id).three_text_translation.is_file()


def test_khumalo_identity_candidate_recovers_without_provider_call(
    tmp_path: Path,
) -> None:
    source_text = (
        "Earlier, we heard one of the Commissioners, "
        "Advocate-Commissioner Khumalo, in fact, talk about how,"
    )
    pipeline, job, root, request, candidate = _recovery_job(
        tmp_path,
        unit_id="unit_0023",
        source_text=source_text,
        protected_spans=["Advocate Sandile Khumalo SC"],
    )
    unit = candidate["units"][0]
    unit["required_facts"] = ["Commissioner Khumalo talked"]
    spoken = [
        (
            "Ekuqaleni, sizwe omunye wamaKhomishana, "
            "u-Advocate Khumalo, empeleni, ekhuluma ngokuthi,"
        ),
        "Ekuqaleni sizwe u-Advocate Khumalo, iKhomishana, ekhuluma ngokuthi,",
        "Sizwe u-Advocate Khumalo ekhuluma ngokuthi,",
    ]
    estimates = [5480, 4720, 3900]
    for variant, spoken_text, estimate in zip(
        unit["variants"], spoken, estimates, strict=True
    ):
        variant["spoken_text"] = spoken_text
        variant["estimated_duration_ms"] = estimate
        variant["preserved_english_spans"] = ["Advocate"]
        variant["omitted_optional_details"] = []

    run_dir = root / "translation" / "cloud_runs" / "20260728T203754.765108Z"
    atomic_write_json(run_dir / "translation_request.json", request)
    atomic_write_json(
        run_dir / "failure.json",
        {
            "error": {
                "details": {
                    "provider": "azure-foundry-claude",
                    "model_requested": "claude-opus-4-8",
                    "model_returned": "claude-opus-4-8",
                    "prompt_version": request["prompt_version"],
                    "candidate_output": candidate,
                    "candidate_output_sha256": "b" * 64,
                }
            }
        },
    )
    atomic_write_json(run_dir / "status.json", {"status": "failed"})
    job.media["translation_run_directory"] = str(run_dir)
    pipeline.jobs.save(job)

    result = pipeline.recover_translation_response(job)

    assert result["provider_http_requests_started"] == 0
    assert result["status"] == "installed"
    recovered = read_json(run_dir / "translation_response.recovered.json")
    validation = validate_multivariant_response(recovered, request=request)
    evidence = [
        item["protected_span_evidence"][0]
        for item in validation["units"][0]["variants"]
    ]
    assert [item["matched_form"] for item in evidence] == [
        "Advocate Khumalo",
        "Advocate Khumalo",
        "Advocate Khumalo",
    ]
    assert all("Khumalo" not in item["approved_forms"] for item in evidence)


def test_missing_protected_identity_in_shorter_variants_reuses_richer_variant() -> None:
    request = {
        "schema_version": "mathula.translation.request.v1",
        "job_id": "job-identity-floor",
        "source_language": "en-ZA",
        "target_language": "isiZulu",
        "target_locale": "zu-ZA",
        "translation_strategy": "contextual_code_switching",
        "prompt_version": "test",
        "source_transcript_sha256": "a" * 64,
        "source_timeline_sha256": "b" * 64,
        "source_timeline_kind": "dubbing_units",
        "request_sha256": "c" * 64,
        "units": [
            {
                "unit_id": "unit_0067",
                "speaker_id": "SPEAKER_00",
                "start_ms": 0,
                "end_ms": 7000,
                "available_duration_ms": 7000,
                "overlap_group": None,
                "source_text": "The matter involved SAPS Crime Intelligence.",
                "protected_spans": ["SAPS Crime Intelligence"],
                "required_facts": [],
                "optional_details": [],
                "duration_targets": {
                    "natural_target_duration_ms": 6300,
                    "concise_target_duration_ms": 5250,
                    "compact_target_duration_ms": 4200,
                },
            }
        ],
    }
    response = {
        "schema_version": "mathula.translation.multivariant.v1",
        "source_language": "en-ZA",
        "target_language": "isiZulu",
        "target_locale": "zu-ZA",
        "translation_strategy": "contextual_code_switching",
        "units": [
            {
                "unit_id": "unit_0067",
                "required_facts": [],
                "optional_details": [],
                "variants": [
                    {
                        "variant_id": "natural",
                        "spoken_text": "Udaba lwalubandakanya i-SAPS Crime Intelligence.",
                        "estimated_duration_ms": 4200,
                        "preserved_english_spans": ["SAPS Crime Intelligence"],
                        "omitted_optional_details": [],
                        "meaning_preserved": True,
                    },
                    {
                        "variant_id": "concise",
                        "spoken_text": "Udaba lwalubandakanya uphiko lwezobunhloli.",
                        "estimated_duration_ms": 3300,
                        "preserved_english_spans": [],
                        "omitted_optional_details": [],
                        "meaning_preserved": True,
                    },
                    {
                        "variant_id": "compact",
                        "spoken_text": "Lwalubandakanya ezobunhloli.",
                        "estimated_duration_ms": 2200,
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

    normalized = normalize_multivariant_response(response, request=request)
    validation = validate_multivariant_response(normalized, request=request)

    variants = normalized["units"][0]["variants"]
    assert [item["spoken_text"] for item in variants] == [
        "Udaba lwalubandakanya i-SAPS Crime Intelligence.",
        "Udaba lwalubandakanya i-SAPS Crime Intelligence.",
        "Udaba lwalubandakanya i-SAPS Crime Intelligence.",
    ]
    assert [item["estimated_duration_ms"] for item in variants] == [4200, 4200, 4200]
    repairs = [
        item
        for item in normalized["units"][0]["warnings"]
        if item.get("code") == "protected_identity_compression_floor_fallback"
    ]
    assert [(item["from_variant_id"], item["to_variant_id"]) for item in repairs] == [
        ("natural", "concise"),
        ("concise", "compact"),
    ]
    assert validation["units"][0]["compression_floor_reached"] is True
    assert validation["units"][0]["equivalent_variant_groups"] == [
        ["natural", "concise", "compact"]
    ]


def test_missing_protected_identity_in_natural_still_fails_closed() -> None:
    request = {
        "schema_version": "mathula.translation.request.v1",
        "job_id": "job-identity-hard-fail",
        "source_language": "en-ZA",
        "target_language": "isiZulu",
        "target_locale": "zu-ZA",
        "translation_strategy": "contextual_code_switching",
        "prompt_version": "test",
        "source_transcript_sha256": "a" * 64,
        "source_timeline_sha256": "b" * 64,
        "source_timeline_kind": "dubbing_units",
        "request_sha256": "c" * 64,
        "units": [
            {
                "unit_id": "unit_0001",
                "speaker_id": "SPEAKER_00",
                "start_ms": 0,
                "end_ms": 7000,
                "available_duration_ms": 7000,
                "overlap_group": None,
                "source_text": "The matter involved SAPS Crime Intelligence.",
                "protected_spans": ["SAPS Crime Intelligence"],
                "required_facts": [],
                "optional_details": [],
                "duration_targets": {
                    "natural_target_duration_ms": 6300,
                    "concise_target_duration_ms": 5250,
                    "compact_target_duration_ms": 4200,
                },
            }
        ],
    }
    response = {
        "schema_version": "mathula.translation.multivariant.v1",
        "source_language": "en-ZA",
        "target_language": "isiZulu",
        "target_locale": "zu-ZA",
        "translation_strategy": "contextual_code_switching",
        "units": [
            {
                "unit_id": "unit_0001",
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
                    for variant_id, text, duration in (
                        ("natural", "Udaba lwalubandakanya uphiko lwezobunhloli.", 4200),
                        ("concise", "Lwalubandakanya uphiko lwezobunhloli.", 3300),
                        ("compact", "Lwalubandakanya ezobunhloli.", 2200),
                    )
                ],
            }
        ],
        "global_terminology": [],
        "warnings": [],
    }

    normalized = normalize_multivariant_response(response, request=request)
    with pytest.raises(MultivariantTranslationError, match="SAPS Crime Intelligence"):
        validate_multivariant_response(normalized, request=request)


def test_compact_undeclared_role_omission_reuses_concise_variant() -> None:
    request = {
        "schema_version": "mathula.translation.request.v1",
        "job_id": "job-undeclared-role-floor",
        "source_language": "en-ZA",
        "target_language": "isiZulu",
        "target_locale": "zu-ZA",
        "translation_strategy": "contextual_code_switching",
        "prompt_version": "test",
        "source_transcript_sha256": "a" * 64,
        "source_timeline_sha256": "b" * 64,
        "source_timeline_kind": "dubbing_units",
        "request_sha256": "c" * 64,
        "units": [
            {
                "unit_id": "unit_0004",
                "speaker_id": "SPEAKER_00",
                "start_ms": 0,
                "end_ms": 8000,
                "available_duration_ms": 8000,
                "overlap_group": None,
                "source_text": (
                    "National Coloured Congress leader Fadiel Adams said the "
                    "matter should be investigated."
                ),
                "protected_spans": ["National Coloured Congress"],
                "required_facts": [],
                "optional_details": [],
                "duration_targets": {
                    "natural_target_duration_ms": 7200,
                    "concise_target_duration_ms": 6000,
                    "compact_target_duration_ms": 4800,
                },
            }
        ],
    }
    response = {
        "schema_version": "mathula.translation.multivariant.v1",
        "source_language": "en-ZA",
        "target_language": "isiZulu",
        "target_locale": "zu-ZA",
        "translation_strategy": "contextual_code_switching",
        "units": [
            {
                "unit_id": "unit_0004",
                "required_facts": [],
                "optional_details": [],
                "variants": [
                    {
                        "variant_id": "natural",
                        "spoken_text": (
                            "Umholi we-National Coloured Congress uFadiel Adams "
                            "uthe lolu daba kufanele luphenywe."
                        ),
                        "estimated_duration_ms": 6200,
                        "preserved_english_spans": ["National Coloured Congress"],
                        "omitted_optional_details": [],
                        "meaning_preserved": True,
                    },
                    {
                        "variant_id": "concise",
                        "spoken_text": (
                            "Umholi we-National Coloured Congress uFadiel Adams "
                            "uthe udaba maluphenywe."
                        ),
                        "estimated_duration_ms": 5100,
                        "preserved_english_spans": ["National Coloured Congress"],
                        "omitted_optional_details": [],
                        "meaning_preserved": True,
                    },
                    {
                        "variant_id": "compact",
                        "spoken_text": (
                            "I-National Coloured Congress ithe udaba maluphenywe."
                        ),
                        "estimated_duration_ms": 3500,
                        "preserved_english_spans": ["National Coloured Congress"],
                        "omitted_optional_details": [
                            "National Coloured Congress leader"
                        ],
                        "meaning_preserved": True,
                    },
                ],
            }
        ],
        "global_terminology": [],
        "warnings": [],
    }

    normalized = normalize_multivariant_response(response, request=request)
    validation = validate_multivariant_response(normalized, request=request)

    variants = normalized["units"][0]["variants"]
    assert variants[2]["spoken_text"] == variants[1]["spoken_text"]
    assert variants[2]["estimated_duration_ms"] == variants[1][
        "estimated_duration_ms"
    ]
    assert variants[2]["omitted_optional_details"] == []
    repairs = [
        item
        for item in normalized["units"][0]["warnings"]
        if item.get("code") == "undeclared_omission_compression_floor_fallback"
    ]
    assert repairs == [
        {
            "code": "undeclared_omission_compression_floor_fallback",
            "from_variant_id": "concise",
            "to_variant_id": "compact",
            "undeclared_details": ["National Coloured Congress leader"],
            "discarded_spoken_text": (
                "I-National Coloured Congress ithe udaba maluphenywe."
            ),
            "replacement_policy": "reuse_nearest_richer_valid_variant_verbatim",
        }
    ]
    assert validation["units"][0]["compression_floor_reached"] is True
    assert validation["units"][0]["equivalent_variant_groups"] == [
        ["concise", "compact"]
    ]


def test_natural_undeclared_omission_still_fails_closed() -> None:
    request = {
        "schema_version": "mathula.translation.request.v1",
        "job_id": "job-natural-undeclared-hard-fail",
        "source_language": "en-ZA",
        "target_language": "isiZulu",
        "target_locale": "zu-ZA",
        "translation_strategy": "contextual_code_switching",
        "prompt_version": "test",
        "source_transcript_sha256": "a" * 64,
        "source_timeline_sha256": "b" * 64,
        "source_timeline_kind": "dubbing_units",
        "request_sha256": "c" * 64,
        "units": [
            {
                "unit_id": "unit_0001",
                "speaker_id": "SPEAKER_00",
                "start_ms": 0,
                "end_ms": 5000,
                "available_duration_ms": 5000,
                "overlap_group": None,
                "source_text": "The minister announced the inquiry.",
                "protected_spans": [],
                "required_facts": [],
                "optional_details": [],
                "duration_targets": {
                    "natural_target_duration_ms": 4500,
                    "concise_target_duration_ms": 3750,
                    "compact_target_duration_ms": 3000,
                },
            }
        ],
    }
    response = {
        "schema_version": "mathula.translation.multivariant.v1",
        "source_language": "en-ZA",
        "target_language": "isiZulu",
        "target_locale": "zu-ZA",
        "translation_strategy": "contextual_code_switching",
        "units": [
            {
                "unit_id": "unit_0001",
                "required_facts": [],
                "optional_details": [],
                "variants": [
                    {
                        "variant_id": "natural",
                        "spoken_text": "Kumenyezelwe uphenyo.",
                        "estimated_duration_ms": 2800,
                        "preserved_english_spans": [],
                        "omitted_optional_details": ["the minister"],
                        "meaning_preserved": True,
                    },
                    {
                        "variant_id": "concise",
                        "spoken_text": "Kumenyezelwe uphenyo manje.",
                        "estimated_duration_ms": 2400,
                        "preserved_english_spans": [],
                        "omitted_optional_details": [],
                        "meaning_preserved": True,
                    },
                    {
                        "variant_id": "compact",
                        "spoken_text": "Up henyo lumenyezelwe.",
                        "estimated_duration_ms": 1900,
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

    normalized = normalize_multivariant_response(response, request=request)
    with pytest.raises(MultivariantTranslationError, match="the minister"):
        validate_multivariant_response(normalized, request=request)


def test_split_commissioner_khumalo_candidate_recovers_without_provider_call(
    tmp_path: Path,
) -> None:
    source_text = (
        "What Commissioner Khumalo said earlier regarding this emerging pattern "
        "of IDAC going beyond the scope of its work and how that may, on his reading,"
    )
    pipeline, job, root, request, candidate = _recovery_job(
        tmp_path,
        unit_id="unit_0029",
        source_text=source_text,
        protected_spans=[
            "Advocate Sandile Khumalo SC",
            "Investigating Directorate Against Corruption",
        ],
    )
    unit = candidate["units"][0]
    unit["required_facts"] = [
        "Commissioner Khumalo said",
        "pattern of IDAC going beyond scope",
        "on his reading",
    ]
    unit["optional_details"] = []
    spoken = [
        (
            "Lokho okushiwo yiKhomishana uKhumalo ngaphambilini mayelana nalo "
            "mkhuba ovelayo we-IDAC wokweqa umkhawulo womsebenzi wayo nokuthi "
            "lokho kungaba, ngokubuka kwakhe,"
        ),
        (
            "Lokho okushiwo yiKhomishana uKhumalo ngalo mkhuba we-IDAC wokweqa "
            "umkhawulo womsebenzi, nokuthi ngokubuka kwakhe,"
        ),
        "Okushiwo uKhumalo ngomkhuba we-IDAC wokweqa umkhawulo, nokuthi ngokubuka kwakhe,",
    ]
    estimates = [7090, 6110, 5060]
    for variant, spoken_text, estimate in zip(
        unit["variants"], spoken, estimates, strict=True
    ):
        variant["spoken_text"] = spoken_text
        variant["estimated_duration_ms"] = estimate
        variant["preserved_english_spans"] = ["IDAC"]
        variant["omitted_optional_details"] = []

    run_dir = root / "translation" / "cloud_runs" / "20260728T211859.936827Z"
    atomic_write_json(run_dir / "translation_request.json", request)
    atomic_write_json(
        run_dir / "failure.json",
        {
            "error": {
                "details": {
                    "provider": "azure-foundry-claude",
                    "model_requested": "claude-opus-4-8",
                    "model_returned": "claude-opus-4-8",
                    "prompt_version": request["prompt_version"],
                    "candidate_output": candidate,
                    "candidate_output_sha256": "c" * 64,
                }
            }
        },
    )
    atomic_write_json(run_dir / "status.json", {"status": "failed"})
    job.media["translation_run_directory"] = str(run_dir)
    pipeline.jobs.save(job)

    result = pipeline.recover_translation_response(job)

    assert result["provider_http_requests_started"] == 0
    assert result["status"] == "installed"
    recovered = read_json(run_dir / "translation_response.recovered.json")
    validation = validate_multivariant_response(recovered, request=request)
    variants = validation["units"][0]["variants"]
    khumalo_evidence = [
        next(
            item
            for item in variant["protected_span_evidence"]
            if item["required_span"] == "Advocate Sandile Khumalo SC"
        )
        for variant in variants
    ]
    assert [item["matched_form"] for item in khumalo_evidence[:2]] == [
        "Khomishana … Khumalo",
        "Khomishana … Khumalo",
    ]
    assert khumalo_evidence[2]["substitute_form"] == "Khumalo"
