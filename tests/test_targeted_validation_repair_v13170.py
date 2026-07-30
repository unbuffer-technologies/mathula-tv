from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from mathula_tv.atomic_io import atomic_write_json, read_json
from mathula_tv.config import Settings
from mathula_tv.errors import ClaudeInvalidStructuredOutput
from mathula_tv.media import checksum
from mathula_tv.multivariant_translation import (
    build_translation_request,
    normalize_multivariant_response,
    validate_multivariant_response,
)
from mathula_tv.one_call_translation import translate_full_transcript_once
from mathula_tv.production_pipeline import ProductionDubbingPipeline


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


def _job(tmp_path: Path):
    pipeline = ProductionDubbingPipeline(_settings(tmp_path))
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    job = pipeline.jobs.create(
        source_filename="source.mp4",
        local_source_path=str(source),
        source_checksum=checksum(source),
        source_duration=8.0,
        target_language="zu-ZA",
    )
    job.state = "analysis_ready"
    pipeline.jobs.save(job)
    root = pipeline.jobs.job_dir(job.job_id)
    source_text = (
        "If we do not find a better solution, we will continue with this one."
    )
    transcript_path = root / "analysis" / "transcript_en.json"
    atomic_write_json(
        transcript_path,
        {
            "schema_version": "transcript-v1",
            "language": "en-ZA",
            "autocorrect": {"rendered": True},
            "segments": [
                {
                    "segment_id": "unit_0018",
                    "speaker": "SPEAKER_00",
                    "start": 0.0,
                    "end": 8.0,
                    "source_text": source_text,
                    "protected_spans": [],
                }
            ],
        },
    )
    atomic_write_json(
        root / "analysis" / "autocorrection_state.json",
        {
            "ready_for_language_ai": True,
            "blocking_review_count": 0,
            "authoritative_transcript_sha256": checksum(transcript_path),
        },
    )
    units_path = root / "dubbing" / "dubbing_units.json"
    atomic_write_json(
        units_path,
        {
            "schema_version": "dubbing-units-v1",
            "units": [
                {
                    "unit_id": "unit_0018",
                    "speaker_id": "SPEAKER_00",
                    "start_ms": 0,
                    "end_ms": 8000,
                    "source_text": source_text,
                    "protected_spans": [],
                }
            ],
        },
    )
    atomic_write_json(
        root / "analysis" / "entity_bindings.json",
        {
            "source_transcript_sha256": checksum(transcript_path),
            "source_dubbing_units_sha256": checksum(units_path),
            "registry_sha256": "test",
            "per_unit_bindings": {},
        },
    )
    pipeline.bind_entities = lambda _job, force=False: read_json(  # type: ignore[method-assign]
        root / "analysis" / "entity_bindings.json"
    )
    pipeline.build_units = lambda _job, force=False: read_json(  # type: ignore[method-assign]
        units_path
    )
    return pipeline, job


def _response(request: dict, *, invalid_natural_omission: bool) -> dict:
    source = request["units"][0]
    targets = source["duration_targets"]
    texts = (
        "Uma singatholi isixazululo esingcono, sizoqhubeka nalesi.",
        "Uma singatholi esingcono, sizoqhubeka nalesi.",
        "Uma singatholi esingcono, sisebenzisa lesi.",
    )
    variants = []
    for variant_id, text, estimate in zip(
        ("natural", "concise", "compact"),
        texts,
        (5600, 4600, 3500),
        strict=True,
    ):
        variants.append(
            {
                "variant_id": variant_id,
                "compression_level": variant_id,
                "target_duration_ms": targets[
                    f"{variant_id}_target_duration_ms"
                ],
                "spoken_text": text,
                "estimated_duration_ms": estimate,
                "preserved_english_spans": [],
                "omitted_optional_details": (
                    ["uma singatholi elingcono"]
                    if invalid_natural_omission and variant_id == "natural"
                    else []
                ),
                "meaning_preserved": True,
            }
        )
    return {
        "schema_version": "mathula.translation.multivariant.v1",
        "source_language": "en-ZA",
        "target_language": "isiZulu",
        "target_locale": "zu-ZA",
        "translation_strategy": "contextual_code_switching",
        "units": [
            {
                "unit_id": source["unit_id"],
                "speaker_id": source["speaker_id"],
                "start_ms": source["start_ms"],
                "end_ms": source["end_ms"],
                "available_duration_ms": source["available_duration_ms"],
                "overlap_group": source.get("overlap_group"),
                "source_text": source["source_text"],
                "protected_spans": [],
                "required_facts": [],
                "optional_details": [],
                "variants": variants,
                "selection_status": "unselected",
                "selected_variant_id": None,
                "warnings": [],
            }
        ],
        "global_terminology": [],
        "warnings": [],
    }


class _RepairingProvider:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.config = SimpleNamespace(
            provider="azure-foundry-claude",
            base_url="https://example.invalid/anthropic",
            model="claude-opus-4-8",
        )

    def with_request_options(self, **_options):
        return self

    def translate_multivariant(self, request: dict):
        self.calls.append(request)
        if "targeted_validation_repair" in request:
            return SimpleNamespace(
                data=_response(request, invalid_natural_omission=False),
                metadata=SimpleNamespace(
                    to_dict=lambda: {
                        "provider": "azure-foundry-claude",
                        "model_returned": "claude-opus-4-8",
                        "provider_attempts": 1,
                        "token_usage": {
                            "input_tokens": 40,
                            "output_tokens": 20,
                        },
                    }
                ),
            )
        candidate = _response(request, invalid_natural_omission=True)
        raise ClaudeInvalidStructuredOutput(
            "Anthropic structured output failed validation after 0 repairs",
            details={
                "provider": "azure-foundry-claude",
                "model_returned": "claude-opus-4-8",
                "validation_stage": "validator",
                "validation_error": (
                    "MultivariantTranslationError: Unit unit_0018 variant "
                    "natural omits undeclared details: "
                    "['uma singatholi elingcono']"
                ),
                "candidate_output": candidate,
                "provider_attempts": 1,
                "token_usage": {
                    "input_tokens": 100,
                    "output_tokens": 80,
                },
            },
        )


class _CompactFailureProvider(_RepairingProvider):
    def translate_multivariant(self, request: dict):
        self.calls.append(request)
        if "targeted_validation_repair" in request:
            raise RuntimeError("targeted repair temporarily unavailable")
        candidate = _response(request, invalid_natural_omission=False)
        candidate["units"][0]["variants"][2][
            "omitted_optional_details"
        ] = ["uma singatholi elingcono"]
        raise ClaudeInvalidStructuredOutput(
            "Anthropic structured output failed validation after 0 repairs",
            details={
                "provider": "azure-foundry-claude",
                "model_returned": "claude-opus-4-8",
                "validation_stage": "validator",
                "validation_error": (
                    "MultivariantTranslationError: Unit unit_0018 variant "
                    "compact omits undeclared details: "
                    "['uma singatholi elingcono']"
                ),
                "candidate_output": candidate,
                "provider_attempts": 1,
                "token_usage": {
                    "input_tokens": 100,
                    "output_tokens": 80,
                },
            },
        )


class _NoCallProvider(_RepairingProvider):
    def translate_multivariant(self, request: dict):
        self.calls.append(request)
        raise AssertionError("saved compact candidate must recover locally")


def test_compact_target_language_omission_metadata_uses_safe_richer_variant() -> None:
    request = build_translation_request(
        transcript={
            "schema_version": "transcript-v1",
            "segments": [
                {
                    "segment_id": "unit_0018",
                    "speaker": "SPEAKER_00",
                    "start": 0.0,
                    "end": 8.0,
                    "source_text": (
                        "If we do not find a better solution, we will continue "
                        "with this one."
                    ),
                    "protected_spans": [],
                }
            ],
        },
        target_locale="zu-ZA",
        job_id="job-target-language-omission-fallback",
    )
    response = _response(request, invalid_natural_omission=False)
    compact = response["units"][0]["variants"][2]
    compact["omitted_optional_details"] = ["uma singatholi elingcono"]
    discarded_compact_text = compact["spoken_text"]

    normalized = normalize_multivariant_response(response, request=request)
    validation = validate_multivariant_response(normalized, request=request)

    variants = normalized["units"][0]["variants"]
    assert validation["valid"] is True
    assert variants[2]["spoken_text"] == variants[1]["spoken_text"]
    assert variants[2]["omitted_optional_details"] == []
    assert {
        "code": "undeclared_omission_compression_floor_fallback",
        "from_variant_id": "concise",
        "to_variant_id": "compact",
        "undeclared_details": ["uma singatholi elingcono"],
        "discarded_spoken_text": discarded_compact_text,
        "replacement_policy": "reuse_nearest_richer_valid_variant_verbatim",
    } in normalized["units"][0]["warnings"]


def test_unit_validator_failure_repairs_only_rejected_unit_and_continues(
    tmp_path: Path,
) -> None:
    pipeline, job = _job(tmp_path)
    provider = _RepairingProvider()

    result = translate_full_transcript_once(
        pipeline,
        job,
        provider,
        batch_size=18,
    )

    assert len(provider.calls) == 2
    repair_request = provider.calls[1]
    assert [item["unit_id"] for item in repair_request["units"]] == [
        "unit_0018"
    ]
    assert repair_request["targeted_validation_repair"]["variant_id"] == (
        "natural"
    )
    assert all(
        not variant["omitted_optional_details"]
        for variant in result["units"][0]["translation_variants"]
    )
    stored = pipeline.jobs.load(job.job_id)
    assert stored.state == "azure_tts_queued"
    chunk_dir = Path(stored.media["translation_batch_directory"]) / "chunk_0001"
    repair_dir = (
        chunk_dir
        / "targeted_validation_repairs"
        / "attempt_0001"
    )
    assert (repair_dir / "completed.json").is_file()
    metadata = read_json(chunk_dir / "provider_metadata.json")
    assert metadata["targeted_validation_repair_count"] == 1


def test_existing_compact_failure_resumes_without_another_provider_call(
    tmp_path: Path,
) -> None:
    pipeline, job = _job(tmp_path)

    with pytest.raises(ClaudeInvalidStructuredOutput):
        translate_full_transcript_once(
            pipeline,
            job,
            _CompactFailureProvider(),
            batch_size=18,
        )

    failed = pipeline.jobs.load(job.job_id)
    assert failed.state == "failed_retryable"
    provider = _NoCallProvider()
    result = translate_full_transcript_once(
        pipeline,
        failed,
        provider,
        batch_size=18,
    )

    assert provider.calls == []
    assert all(
        not variant["omitted_optional_details"]
        for variant in result["units"][0]["translation_variants"]
    )
    stored = pipeline.jobs.load(job.job_id)
    chunk_dir = Path(stored.media["translation_batch_directory"]) / "chunk_0001"
    recovery = read_json(chunk_dir / "recovery.json")
    assert recovery["provider_request_started"] is False
    assert (chunk_dir / "completed.json").is_file()
