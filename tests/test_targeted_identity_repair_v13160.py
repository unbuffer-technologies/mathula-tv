from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import mathula_tv.multivariant_translation as multivariant_translation_module
import mathula_tv.one_call_translation as one_call_translation_module
from mathula_tv.atomic_io import atomic_write_json, read_json
from mathula_tv.config import Settings
from mathula_tv.errors import ClaudeInvalidStructuredOutput
from mathula_tv.media import checksum
from mathula_tv.models import JobManifest
from mathula_tv.multivariant_translation import (
    MultivariantTranslationError,
    identity_requirements_for_unit,
)
from mathula_tv.one_call_translation import translate_full_transcript_once
from mathula_tv.production_pipeline import ProductionDubbingPipeline


IDENTITY = "Investigating Directorate Against Corruption"


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


def _job(tmp_path: Path) -> tuple[ProductionDubbingPipeline, JobManifest]:
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
        "The Investigating Directorate Against Corruption is investigating "
        "the allegation."
    )
    transcript = {
        "schema_version": "transcript-v1",
        "language": "en-ZA",
        "autocorrect": {"rendered": True},
        "segments": [
            {
                "segment_id": "unit_0039",
                "speaker": "SPEAKER_00",
                "start": 0.0,
                "end": 8.0,
                "source_text": source_text,
                "protected_spans": [IDENTITY],
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
    units_path = root / "dubbing" / "dubbing_units.json"
    atomic_write_json(
        units_path,
        {
            "schema_version": "dubbing-units-v1",
            "units": [
                {
                    "unit_id": "unit_0039",
                    "speaker_id": "SPEAKER_00",
                    "start_ms": 0,
                    "end_ms": 8000,
                    "source_text": source_text,
                    "protected_spans": [IDENTITY],
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


def _response(request: dict, *, identity_present: bool) -> dict:
    source = request["units"][0]
    targets = source["duration_targets"]
    if identity_present:
        texts = (
            f"I-{IDENTITY} iphenya lesi simangalo.",
            f"I-{IDENTITY} iphenya isimangalo.",
            f"I-{IDENTITY} iyaphenya.",
        )
        estimates = (5200, 4400, 3300)
    else:
        texts = (
            "Lolu phiko luphenya lesi simangalo.",
            "Lolu phiko luphenya isimangalo.",
            "Luyaphenya.",
        )
        estimates = (3900, 3200, 1600)
    variants = []
    for variant_id, text, duration in zip(
        ("natural", "concise", "compact"),
        texts,
        estimates,
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
                "estimated_duration_ms": duration,
                "preserved_english_spans": (
                    [IDENTITY] if identity_present else []
                ),
                "omitted_optional_details": [],
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
                "protected_spans": [IDENTITY],
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


def _metadata() -> SimpleNamespace:
    return SimpleNamespace(
        to_dict=lambda: {
            "provider": "azure-foundry-claude",
            "model_returned": "claude-opus-4-8",
            "prompt_version": "mathula-multivariant-translation-prompt-v3-compression-floor",
            "provider_attempts": 1,
            "token_usage": {"input_tokens": 20, "output_tokens": 30},
        }
    )


class RepairingProvider:
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
        if "targeted_identity_repair" in request:
            return SimpleNamespace(
                data=_response(request, identity_present=True),
                metadata=_metadata(),
            )
        candidate = _response(request, identity_present=False)
        validation_error = (
            "MultivariantTranslationError: Unit unit_0039 variant natural is "
            "missing protected spans or approved identity aliases: "
            f"['{IDENTITY} (accepted forms: {IDENTITY})']"
        )
        raise ClaudeInvalidStructuredOutput(
            "Anthropic structured output failed validation after 0 repairs",
            details={
                "provider": "azure-foundry-claude",
                "model_returned": "claude-opus-4-8",
                "prompt_version": (
                    "mathula-multivariant-translation-prompt-v3-compression-floor"
                ),
                "validation_stage": "validator",
                "validation_error": validation_error,
                "candidate_output": candidate,
                "provider_attempts": 1,
                "token_usage": {"input_tokens": 100, "output_tokens": 200},
            },
        )


class FailingProvider(RepairingProvider):
    def translate_multivariant(self, request: dict):
        if "targeted_identity_repair" in request:
            raise RuntimeError("targeted repair unavailable")
        return super().translate_multivariant(request)


class RepairOnlyProvider(RepairingProvider):
    def translate_multivariant(self, request: dict):
        self.calls.append(request)
        if "targeted_identity_repair" not in request:
            raise AssertionError("complete failed chunk must not be requested again")
        return SimpleNamespace(
            data=_response(request, identity_present=True),
            metadata=_metadata(),
        )


class NonIdentityFailingProvider(RepairingProvider):
    def translate_multivariant(self, request: dict):
        self.calls.append(request)
        raise RuntimeError("unrelated provider failure")


def test_identity_requirements_reuse_validator_alias_contract() -> None:
    requirements = identity_requirements_for_unit(
        {
            "source_text": f"The {IDENTITY} is investigating.",
            "protected_spans": [IDENTITY],
        },
        target_locale="zu-ZA",
    )
    assert len(requirements) == 1
    assert requirements[0]["required_span"] == IDENTITY
    assert requirements[0]["policy"] == "hard_identity"
    # Production registries may add reviewed aliases such as IDAC and
    # Investigating Directorate. The canonical identity must remain accepted,
    # but the test must not reject those additional safe aliases.
    assert IDENTITY in requirements[0]["accepted_forms"]


def test_failed_chunk_repairs_only_the_rejected_unit_and_checkpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_approved_forms = (
        multivariant_translation_module._approved_translation_identity_forms
    )

    def production_aliases(
        span: str,
        *,
        target_locale: str,
        source_text: str,
    ):
        if span == IDENTITY:
            return (
                IDENTITY,
                "IDAC",
                "Investigating Directorate",
            )
        return real_approved_forms(
            span,
            target_locale=target_locale,
            source_text=source_text,
        )

    monkeypatch.setattr(
        multivariant_translation_module,
        "_approved_translation_identity_forms",
        production_aliases,
    )
    pipeline, job = _job(tmp_path)
    provider = RepairingProvider()

    result = translate_full_transcript_once(
        pipeline,
        job,
        provider,
        batch_size=18,
    )

    assert len(provider.calls) == 1
    stored = pipeline.jobs.load(job.job_id)
    assert stored.state == "azure_tts_queued"
    session_root = Path(stored.media["translation_batch_directory"])
    repair_dir = (
        session_root
        / "chunk_0001"
        / "targeted_identity_repairs"
        / "attempt_0001"
    )
    repair_request = read_json(repair_dir / "repair_request.json")
    assert [item["unit_id"] for item in repair_request["units"]] == [
        "unit_0039"
    ]
    directive = repair_request["targeted_identity_repair"]
    assert directive["unit_id"] == "unit_0039"
    assert directive["variant_id"] == "natural"
    accepted_forms = directive["missing_identity_requirements"][0][
        "accepted_forms"
    ]
    assert IDENTITY in accepted_forms
    assert all(
        any(form in item["spoken_text"] for form in accepted_forms)
        for item in result["units"][0]["translation_variants"]
    )
    assert (session_root / "chunk_0001" / "completed.json").is_file()
    assert (repair_dir / "completed.json").is_file()
    assert read_json(repair_dir / "completed.json")[
        "provider_request_started"
    ] is False
    metadata = read_json(
        session_root / "chunk_0001" / "provider_metadata.json"
    )
    assert metadata["targeted_identity_repair_count"] == 1
    assert metadata["local_identity_alias_repair_count"] == 1


def test_exhausted_targeted_repair_sets_truthful_retryable_state(
    tmp_path: Path,
) -> None:
    pipeline, job = _job(tmp_path)

    with pytest.raises(RuntimeError, match="unrelated provider failure"):
        translate_full_transcript_once(
            pipeline,
            job,
            NonIdentityFailingProvider(),
            batch_size=18,
        )

    stored = pipeline.jobs.load(job.job_id)
    assert stored.state == "failed_retryable"
    assert stored.last_error["retryable"] is True
    assert "same translate command" in stored.last_error["action"]


def test_existing_failed_chunk_resumes_with_only_the_saved_unit_repair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline, job = _job(tmp_path)
    real_local_repair = (
        one_call_translation_module.restore_missing_hard_identities_in_unit
    )

    def disabled_local_repair(*_args, **_kwargs):
        raise MultivariantTranslationError("local repair disabled for fixture")

    monkeypatch.setattr(
        one_call_translation_module,
        "restore_missing_hard_identities_in_unit",
        disabled_local_repair,
    )
    with pytest.raises(ClaudeInvalidStructuredOutput):
        translate_full_transcript_once(
            pipeline,
            job,
            FailingProvider(),
            batch_size=18,
        )

    failed = pipeline.jobs.load(job.job_id)
    chunk_dir = Path(failed.media["translation_batch_directory"]) / "chunk_0001"
    failure = read_json(chunk_dir / "failure.json")
    assert failure["error"]["details"]["candidate_output"]["units"][0][
        "unit_id"
    ] == "unit_0039"
    failure["error"].pop("details")
    atomic_write_json(chunk_dir / "failure.json", failure)
    monkeypatch.setattr(
        one_call_translation_module,
        "restore_missing_hard_identities_in_unit",
        real_local_repair,
    )

    repair_provider = RepairOnlyProvider()
    result = translate_full_transcript_once(
        pipeline,
        failed,
        repair_provider,
        batch_size=18,
    )

    assert len(repair_provider.calls) == 0
    stored = pipeline.jobs.load(job.job_id)
    assert stored.state == "azure_tts_queued"
    assert (chunk_dir / "completed.json").is_file()
    latest_repair = sorted(
        (chunk_dir / "targeted_identity_repairs").glob(
            "attempt_*/completed.json"
        )
    )[-1]
    completed = read_json(latest_repair)
    assert completed["provider_request_started"] is False
    accepted_forms = read_json(
        latest_repair.parent / "repair_request.json"
    )["targeted_identity_repair"]["missing_identity_requirements"][0][
        "accepted_forms"
    ]
    assert all(
        any(form in item["spoken_text"] for form in accepted_forms)
        for item in result["units"][0]["translation_variants"]
    )
