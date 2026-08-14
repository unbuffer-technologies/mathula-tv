from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from mathula_tv.atomic_io import atomic_write_json, read_json
from mathula_tv.config import Settings
from mathula_tv.media import checksum
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


def _response(request: dict) -> dict:
    source = request["units"][0]
    targets = source["duration_targets"]
    variants = []
    for variant_id, text, duration in (
        ("natural", "I-EFF ikhuluma namhlanje.", 1800),
        ("concise", "I-EFF iyakhuluma.", 1400),
        ("compact", "I-EFF iyasho.", 1000),
    ):
        variants.append(
            {
                "variant_id": variant_id,
                "compression_level": variant_id,
                "target_duration_ms": targets[f"{variant_id}_target_duration_ms"],
                "spoken_text": text,
                "estimated_duration_ms": duration,
                "preserved_english_spans": ["EFF"],
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
                "protected_spans": ["EFF"],
                "required_facts": ["The EFF is speaking today."],
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


class Provider:
    def __init__(self) -> None:
        self.calls = 0

    def translate_multivariant(self, request: dict):
        self.calls += 1
        metadata = SimpleNamespace(
            model_returned="claude-test",
            prompt_version="mathula-multivariant-translation-prompt-v2-batched",
            to_dict=lambda: {
                "provider": "anthropic",
                "model_returned": "claude-test",
                "prompt_version": "mathula-multivariant-translation-prompt-v2-batched",
            },
        )
        return SimpleNamespace(data=_response(request), metadata=metadata)


def _job(tmp_path: Path):
    pipeline = ProductionDubbingPipeline(_settings(tmp_path))
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    job = pipeline.jobs.create(
        source_filename="source.mp4",
        local_source_path=str(source),
        source_checksum=checksum(source),
        source_duration=2.0,
        target_language="zu-ZA",
    )
    job.state = "analysis_ready"
    pipeline.jobs.save(job)
    root = pipeline.jobs.job_dir(job.job_id)
    transcript = {
        "schema_version": "transcript-v1",
        "language": "en-ZA",
        "autocorrect": {"rendered": True},
        "segments": [
            {
                "segment_id": "unit_0001",
                "speaker": "SPEAKER_00",
                "start": 0.0,
                "end": 2.0,
                "source_text": "The EFF is speaking today.",
                "protected_spans": ["EFF"],
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
                "unit_id": "unit_0001",
                "speaker_id": "SPEAKER_00",
                "start_ms": 0,
                "end_ms": 2000,
                "source_text": "The EFF is speaking today.",
                "protected_spans": ["EFF"],
            }
        ],
    }
    atomic_write_json(root / "dubbing" / "dubbing_units.json", units)
    atomic_write_json(
        root / "analysis" / "entity_bindings.json",
        {
            "source_transcript_sha256": checksum(transcript_path),
            "source_dubbing_units_sha256": checksum(root / "dubbing" / "dubbing_units.json"),
            "registry_sha256": "test",
            "per_unit_bindings": {},
        },
    )
    pipeline.bind_entities = lambda _job, force=False: read_json(root / "analysis" / "entity_bindings.json")  # type: ignore[method-assign]
    pipeline.build_units = lambda _job, force=False: read_json(root / "dubbing" / "dubbing_units.json")  # type: ignore[method-assign]
    return pipeline, job


def test_cloud_translation_writes_multivariant_and_compatibility_artifacts(tmp_path: Path) -> None:
    pipeline, job = _job(tmp_path)
    provider = Provider()
    result = pipeline.translate(job, provider)
    paths = pipeline.paths(job.job_id)

    assert provider.calls == 1
    assert paths.raw_multivariant_translation.is_file()
    assert paths.three_text_translation.is_file()
    assert result["schema_version"] == "three-text-translation-v2-multivariant"
    assert len(result["units"][0]["translation_variants"]) == 3
    stored = pipeline.jobs.load(job.job_id)
    assert stored.state == "azure_tts_queued"
    assert stored.providers["translation_prompt_version"] == "mathula-multivariant-translation-prompt-v2-batched"


def test_cloud_translation_reuses_current_artifact_without_provider_call(tmp_path: Path) -> None:
    pipeline, job = _job(tmp_path)
    provider = Provider()
    pipeline.translate(job, provider)
    stored = pipeline.jobs.load(job.job_id)
    result = pipeline.translate(stored, provider)
    assert provider.calls == 1
    assert result["schema_version"] == "three-text-translation-v2-multivariant"
