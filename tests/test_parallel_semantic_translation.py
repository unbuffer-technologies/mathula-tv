from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

from mathula_tv.atomic_io import atomic_write_json, read_json
from mathula_tv.config import Settings
from mathula_tv.media import checksum
from mathula_tv.multivariant_translation import PROMPT_VERSION
from mathula_tv.production_pipeline import ProductionDubbingPipeline
from mathula_tv.one_call_translation import _parallel_terminology_conflicts


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
    units = []
    for source in request["units"]:
        variants = []
        for variant_id, prefix, duration in (
            ("natural", "I-EFF ikhuluma namhlanje", 1800),
            ("concise", "I-EFF iyakhuluma", 1400),
            ("compact", "I-EFF iyasho", 1000),
        ):
            variants.append(
                {
                    "variant_id": variant_id,
                    "compression_level": variant_id,
                    "target_duration_ms": source["duration_targets"][
                        f"{variant_id}_target_duration_ms"
                    ],
                    "spoken_text": f"{prefix} {source['unit_id']}.",
                    "estimated_duration_ms": duration,
                    "preserved_english_spans": ["EFF"],
                    "omitted_optional_details": [],
                    "meaning_preserved": True,
                }
            )
        units.append(
            {
                "unit_id": source["unit_id"],
                "speaker_id": source["speaker_id"],
                "start_ms": source["start_ms"],
                "end_ms": source["end_ms"],
                "available_duration_ms": source["available_duration_ms"],
                "overlap_group": source.get("overlap_group"),
                "source_text": source["source_text"],
                "protected_spans": ["EFF"],
                "required_facts": [source["source_text"]],
                "optional_details": [],
                "variants": variants,
                "selection_status": "unselected",
                "selected_variant_id": None,
                "warnings": [],
            }
        )
    return {
        "schema_version": "mathula.translation.multivariant.v1",
        "source_language": "en-ZA",
        "target_language": "isiZulu",
        "target_locale": "zu-ZA",
        "translation_strategy": "contextual_code_switching",
        "units": units,
        "global_terminology": [
            {
                "source_term": "EFF",
                "target_rendering": "i-EFF",
                "preserve_english": True,
                "reason": "acronym",
            }
        ],
        "warnings": [],
    }


class _ProviderView:
    def __init__(self, owner: "ParallelProvider", options: dict) -> None:
        self.owner = owner
        self.options = options

    def translate_multivariant(self, request: dict):
        guard = self.options.get("request_guard")
        if callable(guard):
            guard({"attempt": 1})
        with self.owner.lock:
            self.owner.active += 1
            self.owner.max_active = max(self.owner.max_active, self.owner.active)
            self.owner.calls.append(request)
        time.sleep(0.04)
        with self.owner.lock:
            self.owner.active -= 1
        metadata = SimpleNamespace(
            to_dict=lambda: {
                "provider": "azure-openai-gpt",
                "model_returned": "gpt-test",
                "prompt_version": PROMPT_VERSION,
            }
        )
        return SimpleNamespace(data=self.owner.response(request), metadata=metadata)


class ParallelProvider:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.calls: list[dict] = []

    def with_request_options(self, **kwargs):
        return _ProviderView(self, kwargs)

    def response(self, request: dict) -> dict:
        return _response(request)


class ConflictingParallelProvider(ParallelProvider):
    def response(self, request: dict) -> dict:
        response = _response(request)
        prior_ids = {
            item["unit_id"]
            for item in request["translation_batch"]["prior_translated_context"]
        }
        if (
            request["units"][0]["unit_id"] == "unit_0005"
            and "unit_0003" not in prior_ids
        ):
            response["global_terminology"][0]["target_rendering"] = "iNhlangano"
        return response


def _job(tmp_path: Path, unit_count: int = 5):
    pipeline = ProductionDubbingPipeline(_settings(tmp_path))
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    job = pipeline.jobs.create(
        source_filename=source.name,
        local_source_path=str(source),
        source_checksum=checksum(source),
        source_duration=float(unit_count * 2),
        target_language="zu-ZA",
    )
    job.state = "analysis_ready"
    pipeline.jobs.save(job)
    root = pipeline.jobs.job_dir(job.job_id)
    segments = []
    units = []
    for index in range(unit_count):
        unit_id = f"unit_{index + 1:04d}"
        start_ms = index * 2000
        text = f"The EFF is speaking today in unit {index + 1}."
        segments.append(
            {
                "segment_id": unit_id,
                "speaker": "SPEAKER_00",
                "start": start_ms / 1000,
                "end": (start_ms + 2000) / 1000,
                "source_text": text,
                "protected_spans": ["EFF"],
            }
        )
        units.append(
            {
                "unit_id": unit_id,
                "speaker_id": "SPEAKER_00",
                "start_ms": start_ms,
                "end_ms": start_ms + 2000,
                "source_text": text,
                "protected_spans": ["EFF"],
            }
        )
    transcript_path = root / "analysis" / "transcript_en.json"
    atomic_write_json(
        transcript_path,
        {
            "schema_version": "transcript-v1",
            "language": "en-ZA",
            "autocorrect": {"rendered": True},
            "segments": segments,
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
    atomic_write_json(units_path, {"schema_version": "dubbing-units-v1", "units": units})
    bindings_path = root / "analysis" / "entity_bindings.json"
    atomic_write_json(
        bindings_path,
        {
            "source_transcript_sha256": checksum(transcript_path),
            "source_dubbing_units_sha256": checksum(units_path),
            "registry_sha256": "test",
            "per_unit_bindings": {},
        },
    )
    pipeline.bind_entities = lambda _job, force=False: read_json(bindings_path)  # type: ignore[method-assign]
    pipeline.build_units = lambda _job, force=False: read_json(units_path)  # type: ignore[method-assign]
    return pipeline, job


def test_fresh_translation_uses_validated_seed_then_parallel_wave(tmp_path: Path):
    pipeline, job = _job(tmp_path)
    provider = ParallelProvider()

    result = pipeline.translate(
        job,
        provider,
        batch_size=2,
        parallelism=2,
        restart_batches=True,
    )

    assert len(result["units"]) == 5
    assert len(provider.calls) == 3
    assert provider.max_active == 2
    calls_by_first_unit = {
        call["units"][0]["unit_id"]: call for call in provider.calls
    }
    seed = calls_by_first_unit["unit_0001"]["translation_batch"]
    second = calls_by_first_unit["unit_0003"]["translation_batch"]
    third = calls_by_first_unit["unit_0005"]["translation_batch"]
    assert seed["prior_translated_context"] == []
    assert [item["unit_id"] for item in second["prior_translated_context"]] == [
        "unit_0001",
        "unit_0002",
    ]
    assert third["prior_translated_context"] == second["prior_translated_context"]
    stored = pipeline.jobs.load(job.job_id)
    run_dir = Path(stored.media["translation_run_directory"])
    metadata = read_json(run_dir / "provider_metadata.json")
    assert metadata["parallelism"] == 2
    assert metadata["parallel_strategy"] == "validated_seed_then_parallel_waves_v1"


def test_parallelism_one_keeps_full_serial_continuity(tmp_path: Path):
    pipeline, job = _job(tmp_path)
    provider = ParallelProvider()

    pipeline.translate(
        job,
        provider,
        batch_size=2,
        parallelism=1,
        restart_batches=True,
    )

    assert len(provider.calls) == 3
    assert provider.max_active == 1
    calls_by_first_unit = {
        call["units"][0]["unit_id"]: call for call in provider.calls
    }
    final_context = calls_by_first_unit["unit_0005"]["translation_batch"][
        "prior_translated_context"
    ]
    assert [item["unit_id"] for item in final_context] == [
        "unit_0002",
        "unit_0003",
        "unit_0004",
    ]
    stored = pipeline.jobs.load(job.job_id)
    run_dir = Path(stored.media["translation_run_directory"])
    metadata = read_json(run_dir / "provider_metadata.json")
    assert metadata["parallelism"] == 1
    assert metadata["parallel_strategy"] == "serial_validated_continuity"


def test_parallel_terminology_conflicts_fail_closed():
    conflicts = _parallel_terminology_conflicts(
        [
            {
                "global_terminology": [
                    {
                        "source_term": "Commission",
                        "target_rendering": "iKhomishini",
                        "preserve_english": False,
                    }
                ]
            },
            {
                "global_terminology": [
                    {
                        "source_term": "commission",
                        "target_rendering": "iKhomishana",
                        "preserve_english": False,
                    }
                ]
            },
        ]
    )

    assert len(conflicts) == 1
    assert conflicts[0]["first_chunk"] == 1
    assert conflicts[0]["conflicting_chunk"] == 2


def test_parallel_terminology_ignores_advisory_flag_when_rendering_is_identical():
    conflicts = _parallel_terminology_conflicts(
        [
            {
                "global_terminology": [
                    {
                        "source_term": "cell phone grabber",
                        "target_rendering": "cell phone grabber",
                        "preserve_english": False,
                    }
                ]
            },
            {
                "global_terminology": [
                    {
                        "source_term": "cell phone grabber",
                        "target_rendering": "cell phone grabber",
                        "preserve_english": True,
                    }
                ]
            },
        ]
    )

    assert conflicts == []


def test_parallel_conflict_automatically_repairs_only_the_unsafe_suffix(
    tmp_path: Path,
):
    pipeline, job = _job(tmp_path)
    provider = ConflictingParallelProvider()

    result = pipeline.translate(
        job,
        provider,
        batch_size=2,
        parallelism=2,
        restart_batches=True,
    )

    assert len(result["units"]) == 5
    assert len(provider.calls) == 4
    stored = pipeline.jobs.load(job.job_id)
    run_dir = Path(stored.media["translation_run_directory"])
    batch_dir = Path(stored.media["translation_batch_directory"])
    conflict = read_json(batch_dir / "parallel_terminology_conflicts.json")
    metadata = read_json(run_dir / "provider_metadata.json")
    assert conflict["status"] == "automatic_serial_repair_completed"
    assert conflict["first_serial_repair_chunk"] == 3
    assert metadata["automatic_serial_repair_from_chunk"] == 3
    assert (batch_dir / "chunk_0001" / "completed.json").is_file()
    assert (batch_dir / "chunk_0002" / "completed.json").is_file()
    assert (batch_dir / "chunk_0003" / "completed.json").is_file()
    assert (
        batch_dir / "chunk_0003" / "completed.parallel-conflict.json"
    ).is_file()
