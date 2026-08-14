from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mathula_tv.atomic_io import atomic_write_json, read_json
from mathula_tv.config import Settings
from mathula_tv.media import checksum
from mathula_tv.multivariant_translation import PROMPT_VERSION
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
        translation_batch_size=2,
        translation_context_units=1,
        translation_request_timeout_seconds=17,
        translation_batch_max_retries=2,
    )


def _unit_response(source: dict) -> dict:
    targets = source["duration_targets"]
    suffix = source["unit_id"]
    variants = []
    for variant_id, text, duration in (
        ("natural", f"I-EFF iyakhuluma namhlanje {suffix}.", 1800),
        ("concise", f"I-EFF iyakhuluma {suffix}.", 1400),
        ("compact", f"I-EFF iyasho {suffix}.", 1000),
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


def _response(request: dict) -> dict:
    return {
        "schema_version": "mathula.translation.multivariant.v1",
        "source_language": "en-ZA",
        "target_language": "isiZulu",
        "target_locale": "zu-ZA",
        "translation_strategy": "contextual_code_switching",
        "units": [_unit_response(source) for source in request["units"]],
        "global_terminology": [],
        "warnings": [],
    }


class Provider:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.options: list[dict] = []

    def with_request_options(self, **kwargs):
        self.options.append(kwargs)
        return self

    def translate_multivariant(self, request: dict):
        self.calls.append(request)
        callback = self.options[-1].get("event_callback")
        result = _response(request)
        if callable(callback):
            callback(
                {
                    "event": "request_attempt",
                    "attempt": 1,
                    "max_retries": self.options[-1]["max_retries"],
                    "timeout_seconds": self.options[-1]["timeout_seconds"],
                    "streaming": True,
                }
            )
            callback(
                {
                    "event": "http_response",
                    "attempt": 1,
                    "status_code": 200,
                    "streaming": True,
                }
            )
            callback(
                {
                    "event": "stream_opened",
                    "attempt": 1,
                    "status_code": 200,
                    "read_idle_timeout_seconds": self.options[-1]["timeout_seconds"],
                }
            )
            streamed_text = json.dumps(result, ensure_ascii=False)
            callback(
                {
                    "event": "stream_event",
                    "attempt": 1,
                    "event_type": "content_block_delta",
                    "data": {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": streamed_text},
                    },
                }
            )
            callback(
                {
                    "event": "stream_progress",
                    "attempt": 1,
                    "event_count": 5,
                    "text_characters": len(streamed_text),
                    "output_tokens": 123,
                    "last_event_type": "message_stop",
                }
            )
            callback(
                {
                    "event": "stream_completed",
                    "attempt": 1,
                    "event_count": 5,
                    "text_characters": len(streamed_text),
                    "output_tokens": 123,
                }
            )
        metadata = SimpleNamespace(
            to_dict=lambda: {
                "provider": "azure-foundry-claude",
                "model_returned": "claude-test",
                "prompt_version": PROMPT_VERSION,
            }
        )
        return SimpleNamespace(data=result, metadata=metadata)


def _job(tmp_path: Path, unit_count: int = 5):
    pipeline = ProductionDubbingPipeline(_settings(tmp_path))
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    job = pipeline.jobs.create(
        source_filename="source.mp4",
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
        start = index * 2
        unit_id = f"unit_{index + 1:04d}"
        text = f"The EFF is speaking in unit {index + 1}."
        segments.append(
            {
                "segment_id": unit_id,
                "speaker": "SPEAKER_00",
                "start": float(start),
                "end": float(start + 2),
                "source_text": text,
                "protected_spans": ["EFF"],
            }
        )
        units.append(
            {
                "unit_id": unit_id,
                "speaker_id": "SPEAKER_00",
                "start_ms": start * 1000,
                "end_ms": (start + 2) * 1000,
                "source_text": text,
                "protected_spans": ["EFF"],
            }
        )
    transcript = {
        "schema_version": "transcript-v1",
        "language": "en-ZA",
        "autocorrect": {"rendered": True},
        "segments": segments,
    }
    transcript_path = root / "analysis" / "transcript_en.json"
    atomic_write_json(transcript_path, transcript)
    atomic_write_json(
        root / "analysis" / "autocorrection_state.json",
        {
            "ready_for_language_ai": True,
            "review_count": 0,
            "blocking_review_count": 0,
            "authoritative_transcript_sha256": checksum(transcript_path),
        },
    )
    atomic_write_json(
        root / "dubbing" / "dubbing_units.json",
        {"schema_version": "dubbing-units-v1", "units": units},
    )
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
    return pipeline, job


def test_full_transcript_is_sent_in_exactly_one_provider_request(tmp_path: Path) -> None:
    pipeline, job = _job(tmp_path, unit_count=5)
    provider = Provider()

    result = pipeline.translate(job, provider, request_timeout_seconds=17)

    assert len(provider.calls) == 1
    assert len(provider.calls[0]["units"]) == 5
    assert "translation_batch" not in provider.calls[0]
    assert provider.options[-1]["max_retries"] == 1
    assert provider.options[-1]["timeout_seconds"] == 17
    assert len(result["units"]) == 5

    stored = pipeline.jobs.load(job.job_id)
    assert stored.state == "azure_tts_queued"
    assert "translation_batch_directory" not in stored.media
    run_dir = Path(stored.media["translation_run_directory"])
    persisted_request = read_json(run_dir / "translation_request.json")
    status = read_json(run_dir / "status.json")
    assert len(persisted_request["units"]) == 5
    assert status["status"] == "completed"
    assert status["provider_http_requests_started"] == 1
    partial = run_dir / "translation_response.partial.txt"
    events = run_dir / "provider_stream_events.jsonl"
    progress = read_json(run_dir / "provider_stream_progress.json")
    assert partial.is_file() and partial.stat().st_size > 0
    assert events.is_file() and events.stat().st_size > 0
    assert progress["output_tokens"] == 123
    assert read_json(run_dir / "install_manifest.json")["streaming"] is True


def test_old_batch_options_are_rejected_instead_of_splitting(tmp_path: Path) -> None:
    pipeline, job = _job(tmp_path, unit_count=5)
    provider = Provider()

    with pytest.raises(ValueError, match="Batched Claude translation has been removed"):
        pipeline.translate(job, provider, batch_size=2)

    assert provider.calls == []
