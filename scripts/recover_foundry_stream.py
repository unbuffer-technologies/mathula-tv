#!/usr/bin/env python3
"""Recover a completed Azure Foundry Claude stream without another model call."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from mathula_tv.ai_provider import AnthropicConfig, _parse_multivariant_structured_json
from mathula_tv.atomic_io import atomic_write_json, read_json
from mathula_tv.config import load_settings
from mathula_tv.models import PRODUCTION_STATES, utcnow
from mathula_tv.multivariant_translation import (
    PROMPT_VERSION,
    normalize_multivariant_response,
    prepare_installed_translation,
    validate_multivariant_response,
    write_translation_artifacts,
)
from mathula_tv.orchestrator import Orchestrator
from mathula_tv.production_pipeline import ProductionDubbingPipeline


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_json_object(text: str) -> dict[str, Any]:
    """Use the provider parser so recovery and live translation select the same unit array."""

    return _parse_multivariant_structured_json(text)


def _latest_recoverable_run(job_root: Path) -> Path:
    cloud_runs = job_root / "translation" / "cloud_runs"
    if not cloud_runs.is_dir():
        raise FileNotFoundError(f"No cloud run directory exists at {cloud_runs}")
    for run_dir in sorted(
        (item for item in cloud_runs.iterdir() if item.is_dir()),
        reverse=True,
    ):
        partial = run_dir / "translation_response.partial.txt"
        request = run_dir / "translation_request.json"
        if partial.is_file() and partial.stat().st_size > 0 and request.is_file():
            return run_dir
    raise FileNotFoundError("No completed or partial Foundry stream response was found")


def recover(job_id: str, *, run_dir: Path | None = None) -> dict[str, Any]:
    settings = load_settings()
    app = Orchestrator(settings)
    pipeline = ProductionDubbingPipeline(settings)
    job = app.jobs.load(job_id)
    paths = pipeline.paths(job_id)
    selected_run = run_dir.resolve() if run_dir is not None else _latest_recoverable_run(paths.job_root)

    request_path = selected_run / "translation_request.json"
    partial_path = selected_run / "translation_response.partial.txt"
    request = read_json(request_path)
    raw_text = partial_path.read_text(encoding="utf-8")
    parsed = _parse_json_object(raw_text)
    normalized = normalize_multivariant_response(parsed, request=request)
    validate_multivariant_response(normalized, request=request)

    try:
        config = AnthropicConfig.from_environment()
        model = config.model
        endpoint = config.base_url
    except Exception:
        model = str(getattr(settings, "claude_model", "") or "")
        endpoint = ""

    provider_metadata = {
        "provider": "azure-foundry-claude",
        "model_requested": model,
        "model_returned": model,
        "prompt_version": PROMPT_VERSION,
        "streaming": True,
        "transport": "server_sent_events",
        "recovered_from_completed_stream": True,
        "automatic_model_repairs": 0,
        "provider_http_requests_started_during_recovery": 0,
        "foundry_endpoint": endpoint,
    }
    installed, validation = prepare_installed_translation(
        response=normalized,
        request=request,
        mode="cloud_translation_full_transcript_stream_recovery",
        provider_metadata=provider_metadata,
    )
    written = write_translation_artifacts(
        job_root=paths.job_root,
        target_locale=job.target_language,
        installed=installed,
        pronunciation_dictionary=pipeline._pronunciation_dictionary(job),
        backup_dir=selected_run / "recovery_backups",
    )

    atomic_write_json(selected_run / "translation_response.json", normalized)
    atomic_write_json(selected_run / "provider_metadata.json", provider_metadata)
    atomic_write_json(selected_run / "validation.json", validation)
    atomic_write_json(selected_run / "installed_translation.json", installed)
    recovery_record = {
        "schema_version": "mathula.translation.foundry-stream-recovery.v1",
        "recovered_at": _utcnow(),
        "job_id": job_id,
        "run_directory": str(selected_run),
        "source_partial_response": str(partial_path),
        "source_request": str(request_path),
        "request_sha256": request.get("request_sha256"),
        "unit_count": len(normalized.get("units") or []),
        "provider_http_requests_started_during_recovery": 0,
        "automatic_model_repairs": 0,
        **written,
    }
    atomic_write_json(selected_run / "recovery.json", recovery_record)
    atomic_write_json(
        selected_run / "status.json",
        {
            "schema_version": "mathula.translation.one-call-status.v1",
            "status": "recovered_completed",
            "updated_at": _utcnow(),
            "provider_http_requests_started_during_recovery": 0,
            "translation_target": written["translation_target"],
            "dubbing_target": written["dubbing_target"],
        },
    )

    job.providers.update(
        translation="azure-foundry-claude",
        translation_model=model,
        translation_prompt_version=PROMPT_VERSION,
    )
    job.media["translation_sha256"] = written["dubbing_sha256"]
    job.media["translation_multivariant_sha256"] = written["translation_sha256"]
    job.media["translation_request_sha256"] = str(request.get("request_sha256") or "")
    job.media["translation_run_directory"] = str(selected_run)
    job.media.pop("translation_batch_directory", None)
    job.human_review_flags = list(
        dict.fromkeys(
            [
                *job.human_review_flags,
                "multivariant_translation_requires_human_review",
            ]
        )
    )
    job.last_error = None
    if job.state == "analysis_ready":
        job.transition("translation_running")
    if job.state == "translation_running":
        job.transition("translation_ready")
        job.transition("azure_tts_queued")
    elif job.state == "translation_ready":
        job.transition("azure_tts_queued")
    elif job.state != "azure_tts_queued":
        if job.state in set(PRODUCTION_STATES) - {
            "created",
            "audio_prepared",
            "uploaded",
            "analysis_queued",
            "analysis_running",
        }:
            job.state = "azure_tts_queued"
            job.updated_at = utcnow()
        else:
            raise ValueError(
                f"Cannot install recovered translation while job is in state {job.state}"
            )
    if "translation" not in job.completed_stages:
        job.completed_stages.append("translation")
    app.jobs.save(job)
    return recovery_record


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and install a completed Azure Foundry SSE response from disk. "
            "This command performs zero provider requests."
        )
    )
    parser.add_argument("job_id")
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    result = recover(args.job_id, run_dir=args.run_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
