#!/usr/bin/env python3
"""Resume a legacy truncated translation with Azure OpenAI GPT.

The command salvages every complete unit from the saved SSE text, sends one
explicit continuation request for only the missing units while retaining the
complete English transcript as context, merges both responses, validates the
full source-bound result, and installs the normal Mathula TV artifacts.

It never retries the continuation request and never starts a model repair call.
A rerun reuses a saved continuation response instead of charging for another
provider request.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from mathula_tv.ai_provider import create_production_ai_provider
from mathula_tv.atomic_io import atomic_write_json, read_json
from mathula_tv.config import load_settings
from mathula_tv.models import PRODUCTION_STATES, utcnow
from mathula_tv.multivariant_translation import (
    PROMPT_VERSION,
    merge_multivariant_batch_responses,
    normalize_multivariant_response,
    prepare_installed_translation,
    validate_multivariant_response,
    write_translation_artifacts,
)
from mathula_tv.orchestrator import Orchestrator
from mathula_tv.production_pipeline import ProductionDubbingPipeline


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _latest_recoverable_run(job_root: Path) -> Path:
    cloud_runs = job_root / "translation" / "cloud_runs"
    if not cloud_runs.is_dir():
        raise FileNotFoundError(f"No cloud run directory exists at {cloud_runs}")
    for run_dir in sorted(
        (item for item in cloud_runs.iterdir() if item.is_dir()),
        reverse=True,
    ):
        if (
            (run_dir / "translation_response.partial.txt").is_file()
            and (run_dir / "translation_request.json").is_file()
            and (run_dir / "provider_stream_events.jsonl").is_file()
        ):
            return run_dir
    raise FileNotFoundError("No streamed Foundry translation run was found")


def _array_start(text: str, key: str) -> int:
    marker = f'"{key}"'
    key_pos = text.find(marker)
    if key_pos < 0:
        raise ValueError(f"Legacy output does not contain {marker}")
    colon_pos = text.find(":", key_pos + len(marker))
    if colon_pos < 0:
        raise ValueError(f"Legacy output has no value for {marker}")
    array_pos = text.find("[", colon_pos + 1)
    if array_pos < 0:
        raise ValueError(f"Legacy output value for {marker} is not an array")
    return array_pos


def _complete_array_value(text: str, key: str) -> list[Any]:
    start = _array_start(text, key)
    value, _ = json.JSONDecoder().raw_decode(text, start)
    if not isinstance(value, list):
        raise ValueError(f"Legacy output {key} value is not an array")
    return value


def extract_complete_translation_units(text: str) -> list[dict[str, Any]]:
    """Extract only fully closed unit objects from a truncated JSON array."""

    position = _array_start(text, "units") + 1
    decoder = json.JSONDecoder()
    units: list[dict[str, Any]] = []

    while position < len(text):
        while position < len(text) and text[position] in " \t\r\n,":
            position += 1
        if position >= len(text) or text[position] == "]":
            break
        try:
            value, end = decoder.raw_decode(text, position)
        except json.JSONDecodeError:
            # The first incomplete object is where max_tokens cut the stream.
            break
        if not isinstance(value, Mapping):
            break
        unit_id = str(value.get("unit_id") or "").strip()
        variants = value.get("variants")
        if not unit_id or not isinstance(variants, list):
            break
        units.append(dict(value))
        position = end

    return units


def _stream_stop_details(events_path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for line in events_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        data = record.get("data")
        if not isinstance(data, Mapping):
            continue
        if data.get("type") != "message_delta":
            continue
        delta = data.get("delta")
        usage = data.get("usage")
        if isinstance(delta, Mapping):
            result["stop_reason"] = delta.get("stop_reason")
        if isinstance(usage, Mapping):
            result["usage"] = dict(usage)
    return result


def _subset_request(
    original: Mapping[str, Any],
    units: list[Mapping[str, Any]],
) -> dict[str, Any]:
    request = {key: value for key, value in original.items() if key != "request_sha256"}
    request["units"] = [dict(item) for item in units]
    request["request_sha256"] = _sha256_json(request)
    return request


def _continuation_request(
    original: Mapping[str, Any],
    *,
    completed_units: list[Mapping[str, Any]],
    missing_units: list[Mapping[str, Any]],
) -> dict[str, Any]:
    request = _subset_request(original, missing_units)
    global_context = dict(request.get("global_context") or {})
    global_context["max_tokens_recovery"] = {
        "schema_version": "mathula.translation.max-tokens-context.v1",
        "instruction": (
            "The previous response reached max_tokens. Use the complete English "
            "timeline below for context, but return translations only for the units "
            "in the top-level units array. Do not repeat completed units."
        ),
        "completed_unit_ids": [str(item.get("unit_id") or "") for item in completed_units],
        "requested_unit_ids": [str(item.get("unit_id") or "") for item in missing_units],
        "full_english_timeline": [
            {
                key: item.get(key)
                for key in (
                    "unit_id",
                    "speaker_id",
                    "start_ms",
                    "end_ms",
                    "source_text",
                    "protected_spans",
                )
            }
            for item in original.get("units") or []
            if isinstance(item, Mapping)
        ],
    }
    request["global_context"] = global_context
    request["request_sha256"] = _sha256_json(
        {key: value for key, value in request.items() if key != "request_sha256"}
    )
    return request


def _metadata_dict(response: Any) -> dict[str, Any]:
    metadata = getattr(response, "metadata", None)
    to_dict = getattr(metadata, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
        if isinstance(value, Mapping):
            return dict(value)
    return {
        "provider": "azure-openai-gpt",
        "model_returned": "",
        "prompt_version": PROMPT_VERSION,
    }


def _install(
    *,
    app: Orchestrator,
    pipeline: ProductionDubbingPipeline,
    job: Any,
    run_dir: Path,
    request: Mapping[str, Any],
    response: Mapping[str, Any],
    provider_metadata: Mapping[str, Any],
    provider_requests_started: int,
) -> dict[str, Any]:
    paths = pipeline.paths(job.job_id)
    installed, validation = prepare_installed_translation(
        response=response,
        request=request,
        mode="cloud_translation_foundry_max_tokens_resume",
        provider_metadata={
            **dict(provider_metadata),
            "translation_mode": "full_transcript_max_tokens_resume",
            "automatic_model_repairs": 0,
            "provider_http_requests_started_during_recovery": provider_requests_started,
        },
    )
    written = write_translation_artifacts(
        job_root=paths.job_root,
        target_locale=job.target_language,
        installed=installed,
        pronunciation_dictionary=pipeline._pronunciation_dictionary(job),
        backup_dir=run_dir / "max_tokens_recovery_backups",
    )

    atomic_write_json(run_dir / "translation_response.json", response)
    atomic_write_json(run_dir / "provider_metadata.json", dict(provider_metadata))
    atomic_write_json(run_dir / "validation.json", validation)
    atomic_write_json(run_dir / "installed_translation.json", installed)

    job.providers.update(
        translation="azure-openai-gpt",
        translation_model=str(provider_metadata.get("model_returned") or ""),
        translation_prompt_version=PROMPT_VERSION,
    )
    job.media["translation_sha256"] = written["dubbing_sha256"]
    job.media["translation_multivariant_sha256"] = written["translation_sha256"]
    job.media["translation_request_sha256"] = str(request.get("request_sha256") or "")
    job.media["translation_run_directory"] = str(run_dir)
    job.media.pop("translation_batch_directory", None)
    job.human_review_flags = list(
        dict.fromkeys(
            [
                *job.human_review_flags,
                "multivariant_translation_requires_human_review",
                "foundry_max_tokens_recovery_used",
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
    return written


def resume(
    job_id: str,
    *,
    run_dir: Path | None = None,
    request_timeout_seconds: float | None = None,
) -> dict[str, Any]:
    settings = load_settings()
    app = Orchestrator(settings)
    pipeline = ProductionDubbingPipeline(settings)
    job = app.jobs.load(job_id)
    paths = pipeline.paths(job_id)
    selected_run = (
        run_dir.resolve()
        if run_dir is not None
        else _latest_recoverable_run(paths.job_root)
    )

    completed_record_path = selected_run / "max_tokens_recovery.json"
    if completed_record_path.is_file():
        existing = read_json(completed_record_path)
        if existing.get("status") == "completed":
            return existing

    request_path = selected_run / "translation_request.json"
    partial_path = selected_run / "translation_response.partial.txt"
    events_path = selected_run / "provider_stream_events.jsonl"
    original_request = read_json(request_path)
    raw_text = partial_path.read_text(encoding="utf-8")
    stop_details = _stream_stop_details(events_path)
    if stop_details.get("stop_reason") != "max_tokens":
        raise ValueError(
            "This command is only for a stream whose message_delta stop_reason is max_tokens"
        )

    raw_completed_units = extract_complete_translation_units(raw_text)
    expected_units = [
        dict(item)
        for item in original_request.get("units") or []
        if isinstance(item, Mapping)
    ]
    expected_ids = [str(item.get("unit_id") or "") for item in expected_units]
    completed_ids = [str(item.get("unit_id") or "") for item in raw_completed_units]
    if completed_ids != expected_ids[: len(completed_ids)]:
        raise ValueError(
            "Saved legacy units are not a contiguous prefix of the authoritative request"
        )
    if not completed_ids:
        raise ValueError("No complete translation units could be salvaged")
    if len(completed_ids) >= len(expected_ids):
        raise ValueError("The saved response already contains every requested unit")

    completed_source_units = expected_units[: len(completed_ids)]
    missing_source_units = expected_units[len(completed_ids) :]
    partial_request = _subset_request(original_request, completed_source_units)
    raw_terminology = _complete_array_value(raw_text, "global_terminology")
    partial_payload = {
        "schema_version": "mathula.translation.multivariant.v1",
        "units": raw_completed_units,
        "global_terminology": raw_terminology,
        "warnings": [],
    }
    partial_response = normalize_multivariant_response(
        partial_payload,
        request=partial_request,
    )
    validate_multivariant_response(partial_response, request=partial_request)

    continuation_request = _continuation_request(
        original_request,
        completed_units=completed_source_units,
        missing_units=missing_source_units,
    )
    atomic_write_json(selected_run / "salvaged_response.json", partial_response)
    atomic_write_json(selected_run / "continuation_request.json", continuation_request)
    atomic_write_json(
        selected_run / "max_tokens_recovery_plan.json",
        {
            "schema_version": "mathula.translation.max-tokens-recovery-plan.v1",
            "created_at": _utcnow(),
            "job_id": job_id,
            "original_unit_count": len(expected_ids),
            "salvaged_unit_count": len(completed_ids),
            "salvaged_unit_ids": completed_ids,
            "missing_unit_count": len(missing_source_units),
            "missing_unit_ids": [str(item.get("unit_id") or "") for item in missing_source_units],
            "stop_details": stop_details,
            "continuation_provider_request_limit": 1,
            "automatic_retries": 0,
            "automatic_model_repairs": 0,
            "full_english_transcript_in_continuation_context": True,
        },
    )

    continuation_response_path = selected_run / "continuation_response.json"
    provider_requests_started = 0
    if continuation_response_path.is_file():
        continuation_response = read_json(continuation_response_path)
        validate_multivariant_response(
            continuation_response,
            request=continuation_request,
        )
        provider_metadata = read_json(selected_run / "continuation_provider_metadata.json")
    else:
        provider = create_production_ai_provider()
        timeout = float(
            request_timeout_seconds
            if request_timeout_seconds is not None
            else getattr(
                settings,
                "translation_request_timeout_seconds",
                getattr(settings, "gpt_timeout_seconds", 900),
            )
        )
        if timeout <= 0 or not math.isfinite(timeout):
            raise ValueError("Request timeout must be a positive finite number")

        continuation_events = selected_run / "continuation_stream_events.jsonl"
        continuation_partial = selected_run / "continuation_response.partial.txt"
        continuation_events.write_text("", encoding="utf-8")
        continuation_partial.write_text("", encoding="utf-8")
        started_at = time.monotonic()
        last_printed = 0

        def event_callback(event: dict[str, Any]) -> None:
            nonlocal provider_requests_started, last_printed
            kind = str(event.get("event") or "provider")
            elapsed = time.monotonic() - started_at
            if kind == "request_attempt":
                provider_requests_started = 1
                print(
                    f"[resume {elapsed:05.0f}s] sending {len(missing_source_units)} missing units "
                    "in one Azure OpenAI GPT request",
                    flush=True,
                )
            elif kind == "http_response":
                print(
                    f"[resume {elapsed:05.0f}s] HTTP {event.get('status_code')}; awaiting response",
                    flush=True,
                )
            elif kind == "stream_event":
                record = {
                    "received_at": _utcnow(),
                    "event_type": event.get("event_type"),
                    "data": event.get("data"),
                }
                with continuation_events.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
                    handle.write("\n")
                data = event.get("data")
                if isinstance(data, Mapping) and data.get("type") == "content_block_delta":
                    delta = data.get("delta")
                    if isinstance(delta, Mapping) and delta.get("type") == "text_delta":
                        with continuation_partial.open("a", encoding="utf-8") as handle:
                            handle.write(str(delta.get("text") or ""))
            elif kind == "stream_progress":
                characters = int(event.get("text_characters") or 0)
                if characters >= last_printed + 4096 or characters == 0:
                    last_printed = characters
                    print(
                        f"[resume {elapsed:05.0f}s] {characters} response characters",
                        flush=True,
                    )
            elif kind == "stream_truncated":
                print(
                    f"[resume {elapsed:05.0f}s] ERROR: continuation also reached max_tokens",
                    flush=True,
                )

        provider = provider.with_request_options(
            timeout_seconds=timeout,
            max_retries=1,
            event_callback=event_callback,
        )
        response = provider.translate_multivariant(continuation_request)
        continuation_response = response.data
        provider_metadata = _metadata_dict(response)
        provider_metadata.update(
            {
                "provider": "azure-openai-gpt",
                "streaming": False,
                "transport": "responses_api",
                "thinking_type": getattr(provider.config, "translation_thinking_type", "disabled"),
                "effort": getattr(provider.config, "translation_effort", "low"),
                "max_output_tokens": getattr(provider.config, "translation_max_output_tokens", 100_000),
                "automatic_model_repairs": 0,
            }
        )
        atomic_write_json(continuation_response_path, continuation_response)
        atomic_write_json(
            selected_run / "continuation_provider_metadata.json",
            provider_metadata,
        )

    merged = merge_multivariant_batch_responses(
        request=original_request,
        responses=[partial_response, continuation_response],
    )
    validate_multivariant_response(merged, request=original_request)
    written = _install(
        app=app,
        pipeline=pipeline,
        job=job,
        run_dir=selected_run,
        request=original_request,
        response=merged,
        provider_metadata=provider_metadata,
        provider_requests_started=provider_requests_started,
    )

    record = {
        "schema_version": "mathula.translation.max-tokens-recovery.v1",
        "status": "completed",
        "completed_at": _utcnow(),
        "job_id": job_id,
        "run_directory": str(selected_run),
        "original_unit_count": len(expected_ids),
        "salvaged_unit_count": len(completed_ids),
        "continued_unit_count": len(missing_source_units),
        "provider_http_requests_started_during_recovery": provider_requests_started,
        "automatic_retries": 0,
        "automatic_model_repairs": 0,
        "full_english_transcript_in_continuation_context": True,
        **written,
    }
    atomic_write_json(completed_record_path, record)
    atomic_write_json(
        selected_run / "status.json",
        {
            "schema_version": "mathula.translation.one-call-status.v1",
            "status": "max_tokens_recovery_completed",
            "updated_at": _utcnow(),
            "salvaged_unit_count": len(completed_ids),
            "continued_unit_count": len(missing_source_units),
            "provider_http_requests_started_during_recovery": provider_requests_started,
            "translation_target": written["translation_target"],
            "dubbing_target": written["dubbing_target"],
        },
    )
    return record


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Salvage a max_tokens-truncated Foundry translation and make one explicit "
            "continuation call for only the missing units."
        )
    )
    parser.add_argument("job_id")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--request-timeout-seconds", type=float)
    args = parser.parse_args()
    result = resume(
        args.job_id,
        run_dir=args.run_dir,
        request_timeout_seconds=args.request_timeout_seconds,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
