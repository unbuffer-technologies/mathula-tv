"""Single-request full-transcript translation through Azure AI Foundry.

This module deliberately shares the exact source-bound request built for
``export-translation-package``.  One invocation permits one provider HTTP
request and zero model repair requests.  Failures are persisted and returned to
the operator instead of starting another paid Claude call. Production use is
restricted to the existing Azure Foundry Claude deployment.
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from .atomic_io import atomic_write_json, read_json
from .autocorrect_stage import require_authoritative_transcript
from .media import checksum
from .multivariant_translation import (
    PROMPT_VERSION,
    build_job_translation_request,
    prepare_installed_translation,
    write_translation_artifacts,
)

ProgressCallback = Callable[[dict[str, Any]], None]


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_emit(callback: ProgressCallback | None, event: Mapping[str, Any]) -> None:
    if callback is None:
        return
    try:
        callback(dict(event))
    except Exception:
        # Reporting must never affect translation correctness.
        return


def _provider_metadata(response: Any) -> dict[str, Any]:
    metadata = getattr(response, "metadata", None)
    to_dict = getattr(metadata, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
        if isinstance(value, Mapping):
            return dict(value)
    return {
        "provider": "azure-foundry-claude",
        "model_returned": str(getattr(metadata, "model_returned", "") or ""),
        "prompt_version": str(getattr(metadata, "prompt_version", "") or PROMPT_VERSION),
    }


def translate_full_transcript_once(
    pipeline: Any,
    job: Any,
    provider: Any,
    *,
    force: bool = False,
    batch_size: int | None = None,
    context_units: int | None = None,
    request_timeout_seconds: float | None = None,
    batch_max_retries: int | None = None,
    max_provider_calls: int | None = None,
    restart_batches: bool = False,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Translate the complete authoritative English timeline in one request.

    ``batch_*`` arguments remain in the public method signature only so older
    callers fail with a clear migration error instead of a Python ``TypeError``.
    They can no longer change the translation architecture.
    """

    provider_config = getattr(provider, "config", None)
    configured_provider = str(
        getattr(provider_config, "provider", None)
        or getattr(provider, "provider", None)
        or ""
    ).strip().lower()
    if configured_provider and configured_provider != "azure-foundry-claude":
        raise ValueError(
            "Mathula TV Claude translation must use Azure AI Foundry "
            f"(azure-foundry-claude), not {configured_provider!r}"
        )
    foundry_endpoint = str(getattr(provider_config, "base_url", "") or "")
    foundry_deployment = str(getattr(provider_config, "model", "") or "")
    translation_thinking_type = str(
        getattr(provider_config, "translation_thinking_type", "disabled") or "disabled"
    )
    translation_effort = str(
        getattr(provider_config, "translation_effort", "low") or "low"
    )
    translation_max_output_tokens = int(
        getattr(provider_config, "translation_max_output_tokens", 100_000) or 100_000
    )

    supplied_batch_options = {
        "batch_size": batch_size,
        "context_units": context_units,
        "batch_max_retries": batch_max_retries,
        "restart_batches": restart_batches or None,
    }
    enabled_batch_options = {
        key: value for key, value in supplied_batch_options.items() if value is not None
    }
    if enabled_batch_options:
        raise ValueError(
            "Batched Claude translation has been removed. Run translate without "
            f"batch options; received {enabled_batch_options}."
        )
    if max_provider_calls is not None and int(max_provider_calls) < 1:
        raise ValueError("Translation provider-call budget must allow one request")

    if job.state == "synthesis_queued":
        raise ValueError(
            "Migrate the validated legacy translation; do not regenerate it silently"
        )

    started_at = time.monotonic()

    def emit(
        stage: str,
        message: str,
        *,
        status: str = "running",
        **extra: Any,
    ) -> None:
        _safe_emit(
            progress,
            {
                "stage": stage,
                "message": message,
                "status": status,
                "elapsed_seconds": time.monotonic() - started_at,
                **extra,
            },
        )

    emit("prepare", "Loading the complete authoritative English transcript for Azure Foundry")
    emit(
        "provider",
        (
            "Claude translation mode: "
            f"thinking={translation_thinking_type}, "
            f"effort={translation_effort}, "
            f"max_output_tokens={translation_max_output_tokens}"
        ),
        thinking_type=translation_thinking_type,
        effort=translation_effort,
        max_output_tokens=translation_max_output_tokens,
    )
    paths = pipeline.paths(job.job_id)
    authoritative = require_authoritative_transcript(
        work_dir=pipeline.settings.work_dir,
        job_id=job.job_id,
    )

    # Refresh both deterministic boundaries before building the request.  The
    # same builder is used by export-translation-package, so manual and cloud
    # translation receive the same transcript, units, hashes, context and schema.
    pipeline.bind_entities(job, force=force)
    units = pipeline.build_units(job, force=False)
    units_sha256 = checksum(paths.dubbing_units)

    if (
        paths.raw_multivariant_translation.is_file()
        and paths.three_text_translation.is_file()
        and not force
    ):
        existing = read_json(paths.three_text_translation)
        if (
            existing.get("source_transcript_sha256") == authoritative["sha256"]
            and existing.get("source_dubbing_units_sha256") == units_sha256
            and bool(units.get("units"))
        ):
            emit("complete", "Reusing the current source-bound translation", status="completed")
            return existing

    if job.state not in {"analysis_ready", "translation_running"}:
        raise ValueError(
            f"Claude translation requires analysis_ready or translation_running, not {job.state}; "
            "use import-translation for an intentional reviewed override"
        )

    request, _ = build_job_translation_request(
        job_root=paths.job_root,
        job_id=job.job_id,
        target_locale=job.target_language,
    )
    if request.get("source_transcript_file_sha256") != authoritative["sha256"]:
        raise ValueError(
            "Translation request was not built from the authoritative corrected transcript"
        )
    if request.get("source_timeline_file_sha256") != units_sha256:
        raise ValueError(
            "Translation request was not built from the current deterministic dubbing units"
        )
    if "translation_batch" in request:
        raise ValueError("Full-transcript request unexpectedly contains translation_batch")

    timeout_seconds = float(
        request_timeout_seconds
        if request_timeout_seconds is not None
        else getattr(
            pipeline.settings,
            "translation_request_timeout_seconds",
            pipeline.settings.claude_timeout_seconds,
        )
    )
    if timeout_seconds <= 0 or not math.isfinite(timeout_seconds):
        raise ValueError("Translation request timeout must be a positive finite number")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir = paths.job_root / "translation" / "cloud_runs" / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    atomic_write_json(run_dir / "translation_request.json", request)
    atomic_write_json(
        run_dir / "request_manifest.json",
        {
            "schema_version": "mathula.translation.one-call-request.v1",
            "created_at": _utcnow(),
            "job_id": job.job_id,
            "target_locale": job.target_language,
            "prompt_version": PROMPT_VERSION,
            "translation_mode": "full_transcript_one_call",
            "transport": "server_sent_events",
            "streaming": True,
            "timeout_semantics": "socket_read_idle_timeout_not_total_generation_deadline",
            "provider": "azure-foundry-claude",
            "foundry_endpoint": foundry_endpoint,
            "foundry_deployment": foundry_deployment,
            "unit_count": len(request.get("units") or []),
            "request_sha256": request["request_sha256"],
            "source_transcript_file_sha256": request.get(
                "source_transcript_file_sha256"
            ),
            "source_timeline_file_sha256": request.get(
                "source_timeline_file_sha256"
            ),
            "request_timeout_seconds": timeout_seconds,
            "provider_http_request_limit": 1,
            "automatic_model_repairs": 0,
            "translation_thinking_type": translation_thinking_type,
            "translation_effort": translation_effort,
            "translation_max_output_tokens": translation_max_output_tokens,
        },
    )
    atomic_write_json(
        run_dir / "status.json",
        {
            "schema_version": "mathula.translation.one-call-status.v1",
            "status": "request_prepared",
            "updated_at": _utcnow(),
            "provider_http_requests_started": 0,
        },
    )

    if job.state == "analysis_ready":
        job.transition("translation_running")
        pipeline.jobs.save(job)

    provider_request_started = False
    stream_events_path = run_dir / "provider_stream_events.jsonl"
    partial_response_path = run_dir / "translation_response.partial.txt"
    stream_progress_path = run_dir / "provider_stream_progress.json"
    stream_events_path.touch()
    partial_response_path.touch()

    def append_stream_event(value: Mapping[str, Any]) -> None:
        with stream_events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(value), ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")

    def append_partial_text(value: str) -> None:
        if not value:
            return
        with partial_response_path.open("a", encoding="utf-8") as handle:
            handle.write(value)

    def provider_event(event: dict[str, Any]) -> None:
        nonlocal provider_request_started
        kind = str(event.get("event") or "provider")
        if kind == "request_attempt":
            provider_request_started = True
            atomic_write_json(
                run_dir / "status.json",
                {
                    "schema_version": "mathula.translation.one-call-status.v1",
                    "status": "provider_request_started",
                    "updated_at": _utcnow(),
                    "provider_http_requests_started": 1,
                    "attempt": int(event.get("attempt") or 1),
                    "max_retries": int(event.get("max_retries") or 1),
                    "timeout_seconds": float(
                        event.get("timeout_seconds") or timeout_seconds
                    ),
                    "streaming": bool(event.get("streaming", True)),
                },
            )
            emit(
                "provider",
                "Claude full-transcript streaming request started through Azure Foundry",
                provider_event=kind,
            )
        elif kind == "http_response":
            emit(
                "provider",
                f"Claude returned HTTP {event.get('status_code')}; opening SSE stream",
                provider_event=kind,
                status_code=event.get("status_code"),
            )
        elif kind == "stream_opened":
            atomic_write_json(
                run_dir / "status.json",
                {
                    "schema_version": "mathula.translation.one-call-status.v1",
                    "status": "provider_streaming",
                    "updated_at": _utcnow(),
                    "provider_http_requests_started": 1,
                    "streaming": True,
                    "read_idle_timeout_seconds": event.get(
                        "read_idle_timeout_seconds"
                    ),
                },
            )
            emit(
                "provider",
                "Azure Foundry SSE stream opened; waiting for Claude output",
                provider_event=kind,
            )
        elif kind == "stream_event":
            append_stream_event(
                {
                    "received_at": _utcnow(),
                    "event_type": event.get("event_type"),
                    "data": event.get("data"),
                }
            )
            data = event.get("data")
            if isinstance(data, Mapping) and data.get("type") == "content_block_delta":
                delta = data.get("delta")
                if isinstance(delta, Mapping) and delta.get("type") == "text_delta":
                    append_partial_text(str(delta.get("text") or ""))
        elif kind == "stream_progress":
            progress_record = {
                "schema_version": "mathula.translation.provider-stream-progress.v1",
                "updated_at": _utcnow(),
                "event_count": int(event.get("event_count") or 0),
                "text_characters": int(event.get("text_characters") or 0),
                "output_tokens": int(event.get("output_tokens") or 0),
                "last_event_type": event.get("last_event_type"),
            }
            atomic_write_json(stream_progress_path, progress_record)
            atomic_write_json(
                run_dir / "status.json",
                {
                    "schema_version": "mathula.translation.one-call-status.v1",
                    "status": "provider_streaming",
                    "updated_at": _utcnow(),
                    "provider_http_requests_started": 1,
                    "streaming": True,
                    **{
                        key: progress_record[key]
                        for key in (
                            "event_count",
                            "text_characters",
                            "output_tokens",
                            "last_event_type",
                        )
                    },
                },
            )
            emit(
                "provider",
                (
                    "Claude stream active: "
                    f"{progress_record['text_characters']} response characters, "
                    f"{progress_record['output_tokens']} output tokens"
                ),
                provider_event=kind,
                **progress_record,
            )
        elif kind == "stream_completed":
            emit(
                "provider",
                (
                    "Claude SSE stream completed: "
                    f"{int(event.get('text_characters') or 0)} response characters"
                ),
                status="completed",
                provider_event=kind,
                event_count=event.get("event_count"),
                text_characters=event.get("text_characters"),
                output_tokens=event.get("output_tokens"),
            )
        elif kind == "stream_truncated":
            emit(
                "provider",
                (
                    "Claude reached the output-token limit before finishing: "
                    f"{int(event.get('output_tokens') or 0)} total output tokens, "
                    f"{int(event.get('thinking_tokens') or 0)} thinking tokens"
                ),
                status="warning",
                provider_event=kind,
                stop_reason=event.get("stop_reason"),
                max_tokens=event.get("max_tokens"),
                text_characters=event.get("text_characters"),
            )
        elif kind in {
            "request_timeout",
            "request_connection_error",
            "stream_timeout",
            "stream_interrupted",
        }:
            emit(
                "provider",
                f"Claude request failed: {kind}",
                status="warning",
                provider_event=kind,
            )
        elif kind == "retry_backoff":
            # This must remain impossible because max_retries is forced to one.
            raise RuntimeError(
                "One-call translation attempted an automatic provider retry"
            )

    one_request_provider = provider
    with_options = getattr(provider, "with_request_options", None)
    if callable(with_options):
        one_request_provider = with_options(
            timeout_seconds=timeout_seconds,
            max_retries=1,
            event_callback=provider_event,
        )

    attempt = pipeline.jobs.begin_attempt(
        job,
        "claude_multivariant_translation_one_call",
        forced=force,
    )
    try:
        emit(
            "provider",
            f"Sending {len(request.get('units') or [])} units in one Claude request through Azure Foundry",
        )
        response = one_request_provider.translate_multivariant(request)
        metadata = _provider_metadata(response)
        returned_provider = str(metadata.get("provider") or configured_provider or "").strip().lower()
        if (
            configured_provider
            and returned_provider
            and returned_provider != configured_provider
        ):
            raise ValueError(
                "Translation response metadata does not identify the configured "
                f"Azure Foundry provider: {returned_provider!r}"
            )
        metadata["provider"] = "azure-foundry-claude"
        metadata["foundry_endpoint"] = foundry_endpoint
        metadata["foundry_deployment"] = foundry_deployment
        metadata["streaming"] = True
        metadata["transport"] = "server_sent_events"

        # Persist the returned structured object before local installation work.
        atomic_write_json(run_dir / "translation_response.json", response.data)
        atomic_write_json(run_dir / "provider_metadata.json", metadata)
        atomic_write_json(
            run_dir / "status.json",
            {
                "schema_version": "mathula.translation.one-call-status.v1",
                "status": "provider_response_received",
                "updated_at": _utcnow(),
                "provider_http_requests_started": 1 if provider_request_started else 1,
            },
        )

        installed, validation = prepare_installed_translation(
            response=response.data,
            request=request,
            mode="cloud_translation_full_transcript_one_call",
            provider_metadata={
                **metadata,
                "translation_mode": "full_transcript_one_call",
                "provider_http_request_limit": 1,
                "automatic_model_repairs": 0,
                "request_timeout_seconds": timeout_seconds,
            },
        )
        atomic_write_json(run_dir / "validation.json", validation)

        pronunciation = pipeline._pronunciation_dictionary(job)
        written = write_translation_artifacts(
            job_root=paths.job_root,
            target_locale=job.target_language,
            installed=installed,
            pronunciation_dictionary=pronunciation,
            backup_dir=run_dir / "backups",
        )
        atomic_write_json(run_dir / "installed_translation.json", installed)
        atomic_write_json(
            run_dir / "install_manifest.json",
            {
                "schema_version": "mathula.translation.cloud-install-result.v3-one-call",
                "job_id": job.job_id,
                "prompt_version": PROMPT_VERSION,
                "request_sha256": request["request_sha256"],
                "unit_count": len(request.get("units") or []),
                "provider_http_requests_started": 1,
                "automatic_model_repairs": 0,
                "streaming": True,
                "transport": "server_sent_events",
                "provider_stream_events": str(stream_events_path),
                "partial_response": str(partial_response_path),
                **written,
            },
        )

        job.providers.update(
            translation=metadata.get("provider", "azure-foundry-claude"),
            translation_model=metadata.get("model_returned", ""),
            translation_prompt_version=metadata.get("prompt_version") or PROMPT_VERSION,
        )
        job.media["translation_sha256"] = written["dubbing_sha256"]
        job.media["translation_multivariant_sha256"] = written["translation_sha256"]
        job.media["translation_request_sha256"] = request["request_sha256"]
        job.media["translation_run_directory"] = str(run_dir)
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
        job.transition("translation_ready")
        job.transition("azure_tts_queued")
        if "translation" not in job.completed_stages:
            job.completed_stages.append("translation")
        pipeline.jobs.finish_attempt(
            job,
            "claude_multivariant_translation_one_call",
            attempt,
            "completed",
        )
        pipeline.jobs.save(job)
        atomic_write_json(
            run_dir / "status.json",
            {
                "schema_version": "mathula.translation.one-call-status.v1",
                "status": "completed",
                "updated_at": _utcnow(),
                "provider_http_requests_started": 1,
                "translation_target": written["translation_target"],
                "dubbing_target": written["dubbing_target"],
            },
        )
        emit("complete", "Azure Foundry full-transcript translation is ready for Azure dubbing", status="completed")
        return read_json(paths.three_text_translation)
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            safe_error = {
                "stage": "claude_multivariant_translation_one_call",
                "error_type": "KeyboardInterrupt",
                "message": "Translation was interrupted by the operator",
                "retryable": True,
                "at": _utcnow(),
            }
        elif isinstance(exc, Exception):
            safe_error = pipeline._safe_error(
                "claude_multivariant_translation_one_call", exc
            )
        else:
            safe_error = {
                "stage": "claude_multivariant_translation_one_call",
                "error_type": type(exc).__name__,
                "message": str(exc)[:500],
                "retryable": False,
                "at": _utcnow(),
            }
        failure = {
            "schema_version": "mathula.translation.one-call-failure.v1",
            "failed_at": _utcnow(),
            "job_id": job.job_id,
            "request_sha256": request["request_sha256"],
            "provider_http_requests_started": 1 if provider_request_started else 0,
            "automatic_retry_started": False,
            "automatic_model_repair_started": False,
            "error": safe_error,
        }
        atomic_write_json(run_dir / "failure.json", failure)
        atomic_write_json(
            run_dir / "status.json",
            {
                "schema_version": "mathula.translation.one-call-status.v1",
                "status": "failed",
                "updated_at": _utcnow(),
                "provider_http_requests_started": failure[
                    "provider_http_requests_started"
                ],
                "error": safe_error,
            },
        )
        job.last_error = safe_error
        pipeline.jobs.finish_attempt(
            job,
            "claude_multivariant_translation_one_call",
            attempt,
            "failed",
        )
        pipeline.jobs.save(job)
        emit(
            "failed",
            "Full-transcript translation failed; no automatic second Claude call was made",
            status="warning",
        )
        raise


__all__ = ["translate_full_transcript_once"]
