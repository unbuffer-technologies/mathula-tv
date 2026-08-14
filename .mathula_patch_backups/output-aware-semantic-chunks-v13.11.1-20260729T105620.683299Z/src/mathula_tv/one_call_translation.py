"""Resumable semantic-context translation through Azure AI Foundry Claude.

The public function name is retained for compatibility with the existing
production pipeline.  Internally, translation is no longer one giant response:
contiguous semantic chunks are generated, validated, and checkpointed one at a
time.  Every chunk receives a stable transcript-wide context spine, exact local
source context, previous translated tail, and accumulated terminology.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
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
    merge_multivariant_batch_responses,
    prepare_installed_translation,
    validate_multivariant_response,
    write_translation_artifacts,
)
from .semantic_translation_chunks import build_semantic_batch_requests, sha256_json

ProgressCallback = Callable[[dict[str, Any]], None]
_STAGE = "claude_multivariant_translation_semantic_chunk"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_emit(callback: ProgressCallback | None, event: Mapping[str, Any]) -> None:
    if callback is None:
        return
    try:
        callback(dict(event))
    except Exception:
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


def _safe_error(pipeline: Any, exc: BaseException) -> dict[str, Any]:
    if isinstance(exc, KeyboardInterrupt):
        return {
            "stage": _STAGE,
            "error_type": "KeyboardInterrupt",
            "message": "Translation was interrupted by the operator",
            "retryable": True,
            "at": _utcnow(),
        }
    if isinstance(exc, Exception):
        return pipeline._safe_error(_STAGE, exc)
    return {
        "stage": _STAGE,
        "error_type": type(exc).__name__,
        "message": str(exc)[:500],
        "retryable": False,
        "at": _utcnow(),
    }


def _prior_context(responses: list[Mapping[str, Any]], limit: int = 3) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    for response in responses:
        for unit in response.get("units") or []:
            variants = {
                str(item.get("variant_id") or ""): str(item.get("spoken_text") or "")
                for item in unit.get("variants") or []
                if isinstance(item, Mapping)
            }
            units.append(
                {
                    "unit_id": str(unit.get("unit_id") or ""),
                    "speaker_id": str(unit.get("speaker_id") or ""),
                    "source_text": str(unit.get("source_text") or ""),
                    "natural_translation": variants.get("natural", ""),
                    "concise_translation": variants.get("concise", ""),
                    "compact_translation": variants.get("compact", ""),
                }
            )
    return units[-max(0, limit) :]


def _prior_terminology(responses: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for response in responses:
        for item in response.get("global_terminology") or []:
            if not isinstance(item, Mapping):
                continue
            key = json.dumps(dict(item), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            if key in seen:
                continue
            seen.add(key)
            result.append(dict(item))
    return result


def _aggregate_metadata(items: list[Mapping[str, Any]], **extra: Any) -> dict[str, Any]:
    usage_fields = (
        "input_tokens",
        "output_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
    )

    def usage(item: Mapping[str, Any], field: str) -> int:
        nested = item.get("token_usage")
        if isinstance(nested, Mapping):
            return int(nested.get(field) or 0)
        return int(item.get(field) or 0)

    aggregate_usage = {
        field: sum(usage(item, field) for item in items)
        for field in usage_fields
    }
    first = dict(items[0]) if items else {}
    return {
        **first,
        "provider": str(first.get("provider") or "azure-foundry-claude"),
        "prompt_version": PROMPT_VERSION,
        "translation_mode": "semantic_context_chunks",
        "chunk_count": len(items),
        "chunks": [dict(item) for item in items],
        "token_usage": aggregate_usage,
        "provider_attempts": sum(int(item.get("provider_attempts") or 0) for item in items),
        **extra,
    }


def _load_checkpoint(
    *,
    chunk_dir: Path,
    request: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    completed = chunk_dir / "completed.json"
    response_path = chunk_dir / "translation_response.json"
    metadata_path = chunk_dir / "provider_metadata.json"
    if not (completed.is_file() and response_path.is_file() and metadata_path.is_file()):
        return None
    marker = read_json(completed)
    if marker.get("request_sha256") != request.get("request_sha256"):
        return None
    response = read_json(response_path)
    validate_multivariant_response(response, request=request)
    return response, read_json(metadata_path)


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
    chunk_target_tokens: int | None = None,
    chunk_hard_tokens: int | None = None,
    context_spine_tokens: int | None = None,
) -> dict[str, Any]:
    """Translate the complete authoritative timeline using semantic chunks.

    ``batch_size`` is retained as an optional maximum-unit safety cap; normal
    chunk boundaries are selected by estimated token budget and semantic pauses.
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
    if job.state == "synthesis_queued":
        raise ValueError("Migrate the validated legacy translation; do not regenerate it silently")
    foundry_endpoint = str(getattr(provider_config, "base_url", "") or "")
    foundry_deployment = str(getattr(provider_config, "model", "") or "")

    settings = pipeline.settings
    target_tokens = int(
        chunk_target_tokens
        if chunk_target_tokens is not None
        else os.getenv("MATHULA_TV_TRANSLATION_CHUNK_TARGET_TOKENS", "3200")
    )
    hard_tokens = int(
        chunk_hard_tokens
        if chunk_hard_tokens is not None
        else os.getenv("MATHULA_TV_TRANSLATION_CHUNK_HARD_TOKENS", "4600")
    )
    spine_tokens = int(
        context_spine_tokens
        if context_spine_tokens is not None
        else os.getenv("MATHULA_TV_TRANSLATION_CONTEXT_SPINE_TOKENS", "4500")
    )
    effective_context_units = int(
        context_units
        if context_units is not None
        else os.getenv("MATHULA_TV_TRANSLATION_CONTEXT_UNITS", "3")
    )
    effective_timeout = float(
        request_timeout_seconds
        if request_timeout_seconds is not None
        else getattr(settings, "translation_request_timeout_seconds", settings.claude_timeout_seconds)
    )
    effective_attempts = int(
        batch_max_retries
        if batch_max_retries is not None
        else os.getenv("MATHULA_TV_TRANSLATION_BATCH_MAX_RETRIES", "5")
    )
    provider_call_limit = int(
        max_provider_calls
        if max_provider_calls is not None
        else os.getenv(
            "MATHULA_TV_TRANSLATION_MAX_PROVIDER_CALLS_PER_SESSION",
            str(getattr(settings, "translation_max_provider_calls_per_session", 64)),
        )
    )
    output_ceiling = int(
        os.getenv("MATHULA_TV_TRANSLATION_CHUNK_MAX_OUTPUT_TOKENS", "24000")
    )
    max_units = int(batch_size) if batch_size is not None else None
    if target_tokens <= 0 or hard_tokens < target_tokens:
        raise ValueError("translation chunk token budgets are invalid")
    if spine_tokens < 800:
        raise ValueError("translation context spine token budget must be at least 800")
    if effective_context_units < 0:
        raise ValueError("translation context units must not be negative")
    if effective_timeout <= 0 or not math.isfinite(effective_timeout):
        raise ValueError("translation request timeout must be a positive finite number")
    if effective_attempts <= 0 or provider_call_limit <= 0:
        raise ValueError("translation retry and provider-call limits must be positive")

    started_at = time.monotonic()

    def emit(stage: str, message: str, *, status: str = "running", **extra: Any) -> None:
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

    emit("prepare", "Loading the authoritative transcript and building semantic translation chunks")
    paths = pipeline.paths(job.job_id)
    authoritative = require_authoritative_transcript(
        work_dir=settings.work_dir,
        job_id=job.job_id,
    )
    pipeline.bind_entities(job, force=force)
    units = pipeline.build_units(job, force=False)
    units_sha256 = checksum(paths.dubbing_units)

    if paths.raw_multivariant_translation.is_file() and paths.three_text_translation.is_file() and not force:
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
        raise ValueError("Translation request was not built from the authoritative corrected transcript")
    if request.get("source_timeline_file_sha256") != units_sha256:
        raise ValueError("Translation request was not built from the current deterministic dubbing units")

    batches, plan = build_semantic_batch_requests(
        request,
        target_tokens=target_tokens,
        hard_tokens=hard_tokens,
        context_units=effective_context_units,
        context_spine_tokens=spine_tokens,
        max_units=max_units,
        max_output_tokens=output_ceiling,
    )
    chunk_count = len(batches)
    session_key = hashlib.sha256(
        f"{request['request_sha256']}:{PROMPT_VERSION}:{plan['plan_sha256']}".encode("utf-8")
    ).hexdigest()[:24]
    session_root = paths.job_root / "translation" / "cloud_chunks" / session_key
    if restart_batches and session_root.exists():
        archive_root = paths.job_root / "translation" / "cloud_chunks" / "restarted"
        archive_root.mkdir(parents=True, exist_ok=True)
        shutil.move(
            str(session_root),
            str(archive_root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + session_key)),
        )
    session_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(session_root / "full_translation_request.json", request)
    atomic_write_json(session_root / "chunk_plan.json", plan)
    atomic_write_json(
        session_root / "session_manifest.json",
        {
            "schema_version": "mathula.translation.semantic-chunk-session.v1",
            "created_at": _utcnow(),
            "job_id": job.job_id,
            "target_locale": job.target_language,
            "prompt_version": PROMPT_VERSION,
            "translation_mode": "semantic_context_chunks",
            "session_key": session_key,
            "full_request_sha256": request["request_sha256"],
            "plan_sha256": plan["plan_sha256"],
            "chunk_count": chunk_count,
            "chunk_target_tokens": target_tokens,
            "chunk_hard_tokens": hard_tokens,
            "context_spine_tokens": spine_tokens,
            "context_units": effective_context_units,
            "max_units_per_chunk": max_units,
            "request_timeout_seconds": effective_timeout,
            "provider_attempt_limit_per_chunk": effective_attempts,
            "provider_http_request_limit_per_session": provider_call_limit,
            "chunk_max_output_tokens": output_ceiling,
        },
    )
    emit(
        "prepare",
        f"Prepared {len(request.get('units') or [])} units in {chunk_count} semantic chunks",
        current=0,
        total=chunk_count,
        context_spine_units=plan["context_spine"]["source_unit_count"],
    )

    if job.state == "analysis_ready":
        job.transition("translation_running")
    job.media["translation_batch_directory"] = str(session_root)
    job.media["translation_run_directory"] = str(session_root)
    pipeline.jobs.save(job)

    responses: list[dict[str, Any]] = []
    metadata_items: list[dict[str, Any]] = []
    budget_path = session_root / "provider_call_budget.json"
    provider_http_requests_started = 0
    if budget_path.is_file():
        try:
            previous_budget = read_json(budget_path)
            provider_http_requests_started = int(previous_budget.get("started") or 0)
        except Exception:
            provider_http_requests_started = 0
    attempt = pipeline.jobs.begin_attempt(job, _STAGE, forced=force)

    try:
        for chunk_index, source_batch in enumerate(batches, start=1):
            chunk_started = time.monotonic()
            batch_request = json.loads(json.dumps(source_batch, ensure_ascii=False))
            translation_batch = dict(batch_request.get("translation_batch") or {})
            translation_batch["cacheable_context"] = plan["context_spine"]
            translation_batch["prior_translated_context"] = _prior_context(responses)
            translation_batch["prior_global_terminology"] = _prior_terminology(responses)
            batch_request["translation_batch"] = translation_batch
            batch_request["request_sha256"] = sha256_json(
                {key: value for key, value in batch_request.items() if key != "request_sha256"}
            )

            chunk_dir = session_root / f"chunk_{chunk_index:04d}"
            chunk_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_json(chunk_dir / "translation_request.json", batch_request)
            checkpoint = _load_checkpoint(chunk_dir=chunk_dir, request=batch_request)
            if checkpoint is not None:
                response_data, metadata = checkpoint
                responses.append(response_data)
                metadata_items.append(metadata)
                emit(
                    "chunk",
                    f"Chunk {chunk_index}/{chunk_count} reused from a validated checkpoint",
                    status="completed",
                    current=chunk_index,
                    total=chunk_count,
                    chunk_index=chunk_index,
                    checkpoint_reused=True,
                )
                continue

            stream_events = chunk_dir / "provider_stream_events.jsonl"
            partial_response = chunk_dir / "translation_response.partial.txt"
            stream_events.write_text("", encoding="utf-8")
            partial_response.write_text("", encoding="utf-8")

            def request_guard(event: dict[str, Any]) -> None:
                nonlocal provider_http_requests_started
                if provider_http_requests_started >= provider_call_limit:
                    raise RuntimeError(
                        "MATHULA_TV_TRANSLATION_MAX_PROVIDER_CALLS_PER_SESSION reached before translation completed"
                    )
                provider_http_requests_started += 1
                atomic_write_json(
                    budget_path,
                    {
                        "schema_version": "mathula.translation.provider-call-budget.v1",
                        "updated_at": _utcnow(),
                        "limit": provider_call_limit,
                        "started": provider_http_requests_started,
                        "remaining": provider_call_limit - provider_http_requests_started,
                        "chunk_index": chunk_index,
                        "provider_attempt": int(event.get("attempt") or 1),
                    },
                )

            def provider_event(event: dict[str, Any]) -> None:
                kind = str(event.get("event") or "provider")
                if kind == "stream_event":
                    with stream_events.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
                    data = event.get("data")
                    if isinstance(data, Mapping) and data.get("type") == "content_block_delta":
                        delta = data.get("delta")
                        if isinstance(delta, Mapping) and delta.get("type") == "text_delta":
                            with partial_response.open("a", encoding="utf-8") as handle:
                                handle.write(str(delta.get("text") or ""))
                    return
                if kind == "http_response":
                    status_code = int(event.get("status_code") or 0)
                    message = (
                        f"Chunk {chunk_index}/{chunk_count}: HTTP {status_code}; opening SSE stream"
                        if 200 <= status_code < 300
                        else f"Chunk {chunk_index}/{chunk_count}: HTTP {status_code}; request rejected before SSE"
                    )
                    status = "running" if 200 <= status_code < 300 else "warning"
                elif kind == "retry_backoff":
                    message = (
                        f"Chunk {chunk_index}/{chunk_count}: retrying after "
                        f"{event.get('delay_seconds')}s ({event.get('reason')})"
                    )
                    status = "warning"
                elif kind == "request_attempt":
                    message = (
                        f"Chunk {chunk_index}/{chunk_count}: Claude attempt "
                        f"{event.get('attempt')}/{event.get('max_retries')}"
                    )
                    status = "running"
                elif kind == "stream_progress":
                    message = (
                        f"Chunk {chunk_index}/{chunk_count}: "
                        f"{int(event.get('text_characters') or 0)} response characters"
                    )
                    status = "running"
                elif kind == "stream_completed":
                    message = f"Chunk {chunk_index}/{chunk_count}: SSE response completed"
                    status = "completed"
                else:
                    message = f"Chunk {chunk_index}/{chunk_count}: {kind}"
                    status = "running"
                emit(
                    "provider",
                    message,
                    status=status,
                    current=chunk_index - 1,
                    total=chunk_count,
                    chunk_index=chunk_index,
                    provider_event=kind,
                    **{key: value for key, value in event.items() if key != "event"},
                )

            batch_provider = provider
            with_options = getattr(provider, "with_request_options", None)
            if callable(with_options):
                batch_provider = with_options(
                    timeout_seconds=effective_timeout,
                    max_retries=effective_attempts,
                    event_callback=provider_event,
                    request_guard=request_guard,
                )

            emit(
                "chunk",
                (
                    f"Sending chunk {chunk_index}/{chunk_count}: "
                    f"{len(batch_request.get('units') or [])} units, "
                    f"~{translation_batch.get('estimated_source_tokens')} source tokens, "
                    f"max_output_tokens={translation_batch.get('max_output_tokens')}"
                ),
                current=chunk_index - 1,
                total=chunk_count,
                chunk_index=chunk_index,
            )
            try:
                response = batch_provider.translate_multivariant(batch_request)
                response_data = dict(response.data)
                validation = validate_multivariant_response(response_data, request=batch_request)
                metadata = _provider_metadata(response)
                metadata.update(
                    {
                        "chunk_index": chunk_index,
                        "chunk_count": chunk_count,
                        "request_sha256": batch_request["request_sha256"],
                        "estimated_source_tokens": translation_batch.get("estimated_source_tokens"),
                        "max_output_tokens": translation_batch.get("max_output_tokens"),
                        "context_spine_sha256": translation_batch.get("context_spine_sha256"),
                    }
                )
                atomic_write_json(chunk_dir / "translation_response.json", response_data)
                atomic_write_json(chunk_dir / "provider_metadata.json", metadata)
                atomic_write_json(chunk_dir / "validation.json", validation)
                atomic_write_json(
                    chunk_dir / "completed.json",
                    {
                        "schema_version": "mathula.translation.semantic-chunk-checkpoint.v1",
                        "completed_at": _utcnow(),
                        "chunk_index": chunk_index,
                        "chunk_count": chunk_count,
                        "request_sha256": batch_request["request_sha256"],
                        "response_sha256": validation.get("response_sha256"),
                        "unit_ids": translation_batch.get("requested_unit_ids"),
                    },
                )
                responses.append(response_data)
                metadata_items.append(metadata)
                emit(
                    "chunk",
                    f"Chunk {chunk_index}/{chunk_count} validated and checkpointed",
                    status="completed",
                    current=chunk_index,
                    total=chunk_count,
                    chunk_index=chunk_index,
                    elapsed_chunk_seconds=time.monotonic() - chunk_started,
                )
            except BaseException as exc:
                safe_error = _safe_error(pipeline, exc)
                atomic_write_json(
                    chunk_dir / "failure.json",
                    {
                        "schema_version": "mathula.translation.semantic-chunk-failure.v1",
                        "failed_at": _utcnow(),
                        "chunk_index": chunk_index,
                        "request_sha256": batch_request["request_sha256"],
                        "completed_chunks_preserved": len(responses),
                        "error": safe_error,
                    },
                )
                job.last_error = safe_error
                pipeline.jobs.save(job)
                emit(
                    "failed",
                    (
                        f"Chunk {chunk_index}/{chunk_count} failed; "
                        f"{len(responses)} completed checkpoint(s) remain reusable"
                    ),
                    status="warning",
                    current=len(responses),
                    total=chunk_count,
                    chunk_index=chunk_index,
                )
                raise

        emit("merge", "Merging and validating all semantic translation chunks", current=chunk_count, total=chunk_count)
        merged_response = merge_multivariant_batch_responses(request=request, responses=responses)
        aggregate_metadata = _aggregate_metadata(
            metadata_items,
            chunk_plan_sha256=plan["plan_sha256"],
            context_spine_sha256=plan["context_spine"]["context_spine_sha256"],
            chunk_target_tokens=target_tokens,
            chunk_hard_tokens=hard_tokens,
            context_spine_tokens=spine_tokens,
            context_units=effective_context_units,
            provider_http_requests_started=provider_http_requests_started,
            provider_http_request_limit=provider_call_limit,
            request_timeout_seconds=effective_timeout,
            provider_attempt_limit_per_chunk=effective_attempts,
            prompt_cache_requested=True,
            foundry_endpoint=foundry_endpoint,
            foundry_deployment=foundry_deployment,
            streaming=True,
            transport="server_sent_events",
        )
        installed, validation = prepare_installed_translation(
            response=merged_response,
            request=request,
            mode="cloud_translation_semantic_context_chunks",
            provider_metadata=aggregate_metadata,
        )

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        run_dir = paths.job_root / "translation" / "cloud_runs" / stamp
        run_dir.mkdir(parents=True, exist_ok=False)
        atomic_write_json(run_dir / "translation_request.json", request)
        atomic_write_json(run_dir / "translation_response.json", merged_response)
        atomic_write_json(run_dir / "provider_metadata.json", aggregate_metadata)
        atomic_write_json(run_dir / "validation.json", validation)
        atomic_write_json(run_dir / "chunk_plan.json", plan)
        atomic_write_json(
            run_dir / "chunk_session.json",
            {
                "schema_version": "mathula.translation.semantic-chunk-session-reference.v1",
                "session_root": str(session_root),
                "session_manifest_sha256": checksum(session_root / "session_manifest.json"),
                "chunk_count": chunk_count,
            },
        )

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
                "schema_version": "mathula.translation.cloud-install-result.v4-semantic-chunks",
                "job_id": job.job_id,
                "prompt_version": PROMPT_VERSION,
                "request_sha256": request["request_sha256"],
                "chunk_plan_sha256": plan["plan_sha256"],
                "chunk_count": chunk_count,
                "provider_http_requests_started": provider_http_requests_started,
                "automatic_model_repairs": 0,
                **written,
            },
        )

        job.providers.update(
            translation=aggregate_metadata.get("provider", "azure-foundry-claude"),
            translation_model=aggregate_metadata.get("model_returned", ""),
            translation_prompt_version=aggregate_metadata.get("prompt_version") or PROMPT_VERSION,
        )
        job.media["translation_sha256"] = written["dubbing_sha256"]
        job.media["translation_multivariant_sha256"] = written["translation_sha256"]
        job.media["translation_request_sha256"] = request["request_sha256"]
        job.media["translation_run_directory"] = str(run_dir)
        job.media["translation_batch_directory"] = str(session_root)
        job.human_review_flags = list(
            dict.fromkeys([*job.human_review_flags, "multivariant_translation_requires_human_review"])
        )
        job.last_error = None
        job.transition("translation_ready")
        job.transition("azure_tts_queued")
        if "translation" not in job.completed_stages:
            job.completed_stages.append("translation")
        pipeline.jobs.finish_attempt(job, _STAGE, attempt, "completed")
        pipeline.jobs.save(job)
        atomic_write_json(
            session_root / "completed.json",
            {
                "schema_version": "mathula.translation.semantic-chunk-session-complete.v1",
                "completed_at": _utcnow(),
                "chunk_count": chunk_count,
                "provider_http_requests_started": provider_http_requests_started,
                "translation_target": written["translation_target"],
                "dubbing_target": written["dubbing_target"],
            },
        )
        emit(
            "complete",
            "Semantic-context translation is ready; dub-azure is next",
            status="completed",
            current=chunk_count,
            total=chunk_count,
        )
        return read_json(paths.three_text_translation)
    except BaseException as exc:
        safe_error = _safe_error(pipeline, exc)
        job.last_error = safe_error
        # Keep translation_running. A normal rerun reuses every valid checkpoint.
        pipeline.jobs.finish_attempt(job, _STAGE, attempt, "failed")
        pipeline.jobs.save(job)
        atomic_write_json(
            session_root / "failure.json",
            {
                "schema_version": "mathula.translation.semantic-chunk-session-failure.v1",
                "failed_at": _utcnow(),
                "job_id": job.job_id,
                "full_request_sha256": request["request_sha256"],
                "completed_chunk_count": len(responses),
                "chunk_count": chunk_count,
                "provider_http_requests_started": provider_http_requests_started,
                "error": safe_error,
            },
        )
        raise


__all__ = ["translate_full_transcript_once"]
