"""Content-free, cumulative AI request and token accounting.

The provider adapters retain their native metadata shapes.  This module
normalises those shapes into one per-job report without copying prompts,
transcripts, model output, credentials, or server-tool evidence.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .atomic_io import atomic_write_json, read_json

AI_CONSUMPTION_REPORT_SCHEMA_VERSION = "mathula.ai-consumption-report.v1"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _integer(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _wire_requests(metadata: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    summary = _mapping(metadata.get("request_summary"))
    values = summary.get("wire_requests")
    if not isinstance(values, list):
        values = metadata.get("request_metrics")
    return [item for item in (values or []) if isinstance(item, Mapping)]


def _usage(metadata: Mapping[str, Any]) -> dict[str, int]:
    native = _mapping(metadata.get("token_usage"))
    if not native:
        native = _mapping(metadata.get("usage"))
    input_details = _mapping(native.get("input_tokens_details"))
    output_details = _mapping(native.get("output_tokens_details"))
    input_tokens = _integer(native.get("input_tokens"))
    output_tokens = _integer(native.get("output_tokens"))
    cache_read = _integer(
        native.get("cache_read_input_tokens")
        or input_details.get("cached_tokens")
    )
    cache_creation = _integer(native.get("cache_creation_input_tokens"))
    thinking = _integer(
        native.get("thinking_tokens")
        or output_details.get("thinking_tokens")
        or output_details.get("reasoning_tokens")
    )
    total = _integer(native.get("total_tokens")) or input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total,
        "thinking_tokens": thinking,
        "cache_read_input_tokens": cache_read,
        "cache_creation_input_tokens": cache_creation,
    }


def _record_id(
    *,
    operation: str,
    scope: str,
    metadata: Mapping[str, Any],
) -> str:
    request_sha256 = str(metadata.get("request_sha256") or "")
    identity = (
        {
            "operation": operation,
            "scope": scope,
            "request_sha256": request_sha256,
        }
        if request_sha256
        else {
            "operation": operation,
            "scope": scope,
            "provider": metadata.get("provider"),
            "model": metadata.get("model_returned") or metadata.get("model"),
            "prompt_hash": metadata.get("prompt_hash"),
            "response_id": metadata.get("response_id"),
            "request_timestamp": metadata.get("request_timestamp"),
            "batch_index": metadata.get("batch_index"),
            "chunk_index": metadata.get("chunk_index"),
        }
    )
    encoded = json.dumps(
        identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def consumption_record(
    *,
    operation: str,
    metadata: Mapping[str, Any],
    scope: str = "",
    status: str = "completed",
) -> dict[str, Any]:
    """Normalise one provider result into a content-free report record."""

    source = dict(metadata)
    wires = _wire_requests(source)
    usage = _usage(source)
    request_bytes = sum(
        _integer(item.get("request_body_bytes") or item.get("request_json_bytes"))
        for item in wires
    )
    if not request_bytes:
        request_bytes = _integer(source.get("request_body_bytes"))
    estimated_input = sum(
        _integer(item.get("estimated_input_tokens")) for item in wires
    )
    if not estimated_input:
        estimated_input = _integer(source.get("estimated_input_tokens"))
    schema_bytes = sum(
        _integer(item.get("output_schema_bytes")) for item in wires
    )
    reserved_output = sum(
        _integer(item.get("max_output_tokens")) for item in wires
    )
    response_bytes = _integer(source.get("response_body_bytes"))
    attempts = _integer(
        source.get("provider_attempts") or source.get("request_attempts")
    )
    if not attempts:
        attempts = max(1, len(wires)) if status == "completed" else len(wires)
    record_scope = str(scope or source.get("scope") or "")
    return {
        "record_id": _record_id(
            operation=operation,
            scope=record_scope,
            metadata=source,
        ),
        "recorded_at": _utcnow(),
        "operation": str(operation),
        "scope": record_scope,
        "status": str(status),
        "provider": str(source.get("provider") or "unknown"),
        "model": str(
            source.get("model_returned")
            or source.get("model")
            or source.get("model_requested")
            or ""
        ),
        "prompt_version": str(source.get("prompt_version") or ""),
        "provider_attempts": attempts,
        "repair_count": _integer(source.get("repair_count")),
        "request_count": len(wires) or (1 if request_bytes else 0),
        "request_body_bytes": request_bytes,
        "estimated_input_tokens": estimated_input,
        "output_schema_bytes": schema_bytes,
        "reserved_output_tokens": reserved_output,
        "response_body_bytes": response_bytes,
        **usage,
    }


def _sum(records: Iterable[Mapping[str, Any]], field: str) -> int:
    return sum(_integer(item.get(field)) for item in records)


def _summary(records: list[Mapping[str, Any]], duration_seconds: float) -> dict[str, Any]:
    fields = (
        "provider_attempts",
        "request_count",
        "request_body_bytes",
        "estimated_input_tokens",
        "output_schema_bytes",
        "reserved_output_tokens",
        "response_body_bytes",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "thinking_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
        "repair_count",
    )
    totals = {field: _sum(records, field) for field in fields}
    input_basis = totals["input_tokens"] + totals["cache_read_input_tokens"]
    totals["cache_read_share_percent"] = (
        round(totals["cache_read_input_tokens"] * 100.0 / input_basis, 2)
        if input_basis
        else 0.0
    )
    totals["schema_share_of_request_percent"] = (
        round(
            totals["output_schema_bytes"]
            * 100.0
            / totals["request_body_bytes"],
            2,
        )
        if totals["request_body_bytes"]
        else 0.0
    )
    totals["output_reservation_utilization_percent"] = (
        round(
            totals["output_tokens"]
            * 100.0
            / totals["reserved_output_tokens"],
            2,
        )
        if totals["reserved_output_tokens"]
        else 0.0
    )
    totals["completed_record_count"] = sum(
        1 for item in records if item.get("status") == "completed"
    )
    totals["failed_record_count"] = sum(
        1 for item in records if item.get("status") != "completed"
    )
    totals["extra_provider_attempts"] = max(
        0, totals["provider_attempts"] - totals["completed_record_count"]
    )
    totals["video_duration_seconds"] = round(max(0.0, duration_seconds), 3)
    minutes = duration_seconds / 60.0
    totals["tokens_per_video_minute"] = (
        round(totals["total_tokens"] / minutes, 2) if minutes > 0 else None
    )
    return totals


def _operation_summaries(
    records: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    operations = sorted({str(item.get("operation") or "unknown") for item in records})
    return [
        {
            "operation": operation,
            **_summary(
                [item for item in records if item.get("operation") == operation],
                0.0,
            ),
        }
        for operation in operations
    ]


def _recommendations(
    records: list[Mapping[str, Any]], summary: Mapping[str, Any]
) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    translation = [
        item
        for item in records
        if item.get("operation") == "multivariant_translation"
        and item.get("status") == "completed"
    ]
    if len(translation) >= 2 and _sum(
        translation, "cache_read_input_tokens"
    ) == 0:
        result.append(
            {
                "code": "translation_prompt_cache_not_observed",
                "message": (
                    "No translation cache reads were reported across multiple "
                    "chunks; consider a smaller context spine after confirming "
                    "the Azure deployment supports Anthropic prompt caching."
                ),
            }
        )
    if float(summary.get("schema_share_of_request_percent") or 0.0) >= 25.0:
        result.append(
            {
                "code": "provider_schema_is_large",
                "message": (
                    "Output schemas exceed 25% of request bytes; compact the "
                    "provider-facing schema while retaining strict local validation."
                ),
            }
        )
    reserved = _integer(summary.get("reserved_output_tokens"))
    if reserved and float(
        summary.get("output_reservation_utilization_percent") or 0.0
    ) < 15.0:
        result.append(
            {
                "code": "output_reservation_is_high",
                "message": (
                    "Actual output used less than 15% of reserved output tokens; "
                    "lower the operation-specific maximum after reviewing several jobs."
                ),
            }
        )
    if _integer(summary.get("extra_provider_attempts")):
        result.append(
            {
                "code": "retry_overhead_observed",
                "message": (
                    "Extra provider attempts consumed capacity; inspect failed "
                    "records and local validation repairs before increasing retry limits."
                ),
            }
        )
    return result


def update_ai_consumption_report(
    *,
    job_root: Path,
    records: Iterable[Mapping[str, Any]],
    video_duration_seconds: float | None = None,
) -> dict[str, Any]:
    """Insert or replace records and atomically refresh the per-job report."""

    root = Path(job_root)
    report_path = root / "analysis" / "ai_consumption_report.json"
    existing_records: list[dict[str, Any]] = []
    existing_duration = 0.0
    if report_path.is_file():
        try:
            existing = read_json(report_path)
            existing_records = [
                dict(item)
                for item in existing.get("records") or []
                if isinstance(item, Mapping) and item.get("record_id")
            ]
            existing_duration = float(
                _mapping(existing.get("summary")).get("video_duration_seconds")
                or 0.0
            )
        except (OSError, TypeError, ValueError):
            existing_records = []
    indexed = {str(item["record_id"]): item for item in existing_records}
    for raw in records:
        if not isinstance(raw, Mapping) or not raw.get("record_id"):
            continue
        indexed[str(raw["record_id"])] = dict(raw)
    ordered = sorted(
        indexed.values(),
        key=lambda item: (
            str(item.get("operation") or ""),
            str(item.get("scope") or ""),
            str(item.get("record_id") or ""),
        ),
    )
    duration = (
        max(0.0, float(video_duration_seconds))
        if video_duration_seconds is not None
        else existing_duration
    )
    summary = _summary(ordered, duration)
    report = {
        "schema_version": AI_CONSUMPTION_REPORT_SCHEMA_VERSION,
        "updated_at": _utcnow(),
        "job_id": root.name,
        "privacy": (
            "Counts and provider metadata only; prompts, transcripts, model "
            "responses, credentials, and server-tool evidence are excluded."
        ),
        "summary": summary,
        "operations": _operation_summaries(ordered),
        "recommendations": _recommendations(ordered, summary),
        "records": ordered,
    }
    atomic_write_json(report_path, report)
    return report


__all__ = [
    "AI_CONSUMPTION_REPORT_SCHEMA_VERSION",
    "consumption_record",
    "update_ai_consumption_report",
]
