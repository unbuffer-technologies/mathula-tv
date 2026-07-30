"""Resumable semantic-context translation through Azure AI Foundry Claude.

The public function name is retained for compatibility with the existing
production pipeline.  Internally, translation is no longer one giant response:
contiguous semantic chunks are generated, validated, and checkpointed one at a
time.  Every chunk receives a stable transcript-wide context spine, exact local
source context, previous translated tail, and accumulated terminology.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from .ai_consumption import consumption_record, update_ai_consumption_report
from .ai_provider import expand_compact_multivariant_result
from .atomic_io import atomic_write_json, read_json
from .autocorrect_stage import require_authoritative_transcript
from .media import checksum
from .multivariant_translation import (
    PROMPT_VERSION,
    build_job_translation_request,
    identity_requirements_for_unit,
    merge_multivariant_batch_responses,
    MultivariantTranslationError,
    normalize_multivariant_response,
    prepare_installed_translation,
    restore_missing_hard_identities_in_unit,
    validate_multivariant_response,
    write_translation_artifacts,
)
from .semantic_translation_chunks import (
    adaptive_context_spine_tokens,
    build_semantic_batch_requests,
    estimate_unit_tokens,
    sha256_json,
)

ProgressCallback = Callable[[dict[str, Any]], None]
_STAGE = "claude_multivariant_translation_semantic_chunk"
_TARGETED_IDENTITY_REPAIR_SCHEMA_VERSION = (
    "mathula.translation.targeted-identity-repair.v1"
)
_TARGETED_VALIDATION_REPAIR_SCHEMA_VERSION = (
    "mathula.translation.targeted-validation-repair.v1"
)
_SPLIT_RECOVERY_SCHEMA_VERSION = "mathula.translation.chunk-split-recovery.v1"
_MISSING_IDENTITY_RE = re.compile(
    r"Unit (?P<unit_id>[^\s]+) variant "
    r"(?P<variant_id>natural|concise|compact) is missing protected spans "
    r"or approved identity aliases:",
    re.IGNORECASE,
)
_UNIT_VALIDATION_RE = re.compile(
    r"Unit (?P<unit_id>[^\s:]+)"
    r"(?: variant (?P<variant_id>natural|concise|compact))?",
    re.IGNORECASE,
)


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
    """Return only the translated tail not already repeated by local source context."""

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
                    "natural_translation": variants.get("natural", ""),
                }
            )
    return units[-max(0, limit) :]


def _prior_context_legacy(
    responses: list[Mapping[str, Any]], limit: int = 3
) -> list[dict[str, Any]]:
    """Reproduce the v13.16.5 request identity while resuming its checkpoints."""

    units: list[dict[str, Any]] = []
    for response in responses:
        for unit in response.get("units") or []:
            variants = {
                str(item.get("variant_id") or ""): str(
                    item.get("spoken_text") or ""
                )
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


def _prior_terminology(
    responses: list[Mapping[str, Any]],
    current_request: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Keep only established terms that can affect the active chunk."""

    batch = (
        current_request.get("translation_batch")
        if isinstance(current_request.get("translation_batch"), Mapping)
        else {}
    )
    searchable = " ".join(
        [
            *[
                str(item.get("source_text") or "")
                for item in current_request.get("units") or []
                if isinstance(item, Mapping)
            ],
            *[
                str(item.get("source_text") or "")
                for item in batch.get("context_units") or []
                if isinstance(item, Mapping)
            ],
            *[
                str(span)
                for item in current_request.get("units") or []
                if isinstance(item, Mapping)
                for span in item.get("protected_spans") or []
            ],
        ]
    ).casefold()
    result: list[dict[str, Any]] = []
    seen_terms: set[str] = set()
    for response in responses:
        for item in response.get("global_terminology") or []:
            if not isinstance(item, Mapping):
                continue
            source_term = str(item.get("source_term") or "").strip()
            key = source_term.casefold()
            if not key or key in seen_terms or key not in searchable:
                continue
            seen_terms.add(key)
            result.append(dict(item))
    return result[:40]


def _prior_terminology_legacy(
    responses: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for response in responses:
        for item in response.get("global_terminology") or []:
            if not isinstance(item, Mapping):
                continue
            key = json.dumps(
                dict(item),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            if key in seen:
                continue
            seen.add(key)
            result.append(dict(item))
    return result


def _resumable_session_manifest(
    *,
    job_root: Path,
    request_sha256: str,
) -> dict[str, Any] | None:
    cloud_root = job_root / "translation" / "cloud_chunks"
    if not cloud_root.is_dir():
        return None
    candidates: list[tuple[float, dict[str, Any]]] = []
    for path in cloud_root.glob("*/session_manifest.json"):
        if path.parent.name == "restarted" or (path.parent / "completed.json").is_file():
            continue
        try:
            value = read_json(path)
            if (
                value.get("full_request_sha256") == request_sha256
                and value.get("prompt_version") == PROMPT_VERSION
            ):
                candidates.append((path.stat().st_mtime, dict(value)))
        except (OSError, TypeError, ValueError):
            continue
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


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
        "targeted_identity_repair_count": sum(
            int(item.get("targeted_identity_repair_count") or 0)
            for item in items
        ),
        "targeted_validation_repair_count": sum(
            int(item.get("targeted_validation_repair_count") or 0)
            for item in items
        ),
        "split_chunk_recovery_count": sum(
            int(item.get("split_chunk_recovery_count") or 0)
            for item in items
        ),
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



def _parse_complete_json_object(text: str) -> dict[str, Any] | None:
    """Parse one completed provider JSON object from a saved SSE text stream."""

    value = str(text or "").strip()
    if not value:
        return None
    candidates = [value]
    first = value.find("{")
    last = value.rfind("}")
    if first >= 0 and last > first:
        candidates.append(value[first : last + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(parsed, Mapping):
            return dict(parsed)
    return None


def _recover_failed_candidate(
    *,
    chunk_dir: Path,
    request: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None:
    """Recover a completed paid response after a local validator upgrade.

    Recovery first checks the structured failure details. Older coordinator
    versions did not always preserve ``candidate_output`` there, but the SSE
    stream writer still saved the complete compact JSON in
    ``translation_response.partial.txt``. Both sources are deterministic and
    are revalidated against the exact unchanged chunk request before becoming a
    checkpoint. No provider request is started during recovery.
    """

    failure_path = chunk_dir / "failure.json"
    if not failure_path.is_file():
        return None
    try:
        failure = read_json(failure_path)
        if failure.get("request_sha256") != request.get("request_sha256"):
            return None
        error = failure.get("error")
        details = error.get("details") if isinstance(error, Mapping) else None
        candidate: Mapping[str, Any] | None = None
        if isinstance(details, Mapping) and isinstance(
            details.get("candidate_output"), Mapping
        ):
            candidate = details.get("candidate_output")

        recovery_source = failure_path
        if candidate is None:
            partial_path = chunk_dir / "translation_response.partial.txt"
            if partial_path.is_file():
                parsed = _parse_complete_json_object(
                    partial_path.read_text(encoding="utf-8", errors="replace")
                )
                if parsed is not None:
                    candidate = expand_compact_multivariant_result(parsed)
                    recovery_source = partial_path

        if not isinstance(candidate, Mapping):
            return None

        # A provider-schema failure stores the parsed compact wire candidate
        # before normalisation. Canonicalise/expand it here so harmless nested
        # annotations can be recovered locally without another paid request.
        candidate = expand_compact_multivariant_result(candidate)
        normalized = normalize_multivariant_response(candidate, request=request)
        validation = validate_multivariant_response(normalized, request=request)
        metadata = {
            "provider": str(
                details.get("provider")
                if isinstance(details, Mapping)
                else "azure-foundry-claude"
            )
            or "azure-foundry-claude",
            "model_returned": str(
                details.get("model_returned") if isinstance(details, Mapping) else ""
            ),
            "prompt_version": str(
                details.get("prompt_version")
                if isinstance(details, Mapping)
                else PROMPT_VERSION
            )
            or PROMPT_VERSION,
            "provider_attempts": int(
                (details.get("provider_attempts") if isinstance(details, Mapping) else 0)
                or 0
            ),
            "token_usage": dict(
                details.get("token_usage")
                if isinstance(details, Mapping)
                and isinstance(details.get("token_usage"), Mapping)
                else {}
            ),
            "stop_reason": (
                details.get("stop_reason") if isinstance(details, Mapping) else None
            ),
            "recovered_from_failed_candidate": True,
            "recovery_source": str(recovery_source),
            "recovered_at": _utcnow(),
        }
        return normalized, metadata, validation
    except Exception:
        return None


def _load_saved_validation_failure(
    *,
    chunk_dir: Path,
    request: Mapping[str, Any],
) -> BaseException | None:
    """Load an existing failed chunk candidate for one-unit validation repair.

    This path is what makes the upgrade immediately useful for an already
    failed production job: the rerun must not buy the same complete chunk
    response again merely to rediscover the same unit-level validator error.
    """

    failure_path = chunk_dir / "failure.json"
    if not failure_path.is_file():
        return None
    try:
        failure = read_json(failure_path)
        if failure.get("request_sha256") != request.get("request_sha256"):
            return None
        error = failure.get("error")
        details = error.get("details") if isinstance(error, Mapping) else None
        saved_details = dict(details) if isinstance(details, Mapping) else {}
        candidate: Mapping[str, Any] | None = None
        if isinstance(saved_details.get("candidate_output"), Mapping):
            candidate = saved_details["candidate_output"]
        if candidate is None:
            repair_root = chunk_dir / "targeted_identity_repairs"
            saved_candidates = (
                sorted(
                    repair_root.glob("attempt_*/failed_candidate.json"),
                    reverse=True,
                )
                if repair_root.is_dir()
                else []
            )
            for saved_candidate_path in saved_candidates:
                saved_candidate = read_json(saved_candidate_path)
                if isinstance(saved_candidate, Mapping):
                    candidate = saved_candidate
                    saved_details["saved_targeted_repair_candidate_path"] = str(
                        saved_candidate_path
                    )
                    break
        if candidate is None:
            partial_path = chunk_dir / "translation_response.partial.txt"
            if partial_path.is_file():
                parsed = _parse_complete_json_object(
                    partial_path.read_text(encoding="utf-8", errors="replace")
                )
                if parsed is not None:
                    candidate = expand_compact_multivariant_result(parsed)
        if not isinstance(candidate, Mapping):
            return None

        normalized = normalize_multivariant_response(
            expand_compact_multivariant_result(candidate),
            request=request,
        )
        try:
            validate_multivariant_response(normalized, request=request)
        except MultivariantTranslationError as validation_error:
            issue = _unit_validation_issue(str(validation_error))
            if issue is None:
                return None
            recovered_error = MultivariantTranslationError(str(validation_error))
            recovered_error.details = {
                **saved_details,
                "validation_stage": "validator",
                "validation_error": (
                    f"MultivariantTranslationError: {validation_error}"
                ),
                "candidate_output": normalized,
                "loaded_from_saved_chunk_failure": True,
                "saved_chunk_failure_path": str(failure_path),
            }
            return recovered_error
        return None
    except Exception:
        return None


def _load_saved_split_failure(
    *,
    chunk_dir: Path,
    request: Mapping[str, Any],
) -> BaseException | None:
    """Recover a saved size/truncation failure without repeating the parent call."""

    failure_path = chunk_dir / "failure.json"
    if not failure_path.is_file():
        return None
    try:
        failure = read_json(failure_path)
        if failure.get("request_sha256") != request.get("request_sha256"):
            return None
        error = failure.get("error")
        if not isinstance(error, Mapping):
            return None
        details = error.get("details")
        details = dict(details) if isinstance(details, Mapping) else {}
        saved = RuntimeError(str(error.get("message") or "saved chunk failure"))
        saved.details = details
        if _chunk_split_recovery_reason(saved) is None:
            return None
        details["loaded_from_saved_chunk_failure"] = True
        details["saved_chunk_failure_path"] = str(failure_path)
        return saved
    except Exception:
        return None


def _exception_details(exc: BaseException) -> dict[str, Any]:
    details = getattr(exc, "details", None)
    if isinstance(details, Mapping):
        return dict(details)
    to_dict = getattr(exc, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
        if isinstance(value, Mapping) and isinstance(value.get("details"), Mapping):
            return dict(value["details"])
    return {}


def _missing_identity_issue(
    error: BaseException | str,
) -> dict[str, str] | None:
    details = _exception_details(error) if isinstance(error, BaseException) else {}
    validation_error = str(
        details.get("validation_error")
        or (str(error) if isinstance(error, BaseException) else error)
        or ""
    )
    if details and str(details.get("validation_stage") or "") != "validator":
        return None
    match = _MISSING_IDENTITY_RE.search(validation_error)
    if match is None:
        return None
    return {
        "unit_id": match.group("unit_id"),
        "variant_id": match.group("variant_id").casefold(),
        "validation_error": validation_error,
    }


def _unit_validation_issue(
    error: BaseException | str,
) -> dict[str, str] | None:
    """Resolve a validator rejection to the smallest replaceable source unit."""

    details = _exception_details(error) if isinstance(error, BaseException) else {}
    validation_error = str(
        details.get("validation_error")
        or (str(error) if isinstance(error, BaseException) else error)
        or ""
    )
    if details and str(details.get("validation_stage") or "") != "validator":
        return None
    match = _UNIT_VALIDATION_RE.search(validation_error)
    if match is None:
        return None
    return {
        "unit_id": match.group("unit_id"),
        "variant_id": str(match.group("variant_id") or "").casefold(),
        "validation_error": validation_error,
    }


def _next_repair_directory(chunk_dir: Path) -> Path:
    root = chunk_dir / "targeted_identity_repairs"
    root.mkdir(parents=True, exist_ok=True)
    existing = [
        int(path.name.split("_", 1)[1])
        for path in root.glob("attempt_[0-9][0-9][0-9][0-9]")
        if path.name.split("_", 1)[1].isdigit()
    ]
    attempt = max(existing, default=0) + 1
    path = root / f"attempt_{attempt:04d}"
    path.mkdir(parents=False, exist_ok=False)
    return path


def _next_validation_repair_directory(chunk_dir: Path) -> Path:
    root = chunk_dir / "targeted_validation_repairs"
    root.mkdir(parents=True, exist_ok=True)
    existing = [
        int(path.name.split("_", 1)[1])
        for path in root.glob("attempt_[0-9][0-9][0-9][0-9]")
        if path.name.split("_", 1)[1].isdigit()
    ]
    attempt = max(existing, default=0) + 1
    path = root / f"attempt_{attempt:04d}"
    path.mkdir(parents=False, exist_ok=False)
    return path


def _targeted_identity_repair_request(
    *,
    batch_request: Mapping[str, Any],
    candidate: Mapping[str, Any],
    issue: Mapping[str, str],
) -> dict[str, Any]:
    unit_id = issue["unit_id"]
    variant_id = issue["variant_id"]
    source_unit = next(
        (
            dict(item)
            for item in batch_request.get("units") or []
            if isinstance(item, Mapping)
            and str(item.get("unit_id") or "") == unit_id
        ),
        None,
    )
    failed_unit = next(
        (
            dict(item)
            for item in candidate.get("units") or []
            if isinstance(item, Mapping)
            and str(item.get("unit_id") or "") == unit_id
        ),
        None,
    )
    if source_unit is None or failed_unit is None:
        raise MultivariantTranslationError(
            f"Targeted identity repair could not resolve failed unit {unit_id}"
        )

    requirements = identity_requirements_for_unit(
        source_unit,
        target_locale=str(batch_request.get("target_locale") or ""),
    )
    missing_requirements = [
        item
        for item in requirements
        if str(item.get("required_span") or "") in issue["validation_error"]
    ]
    if not missing_requirements:
        missing_requirements = [
            item for item in requirements if item.get("policy") == "hard_identity"
        ]
    if not missing_requirements:
        raise MultivariantTranslationError(
            f"Targeted identity repair found no hard identity contract for {unit_id}"
        )

    request = copy.deepcopy(dict(batch_request))
    request["units"] = [source_unit]
    translation_batch = dict(request.get("translation_batch") or {})
    translation_batch.update(
        {
            "unit_count": 1,
            "requested_unit_ids": [unit_id],
            "prior_translated_context": list(
                translation_batch.get("prior_translated_context") or []
            ),
            "instructions": [
                *list(translation_batch.get("instructions") or []),
                (
                    "This is a targeted validation repair. Return only the single "
                    "top-level source unit and do not return context-only units."
                ),
                (
                    f"Variant {variant_id} was the first variant rejected by local "
                    "validation. Repair the single unit, preserving any sibling "
                    "variant that already satisfies the identity contract."
                ),
                (
                    "Every natural, concise and compact variant in the repaired unit "
                    "must contain at least one exact accepted form for every missing "
                    "hard identity."
                ),
            ],
        }
    )
    request["translation_batch"] = translation_batch
    request["targeted_identity_repair"] = {
        "schema_version": _TARGETED_IDENTITY_REPAIR_SCHEMA_VERSION,
        "reason": "protected_identity_validation_failure",
        "unit_id": unit_id,
        "variant_id": variant_id,
        "validation_error": issue["validation_error"],
        "missing_identity_requirements": missing_requirements,
        "all_unit_identity_requirements": requirements,
        "failed_unit": failed_unit,
        "mutation_scope": {
            "allowed_unit_ids": [unit_id],
            "first_reported_variant_id": variant_id,
            "repair_all_invalid_variants_in_unit": True,
            "preserve_already_valid_variants": True,
        },
    }
    request["request_sha256"] = sha256_json(
        {key: value for key, value in request.items() if key != "request_sha256"}
    )
    return request


def _merge_targeted_identity_unit(
    *,
    batch_request: Mapping[str, Any],
    candidate: Mapping[str, Any],
    repaired_response: Mapping[str, Any],
    issue: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    unit_id = issue["unit_id"]
    repaired_units = [
        item
        for item in repaired_response.get("units") or []
        if isinstance(item, Mapping)
        and str(item.get("unit_id") or "") == unit_id
    ]
    if len(repaired_units) != 1:
        raise MultivariantTranslationError(
            f"Targeted identity repair returned no unique unit {unit_id}"
        )
    merged = copy.deepcopy(dict(candidate))
    replaced = False
    for index, target_unit in enumerate(merged.get("units") or []):
        if (
            isinstance(target_unit, Mapping)
            and str(target_unit.get("unit_id") or "") == unit_id
        ):
            repaired_unit = copy.deepcopy(dict(repaired_units[0]))
            warnings = list(repaired_unit.get("warnings") or [])
            warnings.append(
                {
                    "code": "targeted_protected_identity_unit_repair",
                    "first_reported_variant_id": issue["variant_id"],
                    "validation_error": issue["validation_error"],
                    "replacement_policy": (
                        "replace_only_failed_unit_from_single_unit_repair"
                    ),
                }
            )
            repaired_unit["warnings"] = warnings
            merged["units"][index] = repaired_unit
            replaced = True
            break
    if not replaced:
        raise MultivariantTranslationError(
            f"Saved chunk candidate contains no unique unit {unit_id}"
        )
    normalized = normalize_multivariant_response(merged, request=batch_request)
    validation = validate_multivariant_response(
        normalized,
        request=batch_request,
    )
    return normalized, validation


def _sum_usage(*items: Mapping[str, Any]) -> dict[str, int]:
    fields = (
        "input_tokens",
        "output_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
    )
    return {
        field: sum(
            int(
                (
                    item.get("token_usage")
                    if isinstance(item.get("token_usage"), Mapping)
                    else {}
                ).get(field)
                or 0
            )
            for item in items
        )
        for field in fields
    }


def _attempt_targeted_identity_repair(
    *,
    provider: Any,
    chunk_dir: Path,
    batch_request: Mapping[str, Any],
    original_error: BaseException,
    emit: Callable[..., None],
    chunk_index: int,
    chunk_count: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None:
    details = _exception_details(original_error)
    issue = _missing_identity_issue(original_error)
    candidate = details.get("candidate_output")
    if issue is None or not isinstance(candidate, Mapping):
        return None

    repair_dir = _next_repair_directory(chunk_dir)
    repair_request = _targeted_identity_repair_request(
        batch_request=batch_request,
        candidate=candidate,
        issue=issue,
    )
    atomic_write_json(repair_dir / "repair_request.json", repair_request)
    atomic_write_json(repair_dir / "failed_candidate.json", candidate)
    original_metadata = {
        key: details.get(key)
        for key in (
            "provider",
            "model_requested",
            "model_returned",
            "prompt_version",
            "prompt_hash",
            "request_timestamp",
            "completion_timestamp",
            "token_usage",
            "provider_attempts",
            "stop_reason",
        )
        if details.get(key) is not None
    }
    original_request_metrics = [
        dict(item)
        for item in details.get("request_metrics") or []
        if isinstance(item, Mapping)
    ]
    if original_request_metrics:
        original_metadata["request_summary"] = {
            "wire_requests": original_request_metrics
        }
    try:
        normalized, validation, local_repairs = (
            restore_missing_hard_identities_in_unit(
                candidate,
                request=batch_request,
                unit_id=issue["unit_id"],
            )
        )
        metadata = {
            **original_metadata,
            "provider": str(
                original_metadata.get("provider") or "azure-foundry-claude"
            ),
            "targeted_identity_repair_count": 1,
            "local_identity_alias_repair_count": 1,
            "targeted_identity_repair": {
                "schema_version": _TARGETED_IDENTITY_REPAIR_SCHEMA_VERSION,
                "unit_id": issue["unit_id"],
                "variant_id": issue["variant_id"],
                "repair_directory": str(repair_dir),
                "initial_validation_error": issue["validation_error"],
                "provider_http_requests_started": 0,
                "repair_mode": "deterministic_reviewed_alias_insertion",
            },
        }
        atomic_write_json(repair_dir / "merged_response.json", normalized)
        atomic_write_json(repair_dir / "validation.json", validation)
        atomic_write_json(repair_dir / "provider_metadata.json", metadata)
        atomic_write_json(
            repair_dir / "local_repairs.json",
            {
                "schema_version": _TARGETED_IDENTITY_REPAIR_SCHEMA_VERSION,
                "repairs": local_repairs,
            },
        )
        atomic_write_json(
            repair_dir / "completed.json",
            {
                "schema_version": _TARGETED_IDENTITY_REPAIR_SCHEMA_VERSION,
                "completed_at": _utcnow(),
                "unit_id": issue["unit_id"],
                "variant_id": issue["variant_id"],
                "request_sha256": repair_request["request_sha256"],
                "response_sha256": validation.get("response_sha256"),
                "provider_request_started": False,
                "repair_mode": "deterministic_reviewed_alias_insertion",
            },
        )
        emit(
            "identity-repair",
            (
                f"Chunk {chunk_index}/{chunk_count}: restored the reviewed "
                f"identity alias locally in {issue['unit_id']} and revalidated "
                "the complete chunk"
            ),
            status="completed",
            current=chunk_index,
            total=chunk_count,
            chunk_index=chunk_index,
            unit_id=issue["unit_id"],
            variant_id=issue["variant_id"],
            provider_request_started=False,
        )
        return normalized, metadata, validation
    except MultivariantTranslationError as local_repair_error:
        atomic_write_json(
            repair_dir / "local_repair_failure.json",
            {
                "schema_version": _TARGETED_IDENTITY_REPAIR_SCHEMA_VERSION,
                "failed_at": _utcnow(),
                "error_type": type(local_repair_error).__name__,
                "message": str(local_repair_error)[:500],
                "provider_fallback_required": True,
            },
        )
    emit(
        "identity-repair",
        (
            f"Chunk {chunk_index}/{chunk_count}: repairing only "
            f"{issue['unit_id']} (first rejected variant: {issue['variant_id']})"
        ),
        status="running",
        current=chunk_index - 1,
        total=chunk_count,
        chunk_index=chunk_index,
        unit_id=issue["unit_id"],
        variant_id=issue["variant_id"],
    )
    try:
        repaired = provider.translate_multivariant(repair_request)
        repaired_data = dict(repaired.data)
        normalized, validation = _merge_targeted_identity_unit(
            batch_request=batch_request,
            candidate=candidate,
            repaired_response=repaired_data,
            issue=issue,
        )
        repair_metadata = _provider_metadata(repaired)
        repair_request_summary = (
            repair_metadata.get("request_summary")
            if isinstance(repair_metadata.get("request_summary"), Mapping)
            else {}
        )
        repair_wire_requests = [
            dict(item)
            for item in repair_request_summary.get("wire_requests") or []
            if isinstance(item, Mapping)
        ]
        metadata = {
            **original_metadata,
            "provider": str(
                repair_metadata.get("provider")
                or original_metadata.get("provider")
                or "azure-foundry-claude"
            ),
            "model_returned": str(
                repair_metadata.get("model_returned")
                or original_metadata.get("model_returned")
                or ""
            ),
            "token_usage": _sum_usage(original_metadata, repair_metadata),
            "provider_attempts": int(
                original_metadata.get("provider_attempts") or 0
            )
            + int(repair_metadata.get("provider_attempts") or 0),
            "request_summary": {
                "wire_requests": [
                    *original_request_metrics,
                    *repair_wire_requests,
                ]
            },
            "targeted_identity_repair_count": 1,
            "targeted_identity_repair": {
                "schema_version": _TARGETED_IDENTITY_REPAIR_SCHEMA_VERSION,
                "unit_id": issue["unit_id"],
                "variant_id": issue["variant_id"],
                "repair_directory": str(repair_dir),
                "initial_validation_error": issue["validation_error"],
                "provider_http_requests_started": 1,
            },
        }
        atomic_write_json(repair_dir / "repair_response.json", repaired_data)
        atomic_write_json(repair_dir / "merged_response.json", normalized)
        atomic_write_json(repair_dir / "validation.json", validation)
        atomic_write_json(repair_dir / "provider_metadata.json", metadata)
        atomic_write_json(
            repair_dir / "completed.json",
            {
                "schema_version": _TARGETED_IDENTITY_REPAIR_SCHEMA_VERSION,
                "completed_at": _utcnow(),
                "unit_id": issue["unit_id"],
                "variant_id": issue["variant_id"],
                "request_sha256": repair_request["request_sha256"],
                "response_sha256": validation.get("response_sha256"),
            },
        )
        emit(
            "identity-repair",
            (
                f"Chunk {chunk_index}/{chunk_count}: repaired "
                f"{issue['unit_id']} variant {issue['variant_id']} and revalidated "
                "the complete chunk"
            ),
            status="completed",
            current=chunk_index,
            total=chunk_count,
            chunk_index=chunk_index,
            unit_id=issue["unit_id"],
            variant_id=issue["variant_id"],
        )
        return normalized, metadata, validation
    except BaseException as repair_error:
        atomic_write_json(
            repair_dir / "failure.json",
            {
                "schema_version": _TARGETED_IDENTITY_REPAIR_SCHEMA_VERSION,
                "failed_at": _utcnow(),
                "unit_id": issue["unit_id"],
                "variant_id": issue["variant_id"],
                "request_sha256": repair_request["request_sha256"],
                "initial_error": {
                    "error_type": type(original_error).__name__,
                    "message": str(original_error)[:500],
                },
                "repair_error": {
                    "error_type": type(repair_error).__name__,
                    "message": str(repair_error)[:500],
                },
            },
        )
        original_details = getattr(original_error, "details", None)
        if isinstance(original_details, dict):
            original_details["targeted_identity_repair_failed"] = {
                "error_type": type(repair_error).__name__,
                "message": str(repair_error)[:500],
                "repair_directory": str(repair_dir),
            }
        raise original_error from repair_error


def _targeted_validation_repair_request(
    *,
    batch_request: Mapping[str, Any],
    candidate: Mapping[str, Any],
    issue: Mapping[str, str],
) -> dict[str, Any]:
    """Build a one-unit request for a recoverable local-validator rejection."""

    unit_id = issue["unit_id"]
    source_unit = next(
        (
            dict(item)
            for item in batch_request.get("units") or []
            if isinstance(item, Mapping)
            and str(item.get("unit_id") or "") == unit_id
        ),
        None,
    )
    failed_unit = next(
        (
            dict(item)
            for item in candidate.get("units") or []
            if isinstance(item, Mapping)
            and str(item.get("unit_id") or "") == unit_id
        ),
        None,
    )
    if source_unit is None or failed_unit is None:
        raise MultivariantTranslationError(
            f"Targeted validation repair could not resolve failed unit {unit_id}"
        )

    request = copy.deepcopy(dict(batch_request))
    request["units"] = [source_unit]
    translation_batch = dict(request.get("translation_batch") or {})
    variant_id = issue.get("variant_id") or "unit"
    translation_batch.update(
        {
            "unit_count": 1,
            "requested_unit_ids": [unit_id],
            "instructions": [
                *list(translation_batch.get("instructions") or []),
                (
                    "This is a targeted validator repair. Return exactly the "
                    "single requested source unit and all three variants."
                ),
                (
                    f"The local validator rejected {unit_id} at {variant_id}: "
                    f"{issue['validation_error']}"
                ),
                (
                    "Use the failed unit as the starting point. Change only what "
                    "is necessary to satisfy the reported validation contract. "
                    "Preserve immutable source fields, required facts, protected "
                    "identities, attribution, uncertainty and negation."
                ),
            ],
        }
    )
    request["translation_batch"] = translation_batch
    request["targeted_validation_repair"] = {
        "schema_version": _TARGETED_VALIDATION_REPAIR_SCHEMA_VERSION,
        "reason": "unit_level_validator_rejection",
        "unit_id": unit_id,
        "variant_id": issue.get("variant_id") or None,
        "validation_error": issue["validation_error"],
        "failed_unit": failed_unit,
        "mutation_scope": {
            "allowed_unit_ids": [unit_id],
            "return_all_variants": True,
            "replace_only_failed_unit": True,
        },
    }
    request["request_sha256"] = sha256_json(
        {key: value for key, value in request.items() if key != "request_sha256"}
    )
    return request


def _merge_targeted_validation_unit(
    *,
    batch_request: Mapping[str, Any],
    candidate: Mapping[str, Any],
    repaired_response: Mapping[str, Any],
    issue: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    unit_id = issue["unit_id"]
    repaired_units = [
        item
        for item in repaired_response.get("units") or []
        if isinstance(item, Mapping)
        and str(item.get("unit_id") or "") == unit_id
    ]
    if len(repaired_units) != 1:
        raise MultivariantTranslationError(
            f"Targeted validation repair returned no unique unit {unit_id}"
        )

    merged = copy.deepcopy(dict(candidate))
    replaced = False
    for index, target_unit in enumerate(merged.get("units") or []):
        if (
            isinstance(target_unit, Mapping)
            and str(target_unit.get("unit_id") or "") == unit_id
        ):
            repaired_unit = copy.deepcopy(dict(repaired_units[0]))
            repaired_unit["warnings"] = [
                *list(repaired_unit.get("warnings") or []),
                {
                    "code": "targeted_unit_validation_repair",
                    "first_reported_variant_id": issue.get("variant_id") or None,
                    "validation_error": issue["validation_error"],
                    "replacement_policy": (
                        "replace_only_rejected_unit_from_single_unit_repair"
                    ),
                },
            ]
            merged["units"][index] = repaired_unit
            replaced = True
            break
    if not replaced:
        raise MultivariantTranslationError(
            f"Saved chunk candidate contains no unique unit {unit_id}"
        )

    normalized = normalize_multivariant_response(merged, request=batch_request)
    validation = validate_multivariant_response(
        normalized,
        request=batch_request,
    )
    return normalized, validation


def _attempt_targeted_validation_repair(
    *,
    provider: Any,
    chunk_dir: Path,
    batch_request: Mapping[str, Any],
    original_error: BaseException,
    emit: Callable[..., None],
    chunk_index: int,
    chunk_count: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None:
    """Repair one rejected unit instead of repeating or terminating its chunk."""

    if _missing_identity_issue(original_error) is not None:
        return None
    details = _exception_details(original_error)
    issue = _unit_validation_issue(original_error)
    candidate = details.get("candidate_output")
    if issue is None or not isinstance(candidate, Mapping):
        return None

    repair_dir = _next_validation_repair_directory(chunk_dir)
    repair_request = _targeted_validation_repair_request(
        batch_request=batch_request,
        candidate=candidate,
        issue=issue,
    )
    atomic_write_json(repair_dir / "repair_request.json", repair_request)
    atomic_write_json(repair_dir / "failed_candidate.json", candidate)
    emit(
        "validation-repair",
        (
            f"Chunk {chunk_index}/{chunk_count}: repairing only "
            f"{issue['unit_id']} after a local validator rejection"
        ),
        status="running",
        current=chunk_index - 1,
        total=chunk_count,
        chunk_index=chunk_index,
        unit_id=issue["unit_id"],
        variant_id=issue.get("variant_id") or None,
    )

    original_metadata = {
        key: details.get(key)
        for key in (
            "provider",
            "model_requested",
            "model_returned",
            "prompt_version",
            "prompt_hash",
            "request_timestamp",
            "completion_timestamp",
            "token_usage",
            "provider_attempts",
            "stop_reason",
        )
        if details.get(key) is not None
    }
    original_request_metrics = [
        dict(item)
        for item in details.get("request_metrics") or []
        if isinstance(item, Mapping)
    ]
    try:
        repaired = provider.translate_multivariant(repair_request)
        repaired_data = dict(repaired.data)
        normalized, validation = _merge_targeted_validation_unit(
            batch_request=batch_request,
            candidate=candidate,
            repaired_response=repaired_data,
            issue=issue,
        )
        repair_metadata = _provider_metadata(repaired)
        repair_request_summary = (
            repair_metadata.get("request_summary")
            if isinstance(repair_metadata.get("request_summary"), Mapping)
            else {}
        )
        repair_wire_requests = [
            dict(item)
            for item in repair_request_summary.get("wire_requests") or []
            if isinstance(item, Mapping)
        ]
        metadata = {
            **original_metadata,
            "provider": str(
                repair_metadata.get("provider")
                or original_metadata.get("provider")
                or "azure-foundry-claude"
            ),
            "model_returned": str(
                repair_metadata.get("model_returned")
                or original_metadata.get("model_returned")
                or ""
            ),
            "token_usage": _sum_usage(original_metadata, repair_metadata),
            "provider_attempts": int(
                original_metadata.get("provider_attempts") or 0
            )
            + int(repair_metadata.get("provider_attempts") or 0),
            "request_summary": {
                "wire_requests": [
                    *original_request_metrics,
                    *repair_wire_requests,
                ]
            },
            "targeted_validation_repair_count": 1,
            "targeted_validation_repair": {
                "schema_version": _TARGETED_VALIDATION_REPAIR_SCHEMA_VERSION,
                "unit_id": issue["unit_id"],
                "variant_id": issue.get("variant_id") or None,
                "repair_directory": str(repair_dir),
                "initial_validation_error": issue["validation_error"],
                "provider_http_requests_started": 1,
            },
        }
        atomic_write_json(repair_dir / "repair_response.json", repaired_data)
        atomic_write_json(repair_dir / "merged_response.json", normalized)
        atomic_write_json(repair_dir / "validation.json", validation)
        atomic_write_json(repair_dir / "provider_metadata.json", metadata)
        atomic_write_json(
            repair_dir / "completed.json",
            {
                "schema_version": _TARGETED_VALIDATION_REPAIR_SCHEMA_VERSION,
                "completed_at": _utcnow(),
                "unit_id": issue["unit_id"],
                "variant_id": issue.get("variant_id") or None,
                "request_sha256": repair_request["request_sha256"],
                "response_sha256": validation.get("response_sha256"),
            },
        )
        emit(
            "validation-repair",
            (
                f"Chunk {chunk_index}/{chunk_count}: repaired "
                f"{issue['unit_id']} and revalidated the complete chunk"
            ),
            status="completed",
            current=chunk_index,
            total=chunk_count,
            chunk_index=chunk_index,
            unit_id=issue["unit_id"],
            variant_id=issue.get("variant_id") or None,
        )
        return normalized, metadata, validation
    except BaseException as repair_error:
        atomic_write_json(
            repair_dir / "failure.json",
            {
                "schema_version": _TARGETED_VALIDATION_REPAIR_SCHEMA_VERSION,
                "failed_at": _utcnow(),
                "unit_id": issue["unit_id"],
                "variant_id": issue.get("variant_id") or None,
                "request_sha256": repair_request["request_sha256"],
                "initial_error": {
                    "error_type": type(original_error).__name__,
                    "message": str(original_error)[:500],
                },
                "repair_error": {
                    "error_type": type(repair_error).__name__,
                    "message": str(repair_error)[:500],
                },
            },
        )
        original_details = getattr(original_error, "details", None)
        if isinstance(original_details, dict):
            original_details["targeted_validation_repair_failed"] = {
                "error_type": type(repair_error).__name__,
                "message": str(repair_error)[:500],
                "repair_directory": str(repair_dir),
            }
        raise original_error from repair_error


def _chunk_split_recovery_reason(exc: BaseException) -> str | None:
    details = getattr(exc, "details", None)
    details = details if isinstance(details, Mapping) else {}
    stop_reason = str(details.get("stop_reason") or "").strip()
    if stop_reason in {"max_tokens", "model_context_window_exceeded"}:
        return stop_reason
    if int(details.get("status_code") or 0) == 413:
        return "request_too_large"
    message = str(exc).casefold()
    if "request too large" in message or "request_too_large" in message:
        return "request_too_large"
    if "context window" in message:
        return "model_context_window_exceeded"
    return None


def _split_child_request(
    request: Mapping[str, Any],
    *,
    units: list[Mapping[str, Any]],
    part_index: int,
    part_count: int,
    depth: int,
) -> dict[str, Any]:
    child = copy.deepcopy(dict(request))
    child["units"] = [copy.deepcopy(dict(item)) for item in units]
    unit_ids = [str(item.get("unit_id") or "") for item in units]
    batch = dict(child.get("translation_batch") or {})
    batch["requested_unit_ids"] = unit_ids
    batch["unit_count"] = len(units)
    batch["estimated_source_tokens"] = sum(
        estimate_unit_tokens(item) for item in units
    )
    original_limit = int(batch.get("max_output_tokens") or 12_000)
    batch["max_output_tokens"] = min(
        original_limit,
        max(6_000, 2_200 + len(units) * 430),
    )
    batch["split_recovery"] = {
        "schema_version": _SPLIT_RECOVERY_SCHEMA_VERSION,
        "depth": depth,
        "part_index": part_index,
        "part_count": part_count,
        "parent_request_sha256": str(request.get("request_sha256") or ""),
    }
    selected = set(unit_ids)
    context = []
    for item in batch.get("context_units") or []:
        if not isinstance(item, Mapping):
            continue
        context_item = dict(item)
        context_item["translate_this_unit"] = (
            str(context_item.get("unit_id") or "") in selected
        )
        context.append(context_item)
    if context:
        batch["context_units"] = context
    child["translation_batch"] = batch
    child["request_sha256"] = sha256_json(
        {key: value for key, value in child.items() if key != "request_sha256"}
    )
    return child


def _failure_metadata(exc: BaseException, reason: str) -> dict[str, Any]:
    details = getattr(exc, "details", None)
    details = details if isinstance(details, Mapping) else {}
    return {
        "provider": str(details.get("provider") or "azure-foundry-claude"),
        "model_returned": str(details.get("model_returned") or ""),
        "prompt_version": str(details.get("prompt_version") or PROMPT_VERSION),
        "provider_attempts": int(details.get("provider_attempts") or 0),
        "token_usage": dict(
            details.get("token_usage")
            if isinstance(details.get("token_usage"), Mapping)
            else {}
        ),
        "status": "failed_recovered_by_split",
        "split_recovery_reason": reason,
    }


def _attempt_chunk_split_recovery(
    *,
    provider: Any,
    chunk_dir: Path,
    batch_request: Mapping[str, Any],
    original_error: BaseException,
    emit: Callable[..., None],
    chunk_index: int,
    chunk_count: int,
    depth: int = 1,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None:
    """Serially split only a transport-truncated chunk and merge it back."""

    reason = _chunk_split_recovery_reason(original_error)
    units = [
        dict(item)
        for item in batch_request.get("units") or []
        if isinstance(item, Mapping)
    ]
    if reason is None or len(units) < 2:
        return None

    midpoint = max(1, len(units) // 2)
    groups = [units[:midpoint], units[midpoint:]]
    recovery_root = chunk_dir / "split_recovery" / f"depth_{depth:02d}"
    recovery_root.mkdir(parents=True, exist_ok=True)
    emit(
        "chunk-split",
        (
            f"Chunk {chunk_index}/{chunk_count}: {reason}; retrying only this "
            f"chunk as {len(groups)} smaller serial parts"
        ),
        status="warning",
        current=chunk_index - 1,
        total=chunk_count,
        chunk_index=chunk_index,
        reason=reason,
        depth=depth,
    )

    responses: list[dict[str, Any]] = []
    metadata_items: list[dict[str, Any]] = [_failure_metadata(original_error, reason)]
    for part_index, group in enumerate(groups, start=1):
        child_request = _split_child_request(
            batch_request,
            units=group,
            part_index=part_index,
            part_count=len(groups),
            depth=depth,
        )
        part_dir = recovery_root / f"part_{part_index:02d}"
        part_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(part_dir / "translation_request.json", child_request)
        checkpoint = _load_checkpoint(
            chunk_dir=part_dir,
            request=child_request,
        )
        if checkpoint is not None:
            response_data, metadata = checkpoint
            validation = validate_multivariant_response(
                response_data,
                request=child_request,
            )
        else:
            try:
                provider_response = provider.translate_multivariant(child_request)
                response_data = dict(provider_response.data)
                validation = validate_multivariant_response(
                    response_data,
                    request=child_request,
                )
                metadata = _provider_metadata(provider_response)
            except Exception as child_error:
                repaired = _attempt_targeted_identity_repair(
                    provider=provider,
                    chunk_dir=part_dir,
                    batch_request=child_request,
                    original_error=child_error,
                    emit=emit,
                    chunk_index=chunk_index,
                    chunk_count=chunk_count,
                )
                if repaired is None:
                    repaired = _attempt_targeted_validation_repair(
                        provider=provider,
                        chunk_dir=part_dir,
                        batch_request=child_request,
                        original_error=child_error,
                        emit=emit,
                        chunk_index=chunk_index,
                        chunk_count=chunk_count,
                    )
                if repaired is None:
                    repaired = _attempt_chunk_split_recovery(
                        provider=provider,
                        chunk_dir=part_dir,
                        batch_request=child_request,
                        original_error=child_error,
                        emit=emit,
                        chunk_index=chunk_index,
                        chunk_count=chunk_count,
                        depth=depth + 1,
                    )
                if repaired is None:
                    raise
                response_data, metadata, validation = repaired

            atomic_write_json(part_dir / "translation_response.json", response_data)
            atomic_write_json(part_dir / "provider_metadata.json", metadata)
            atomic_write_json(part_dir / "validation.json", validation)
            atomic_write_json(
                part_dir / "completed.json",
                {
                    "schema_version": _SPLIT_RECOVERY_SCHEMA_VERSION,
                    "completed_at": _utcnow(),
                    "request_sha256": child_request["request_sha256"],
                    "response_sha256": validation.get("response_sha256"),
                    "unit_ids": (
                        child_request.get("translation_batch") or {}
                    ).get("requested_unit_ids"),
                },
            )
        responses.append(response_data)
        metadata_items.append(metadata)

    merged = merge_multivariant_batch_responses(
        request=batch_request,
        responses=responses,
    )
    validation = validate_multivariant_response(merged, request=batch_request)
    metadata = _aggregate_metadata(
        metadata_items,
        split_chunk_recovery_count=1
        + sum(
            int(item.get("split_chunk_recovery_count") or 0)
            for item in metadata_items
        ),
        split_recovery_reason=reason,
        split_part_count=len(groups),
    )
    atomic_write_json(
        recovery_root / "completed.json",
        {
            "schema_version": _SPLIT_RECOVERY_SCHEMA_VERSION,
            "completed_at": _utcnow(),
            "reason": reason,
            "depth": depth,
            "parent_request_sha256": batch_request.get("request_sha256"),
            "response_sha256": validation.get("response_sha256"),
            "part_count": len(groups),
        },
    )
    emit(
        "chunk-split",
        (
            f"Chunk {chunk_index}/{chunk_count}: smaller serial parts merged "
            "and the original chunk revalidated"
        ),
        status="completed",
        current=chunk_index,
        total=chunk_count,
        chunk_index=chunk_index,
        reason=reason,
        depth=depth,
    )
    return merged, metadata, validation


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
        else os.getenv("MATHULA_TV_TRANSLATION_CHUNK_TARGET_TOKENS", "1800")
    )
    hard_tokens = int(
        chunk_hard_tokens
        if chunk_hard_tokens is not None
        else os.getenv("MATHULA_TV_TRANSLATION_CHUNK_HARD_TOKENS", "2600")
    )
    configured_spine_tokens = os.getenv(
        "MATHULA_TV_TRANSLATION_CONTEXT_SPINE_TOKENS"
    )
    spine_tokens = (
        int(context_spine_tokens)
        if context_spine_tokens is not None
        else (
            int(configured_spine_tokens)
            if configured_spine_tokens not in (None, "")
            else None
        )
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
        os.getenv("MATHULA_TV_TRANSLATION_CHUNK_MAX_OUTPUT_TOKENS", "32000")
    )
    max_units = int(
        batch_size
        if batch_size is not None
        else os.getenv("MATHULA_TV_TRANSLATION_MAX_UNITS_PER_CHUNK", "18")
    )
    if target_tokens <= 0 or hard_tokens < target_tokens:
        raise ValueError("translation chunk token budgets are invalid")
    if spine_tokens is not None and spine_tokens < 1000:
        raise ValueError("translation context spine token budget must be at least 1000")
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

    retrying_failed_translation = (
        job.state == "failed_retryable"
        and str((job.last_error or {}).get("stage") or "")
        in {_STAGE, "translation"}
    )
    if job.state not in {"analysis_ready", "translation_running"} and not retrying_failed_translation:
        raise ValueError(
            "Claude translation requires analysis_ready, translation_running, or "
            f"a retryable translation failure, not {job.state}; "
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

    resumable_manifest = (
        _resumable_session_manifest(
            job_root=paths.job_root,
            request_sha256=str(request["request_sha256"]),
        )
        if spine_tokens is None and not restart_batches
        else None
    )
    resumed_existing_session = resumable_manifest is not None
    if resumable_manifest is not None:
        previous_spine_tokens = int(
            resumable_manifest.get("context_spine_tokens") or 0
        )
        if previous_spine_tokens >= 1000:
            spine_tokens = previous_spine_tokens
    adaptive_spine = spine_tokens is None
    if spine_tokens is None:
        spine_tokens = adaptive_context_spine_tokens(request)
    continuity_payload_policy = (
        str(resumable_manifest.get("continuity_payload_policy") or "")
        if resumable_manifest is not None
        else ""
    ) or (
        "legacy_full_three_variant_tail_v1"
        if resumed_existing_session
        else "relevant_natural_tail_v1"
    )

    source_duration_seconds = (
        max(
            (int(item.get("end_ms") or 0) for item in request.get("units") or []),
            default=0,
        )
        / 1000.0
    )

    def record_chunk_consumption(
        metadata: Mapping[str, Any],
        *,
        chunk_index: int,
        status: str = "completed",
    ) -> None:
        try:
            report = update_ai_consumption_report(
                job_root=paths.job_root,
                records=[
                    consumption_record(
                        operation="multivariant_translation",
                        metadata=metadata,
                        scope=f"chunk_{chunk_index:04d}",
                        status=status,
                    )
                ],
                video_duration_seconds=source_duration_seconds,
            )
            summary = report.get("summary") or {}
            emit(
                "consumption",
                (
                    "AI consumption report updated: "
                    f"input={int(summary.get('input_tokens') or 0):,}, "
                    f"output={int(summary.get('output_tokens') or 0):,}, "
                    f"cache-read={int(summary.get('cache_read_input_tokens') or 0):,}"
                ),
                current=chunk_index,
                total=len(batches),
                chunk_index=chunk_index,
                report_path=str(
                    paths.job_root / "analysis" / "ai_consumption_report.json"
                ),
            )
        except Exception:
            # Consumption accounting must never invalidate a paid, validated
            # translation result or block checkpoint recovery.
            return

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
            "context_spine_policy": (
                "resumed_existing_session"
                if resumed_existing_session
                else (
                    "adaptive_transcript_complexity"
                    if adaptive_spine
                    else "explicit_fixed_budget"
                )
            ),
            "continuity_payload_policy": continuity_payload_policy,
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
        context_spine_tokens=spine_tokens,
        context_spine_policy=(
            "resumed_existing_session"
            if resumed_existing_session
            else (
                "adaptive_transcript_complexity"
                if adaptive_spine
                else "explicit_fixed_budget"
            )
        ),
        continuity_payload_policy=continuity_payload_policy,
    )

    if job.state == "analysis_ready":
        job.transition("translation_running")
    elif retrying_failed_translation:
        job.transition("translation_running")
        job.last_error = None
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
            if continuity_payload_policy == "legacy_full_three_variant_tail_v1":
                translation_batch["prior_translated_context"] = (
                    _prior_context_legacy(responses)
                )
                translation_batch["prior_global_terminology"] = (
                    _prior_terminology_legacy(responses)
                )
            else:
                translation_batch["prior_translated_context"] = _prior_context(
                    responses
                )
                translation_batch["prior_global_terminology"] = _prior_terminology(
                    responses,
                    batch_request,
                )
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
                record_chunk_consumption(
                    metadata,
                    chunk_index=chunk_index,
                )
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

            recovered = _recover_failed_candidate(
                chunk_dir=chunk_dir,
                request=batch_request,
            )
            if recovered is not None:
                response_data, metadata, validation = recovered
                metadata.update(
                    {
                        "chunk_index": chunk_index,
                        "chunk_count": chunk_count,
                        "request_sha256": batch_request["request_sha256"],
                        "estimated_source_tokens": translation_batch.get(
                            "estimated_source_tokens"
                        ),
                        "max_output_tokens": translation_batch.get("max_output_tokens"),
                        "context_spine_sha256": translation_batch.get(
                            "context_spine_sha256"
                        ),
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
                        "recovered_from_failed_candidate": True,
                    },
                )
                atomic_write_json(
                    chunk_dir / "recovery.json",
                    {
                        "schema_version": "mathula.translation.semantic-chunk-recovery.v1",
                        "recovered_at": _utcnow(),
                        "request_sha256": batch_request["request_sha256"],
                        "source_failure": str(chunk_dir / "failure.json"),
                        "provider_request_started": False,
                    },
                )
                responses.append(response_data)
                metadata_items.append(metadata)
                record_chunk_consumption(
                    metadata,
                    chunk_index=chunk_index,
                )
                emit(
                    "chunk",
                    (
                        f"Chunk {chunk_index}/{chunk_count} recovered locally from "
                        "the saved provider candidate and checkpointed"
                    ),
                    status="completed",
                    current=chunk_index,
                    total=chunk_count,
                    chunk_index=chunk_index,
                    checkpoint_reused=False,
                    failed_candidate_recovered=True,
                )
                continue

            saved_validation_failure = _load_saved_validation_failure(
                chunk_dir=chunk_dir,
                request=batch_request,
            )
            saved_split_failure = _load_saved_split_failure(
                chunk_dir=chunk_dir,
                request=batch_request,
            )
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
                elif kind == "request_metrics":
                    request_bytes = int(
                        event.get("request_body_bytes") or 0
                    )
                    payload_bytes = int(event.get("payload_bytes") or 0)
                    schema_bytes = int(
                        event.get("output_schema_bytes") or 0
                    )
                    estimated_tokens = int(
                        event.get("estimated_input_tokens") or 0
                    )
                    message = (
                        f"Chunk {chunk_index}/{chunk_count}: sending "
                        f"{request_bytes / 1024:.1f} KiB JSON "
                        f"(~{estimated_tokens:,} input tokens estimated); "
                        f"payload={payload_bytes / 1024:.1f} KiB, "
                        f"schema={schema_bytes / 1024:.1f} KiB, "
                        f"max_output_tokens={event.get('max_output_tokens')}, "
                        "structured-output="
                        f"{'native' if event.get('native_structured_outputs') else 'prompt+local'}"
                    )
                    status = "running"
                elif kind == "structured_output_schema":
                    message = (
                        f"Chunk {chunk_index}/{chunk_count}: Claude schema "
                        f"preflight passed ({int(event.get('schema_bytes') or 0) / 1024:.1f} KiB, "
                        f"{int(event.get('optional_parameters') or 0)} optional, "
                        f"{int(event.get('union_parameters') or 0)} unions)"
                    )
                    status = "running"
                elif kind == "request_capability_fallback":
                    capability = str(event.get("capability") or "request option")
                    labels = {
                        "foundry_structured_outputs": "native Structured Outputs",
                        "foundry_effort": "effort control",
                        "foundry_thinking": "thinking control",
                        "foundry_prompt_cache": "prompt caching",
                    }
                    message = (
                        f"Chunk {chunk_index}/{chunk_count}: deployment does not "
                        f"support {labels.get(capability, capability)}; retrying "
                        "without only that optional request feature"
                    )
                    status = "warning"
                elif kind == "response_usage":
                    response_bytes = int(
                        event.get("response_body_bytes") or 0
                    )
                    message = (
                        f"Chunk {chunk_index}/{chunk_count}: Claude usage "
                        f"input={int(event.get('input_tokens') or 0):,}, "
                        f"output={int(event.get('output_tokens') or 0):,}, "
                        f"thinking={int(event.get('thinking_tokens') or 0):,}, "
                        f"cache-read={int(event.get('cache_read_input_tokens') or 0):,}, "
                        f"cache-write={int(event.get('cache_creation_input_tokens') or 0):,}; "
                        f"response={response_bytes / 1024:.1f} KiB"
                    )
                    status = "completed"
                elif kind == "translation_wire_contract":
                    message = (
                        f"Chunk {chunk_index}/{chunk_count}: compact response enabled; "
                        f"provider max_output_tokens={event.get('max_output_tokens')}"
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
                elif kind == "stream_json_salvaged":
                    message = (
                        f"Chunk {chunk_index}/{chunk_count}: complete JSON salvaged "
                        "from an interrupted SSE stream; validating locally"
                    )
                    status = "warning"
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
                    (
                        f"Reusing the saved Chunk {chunk_index}/{chunk_count} "
                        + (
                            "transport failure and sending only smaller serial parts"
                            if saved_split_failure is not None
                            else "candidate and sending only its rejected unit"
                        )
                    )
                        if (
                            saved_validation_failure is not None
                            or saved_split_failure is not None
                        )
                        else (
                        f"Sending chunk {chunk_index}/{chunk_count}: "
                        f"{len(batch_request.get('units') or [])} units, "
                        f"~{translation_batch.get('estimated_source_tokens')} source tokens, "
                        f"max_output_tokens={translation_batch.get('max_output_tokens')}"
                    )
                ),
                current=chunk_index - 1,
                total=chunk_count,
                chunk_index=chunk_index,
            )
            try:
                if saved_split_failure is not None:
                    repaired = _attempt_chunk_split_recovery(
                        provider=batch_provider,
                        chunk_dir=chunk_dir,
                        batch_request=batch_request,
                        original_error=saved_split_failure,
                        emit=emit,
                        chunk_index=chunk_index,
                        chunk_count=chunk_count,
                    )
                    if repaired is None:
                        raise saved_split_failure
                    response_data, metadata, validation = repaired
                elif saved_validation_failure is not None:
                    repaired = _attempt_targeted_identity_repair(
                        provider=batch_provider,
                        chunk_dir=chunk_dir,
                        batch_request=batch_request,
                        original_error=saved_validation_failure,
                        emit=emit,
                        chunk_index=chunk_index,
                        chunk_count=chunk_count,
                    )
                    if repaired is None:
                        repaired = _attempt_targeted_validation_repair(
                            provider=batch_provider,
                            chunk_dir=chunk_dir,
                            batch_request=batch_request,
                            original_error=saved_validation_failure,
                            emit=emit,
                            chunk_index=chunk_index,
                            chunk_count=chunk_count,
                        )
                    if repaired is None:
                        raise saved_validation_failure
                    response_data, metadata, validation = repaired
                else:
                    try:
                        response = batch_provider.translate_multivariant(batch_request)
                        response_data = dict(response.data)
                        validation = validate_multivariant_response(
                            response_data,
                            request=batch_request,
                        )
                        metadata = _provider_metadata(response)
                    except BaseException as initial_error:
                        repaired = _attempt_targeted_identity_repair(
                            provider=batch_provider,
                            chunk_dir=chunk_dir,
                            batch_request=batch_request,
                            original_error=initial_error,
                            emit=emit,
                            chunk_index=chunk_index,
                            chunk_count=chunk_count,
                        )
                        if repaired is None:
                            repaired = _attempt_targeted_validation_repair(
                                provider=batch_provider,
                                chunk_dir=chunk_dir,
                                batch_request=batch_request,
                                original_error=initial_error,
                                emit=emit,
                                chunk_index=chunk_index,
                                chunk_count=chunk_count,
                            )
                        if repaired is None:
                            repaired = _attempt_chunk_split_recovery(
                                provider=batch_provider,
                                chunk_dir=chunk_dir,
                                batch_request=batch_request,
                                original_error=initial_error,
                                emit=emit,
                                chunk_index=chunk_index,
                                chunk_count=chunk_count,
                            )
                        if repaired is None:
                            raise
                        response_data, metadata, validation = repaired
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
                record_chunk_consumption(
                    metadata,
                    chunk_index=chunk_index,
                )
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
                failure_metadata = dict(_exception_details(exc))
                failure_metadata.update(
                    {
                        "request_sha256": batch_request["request_sha256"],
                        "chunk_index": chunk_index,
                        "provider": str(
                            failure_metadata.get("provider")
                            or "azure-foundry-claude"
                        ),
                    }
                )
                record_chunk_consumption(
                    failure_metadata,
                    chunk_index=chunk_index,
                    status="failed",
                )
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
            context_spine_policy=(
                "resumed_existing_session"
                if resumed_existing_session
                else (
                    "adaptive_transcript_complexity"
                    if adaptive_spine
                    else "explicit_fixed_budget"
                )
            ),
            continuity_payload_policy=continuity_payload_policy,
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
                "targeted_identity_repair_count": int(
                    aggregate_metadata.get("targeted_identity_repair_count") or 0
                ),
                "targeted_validation_repair_count": int(
                    aggregate_metadata.get("targeted_validation_repair_count")
                    or 0
                ),
                "split_chunk_recovery_count": int(
                    aggregate_metadata.get("split_chunk_recovery_count") or 0
                ),
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
                "split_chunk_recovery_count": int(
                    aggregate_metadata.get("split_chunk_recovery_count") or 0
                ),
                "translation_target": written["translation_target"],
                "dubbing_target": written["dubbing_target"],
            },
        )
        consumption_report_path = (
            paths.job_root / "analysis" / "ai_consumption_report.json"
        )
        if consumption_report_path.is_file():
            try:
                consumption_summary = (
                    read_json(consumption_report_path).get("summary") or {}
                )
                request_megabytes = (
                    int(consumption_summary.get("request_body_bytes") or 0)
                    / (1024 * 1024)
                )
                tokens_per_minute = consumption_summary.get(
                    "tokens_per_video_minute"
                )
                tokens_per_minute_text = (
                    f"{float(tokens_per_minute):,.0f}"
                    if isinstance(tokens_per_minute, (int, float))
                    else "n/a"
                )
                emit(
                    "consumption-summary",
                    (
                        "Cumulative AI job usage: "
                        f"input={int(consumption_summary.get('input_tokens') or 0):,}, "
                        f"output={int(consumption_summary.get('output_tokens') or 0):,}, "
                        f"cache-read={int(consumption_summary.get('cache_read_input_tokens') or 0):,}, "
                        f"wire={request_megabytes:.2f} MiB, "
                        f"tokens/video-minute={tokens_per_minute_text}"
                    ),
                    status="completed",
                    current=chunk_count,
                    total=chunk_count,
                    report_path=str(consumption_report_path),
                )
            except (OSError, TypeError, ValueError):
                pass
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
        safe_error["retryable"] = True
        safe_error["action"] = (
            "Rerun the same translate command; validated chunk checkpoints "
            "will be reused"
        )
        job.last_error = safe_error
        if job.state == "translation_running":
            job.transition("failed_retryable")
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
