"""Azure OpenAI Responses API web-search client for transcript name research.

This module is intentionally independent of Mathula TV's Claude Messages client.
Azure-hosted Claude deployments may reject Anthropic server-tool definitions even
when ordinary Messages calls work. Name research therefore uses the Azure OpenAI
Responses API and its native ``web_search`` tool, while translation and editorial
work remain on Claude.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import jsonschema
import requests


class AzureWebResearchError(RuntimeError):
    """Raised when Azure Responses web research cannot complete safely."""

    def __init__(
        self,
        message: str,
        *,
        diagnostic_paths: Sequence[str] = (),
    ) -> None:
        super().__init__(message)
        self.diagnostic_paths = tuple(str(path) for path in diagnostic_paths if path)


@dataclass(frozen=True)
class AzureWebResearchConfig:
    endpoint: str
    api_key: str
    deployment: str
    timeout_seconds: float = 900.0
    max_retries: int = 2
    max_output_tokens: int = 0
    web_search_tool: str = "auto"

    @classmethod
    def from_environment(
        cls, environ: Mapping[str, str] | None = None
    ) -> "AzureWebResearchConfig":
        env = os.environ if environ is None else environ
        endpoint = (
            env.get("AZURE_AI_ENDPOINT")
            or env.get("AZURE_OPENAI_ENDPOINT")
            or ""
        ).rstrip("/")
        api_key = (
            env.get("AZURE_AI_KEY")
            or env.get("AZURE_OPENAI_API_KEY")
            or ""
        )
        deployment = (
            env.get("MATHULA_TV_AUTOCORRECT_RESEARCH_DEPLOYMENT")
            or env.get("AZURE_AI_DEPLOYMENT")
            or env.get("AZURE_OPENAI_DEPLOYMENT")
            or ""
        )
        missing = [
            name
            for name, value in (
                ("AZURE_AI_ENDPOINT or AZURE_OPENAI_ENDPOINT", endpoint),
                ("AZURE_AI_KEY or AZURE_OPENAI_API_KEY", api_key),
                (
                    "MATHULA_TV_AUTOCORRECT_RESEARCH_DEPLOYMENT, "
                    "AZURE_AI_DEPLOYMENT, or AZURE_OPENAI_DEPLOYMENT",
                    deployment,
                ),
            )
            if not value
        ]
        if missing:
            raise AzureWebResearchError(
                "Missing Azure web-research configuration: " + ", ".join(missing)
            )
        tool = env.get("MATHULA_TV_AUTOCORRECT_WEB_SEARCH_TOOL", "auto").strip()
        if tool not in {"auto", "web_search", "web_search_preview"}:
            raise AzureWebResearchError(
                "MATHULA_TV_AUTOCORRECT_WEB_SEARCH_TOOL must be auto, "
                "web_search, or web_search_preview"
            )
        return cls(
            endpoint=endpoint,
            api_key=api_key,
            deployment=deployment,
            timeout_seconds=float(
                env.get("MATHULA_TV_AUTOCORRECT_RESEARCH_TIMEOUT_SECONDS", "900")
            ),
            max_retries=int(
                env.get("MATHULA_TV_AUTOCORRECT_RESEARCH_MAX_RETRIES", "2")
            ),
            max_output_tokens=int(
                env.get("MATHULA_TV_AUTOCORRECT_RESEARCH_MAX_OUTPUT_TOKENS", "0")
            ),
            web_search_tool=tool,
        )

    @property
    def url(self) -> str:
        base = self.endpoint.rstrip("/")
        if base.endswith("/openai/v1"):
            return f"{base}/responses"
        return f"{base}/openai/v1/responses"

    def safe_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("api_key", None)
        data["url"] = self.url
        return data


@dataclass(frozen=True)
class AzureWebResearchMetadata:
    provider: str
    model: str
    response_id: str | None
    status: str | None
    usage: Mapping[str, Any]
    web_search_tool: str
    server_tool_evidence: tuple[dict[str, Any], ...]
    request_attempts: int
    elapsed_seconds: float
    schema_repair_used: bool = False
    schema_repair_response_id: str | None = None
    schema_repair_elapsed_seconds: float = 0.0
    request_body_bytes: int = 0
    estimated_input_tokens: int = 0
    response_body_bytes: int = 0
    response_text_characters: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "response_id": self.response_id,
            "status": self.status,
            "usage": dict(self.usage),
            "web_search_tool": self.web_search_tool,
            "server_tool_evidence": [dict(item) for item in self.server_tool_evidence],
            "request_attempts": self.request_attempts,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "schema_repair_used": self.schema_repair_used,
            "schema_repair_response_id": self.schema_repair_response_id,
            "schema_repair_elapsed_seconds": round(
                self.schema_repair_elapsed_seconds, 3
            ),
            "request_body_bytes": self.request_body_bytes,
            "estimated_input_tokens": self.estimated_input_tokens,
            "response_body_bytes": self.response_body_bytes,
            "response_text_characters": self.response_text_characters,
        }


@dataclass(frozen=True)
class AzureWebResearchResponse:
    data: dict[str, Any]
    metadata: AzureWebResearchMetadata


def _safe_error_body(response: Any, limit: int = 2000) -> str:
    try:
        value = response.json()
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        text = str(getattr(response, "text", "") or "")
    text = re.sub(r"(?i)(api[-_ ]?key|authorization|bearer)\s*[:=]\s*[^\s,}\]]+", r"\1=<redacted>", text)
    return text[:limit]


def _request_consumption_metrics(body: Mapping[str, Any]) -> dict[str, Any]:
    serialized = json.dumps(
        body,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return {
        "request_json_chars": len(serialized),
        "request_json_bytes": len(serialized.encode("utf-8")),
        "estimated_input_tokens": max(1, (len(serialized) + 3) // 4),
        "token_estimation_method": "serialized_json_characters_divided_by_4",
    }


def _response_consumption_metrics(
    envelope: Mapping[str, Any],
    *,
    output_text: str = "",
) -> dict[str, Any]:
    usage = envelope.get("usage")
    usage_map = usage if isinstance(usage, Mapping) else {}
    input_details = usage_map.get("input_tokens_details")
    input_map = input_details if isinstance(input_details, Mapping) else {}
    output_details = usage_map.get("output_tokens_details")
    output_map = output_details if isinstance(output_details, Mapping) else {}
    serialized = json.dumps(
        envelope,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    input_tokens = int(usage_map.get("input_tokens") or 0)
    output_tokens = int(usage_map.get("output_tokens") or 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "thinking_tokens": int(
            output_map.get("reasoning_tokens")
            or output_map.get("thinking_tokens")
            or 0
        ),
        "cache_read_input_tokens": int(
            input_map.get("cached_tokens")
            or usage_map.get("cache_read_input_tokens")
            or 0
        ),
        "total_tokens": int(
            usage_map.get("total_tokens")
            or input_tokens + output_tokens
        ),
        "response_body_bytes": len(serialized.encode("utf-8")),
        "response_text_characters": len(output_text),
    }


def _strip_json_wrapper(text: str) -> str:
    value = str(text or "").strip()
    value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s*```$", "", value)
    start = value.find("{")
    end = value.rfind("}")
    if start >= 0 and end > start:
        value = value[start : end + 1]
    return value


_SENSITIVE_KEY_RE = re.compile(r"(?i)(api[-_ ]?key|authorization|bearer|secret|token)")


def _redact_diagnostic(value: Any) -> Any:
    """Recursively redact credentials while retaining the failed research data."""

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            name = str(key)
            result[name] = "<redacted>" if _SENSITIVE_KEY_RE.search(name) else _redact_diagnostic(item)
        return result
    if isinstance(value, list):
        return [_redact_diagnostic(item) for item in value]
    if isinstance(value, tuple):
        return [_redact_diagnostic(item) for item in value]
    if isinstance(value, str):
        return re.sub(
            r"(?i)(api[-_ ]?key|authorization|bearer)\\s*[:=]\\s*[^\\s,}\\]]+",
            r"\\1=<redacted>",
            value,
        )
    return value


def _diagnostic_headers(response: Any) -> dict[str, str]:
    headers = getattr(response, "headers", {}) or {}
    allowed = {
        "apim-request-id",
        "content-type",
        "date",
        "request-id",
        "x-ms-request-id",
        "x-request-id",
    }
    return {
        str(key): str(value)
        for key, value in dict(headers).items()
        if str(key).casefold() in allowed
    }


def _persist_failure_diagnostic(
    diagnostic_dir: Path | None,
    *,
    outcome: str,
    tool_type: str,
    attempt: int,
    request_body: Mapping[str, Any],
    diagnostic_context: Mapping[str, Any] | None = None,
    http_status: int | None = None,
    response_headers: Mapping[str, Any] | None = None,
    response_envelope: Any = None,
    response_text: str | None = None,
    output_text: str | None = None,
    parsed_output: Any = None,
    normalized_output: Any = None,
    dropped_fields: Sequence[str] = (),
    canonicalized_fields: Sequence[str] = (),
    error: str = "",
    validation_error: Mapping[str, Any] | None = None,
    elapsed_seconds: float | None = None,
) -> str | None:
    """Persist a failed provider attempt for diagnosis, never for result reuse."""

    if diagnostic_dir is None:
        return None
    root = Path(diagnostic_dir)
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    safe_outcome = re.sub(r"[^a-z0-9_-]+", "-", outcome.casefold()).strip("-")
    safe_tool = re.sub(r"[^a-z0-9_-]+", "-", tool_type.casefold()).strip("-")
    path = root / f"{stamp}-{safe_tool}-attempt-{attempt:02d}-{safe_outcome}.json"
    record = {
        "schema_version": "mathula-azure-web-research-failure-diagnostic-v1",
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "outcome": outcome,
        "tool_type": tool_type,
        "attempt": attempt,
        "http_status": http_status,
        "elapsed_seconds": (
            round(float(elapsed_seconds), 3) if elapsed_seconds is not None else None
        ),
        "error": error,
        "validation_error": dict(validation_error or {}),
        "diagnostic_context": dict(diagnostic_context or {}),
        "request": dict(request_body),
        "response_headers": dict(response_headers or {}),
        "response_envelope": response_envelope,
        "response_text": response_text,
        "output_text": output_text,
        "parsed_output": parsed_output,
        "normalized_output": normalized_output,
        "dropped_fields": list(dropped_fields),
        "canonicalized_fields": list(canonicalized_fields),
        "reuse_policy": "diagnostic_only_never_used_as_research_result",
    }
    record = _redact_diagnostic(record)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return str(path)



def _expected_const(schema: Mapping[str, Any], key: str) -> Any:
    properties = schema.get("properties")
    if not isinstance(properties, Mapping):
        return None
    field = properties.get(key)
    if not isinstance(field, Mapping):
        return None
    return field.get("const")


def _prune_to_schema(value: Any, schema: Mapping[str, Any]) -> Any:
    """Discard model commentary fields while retaining strict validation.

    Azure Responses web search does not currently receive a server-enforced JSON
    schema in Mathula's minimal request envelope. GPT may therefore return the
    requested object plus harmless explanatory keys. We remove only keys that the
    caller's schema explicitly forbids, then run the unchanged strict validator.
    Required fields, types, enums, confidence limits, and nested contracts remain
    fully enforced after normalization.
    """

    schema_type = schema.get("type")
    if schema_type == "object" and isinstance(value, Mapping):
        properties = schema.get("properties")
        if not isinstance(properties, Mapping):
            return dict(value)
        result: dict[str, Any] = {}
        for key, child_schema in properties.items():
            if key not in value:
                continue
            child = child_schema if isinstance(child_schema, Mapping) else {}
            result[str(key)] = _prune_to_schema(value[key], child)
        return result
    if schema_type == "array" and isinstance(value, list):
        item_schema = schema.get("items")
        child = item_schema if isinstance(item_schema, Mapping) else {}
        return [_prune_to_schema(item, child) for item in value]
    return value



def _first_present(item: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in item and item.get(name) not in (None, "", []):
            return item.get(name)
    return None


def _normalise_url_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        url = item.get("url") if isinstance(item, Mapping) else item
        text = str(url or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def _normalise_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _normalise_confidence(value: Any) -> Any:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value or "").strip()
    if not text:
        return value
    try:
        if text.endswith("%"):
            return float(text[:-1].strip()) / 100.0
        number = float(text)
        return number / 100.0 if number > 1.0 and number <= 100.0 else number
    except ValueError:
        return value


def _normalise_correction_items(
    prepared: dict[str, Any], payload: Mapping[str, Any] | None
) -> tuple[dict[str, Any], tuple[str, ...]]:
    corrections = prepared.get("corrections")
    if not isinstance(corrections, list):
        return prepared, ()
    candidates: list[dict[str, Any]] = []
    if isinstance(payload, Mapping):
        raw_candidates = (
            payload.get("candidate_entities")
            if isinstance(payload.get("candidate_entities"), list)
            else payload.get("candidate_name_spans")
        )
        if isinstance(raw_candidates, list):
            candidates = [
                dict(item)
                for item in raw_candidates
                if isinstance(item, Mapping)
            ]
    aliases_used: set[str] = set()
    normalized: list[Any] = []
    scalar_unresolved: list[dict[str, str]] = []
    candidate_ids = {
        str(item.get("entity_id") or "").strip(): item
        for item in candidates
        if str(item.get("entity_id") or "").strip()
    }
    for raw_item in corrections:
        if not isinstance(raw_item, Mapping):
            scalar = str(raw_item or "").strip()
            if scalar in candidate_ids:
                # A bare supplied entity ID contains no researched canonical
                # spelling or evidence. It is safe to preserve the cluster as
                # unresolved instead of failing the complete batch or
                # pretending the identifier is a correction object.
                scalar_unresolved.append(
                    {
                        "entity_id": scalar,
                        "reason": (
                            "research_model_returned_entity_id_without_"
                            "supported_correction"
                        ),
                    }
                )
                aliases_used.add("scalar_correction_moved_to_unresolved")
                continue
            normalized.append(raw_item)
            continue
        item = dict(raw_item)
        alias_map = {
            "segment_id": ("segment", "segmentId", "candidate_segment_id"),
            "raw_text": (
                "heard_text", "original_text", "transcribed_text", "candidate_text",
                "source_name", "misspelled_text", "incorrect_text",
            ),
            "canonical_text": (
                "canonical_replacement", "corrected_text", "resolved_text",
                "replacement_text", "corrected_name", "resolved_name",
                "canonical_name", "correct_spelling", "corrected_spelling",
                "replacement", "corrected_to",
            ),
            "entity_type": ("type", "name_type", "entity_kind"),
            "confidence": ("confidence_score", "score", "probability"),
            "source_urls": (
                "sources", "citations", "evidence", "evidence_urls", "urls"
            ),
            "story_match_terms": (
                "context_terms", "matching_terms", "story_clues", "same_story_clues",
            ),
            "reason": ("explanation", "rationale", "justification"),
        }
        for canonical, aliases in alias_map.items():
            if item.get(canonical) not in (None, "", []):
                continue
            value = _first_present(item, aliases)
            if value not in (None, "", []):
                item[canonical] = value
                aliases_used.add(canonical)

        # Fill only candidate identity fields that are unambiguous in this exact batch.
        raw_value = str(item.get("raw_text") or "").strip().casefold()
        if raw_value and not item.get("segment_id"):
            matches = [
                candidate
                for candidate in candidates
                if str(candidate.get("raw_text") or "").strip().casefold() == raw_value
            ]
            if len(matches) == 1 and matches[0].get("segment_id"):
                item["segment_id"] = matches[0]["segment_id"]
                aliases_used.add("segment_id_from_candidate")

        entity_type = str(item.get("entity_type") or "").strip().casefold()
        type_aliases = {
            "organization": "organisation",
            "organisation": "organisation",
            "org": "organisation",
            "person": "person",
            "place": "place",
            "location": "place",
            "case": "case",
            "court_case": "case",
            "other": "other_name",
            "name": "other_name",
            "other_name": "other_name",
        }
        if entity_type in type_aliases:
            item["entity_type"] = type_aliases[entity_type]
        if "confidence" in item:
            item["confidence"] = _normalise_confidence(item.get("confidence"))
        if "source_urls" in item:
            item["source_urls"] = _normalise_url_list(item.get("source_urls"))
        if "story_match_terms" in item:
            item["story_match_terms"] = _normalise_string_list(item.get("story_match_terms"))
        normalized.append(item)
    prepared["corrections"] = normalized
    if scalar_unresolved:
        existing = (
            list(prepared.get("unresolved_candidates") or [])
            if isinstance(prepared.get("unresolved_candidates"), list)
            else []
        )
        existing_ids = {
            str(item.get("entity_id") or "").strip()
            for item in existing
            if isinstance(item, Mapping)
        }
        prepared["unresolved_candidates"] = [
            *existing,
            *[
                item
                for item in scalar_unresolved
                if item["entity_id"] not in existing_ids
            ],
        ]
    return prepared, tuple(sorted(aliases_used))


def _normalise_research_output(
    data: Mapping[str, Any],
    output_schema: Mapping[str, Any],
    payload: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], tuple[str, ...], tuple[str, ...]]:
    """Normalise common GPT wrapper aliases before strict schema validation.

    This is intentionally narrow. Accepted corrections still pass Mathula's later
    candidate binding, same-story, confidence, phonetic, and actual-citation checks.
    Model-provided top-level ``sources`` are discarded because only URLs observed
    in the Azure response envelope count as server-tool evidence.
    """

    expected = output_schema.get("properties")
    if not isinstance(expected, Mapping):
        return dict(data), (), ()

    source: Mapping[str, Any] = data
    expected_keys = set(str(key) for key in expected)
    if not expected_keys.intersection(source):
        for wrapper in ("result", "data", "research_result", "name_research"):
            candidate = source.get(wrapper)
            if isinstance(candidate, Mapping):
                source = candidate
                break

    prepared = dict(source)
    aliases = {
        "corrections": (
            "resolved_corrections",
            "accepted_corrections",
            "name_corrections",
        ),
        "unresolved_candidates": (
            "unresolved",
            "unresolved_names",
            "unresolved_corrections",
        ),
    }
    for canonical, alternatives in aliases.items():
        if canonical in prepared:
            continue
        for alias in alternatives:
            if alias in prepared:
                prepared[canonical] = prepared[alias]
                break

    prepared, correction_aliases = _normalise_correction_items(prepared, payload)

    canonicalized: list[str] = list(correction_aliases)
    schema_version = _expected_const(output_schema, "schema_version")
    if schema_version is not None and prepared.get("schema_version") != schema_version:
        # The schema version is an application transport contract, not researched
        # evidence. GPT sometimes echoes a batch/artifact schema name instead of
        # the requested response schema. Canonicalise it before strict validation;
        # all substantive correction fields remain model supplied and validated.
        prepared["schema_version"] = schema_version
        canonicalized.append("schema_version")
    if "corrections" not in prepared:
        prepared["corrections"] = []
    if "unresolved_candidates" not in prepared:
        prepared["unresolved_candidates"] = []

    dropped = tuple(sorted(str(key) for key in prepared if key not in expected_keys))
    normalised = _prune_to_schema(prepared, output_schema)
    if not isinstance(normalised, dict):
        normalised = dict(prepared)
    return normalised, dropped, tuple(canonicalized)

def _response_output_text(response: Mapping[str, Any]) -> str:
    direct = response.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct
    parts: list[str] = []
    for item in response.get("output") or []:
        if not isinstance(item, Mapping) or item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if not isinstance(content, Mapping):
                continue
            if content.get("type") in {"output_text", "text"}:
                text = content.get("text")
                if isinstance(text, str):
                    parts.append(text)
    return "\n".join(parts).strip()


def _collect_evidence(response: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    found: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(url: Any, title: Any = "", source_type: str = "url_citation") -> None:
        value = str(url or "").strip()
        if not value or value in seen:
            return
        seen.add(value)
        found.append(
            {
                "url": value,
                "title": str(title or "").strip(),
                "type": source_type,
            }
        )

    for item in response.get("output") or []:
        if not isinstance(item, Mapping):
            continue
        if item.get("type") == "web_search_call":
            action = item.get("action")
            if isinstance(action, Mapping):
                for source in action.get("sources") or []:
                    if isinstance(source, Mapping):
                        add(source.get("url"), source.get("title"), "web_search_source")
        if item.get("type") == "message":
            for content in item.get("content") or []:
                if not isinstance(content, Mapping):
                    continue
                for annotation in content.get("annotations") or []:
                    if not isinstance(annotation, Mapping):
                        continue
                    if annotation.get("type") == "url_citation":
                        add(
                            annotation.get("url"),
                            annotation.get("title"),
                            "url_citation",
                        )
    return tuple(found)


def _tool_sequence(configured: str) -> tuple[str, ...]:
    if configured == "auto":
        return ("web_search", "web_search_preview")
    return (configured,)


_STRUCTURED_OUTPUT_UNSUPPORTED_KEYS = {
    "minLength",
    "maxLength",
    "pattern",
    "format",
    "minimum",
    "maximum",
    "multipleOf",
    "minItems",
    "maxItems",
    "uniqueItems",
    "contains",
    "minContains",
    "maxContains",
}


def _structured_output_schema(value: Any) -> Any:
    """Return the Azure Structured Outputs-compatible subset of a schema.

    Runtime validation still uses the original, stricter JSON Schema. This copy
    exists only to force the model to emit all required fields. Azure structured
    outputs do not support several length/range keywords, and expose enum rather
    than const in the documented subset.
    """

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if key in _STRUCTURED_OUTPUT_UNSUPPORTED_KEYS:
                continue
            if key == "const":
                result["enum"] = [_structured_output_schema(item)]
                continue
            result[str(key)] = _structured_output_schema(item)
        properties = result.get("properties")
        if result.get("type") == "object" and isinstance(properties, Mapping):
            # Azure strict structured outputs require every declared property
            # at every object depth to appear in ``required``. Application
            # validation still uses the original schema, so this transport
            # copy may safely require optional explanatory fields; unsupported
            # minLength/range constraints were removed above and the model can
            # emit empty strings/lists when no optional evidence is available.
            result["required"] = [str(key) for key in properties]
            result["additionalProperties"] = False
        return result
    if isinstance(value, list):
        return [_structured_output_schema(item) for item in value]
    if isinstance(value, tuple):
        return [_structured_output_schema(item) for item in value]
    return value


def _structured_text_format(output_schema: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "format": {
            "type": "json_schema",
            "name": "mathula_autocorrect_name_research",
            "description": (
                "Strict grounded proper-name corrections for one transcript batch"
            ),
            "strict": True,
            "schema": _structured_output_schema(dict(output_schema)),
        }
    }


class AzureResponsesWebResearchProvider:
    """One-call JSON research provider backed by Azure Responses web search."""

    def __init__(
        self,
        config: AzureWebResearchConfig,
        *,
        session: Any = None,
        sleep: Any = time.sleep,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.config = config
        self.session = session or requests.Session()
        self.sleep = sleep
        self.progress_callback = progress_callback

    def _emit(self, event: str, **details: Any) -> None:
        callback = self.progress_callback
        if callback is None:
            return
        try:
            callback({"event": event, **details})
        except Exception:
            # Progress reporting must never change research correctness.
            return

    @classmethod
    def from_environment(
        cls, environ: Mapping[str, str] | None = None
    ) -> "AzureResponsesWebResearchProvider":
        return cls(AzureWebResearchConfig.from_environment(environ))

    def _body(
        self,
        *,
        system_prompt: str,
        payload: Mapping[str, Any],
        tool_type: str,
    ) -> dict[str, Any]:
        # Keep the Azure request envelope identical to the minimal Responses
        # contract proven against the production resource. The full task
        # instructions live inside ``input`` so optional Azure fields cannot
        # make an otherwise valid web-search request fail with HTTP 400.
        user_text = (
            system_prompt.strip()
            + "\n\nReturn one valid JSON object only. Research the supplied "
              "proper-name candidates using web search.\n\nINPUT:\n"
            + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        )
        body: dict[str, Any] = {
            "model": self.config.deployment,
            "tools": [{"type": tool_type}],
            "include": ["web_search_call.action.sources"],
            "input": user_text,
        }
        # Keep the web-search generation free-form. Azure reliably performs the
        # search with this compact envelope. If GPT returns a proposal rather
        # than the full application schema, a separate no-search structured
        # output pass repairs only the transport shape.
        return body

    def _repair_json_to_schema(
        self,
        *,
        parsed_output: Mapping[str, Any],
        payload: Mapping[str, Any],
        output_schema: Mapping[str, Any],
        actual_evidence: Sequence[Mapping[str, Any]],
        headers: Mapping[str, str],
        save_failure: Callable[..., str | None],
    ) -> tuple[dict[str, Any], Mapping[str, Any], float, int]:
        """Convert a grounded proposal into the strict application schema.

        This pass has no web-search tool. It cannot discover new entities or
        sources; it only maps the already-grounded proposal and batch context
        into the contract required by Mathula's deterministic validator.
        """

        candidate_entities = (
            payload.get("candidate_entities")
            if isinstance(payload.get("candidate_entities"), list)
            else payload.get("candidate_name_spans") or []
        )
        compact_context = {
            "candidate_entities": candidate_entities,
            "source_metadata": payload.get("source_metadata") or {},
            "research_profile": {
                "name": (payload.get("research_profile") or {}).get("name")
                if isinstance(payload.get("research_profile"), Mapping)
                else None,
                "authoritative_domains": (
                    (payload.get("research_profile") or {}).get(
                        "authoritative_domains"
                    )
                    if isinstance(payload.get("research_profile"), Mapping)
                    else []
                ),
            },
        }
        grounded_sources = [
            {
                "url": str(item.get("url") or "").strip(),
                "title": str(item.get("title") or "").strip(),
                "type": str(item.get("type") or "").strip(),
            }
            for item in actual_evidence
            if isinstance(item, Mapping) and str(item.get("url") or "").strip()
        ]
        repair_input = (
            "You are a strict JSON transport repair step for Mathula TV. "
            "Do not search the web and do not introduce any correction that is "
            "not already proposed below. Convert the proposal into the supplied "
            "JSON schema. Use only URLs listed in actual_grounded_sources. If a "
            "proposed correction cannot be supported from the supplied context "
            "and grounded sources, move that candidate to unresolved_candidates. "
            "Every supplied entity_id must appear exactly once, either as a "
            "correction or as unresolved. Preserve entity_id exactly and do not "
            "create new entities. entity_type must use one of the values allowed "
            "by the supplied schema. confidence "
            "must be a calibrated number from 0 to 1. story_match_terms must be "
            "concrete same-story clues, not generic words. Return JSON only.\n\n"
            + json.dumps(
                {
                    "proposal": dict(parsed_output),
                    "actual_grounded_sources": grounded_sources,
                    "batch_context": compact_context,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
        body: dict[str, Any] = {
            "model": self.config.deployment,
            "input": repair_input,
            "text": _structured_text_format(output_schema),
        }
        request_metrics = _request_consumption_metrics(body)
        self._emit(
            "schema_repair_started",
            **request_metrics,
            candidate_count=len(compact_context["candidate_entities"]),
            source_count=len(grounded_sources),
        )
        started = time.monotonic()
        failures: list[str] = []
        attempts = 0
        for attempt in range(1, max(1, self.config.max_retries) + 1):
            attempts += 1
            attempt_started = time.monotonic()
            try:
                response = self.session.post(
                    self.config.url,
                    headers=dict(headers),
                    json=body,
                    timeout=self.config.timeout_seconds,
                )
            except (requests.Timeout, TimeoutError):
                message = f"schema repair timeout on attempt {attempt}"
                failures.append(message)
                diagnostic_path = save_failure(
                    outcome="schema_repair_timeout",
                    tool_type="structured_output_repair",
                    attempt=attempt,
                    request_body=body,
                    error=message,
                    elapsed_seconds=time.monotonic() - attempt_started,
                )
                self._emit(
                    "schema_repair_attempt_failed",
                    attempt=attempt,
                    error=message,
                    diagnostic_path=diagnostic_path,
                )
                if attempt < self.config.max_retries:
                    self.sleep(min(2 ** (attempt - 1), 10))
                    continue
                break
            except (requests.ConnectionError, ConnectionError, OSError) as exc:
                message = (
                    "schema repair connection failure on attempt "
                    f"{attempt}: {type(exc).__name__}"
                )
                failures.append(message)
                diagnostic_path = save_failure(
                    outcome="schema_repair_connection_failure",
                    tool_type="structured_output_repair",
                    attempt=attempt,
                    request_body=body,
                    error=message,
                    elapsed_seconds=time.monotonic() - attempt_started,
                )
                self._emit(
                    "schema_repair_attempt_failed",
                    attempt=attempt,
                    error=message,
                    diagnostic_path=diagnostic_path,
                )
                if attempt < self.config.max_retries:
                    self.sleep(min(2 ** (attempt - 1), 10))
                    continue
                break

            status = int(getattr(response, "status_code", 0))
            try:
                envelope = response.json()
            except Exception:
                envelope = None
            if not 200 <= status < 300:
                message = (
                    f"schema repair HTTP {status}: {_safe_error_body(response)}"
                )
                failures.append(message)
                diagnostic_path = save_failure(
                    outcome="schema_repair_http_error",
                    tool_type="structured_output_repair",
                    attempt=attempt,
                    request_body=body,
                    http_status=status,
                    response_headers=_diagnostic_headers(response),
                    response_envelope=envelope,
                    response_text=str(getattr(response, "text", "") or ""),
                    error=message,
                    elapsed_seconds=time.monotonic() - attempt_started,
                )
                self._emit(
                    "schema_repair_attempt_failed",
                    attempt=attempt,
                    error=message,
                    diagnostic_path=diagnostic_path,
                )
                if (status == 429 or status >= 500) and attempt < self.config.max_retries:
                    self.sleep(min(2 ** (attempt - 1), 10))
                    continue
                break
            if not isinstance(envelope, Mapping):
                message = "schema repair response envelope is not an object"
                failures.append(message)
                diagnostic_path = save_failure(
                    outcome="schema_repair_non_object_envelope",
                    tool_type="structured_output_repair",
                    attempt=attempt,
                    request_body=body,
                    http_status=status,
                    response_headers=_diagnostic_headers(response),
                    response_envelope=envelope,
                    error=message,
                    elapsed_seconds=time.monotonic() - attempt_started,
                )
                self._emit(
                    "schema_repair_attempt_failed",
                    attempt=attempt,
                    error=message,
                    diagnostic_path=diagnostic_path,
                )
                break

            output_text = _response_output_text(envelope)
            try:
                repaired = json.loads(_strip_json_wrapper(output_text))
            except (json.JSONDecodeError, TypeError) as exc:
                message = f"schema repair returned invalid JSON: {exc}"
                failures.append(message)
                diagnostic_path = save_failure(
                    outcome="schema_repair_invalid_json",
                    tool_type="structured_output_repair",
                    attempt=attempt,
                    request_body=body,
                    http_status=status,
                    response_headers=_diagnostic_headers(response),
                    response_envelope=envelope,
                    output_text=output_text,
                    error=message,
                    elapsed_seconds=time.monotonic() - attempt_started,
                )
                self._emit(
                    "schema_repair_attempt_failed",
                    attempt=attempt,
                    error=message,
                    diagnostic_path=diagnostic_path,
                )
                break
            if not isinstance(repaired, Mapping):
                message = "schema repair JSON root is not an object"
                failures.append(message)
                diagnostic_path = save_failure(
                    outcome="schema_repair_non_object_json",
                    tool_type="structured_output_repair",
                    attempt=attempt,
                    request_body=body,
                    http_status=status,
                    response_headers=_diagnostic_headers(response),
                    response_envelope=envelope,
                    output_text=output_text,
                    parsed_output=repaired,
                    error=message,
                    elapsed_seconds=time.monotonic() - attempt_started,
                )
                self._emit(
                    "schema_repair_attempt_failed",
                    attempt=attempt,
                    error=message,
                    diagnostic_path=diagnostic_path,
                )
                break
            normalized, dropped, canonicalized = _normalise_research_output(
                repaired, output_schema, payload
            )
            try:
                jsonschema.Draft202012Validator(dict(output_schema)).validate(
                    normalized
                )
            except jsonschema.ValidationError as exc:
                message = (
                    "schema repair output validation failed: " + exc.message
                )
                failures.append(message)
                diagnostic_path = save_failure(
                    outcome="schema_repair_output_schema_validation_failed",
                    tool_type="structured_output_repair",
                    attempt=attempt,
                    request_body=body,
                    http_status=status,
                    response_headers=_diagnostic_headers(response),
                    response_envelope=envelope,
                    output_text=output_text,
                    parsed_output=dict(repaired),
                    normalized_output=normalized,
                    dropped_fields=dropped,
                    canonicalized_fields=canonicalized,
                    error=message,
                    validation_error={
                        "message": exc.message,
                        "path": list(exc.absolute_path),
                        "schema_path": list(exc.absolute_schema_path),
                        "validator": str(exc.validator or ""),
                    },
                    elapsed_seconds=time.monotonic() - attempt_started,
                )
                self._emit(
                    "schema_repair_attempt_failed",
                    attempt=attempt,
                    error=message,
                    diagnostic_path=diagnostic_path,
                )
                break

            elapsed = time.monotonic() - started
            self._emit(
                "schema_repair_completed",
                attempt=attempt,
                elapsed_seconds=elapsed,
                response_id=str(envelope.get("id") or ""),
                **_response_consumption_metrics(
                    envelope,
                    output_text=output_text,
                ),
            )
            return dict(normalized), envelope, elapsed, attempts

        raise AzureWebResearchError(
            "Azure structured-output schema repair failed: "
            + (" | ".join(failures[-4:]) or "no response details")
        )

    def research_json(
        self,
        *,
        system_prompt: str,
        payload: Mapping[str, Any],
        output_schema: Mapping[str, Any],
        diagnostic_dir: Path | None = None,
        diagnostic_context: Mapping[str, Any] | None = None,
    ) -> AzureWebResearchResponse:
        headers = {
            "Content-Type": "application/json",
            "api-key": self.config.api_key,
        }
        failures: list[str] = []
        diagnostic_paths: list[str] = []
        request_attempts = 0
        started = time.monotonic()

        def save_failure(**details: Any) -> str | None:
            path = _persist_failure_diagnostic(
                diagnostic_dir,
                diagnostic_context=diagnostic_context,
                **details,
            )
            if path:
                diagnostic_paths.append(path)
            return path

        for tool_type in _tool_sequence(self.config.web_search_tool):
            received_successful_http = False
            for attempt in range(1, max(1, self.config.max_retries) + 1):
                request_attempts += 1
                body = self._body(
                    system_prompt=system_prompt,
                    payload=payload,
                    tool_type=tool_type,
                )
                self._emit(
                    "request_attempt_started",
                    tool_type=tool_type,
                    attempt=attempt,
                    max_retries=max(1, self.config.max_retries),
                    **_request_consumption_metrics(body),
                )
                attempt_started = time.monotonic()
                try:
                    response = self.session.post(
                        self.config.url,
                        headers=headers,
                        json=body,
                        timeout=self.config.timeout_seconds,
                    )
                except (requests.Timeout, TimeoutError):
                    message = f"{tool_type}: timeout on attempt {attempt}"
                    failures.append(message)
                    diagnostic_path = save_failure(
                        outcome="timeout",
                        tool_type=tool_type,
                        attempt=attempt,
                        request_body=body,
                        error=message,
                        elapsed_seconds=time.monotonic() - attempt_started,
                    )
                    self._emit(
                        "request_attempt_failed",
                        tool_type=tool_type,
                        attempt=attempt,
                        elapsed_seconds=time.monotonic() - attempt_started,
                        error=message,
                        diagnostic_path=diagnostic_path,
                    )
                    if attempt >= self.config.max_retries:
                        break
                    delay = min(2 ** (attempt - 1), 10)
                    self._emit("request_retry_scheduled", delay_seconds=delay)
                    self.sleep(delay)
                    continue
                except (requests.ConnectionError, ConnectionError, OSError) as exc:
                    message = (
                        f"{tool_type}: connection failure on attempt {attempt}: "
                        f"{type(exc).__name__}"
                    )
                    failures.append(message)
                    diagnostic_path = save_failure(
                        outcome="connection_failure",
                        tool_type=tool_type,
                        attempt=attempt,
                        request_body=body,
                        error=message,
                        elapsed_seconds=time.monotonic() - attempt_started,
                    )
                    self._emit(
                        "request_attempt_failed",
                        tool_type=tool_type,
                        attempt=attempt,
                        elapsed_seconds=time.monotonic() - attempt_started,
                        error=message,
                        diagnostic_path=diagnostic_path,
                    )
                    if attempt >= self.config.max_retries:
                        break
                    delay = min(2 ** (attempt - 1), 10)
                    self._emit("request_retry_scheduled", delay_seconds=delay)
                    self.sleep(delay)
                    continue

                status = int(getattr(response, "status_code", 0))
                if status == 429 or 500 <= status < 600:
                    message = f"{tool_type}: HTTP {status}: {_safe_error_body(response)}"
                    failures.append(message)
                    try:
                        response_payload = response.json()
                    except Exception:
                        response_payload = None
                    diagnostic_path = save_failure(
                        outcome="http_error",
                        tool_type=tool_type,
                        attempt=attempt,
                        request_body=body,
                        http_status=status,
                        response_headers=_diagnostic_headers(response),
                        response_envelope=response_payload,
                        response_text=str(getattr(response, "text", "") or ""),
                        error=message,
                        elapsed_seconds=time.monotonic() - attempt_started,
                    )
                    self._emit(
                        "request_attempt_failed",
                        tool_type=tool_type,
                        attempt=attempt,
                        http_status=status,
                        elapsed_seconds=time.monotonic() - attempt_started,
                        error=message,
                        diagnostic_path=diagnostic_path,
                    )
                    if attempt < self.config.max_retries:
                        delay = min(2 ** (attempt - 1), 10)
                        self._emit("request_retry_scheduled", delay_seconds=delay)
                        self.sleep(delay)
                        continue
                    break
                if not 200 <= status < 300:
                    message = f"{tool_type}: HTTP {status}: {_safe_error_body(response)}"
                    failures.append(message)
                    try:
                        response_payload = response.json()
                    except Exception:
                        response_payload = None
                    diagnostic_path = save_failure(
                        outcome="http_error",
                        tool_type=tool_type,
                        attempt=attempt,
                        request_body=body,
                        http_status=status,
                        response_headers=_diagnostic_headers(response),
                        response_envelope=response_payload,
                        response_text=str(getattr(response, "text", "") or ""),
                        error=message,
                        elapsed_seconds=time.monotonic() - attempt_started,
                    )
                    self._emit(
                        "request_attempt_failed",
                        tool_type=tool_type,
                        attempt=attempt,
                        http_status=status,
                        elapsed_seconds=time.monotonic() - attempt_started,
                        error=message,
                        diagnostic_path=diagnostic_path,
                    )
                    # In auto mode, a 400 may mean only this alias is unsupported.
                    break

                received_successful_http = True
                try:
                    envelope = response.json()
                except (ValueError, TypeError):
                    message = f"{tool_type}: malformed JSON response envelope"
                    failures.append(message)
                    diagnostic_path = save_failure(
                        outcome="malformed_response_envelope", tool_type=tool_type, attempt=attempt,
                        request_body=body, http_status=status, response_headers=_diagnostic_headers(response),
                        response_text=str(getattr(response, "text", "") or ""), error=message,
                        elapsed_seconds=time.monotonic() - attempt_started,
                    )
                    self._emit("response_validation_failed", error=message, diagnostic_path=diagnostic_path)
                    break
                if not isinstance(envelope, Mapping):
                    message = f"{tool_type}: response envelope is not an object"
                    failures.append(message)
                    diagnostic_path = save_failure(
                        outcome="non_object_response_envelope", tool_type=tool_type, attempt=attempt,
                        request_body=body, http_status=status, response_headers=_diagnostic_headers(response),
                        response_envelope=envelope, error=message, elapsed_seconds=time.monotonic() - attempt_started,
                    )
                    self._emit("response_validation_failed", error=message, diagnostic_path=diagnostic_path)
                    break
                if envelope.get("status") in {"failed", "incomplete"}:
                    message = (
                        f"{tool_type}: response status {envelope.get('status')}: "
                        f"{json.dumps(envelope.get('error') or envelope.get('incomplete_details') or {}, ensure_ascii=False)[:1000]}"
                    )
                    failures.append(message)
                    diagnostic_path = save_failure(
                        outcome="azure_response_failed", tool_type=tool_type, attempt=attempt,
                        request_body=body, http_status=status, response_headers=_diagnostic_headers(response),
                        response_envelope=envelope, error=message, elapsed_seconds=time.monotonic() - attempt_started,
                    )
                    self._emit("response_validation_failed", error=message, diagnostic_path=diagnostic_path)
                    break

                output_text = _response_output_text(envelope)
                if not output_text:
                    message = f"{tool_type}: response contained no output text"
                    failures.append(message)
                    diagnostic_path = save_failure(
                        outcome="missing_output_text", tool_type=tool_type, attempt=attempt,
                        request_body=body, http_status=status, response_headers=_diagnostic_headers(response),
                        response_envelope=envelope, error=message, elapsed_seconds=time.monotonic() - attempt_started,
                    )
                    self._emit("response_validation_failed", error=message, diagnostic_path=diagnostic_path)
                    break
                try:
                    data = json.loads(_strip_json_wrapper(output_text))
                except json.JSONDecodeError as exc:
                    message = (
                        f"{tool_type}: invalid JSON output: {exc}; "
                        f"output={output_text[:1000]}"
                    )
                    failures.append(message)
                    diagnostic_path = save_failure(
                        outcome="invalid_json_output", tool_type=tool_type, attempt=attempt,
                        request_body=body, http_status=status, response_headers=_diagnostic_headers(response),
                        response_envelope=envelope, output_text=output_text, error=message,
                        elapsed_seconds=time.monotonic() - attempt_started,
                    )
                    self._emit("response_validation_failed", error=message, diagnostic_path=diagnostic_path)
                    break
                if not isinstance(data, dict):
                    message = f"{tool_type}: JSON output root is not an object"
                    failures.append(message)
                    diagnostic_path = save_failure(
                        outcome="non_object_json_output", tool_type=tool_type, attempt=attempt,
                        request_body=body, http_status=status, response_headers=_diagnostic_headers(response),
                        response_envelope=envelope, output_text=output_text, parsed_output=data, error=message,
                        elapsed_seconds=time.monotonic() - attempt_started,
                    )
                    self._emit("response_validation_failed", error=message, diagnostic_path=diagnostic_path)
                    break
                parsed_output = dict(data)
                data, dropped_fields, canonicalized_fields = _normalise_research_output(
                    data, output_schema, payload
                )
                if dropped_fields or canonicalized_fields:
                    self._emit(
                        "response_normalized",
                        tool_type=tool_type,
                        dropped_fields=list(dropped_fields),
                        canonicalized_fields=list(canonicalized_fields),
                    )
                schema_repair_used = False
                schema_repair_response_id: str | None = None
                schema_repair_elapsed_seconds = 0.0
                try:
                    jsonschema.Draft202012Validator(dict(output_schema)).validate(data)
                except jsonschema.ValidationError as exc:
                    message = (
                        f"{tool_type}: output schema validation failed: {exc.message}"
                    )
                    validation_details = {
                        "message": exc.message,
                        "path": list(exc.absolute_path),
                        "schema_path": list(exc.absolute_schema_path),
                        "validator": str(exc.validator or ""),
                    }
                    diagnostic_path = save_failure(
                        outcome="output_schema_validation_repair_required",
                        tool_type=tool_type,
                        attempt=attempt,
                        request_body=body,
                        http_status=status,
                        response_headers=_diagnostic_headers(response),
                        response_envelope=envelope,
                        output_text=output_text,
                        parsed_output=parsed_output,
                        normalized_output=data,
                        dropped_fields=dropped_fields,
                        canonicalized_fields=canonicalized_fields,
                        error=message,
                        validation_error=validation_details,
                        elapsed_seconds=time.monotonic() - attempt_started,
                    )
                    self._emit(
                        "response_schema_repair_required",
                        error=message,
                        diagnostic_path=diagnostic_path,
                        validation_path=list(exc.absolute_path),
                    )
                    try:
                        (
                            data,
                            repair_envelope,
                            schema_repair_elapsed_seconds,
                            repair_attempts,
                        ) = self._repair_json_to_schema(
                            parsed_output=parsed_output,
                            payload=payload,
                            output_schema=output_schema,
                            actual_evidence=_collect_evidence(envelope),
                            headers=headers,
                            save_failure=save_failure,
                        )
                    except AzureWebResearchError as repair_exc:
                        combined = f"{message}; {repair_exc}"
                        failures.append(combined)
                        self._emit(
                            "response_validation_failed",
                            error=combined,
                            diagnostic_path=(
                                repair_exc.diagnostic_paths[-1]
                                if repair_exc.diagnostic_paths
                                else diagnostic_path
                            ),
                            validation_path=list(exc.absolute_path),
                        )
                        break
                    schema_repair_used = True
                    request_attempts += repair_attempts
                    schema_repair_response_id = str(
                        repair_envelope.get("id") or ""
                    ) or None

                elapsed = time.monotonic() - started
                successful_request_metrics = _request_consumption_metrics(body)
                successful_response_metrics = _response_consumption_metrics(
                    envelope,
                    output_text=output_text,
                )
                metadata = AzureWebResearchMetadata(
                    provider="azure-openai-responses-web-search",
                    model=str(envelope.get("model") or self.config.deployment),
                    response_id=(
                        str(envelope.get("id")) if envelope.get("id") else None
                    ),
                    status=(
                        str(envelope.get("status"))
                        if envelope.get("status") is not None
                        else None
                    ),
                    usage=(
                        dict(envelope.get("usage"))
                        if isinstance(envelope.get("usage"), Mapping)
                        else {}
                    ),
                    web_search_tool=tool_type,
                    server_tool_evidence=_collect_evidence(envelope),
                    request_attempts=request_attempts,
                    elapsed_seconds=elapsed,
                    schema_repair_used=schema_repair_used,
                    schema_repair_response_id=schema_repair_response_id,
                    schema_repair_elapsed_seconds=schema_repair_elapsed_seconds,
                    request_body_bytes=int(
                        successful_request_metrics["request_json_bytes"]
                    ),
                    estimated_input_tokens=int(
                        successful_request_metrics["estimated_input_tokens"]
                    ),
                    response_body_bytes=int(
                        successful_response_metrics["response_body_bytes"]
                    ),
                    response_text_characters=int(
                        successful_response_metrics[
                            "response_text_characters"
                        ]
                    ),
                )
                self._emit(
                    "request_completed",
                    tool_type=tool_type,
                    request_attempts=request_attempts,
                    elapsed_seconds=elapsed,
                    source_count=len(metadata.server_tool_evidence),
                    **successful_response_metrics,
                )
                return AzureWebResearchResponse(data=data, metadata=metadata)

            # A 2xx response proves the current tool alias is supported. If its
            # output is semantically invalid, trying the preview alias repeats the
            # same paid search and cannot repair the transport contract.
            if received_successful_http:
                break

        joined = " | ".join(failures[-6:]) or "no response details"
        elapsed = time.monotonic() - started
        self._emit("request_failed", elapsed_seconds=elapsed, error=joined)
        raise AzureWebResearchError(
            "Azure Responses web research failed: " + joined,
            diagnostic_paths=diagnostic_paths,
        )


__all__ = [
    "AzureResponsesWebResearchProvider",
    "AzureWebResearchConfig",
    "AzureWebResearchError",
    "AzureWebResearchMetadata",
    "AzureWebResearchResponse",
]
