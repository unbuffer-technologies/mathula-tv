"""Human-mediated ChatGPT provider for resumable Mathula TV AI calls.

Manual mode never calls a paid model endpoint.  It exports the exact structured
request contract to disk, requires a response envelope carrying the matching
request id, then runs the same local schema/normalizer/validator chain used by
the production provider before returning an :class:`AIResponse`.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import jsonschema

from .ai_provider import (
    AIResponse,
    AIResponseMetadata,
    AIUsage,
    AnthropicClaudeProvider,
    StructuredAIRequest,
    _canonical_json,
    _drop_schema_forbidden_properties,
    _normalise_enum_casing,
    _prompt_hash,
    safe_request_summary,
)
from .errors import AIInvalidStructuredOutput, ErrorDescriptor, PipelineError


MANUAL_CHATGPT_PROVIDER = "manual-chatgpt"
MANUAL_REQUEST_SCHEMA_VERSION = "mathula-manual-ai-request-v1"
MANUAL_RESPONSE_SCHEMA_VERSION = "mathula-manual-ai-response-v1"
DEFAULT_MANUAL_MODEL = "gpt-5.6-sol"
_MANUAL_PROVIDER_ALIASES = {
    MANUAL_CHATGPT_PROVIDER,
    "manual",
    "chatgpt-manual",
    "manual-gpt",
}


class ManualAIResponseRequired(PipelineError):
    descriptor = ErrorDescriptor(
        "manual_ai_response_required",
        "ai",
        True,
        "Give the exported request JSON to ChatGPT, install the matching response, and rerun the same command",
    )


def is_manual_ai_provider(value: str | None) -> bool:
    return str(value or "").strip().casefold() in _MANUAL_PROVIDER_ALIASES


def _positive_float(name: str, value: str, default: float) -> float:
    raw = str(value or "").strip()
    if not raw:
        return default
    parsed = float(raw)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _positive_int(name: str, value: str, default: int) -> int:
    raw = str(value or "").strip()
    if not raw:
        return default
    parsed = int(raw)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _nonnegative_int(name: str, value: str, default: int) -> int:
    raw = str(value or "").strip()
    if not raw:
        return default
    parsed = int(raw)
    if parsed < 0:
        raise ValueError(f"{name} must not be negative")
    return parsed


@dataclass(frozen=True)
class ManualChatGPTConfig:
    """Provider-shape-compatible configuration with no cloud credentials."""

    model: str = DEFAULT_MANUAL_MODEL
    provider: str = MANUAL_CHATGPT_PROVIDER
    endpoint: str = ""
    timeout_seconds: float = 900.0
    max_retries: int = 1
    max_repairs: int = 0
    effort: str = "high"
    max_output_tokens: int = 50_000
    translation_effort: str = "low"
    translation_thinking_type: str = "disabled"
    translation_max_output_tokens: int = 100_000
    seo_effort: str = "medium"
    seo_thinking_type: str = "adaptive"
    seo_max_output_tokens: int = 12_000
    hook_effort: str = "low"
    hook_thinking_type: str = "disabled"
    hook_max_output_tokens: int = 12_000
    editorial_effort: str = "high"
    editorial_thinking_type: str = "adaptive"
    editorial_max_output_tokens: int = 12_000
    root_dir: Path = Path("working/manual_ai")

    @classmethod
    def from_environment(
        cls, environ: Mapping[str, str] | None = None
    ) -> "ManualChatGPTConfig":
        env = environ if environ is not None else os.environ
        model = str(env.get("MATHULA_TV_MANUAL_AI_MODEL") or DEFAULT_MANUAL_MODEL).strip()
        if not model:
            model = DEFAULT_MANUAL_MODEL
        root = Path(
            str(env.get("MATHULA_TV_MANUAL_AI_DIR") or "working/manual_ai").strip()
        ).expanduser()
        return cls(
            model=model,
            timeout_seconds=_positive_float(
                "MATHULA_TV_MANUAL_AI_TIMEOUT_SECONDS",
                str(env.get("MATHULA_TV_MANUAL_AI_TIMEOUT_SECONDS") or "900"),
                900.0,
            ),
            max_retries=1,
            max_repairs=_nonnegative_int(
                "MATHULA_TV_MAX_TRANSLATION_REPAIRS",
                str(env.get("MATHULA_TV_MAX_TRANSLATION_REPAIRS") or "0"),
                0,
            ),
            effort=str(env.get("MATHULA_TV_GPT_EFFORT") or "high").strip().lower(),
            max_output_tokens=_positive_int(
                "MATHULA_TV_GPT_MAX_OUTPUT_TOKENS",
                str(env.get("MATHULA_TV_GPT_MAX_OUTPUT_TOKENS") or "50000"),
                50_000,
            ),
            translation_effort=str(
                env.get("MATHULA_TV_TRANSLATION_GPT_EFFORT") or "low"
            ).strip().lower(),
            translation_thinking_type=str(
                env.get("MATHULA_TV_TRANSLATION_GPT_THINKING") or "disabled"
            ).strip().lower(),
            translation_max_output_tokens=_positive_int(
                "MATHULA_TV_TRANSLATION_MAX_OUTPUT_TOKENS",
                str(env.get("MATHULA_TV_TRANSLATION_MAX_OUTPUT_TOKENS") or "100000"),
                100_000,
            ),
            seo_effort=str(env.get("MATHULA_TV_SEO_GPT_EFFORT") or "medium").strip().lower(),
            seo_thinking_type=str(
                env.get("MATHULA_TV_SEO_GPT_THINKING") or "adaptive"
            ).strip().lower(),
            seo_max_output_tokens=_positive_int(
                "MATHULA_TV_SEO_MAX_OUTPUT_TOKENS",
                str(env.get("MATHULA_TV_SEO_MAX_OUTPUT_TOKENS") or "12000"),
                12_000,
            ),
            hook_effort=str(env.get("MATHULA_TV_HOOK_GPT_EFFORT") or "low").strip().lower(),
            hook_thinking_type=str(
                env.get("MATHULA_TV_HOOK_GPT_THINKING") or "disabled"
            ).strip().lower(),
            hook_max_output_tokens=_positive_int(
                "MATHULA_TV_HOOK_MAX_OUTPUT_TOKENS",
                str(env.get("MATHULA_TV_HOOK_MAX_OUTPUT_TOKENS") or "12000"),
                12_000,
            ),
            editorial_effort=str(
                env.get("MATHULA_TV_EDITORIAL_GPT_EFFORT") or "high"
            ).strip().lower(),
            editorial_thinking_type=str(
                env.get("MATHULA_TV_EDITORIAL_GPT_THINKING") or "adaptive"
            ).strip().lower(),
            editorial_max_output_tokens=_positive_int(
                "MATHULA_TV_EDITORIAL_MAX_OUTPUT_TOKENS",
                str(env.get("MATHULA_TV_EDITORIAL_MAX_OUTPUT_TOKENS") or "12000"),
                12_000,
            ),
            root_dir=root,
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_operation_name(operation: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", operation.strip()).strip("-.")
    return cleaned[:80] or "ai-request"


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AIInvalidStructuredOutput(
            f"Manual AI response could not be read as JSON: {path}",
            details={"response_path": str(path), "error_type": type(exc).__name__},
        ) from exc
    if not isinstance(value, dict):
        raise AIInvalidStructuredOutput(
            "Manual AI response envelope must be a JSON object",
            details={"response_path": str(path)},
        )
    return value


def _redact_export_secrets(value: Any) -> Any:
    """Protect accidental credentials while preserving transcript/token-budget data."""

    secret_key = re.compile(
        r"(?i)^(?:api[_-]?key|authorization|password|secret|credential|cookie|access[_-]?token|refresh[_-]?token)$"
    )
    if isinstance(value, Mapping):
        return {
            str(key): ("<redacted>" if secret_key.fullmatch(str(key)) else _redact_export_secrets(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_export_secrets(item) for item in value]
    if isinstance(value, tuple):
        return [_redact_export_secrets(item) for item in value]
    return value


def manual_request_id(request: StructuredAIRequest) -> str:
    summary = safe_request_summary(request)
    material = {
        "operation": request.operation,
        "prompt_hash": _prompt_hash(request),
        "payload_sha256": summary["payload_sha256"],
        "response_schema_version": request.response_schema_version,
        "provider_output_schema": request.provider_output_schema,
        "output_schema": request.output_schema,
    }
    return hashlib.sha256(_canonical_json(material).encode("utf-8")).hexdigest()


def _paths(root: Path, request: StructuredAIRequest) -> dict[str, Path]:
    request_id = manual_request_id(request)
    stem = f"{_safe_operation_name(request.operation)}-{request_id[:20]}"
    return {
        "request": root / "requests" / f"{stem}.json",
        "response": root / "responses" / f"{stem}.json",
        "accepted": root / "accepted" / f"{stem}.json",
    }


def _request_envelope(
    request: StructuredAIRequest,
    *,
    request_id: str,
    response_path: Path,
    created_at: str,
    model: str,
) -> dict[str, Any]:
    provider_schema = request.provider_output_schema or request.output_schema
    return {
        "schema_version": MANUAL_REQUEST_SCHEMA_VERSION,
        "request_id": request_id,
        "created_at": created_at,
        "provider": MANUAL_CHATGPT_PROVIDER,
        "model": model,
        "operation": request.operation,
        "prompt_version": request.prompt_version,
        "response_schema_version": request.response_schema_version,
        "prompt_hash": _prompt_hash(request),
        "request_summary": safe_request_summary(request),
        "instructions": (
            "Give this complete JSON file to ChatGPT. ChatGPT must follow the system/user "
            "instructions and return a mathula-manual-ai-response-v1 envelope whose request_id "
            "matches exactly and whose result matches provider_response_schema. Do not edit the request_id."
        ),
        "expected_response_path": str(response_path),
        "response_envelope_contract": {
            "schema_version": MANUAL_RESPONSE_SCHEMA_VERSION,
            "request_id": request_id,
            "result": "<JSON object matching provider_response_schema>",
            "server_tool_evidence": "<optional JSON array>",
        },
        "system_prompt": request.system_prompt,
        "cacheable_user_prefix": request.cacheable_user_prefix,
        "user_prompt": request.user_prompt,
        "payload": _redact_export_secrets(copy.deepcopy(dict(request.payload))),
        "provider_response_schema": copy.deepcopy(dict(provider_schema)),
        "final_output_schema": copy.deepcopy(dict(request.output_schema)),
        "server_tools": _redact_export_secrets(copy.deepcopy(list(request.server_tools))),
        "tool_choice": _redact_export_secrets(copy.deepcopy(request.tool_choice)),
        "generation_hints": {
            "effort": request.effort,
            "thinking_type": request.thinking_type,
            "max_output_tokens": request.max_output_tokens,
            "provider_json_schema": request.provider_json_schema,
        },
    }


def _validate_manual_result(
    request: StructuredAIRequest,
    raw_result: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    value: Any = copy.deepcopy(raw_result)
    dropped_property_paths: list[str] = []
    application_normalizer_applied = False
    validation_stage = "provider_schema"
    try:
        if request.provider_output_schema is not None:
            value, dropped = _drop_schema_forbidden_properties(
                value, request.provider_output_schema
            )
            dropped_property_paths.extend(dropped)
            value = _normalise_enum_casing(value, request.provider_output_schema)
            jsonschema.validate(value, dict(request.provider_output_schema))
        if request.normalizer is not None:
            validation_stage = "normalize"
            before = copy.deepcopy(value)
            value = request.normalizer(value)
            application_normalizer_applied = value != before
        validation_stage = "schema"
        value, dropped = _drop_schema_forbidden_properties(value, request.output_schema)
        dropped_property_paths.extend(dropped)
        value = _normalise_enum_casing(value, request.output_schema)
        jsonschema.validate(value, dict(request.output_schema))
        if request.validator is not None:
            validation_stage = "validator"
            request.validator(value)
    except (
        AIInvalidStructuredOutput,
        jsonschema.ValidationError,
        ValueError,
        KeyError,
        TypeError,
    ) as exc:
        raise AIInvalidStructuredOutput(
            f"Manual ChatGPT response failed {validation_stage}: {type(exc).__name__}: {str(exc)[:500]}",
            details={
                "provider": MANUAL_CHATGPT_PROVIDER,
                "operation": request.operation,
                "prompt_version": request.prompt_version,
                "schema_version": request.response_schema_version,
                "validation_stage": validation_stage,
                "validation_error": f"{type(exc).__name__}: {str(exc)[:500]}",
                "candidate_output": raw_result,
            },
        ) from exc
    if not isinstance(value, dict):
        raise AIInvalidStructuredOutput(
            "Manual ChatGPT normalized response must be a JSON object"
        )
    return value, {
        "dropped_property_count": len(dropped_property_paths),
        "dropped_property_paths": dropped_property_paths[:20],
        "application_normalizer_applied": application_normalizer_applied,
    }


@dataclass
class ManualChatGPTProvider(AnthropicClaudeProvider):
    """Disk-backed manual provider that reuses all Mathula operation builders."""

    config: ManualChatGPTConfig
    provider: str = field(default=MANUAL_CHATGPT_PROVIDER, init=False)

    def __post_init__(self) -> None:
        self.provider = MANUAL_CHATGPT_PROVIDER
        self.session = None

    @property
    def provider_display_name(self) -> str:
        return "Manual ChatGPT"

    @property
    def url(self) -> str:
        return "manual://chatgpt"

    def with_request_options(
        self,
        *,
        timeout_seconds: float | None = None,
        max_retries: int | None = None,
        event_callback: Callable[[dict[str, Any]], None] | None = None,
        request_guard: Callable[[dict[str, Any]], None] | None = None,
    ) -> "ManualChatGPTProvider":
        config = replace(
            self.config,
            timeout_seconds=(
                self.config.timeout_seconds
                if timeout_seconds is None
                else float(timeout_seconds)
            ),
            max_retries=1,
        )
        return ManualChatGPTProvider(
            config=config,
            sleep=self.sleep,
            clock=self.clock,
            event_callback=event_callback,
            request_guard=request_guard,
            capability_state=self.capability_state,
        )

    def complete_structured(self, request: StructuredAIRequest) -> AIResponse:
        root = self.config.root_dir
        paths = _paths(root, request)
        request_id = manual_request_id(request)
        created_at = _utc_now()
        if paths["request"].is_file():
            try:
                existing = json.loads(paths["request"].read_text(encoding="utf-8"))
                if isinstance(existing, Mapping) and existing.get("created_at"):
                    created_at = str(existing["created_at"])
            except (OSError, json.JSONDecodeError):
                pass
        envelope = _request_envelope(
            request,
            request_id=request_id,
            response_path=paths["response"],
            created_at=created_at,
            model=self.config.model,
        )
        _atomic_write_json(paths["request"], envelope)

        if not paths["response"].is_file():
            self._emit_event(
                event="manual_ai_response_required",
                operation=request.operation,
                request_id=request_id,
                request_path=str(paths["request"]),
                response_path=str(paths["response"]),
            )
            raise ManualAIResponseRequired(
                "Manual ChatGPT response required. "
                f"Request: {paths['request']}. "
                f"Install the matching response at {paths['response']} and rerun the same command.",
                details={
                    "provider": MANUAL_CHATGPT_PROVIDER,
                    "operation": request.operation,
                    "request_id": request_id,
                    "request_path": str(paths["request"]),
                    "response_path": str(paths["response"]),
                },
            )

        response_envelope = _read_json_object(paths["response"])
        if response_envelope.get("schema_version") != MANUAL_RESPONSE_SCHEMA_VERSION:
            raise AIInvalidStructuredOutput(
                f"Manual response schema_version must be {MANUAL_RESPONSE_SCHEMA_VERSION}",
                details={"response_path": str(paths["response"])},
            )
        if str(response_envelope.get("request_id") or "") != request_id:
            raise AIInvalidStructuredOutput(
                "Manual response request_id does not match the pending request",
                details={
                    "expected_request_id": request_id,
                    "received_request_id": response_envelope.get("request_id"),
                    "response_path": str(paths["response"]),
                },
            )
        raw_result = response_envelope.get("result")
        if not isinstance(raw_result, dict):
            raise AIInvalidStructuredOutput(
                "Manual response result must be a JSON object",
                details={"response_path": str(paths["response"])},
            )

        # Count a manual completion only when a matching response is actually
        # being consumed; merely exporting a request must not exhaust a call budget.
        self._run_request_guard(
            event="request_attempt",
            attempt=1,
            max_retries=1,
            timeout_seconds=self.config.timeout_seconds,
            manual=True,
        )
        self._emit_event(
            event="request_attempt",
            attempt=1,
            max_retries=1,
            timeout_seconds=self.config.timeout_seconds,
            manual=True,
            operation=request.operation,
        )

        value, normalization = _validate_manual_result(request, raw_result)
        completed_at = _utc_now()
        evidence_value = response_envelope.get("server_tool_evidence") or []
        evidence = tuple(
            dict(item) for item in evidence_value if isinstance(item, Mapping)
        ) if isinstance(evidence_value, list) else ()
        metadata = AIResponseMetadata(
            provider=MANUAL_CHATGPT_PROVIDER,
            model_requested=self.config.model,
            model_returned=str(response_envelope.get("model") or self.config.model),
            prompt_version=request.prompt_version,
            prompt_hash=_prompt_hash(request),
            schema_version=request.response_schema_version,
            request_timestamp=created_at,
            completion_timestamp=completed_at,
            token_usage=AIUsage(),
            stop_reason="manual_handoff",
            refusal={"detected": False, "reason": None},
            repair_count=0,
            provider_attempts=1,
            request_summary={
                **safe_request_summary(request),
                "manual_ai": {
                    "request_id": request_id,
                    "request_path": str(paths["request"]),
                    "response_path": str(paths["response"]),
                    "accepted_path": str(paths["accepted"]),
                },
                "schema_normalization": normalization,
                "wire_requests": [],
            },
            server_tool_evidence=evidence,
        )
        accepted = {
            "schema_version": "mathula-manual-ai-accepted-v1",
            "request_id": request_id,
            "accepted_at": completed_at,
            "response_sha256": hashlib.sha256(
                paths["response"].read_bytes()
            ).hexdigest(),
            "artifact": AIResponse(data=value, metadata=metadata).to_artifact(),
        }
        _atomic_write_json(paths["accepted"], accepted)
        self._emit_event(
            event="response_usage",
            operation=request.operation,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            provider_attempts=1,
            manual=True,
        )
        return AIResponse(data=value, metadata=metadata)


def create_manual_chatgpt_provider(
    environ: Mapping[str, str] | None = None,
    *,
    sleep: Callable[[float], None] | None = None,
) -> ManualChatGPTProvider:
    kwargs: dict[str, Any] = {"config": ManualChatGPTConfig.from_environment(environ)}
    if sleep is not None:
        kwargs["sleep"] = sleep
    return ManualChatGPTProvider(**kwargs)


def manual_ai_root_for_job(work_dir: Path, job_id: str) -> Path:
    return Path(work_dir) / "jobs" / str(job_id) / "manual_ai"


def pending_manual_requests(root: Path) -> list[dict[str, Any]]:
    request_dir = root / "requests"
    response_dir = root / "responses"
    pending: list[dict[str, Any]] = []
    if not request_dir.is_dir():
        return pending
    for request_path in sorted(request_dir.glob("*.json")):
        try:
            request = json.loads(request_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        response_path = response_dir / request_path.name
        if response_path.is_file():
            continue
        pending.append(
            {
                "request_id": request.get("request_id"),
                "operation": request.get("operation"),
                "request_path": str(request_path),
                "expected_response_path": str(response_path),
                "created_at": request.get("created_at"),
            }
        )
    return pending


def install_manual_response(root: Path, response_json: Path) -> dict[str, Any]:
    response = _read_json_object(response_json)
    if response.get("schema_version") != MANUAL_RESPONSE_SCHEMA_VERSION:
        raise ValueError(
            f"Response schema_version must be {MANUAL_RESPONSE_SCHEMA_VERSION}"
        )
    request_id = str(response.get("request_id") or "").strip()
    if not request_id:
        raise ValueError("Manual response request_id is required")
    request_dir = root / "requests"
    matches: list[Path] = []
    for request_path in request_dir.glob("*.json") if request_dir.is_dir() else ():
        try:
            request = json.loads(request_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if str(request.get("request_id") or "") == request_id:
            matches.append(request_path)
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one pending request for {request_id}, found {len(matches)}"
        )
    destination = root / "responses" / matches[0].name
    _atomic_write_json(destination, response)
    return {
        "installed": True,
        "request_id": request_id,
        "operation": json.loads(matches[0].read_text(encoding="utf-8")).get("operation"),
        "response_path": str(destination),
    }


__all__ = [
    "DEFAULT_MANUAL_MODEL",
    "MANUAL_CHATGPT_PROVIDER",
    "MANUAL_REQUEST_SCHEMA_VERSION",
    "MANUAL_RESPONSE_SCHEMA_VERSION",
    "ManualAIResponseRequired",
    "ManualChatGPTConfig",
    "ManualChatGPTProvider",
    "create_manual_chatgpt_provider",
    "install_manual_response",
    "is_manual_ai_provider",
    "manual_ai_root_for_job",
    "manual_request_id",
    "pending_manual_requests",
]
