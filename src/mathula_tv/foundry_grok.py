"""Azure AI Foundry Grok transport for the native dubbing experiment.

This adapter intentionally uses the stable OpenAI v1 Chat Completions API and
never attaches tools.  Web research is therefore impossible unless a human
explicitly installs a reviewed override artifact into the job.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import jsonschema
import requests

from .errors import AIAuthenticationFailure, AIInvalidStructuredOutput, AIRateLimit, AITimeout


FOUNDRY_GROK_PROVIDER = "azure-foundry-grok"
FOUNDRY_GROK_PROVIDER_VERSION = "mathula-foundry-grok-openai-v1-chat-v1"
_TRANSIENT = {408, 409, 425, 429, 500, 502, 503, 504, 529}
_ENDPOINT = re.compile(
    r"https://[A-Za-z0-9][A-Za-z0-9.-]*(?:\.services\.ai\.azure\.com|\.openai\.azure\.com)\Z"
)
# Confirmed real 2026-08-30 against gpt-5.6-sol-1: "none"/"low"/"medium"/"high"/
# "xhigh" are accepted; "minimal" (a value some other OpenAI-family reasoning
# models use) is rejected with a 400 on this deployment.
_VALID_REASONING_EFFORTS = {"none", "low", "medium", "high", "xhigh"}


_GROK_EDITORIAL_COMPACT_PROMPT = r"""You are Mathula TV's audience-development editor.

Use only the supplied evidence. Return exactly ONE JSON object containing ALL of the
following top-level fields together in a single response -- never return a bare array,
and never return only some of these fields on their own:
- opening_boundary_block_id, opening_payoff_block_id, opening_relationship,
  opening_boundary_rationale: the opening cut, using only supplied block IDs. Never cut
  between a meaningful question and its direct response.
- candidates: exactly 3 distinct candidate objects. Each candidate is ONE unified object
  combining its title/hook text AND its supporting evidence -- not two separate lists.
  Each candidate needs its own candidate_id, selected_block_id, evidence_block_ids,
  editorial_angle, hook_strategy, withheld_answer, hook_text, accessible_hook_text,
  title_lead, rationale, confidence, grounded, preserves_attribution, no_sensationalism,
  scores, and human_review_flags -- follow output_contract's exact field names and enums
  for every one of these on every candidate.
- primary_story_angle, caption, search_keywords: English discovery metadata.
- topic_hashtags: exactly 4 distinct story hashtags.
- featured_person_names: up to 2 canonical featured people when materially central.
- human_review_flags: a top-level array, separate from each candidate's own.

Preserve attribution, allegation/uncertainty, names, numbers and evidential limits. Never
invent guilt, corruption, lying, motives, admissions or facts. Prefer a named actor plus a
verified contradiction, document, consequence, decision, or unanswered question over
generic newsroom framing. Each candidate's hook_text and accessible_hook_text are isiZulu;
caption and search_keywords are English. Do not rewrite approved dubbed speech. Do not use
tools or web search. Return JSON only."""


@dataclass(frozen=True)
class FoundryGrokConfig:
    api_key: str = field(repr=False)
    endpoint: str
    deployment: str
    timeout_seconds: float = 900.0
    max_retries: int = 3
    # Confirmed real production case: a schema-invalid first attempt (wrong top-level
    # JSON shape) survived one repair round with a DIFFERENT, still-invalid shape,
    # exhausting the old default of 1 and crashing the whole run. 2 gives a real
    # second chance without being unbounded -- each round costs a genuine network
    # round trip, so this stays a rare safety net, not a routine expectation.
    max_schema_repairs: int = 2
    max_output_tokens: int = 8192
    temperature: float = 0.1
    # "low" by default 2026-08-30: this pipeline's translation/QA/compaction
    # calls are high-volume, well-specified, extractive-shaped tasks (follow
    # the grammar rules, fill the schema) that don't need deep deliberation --
    # confirmed real that gpt-5.6-sol-1 accepts none/low/medium/high/xhigh
    # (NOT "minimal", a 400). Reserve higher effort for tasks that genuinely
    # benefit from it (e.g. SEO title generation, a separate provider path --
    # see ai_provider.py). None when empty (omit the param entirely) so a
    # non-reasoning deployment (e.g. grok-4.3) is unaffected.
    reasoning_effort: str = "low"

    def __post_init__(self) -> None:
        endpoint = self.endpoint.strip().rstrip("/")
        for suffix in ("/openai/v1/chat/completions", "/openai/v1"):
            if endpoint.endswith(suffix):
                endpoint = endpoint[: -len(suffix)]
                break
        object.__setattr__(self, "endpoint", endpoint)
        if not self.api_key:
            raise ValueError("AZURE_AI_KEY is missing")
        if not endpoint or not _ENDPOINT.fullmatch(endpoint):
            raise ValueError(
                "AZURE_AI_ENDPOINT must be an Azure Foundry/OpenAI resource endpoint"
            )
        if not self.deployment.strip():
            raise ValueError(
                "AZURE_FOUNDRY_GROK_DEPLOYMENT is missing; set it to your Grok deployment name"
            )
        if self.timeout_seconds <= 0 or not math.isfinite(self.timeout_seconds):
            raise ValueError("MATHULA_TV_GROK_TIMEOUT_SECONDS must be positive")
        if not 0 <= self.max_retries <= 10:
            raise ValueError("MATHULA_TV_GROK_MAX_RETRIES must be between 0 and 10")
        if not 0 <= self.max_schema_repairs <= 3:
            raise ValueError("MATHULA_TV_GROK_SCHEMA_REPAIRS must be between 0 and 3")
        if self.max_output_tokens <= 0:
            raise ValueError("MATHULA_TV_GROK_MAX_OUTPUT_TOKENS must be positive")
        if not 0 <= self.temperature <= 2:
            raise ValueError("MATHULA_TV_GROK_TEMPERATURE must be between 0 and 2")
        if self.reasoning_effort and self.reasoning_effort not in _VALID_REASONING_EFFORTS:
            raise ValueError(
                "MATHULA_TV_GROK_REASONING_EFFORT must be one of "
                f"{sorted(_VALID_REASONING_EFFORTS)} or empty"
            )

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> "FoundryGrokConfig":
        env = environ if environ is not None else os.environ
        deployment = (
            env.get("AZURE_FOUNDRY_GROK_DEPLOYMENT", "").strip()
            or env.get("MATHULA_TV_GROK_DEPLOYMENT", "").strip()
        )
        return cls(
            api_key=env.get("AZURE_AI_KEY", "").strip(),
            endpoint=env.get("AZURE_AI_ENDPOINT", "").strip(),
            deployment=deployment,
            timeout_seconds=float(env.get("MATHULA_TV_GROK_TIMEOUT_SECONDS", "900")),
            max_retries=int(env.get("MATHULA_TV_GROK_MAX_RETRIES", "3")),
            max_schema_repairs=int(env.get("MATHULA_TV_GROK_SCHEMA_REPAIRS", "2")),
            max_output_tokens=int(env.get("MATHULA_TV_GROK_MAX_OUTPUT_TOKENS", "8192")),
            temperature=float(env.get("MATHULA_TV_GROK_TEMPERATURE", "0.1")),
            reasoning_effort=env.get("MATHULA_TV_GROK_REASONING_EFFORT", "low").strip(),
        )


def _describe_validation_error(exc: Exception) -> str:
    """One-line summary of why structured output failed validation.

    jsonschema.ValidationError's own str() is a multi-line dump of the full failing
    instance and schema -- useless in a progress log. exc.message is the actual
    one-line human-readable reason (e.g. "'self_check' is a required property" or
    "Additional properties are not allowed ('self_check' was unexpected)"), and
    exc.absolute_path names exactly where in the response it failed. Surfacing this
    (previously silently sent only to Grok's own repair request, never to the
    console) is what actually lets a real, systematic failure -- e.g. every call
    needing a repair round because of the same recurring field -- be diagnosed
    instead of just retried.
    """
    if isinstance(exc, jsonschema.ValidationError):
        location = "/".join(str(part) for part in exc.absolute_path) or "<root>"
        return f"{exc.message} (at {location})"
    return str(exc)


@dataclass(frozen=True)
class GrokJSONResponse:
    data: dict[str, Any]
    model: str
    input_tokens: int
    output_tokens: int
    attempts: int
    cached_tokens: int = 0


class GrokOutputTruncated(AIInvalidStructuredOutput):
    """The model stopped because its output-token ceiling was reached."""

    def __init__(
        self,
        message: str,
        *,
        model: str,
        input_tokens: int,
        output_tokens: int,
        attempts: int,
    ) -> None:
        super().__init__(message)
        self.model = model
        self.input_tokens = int(input_tokens)
        self.output_tokens = int(output_tokens)
        self.attempts = int(attempts)


class FoundryGrokProvider:
    """Tool-free Foundry Grok provider with strict local JSON validation."""

    provider = FOUNDRY_GROK_PROVIDER

    def __init__(
        self,
        config: FoundryGrokConfig,
        *,
        session: Any = None,
        sleep: Callable[[float], None] = time.sleep,
        progress: Callable[[str], None] | None = None,
        heartbeat_seconds: float = 15.0,
    ) -> None:
        self.config = config
        self.session = session or requests.Session()
        self.sleep = sleep
        self.progress = progress
        self.heartbeat_seconds = max(1.0, float(heartbeat_seconds))

    def _emit(self, message: str) -> None:
        if self.progress is not None:
            self.progress(message)

    @property
    def endpoint(self) -> str:
        return f"{self.config.endpoint}/openai/v1/chat/completions"

    def complete_json(
        self,
        *,
        operation: str,
        system_prompt: str,
        payload: Mapping[str, Any],
        schema: Mapping[str, Any],
        max_output_tokens: int | None = None,
    ) -> GrokJSONResponse:
        """Return schema-valid JSON. No tools, web search, or hidden fallback."""
        output_contract = _compact_schema_contract(schema)
        user_text = json.dumps(
            {"input": payload, "output_contract": output_contract},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        base_system = (
            system_prompt.rstrip()
            + "\n\nReturn exactly one JSON object and no markdown. "
            + "Follow output_contract exactly. Local code will enforce the full schema."
        )
        messages: list[dict[str, str]] = [
            {"role": "system", "content": base_system},
            {"role": "user", "content": user_text},
        ]
        total_attempts = 0
        last_error: Exception | None = None
        response_model = self.config.deployment
        input_tokens = 0
        output_tokens = 0
        cached_tokens = 0
        for schema_round in range(self.config.max_schema_repairs + 1):
            raw, attempts = self._send(
                messages,
                operation=operation,
                max_output_tokens=max_output_tokens or self.config.max_output_tokens,
            )
            total_attempts += attempts
            response_model = str(raw.get("model") or self.config.deployment)
            usage = raw.get("usage") if isinstance(raw.get("usage"), Mapping) else {}
            input_tokens += int(usage.get("prompt_tokens") or 0)
            output_tokens += int(usage.get("completion_tokens") or 0)
            prompt_tokens_details = usage.get("prompt_tokens_details")
            if isinstance(prompt_tokens_details, Mapping):
                cached_tokens += int(prompt_tokens_details.get("cached_tokens") or 0)
            choices = raw.get("choices")
            first = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], Mapping) else {}
            if str(first.get("finish_reason") or "") == "length":
                self._emit(
                    f"[grok] {operation}: model output limit reached after "
                    f"{output_tokens:,} completion tokens"
                )
                raise GrokOutputTruncated(
                    f"Foundry Grok response hit max_tokens for {operation}",
                    model=response_model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    attempts=total_attempts,
                )
            try:
                content = self._content(raw)
                data = self._parse_json_object(content)
                jsonschema.validate(data, schema)
                cache_note = f", cached={cached_tokens:,}" if cached_tokens else ""
                self._emit(
                    f"[grok] {operation}: response validated; "
                    f"input={input_tokens:,} output={output_tokens:,} tokens{cache_note}, attempts={total_attempts}"
                )
                return GrokJSONResponse(
                    data=data,
                    model=response_model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    attempts=total_attempts,
                    cached_tokens=cached_tokens,
                )
            except (AIInvalidStructuredOutput, jsonschema.ValidationError) as exc:
                last_error = exc
                if schema_round >= self.config.max_schema_repairs:
                    break
                # This is format/schema repair only. It does not ask Grok to
                # reinterpret semantics or use external tools.
                self._emit(
                    f"[grok] {operation}: schema validation failed -- {_describe_validation_error(exc)} -- "
                    f"requesting JSON-only repair ({schema_round + 1}/{self.config.max_schema_repairs})"
                )
                # Token-saving repair: do NOT resend the original (potentially huge)
                # transcript/payload. The prior model output already contains the intended
                # semantics; repair only that output against the compact contract.
                previous_content = self._content(raw)
                repair_user = json.dumps(
                    {
                        "invalid_output": previous_content,
                        "validation_error": str(exc)[:800],
                        "output_contract": output_contract,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "Repair the supplied JSON only. Preserve its semantic content. "
                            "Do not infer new facts. Return one JSON object matching output_contract."
                        ),
                    },
                    {"role": "user", "content": repair_user},
                ]
        raise AIInvalidStructuredOutput(
            f"Foundry Grok returned invalid structured output for {operation}: {last_error}"
        )

    def complete_structured(self, request: Any) -> Any:
        """Compatibility adapter for Mathula structured editorial calls.

        This path deliberately rejects server tools/tool_choice so the native
        Grok pipeline cannot silently re-enable web search. Provider wire JSON
        is validated locally, normalized with the existing Mathula normalizer,
        then validated again against the final application schema.
        """
        if getattr(request, "server_tools", ()) or getattr(request, "tool_choice", None):
            raise ValueError(
                "Foundry Grok native mode does not permit server tools; "
                "web research remains a manual override"
            )
        from .ai_provider import (
            AIResponse,
            AIResponseMetadata,
            AIUsage,
            safe_request_summary,
        )
        from .models import utcnow

        request_timestamp = utcnow()
        provider_schema = (
            getattr(request, "provider_output_schema", None)
            or request.output_schema
        )
        grok_system_prompt = request.system_prompt
        grok_payload = request.payload
        if str(request.operation) == "editorial_package":
            # The production editorial request intentionally contains overlapping
            # transcripts and a very long policy prompt. Grok gets a compact,
            # evidence-grounded wire form; local normalization/validation remains
            # identical to production.
            grok_system_prompt = _GROK_EDITORIAL_COMPACT_PROMPT
            grok_payload = _compact_editorial_payload(request.payload)
        requested_output_tokens = (
            request.max_output_tokens
            if getattr(request, "max_output_tokens", None) is not None
            else self.config.max_output_tokens
        )
        if str(request.operation) == "editorial_package":
            requested_output_tokens = min(int(requested_output_tokens), 3500)
        result = self.complete_json(
            operation=request.operation,
            system_prompt=grok_system_prompt,
            payload=grok_payload,
            schema=provider_schema,
            max_output_tokens=int(requested_output_tokens),
        )
        data = dict(result.data)
        normalizer = getattr(request, "normalizer", None)
        if normalizer is not None:
            data = dict(normalizer(data))
        jsonschema.validate(data, request.output_schema)
        validator = getattr(request, "validator", None)
        if validator is not None:
            validator(data)

        prompt_material = json.dumps(
            {
                "prompt_version": request.prompt_version,
                "system_prompt": request.system_prompt,
                "response_schema_version": request.response_schema_version,
                "output_schema": request.output_schema,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        metadata = AIResponseMetadata(
            provider=self.provider,
            model_requested=self.config.deployment,
            model_returned=result.model,
            prompt_version=request.prompt_version,
            prompt_hash=hashlib.sha256(prompt_material).hexdigest(),
            schema_version=request.response_schema_version,
            request_timestamp=request_timestamp,
            completion_timestamp=utcnow(),
            token_usage=AIUsage(
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
            ),
            stop_reason=None,
            refusal={},
            repair_count=0,
            provider_attempts=result.attempts,
            request_summary=safe_request_summary(request),
            server_tool_evidence=(),
        )
        return AIResponse(data=data, metadata=metadata)

    def _send(
        self,
        messages: list[dict[str, str]],
        *,
        operation: str,
        max_output_tokens: int,
    ) -> tuple[dict[str, Any], int]:
        body: dict[str, Any] = {
            "model": self.config.deployment,
            "messages": messages,
            # "max_completion_tokens", not the older "max_tokens" -- confirmed
            # real 2026-08-30: reasoning-class deployments (gpt-5.6-sol-1) hard
            # -400 on "max_tokens" ("Use 'max_completion_tokens' instead").
            # Also confirmed the newer name works fine against grok-4.3, so
            # this is a safe, unconditional change either way.
            "max_completion_tokens": int(max_output_tokens),
        }
        # Confirmed real 2026-08-30: gpt-5.6-sol-1 hard-400s on any temperature
        # other than its default (1) ("Only the default (1) value is
        # supported") -- a genuine reasoning-model constraint, not a naming
        # difference. grok-4.3 accepts a low temperature fine, so only omit
        # the param (falling back to the API's own default) when configured
        # at/above 1.0; a genuinely low temperature is still sent as before.
        if self.config.temperature < 1.0:
            body["temperature"] = self.config.temperature
        if self.config.reasoning_effort:
            body["reasoning_effort"] = self.config.reasoning_effort
        headers = {
            "api-key": self.config.api_key,
            "Content-Type": "application/json",
            "User-Agent": "mathula-tv-foundry-grok",
        }
        last_exc: Exception | None = None
        for attempt in range(1, self.config.max_retries + 2):
            self._emit(
                f"[grok] {operation}: sending Foundry request "
                f"(attempt {attempt}/{self.config.max_retries + 1}, model={self.config.deployment})"
            )
            started = time.monotonic()
            done = threading.Event()
            heartbeat: threading.Thread | None = None
            if self.progress is not None:
                def _heartbeat() -> None:
                    while not done.wait(self.heartbeat_seconds):
                        elapsed = int(time.monotonic() - started)
                        self._emit(
                            f"[grok] {operation}: still waiting for Foundry response "
                            f"({elapsed}s, attempt {attempt})"
                        )
                heartbeat = threading.Thread(target=_heartbeat, daemon=True)
                heartbeat.start()
            try:
                response = self.session.post(
                    self.endpoint,
                    headers=headers,
                    json=body,
                    timeout=self.config.timeout_seconds,
                )
            except requests.Timeout as exc:
                done.set()
                if heartbeat is not None:
                    heartbeat.join(timeout=0.2)
                self._emit(f"[grok] {operation}: request timed out on attempt {attempt}")
                last_exc = exc
                if attempt > self.config.max_retries:
                    raise AITimeout("Foundry Grok request timed out") from exc
                self.sleep(min(2 ** (attempt - 1), 8))
                continue
            except requests.ConnectionError as exc:
                done.set()
                if heartbeat is not None:
                    heartbeat.join(timeout=0.2)
                self._emit(f"[grok] {operation}: connection error on attempt {attempt}")
                last_exc = exc
                if attempt > self.config.max_retries:
                    raise AIInvalidStructuredOutput("Foundry Grok connection failed") from exc
                self.sleep(min(2 ** (attempt - 1), 8))
                continue
            finally:
                done.set()
                if heartbeat is not None:
                    heartbeat.join(timeout=0.2)

            elapsed = time.monotonic() - started
            self._emit(
                f"[grok] {operation}: HTTP {int(response.status_code)} after {elapsed:.1f}s "
                f"(attempt {attempt})"
            )
            status = int(response.status_code)
            if status in (401, 403):
                raise AIAuthenticationFailure(
                    "Foundry Grok authentication failed; verify AZURE_AI_KEY, AZURE_AI_ENDPOINT and deployment access"
                )
            if status == 429:
                if attempt > self.config.max_retries:
                    raise AIRateLimit("Foundry Grok rate limit exhausted")
                wait = self._retry_after(response, attempt)
                self._emit(f"[grok] {operation}: rate limited; retrying in {wait:.1f}s")
                self.sleep(wait)
                continue
            if status in _TRANSIENT:
                if attempt > self.config.max_retries:
                    raise AIInvalidStructuredOutput(
                        f"Foundry Grok transient HTTP {status} exhausted retries"
                    )
                wait = self._retry_after(response, attempt)
                self._emit(f"[grok] {operation}: transient HTTP {status}; retrying in {wait:.1f}s")
                self.sleep(wait)
                continue
            if status >= 400:
                text = str(getattr(response, "text", ""))[:1500]
                raise AIInvalidStructuredOutput(
                    f"Foundry Grok HTTP {status}: {text}"
                )
            try:
                parsed = response.json()
            except Exception as exc:
                raise AIInvalidStructuredOutput("Foundry Grok returned malformed JSON transport data") from exc
            if not isinstance(parsed, dict):
                raise AIInvalidStructuredOutput("Foundry Grok transport response is not an object")
            return parsed, attempt
        raise AIInvalidStructuredOutput(f"Foundry Grok request failed: {last_exc}")

    @staticmethod
    def _retry_after(response: Any, attempt: int) -> float:
        value = str(getattr(response, "headers", {}).get("Retry-After") or "").strip()
        try:
            if value:
                return max(0.0, min(float(value), 60.0))
        except ValueError:
            pass
        return min(2 ** (attempt - 1), 8)

    @staticmethod
    def _content(response: Mapping[str, Any]) -> str:
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise AIInvalidStructuredOutput("Foundry Grok response has no choices")
        first = choices[0] if isinstance(choices[0], Mapping) else {}
        finish_reason = str(first.get("finish_reason") or "")
        if finish_reason == "length":
            raise AIInvalidStructuredOutput("Foundry Grok response hit max_tokens")
        message = first.get("message") if isinstance(first.get("message"), Mapping) else {}
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise AIInvalidStructuredOutput("Foundry Grok response has no text content")
        return content.strip()

    @staticmethod
    def _parse_json_object(text: str) -> dict[str, Any]:
        value = text.strip()
        if value.startswith("```"):
            value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
            value = re.sub(r"\s*```$", "", value)
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            start, end = value.find("{"), value.rfind("}")
            if start >= 0 and end > start:
                try:
                    parsed = json.loads(value[start : end + 1])
                except json.JSONDecodeError:
                    raise AIInvalidStructuredOutput("Foundry Grok output is not valid JSON") from exc
            else:
                raise AIInvalidStructuredOutput("Foundry Grok output is not valid JSON") from exc
        if not isinstance(parsed, dict):
            raise AIInvalidStructuredOutput("Foundry Grok structured output must be an object")
        return parsed



def _compact_schema_contract(schema: Mapping[str, Any]) -> Any:
    """Return a small human-readable contract while retaining full local validation.

    Sending the complete JSON Schema on every call can cost thousands of prompt
    tokens. Required keys, constants, enums and nesting are sufficient guidance;
    jsonschema still validates the original full schema after the response.
    """
    if "const" in schema:
        return schema["const"]
    enum = schema.get("enum")
    if isinstance(enum, Sequence) and not isinstance(enum, (str, bytes)):
        return {"one_of": list(enum)}
    schema_type = schema.get("type")
    if schema_type == "object" or isinstance(schema.get("properties"), Mapping):
        properties = schema.get("properties") if isinstance(schema.get("properties"), Mapping) else {}
        required = schema.get("required") if isinstance(schema.get("required"), list) else list(properties)
        return {
            str(key): _compact_schema_contract(properties.get(key, {}))
            for key in required
            if key in properties
        }
    if schema_type == "array":
        items = schema.get("items") if isinstance(schema.get("items"), Mapping) else {}
        return [_compact_schema_contract(items)]
    if schema_type == "boolean":
        return "boolean"
    if schema_type in {"integer", "number"}:
        return schema_type
    if schema_type == "string":
        return "string"
    if isinstance(schema.get("anyOf"), list):
        return {"any_of": [_compact_schema_contract(x) for x in schema["anyOf"] if isinstance(x, Mapping)]}
    if isinstance(schema.get("oneOf"), list):
        return {"one_of": [_compact_schema_contract(x) for x in schema["oneOf"] if isinstance(x, Mapping)]}
    return "value"


def _compact_editorial_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Deduplicate the native Grok editorial wire payload.

    The normal production request carries English + rendered bilingual transcript
    copies plus verbose requirements. For Grok native publication we keep one
    English evidence surface, early opening candidates and the cached native
    context ledger/story beats.
    """
    transcript = payload.get("complete_english_transcript")
    transcript_segments = []
    if isinstance(transcript, Mapping):
        raw = transcript.get("segments")
        if isinstance(raw, list):
            for item in raw:
                if not isinstance(item, Mapping):
                    continue
                seg_id = str(item.get("segment_id") or "").strip()
                text = str(item.get("source_text") or "").strip()
                if seg_id and text:
                    transcript_segments.append({
                        "id": seg_id,
                        "spk": str(item.get("speaker_id") or ""),
                        "text": text,
                    })

    candidates = []
    candidate_ids: set[str] = set()
    raw_candidates = payload.get("candidate_boundaries")
    if isinstance(raw_candidates, list):
        for item in raw_candidates:
            if not isinstance(item, Mapping):
                continue
            block_id = str(item.get("block_id") or "").strip()
            if not block_id:
                continue
            candidate_ids.add(block_id)
            candidates.append({
                "id": block_id,
                "start": item.get("start_seconds"),
                "end": item.get("end_seconds"),
            })

    grounded = payload.get("grounded_context") if isinstance(payload.get("grounded_context"), Mapping) else {}
    ledger = grounded.get("native_context_ledger") if isinstance(grounded.get("native_context_ledger"), Mapping) else {}
    beat_ids: set[str] = set()
    beats = ledger.get("story_beats") if isinstance(ledger, Mapping) else None
    if isinstance(beats, list):
        for beat in beats:
            if not isinstance(beat, Mapping):
                continue
            ids = beat.get("segment_ids")
            if isinstance(ids, list):
                beat_ids.update(str(x) for x in ids if str(x).strip())

    # Include story-beat evidence plus one neighbouring segment on each side.
    wanted = set(candidate_ids) | beat_ids
    if wanted and transcript_segments:
        positions = {item["id"]: i for i, item in enumerate(transcript_segments)}
        for seg_id in list(wanted):
            pos = positions.get(seg_id)
            if pos is None:
                continue
            for neighbour in (pos - 1, pos, pos + 1):
                if 0 <= neighbour < len(transcript_segments):
                    wanted.add(transcript_segments[neighbour]["id"])
        evidence = [item for item in transcript_segments if item["id"] in wanted]
    else:
        # Older Pass-1 checkpoints may not have a ledger. Still remove the
        # duplicate rendered isiZulu transcript and send English only once.
        evidence = transcript_segments

    compact: dict[str, Any] = {
        "job_id": payload.get("job_id"),
        "target_language": payload.get("target_language"),
        "source_duration_seconds": payload.get("source_duration_seconds"),
        "maximum_intro_cut_seconds": payload.get("maximum_intro_cut_seconds"),
        "context_ledger": ledger,
        "candidate_boundaries": candidates,
        "evidence_segments": evidence,
    }
    classification = payload.get("classification")
    if isinstance(classification, Mapping) and classification:
        # Classification is usually tiny; keep only shallow scalar/list values.
        compact["classification"] = {
            str(k): (v[:20] if isinstance(v, list) else v)
            for k, v in classification.items()
            if isinstance(v, (str, int, float, bool))
            or (isinstance(v, list) and all(isinstance(x, (str, int, float, bool)) for x in v[:20]))
        }
    return compact


def create_foundry_grok_provider(
    environ: Mapping[str, str] | None = None,
    *,
    session: Any = None,
    sleep: Callable[[float], None] = time.sleep,
    progress: Callable[[str], None] | None = None,
    heartbeat_seconds: float = 15.0,
) -> FoundryGrokProvider:
    return FoundryGrokProvider(
        FoundryGrokConfig.from_environment(environ),
        session=session,
        sleep=sleep,
        progress=progress,
        heartbeat_seconds=heartbeat_seconds,
    )


__all__ = [
    "FOUNDRY_GROK_PROVIDER",
    "FOUNDRY_GROK_PROVIDER_VERSION",
    "FoundryGrokConfig",
    "FoundryGrokProvider",
    "GrokJSONResponse",
    "GrokOutputTruncated",
    "create_foundry_grok_provider",
]
