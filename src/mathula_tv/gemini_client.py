"""Gemini behind the same ``complete()`` call the dubbing agent and its judges use for Azure OpenAI.

The agent keeps a neutral chat-style history (system / user / assistant with tool_calls / tool).  This client translates it to the Gemini ``generateContent``
wire format on every call and translates the answer back to the chat shape the loop already reads, so nothing else in the agent changes.

Two facts the probes of 2026-11-19 on ``gemini-3.8-flash`` established, and the design follows them:
* a ``functionCall`` part carries a ``thoughtSignature`` and the API answers the next turn with HTTP 400 when it is missing, so the model's own parts ride along on the
  assistant message under ``_raw`` and are replayed verbatim (as the Azure Responses client does with its reasoning items);
* ``parametersJsonSchema`` takes the agent's JSON-schema tool definitions as they are, and Google Search can be offered in the same request as function tools.

Two ways to reach the model, the same request body either way:
* the Gemini API with a key (``GEMINI_API_KEY``): ``generativelanguage.googleapis.com``; a key made on the Google Cloud console and restricted to Vertex AI is refused
  there (``API_KEY_SERVICE_BLOCKED`` / ``SERVICE_DISABLED``) and works on the Vertex AI express endpoint instead: set ``GEMINI_API_FLAVOR=vertex``;
* Vertex AI on a Google Cloud project (``GOOGLE_CLOUD_PROJECT``, location ``global`` unless ``GOOGLE_CLOUD_LOCATION``): a bearer token from google-auth when it is
  installed, else ``GOOGLE_ACCESS_TOKEN``, else ``gcloud auth print-access-token``.

Select it with ``MATHULA_AI_PROVIDER=gemini`` (``dub_agent.native_client_config`` reads it); ``GEMINI_MODEL`` names the model, ``GEMINI_THINKING`` the effort.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

DEFAULT_MODEL = "gemini-3.8-flash"
THINKING_LEVELS = ("minimal", "low", "medium", "high")          # low and high were probed live on gemini-3.8-flash; medium and minimal were not


@dataclass(frozen=True)
class GeminiConfig:
    """Shaped like the Azure config the agent code reads (``deployment``, ``reasoning_effort``, ``timeout_seconds``) so existing summaries and traces keep working."""
    model: str = DEFAULT_MODEL
    api_key: str | None = None
    project: str | None = None
    location: str = "global"
    api_flavor: str = "gemini"                  # what the API key belongs to: "gemini" (Google AI Studio key) or "vertex" (a Google Cloud key restricted to Vertex AI)
    max_output_tokens: int = 65_536             # the model's own limit (gemini-3.8-flash accepts 1 to 65,536; 131,072 is refused).  A ceiling, not a spend: only generated tokens are billed.  The
                                                # 16,000 the Azure clients default to cut a full-clip submission off twice (prod-114, finish 'length'): thinking tokens count against it too.
    thinking_level: str | None = "low"          # the Azure GPT side runs at low effort (MATHULA_TV_TRANSLATION_GPT_EFFORT=low), so Gemini does too: a like-for-like comparison and no thinking tokens billed
    timeout_seconds: float = 420.0
    explicit_cache: bool = False                # the agent loop keeps an explicit context cache (gemini_cache); needs a Vertex project, and uses it instead of the API key
    cache_ttl_seconds: int = 1800
    cache_min_tokens: int = 8192

    @property
    def uses_project(self) -> bool:
        """The request goes to the project's Vertex path with a bearer token: always without an API key, and with explicit caching even when a key is set (a key cannot make caches)."""
        return bool(self.project) and (not self.api_key or self.explicit_cache)

    @property
    def deployment(self) -> str:
        return self.model

    @property
    def reasoning_effort(self) -> str | None:
        return self.thinking_level

    @property
    def endpoint(self) -> str:
        return "https://generativelanguage.googleapis.com" if self.api_key and self.api_flavor != "vertex" and not self.uses_project else "https://aiplatform.googleapis.com"

    def project_root(self) -> str:
        return f"https://aiplatform.googleapis.com/v1/projects/{self.project}/locations/{self.location}"

    def model_path(self) -> str:
        return f"projects/{self.project}/locations/{self.location}/publishers/google/models/{self.model}"

    def url(self) -> str:
        if self.uses_project:
            return f"{self.project_root()}/publishers/google/models/{self.model}:generateContent"
        if self.api_key and self.api_flavor == "vertex":          # Vertex AI express mode: the key alone, no project path and no bearer token
            return f"https://aiplatform.googleapis.com/v1/publishers/google/models/{self.model}:generateContent"
        if self.api_key:
            return f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        if not self.project:
            raise RuntimeError("Gemini needs GEMINI_API_KEY, or GOOGLE_CLOUD_PROJECT for Vertex AI")
        return f"https://aiplatform.googleapis.com/v1/projects/{self.project}/locations/{self.location}/publishers/google/models/{self.model}:generateContent"


def gemini_config_from_environment(environ: Mapping[str, str]) -> GeminiConfig:
    level = (environ.get("GEMINI_THINKING") or "low").strip().lower()
    if level not in THINKING_LEVELS:
        raise ValueError(f"GEMINI_THINKING must be one of {THINKING_LEVELS}, got {level!r}")
    flavor = (environ.get("GEMINI_API_FLAVOR") or "gemini").strip().lower()
    if flavor not in ("gemini", "vertex"):
        raise ValueError(f"GEMINI_API_FLAVOR must be 'gemini' or 'vertex', got {flavor!r}")
    return GeminiConfig(model=(environ.get("GEMINI_MODEL") or DEFAULT_MODEL).strip(), api_key=(environ.get("GEMINI_API_KEY") or "").strip() or None,
                        project=(environ.get("GOOGLE_CLOUD_PROJECT") or "").strip() or None, location=(environ.get("GOOGLE_CLOUD_LOCATION") or "global").strip(),
                        api_flavor=flavor, thinking_level=level, max_output_tokens=int(environ.get("GEMINI_MAX_OUTPUT") or 65_536),
                        explicit_cache=(environ.get("GEMINI_EXPLICIT_CACHE") or "").strip().lower() in ("1", "true", "yes", "on") and bool((environ.get("GOOGLE_CLOUD_PROJECT") or "").strip()),
                        cache_ttl_seconds=int(environ.get("GEMINI_CACHE_TTL") or 1800), cache_min_tokens=int(environ.get("GEMINI_CACHE_MIN_TOKENS") or 8192))


class _Token:
    """A Vertex AI bearer token, refreshed shortly before it expires."""

    def __init__(self) -> None:
        self.value: str | None = None
        self.expires = 0.0

    def get(self) -> str:
        if self.value and time.time() < self.expires:
            return self.value
        value = None
        try:
            import google.auth
            import google.auth.transport.requests

            credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
            credentials.refresh(google.auth.transport.requests.Request())
            value = credentials.token
        except Exception:
            value = (os.environ.get("GOOGLE_ACCESS_TOKEN") or "").strip() or None
        if not value:
            for command in (["gcloud", "auth", "print-access-token"], ["powershell", "-NoProfile", "-Command", "gcloud auth print-access-token"]):
                try:
                    lines = subprocess.run(command, capture_output=True, text=True, timeout=60).stdout.strip().splitlines()
                    if lines and not lines[-1].startswith("ERROR"):
                        value = lines[-1].strip()
                        break
                except (OSError, subprocess.SubprocessError):
                    continue
        if not value:
            raise RuntimeError("no Google credentials: install google-auth with a service account, or set GOOGLE_ACCESS_TOKEN, or sign in with gcloud")
        self.value, self.expires = value, time.time() + 45 * 60
        return value


def to_contents(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Chat history -> (Gemini contents, systemInstruction).  Consecutive tool results become ONE user turn of functionResponse parts (the API wants a response
    for every call of the model turn before it); consecutive user texts merge; an assistant message replays its ``_raw`` parts verbatim when it has them."""
    system: list[str] = []
    contents: list[dict[str, Any]] = []
    names: dict[str, str] = {}

    def push(role: str, parts: list[dict[str, Any]]) -> None:
        # Function results of one model turn share ONE user turn; any other user text is a user turn of its own and is never mixed into a function-response turn.  Probed on
        # 2026-11-19 (gemini-3.8-flash): text after function responses in the LAST user turn is refused with 400 "Requests ending with a model turn are not supported"
        # (a harness note after a tool result did exactly that and ended run prod-114); a text turn of its own, text before the responses, and the note inside a response are accepted.
        kind = "response" if parts and "functionResponse" in parts[0] else "text"
        if contents and role == "user" and contents[-1]["role"] == "user" and contents[-1]["_kind"] == kind:
            contents[-1]["parts"].extend(parts)
        else:
            contents.append({"role": role, "parts": parts, "_kind": kind})

    for message in messages:
        role = message["role"]
        if role == "system":
            if message.get("content"):
                system.append(str(message["content"]))
        elif role == "tool":
            call_id = str(message.get("tool_call_id") or "")
            part: dict[str, Any] = {"functionResponse": {"name": names.get(call_id, message.get("name") or "tool"), "response": {"result": message["content"]}}}
            if call_id:
                part["functionResponse"]["id"] = call_id
            push("user", [part])
        elif role == "assistant":
            for call in message.get("tool_calls") or []:
                names[str(call["id"])] = call["function"]["name"]
            if message.get("_raw"):
                contents.append({"role": "model", "parts": message["_raw"], "_kind": "model"})
                continue
            parts: list[dict[str, Any]] = []
            if message.get("content"):
                parts.append({"text": message["content"]})
            for call in message.get("tool_calls") or []:
                try:
                    args = json.loads(call["function"].get("arguments") or "{}")
                except ValueError:
                    args = {}
                parts.append({"functionCall": {"name": call["function"]["name"], "args": args, "id": call["id"]}})
            if parts:
                contents.append({"role": "model", "parts": parts, "_kind": "model"})
        else:
            push("user", [{"text": str(message["content"])}])
    instruction = {"parts": [{"text": "\n\n".join(system)}]} if system else None
    return [{k: v for k, v in c.items() if k != "_kind"} for c in contents], instruction


def to_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Chat-completions tool specs -> one Gemini ``functionDeclarations`` tool (the JSON schema passes through as ``parametersJsonSchema``)."""
    declarations = []
    for tool in tools or []:
        function = tool["function"]
        declaration: dict[str, Any] = {"name": function["name"], "description": function.get("description", "")}
        parameters = function.get("parameters")
        if parameters and (parameters.get("properties") or parameters.get("type") not in (None, "object")):
            declaration["parametersJsonSchema"] = parameters
        declarations.append(declaration)
    return [{"functionDeclarations": declarations}] if declarations else []


def to_chat_shape(response: dict[str, Any]) -> dict[str, Any]:
    """Gemini answer -> the chat shape: choices[0].message (content, tool_calls, ``_raw`` = the model's parts), finish_reason, usage."""
    candidates = response.get("candidates") or []
    if not candidates:
        raise RuntimeError(f"Gemini returned no candidate: {json.dumps(response.get('promptFeedback') or response)[:400]}")
    candidate = candidates[0]
    parts = (candidate.get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts if "text" in p and not p.get("thought"))
    thoughts = "\n".join(p.get("text", "") for p in parts if p.get("thought") and p.get("text"))
    calls = []
    for index, part in enumerate(p for p in parts if "functionCall" in p):
        call = part["functionCall"]
        calls.append({"id": str(call.get("id") or f"call_{index}"), "type": "function",
                      "function": {"name": call["name"], "arguments": json.dumps(call.get("args") or {}, ensure_ascii=False)}})
    message: dict[str, Any] = {"role": "assistant", "content": text or None, "_raw": parts}
    if thoughts:
        message["_reasoning"] = thoughts
    queries = (candidate.get("groundingMetadata") or {}).get("webSearchQueries") or []
    if queries:
        message["_web"] = [{"action": "search", "queries": [str(q)[:200] for q in queries[:4]]}]
    if calls:
        message["tool_calls"] = calls
    finish = str(candidate.get("finishReason") or "STOP")
    usage = response.get("usageMetadata") or {}
    return {
        "choices": [{"message": message, "finish_reason": "tool_calls" if calls else ("length" if finish == "MAX_TOKENS" else finish.lower())}],
        "usage": {"prompt_tokens": int(usage.get("promptTokenCount") or 0),
                  "completion_tokens": int(usage.get("candidatesTokenCount") or 0) + int(usage.get("thoughtsTokenCount") or 0),
                  "cached_tokens": int(usage.get("cachedContentTokenCount") or 0),
                  "thoughts_tokens": int(usage.get("thoughtsTokenCount") or 0)},
    }


class GeminiClient:
    """``complete()`` for the agent loop and the judges, over plain HTTP with the transport-only retries the Azure clients use."""

    def __init__(self, config: GeminiConfig, *, retries: int = 3, sleep: Callable[[float], None] = time.sleep) -> None:
        self.config = config
        self.retries = retries
        self.sleep = sleep
        self.reasoning_summary: str | None = None          # when set, thought summaries come back and are logged to the trace
        self.web_budget = 0                                  # Google Search lookups the agent may make in a run; 0 = not offered
        self.web_used = 0
        self.last_request: dict[str, Any] | None = None
        self.last_explicit: dict[str, Any] | None = None     # what the explicit cache did for the last call (state, cache_tokens): cache_watch reads it
        self._token = _Token()
        self.cache: Any = None
        if config.explicit_cache and config.project:
            from .gemini_cache import ExplicitCache

            self.cache = ExplicitCache(model_path=config.model_path(), create=self._create_cache, delete=self._delete_cache, ttl_seconds=config.cache_ttl_seconds, min_tokens=config.cache_min_tokens)
        self._last_prompt_tokens = 0

    def _create_cache(self, payload: dict[str, Any]) -> dict[str, Any]:
        from .dub_agent import post_with_retry

        return post_with_retry(f"{self.config.project_root()}/cachedContents", self._headers(), payload, timeout=120.0, retries=1, sleep=self.sleep)

    def _delete_cache(self, name: str) -> None:
        import requests

        requests.delete(f"https://aiplatform.googleapis.com/v1/{name}", headers=self._headers(), timeout=30)

    def close(self) -> None:
        """Release the explicit cache (it expires by itself otherwise)."""
        if self.cache is not None:
            self.cache.close()

    def cache_report(self) -> dict[str, Any] | None:
        return self.cache.report() if self.cache is not None else None

    def drain_cache_events(self) -> list[dict[str, Any]]:
        return self.cache.drain_events() if self.cache is not None else []

    def _headers(self) -> dict[str, str]:
        if self.config.api_key and not self.config.uses_project:
            return {"x-goog-api-key": self.config.api_key, "Content-Type": "application/json"}
        headers = {"Authorization": f"Bearer {self._token.get()}", "Content-Type": "application/json"}
        if self.config.project:
            headers["x-goog-user-project"] = self.config.project
        return headers

    def build_body(self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None, json_mode: bool, max_output_tokens: int) -> dict[str, Any]:
        contents, instruction = to_contents(messages)
        generation: dict[str, Any] = {"maxOutputTokens": max_output_tokens}
        thinking: dict[str, Any] = {}
        if self.config.thinking_level:
            thinking["thinkingLevel"] = self.config.thinking_level
        if self.reasoning_summary:
            thinking["includeThoughts"] = True
        if thinking:
            generation["thinkingConfig"] = thinking
        body: dict[str, Any] = {"contents": contents, "generationConfig": generation}
        if instruction:
            body["systemInstruction"] = instruction
        declared = to_tools(tools)
        if self.web_budget - self.web_used > 0:
            declared = declared + [{"googleSearch": {}}]
        if declared:
            body["tools"] = declared
        elif json_mode:                                      # JSON mode and function tools do not mix; the agent only asks for JSON on tool-less calls
            generation["responseMimeType"] = "application/json"
        return body

    def complete_raw(self, messages: list[dict[str, Any]], *, web_search: bool = False, json_mode: bool = False, max_output_tokens: int | None = None) -> dict[str, Any]:
        """The Gemini answer as it came (candidates, groundingMetadata, usageMetadata), for callers that need the citations ``to_chat_shape`` drops.  ``web_search`` grounds it in Google Search."""
        from .dub_agent import REQUEST_TIMEOUT_SECONDS, post_with_retry

        body = self.build_body(messages, tools=None, json_mode=json_mode, max_output_tokens=int(max_output_tokens or self.config.max_output_tokens))
        if web_search:
            body["tools"] = [{"googleSearch": {}}]
            body["generationConfig"].pop("responseMimeType", None)          # JSON mode and Search do not mix: the JSON is asked for in the prompt and checked by the caller
        return post_with_retry(self.config.url(), self._headers(), body, timeout=min(self.config.timeout_seconds, REQUEST_TIMEOUT_SECONDS), retries=self.retries, sleep=self.sleep)

    def complete(self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None = None, json_mode: bool = False, max_output_tokens: int | None = None) -> dict[str, Any]:
        from .dub_agent import REQUEST_TIMEOUT_SECONDS, post_with_retry

        body = self.build_body(messages, tools=tools, json_mode=json_mode, max_output_tokens=int(max_output_tokens or self.config.max_output_tokens))
        self.last_request = {"tools": body.get("tools"), "system": body.get("systemInstruction"), "contents": body["contents"]}          # what the provider receives: cache_watch reads it
        timeout = min(self.config.timeout_seconds, REQUEST_TIMEOUT_SECONDS)
        sent, self.last_explicit = body, None
        if self.cache is not None and body.get("tools"):          # only the agent loop (it declares tools) keeps a cache
            sent, self.last_explicit = self.cache.prepare(body, self._last_prompt_tokens)
        try:
            raw = post_with_retry(self.config.url(), self._headers(), sent, timeout=timeout, retries=self.retries, sleep=self.sleep)
        except RuntimeError as error:
            if sent is body or "HTTP 4" not in str(error):
                raise
            self.cache.request_failed(error)                      # the cache was refused: send the plain request, once
            self.last_explicit = {"state": "fallback"}
            raw = post_with_retry(self.config.url(), self._headers(), body, timeout=timeout, retries=self.retries, sleep=self.sleep)
        shaped = to_chat_shape(raw)
        self._last_prompt_tokens = int(shaped["usage"].get("prompt_tokens") or 0)
        if self.last_explicit and self.last_explicit.get("state") == "used":
            self.cache.observe(shaped["usage"])
        self.web_used += len((shaped["choices"][0]["message"].get("_web") or [{}])[0].get("queries") or [])
        return shaped
