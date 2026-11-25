"""Name and pronunciation web research on Gemini with Google Search grounding, in place of the Azure OpenAI Responses ``web_search`` tool.

``GeminiWebResearchProvider.research_json`` has the interface ``AzureResponsesWebResearchProvider`` had (a system prompt, a payload, an output schema -> data + metadata with the evidence URLs), and it
reuses that module's own answer handling (the JSON wrapper strip, the alias normalisation, the pruning to the schema), so the validators downstream (``validate_researched_corrections``: candidate
binding, same-story, confidence, and "the cited URL was really observed") see exactly the shape they were built for.

* the grounded call returns prose-or-JSON text (Search and JSON mode do not mix), so the schema is in the prompt and the answer is checked here against it; one JSON-only repair call (no search) is made
  when it does not validate, and no more: a hard ceiling on cost and time, as before;
* evidence = the pages Gemini says it grounded on.  Google returns them as redirect links, so each is followed to the page it points at, and both forms are kept: a URL the model cites that the
  search really returned still counts, one it invented does not;
* selected with MATHULA_AI_PROVIDER=gemini by ``create_web_research_provider`` (native_dub and autocorrect_research call it).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import jsonschema
import requests

from .azure_web_research import (
    AzureResponsesWebResearchProvider,
    AzureWebResearchError,
    AzureWebResearchMetadata,
    AzureWebResearchResponse,
    _normalise_research_output,
    _strip_json_wrapper,
)

GEMINI_WEB_RESEARCH_PROVIDER = "gemini-web-research"
MAX_EVIDENCE_PAGES = 12
GROUNDING_HOST = "vertexaisearch.cloud.google.com"
RESEARCH_INSTRUCTION = ("\n\nVerify with Google Search before you answer. Return exactly one JSON object and no markdown, no prose, matching this JSON schema. "
                        "Cite only pages you actually found.\n")


def resolve_redirect(url: str, *, session: Any = None, timeout: float = 8.0) -> str:
    """Where a grounding redirect link really points (the link itself when it cannot be followed)."""
    if GROUNDING_HOST not in url:
        return url
    try:
        response = (session or requests).get(url, allow_redirects=True, stream=True, timeout=timeout)
        final = str(getattr(response, "url", "") or url)
        getattr(response, "close", lambda: None)()
        return final
    except Exception:          # noqa: BLE001 - an unfollowable link is kept as it is
        return url


def grounding_evidence(raw: Mapping[str, Any], *, session: Any = None) -> tuple[dict[str, Any], ...]:
    """The pages the answer was grounded on, as ``url_citation`` entries (the resolved page, and the redirect link when it differed)."""
    candidates = raw.get("candidates") or [{}]
    chunks = ((candidates[0].get("groundingMetadata") or {}).get("groundingChunks") or [])[:MAX_EVIDENCE_PAGES]
    found: list[dict[str, Any]] = []
    seen: set[str] = set()
    for chunk in chunks:
        web = chunk.get("web") if isinstance(chunk, Mapping) else None
        if not isinstance(web, Mapping) or not web.get("uri"):
            continue
        link, title = str(web["uri"]), str(web.get("title") or "")
        for url in dict.fromkeys([resolve_redirect(link, session=session), link]):
            if url not in seen:
                seen.add(url)
                found.append({"type": "url_citation", "url": url, "title": title})
    return tuple(found)


class GeminiWebResearchProvider:
    provider = GEMINI_WEB_RESEARCH_PROVIDER

    def __init__(self, client: Any, *, model: str, max_output_tokens: int = 16_384, session: Any = None, sleep: Callable[[float], None] = time.sleep,
                 progress: Callable[[str], None] | None = None) -> None:
        self.client, self.model, self.max_output_tokens = client, model, int(max_output_tokens)
        self.session, self.sleep, self.progress = session, sleep, progress

    def _emit(self, message: str) -> None:
        if self.progress is not None:
            self.progress(message)

    def _text_and_usage(self, raw: Mapping[str, Any]) -> tuple[str, dict[str, int]]:
        from .gemini_client import to_chat_shape

        shaped = to_chat_shape(dict(raw))
        text = str(((shaped.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
        usage = shaped.get("usage") or {}
        return text, {"input_tokens": int(usage.get("prompt_tokens") or 0), "output_tokens": int(usage.get("completion_tokens") or 0), "cached_tokens": int(usage.get("cached_tokens") or 0)}

    @staticmethod
    def _parse(text: str, output_schema: Mapping[str, Any], payload: Mapping[str, Any]) -> dict[str, Any]:
        data = json.loads(_strip_json_wrapper(text))
        if not isinstance(data, dict):
            raise ValueError("the answer is not a JSON object")
        data, _dropped, _canonicalized = _normalise_research_output(data, output_schema, payload)
        jsonschema.Draft202012Validator(dict(output_schema)).validate(data)
        return data

    def research_json(self, *, system_prompt: str, payload: Mapping[str, Any], output_schema: Mapping[str, Any], diagnostic_dir: Path | None = None,
                      diagnostic_context: Mapping[str, Any] | None = None) -> AzureWebResearchResponse:
        started = time.monotonic()
        schema_text = json.dumps(output_schema, ensure_ascii=False, separators=(",", ":"))
        messages = [{"role": "system", "content": system_prompt.rstrip() + RESEARCH_INSTRUCTION + schema_text},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}]
        self._emit(f"[gemini web research] searching (model={self.model})")
        try:
            raw = self.client.complete_raw(messages, web_search=True, max_output_tokens=self.max_output_tokens)
            text, usage = self._text_and_usage(raw)
        except Exception as exc:
            raise AzureWebResearchError(f"Gemini grounded research failed: {type(exc).__name__}: {str(exc)[:300]}") from exc
        evidence = grounding_evidence(raw, session=self.session)
        attempts, repaired, error = 1, False, ""
        try:
            data = self._parse(text, output_schema, payload)
        except (ValueError, jsonschema.ValidationError) as first:
            error = str(first)[:600]
            repaired = True
            attempts += 1
            try:        # one JSON-only repair of the answer already given: no new search, no new facts
                repair = self.client.complete_raw(
                    [{"role": "system", "content": "Repair the supplied JSON only. Keep its content; do not add facts or sources. Return one JSON object matching the schema.\n" + schema_text},
                     {"role": "user", "content": json.dumps({"invalid_output": text[:20_000], "validation_error": error}, ensure_ascii=False)}],
                    json_mode=True, max_output_tokens=self.max_output_tokens)
                repaired_text, repair_usage = self._text_and_usage(repair)
                usage = {k: usage[k] + repair_usage[k] for k in usage}
                data = self._parse(repaired_text, output_schema, payload)
            except Exception as exc:
                raise AzureWebResearchError(f"Gemini research returned invalid structured output: {error}; repair failed: {type(exc).__name__}") from exc
        metadata = AzureWebResearchMetadata(
            provider=self.provider, model=self.model, response_id=None, status="completed", usage=usage, web_search_tool="googleSearch", server_tool_evidence=evidence,
            request_attempts=attempts, elapsed_seconds=time.monotonic() - started, schema_repair_used=repaired, response_text_characters=len(text))
        self._emit(f"[gemini web research] done: {len(evidence)} page(s) cited, input={usage['input_tokens']:,} output={usage['output_tokens']:,} tokens")
        return AzureWebResearchResponse(data=data, metadata=metadata)


def create_web_research_provider(environ: Mapping[str, str] | None = None, *, progress: Callable[[str], None] | None = None) -> Any:
    """Gemini with Google Search when MATHULA_AI_PROVIDER=gemini, else the Azure Responses web_search provider."""
    env = dict(os.environ if environ is None else environ)
    if (env.get("MATHULA_AI_PROVIDER") or "azure").strip().lower() == "gemini":
        import dataclasses

        from .gemini_client import GeminiClient, gemini_config_from_environment

        config = gemini_config_from_environment(env)
        config = dataclasses.replace(config, thinking_level=(env.get("GEMINI_NATIVE_THINKING") or "low").strip().lower(), explicit_cache=False)
        return GeminiWebResearchProvider(GeminiClient(config), model=config.model, progress=progress)
    return AzureResponsesWebResearchProvider.from_environment() if environ is None else AzureResponsesWebResearchProvider.from_environment(env)
