"""Gemini as the model of the native pipeline (pass 1, translation, QA, compaction), in place of the Azure OpenAI GPT deployment.

The native pipeline talks to its model through ``FoundryGrokProvider`` (``complete_json``: a system prompt, a payload, an output schema; the answer is parsed and checked against the schema locally,
with a bounded format-repair loop).  Only its transport is Azure-specific (``_send``: one chat-completions request), and the Gemini client already returns the chat-completions shape, so this subclass
replaces that one method and nothing else: the prompts, the schema contract, the local validation and the repair rounds are exactly what ran on GPT.

Selected with MATHULA_AI_PROVIDER=gemini (see foundry_grok.create_native_gpt_provider), the same switch the agent uses.  Gemini's thinking tokens count against the output limit, so the default
output cap here is higher than the GPT one (MATHULA_TV_NATIVE_GPT_MAX_OUTPUT_TOKENS overrides it) and the thinking level is low (GEMINI_NATIVE_THINKING overrides it): these calls are
well-specified extraction and translation steps, not deliberation.
"""
from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .foundry_grok import FoundryGrokProvider

GEMINI_NATIVE_PROVIDER = "gemini-native"
DEFAULT_OUTPUT_TOKENS = 32_768


@dataclass(frozen=True)
class GeminiNativeConfig:
    """The fields ``FoundryGrokProvider`` reads, for a Gemini model (the credentials and endpoint live in the Gemini client's own config)."""

    deployment: str
    api_key: str = "gemini"
    endpoint: str = "gemini"
    timeout_seconds: float = 900.0
    max_retries: int = 3
    max_schema_repairs: int = 2
    max_output_tokens: int = DEFAULT_OUTPUT_TOKENS
    temperature: float = 1.0
    reasoning_effort: str = ""


class GeminiNativeProvider(FoundryGrokProvider):
    def __init__(self, client: Any, config: GeminiNativeConfig, *, progress: Callable[[str], None] | None = None, **kwargs: Any) -> None:
        super().__init__(config, progress=progress, provider_name=GEMINI_NATIVE_PROVIDER, log_label="gemini", **kwargs)
        self.client = client

    def _send(self, messages: list[dict[str, str]], *, operation: str, max_output_tokens: int) -> tuple[dict[str, Any], int]:
        """One JSON-mode Gemini request, answered in the chat-completions shape the parent reads (transport retries are the client's)."""
        self._emit(f"[{self._label}] {operation}: sending Gemini request (model={self.config.deployment})")
        raw = self.client.complete(messages, json_mode=True, max_output_tokens=int(max_output_tokens))
        usage = dict(raw.get("usage") or {})
        raw = {**raw, "model": self.config.deployment,
               "usage": {"prompt_tokens": int(usage.get("prompt_tokens") or 0), "completion_tokens": int(usage.get("completion_tokens") or 0),
                         "prompt_tokens_details": {"cached_tokens": int(usage.get("cached_tokens") or 0)}}}
        return raw, 1


def gemini_native_config_from_environment(environ: Mapping[str, str] | None = None) -> GeminiNativeConfig:
    from .gemini_client import gemini_config_from_environment

    env = dict(os.environ if environ is None else environ)
    return GeminiNativeConfig(
        deployment=gemini_config_from_environment(env).model,
        timeout_seconds=float(env.get("MATHULA_TV_GPT_TIMEOUT_SECONDS", "900")),
        max_retries=int(env.get("MATHULA_TV_GPT_MAX_RETRIES", "3")),
        max_schema_repairs=int(env.get("MATHULA_TV_MAX_TRANSLATION_REPAIRS", "2")),
        max_output_tokens=int(env.get("MATHULA_TV_NATIVE_GPT_MAX_OUTPUT_TOKENS") or DEFAULT_OUTPUT_TOKENS))


def create_gemini_native_provider(environ: Mapping[str, str] | None = None, *, progress: Callable[[str], None] | None = None, client: Any = None, **kwargs: Any) -> GeminiNativeProvider:
    from .gemini_client import GeminiClient, gemini_config_from_environment

    env = dict(os.environ if environ is None else environ)
    if client is None:
        gemini = gemini_config_from_environment(env)
        client = GeminiClient(dataclasses.replace(gemini, thinking_level=(env.get("GEMINI_NATIVE_THINKING") or "low").strip().lower(), explicit_cache=False))
    return GeminiNativeProvider(client, gemini_native_config_from_environment(env), progress=progress, **kwargs)
