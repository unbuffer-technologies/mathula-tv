"""Constrained transcript candidate ranking through Azure OpenAI GPT."""

from __future__ import annotations

import os
from typing import Any

from .ai_provider import StructuredAIRequest, create_production_ai_provider
from .autocorrect import AI_RESPONSE_SCHEMA, AutocorrectError, validate_ai_response


AUTOCORRECT_RANKING_PROMPT_VERSION = "gpt-autocorrect-ranking-v1-v13.18.54"

_AUTOCORRECT_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "decisions"],
    "properties": {
        "schema_version": {"const": AI_RESPONSE_SCHEMA},
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "correction_id",
                    "decision",
                    "candidate_id",
                    "confidence",
                    "reason",
                    "question",
                ],
                "properties": {
                    "correction_id": {"type": "string", "minLength": 1},
                    "decision": {
                        "enum": ["ACCEPT_CANDIDATE", "KEEP_AZURE", "ASK_USER"]
                    },
                    "candidate_id": {"type": ["string", "null"]},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "reason": {"type": "string"},
                    "question": {"type": "string"},
                },
            },
        },
    },
}

_SYSTEM_PROMPT = (
    "You are a constrained transcript autocorrect judge. For every supplied "
    "correction, choose exactly one of ACCEPT_CANDIDATE using a supplied "
    "candidate_id, KEEP_AZURE, or ASK_USER. Never rewrite transcript text. "
    "Never invent a spelling or candidate. Use ASK_USER when the phrase may "
    "be emerging slang, a new proper noun, or remains contextually ambiguous."
)


def rank_with_gpt(
    request: dict[str, Any],
    *,
    model: str | None = None,
    max_tokens: int = 8_000,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Rank supplied candidates with GPT; local validation remains authoritative."""

    environment = dict(os.environ)
    environment["MATHULA_TV_AI_PROVIDER"] = "azure-openai-gpt"
    if model:
        environment["AZURE_AI_DEPLOYMENT"] = model
        environment["AZURE_OPENAI_CHAT_DEPLOYMENT"] = model
    try:
        provider = create_production_ai_provider(environment)
        response = provider.complete_structured(
            StructuredAIRequest(
                operation="autocorrect_candidate_ranking",
                payload=request,
                output_schema=_AUTOCORRECT_RESPONSE_SCHEMA,
                prompt_version=AUTOCORRECT_RANKING_PROMPT_VERSION,
                system_prompt=_SYSTEM_PROMPT,
                response_schema_version=AI_RESPONSE_SCHEMA,
                provider_json_schema=True,
                stream_response=False,
                effort="low",
                max_output_tokens=max_tokens,
                max_repairs=0,
            )
        )
        parsed = dict(response.data)
        validated = validate_ai_response(request, parsed)
    except AutocorrectError:
        raise
    except Exception as exc:
        raise AutocorrectError(f"Azure OpenAI GPT ranking failed: {exc}") from exc

    metadata = response.metadata.to_dict()
    metadata.update(
        {
            "provider": "azure-openai-gpt",
            "deployment": provider.config.model,
            "response": parsed,
        }
    )
    return validated, metadata


def rank_with_anthropic(
    request: dict[str, Any],
    *,
    model: str | None = None,
    max_tokens: int = 8_000,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Deprecated API alias; it intentionally routes to GPT, never Anthropic."""

    return rank_with_gpt(request, model=model, max_tokens=max_tokens)


__all__ = ["AUTOCORRECT_RANKING_PROMPT_VERSION", "rank_with_gpt"]
