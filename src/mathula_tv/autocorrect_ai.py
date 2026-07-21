"""Optional constrained Anthropic ranker for autocorrect candidates."""

from __future__ import annotations

import json
import os
import re
from typing import Any

import requests

from .autocorrect import (
    AutocorrectError,
    validate_ai_response,
)


def _parse_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise AutocorrectError(
            f"AI response is not valid JSON: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise AutocorrectError("AI response must be a JSON object")
    return value


def rank_with_anthropic(
    request: dict[str, Any],
    *,
    model: str,
    max_tokens: int = 8000,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Rank supplied candidates without allowing free-form rewrites."""

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise AutocorrectError("ANTHROPIC_API_KEY is not configured")

    endpoint = os.environ.get(
        "MATHULA_TV_ANTHROPIC_MESSAGES_ENDPOINT",
        "https://api.anthropic.com/v1/messages",
    )
    timeout = int(
        os.environ.get("MATHULA_TV_CLAUDE_TIMEOUT_SECONDS", "900")
    )
    system = (
        "You are a constrained transcript autocorrect judge. "
        "For each correction, select only a supplied candidate ID, "
        "KEEP_AZURE, or ASK_USER. Never rewrite transcript text, never "
        "invent a candidate, and use ASK_USER for new popular terms or "
        "ambiguity. Return JSON only."
    )
    response = requests.post(
        endpoint,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": max_tokens,
            "temperature": 0,
            "system": system,
            "messages": [
                {
                    "role": "user",
                    "content": json.dumps(
                        request,
                        ensure_ascii=False,
                        indent=2,
                    ),
                }
            ],
        },
        timeout=timeout,
    )
    try:
        payload = response.json()
    except ValueError as exc:
        raise AutocorrectError(
            f"Anthropic returned non-JSON HTTP {response.status_code}"
        ) from exc
    if not response.ok:
        message = (
            payload.get("error", {}).get("message")
            if isinstance(payload, dict)
            else None
        )
        raise AutocorrectError(
            f"Anthropic HTTP {response.status_code}: "
            f"{message or 'request failed'}"
        )

    content = payload.get("content", [])
    response_text = "\n".join(
        str(block.get("text", ""))
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ).strip()
    if not response_text:
        raise AutocorrectError("AI returned no text")

    parsed = _parse_json(response_text)
    validated = validate_ai_response(request, parsed)
    metadata = {
        "model": payload.get("model", model),
        "message_id": payload.get("id"),
        "stop_reason": payload.get("stop_reason"),
        "usage": payload.get("usage", {}),
        "response": parsed,
    }
    return validated, metadata
