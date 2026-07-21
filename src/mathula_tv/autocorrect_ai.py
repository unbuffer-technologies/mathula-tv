"""Constrained Claude candidate ranking through Azure AI Foundry."""

from __future__ import annotations

import json
import os
import re
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import requests

from .autocorrect import (
    AutocorrectError,
    validate_ai_response,
)


def _parse_json(text: str) -> dict[str, Any]:
    stripped = text.strip()

    if stripped.startswith("```"):
        stripped = re.sub(
            r"^```(?:json)?\s*",
            "",
            stripped,
        )
        stripped = re.sub(
            r"\s*```$",
            "",
            stripped,
        )

    try:
        value = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise AutocorrectError(
            f"Claude response is not valid JSON: {exc}"
        ) from exc

    if not isinstance(value, dict):
        raise AutocorrectError(
            "Claude response must be a JSON object"
        )

    return value


def _foundry_messages_endpoint(raw_endpoint: str) -> str:
    """Convert any Foundry resource URL to its Claude Messages URL."""

    endpoint = raw_endpoint.strip()
    if not endpoint:
        raise AutocorrectError(
            "MATHULA_TV_FOUNDRY_CLAUDE_ENDPOINT is empty"
        )

    parsed = urlsplit(endpoint)

    if parsed.scheme not in {"http", "https"}:
        raise AutocorrectError(
            "Foundry Claude endpoint must be an HTTP or HTTPS URL"
        )

    if not parsed.netloc:
        raise AutocorrectError(
            "Foundry Claude endpoint has no hostname"
        )

    path = parsed.path.rstrip("/")

    if path.endswith("/anthropic/v1/messages"):
        final_path = path
    elif path.endswith("/anthropic"):
        final_path = f"{path}/v1/messages"
    elif "/anthropic/" in path:
        prefix = path.split("/anthropic/", 1)[0]
        final_path = f"{prefix}/anthropic/v1/messages"
    else:
        # Environment values sometimes contain the resource endpoint,
        # project endpoint, or deployment endpoint. Claude's Messages API
        # always lives at the resource origin under /anthropic/v1/messages.
        final_path = "/anthropic/v1/messages"

    return urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            final_path,
            "",
            "",
        )
    )


def _configuration(
    requested_model: str,
) -> tuple[str, str, str, str]:
    """Return provider, endpoint, API key and deployment."""

    foundry_endpoint = os.environ.get(
        "MATHULA_TV_FOUNDRY_CLAUDE_ENDPOINT"
    )
    foundry_key = os.environ.get(
        "MATHULA_TV_FOUNDRY_CLAUDE_API_KEY"
    )
    foundry_deployment = os.environ.get(
        "MATHULA_TV_FOUNDRY_CLAUDE_DEPLOYMENT"
    )

    foundry_values = {
        "MATHULA_TV_FOUNDRY_CLAUDE_ENDPOINT": foundry_endpoint,
        "MATHULA_TV_FOUNDRY_CLAUDE_API_KEY": foundry_key,
        "MATHULA_TV_FOUNDRY_CLAUDE_DEPLOYMENT": foundry_deployment,
    }

    if any(foundry_values.values()):
        missing = [
            name
            for name, value in foundry_values.items()
            if not value
        ]

        if missing:
            raise AutocorrectError(
                "Incomplete Azure AI Foundry Claude configuration: "
                + ", ".join(missing)
            )

        return (
            "azure_ai_foundry",
            _foundry_messages_endpoint(
                str(foundry_endpoint)
            ),
            str(foundry_key),
            str(foundry_deployment),
        )

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

    if anthropic_key:
        endpoint = os.environ.get(
            "MATHULA_TV_ANTHROPIC_MESSAGES_ENDPOINT",
            "https://api.anthropic.com/v1/messages",
        )

        return (
            "anthropic",
            endpoint,
            anthropic_key,
            requested_model,
        )

    raise AutocorrectError(
        "Claude is not configured. Configure the three "
        "MATHULA_TV_FOUNDRY_CLAUDE_* variables or "
        "ANTHROPIC_API_KEY."
    )


def rank_with_anthropic(
    request: dict[str, Any],
    *,
    model: str,
    max_tokens: int = 8000,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Rank supplied candidates without permitting transcript rewrites."""

    provider, endpoint, api_key, deployment = _configuration(
        model
    )

    timeout = int(
        os.environ.get(
            "MATHULA_TV_CLAUDE_TIMEOUT_SECONDS",
            "900",
        )
    )

    system = (
        "You are a constrained transcript autocorrect judge. "
        "For every supplied correction, choose exactly one of: "
        "ACCEPT_CANDIDATE using a supplied candidate_id, "
        "KEEP_AZURE, or ASK_USER. "
        "Never rewrite transcript text. "
        "Never invent a spelling or candidate. "
        "Use ASK_USER when the phrase may be emerging slang, "
        "a new proper noun, or remains contextually ambiguous. "
        "Return valid JSON only, exactly matching the requested schema."
    )

    try:
        response = requests.post(
            endpoint,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
                "accept": "application/json",
            },
            json={
                "model": deployment,
                "max_tokens": max_tokens,
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
    except requests.RequestException as exc:
        raise AutocorrectError(
            f"Claude request failed: {exc}"
        ) from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise AutocorrectError(
            "Claude returned a non-JSON response with "
            f"HTTP status {response.status_code}"
        ) from exc

    if not response.ok:
        error_message = None

        if isinstance(payload, dict):
            error = payload.get("error")

            if isinstance(error, dict):
                error_message = (
                    error.get("message")
                    or error.get("detail")
                )
            elif error:
                error_message = str(error)

            error_message = (
                error_message
                or payload.get("message")
                or payload.get("detail")
            )

        raise AutocorrectError(
            f"Claude HTTP {response.status_code}: "
            f"{error_message or 'request failed'}"
        )

    content = payload.get("content")

    if not isinstance(content, list):
        raise AutocorrectError(
            "Claude response has no content list"
        )

    response_text = "\n".join(
        str(block.get("text", ""))
        for block in content
        if isinstance(block, dict)
        and block.get("type") == "text"
    ).strip()

    if not response_text:
        raise AutocorrectError(
            "Claude returned no text response"
        )

    parsed = _parse_json(response_text)
    validated = validate_ai_response(
        request,
        parsed,
    )

    metadata = {
        "provider": provider,
        "endpoint_host": urlsplit(endpoint).netloc,
        "deployment": deployment,
        "model_returned": payload.get("model"),
        "message_id": payload.get("id"),
        "stop_reason": payload.get("stop_reason"),
        "usage": payload.get("usage", {}),
        "response": parsed,
    }

    return validated, metadata
