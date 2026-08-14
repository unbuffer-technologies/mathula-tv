import json

import pytest

from mathula_tv.ai_provider import (
    TURN_REPAIR_SCHEMA,
    AnthropicClaudeProvider,
    AnthropicConfig,
)
from mathula_tv.errors import ClaudeInvalidStructuredOutput


class Response:
    status_code = 200
    headers = {}

    def __init__(self, body):
        self._body = body

    def json(self):
        return self._body


class Session:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return Response(
            {
                "model": "claude-opus-4-8",
                "content": [{"type": "text", "text": json.dumps(self.output)}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 20},
            }
        )


def _provider(output):
    return AnthropicClaudeProvider(
        AnthropicConfig(api_key="test", max_retries=1, max_repairs=0),
        session=Session(output),
        sleep=lambda _delay: None,
    )


def _payload():
    return {
        "source_unit": {
            "unit_id": "unit_0001",
            "source_text": "This sentence is too long for the timing window.",
        },
        "constraints": {
            "protected_verbatim_phrases_must_be_preserved": [],
        },
    }


def test_turn_repair_validates_normalized_single_unit_shape():
    provider = _provider(
        {
            "schema_version": "claude-turn-repair-v1",
            "unit": {
                "unit_id": "unit_0001",
                "translation": "Lo musho mude kakhulu esikhathini esikhona.",
            },
        }
    )

    result = provider.repair_turn(_payload(), output_schema=TURN_REPAIR_SCHEMA)

    assert result.data["schema_version"] == "claude-turn-repair-v1"
    assert result.data["unit"]["unit_id"] == "unit_0001"
    assert result.data["unit"]["faithful_translation"]
    assert result.data["unit"]["spoken_text"]
    assert result.data["unit"]["tts_text"]


def test_turn_repair_rejects_a_different_unit_id():
    provider = _provider(
        {
            "schema_version": "claude-turn-repair-v1",
            "unit": {
                "unit_id": "unit_9999",
                "translation": "Ungashintshi i-ID.",
            },
        }
    )

    with pytest.raises(ClaudeInvalidStructuredOutput, match="translation is missing units"):
        provider.repair_turn(_payload(), output_schema=TURN_REPAIR_SCHEMA)
