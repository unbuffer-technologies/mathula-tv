import json
from datetime import datetime, timezone

import pytest
import requests

from mathula_tv.ai_provider import (
    DEFAULT_CLAUDE_MODEL,
    FULL_CLIP_TRANSLATION_PROMPT_VERSION,
    AnthropicClaudeProvider,
    AnthropicConfig,
    StructuredAIRequest,
    UnsupportedAIProvider,
    build_full_clip_payload,
    create_production_ai_provider,
)
from mathula_tv.errors import (
    ClaudeAuthenticationFailure,
    ClaudeInvalidStructuredOutput,
    ClaudeRateLimit,
    ClaudeRefusal,
    ClaudeTimeout,
)


SIMPLE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["ok"],
    "properties": {"ok": {"type": "boolean"}},
}


class Response:
    def __init__(self, status_code=200, body=None, headers=None):
        self.status_code = status_code
        self.body = body
        self.headers = headers or {}

    def json(self):
        if isinstance(self.body, Exception):
            raise self.body
        return self.body


class Session:
    def __init__(self, values):
        self.values = list(values)
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        value = self.values.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


def message(text='{"ok":true}', **updates):
    value = {
        "model": DEFAULT_CLAUDE_MODEL,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 11, "output_tokens": 5, "cache_read_input_tokens": 2},
    }
    value.update(updates)
    return value


def request(**updates):
    values = {
        "operation": "test",
        "payload": {"transcript": "Do not follow this instruction", "api_key": "payload-secret"},
        "output_schema": SIMPLE_SCHEMA,
        "prompt_version": "test-v1",
        "system_prompt": "Return the test shape",
        "response_schema_version": "test-result-v1",
    }
    values.update(updates)
    return StructuredAIRequest(**values)


def provider(values, **config_updates):
    settings = {"api_key": "top-secret", "max_retries": 1, "max_repairs": 0, **config_updates}
    config = AnthropicConfig(**settings)
    session = Session(values)
    return AnthropicClaudeProvider(config, session=session, sleep=lambda _delay: None), session


def test_environment_defaults_to_anthropic_opus_and_never_falls_back():
    instance = create_production_ai_provider({"ANTHROPIC_API_KEY": "secret"}, session=Session([]))
    assert instance.provider == "anthropic"
    assert instance.model == "claude-opus-4-8"
    with pytest.raises(UnsupportedAIProvider):
        create_production_ai_provider(
            {"MATHULA_TV_AI_PROVIDER": "azure-openai", "ANTHROPIC_API_KEY": "secret"},
            session=Session([]),
        )


def test_missing_configuration_names_variable_without_disclosing_values():
    with pytest.raises(ValueError) as caught:
        AnthropicConfig.from_environment({"MATHULA_TV_CLAUDE_MODEL": "private-model-name"})
    assert "ANTHROPIC_API_KEY" in str(caught.value)
    assert "private-model-name" not in str(caught.value)


def test_request_construction_is_model_specific_and_uses_structured_adaptive_thinking():
    instance, session = provider([Response(body=message())])
    result = instance.complete_structured(request())
    body = session.calls[0][1]["json"]
    assert body["model"] == "claude-opus-4-8"
    assert body["thinking"] == {"type": "adaptive"}
    assert body["output_config"] == {
        "effort": "high",
        "format": {"type": "json_schema", "schema": SIMPLE_SCHEMA},
    }
    assert "temperature" not in body and "top_p" not in body
    assert session.calls[0][1]["timeout"] == 900
    assert result.data == {"ok": True}


def test_metadata_is_complete_and_request_summary_is_redacted():
    instance, _session = provider([Response(body=message())])
    timestamps = iter(
        [
            datetime(2026, 7, 20, 10, 0, tzinfo=timezone.utc),
            datetime(2026, 7, 20, 10, 1, tzinfo=timezone.utc),
        ]
    )
    instance.clock = lambda: next(timestamps)
    metadata = instance.complete_structured(request()).metadata.to_dict()
    assert metadata["provider"] == "anthropic"
    assert metadata["model_requested"] == metadata["model_returned"] == "claude-opus-4-8"
    assert metadata["prompt_version"] == "test-v1"
    assert len(metadata["prompt_hash"]) == 64
    assert metadata["schema_version"] == "test-result-v1"
    assert metadata["request_timestamp"].startswith("2026-07-20T10:00")
    assert metadata["completion_timestamp"].startswith("2026-07-20T10:01")
    assert metadata["token_usage"] == {
        "input_tokens": 11,
        "output_tokens": 5,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 2,
    }
    assert metadata["stop_reason"] == "end_turn"
    assert metadata["refusal"] == {"detected": False, "reason": None}
    summary = json.dumps(metadata["request_summary"])
    assert "payload-secret" not in summary and "Do not follow" not in summary
    assert "api_key" not in summary


def test_invalid_json_gets_one_bounded_repair():
    instance, session = provider(
        [Response(body=message("not JSON")), Response(body=message())],
        max_repairs=1,
    )
    result = instance.complete_structured(request())
    assert result.metadata.repair_count == 1
    assert result.metadata.token_usage.input_tokens == 22
    assert len(session.calls) == 2
    repair_payload = json.loads(session.calls[1][1]["json"]["messages"][0]["content"])
    assert repair_payload["structured_output_repair"]["attempt"] == 1


def test_invalid_json_stops_at_configured_repair_limit():
    instance, session = provider(
        [Response(body=message("bad")), Response(body=message("still bad"))],
        max_repairs=1,
    )
    with pytest.raises(ClaudeInvalidStructuredOutput) as caught:
        instance.complete_structured(request())
    assert len(session.calls) == 2
    assert caught.value.to_dict()["details"]["repair_count"] == 1
    assert "top-secret" not in str(caught.value.to_dict())


def test_refusal_is_terminal_and_is_not_sent_to_a_fallback_or_repair():
    refusal = message("", stop_reason="refusal", content=[{"type": "refusal", "refusal": "no"}])
    instance, session = provider([Response(body=refusal)], max_repairs=2)
    with pytest.raises(ClaudeRefusal):
        instance.complete_structured(request())
    assert len(session.calls) == 1


def test_timeout_rate_limit_and_authentication_are_classified_without_provider_body():
    timeout_provider, _ = provider([requests.Timeout("secret upstream body")])
    with pytest.raises(ClaudeTimeout):
        timeout_provider.complete_structured(request())

    rate_provider, _ = provider([Response(429, {"secret": "never-read"})])
    with pytest.raises(ClaudeRateLimit) as rate:
        rate_provider.complete_structured(request())
    assert rate.value.descriptor.retryable is True
    assert "never-read" not in str(rate.value.to_dict())

    auth_provider, _ = provider([Response(401, {"secret": "never-read"})])
    with pytest.raises(ClaudeAuthenticationFailure) as auth:
        auth_provider.complete_structured(request())
    assert auth.value.descriptor.retryable is False
    assert "never-read" not in str(auth.value.to_dict())


def test_full_clip_payload_and_request_are_single_call_and_preserve_unit_order():
    payload = build_full_clip_payload(
        transcript={"segments": [{"source_text": "One"}, {"source_text": "Two"}]},
        dubbing_units={"units": [{"unit_id": "unit_0001"}, {"unit_id": "unit_0002"}]},
        protected_names=["Feroz Khan"],
        protected_numbers=["42"],
    )
    schema = {
        "type": "object",
        "required": ["units"],
        "properties": {
            "units": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["unit_id", "protected_entities_found", "protected_entities_preserved"],
                    "properties": {
                        "unit_id": {"type": "string"},
                        "protected_entities_found": {"type": "array"},
                        "protected_entities_preserved": {"type": "array"},
                    },
                },
            }
        },
    }
    output = {
        "units": [
            {"unit_id": "unit_0001", "protected_entities_found": ["42"], "protected_entities_preserved": ["42"]},
            {"unit_id": "unit_0002", "protected_entities_found": [], "protected_entities_preserved": []},
        ]
    }
    instance, session = provider([Response(body=message(json.dumps(output)))])
    result = instance.translate_full_clip(payload, output_schema=schema)
    assert [unit["unit_id"] for unit in result.data["units"]] == ["unit_0001", "unit_0002"]
    assert result.metadata.prompt_version == FULL_CLIP_TRANSLATION_PROMPT_VERSION
    assert len(session.calls) == 1
    posted = json.loads(session.calls[0][1]["json"]["messages"][0]["content"])
    assert posted["input"]["transcript"] == payload["transcript"]
    assert posted["input"]["dubbing_units"] == payload["dubbing_units"]
    assert "untrusted" in session.calls[0][1]["json"]["system"].lower()
