import json

import pytest

from mathula_tv.ai_provider import (
    AZURE_OPENAI_GPT_PROVIDER,
    AzureOpenAIGPTConfig,
    AzureOpenAIGPTProvider,
    StructuredAIRequest,
    create_production_ai_provider,
)
from mathula_tv.cli import create_translation_backend
from mathula_tv.errors import AIInvalidStructuredOutput, AIRateLimit


SCHEMA = {
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
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.responses.pop(0)


def request(**updates):
    values = {
        "operation": "provider_contract",
        "payload": {"private_transcript": "test"},
        "output_schema": SCHEMA,
        "prompt_version": "gpt-contract-test-v1",
        "system_prompt": "Return JSON.",
        "response_schema_version": "provider-contract-v1",
        "max_repairs": 0,
    }
    values.update(updates)
    return StructuredAIRequest(**values)


def response(text='{"ok":true}', **updates):
    value = {
        "id": "resp_test",
        "model": "gpt-5.6-sol-1",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "status": "completed",
                "content": [{"type": "output_text", "text": text}],
            }
        ],
        "usage": {
            "input_tokens": 13,
            "output_tokens": 7,
            "input_tokens_details": {"cached_tokens": 3},
            "output_tokens_details": {"reasoning_tokens": 2},
        },
    }
    value.update(updates)
    return value


def environment(**updates):
    values = {
        "AZURE_AI_ENDPOINT": "https://unit-test.services.ai.azure.com",
        "AZURE_AI_KEY": "not-a-real-secret",
        "AZURE_AI_DEPLOYMENT": "gpt-5.6-sol-1",
        "AZURE_OPENAI_CHAT_DEPLOYMENT": "gpt-5.6-sol-1",
    }
    values.update(updates)
    return values


def provider(responses, **config_updates):
    values = {
        "api_key": "not-a-real-secret",
        "endpoint": "https://unit-test.services.ai.azure.com",
        "model": "gpt-5.6-sol-1",
        "max_retries": 1,
        "max_repairs": 0,
    }
    values.update(config_updates)
    session = Session(responses)
    return (
        AzureOpenAIGPTProvider(
            AzureOpenAIGPTConfig(**values),
            session=session,
            sleep=lambda _delay: None,
        ),
        session,
    )


def test_factory_is_gpt_only_and_never_reads_claude_configuration():
    instance = create_production_ai_provider(
        {**environment(), "ANTHROPIC_API_KEY": "must-not-be-used"},
        session=Session([]),
    )
    assert isinstance(instance, AzureOpenAIGPTProvider)
    assert instance.provider == AZURE_OPENAI_GPT_PROVIDER
    assert instance.model == "gpt-5.6-sol-1"

    stale_claude_env = create_production_ai_provider(
        {
            **environment(),
            "MATHULA_TV_AI_PROVIDER": "azure-foundry-claude",
            "MATHULA_TV_FOUNDRY_CLAUDE_API_KEY": "must-not-be-used",
        },
        session=Session([]),
    )
    assert stale_claude_env.provider == AZURE_OPENAI_GPT_PROVIDER


def test_conflicting_deployments_fail_before_a_paid_request():
    with pytest.raises(ValueError, match="disagree"):
        AzureOpenAIGPTConfig.from_environment(
            environment(AZURE_OPENAI_CHAT_DEPLOYMENT="different-deployment")
        )


def test_responses_transport_uses_foundry_url_api_key_and_strict_schema():
    instance, session = provider([Response(body=response())])
    result = instance.complete_structured(request())

    url, call = session.calls[0]
    body = call["json"]
    assert url == "https://unit-test.services.ai.azure.com/openai/v1/responses"
    assert call["headers"] == {
        "api-key": "not-a-real-secret",
        "content-type": "application/json",
        "accept": "application/json",
    }
    assert body["model"] == "gpt-5.6-sol-1"
    assert body["store"] is False
    assert body["reasoning"] == {"effort": "high"}
    assert body["text"]["format"] == {
        "type": "json_schema",
        "name": "provider_contract",
        "strict": True,
        "schema": SCHEMA,
    }
    assert "messages" not in body and "anthropic-version" not in call["headers"]
    assert result.data == {"ok": True}
    assert result.metadata.provider == AZURE_OPENAI_GPT_PROVIDER
    assert result.metadata.token_usage.cache_read_input_tokens == 3
    assert result.metadata.stop_reason == "end_turn"


def test_optional_schema_falls_back_to_json_object_without_changing_semantics():
    optional_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"ok": {"type": "boolean"}},
    }
    instance, _session = provider([])
    body = instance.build_request(request(output_schema=optional_schema))
    assert body["text"]["format"] == {"type": "json_object"}
    prompt = body["input"][0]["content"][0]["text"]
    assert json.dumps(optional_schema, sort_keys=True, separators=(",", ":")) in prompt

    normalized = instance.build_request(
        request(
            output_schema=optional_schema,
            normalizer=lambda value: {"ok": bool(value.get("ok"))},
        )
    )
    strict_schema = normalized["text"]["format"]["schema"]
    assert normalized["text"]["format"]["type"] == "json_schema"
    assert strict_schema["required"] == ["ok"]
    assert strict_schema["properties"]["ok"] == {
        "anyOf": [{"type": "boolean"}, {"type": "null"}]
    }


def test_schema_over_azure_depth_limit_uses_json_object_mode():
    nested = {"type": "string"}
    for index in range(6):
        nested = {
            "type": "object",
            "additionalProperties": False,
            "required": [f"level_{index}"],
            "properties": {f"level_{index}": nested},
        }
    instance, _session = provider([])
    body = instance.build_request(request(output_schema=nested))
    assert body["text"]["format"] == {"type": "json_object"}


def test_semantic_fit_budget_reserves_reasoning_and_structured_output_capacity():
    instance, _session = provider([])
    events = []
    instance.event_callback = events.append
    body = instance.build_request(
        request(
            operation="semantic_fit_portfolio_batch",
            max_output_tokens=4_096,
            effort="low",
        )
    )
    assert body["max_output_tokens"] == 16_384
    assert body["reasoning"] == {"effort": "low"}
    assert events == [
        {
            "event": "output_budget_adjusted",
            "operation": "semantic_fit_portfolio_batch",
            "requested_max_output_tokens": 4_096,
            "effective_max_output_tokens": 16_384,
            "reason": "gpt_reasoning_and_structured_output_reserve",
        }
    ]


def test_gpt_model_output_ceiling_is_validated_locally():
    with pytest.raises(ValueError, match="must not exceed 128000"):
        AzureOpenAIGPTConfig(
            api_key="not-a-real-secret",
            endpoint="https://unit-test.services.ai.azure.com",
            model="gpt-5.6-sol-1",
            max_output_tokens=128_001,
        )


def test_incomplete_and_rate_limited_responses_fail_with_gpt_diagnostics():
    incomplete = response(
        text="",
        status="incomplete",
        incomplete_details={"reason": "max_output_tokens"},
        output=[],
    )
    instance, _session = provider([Response(body=incomplete)])
    with pytest.raises(AIInvalidStructuredOutput, match="Azure OpenAI GPT.*max_tokens"):
        instance.complete_structured(request())

    instance, _session = provider(
        [Response(status_code=429, body={"error": {"message": "slow down"}})]
    )
    with pytest.raises(AIRateLimit, match="Azure OpenAI GPT rate limit"):
        instance.complete_structured(request())


def test_cli_rejects_claude_in_this_release(monkeypatch):
    for name in (
        "MATHULA_TV_AI_PROVIDER",
        "AZURE_AI_ENDPOINT",
        "AZURE_AI_KEY",
        "AZURE_AI_DEPLOYMENT",
        "AZURE_OPENAI_CHAT_DEPLOYMENT",
    ):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="GPT-only"):
        create_translation_backend("azure-foundry-claude")
