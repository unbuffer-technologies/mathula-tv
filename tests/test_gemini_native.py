from __future__ import annotations

import json

import pytest

from mathula_tv.foundry_grok import GrokOutputTruncated, create_native_gpt_provider
from mathula_tv.gemini_native import GEMINI_NATIVE_PROVIDER, GeminiNativeConfig, GeminiNativeProvider

SCHEMA = {"type": "object", "properties": {"words": {"type": "array", "items": {"type": "string"}}}, "required": ["words"], "additionalProperties": False}


def reply(content, finish="stop", prompt=100, out=20, cached=40):
    return {"choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": finish}], "usage": {"prompt_tokens": prompt, "completion_tokens": out, "cached_tokens": cached, "thoughts_tokens": 5}}


class FakeClient:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def complete(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        return self.replies.pop(0)


def provider(client, **kw):
    return GeminiNativeProvider(client, GeminiNativeConfig(deployment="gemini-test", **kw))


def ask(p, **kw):
    return p.complete_json(operation="pass1", system_prompt="Restore the transcript.", payload={"segments": ["a"]}, schema=SCHEMA, **kw)


def test_a_valid_answer_comes_back_checked_with_its_usage_and_the_request_is_json_mode():
    client = FakeClient([reply('{"words": ["hello"]}')])
    result = ask(provider(client), max_output_tokens=5000)
    assert result.data == {"words": ["hello"]} and result.model == "gemini-test" and result.attempts == 1
    assert (result.input_tokens, result.output_tokens, result.cached_tokens) == (100, 20, 40)
    call = client.calls[0]
    assert call["json_mode"] is True and call["max_output_tokens"] == 5000
    assert call["messages"][0]["role"] == "system" and "Restore the transcript." in call["messages"][0]["content"] and call["messages"][1]["role"] == "user"


def test_a_schema_violation_gets_the_same_bounded_repair_round_as_on_gpt():
    client = FakeClient([reply('{"wrong": 1}'), reply('{"words": ["fixed"]}')])
    result = ask(provider(client))
    assert result.data == {"words": ["fixed"]} and len(client.calls) == 2
    repair = json.loads(client.calls[1]["messages"][1]["content"])
    assert repair["invalid_output"] == '{"wrong": 1}' and "validation_error" in repair                # only the bad output is resent, not the whole transcript
    with pytest.raises(Exception):
        ask(provider(FakeClient([reply("not json")] * 3)))                                            # and it stops after max_schema_repairs (2): three calls in all


def test_a_cut_off_answer_is_reported_as_truncated_not_as_bad_json():
    with pytest.raises(GrokOutputTruncated):
        ask(provider(FakeClient([reply('{"words": ["he', finish="length")])))


def test_the_native_factory_returns_gemini_only_when_asked_to():
    gemini = create_native_gpt_provider({"MATHULA_AI_PROVIDER": "gemini", "GEMINI_API_KEY": "k", "GEMINI_MODEL": "gemini-test-model"})
    assert isinstance(gemini, GeminiNativeProvider) and gemini.provider == GEMINI_NATIVE_PROVIDER and gemini.config.deployment == "gemini-test-model"
    assert gemini.client.config.thinking_level == "low" and gemini.config.max_output_tokens >= 32_768
    azure = create_native_gpt_provider({"AZURE_AI_KEY": "k", "AZURE_AI_ENDPOINT": "https://x.openai.azure.com", "AZURE_AI_DEPLOYMENT": "gpt-test"})
    assert not isinstance(azure, GeminiNativeProvider)
