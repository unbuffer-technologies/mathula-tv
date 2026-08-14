from mathula_tv.ai_provider import (
    AZURE_FOUNDRY_CLAUDE_PROVIDER,
    AnthropicClaudeProvider,
    AnthropicConfig,
    StructuredAIRequest,
    _retry_delay_from_headers,
)


def _config():
    return AnthropicConfig(
        api_key="test-key",
        provider=AZURE_FOUNDRY_CLAUDE_PROVIDER,
        model="claude-opus-4-8",
        base_url="https://mathula-test.services.ai.azure.com/anthropic",
        max_retries=5,
        translation_max_output_tokens=100_000,
    )


def test_foundry_message_uses_stable_cacheable_prefix():
    provider = AnthropicClaudeProvider(_config())
    request = StructuredAIRequest(
        operation="test",
        payload={"value": 1},
        output_schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["ok"],
            "properties": {"ok": {"type": "boolean"}},
        },
        prompt_version="test-v1",
        system_prompt="System",
        user_prompt="Dynamic request",
        cacheable_user_prefix="Stable complete-story context",
        cache_ttl="1h",
        response_schema_version="test-response-v1",
        max_output_tokens=9_000,
    )
    body = provider.build_request(request)
    content = body["messages"][0]["content"]
    assert body["max_tokens"] == 9_000
    assert content[0]["text"] == "Stable complete-story context"
    assert content[0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert content[1]["text"].startswith("Dynamic request")


def test_azure_retry_after_milliseconds_is_honoured():
    assert _retry_delay_from_headers({"retry-after-ms": "1250"}) == 1.25
    assert _retry_delay_from_headers({"Retry-After": "3"}) == 3.0


def test_multivariant_chunk_uses_dynamic_output_limit_and_removes_cached_spine(monkeypatch):
    monkeypatch.setenv("MATHULA_TV_TRANSLATION_PROMPT_CACHE", "1")
    monkeypatch.setenv("MATHULA_TV_TRANSLATION_CACHE_TTL", "1h")

    class CaptureProvider(AnthropicClaudeProvider):
        def complete_structured(self, request):
            return request

    provider = CaptureProvider(_config())
    request = provider.translate_multivariant(
        {
            "target_locale": "zu-ZA",
            "translation_batch": {
                "cacheable_context": {"complete_timeline_coverage": True},
                "max_output_tokens": 12_345,
                "requested_unit_ids": ["unit_0001"],
            },
            "units": [{"unit_id": "unit_0001", "source_text": "Hello"}],
        }
    )
    assert request.max_output_tokens == 12_345
    assert request.cache_ttl == "1h"
    assert "complete_timeline_coverage" in request.cacheable_user_prefix
    assert "cacheable_context" not in request.user_prompt


def test_context_spine_remains_in_dynamic_prompt_when_cache_is_disabled(monkeypatch):
    monkeypatch.setenv("MATHULA_TV_TRANSLATION_PROMPT_CACHE", "0")

    class CaptureProvider(AnthropicClaudeProvider):
        def complete_structured(self, request):
            return request

    provider = CaptureProvider(_config())
    request = provider.translate_multivariant(
        {
            "target_locale": "zu-ZA",
            "translation_batch": {
                "cacheable_context": {"complete_timeline_coverage": True},
                "max_output_tokens": 8_000,
            },
            "units": [{"unit_id": "unit_0001", "source_text": "Hello"}],
        }
    )
    assert request.cacheable_user_prefix is None
    assert "complete_story_context_spine" in request.user_prompt
