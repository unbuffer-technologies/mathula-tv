import json
from datetime import datetime, timezone

import pytest
import requests

from mathula_tv.ai_provider import (
    AZURE_FOUNDRY_CLAUDE_PROVIDER,
    DEFAULT_CLAUDE_MODEL,
    FULL_CLIP_TRANSLATION_PROMPT_VERSION,
    SEMANTIC_LANGUAGE_ANALYSIS_SCHEMA,
    build_semantic_language_payload,
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


def test_environment_defaults_to_gpt_and_never_falls_back_to_anthropic():
    with pytest.raises(ValueError, match="AZURE_AI_KEY"):
        create_production_ai_provider(
            {"ANTHROPIC_API_KEY": "must-not-be-used"}, session=Session([])
        )
    instance = create_production_ai_provider(
        {
            "AZURE_AI_ENDPOINT": "https://unit-test.services.ai.azure.com",
            "AZURE_AI_KEY": "secret",
            "AZURE_AI_DEPLOYMENT": "gpt-5.6-sol-1",
            "MATHULA_TV_AI_PROVIDER": "azure-foundry-claude",
            "ANTHROPIC_API_KEY": "must-not-be-used",
        },
        session=Session([]),
    )
    assert instance.provider == "azure-openai-gpt"
    assert instance.model == "gpt-5.6-sol-1"


def test_missing_configuration_names_variable_without_disclosing_values():
    with pytest.raises(ValueError) as caught:
        AnthropicConfig.from_environment({"MATHULA_TV_CLAUDE_MODEL": "private-model-name"})
    assert "ANTHROPIC_API_KEY" in str(caught.value)
    assert "private-model-name" not in str(caught.value)


def test_azure_foundry_claude_uses_messages_endpoint_and_api_key_header():
    config = AnthropicConfig.from_environment(
        {
            "MATHULA_TV_AI_PROVIDER": AZURE_FOUNDRY_CLAUDE_PROVIDER,
            "MATHULA_TV_FOUNDRY_CLAUDE_ENDPOINT": "https://example.services.ai.azure.com/anthropic/v1/messages",
            "MATHULA_TV_FOUNDRY_CLAUDE_DEPLOYMENT": "claude-opus-4-8",
            "MATHULA_TV_FOUNDRY_CLAUDE_API_KEY": "foundry-secret",
        }
    )
    instance = AnthropicClaudeProvider(config, session=Session([Response(body=message())]), sleep=lambda _delay: None)
    instance.complete_structured(request())
    url, kwargs = instance.session.calls[0]
    assert url == "https://example.services.ai.azure.com/anthropic/v1/messages"
    assert kwargs["headers"]["x-api-key"] == "foundry-secret"
    assert instance.provider == AZURE_FOUNDRY_CLAUDE_PROVIDER
    assert kwargs["json"]["output_config"] == {
        "effort": "high",
        "format": {"type": "json_schema", "schema": SIMPLE_SCHEMA},
    }
    payload = json.loads(kwargs["json"]["messages"][0]["content"])
    assert "required_output_schema" not in payload


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


def test_markdown_fenced_json_is_accepted_from_compatible_endpoint():
    instance, _session = provider([Response(body=message("```json\n{\"ok\": true}\n```"))])
    assert instance.complete_structured(request()).data == {"ok": True}


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


def test_english_code_switch_phrase_is_protected_verbatim():
    payload = build_full_clip_payload(
        transcript={"segments": [{"source_text": "The Big W West Side."}]},
        dubbing_units={
            "units": [
                {
                    "unit_id": "unit_0001",
                    "source_text": "The Big W West Side.",
                }
            ]
        },
        pronunciation_dictionary={
            "entries": [],
            "job_overrides": [
                {
                    "kind": "english_code_switch",
                    "display_text": "Big W",
                    "spoken_text": "Big W",
                    "tts_text": "Big double-you",
                },
                {
                    "kind": "english_code_switch",
                    "display_text": "West Side",
                    "spoken_text": "West Side",
                    "tts_text": "West Side",
                },
            ],
        },
    )
    assert payload["protected"]["verbatim_phrases"] == ["Big W", "West Side"]

    schema = {
        "type": "object",
        "required": ["units"],
        "properties": {
            "units": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": [
                        "unit_id",
                        "faithful_translation",
                        "spoken_text",
                        "protected_entities_found",
                        "protected_entities_preserved",
                    ],
                    "properties": {
                        "unit_id": {"type": "string"},
                        "faithful_translation": {"type": "string"},
                        "spoken_text": {"type": "string"},
                        "protected_entities_found": {"type": "array"},
                        "protected_entities_preserved": {"type": "array"},
                    },
                },
            }
        },
    }
    bad_output = {
        "units": [
            {
                "unit_id": "unit_0001",
                "faithful_translation": "Uhlangothi olusentshonalanga.",
                "spoken_text": "Uhlangothi olusentshonalanga.",
                "protected_entities_found": [],
                "protected_entities_preserved": [],
            }
        ]
    }
    instance, _session = provider([Response(body=message(json.dumps(bad_output)))])
    with pytest.raises(ClaudeInvalidStructuredOutput):
        instance.translate_full_clip(payload, output_schema=schema)


def test_semantic_analysis_locks_cross_unit_wordplay_and_translation_obeys_owner():
    semantic_payload = build_semantic_language_payload(
        transcript={"segments": [{"source_text": "The Big W West Side."}]},
        dubbing_units={
            "units": [
                {"unit_id": "unit_0007", "source_text": "The Big"},
                {"unit_id": "unit_0008", "source_text": "W West Side."},
            ]
        },
    )
    semantic_output = {
        "schema_version": "semantic-language-analysis-v1",
        "spans": [
            {
                "span_id": "span_0001",
                "source_text": "Big W",
                "source_unit_ids": ["unit_0007", "unit_0008"],
                "output_unit_id": "unit_0007",
                "category": "wordplay",
                "policy": "spell_out",
                "display_text": "Big W",
                "tts_text": "Big double-you",
                "rationale": "W links Big W, West Side and Woolworths",
                "confidence": 0.96,
                "human_review_required": False,
            },
            {
                "span_id": "span_0002",
                "source_text": "West Side",
                "source_unit_ids": ["unit_0008"],
                "output_unit_id": "unit_0008",
                "category": "wordplay",
                "policy": "preserve_verbatim",
                "display_text": "West Side",
                "tts_text": "West Side",
                "rationale": "English wordplay anchor",
                "confidence": 0.94,
                "human_review_required": False,
            },
        ],
    }
    semantic_provider, _ = provider([Response(body=message(json.dumps(semantic_output)))])
    analyzed = semantic_provider.analyze_semantic_language(semantic_payload)
    assert analyzed.data["spans"][0]["source_unit_ids"] == ["unit_0007", "unit_0008"]

    translation_payload = build_full_clip_payload(
        transcript=semantic_payload["transcript"],
        dubbing_units=semantic_payload["dubbing_units"],
        semantic_annotations=semantic_output,
    )
    translated = {
        "units": [
            {
                "unit_id": "unit_0007",
                "faithful_translation": "Uhehwe yi-Big W.",
                "spoken_text": "Uhehwe yi-Big W.",
                "tts_text": "Uhehwe yi-Big double-you.",
                "protected_entities_found": [],
                "protected_entities_preserved": [],
                "human_review_flags": [],
                "semantic_span_ids_applied": ["span_0001"],
            },
            {
                "unit_id": "unit_0008",
                "faithful_translation": "West Side.",
                "spoken_text": "West Side.",
                "tts_text": "West Side.",
                "protected_entities_found": [],
                "protected_entities_preserved": [],
                "human_review_flags": [],
                "semantic_span_ids_applied": ["span_0002"],
            },
        ]
    }
    schema = {
        "type": "object",
        "required": ["units"],
        "properties": {
            "units": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": [
                        "unit_id", "faithful_translation", "spoken_text", "tts_text",
                        "protected_entities_found", "protected_entities_preserved",
                        "human_review_flags", "semantic_span_ids_applied"
                    ],
                    "properties": {
                        "unit_id": {"type": "string"},
                        "faithful_translation": {"type": "string"},
                        "spoken_text": {"type": "string"},
                        "tts_text": {"type": "string"},
                        "protected_entities_found": {"type": "array"},
                        "protected_entities_preserved": {"type": "array"},
                        "human_review_flags": {"type": "array"},
                        "semantic_span_ids_applied": {"type": "array"},
                    },
                },
            }
        },
    }
    translation_provider, _ = provider([Response(body=message(json.dumps(translated)))])
    result = translation_provider.translate_full_clip(translation_payload, output_schema=schema)
    assert result.data["units"][0]["tts_text"].endswith("Big double-you.")


def test_semantic_analysis_rejects_invented_span():
    payload = build_semantic_language_payload(
        transcript={"segments": [{"source_text": "Ordinary sentence."}]},
        dubbing_units={"units": [{"unit_id": "unit_0001", "source_text": "Ordinary sentence."}]},
    )
    invented = {
        "schema_version": "semantic-language-analysis-v1",
        "spans": [
            {
                "span_id": "span_0001",
                "source_text": "Big W",
                "source_unit_ids": ["unit_0001"],
                "output_unit_id": "unit_0001",
                "category": "wordplay",
                "policy": "preserve_verbatim",
                "display_text": "Big W",
                "tts_text": "Big W",
                "rationale": "invented",
                "confidence": 0.9,
                "human_review_required": False,
            }
        ],
    }
    instance, _ = provider([Response(body=message(json.dumps(invented)))])
    with pytest.raises(ClaudeInvalidStructuredOutput):
        instance.analyze_semantic_language(payload, output_schema=SEMANTIC_LANGUAGE_ANALYSIS_SCHEMA)
