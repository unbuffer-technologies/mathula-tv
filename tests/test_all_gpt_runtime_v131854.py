from types import SimpleNamespace

import pytest

import mathula_tv.autocorrect_ai as autocorrect_ai
from mathula_tv.ai_provider import (
    AzureOpenAIGPTConfig,
    AzureOpenAIGPTProvider,
    PRODUCTION_AI_PROVIDER,
    SEMANTIC_FIT_REVIEW_BATCH_SCHEMA,
    StructuredAIRequest,
    _openai_output_schema,
    create_production_ai_provider,
)
from mathula_tv.cli import create_translation_backend
from mathula_tv.config import load_settings
from mathula_tv.one_call_translation import translate_full_transcript_once


ENVIRONMENT = {
    "AZURE_AI_ENDPOINT": "https://unit-test.services.ai.azure.com",
    "AZURE_AI_KEY": "unit-test-key",
    "AZURE_AI_DEPLOYMENT": "gpt-5.6-sol-1",
    "AZURE_OPENAI_CHAT_DEPLOYMENT": "gpt-5.6-sol-1",
}


def test_settings_and_factory_ignore_stale_claude_selector(tmp_path, monkeypatch):
    for name, value in {
        **ENVIRONMENT,
        "MATHULA_TV_AI_PROVIDER": "azure-foundry-claude",
        "ANTHROPIC_API_KEY": "must-not-be-read",
        "MATHULA_TV_FOUNDRY_CLAUDE_API_KEY": "must-not-be-read",
    }.items():
        monkeypatch.setenv(name, value)
    settings = load_settings(tmp_path)
    provider = create_production_ai_provider(dict(ENVIRONMENT))
    assert settings.ai_provider == PRODUCTION_AI_PROVIDER == "azure-openai-gpt"
    assert settings.gpt_model == "gpt-5.6-sol-1"
    assert not any("claude" in key for key in settings.safe_snapshot())
    assert isinstance(provider, AzureOpenAIGPTProvider)
    assert provider.url.endswith("/openai/v1/responses")


def test_cli_has_no_selectable_claude_runtime():
    with pytest.raises(ValueError, match="GPT-only"):
        create_translation_backend("azure-foundry-claude")
    with pytest.raises(ValueError, match="GPT-only"):
        create_translation_backend("anthropic")


def test_translation_orchestrator_accepts_gpt_and_rejects_non_gpt_before_io():
    pipeline = SimpleNamespace()
    job = SimpleNamespace(state="synthesis_queued")
    gpt = SimpleNamespace(
        provider="azure-openai-gpt",
        config=SimpleNamespace(provider="azure-openai-gpt", endpoint="", model="gpt"),
    )
    with pytest.raises(ValueError, match="Migrate the validated legacy translation"):
        translate_full_transcript_once(pipeline, job, gpt)

    legacy = SimpleNamespace(
        provider="azure-foundry-claude",
        config=SimpleNamespace(provider="azure-foundry-claude"),
    )
    with pytest.raises(ValueError, match="must use Azure OpenAI GPT"):
        translate_full_transcript_once(pipeline, job, legacy)


def test_gpt_web_search_tool_is_mapped_to_responses_api_shape():
    provider = AzureOpenAIGPTProvider(
        AzureOpenAIGPTConfig(
            api_key="unit-test-key",
            endpoint="https://unit-test.services.ai.azure.com",
            model="gpt-5.6-sol-1",
            max_retries=1,
        ),
        session=SimpleNamespace(),
    )
    request = StructuredAIRequest(
        operation="research_contract",
        payload={"term": "example"},
        output_schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["ok"],
            "properties": {"ok": {"type": "boolean"}},
        },
        prompt_version="gpt-research-contract-v1",
        system_prompt="Research and return JSON.",
        response_schema_version="research-contract-v1",
        server_tools=({"type": "web_search_20250305", "name": "web_search"},),
    )
    body = provider.build_request(request)
    assert body["tools"] == [{"type": "web_search"}]
    assert "messages" not in body


def test_semantic_fit_review_batch_const_has_explicit_azure_type():
    wire_schema = _openai_output_schema(SEMANTIC_FIT_REVIEW_BATCH_SCHEMA)
    assert wire_schema["properties"]["schema_version"] == {
        "const": "claude-semantic-fit-review-batch-v1",
        "type": "string",
    }


def test_openai_schema_infers_all_unambiguous_literal_types():
    assert _openai_output_schema({"const": True})["type"] == "boolean"
    assert _openai_output_schema({"const": 3})["type"] == "integer"
    assert _openai_output_schema({"const": 3.5})["type"] == "number"
    assert _openai_output_schema({"const": None})["type"] == "null"
    assert _openai_output_schema({"enum": ["a", "b"]})["type"] == "string"
    mixed = _openai_output_schema({"enum": ["a", 1]})
    assert "type" not in mixed


def test_every_runtime_operation_family_builds_only_responses_requests():
    provider = AzureOpenAIGPTProvider(
        AzureOpenAIGPTConfig(
            api_key="unit-test-key",
            endpoint="https://unit-test.services.ai.azure.com",
            model="gpt-5.6-sol-1",
            max_retries=1,
        ),
        session=SimpleNamespace(),
    )
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["ok"],
        "properties": {"ok": {"type": "boolean"}},
    }
    operations = (
        "multivariant_translation",
        "semantic_language_analysis",
        "context_analysis",
        "contextual_voice_family_inference",
        "autocorrect_entity_inventory",
        "autocorrect_candidate_ranking",
        "seo_generation",
        "editorial_package",
        "semantic_fit_portfolio_batch",
        "semantic_fit_review_batch",
    )
    for operation in operations:
        body = provider.build_request(
            StructuredAIRequest(
                operation=operation,
                payload={"fixture": True},
                output_schema=schema,
                prompt_version=f"gpt-{operation}-simulation-v1",
                system_prompt="Return the source-bound JSON result.",
                response_schema_version="simulation-v1",
                max_output_tokens=20_000,
            )
        )
        assert body["model"] == "gpt-5.6-sol-1"
        assert body["store"] is False
        assert "input" in body and "instructions" in body
        assert "messages" not in body
        assert "anthropic_version" not in body


def test_autocorrect_compatibility_alias_routes_to_gpt(monkeypatch):
    request = {
        "spans": [
            {
                "correction_id": "c1",
                "candidates": [{"candidate_id": "candidate-1"}],
            }
        ]
    }
    response_data = {
        "schema_version": "mathula-autocorrect-ai-response-v1",
        "decisions": [
            {
                "correction_id": "c1",
                "decision": "ACCEPT_CANDIDATE",
                "candidate_id": "candidate-1",
                "confidence": 0.9,
                "reason": "context",
                "question": "",
            }
        ],
    }
    metadata = SimpleNamespace(
        to_dict=lambda: {
            "provider": "azure-openai-gpt",
            "model_returned": "gpt-5.6-sol-1",
        }
    )

    class Provider:
        config = SimpleNamespace(model="gpt-5.6-sol-1")

        def complete_structured(self, ai_request):
            assert ai_request.operation == "autocorrect_candidate_ranking"
            return SimpleNamespace(data=response_data, metadata=metadata)

    monkeypatch.setattr(
        autocorrect_ai, "create_production_ai_provider", lambda _environment: Provider()
    )
    advice, generation = autocorrect_ai.rank_with_anthropic(
        request, model="gpt-5.6-sol-1"
    )
    assert advice["c1"]["candidate_id"] == "candidate-1"
    assert generation["provider"] == "azure-openai-gpt"
