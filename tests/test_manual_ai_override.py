import json
from pathlib import Path

import pytest

from mathula_tv.ai_provider import StructuredAIRequest, create_production_ai_provider
from mathula_tv.cli import create_translation_backend
from mathula_tv.errors import AIInvalidStructuredOutput
from mathula_tv.manual_ai import (
    MANUAL_RESPONSE_SCHEMA_VERSION,
    ManualAIResponseRequired,
    ManualChatGPTProvider,
    install_manual_response,
    pending_manual_requests,
)


SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["ok"],
    "properties": {"ok": {"type": "boolean"}},
}


def request():
    return StructuredAIRequest(
        operation="manual_test",
        payload={"source_text": "Hello", "api_key": "must-not-export"},
        output_schema=SCHEMA,
        prompt_version="manual-test-v1",
        system_prompt="Return the required JSON object.",
        response_schema_version="manual-test-result-v1",
    )


def test_factory_manual_override_does_not_require_azure_credentials(tmp_path):
    provider = create_production_ai_provider(
        {
            "MATHULA_TV_AI_PROVIDER": "manual-chatgpt",
            "MATHULA_TV_MANUAL_AI_DIR": str(tmp_path / "manual_ai"),
        }
    )
    assert isinstance(provider, ManualChatGPTProvider)
    assert provider.provider == "manual-chatgpt"
    assert provider.model == "gpt-5.6-sol"


def test_manual_provider_exports_then_resumes_through_schema_validation(tmp_path):
    root = tmp_path / "manual_ai"
    provider = create_production_ai_provider(
        {
            "MATHULA_TV_AI_PROVIDER": "manual-chatgpt",
            "MATHULA_TV_MANUAL_AI_DIR": str(root),
        }
    )
    with pytest.raises(ManualAIResponseRequired):
        provider.complete_structured(request())

    pending = pending_manual_requests(root)
    assert len(pending) == 1
    exported = json.loads(Path(pending[0]["request_path"]).read_text(encoding="utf-8"))
    assert exported["schema_version"] == "mathula-manual-ai-request-v1"
    assert exported["payload"]["source_text"] == "Hello"
    assert exported["payload"]["api_key"] == "<redacted>"

    response_file = tmp_path / "from-chatgpt.json"
    response_file.write_text(
        json.dumps(
            {
                "schema_version": MANUAL_RESPONSE_SCHEMA_VERSION,
                "request_id": exported["request_id"],
                "result": {"ok": True},
            }
        ),
        encoding="utf-8",
    )
    installed = install_manual_response(root, response_file)
    assert Path(installed["response_path"]).is_file()

    response = provider.complete_structured(request())
    assert response.data == {"ok": True}
    assert response.metadata.provider == "manual-chatgpt"
    assert response.metadata.provider_attempts == 1
    assert response.metadata.token_usage.input_tokens == 0
    assert list((root / "accepted").glob("*.json"))


def test_manual_provider_rejects_wrong_request_id(tmp_path):
    root = tmp_path / "manual_ai"
    provider = create_production_ai_provider(
        {
            "MATHULA_TV_AI_PROVIDER": "manual-chatgpt",
            "MATHULA_TV_MANUAL_AI_DIR": str(root),
        }
    )
    with pytest.raises(ManualAIResponseRequired):
        provider.complete_structured(request())
    pending = pending_manual_requests(root)[0]
    response_path = Path(pending["expected_response_path"])
    response_path.parent.mkdir(parents=True, exist_ok=True)
    response_path.write_text(
        json.dumps(
            {
                "schema_version": MANUAL_RESPONSE_SCHEMA_VERSION,
                "request_id": "wrong",
                "result": {"ok": True},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(AIInvalidStructuredOutput, match="request_id"):
        provider.complete_structured(request())


def test_cli_translation_backend_accepts_manual_provider_without_azure_env(tmp_path, monkeypatch):
    for name in (
        "AZURE_AI_KEY",
        "AZURE_AI_ENDPOINT",
        "AZURE_AI_DEPLOYMENT",
        "AZURE_OPENAI_CHAT_DEPLOYMENT",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MATHULA_TV_MANUAL_AI_DIR", str(tmp_path / "manual_ai"))
    backend = create_translation_backend("manual-chatgpt")
    assert backend.provider == "manual-chatgpt"


def test_cli_parser_exposes_manual_ai_workflow_and_dub_override():
    from mathula_tv.cli import parser

    dub = parser().parse_args(
        ["dub-azure", "job-1", "--ai-provider", "manual-chatgpt", "--live-operation"]
    )
    assert dub.ai_provider == "manual-chatgpt"
    pending = parser().parse_args(["manual-ai", "job-1", "pending"])
    assert pending.job_id == "job-1"
    assert pending.action == "pending"
