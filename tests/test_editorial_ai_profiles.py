from __future__ import annotations

from pathlib import Path

from mathula_tv.ai_provider import (
    AZURE_FOUNDRY_CLAUDE_PROVIDER,
    AnthropicClaudeProvider,
    AnthropicConfig,
)
from mathula_tv.config import load_settings


def _environment() -> dict[str, str]:
    return {
        "MATHULA_TV_AI_PROVIDER": AZURE_FOUNDRY_CLAUDE_PROVIDER,
        "MATHULA_TV_FOUNDRY_CLAUDE_API_KEY": "test-key",
        "MATHULA_TV_FOUNDRY_CLAUDE_ENDPOINT": (
            "https://mathula-test.services.ai.azure.com/anthropic"
        ),
        "MATHULA_TV_FOUNDRY_CLAUDE_DEPLOYMENT": "claude-opus-4-8",
        "MATHULA_TV_CLAUDE_EFFORT": "high",
        "MATHULA_TV_SEO_CLAUDE_THINKING": "adaptive",
        "MATHULA_TV_SEO_CLAUDE_EFFORT": "medium",
        "MATHULA_TV_SEO_MAX_OUTPUT_TOKENS": "12000",
        "MATHULA_TV_HOOK_CLAUDE_THINKING": "disabled",
        "MATHULA_TV_HOOK_CLAUDE_EFFORT": "low",
        "MATHULA_TV_HOOK_MAX_OUTPUT_TOKENS": "9000",
        "MATHULA_TV_EDITORIAL_CLAUDE_THINKING": "adaptive",
        "MATHULA_TV_EDITORIAL_CLAUDE_EFFORT": "high",
        "MATHULA_TV_EDITORIAL_MAX_OUTPUT_TOKENS": "32000",
    }


def test_editorial_profiles_are_loaded_independently() -> None:
    config = AnthropicConfig.from_environment(_environment())

    assert config.effort == "high"
    assert config.seo_thinking_type == "adaptive"
    assert config.seo_effort == "medium"
    assert config.seo_max_output_tokens == 12_000
    assert config.hook_thinking_type == "disabled"
    assert config.hook_effort == "low"
    assert config.hook_max_output_tokens == 9_000
    assert config.editorial_thinking_type == "adaptive"
    assert config.editorial_effort == "high"
    assert config.editorial_max_output_tokens == 32_000


def test_seo_request_uses_seo_profile() -> None:
    config = AnthropicConfig.from_environment(_environment())
    provider = AnthropicClaudeProvider(config)
    captured = {}

    def fake_complete(request):
        captured["request"] = request
        return None

    provider.complete_structured = fake_complete  # type: ignore[method-assign]
    provider.generate_seo(
        {"full_transcript": "context"},
        output_schema={"type": "object", "additionalProperties": True},
    )

    request = captured["request"]
    assert request.effort == "medium"
    assert request.thinking_type == "adaptive"
    assert request.max_output_tokens == 12_000

    body = provider.build_request(request)
    assert body["thinking"] == {"type": "adaptive"}
    assert body["output_config"] == {"effort": "medium"}
    assert body["max_tokens"] == 12_000


def test_settings_snapshot_contains_editorial_profiles(monkeypatch, tmp_path: Path) -> None:
    for key, value in _environment().items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("MATHULA_TV_WORK_DIR", str(tmp_path / "working"))

    settings = load_settings(tmp_path)

    assert settings.seo_claude_thinking == "adaptive"
    assert settings.seo_claude_effort == "medium"
    assert settings.seo_max_output_tokens == 12_000
    assert settings.hook_claude_thinking == "disabled"
    assert settings.hook_claude_effort == "low"
    assert settings.hook_max_output_tokens == 9_000
    assert settings.editorial_claude_thinking == "adaptive"
    assert settings.editorial_claude_effort == "high"
    assert settings.editorial_max_output_tokens == 32_000
