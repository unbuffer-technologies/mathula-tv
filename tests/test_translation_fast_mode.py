from __future__ import annotations

from pathlib import Path

from mathula_tv.ai_provider import (
    AZURE_FOUNDRY_CLAUDE_PROVIDER,
    AnthropicClaudeProvider,
    AnthropicConfig,
)
from mathula_tv.config import load_settings
from mathula_tv.multivariant_translation import build_translation_request


def _environment() -> dict[str, str]:
    return {
        "MATHULA_TV_AI_PROVIDER": AZURE_FOUNDRY_CLAUDE_PROVIDER,
        "MATHULA_TV_FOUNDRY_CLAUDE_API_KEY": "test-key",
        "MATHULA_TV_FOUNDRY_CLAUDE_ENDPOINT": (
            "https://mathula-test.services.ai.azure.com/anthropic"
        ),
        "MATHULA_TV_FOUNDRY_CLAUDE_DEPLOYMENT": "claude-opus-4-8",
        # Keep the general Claude profile deliberately expensive. Translation
        # must not inherit it.
        "MATHULA_TV_CLAUDE_EFFORT": "high",
        "MATHULA_TV_CLAUDE_MAX_OUTPUT_TOKENS": "50000",
        "MATHULA_TV_TRANSLATION_CLAUDE_THINKING": "disabled",
        "MATHULA_TV_TRANSLATION_CLAUDE_EFFORT": "low",
        "MATHULA_TV_TRANSLATION_MAX_OUTPUT_TOKENS": "100000",
    }


def _request() -> dict:
    return build_translation_request(
        transcript={
            "language": "en-ZA",
            "segments": [
                {
                    "segment_id": "seg-1",
                    "speaker": "SPEAKER_00",
                    "start": 0,
                    "end": 2,
                    "source_text": "The EFF speaks.",
                    "protected_spans": ["EFF"],
                }
            ],
        },
        target_locale="zu-ZA",
        job_id="job-fast-mode",
    )


def test_translation_environment_overrides_global_hard_mode() -> None:
    config = AnthropicConfig.from_environment(_environment())

    assert config.effort == "high"
    assert config.max_output_tokens == 50_000
    assert config.translation_thinking_type == "disabled"
    assert config.translation_effort == "low"
    assert config.translation_max_output_tokens == 100_000


def test_translation_request_sends_fast_profile_to_foundry() -> None:
    config = AnthropicConfig.from_environment(_environment())
    provider = AnthropicClaudeProvider(config)
    payload = _request()

    # Build the same request object used by translate_multivariant without
    # issuing a network call.
    from mathula_tv.ai_provider import StructuredAIRequest
    from mathula_tv.ai_provider import (
        MULTIVARIANT_TRANSLATION_PROMPT_VERSION,
        MULTIVARIANT_TRANSLATION_SCHEMA,
        MULTIVARIANT_TRANSLATION_SCHEMA_VERSION,
    )
    from mathula_tv.multivariant_translation import render_system_prompt, render_user_prompt

    request = StructuredAIRequest(
        operation="multivariant_translation",
        payload=payload,
        output_schema=MULTIVARIANT_TRANSLATION_SCHEMA,
        prompt_version=MULTIVARIANT_TRANSLATION_PROMPT_VERSION,
        system_prompt=render_system_prompt("zu-ZA"),
        user_prompt=render_user_prompt(payload),
        response_schema_version=MULTIVARIANT_TRANSLATION_SCHEMA_VERSION,
        max_repairs=0,
        provider_json_schema=True,
        stream_response=True,
        effort=config.translation_effort,
        thinking_type=config.translation_thinking_type,
        max_output_tokens=config.translation_max_output_tokens,
    )
    body = provider.build_request(request)

    assert body["thinking"] == {"type": "disabled"}
    assert body["output_config"] == {"effort": "low"}
    assert body["max_tokens"] == 100_000
    assert body["stream"] is True


def test_settings_snapshot_includes_translation_profile(monkeypatch, tmp_path: Path) -> None:
    env = _environment()
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("MATHULA_TV_WORK_DIR", str(tmp_path / "working"))

    settings = load_settings(tmp_path)

    assert settings.translation_claude_thinking == "disabled"
    assert settings.translation_claude_effort == "low"
    assert settings.translation_max_output_tokens == 100_000
