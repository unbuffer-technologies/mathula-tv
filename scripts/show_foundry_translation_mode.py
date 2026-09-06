#!/usr/bin/env python3
"""Print the resolved, secret-free Azure GPT translation profile."""

from __future__ import annotations

from mathula_tv.config import load_settings
from mathula_tv.ai_provider import AzureOpenAIGPTConfig


def main() -> int:
    # load_settings() loads the project .env without overriding exported values.
    load_settings()
    config = AzureOpenAIGPTConfig.from_environment()
    print(f"provider: {config.provider}")
    print(f"endpoint: {config.endpoint}")
    print(f"deployment: {config.model}")
    print(f"translation_thinking: {config.translation_thinking_type}")
    print(f"translation_effort: {config.translation_effort}")
    print(f"translation_max_output_tokens: {config.translation_max_output_tokens}")
    print("transport: responses_api")
    print("streaming: disabled")
    print("automatic_model_repairs: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
