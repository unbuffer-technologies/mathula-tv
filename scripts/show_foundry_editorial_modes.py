#!/usr/bin/env python3
from __future__ import annotations

from mathula_tv.ai_provider import AzureOpenAIGPTConfig


def main() -> int:
    config = AzureOpenAIGPTConfig.from_environment()
    print(f"provider: {config.provider}")
    print(f"deployment: {config.model}")
    print(
        "translation: "
        f"thinking={config.translation_thinking_type}, "
        f"effort={config.translation_effort}, "
        f"max_output_tokens={config.translation_max_output_tokens}"
    )
    print(
        "editorial: "
        f"thinking={config.editorial_thinking_type}, "
        f"effort={config.editorial_effort}, "
        f"max_output_tokens={config.editorial_max_output_tokens}"
    )
    print("editorial_calls_per_publication: 1")
    print("automatic_editorial_repairs: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
