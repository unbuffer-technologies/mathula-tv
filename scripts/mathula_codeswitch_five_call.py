#!/usr/bin/env python3
"""Compatibility entry point for the retired five-model translation chain.

The historical implementation called Anthropic directly. It is intentionally
unavailable in the GPT-only release because it bypassed the production chunk
checkpoints, source-bound validators, and single provider factory.
"""

from __future__ import annotations

import sys


def main() -> int:
    print(
        "The five-call translation experiment has been retired in v13.18.54. "
        "Use `mathula-tv translate JOB_ID --live-operation`; that command uses "
        "Azure OpenAI GPT and preserves the production validation gates.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
