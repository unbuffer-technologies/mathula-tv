#!/usr/bin/env python3
"""Install Mathula TV's pinned SpeechBrain ECAPA model into the local HF cache."""

from __future__ import annotations

from huggingface_hub import snapshot_download

from mathula_tv.known_speakers import (
    SPEECHBRAIN_MODEL_ID,
    SPEECHBRAIN_MODEL_REVISION,
)


def main() -> int:
    path = snapshot_download(
        repo_id=SPEECHBRAIN_MODEL_ID,
        revision=SPEECHBRAIN_MODEL_REVISION,
    )
    print(
        f"Installed {SPEECHBRAIN_MODEL_ID}@{SPEECHBRAIN_MODEL_REVISION} in {path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
