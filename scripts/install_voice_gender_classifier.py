#!/usr/bin/env python3
"""Install the pinned voice-family classifier into the Hugging Face cache."""
from __future__ import annotations

import argparse
import json

from huggingface_hub import snapshot_download

from mathula_tv.voice_family_classifier import (
    VOICE_GENDER_MODEL_ID,
    VOICE_GENDER_MODEL_REVISION,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download Mathula TV's pinned voice-family classifier once. "
            "Normal dubbing runs are local-only."
        )
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload the pinned snapshot even when it is already cached.",
    )
    args = parser.parse_args()
    snapshot = snapshot_download(
        repo_id=VOICE_GENDER_MODEL_ID,
        revision=VOICE_GENDER_MODEL_REVISION,
        force_download=args.force,
    )
    print(
        json.dumps(
            {
                "status": "installed",
                "model": VOICE_GENDER_MODEL_ID,
                "revision": VOICE_GENDER_MODEL_REVISION,
                "snapshot": snapshot,
                "runtime_mode": "local_files_only",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
