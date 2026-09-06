#!/usr/bin/env python3
"""Print the personas used by the active direct Azure renderer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _cell(value: Any) -> str:
    return "-" if value is None else str(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("job_id")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    path = (
        args.repo.resolve()
        / "working"
        / "jobs"
        / args.job_id
        / "direct_dub"
        / "voice_resolution.json"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if args.json:
        print(json.dumps(artifact, ensure_ascii=False, indent=2))
        return 0

    print(f"Artifact: {path}")
    print(f"Persona version: {artifact.get('persona_version', '-')}")
    print("speaker\tfamily\tvoice\tslot\tpitch\trate\tvolume\tadjusted\tresolution")
    for item in artifact.get("speakers", []):
        if item.get("status") != "resolved":
            continue
        persona = item.get("base_prosody") or {}
        print(
            "\t".join(
                [
                    _cell(item.get("speaker_id")),
                    _cell(item.get("voice_family")),
                    _cell(item.get("selected_voice")),
                    _cell(persona.get("persona_slot")),
                    _cell(persona.get("pitch_percent")),
                    _cell(persona.get("rate_percent")),
                    _cell(persona.get("volume_percent")),
                    _cell(persona.get("separation_adjusted")),
                    _cell(item.get("resolution_status")),
                ]
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
