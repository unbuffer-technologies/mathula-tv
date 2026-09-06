#!/usr/bin/env python3
'''Print the resolved Azure persona assigned to every speaker in one Mathula job.'''

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def find_assignment_file(job_root: Path) -> Path:
    preferred = (
        job_root / "dubbing/azure_voice_assignments.json",
        job_root / "dubbing/azure_tts/azure_voice_assignments.json",
        job_root / "analysis/azure_voice_assignments.json",
    )
    for path in preferred:
        if path.is_file():
            return path

    candidates = sorted(job_root.rglob("*voice*assign*.json"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"No Azure voice-assignment artifact found below {job_root}"
        )
    raise RuntimeError(
        "Multiple assignment artifacts were found; pass --file explicitly:\n"
        + "\n".join(str(path) for path in candidates)
    )


def cell(value: Any) -> str:
    return "-" if value is None else str(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("job_id", nargs="?")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--file", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.file:
        path = args.file.resolve()
    else:
        if not args.job_id:
            raise SystemExit("JOB_ID or --file is required")
        path = find_assignment_file(
            args.repo.resolve() / "working/jobs" / args.job_id
        )

    data = json.loads(path.read_text(encoding="utf-8"))
    assignments = data.get("assignments")
    if not isinstance(assignments, list):
        raise RuntimeError(f"Invalid assignment artifact: {path}")

    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return 0

    print(f"Artifact: {path}")
    print(
        "speaker\tfamily\tvoice\tslot\tpitch\trate\tvolume\t"
        "v1_pitch\tv1_rate\tadjusted"
    )
    for assignment in assignments:
        prosody = assignment.get("base_prosody") or {}
        old = prosody.get("persona_v1") or {}
        print(
            "\t".join(
                [
                    cell(assignment.get("speaker_id")),
                    cell(assignment.get("anchor_voice_family")),
                    cell(assignment.get("selected_voice")),
                    cell(prosody.get("persona_slot")),
                    cell(prosody.get("pitch_percent")),
                    cell(prosody.get("rate_percent")),
                    cell(prosody.get("volume_percent")),
                    cell(old.get("pitch_percent")),
                    cell(old.get("rate_percent")),
                    cell(prosody.get("separation_adjusted")),
                ]
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
