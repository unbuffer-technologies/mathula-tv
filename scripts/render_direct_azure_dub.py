#!/usr/bin/env python3
"""Render a dub directly from translation/transcript_zu.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mathula_tv.config import load_settings
from mathula_tv.direct_azure_dub import (
    DirectAzureDubRenderer,
    DirectDubOptions,
    create_direct_azure_backend,
)
from mathula_tv.job_store import JobStore


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        description=(
            "Synthesize, time-fit and render the approved segment-level "
            "translation without production orchestration or translation repair. "
            "The default output is dialogue-only."
        )
    )
    command.add_argument("job_id")
    command.add_argument("--translation", type=Path)
    command.add_argument("--voice-map", type=Path)
    command.add_argument("--clean-background-stem", type=Path)
    command.add_argument(
        "--include-background",
        action="store_true",
        help=(
            "Opt in to mixing background audio. By default the final video "
            "contains only the translated dialogue."
        ),
    )
    command.add_argument("--same-speaker-gap-ms", type=int, default=1200)
    command.add_argument("--max-azure-rate", type=int, default=25)
    command.add_argument("--max-time-stretch", type=float, default=1.35)
    command.add_argument("--male-voice", default="zu-ZA-ThembaNeural")
    command.add_argument("--female-voice", default="zu-ZA-ThandoNeural")
    command.add_argument("--min-voice-confidence", type=float, default=0.70)
    command.add_argument("--resolve-voices-only", action="store_true")
    command.add_argument("--force", action="store_true")
    return command


def main() -> int:
    args = parser().parse_args()
    settings = load_settings()
    jobs = JobStore(settings.work_dir)
    job = jobs.load(args.job_id)
    if args.resolve_voices_only:
        renderer = DirectAzureDubRenderer(settings, object())
        translation_path = args.translation or (
            jobs.job_dir(args.job_id) / "translation" / "transcript_zu.json"
        )
        translation = json.loads(translation_path.read_text(encoding="utf-8"))
        required_speaker_ids = sorted(
            {
                str(item.get("speaker_id") or item.get("speaker") or "").strip()
                for item in translation.get("segments", [])
                if str(item.get("speaker_id") or item.get("speaker") or "").strip()
            }
        )
        output_path = jobs.job_dir(args.job_id) / "direct_dub" / "voice_resolution.json"
        assignments = renderer._voice_assignments(
            job,
            args.voice_map,
            DirectDubOptions(
                same_speaker_gap_ms=args.same_speaker_gap_ms,
                max_azure_rate_percent=args.max_azure_rate,
                max_time_stretch_ratio=args.max_time_stretch,
                male_voice=args.male_voice,
                female_voice=args.female_voice,
                min_voice_family_confidence=args.min_voice_confidence,
                include_background=args.include_background,
            ),
            required_speaker_ids=required_speaker_ids,
            output_path=output_path,
            refresh_voice_analysis=args.force,
        )
        print(json.dumps({
            "voice_resolution_path": str(output_path),
            "assignments": assignments,
        }, ensure_ascii=False, indent=2))
        return 0

    backend = create_direct_azure_backend(
        settings,
        max_rate_percent=args.max_azure_rate,
    )
    renderer = DirectAzureDubRenderer(settings, backend)
    result = renderer.render(
        job,
        translation_path=args.translation,
        voice_map_path=args.voice_map,
        clean_background_stem=args.clean_background_stem,
        options=DirectDubOptions(
            same_speaker_gap_ms=args.same_speaker_gap_ms,
            max_azure_rate_percent=args.max_azure_rate,
            max_time_stretch_ratio=args.max_time_stretch,
            male_voice=args.male_voice,
            female_voice=args.female_voice,
            min_voice_family_confidence=args.min_voice_confidence,
            include_background=(
                args.include_background or args.clean_background_stem is not None
            ),
        ),
        force=args.force,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
