"""CLI wiring for :mod:`mathula_tv.viral_moments`.

This module is intentionally small so the main Mathula TV CLI needs only two
stable integration calls: ``add_viral_moments_command`` during parser creation
and ``run_viral_moments_command`` during dispatch.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from .viral_moments import ViralMomentsOptions, run_viral_moments_for_job


def add_viral_moments_command(commands: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Register ``mathula-tv viral-clips`` on the production CLI."""

    command = commands.add_parser(
        "viral-clips",
        help="Rank and render viral source moments from a long Mathula TV job",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    command.add_argument("job_id")
    command.add_argument("--output-dir", type=Path)
    command.add_argument("--force", action="store_true")
    command.add_argument(
        "--force-transcribe",
        action="store_true",
        help="Ignore an existing English transcript and run Azure STT again",
    )
    command.add_argument(
        "--live-operation",
        action="store_true",
        help="Authorize Azure provider calls when transcription or AI ranking is required",
    )
    command.add_argument("--dry-run", action="store_true")
    command.add_argument("--no-ai", action="store_true", help="Use deterministic ranking only")
    command.add_argument("--locale", default="en-ZA")
    command.add_argument("--max-speakers", type=int, default=12)
    command.add_argument("--clip-count", type=int, default=15)
    command.add_argument("--min-clip-seconds", type=float, default=22.0)
    command.add_argument("--target-clip-seconds", type=float, default=52.0)
    command.add_argument("--max-clip-seconds", type=float, default=90.0)
    command.add_argument("--candidate-limit", type=int, default=60)
    command.add_argument("--ai-batch-size", type=int, default=10)
    command.add_argument("--phrases-file", type=Path)
    command.add_argument("--ffmpeg-preset", default="veryfast")
    command.add_argument("--crf", type=int, default=20)
    command.add_argument("--start-padding-ms", type=int, default=350)
    command.add_argument("--end-padding-ms", type=int, default=650)
    command.add_argument("--speech-timeout-seconds", type=int, default=7200)
    command.add_argument("--openai-timeout-seconds", type=int, default=900)
    command.add_argument("--downstream-target-language", default="zu-ZA")
    return command


def _job_root(settings: Any, job_id: str) -> Path:
    return Path(settings.work_dir).resolve() / "jobs" / job_id


def _has_reusable_transcript(job_root: Path) -> bool:
    return any(
        path.is_file()
        for path in (
            job_root / "analysis/transcript_en.json",
            job_root / "analysis/transcript_en_raw.json",
            job_root / "analysis/azure_diarization.json",
            job_root / "analysis/azure_stt.json",
        )
    )


def _azure_openai_is_configured() -> bool:
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    key = (
        os.getenv("AZURE_OPENAI_API_KEY", "").strip()
        or os.getenv("AZURE_OPENAI_KEY", "").strip()
    )
    deployment = (
        os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip()
        or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "").strip()
        or os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "").strip()
    )
    return bool(endpoint and key and deployment)


def _provider_calls_required(args: argparse.Namespace, settings: Any) -> list[str]:
    reasons: list[str] = []
    job_root = _job_root(settings, args.job_id)
    if args.force_transcribe or not _has_reusable_transcript(job_root):
        reasons.append("Azure Speech transcription")
    if not args.no_ai and not args.dry_run and _azure_openai_is_configured():
        reasons.append("Azure OpenAI editorial ranking")
    return reasons


def run_viral_moments_command(args: argparse.Namespace, settings: Any) -> dict[str, Any]:
    """Validate provider authorization and execute the job-local module."""

    reasons = _provider_calls_required(args, settings)
    if args.dry_run and "Azure Speech transcription" in reasons:
        raise ValueError(
            "viral-clips --dry-run requires an existing English transcript; "
            "remove --force-transcribe or run Azure STT with --live-operation"
        )
    if reasons and not args.live_operation:
        joined = " and ".join(reasons)
        raise ValueError(
            f"viral-clips requires --live-operation for {joined}; "
            "use --no-ai to disable editorial AI ranking"
        )

    phrases_file = args.phrases_file
    if phrases_file is None:
        default_phrases = Path("config/madlanga_viral_phrases.txt")
        if default_phrases.is_file():
            phrases_file = default_phrases

    options = ViralMomentsOptions(
        locale=args.locale,
        max_speakers=args.max_speakers,
        clip_count=args.clip_count,
        min_clip_seconds=args.min_clip_seconds,
        target_clip_seconds=args.target_clip_seconds,
        max_clip_seconds=args.max_clip_seconds,
        candidate_limit=args.candidate_limit,
        ai_batch_size=args.ai_batch_size,
        phrases_file=phrases_file,
        no_ai=(args.no_ai or args.dry_run),
        dry_run=args.dry_run,
        ffmpeg_preset=args.ffmpeg_preset,
        crf=args.crf,
        start_padding_ms=args.start_padding_ms,
        end_padding_ms=args.end_padding_ms,
        speech_timeout_seconds=args.speech_timeout_seconds,
        openai_timeout_seconds=args.openai_timeout_seconds,
        force=args.force,
        downstream_target_language=args.downstream_target_language,
    )
    return run_viral_moments_for_job(
        job_id=args.job_id,
        work_dir=Path(settings.work_dir),
        options=options,
        output_dir=args.output_dir,
        force_transcribe=args.force_transcribe,
    )


__all__ = ["add_viral_moments_command", "run_viral_moments_command"]
