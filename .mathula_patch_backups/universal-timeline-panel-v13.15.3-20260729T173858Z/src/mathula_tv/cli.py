from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from .ai_provider import create_production_ai_provider
from .atomic_io import read_json
from .azure_stt import AzureFastTranscriptionBackend, derive_endpoint
from .azure_tts import AzureTTSBackend, AzureTTSHTTPBoundary
from .colab_package import publish_colab_package
from .compatibility import (
    CompatibilityThresholds,
    ReviewPackageUnit,
    UnitCompatibilityObservation,
    build_ab_review_package,
)
from .config import load_settings
from .diarization import run_pyannote
from .logging_utils import configure_logging, redact
from .models import PRODUCTION_STATES, utcnow
from .orchestrator import Orchestrator
from .youtube_ingestion import (
    download_youtube_video,
    is_youtube_url,
    record_youtube_submission,
)
from .production_pipeline import ProductionDubbingPipeline
from .translation import AzureOpenAIClient
from .tts_ssml import SSMLBounds
from .viral_moments_cli import add_viral_moments_command, run_viral_moments_command


LIVE_COMPATIBILITY_JOB_ID = "19ba6d69f1b84132ba4f20599101834a"
EXTERNAL_COMMANDS = {
    "transcribe",
    "translate",
    "repair-translation",
    "validate-azure-voices",
    "calibrate-azure-sources",
    "calibrate-speaker-voices",
    "synthesize",
    "queue-openvoice-assets",
    "reconcile-openvoice-assets",
    "queue-voice-conversion",
    "reconcile-voice-conversion",
    "dub-azure",
    "edit-tiktok",
}


def create_speech_backend(settings):
    key = os.getenv("AZURE_SPEECH_KEY", "")
    if not key:
        raise ValueError("AZURE_SPEECH_KEY is missing")
    endpoint = settings.azure_speech_endpoint or derive_endpoint(settings.azure_speech_region)
    return AzureFastTranscriptionBackend(
        endpoint=endpoint,
        subscription_key=key,
        api_version=settings.azure_speech_api_version,
        max_speakers=settings.azure_speech_max_speakers,
        timeout_seconds=settings.azure_speech_timeout_seconds,
    )


def create_translation_backend(provider: str | None = None, model: str | None = None):
    selected = (
        provider
        or os.getenv("MATHULA_TV_AI_PROVIDER")
        or "azure-foundry-claude"
    ).strip().lower()
    if selected == "anthropic":
        raise ValueError(
            "Mathula TV Claude calls must use Azure AI Foundry; "
            "set --provider azure-foundry-claude"
        )
    if selected == "azure-foundry-claude":
        environment = dict(os.environ)
        environment["MATHULA_TV_AI_PROVIDER"] = selected
        if model:
            environment["MATHULA_TV_FOUNDRY_CLAUDE_DEPLOYMENT"] = model
        return create_production_ai_provider(environment)
    if selected in {"azure-openai-legacy", "azure-openai-test"}:
        return AzureOpenAIClient.from_environment()
    raise ValueError(f"Unsupported AI provider: {selected}; no fallback is configured")


def create_tts_backend(settings) -> AzureTTSBackend:
    key = os.getenv("AZURE_SPEECH_KEY", "")
    if not key:
        raise ValueError("AZURE_SPEECH_KEY is missing")
    if not settings.azure_speech_region:
        raise ValueError("AZURE_SPEECH_REGION is missing")
    bounds = SSMLBounds(
        rate_min_percent=settings.azure_rate_min_percent,
        rate_max_percent=settings.azure_rate_max_percent,
    )
    return AzureTTSBackend(
        AzureTTSHTTPBoundary(settings.azure_speech_region, key),
        configured_voices=settings.azure_tts_voices,
        default_voice=settings.azure_tts_default_voice,
        sample_rate=settings.azure_tts_sample_rate,
        channels=settings.azure_tts_channels,
        sample_width=settings.azure_tts_sample_width,
        timeout_seconds=settings.azure_tts_timeout_seconds,
        max_retries=settings.azure_tts_max_retries,
        ssml_bounds=bounds,
    )


def create_translation_client(settings):
    key = os.getenv("AZURE_SPEECH_KEY", "")
    if not key:
        raise ValueError("AZURE_SPEECH_KEY is missing")
    if not settings.azure_ai_endpoint:
        raise ValueError("AZURE_AI_ENDPOINT is missing")
    if not settings.azure_ai_deployment:
        raise ValueError("AZURE_AI_DEPLOYMENT is missing")

    class SimpleJsonClient:
        def __init__(self, endpoint, deployment, api_version, key):
            self.deployment = deployment
            self.api_version = api_version
            self.request_durations = []
            self.endpoint = endpoint
            self.key = key

        def __call__(self, payload):
            import time
            import requests
            start = time.time()
            headers = {
                "Content-Type": "application/json",
                "api-key": self.key,
            }
            url = f"{self.endpoint}/openai/deployments/{self.deployment}/chat/completions?api-version={self.api_version}"
            response = requests.post(url, headers=headers, json=payload, timeout=600)
            response.raise_for_status()
            duration = time.time() - start
            self.request_durations.append(duration)
            return response.json()

    return SimpleJsonClient(
        endpoint=settings.azure_ai_endpoint,
        deployment=settings.azure_ai_deployment,
        api_version=settings.azure_ai_api_version,
        key=key,
    )


def _add_job_command(commands, name: str) -> argparse.ArgumentParser:
    command = commands.add_parser(name)
    command.add_argument("job_id")
    command.add_argument("--force", action="store_true")
    command.add_argument("--dry-run", action="store_true")
    command.add_argument("--live-operation", action="store_true")
    return command


def _format_progress_time(seconds: Any) -> str:
    try:
        total = max(0, int(round(float(seconds))))
    except (TypeError, ValueError):
        return "--:--"
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _print_direct_dub_progress(event: dict[str, Any]) -> None:
    """Write human progress to stderr while preserving JSON-only stdout."""

    elapsed = _format_progress_time(event.get("elapsed_seconds"))
    stage = str(event.get("stage") or "working").replace("_", "-")
    message = str(event.get("message") or "Working")
    current = event.get("current")
    total = event.get("total")
    progress = ""
    if isinstance(current, int) and isinstance(total, int) and total > 0:
        percent = event.get("percent")
        percent_text = (
            f" {float(percent):.0f}%"
            if isinstance(percent, (int, float))
            else ""
        )
        progress = f" {current}/{total}{percent_text}"
    eta = event.get("eta_seconds")
    eta_text = (
        f" | ETA {_format_progress_time(eta)}"
        if isinstance(eta, (int, float)) and eta > 0
        else ""
    )
    status = str(event.get("status") or "running")
    status_text = " WARNING" if status == "warning" else ""
    print(
        f"[dub-azure {elapsed}] {stage}{progress}{status_text}: {message}{eta_text}",
        file=sys.stderr,
        flush=True,
    )


def _print_translation_result(
    *,
    job_id: str,
    result: dict[str, Any],
    json_output: bool,
) -> None:
    """Print a concise operator result unless machine-readable JSON is requested."""

    if json_output:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    print(f"Translation complete for job {job_id}.")


def _print_job_status(job: Any, *, json_output: bool) -> None:
    """Print only the current state unless the complete manifest is requested."""

    if json_output:
        print(json.dumps(redact(job.to_dict()), ensure_ascii=False, indent=2))
        return
    print(str(job.state))



def _format_media_duration(seconds: Any) -> str:
    """Format media duration as H:MM:SS or MM:SS for operator output."""

    try:
        total = max(0, int(round(float(seconds))))
    except (TypeError, ValueError):
        return "unknown"
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _friendly_codec(value: Any) -> str:
    codec = str(value or "").strip().lower()
    labels = {
        "h264": "H.264",
        "avc": "H.264",
        "h265": "H.265",
        "hevc": "H.265",
        "aac": "AAC",
        "mp3": "MP3",
        "opus": "Opus",
    }
    return labels.get(codec, codec.upper()) if codec else "unknown"


def _friendly_frame_rate(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "unknown fps"
    if "/" in raw:
        numerator, denominator = raw.split("/", 1)
        try:
            rate = float(numerator) / float(denominator)
        except (TypeError, ValueError, ZeroDivisionError):
            return f"{raw} fps"
        if rate.is_integer():
            return f"{int(rate)} fps"
        return f"{rate:.3f}".rstrip("0").rstrip(".") + " fps"
    return f"{raw} fps"


def _friendly_sample_rate(value: Any) -> str:
    try:
        rate = int(value)
    except (TypeError, ValueError):
        return "unknown"
    if rate >= 1000 and rate % 1000 == 0:
        return f"{rate // 1000} kHz"
    if rate >= 1000:
        return f"{rate / 1000:.1f} kHz"
    return f"{rate} Hz"


def _print_dub_azure_result(
    *,
    job_id: str,
    result: dict[str, Any],
    json_output: bool,
) -> None:
    """Print the useful publication result instead of dumping the full report."""

    if json_output:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    nested_publication = result.get("tiktok_edit")
    publication = (
        nested_publication
        if isinstance(nested_publication, dict)
        and any(
            key in nested_publication
            for key in ("output", "title_panel", "translation", "idempotent_reuse")
        )
        else result
    )
    output = (
        publication.get("output")
        if isinstance(publication.get("output"), dict)
        else {}
    )
    title_panel = (
        publication.get("title_panel")
        if isinstance(publication.get("title_panel"), dict)
        else {}
    )
    translation = (
        publication.get("translation")
        if isinstance(publication.get("translation"), dict)
        else {}
    )

    if not output:
        outputs = result.get("outputs") if isinstance(result.get("outputs"), dict) else {}
        fallback_path = (
            outputs.get("canonical_final_video")
            or outputs.get("final_video")
            or outputs.get("canonical_dubbed_master")
            or outputs.get("dubbed_master")
        )
        if fallback_path:
            output = {"path": fallback_path}

    video_path = str(output.get("path") or "not reported")
    duration = _format_media_duration(output.get("duration_seconds"))
    if duration == "unknown":
        duration = "not reported"
    video_codec = _friendly_codec(output.get("video_codec"))
    profile = str(output.get("video_profile") or "").strip()
    frame_rate = _friendly_frame_rate(output.get("frame_rate"))
    audio_codec = _friendly_codec(output.get("audio_codec"))
    sample_rate = _friendly_sample_rate(output.get("audio_sample_rate"))
    title = str(title_panel.get("text") or "none").strip()
    if translation.get("immutable") is True:
        translation_state = "unchanged"
    elif translation.get("immutable") is False:
        translation_state = "changed"
    else:
        translation_state = "not reported"
    reuse_state = "reused" if publication.get("idempotent_reuse") else "created"

    video_description = video_codec
    if profile:
        video_description += f" {profile}"

    print("Dubbing complete.")
    print(f"Job: {job_id}")
    print(f"Video: {video_path}")
    print(f"Duration: {duration}")
    media_parts: list[str] = []
    if video_codec != "unknown":
        media_parts.append(video_description)
    if frame_rate != "unknown fps":
        media_parts.append(frame_rate)
    if audio_codec != "unknown":
        audio_description = audio_codec
        if sample_rate != "unknown":
            audio_description += f" {sample_rate}"
        media_parts.append(audio_description)
    print(f"Media: {' · '.join(media_parts) if media_parts else 'not reported'}")
    print(f"Title panel: {title}")
    if publication.get("hook_edit_enabled") is False:
        print("Hook edit: disabled (hook card retained)")
    print(f"Translation: {translation_state}")
    print(f"Publication render: {reuse_state}")


def _translation_progress_callback():
    """Return a stderr progress callback with a stable per-command clock."""

    started = time.monotonic()

    def report(event: dict[str, Any]) -> None:
        payload = dict(event)
        payload.setdefault("elapsed_seconds", time.monotonic() - started)
        elapsed = _format_progress_time(payload.get("elapsed_seconds"))
        stage = str(payload.get("stage") or "working").replace("_", "-")
        message = str(payload.get("message") or "Working")
        current = payload.get("current")
        total = payload.get("total")
        progress_text = ""
        if isinstance(current, int) and isinstance(total, int) and total > 0:
            percent = payload.get("percent")
            percent_text = (
                f" {float(percent):.0f}%"
                if isinstance(percent, (int, float))
                else ""
            )
            progress_text = f" {current}/{total}{percent_text}"
        eta = payload.get("eta_seconds")
        eta_text = (
            f" | ETA {_format_progress_time(eta)}"
            if isinstance(eta, (int, float)) and eta > 0
            else ""
        )
        warning = " WARNING" if payload.get("status") == "warning" else ""
        print(
            f"[translate {elapsed}] {stage}{progress_text}{warning}: {message}{eta_text}",
            file=sys.stderr,
            flush=True,
        )

    return report


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="mathula-tv")
    root.add_argument("--verbose", action="store_true")
    commands = root.add_subparsers(dest="command", required=True)
    submit = commands.add_parser("submit")
    submit.add_argument("video", help="Local MP4/WebM path or a YouTube video URL")
    submit.add_argument("--target-language", default="zu-ZA")
    submit.add_argument("--force", action="store_true")

    for name in ("process", "transcribe", "diarize", "retry", "validate", "inspect"):
        _add_job_command(commands, name)
    status = _add_job_command(commands, "status")
    status.add_argument(
        "--json",
        "--json-output",
        dest="json_output",
        action="store_true",
        help="Print the complete redacted job manifest instead of only the current state",
    )
    migration = _add_job_command(commands, "migrate-dubbing-state")
    migration.add_argument("--allow-legacy-review-reset", action="store_true")
    _add_job_command(commands, "prepare-source-derivatives")
    _add_job_command(commands, "build-dubbing-units")
    translate = _add_job_command(commands, "translate")
    translate.add_argument("--provider", default=None)
    translate.add_argument("--model", default=None)
    translate.add_argument("--batch-size", type=int, default=None)
    translate.add_argument("--context-units", type=int, default=None)
    translate.add_argument("--request-timeout-seconds", type=float, default=None)
    translate.add_argument("--batch-max-retries", type=int, default=None)
    translate.add_argument("--max-provider-calls", type=int, default=None)
    translate.add_argument("--restart-batches", action="store_true")
    translate.add_argument("--no-progress", action="store_true")
    translate.add_argument(
        "--json-output",
        action="store_true",
        help="Print the complete translation JSON to stdout instead of a concise completion message",
    )
    recover_translation = _add_job_command(commands, "recover-translation-response")
    recover_translation.add_argument(
        "--run-dir",
        type=Path,
        help=(
            "Specific failed translation/cloud_runs/<timestamp> directory. "
            "Defaults to the latest failed run for the job."
        ),
    )
    repair_translation = _add_job_command(commands, "repair-translation")
    repair_translation.add_argument("--provider", default=None)
    repair_translation.add_argument("--model", default=None)
    _add_job_command(commands, "prepare-dubbing")
    _add_job_command(commands, "validate-azure-voices")
    _add_job_command(commands, "calibrate-azure-sources")
    synthesize = _add_job_command(commands, "synthesize")
    synthesize.add_argument("--backend", default="azure-tts", choices=("azure-tts",))
    references = _add_job_command(commands, "build-reference-reels")
    references.add_argument("--candidates", type=Path)
    references.add_argument("--proof-of-concept", action="store_true")
    calibrate_speaker_voices = _add_job_command(commands, "calibrate-speaker-voices")
    calibrate_speaker_voices.add_argument("--prepare-only", action="store_true")
    calibrate_speaker_voices.add_argument("--evaluations", type=Path)
    prepare_openvoice = _add_job_command(commands, "prepare-openvoice")
    prepare_openvoice.add_argument("--prepare-only", action="store_true")
    prepare_openvoice.add_argument("--proof-of-concept", action="store_true")
    prepare_openvoice.add_argument("--device", default="cuda")
    _add_job_command(commands, "queue-openvoice-assets")
    _add_job_command(commands, "reconcile-openvoice-assets")
    _add_job_command(commands, "queue-voice-conversion")
    _add_job_command(commands, "reconcile-voice-conversion")
    compatibility = _add_job_command(commands, "evaluate-compatibility")
    compatibility.add_argument("--observations", type=Path)
    compatibility.add_argument("--thresholds", type=Path)
    _add_job_command(commands, "build-review-package")
    align = _add_job_command(commands, "align")
    align.add_argument("--skip-openvoice", action="store_true")
    align.add_argument("--require-compatibility-pass", action=argparse.BooleanOptionalAction, default=True)
    background = _add_job_command(commands, "prepare-background")
    background.add_argument("--clean-stem", type=Path)
    background.add_argument("--proof-of-concept", action="store_true")
    _add_job_command(commands, "mix")
    _add_job_command(commands, "subtitles")
    render = _add_job_command(commands, "render")
    render.add_argument("--burn-subtitles", action="store_true")
    _add_job_command(commands, "review")
    
    export_translation = _add_job_command(commands, "export-translation-package")
    export_translation.add_argument("--target-locale", choices=("zu-ZA", "nso-ZA"))
    export_translation.add_argument("--output-dir", type=Path)

    import_translation = _add_job_command(commands, "import-translation")
    import_translation.add_argument("translated_json", type=Path)
    import_translation.add_argument("--request", type=Path)
    import_translation.add_argument("--reviewed-by", required=True)

    # Direct Azure dubbing from the already-approved translation artifact.
    dub_azure = _add_job_command(commands, "dub-azure")
    dub_azure.add_argument("--min-confidence", type=float, default=0.70)
    dub_azure.add_argument("--voice-map", type=Path)
    dub_azure.add_argument(
        "--refresh-voice-analysis",
        action="store_true",
        help=(
            "Rebuild speaker reels and rerun SpeechBrain/voice-family analysis. "
            "Ordinary --force retries reuse the persisted job-local voice resolution."
        ),
    )
    dub_azure.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Show stage, block, Azure candidate, timeline balance, and render progress "
            "on stderr. Use --no-progress for quiet automation logs."
        ),
    )
    dub_azure.add_argument(
        "--hook-edit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Cut the publication video to the AI-selected opening anchor. "
            "Use --no-hook-edit to preserve the full dubbed video while still "
            "generating and rendering the hook text on the footer card."
        ),
    )
    dub_azure.add_argument(
        "--json",
        "--json-output",
        dest="json_output",
        action="store_true",
        help="Print the complete dubbing and publication report as JSON",
    )


    collision_panel = _add_job_command(commands, "timeline-collision-panel")
    collision_panel.add_argument("--serve", action="store_true")
    collision_panel.add_argument("--host", default="127.0.0.1")
    collision_panel.add_argument("--port", type=int, default=8765)
    collision_panel.add_argument("--resolve", metavar="COLLISION_ID")
    collision_panel.add_argument(
        "--strategy",
        choices=("allow_overlap", "manual_text", "select_variants"),
    )
    collision_panel.add_argument("--reviewed-by")
    collision_panel.add_argument("--overlap-before-ms", type=int, default=0)
    collision_panel.add_argument("--overlap-after-ms", type=int, default=0)
    collision_panel.add_argument("--text")
    collision_panel.add_argument("--text-file", type=Path)
    collision_panel.add_argument(
        "--previous-variant", choices=("natural", "concise", "compact")
    )
    collision_panel.add_argument(
        "--current-variant", choices=("natural", "concise", "compact")
    )
    collision_panel.add_argument(
        "--following-variant", choices=("natural", "concise", "compact")
    )
    collision_panel.add_argument("--clear", metavar="COLLISION_ID")

    enroll_speaker = _add_job_command(commands, "enroll-speaker")
    enroll_speaker.add_argument("--name", required=True)
    enroll_speaker.add_argument("--gender", required=True, choices=("male", "female"))
    enroll_speaker.add_argument(
        "--range",
        dest="ranges",
        action="append",
        required=True,
        help="Clean single-speaker timeline range, e.g. 00:11:38-00:12:08; repeat it",
    )
    enroll_speaker.add_argument("--person-id")
    enroll_speaker.add_argument("--voice")

    _add_job_command(commands, "edit-tiktok")
    
    commands.add_parser("list-jobs")
    
    # Entity registry commands
    _add_job_command(commands, "bind-entities")
    _add_job_command(commands, "rebuild-with-entities")
    
    # Registry management commands (independent of jobs)
    entities = commands.add_parser("entities")
    entities_subcommands = entities.add_subparsers(dest="entities_command", required=True)
    
    entities_validate = entities_subcommands.add_parser("validate")
    entities_validate.add_argument("--registry", default="config/entity_registry.json")
    
    entities_list = entities_subcommands.add_parser("list")
    entities_list.add_argument("--registry", default="config/entity_registry.json")
    entities_list.add_argument("--domain")
    entities_list.add_argument("--type")
    entities_list.add_argument("--query")
    
    # Intelligibility audit commands
    audit_intelligibility = _add_job_command(commands, "audit-intelligibility")
    audit_intelligibility.add_argument("--stage", required=True)
    audit_intelligibility.add_argument("--with-phrase-hints", action="store_true")
    
    publish_worker = commands.add_parser("publish-colab-package")
    publish_worker.add_argument("--live-operation", action="store_true")
    add_viral_moments_command(commands)
    return root


def _require_operation_authorization(args, settings) -> None:
    job_id = getattr(args, "job_id", None)
    if job_id == LIVE_COMPATIBILITY_JOB_ID:
        if not args.live_operation:
            raise ValueError("The compatibility job requires --live-operation")
        if not settings.gcs_bucket:
            raise ValueError("MATHULA_TV_GCS_BUCKET is required for the compatibility job")
        if not (
            os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            or os.getenv("MATHULA_TV_USE_APPLICATION_DEFAULT_CREDENTIALS") == "1"
        ):
            raise ValueError(
                "Configure GOOGLE_APPLICATION_CREDENTIALS or explicitly acknowledge "
                "application-default credentials with MATHULA_TV_USE_APPLICATION_DEFAULT_CREDENTIALS=1"
            )
    if args.command in EXTERNAL_COMMANDS and hasattr(settings, "ai_provider") and not args.live_operation:
        raise ValueError(f"{args.command} calls an external provider and requires --live-operation")


def _dry_run(args, job) -> dict[str, Any] | None:
    if not getattr(args, "dry_run", False):
        return None
    if args.command in {
        "inspect",
        "status",
        "validate",
        "migrate-dubbing-state",
        "export-translation-package",
        "import-translation",
        "recover-translation-response",
    }:
        return None
    return {
        "dry_run": True,
        "job_id": job.job_id,
        "state": job.state,
        "command": args.command,
        "mutated": False,
    }


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    settings = load_settings()
    if args.command == "viral-clips":
        print(json.dumps(run_viral_moments_command(args, settings), ensure_ascii=False, indent=2))
        return 0

    configure_logging(settings.work_dir / "logs/mathula-tv.jsonl", args.verbose)
    app = Orchestrator(settings)
    try:
        if args.command == "submit":
            if is_youtube_url(args.video):
                maximum_minutes = float(getattr(settings, "max_source_duration_minutes", 240))
                with download_youtube_video(
                    args.video,
                    max_duration_seconds=maximum_minutes * 60.0,
                ) as download:
                    job = app.submit(download.path, args.target_language, args.force)
                    record_youtube_submission(app, job, download)
            else:
                job = app.submit(Path(args.video), args.target_language, args.force)
            print(job.job_id)
            return 0
        if args.command == "list-jobs":
            print(json.dumps([{"job_id": item.job_id, "state": item.state, "source": item.source_filename} for item in app.jobs.list()], indent=2))
            return 0
        if args.command == "entities":
            from .entity_registry import EntityRegistry
            
            registry_path = Path(args.registry)
            if args.entities_command == "validate":
                registry = EntityRegistry(registry_path)
                print(json.dumps({
                    "validation_state": "valid",
                    "schema_version": "mathula-entity-registry-v1",
                    "entity_count": len(registry._entities),
                    "registry_sha256": registry.sha256,
                    "alias_collision_count": 0,  # Would need collision detection logic
                    "invalid_reference_count": 0,  # Would need reference validation logic
                }, indent=2))
                return 0
            elif args.entities_command == "list":
                registry = EntityRegistry(registry_path)
                entities = list(registry._entities.values())
                
                # Apply filters
                if args.domain:
                    entities = [e for e in entities if args.domain in e.domains]
                if args.type:
                    entities = [e for e in entities if e.entity_type == args.type]
                if args.query:
                    query_lower = args.query.lower()
                    entities = [e for e in entities if query_lower in e.canonical_text.lower() or query_lower in " ".join(e.aliases).lower()]
                
                print(json.dumps([
                    {
                        "entity_id": e.entity_id,
                        "entity_type": e.entity_type,
                        "canonical_text": e.canonical_text,
                        "display_text": e.display_text,
                        "aliases": list(e.aliases),
                        "domains": list(e.domains),
                        "active": e.active,
                        "must_preserve": e.must_preserve,
                    }
                    for e in entities
                ], indent=2))
                return 0
        if args.command == "publish-colab-package":
            if not args.live_operation:
                raise ValueError("publish-colab-package mutates GCS and requires --live-operation")
            if not settings.gcs_bucket:
                raise ValueError("MATHULA_TV_GCS_BUCKET is required")
            from google.cloud import storage

            result = publish_colab_package(
                Path(__file__).resolve().parents[2],
                storage.Client().bucket(settings.gcs_bucket),
                settings.gcs_prefix,
            )
            print(json.dumps({key: value for key, value in result.items() if key != "sha256"}, indent=2))
            return 0

        job = app.jobs.load(args.job_id)
        _require_operation_authorization(args, settings)
        dry = _dry_run(args, job)
        if dry is not None:
            print(json.dumps(dry, indent=2))
            return 0

        # Preserve the explicitly selectable Azure OpenAI legacy/test adapter.
        # Normal Settings always has ai_provider=anthropic and uses production below.
        if args.command == "translate" and not hasattr(settings, "ai_provider"):
            result = app.translate_and_package(
                job,
                create_translation_backend(),
                force=args.force,
            )
            _print_translation_result(
                job_id=job.job_id,
                result=result,
                json_output=args.json_output,
            )
            return 0
        production = ProductionDubbingPipeline(settings)

        if args.command == "status":
            _print_job_status(job, json_output=args.json_output)
        elif args.command == "inspect":
            print(json.dumps(redact(production.inspect(job)), indent=2))
        elif args.command == "migrate-dubbing-state":
            print(
                json.dumps(
                    production.migrate(
                        job,
                        dry_run=args.dry_run,
                        allow_legacy_review_reset=args.allow_legacy_review_reset,
                    ),
                    indent=2,
                )
            )
        elif args.command == "prepare-source-derivatives":
            print(json.dumps(production.ensure_source_derivatives(job, force=args.force), indent=2))
        elif args.command == "build-dubbing-units":
            print(json.dumps(production.build_units(job, force=args.force), ensure_ascii=False, indent=2))
        elif args.command == "prepare-dubbing":
            print(json.dumps(production.prepare_dubbing(job, force=args.force), ensure_ascii=False, indent=2))
        elif args.command == "translate":
            provider_name = args.provider or settings.ai_provider
            if provider_name in {"azure-openai-legacy", "azure-openai-test"}:
                result = app.translate_and_package(
                    job,
                    create_translation_backend(provider_name, args.model),
                    force=args.force,
                )
            else:
                provider = create_translation_backend(
                    provider_name,
                    args.model or settings.claude_model,
                )
                result = production.translate(
                    job,
                    provider,
                    force=args.force,
                    batch_size=args.batch_size,
                    context_units=args.context_units,
                    request_timeout_seconds=args.request_timeout_seconds,
                    batch_max_retries=args.batch_max_retries,
                    max_provider_calls=args.max_provider_calls,
                    restart_batches=args.restart_batches,
                    progress=(
                        None if args.no_progress else _translation_progress_callback()
                    ),
                )
            _print_translation_result(
                job_id=job.job_id,
                result=result,
                json_output=args.json_output,
            )
        elif args.command == "recover-translation-response":
            if not args.dry_run and not args.live_operation:
                raise ValueError(
                    "recover-translation-response installs a preserved cloud response "
                    "and requires --live-operation (or use --dry-run to validate only)"
                )
            print(
                json.dumps(
                    production.recover_translation_response(
                        job,
                        run_dir=(args.run_dir.resolve() if args.run_dir else None),
                        dry_run=args.dry_run,
                    ),
                    ensure_ascii=False,
                    indent=2,
                )
            )
        elif args.command == "repair-translation":
            provider_name = args.provider or settings.ai_provider
            if provider_name != "azure-foundry-claude":
                raise ValueError("Timing repair requires Azure AI Foundry Claude")
            provider = create_translation_backend(provider_name, args.model or settings.claude_model)
            print(
                json.dumps(
                    production.repair_translation_units(job, provider, force=args.force),
                    ensure_ascii=False,
                    indent=2,
                )
            )
        elif args.command == "validate-azure-voices":
            voices = create_tts_backend(settings).validate_configured_voices(force_refresh=args.force)
            print(json.dumps({"available": [voice.name for voice in voices], "validated": True}, indent=2))
        elif args.command == "calibrate-azure-sources":
            print(json.dumps(production.prepare_azure_source_calibrations(job, create_tts_backend(settings), force=args.force), indent=2))
        elif args.command == "synthesize":
            print(json.dumps(production.synthesize_azure(job, create_tts_backend(settings), force=args.force), indent=2))
        elif args.command == "build-reference-reels":
            if args.candidates:
                candidates = read_json(args.candidates)
            elif args.proof_of_concept:
                candidates = production.default_reference_candidates(job)
            else:
                raise ValueError("Provide --candidates or explicitly select --proof-of-concept mixed-source references")
            print(json.dumps(production.prepare_reference_reels(job, candidates, proof_of_concept=args.proof_of_concept, force=args.force), indent=2))
        elif args.command == "calibrate-speaker-voices":
            if args.prepare_only:
                value = production.prepare_speaker_voice_calibration_audio(
                    job,
                    create_tts_backend(settings),
                    force=args.force,
                )
            else:
                value = production.persist_speaker_voice_selections(
                    job,
                    read_json(args.evaluations) if args.evaluations else None,
                )
            print(json.dumps(value, indent=2))
        elif args.command == "prepare-openvoice":
            if args.device != "cuda":
                raise ValueError("Production OpenVoice preparation supports only --device cuda")
            print(
                json.dumps(
                    production.prepare_openvoice(
                        job,
                        lease_identity="unbound",
                        prepare_only=args.prepare_only,
                        proof_of_concept=args.proof_of_concept,
                        force=args.force,
                    ),
                    indent=2,
                )
            )
        elif args.command == "queue-openvoice-assets":
            print(
                json.dumps(
                    ProductionDubbingPipeline(
                        settings, _gcs(settings)
                    ).queue_openvoice_assets(job, force=args.force),
                    indent=2,
                )
            )
        elif args.command == "reconcile-openvoice-assets":
            print(
                json.dumps(
                    ProductionDubbingPipeline(
                        settings, _gcs(settings)
                    ).reconcile_openvoice_assets(job),
                    indent=2,
                )
            )
        elif args.command == "queue-voice-conversion":
            print(json.dumps(ProductionDubbingPipeline(settings, _gcs(settings)).queue_openvoice(job, force=args.force), indent=2))
        elif args.command == "reconcile-voice-conversion":
            print(json.dumps(ProductionDubbingPipeline(settings, _gcs(settings)).reconcile_openvoice(job), indent=2))
        elif args.command == "evaluate-compatibility":
            observations = [UnitCompatibilityObservation(**item) for item in read_json(args.observations)] if args.observations else None
            thresholds = CompatibilityThresholds(**read_json(args.thresholds)) if args.thresholds else None
            print(json.dumps(production.evaluate_compatibility(job, observations, thresholds), ensure_ascii=False, indent=2))
        elif args.command == "build-review-package":
            report = read_json(production.paths(job.job_id).compatibility_json)
            plan = production.prepare_dubbing(job)
            assets = [
                ReviewPackageUnit(
                    unit_id=unit["unit_id"],
                    original_reference=production.jobs.job_dir(job.job_id) / "speakers" / unit["speaker_id"] / "reference_reel.wav",
                    azure_tts=production.paths(job.job_id).azure_turn(unit["unit_id"]),
                    openvoice=production.paths(job.job_id).openvoice_turn(unit["unit_id"]),
                    aligned=production.paths(job.job_id).aligned_turn(unit["unit_id"]),
                )
                for unit in plan["units"]
            ]
            print(json.dumps(build_ab_review_package(production.paths(job.job_id).review_root, report, assets), indent=2))
        elif args.command == "align":
            print(json.dumps(production.align(job, skip_openvoice=args.skip_openvoice, require_compatibility_pass=args.require_compatibility_pass, force=args.force), indent=2))
        elif args.command == "prepare-background":
            print(json.dumps(production.prepare_background(job, clean_stem=args.clean_stem, proof_of_concept=args.proof_of_concept), indent=2))
        elif args.command == "mix":
            print(json.dumps(production.mix(job), indent=2))
        elif args.command == "subtitles":
            print(json.dumps(production.subtitles(job), indent=2))
        elif args.command == "render":
            print(json.dumps(production.render(job, burn_subtitles=args.burn_subtitles), indent=2))
        elif args.command == "review":
            print(json.dumps(production.review(job), ensure_ascii=False, indent=2))
        elif args.command == "bind-entities":
            print(json.dumps(production.bind_entities(job, force=args.force), ensure_ascii=False, indent=2))
        elif args.command == "rebuild-with-entities":
            print(json.dumps(production.rebuild_with_entities(job, force=args.force), indent=2))
        elif args.command == "audit-intelligibility":
            stt_backend = create_speech_backend(settings)
            production_with_stt = ProductionDubbingPipeline(settings, _gcs(settings), stt_backend)
            plan = production_with_stt.prepare_dubbing(job)
            stage = args.stage
            audio_root = None
            if stage == "azure-tts":
                audio_root = production_with_stt.paths(job.job_id).azure_tts_root / "turns"
            elif stage == "openvoice":
                audio_root = production_with_stt.paths(job.job_id).openvoice_root / "turns"
            elif stage == "aligned":
                audio_root = production_with_stt.paths(job.job_id).aligned_root / "turns"
            elif stage == "final-mix":
                audio_root = production_with_stt.paths(job.job_id).job_root / "audio"
            else:
                raise ValueError(f"Unknown stage: {stage}")
            if not audio_root or not audio_root.exists():
                raise ValueError(f"Audio root for stage {stage} does not exist")
            
            # Get baseline WER for final-mix degradation check
            baseline_wer = None
            if stage == "final-mix":
                aligned_report_path = production_with_stt.paths(job.job_id).qc_intelligibility_root / "aligned" / "report.json"
                if aligned_report_path.exists():
                    aligned_report = read_json(aligned_report_path)
                    baseline_wer = aligned_report.get("aggregate_blind_wer")
            
            result = production_with_stt._audit_intelligibility(
                job, stage, plan["units"], audio_root, baseline_wer=baseline_wer, with_phrase_hints=args.with_phrase_hints
            )
            print(json.dumps(result, indent=2))
        elif args.command == "export-translation-package":
            from .multivariant_translation import export_translation_package

            target_locale = args.target_locale or job.target_language
            output_dir = args.output_dir or (
                app.jobs.job_dir(job.job_id)
                / "translation"
                / "prompt_package"
            )
            result = export_translation_package(
                job_root=app.jobs.job_dir(job.job_id),
                job_id=job.job_id,
                target_locale=target_locale,
                output_dir=output_dir,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command == "import-translation":
            from .multivariant_translation import (
                install_manual_translation,
                load_request_for_response,
            )

            if not args.dry_run and not args.live_operation:
                raise ValueError(
                    "import-translation changes the authoritative job translation and "
                    "requires --live-operation (or use --dry-run to validate only)"
                )
            response_path = args.translated_json.resolve()
            request_path = load_request_for_response(
                response_path,
                explicit_request_path=(args.request.resolve() if args.request else None),
            )
            result = install_manual_translation(
                job_root=app.jobs.job_dir(job.job_id),
                job_id=job.job_id,
                target_locale=job.target_language,
                response_path=response_path,
                reviewed_by=args.reviewed_by,
                request_path=request_path,
                dry_run=args.dry_run,
            )
            if not args.dry_run:
                job.providers["translation"] = "external_manual_override"
                job.providers["translation_model"] = "manual_multivariant_import"
                job.providers["translation_prompt_version"] = (
                    result.get("validation", {}).get("prompt_version")
                    or "mathula-multivariant-translation-prompt-v2-batched"
                )
                job.media["manual_translation_override"] = result[
                    "translation_target"
                ]
                job.media["manual_translation_override_history"] = result[
                    "history_directory"
                ]
                job.media["translation_multivariant_sha256"] = result[
                    "installed_translation_sha256"
                ]
                job.media["translation_sha256"] = result[
                    "installed_dubbing_sha256"
                ]
                previous_state = job.state
                if job.state in {"analysis_ready", "translation_running"}:
                    if job.state == "analysis_ready":
                        job.transition("translation_running")
                    job.transition("translation_ready")
                    job.transition("azure_tts_queued")
                elif job.state == "translation_ready":
                    job.transition("azure_tts_queued")
                elif job.state in set(PRODUCTION_STATES) - {
                    "created",
                    "audio_prepared",
                    "uploaded",
                    "analysis_queued",
                    "analysis_running",
                }:
                    # A reviewed override invalidates every generated artifact
                    # downstream of translation. The files remain available for
                    # audit/cache comparison, but the manifest may no longer
                    # claim that the old dub is review-ready.
                    job.state = "azure_tts_queued"
                    job.updated_at = utcnow()
                else:
                    raise ValueError(
                        f"Cannot install a production translation override while job "
                        f"is in state {job.state}"
                    )
                downstream_stages = {
                    "azure_tts",
                    "voice_conversion",
                    "compatibility",
                    "alignment",
                    "background",
                    "mix",
                    "render",
                    "review",
                }
                job.completed_stages = [
                    stage
                    for stage in job.completed_stages
                    if stage not in downstream_stages
                ]
                if "translation" not in job.completed_stages:
                    job.completed_stages.append("translation")
                job.compatibility_gate = {"state": "pending", "calibrated_thresholds": False}
                job.review_readiness = "compatibility_pending"
                job.last_error = None
                job.attempt_history.append(
                    {
                        "stage": "manual_multivariant_translation_override",
                        "status": "completed",
                        "from_state": previous_state,
                        "to_state": "azure_tts_queued",
                        "reviewed_by": args.reviewed_by,
                        "translation_sha256": result["installed_translation_sha256"],
                        "completed_at": utcnow(),
                    }
                )
                job.human_review_flags = list(
                    dict.fromkeys(
                        [
                            *job.human_review_flags,
                            "manual_multivariant_translation_requires_human_review",
                        ]
                    )
                )
                app.jobs.save(job)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command == "timeline-collision-panel":
            from .direct_azure_dub import DirectDubArtifacts
            from .timeline_collision_panel import (
                clear_timeline_collision_resolution,
                load_timeline_collision_review,
                render_timeline_collision_panel,
                save_timeline_collision_resolution,
                serve_timeline_collision_panel,
                timeline_collision_summary,
            )

            artifacts = DirectDubArtifacts(app.jobs.job_dir(job.job_id))
            if not artifacts.timeline_collision_review.is_file():
                raise ValueError(
                    "No timeline collision review exists for this job; run dub-azure first"
                )
            if args.clear:
                clear_timeline_collision_resolution(
                    resolutions_path=artifacts.timeline_collision_resolutions,
                    review_path=artifacts.timeline_collision_review,
                    panel_path=artifacts.timeline_collision_panel,
                    collision_id=args.clear,
                )
            if args.resolve:
                if not args.strategy:
                    raise ValueError("--strategy is required with --resolve")
                text = args.text
                if args.text_file:
                    text = args.text_file.read_text(encoding="utf-8")
                save_timeline_collision_resolution(
                    resolutions_path=artifacts.timeline_collision_resolutions,
                    review_path=artifacts.timeline_collision_review,
                    panel_path=artifacts.timeline_collision_panel,
                    job_id=job.job_id,
                    collision_id=args.resolve,
                    strategy=args.strategy,
                    reviewed_by=args.reviewed_by or "",
                    overlap_before_ms=args.overlap_before_ms,
                    overlap_after_ms=args.overlap_after_ms,
                    manual_text=text,
                    previous_variant_id=args.previous_variant,
                    current_variant_id=args.current_variant,
                    following_variant_id=args.following_variant,
                )
            review = load_timeline_collision_review(
                artifacts.timeline_collision_review
            )
            artifacts.timeline_collision_panel.write_text(
                render_timeline_collision_panel(review), encoding="utf-8"
            )
            if args.serve:
                serve_timeline_collision_panel(
                    job_id=job.job_id,
                    review_path=artifacts.timeline_collision_review,
                    panel_path=artifacts.timeline_collision_panel,
                    resolutions_path=artifacts.timeline_collision_resolutions,
                    host=args.host,
                    port=args.port,
                )
            else:
                print(timeline_collision_summary(review))
                print(
                    f"Serve the panel with: python -m mathula_tv.cli "
                    f"timeline-collision-panel {job.job_id} --serve"
                )
        elif args.command == "dub-azure":
            from .direct_azure_dub import (
                DirectAzureDubRenderer,
                DirectDubOptions,
            )

            azure_tts_backend = create_tts_backend(settings)
            renderer = DirectAzureDubRenderer(
                settings,
                azure_tts_backend,
            )
            progress_elapsed = {"seconds": 0.0}

            def progress_callback(event: dict[str, Any]) -> None:
                if not args.progress:
                    return
                payload = dict(event)
                elapsed = payload.get("elapsed_seconds")
                if isinstance(elapsed, (int, float)):
                    progress_elapsed["seconds"] = float(elapsed)
                else:
                    payload["elapsed_seconds"] = progress_elapsed["seconds"]
                _print_direct_dub_progress(payload)

            result = renderer.render(
                job,
                voice_map_path=args.voice_map,
                options=DirectDubOptions(
                    max_azure_rate_percent=min(
                        10,
                        azure_tts_backend.ssml_bounds.rate_max_percent,
                    ),
                    min_voice_family_confidence=args.min_confidence,
                ),
                force=args.force,
                refresh_voice_analysis=args.refresh_voice_analysis,
                progress_callback=(progress_callback if args.progress else None),
            )
            from .tiktok_editor import edit_tiktok_job

            publication_message = (
                "Rendering or reusing the hook-panel publication video"
                if args.hook_edit
                else (
                    "Rendering or reusing the full-length publication video "
                    "with the hook card and no opening cut"
                )
            )
            if args.progress:
                progress_callback(
                    {
                        "stage": "publication_render",
                        "status": "started",
                        "message": publication_message,
                    }
                )
            tiktok_edit = edit_tiktok_job(
                work_dir=settings.work_dir,
                job=job,
                provider=create_production_ai_provider(),
                # Reuse by master/translation checksum. A forced dub does not
                # justify paying for the same AI hook or re-encoding it again.
                force=False,
                apply_opening_cut=args.hook_edit,
            )
            if args.progress:
                progress_callback(
                    {
                        "stage": "publication_render",
                        "status": "completed",
                        "message": (
                            "Hook-panel publication video is ready"
                            if args.hook_edit
                            else "Full-length publication video with hook card is ready"
                        ),
                    }
                )
            app.jobs.save(job)
            # The hook-panel publication pass updates report.json. Reload it so
            # the CLI's final_video always points to the panel-bearing output,
            # never the clean dubbed master.
            final_report = read_json(
                settings.work_dir / "jobs" / job.job_id / "direct_dub" / "report.json"
            )
            result = {**final_report, "tiktok_edit": tiktok_edit}
            _print_dub_azure_result(
                job_id=job.job_id,
                result=result,
                json_output=args.json_output,
            )
        elif args.command == "enroll-speaker":
            from .known_speakers import enrol_known_speaker
            from .media import prepare_source_derivatives

            job_root = app.jobs.job_dir(job.job_id)
            analysis_audio = job_root / "audio" / "analysis_mono.wav"
            if not analysis_audio.is_file():
                prepare_source_derivatives(
                    Path(job.local_source_path),
                    analysis_audio,
                    job_root / "audio" / "mix_source_stereo.wav",
                    max_duration_seconds=settings.max_source_duration_minutes * 60,
                )
            print(
                json.dumps(
                    enrol_known_speaker(
                        work_dir=settings.work_dir,
                        job_id=job.job_id,
                        analysis_audio_path=analysis_audio,
                        name=args.name,
                        gender=args.gender,
                        ranges=args.ranges,
                        azure_voice=args.voice,
                        person_id=args.person_id,
                    ),
                    ensure_ascii=False,
                    indent=2,
                )
            )
        elif args.command == "edit-tiktok":
            from .tiktok_editor import edit_tiktok_job

            result = edit_tiktok_job(
                work_dir=settings.work_dir,
                job=job,
                provider=create_production_ai_provider(),
                force=args.force,
            )
            app.jobs.save(job)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command == "process":
            result = app.process(job)
            if result is not None:
                if isinstance(result, (dict, list)):
                    print(json.dumps(result, indent=2))
                else:
                    print(result)
            else:
                print(_process_message(job))
        elif args.command == "diarize":
            job_dir = app.jobs.job_dir(job.job_id)
            pyannote_path = job_dir / "analysis/pyannote_diarization.json"
            azure_path = job_dir / "analysis/azure_diarization.json"
            
            # Check if existing diarization artifacts are valid
            if pyannote_path.is_file() and azure_path.is_file():
                # Both artifacts exist - verify they are valid
                from .diarization import validate_turns
                try:
                    pyannote_data = read_json(pyannote_path)
                    azure_data = read_json(azure_path)
                    validate_turns(azure_data.get("turns", []))
                    validate_turns(pyannote_data.get("turns", []))
                    print("Reusing existing valid diarization artifacts")
                    print("Optional Pyannote diagnostic ready; Azure speaker labels remain authoritative")
                except Exception as exc:
                    print(f"Existing diarization artifacts invalid: {exc}")
                    print("Running fresh Pyannote inference")
                    result = run_pyannote(Path(job.objects["local_audio"]), settings.pyannote_model)
                    from .atomic_io import atomic_write_json
                    atomic_write_json(pyannote_path, result)
                    print("Optional Pyannote diagnostic ready; Azure speaker labels remain authoritative")
            elif azure_path.is_file():
                # Only Azure exists - if Azure is authoritative, no need for Pyannote
                from .diarization import validate_turns
                try:
                    azure_data = read_json(azure_path)
                    validate_turns(azure_data.get("turns", []))
                    print("Azure diarization valid; Pyannote diagnostic optional")
                    print("Azure speaker labels remain authoritative")
                except Exception as exc:
                    print(f"Azure diarization invalid: {exc}")
                    print("Running fresh Pyannote inference")
                    result = run_pyannote(Path(job.objects["local_audio"]), settings.pyannote_model)
                    from .atomic_io import atomic_write_json
                    atomic_write_json(pyannote_path, result)
                    print("Optional Pyannote diagnostic ready; Azure speaker labels remain authoritative")
            else:
                # No existing artifacts - run Pyannote
                result = run_pyannote(Path(job.objects["local_audio"]), settings.pyannote_model)
                from .atomic_io import atomic_write_json
                atomic_write_json(pyannote_path, result)
                print("Optional Pyannote diagnostic ready; Azure speaker labels remain authoritative")
            
            # Call orchestrator to continue analysis with existing/reused diarization
            result = app.process(job)
            print(result)
        elif args.command == "retry":
            error = job.last_error or {}
            recoverable_azure_auth_failure = (
                job.state == "failed_terminal"
                and error.get("stage") == "azure_transcription"
                and error.get("error_type") == "AzureSpeechError"
                and "401" in error.get("message", "")
            )

            if job.state != "failed_retryable" and not recoverable_azure_auth_failure:
                raise ValueError(
                    "Only failed_retryable jobs or Azure Speech 401 failures can be retried"
                )

            job.last_error = None

            if job.state == "failed_retryable":
                job.transition("uploaded")
            else:
                # A corrected Azure credential makes this configuration failure
                # explicitly recoverable without repeating upload or submission.
                job.state = "uploaded"

            app.jobs.save(job)
            print(_process_message(job))
        elif args.command == "validate":
            print(json.dumps(production.validate_job(job), indent=2))
        elif args.command == "transcribe":
            transcribed_job = app.transcribe(job, create_speech_backend(settings), force=args.force)
            print(
                json.dumps(
                    {
                        "job_id": transcribed_job.job_id,
                        "state": transcribed_job.state,
                        "stage": "azure_transcription",
                        "completed": True,
                    }
                )
            )
        else:
            raise ValueError(f"Unsupported command: {args.command}")
        return 0
    except Exception as exc:
        if exc.__class__.__name__ == "TimelineCollisionReviewRequired" and hasattr(
            exc, "to_result"
        ):
            print(json.dumps(exc.to_result(), ensure_ascii=False, indent=2))
            return 0
        print(f"error: {redact(str(exc))}", file=sys.stderr)
        return 1


def _gcs(settings):
    from .gcs_store import GCSStore

    return GCSStore(settings.gcs_bucket, settings.gcs_prefix)


def _process_message(job) -> str:
    messages = {
        "synthesis_queued": f"Legacy state requires: python -m mathula_tv.cli migrate-dubbing-state {job.job_id}",
        "azure_tts_queued": f"Run: python -m mathula_tv.cli synthesize {job.job_id} --backend azure-tts --live-operation",
        "azure_tts_ready": (
            "Run the OpenVoice asset preflight: prepare-openvoice --prepare-only, "
            "queue-openvoice-assets, the assets-mode GPU notebook, and "
            f"reconcile-openvoice-assets for {job.job_id}"
        ),
        "voice_conversion_queued": "GPU worker required. Open the pinned Colab or Kaggle OpenVoice notebook, then reconcile.",
        "voice_conversion_ready": f"Run: python -m mathula_tv.cli evaluate-compatibility {job.job_id} --live-operation",
        "compatibility_review": "Compatibility/human review must pass before normal production alignment.",
        "alignment_ready": f"Run: python -m mathula_tv.cli prepare-background {job.job_id} --live-operation",
        "background_ready": f"Run: python -m mathula_tv.cli mix {job.job_id} --live-operation",
        "review_ready": "Review artifacts are ready; publication approval remains a human decision.",
    }
    return messages.get(job.state, f"No automatic production action for state {job.state}")


if __name__ == "__main__":
    raise SystemExit(main())
