from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from .ai_provider import create_production_ai_provider
from .foundry_grok import create_foundry_grok_provider
from .atomic_io import atomic_write_json, read_json
from .azure_stt import AzureFastTranscriptionBackend, derive_endpoint
from .azure_tts import AzureTTSBackend, AzureTTSHTTPBoundary
from .azure_tts_sdk import AzureSpeechSDKLiveBoundary
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
from .media import checksum
from .models import PRODUCTION_STATES, utcnow
from .orchestrator import Orchestrator
from .youtube_ingestion import (
    download_youtube_video,
    is_youtube_url,
    record_youtube_submission,
)
from .production_pipeline import ProductionDubbingPipeline
from .tts_ssml import SSMLBounds
from .viral_moments_cli import add_viral_moments_command, run_viral_moments_command


LIVE_COMPATIBILITY_JOB_ID = "19ba6d69f1b84132ba4f20599101834a"
EXTERNAL_COMMANDS = {
    "transcribe",
    "resume-autocorrect-research",
    "repair-speaker-turns",
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
    "native-translate",
    "native-qa",
    "native-dub",
    "optimize-dub",
    "edit-tiktok",
}
_POST_AUTOCORRECT_STAGES = {
    "classification",
    "context",
    "entity_binding",
    "translation",
    "azure_tts",
    "voice_conversion",
    "compatibility",
    "alignment",
    "background",
    "mix",
    "render",
    "review",
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
        or "azure-openai-gpt"
    ).strip().lower()
    if selected in {"anthropic", "azure-foundry-claude"}:
        raise ValueError(
            "v13.18.54 is GPT-only; use v13.18.51 or older for Claude"
        )
    if selected in {"manual-chatgpt", "manual", "chatgpt-manual", "manual-gpt"}:
        environment = dict(os.environ)
        environment["MATHULA_TV_AI_PROVIDER"] = "manual-chatgpt"
        if model:
            environment["MATHULA_TV_MANUAL_AI_MODEL"] = model
        return create_production_ai_provider(environment)
    if selected in {"azure-openai-gpt", "azure-openai", "gpt"}:
        environment = dict(os.environ)
        environment["MATHULA_TV_AI_PROVIDER"] = "azure-openai-gpt"
        if model:
            environment["AZURE_OPENAI_CHAT_DEPLOYMENT"] = model
            environment["AZURE_AI_DEPLOYMENT"] = model
        return create_production_ai_provider(environment)
    raise ValueError(f"Unsupported AI provider: {selected}; no fallback is configured")


def create_native_grok_backend(
    model: str | None = None,
    *,
    progress: Callable[[str], None] | None = None,
):
    environment = dict(os.environ)
    if model:
        environment["AZURE_FOUNDRY_GROK_DEPLOYMENT"] = model
    return create_foundry_grok_provider(environment, progress=progress)


def _print_native_progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _print_referent_audit_summary(summary: Mapping[str, Any] | None) -> None:
    if not summary:
        return
    unresolved = summary.get("unresolved_group_ids") or []
    print(
        f"Referent audit: {int(summary.get('groups_checked') or 0)} group(s) checked, "
        f"{int(summary.get('groups_flagged') or 0)} flagged, "
        f"{int(summary.get('repairs_applied') or 0)} repaired"
        + (f", {len(unresolved)} unresolved" if unresolved else "")
        + "."
    )


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


def create_sdk_boundary(settings) -> AzureSpeechSDKLiveBoundary:
    """Construct the Speech SDK boundary for grouped bookmark synthesis.

    Reuses the exact same AZURE_SPEECH_REGION/AZURE_SPEECH_KEY credentials and TTS
    audio format already configured for the REST path (see create_tts_backend) --
    no separate credential plumbing. Only called when enable_sdk_group_synthesis is
    on, so the lazy Speech SDK import (see azure_tts_sdk.build_speech_config) is
    never triggered for callers who never opt into this path.
    """
    key = os.getenv("AZURE_SPEECH_KEY", "")
    if not key:
        raise ValueError("AZURE_SPEECH_KEY is missing")
    if not settings.azure_speech_region:
        raise ValueError("AZURE_SPEECH_REGION is missing")
    return AzureSpeechSDKLiveBoundary(
        region=settings.azure_speech_region,
        subscription_key=key,
        sample_rate=settings.azure_tts_sample_rate,
        channels=settings.azure_tts_channels,
        sample_width=settings.azure_tts_sample_width,
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


class _PhaseDurationTracker:
    """Measure independently observed phases against one wall-clock total."""

    _TERMINAL_STATUSES = {
        "completed",
        "failed",
        "review_required",
        "reused",
    }

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._started_at = clock()
        self._active: dict[str, float] = {}
        self._durations: dict[str, float] = {}
        self._order: list[str] = []

    def observe(self, event: Mapping[str, Any]) -> None:
        stage = str(event.get("stage") or "").strip()
        if not stage:
            return
        status = str(event.get("status") or "running").strip().lower()
        now = self._clock()
        if stage not in self._order:
            self._order.append(stage)

        if status == "started":
            previous_start = self._active.get(stage)
            if previous_start is not None:
                self._durations[stage] = self._durations.get(stage, 0.0) + max(
                    0.0,
                    now - previous_start,
                )
            self._active[stage] = now
        elif stage not in self._active and stage not in self._durations:
            # Some reusable stages report only one completed event. Preserve
            # them in the summary even though no measurable interval exists.
            self._active[stage] = now

        if status in self._TERMINAL_STATUSES:
            stage_started_at = self._active.pop(stage, None)
            if stage_started_at is not None:
                self._durations[stage] = self._durations.get(stage, 0.0) + max(
                    0.0,
                    now - stage_started_at,
                )

    def finish(self) -> dict[str, Any]:
        finished_at = self._clock()
        for stage, stage_started_at in list(self._active.items()):
            self._durations[stage] = self._durations.get(stage, 0.0) + max(
                0.0,
                finished_at - stage_started_at,
            )
        self._active.clear()
        total_seconds = max(0.0, finished_at - self._started_at)
        phases = [
            {
                "stage": stage,
                "duration_seconds": self._durations.get(stage, 0.0),
                "share_percent": (
                    (self._durations.get(stage, 0.0) / total_seconds) * 100.0
                    if total_seconds > 0
                    else 0.0
                ),
            }
            for stage in self._order
        ]
        return {
            "total_seconds": total_seconds,
            "phases": phases,
        }


def _print_direct_dub_phase_summary(summary: Mapping[str, Any]) -> None:
    """Print a final timing report without contaminating JSON stdout."""

    phases = summary.get("phases")
    if not isinstance(phases, list):
        phases = []
    print("[dub-azure summary] Phase durations", file=sys.stderr)
    for phase in phases:
        if not isinstance(phase, Mapping):
            continue
        stage = str(phase.get("stage") or "working").replace("_", "-")
        if stage == "publication-render":
            stage = "publication-render (aggregate)"
        duration = _format_progress_time(phase.get("duration_seconds"))
        share = phase.get("share_percent")
        share_text = (
            f"{float(share):5.1f}%"
            if isinstance(share, (int, float))
            else "    --"
        )
        print(
            f"  {stage:<34} {duration:>8}  {share_text}",
            file=sys.stderr,
        )
    print(
        f"  {'TOTAL WALL TIME':<34} "
        f"{_format_progress_time(summary.get('total_seconds')):>8}  100.0%",
        file=sys.stderr,
    )
    print(
        "  Note: aggregate and nested phases overlap; TOTAL WALL TIME is not "
        "the sum of phase rows.",
        file=sys.stderr,
        flush=True,
    )


def _publication_ai_progress_payload(
    event: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Convert content-free provider telemetry into operator progress."""

    kind = str(event.get("event") or "")
    status = "running"
    if kind == "request_metrics":
        request_bytes = int(event.get("request_body_bytes") or 0)
        payload_bytes = int(event.get("payload_bytes") or 0)
        schema_bytes = int(event.get("output_schema_bytes") or 0)
        estimated_tokens = int(event.get("estimated_input_tokens") or 0)
        message = (
            f"Editorial AI request: {request_bytes / 1024:.1f} KiB JSON "
            f"(~{estimated_tokens:,} input tokens estimated); "
            f"payload={payload_bytes / 1024:.1f} KiB, "
            f"schema={schema_bytes / 1024:.1f} KiB, "
            f"max_output_tokens={event.get('max_output_tokens')}"
        )
    elif kind == "response_usage":
        response_bytes = int(event.get("response_body_bytes") or 0)
        message = (
            "Editorial AI usage: "
            f"input={int(event.get('input_tokens') or 0):,}, "
            f"output={int(event.get('output_tokens') or 0):,}, "
            f"thinking={int(event.get('thinking_tokens') or 0):,}, "
            f"cache-read={int(event.get('cache_read_input_tokens') or 0):,}, "
            f"cache-write={int(event.get('cache_creation_input_tokens') or 0):,}; "
            f"response={response_bytes / 1024:.1f} KiB"
        )
        status = "completed"
    elif kind == "request_attempt":
        message = (
            "Editorial AI request attempt "
            f"{event.get('attempt')}/{event.get('max_retries')}"
        )
    elif kind == "http_response":
        status_code = int(event.get("status_code") or 0)
        message = f"Editorial AI returned HTTP {status_code}"
        if not 200 <= status_code < 300:
            status = "warning"
    elif kind == "stream_opened":
        message = "Editorial AI SSE stream opened"
    elif kind == "stream_progress":
        message = (
            "Editorial AI response: "
            f"{int(event.get('text_characters') or 0):,} characters, "
            f"{int(event.get('output_tokens') or 0):,} output tokens"
        )
    elif kind == "stream_completed":
        message = (
            "Editorial AI SSE response completed: "
            f"{int(event.get('text_characters') or 0):,} characters"
        )
        status = "completed"
    elif kind == "retry_backoff":
        message = (
            "Editorial AI retrying after "
            f"{event.get('delay_seconds')}s ({event.get('reason')})"
        )
        status = "warning"
    elif kind == "structured_output_normalized":
        count = int(event.get("dropped_property_count") or 0)
        application_normalizer_applied = bool(
            event.get("application_normalizer_applied")
        )
        paths = [
            str(value)
            for value in (event.get("dropped_property_paths") or [])
        ]
        path_text = ", ".join(paths[:3])
        if len(paths) > 3:
            path_text += f", +{len(paths) - 3} more"
        actions: list[str] = []
        if count:
            actions.append(
                f"removed {count} schema-forbidden "
                f"propert{'y' if count == 1 else 'ies'}"
                + (f": {path_text}" if path_text else "")
            )
        if application_normalizer_applied:
            actions.append("restored safely derivable omitted metadata")
        message = "Editorial AI " + " and ".join(actions)
        message += "; strict local validation continues"
        status = "warning"
    else:
        # Do not print every low-level SSE delta.
        return None
    return {
        "stage": "publication_ai",
        "status": status,
        "message": message,
        "provider_event": kind,
        **{key: value for key, value in event.items() if key != "event"},
    }


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


def _native_dub_rush_rows(groups: Any) -> list[dict[str, Any]]:
    """Extract each rendered phrase group's real, applied rush percentage.

    ``speed_fit.speed_percent`` is the actual pitch-preserving atempo speed-up
    _run_batch_adaptive_timing_controller applied at render time to hit the
    real source mouth-close window -- 0.0 when a group needed no correction
    (fit naturally or came in short). This is the real, final number, not the
    candidate-pool's earlier translation-time estimate, which can differ once
    turn rebalancing/compaction and the actual synthesized audio are in.
    """

    rows: list[dict[str, Any]] = []
    for group in groups or []:
        speed_fit = group.get("speed_fit") or {}
        rows.append({
            "group_id": group.get("group_id"),
            "speed_percent": float(speed_fit.get("speed_percent") or 0.0),
            "speed_fit_mode": speed_fit.get("speed_fit_mode") or "none",
            "mouth_close_sync_ok": bool(group.get("mouth_close_sync_ok", True)),
        })
    return rows


def _print_native_dub_rush_summary(groups: Any) -> None:
    """Print every rendered phrase group's real applied rush percentage.

    Ordered worst-first so the sentences that actually needed compression are
    immediately visible; groups that fit naturally (0.0%) still get listed so
    the summary genuinely covers "all turns," not just the flagged ones.
    """

    rows = _native_dub_rush_rows(groups)
    if not rows:
        return
    rows.sort(key=lambda row: -row["speed_percent"])
    print(f"Rush by turn ({len(rows)} total, worst first):")
    for row in rows:
        flag = "" if row["mouth_close_sync_ok"] else "  [OUT OF SYNC]"
        print(
            f"  {row['group_id']:<22} {row['speed_percent']:>6.1f}%  ({row['speed_fit_mode']}){flag}"
        )


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
    post_tts_qa = (
        result.get("post_tts_qa")
        if isinstance(result.get("post_tts_qa"), dict)
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
    if post_tts_qa:
        qa_state = str(post_tts_qa.get("state") or "not reported").strip()
        if qa_state.casefold() == "disabled":
            qa_state = "disabled"
        elif post_tts_qa.get("publication_authorized") is True:
            qa_state = "passed"
        elif post_tts_qa.get("publication_authorized") is False:
            qa_state = "failed"
        qa_parts = [qa_state]
        score = post_tts_qa.get("best_production_score")
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            qa_parts.append(f"score {float(score):.2f}/100")
        wer = post_tts_qa.get("aggregate_word_error_rate")
        if isinstance(wer, (int, float)) and not isinstance(wer, bool):
            qa_parts.append(f"WER {float(wer):.3f}")
        blockers = post_tts_qa.get("production_blocker_count")
        if isinstance(blockers, int) and not isinstance(blockers, bool):
            qa_parts.append(f"{blockers} blocker(s)")
        print(f"Post-TTS QA: {' · '.join(qa_parts)}")


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

    job_commands = {}
    for name in ("process", "transcribe", "diarize", "retry", "validate", "inspect"):
        job_commands[name] = _add_job_command(commands, name)
    job_commands["process"].add_argument(
        "--ai-provider",
        choices=("azure-openai-gpt", "manual-chatgpt"),
        default=None,
        help=(
            "AI provider for analysis/autocorrect research. manual-chatgpt "
            "exports resumable request JSON instead of calling Azure AI."
        ),
    )
    resume_autocorrect = _add_job_command(commands, "resume-autocorrect-research")
    resume_autocorrect.add_argument(
        "--ai-provider",
        choices=("azure-openai-gpt", "manual-chatgpt"),
        default=None,
        help=(
            "AI provider for resumed entity inventory/web research. "
            "manual-chatgpt exports resumable request JSON."
        ),
    )
    _add_job_command(commands, "repair-speaker-turns")
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
    native_web = _add_job_command(commands, "native-web-override")
    native_web.add_argument("override_json", type=Path)

    native_translate = _add_job_command(commands, "native-translate")
    native_translate.add_argument("--model", default=None, help="Foundry Grok deployment name")
    native_translate.add_argument(
        "--pass",
        dest="native_pass",
        choices=("all", "1", "2"),
        default="all",
        help="Run both native AI passes or only one pass",
    )
    native_translate.add_argument(
        "--rolling-syllable",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use causal rolling translation: two locked prior isiZulu segments, one source "
            "lookahead, and duration-derived isiZulu syllable budgets before Azure TTS."
        ),
    )
    native_translate.add_argument(
        "--temporal-mask",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Compose fixed four-block English/Zulu semantic windows, target slightly-long isiZulu by syllables, "
            "Azure-measure at rate 0, and treat >+12%% final acceleration as warning-only."
        ),
    )
    native_translate.add_argument(
        "--temporal-mask-batch-size",
        type=int,
        default=4,
        help="Compatibility hint; temporal-mask composition now uses fixed four-block semantic windows (default 4).",
    )
    native_translate.add_argument(
        "--temporal-mask-candidates",
        type=int,
        default=5,
        help=(
            "Ceiling on the natural-to-concise compaction ladder per window (1-8): Grok always "
            "starts from candidate 'a' (natural, no length target) and compacts further only as "
            "needed, stopping itself once a further candidate would repeat or damage one -- this is "
            "a maximum, not a count it must fill. Every candidate returned is measured against real "
            "Azure synthesis to pick the best fit (default 5)."
        ),
    )
    native_translate.add_argument(
        "--temporal-mask-residual-rounds",
        type=int,
        default=0,
        help="Compatibility option; Python temporal-mask residual rounds are disabled (always 0).",
    )
    native_translate.add_argument(
        "--temporal-mask-tts-workers",
        type=int,
        default=6,
        help="Compatibility option; Azure TTS is not called by temporal-mask Pass 2 (timing is deferred to native-dub).",
    )

    native_translate.add_argument(
        "--rolling-syllable-batch-size",
        type=int,
        default=8,
        help=(
            "Maximum sequential segments per Grok rolling-syllable request (default 8). "
            "The model processes them in order inside one call."
        ),
    )
    native_translate.add_argument(
        "--rolling-syllables-per-second",
        type=float,
        default=5.3,
        help="Azure isiZulu content-rate calibration used for syllable planning (default 5.3 syllables/s).",
    )
    native_translate.add_argument(
        "--rolling-azure-overhead-ms",
        type=int,
        default=900,
        help="Fixed per-utterance Azure TTS duration overhead used by the affine syllable model (default 900 ms).",
    )
    native_translate.add_argument(
        "--rolling-syllable-tolerance-percent",
        type=int,
        default=12,
        help="Preferred +/- syllable planning range around the duration-derived target (default 12%%).",
    )
    native_translate.add_argument(
        "--audit-referents",
        dest="audit_referents",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run an independent post-Pass-2 audit call to verify every name, role/title, "
            "number, and negation survives translation, and repair confirmed issues "
            "(default: enabled). The inline Pass-2 self-QA runs in the same call as the "
            "translation and cannot reliably catch a referent it is already confident about."
        ),
    )
    native_translate.add_argument(
        "--referent-audit-batch-size",
        type=int,
        default=12,
        help="Groups per referent-audit/repair provider call (default 12).",
    )
    native_translate.add_argument(
        "--max-referent-repair-rounds",
        type=int,
        default=1,
        help="Bounded repair rounds for groups the referent audit flags (default 1).",
    )
    native_translate.add_argument(
        "--referent-audit-workers",
        type=int,
        default=4,
        help=(
            "Concurrent audit/repair provider calls (default 4). Unlike Pass 2's "
            "temporal-mask windows, every referent-audit batch is fully independent "
            "of every other, so this is safe to raise for a faster audit -- bounded "
            "only to stay a good citizen of the same rate-limited Foundry endpoint "
            "Pass 2 also calls."
        ),
    )
    native_translate.add_argument(
        "--candidate-pool",
        action="store_true",
        help=(
            "Build the full candidate-pool translation artifact (glossary + Grok "
            "translation + QA check/repair + Azure TTS measurement + discourse-segment "
            "timing rebalance) and write it to pass2_candidate_pool.json ONLY -- never "
            "touches pass2_natural_translation.json. Requires pass 1 to already be "
            "complete. Mutually exclusive with --rolling-syllable/--temporal-mask."
        ),
    )
    native_translate.add_argument(
        # Mirrors native_dub.DEFAULT_CANDIDATE_POOL_WORKERS -- kept a literal since
        # this module lazily imports native_dub inside command dispatch, not at
        # parser-construction time.
        "--candidate-pool-workers", type=int, default=4,
        help="Concurrent Grok/QA provider calls for --candidate-pool (default 4).",
    )
    native_translate.add_argument(
        "--candidate-pool-tts-workers", type=int, default=6,
        help="Concurrent Azure TTS measurement calls for --candidate-pool (default 6).",
    )
    native_translate.add_argument(
        "--commit",
        action="store_true",
        help=(
            "After building (or reusing) the candidate pool, commit its selected winners "
            "into pass2_natural_translation.json -- the SAME temporal_mask_groups shape "
            "the --temporal-mask path writes, so native-qa/native-dub work unchanged. "
            "Only used with --candidate-pool; refuses to commit an incomplete pool."
        ),
    )
    native_translate.add_argument(
        "--skip-turn-rebalance",
        action="store_true",
        help=(
            "Diagnostic only (--candidate-pool): skip Phase 13's speaker-turn window "
            "reallocation entirely, so every sentence's required_speed_percent reflects its own "
            "raw natural-translation timing. Isolates the concise/compact compaction mechanism "
            "for independent tuning."
        ),
    )
    native_translate.add_argument(
        "--skip-turn-block-translation",
        action="store_true",
        help=(
            "Diagnostic only (--candidate-pool): skip Phase 16's whole-turn-block translation "
            "entirely and fall back to the older, proven per-sentence translation for EVERY "
            "sentence -- real A/B comparison against a job's own baseline."
        ),
    )
    native_translate.add_argument(
        "--skip-pronunciation-research",
        action="store_true",
        default=os.getenv("MATHULA_TV_SKIP_PRONUNCIATION_RESEARCH", "0") == "1",
        help=(
            "Diagnostic only (--candidate-pool): skip Phase 17's web-search-backed pronunciation "
            "research entirely (real Azure Responses API call, real cost) -- entities keep "
            "whatever pronunciation the hand-curated dictionaries already give them. Defaults on "
            "when MATHULA_TV_SKIP_PRONUNCIATION_RESEARCH=1 is set (e.g. web research is on a "
            "separate, currently-exhausted deployment/quota from the main translation model) -- "
            "a deterministic, zero-cost disable rather than waiting for every call to fail and "
            "degrade on its own."
        ),
    )
    native_translate.add_argument(
        "--skip-pronunciation-round-trip",
        action="store_true",
        help=(
            "Diagnostic only (--candidate-pool): skip Phase 18's automated TTS+STT round-trip "
            "verification of pronunciation research corrections (real Azure TTS/STT calls, real "
            "cost) even when Speech credentials are configured -- corrections are applied on the "
            "research call's own confidence alone, exactly like before Phase 18 existed."
        ),
    )
    native_translate.add_argument(
        "--skip-speaker-seriousness-mode",
        action="store_true",
        help=(
            "Diagnostic only (--candidate-pool): skip the per-speaker seriousness-mode "
            "classification (real cost) -- every speaker defaults to 'cross_examination', the "
            "strictest tier, so the parallel-list-item compaction exception never applies to "
            "anyone."
        ),
    )
    native_translate.add_argument(
        "--json",
        "--json-output",
        dest="json_output",
        action="store_true",
        help="Print the complete native translation artifact instead of the concise summary",
    )

    native_qa = _add_job_command(commands, "native-qa")
    native_qa.add_argument("--model", default=None, help="Foundry Grok deployment used for back-translation QA")
    native_qa.add_argument(
        "--batch-size",
        type=int,
        default=15,
        help="Sentences per back-translation QA batch (default 15)",
    )
    native_qa.add_argument(
        "--repair",
        dest="repair",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Automatically repair sentences the back-translation audit flags, then re-verify "
            "the fix and mutate the canonical Pass-2 translation in place (default: enabled). "
            "--no-repair restores the old flag-only behavior."
        ),
    )
    native_qa.add_argument(
        "--max-qa-repair-rounds",
        type=int,
        default=1,
        help="Bounded repair rounds for sentences the back-translation audit flags (default 1).",
    )
    native_qa.add_argument(
        "--direct-adequacy",
        dest="direct_adequacy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Also judge each sentence directly (source vs. target, no back-translation "
            "step) as a genuinely independent second check -- catches drift a same-model "
            "back-translation round-trip can self-consistently miss, at the cost of "
            "roughly doubling this command's Grok calls (default: enabled). "
            "--no-direct-adequacy restores the back-translation-only behavior."
        ),
    )
    native_qa.add_argument(
        "--qa-audit-workers",
        type=int,
        default=4,
        help=(
            "Parallel workers for the back-translation/direct-adequacy audit batches "
            "(default 4). Batches are fully independent -- each is its own provider "
            "call over a distinct slice of sentences -- so this is a pure throughput knob."
        ),
    )
    native_qa.add_argument(
        "--json",
        "--json-output",
        dest="json_output",
        action="store_true",
        help="Print the complete back-translation QA artifact instead of the concise summary",
    )

    native_dub = _add_job_command(commands, "native-dub")
    native_dub.add_argument("--model", default=None, help="Foundry Grok deployment used for native editorial hook and optional timing repair")
    native_dub.add_argument("--voice-map", type=Path)
    native_dub.add_argument(
        "--hook-edit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use the production AI-selected opening cut. --no-hook-edit keeps "
            "the full native dub but still renders the persistent footer hook overlay."
        ),
    )
    native_dub.add_argument(
        "--hook-overlay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Render the footer hook title-card overlay and run its expensive publication "
            "encode (a full libx264 re-encode of the whole video, unrelated to --hook-edit's "
            "opening cut). --no-hook-overlay skips it entirely, producing just the plain "
            "dubbed master -- untouched source video, dubbed audio, no title cards, no "
            "editorial/Grok hook call -- for fast, cheap iteration on audio timing alone. "
            "Combine with --lite to also cap CPU during iteration."
        ),
    )
    native_dub.add_argument("--repair-threshold-ms", type=int, default=0, help="Deprecated hard-sync compatibility option; native hard sync repairs any true overflow")
    native_dub.add_argument("--max-join-gap-ms", type=int, default=0, help="Same-speaker grouping gap. Hard-sync native mode never consumes a positive source gap by default")
    native_dub.add_argument("--max-phrase-source-ms", type=int, default=45000)
    native_dub.add_argument(
        "--preferred-raw-speed-percent",
        type=int,
        default=6,
        help=(
            "Soft Pass-2/raw-TTS target: generate speech this much longer than the source "
            "mouth window so final sync uses a modest speed-up (0-12; default 6)"
        ),
    )
    native_dub.add_argument(
        "--max-natural-speed-percent",
        type=int,
        # Must be kept in sync manually with native_dub.DEFAULT_MAX_NATURAL_
        # SPEED_PERCENT -- native_dub is deliberately lazy-imported throughout
        # this file (see the per-command imports below) rather than at module
        # top level, so this default can't just reference that constant
        # directly without undoing that. Real gap found 2026-09-08: this
        # literal silently drifted from the constant (12 vs the constant's
        # since-lowered 10) and a real render kept warning at the stale 12%
        # bar even after the shared constant changed everywhere else.
        default=10,
        help=(
            "Red console quality-warning threshold for pitch-preserving post-TTS speed-up. "
            "Higher required speed is allowed and non-fatal (0-15; default 10)."
        ),
    )
    native_dub.add_argument(
        "--mouth-close-lag-tolerance-ms",
        type=int,
        default=40,
        help=(
            "Maximum final audio-vs-mouth-close error in milliseconds before wording recast "
            "is required (0-80; default 40)"
        ),
    )
    native_dub.add_argument(
        "--protected-gap-overflow",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Allow bounded use of existing post-articulation source silence when exact mouth-close "
            "would require excessive speed; never moves the next source onset"
        ),
    )
    native_dub.add_argument(
        "--max-protected-gap-overflow-ms",
        type=int,
        default=400,
        help="Maximum post-mouth-close source gap Mathula may occupy (0-1000 ms; default 400)",
    )
    native_dub.add_argument(
        "--min-protected-pause-ms",
        type=int,
        default=120,
        help="Minimum existing pause preserved before the next source onset (0-2000 ms; default 120)",
    )
    native_dub.add_argument(
        "--lite",
        action="store_true",
        help=(
            "Cap ffmpeg's own encode threads at half the machine's logical CPU count and "
            "lower the ffmpeg subprocess's OS scheduling priority, so a background render "
            "doesn't peg every core on your machine. An approximation, not a literal "
            "enforced percentage ceiling -- see render_native_dub's docstring."
        ),
    )
    native_dub.add_argument(
        "--json",
        "--json-output",
        dest="json_output",
        action="store_true",
        help="Print the complete native dub result instead of the concise summary",
    )

    translate = _add_job_command(commands, "translate")
    translate.add_argument("--provider", default=None)
    translate.add_argument("--model", default=None)
    translate.add_argument("--batch-size", type=int, default=None)
    translate.add_argument(
        "--parallelism",
        type=int,
        choices=range(1, 5),
        default=None,
        help=(
            "Concurrent translation chunks after the first validated continuity "
            "seed (default: 2; use 1 for fully serial translation)"
        ),
    )
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
    manual_ai = _add_job_command(commands, "manual-ai")
    manual_ai.add_argument("action", choices=("pending", "install"))
    manual_ai.add_argument("response_json", nargs="?", type=Path)
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
        "--ai-provider",
        choices=("azure-openai-gpt", "manual-chatgpt"),
        default=None,
        help=(
            "AI provider for grammar/timing/publication calls. manual-chatgpt "
            "exports resumable request JSON instead of calling Azure OpenAI."
        ),
    )
    dub_azure.add_argument("--timeline-panel-host", default="127.0.0.1")
    dub_azure.add_argument("--timeline-panel-port", type=int, default=8765)
    dub_azure.add_argument(
        "--timing-mode",
        choices=("balanced", "semantic-fit", "performance-plan"),
        default="balanced",
        help=(
            "balanced keeps prepared variants and bounded tempo repair; "
            "semantic-fit locks speaker tempo and iteratively searches for an "
            "Azure-measured, semantically reviewed delivery; performance-plan "
            "uses Azure word-timed speech islands as the timing authority and "
            "treats STT blocks only as audit members"
        ),
    )
    dub_azure.add_argument(
        "--semantic-fit-max-attempts",
        type=int,
        default=12,
        help=(
            "Maximum measured AI candidates per block in semantic-fit mode; "
            "performance-plan uses at most four one-candidate measured "
            "feedback rounds"
        ),
    )
    dub_azure.add_argument(
        "--semantic-fit-tolerance-ms",
        type=int,
        default=250,
        help="Allowed duration shortfall around the fixed semantic-fit target",
    )
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
        "--qa-stt",
        action=argparse.BooleanOptionalAction,
        default=os.getenv(
            "MATHULA_TV_POST_TTS_QA_ENABLED", "1"
        ).strip().casefold()
        not in {"0", "false", "no", "off"},
        help=(
            "Transcribe and score the rendered dub before publication; "
            "publication fails closed when production thresholds are not met"
        ),
    )
    dub_azure.add_argument(
        "--qa-stt-max-rounds",
        type=int,
        default=int(os.getenv("MATHULA_TV_POST_TTS_QA_MAX_ROUNDS", "1")),
        help="Maximum approved-variant repair rounds after the initial STT audit",
    )
    dub_azure.add_argument(
        "--qa-minimum-production-score",
        type=float,
        default=float(
            os.getenv(
                "MATHULA_TV_POST_TTS_QA_MINIMUM_PRODUCTION_SCORE",
                "82",
            )
        ),
        help="Minimum post-TTS production quality score",
    )
    dub_azure.add_argument(
        "--qa-maximum-aggregate-wer",
        type=float,
        default=float(
            os.getenv(
                "MATHULA_TV_POST_TTS_QA_MAXIMUM_AGGREGATE_WER",
                "0.40",
            )
        ),
        help="Maximum delivery-aware aggregate word error rate",
    )
    dub_azure.add_argument(
        "--json",
        "--json-output",
        dest="json_output",
        action="store_true",
        help="Print the complete dubbing and publication report as JSON",
    )

    optimize_dub = _add_job_command(commands, "optimize-dub")
    optimize_dub.add_argument("--max-rounds", type=int, default=2)
    optimize_dub.add_argument("--audit-only", action="store_true")
    optimize_dub.add_argument("--min-confidence", type=float, default=0.70)
    optimize_dub.add_argument("--voice-map", type=Path)
    optimize_dub.add_argument(
        "--ai-provider",
        choices=("azure-openai-gpt", "manual-chatgpt"),
        default=None,
        help="AI provider for grammar/timing/mastering calls.",
    )
    optimize_dub.add_argument("--minimum-production-score", type=float, default=82.0)
    optimize_dub.add_argument("--maximum-aggregate-wer", type=float, default=0.40)
    optimize_dub.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    optimize_dub.add_argument(
        "--json",
        "--json-output",
        dest="json_output",
        action="store_true",
    )


    collision_panel = commands.add_parser("timeline-collision-panel")
    collision_panel.add_argument("legacy_job_id", nargs="?")
    collision_panel.add_argument("--job-id")
    collision_panel.add_argument("--force", action="store_true")
    collision_panel.add_argument("--dry-run", action="store_true")
    collision_panel.add_argument("--live-operation", action="store_true")
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

    entities_audit = entities_subcommands.add_parser("audit-pronunciations")
    entities_audit.add_argument("--registry", default="config/entity_registry.json")
    entities_audit.add_argument(
        "--output-dir",
        type=Path,
        default=Path("working/pronunciation_audits/current"),
    )
    entities_audit.add_argument("--force", action="store_true")
    entities_audit.add_argument("--live-operation", action="store_true")
    entities_audit.add_argument("--rate-percent", type=int, default=5)
    entities_audit.add_argument("--atempo-ratio", type=float, default=1.03)
    
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


def _reset_for_autocorrect_research(job: Any) -> str:
    """Return a job to analysis without repeating upload or transcription."""

    previous_state = str(job.state)
    job.state = "analysis_running"
    job.completed_stages = [
        stage
        for stage in job.completed_stages
        if stage not in _POST_AUTOCORRECT_STAGES
    ]
    job.review_readiness = "autocorrection_research_rerun"
    job.last_error = None
    job.attempt_history.append(
        {
            "stage": "autocorrect_name_research_rerun",
            "status": "started",
            "from_state": previous_state,
            "to_state": "analysis_running",
            "started_at": utcnow(),
        }
    )
    return previous_state


def _reset_for_contextual_speaker_repair(job: Any) -> str:
    """Rebuild analysis from saved Azure words without retranscription."""

    previous_state = str(job.state)
    job.state = "analysis_running"
    job.completed_stages = [
        stage
        for stage in job.completed_stages
        if stage not in _POST_AUTOCORRECT_STAGES
    ]
    job.review_readiness = "contextual_speaker_turn_repair"
    job.last_error = None
    job.attempt_history.append(
        {
            "stage": "contextual_speaker_turn_repair",
            "status": "started",
            "from_state": previous_state,
            "to_state": "analysis_running",
            "started_at": utcnow(),
        }
    )
    return previous_state


def main(argv: list[str] | None = None) -> int:
    # load_settings() loads .env into os.environ (via load_dotenv) as a side
    # effect. Must run BEFORE parser() is built: real bug found 2026-09-10 --
    # an argparse default reading os.getenv(...) directly (e.g.
    # --skip-pronunciation-research's MATHULA_TV_SKIP_PRONUNCIATION_RESEARCH
    # toggle) silently always saw the pre-dotenv environment when parser() was
    # constructed first, so a value set only in .env (not actually exported in
    # the shell) never took effect.
    settings = load_settings()
    args = parser().parse_args(argv)
    if args.command == "viral-clips":
        print(json.dumps(run_viral_moments_command(args, settings), ensure_ascii=False, indent=2))
        return 0

    configure_logging(settings.work_dir / "logs/mathula-tv.jsonl", args.verbose)
    app = Orchestrator(settings)
    phase_timing: _PhaseDurationTracker | None = None
    phase_timing_reported = False
    try:
        if args.command == "submit":
            if is_youtube_url(args.video):
                maximum_minutes = float(getattr(settings, "max_source_duration_minutes", 240))
                print("[submit] Reading YouTube metadata...", file=sys.stderr, flush=True)
                with download_youtube_video(
                    args.video,
                    max_duration_seconds=maximum_minutes * 60.0,
                    cookies_file=settings.youtube_cookies_file or None,
                ) as download:
                    print(
                        f"[submit] Download complete: {download.path.name} "
                        f"({download.path.stat().st_size / (1024 * 1024):.1f} MiB)",
                        file=sys.stderr,
                        flush=True,
                    )
                    print(
                        "[submit] Preparing Mathula job and source audio...",
                        file=sys.stderr,
                        flush=True,
                    )
                    job = app.submit(download.path, args.target_language, args.force)
                    print(
                        f"[submit] Job prepared: {job.job_id}; saving YouTube provenance...",
                        file=sys.stderr,
                        flush=True,
                    )
                    record_youtube_submission(app, job, download)
            else:
                print("[submit] Preparing Mathula job and source audio...", file=sys.stderr, flush=True)
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
            elif args.entities_command == "audit-pronunciations":
                if not args.live_operation:
                    raise ValueError(
                        "entities audit-pronunciations calls Azure TTS/STT and "
                        "requires --live-operation"
                    )
                from .entity_pronunciation_audit import (
                    EntityPronunciationAuditOptions,
                    audit_entity_pronunciations,
                )
                from .direct_azure_dub import create_direct_azure_backend

                def pronunciation_progress(event: dict[str, Any]) -> None:
                    state = "PASS" if event["passed"] else "FAIL"
                    print(
                        f"[entity-pronunciation] {event['current']}/{event['total']} "
                        f"{state} {event['entity_id']} {event['voice']}: "
                        f"{str(event['recognized_text'])[:180]}",
                        file=sys.stderr,
                        flush=True,
                    )

                report = audit_entity_pronunciations(
                    registry_path=Path(args.registry),
                    output_dir=args.output_dir,
                    tts_backend=create_direct_azure_backend(
                        settings,
                        max_rate_percent=max(5, args.rate_percent),
                    ),
                    stt_backend=create_speech_backend(settings),
                    options=EntityPronunciationAuditOptions(
                        voices=tuple(settings.azure_tts_voices),
                        rate_percent=args.rate_percent,
                        atempo_ratio=args.atempo_ratio,
                        force=args.force,
                    ),
                    progress=pronunciation_progress,
                )
                print(
                    json.dumps(
                        {
                            key: value
                            for key, value in report.items()
                            if key != "observations"
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
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

        if args.command == "timeline-collision-panel":
            from .timeline_collision_panel import (
                clear_timeline_collision_resolution,
                save_timeline_collision_resolution,
                serve_universal_timeline_collision_panel,
                timeline_collision_job_paths,
                universal_timeline_collision_summary,
            )

            target_job_id = args.job_id or args.legacy_job_id
            if args.clear or args.resolve:
                if not target_job_id:
                    raise ValueError(
                        "--job-id is required only for terminal resolution or clear actions"
                    )
                paths = timeline_collision_job_paths(app.jobs.root, target_job_id)
                if not paths["review"].is_file():
                    raise ValueError(
                        f"No timeline collision review exists for job {target_job_id}"
                    )
                if args.clear:
                    clear_timeline_collision_resolution(
                        resolutions_path=paths["resolutions"],
                        review_path=paths["review"],
                        panel_path=paths["panel"],
                        collision_id=args.clear,
                    )
                if args.resolve:
                    if not args.strategy:
                        raise ValueError("--strategy is required with --resolve")
                    text_value = args.text
                    if args.text_file:
                        text_value = args.text_file.read_text(encoding="utf-8")
                    save_timeline_collision_resolution(
                        resolutions_path=paths["resolutions"],
                        review_path=paths["review"],
                        panel_path=paths["panel"],
                        job_id=target_job_id,
                        collision_id=args.resolve,
                        strategy=args.strategy,
                        reviewed_by=args.reviewed_by or "",
                        overlap_before_ms=args.overlap_before_ms,
                        overlap_after_ms=args.overlap_after_ms,
                        manual_text=text_value,
                        previous_variant_id=args.previous_variant,
                        current_variant_id=args.current_variant,
                        following_variant_id=args.following_variant,
                    )
            if args.serve:
                serve_universal_timeline_collision_panel(
                    jobs_root=app.jobs.root,
                    host=args.host,
                    port=args.port,
                )
            else:
                print(universal_timeline_collision_summary(app.jobs.root))
                print(
                    "Serve the universal panel with: "
                    "python -m mathula_tv.cli timeline-collision-panel --serve"
                )
            return 0

        job = app.jobs.load(args.job_id)

        # A CLI provider selector takes precedence over a stale shell value.
        # When manual mode is active, keep every request for this invocation
        # inside the current job so translation, grammar, timing and publication
        # calls can all resume from the same deterministic handoff directory.
        from .manual_ai import is_manual_ai_provider, manual_ai_root_for_job

        explicit_ai_provider = (
            getattr(args, "provider", None)
            or getattr(args, "ai_provider", None)
        )
        if explicit_ai_provider is not None:
            if is_manual_ai_provider(explicit_ai_provider):
                os.environ["MATHULA_TV_AI_PROVIDER"] = "manual-chatgpt"
            elif str(explicit_ai_provider).strip().casefold() in {
                "azure-openai-gpt", "azure-openai", "gpt"
            }:
                os.environ["MATHULA_TV_AI_PROVIDER"] = "azure-openai-gpt"
        if is_manual_ai_provider(os.getenv("MATHULA_TV_AI_PROVIDER")):
            os.environ.setdefault(
                "MATHULA_TV_MANUAL_AI_DIR",
                str(manual_ai_root_for_job(settings.work_dir, job.job_id)),
            )

        _require_operation_authorization(args, settings)
        dry = _dry_run(args, job)
        if dry is not None:
            print(json.dumps(dry, indent=2))
            return 0

        if args.command == "native-web-override":
            from .native_dub import install_manual_web_overrides

            result = install_manual_web_overrides(
                app.jobs.job_dir(job.job_id),
                args.override_json.resolve(),
            )
            job.media["native_manual_web_overrides"] = str(
                app.jobs.job_dir(job.job_id) / "native_dub" / "manual_web_overrides.json"
            )
            app.jobs.save(job)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0

        if args.command == "native-translate":
            from .native_dub import run_native_pass1, run_native_pass2, run_native_translation

            if bool(args.candidate_pool):
                from .native_dub import build_candidate_pool, commit_candidate_pool

                if args.rolling_syllable or args.temporal_mask:
                    raise ValueError(
                        "--candidate-pool cannot be combined with --rolling-syllable or --temporal-mask"
                    )
                if args.native_pass == "1":
                    raise ValueError(
                        "--candidate-pool requires pass 1 to already be complete; omit --pass 1"
                    )
                if not 1 <= int(args.candidate_pool_workers) <= 12:
                    raise ValueError("--candidate-pool-workers must be between 1 and 12")
                if not 1 <= int(args.candidate_pool_tts_workers) <= 12:
                    raise ValueError("--candidate-pool-tts-workers must be between 1 and 12")
                progress = _print_native_progress if args.live_operation else None
                provider = create_native_grok_backend(args.model, progress=progress)
                job_root = app.jobs.job_dir(job.job_id)
                tts_backend = create_tts_backend(settings)
                pronunciation_stt_backend = None
                if not bool(args.skip_pronunciation_round_trip):
                    try:
                        pronunciation_stt_backend = create_speech_backend(settings)
                    except Exception as exc:  # noqa: BLE001 - optional enhancement, never a hard requirement
                        _print_native_progress(
                            f"[native pronunciation] Speech credentials unavailable ({exc}); "
                            "skipping Phase 18's TTS+STT round-trip verification."
                        )
                result = build_candidate_pool(
                    job_root=job_root,
                    provider=provider,
                    tts=tts_backend,
                    stt_backend=pronunciation_stt_backend,
                    skip_pronunciation_round_trip=bool(args.skip_pronunciation_round_trip),
                    candidate_pool_workers=int(args.candidate_pool_workers),
                    candidate_pool_tts_workers=int(args.candidate_pool_tts_workers),
                    force=args.force,
                    skip_turn_rebalance=bool(args.skip_turn_rebalance),
                    skip_turn_block_translation=bool(args.skip_turn_block_translation),
                    skip_pronunciation_research=bool(args.skip_pronunciation_research),
                    skip_speaker_seriousness_mode=bool(args.skip_speaker_seriousness_mode),
                    progress=progress,
                )
                job.media["native_candidate_pool"] = str(
                    job_root / "native_dub" / "pass2_candidate_pool.json"
                )

                commit_result = None
                if bool(args.commit):
                    commit_result = commit_candidate_pool(
                        job_root=job_root, force=args.force, progress=progress,
                    )
                    job.media["native_translation"] = str(
                        job_root / "native_dub" / "pass2_natural_translation.json"
                    )
                    if "native_pass2" not in job.completed_stages:
                        job.completed_stages.append("native_pass2")

                app.jobs.save(job)
                if args.json_output:
                    print(json.dumps({"candidate_pool": result, "committed": commit_result}, ensure_ascii=False, indent=2))
                else:
                    records = result.get("candidate_pool") or []
                    selected = sum(1 for r in records if r.get("selected_candidate_id"))
                    review = sum(1 for r in records if r.get("requires_review"))
                    usage = result.get("usage") or {}
                    print(
                        f"Candidate pool complete: {selected}/{len(records)} sentence(s) have a "
                        f"selected winner, {review} flagged requires_review."
                    )
                    print(
                        f"Grok tokens: in={int(usage.get('input_tokens') or 0):,}, "
                        f"out={int(usage.get('output_tokens') or 0):,}."
                    )
                    print(f"Saved: {job_root / 'native_dub' / 'pass2_candidate_pool.json'}")
                    if commit_result is not None:
                        mask_count = int((commit_result.get("summary") or {}).get("mask_count") or 0)
                        print(
                            f"Committed {mask_count} sentence(s) to pass2_natural_translation.json -- "
                            "run native-dub to render."
                        )
                return 0

            if args.rolling_syllable and args.temporal_mask:
                raise ValueError("Choose either --rolling-syllable or --temporal-mask, not both")
            if not 1 <= int(args.temporal_mask_batch_size) <= 12:
                raise ValueError("--temporal-mask-batch-size must be between 1 and 12")
            if args.temporal_mask and not (1 <= int(args.temporal_mask_candidates) <= 8):
                raise ValueError(
                    "--temporal-mask-candidates must be between 1 and 8; it is a ceiling on the "
                    "natural-to-concise compaction ladder Grok may return per window, not an exact "
                    "count it must fill"
                )
            if args.temporal_mask and int(args.temporal_mask_residual_rounds) != 0:
                raise ValueError("--temporal-mask-residual-rounds must be 0; temporal-mask revisions now occur inside one Grok call per window")
            if not 1 <= int(args.temporal_mask_tts_workers) <= 12:
                raise ValueError("--temporal-mask-tts-workers must be between 1 and 12")
            if not 1 <= int(args.rolling_syllable_batch_size) <= 16:
                raise ValueError("--rolling-syllable-batch-size must be between 1 and 16")
            if not 2.0 <= float(args.rolling_syllables_per_second) <= 8.0:
                raise ValueError("--rolling-syllables-per-second must be between 2.0 and 8.0")
            if not 0 <= int(args.rolling_azure_overhead_ms) <= 2000:
                raise ValueError("--rolling-azure-overhead-ms must be between 0 and 2000")
            if not 5 <= int(args.rolling_syllable_tolerance_percent) <= 30:
                raise ValueError("--rolling-syllable-tolerance-percent must be between 5 and 30")
            if not 1 <= int(args.referent_audit_batch_size) <= 24:
                raise ValueError("--referent-audit-batch-size must be between 1 and 24")
            if not 0 <= int(args.max_referent_repair_rounds) <= 3:
                raise ValueError("--max-referent-repair-rounds must be between 0 and 3")
            if not 1 <= int(args.referent_audit_workers) <= 12:
                raise ValueError("--referent-audit-workers must be between 1 and 12")
            progress = _print_native_progress if args.live_operation else None
            provider = create_native_grok_backend(args.model, progress=progress)
            job_root = app.jobs.job_dir(job.job_id)
            temporal_tts = (
                create_tts_backend(settings)
                if bool(args.temporal_mask) and args.native_pass != "1"
                else None
            )
            if args.native_pass == "1":
                result = run_native_pass1(
                    job_root=job_root, provider=provider, force=args.force, progress=progress
                )
            elif args.native_pass == "2":
                result = run_native_pass2(
                    job_root=job_root,
                    provider=provider,
                    force=args.force,
                    progress=progress,
                    audit_referents=bool(args.audit_referents),
                    referent_audit_batch_size=int(args.referent_audit_batch_size),
                    max_referent_repair_rounds=int(args.max_referent_repair_rounds),
                    referent_audit_workers=int(args.referent_audit_workers),
                    rolling_syllable=bool(args.rolling_syllable),
                    rolling_syllable_batch_size=int(args.rolling_syllable_batch_size),
                    temporal_mask=bool(args.temporal_mask),
                    temporal_mask_batch_size=int(args.temporal_mask_batch_size),
                    temporal_mask_candidates=int(args.temporal_mask_candidates),
                    temporal_mask_residual_rounds=int(args.temporal_mask_residual_rounds),
                    temporal_mask_tts_workers=int(args.temporal_mask_tts_workers),
                    job=job,
                    tts=temporal_tts,
                    content_syllables_per_second=float(args.rolling_syllables_per_second),
                    azure_utterance_overhead_ms=int(args.rolling_azure_overhead_ms),
                    syllable_tolerance_percent=int(args.rolling_syllable_tolerance_percent),
                )
            else:
                result = run_native_translation(
                    job_root=job_root,
                    provider=provider,
                    force=args.force,
                    progress=progress,
                    audit_referents=bool(args.audit_referents),
                    referent_audit_batch_size=int(args.referent_audit_batch_size),
                    max_referent_repair_rounds=int(args.max_referent_repair_rounds),
                    referent_audit_workers=int(args.referent_audit_workers),
                    rolling_syllable=bool(args.rolling_syllable),
                    rolling_syllable_batch_size=int(args.rolling_syllable_batch_size),
                    temporal_mask=bool(args.temporal_mask),
                    temporal_mask_batch_size=int(args.temporal_mask_batch_size),
                    temporal_mask_candidates=int(args.temporal_mask_candidates),
                    temporal_mask_residual_rounds=int(args.temporal_mask_residual_rounds),
                    temporal_mask_tts_workers=int(args.temporal_mask_tts_workers),
                    job=job,
                    tts=temporal_tts,
                    content_syllables_per_second=float(args.rolling_syllables_per_second),
                    azure_utterance_overhead_ms=int(args.rolling_azure_overhead_ms),
                    syllable_tolerance_percent=int(args.rolling_syllable_tolerance_percent),
                )
            job.providers["native_dub_ai"] = "azure-foundry-grok"
            job.providers["native_dub_model"] = provider.config.deployment
            if bool(args.temporal_mask):
                job.providers["native_dub_tts"] = "azure-speech-rate-0-temporal-mask-measurement"
            job.media["native_dub_root"] = str(job_root / "native_dub")
            job.media["native_translation"] = str(job_root / "native_dub" / "pass2_natural_translation.json")
            if args.native_pass in ("1", "all") and "native_pass1" not in job.completed_stages:
                job.completed_stages.append("native_pass1")
            if args.native_pass in ("2", "all") and "native_pass2" not in job.completed_stages:
                job.completed_stages.append("native_pass2")
            app.jobs.save(job)
            if args.json_output:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                native_root = job_root / "native_dub"
                if args.native_pass == "1":
                    usage = result.get("usage") or {}
                    print(
                        f"Native Pass 1 complete: {len(result.get('segments') or [])} segment(s), "
                        f"{len(result.get('corrections') or [])} STT correction(s)."
                    )
                    print(
                        f"Grok: {result.get('model') or provider.config.deployment}; "
                        f"tokens in={int(usage.get('input_tokens') or 0):,}, "
                        f"out={int(usage.get('output_tokens') or 0):,}."
                    )
                    print(f"Saved: {native_root / 'pass1_restored_transcript.json'}")
                elif args.native_pass == "2":
                    usage = result.get("usage") or {}
                    if result.get("temporal_mask_mode"):
                        summary = result.get("summary") or {}
                        print(
                            f"Native Pass 2 complete: {int(summary.get('mask_count') or 0)} temporal mask(s) committed from "
                            f"{int(summary.get('source_segment_count') or 0)} source segment(s); "
                            f"residual calls={int(summary.get('residual_calls') or 0)}, "
                            f"speculative rollback masks={int(summary.get('rollback_mask_count') or 0)}."
                        )
                        undersized = int(summary.get("undersized_syllable_count") or 0)
                        oversized = int(summary.get("oversized_syllable_count") or 0)
                        if undersized or oversized:
                            print(
                                f"Syllable budget: {undersized} undersized (kept, normal speed), "
                                f"{oversized} oversized (kept, native-dub will rush and warn)."
                            )
                    elif result.get("rolling_syllable_mode"):
                        summary = result.get("syllable_summary") or {}
                        print(
                            f"Native Pass 2 complete: {len(result.get('translations') or [])} rolling-context isiZulu segment(s); "
                            f"syllable-fit={int(summary.get('fit_count') or 0)}/{int(summary.get('segment_count') or 0)}, "
                            f"refinement calls={int(summary.get('refinement_calls') or 0)}."
                        )
                    else:
                        print(
                            f"Native Pass 2 complete: {len(result.get('translations') or [])} natural isiZulu segment(s), "
                            "1 translation per segment with soft raw-duration intent."
                        )
                    cached_tokens = int(usage.get("cached_input_tokens") or 0)
                    cache_note = f", cached={cached_tokens:,}" if cached_tokens else ""
                    print(
                        f"Grok: {result.get('model') or provider.config.deployment}; "
                        f"tokens in={int(usage.get('input_tokens') or 0):,}{cache_note}, "
                        f"out={int(usage.get('output_tokens') or 0):,}."
                    )
                    print(f"Saved: {native_root / 'pass2_natural_translation.json'}")
                    _print_referent_audit_summary(result.get("referent_audit_summary"))
                else:
                    print(
                        f"Native translation complete: {int(result.get('pass1_corrections') or 0)} STT correction(s), "
                        f"{int(result.get('pass2_segments') or 0)} natural isiZulu segment(s)."
                    )
                    print(f"Pass 1: {result.get('pass1_path')}")
                    print(f"Pass 2: {result.get('pass2_path')}")
                    _print_referent_audit_summary(result.get("referent_audit_summary"))
            return 0

        if args.command == "native-qa":
            from .native_dub import run_native_qa_back_translation

            if not 1 <= int(args.batch_size) <= 40:
                raise ValueError("--batch-size must be between 1 and 40")
            if not 1 <= int(args.qa_audit_workers) <= 12:
                raise ValueError("--qa-audit-workers must be between 1 and 12")
            progress = _print_native_progress if args.live_operation else None
            provider = create_native_grok_backend(args.model, progress=progress)
            result = run_native_qa_back_translation(
                job_root=app.jobs.job_dir(job.job_id),
                provider=provider,
                force=args.force,
                batch_size=int(args.batch_size),
                repair=bool(args.repair),
                max_repair_rounds=int(args.max_qa_repair_rounds),
                direct_adequacy=bool(args.direct_adequacy),
                qa_audit_workers=int(args.qa_audit_workers),
                progress=progress,
            )
            job.review_readiness = (
                "needs_translation_review" if result.get("requires_review") else job.review_readiness
            )
            job.updated_at = utcnow()
            app.jobs.save(job)
            if args.json_output:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                repairs_applied = len(result.get("repairs") or [])
                accepted_minor = int(result.get("accepted_minor_count") or 0)
                print(
                    f"Native QA back-translation complete: {repairs_applied} repair(s) applied, "
                    f"{int(result.get('flagged_count') or 0)}/"
                    f"{int(result.get('sentence_count') or 0)} sentence(s) still flagged for review"
                    + (f" ({accepted_minor} minor finding(s) accepted without repair)." if accepted_minor else ".")
                )
                print(f"Grok: {result.get('model')}; tokens in={int((result.get('usage') or {}).get('input_tokens') or 0):,}, "
                      f"out={int((result.get('usage') or {}).get('output_tokens') or 0):,}.")
                for item in (result.get("flagged") or [])[:10]:
                    print(
                        f"  {item['ref']} [{item.get('severity', '?')}/{item.get('category', '?')}]: "
                        f"{item.get('discrepancy_summary') or 'no verdict returned'}"
                    )
                print(f"Saved: {app.jobs.job_dir(job.job_id) / 'native_dub' / 'qa_back_translation.json'}")
            return 0

        if args.command == "native-dub":
            from .native_dub import render_native_dub

            if args.repair_threshold_ms < 0:
                raise ValueError("--repair-threshold-ms must not be negative")
            if args.max_join_gap_ms < 0:
                raise ValueError("--max-join-gap-ms must not be negative")
            if args.max_phrase_source_ms <= 0:
                raise ValueError("--max-phrase-source-ms must be positive")
            if not 0 <= args.preferred_raw_speed_percent <= 12:
                raise ValueError("--preferred-raw-speed-percent must be between 0 and 12")
            if not 0 <= args.max_natural_speed_percent <= 15:
                raise ValueError("--max-natural-speed-percent must be between 0 and 15")
            if args.preferred_raw_speed_percent > args.max_natural_speed_percent:
                raise ValueError("--preferred-raw-speed-percent cannot exceed the configured rush warning threshold")
            if not 0 <= args.mouth_close_lag_tolerance_ms <= 80:
                raise ValueError("--mouth-close-lag-tolerance-ms must be between 0 and 80")
            if not 0 <= args.max_protected_gap_overflow_ms <= 1000:
                raise ValueError("--max-protected-gap-overflow-ms must be between 0 and 1000")
            if not 0 <= args.min_protected_pause_ms <= 2000:
                raise ValueError("--min-protected-pause-ms must be between 0 and 2000")
            progress = _print_native_progress if args.live_operation else None
            provider = create_native_grok_backend(args.model, progress=progress)
            sdk_group_synthesis_enabled = bool(getattr(settings, "enable_sdk_group_synthesis", False))
            turn_group_synthesis_enabled = bool(getattr(settings, "enable_turn_group_synthesis", False))
            needs_sdk_boundary = sdk_group_synthesis_enabled or turn_group_synthesis_enabled
            result = render_native_dub(
                job=job,
                job_root=app.jobs.job_dir(job.job_id),
                tts=create_tts_backend(settings),
                timing_repair_provider=provider,
                editorial_provider=provider,
                force=args.force,
                apply_opening_cut=args.hook_edit,
                render_hook_overlay=bool(args.hook_overlay),
                voice_map_path=(args.voice_map.resolve() if args.voice_map else None),
                max_join_gap_ms=args.max_join_gap_ms,
                max_phrase_source_ms=args.max_phrase_source_ms,
                repair_threshold_ms=args.repair_threshold_ms,
                preferred_raw_speed_percent=args.preferred_raw_speed_percent,
                max_natural_speed_percent=args.max_natural_speed_percent,
                mouth_close_lag_tolerance_ms=args.mouth_close_lag_tolerance_ms,
                protected_gap_overflow=args.protected_gap_overflow,
                max_protected_gap_overflow_ms=args.max_protected_gap_overflow_ms,
                min_protected_pause_ms=args.min_protected_pause_ms,
                enable_sdk_group_synthesis=sdk_group_synthesis_enabled,
                enable_turn_group_synthesis=turn_group_synthesis_enabled,
                sdk_boundary=(create_sdk_boundary(settings) if needs_sdk_boundary else None),
                lite=bool(args.lite),
                progress=progress,
            )
            job.providers["native_dub_ai"] = "azure-foundry-grok"
            job.providers["native_dub_model"] = provider.config.deployment
            job.providers["native_dub_tts"] = "azure-speech-rate-0-plus-bounded-hard-sync-atempo"
            job.media["native_dub_master"] = str(app.jobs.job_dir(job.job_id) / "native_dub" / "native_dubbed.mp4")
            job.media["native_dub_video"] = str(result["output_video"])
            job.media["native_dub_hook_manifest"] = str(app.jobs.job_dir(job.job_id) / "native_dub" / "tiktok_edit_manifest.json")
            job.media["native_dub_timing_plan"] = str(app.jobs.job_dir(job.job_id) / "native_dub" / "timing_plan.json")
            if "native_dub" not in job.completed_stages:
                job.completed_stages.append("native_dub")
            overlaps = ((result.get("dialogue") or {}).get("overlaps")) or []
            if any(overlap.get("human_review_required") for overlap in overlaps) or result.get(
                "timing_quality_review_required"
            ):
                job.review_readiness = "needs_timing_review"
            else:
                job.review_readiness = "review_ready"
            job.state = "review_ready"
            job.updated_at = utcnow()
            app.jobs.save(job)
            if args.json_output:
                print(json.dumps({
                    "schema_version": result["schema_version"],
                    "output_video": result["output_video"],
                    "phrase_group_count": result["phrase_group_count"],
                    "timing_repair_count": len(result["timing_repairs"]),
                    "video_retimed": result["video_retimed"],
                    "tts_rate_percent": result["tts_rate_percent"],
                    "preferred_raw_speed_percent": result.get("preferred_raw_speed_percent", 0),
                    "max_natural_speed_percent": result.get("max_natural_speed_percent", 0),
                    "mouth_close_target_tolerance_ms": result.get("mouth_close_target_tolerance_ms", 0),
                    "timing_policy": result.get("timing_policy"),
                    "hook_overlay_enabled": result.get("hook_overlay_enabled", False),
                    "hook_edit_enabled": result.get("hook_edit_enabled", False),
                    "native_master_video": result.get("native_master_video"),
                    "timing_quality_review_required": result.get("timing_quality_review_required", False),
                    "timing_quality_flags": result.get("timing_quality_flags", []),
                    "rush_by_turn": _native_dub_rush_rows(result.get("groups")),
                }, ensure_ascii=False, indent=2))
            else:
                print(
                    f"Native dub complete: {int(result.get('phrase_group_count') or 0)} phrase group(s), "
                    f"{len(result.get('timing_repairs') or [])} timing repair(s), "
                    f"TTS baseline rate {int(result.get('tts_rate_percent') or 0):+d}%, "
                    f"raw target +{int(result.get('preferred_raw_speed_percent') or 0)}%, "
                    f"speed cap +{int(result.get('max_natural_speed_percent') or 0)}%, "
                    f"mouth-close tolerance {int(result.get('mouth_close_target_tolerance_ms') or 0)} ms."
                )
                print(
                    "Video retimed: " + ("yes" if result.get("video_retimed") else "no")
                    + "; hook overlay: " + ("yes" if result.get("hook_overlay_enabled") else "no")
                    + "; opening hook edit: " + ("yes" if result.get("hook_edit_enabled") else "no")
                )
                if result.get("native_master_video"):
                    print(f"Native master: {result['native_master_video']}")
                if result.get("timing_quality_review_required"):
                    flags = result.get("timing_quality_flags") or []
                    worst = max(flags, key=lambda item: item.get("natural_overrun_ms", 0), default=None)
                    print(
                        f"NEEDS TIMING REVIEW: {len(flags)} unit(s) exceeded the timing-quality threshold"
                        + (f" (worst: {worst['group_id']} at {worst['natural_overrun_ms']} ms / "
                           f"{worst['raw_speed_percent']:.1f}%)" if worst else "")
                        + "."
                    )
                print(f"Final output: {result['output_video']}")
                _print_native_dub_rush_summary(result.get("groups"))
            return 0

        # Lightweight test settings may omit provider fields; production always
        # uses the GPT-only provider factory below.
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
        elif args.command == "manual-ai":
            from .manual_ai import (
                install_manual_response,
                manual_ai_root_for_job,
                pending_manual_requests,
            )

            manual_root = manual_ai_root_for_job(settings.work_dir, job.job_id)
            if args.action == "pending":
                print(
                    json.dumps(
                        {
                            "job_id": job.job_id,
                            "manual_ai_dir": str(manual_root),
                            "pending": pending_manual_requests(manual_root),
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
            else:
                if args.response_json is None:
                    raise ValueError("manual-ai install requires response_json")
                result = install_manual_response(
                    manual_root, args.response_json.resolve()
                )
                print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command == "translate":
            provider_name = (
                args.provider
                or (
                    "manual-chatgpt"
                    if is_manual_ai_provider(os.getenv("MATHULA_TV_AI_PROVIDER"))
                    else "azure-openai-gpt"
                )
            )
            provider = create_translation_backend(
                provider_name,
                args.model
                or os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
                or os.getenv("AZURE_AI_DEPLOYMENT"),
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
                parallelism=args.parallelism,
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
            provider_name = (
                args.provider
                or (
                    "manual-chatgpt"
                    if is_manual_ai_provider(os.getenv("MATHULA_TV_AI_PROVIDER"))
                    else "azure-openai-gpt"
                )
            )
            if provider_name not in {
                "azure-openai-gpt", "azure-openai", "gpt",
                "manual-chatgpt", "manual", "chatgpt-manual", "manual-gpt",
            }:
                raise ValueError(
                    "Timing repair requires Azure OpenAI GPT or manual-chatgpt"
                )
            provider = create_translation_backend(
                provider_name,
                args.model
                or os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
                or os.getenv("AZURE_AI_DEPLOYMENT"),
            )
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
        elif args.command == "dub-azure":
            from .direct_azure_dub import (
                DirectAzureDubRenderer,
                DirectDubOptions,
                create_direct_azure_backend,
            )

            azure_tts_backend = create_direct_azure_backend(
                settings,
                max_rate_percent=settings.azure_rate_max_percent,
            )
            renderer = DirectAzureDubRenderer(
                settings,
                azure_tts_backend,
                # Created lazily for unresolved transcript-context voice
                # inference, or later if the finalization conductor is needed.
                timing_repair_provider_factory=create_production_ai_provider,
            )
            progress_elapsed = {"seconds": 0.0}
            phase_timing = _PhaseDurationTracker()

            def progress_callback(event: dict[str, Any]) -> None:
                if not args.progress:
                    return
                payload = dict(event)
                phase_timing.observe(payload)
                elapsed = payload.get("elapsed_seconds")
                if isinstance(elapsed, (int, float)):
                    progress_elapsed["seconds"] = float(elapsed)
                else:
                    payload["elapsed_seconds"] = progress_elapsed["seconds"]
                _print_direct_dub_progress(payload)

            from .grammar_quality import (
                GrammarQualityGateError,
                ensure_job_grammar_quality,
            )

            if args.progress:
                progress_callback(
                    {
                        "stage": "grammar-editor",
                        "message": (
                            "Running or reusing the full-context grammar editor "
                            "and independent grammar QA"
                        ),
                    }
                )
            grammar_gate_path = (
                settings.work_dir
                / "jobs"
                / job.job_id
                / "translation"
                / "grammar_quality_gate.json"
            )
            grammar_provider = create_production_ai_provider()
            try:
                grammar_gate = ensure_job_grammar_quality(
                    job_root=settings.work_dir / "jobs" / job.job_id,
                    job_id=job.job_id,
                    target_locale=job.target_language,
                    provider=grammar_provider,
                    pronunciation_dictionary=production._pronunciation_dictionary(job),
                )
            except GrammarQualityGateError as exc:
                job.human_review_flags = list(
                    dict.fromkeys(
                        [*job.human_review_flags, "contextual_grammar_qa_failed"]
                    )
                )
                job.last_error = {
                    "stage": "contextual_grammar_qa",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "retryable": True,
                    "gate_path": str(grammar_gate_path),
                }
                app.jobs.save(job)
                raise
            grammar_provider_name = str(
                getattr(grammar_provider, "provider", "azure-openai-gpt")
            )
            job.providers["grammar_editor"] = grammar_provider_name
            job.providers["grammar_qa"] = f"{grammar_provider_name}-independent-pass"
            if grammar_gate.get("translation_changed") is True:
                job.media["translation_multivariant_sha256"] = str(
                    grammar_gate["translation_sha256"]
                )
                job.media["translation_sha256"] = str(
                    grammar_gate["dubbing_sha256"]
                )
            job.media["grammar_quality_gate"] = str(grammar_gate_path)
            job.media["grammar_quality_gate_sha256"] = checksum(grammar_gate_path)
            job.human_review_flags = [
                value
                for value in job.human_review_flags
                if value != "contextual_grammar_qa_failed"
            ]
            if "contextual_grammar_qa" not in job.completed_stages:
                job.completed_stages.append("contextual_grammar_qa")
            if (
                isinstance(job.last_error, Mapping)
                and job.last_error.get("stage") == "contextual_grammar_qa"
            ):
                job.last_error = None
            app.jobs.save(job)
            if args.progress:
                progress_callback(
                    {
                        "stage": "grammar-qa",
                        "status": "completed",
                        "message": (
                            "Contextual grammar QA passed; synthesis is authorized"
                        ),
                    }
                )

            direct_options = DirectDubOptions(
                timing_mode=args.timing_mode,
                semantic_fit_max_attempts=args.semantic_fit_max_attempts,
                semantic_fit_tolerance_ms=args.semantic_fit_tolerance_ms,
                max_azure_rate_percent=min(
                    10,
                    azure_tts_backend.ssml_bounds.rate_max_percent,
                ),
                min_voice_family_confidence=args.min_confidence,
                # The publication pass immediately follows and performs the
                # single required CFR H.264 encode. Keep the intermediate
                # clean master as a fast video-copy remux.
                clean_master_video_mode=os.getenv(
                    "MATHULA_TV_CLEAN_MASTER_VIDEO_MODE",
                    "stream_copy",
                ).strip().lower(),
                finalization_ai_rounds=int(
                    os.getenv("MATHULA_TV_FINALIZATION_AI_ROUNDS", "2")
                ),
            )
            result = renderer.render(
                job,
                voice_map_path=args.voice_map,
                options=direct_options,
                force=(
                    args.force
                    or grammar_gate.get("translation_changed") is True
                ),
                refresh_voice_analysis=args.refresh_voice_analysis,
                progress_callback=(progress_callback if args.progress else None),
            )
            if grammar_gate.get("translation_changed") is True:
                grammar_gate = {
                    **grammar_gate,
                    "translation_changed": False,
                    "rendered_translation_sha256": grammar_gate.get(
                        "translation_sha256"
                    ),
                    "rendered_at": utcnow(),
                }
                atomic_write_json(grammar_gate_path, grammar_gate)
                job.media["grammar_quality_gate_sha256"] = checksum(
                    grammar_gate_path
                )
                app.jobs.save(job)

            post_tts_qa: dict[str, Any]
            if args.qa_stt:
                from .dub_mastering import (
                    DubMasteringLoop,
                    DubMasteringThresholds,
                    build_post_tts_quality_gate,
                )
                qa_elapsed_base = progress_elapsed["seconds"]
                qa_started_at = time.monotonic()

                def qa_progress(event: dict[str, Any]) -> None:
                    payload = dict(event)
                    payload["elapsed_seconds"] = (
                        qa_elapsed_base
                        + max(0.0, time.monotonic() - qa_started_at)
                    )
                    progress_callback(payload)

                qa_loop = DubMasteringLoop(
                    stt_backend=create_speech_backend(settings),
                    renderer=renderer,
                    qa_review_provider_factory=create_production_ai_provider,
                    progress_callback=(qa_progress if args.progress else None),
                )
                qa_result = qa_loop.run(
                    job,
                    options=direct_options,
                    max_rounds=args.qa_stt_max_rounds,
                    audit_only=False,
                    thresholds=DubMasteringThresholds(
                        minimum_production_score=(
                            args.qa_minimum_production_score
                        ),
                        maximum_aggregate_wer=(
                            args.qa_maximum_aggregate_wer
                        ),
                    ),
                    voice_map_path=args.voice_map,
                )
                mastering_report_path = (
                    settings.work_dir
                    / "jobs"
                    / job.job_id
                    / "direct_dub"
                    / "mastering"
                    / "report.json"
                )
                gate_path = mastering_report_path.with_name(
                    "post_tts_quality_gate.json"
                )
                post_tts_qa = build_post_tts_quality_gate(
                    qa_result,
                    report_path=mastering_report_path,
                    grammar_gate=grammar_gate,
                )
                atomic_write_json(gate_path, post_tts_qa)
                direct_report_path = (
                    settings.work_dir
                    / "jobs"
                    / job.job_id
                    / "direct_dub"
                    / "report.json"
                )
                direct_report = read_json(direct_report_path)
                direct_report["post_tts_qa"] = {
                    **post_tts_qa,
                    "gate_path": str(gate_path),
                }
                atomic_write_json(direct_report_path, direct_report)
                job.providers["post_tts_qa"] = "azure-speech"
                job.media["post_tts_qa_report"] = str(mastering_report_path)
                job.media["post_tts_qa_report_sha256"] = checksum(
                    mastering_report_path
                )
                job.media["post_tts_qa_gate"] = str(gate_path)
                job.media["post_tts_qa_gate_sha256"] = checksum(gate_path)
                if post_tts_qa["publication_authorized"]:
                    if "post_tts_stt_qa" not in job.completed_stages:
                        job.completed_stages.append("post_tts_stt_qa")
                    job.human_review_flags = [
                        value
                        for value in job.human_review_flags
                        if value != "post_tts_stt_qa_failed"
                    ]
                    if (
                        isinstance(job.last_error, Mapping)
                        and job.last_error.get("stage") == "post_tts_stt_qa"
                    ):
                        job.last_error = None
                else:
                    job.human_review_flags = list(
                        dict.fromkeys(
                            [
                                *job.human_review_flags,
                                "post_tts_stt_qa_failed",
                            ]
                        )
                    )
                    job.last_error = {
                        "stage": "post_tts_stt_qa",
                        "error_type": "ProductionQualityGateFailed",
                        "message": (
                            "Rendered dub did not pass the post-TTS Azure STT "
                            "production quality gate"
                        ),
                        "retryable": True,
                        "report_path": str(mastering_report_path),
                        "gate_path": str(gate_path),
                    }
                app.jobs.save(job)
                if not post_tts_qa["publication_authorized"]:
                    raise RuntimeError(
                        "Post-TTS Azure STT production QA failed: "
                        f"score {post_tts_qa['best_production_score']:.1f}/100, "
                        f"WER {post_tts_qa['aggregate_word_error_rate']:.3f}, "
                        f"{post_tts_qa['production_blocker_count']} blocker(s); "
                        f"publication stopped. Review {gate_path}"
                    )
                # Mastering may have selected another already-approved variant
                # and rerendered the mix. Reload the authoritative dub report.
                result = read_json(
                    settings.work_dir
                    / "jobs"
                    / job.job_id
                    / "direct_dub"
                    / "report.json"
                )
            else:
                post_tts_qa = {
                    "state": "disabled",
                    "publication_authorized": True,
                    "reason": "explicit_no_qa_stt",
                }
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
            publication_elapsed_base = progress_elapsed["seconds"]
            publication_started_at = time.monotonic()

            def publication_progress(
                event: dict[str, Any],
            ) -> None:
                payload = dict(event)
                stage_elapsed = payload.get("stage_elapsed_seconds")
                wall_elapsed = max(
                    0.0,
                    time.monotonic() - publication_started_at,
                )
                reported_elapsed = (
                    max(wall_elapsed, float(stage_elapsed))
                    if isinstance(stage_elapsed, (int, float))
                    else wall_elapsed
                )
                payload["elapsed_seconds"] = (
                    publication_elapsed_base + reported_elapsed
                )
                progress_callback(payload)

            publication_provider = create_production_ai_provider()
            with_request_options = getattr(
                publication_provider,
                "with_request_options",
                None,
            )
            if args.progress and callable(with_request_options):
                def publication_provider_event(
                    event: dict[str, Any],
                ) -> None:
                    payload = _publication_ai_progress_payload(event)
                    if payload is not None:
                        publication_progress(payload)

                publication_provider = with_request_options(
                    event_callback=publication_provider_event,
                )
            tiktok_edit = edit_tiktok_job(
                work_dir=settings.work_dir,
                job=job,
                provider=publication_provider,
                # Reuse by master/translation checksum. A forced dub does not
                # justify paying for the same AI hook or re-encoding it again.
                force=False,
                apply_opening_cut=args.hook_edit,
                progress_callback=(
                    publication_progress if args.progress else None
                ),
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
            result = {
                **final_report,
                "post_tts_qa": post_tts_qa,
                "tiktok_edit": tiktok_edit,
            }
            if args.progress:
                _print_direct_dub_phase_summary(phase_timing.finish())
                phase_timing_reported = True
            _print_dub_azure_result(
                job_id=job.job_id,
                result=result,
                json_output=args.json_output,
            )
        elif args.command == "optimize-dub":
            from .direct_azure_dub import DirectAzureDubRenderer, DirectDubOptions
            from .dub_mastering import (
                DubMasteringLoop,
                DubMasteringThresholds,
                build_post_tts_quality_gate,
            )
            from .grammar_quality import (
                GrammarQualityGateError,
                ensure_job_grammar_quality,
            )

            grammar_gate_path = (
                settings.work_dir
                / "jobs"
                / job.job_id
                / "translation"
                / "grammar_quality_gate.json"
            )
            grammar_provider = create_production_ai_provider()
            try:
                grammar_gate = ensure_job_grammar_quality(
                    job_root=settings.work_dir / "jobs" / job.job_id,
                    job_id=job.job_id,
                    target_locale=job.target_language,
                    provider=grammar_provider,
                    pronunciation_dictionary=production._pronunciation_dictionary(job),
                )
            except GrammarQualityGateError as exc:
                job.human_review_flags = list(
                    dict.fromkeys(
                        [*job.human_review_flags, "contextual_grammar_qa_failed"]
                    )
                )
                job.last_error = {
                    "stage": "contextual_grammar_qa",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "retryable": True,
                    "gate_path": str(grammar_gate_path),
                }
                app.jobs.save(job)
                raise
            if grammar_gate.get("translation_changed") is True:
                job.media["translation_multivariant_sha256"] = str(
                    grammar_gate["translation_sha256"]
                )
                job.media["translation_sha256"] = str(
                    grammar_gate["dubbing_sha256"]
                )
                app.jobs.save(job)
            if grammar_gate.get("translation_changed") is True:
                raise RuntimeError(
                    "The grammar editor changed the translation; run dub-azure to "
                    "rerender it before mastering"
                )

            azure_tts_backend = create_tts_backend(settings)
            stt_backend = create_speech_backend(settings)
            renderer = DirectAzureDubRenderer(
                settings,
                azure_tts_backend,
                timing_repair_provider_factory=create_production_ai_provider,
            )
            mastering_started = time.monotonic()

            def mastering_progress(event: dict[str, Any]) -> None:
                if not args.progress:
                    return
                payload = dict(event)
                payload.setdefault(
                    "elapsed_seconds",
                    max(0.0, time.monotonic() - mastering_started),
                )
                _print_direct_dub_progress(payload)

            loop = DubMasteringLoop(
                stt_backend=stt_backend,
                renderer=renderer,
                qa_review_provider_factory=create_production_ai_provider,
                progress_callback=(mastering_progress if args.progress else None),
            )
            result = loop.run(
                job,
                options=DirectDubOptions(
                    max_azure_rate_percent=min(
                        10,
                        azure_tts_backend.ssml_bounds.rate_max_percent,
                    ),
                    min_voice_family_confidence=args.min_confidence,
                    clean_master_video_mode=os.getenv(
                        "MATHULA_TV_CLEAN_MASTER_VIDEO_MODE",
                        "stream_copy",
                    ).strip().lower(),
                    finalization_ai_rounds=int(
                        os.getenv("MATHULA_TV_FINALIZATION_AI_ROUNDS", "2")
                    ),
                ),
                max_rounds=args.max_rounds,
                audit_only=args.audit_only,
                thresholds=DubMasteringThresholds(
                    minimum_production_score=args.minimum_production_score,
                    maximum_aggregate_wer=args.maximum_aggregate_wer,
                ),
                voice_map_path=args.voice_map,
            )
            mastering_report_path = (
                settings.work_dir
                / "jobs"
                / job.job_id
                / "direct_dub"
                / "mastering"
                / "report.json"
            )
            gate_path = mastering_report_path.with_name("post_tts_quality_gate.json")
            quality_gate = build_post_tts_quality_gate(
                result,
                report_path=mastering_report_path,
                grammar_gate=grammar_gate,
            )
            atomic_write_json(gate_path, quality_gate)
            direct_report_path = mastering_report_path.parent.parent / "report.json"
            if direct_report_path.is_file():
                direct_report = read_json(direct_report_path)
                direct_report["post_tts_qa"] = {
                    **quality_gate,
                    "gate_path": str(gate_path),
                }
                atomic_write_json(direct_report_path, direct_report)
            result = {**result, "post_tts_qa": quality_gate}
            if args.json_output:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                best_round = next(
                    (
                        value
                        for value in result.get("rounds") or []
                        if int(value.get("round") or 0)
                        == int(result.get("best_round") or 0)
                    ),
                    {},
                )
                print(
                    f"Dub mastering {result['state']} for job {job.job_id}: "
                    f"best score {result['best_production_score']:.1f}/100; "
                    f"report {settings.work_dir / 'jobs' / job.job_id / 'direct_dub' / 'mastering' / 'report.json'}"
                )
                dimensions = best_round.get("quality_dimensions") or {}
                if dimensions:
                    print(
                        "Quality dimensions: "
                        + ", ".join(
                            f"{name.replace('_', ' ')}={float(value):.1f}"
                            for name, value in dimensions.items()
                        )
                    )
                print(
                    "Production blockers in best round: "
                    f"{int(best_round.get('production_blocker_count') or 0)}"
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
        elif args.command == "repair-speaker-turns":
            job_root = app.jobs.job_dir(job.job_id)
            azure_diarization = job_root / "analysis" / "azure_diarization.json"
            if not azure_diarization.is_file():
                raise ValueError(
                    "Cannot rebuild contextual speaker turns without "
                    f"{azure_diarization}"
                )
            previous_state = _reset_for_contextual_speaker_repair(job)
            app.jobs.save(job)
            result = app.process(job)
            repaired_transcript = read_json(
                job_root / "analysis" / "transcript_en.json"
            )
            repairs = repaired_transcript.get("speaker_turn_repairs", [])
            print(
                json.dumps(
                    {
                        "job_id": job.job_id,
                        "from_state": previous_state,
                        "state": job.state,
                        "speaker_turn_repair_count": len(repairs),
                        "speaker_turn_repairs": repairs,
                        "analysis_result": result,
                        "next_step": (
                            "Run translate with --live-operation; changed source "
                            "identity invalidates stale translation checkpoints"
                        ),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        elif args.command == "resume-autocorrect-research":
            job_root = app.jobs.job_dir(job.job_id)
            azure_diarization = (
                job_root / "analysis" / "azure_diarization.json"
            )
            research_artifact = (
                job_root / "analysis" / "autocorrection_name_research.json"
            )
            if not azure_diarization.is_file():
                raise ValueError(
                    "Cannot resume autocorrect research without "
                    f"{azure_diarization}"
                )
            if not research_artifact.is_file():
                raise ValueError(
                    "No previous autocorrect research artifact exists for this job"
                )
            artifact = read_json(research_artifact)
            if artifact.get("status") in {
                "completed",
                "no_name_candidates",
                "no_research_candidates",
            } and not args.force:
                raise ValueError(
                    "Autocorrect research is already complete; this command "
                    "resumes only failed or incomplete research unless --force "
                    "is supplied to rebuild and reapply cached evidence"
                )
            previous_state = _reset_for_autocorrect_research(job)
            app.jobs.save(job)
            result = app.process(job)
            print(
                result
                or (
                    "Autocorrect research resumed from "
                    f"{previous_state}; analysis rebuilt"
                )
            )
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
                    atomic_write_json(pyannote_path, result)
                    print("Optional Pyannote diagnostic ready; Azure speaker labels remain authoritative")
            else:
                # No existing artifacts - run Pyannote
                result = run_pyannote(Path(job.objects["local_audio"]), settings.pyannote_model)
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
            exc, "review"
        ):
            from .timeline_collision_panel import (
                wait_for_timeline_collision_resolutions,
            )

            cycle_count = int(
                getattr(main, "_timeline_collision_review_cycles", 0)
            )
            if cycle_count >= 20:
                raise RuntimeError(
                    "More than 20 timeline collision review cycles were requested"
                ) from exc
            setattr(main, "_timeline_collision_review_cycles", cycle_count + 1)
            try:
                wait_for_timeline_collision_resolutions(
                    review=exc.review,
                    jobs_root=app.jobs.root,
                    host=getattr(args, "timeline_panel_host", "127.0.0.1"),
                    port=int(getattr(args, "timeline_panel_port", 8765)),
                )
                # Re-enter the same command after the matching decision is saved.
                # The direct renderer reuses its Azure and finalized-block caches.
                return main()
            finally:
                setattr(main, "_timeline_collision_review_cycles", cycle_count)
        if (
            phase_timing is not None
            and getattr(args, "progress", False)
            and not phase_timing_reported
        ):
            _print_direct_dub_phase_summary(phase_timing.finish())
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
