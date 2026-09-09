from __future__ import annotations

import json
import os
import re
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


@dataclass(frozen=True)
class Settings:
    work_dir: Path
    gcs_bucket: str
    gcs_prefix: str
    commission_master_case_path: Path
    politics_context_path: Path
    azure_speech_endpoint: str
    azure_speech_region: str
    azure_speech_locale: str
    azure_speech_api_version: str
    azure_speech_max_speakers: int
    azure_speech_timeout_seconds: float
    azure_ai_endpoint: str
    azure_ai_deployment: str
    azure_ai_api_version: str
    pyannote_model: str
    speaker_tolerance_seconds: float = 0.35
    lease_seconds: int = 900
    ai_provider: str = "azure-openai-gpt"
    # Deprecated names are retained only so older callers can construct
    # Settings and read historical job snapshots. Runtime code uses the GPT
    # fields below and never reads Claude credentials or endpoints.
    claude_model: str = "claude-opus-4-8"
    claude_timeout_seconds: float = 900
    claude_max_retries: int = 3
    claude_effort: str = "high"
    translation_claude_effort: str = "low"
    translation_claude_thinking: str = "disabled"
    translation_max_output_tokens: int = 100_000
    seo_claude_effort: str = "medium"
    seo_claude_thinking: str = "adaptive"
    seo_max_output_tokens: int = 12_000
    hook_claude_effort: str = "low"
    hook_claude_thinking: str = "disabled"
    hook_max_output_tokens: int = 12_000
    editorial_claude_effort: str = "high"
    editorial_claude_thinking: str = "adaptive"
    editorial_max_output_tokens: int = 12_000
    translation_batch_size: int = 6
    translation_context_units: int = 2
    translation_request_timeout_seconds: float = 300
    translation_batch_max_retries: int = 2
    translation_max_provider_calls_per_session: int = 64
    azure_tts_voices: tuple[str, ...] = ("zu-ZA-ThandoNeural", "zu-ZA-ThembaNeural")
    azure_tts_default_voice: str = "zu-ZA-ThandoNeural"
    azure_tts_sample_rate: int = 24000
    azure_tts_channels: int = 1
    azure_tts_sample_width: int = 2
    azure_tts_timeout_seconds: float = 120
    azure_tts_max_retries: int = 3
    azure_rate_min_percent: int = -12
    azure_rate_max_percent: int = 15
    azure_rate_search_attempts: int = 3
    preferred_end_tolerance_ms: int = 120
    post_conversion_min_speed: float = 0.94
    post_conversion_max_speed: float = 1.06
    final_timing_tolerance_ms: int = 120
    openvoice_revision: str = ""
    openvoice_checkpoint_hashes: tuple[tuple[str, str], ...] = ()
    openvoice_max_attempts: int = 2
    target_lufs: float = -14.0
    true_peak_dbtp: float = -1.5
    target_lra: float = 11.0
    max_source_duration_minutes: int = 240
    max_claude_calls_per_job: int = 20
    max_translation_repairs_per_unit: int = 2
    max_azure_tts_attempts_per_unit: int = 3
    max_openvoice_attempts_per_unit: int = 2
    max_failed_units: int = 2
    max_job_runtime_seconds: int = 14400
    max_artifact_download_bytes: int = 2 * 1024 * 1024 * 1024
    pyannote_enabled: bool = False
    max_clean_unit_wer: float = 0.35
    max_openvoice_wer_degradation: float = 0.10
    max_final_mix_wer_degradation: float = 0.15
    min_protected_entity_similarity: float = 0.85
    gpt_model: str = "gpt-5.6-sol-1"
    gpt_timeout_seconds: float = 900
    gpt_max_retries: int = 3
    gpt_effort: str = "high"
    translation_gpt_effort: str = "low"
    seo_gpt_effort: str = "medium"
    hook_gpt_effort: str = "low"
    editorial_gpt_effort: str = "high"
    max_ai_calls_per_job: int = 20
    # New, unvalidated fast path (see azure_tts_sdk.py): grouped bookmark synthesis
    # of speech islands via the Speech SDK instead of one REST call per island.
    # Defaults off (unlike other native-dub timing features) because it touches
    # real Azure billing/audio quality and has not yet been validated on a real job.
    enable_sdk_group_synthesis: bool = False
    # Phase 14: synthesize a whole multi-sentence speaker turn in ONE bookmarked
    # Speech SDK call and speed-fit it as one unit, instead of fitting each
    # sentence in the turn independently -- fixes a confirmed real defect where
    # a turn's last sentence drifts audibly past the true mouth-close (see
    # native_dub.py's _synthesize_speaker_turns_with_bookmarks). Independent of
    # enable_sdk_group_synthesis (a different, still-unvalidated mechanism for a
    # different granularity) so each can be rolled out/validated separately.
    enable_turn_group_synthesis: bool = False
    # Real yt-dlp defect confirmed 2026-09-09: some YouTube videos return
    # LOGIN_REQUIRED to every player client (independent of the bgutil PO-token
    # provider, which handles the generic bot-check but not this gate) even
    # though the video is genuinely public. A real browser cookies.txt is the
    # only confirmed fix; empty means submit yt-dlp calls without --cookies.
    youtube_cookies_file: str = ""

    def safe_snapshot(self) -> dict:
        legacy_keys = {
            "claude_model",
            "claude_timeout_seconds",
            "claude_max_retries",
            "claude_effort",
            "translation_claude_effort",
            "translation_claude_thinking",
            "seo_claude_effort",
            "seo_claude_thinking",
            "hook_claude_effort",
            "hook_claude_thinking",
            "editorial_claude_effort",
            "editorial_claude_thinking",
            "max_claude_calls_per_job",
        }
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(self).items()
            if key not in legacy_keys
        }


def _resolve_windows_safe_path(default: Path, raw_value: str | None, *, platform_name: str | None = None) -> Path:
    """Resolve a configured path without silently accepting a stale value from the wrong platform.

    A common migration failure is carrying a Linux value such as
    ``/home/user/mathula-tv/working`` into a Windows ``.env``.  ``pathlib`` on
    Windows interprets that as a root-relative path on the current drive, e.g.
    ``C:\\home\\user\\...``, which silently sends job artifacts, commission
    context, or politics context to the wrong (nonexistent) tree.  On Windows,
    reject POSIX/root-relative values that do not name a drive or UNC share
    and fall back to ``default`` instead.

    The reverse migration failure is just as real -- a shared ``.env`` carrying
    a Windows value such as ``C:\\mathula-tv\\working`` onto a Linux/remote
    server.  ``pathlib.PosixPath`` never treats backslashes as separators, so
    the whole drive-letter string becomes one literal path component; worse,
    if anything upstream (a shell, a ``.env`` loader) has already eaten the
    backslashes as unrecognised escapes, the result silently mangles into
    something like ``C:mathula-tvworking`` -- confirmed as a real production
    failure (job submission wrote ffprobe input paths under a nonexistent
    ``C:mathula-tvworking/jobs/...`` tree on a Linux box). On any non-Windows
    platform, reject a drive-letter or UNC value the same way and fall back to
    ``default``.
    """

    platform_name = platform_name or os.name
    raw = str(raw_value or "").strip()
    if not raw:
        return default

    if platform_name == "nt":
        normalized = raw.replace("\\", "/")
        has_drive = len(normalized) >= 3 and normalized[0].isalpha() and normalized[1:3] == ":/"
        is_unc = raw.startswith("\\\\") or normalized.startswith("//")
        is_root_relative = normalized.startswith("/") and not is_unc
        if is_root_relative and not has_drive:
            warnings.warn(
                "Ignoring non-Windows path %r on Windows; using default %s" % (raw, default),
                RuntimeWarning,
                stacklevel=2,
            )
            return default
    else:
        normalized = raw.replace("\\", "/")
        has_drive = len(normalized) >= 3 and normalized[0].isalpha() and normalized[1:3] == ":/"
        is_unc = raw.startswith("\\\\") or normalized.startswith("//")
        if has_drive or is_unc:
            warnings.warn(
                "Ignoring Windows-style path %r on a non-Windows platform; using default %s" % (raw, default),
                RuntimeWarning,
                stacklevel=2,
            )
            return default

    return Path(raw)


def _resolve_work_dir(root: Path, raw_value: str | None = None, *, platform_name: str | None = None) -> Path:
    return _resolve_windows_safe_path(root / "working", raw_value, platform_name=platform_name)


def load_settings(project_root: Path | None = None) -> Settings:
    root = project_root or Path(__file__).resolve().parents[2]
    if load_dotenv:
        load_dotenv(root / ".env", override=False)
    work = _resolve_work_dir(root, os.getenv("MATHULA_TV_WORK_DIR"))
    commission_default = (
        Path("/home/mokgethwa/commission-ai/working/case_state/master_case_state.json")
        if os.name != "nt"
        else work / "context/commission_master_case_state.json"
    )
    voices = tuple(
        item.strip()
        for item in os.getenv("MATHULA_TV_AZURE_TTS_VOICES", "zu-ZA-ThandoNeural,zu-ZA-ThembaNeural").split(",")
        if item.strip()
    )
    return Settings(
        work_dir=work,
        gcs_bucket=os.getenv("MATHULA_TV_GCS_BUCKET", ""),
        gcs_prefix=os.getenv("MATHULA_TV_GCS_PREFIX", "mathula-tv").strip("/"),
        commission_master_case_path=_resolve_windows_safe_path(
            commission_default, os.getenv("COMMISSION_MASTER_CASE_PATH")
        ),
        politics_context_path=_resolve_windows_safe_path(
            work / "context/politics_context.json", os.getenv("POLITICS_CONTEXT_PATH")
        ),
        azure_speech_endpoint=os.getenv("AZURE_SPEECH_ENDPOINT", ""),
        azure_speech_region=os.getenv("AZURE_SPEECH_REGION", ""),
        azure_speech_locale=os.getenv("AZURE_SPEECH_LOCALE", "en-ZA"),
        azure_speech_api_version=os.getenv("AZURE_SPEECH_API_VERSION", "2025-10-15"),
        azure_speech_max_speakers=int(os.getenv("AZURE_SPEECH_MAX_SPEAKERS", "10")),
        azure_speech_timeout_seconds=float(os.getenv("AZURE_SPEECH_TIMEOUT_SECONDS", "600")),
        azure_ai_endpoint=(os.getenv("AZURE_AI_ENDPOINT") or os.getenv("AZURE_OPENAI_ENDPOINT") or "").rstrip("/"),
        azure_ai_deployment=os.getenv("AZURE_AI_DEPLOYMENT") or os.getenv("AZURE_OPENAI_DEPLOYMENT") or "",
        azure_ai_api_version=os.getenv("AZURE_AI_API_VERSION") or os.getenv("AZURE_OPENAI_API_VERSION") or "preview",
        pyannote_model=os.getenv("PYANNOTE_MODEL", "pyannote/speaker-diarization-3.1"),
        speaker_tolerance_seconds=float(os.getenv("MATHULA_TV_SPEAKER_TOLERANCE_SECONDS", "0.35")),
        lease_seconds=int(os.getenv("MATHULA_TV_LEASE_SECONDS", "900")),
        # GPT is the sole runtime provider. A stale historical selector in an
        # existing .env is intentionally ignored by the provider factory too.
        ai_provider="azure-openai-gpt",
        claude_model="claude-opus-4-8",
        claude_timeout_seconds=900,
        claude_max_retries=3,
        claude_effort="high",
        translation_claude_effort="low",
        translation_claude_thinking="disabled",
        translation_max_output_tokens=int(
            os.getenv("MATHULA_TV_TRANSLATION_MAX_OUTPUT_TOKENS", "100000")
        ),
        seo_claude_effort="medium",
        seo_claude_thinking="adaptive",
        seo_max_output_tokens=int(
            os.getenv(
                "MATHULA_TV_SEO_MAX_OUTPUT_TOKENS",
                "12000",
            )
        ),
        hook_claude_effort="low",
        hook_claude_thinking="disabled",
        hook_max_output_tokens=int(
            os.getenv(
                "MATHULA_TV_HOOK_MAX_OUTPUT_TOKENS",
                "12000",
            )
        ),
        editorial_claude_effort="high",
        editorial_claude_thinking="adaptive",
        editorial_max_output_tokens=int(
            os.getenv(
                "MATHULA_TV_EDITORIAL_MAX_OUTPUT_TOKENS",
                "12000",
            )
        ),
        translation_batch_size=int(os.getenv("MATHULA_TV_TRANSLATION_BATCH_SIZE", "6")),
        translation_context_units=int(os.getenv("MATHULA_TV_TRANSLATION_CONTEXT_UNITS", "2")),
        translation_request_timeout_seconds=float(os.getenv("MATHULA_TV_TRANSLATION_REQUEST_TIMEOUT_SECONDS", "300")),
        translation_batch_max_retries=int(os.getenv("MATHULA_TV_TRANSLATION_BATCH_MAX_RETRIES", "2")),
        translation_max_provider_calls_per_session=int(
            os.getenv("MATHULA_TV_TRANSLATION_MAX_PROVIDER_CALLS_PER_SESSION", "64")
        ),
        azure_tts_voices=voices,
        azure_tts_default_voice=os.getenv("MATHULA_TV_AZURE_TTS_DEFAULT_VOICE", "zu-ZA-ThandoNeural"),
        azure_tts_sample_rate=int(os.getenv("MATHULA_TV_AZURE_TTS_SAMPLE_RATE", "24000")),
        azure_tts_channels=int(os.getenv("MATHULA_TV_AZURE_TTS_CHANNELS", "1")),
        azure_tts_sample_width=int(os.getenv("MATHULA_TV_AZURE_TTS_SAMPLE_WIDTH", "2")),
        azure_tts_timeout_seconds=float(os.getenv("MATHULA_TV_AZURE_TTS_TIMEOUT_SECONDS", "120")),
        azure_tts_max_retries=int(os.getenv("MATHULA_TV_AZURE_TTS_MAX_RETRIES", "3")),
        azure_rate_min_percent=int(os.getenv("MATHULA_TV_AZURE_RATE_MIN_PERCENT", "-12")),
        azure_rate_max_percent=int(os.getenv("MATHULA_TV_AZURE_RATE_MAX_PERCENT", "15")),
        azure_rate_search_attempts=int(os.getenv("MATHULA_TV_AZURE_RATE_SEARCH_ATTEMPTS", "3")),
        preferred_end_tolerance_ms=int(os.getenv("MATHULA_TV_PREFERRED_END_TOLERANCE_MS", "120")),
        post_conversion_min_speed=float(os.getenv("MATHULA_TV_POST_CONVERSION_MIN_SPEED", "0.94")),
        post_conversion_max_speed=float(os.getenv("MATHULA_TV_POST_CONVERSION_MAX_SPEED", "1.06")),
        final_timing_tolerance_ms=int(os.getenv("MATHULA_TV_FINAL_TIMING_TOLERANCE_MS", "120")),
        openvoice_revision=os.getenv("MATHULA_TV_OPENVOICE_REVISION", "").strip(),
        openvoice_checkpoint_hashes=_openvoice_checkpoint_hashes(),
        openvoice_max_attempts=int(os.getenv("MATHULA_TV_OPENVOICE_MAX_ATTEMPTS", "2")),
        target_lufs=float(os.getenv("MATHULA_TV_TARGET_LUFS", "-14")),
        true_peak_dbtp=float(os.getenv("MATHULA_TV_TRUE_PEAK_DBTP", "-1.5")),
        target_lra=float(os.getenv("MATHULA_TV_TARGET_LRA", "11")),
        max_source_duration_minutes=int(os.getenv("MATHULA_TV_MAX_SOURCE_DURATION_MINUTES", "240")),
        max_claude_calls_per_job=int(os.getenv("MATHULA_TV_MAX_AI_CALLS_PER_JOB", "20")),
        max_translation_repairs_per_unit=int(os.getenv("MATHULA_TV_MAX_TRANSLATION_REPAIRS_PER_UNIT", "2")),
        max_azure_tts_attempts_per_unit=int(os.getenv("MATHULA_TV_MAX_AZURE_TTS_ATTEMPTS_PER_UNIT", "3")),
        max_openvoice_attempts_per_unit=int(os.getenv("MATHULA_TV_MAX_OPENVOICE_ATTEMPTS_PER_UNIT", "2")),
        max_failed_units=int(os.getenv("MATHULA_TV_MAX_FAILED_UNITS", "2")),
        max_job_runtime_seconds=int(os.getenv("MATHULA_TV_MAX_JOB_RUNTIME_SECONDS", "14400")),
        max_artifact_download_bytes=int(os.getenv("MATHULA_TV_MAX_ARTIFACT_DOWNLOAD_BYTES", str(2 * 1024 * 1024 * 1024))),
        pyannote_enabled=os.getenv("MATHULA_TV_ENABLE_PYANNOTE_DIAGNOSTIC", "0") == "1",
        enable_sdk_group_synthesis=os.getenv("MATHULA_TV_ENABLE_SDK_GROUP_SYNTHESIS", "0") == "1",
        enable_turn_group_synthesis=os.getenv("MATHULA_TV_ENABLE_TURN_GROUP_SYNTHESIS", "0") == "1",
        youtube_cookies_file=os.getenv("MATHULA_TV_YOUTUBE_COOKIES_FILE", "").strip(),
        max_clean_unit_wer=float(os.getenv("MATHULA_TV_MAX_CLEAN_UNIT_WER", "0.35")),
        max_openvoice_wer_degradation=float(os.getenv("MATHULA_TV_MAX_OPENVOICE_WER_DEGRADATION", "0.10")),
        max_final_mix_wer_degradation=float(os.getenv("MATHULA_TV_MAX_FINAL_MIX_WER_DEGRADATION", "0.15")),
        min_protected_entity_similarity=float(os.getenv("MATHULA_TV_MIN_PROTECTED_ENTITY_SIMILARITY", "0.85")),
        gpt_model=(
            os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
            or os.getenv("AZURE_AI_DEPLOYMENT")
            or "gpt-5.6-sol-1"
        ),
        gpt_timeout_seconds=float(
            os.getenv("MATHULA_TV_GPT_TIMEOUT_SECONDS", "900")
        ),
        gpt_max_retries=int(
            os.getenv("MATHULA_TV_GPT_MAX_RETRIES", "3")
        ),
        gpt_effort=os.getenv("MATHULA_TV_GPT_EFFORT", "high").strip().lower(),
        translation_gpt_effort=os.getenv(
            "MATHULA_TV_TRANSLATION_GPT_EFFORT", "low"
        ).strip().lower(),
        seo_gpt_effort=os.getenv(
            "MATHULA_TV_SEO_GPT_EFFORT", "medium"
        ).strip().lower(),
        hook_gpt_effort=os.getenv(
            "MATHULA_TV_HOOK_GPT_EFFORT", "low"
        ).strip().lower(),
        editorial_gpt_effort=os.getenv(
            "MATHULA_TV_EDITORIAL_GPT_EFFORT", "high"
        ).strip().lower(),
        max_ai_calls_per_job=int(
            os.getenv("MATHULA_TV_MAX_AI_CALLS_PER_JOB", "20")
        ),
    )


def _openvoice_checkpoint_hashes() -> tuple[tuple[str, str], ...]:
    hashes_raw = os.getenv(
        "MATHULA_TV_OPENVOICE_CHECKPOINT_HASHES_JSON", ""
    ).strip()
    raw = hashes_raw or os.getenv(
        "MATHULA_TV_OPENVOICE_CHECKPOINTS_JSON", ""
    ).strip()
    if not raw:
        return ()
    try:
        values = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("OpenVoice checkpoint configuration must be valid JSON") from exc
    if hashes_raw and isinstance(values, dict):
        values = [
            {"name": name, "sha256": digest}
            for name, digest in values.items()
        ]
    if not isinstance(values, list) or not values:
        raise ValueError(
            "OpenVoice checkpoint hashes must be a non-empty object or list"
        )
    identities: dict[str, str] = {}
    for item in values:
        if not isinstance(item, dict):
            raise ValueError("Every OpenVoice checkpoint specification must be an object")
        name = str(item.get("name") or "")
        digest = str(item.get("sha256") or "")
        if (
            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", name)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or name in identities
        ):
            raise ValueError("OpenVoice checkpoint names and hashes must be unique and valid")
        identities[name] = digest
    return tuple(sorted(identities.items()))
