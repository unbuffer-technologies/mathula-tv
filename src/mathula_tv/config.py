from __future__ import annotations

import json
import os
import re
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
    ai_provider: str = "anthropic"
    claude_model: str = "claude-opus-4-8"
    claude_timeout_seconds: float = 900
    claude_max_retries: int = 3
    claude_effort: str = "high"
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

    def safe_snapshot(self) -> dict:
        return {k: str(v) if isinstance(v, Path) else v for k, v in asdict(self).items()}


def load_settings(project_root: Path | None = None) -> Settings:
    root = project_root or Path(__file__).resolve().parents[2]
    if load_dotenv:
        load_dotenv(root / ".env", override=False)
    work = Path(os.getenv("MATHULA_TV_WORK_DIR", str(root / "working")))
    voices = tuple(
        item.strip()
        for item in os.getenv("MATHULA_TV_AZURE_TTS_VOICES", "zu-ZA-ThandoNeural,zu-ZA-ThembaNeural").split(",")
        if item.strip()
    )
    return Settings(
        work_dir=work,
        gcs_bucket=os.getenv("MATHULA_TV_GCS_BUCKET", ""),
        gcs_prefix=os.getenv("MATHULA_TV_GCS_PREFIX", "mathula-tv").strip("/"),
        commission_master_case_path=Path(os.getenv("COMMISSION_MASTER_CASE_PATH", "/home/mokgethwa/commission-ai/working/case_state/master_case_state.json")),
        politics_context_path=Path(os.getenv("POLITICS_CONTEXT_PATH", str(work / "context/politics_context.json"))),
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
        ai_provider=os.getenv("MATHULA_TV_AI_PROVIDER", "anthropic").strip().lower(),
        claude_model=os.getenv("MATHULA_TV_CLAUDE_MODEL", "claude-opus-4-8"),
        claude_timeout_seconds=float(os.getenv("MATHULA_TV_CLAUDE_TIMEOUT_SECONDS", "900")),
        claude_max_retries=int(os.getenv("MATHULA_TV_CLAUDE_MAX_RETRIES", "3")),
        claude_effort=os.getenv("MATHULA_TV_CLAUDE_EFFORT", "high"),
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
        max_claude_calls_per_job=int(os.getenv("MATHULA_TV_MAX_CLAUDE_CALLS_PER_JOB", "20")),
        max_translation_repairs_per_unit=int(os.getenv("MATHULA_TV_MAX_TRANSLATION_REPAIRS_PER_UNIT", "2")),
        max_azure_tts_attempts_per_unit=int(os.getenv("MATHULA_TV_MAX_AZURE_TTS_ATTEMPTS_PER_UNIT", "3")),
        max_openvoice_attempts_per_unit=int(os.getenv("MATHULA_TV_MAX_OPENVOICE_ATTEMPTS_PER_UNIT", "2")),
        max_failed_units=int(os.getenv("MATHULA_TV_MAX_FAILED_UNITS", "2")),
        max_job_runtime_seconds=int(os.getenv("MATHULA_TV_MAX_JOB_RUNTIME_SECONDS", "14400")),
        max_artifact_download_bytes=int(os.getenv("MATHULA_TV_MAX_ARTIFACT_DOWNLOAD_BYTES", str(2 * 1024 * 1024 * 1024))),
        pyannote_enabled=os.getenv("MATHULA_TV_ENABLE_PYANNOTE_DIAGNOSTIC", "0") == "1",
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
