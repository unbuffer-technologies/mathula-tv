"""Speech-to-text and diarization from the Cloud Run service (gcp/stt_service), in place of Azure Fast Transcription.

``CloudSTTBackend`` is a ``SpeechBackend`` (``transcribe(audio, locale) -> raw``), and the raw it returns is Azure's Fast Transcription shape, so ``azure_stt.normalize`` and everything after it read it unchanged.
Selected with MATHULA_STT_PROVIDER=whisper (see cli.create_speech_backend); STT_URL is the service URL, reached with a Google ID token like the OmniVoice service.

* English only (any en-*): Whisper large-v3 with pyannote's speaker turns.  Any other locale is refused, so a stage that asks for one (the check of the dub's own Zulu speech) fails visibly instead of being given a wrong transcription;
* the audio is sent as 16 kHz mono Opus: a speech recogniser loses nothing to it and a long recording stays under Cloud Run's 32 MiB request limit;
* ``phrase_list`` (the names and terms the pipeline expects) is sent as the recogniser's spelling hint.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import requests

from .azure_stt import AzureSpeechError, validate_raw_response

TRANSIENT_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


def gcp_id_token(url: str) -> str:
    """A Google ID token for a private Cloud Run service (the service account in GOOGLE_APPLICATION_CREDENTIALS)."""
    import google.auth.transport.requests
    import google.oauth2.id_token

    return google.oauth2.id_token.fetch_id_token(google.auth.transport.requests.Request(), url)


def opus_16k_mono(source: Path, target: Path) -> Path:
    done = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(source), "-ac", "1", "-ar", "16000", "-c:a", "libopus", "-b:a", "32k", str(target)], capture_output=True, text=True)
    if done.returncode or not target.is_file():
        raise AzureSpeechError(f"ffmpeg could not prepare the audio: {done.stderr[-200:]}")
    return target


@dataclass
class CloudSTTBackend:
    url: str
    max_speakers: int = 10
    timeout_seconds: float = 1800.0                 # a cold start loads the models first; a long recording takes minutes
    retries: int = 3
    provider: str = "whisper-pyannote-cloud-run"
    api_version: str = "v1"
    request_duration_seconds: float | None = None
    phrase_list: tuple[str, ...] = ()
    token: Callable[[str], str] = field(default=gcp_id_token, repr=False)
    session: Any = None
    sleep: Callable[[float], None] = time.sleep

    def __post_init__(self) -> None:
        self.url = self.url.rstrip("/")
        self.session = self.session or requests.Session()

    @classmethod
    def from_environment(cls, env: Mapping[str, str] | None = None, **kwargs: Any) -> "CloudSTTBackend":
        import os

        env = env if env is not None else os.environ
        url = (env.get("STT_URL") or "").strip()
        if not url:
            raise ValueError("STT_URL is not set (the Cloud Run URL of mathula-stt)")
        return cls(url=url, **kwargs)

    def _options(self, locale: str) -> dict[str, Any]:
        if not locale.lower().startswith("en"):
            raise AzureSpeechError(f"The Whisper service transcribes English only (asked for {locale}); set MATHULA_STT_PROVIDER=azure for another language", retryable=False)
        options: dict[str, Any] = {"locale": locale, "diarize": True, "max_speakers": self.max_speakers}
        if self.phrase_list:
            options["phrases"] = [p for p in self.phrase_list if str(p).strip()]
        return options

    def transcribe(self, audio: Path, locale: str = "en-ZA") -> dict[str, Any]:
        audio = Path(audio)
        options = self._options(locale)
        if not audio.is_file() or not audio.stat().st_size:
            raise AzureSpeechError(f"Audio is missing or empty: {audio}")
        started = time.perf_counter()
        last: Exception | None = None
        with tempfile.TemporaryDirectory() as folder:
            upload = opus_16k_mono(audio, Path(folder) / "audio.ogg")
            for attempt in range(1, self.retries + 1):
                try:
                    with upload.open("rb") as handle:
                        response = self.session.post(f"{self.url}/v1/transcribe", headers={"Authorization": f"Bearer {self.token(self.url)}"},
                                                     files={"audio": (upload.name, handle, "audio/ogg")}, data={"params": json.dumps(options)}, timeout=(30, self.timeout_seconds))
                    if response.status_code >= 400:
                        retryable = response.status_code in TRANSIENT_STATUS
                        error = AzureSpeechError(f"STT service HTTP {response.status_code}: {response.text[:300]}", retryable=retryable, status_code=response.status_code)
                        if not retryable or attempt >= self.retries:
                            raise error
                        last = error
                        self.sleep(min(2 ** attempt * 5, 60))
                        continue
                    try:
                        raw = response.json()
                    except ValueError as exc:
                        raise AzureSpeechError("STT service returned malformed JSON") from exc
                    validate_raw_response(raw)
                    self.request_duration_seconds = time.perf_counter() - started
                    return raw
                except (requests.Timeout, requests.ConnectionError) as exc:
                    last = exc
                    if attempt >= self.retries:
                        raise AzureSpeechError(f"STT service {'timed out' if isinstance(exc, requests.Timeout) else 'connection failed'} after {self.retries} attempts", retryable=True) from exc
                    self.sleep(min(2 ** attempt * 5, 60))
        raise AzureSpeechError(f"STT request failed: {type(last).__name__}", retryable=True)
