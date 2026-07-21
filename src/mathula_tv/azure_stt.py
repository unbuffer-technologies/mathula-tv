from __future__ import annotations

import email.utils
import json
import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol
from urllib.parse import urlsplit, urlunsplit

import requests

from .media import probe

FAST_MAX_BYTES = 500 * 1024 * 1024
PRODUCTION_MAX_DURATION_SECONDS = 240 * 60
FAST_MAX_DURATION_SECONDS = PRODUCTION_MAX_DURATION_SECONDS
TRANSIENT_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class SpeechBackend(Protocol):
    provider: str
    api_version: str
    request_duration_seconds: float | None

    def transcribe(self, audio: Path, locale: str) -> dict[str, Any]: ...


class BatchHTTPBoundary(Protocol):
    def submit(self, audio: Path, definition: dict[str, Any], timeout_seconds: float) -> str: ...

    def wait(self, operation_id: str, timeout_seconds: float) -> dict[str, Any]: ...


class AzureSpeechError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool = False, status_code: int | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code


def derive_endpoint(region: str) -> str:
    region = region.strip().lower()
    if not region or any(not (part.isalnum() or part == "-") for part in region):
        raise ValueError("AZURE_SPEECH_REGION is missing or invalid")
    return f"https://{region}.api.cognitive.microsoft.com"


def validate_endpoint(endpoint: str) -> str:
    candidate = endpoint.strip().rstrip("/")
    parsed = urlsplit(candidate)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("AZURE_SPEECH_ENDPOINT must be an HTTPS origin without credentials, query, or fragment")
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))


def safe_endpoint(endpoint: str) -> str:
    parsed = urlsplit(endpoint)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def _request_definition_without_phrases(locale: str, max_speakers: int) -> dict[str, Any]:
    if not locale or len(locale) > 20:
        raise ValueError("AZURE_SPEECH_LOCALE is invalid")
    if not 1 <= max_speakers <= 35:
        raise ValueError("AZURE_SPEECH_MAX_SPEAKERS must be between 1 and 35")
    # Fast Transcription returns word timestamps by default. Its current request
    # schema has no wordLevelTimestampsEnabled property (that is a batch option).
    return {"locales": [locale], "diarization": {"enabled": True, "maxSpeakers": max_speakers}}

def request_definition(
    locale: str = "en-ZA",
    *args: Any,
    phrases: list[str] | tuple[str, ...] | None = None,
    biasing_weight: float | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build the Fast transcription definition with optional entity phrases."""

    definition = _request_definition_without_phrases(locale, *args, **kwargs)
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in phrases or ():
        phrase = " ".join(str(value).split()).strip()
        key = phrase.casefold()
        if phrase and key not in seen:
            seen.add(key)
            cleaned.append(phrase)
    if cleaned:
        phrase_list: dict[str, Any] = {"phrases": cleaned}
        if biasing_weight is not None:
            if not 0.0 <= float(biasing_weight) <= 2.0:
                raise ValueError("Azure phrase-list biasing weight must be between 0.0 and 2.0")
            phrase_list["biasingWeight"] = float(biasing_weight)
        definition["phraseList"] = phrase_list
    return definition



def _error_message(response: requests.Response) -> str:
    try:
        body = response.json()
        detail = body.get("error", body) if isinstance(body, dict) else body
        if isinstance(detail, dict):
            detail = detail.get("message") or detail.get("code") or "request rejected"
        text = str(detail)
    except (ValueError, TypeError):
        text = (response.text or "request rejected").strip()
    text = " ".join(text.split())[:300]
    return text or "request rejected"


def _retry_after(value: str | None, now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            parsed = email.utils.parsedate_to_datetime(value)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return max(0.0, (parsed - now()).total_seconds())
        except (TypeError, ValueError):
            return None


@dataclass
class AzureFastTranscriptionBackend:
    endpoint: str
    subscription_key: str
    api_version: str = "2025-10-15"
    max_speakers: int = 10
    timeout_seconds: float = 600
    connect_timeout_seconds: float = 15
    retries: int = 3
    session: Any = None
    sleep: Callable[[float], None] = time.sleep
    provider: str = "azure-speech-fast-transcription"
    request_duration_seconds: float | None = None

    phrase_list: tuple[str, ...] = ()
    phrase_biasing_weight: float | None = None

    def __post_init__(self) -> None:
        self.endpoint = validate_endpoint(self.endpoint)
        if not self.subscription_key:
            raise ValueError("AZURE_SPEECH_KEY is missing")
        if self.api_version != "2025-10-15":
            raise ValueError("AZURE_SPEECH_API_VERSION must be 2025-10-15")
        if self.timeout_seconds <= 0 or self.connect_timeout_seconds <= 0:
            raise ValueError("Azure Speech timeouts must be positive")
        request_definition("en-ZA", self.max_speakers)
        self.session = self.session or requests.Session()

    @property
    def url(self) -> str:
        return f"{self.endpoint}/speechtotext/transcriptions:transcribe?api-version={self.api_version}"

    def transcribe(self, audio: Path, locale: str = "en-ZA") -> dict[str, Any]:
        if not audio.is_file() or not audio.stat().st_size:
            raise AzureSpeechError(f"Audio is missing or empty: {audio}")
        try:
            with audio.open("rb"):
                pass
        except OSError as exc:
            raise AzureSpeechError(f"Audio is not readable: {audio}") from exc
        size = audio.stat().st_size
        if size >= FAST_MAX_BYTES:
            raise AzureSpeechError("Audio is too large for Fast Transcription; a future asynchronous batch backend is required")
        try:
            duration = float(probe(audio)["duration"])
        except Exception as exc:
            raise AzureSpeechError(f"Invalid WAV/audio input: {type(exc).__name__}") from exc
        if not math.isfinite(duration) or duration <= 0:
            raise AzureSpeechError("Audio has invalid or zero duration")
        if duration >= FAST_MAX_DURATION_SECONDS:
            raise AzureSpeechError("Audio exceeds the reviewed 240-minute production limit; chunked speaker stitching is not implemented")

        definition = request_definition(locale, self.max_speakers, phrases=self.phrase_list, biasing_weight=self.phrase_biasing_weight)
        started = time.perf_counter()
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                with audio.open("rb") as handle:
                    response = self.session.post(
                        self.url,
                        headers={"Ocp-Apim-Subscription-Key": self.subscription_key},
                        files={"audio": (audio.name, handle, "audio/wav")},
                        data={"definition": json.dumps(definition, separators=(",", ":"))},
                        timeout=(self.connect_timeout_seconds, self.timeout_seconds),
                    )
                if response.status_code >= 400:
                    retryable = response.status_code in TRANSIENT_STATUS
                    message = _error_message(response)
                    if response.status_code in {401, 403}:
                        message = "Azure Speech authentication/authorization failed"
                    error = AzureSpeechError(f"Azure Speech HTTP {response.status_code}: {message}", retryable=retryable, status_code=response.status_code)
                    if not retryable or attempt >= self.retries:
                        raise error
                    delay = _retry_after(response.headers.get("Retry-After")) if response.status_code == 429 else None
                    self.sleep(delay if delay is not None else min(2 ** (attempt - 1), 30))
                    last_error = error
                    continue
                try:
                    raw = response.json()
                except ValueError as exc:
                    raise AzureSpeechError("Azure Speech returned malformed JSON") from exc
                validate_raw_response(raw)
                self.request_duration_seconds = time.perf_counter() - started
                logging.info("Azure Fast Transcription completed in %.2fs via %s", self.request_duration_seconds, safe_endpoint(self.endpoint))
                return raw
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_error = exc
                if attempt >= self.retries:
                    kind = "timed out" if isinstance(exc, requests.Timeout) else "connection failed"
                    raise AzureSpeechError(f"Azure Speech {kind} after {self.retries} attempts", retryable=True) from exc
                self.sleep(min(2 ** (attempt - 1), 30))
            except AzureSpeechError:
                raise
        raise AzureSpeechError(f"Azure Speech request failed: {type(last_error).__name__}", retryable=True)


@dataclass
class AzureBatchTranscriptionBackend:
    """Injectable long-file boundary; no public or signed URL is persisted."""

    boundary: BatchHTTPBoundary
    api_version: str = "2025-10-15"
    max_speakers: int = 10
    timeout_seconds: float = 7200
    provider: str = "azure-speech-batch-transcription"
    request_duration_seconds: float | None = None

    def transcribe(self, audio: Path, locale: str = "en-ZA") -> dict[str, Any]:
        info = probe(audio)
        duration = float(info["duration"])
        if duration <= 0 or duration > PRODUCTION_MAX_DURATION_SECONDS:
            raise AzureSpeechError("Audio exceeds the reviewed 240-minute production limit")
        definition = {
            "locale": locale,
            "diarization": {"enabled": True, "maxSpeakers": self.max_speakers},
            "wordLevelTimestampsEnabled": True,
            "punctuationMode": "DictatedAndAutomatic",
        }
        started = time.perf_counter()
        try:
            operation_id = self.boundary.submit(audio, definition, self.timeout_seconds)
            if not operation_id or any(term in operation_id.lower() for term in ("sig=", "token=", "key=")):
                raise AzureSpeechError("Azure Batch returned an invalid operation identifier")
            raw = self.boundary.wait(operation_id, self.timeout_seconds)
            validate_raw_response(raw)
        except AzureSpeechError:
            raise
        except TimeoutError as exc:
            raise AzureSpeechError("Azure Batch transcription timed out", retryable=True) from exc
        except Exception as exc:
            raise AzureSpeechError(
                f"Azure Batch transcription failed: {type(exc).__name__}", retryable=True
            ) from exc
        self.request_duration_seconds = time.perf_counter() - started
        return raw


@dataclass
class AzureSTTRouter:
    """Select Fast where eligible and an injected Batch backend otherwise."""

    fast: SpeechBackend
    batch: SpeechBackend | None = None
    provider: str = "azure-speech-router"
    api_version: str = "2025-10-15"
    request_duration_seconds: float | None = None

    def transcribe(self, audio: Path, locale: str = "en-ZA") -> dict[str, Any]:
        duration = float(probe(audio)["duration"])
        if duration > PRODUCTION_MAX_DURATION_SECONDS:
            raise AzureSpeechError("Audio exceeds the reviewed 240-minute production limit")
        backend = self.fast if audio.stat().st_size < FAST_MAX_BYTES and duration < FAST_MAX_DURATION_SECONDS else self.batch
        if backend is self.batch and getattr(self.fast, "phrase_list", ()):
            raise AzureSpeechError(
                "Entity phrase lists require Azure Fast transcription; "
                "the source is outside the Fast API size or duration limit"
            )
        if backend is None:
            raise AzureSpeechError("Audio requires Azure Batch transcription, but no reviewed batch boundary is configured")
        raw = backend.transcribe(audio, locale)
        self.request_duration_seconds = backend.request_duration_seconds
        self.provider = backend.provider
        self.api_version = backend.api_version
        return raw


def validate_raw_response(raw: Any) -> None:
    if not isinstance(raw, dict) or not isinstance(raw.get("phrases"), list):
        raise AzureSpeechError("Azure Speech response is missing phrases")
    combined = raw.get("combinedPhrases", [])
    text = " ".join(str(item.get("text", "")) for item in combined if isinstance(item, dict)).strip()
    if not text:
        text = " ".join(str(item.get("text", "")) for item in raw["phrases"] if isinstance(item, dict)).strip()
    if not text:
        raise AzureSpeechError("Azure Speech returned an empty transcription")


def _seconds(value: Any) -> float:
    return float(value or 0) / 1000.0


def normalize(
    raw: dict[str, Any],
    *,
    raw_reference: str = "analysis/azure_stt.json",
    provider: str = "azure-speech-fast-transcription",
) -> dict[str, Any]:
    validate_raw_response(raw)
    words: list[dict[str, Any]] = []
    phrases: list[dict[str, Any]] = []
    turns: list[dict[str, Any]] = []
    for index, phrase in enumerate(raw["phrases"]):
        phrase_id = f"phrase_{index + 1:05d}"
        start = _seconds(phrase.get("offsetMilliseconds"))
        duration = _seconds(phrase.get("durationMilliseconds"))
        speaker_number = phrase.get("speaker")
        speaker = f"AZURE_{speaker_number}" if speaker_number is not None else "UNKNOWN"
        normalized_phrase = {
            "phrase_id": phrase_id,
            "text": phrase.get("text", ""),
            "start": start,
            "end": start + duration,
            "duration": duration,
            "locale": phrase.get("locale"),
            "confidence": phrase.get("confidence"),
            "azure_speaker": speaker_number,
            "speaker": speaker,
            "channel": phrase.get("channel"),
        }
        phrases.append(normalized_phrase)
        for word in phrase.get("words") or []:
            word_start = _seconds(word.get("offsetMilliseconds"))
            word_duration = _seconds(word.get("durationMilliseconds"))
            words.append({
                "word_id": f"word_{len(words) + 1:06d}",
                "source_phrase_id": phrase_id,
                "text": word.get("text", ""),
                "start": word_start,
                "end": word_start + word_duration,
                "duration": word_duration,
                "confidence": word.get("confidence", phrase.get("confidence")),
                "locale": phrase.get("locale"),
                "azure_speaker": speaker_number,
                "speaker": speaker,
            })
        if duration > 0:
            turns.append({"phrase_id": phrase_id, "speaker": speaker, "azure_speaker": speaker_number, "start": start, "end": start + duration, "duration": duration, "overlap": False, "confidence": phrase.get("confidence"), "source": "azure"})
    return {
        "schema_version": "azure-stt-normalized-v1",
        "provider": provider,
        "raw_response_reference": raw_reference,
        "duration": _seconds(raw.get("durationMilliseconds")),
        "complete_text": "\n".join(str(p.get("text", "")) for p in raw.get("combinedPhrases", [])).strip(),
        "phrases": phrases,
        "words": words,
        "turns": turns,
        "warnings": ["Azure diarization is authoritative; speaker identifiers are job-local"],
    }


class AzureSpeechTranscriber:
    def __init__(self, backend: SpeechBackend):
        self.backend = backend

    def transcribe(self, audio: Path, locale: str = "en-ZA", *, raw_reference: str = "analysis/azure_stt.json") -> tuple[dict, dict]:
        raw = self.backend.transcribe(audio, locale)
        return raw, normalize(
            raw,
            raw_reference=raw_reference,
            provider=self.backend.provider,
        )
