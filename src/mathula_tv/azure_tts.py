from __future__ import annotations

import email.utils
import hashlib
import io
import json
import math
import os
import re
import struct
import tempfile
import time
import wave
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence, TypeVar

import requests

from .atomic_io import read_json
from .errors import PipelineError
from .logging_utils import redact
from .tts_ssml import (
    SSMLBounds,
    SSMLDocument,
    SSMLPart,
    SSMLValidationError,
    SafeSSMLBuilder,
    validate_document,
)


TRANSIENT_HTTP_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
TRANSIENT_CANCELLATION_CODES = {
    "connectionfailure",
    "servicetimeout",
    "servicenotavailable",
    "tooManyrequests".lower(),
    "runtimeerror",
}
OUTPUT_FORMATS = {
    (8_000, 1, 2): "riff-8khz-16bit-mono-pcm",
    (16_000, 1, 2): "riff-16khz-16bit-mono-pcm",
    (24_000, 1, 2): "riff-24khz-16bit-mono-pcm",
    (48_000, 1, 2): "riff-48khz-16bit-mono-pcm",
}
AZURE_REST_MAX_AUDIO_DURATION_MS = 10 * 60 * 1000
_REGION = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z")
_IDENTIFIER = re.compile(r"[^\x00-\x1f\x7f]{1,200}\Z")
_T = TypeVar("_T")


class AzureTTSError(PipelineError):
    def __init__(
        self,
        message: str,
        *,
        category: str,
        retryable: bool = False,
        status_code: int | None = None,
        retry_after_seconds: float | None = None,
        cancellation_details: Mapping[str, Any] | None = None,
    ):
        self.category = category
        self.retryable = retryable
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds
        self.cancellation_details = dict(cancellation_details or {})
        super().__init__(
            message,
            details={
                "category": category,
                "status_code": status_code,
                "retry_after_seconds": retry_after_seconds,
                "cancellation_details": self.cancellation_details,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": f"azure_tts_{self.category}",
            "stage": "azure_tts",
            "retryable": self.retryable,
            "message": self.safe_message,
            "action": "Retry if transient; otherwise review Azure TTS configuration and the affected unit",
            "details": self.details,
        }


class AzureTTSIdempotencyError(AzureTTSError):
    def __init__(self, message: str, *, conflict_kind: str = "artifact_integrity"):
        self.conflict_kind = conflict_kind
        super().__init__(message, category="idempotency_conflict", retryable=False)


class AzureTTSTransportError(RuntimeError):
    def __init__(self, message: str, *, category: str = "connection", retryable: bool = True):
        super().__init__(message)
        self.category = category
        self.retryable = retryable


@dataclass(frozen=True)
class AzureCancellationDetails:
    reason: str
    error_code: str | None = None
    details: str | None = None


@dataclass(frozen=True)
class AzureTTSResponse:
    status_code: int
    body: bytes = b""
    headers: Mapping[str, str] = field(default_factory=dict)
    cancellation: AzureCancellationDetails | None = None


class AzureTTSBoundary(Protocol):
    def get_voices(self, locale: str, *, timeout_seconds: float) -> AzureTTSResponse: ...

    def synthesize_ssml(
        self, ssml: str, *, output_format: str, timeout_seconds: float
    ) -> AzureTTSResponse: ...


@dataclass
class AzureTTSHTTPBoundary:
    """Thin production REST boundary; business behavior lives in AzureTTSBackend."""

    region: str
    subscription_key: str = field(repr=False)
    session: Any = field(default=None, repr=False)
    connect_timeout_seconds: float = 15
    user_agent: str = "mathula-tv"

    def __post_init__(self) -> None:
        self.region = self.region.strip().lower()
        if not _REGION.fullmatch(self.region):
            raise ValueError("AZURE_SPEECH_REGION is missing or invalid")
        if not self.subscription_key:
            raise ValueError("AZURE_SPEECH_KEY is missing")
        if self.connect_timeout_seconds <= 0:
            raise ValueError("Azure TTS connect timeout must be positive")
        if not self.user_agent or len(self.user_agent) >= 255 or "\n" in self.user_agent or "\r" in self.user_agent:
            raise ValueError("Azure TTS user agent is invalid")
        self.session = self.session or requests.Session()

    @property
    def origin(self) -> str:
        return f"https://{self.region}.tts.speech.microsoft.com"

    def get_voices(self, locale: str, *, timeout_seconds: float) -> AzureTTSResponse:
        try:
            response = self.session.get(
                f"{self.origin}/cognitiveservices/voices/list",
                headers={"Ocp-Apim-Subscription-Key": self.subscription_key},
                timeout=(self.connect_timeout_seconds, timeout_seconds),
            )
        except requests.Timeout as exc:
            raise AzureTTSTransportError("Azure TTS voice discovery timed out", category="timeout") from exc
        except requests.ConnectionError as exc:
            raise AzureTTSTransportError("Azure TTS voice discovery connection failed") from exc
        return _http_response(response, secret=self.subscription_key)

    def synthesize_ssml(
        self, ssml: str, *, output_format: str, timeout_seconds: float
    ) -> AzureTTSResponse:
        try:
            response = self.session.post(
                f"{self.origin}/cognitiveservices/v1",
                headers={
                    "Ocp-Apim-Subscription-Key": self.subscription_key,
                    "Content-Type": "application/ssml+xml",
                    "X-Microsoft-OutputFormat": output_format,
                    "User-Agent": self.user_agent,
                },
                data=ssml.encode("utf-8"),
                timeout=(self.connect_timeout_seconds, timeout_seconds),
            )
        except requests.Timeout as exc:
            raise AzureTTSTransportError("Azure TTS synthesis timed out", category="timeout") from exc
        except requests.ConnectionError as exc:
            raise AzureTTSTransportError("Azure TTS synthesis connection failed") from exc
        return _http_response(response, secret=self.subscription_key)


def _http_response(response: Any, *, secret: str) -> AzureTTSResponse:
    headers = {
        str(key): str(value)
        for key, value in response.headers.items()
        if str(key).lower() in {"retry-after", "content-type", "x-requestid", "x-microsoft-requestid"}
    }
    body = bytes(response.content)
    if int(response.status_code) >= 400 and secret:
        body = body.replace(secret.encode("utf-8"), b"[REDACTED]")
    return AzureTTSResponse(status_code=int(response.status_code), body=body, headers=headers)


@dataclass(frozen=True)
class AzureVoice:
    name: str
    locale: str
    display_name: str | None = None
    local_name: str | None = None
    gender: str | None = None
    voice_type: str | None = None
    status: str | None = None


@dataclass(frozen=True)
class AzureTTSRequest:
    turn_id: str
    speaker_id: str
    text: str
    preferred_duration_ms: int
    maximum_duration_ms: int
    voice: str | None = None
    language: str = "zu-ZA"
    rate_percent: int = 0
    pitch_percent: int = 0
    volume_percent: int = 0
    parts: tuple[SSMLPart, ...] = ()
    ssml_document: SSMLDocument | None = None

    def with_rate(self, rate_percent: int) -> "AzureTTSRequest":
        if self.ssml_document is not None:
            raise ValueError("A prebuilt SSML request cannot be assigned a different rate")
        return replace(self, rate_percent=rate_percent)


@dataclass(frozen=True)
class AzureTTSResult:
    backend: str
    turn_id: str
    speaker_id: str
    voice: str
    language: str
    input_text_hash: str
    ssml_hash: str
    output_path: str
    sample_rate: int
    channels: int
    sample_width: int
    duration_ms: int
    preferred_duration_ms: int
    maximum_duration_ms: int
    rate_percent: int
    attempt: int
    sha256: str
    warnings: tuple[str, ...]
    request_hash: str
    ssml_path: str
    manifest_path: str
    ssml_summary: str
    trimmed_leading_ms: int = 0
    trimmed_trailing_ms: int = 0
    idempotent_reuse: bool = False

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["warnings"] = list(self.warnings)
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, idempotent_reuse: bool = False) -> "AzureTTSResult":
        fields = {
            "backend",
            "turn_id",
            "speaker_id",
            "voice",
            "language",
            "input_text_hash",
            "ssml_hash",
            "output_path",
            "sample_rate",
            "channels",
            "sample_width",
            "duration_ms",
            "preferred_duration_ms",
            "maximum_duration_ms",
            "rate_percent",
            "attempt",
            "sha256",
            "warnings",
            "request_hash",
            "ssml_path",
            "manifest_path",
            "ssml_summary",
        }
        if not fields.issubset(value):
            raise AzureTTSIdempotencyError("Azure TTS manifest is incomplete")
        return cls(
            **{name: value[name] for name in fields if name != "warnings"},
            warnings=tuple(str(item) for item in value["warnings"]),
            trimmed_leading_ms=int(value.get("trimmed_leading_ms", 0)),
            trimmed_trailing_ms=int(value.get("trimmed_trailing_ms", 0)),
            idempotent_reuse=idempotent_reuse,
        )


class AzureTTSBackend:
    provider = "azure_tts"
    manifest_schema_version = "azure-tts-result-v1"

    def __init__(
        self,
        boundary: AzureTTSBoundary,
        *,
        configured_voices: Sequence[str],
        default_voice: str,
        sample_rate: int = 24_000,
        channels: int = 1,
        sample_width: int = 2,
        timeout_seconds: float = 120,
        max_retries: int = 3,
        ssml_bounds: SSMLBounds | None = None,
        verified_emphasis_voices: Sequence[str] = (),
        trim_digital_silence_ms: int = 2000,
        maximum_response_bytes: int = 100 * 1024 * 1024,
        sleep: Callable[[float], None] = time.sleep,
    ):
        voices = tuple(dict.fromkeys(str(voice).strip() for voice in configured_voices if str(voice).strip()))
        if not voices:
            raise ValueError("MATHULA_TV_AZURE_TTS_VOICES must contain at least one voice")
        if default_voice not in voices:
            raise ValueError("The Azure TTS default voice must be in the configured voice allowlist")
        if (sample_rate, channels, sample_width) not in OUTPUT_FORMATS:
            raise ValueError("Azure TTS output must use a supported mono PCM16 WAV format")
        if timeout_seconds <= 0:
            raise ValueError("Azure TTS timeout must be positive")
        if isinstance(max_retries, bool) or not 0 <= max_retries <= 10:
            raise ValueError("Azure TTS max retries must be between 0 and 10")
        if trim_digital_silence_ms < 0:
            raise ValueError("Azure TTS digital-silence trim bound cannot be negative")
        if maximum_response_bytes <= 44:
            raise ValueError("Azure TTS maximum response size is too small")
        self.boundary = boundary
        self.configured_voices = voices
        self.default_voice = default_voice
        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_width = sample_width
        self.output_format = OUTPUT_FORMATS[(sample_rate, channels, sample_width)]
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.ssml_bounds = ssml_bounds or SSMLBounds()
        self.verified_emphasis_voices = tuple(dict.fromkeys(verified_emphasis_voices))
        self.trim_digital_silence_ms = trim_digital_silence_ms
        self.maximum_response_bytes = maximum_response_bytes
        self.sleep = sleep
        self.ssml_builder = SafeSSMLBuilder(
            allowed_voices=voices,
            bounds=self.ssml_bounds,
            verified_emphasis_voices=self.verified_emphasis_voices,
        )
        self._voice_cache: dict[str, tuple[AzureVoice, ...]] = {}

    def discover_voices(self, language: str = "zu-ZA", *, force_refresh: bool = False) -> tuple[AzureVoice, ...]:
        cache_key = language.casefold()
        if not force_refresh and cache_key in self._voice_cache:
            return self._voice_cache[cache_key]

        def parse(response: AzureTTSResponse) -> tuple[AzureVoice, ...]:
            self._raise_for_response(response, operation="voice discovery")
            if len(response.body) > 5 * 1024 * 1024:
                raise AzureTTSError(
                    "Azure TTS voice discovery response is too large",
                    category="malformed_response",
                    retryable=False,
                )
            try:
                raw = json.loads(response.body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise AzureTTSError(
                    "Azure TTS voice discovery returned malformed JSON",
                    category="malformed_response",
                    retryable=True,
                ) from exc
            if not isinstance(raw, list):
                raise AzureTTSError(
                    "Azure TTS voice discovery response is not a list",
                    category="malformed_response",
                    retryable=True,
                )
            found: dict[str, AzureVoice] = {}
            for item in raw:
                if not isinstance(item, dict):
                    continue
                name = item.get("ShortName")
                locale = item.get("Locale")
                if not isinstance(name, str) or not name or not isinstance(locale, str) or not locale:
                    continue
                if locale.casefold() != cache_key:
                    continue
                found[name] = AzureVoice(
                    name=name,
                    locale=locale,
                    display_name=_optional_text(item.get("DisplayName")),
                    local_name=_optional_text(item.get("LocalName")),
                    gender=_optional_text(item.get("Gender")),
                    voice_type=_optional_text(item.get("VoiceType")),
                    status=_optional_text(item.get("Status")),
                )
            return tuple(found[name] for name in sorted(found))

        voices, _ = self._with_retries(
            lambda: self.boundary.get_voices(language, timeout_seconds=self.timeout_seconds),
            parse,
        )
        self._voice_cache[cache_key] = voices
        return voices

    def validate_configured_voices(
        self, language: str = "zu-ZA", *, force_refresh: bool = False
    ) -> tuple[AzureVoice, ...]:
        discovered = self.discover_voices(language, force_refresh=force_refresh)
        available = {voice.name for voice in discovered}
        missing = [voice for voice in self.configured_voices if voice not in available]
        if missing:
            raise AzureTTSError(
                "Configured Azure TTS voice is unavailable: " + ", ".join(missing),
                category="voice_unavailable",
                retryable=False,
            )
        return tuple(voice for voice in discovered if voice.name in self.configured_voices)

    def validate_voice(self, voice: str, language: str = "zu-ZA") -> AzureVoice:
        if voice not in self.configured_voices:
            raise AzureTTSError(
                f"Azure TTS voice is not configured: {voice}",
                category="voice_unavailable",
                retryable=False,
            )
        for discovered in self.discover_voices(language):
            if discovered.name == voice:
                return discovered
        raise AzureTTSError(
            f"Azure TTS voice is unavailable for {language}: {voice}",
            category="voice_unavailable",
            retryable=False,
        )

    def synthesize(self, request: AzureTTSRequest, output_path: Path, *, force: bool = False) -> AzureTTSResult:
        output_path = Path(output_path)
        self._validate_request(request, output_path)
        voice = request.voice or self.default_voice
        document = self._ssml_document(request, voice)
        input_text_hash = hashlib.sha256(request.text.encode("utf-8")).hexdigest()
        ssml_path, manifest_path = self._artifact_paths(output_path)
        request_hash = self._request_hash(request, voice, document, input_text_hash)

        if not force:
            existing = self._reuse_existing(
                output_path,
                ssml_path,
                manifest_path,
                request=request,
                voice=voice,
                input_text_hash=input_text_hash,
                document=document,
                request_hash=request_hash,
            )
            if existing is not None:
                return existing

        self.validate_voice(voice, request.language)

        def parse(response: AzureTTSResponse) -> tuple[bytes, dict[str, int]]:
            self._raise_for_response(response, operation="synthesis")
            if len(response.body) > self.maximum_response_bytes:
                raise AzureTTSError(
                    "Azure TTS audio response exceeds the configured size limit",
                    category="invalid_output",
                    retryable=False,
                )
            try:
                return canonicalize_pcm_wav(
                    response.body,
                    sample_rate=self.sample_rate,
                    channels=self.channels,
                    sample_width=self.sample_width,
                    trim_digital_silence_ms=self.trim_digital_silence_ms,
                )
            except ValueError as exc:
                raise AzureTTSError(
                    f"Azure TTS returned invalid WAV audio: {exc}",
                    category="invalid_output",
                    retryable=True,
                ) from exc

        (audio, metadata), attempt = self._with_retries(
            lambda: self.boundary.synthesize_ssml(
                document.xml,
                output_format=self.output_format,
                timeout_seconds=self.timeout_seconds,
            ),
            parse,
        )
        output_hash = hashlib.sha256(audio).hexdigest()
        warnings: list[str] = []
        if metadata["trimmed_leading_ms"]:
            warnings.append(f"trimmed_leading_digital_silence_ms={metadata['trimmed_leading_ms']}")
        if metadata["trimmed_trailing_ms"]:
            warnings.append(f"trimmed_trailing_digital_silence_ms={metadata['trimmed_trailing_ms']}")
        if metadata["duration_ms"] > request.maximum_duration_ms:
            warnings.append("exceeds_maximum_duration_ms")
        elif metadata["duration_ms"] > request.preferred_duration_ms:
            warnings.append("exceeds_preferred_duration_ms")
        result = AzureTTSResult(
            backend=self.provider,
            turn_id=request.turn_id,
            speaker_id=request.speaker_id,
            voice=voice,
            language=request.language,
            input_text_hash=input_text_hash,
            ssml_hash=document.sha256,
            output_path=str(output_path),
            sample_rate=self.sample_rate,
            channels=self.channels,
            sample_width=self.sample_width,
            duration_ms=metadata["duration_ms"],
            preferred_duration_ms=request.preferred_duration_ms,
            maximum_duration_ms=request.maximum_duration_ms,
            rate_percent=request.rate_percent,
            attempt=attempt,
            sha256=output_hash,
            warnings=tuple(warnings),
            request_hash=request_hash,
            ssml_path=str(ssml_path),
            manifest_path=str(manifest_path),
            ssml_summary=document.summary,
            trimmed_leading_ms=metadata["trimmed_leading_ms"],
            trimmed_trailing_ms=metadata["trimmed_trailing_ms"],
        )
        manifest = {
            "schema_version": self.manifest_schema_version,
            "request_hash": request_hash,
            "result": result.to_dict(),
        }
        # The manifest is promoted last and therefore acts as the completeness marker.
        _atomic_write_bytes(output_path, audio)
        _atomic_write_bytes(ssml_path, (document.xml + "\n").encode("utf-8"))
        _atomic_write_bytes(
            manifest_path,
            (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
        return result

    def _ssml_document(self, request: AzureTTSRequest, voice: str) -> SSMLDocument:
        if request.ssml_document is not None:
            if request.parts:
                raise SSMLValidationError("Azure TTS request cannot contain both typed parts and a prebuilt SSML document")
            document = request.ssml_document
            validate_document(
                document,
                bounds=self.ssml_bounds,
                allowed_voices=self.configured_voices,
                verified_emphasis_voices=self.verified_emphasis_voices,
            )
            expected = (
                voice,
                request.language,
                request.rate_percent,
                request.pitch_percent,
                request.volume_percent,
            )
            actual = (
                document.voice,
                document.language,
                document.rate_percent,
                document.pitch_percent,
                document.volume_percent,
            )
            if actual != expected:
                raise SSMLValidationError("Prebuilt SSML delivery settings do not match the Azure TTS request")
            return document
        if request.parts:
            return self.ssml_builder.build(
                request.parts,
                voice=voice,
                language=request.language,
                rate_percent=request.rate_percent,
                pitch_percent=request.pitch_percent,
                volume_percent=request.volume_percent,
            )
        return self.ssml_builder.build_plain_text(
            request.text,
            voice=voice,
            language=request.language,
            rate_percent=request.rate_percent,
            pitch_percent=request.pitch_percent,
            volume_percent=request.volume_percent,
        )

    def _validate_request(self, request: AzureTTSRequest, output_path: Path) -> None:
        if not isinstance(request, AzureTTSRequest):
            raise TypeError("Azure TTS request has an invalid type")
        for name, value in (("turn_id", request.turn_id), ("speaker_id", request.speaker_id)):
            if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
                raise ValueError(f"Azure TTS {name} is missing or invalid")
        if not isinstance(request.text, str) or not request.text.strip():
            raise ValueError("Azure TTS input text cannot be empty")
        if len(request.text) > self.ssml_bounds.max_text_characters:
            raise ValueError("Azure TTS input text exceeds the configured character limit")
        for duration_name, duration_value in (
            ("preferred_duration_ms", request.preferred_duration_ms),
            ("maximum_duration_ms", request.maximum_duration_ms),
        ):
            if (
                isinstance(duration_value, bool)
                or not isinstance(duration_value, int)
                or duration_value <= 0
            ):
                raise ValueError(f"Azure TTS {duration_name} must be a positive integer")
        if request.preferred_duration_ms > request.maximum_duration_ms:
            raise ValueError("Azure TTS preferred duration cannot exceed maximum duration")
        if request.maximum_duration_ms > AZURE_REST_MAX_AUDIO_DURATION_MS:
            raise ValueError("Azure TTS unit exceeds the REST service's 10-minute audio limit")
        if output_path.suffix.lower() != ".wav":
            raise ValueError("Azure TTS output path must end in .wav")

    def _request_hash(
        self, request: AzureTTSRequest, voice: str, document: SSMLDocument, input_text_hash: str
    ) -> str:
        payload = {
            "schema_version": self.manifest_schema_version,
            "turn_id": request.turn_id,
            "speaker_id": request.speaker_id,
            "voice": voice,
            "language": request.language,
            "input_text_hash": input_text_hash,
            "ssml_hash": document.sha256,
            "preferred_duration_ms": request.preferred_duration_ms,
            "maximum_duration_ms": request.maximum_duration_ms,
            "rate_percent": request.rate_percent,
            "pitch_percent": request.pitch_percent,
            "volume_percent": request.volume_percent,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "sample_width": self.sample_width,
            "output_format": self.output_format,
        }
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _artifact_paths(output_path: Path) -> tuple[Path, Path]:
        return output_path.with_suffix(".azure_tts.ssml"), output_path.with_suffix(".azure_tts.json")

    def _reuse_existing(
        self,
        output_path: Path,
        ssml_path: Path,
        manifest_path: Path,
        *,
        request: AzureTTSRequest,
        voice: str,
        input_text_hash: str,
        document: SSMLDocument,
        request_hash: str,
    ) -> AzureTTSResult | None:
        exists = (output_path.exists(), ssml_path.exists(), manifest_path.exists())
        if not any(exists):
            return None
        if not all(exists):
            raise AzureTTSIdempotencyError(
                "Azure TTS output is incomplete; use forced regeneration after reviewing the partial artifacts"
            )
        try:
            manifest = read_json(manifest_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise AzureTTSIdempotencyError("Azure TTS manifest is unreadable") from exc
        if not isinstance(manifest, dict) or manifest.get("schema_version") != self.manifest_schema_version:
            raise AzureTTSIdempotencyError("Azure TTS manifest schema is invalid")
        try:
            stored_ssml = ssml_path.read_text(encoding="utf-8").removesuffix("\n")
            audio = output_path.read_bytes()
        except OSError as exc:
            raise AzureTTSIdempotencyError("Azure TTS output artifacts are unreadable") from exc
        result_value = manifest.get("result")
        if not isinstance(result_value, dict):
            raise AzureTTSIdempotencyError("Azure TTS manifest result is invalid")
        try:
            result = AzureTTSResult.from_dict(result_value, idempotent_reuse=True)
        except (KeyError, TypeError, ValueError) as exc:
            raise AzureTTSIdempotencyError("Azure TTS manifest result is malformed") from exc
        request_hash_matches = manifest.get("request_hash") == request_hash
        stored_ssml_hash = hashlib.sha256(stored_ssml.encode("utf-8")).hexdigest()
        if stored_ssml_hash != result.ssml_hash:
            raise AzureTTSIdempotencyError(
                "Azure TTS SSML artifact failed hash verification"
            )
        if stored_ssml != document.xml or stored_ssml_hash != document.sha256:
            raise AzureTTSIdempotencyError(
                "Azure TTS output belongs to a different synthesis request",
                conflict_kind="request_mismatch",
            )
        expected = {
            "backend": self.provider,
            "turn_id": request.turn_id,
            "speaker_id": request.speaker_id,
            "voice": voice,
            "language": request.language,
            "input_text_hash": input_text_hash,
            "ssml_hash": document.sha256,
            "output_path": str(output_path),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "sample_width": self.sample_width,
            "preferred_duration_ms": request.preferred_duration_ms,
            "maximum_duration_ms": request.maximum_duration_ms,
            "rate_percent": request.rate_percent,
            "request_hash": request_hash,
            "ssml_path": str(ssml_path),
            "manifest_path": str(manifest_path),
            "ssml_summary": document.summary,
        }
        # Preferred/maximum duration describe the target timeline but do not
        # affect Azure's SSML or returned waveform. A cadence-plan change may
        # therefore safely reuse byte-identical speech when every synthesis
        # identity field and the stored SSML still match. Older manifests retain
        # their original timing metadata and request hash for auditability.
        timing_metadata = {
            "preferred_duration_ms",
            "maximum_duration_ms",
            "request_hash",
        }
        identity_expected = (
            expected
            if request_hash_matches
            else {
                name: value
                for name, value in expected.items()
                if name not in timing_metadata
            }
        )
        if any(
            getattr(result, name) != value
            for name, value in identity_expected.items()
        ):
            conflict_kind = (
                "artifact_integrity"
                if request_hash_matches
                else "request_mismatch"
            )
            raise AzureTTSIdempotencyError(
                "Azure TTS manifest does not match the deterministic request metadata",
                conflict_kind=conflict_kind,
            )
        if not request_hash_matches and stored_ssml != document.xml:
            raise AzureTTSIdempotencyError(
                "Azure TTS output belongs to a different synthesis request",
                conflict_kind="request_mismatch",
            )
        if request_hash_matches and any(
            getattr(result, name) != value for name, value in expected.items()
        ):
            raise AzureTTSIdempotencyError("Azure TTS manifest does not match the deterministic request metadata")
        if not isinstance(result.attempt, int) or result.attempt < 1:
            raise AzureTTSIdempotencyError("Azure TTS manifest attempt is invalid")
        if hashlib.sha256(audio).hexdigest() != result.sha256:
            raise AzureTTSIdempotencyError("Azure TTS WAV failed SHA-256 verification")
        try:
            _, metadata = canonicalize_pcm_wav(
                audio,
                sample_rate=self.sample_rate,
                channels=self.channels,
                sample_width=self.sample_width,
                trim_digital_silence_ms=0,
            )
        except ValueError as exc:
            raise AzureTTSIdempotencyError("Azure TTS WAV failed format validation") from exc
        if metadata["duration_ms"] != result.duration_ms:
            raise AzureTTSIdempotencyError("Azure TTS WAV duration does not match its manifest")
        return result

    def _with_retries(
        self,
        operation: Callable[[], AzureTTSResponse],
        parse: Callable[[AzureTTSResponse], _T],
    ) -> tuple[_T, int]:
        attempts = self.max_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                response = operation()
                value = parse(response)
                return value, attempt
            except AzureTTSTransportError as exc:
                error = AzureTTSError(
                    str(exc),
                    category=exc.category,
                    retryable=exc.retryable,
                )
            except AzureTTSError as exc:
                error = exc
            if not error.retryable or attempt >= attempts:
                raise error
            delay = error.retry_after_seconds
            self.sleep(delay if delay is not None else min(2 ** (attempt - 1), 30))
        raise AssertionError("unreachable")

    def _raise_for_response(self, response: AzureTTSResponse, *, operation: str) -> None:
        if not isinstance(response, AzureTTSResponse):
            raise AzureTTSError(
                f"Azure TTS {operation} boundary returned an invalid response",
                category="malformed_response",
                retryable=False,
            )
        if response.cancellation is not None:
            cancellation = response.cancellation
            code = (cancellation.error_code or "").replace("_", "").casefold()
            details = _safe_detail((cancellation.details or "").encode("utf-8"))
            retryable = code in TRANSIENT_CANCELLATION_CODES
            safe = {"reason": cancellation.reason, "error_code": cancellation.error_code, "details": details}
            raise AzureTTSError(
                f"Azure TTS {operation} was cancelled ({cancellation.error_code or cancellation.reason})",
                category="cancellation",
                retryable=retryable,
                cancellation_details=safe,
            )
        status = response.status_code
        if 200 <= status < 300:
            return
        retryable = status in TRANSIENT_HTTP_STATUS
        retry_after = _retry_after(_header(response.headers, "Retry-After")) if status == 429 else None
        if status in {401, 403}:
            raise AzureTTSError(
                "Azure TTS authentication/authorization failed",
                category="authentication",
                retryable=False,
                status_code=status,
            )
        category = "rate_limit" if status == 429 else "service"
        detail = _safe_detail(response.body)
        suffix = f": {detail}" if detail else ""
        raise AzureTTSError(
            f"Azure TTS {operation} HTTP {status}{suffix}",
            category=category,
            retryable=retryable,
            status_code=status,
            retry_after_seconds=retry_after,
        )


# Real Azure TTS output rarely lands on literal 0x0000 for its "silent" lead-in/
# trail -- it's low-level dither/quantization noise instead (confirmed real case:
# a 16-bit PCM tail alternating between -1 and 0 for over 800ms). An exact-byte
# match therefore misses the overwhelming majority of real silence. Real speech
# in every clip inspected had an RMS in the hundreds to thousands (out of a
# +/-32768 range); this threshold is far below that -- it can only ever remove
# genuine noise-floor silence, never real speech, however quiet.
SILENCE_AMPLITUDE_THRESHOLD = 32


def _frame_is_near_silent(frame_bytes: bytes, *, sample_width: int, threshold: int) -> bool:
    if sample_width == 2:
        values = struct.unpack(f"<{len(frame_bytes) // 2}h", frame_bytes)
    elif sample_width == 1:
        values = [byte - 128 for byte in frame_bytes]
    else:
        # Uncommon sample width: fall back to the original exact-zero definition
        # rather than guessing at an unfamiliar PCM layout's amplitude range.
        return frame_bytes == b"\x00" * len(frame_bytes)
    return all(-threshold <= value <= threshold for value in values)


def canonicalize_pcm_wav(
    data: bytes,
    *,
    sample_rate: int,
    channels: int,
    sample_width: int,
    trim_digital_silence_ms: int,
    silence_amplitude_threshold: int = SILENCE_AMPLITUDE_THRESHOLD,
) -> tuple[bytes, dict[str, int]]:
    if not data:
        raise ValueError("response is empty")
    try:
        with wave.open(io.BytesIO(data), "rb") as source:
            actual = (source.getframerate(), source.getnchannels(), source.getsampwidth())
            if actual != (sample_rate, channels, sample_width):
                raise ValueError(
                    f"expected {sample_rate} Hz/{channels} channel(s)/{sample_width}-byte samples, got "
                    f"{actual[0]} Hz/{actual[1]} channel(s)/{actual[2]}-byte samples"
                )
            if source.getcomptype() != "NONE":
                raise ValueError("audio is compressed instead of PCM")
            frame_count = source.getnframes()
            frames = source.readframes(frame_count)
    except (wave.Error, EOFError) as exc:
        raise ValueError("response is not a readable WAV") from exc
    frame_width = channels * sample_width
    if frame_count <= 0 or len(frames) != frame_count * frame_width:
        raise ValueError("audio contains no complete PCM frames")
    leading = 0
    while leading < frame_count and _frame_is_near_silent(
        frames[leading * frame_width : (leading + 1) * frame_width],
        sample_width=sample_width,
        threshold=silence_amplitude_threshold,
    ):
        leading += 1
    trailing = 0
    while trailing < frame_count - leading and _frame_is_near_silent(
        frames[(frame_count - trailing - 1) * frame_width : (frame_count - trailing) * frame_width],
        sample_width=sample_width,
        threshold=silence_amplitude_threshold,
    ):
        trailing += 1
    if leading + trailing >= frame_count:
        raise ValueError("audio contains only digital silence")
    trim_limit_frames = int(sample_rate * trim_digital_silence_ms / 1000)
    trimmed_leading = min(leading, trim_limit_frames)
    trimmed_trailing = min(trailing, trim_limit_frames)
    first = trimmed_leading * frame_width
    last_frame = frame_count - trimmed_trailing
    canonical_frames = frames[first : last_frame * frame_width]
    canonical_count = last_frame - trimmed_leading
    output = io.BytesIO()
    with wave.open(output, "wb") as target:
        target.setparams((channels, sample_width, sample_rate, 0, "NONE", "not compressed"))
        target.writeframes(canonical_frames)
    return output.getvalue(), {
        "duration_ms": int(round(canonical_count * 1000 / sample_rate)),
        "trimmed_leading_ms": int(round(trimmed_leading * 1000 / sample_rate)),
        "trimmed_trailing_ms": int(round(trimmed_trailing * 1000 / sample_rate)),
    }


def _atomic_write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            directory_descriptor = os.open(path.parent, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _safe_detail(body: bytes) -> str:
    try:
        decoded = body.decode("utf-8", errors="replace")
    except (AttributeError, TypeError):
        return ""
    try:
        parsed = json.loads(decoded)
        if isinstance(parsed, dict):
            candidate: Any = parsed.get("error", parsed)
            if isinstance(candidate, dict):
                candidate = candidate.get("message") or candidate.get("code") or "request rejected"
            decoded = str(candidate)
    except json.JSONDecodeError:
        pass
    safe = redact(decoded)
    text = safe if isinstance(safe, str) else str(safe)
    return " ".join(text.split())[:300]


def _header(headers: Mapping[str, str], name: str) -> str | None:
    wanted = name.casefold()
    return next((str(value) for key, value in headers.items() if str(key).casefold() == wanted), None)


def _retry_after(value: str | None, now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> float | None:
    if not value:
        return None
    try:
        numeric = float(value)
        return max(0.0, numeric) if math.isfinite(numeric) else None
    except ValueError:
        try:
            parsed = email.utils.parsedate_to_datetime(value)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return max(0.0, (parsed - now()).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


def _optional_text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None
