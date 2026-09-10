"""Bookmark-based grouped Azure Speech SDK synthesis for speech islands.

Speech islands (see speech_islands.py) are normally synthesized as N separate Azure
TTS REST calls per phrase group -- one per island. Real production data showed a
fixed ~900ms Azure per-utterance overhead baked into every call's audio duration, so
synthesizing a K-island group as K calls pays that overhead K times where one payment
would do. The Speech SDK supports SSML <bookmark> tags plus a BookmarkReached event
reporting the real offset where each bookmark landed in ONE continuous synthesis
result, letting a whole group synthesize in a single call while still recovering real
per-island boundaries afterward (see speech_islands.slice_wav_by_bookmarks).

This module is a new file, not an extension of azure_tts.py: the SDK import is an
optional native/platform wheel that must stay lazy so REST-only workflows, CI, and
azure_tts.py's own test surface are unaffected by its presence or absence, and the
SDK's blocking speak_ssml_async().get() + event-based completion model is a different
shape than AzureTTSBoundary's stateless REST request/response, which would blur two
different error models if forced into one retry wrapper.

Repair (Grok-driven re-synthesis of individual islands that overran/underran their
timing budget) deliberately keeps using the existing per-island REST path unchanged --
this module covers only the initial whole-group measurement pass. See native_dub.py's
_presynthesize_islands_with_bookmarks for how the two paths combine.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from .atomic_io import atomic_write_json, read_json
from .azure_tts import (
    TRANSIENT_CANCELLATION_CODES,
    AzureTTSError,
    AzureTTSIdempotencyError,
    canonicalize_pcm_wav,
)
from .logging_utils import redact
from .speech_islands import slice_wav_by_bookmarks
from .tts_ssml import SSMLBounds, build_group_bookmark_ssml


GROUP_MANIFEST_SCHEMA_VERSION = "azure-tts-sdk-group-result-v1"
SLICE_MANIFEST_SCHEMA_VERSION = "azure-tts-sdk-slice-v1"
GROUP_WAV_SUFFIX = ".sdk-group.wav"
SLICE_WAV_SUFFIX = ".sdk-slice.wav"

_REGION = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z")
_GROUP_ID = re.compile(r"[^\x00-\x1f\x7f]{1,200}\Z")
_TICKS_PER_MS = 10_000

# Mirrors azure_tts.py's REST OUTPUT_FORMATS table, but keyed to the SDK's
# SpeechSynthesisOutputFormat enum member names. Deliberately restricted to the
# Riff-prefixed members: Azure's REST "riff-*" formats already round-trip cleanly
# through wave.open/canonicalize_pcm_wav, and the SDK's "Riff*" members are the same
# RIFF-container family, so this sidesteps the (otherwise open) question of whether a
# raw/headerless SDK output format would need a hand-built WAV header.
_SDK_OUTPUT_FORMAT_NAMES = {
    (8_000, 1, 2): "Riff8Khz16BitMonoPcm",
    (16_000, 1, 2): "Riff16Khz16BitMonoPcm",
    (24_000, 1, 2): "Riff24Khz16BitMonoPcm",
    (48_000, 1, 2): "Riff48Khz16BitMonoPcm",
}


def _ticks_to_ms(ticks: int) -> int:
    return int(round(ticks / _TICKS_PER_MS))


@dataclass(frozen=True)
class SDKBookmarkEvent:
    mark: str
    audio_offset_ticks: int


@dataclass(frozen=True)
class SDKGroupSynthesisResponse:
    """Boundary-level result of one bookmarked group synthesis call."""

    audio_data: bytes
    bookmarks: tuple[SDKBookmarkEvent, ...]
    cancelled: bool = False
    cancellation_reason: str | None = None
    cancellation_error_code: str | None = None
    cancellation_details: str | None = None


class AzureSpeechSDKBoundary(Protocol):
    def synthesize_group(self, ssml: str, *, timeout_seconds: float) -> SDKGroupSynthesisResponse: ...


def _import_speech_sdk() -> Any:
    try:
        import azure.cognitiveservices.speech as speechsdk
    except ImportError as exc:
        raise AzureTTSError(
            'Azure Speech SDK is not installed (pip install "mathula-tv[speech-sdk]")',
            category="sdk_unavailable",
            retryable=False,
        ) from exc
    return speechsdk


def build_speech_config(
    *, region: str, subscription_key: str, sample_rate: int, channels: int, sample_width: int
) -> Any:
    """Construct a real azure.cognitiveservices.speech.SpeechConfig for bookmark synthesis.

    Lazily imports the SDK so modules that never use this path (REST-only workflows,
    CI, tests) are unaffected by whether the optional azure-cognitiveservices-speech
    wheel is installed. Reuses the exact same AZURE_SPEECH_REGION/AZURE_SPEECH_KEY
    values already read for the REST path -- no new credential plumbing.
    """
    speechsdk = _import_speech_sdk()
    normalized_region = str(region).strip().lower()
    if not _REGION.fullmatch(normalized_region):
        raise ValueError("AZURE_SPEECH_REGION is missing or invalid")
    if not subscription_key:
        raise ValueError("AZURE_SPEECH_KEY is missing")
    if (sample_rate, channels, sample_width) not in _SDK_OUTPUT_FORMAT_NAMES:
        raise ValueError("Azure Speech SDK output must use a supported mono PCM16 WAV format")
    speech_config = speechsdk.SpeechConfig(subscription=subscription_key, region=normalized_region)
    format_name = _SDK_OUTPUT_FORMAT_NAMES[(sample_rate, channels, sample_width)]
    speech_config.set_speech_synthesis_output_format(getattr(speechsdk.SpeechSynthesisOutputFormat, format_name))
    return speech_config


@dataclass
class AzureSpeechSDKLiveBoundary:
    """Thin production boundary wrapping one blocking bookmarked SDK synthesis call.

    Kept separate from AzureTTSHTTPBoundary: the SDK's speak_ssml_async().get() +
    event-based bookmark reporting is a different shape than the stateless REST
    request/response boundary. synthesize_group_with_bookmarks depends only on the
    AzureSpeechSDKBoundary Protocol, so tests substitute a fake boundary here without
    ever importing the real SDK.
    """

    region: str
    subscription_key: str = field(repr=False)
    sample_rate: int = 24_000
    channels: int = 1
    sample_width: int = 2

    def __post_init__(self) -> None:
        self._speech_config = build_speech_config(
            region=self.region,
            subscription_key=self.subscription_key,
            sample_rate=self.sample_rate,
            channels=self.channels,
            sample_width=self.sample_width,
        )

    def synthesize_group(self, ssml: str, *, timeout_seconds: float) -> SDKGroupSynthesisResponse:
        # NOTE: the SDK's blocking ResultFuture.get() has no public timeout
        # parameter in the standard usage pattern; timeout_seconds is accepted here
        # for AzureSpeechSDKBoundary/test-fake parity but not currently enforced.
        # Verify against the SDK's actual connection/synthesis timeout properties
        # (e.g. speech_config.set_property) before relying on this in production.
        speechsdk = _import_speech_sdk()
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=self._speech_config, audio_config=None)
        bookmarks: list[SDKBookmarkEvent] = []
        synthesizer.bookmark_reached.connect(
            lambda evt: bookmarks.append(SDKBookmarkEvent(mark=evt.text, audio_offset_ticks=int(evt.audio_offset)))
        )
        result = synthesizer.speak_ssml_async(ssml).get()
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            return SDKGroupSynthesisResponse(audio_data=bytes(result.audio_data), bookmarks=tuple(bookmarks))
        if result.reason == speechsdk.ResultReason.Canceled:
            # Real bug, confirmed against the installed SDK (1.51.2): there is no
            # CancellationDetails.from_result classmethod for a synthesis result in
            # this version (that class/API belongs to speech RECOGNITION results).
            # A synthesis result's cancellation details are exposed directly as a
            # property, matching Microsoft's own synthesis quickstart samples.
            details = result.cancellation_details
            return SDKGroupSynthesisResponse(
                audio_data=b"",
                bookmarks=tuple(bookmarks),
                cancelled=True,
                cancellation_reason=str(details.reason) if details.reason is not None else None,
                cancellation_error_code=str(details.error_code) if details.error_code is not None else None,
                cancellation_details=details.error_details,
            )
        raise AzureTTSError(
            f"Azure Speech SDK returned an unexpected synthesis result reason: {result.reason}",
            category="malformed_response",
            retryable=True,
        )


@dataclass(frozen=True)
class GroupBookmarkSynthesisResult:
    parent_group_id: str
    voice: str
    language: str
    rate_percent: int
    pitch_percent: int
    volume_percent: int
    island_ids: tuple[str, ...]
    bookmark_offsets_ms: Mapping[str, int]
    group_wav_path: str
    group_duration_ms: int
    group_sha256: str
    request_hash: str
    ssml_path: str
    manifest_path: str
    attempt: int
    idempotent_reuse: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "parent_group_id": self.parent_group_id,
            "voice": self.voice,
            "language": self.language,
            "rate_percent": self.rate_percent,
            "pitch_percent": self.pitch_percent,
            "volume_percent": self.volume_percent,
            "island_ids": list(self.island_ids),
            "bookmark_offsets_ms": dict(self.bookmark_offsets_ms),
            "group_wav_path": self.group_wav_path,
            "group_duration_ms": self.group_duration_ms,
            "group_sha256": self.group_sha256,
            "request_hash": self.request_hash,
            "ssml_path": self.ssml_path,
            "manifest_path": self.manifest_path,
            "attempt": self.attempt,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, idempotent_reuse: bool = False) -> "GroupBookmarkSynthesisResult":
        fields = {
            "parent_group_id",
            "voice",
            "language",
            "rate_percent",
            "pitch_percent",
            "volume_percent",
            "island_ids",
            "bookmark_offsets_ms",
            "group_wav_path",
            "group_duration_ms",
            "group_sha256",
            "request_hash",
            "ssml_path",
            "manifest_path",
            "attempt",
        }
        if not fields.issubset(value):
            raise AzureTTSIdempotencyError("Azure Speech SDK group manifest is incomplete")
        return cls(
            parent_group_id=str(value["parent_group_id"]),
            voice=str(value["voice"]),
            language=str(value["language"]),
            rate_percent=int(value["rate_percent"]),
            pitch_percent=int(value["pitch_percent"]),
            volume_percent=int(value["volume_percent"]),
            island_ids=tuple(str(item) for item in value["island_ids"]),
            bookmark_offsets_ms={str(k): int(v) for k, v in dict(value["bookmark_offsets_ms"]).items()},
            group_wav_path=str(value["group_wav_path"]),
            group_duration_ms=int(value["group_duration_ms"]),
            group_sha256=str(value["group_sha256"]),
            request_hash=str(value["request_hash"]),
            ssml_path=str(value["ssml_path"]),
            manifest_path=str(value["manifest_path"]),
            attempt=int(value["attempt"]),
            idempotent_reuse=idempotent_reuse,
        )


@dataclass(frozen=True)
class IslandSliceResult:
    island_id: str
    parent_group_id: str
    path: str
    start_ms: int
    end_ms: int
    duration_ms: int
    sha256: str
    manifest_path: str
    idempotent_reuse: bool = False


def _artifact_paths(output_path: Path) -> tuple[Path, Path]:
    return output_path.with_suffix(".azure_tts_sdk.ssml"), output_path.with_suffix(".azure_tts_sdk.json")


def _slice_paths(output_dir: Path, island_id: str, *, output_suffix: str) -> tuple[Path, Path]:
    wav_path = output_dir / f"{island_id}{output_suffix}"
    manifest_path = wav_path.with_suffix(".json")
    return wav_path, manifest_path


def _group_request_hash(
    *,
    parent_group_id: str,
    islands: Sequence[tuple[str, str]],
    voice: str,
    language: str,
    rate_percent: int,
    pitch_percent: int,
    volume_percent: int,
    sample_rate: int,
    channels: int,
    sample_width: int,
) -> str:
    payload = {
        "schema_version": GROUP_MANIFEST_SCHEMA_VERSION,
        "parent_group_id": str(parent_group_id),
        "islands": [
            [str(island_id), hashlib.sha256(text.encode("utf-8")).hexdigest()] for island_id, text in islands
        ],
        "voice": voice,
        "language": language,
        "rate_percent": rate_percent,
        "pitch_percent": pitch_percent,
        "volume_percent": volume_percent,
        "sample_rate": sample_rate,
        "channels": channels,
        "sample_width": sample_width,
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _safe_sdk_detail(value: str | None) -> str:
    if not value:
        return ""
    safe = redact(value)
    text = safe if isinstance(safe, str) else str(safe)
    return " ".join(text.split())[:300]


def _with_retries(
    operation: Callable[[], SDKGroupSynthesisResponse],
    *,
    max_retries: int,
    sleep: Callable[[float], None],
) -> tuple[SDKGroupSynthesisResponse, int]:
    attempts = max_retries + 1
    for attempt in range(1, attempts + 1):
        response = operation()
        if not response.cancelled:
            return response, attempt
        # Shared with azure_tts.py's REST cancellation taxonomy: both derive from
        # Microsoft's CancellationErrorCode family. Verify the exact SDK enum member
        # spellings against a real cancellation before depending on this in
        # production (see the SDK migration plan's open questions).
        code = (response.cancellation_error_code or "").replace("_", "").casefold()
        retryable = code in TRANSIENT_CANCELLATION_CODES
        detail = _safe_sdk_detail(response.cancellation_details)
        error = AzureTTSError(
            f"Azure Speech SDK group synthesis was cancelled "
            f"({response.cancellation_error_code or response.cancellation_reason})",
            category="cancellation",
            retryable=retryable,
            cancellation_details={
                "reason": response.cancellation_reason,
                "error_code": response.cancellation_error_code,
                "details": detail,
            },
        )
        if not retryable or attempt >= attempts:
            raise error
        sleep(min(2 ** (attempt - 1), 30))
    raise AssertionError("unreachable")


def _extract_bookmark_offsets_ms(response: SDKGroupSynthesisResponse, island_ids: Sequence[str]) -> dict[str, int]:
    marks_in_order = [event.mark for event in response.bookmarks]
    if marks_in_order != list(island_ids):
        raise AzureTTSError(
            "Azure Speech SDK bookmark events do not match the requested islands in order: "
            f"expected {list(island_ids)!r}, got {marks_in_order!r}",
            category="malformed_response",
            retryable=False,
        )
    offsets_ms: dict[str, int] = {}
    previous_ms = -1
    for event in response.bookmarks:
        ms = _ticks_to_ms(event.audio_offset_ticks)
        if ms <= previous_ms:
            raise AzureTTSError(
                f"Azure Speech SDK bookmark for island {event.mark!r} is out of order or duplicated",
                category="malformed_response",
                retryable=False,
            )
        offsets_ms[event.mark] = ms
        previous_ms = ms
    return offsets_ms


def _reuse_existing_group(
    output_path: Path,
    ssml_path: Path,
    manifest_path: Path,
    *,
    xml: str,
    ssml_hash: str,
    request_hash: str,
    island_ids: Sequence[str],
    sample_rate: int,
    channels: int,
    sample_width: int,
) -> GroupBookmarkSynthesisResult | None:
    exists = (output_path.exists(), ssml_path.exists(), manifest_path.exists())
    if not any(exists):
        return None
    if not all(exists):
        raise AzureTTSIdempotencyError(
            "Azure Speech SDK group output is incomplete; use forced regeneration after reviewing the partial artifacts"
        )
    try:
        manifest = read_json(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise AzureTTSIdempotencyError("Azure Speech SDK group manifest is unreadable") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != GROUP_MANIFEST_SCHEMA_VERSION:
        raise AzureTTSIdempotencyError("Azure Speech SDK group manifest schema is invalid")
    try:
        stored_ssml = ssml_path.read_text(encoding="utf-8").removesuffix("\n")
        audio = output_path.read_bytes()
    except OSError as exc:
        raise AzureTTSIdempotencyError("Azure Speech SDK group output artifacts are unreadable") from exc
    result_value = manifest.get("result")
    if not isinstance(result_value, dict):
        raise AzureTTSIdempotencyError("Azure Speech SDK group manifest result is invalid")
    try:
        result = GroupBookmarkSynthesisResult.from_dict(result_value, idempotent_reuse=True)
    except (KeyError, TypeError, ValueError) as exc:
        raise AzureTTSIdempotencyError("Azure Speech SDK group manifest result is malformed") from exc
    if manifest.get("request_hash") != request_hash or result.request_hash != request_hash:
        raise AzureTTSIdempotencyError(
            "Azure Speech SDK group output belongs to a different synthesis request",
            conflict_kind="request_mismatch",
        )
    if stored_ssml != xml or manifest.get("ssml_hash") != ssml_hash:
        raise AzureTTSIdempotencyError("Azure Speech SDK group SSML artifact failed hash verification")
    if tuple(result.island_ids) != tuple(island_ids):
        raise AzureTTSIdempotencyError(
            "Azure Speech SDK group manifest island order does not match the requested islands",
            conflict_kind="request_mismatch",
        )
    if hashlib.sha256(audio).hexdigest() != result.group_sha256:
        raise AzureTTSIdempotencyError("Azure Speech SDK group WAV failed SHA-256 verification")
    try:
        _, metadata = canonicalize_pcm_wav(
            audio,
            sample_rate=sample_rate,
            channels=channels,
            sample_width=sample_width,
            trim_digital_silence_ms=0,
        )
    except ValueError as exc:
        raise AzureTTSIdempotencyError("Azure Speech SDK group WAV failed format validation") from exc
    if metadata["duration_ms"] != result.group_duration_ms:
        raise AzureTTSIdempotencyError("Azure Speech SDK group WAV duration does not match its manifest")
    return result


def synthesize_group_with_bookmarks(
    boundary: AzureSpeechSDKBoundary,
    islands: Sequence[tuple[str, str]],
    *,
    parent_group_id: str,
    voice: str,
    allowed_voices: Iterable[str],
    output_path: Path,
    force: bool = False,
    language: str = "zu-ZA",
    rate_percent: int = 0,
    pitch_percent: int = 0,
    volume_percent: int = 0,
    sample_rate: int = 24_000,
    channels: int = 1,
    sample_width: int = 2,
    ssml_bounds: SSMLBounds | None = None,
    trim_digital_silence_ms: int = 2000,
    max_retries: int = 3,
    timeout_seconds: float = 120,
    sleep: Callable[[float], None] = time.sleep,
) -> GroupBookmarkSynthesisResult:
    """Synthesize a whole group's speech islands in one bookmarked Speech SDK call.

    Idempotent and content-addressed like AzureTTSBackend.synthesize, but for a
    disjoint artifact family ({parent_group_id}.sdk-group.wav / .azure_tts_sdk.json)
    so no rerun can ever collide with existing per-island REST artifacts.
    """
    output_path = Path(output_path)
    if not _GROUP_ID.fullmatch(str(parent_group_id)):
        raise ValueError("Azure Speech SDK parent_group_id is missing or invalid")
    if not islands:
        raise ValueError("Azure Speech SDK group synthesis requires at least one island")
    island_ids = tuple(str(island_id) for island_id, _ in islands)
    if len(set(island_ids)) != len(island_ids):
        raise ValueError("Azure Speech SDK group synthesis received duplicate island ids")
    if output_path.suffix.lower() != ".wav":
        raise ValueError("Azure Speech SDK group output path must end in .wav")
    resolved_bounds = ssml_bounds or SSMLBounds()

    xml, ssml_hash = build_group_bookmark_ssml(
        islands,
        voice=voice,
        allowed_voices=allowed_voices,
        bounds=resolved_bounds,
        language=language,
        rate_percent=rate_percent,
        pitch_percent=pitch_percent,
        volume_percent=volume_percent,
    )
    ssml_path, manifest_path = _artifact_paths(output_path)
    request_hash = _group_request_hash(
        parent_group_id=parent_group_id,
        islands=islands,
        voice=voice,
        language=language,
        rate_percent=rate_percent,
        pitch_percent=pitch_percent,
        volume_percent=volume_percent,
        sample_rate=sample_rate,
        channels=channels,
        sample_width=sample_width,
    )

    if not force:
        existing = _reuse_existing_group(
            output_path,
            ssml_path,
            manifest_path,
            xml=xml,
            ssml_hash=ssml_hash,
            request_hash=request_hash,
            island_ids=island_ids,
            sample_rate=sample_rate,
            channels=channels,
            sample_width=sample_width,
        )
        if existing is not None:
            return existing

    response, attempt = _with_retries(
        lambda: boundary.synthesize_group(xml, timeout_seconds=timeout_seconds),
        max_retries=max_retries,
        sleep=sleep,
    )
    try:
        audio, metadata = canonicalize_pcm_wav(
            response.audio_data,
            sample_rate=sample_rate,
            channels=channels,
            sample_width=sample_width,
            trim_digital_silence_ms=trim_digital_silence_ms,
        )
    except ValueError as exc:
        raise AzureTTSError(
            f"Azure Speech SDK returned invalid WAV audio: {exc}", category="invalid_output", retryable=True
        ) from exc

    raw_offsets_ms = _extract_bookmark_offsets_ms(response, island_ids)
    # Bookmark offsets are relative to the RAW synthesized audio; canonicalization
    # trims leading digital silence off the front, so the WAV slice_wav_by_bookmarks
    # will actually operate on is shorter at the start by that trimmed amount. Shift
    # offsets left to stay aligned with the canonicalized WAV.
    shift_ms = metadata["trimmed_leading_ms"]
    bookmark_offsets_ms = {island_id: max(0, offset_ms - shift_ms) for island_id, offset_ms in raw_offsets_ms.items()}

    audio_sha256 = hashlib.sha256(audio).hexdigest()
    result = GroupBookmarkSynthesisResult(
        parent_group_id=str(parent_group_id),
        voice=voice,
        language=language,
        rate_percent=rate_percent,
        pitch_percent=pitch_percent,
        volume_percent=volume_percent,
        island_ids=island_ids,
        bookmark_offsets_ms=bookmark_offsets_ms,
        group_wav_path=str(output_path),
        group_duration_ms=metadata["duration_ms"],
        group_sha256=audio_sha256,
        request_hash=request_hash,
        ssml_path=str(ssml_path),
        manifest_path=str(manifest_path),
        attempt=attempt,
    )
    manifest = {
        "schema_version": GROUP_MANIFEST_SCHEMA_VERSION,
        "request_hash": request_hash,
        "ssml_hash": ssml_hash,
        "result": result.to_dict(),
    }
    # The manifest is promoted last and therefore acts as the completeness marker,
    # matching AzureTTSBackend.synthesize's existing artifact-write ordering.
    _atomic_write_bytes(output_path, audio)
    _atomic_write_bytes(ssml_path, (xml + "\n").encode("utf-8"))
    atomic_write_json(manifest_path, manifest)
    return result


def _reuse_existing_slices(
    group_result: GroupBookmarkSynthesisResult, output_dir: Path, *, output_suffix: str
) -> dict[str, IslandSliceResult] | None:
    results: dict[str, IslandSliceResult] = {}
    any_exists = False
    for island_id in group_result.island_ids:
        wav_path, manifest_path = _slice_paths(output_dir, island_id, output_suffix=output_suffix)
        exists = (wav_path.exists(), manifest_path.exists())
        if not any(exists):
            continue
        any_exists = True
        if not all(exists):
            raise AzureTTSIdempotencyError(
                f"Azure Speech SDK island slice is incomplete for {island_id!r}; use forced regeneration"
            )
        try:
            manifest = read_json(manifest_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise AzureTTSIdempotencyError(f"Azure Speech SDK slice manifest is unreadable for {island_id!r}") from exc
        if (
            not isinstance(manifest, dict)
            or manifest.get("schema_version") != SLICE_MANIFEST_SCHEMA_VERSION
            or manifest.get("parent_request_hash") != group_result.request_hash
            or manifest.get("island_id") != island_id
        ):
            raise AzureTTSIdempotencyError(
                f"Azure Speech SDK slice manifest does not match the current group request for {island_id!r}",
                conflict_kind="request_mismatch",
            )
        try:
            audio = wav_path.read_bytes()
        except OSError as exc:
            raise AzureTTSIdempotencyError(f"Azure Speech SDK slice audio is unreadable for {island_id!r}") from exc
        if hashlib.sha256(audio).hexdigest() != manifest.get("sha256"):
            raise AzureTTSIdempotencyError(f"Azure Speech SDK slice WAV failed SHA-256 verification for {island_id!r}")
        start_ms = int(manifest["start_ms"])
        end_ms = int(manifest["end_ms"])
        results[island_id] = IslandSliceResult(
            island_id=island_id,
            parent_group_id=group_result.parent_group_id,
            path=str(wav_path),
            start_ms=start_ms,
            end_ms=end_ms,
            duration_ms=end_ms - start_ms,
            sha256=str(manifest["sha256"]),
            manifest_path=str(manifest_path),
            idempotent_reuse=True,
        )
    if not any_exists:
        return None
    if set(results) != set(group_result.island_ids):
        raise AzureTTSIdempotencyError(
            "Azure Speech SDK island slices are incomplete for this group; use forced regeneration"
        )
    return results


def write_group_slices(
    group_result: GroupBookmarkSynthesisResult,
    output_dir: Path,
    *,
    force: bool = False,
    output_suffix: str = SLICE_WAV_SUFFIX,
) -> dict[str, IslandSliceResult]:
    """Split a synthesized group WAV into per-island WAVs, with disk-level idempotency.

    Cheap and fully local (no Azure billing), so unlike synthesize_group_with_bookmarks
    an incomplete or absent slice set is simply regenerated rather than treated as a
    hard idempotency conflict; a PARTIAL but individually-invalid set still raises,
    since that signals real corruption rather than "never sliced yet".
    """
    output_dir = Path(output_dir)
    if not force:
        reused = _reuse_existing_slices(group_result, output_dir, output_suffix=output_suffix)
        if reused is not None:
            return reused

    sliced = slice_wav_by_bookmarks(
        Path(group_result.group_wav_path),
        group_result.island_ids,
        group_result.bookmark_offsets_ms,
        output_dir,
        output_suffix=output_suffix,
    )
    results: dict[str, IslandSliceResult] = {}
    for island_id, info in sliced.items():
        slice_path = Path(info["path"])
        audio = slice_path.read_bytes()
        sha256 = hashlib.sha256(audio).hexdigest()
        _, manifest_path = _slice_paths(output_dir, island_id, output_suffix=output_suffix)
        manifest = {
            "schema_version": SLICE_MANIFEST_SCHEMA_VERSION,
            "parent_group_id": group_result.parent_group_id,
            "parent_request_hash": group_result.request_hash,
            "island_id": island_id,
            "start_ms": info["start_ms"],
            "end_ms": info["end_ms"],
            "sha256": sha256,
        }
        atomic_write_json(manifest_path, manifest)
        results[island_id] = IslandSliceResult(
            island_id=island_id,
            parent_group_id=group_result.parent_group_id,
            path=str(slice_path),
            start_ms=info["start_ms"],
            end_ms=info["end_ms"],
            duration_ms=info["duration_ms"],
            sha256=sha256,
            manifest_path=str(manifest_path),
        )
    return results


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
