"""Background-source selection and residual-English quality control."""

from __future__ import annotations

import shutil
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .atomic_io import atomic_write_json
from .errors import BackgroundStemUnavailable, ResidualEnglishDetected
from .media import checksum


PUBLICATION_MODES = {"clean_background_stem", "separated_background", "reconstructed_ambience"}
ALL_MODES = PUBLICATION_MODES | {"original_mix_ducked_review_only"}


class SeparationProvider(Protocol):
    provider: str
    revision: str

    def separate(self, source: Path, output: Path) -> dict[str, Any]: ...


class ResidualSpeechDetector(Protocol):
    def detect(self, audio: Path) -> list[dict[str, Any]]: ...


@dataclass(frozen=True)
class BackgroundCandidate:
    mode: str
    path: Path
    provider: str
    provider_metadata: dict[str, Any]


def prepare_background(
    mix_source: Path,
    output: Path,
    *,
    source_dialogue: list[dict[str, Any]],
    clean_stem: Path | None = None,
    cached_separation: Path | None = None,
    separation_provider: SeparationProvider | None = None,
    reconstructed_ambience: Path | None = None,
    allow_review_ducking: bool = False,
) -> dict[str, Any]:
    """Choose the strongest explicitly available background source."""
    candidate: BackgroundCandidate | None = None
    if clean_stem and clean_stem.is_file():
        candidate = BackgroundCandidate("clean_background_stem", clean_stem, "external", {})
    elif cached_separation and cached_separation.is_file():
        candidate = BackgroundCandidate("separated_background", cached_separation, "local_cache", {})
    elif separation_provider is not None:
        temporary = output.with_name(f".{output.name}.separation.partial")
        metadata = separation_provider.separate(mix_source, temporary)
        candidate = BackgroundCandidate(
            "separated_background",
            temporary,
            separation_provider.provider,
            {"revision": separation_provider.revision, **metadata},
        )
    elif reconstructed_ambience and reconstructed_ambience.is_file():
        candidate = BackgroundCandidate("reconstructed_ambience", reconstructed_ambience, "external", {})
    elif allow_review_ducking:
        temporary = output.with_name(f".{output.name}.ducked.partial.wav")
        duck_dialogue_regions(mix_source, temporary, source_dialogue)
        candidate = BackgroundCandidate("original_mix_ducked_review_only", temporary, "internal", {})

    if candidate is None:
        raise BackgroundStemUnavailable(
            "No clean stem, separation output, or reconstructed ambience is available",
            details={"review_ducking_enabled": allow_review_ducking},
        )
    if candidate.mode not in ALL_MODES:
        raise ValueError("Unknown background mode")
    _validate_stereo_pcm(candidate.path)
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = output.with_name(f".{output.name}.partial")
    shutil.copyfile(candidate.path, staged)
    staged.replace(output)
    return {
        "schema_version": "background-manifest-v1",
        "mode": candidate.mode,
        "provider": candidate.provider,
        "provider_metadata": candidate.provider_metadata,
        "source_sha256": checksum(mix_source),
        "output_path": str(output),
        "output_sha256": checksum(output),
        "publication_candidate": candidate.mode in PUBLICATION_MODES,
        "warnings": (
            ["Ducking reduces but does not remove original English dialogue; internal review only"]
            if candidate.mode == "original_mix_ducked_review_only"
            else []
        ),
    }


def residual_dialogue_qc(
    background: Path,
    source_transcript: str,
    detector: ResidualSpeechDetector,
    *,
    max_residual_seconds: float = 0.5,
) -> dict[str, Any]:
    detections = detector.detect(background)
    source_words = _words(source_transcript)
    matched: list[str] = []
    residual_seconds = 0.0
    confidences: list[float] = []
    for item in detections:
        text_words = _words(str(item.get("text", "")))
        overlap = sorted(source_words & text_words)
        if overlap:
            matched.extend(overlap)
            residual_seconds += max(0.0, float(item.get("end", 0)) - float(item.get("start", 0)))
            confidences.append(float(item.get("confidence", 0)))
    passed = residual_seconds <= max_residual_seconds
    return {
        "schema_version": "residual-dialogue-qc-v1",
        "background_sha256": checksum(background),
        "detected_segment_count": len(detections),
        "residual_speech_duration_seconds": round(residual_seconds, 3),
        "confidence": round(sum(confidences) / len(confidences), 4) if confidences else 0.0,
        "matched_source_words": sorted(set(matched)),
        "threshold_seconds": max_residual_seconds,
        "passed": passed,
        "status": "passed" if passed else "needs_background_review",
        "warnings": ["Retranscription is a QC signal, not absolute truth"],
    }


def require_publication_background(manifest: dict[str, Any], residual_qc: dict[str, Any]) -> None:
    if manifest.get("mode") not in PUBLICATION_MODES:
        raise ResidualEnglishDetected("Review-only ducked source cannot enter publication review")
    if not residual_qc.get("passed"):
        raise ResidualEnglishDetected(
            "Residual source dialogue exceeds the configured threshold",
            details={
                "residual_speech_duration_seconds": residual_qc.get("residual_speech_duration_seconds"),
                "threshold_seconds": residual_qc.get("threshold_seconds"),
            },
        )


def duck_dialogue_regions(
    source: Path,
    output: Path,
    regions: list[dict[str, Any]],
    *,
    gain: float = 0.12,
    fade_ms: int = 80,
) -> None:
    if not 0 <= gain <= 1:
        raise ValueError("Ducking gain must be between zero and one")
    with wave.open(str(source), "rb") as src:
        params = src.getparams()
        if params.nchannels != 2 or params.sampwidth != 2:
            raise ValueError("Background source must be stereo PCM16")
        raw = bytearray(src.readframes(params.nframes))
    frame_count = len(raw) // (params.nchannels * params.sampwidth)
    envelope = [1.0] * frame_count
    fade_frames = max(1, round(fade_ms * params.framerate / 1000))
    for item in regions:
        start = max(0, round(float(item["start"]) * params.framerate))
        end = min(frame_count, round(float(item["end"]) * params.framerate))
        for frame in range(max(0, start - fade_frames), min(frame_count, end + fade_frames)):
            if frame < start:
                amount = (frame - (start - fade_frames)) / fade_frames
                value = 1 - amount * (1 - gain)
            elif frame >= end:
                amount = (frame - end) / fade_frames
                value = gain + amount * (1 - gain)
            else:
                value = gain
            envelope[frame] = min(envelope[frame], max(gain, min(1.0, value)))
    for frame, frame_gain in enumerate(envelope):
        for channel in range(2):
            offset = (frame * 2 + channel) * 2
            value = int.from_bytes(raw[offset : offset + 2], "little", signed=True)
            scaled = max(-32768, min(32767, round(value * frame_gain)))
            raw[offset : offset + 2] = scaled.to_bytes(2, "little", signed=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output), "wb") as dst:
        dst.setparams((2, 2, params.framerate, 0, "NONE", "not compressed"))
        dst.writeframes(raw)


def write_background_manifest(path: Path, manifest: dict[str, Any], residual_qc: dict[str, Any]) -> None:
    atomic_write_json(path, {**manifest, "residual_dialogue_qc": residual_qc})


def _validate_stereo_pcm(path: Path) -> None:
    try:
        with wave.open(str(path), "rb") as handle:
            if handle.getnchannels() != 2 or handle.getsampwidth() != 2 or handle.getnframes() <= 0:
                raise ValueError("Background must be non-empty stereo PCM16")
    except (wave.Error, EOFError) as exc:
        raise ValueError(f"Malformed background WAV: {path.name}") from exc


def _words(text: str) -> set[str]:
    return {"".join(character for character in token.lower() if character.isalnum()) for token in text.split()} - {""}
