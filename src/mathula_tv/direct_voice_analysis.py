"""Self-contained canonical speaker voice-family analysis for direct dubbing.

The direct renderer cannot depend on the removed production orchestrator. This
module creates canonical speaker WAV reels directly from the analysis audio and
raw diarization turns, runs the existing ECAPA classifier, and persists results
under canonical speaker IDs.
"""

from __future__ import annotations

import hashlib
import json
import wave
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .atomic_io import atomic_write_json


DIRECT_VOICE_ANALYSIS_SCHEMA_VERSION = "mathula-direct-voice-analysis-v2-short-family-reels"

# Known-speaker identity matching remains conservative, but binary voice-family
# classification can safely use shorter clean evidence.  The ECAPA family model
# has no fixed one-second input requirement; the previous 1.0s guard was local.
MINIMUM_KNOWN_SPEAKER_REEL_SECONDS = 1.0
MINIMUM_VOICE_FAMILY_REEL_SECONDS = 0.75
MINIMUM_KNOWN_SPEAKER_TURN_SECONDS = 0.20
MINIMUM_VOICE_FAMILY_TURN_SECONDS = 0.10


class DirectVoiceAnalysisError(RuntimeError):
    """Raised when canonical speaker evidence cannot be produced safely."""


def ensure_canonical_speaker_reels(
    *,
    analysis_audio_path: Path,
    diarization: Mapping[str, Any],
    speaker_mapping: Mapping[str, Any],
    required_speaker_ids: Sequence[str],
    speakers_root: Path,
    canonical_speaker_turns: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    max_audio_seconds_per_speaker: float = 60.0,
    force: bool = False,
) -> dict[str, dict[str, Any]]:
    """Create reusable canonical speaker reels without classifying gender."""
    if not analysis_audio_path.is_file():
        raise DirectVoiceAnalysisError(
            f"Analysis audio is missing: {analysis_audio_path}"
        )
    raw_to_canonical = {
        str(raw): str(canonical)
        for raw, canonical in speaker_mapping.get("raw_to_canonical", {}).items()
    }
    canonical_to_raw = {
        str(canonical): [str(raw) for raw in raw_ids]
        for canonical, raw_ids in speaker_mapping.get("canonical_to_raw", {}).items()
    }
    turns = [
        item for item in diarization.get("turns", []) if isinstance(item, Mapping)
    ]
    speakers_root.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}
    for canonical_id in required_speaker_ids:
        canonical_id = str(canonical_id)
        raw_ids = canonical_to_raw.get(canonical_id, [])
        if not raw_ids:
            raw_ids = sorted(
                raw
                for raw, canonical in raw_to_canonical.items()
                if canonical == canonical_id
            )
        speaker_wav = speakers_root / f"{canonical_id}.wav"
        extraction: dict[str, Any] | None = None
        try:
            contextual_turns = list(
                (canonical_speaker_turns or {}).get(canonical_id, ())
            )
            reusable = (
                not force
                and not contextual_turns
                and speaker_wav.is_file()
                and speaker_wav.stat().st_size > 0
                and _wav_duration_seconds(speaker_wav)
                >= MINIMUM_KNOWN_SPEAKER_REEL_SECONDS
            )
            if not reusable:
                selected_turns = contextual_turns or [
                    turn for turn in turns
                    if str(turn.get("speaker") or "").strip() in set(raw_ids)
                ]
                extraction = _write_speaker_reel(
                    source_path=analysis_audio_path,
                    output_path=speaker_wav,
                    turns=selected_turns,
                    max_audio_seconds=max_audio_seconds_per_speaker,
                    minimum_audio_seconds=MINIMUM_KNOWN_SPEAKER_REEL_SECONDS,
                    minimum_turn_seconds=MINIMUM_KNOWN_SPEAKER_TURN_SECONDS,
                )
            results[canonical_id] = {
                "speaker_id": canonical_id,
                "raw_speaker_ids": raw_ids,
                "speaker_audio_path": str(speaker_wav),
                "speaker_audio_sha256": _sha256(speaker_wav),
                "extraction": extraction,
                "turn_source": (
                    "contextual_transcript_segments"
                    if contextual_turns
                    else "raw_diarization"
                ),
                "reused": extraction is None,
                "status": "ready",
            }
        except Exception as exc:
            speaker_wav.unlink(missing_ok=True)
            results[canonical_id] = {
                "speaker_id": canonical_id,
                "raw_speaker_ids": raw_ids,
                "speaker_audio_path": None,
                "speaker_audio_sha256": None,
                "extraction": None,
                "reused": False,
                "status": "unavailable",
                "error": str(exc),
            }
    return results


def ensure_canonical_voice_family_analysis(
    *,
    job_id: str,
    analysis_audio_path: Path,
    diarization: Mapping[str, Any],
    speaker_mapping: Mapping[str, Any],
    required_speaker_ids: Sequence[str],
    output_path: Path,
    speakers_root: Path,
    canonical_speaker_turns: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    confidence_threshold: float = 0.70,
    max_audio_seconds_per_speaker: float = 60.0,
    force: bool = False,
    classifier: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Create or refresh canonical ECAPA voice-family evidence.

    Existing high-confidence canonical evidence is reused unless ``force`` is
    true. Missing or unresolved required speakers are rebuilt from raw
    diarization turns mapped to canonical IDs.
    """
    if not analysis_audio_path.is_file():
        raise DirectVoiceAnalysisError(
            f"Analysis audio is missing: {analysis_audio_path}"
        )
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1")
    if max_audio_seconds_per_speaker <= 0:
        raise ValueError("max_audio_seconds_per_speaker must be positive")

    existing = _read_json_if_present(output_path)
    existing_speakers = _speaker_records(existing, speaker_mapping)
    required = [str(value) for value in required_speaker_ids]

    unresolved = [
        speaker_id
        for speaker_id in required
        if force
        or not _is_resolved(existing_speakers.get(speaker_id), confidence_threshold)
    ]
    if not unresolved:
        return existing
    if classifier is None:
        try:
            from .voice_family_classifier import classify_voice_family
        except ImportError as exc:
            raise DirectVoiceAnalysisError(
                "Voice-family analysis requires its optional PyTorch classifier "
                "dependencies"
            ) from exc
        classifier = classify_voice_family

    raw_to_canonical = {
        str(raw): str(canonical)
        for raw, canonical in speaker_mapping.get("raw_to_canonical", {}).items()
    }
    canonical_to_raw = {
        str(canonical): [str(raw) for raw in raw_ids]
        for canonical, raw_ids in speaker_mapping.get("canonical_to_raw", {}).items()
    }
    turns = [
        item
        for item in diarization.get("turns", [])
        if isinstance(item, Mapping)
    ]

    speakers_root.mkdir(parents=True, exist_ok=True)
    results = dict(existing_speakers)
    failures: dict[str, dict[str, Any]] = {}

    for canonical_id in unresolved:
        raw_ids = canonical_to_raw.get(canonical_id, [])
        if not raw_ids:
            raw_ids = sorted(
                raw for raw, canonical in raw_to_canonical.items()
                if canonical == canonical_id
            )
        contextual_turns = list(
            (canonical_speaker_turns or {}).get(canonical_id, ())
        )
        selected_turns = contextual_turns or [
            turn for turn in turns
            if str(turn.get("speaker") or "").strip() in set(raw_ids)
        ]
        speaker_wav = speakers_root / f"{canonical_id}.wav"

        try:
            extraction = _write_speaker_reel(
                source_path=analysis_audio_path,
                output_path=speaker_wav,
                turns=selected_turns,
                max_audio_seconds=max_audio_seconds_per_speaker,
                minimum_audio_seconds=MINIMUM_VOICE_FAMILY_REEL_SECONDS,
                minimum_turn_seconds=MINIMUM_VOICE_FAMILY_TURN_SECONDS,
            )
            classification = classifier(
                speaker_audio_path=speaker_wav,
                speaker_id=canonical_id,
                confidence_threshold=confidence_threshold,
            )
            record = classification.to_dict()
            record.update(
                {
                    "speaker_id": canonical_id,
                    "raw_speaker_ids": raw_ids,
                    "raw_speaker_id": raw_ids[0] if len(raw_ids) == 1 else None,
                    "speaker_audio_path": str(speaker_wav),
                    "speaker_audio_sha256": _sha256(speaker_wav),
                    "extraction": extraction,
                    "turn_source": (
                        "contextual_transcript_segments"
                        if contextual_turns
                        else "raw_diarization"
                    ),
                    "evidence": {
                        "method": classification.method,
                        "model": classification.model,
                        "device": classification.device,
                        "ecapa_probabilities": classification.probabilities,
                        "canonical_mapping_source": "raw_to_canonical_timeline_mapping",
                    },
                }
            )
            results[canonical_id] = record
        except Exception as exc:
            failures[canonical_id] = {
                "speaker_id": canonical_id,
                "raw_speaker_ids": raw_ids,
                "error": str(exc),
            }
            results[canonical_id] = {
                "speaker_id": canonical_id,
                "raw_speaker_ids": raw_ids,
                "voice_family": "unknown",
                "confidence": 0.0,
                "evidence": {
                    "reason": "direct_ecapa_analysis_failed",
                    "error": str(exc),
                },
            }

    artifact = {
        "schema_version": DIRECT_VOICE_ANALYSIS_SCHEMA_VERSION,
        "job_id": job_id,
        "audio_path": str(analysis_audio_path),
        "audio_sha256": _sha256(analysis_audio_path),
        "speaker_mapping": {
            "raw_to_canonical": raw_to_canonical,
            "canonical_to_raw": canonical_to_raw,
        },
        "confidence_threshold": confidence_threshold,
        "speakers": results,
        "failures": failures,
    }
    atomic_write_json(output_path, artifact)
    return artifact


def _write_speaker_reel(
    *,
    source_path: Path,
    output_path: Path,
    turns: Sequence[Mapping[str, Any]],
    max_audio_seconds: float,
    minimum_audio_seconds: float = MINIMUM_KNOWN_SPEAKER_REEL_SECONDS,
    minimum_turn_seconds: float = MINIMUM_KNOWN_SPEAKER_TURN_SECONDS,
) -> dict[str, Any]:
    """Concatenate mapped diarization turns into one canonical PCM WAV reel."""
    if minimum_audio_seconds <= 0:
        raise ValueError("minimum_audio_seconds must be positive")
    if minimum_turn_seconds <= 0:
        raise ValueError("minimum_turn_seconds must be positive")
    with wave.open(str(source_path), "rb") as source:
        channels = source.getnchannels()
        sample_width = source.getsampwidth()
        sample_rate = source.getframerate()
        frame_count = source.getnframes()
        if channels != 1:
            raise DirectVoiceAnalysisError(
                f"Analysis audio must be mono, got {channels} channels"
            )
        if sample_width != 2:
            raise DirectVoiceAnalysisError(
                f"Analysis audio must be PCM16, got sample width {sample_width}"
            )

        bounded: list[tuple[int, int]] = []
        for turn in turns:
            bounds = _turn_bounds_seconds(turn)
            if bounds is None:
                continue
            start_seconds, end_seconds = bounds
            start_frame = max(0, min(frame_count, round(start_seconds * sample_rate)))
            end_frame = max(start_frame, min(frame_count, round(end_seconds * sample_rate)))
            if end_frame - start_frame < round(minimum_turn_seconds * sample_rate):
                continue
            bounded.append((start_frame, end_frame))

        if not bounded:
            raise DirectVoiceAnalysisError(
                "No usable diarization turns were available for the canonical speaker"
            )

        # Prefer longer turns first to reduce classifier contamination from tiny
        # fragments, but restore chronological order before concatenation.
        max_frames = round(max_audio_seconds * sample_rate)
        selected: list[tuple[int, int]] = []
        total = 0
        for start_frame, end_frame in sorted(
            bounded,
            key=lambda item: (item[1] - item[0], -item[0]),
            reverse=True,
        ):
            remaining = max_frames - total
            if remaining <= 0:
                break
            clipped_end = min(end_frame, start_frame + remaining)
            if clipped_end > start_frame:
                selected.append((start_frame, clipped_end))
                total += clipped_end - start_frame
        selected.sort()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(f".{output_path.name}.partial")
        with wave.open(str(temporary), "wb") as destination:
            destination.setnchannels(1)
            destination.setsampwidth(sample_width)
            destination.setframerate(sample_rate)
            for start_frame, end_frame in selected:
                source.setpos(start_frame)
                destination.writeframes(source.readframes(end_frame - start_frame))
        temporary.replace(output_path)

    duration_seconds = total / sample_rate
    if duration_seconds < minimum_audio_seconds:
        output_path.unlink(missing_ok=True)
        raise DirectVoiceAnalysisError(
            "Canonical speaker reel is too short: "
            f"{duration_seconds:.3f}s; minimum is {minimum_audio_seconds:.3f}s"
        )
    return {
        "turn_count_available": len(bounded),
        "turn_count_selected": len(selected),
        "duration_seconds": round(duration_seconds, 6),
        "sample_rate": sample_rate,
        "channels": 1,
        "sample_width": sample_width,
        "maximum_duration_seconds": max_audio_seconds,
        "minimum_duration_seconds": minimum_audio_seconds,
        "minimum_turn_seconds": minimum_turn_seconds,
    }


def _wav_duration_seconds(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as wav:
            sample_rate = wav.getframerate()
            if sample_rate <= 0:
                return 0.0
            return wav.getnframes() / sample_rate
    except (OSError, wave.Error):
        return 0.0


def _turn_bounds_seconds(item: Mapping[str, Any]) -> tuple[float, float] | None:
    try:
        if item.get("start") is not None and item.get("end") is not None:
            start = float(item["start"])
            end = float(item["end"])
        elif item.get("start_ms") is not None and item.get("end_ms") is not None:
            start = float(item["start_ms"]) / 1000.0
            end = float(item["end_ms"]) / 1000.0
        else:
            return None
    except (TypeError, ValueError):
        return None
    if end <= start:
        return None
    return start, end


def _speaker_records(
    value: Mapping[str, Any],
    speaker_mapping: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    raw: Any = value.get("speakers", {}) if isinstance(value, Mapping) else {}
    raw_to_canonical = {
        str(key): str(mapped)
        for key, mapped in speaker_mapping.get("raw_to_canonical", {}).items()
    }
    records: dict[str, dict[str, Any]] = {}

    def add(record_key: str, item: Mapping[str, Any]) -> None:
        declared = str(item.get("speaker_id") or record_key).strip()
        canonical = declared if declared.startswith("SPEAKER_") else raw_to_canonical.get(declared)
        if not canonical:
            return
        record = dict(item)
        record["speaker_id"] = canonical
        if declared != canonical:
            record.setdefault("raw_speaker_id", declared)
        previous = records.get(canonical)
        if previous is None or _record_rank(record) > _record_rank(previous):
            records[canonical] = record

    if isinstance(raw, Mapping):
        for key, item in raw.items():
            if isinstance(item, Mapping):
                add(str(key), item)
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, Mapping) and item.get("speaker_id"):
                add(str(item["speaker_id"]), item)
    return records


def _record_rank(item: Mapping[str, Any]) -> tuple[int, float]:
    family = str(item.get("voice_family") or "unknown").lower()
    try:
        confidence = float(item.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return (1 if family in {"masculine", "feminine"} else 0, confidence)


def _is_resolved(item: Mapping[str, Any] | None, threshold: float) -> bool:
    if not item:
        return False
    family = str(item.get("voice_family") or "unknown").lower()
    try:
        confidence = float(item.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return family in {"masculine", "feminine"} and confidence >= threshold


def _read_json_if_present(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "DIRECT_VOICE_ANALYSIS_SCHEMA_VERSION",
    "DirectVoiceAnalysisError",
    "ensure_canonical_speaker_reels",
    "ensure_canonical_voice_family_analysis",
]
