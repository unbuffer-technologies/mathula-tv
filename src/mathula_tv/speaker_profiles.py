"""Authoritative speaker context resolution for Azure TTS dubbing."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .atomic_io import atomic_write_json, read_json


SPEAKER_PROFILES_SCHEMA_VERSION = "mathula-speaker-profiles-v1"
_WORD_RE = re.compile(r"[^\W_]+(?:[’'-][^\W_]+)*", re.UNICODE)


@dataclass(frozen=True)
class SpeakerProfile:
    """Authoritative speaker context for voice and delivery resolution."""

    speaker_id: str
    identified_name: str | None
    identity_status: str  # "resolved" | "unresolved" | "inferred"
    identity_source: str | None
    identity_confidence: float | None
    identity_gender: str | None  # "male" | "female" | "unknown"
    identity_gender_source: str | None
    acoustic_voice_family: str  # "masculine" | "feminine" | "unknown"
    acoustic_confidence: float
    tts_voice_family: str  # "masculine" | "feminine" | "unknown"
    tts_voice_family_source: str
    evidence: dict[str, Any] = field(default_factory=dict)
    source_delivery: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "speaker_id": self.speaker_id,
            "identified_name": self.identified_name,
            "identity_status": self.identity_status,
            "identity_source": self.identity_source,
            "identity_confidence": self.identity_confidence,
            "identity_gender": self.identity_gender,
            "identity_gender_source": self.identity_gender_source,
            "acoustic_voice_family": self.acoustic_voice_family,
            "acoustic_confidence": self.acoustic_confidence,
            "tts_voice_family": self.tts_voice_family,
            "tts_voice_family_source": self.tts_voice_family_source,
            "evidence": self.evidence,
            "source_delivery": self.source_delivery,
        }


@dataclass
class SpeakerProfilesArtifact:
    """Complete speaker context artifact for a job."""

    schema_version: str = SPEAKER_PROFILES_SCHEMA_VERSION
    job_id: str = ""
    speakers: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "job_id": self.job_id,
            "speakers": self.speakers,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SpeakerProfilesArtifact":
        return cls(
            schema_version=data.get("schema_version", SPEAKER_PROFILES_SCHEMA_VERSION),
            job_id=data.get("job_id", ""),
            speakers=data.get("speakers", []),
        )


def build_speaker_profiles(
    job_id: str,
    transcript_path: Path,
    diarization_path: Path,
    speaker_registry_path: Path | None = None,
    entity_registry_path: Path | None = None,
    acoustic_analysis_results: dict[str, Any] | None = None,
) -> SpeakerProfilesArtifact:
    """Build authoritative speaker profiles from identity and acoustic evidence.

    The artifact stores source-delivery measurements separately from voice-family
    classification. Voice resolution can therefore use Themba or Thando as a
    family anchor while retaining stable per-speaker pitch and rate differences.
    """
    transcript = read_json(transcript_path)
    diarization = read_json(diarization_path)

    speakers_from_transcript = {
        str(speaker_id)
        for segment in transcript.get("segments", [])
        if (speaker_id := segment.get("speaker_id", segment.get("speaker")))
    }

    speaker_registry: dict[str, Any] = {}
    if speaker_registry_path and speaker_registry_path.is_file():
        try:
            speaker_registry = read_json(speaker_registry_path)
        except Exception:
            speaker_registry = {}

    entity_registry: dict[str, Any] = {}
    if entity_registry_path and entity_registry_path.is_file():
        try:
            entity_registry = read_json(entity_registry_path)
        except Exception:
            entity_registry = {}

    acoustic_analysis = acoustic_analysis_results or {}
    if isinstance(acoustic_analysis.get("speakers"), dict):
        acoustic_analysis = acoustic_analysis["speakers"]

    profiles = []
    for speaker_id in sorted(speakers_from_transcript):
        profile = _resolve_single_speaker(
            speaker_id=speaker_id,
            transcript=transcript,
            diarization=diarization,
            speaker_registry=speaker_registry,
            entity_registry=entity_registry,
            acoustic_analysis=acoustic_analysis,
            job_id=job_id,
        )
        profiles.append(profile.to_dict())

    return SpeakerProfilesArtifact(
        schema_version=SPEAKER_PROFILES_SCHEMA_VERSION,
        job_id=job_id,
        speakers=profiles,
    )


def _resolve_single_speaker(
    speaker_id: str,
    transcript: dict[str, Any],
    diarization: dict[str, Any],
    speaker_registry: dict[str, Any],
    entity_registry: dict[str, Any],
    acoustic_analysis: dict[str, Any],
    job_id: str,
) -> SpeakerProfile:
    """Resolve identity, family and delivery measurements for one speaker."""
    identified_name = None
    identity_status = "unresolved"
    identity_source = None
    identity_confidence = None
    identity_gender = None
    identity_gender_source = None

    if speaker_id in speaker_registry:
        registry_entry = speaker_registry[speaker_id]
        identified_name = registry_entry.get("name")
        identity_status = "resolved"
        identity_source = "authoritative_speaker_registry"
        identity_confidence = registry_entry.get("confidence", 1.0)
        identity_gender = registry_entry.get("gender")
        identity_gender_source = "authoritative_speaker_registry"

    if identity_status == "unresolved" and entity_registry:
        # Person entities remain supporting evidence only; names alone do not bind
        # a diarized speaker to a real person.
        for entity in entity_registry.values():
            if isinstance(entity, dict) and entity.get("type") == "person":
                break

    acoustic_result = acoustic_analysis.get(speaker_id, {})
    if not isinstance(acoustic_result, dict):
        acoustic_result = {}
    acoustic_voice_family = acoustic_result.get("voice_family", "unknown")
    acoustic_confidence = float(acoustic_result.get("confidence", 0.0) or 0.0)
    acoustic_method = acoustic_result.get("evidence", {}).get("method", "unknown")

    if identity_gender in ("male", "female"):
        tts_voice_family = "masculine" if identity_gender == "male" else "feminine"
        tts_voice_family_source = "authoritative_identity_gender"
    elif acoustic_voice_family in ("masculine", "feminine") and acoustic_confidence >= 0.7:
        tts_voice_family = acoustic_voice_family
        tts_voice_family_source = (
            "ecapa_voice_family_classifier"
            if acoustic_method == "ecapa_voice_family_classifier"
            else "cpu_acoustic_analysis"
        )
    else:
        tts_voice_family = "unknown"
        tts_voice_family_source = "insufficient_evidence"

    source_delivery = _source_delivery_features(
        speaker_id=speaker_id,
        transcript=transcript,
        acoustic_result=acoustic_result,
    )

    transcript_speakers = {
        str(value)
        for segment in transcript.get("segments", [])
        if (value := segment.get("speaker_id", segment.get("speaker")))
    }
    diarization_speakers = {
        str(value)
        for turn in diarization.get("turns", [])
        if (value := turn.get("speaker"))
    }
    evidence = {
        "job_id": job_id,
        "speaker_id": speaker_id,
        "transcript_speaker_count": len(transcript_speakers),
        "diarization_speaker_count": len(diarization_speakers),
        "speaker_registry_match": speaker_id in speaker_registry,
        "acoustic_analysis_available": speaker_id in acoustic_analysis,
    }
    if identity_gender:
        evidence["identity_gender"] = identity_gender
    if acoustic_voice_family != "unknown":
        evidence["acoustic_voice_family"] = acoustic_voice_family
        evidence["acoustic_confidence"] = acoustic_confidence

    return SpeakerProfile(
        speaker_id=speaker_id,
        identified_name=identified_name,
        identity_status=identity_status,
        identity_source=identity_source,
        identity_confidence=identity_confidence,
        identity_gender=identity_gender,
        identity_gender_source=identity_gender_source,
        acoustic_voice_family=acoustic_voice_family,
        acoustic_confidence=acoustic_confidence,
        tts_voice_family=tts_voice_family,
        tts_voice_family_source=tts_voice_family_source,
        evidence=evidence,
        source_delivery=source_delivery,
    )


def _source_delivery_features(
    *,
    speaker_id: str,
    transcript: dict[str, Any],
    acoustic_result: dict[str, Any],
) -> dict[str, Any]:
    """Collect conservative source pitch and speech-rate measurements."""
    segments = [
        segment
        for segment in transcript.get("segments", [])
        if str(segment.get("speaker_id", segment.get("speaker")) or "") == speaker_id
    ]
    word_count = 0
    speech_seconds = 0.0
    for segment in segments:
        text = str(
            segment.get("source_text")
            or segment.get("text")
            or segment.get("display_text")
            or ""
        )
        word_count += len(_WORD_RE.findall(text))
        speech_seconds += _segment_duration_seconds(segment)

    result: dict[str, Any] = {
        "source_segment_count": len(segments),
        "source_word_count": word_count,
        "source_speech_seconds": round(speech_seconds, 3),
    }
    if speech_seconds > 0 and word_count > 0:
        result["source_speech_rate_wps"] = round(word_count / speech_seconds, 4)

    for key in (
        "median_f0_hz",
        "f0_percentile_25",
        "f0_percentile_75",
        "voiced_frame_ratio",
        "usable_duration_ms",
        "sample_count",
    ):
        value = acoustic_result.get(key)
        if value is not None:
            result[key] = value
    return result


def _segment_duration_seconds(segment: dict[str, Any]) -> float:
    if segment.get("start_ms") is not None and segment.get("end_ms") is not None:
        return max(0.0, (float(segment["end_ms"]) - float(segment["start_ms"])) / 1000.0)
    if segment.get("start") is not None and segment.get("end") is not None:
        return max(0.0, float(segment["end"]) - float(segment["start"]))
    if segment.get("duration_ms") is not None:
        return max(0.0, float(segment["duration_ms"]) / 1000.0)
    return 0.0


def write_speaker_profiles(
    artifact: SpeakerProfilesArtifact,
    output_path: Path,
) -> None:
    """Write speaker profiles artifact to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = artifact.to_dict()
    data["artifact_sha256"] = checksum_from_dict(data)
    atomic_write_json(output_path, data)


def checksum_from_dict(data: dict[str, Any]) -> str:
    """Compute SHA256 checksum from dictionary."""
    data_copy = {k: v for k, v in data.items() if k != "artifact_sha256"}
    canonical = json.dumps(data_copy, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = [
    "SPEAKER_PROFILES_SCHEMA_VERSION",
    "SpeakerProfile",
    "SpeakerProfilesArtifact",
    "build_speaker_profiles",
    "write_speaker_profiles",
    "checksum_from_dict",
]
