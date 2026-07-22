"""Authoritative speaker context resolution for Azure TTS dubbing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .atomic_io import atomic_write_json, read_json
from .media import checksum


SPEAKER_PROFILES_SCHEMA_VERSION = "mathula-speaker-profiles-v1"


@dataclass(frozen=True)
class SpeakerProfile:
    """Authoritative speaker context for voice resolution."""
    
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
    """Build authoritative speaker profiles from available context.
    
    Resolution priority:
    1. Authoritative speaker registry/seeds
    2. Entity/person metadata
    3. Job-local speaker metadata (filename, etc.)
    4. CPU-only acoustic analysis
    5. Unresolved (fallback)
    
    Args:
        job_id: Job identifier
        transcript_path: Path to transcript_en.json
        diarization_path: Path to authoritative diarization (azure_diarization.json)
        speaker_registry_path: Optional path to authoritative speaker registry
        entity_registry_path: Optional path to entity registry
        acoustic_analysis_results: Optional CPU-only acoustic analysis results
    
    Returns:
        SpeakerProfilesArtifact with resolved speaker context
    """
    transcript = read_json(transcript_path)
    diarization = read_json(diarization_path)
    
    # Extract unique speakers from transcript
    speakers_from_transcript = set()
    for segment in transcript.get("segments", []):
        speaker_id = segment.get("speaker_id", segment.get("speaker"))
        if speaker_id:
            speakers_from_transcript.add(str(speaker_id))
    
    # Load authoritative sources if available
    speaker_registry = {}
    if speaker_registry_path and speaker_registry_path.is_file():
        try:
            speaker_registry = read_json(speaker_registry_path)
        except Exception:
            speaker_registry = {}
    
    entity_registry = {}
    if entity_registry_path and entity_registry_path.is_file():
        try:
            entity_registry = read_json(entity_registry_path)
        except Exception:
            entity_registry = {}
    
    # Build profiles for each speaker
    profiles = []
    for speaker_id in sorted(speakers_from_transcript):
        profile = _resolve_single_speaker(
            speaker_id=speaker_id,
            transcript=transcript,
            diarization=diarization,
            speaker_registry=speaker_registry,
            entity_registry=entity_registry,
            acoustic_analysis=acoustic_analysis_results or {},
            job_id=job_id,
        )
        profiles.append(profile.to_dict())
    
    artifact = SpeakerProfilesArtifact(
        schema_version=SPEAKER_PROFILES_SCHEMA_VERSION,
        job_id=job_id,
        speakers=profiles,
    )
    
    return artifact


def _resolve_single_speaker(
    speaker_id: str,
    transcript: dict[str, Any],
    diarization: dict[str, Any],
    speaker_registry: dict[str, Any],
    entity_registry: dict[str, Any],
    acoustic_analysis: dict[str, Any],
    job_id: str,
) -> SpeakerProfile:
    """Resolve context for a single speaker."""
    
    # Start with unresolved defaults
    identified_name = None
    identity_status = "unresolved"
    identity_source = None
    identity_confidence = None
    identity_gender = None
    identity_gender_source = None
    
    # 1. Check speaker registry (highest priority)
    if speaker_id in speaker_registry:
        registry_entry = speaker_registry[speaker_id]
        identified_name = registry_entry.get("name")
        identity_status = "resolved"
        identity_source = "authoritative_speaker_registry"
        identity_confidence = registry_entry.get("confidence", 1.0)
        identity_gender = registry_entry.get("gender")
        identity_gender_source = "authoritative_speaker_registry"
    
    # 2. Check entity registry for person entities
    if identity_status == "unresolved" and entity_registry:
        # Look for person entities that might be this speaker
        # This is heuristic - we don't assume name presence = speaker identity
        # Just record as potential evidence
        for entity_id, entity in entity_registry.items():
            if entity.get("type") == "person":
                # Record as evidence but don't resolve identity from name alone
                pass
    
    # 3. Check job-local metadata (filename, etc.)
    # Do not infer identity from filename text alone
    
    # 4. Acoustic analysis (CPU-only)
    acoustic_result = acoustic_analysis.get(speaker_id, {})
    acoustic_voice_family = acoustic_result.get("voice_family", "unknown")
    acoustic_confidence = acoustic_result.get("confidence", 0.0)
    
    # 5. Determine TTS voice family
    # Priority: authoritative gender > acoustic analysis > unknown
    if identity_gender in ("male", "female"):
        tts_voice_family = "masculine" if identity_gender == "male" else "feminine"
        tts_voice_family_source = "authoritative_identity_gender"
    elif acoustic_voice_family in ("masculine", "feminine") and acoustic_confidence >= 0.7:
        tts_voice_family = acoustic_voice_family
        tts_voice_family_source = "cpu_acoustic_analysis"
    else:
        tts_voice_family = "unknown"
        tts_voice_family_source = "insufficient_evidence"
    
    # Build evidence record
    evidence = {
        "job_id": job_id,
        "speaker_id": speaker_id,
        "transcript_speaker_count": len(set(
            s.get("speaker_id", s.get("speaker")) 
            for s in transcript.get("segments", [])
        )),
        "diarization_speaker_count": len(set(
            t.get("speaker") for t in diarization.get("turns", [])
        )),
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
    )


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
    # Remove checksum field if present to avoid circular dependency
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
