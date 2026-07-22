"""Deterministic Azure voice resolution from speaker context."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .atomic_io import atomic_write_json, read_json
from .azure_voice_catalogue import get_catalogue, resolve_voice_for_family
from .media import checksum
from .speaker_profiles import SpeakerProfilesArtifact


VOICE_ASSIGNMENTS_SCHEMA_VERSION = "azure-voice-assignments-v1"


@dataclass
class VoiceAssignment:
    """Azure voice assignment for a single speaker."""
    
    speaker_id: str
    target_locale: str
    tts_voice_family: str
    selected_voice: str
    resolution_status: str  # "resolved" | "fallback" | "failed"
    resolution_reason: str
    evidence_source: str
    confidence: float
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "speaker_id": self.speaker_id,
            "target_locale": self.target_locale,
            "tts_voice_family": self.tts_voice_family,
            "selected_voice": self.selected_voice,
            "resolution_status": self.resolution_status,
            "resolution_reason": self.resolution_reason,
            "evidence_source": self.evidence_source,
            "confidence": self.confidence,
        }


@dataclass
class VoiceAssignmentsArtifact:
    """Complete voice assignments for a job."""
    
    schema_version: str = VOICE_ASSIGNMENTS_SCHEMA_VERSION
    job_id: str = ""
    target_locale: str = ""
    speaker_profiles_checksum: str = ""
    assignments: list[dict[str, Any]] = field(default_factory=list)
    unresolved_speakers: list[str] = field(default_factory=list)
    diagnostic_info: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "job_id": self.job_id,
            "target_locale": self.target_locale,
            "speaker_profiles_checksum": self.speaker_profiles_checksum,
            "assignments": self.assignments,
            "unresolved_speakers": self.unresolved_speakers,
            "diagnostic_info": self.diagnostic_info,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VoiceAssignmentsArtifact":
        return cls(
            schema_version=data.get("schema_version", VOICE_ASSIGNMENTS_SCHEMA_VERSION),
            job_id=data.get("job_id", ""),
            target_locale=data.get("target_locale", ""),
            speaker_profiles_checksum=data.get("speaker_profiles_checksum", ""),
            assignments=data.get("assignments", []),
            unresolved_speakers=data.get("unresolved_speakers", []),
            diagnostic_info=data.get("diagnostic_info", {}),
        )


def resolve_azure_voices(
    job_id: str,
    speaker_profiles: SpeakerProfilesArtifact,
    target_locale: str,
    fallback_allowed: bool = False,
    min_confidence: float = 0.5,
) -> VoiceAssignmentsArtifact:
    """Resolve Azure voices for all speakers from speaker profiles.
    
    This is a deterministic, server-owned voice resolution process.
    Each speaker gets exactly one stable voice assignment.
    
    Args:
        job_id: Job identifier
        speaker_profiles: Speaker profiles artifact
        target_locale: Target locale (e.g., "zu-ZA")
        fallback_allowed: If True, allow fallback to any available voice
        min_confidence: Minimum confidence threshold for resolution
    
    Returns:
        VoiceAssignmentsArtifact with voice assignments
    
    Raises:
        ValueError: If any speaker cannot be resolved and fallback is not allowed
    """
    # Validate catalogue supports locale
    try:
        catalogue = get_catalogue(target_locale)
    except ValueError as exc:
        raise ValueError(f"Unsupported target locale '{target_locale}': {exc}") from exc
    
    assignments = []
    unresolved_speakers = []
    diagnostic_info = {
        "catalogue_locale": target_locale,
        "available_families": sorted(catalogue.available_families()),
        "available_voices": [v.name for v in catalogue.voices],
        "total_speakers": len(speaker_profiles.speakers),
    }
    
    for speaker_profile in speaker_profiles.speakers:
        speaker_id = speaker_profile["speaker_id"]
        tts_voice_family = speaker_profile.get("tts_voice_family", "unknown")
        tts_voice_family_source = speaker_profile.get("tts_voice_family_source", "unknown")
        confidence = speaker_profile.get("acoustic_confidence", 0.0)
        
        # Determine evidence source
        evidence_source = speaker_profile.get("tts_voice_family_source", "unknown")
        
        # Try to resolve voice
        try:
            if tts_voice_family == "unknown":
                # Cannot resolve without voice family
                if fallback_allowed:
                    voice_name, resolution_reason = resolve_voice_for_family(
                        target_locale,
                        "masculine",  # Default fallback
                        fallback_allowed=True,
                    )
                    resolution_status = "fallback"
                    confidence = 0.1  # Low confidence for fallback
                else:
                    raise ValueError(
                        f"Speaker {speaker_id} has unknown voice family and fallback is not allowed"
                    )
            else:
                voice_name, resolution_reason = resolve_voice_for_family(
                    target_locale,
                    tts_voice_family,
                    fallback_allowed=fallback_allowed,
                )
                resolution_status = "resolved"
            
            # Check confidence threshold
            if resolution_status == "resolved" and confidence < min_confidence:
                if fallback_allowed:
                    voice_name, resolution_reason = resolve_voice_for_family(
                        target_locale,
                        tts_voice_family,
                        fallback_allowed=True,
                    )
                    resolution_status = "fallback"
                else:
                    raise ValueError(
                        f"Speaker {speaker_id} confidence {confidence} below threshold {min_confidence}"
                    )
            
            assignment = VoiceAssignment(
                speaker_id=speaker_id,
                target_locale=target_locale,
                tts_voice_family=tts_voice_family,
                selected_voice=voice_name,
                resolution_status=resolution_status,
                resolution_reason=resolution_reason,
                evidence_source=evidence_source,
                confidence=confidence,
            )
            assignments.append(assignment.to_dict())
            
        except ValueError as exc:
            # Resolution failed
            unresolved_speakers.append(speaker_id)
            diagnostic_info.setdefault("resolution_failures", {})[speaker_id] = {
                "reason": str(exc),
                "tts_voice_family": tts_voice_family,
                "confidence": confidence,
                "evidence_source": evidence_source,
            }
    
    # Check if all speakers resolved
    if unresolved_speakers and not fallback_allowed:
        raise ValueError(
            f"Failed to resolve voices for {len(unresolved_speakers)} speaker(s): "
            f"{', '.join(unresolved_speakers)}. "
            f"Use fallback_allowed=True to enable fallback resolution."
        )
    
    # Compute speaker profiles checksum
    speaker_profiles_checksum = _compute_checksum(speaker_profiles.to_dict())
    
    artifact = VoiceAssignmentsArtifact(
        schema_version=VOICE_ASSIGNMENTS_SCHEMA_VERSION,
        job_id=job_id,
        target_locale=target_locale,
        speaker_profiles_checksum=speaker_profiles_checksum,
        assignments=assignments,
        unresolved_speakers=unresolved_speakers,
        diagnostic_info=diagnostic_info,
    )
    
    return artifact


def write_voice_assignments(
    artifact: VoiceAssignmentsArtifact,
    output_path: Path,
) -> None:
    """Write voice assignments artifact to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = artifact.to_dict()
    data["artifact_sha256"] = _compute_checksum(data)
    atomic_write_json(output_path, data)


def load_voice_assignments(path: Path) -> VoiceAssignmentsArtifact:
    """Load voice assignments artifact from disk."""
    data = read_json(path)
    return VoiceAssignmentsArtifact.from_dict(data)


def validate_assignments_checksum(
    artifact: VoiceAssignmentsArtifact,
    speaker_profiles: SpeakerProfilesArtifact,
) -> bool:
    """Validate that voice assignments match speaker profiles checksum.
    
    Args:
        artifact: Voice assignments artifact
        speaker_profiles: Speaker profiles artifact
    
    Returns:
        True if checksums match, False otherwise
    """
    expected_checksum = _compute_checksum(speaker_profiles.to_dict())
    return artifact.speaker_profiles_checksum == expected_checksum


def get_voice_for_speaker(
    speaker_id: str,
    assignments: VoiceAssignmentsArtifact,
) -> str | None:
    """Get the assigned Azure voice for a specific speaker.
    
    Args:
        speaker_id: Speaker identifier
        assignments: Voice assignments artifact
    
    Returns:
        Voice name if assigned, None if speaker not found
    """
    for assignment in assignments.assignments:
        if assignment["speaker_id"] == speaker_id:
            return assignment["selected_voice"]
    return None


def _compute_checksum(data: dict[str, Any]) -> str:
    """Compute SHA256 checksum from dictionary."""
    # Remove checksum field if present to avoid circular dependency
    data_copy = {k: v for k, v in data.items() if k not in ("artifact_sha256", "speaker_profiles_checksum")}
    canonical = json.dumps(data_copy, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = [
    "VOICE_ASSIGNMENTS_SCHEMA_VERSION",
    "VoiceAssignment",
    "VoiceAssignmentsArtifact",
    "resolve_azure_voices",
    "write_voice_assignments",
    "load_voice_assignments",
    "validate_assignments_checksum",
    "get_voice_for_speaker",
]
