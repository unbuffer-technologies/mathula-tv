"""Tests for Azure voice resolution."""

import json
import pytest

from mathula_tv.voice_resolver import (
    VOICE_ASSIGNMENTS_SCHEMA_VERSION,
    VoiceAssignment,
    VoiceAssignmentsArtifact,
    resolve_azure_voices,
    write_voice_assignments,
    load_voice_assignments,
    validate_assignments_checksum,
    get_voice_for_speaker,
)
from mathula_tv.speaker_profiles import SpeakerProfilesArtifact


def test_voice_assignment_creation():
    """Test VoiceAssignment dataclass creation."""
    assignment = VoiceAssignment(
        speaker_id="SPEAKER_00",
        target_locale="zu-ZA",
        tts_voice_family="masculine",
        selected_voice="zu-ZA-ThembaNeural",
        resolution_status="resolved",
        resolution_reason="resolved_by_voice_family",
        evidence_source="authoritative_identity_gender",
        confidence=0.95,
    )
    
    data = assignment.to_dict()
    assert data["speaker_id"] == "SPEAKER_00"
    assert data["selected_voice"] == "zu-ZA-ThembaNeural"
    assert data["resolution_status"] == "resolved"


def test_voice_assignments_artifact_creation():
    """Test VoiceAssignmentsArtifact creation."""
    artifact = VoiceAssignmentsArtifact(
        schema_version=VOICE_ASSIGNMENTS_SCHEMA_VERSION,
        job_id="test-job",
        target_locale="zu-ZA",
        speaker_profiles_checksum="abc123",
        assignments=[
            {
                "speaker_id": "SPEAKER_00",
                "selected_voice": "zu-ZA-ThembaNeural",
                "resolution_status": "resolved",
            }
        ],
    )
    
    data = artifact.to_dict()
    assert data["schema_version"] == VOICE_ASSIGNMENTS_SCHEMA_VERSION
    assert data["job_id"] == "test-job"
    assert len(data["assignments"]) == 1


def test_voice_assignments_artifact_from_dict():
    """Test VoiceAssignmentsArtifact deserialization."""
    data = {
        "schema_version": VOICE_ASSIGNMENTS_SCHEMA_VERSION,
        "job_id": "test-job",
        "target_locale": "zu-ZA",
        "speaker_profiles_checksum": "abc123",
        "assignments": [],
    }
    
    artifact = VoiceAssignmentsArtifact.from_dict(data)
    assert artifact.job_id == "test-job"
    assert artifact.target_locale == "zu-ZA"


def test_resolve_azure_voices():
    """Test resolving Azure voices from speaker profiles."""
    speaker_profiles = SpeakerProfilesArtifact(
        schema_version="mathula-speaker-profiles-v1",
        job_id="test-job",
        speakers=[
            {
                "speaker_id": "SPEAKER_00",
                "tts_voice_family": "masculine",
                "tts_voice_family_source": "authoritative_identity_gender",
                "acoustic_confidence": 0.95,
            },
            {
                "speaker_id": "SPEAKER_01",
                "tts_voice_family": "feminine",
                "tts_voice_family_source": "authoritative_identity_gender",
                "acoustic_confidence": 0.95,
            },
        ],
    )
    
    artifact = resolve_azure_voices(
        job_id="test-job",
        speaker_profiles=speaker_profiles,
        target_locale="zu-ZA",
        fallback_allowed=False,
        min_confidence=0.5,
    )
    
    assert artifact.job_id == "test-job"
    assert artifact.target_locale == "zu-ZA"
    assert len(artifact.assignments) == 2
    assert len(artifact.unresolved_speakers) == 0
    
    # Check voice assignments
    voice_by_speaker = {a["speaker_id"]: a["selected_voice"] for a in artifact.assignments}
    assert voice_by_speaker["SPEAKER_00"] == "zu-ZA-ThembaNeural"
    assert voice_by_speaker["SPEAKER_01"] == "zu-ZA-ThandoNeural"


def test_resolve_azure_voices_with_fallback():
    """Test voice resolution with fallback."""
    speaker_profiles = SpeakerProfilesArtifact(
        schema_version="mathula-speaker-profiles-v1",
        job_id="test-job",
        speakers=[
            {
                "speaker_id": "SPEAKER_00",
                "tts_voice_family": "unknown",
                "tts_voice_family_source": "insufficient_evidence",
                "acoustic_confidence": 0.0,
            },
        ],
    )
    
    artifact = resolve_azure_voices(
        job_id="test-job",
        speaker_profiles=speaker_profiles,
        target_locale="zu-ZA",
        fallback_allowed=True,
        min_confidence=0.5,
    )
    
    assert len(artifact.assignments) == 1
    assert artifact.assignments[0]["resolution_status"] == "fallback"


def test_resolve_azure_voices_no_fallback():
    """Test error when fallback not allowed."""
    speaker_profiles = SpeakerProfilesArtifact(
        schema_version="mathula-speaker-profiles-v1",
        job_id="test-job",
        speakers=[
            {
                "speaker_id": "SPEAKER_00",
                "tts_voice_family": "unknown",
                "tts_voice_family_source": "insufficient_evidence",
                "acoustic_confidence": 0.0,
            },
        ],
    )
    
    with pytest.raises(ValueError, match="Failed to resolve voices"):
        resolve_azure_voices(
            job_id="test-job",
            speaker_profiles=speaker_profiles,
            target_locale="zu-ZA",
            fallback_allowed=False,
            min_confidence=0.5,
        )


def test_write_and_load_voice_assignments():
    """Test writing and loading voice assignments."""
    import tempfile
    from pathlib import Path
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        output_path = tmpdir / "voice_assignments.json"
        
        artifact = VoiceAssignmentsArtifact(
            schema_version=VOICE_ASSIGNMENTS_SCHEMA_VERSION,
            job_id="test-job",
            target_locale="zu-ZA",
            speaker_profiles_checksum="abc123",
            assignments=[],
        )
        
        write_voice_assignments(artifact, output_path)
        
        assert output_path.exists()
        loaded = load_voice_assignments(output_path)
        assert loaded.job_id == "test-job"
        assert loaded.target_locale == "zu-ZA"


def test_validate_assignments_checksum():
    """Test checksum validation."""
    speaker_profiles = SpeakerProfilesArtifact(
        schema_version="mathula-speaker-profiles-v1",
        job_id="test-job",
        speakers=[],
    )
    
    artifact = resolve_azure_voices(
        job_id="test-job",
        speaker_profiles=speaker_profiles,
        target_locale="zu-ZA",
        fallback_allowed=True,
    )
    
    # Valid checksum
    assert validate_assignments_checksum(artifact, speaker_profiles) is True
    
    # Invalid checksum
    artifact.speaker_profiles_checksum = "invalid"
    assert validate_assignments_checksum(artifact, speaker_profiles) is False


def test_get_voice_for_speaker():
    """Test getting voice for specific speaker."""
    artifact = VoiceAssignmentsArtifact(
        schema_version=VOICE_ASSIGNMENTS_SCHEMA_VERSION,
        job_id="test-job",
        target_locale="zu-ZA",
        speaker_profiles_checksum="abc123",
        assignments=[
            {
                "speaker_id": "SPEAKER_00",
                "selected_voice": "zu-ZA-ThembaNeural",
                "resolution_status": "resolved",
            },
        ],
    )
    
    voice = get_voice_for_speaker("SPEAKER_00", artifact)
    assert voice == "zu-ZA-ThembaNeural"
    
    voice = get_voice_for_speaker("SPEAKER_99", artifact)
    assert voice is None
