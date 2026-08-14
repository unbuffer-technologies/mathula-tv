"""Tests for speaker profile resolution."""

import json
import tempfile
from pathlib import Path

import pytest

from mathula_tv.speaker_profiles import (
    SPEAKER_PROFILES_SCHEMA_VERSION,
    SpeakerProfile,
    SpeakerProfilesArtifact,
    build_speaker_profiles,
    checksum_from_dict,
    write_speaker_profiles,
)


def test_speaker_profile_creation():
    """Test SpeakerProfile dataclass creation."""
    profile = SpeakerProfile(
        speaker_id="SPEAKER_00",
        identified_name="John Doe",
        identity_status="resolved",
        identity_source="authoritative_speaker_registry",
        identity_confidence=1.0,
        identity_gender="male",
        identity_gender_source="authoritative_speaker_registry",
        acoustic_voice_family="masculine",
        acoustic_confidence=0.95,
        tts_voice_family="masculine",
        tts_voice_family_source="authoritative_identity_gender",
        evidence={"test": True},
    )
    
    data = profile.to_dict()
    assert data["speaker_id"] == "SPEAKER_00"
    assert data["identified_name"] == "John Doe"
    assert data["identity_status"] == "resolved"
    assert data["tts_voice_family"] == "masculine"


def test_speaker_profiles_artifact_creation():
    """Test SpeakerProfilesArtifact creation."""
    artifact = SpeakerProfilesArtifact(
        schema_version=SPEAKER_PROFILES_SCHEMA_VERSION,
        job_id="test-job",
        speakers=[
            {
                "speaker_id": "SPEAKER_00",
                "tts_voice_family": "masculine",
                "acoustic_confidence": 0.95,
            }
        ],
    )
    
    data = artifact.to_dict()
    assert data["schema_version"] == SPEAKER_PROFILES_SCHEMA_VERSION
    assert data["job_id"] == "test-job"
    assert len(data["speakers"]) == 1


def test_speaker_profiles_artifact_from_dict():
    """Test SpeakerProfilesArtifact deserialization."""
    data = {
        "schema_version": SPEAKER_PROFILES_SCHEMA_VERSION,
        "job_id": "test-job",
        "speakers": [
            {
                "speaker_id": "SPEAKER_00",
                "tts_voice_family": "masculine",
                "acoustic_confidence": 0.95,
            }
        ],
    }
    
    artifact = SpeakerProfilesArtifact.from_dict(data)
    assert artifact.job_id == "test-job"
    assert len(artifact.speakers) == 1


def test_build_speaker_profiles():
    """Test building speaker profiles from transcript and diarization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create mock transcript
        transcript = {
            "schema_version": "transcript-v1",
            "segments": [
                {"speaker_id": "SPEAKER_00", "start_ms": 0, "end_ms": 5000, "source_text": "Hello"},
                {"speaker_id": "SPEAKER_01", "start_ms": 5000, "end_ms": 10000, "source_text": "World"},
            ],
        }
        transcript_path = tmpdir / "transcript.json"
        transcript_path.write_text(json.dumps(transcript))
        
        # Create mock diarization
        diarization = {
            "schema_version": "diarization-v1",
            "turns": [
                {"speaker": "SPEAKER_00", "start": 0.0, "end": 5.0},
                {"speaker": "SPEAKER_01", "start": 5.0, "end": 10.0},
            ],
        }
        diarization_path = tmpdir / "diarization.json"
        diarization_path.write_text(json.dumps(diarization))
        
        # Build profiles
        artifact = build_speaker_profiles(
            job_id="test-job",
            transcript_path=transcript_path,
            diarization_path=diarization_path,
        )
        
        assert artifact.job_id == "test-job"
        assert len(artifact.speakers) == 2
        speaker_ids = {s["speaker_id"] for s in artifact.speakers}
        assert speaker_ids == {"SPEAKER_00", "SPEAKER_01"}


def test_checksum_from_dict():
    """Test checksum computation."""
    data1 = {"a": 1, "b": 2}
    data2 = {"a": 1, "b": 2}
    data3 = {"a": 1, "b": 3}
    
    assert checksum_from_dict(data1) == checksum_from_dict(data2)
    assert checksum_from_dict(data1) != checksum_from_dict(data3)


def test_write_speaker_profiles():
    """Test writing speaker profiles to disk."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        output_path = tmpdir / "speaker_profiles.json"
        
        artifact = SpeakerProfilesArtifact(
            schema_version=SPEAKER_PROFILES_SCHEMA_VERSION,
            job_id="test-job",
            speakers=[],
        )
        
        write_speaker_profiles(artifact, output_path)
        
        assert output_path.exists()
        data = json.loads(output_path.read_text())
        assert data["job_id"] == "test-job"
        assert "artifact_sha256" in data



def test_build_speaker_profiles_records_source_delivery_measurements(tmp_path):
    transcript = {
        "segments": [
            {
                "speaker_id": "SPEAKER_00",
                "start_ms": 0,
                "end_ms": 2000,
                "source_text": "one two three four",
            },
            {
                "speaker_id": "SPEAKER_00",
                "start_ms": 3000,
                "end_ms": 5000,
                "source_text": "five six",
            },
        ]
    }
    diarization = {"turns": [{"speaker": "SPEAKER_00", "start": 0.0, "end": 5.0}]}
    transcript_path = tmp_path / "transcript.json"
    diarization_path = tmp_path / "diarization.json"
    transcript_path.write_text(json.dumps(transcript), encoding="utf-8")
    diarization_path.write_text(json.dumps(diarization), encoding="utf-8")

    artifact = build_speaker_profiles(
        "job",
        transcript_path,
        diarization_path,
        acoustic_analysis_results={
            "SPEAKER_00": {
                "voice_family": "masculine",
                "confidence": 0.9,
                "median_f0_hz": 128.0,
                "f0_percentile_25": 110.0,
                "f0_percentile_75": 145.0,
                "voiced_frame_ratio": 0.75,
                "usable_duration_ms": 4000,
                "sample_count": 100,
                "evidence": {},
            }
        },
    )
    delivery = artifact.speakers[0]["source_delivery"]
    assert delivery["source_word_count"] == 6
    assert delivery["source_speech_seconds"] == 4.0
    assert delivery["source_speech_rate_wps"] == 1.5
    assert delivery["median_f0_hz"] == 128.0
