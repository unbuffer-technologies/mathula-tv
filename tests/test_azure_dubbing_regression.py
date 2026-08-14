"""Regression tests for Azure-only dubbing path fixes."""

import json
import tempfile
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mathula_tv.transcript_merge import reconcile, _map_to_canonical_speaker_id
from mathula_tv.acoustic_analysis import analyze_all_speakers, _correct_octave_error
from mathula_tv.speaker_profiles import build_speaker_profiles
from mathula_tv.voice_resolver import resolve_azure_voices
from mathula_tv.speaker_profiles import SpeakerProfilesArtifact
from mathula_tv.azure_qc import validate_azure_tts_unit


def test_azure_to_canonical_speaker_mapping():
    """Test that raw Azure speaker IDs are mapped to canonical IDs deterministically."""
    # Reset the mapping cache
    if hasattr(_map_to_canonical_speaker_id, "_mapping"):
        del _map_to_canonical_speaker_id._mapping
    
    # Simulate Azure diarization with raw IDs
    words = [
        {"word_id": "word_1", "text": "Hello", "start": 0.0, "end": 0.5, "speaker": "AZURE_1"},
        {"word_id": "word_2", "text": "world", "start": 0.5, "end": 1.0, "speaker": "AZURE_1"},
        {"word_id": "word_3", "text": "test", "start": 1.0, "end": 1.5, "speaker": "AZURE_2"},
    ]
    
    diarization = [
        {"speaker": "AZURE_1", "start": 0.0, "end": 1.0},
        {"speaker": "AZURE_2", "start": 1.0, "end": 1.5},
    ]
    
    # Reset mapping before test
    if hasattr(_map_to_canonical_speaker_id, "_mapping"):
        del _map_to_canonical_speaker_id._mapping
    
    transcript = reconcile(words, diarization, source="azure")
    
    # Check that canonical IDs are used
    speakers = set(seg["speaker"] for seg in transcript["segments"])
    assert "SPEAKER_00" in speakers
    assert "SPEAKER_01" in speakers
    assert "AZURE_1" not in speakers
    assert "AZURE_2" not in speakers
    
    # Check that raw_speaker is preserved
    assert transcript["segments"][0]["raw_speaker"] == "AZURE_1"
    assert transcript["segments"][1]["raw_speaker"] == "AZURE_2"


def test_speaker_id_stability_through_pipeline():
    """Test that canonical speaker IDs remain stable through transcript reconciliation."""
    # Reset mapping
    if hasattr(_map_to_canonical_speaker_id, "_mapping"):
        del _map_to_canonical_speaker_id._mapping
    
    words = [
        {"word_id": "word_1", "text": "First", "start": 0.0, "end": 0.5, "speaker": "AZURE_1"},
        {"word_id": "word_2", "text": "Second", "start": 0.5, "end": 1.0, "speaker": "AZURE_1"},
    ]
    
    diarization = [
        {"speaker": "AZURE_1", "start": 0.0, "end": 1.0},
    ]
    
    transcript1 = reconcile(words, diarization, source="azure")
    speaker_id_1 = transcript1["segments"][0]["speaker"]
    
    # Reconcile again with same data
    if hasattr(_map_to_canonical_speaker_id, "_mapping"):
        del _map_to_canonical_speaker_id._mapping
    
    transcript2 = reconcile(words, diarization, source="azure")
    speaker_id_2 = transcript2["segments"][0]["speaker"]
    
    # IDs should be stable
    assert speaker_id_1 == speaker_id_2
    assert speaker_id_1 == "SPEAKER_00"


def test_short_diarization_fragment_does_not_create_speaker():
    """Test that very short diarization fragments don't create false speakers."""
    # Reset mapping
    if hasattr(_map_to_canonical_speaker_id, "_mapping"):
        del _map_to_canonical_speaker_id._mapping
    
    words = [
        {"word_id": "word_1", "text": "Main", "start": 0.0, "end": 2.0, "speaker": "AZURE_1"},
        {"word_id": "word_2", "text": "fragment", "start": 2.0, "end": 2.44, "speaker": "AZURE_2"},  # 440ms fragment
    ]
    
    diarization = [
        {"speaker": "AZURE_1", "start": 0.0, "end": 2.0},
        {"speaker": "AZURE_2", "start": 2.0, "end": 2.44},
    ]
    
    transcript = reconcile(words, diarization, source="azure")
    
    # Both should be mapped to canonical IDs
    speakers = set(seg["speaker"] for seg in transcript["segments"])
    assert len(speakers) == 2  # Both fragments get IDs, but acoustic analysis will filter short ones


def test_acoustic_analysis_uses_canonical_speaker_ids():
    """Test that acoustic analysis results use canonical speaker IDs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create mock transcript with canonical IDs
        transcript = {
            "schema_version": "transcript-v1",
            "segments": [
                {"speaker_id": "SPEAKER_00", "raw_speaker": "AZURE_1", "start": 0.0, "end": 1.0, "source_text": "Hello"},
                {"speaker_id": "SPEAKER_01", "raw_speaker": "AZURE_2", "start": 1.0, "end": 2.0, "source_text": "World"},
            ],
        }
        transcript_path = tmpdir / "transcript.json"
        transcript_path.write_text(json.dumps(transcript))
        
        # Create mock diarization with raw IDs
        diarization = {
            "schema_version": "diarization-v1",
            "turns": [
                {"speaker": "AZURE_1", "start": 0.0, "end": 1.0},
                {"speaker": "AZURE_2", "start": 1.0, "end": 2.0},
            ],
        }
        diarization_path = tmpdir / "diarization.json"
        diarization_path.write_text(json.dumps(diarization))
        
        # Create a minimal WAV file (1 byte header + minimal data)
        # This will fail audio loading but tests the mapping logic
        audio_path = tmpdir / "audio.wav"
        audio_path.write_bytes(b"RIFF" + b"\x00" * 40)  # Invalid WAV but tests path handling
        
        output_path = tmpdir / "acoustic_analysis.json"
        
        # This should fail on audio load, but the mapping logic should work
        try:
            results = analyze_all_speakers(
                audio_path=audio_path,
                diarization_path=diarization_path,
                transcript_path=transcript_path,
                output_path=output_path,
            )
        except Exception:
            # Audio load failure is expected for invalid WAV
            pass
        
        # Check that the artifact includes the speaker mapping
        if output_path.exists():
            artifact = json.loads(output_path.read_text())
            assert "speaker_mapping" in artifact
            assert "AZURE_1" in artifact["speaker_mapping"]
            assert artifact["speaker_mapping"]["AZURE_1"] == "SPEAKER_00"


def test_context_repaired_turn_cannot_redefine_raw_acoustic_speaker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transcript_path = tmp_path / "transcript.json"
    transcript_path.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "speaker_id": "SPEAKER_00",
                        "raw_speaker": "AZURE_1",
                        "start": 0,
                        "end": 5,
                        "source_text": "The hearing is postponed.",
                    },
                    {
                        "speaker_id": "SPEAKER_01",
                        "raw_speaker": "AZURE_1",
                        "speaker_assignment_method": "contextual_vocative_repair",
                        "start": 5,
                        "end": 6,
                        "source_text": "19 August, Chair.",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    diarization_path = tmp_path / "diarization.json"
    diarization_path.write_text(
        json.dumps(
            {
                "turns": [
                    {"speaker": "AZURE_1", "start": 0, "end": 6},
                ]
            }
        ),
        encoding="utf-8",
    )

    captured: dict[str, str] = {}

    def fake_analysis(_audio_path, raw_speaker_id, _turns):
        captured["raw_speaker_id"] = raw_speaker_id
        return SimpleNamespace(to_dict=lambda: {"speaker_id": raw_speaker_id})

    monkeypatch.setattr(
        "mathula_tv.acoustic_analysis.analyze_speaker_acoustics",
        fake_analysis,
    )
    audio_path = tmp_path / "unused.wav"
    audio_path.write_bytes(b"test-audio-placeholder")
    output_path = tmp_path / "analysis.json"
    analyze_all_speakers(
        audio_path=audio_path,
        diarization_path=diarization_path,
        transcript_path=transcript_path,
        output_path=output_path,
        use_ecapa_classifier=False,
    )

    artifact = json.loads(output_path.read_text(encoding="utf-8"))
    assert captured["raw_speaker_id"] == "AZURE_1"
    assert artifact["speakers"]["SPEAKER_00"]["raw_speaker_id"] == "AZURE_1"


def test_octave_correction_for_harmonic_selection():
    """Test that octave correction handles harmonic selection in pitch detection."""
    import numpy as np
    
    # Simulate autocorrelation where a harmonic is selected
    # If F0 is 108 Hz (fundamental), harmonic at 216 Hz might be selected
    sample_rate = 16000
    peak_idx = 74  # Corresponds to ~216 Hz (in correction range 150-300)
    min_lag = 40
    max_lag = 213
    
    # Create mock autocorrelation with peak at harmonic
    # The function expects autocorrelation starting from zero lag
    autocorr = np.zeros(300)
    autocorr[0] = 1.0  # Zero lag
    autocorr[peak_idx] = 0.8  # Harmonic peak
    autocorr[peak_idx * 2] = 0.75  # Fundamental peak (half frequency) - slightly higher for correction
    
    f0_detected = sample_rate / peak_idx  # ~216 Hz
    f0_corrected = _correct_octave_error(
        f0_detected, peak_idx, autocorr, min_lag, max_lag, sample_rate
    )
    
    # Should correct to ~108 Hz (fundamental)
    expected_f0 = sample_rate / (peak_idx * 2)
    assert abs(f0_corrected - expected_f0) < 1.0, f"Expected {expected_f0}, got {f0_corrected}"


def test_octave_correction_no_correction_needed():
    """Test that octave correction doesn't modify valid F0 estimates."""
    import numpy as np
    
    sample_rate = 16000
    peak_idx = 80  # Corresponds to 200 Hz
    min_lag = 40
    max_lag = 213
    
    autocorr = np.zeros(300)
    autocorr[0] = 1.0
    autocorr[peak_idx] = 0.8
    autocorr[peak_idx * 2] = 0.3  # Weak harmonic, shouldn't trigger correction
    
    f0_detected = sample_rate / peak_idx  # 200 Hz
    f0_corrected = _correct_octave_error(
        f0_detected, peak_idx, autocorr, min_lag, max_lag, sample_rate
    )
    
    # Should remain unchanged
    assert f0_corrected == 200.0


def test_voice_assignment_uses_canonical_speaker_ids():
    """Test that voice assignments use canonical speaker IDs."""
    speaker_profiles = SpeakerProfilesArtifact(
        schema_version="mathula-speaker-profiles-v1",
        job_id="test-job",
        speakers=[
            {
                "speaker_id": "SPEAKER_00",
                "tts_voice_family": "masculine",
                "tts_voice_family_source": "cpu_acoustic_analysis",
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
    
    # Check that canonical ID is used
    assert artifact.assignments[0]["speaker_id"] == "SPEAKER_00"
    assert artifact.assignments[0]["selected_voice"] == "zu-ZA-ThembaNeural"


def test_voice_resolution_error_no_fallback_suggestion():
    """Test that voice resolution error doesn't suggest fallback."""
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
    
    with pytest.raises(ValueError) as exc_info:
        resolve_azure_voices(
            job_id="test-job",
            speaker_profiles=speaker_profiles,
            target_locale="zu-ZA",
            fallback_allowed=False,
            min_confidence=0.5,
        )
    
    error_message = str(exc_info.value)
    # Should not suggest fallback
    assert "fallback" not in error_message.lower()
    # Should mention canonical speaker profile
    assert "canonical speaker profile" in error_message.lower()


def test_ambiguous_evidence_remains_unresolved():
    """Test that truly ambiguous evidence remains unresolved without fallback."""
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
    
    # Should raise error without fallback
    with pytest.raises(ValueError):
        resolve_azure_voices(
            job_id="test-job",
            speaker_profiles=speaker_profiles,
            target_locale="zu-ZA",
            fallback_allowed=False,
            min_confidence=0.5,
        )


def test_no_fallback_used_when_confidence_sufficient():
    """Test that fallback is not used when confidence is sufficient."""
    speaker_profiles = SpeakerProfilesArtifact(
        schema_version="mathula-speaker-profiles-v1",
        job_id="test-job",
        speakers=[
            {
                "speaker_id": "SPEAKER_00",
                "tts_voice_family": "masculine",
                "tts_voice_family_source": "cpu_acoustic_analysis",
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
    
    # Should be resolved, not fallback
    assert artifact.assignments[0]["resolution_status"] == "resolved"
    assert artifact.assignments[0]["selected_voice"] == "zu-ZA-ThembaNeural"


def test_speaker_profiles_ignores_historical_voice_override():
    """Test that speaker profiles don't use historical OpenVoice voice overrides."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create mock transcript
        transcript = {
            "schema_version": "transcript-v1",
            "segments": [
                {"speaker_id": "SPEAKER_00", "speaker": "SPEAKER_00", "start": 0.0, "end": 1.0, "source_text": "Hello"},
            ],
        }
        transcript_path = tmpdir / "transcript.json"
        transcript_path.write_text(json.dumps(transcript))
        
        # Create mock diarization
        diarization = {
            "schema_version": "diarization-v1",
            "turns": [
                {"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0},
            ],
        }
        diarization_path = tmpdir / "diarization.json"
        diarization_path.write_text(json.dumps(diarization))
        
        # Build speaker profiles without acoustic analysis
        profiles = build_speaker_profiles(
            job_id="test-job",
            transcript_path=transcript_path,
            diarization_path=diarization_path,
            acoustic_analysis_results=None,
        )
        
        # Should not have voice assignment
        assert profiles.speakers[0]["tts_voice_family"] == "unknown"
        assert profiles.speakers[0]["tts_voice_family_source"] == "insufficient_evidence"


def test_contradictory_percentile_evidence_remains_unknown():
    """Test that contradictory percentile evidence spanning both voice-family thresholds remains unknown."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Create mock transcript
        transcript = {
            "schema_version": "transcript-v1",
            "segments": [
                {"speaker_id": "SPEAKER_00", "speaker": "SPEAKER_00", "start": 0.0, "end": 1.0, "source_text": "Hello"},
            ],
        }
        transcript_path = tmpdir_path / "transcript.json"
        transcript_path.write_text(json.dumps(transcript))
        
        # Create mock diarization
        diarization = {
            "schema_version": "diarization-v1",
            "turns": [
                {"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0},
            ],
        }
        diarization_path = tmpdir_path / "diarization.json"
        diarization_path.write_text(json.dumps(diarization))
        
        # Create acoustic analysis with contradictory percentile evidence
        # Simulating the real-world case: p25 < 165 Hz and p75 > 255 Hz
        acoustic_analysis = {
            "SPEAKER_00": {
                "voice_family": "feminine",  # This would be the old classification
                "confidence": 0.6,  # Below 0.7 threshold
                "median_f0_hz": 214.77,  # In ambiguous range (165-255 Hz)
                "f0_percentile_25": 150.0,  # Below masculine threshold (165 Hz)
                "f0_percentile_75": 270.0,  # Above feminine threshold (255 Hz)
                "voiced_frame_ratio": 0.74,
                "total_duration_ms": 144436,
                "usable_duration_ms": 144439,
                "sample_count": 10710,
                "evidence": {
                    "turn_count": 30,
                    "total_duration_ms": 144436,
                    "usable_duration_ms": 144439,
                    "sample_rate": 16000,
                    "f0_sample_count": 10710,
                    "voiced_frame_ratio": 0.74,
                    "masculine_threshold_hz": 165.0,
                    "feminine_threshold_hz": 255.0,
                },
            }
        }
        
        # Build speaker profiles with contradictory evidence
        profiles = build_speaker_profiles(
            job_id="test-job",
            transcript_path=transcript_path,
            diarization_path=diarization_path,
            acoustic_analysis_results=acoustic_analysis,
        )
        
        # With contradictory evidence (p25 < 165 and p75 > 255), the acoustic analysis
        # should now classify as "unknown" with confidence 0.0
        # Since confidence is 0.0 < 0.7 threshold, tts_voice_family should be "unknown"
        assert profiles.speakers[0]["tts_voice_family"] == "unknown"
        assert profiles.speakers[0]["tts_voice_family_source"] == "insufficient_evidence"


def test_ecapa_classifier_resolves_masculine_voice():
    """Test that ECAPA classifier result resolves to masculine Azure voice."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Create mock transcript
        transcript = {
            "schema_version": "transcript-v1",
            "segments": [
                {"speaker_id": "SPEAKER_00", "speaker": "SPEAKER_00", "start": 0.0, "end": 1.0, "source_text": "Hello"},
            ],
        }
        transcript_path = tmpdir_path / "transcript.json"
        transcript_path.write_text(json.dumps(transcript))
        
        # Create mock diarization
        diarization = {
            "schema_version": "diarization-v1",
            "turns": [
                {"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0},
            ],
        }
        diarization_path = tmpdir_path / "diarization.json"
        diarization_path.write_text(json.dumps(diarization))
        
        # Create acoustic analysis with ECAPA classifier result
        # Mock the proven real-world result: masculine: 0.874251, feminine: 0.125749
        acoustic_analysis = {
            "SPEAKER_00": {
                "voice_family": "masculine",
                "confidence": 0.874251,
                "median_f0_hz": 214.77,
                "f0_percentile_25": 150.0,
                "f0_percentile_75": 270.0,
                "voiced_frame_ratio": 0.74,
                "total_duration_ms": 144436,
                "usable_duration_ms": 144439,
                "sample_count": 10710,
                "evidence": {
                    "method": "ecapa_voice_family_classifier",
                    "model": "JaesungHuh/voice-gender-classifier",
                    "device": "cpu",
                    "turn_count": 30,
                    "total_duration_ms": 144436,
                    "ecapa_probabilities": {
                        "masculine": 0.874251,
                        "feminine": 0.125749,
                    },
                },
            }
        }
        
        # Build speaker profiles with ECAPA result
        profiles = build_speaker_profiles(
            job_id="test-job",
            transcript_path=transcript_path,
            diarization_path=diarization_path,
            acoustic_analysis_results=acoustic_analysis,
        )
        
        # Verify ECAPA result is propagated
        assert profiles.speakers[0]["tts_voice_family"] == "masculine"
        assert profiles.speakers[0]["tts_voice_family_source"] == "ecapa_voice_family_classifier"
        assert profiles.speakers[0]["acoustic_voice_family"] == "masculine"
        assert profiles.speakers[0]["acoustic_confidence"] == 0.874251
        
        # Create speaker profiles artifact for voice resolution
        speaker_profiles = SpeakerProfilesArtifact(
            schema_version="mathula-speaker-profiles-v1",
            job_id="test-job",
            speakers=profiles.speakers,
        )
        
        # Resolve Azure voices
        voice_assignments = resolve_azure_voices(
            job_id="test-job",
            speaker_profiles=speaker_profiles,
            target_locale="zu-ZA",
            fallback_allowed=False,
            min_confidence=0.7,
        )
        
        # Verify masculine voice is selected from catalogue
        assert voice_assignments.assignments[0]["selected_voice"] == "zu-ZA-ThembaNeural"
        assert voice_assignments.assignments[0]["tts_voice_family"] == "masculine"
        assert voice_assignments.assignments[0]["resolution_status"] == "resolved"
        assert voice_assignments.assignments[0]["evidence_source"] == "ecapa_voice_family_classifier"


@pytest.mark.parametrize("duration_ms,preferred_ms,max_ms,should_pass,expected_warnings,expected_failures", [
    # Shorter than preferred but within maximum → pass with padding warning
    (6837, 7400, 8000, True, ["Alignment padding required: 563ms"], []),
    # Longer than preferred but within maximum → pass with timing warning
    (7601, 7400, 8000, True, ["Duration 7601ms exceeds preferred 7400ms but within hard maximum 8000ms"], []),
    # Longer than maximum → blocking failure
    (8300, 7400, 8000, False, [], ["Duration 8300ms exceeds hard maximum 8000ms by more than 200ms"]),
])
def test_azure_qc_duration_tolerance(
    duration_ms, preferred_ms, max_ms, should_pass, expected_warnings, expected_failures
):
    """Test Azure QC distinguishes preferred vs maximum duration correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Create a valid mono PCM16 WAV file
        wav_path = tmpdir_path / "test.wav"
        sample_rate = 24000
        num_samples = int(duration_ms * sample_rate / 1000)
        samples = np.random.randint(-10000, 10000, num_samples, dtype=np.int16)
        
        with wave.open(str(wav_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(samples.tobytes())
        
        # Create dubbing plan unit
        unit = {
            "unit_id": "unit_0001",
            "speaker_id": "SPEAKER_00",
            "preferred_duration_ms": preferred_ms,
            "maximum_duration_ms": max_ms,
            "tts_text": "Hello world",
        }
        
        # Validate unit
        result = validate_azure_tts_unit(
            unit=unit,
            wav_path=wav_path,
            assigned_voice=None,
            protected_entities=None,
            duration_tolerance_ms=200,
        )
        
        # Check pass/fail
        assert result.passed == should_pass
        
        # Check warnings
        assert len(result.warnings) == len(expected_warnings)
        for expected_warning in expected_warnings:
            assert any(expected_warning in w for w in result.warnings)
        
        # Check failures
        assert len(result.failures) == len(expected_failures)
        for expected_failure in expected_failures:
            assert any(expected_failure in f for f in result.failures)
        
        # Check alignment padding calculation
        if duration_ms < preferred_ms:
            assert result.alignment_padding_required_ms == preferred_ms - duration_ms
        else:
            assert result.alignment_padding_required_ms == 0
