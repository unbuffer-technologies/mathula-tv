"""Tests for intelligibility audit module."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mathula_tv.config import Settings
from mathula_tv.intelligibility import (
    IntelligibilityAuditReport,
    IntelligibilityAuditor,
    UnitIntelligibilityObservation,
)


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    from mathula_tv.config import Settings
    return Settings(
        work_dir=Path("/tmp/work"),
        gcs_bucket="test-bucket",
        gcs_prefix="test-prefix",
        commission_master_case_path=Path("/tmp/commission.json"),
        politics_context_path=Path("/tmp/politics.json"),
        azure_speech_endpoint="https://test.endpoint",
        azure_speech_region="eastus",
        azure_speech_locale="zu-ZA",
        azure_speech_api_version="2023-01-01",
        azure_speech_max_speakers=2,
        azure_speech_timeout_seconds=30,
        azure_ai_endpoint="https://test-ai.endpoint",
        azure_ai_deployment="test-deployment",
        azure_ai_api_version="2023-01-01",
        pyannote_model="test-model",
        max_clean_unit_wer=0.35,
        max_openvoice_wer_degradation=0.10,
        max_final_mix_wer_degradation=0.15,
        min_protected_entity_similarity=0.85,
    )


@pytest.fixture
def mock_stt_backend():
    """Mock STT backend for testing."""
    backend = MagicMock()
    backend.transcribe.return_value = {
        "DisplayText": "Test transcription",
        "combinedResults": [{"lexical": "test transcription"}],
        "NBest": [{"Lexical": "test transcription"}],
    }
    return backend


def test_intelligibility_auditor_normalize_text(mock_settings):
    """Test text normalization for WER calculation."""
    backend = MagicMock()
    auditor = IntelligibilityAuditor(mock_settings, backend)
    
    # Test basic normalization
    assert auditor.normalize_text("Hello World") == "hello world"
    assert auditor.normalize_text("  Hello  World  ") == "hello world"
    
    # Test punctuation removal
    assert auditor.normalize_text("Hello, World!") == "hello world"
    
    # Test Unicode normalization
    assert auditor.normalize_text("Hello\u2019World") == "helloworld"


def test_intelligibility_auditor_calculate_wer(mock_settings):
    """Test WER calculation."""
    backend = MagicMock()
    auditor = IntelligibilityAuditor(mock_settings, backend)
    
    # Perfect match
    wer = auditor.calculate_wer("hello world", "hello world")
    assert wer == 0.0
    
    # One substitution
    wer = auditor.calculate_wer("hello world", "hello there")
    assert wer == 0.5
    
    # One insertion
    wer = auditor.calculate_wer("hello world", "hello big world")
    assert wer == 0.5
    
    # One deletion
    wer = auditor.calculate_wer("hello world", "hello")
    assert wer == 0.5
    
    # Empty reference
    wer = auditor.calculate_wer("", "hello")
    assert wer == 1.0
    
    # Empty hypothesis with non-empty reference
    wer = auditor.calculate_wer("hello", "")
    assert wer == 1.0


def test_intelligibility_auditor_check_protected_entity(mock_settings):
    """Test protected entity recognition checking."""
    backend = MagicMock()
    auditor = IntelligibilityAuditor(mock_settings, backend)
    
    # Exact match
    recognized, phrase = auditor.check_protected_entity("John Doe", "John Doe testified")
    assert recognized is True
    assert phrase == "John Doe"
    
    # Alias match
    recognized, phrase = auditor.check_protected_entity(
        "John Doe", "J. Doe testified", aliases=["J. Doe"]
    )
    assert recognized is True
    assert phrase == "J. Doe"
    
    # STT phrase match
    recognized, phrase = auditor.check_protected_entity(
        "John Doe", "Mr. Doe testified", stt_phrases=["Mr. Doe"]
    )
    assert recognized is True
    assert phrase == "Mr. Doe"
    
    # No match
    recognized, phrase = auditor.check_protected_entity("John Doe", "Someone testified")
    assert recognized is False
    assert phrase == ""


def test_intelligibility_auditor_audit_unit(mock_settings, mock_stt_backend, tmp_path):
    """Test auditing a single unit."""
    auditor = IntelligibilityAuditor(mock_settings, mock_stt_backend)
    
    # Create a dummy audio file
    audio_file = tmp_path / "test.wav"
    audio_file.write_bytes(b"dummy audio data")
    
    protected_entities = {
        "person_john_doe": {
            "canonical_text": "John Doe",
            "aliases": ["J. Doe"],
            "stt_phrases": ["John Doe", "J. Doe"],
        }
    }
    
    observation = auditor.audit_unit(
        unit_id="unit_1",
        expected_text="John Doe testified",
        audio_path=audio_file,
        protected_entities=protected_entities,
        locale="zu-ZA",
    )
    
    assert observation.unit_id == "unit_1"
    assert observation.expected_text == "John Doe testified"
    # _extract_transcription returns lowercase from "lexical" field
    assert observation.transcribed_text == "test transcription"
    assert observation.word_error_rate >= 0.0


def test_intelligibility_auditor_audit_stage(mock_settings, mock_stt_backend, tmp_path):
    """Test auditing an entire stage."""
    auditor = IntelligibilityAuditor(mock_settings, mock_stt_backend)
    
    # Create dummy audio files
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    (audio_root / "unit_1.wav").write_bytes(b"dummy audio 1")
    (audio_root / "unit_2.wav").write_bytes(b"dummy audio 2")
    
    units = [
        {"unit_id": "unit_1", "tts_text": "John Doe testified"},
        {"unit_id": "unit_2", "tts_text": "Test Corporation reported"},
    ]
    
    protected_entities = {
        "unit_1": {
            "person_john_doe": {
                "canonical_text": "John Doe",
                "aliases": ["J. Doe"],
                "stt_phrases": ["John Doe"],
            }
        },
        "unit_2": {
            "organisation_test_corp": {
                "canonical_text": "Test Corporation",
                "aliases": ["Test Corp"],
                "stt_phrases": ["Test Corporation"],
            }
        },
    }
    
    report = auditor.audit_stage(
        job_id="test_job",
        stage="azure-tts",
        units=units,
        audio_root=audio_root,
        locale="zu-ZA",
        protected_entities=protected_entities,
    )
    
    assert report.job_id == "test_job"
    assert report.stage == "azure-tts"
    assert report.locale == "zu-ZA"
    assert len(report.per_unit_observations) == 2
    assert report.state in {"passed", "failed"}


def test_intelligibility_auditor_write_report(mock_settings, mock_stt_backend, tmp_path):
    """Test writing intelligibility reports."""
    auditor = IntelligibilityAuditor(mock_settings, mock_stt_backend)
    
    report = IntelligibilityAuditReport(
        schema_version="mathula-stt-intelligibility-audit-v1",
        job_id="test_job",
        stage="azure-tts",
        locale="zu-ZA",
        generated_at="2026-07-21T00:00:00+02:00",
        aggregate_blind_wer=0.25,
        aggregate_hinted_wer=None,
        per_unit_observations={},
        protected_entities_summary={
            "total_expected": 1,
            "total_recognized": 1,
            "recognition_rate": 1.0,
        },
        pass_fail_reasons=[],
        state="passed",
        audio_paths={"stage_root": "/path/to/audio"},
    )
    
    json_path, md_path = auditor.write_report(report, tmp_path)
    
    assert json_path.exists()
    assert md_path.exists()
    assert json_path.suffix == ".json"
    assert md_path.suffix == ".md"


def test_intelligibility_auditor_get_max_wer_for_stage(mock_settings):
    """Test getting max WER for different stages."""
    auditor = IntelligibilityAuditor(mock_settings, MagicMock())
    
    assert auditor._get_max_wer_for_stage("azure-tts") == 0.35
    assert auditor._get_max_wer_for_stage("openvoice") == 0.10
    assert auditor._get_max_wer_for_stage("aligned") == 0.10
    assert auditor._get_max_wer_for_stage("final-mix") == 0.15
    assert auditor._get_max_wer_for_stage("unknown") == 0.35  # Default


def test_unit_intelligibility_observation():
    """Test UnitIntelligibilityObservation dataclass."""
    observation = UnitIntelligibilityObservation(
        unit_id="unit_1",
        expected_text="John Doe testified",
        transcribed_text="John Doe testified",
        word_error_rate=0.0,
        protected_entities_expected=["person_john_doe"],
        protected_entities_recognized=["person_john_doe"],
        protected_entities_missing=[],
        best_recognized_phrases={"person_john_doe": "John Doe"},
        duration_seconds=5.0,
        audio_sha256="abc123",
    )
    
    assert observation.unit_id == "unit_1"
    assert observation.word_error_rate == 0.0
    assert len(observation.protected_entities_missing) == 0


def test_intelligibility_audit_report():
    """Test IntelligibilityAuditReport dataclass."""
    report = IntelligibilityAuditReport(
        schema_version="mathula-stt-intelligibility-audit-v1",
        job_id="test_job",
        stage="azure-tts",
        locale="zu-ZA",
        generated_at="2026-07-21T00:00:00+02:00",
        aggregate_blind_wer=0.25,
        aggregate_hinted_wer=0.20,
        per_unit_observations={},
        protected_entities_summary={
            "total_expected": 1,
            "total_recognized": 1,
            "recognition_rate": 1.0,
        },
        pass_fail_reasons=[],
        state="passed",
        audio_paths={"stage_root": "/path/to/audio"},
    )
    
    assert report.job_id == "test_job"
    assert report.aggregate_blind_wer == 0.25
    assert report.state == "passed"


def test_intelligibility_auditor_markdown_report(mock_settings):
    """Test Markdown report generation."""
    backend = MagicMock()
    auditor = IntelligibilityAuditor(mock_settings, backend)
    
    report = IntelligibilityAuditReport(
        schema_version="mathula-stt-intelligibility-audit-v1",
        job_id="test_job",
        stage="azure-tts",
        locale="zu-ZA",
        generated_at="2026-07-21T00:00:00+02:00",
        aggregate_blind_wer=0.25,
        aggregate_hinted_wer=None,
        per_unit_observations={
            "unit_1": {
                "unit_id": "unit_1",
                "word_error_rate": 0.1,
                "protected_entities_expected": ["person_john_doe"],
                "protected_entities_recognized": ["person_john_doe"],
                "protected_entities_missing": [],
            }
        },
        protected_entities_summary={
            "total_expected": 1,
            "total_recognized": 1,
            "recognition_rate": 1.0,
        },
        pass_fail_reasons=[],
        state="passed",
        audio_paths={"stage_root": "/path/to/audio"},
    )
    
    markdown = auditor._markdown_report(report)
    assert "test_job" in markdown
    assert "azure-tts" in markdown
    assert "0.2500" in markdown


def test_final_mix_wer_degradation_detection(mock_settings, mock_stt_backend):
    """Test final-mix WER degradation detection."""
    auditor = IntelligibilityAuditor(mock_settings, mock_stt_backend)
    
    # Known incident: aligned WER 0.20, final mix WER 1.0756
    # Degradation: 0.8756 exceeds threshold of 0.15
    units = [
        {"unit_id": "unit_1", "tts_text": "test text"}
    ]
    
    # Mock final mix transcription with high WER
    mock_stt_backend.transcribe.return_value = {
        "DisplayText": "completely wrong transcription with many errors",
        "NBest": [{"Lexical": "completely wrong transcription with many errors"}]
    }
    
    # Create temporary audio file
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = Path(tmpdir) / "unit_1.wav"
        audio_path.touch()
        
        report = auditor.audit_stage(
            job_id="test_job",
            stage="final-mix",
            units=units,
            audio_root=Path(tmpdir),
            locale="zu-ZA",
            baseline_wer=0.20,  # Aligned stage WER
        )
        
        # Should fail due to excessive degradation
        assert report.state == "failed"
        assert any("degradation" in reason.lower() for reason in report.pass_fail_reasons)


def test_hinted_stt_diagnostic_only(mock_settings, mock_stt_backend):
    """Test that hinted STT is diagnostic-only and doesn't affect pass/fail."""
    auditor = IntelligibilityAuditor(mock_settings, mock_stt_backend)
    
    units = [
        {"unit_id": "unit_1", "tts_text": "Madlanga Commission"}
    ]
    
    # Mock blind transcription that fails
    mock_stt_backend.transcribe.return_value = {
        "DisplayText": "wrong text",
        "NBest": [{"Lexical": "wrong text"}]
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = Path(tmpdir) / "unit_1.wav"
        audio_path.touch()
        
        # Blind audit should fail
        report = auditor.audit_stage(
            job_id="test_job",
            stage="azure-tts",
            units=units,
            audio_root=Path(tmpdir),
            locale="zu-ZA",
            with_phrase_hints=False,
        )
        
        assert report.state == "failed"
        
        # Even with phrase hints enabled, blind WER determines pass/fail
        # (hinted is diagnostic only)
        report_hinted = auditor.audit_stage(
            job_id="test_job",
            stage="azure-tts",
            units=units,
            audio_root=Path(tmpdir),
            locale="zu-ZA",
            with_phrase_hints=True,
        )
        
        # Should still fail because blind WER is authoritative
        assert report_hinted.state == "failed"
