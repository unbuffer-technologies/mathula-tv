from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mathula_tv.direct_azure_dub import (
    DirectAzureDubRenderer,
    DirectDubError,
    DirectDubOptions,
)
from mathula_tv.known_speakers import KnownSpeakerMatch


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


class _Metadata:
    def to_dict(self) -> dict[str, object]:
        return {"provider": "test", "model_returned": "context-model"}


class _ContextVoiceProvider:
    def __init__(self, speakers: list[dict[str, object]]) -> None:
        self.speakers = speakers

    def complete_structured(self, request):
        return SimpleNamespace(
            data={
                "schema_version": "mathula-contextual-voice-family-inference-v1",
                "speakers": self.speakers,
            },
            metadata=_Metadata(),
        )


def _renderer(tmp_path: Path) -> DirectAzureDubRenderer:
    settings = SimpleNamespace(work_dir=tmp_path)
    return DirectAzureDubRenderer(settings, SimpleNamespace())


def _job_files(tmp_path: Path, *, confidence: float = 0.94) -> Path:
    analysis = tmp_path / "jobs" / "job1" / "analysis"
    _write(
        analysis / "transcript_en.json",
        {
            "segments": [
                {"speaker_id": "SPEAKER_00", "start": 0.0, "end": 10.0},
                {"speaker_id": "SPEAKER_01", "start": 10.0, "end": 20.0},
            ]
        },
    )
    _write(
        analysis / "azure_diarization.json",
        {
            "turns": [
                {"speaker": "AZURE_1", "start": 0.0, "end": 10.0},
                {"speaker": "AZURE_2", "start": 10.0, "end": 20.0},
            ]
        },
    )
    _write(
        analysis / "acoustic_analysis.json",
        {
            "schema_version": "acoustic-analysis-v1",
            "speakers": {
                "AZURE_1": {
                    "speaker_id": "AZURE_1",
                    "voice_family": "feminine",
                    "confidence": confidence,
                    "evidence": {"method": "ecapa_voice_family_classifier"},
                },
                "AZURE_2": {
                    "speaker_id": "AZURE_2",
                    "voice_family": "feminine",
                    "confidence": confidence,
                    "evidence": {"method": "ecapa_voice_family_classifier"},
                },
            },
        },
    )
    _write(
        analysis / "speaker_profiles.json",
        {
            "speakers": [
                {"speaker_id": "SPEAKER_00", "tts_voice_family": "unknown"},
                {"speaker_id": "SPEAKER_01", "tts_voice_family": "unknown"},
            ]
        },
    )
    # This stale artifact is the old bug: unknown speakers were persisted as Themba.
    _write(
        analysis / "azure_voice_assignments.json",
        {
            "assignments": [
                {"speaker_id": "SPEAKER_00", "selected_voice": "zu-ZA-ThembaNeural"},
                {"speaker_id": "SPEAKER_01", "selected_voice": "zu-ZA-ThembaNeural"},
            ]
        },
    )
    return analysis


def test_mapped_classifier_evidence_overrides_stale_themba_assignments(tmp_path) -> None:
    _job_files(tmp_path)
    renderer = _renderer(tmp_path)
    output = tmp_path / "jobs" / "job1" / "direct_dub" / "voice_resolution.json"

    result = renderer._voice_assignments(
        SimpleNamespace(job_id="job1"),
        None,
        DirectDubOptions(),
        required_speaker_ids=["SPEAKER_00", "SPEAKER_01"],
        output_path=output,
    )

    assert result["SPEAKER_00"]["selected_voice"] == "zu-ZA-ThandoNeural"
    assert result["SPEAKER_01"]["selected_voice"] == "zu-ZA-ThandoNeural"
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["speaker_id_mapping"]["raw_to_canonical"] == {
        "AZURE_1": "SPEAKER_00",
        "AZURE_2": "SPEAKER_01",
    }
    assert artifact["legacy_azure_assignments"]["ignored"] is True
    assert artifact["unknown_defaults_to_masculine"] is False
    assert artifact["persona_version"] == "mathula-direct-azure-persona-v3-volume-neutral"
    personas = {
        item["speaker_id"]: item["base_prosody"]
        for item in artifact["speakers"]
        if item["status"] == "resolved"
    }
    assert personas["SPEAKER_00"]["persona_slot"] != personas["SPEAKER_01"]["persona_slot"]
    assert abs(
        personas["SPEAKER_00"]["pitch_percent"]
        - personas["SPEAKER_01"]["pitch_percent"]
    ) >= 8


def test_unknown_family_fails_instead_of_defaulting_to_themba(tmp_path) -> None:
    _job_files(tmp_path, confidence=0.40)
    renderer = _renderer(tmp_path)
    output = tmp_path / "jobs" / "job1" / "direct_dub" / "voice_resolution.json"

    with pytest.raises(DirectDubError, match="no longer default to Themba"):
        renderer._voice_assignments(
            SimpleNamespace(job_id="job1"),
            None,
            DirectDubOptions(min_voice_family_confidence=0.70),
            required_speaker_ids=["SPEAKER_00", "SPEAKER_01"],
            output_path=output,
        )

    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["unresolved_speakers"] == ["SPEAKER_00", "SPEAKER_01"]


def test_transcript_ai_resolves_inconclusive_acoustic_voice_family(tmp_path) -> None:
    analysis = _job_files(tmp_path, confidence=0.66)
    transcript = json.loads((analysis / "transcript_en.json").read_text())
    transcript["segments"][0].update(
        {
            "segment_id": "seg-1",
            "source_text": "I can answer that question, ma'am.",
        }
    )
    transcript["segments"][1].update(
        {
            "segment_id": "seg-2",
            "source_text": "Thank you, sir. Please continue.",
        }
    )
    _write(analysis / "transcript_en.json", transcript)
    acoustic = json.loads((analysis / "acoustic_analysis.json").read_text())
    acoustic["speakers"]["AZURE_2"]["confidence"] = 0.98
    _write(analysis / "acoustic_analysis.json", acoustic)

    provider = _ContextVoiceProvider(
        [
            {
                "speaker_id": "SPEAKER_00",
                "voice_family": "masculine",
                "confidence": 0.93,
                "inference_basis": "explicit_title_or_address",
                "evidence_segment_ids": ["seg-2"],
                "evidence_summary": "SPEAKER_01 addresses SPEAKER_00 as sir.",
            }
        ]
    )
    renderer = DirectAzureDubRenderer(
        SimpleNamespace(work_dir=tmp_path),
        SimpleNamespace(),
        timing_repair_provider_factory=lambda: provider,
    )
    output = tmp_path / "jobs" / "job1" / "direct_dub" / "voice_resolution.json"

    result = renderer._voice_assignments(
        SimpleNamespace(job_id="job1"),
        None,
        DirectDubOptions(min_voice_family_confidence=0.70),
        required_speaker_ids=["SPEAKER_00", "SPEAKER_01"],
        output_path=output,
    )

    assert result["SPEAKER_00"]["selected_voice"] == "zu-ZA-ThembaNeural"
    assert result["SPEAKER_00"]["voice_family_source"] == (
        "ai_transcript_context"
    )
    assert result["SPEAKER_00"]["resolution_status"] == (
        "contextual_transcript_voice_family"
    )
    assert result["SPEAKER_01"]["selected_voice"] == "zu-ZA-ThandoNeural"


def test_verified_speechbrain_match_precedes_anonymous_gender(
    tmp_path, monkeypatch
) -> None:
    analysis = _job_files(tmp_path)
    acoustic = json.loads((analysis / "acoustic_analysis.json").read_text())
    speaker_audio = tmp_path / "jobs" / "job1" / "audio" / "speakers" / "SPEAKER_00.wav"
    speaker_audio.parent.mkdir(parents=True)
    speaker_audio.write_bytes(b"speaker")
    acoustic["speakers"]["AZURE_1"]["speaker_audio_path"] = str(speaker_audio)
    _write(analysis / "acoustic_analysis.json", acoustic)
    _write(
        tmp_path / "known_speakers" / "registry.json",
        {"schema_version": "mathula-known-speakers-v1", "speakers": {}},
    )

    monkeypatch.setattr(
        "mathula_tv.direct_azure_dub.ensure_canonical_speaker_reels",
        lambda **kwargs: {
            "SPEAKER_00": {"speaker_audio_path": str(speaker_audio)}
        },
    )
    monkeypatch.setattr(
        DirectAzureDubRenderer,
        "_ensure_analysis_audio",
        lambda self, job: speaker_audio,
    )
    monkeypatch.setattr(
        "mathula_tv.direct_azure_dub.identify_known_speaker",
        lambda path, **kwargs: (
            KnownSpeakerMatch(
                person_id="samkelo_maseko",
                display_name="Samkelo Maseko",
                gender="male",
                azure_voice="zu-ZA-ThembaNeural",
                score=0.91,
                runner_up_score=None,
                threshold=0.65,
                margin=0.10,
                model_id="speechbrain/spkrec-ecapa-voxceleb",
                model_revision="revision",
            )
            if path == speaker_audio
            else None
        ),
    )

    output = tmp_path / "jobs" / "job1" / "direct_dub" / "voice_resolution.json"
    result = _renderer(tmp_path)._voice_assignments(
        SimpleNamespace(job_id="job1"),
        None,
        DirectDubOptions(),
        required_speaker_ids=["SPEAKER_00"],
        output_path=output,
    )

    assert result["SPEAKER_00"]["selected_voice"] == "zu-ZA-ThembaNeural"
    assert result["SPEAKER_00"]["resolution_status"] == "known_speaker_signature"
    assert result["SPEAKER_00"]["known_speaker"]["display_name"] == "Samkelo Maseko"
