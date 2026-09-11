from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from mathula_tv.models import JobManifest
from mathula_tv.native_dub import ensure_native_voice_assignments


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


class _Provider:
    provider = "azure-foundry-grok"

    def __init__(self, voices: list[dict] | None = None) -> None:
        self.config = SimpleNamespace(deployment="grok-4.3", max_output_tokens=8192)
        self.voices = voices
        self.calls: list[dict] = []

    def complete_json(self, **kwargs):
        self.calls.append(kwargs)
        if self.voices is None:
            raise AssertionError("Grok must not be called")
        return SimpleNamespace(
            data={"voices": self.voices},
            input_tokens=111,
            output_tokens=22,
            attempts=1,
        )


def _job(job_id: str = "job1") -> JobManifest:
    return JobManifest(
        job_id=job_id,
        source_filename="x.mp4",
        local_source_path="x.mp4",
        source_checksum="x",
        source_duration=3.0,
    )


def _segments() -> list[dict]:
    return [
        {
            "segment_id": "a",
            "speaker_id": "SPEAKER_00",
            "start_ms": 0,
            "end_ms": 1000,
            "source_text": "Good morning commissioner.",
            "restored_text": "Good morning commissioner.",
        },
        {
            "segment_id": "b",
            "speaker_id": "SPEAKER_01",
            "start_ms": 1000,
            "end_ms": 2000,
            "source_text": "Thank you, sir.",
            "restored_text": "Thank you, sir.",
        },
    ]


def _analysis_inputs(job_root: Path) -> None:
    audio = job_root / "audio" / "analysis_mono.wav"
    audio.parent.mkdir(parents=True, exist_ok=True)
    audio.write_bytes(b"wav-placeholder")
    _write(
        job_root / "analysis" / "azure_diarization.json",
        {
            "turns": [
                {"speaker": "AZURE_1", "start": 0.0, "end": 1.0},
                {"speaker": "AZURE_2", "start": 1.0, "end": 2.0},
            ]
        },
    )


def test_native_reuses_legacy_direct_job_cache_without_grok(tmp_path: Path) -> None:
    job_root = tmp_path / "jobs" / "job1"
    output = job_root / "direct_dub" / "voice_resolution.json"
    _write(
        output,
        {
            "schema_version": "mathula-direct-voice-resolution-v4-cached",
            "assignments": {
                "SPEAKER_00": {
                    "speaker_id": "SPEAKER_00",
                    "selected_voice": "zu-ZA-ThembaNeural",
                    "voice_family": "masculine",
                    "base_prosody": {"rate_percent": 7, "pitch_percent": -2},
                },
                "SPEAKER_01": {
                    "speaker_id": "SPEAKER_01",
                    "selected_voice": "zu-ZA-ThandoNeural",
                    "voice_family": "feminine",
                    "base_prosody": {"rate_percent": -3, "pitch_percent": 2},
                },
            },
        },
    )
    provider = _Provider()
    path = ensure_native_voice_assignments(
        job=_job(),
        job_root=job_root,
        segments=_segments(),
        context_ledger={},
        provider=provider,
    )
    assert path == output
    assert provider.calls == []


def test_native_runs_legacy_ecapa_classifier_before_grok(tmp_path: Path, monkeypatch) -> None:
    job_root = tmp_path / "jobs" / "job1"
    _analysis_inputs(job_root)
    monkeypatch.setattr(
        "mathula_tv.native_dub.resolve_contextual_speaker_identities",
        lambda **kwargs: {},
    )
    calls: list[list[str]] = []

    def fake_classifier(**kwargs):
        calls.append(list(kwargs["required_speaker_ids"]))
        return {
            "speakers": {
                "SPEAKER_00": {
                    "speaker_id": "SPEAKER_00",
                    "voice_family": "masculine",
                    "confidence": 0.97,
                    "evidence": {"method": "ecapa_voice_family_classifier"},
                },
                "SPEAKER_01": {
                    "speaker_id": "SPEAKER_01",
                    "voice_family": "feminine",
                    "confidence": 0.96,
                    "evidence": {"method": "ecapa_voice_family_classifier"},
                },
            }
        }

    monkeypatch.setattr(
        "mathula_tv.native_dub.ensure_canonical_voice_family_analysis",
        fake_classifier,
    )
    provider = _Provider()
    path = ensure_native_voice_assignments(
        job=_job(),
        job_root=job_root,
        segments=_segments(),
        context_ledger={},
        provider=provider,
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    assert calls == [["SPEAKER_00", "SPEAKER_01"]]
    assert provider.calls == []
    assert value["assignments"]["SPEAKER_00"]["voice_family_source"] == "ecapa_voice_family_classifier"
    assert value["assignments"]["SPEAKER_01"]["selected_voice"] == "zu-ZA-ThandoNeural"
    assert value["tts_rate_percent"] == 0
    assert all(
        item["base_prosody"]["rate_percent"] == 0
        for item in value["assignments"].values()
    )


def test_native_sends_only_legacy_unresolved_speakers_to_grok(tmp_path: Path, monkeypatch) -> None:
    job_root = tmp_path / "jobs" / "job1"
    _analysis_inputs(job_root)
    monkeypatch.setattr(
        "mathula_tv.native_dub.resolve_contextual_speaker_identities",
        lambda **kwargs: {},
    )

    def fake_classifier(**kwargs):
        return {
            "speakers": {
                "SPEAKER_00": {
                    "speaker_id": "SPEAKER_00",
                    "voice_family": "masculine",
                    "confidence": 0.98,
                    "evidence": {"method": "ecapa_voice_family_classifier"},
                },
                "SPEAKER_01": {
                    "speaker_id": "SPEAKER_01",
                    "voice_family": "unknown",
                    "confidence": 0.0,
                    "evidence": {"method": "ecapa_voice_family_classifier"},
                },
            }
        }

    monkeypatch.setattr(
        "mathula_tv.native_dub.ensure_canonical_voice_family_analysis",
        fake_classifier,
    )
    provider = _Provider([
        {"speaker_id": "SPEAKER_01", "voice_family": "feminine", "confidence": 0.94}
    ])
    path = ensure_native_voice_assignments(
        job=_job(),
        job_root=job_root,
        segments=_segments(),
        context_ledger={"speaker_roles": [], "entities": []},
        provider=provider,
    )
    assert len(provider.calls) == 1
    sent = provider.calls[0]["payload"]["speakers"]
    assert [item["speaker_id"] for item in sent] == ["SPEAKER_01"]
    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["assignments"]["SPEAKER_00"]["voice_family_source"] == "ecapa_voice_family_classifier"
    assert value["assignments"]["SPEAKER_01"]["voice_family_source"] == "grok_transcript_context_fallback"
    assert value["usage"]["input_tokens"] == 111


def test_native_reuses_historical_mapping_by_identity_not_diarization_id(tmp_path: Path, monkeypatch) -> None:
    old = tmp_path / "jobs" / "oldjob" / "direct_dub" / "voice_resolution.json"
    _write(
        old,
        {
            "schema_version": "mathula-direct-voice-resolution-v4-cached",
            "assignments": {
                "SPEAKER_09": {
                    "speaker_id": "SPEAKER_09",
                    "identified_name": "Fadiel Adams",
                    "selected_voice": "zu-ZA-ThembaNeural",
                    "voice_family": "masculine",
                    "voice_family_confidence": 0.99,
                    "resolution_status": "contextual_speaker_identity",
                }
            },
        },
    )
    job_root = tmp_path / "jobs" / "job1"
    monkeypatch.setattr(
        "mathula_tv.native_dub.resolve_contextual_speaker_identities",
        lambda **kwargs: {
            "SPEAKER_00": {
                "speaker_id": "SPEAKER_00",
                "identified_name": "Fadiel Adams",
                "voice_family": "masculine",
                "voice_family_confidence": 0.99,
                "voice_family_source": "contextual_transcript_identity_gender",
            },
            "SPEAKER_01": {
                "speaker_id": "SPEAKER_01",
                "voice_family": "feminine",
                "voice_family_confidence": 0.99,
                "voice_family_source": "contextual_transcript_identity_gender",
            },
        },
    )
    provider = _Provider()
    path = ensure_native_voice_assignments(
        job=_job(),
        job_root=job_root,
        segments=_segments(),
        context_ledger={},
        provider=provider,
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    assert provider.calls == []
    assert value["assignments"]["SPEAKER_00"]["resolution_status"] == "historical_identified_voice_cache"
    assert value["assignments"]["SPEAKER_00"]["identified_name"] == "Fadiel Adams"
    assert value["assignments"]["SPEAKER_01"]["resolution_status"] == "contextual_transcript_voice_family"


def test_native_surfaces_isolated_classifier_failures_before_grok(tmp_path: Path, monkeypatch) -> None:
    job_root = tmp_path / "jobs" / "job1"
    _analysis_inputs(job_root)
    monkeypatch.setattr(
        "mathula_tv.native_dub.resolve_contextual_speaker_identities",
        lambda **kwargs: {},
    )

    def fake_classifier(**kwargs):
        # The legacy helper isolates per-speaker failures instead of raising.
        # Native dub must still detect this as classifier unavailability and
        # stop before spending a Grok call.
        return {
            "speakers": {
                speaker: {
                    "speaker_id": speaker,
                    "voice_family": "unknown",
                    "confidence": 0.0,
                }
                for speaker in kwargs["required_speaker_ids"]
            },
            "failures": {
                speaker: {
                    "speaker_id": speaker,
                    "error": "Voice-family model is not installed locally",
                }
                for speaker in kwargs["required_speaker_ids"]
            },
        }

    monkeypatch.setattr(
        "mathula_tv.native_dub.ensure_canonical_voice_family_analysis",
        fake_classifier,
    )
    provider = _Provider([
        {"speaker_id": "SPEAKER_00", "voice_family": "masculine", "confidence": 0.99},
        {"speaker_id": "SPEAKER_01", "voice_family": "feminine", "confidence": 0.99},
    ])

    import pytest
    with pytest.raises(ValueError) as excinfo:
        ensure_native_voice_assignments(
            job=_job(),
            job_root=job_root,
            segments=_segments(),
            context_ledger={},
            provider=provider,
        )
    assert "classifier is unavailable" in str(excinfo.value)
    assert "install_voice_gender_classifier.py" in str(excinfo.value)
    assert provider.calls == []


def test_native_pass1_timings_are_primary_voice_reel_source(tmp_path: Path, monkeypatch) -> None:
    job_root = tmp_path / "jobs" / "job1"
    audio = job_root / "audio" / "analysis_mono.wav"
    audio.parent.mkdir(parents=True, exist_ok=True)
    audio.write_bytes(b"wav-placeholder")
    # Deliberately omit legacy turns: the classifier should still receive the
    # canonical Pass-1 segment timings as its reel source.
    _write(job_root / "analysis" / "azure_diarization.json", {"words": []})
    monkeypatch.setattr(
        "mathula_tv.native_dub.resolve_contextual_speaker_identities",
        lambda **kwargs: {},
    )
    observed: dict = {}

    def fake_classifier(**kwargs):
        observed.update(kwargs)
        return {
            "speakers": {
                "SPEAKER_00": {
                    "speaker_id": "SPEAKER_00",
                    "voice_family": "masculine",
                    "confidence": 0.97,
                    "evidence": {"method": "ecapa_voice_family_classifier"},
                },
                "SPEAKER_01": {
                    "speaker_id": "SPEAKER_01",
                    "voice_family": "feminine",
                    "confidence": 0.96,
                    "evidence": {"method": "ecapa_voice_family_classifier"},
                },
            },
            "failures": {},
        }

    monkeypatch.setattr(
        "mathula_tv.native_dub.ensure_canonical_voice_family_analysis",
        fake_classifier,
    )
    provider = _Provider()
    ensure_native_voice_assignments(
        job=_job(),
        job_root=job_root,
        segments=_segments(),
        context_ledger={},
        provider=provider,
    )
    assert provider.calls == []
    assert set(observed["canonical_speaker_turns"]) == {"SPEAKER_00", "SPEAKER_01"}
    assert observed["canonical_speaker_turns"]["SPEAKER_00"][0]["start_ms"] == 0
    assert observed["canonical_speaker_turns"]["SPEAKER_01"][0]["start_ms"] == 1000
