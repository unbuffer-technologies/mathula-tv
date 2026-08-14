from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mathula_tv.contextual_speaker_identity import (
    CONTEXTUAL_SPEAKER_POLICY_VERSION,
    resolve_contextual_speaker_identities,
)
from mathula_tv.direct_azure_dub import DirectAzureDubRenderer, DirectDubOptions


INTRODUCTION = (
    "We are joined now by the uncle of Ayabonga Gaka, Mr. Bongile Gaka, "
    "who can share how the family is feeling."
)


def _segment(
    segment_id: str,
    speaker_id: str,
    raw_speaker: str,
    start: float,
    end: float,
    text: str,
) -> dict[str, object]:
    return {
        "segment_id": segment_id,
        "speaker_id": speaker_id,
        "speaker": speaker_id,
        "raw_speaker": raw_speaker,
        "start": start,
        "end": end,
        "source_text": text,
    }


def _research() -> dict[str, object]:
    return {
        "rejected_corrections": [
            {
                "entity_id": "person_ayabonga_gaka",
                "entity_type": "person",
                "representative_text": "Aya Bonga Gaka",
                "canonical_text": "Ayabonga Gaqa",
                "aliases": ["Aya Bonga Gaka", "Ayabonga Gaka", "Ayabonga"],
                "confidence": 0.98,
                "validation_reason": (
                    "insufficient_independent_or_authoritative_sources"
                ),
                "mentions": [{"raw_text": "Ayabonga Gaka"}],
            },
            {
                "entity_id": "person_bongile_gaka",
                "entity_type": "person",
                "representative_text": "Bongile Gaka",
                "canonical_text": "Bongile Gaqa",
                "aliases": ["Bongile Gaka", "Bongile"],
                "confidence": 0.82,
                "validation_reason": "confidence_below_0.95",
                "mentions": [{"raw_text": "Bongile Gaka"}],
            },
        ]
    }


def _transcript() -> dict[str, object]:
    return {
        "segments": [
            _segment("seg-1", "SPEAKER_01", "AZURE_2", 46.11, 118.43, INTRODUCTION),
            _segment(
                "seg-2",
                "SPEAKER_02",
                "AZURE_3",
                122.27,
                134.11,
                "Ayabonga was my nephew and the family is devastated.",
            ),
        ]
    }


def _write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def test_unique_titled_guest_wins_over_other_person_in_same_handoff(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "contextual_speaker_identities.json"
    registry_path = tmp_path / "registry.json"

    result = resolve_contextual_speaker_identities(
        job_id="ce9bbbb623b34872abae0428f004ae2e",
        transcript=_transcript(),
        name_research=_research(),
        required_speaker_ids=["SPEAKER_02"],
        output_path=output_path,
        registry_path=registry_path,
    )

    speaker = result["SPEAKER_02"]
    assert speaker["identified_name"] == "Bongile Gaka"
    assert speaker["identified_name"] != "Bongile Gaqa"
    assert speaker["gender"] == "male"
    assert speaker["voice_family"] == "masculine"
    assert speaker["identity_source"] == (
        "explicit_broadcast_boundary_transcript_alias"
    )
    assert speaker["registry_eligible"] is False
    proposal = speaker["proposals"][0]
    assert proposal["binding_mode"] == "explicit_titled_presenter_introduction"
    assert proposal["candidate_source"] == "rejected_correction_transcript_alias"
    assert proposal["rejected_canonical_text"] == "Bongile Gaqa"
    assert not registry_path.exists()

    audit = json.loads(output_path.read_text(encoding="utf-8"))
    assert audit["policy_version"] == CONTEXTUAL_SPEAKER_POLICY_VERSION
    assert audit["safety_contract"]["rejected_canonical_corrections_are_applied"] is False
    assert audit["safety_contract"]["acoustic_threshold_lowered"] is False


def test_rejected_transcript_alias_cannot_bind_without_explicit_handoff(
    tmp_path: Path,
) -> None:
    transcript = {
        "segments": [
            _segment(
                "seg-1",
                "SPEAKER_01",
                "AZURE_2",
                0,
                10,
                "The family of Ayabonga Gaka spoke with Mr. Bongile Gaka.",
            ),
            _segment("seg-2", "SPEAKER_02", "AZURE_3", 10.1, 20, "It is tragic."),
        ]
    }
    result = resolve_contextual_speaker_identities(
        job_id="job-no-handoff",
        transcript=transcript,
        name_research=_research(),
        required_speaker_ids=["SPEAKER_02"],
        output_path=tmp_path / "audit.json",
        registry_path=tmp_path / "registry.json",
    )

    assert result == {}


def test_direct_dub_assigns_themba_without_lowering_acoustic_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_id = "ce9bbbb623b34872abae0428f004ae2e"
    analysis = tmp_path / "jobs" / job_id / "analysis"
    transcript = _transcript()
    _write(analysis / "transcript_en.json", transcript)
    _write(
        analysis / "azure_diarization.json",
        {
            "turns": [
                {"speaker": "AZURE_2", "start": 46.11, "end": 118.43},
                {"speaker": "AZURE_3", "start": 122.27, "end": 134.11},
            ]
        },
    )
    _write(
        analysis / "acoustic_analysis.json",
        {
            "speakers": {
                "AZURE_2": {
                    "speaker_id": "AZURE_2",
                    "voice_family": "feminine",
                    "confidence": 0.99,
                },
                "AZURE_3": {
                    "speaker_id": "AZURE_3",
                    "voice_family": "unknown",
                    "confidence": 0.0,
                    "probabilities": {
                        "masculine": 0.6975021958351135,
                        "feminine": 0.3024978041648865,
                    },
                },
            }
        },
    )
    _write(analysis / "autocorrection_name_research.json", _research())
    monkeypatch.setattr(
        "mathula_tv.direct_azure_dub.ensure_canonical_voice_family_analysis",
        lambda **kwargs: pytest.fail(
            "explicit contextual identity must avoid an acoustic classifier rerun"
        ),
    )

    options = DirectDubOptions()
    assert options.min_voice_family_confidence == pytest.approx(0.70)
    renderer = DirectAzureDubRenderer(
        SimpleNamespace(work_dir=tmp_path), SimpleNamespace()
    )
    output = tmp_path / "jobs" / job_id / "direct_dub" / "voice_resolution.json"
    result = renderer._voice_assignments(
        SimpleNamespace(job_id=job_id),
        None,
        options,
        required_speaker_ids=["SPEAKER_01", "SPEAKER_02"],
        output_path=output,
    )

    speaker = result["SPEAKER_02"]
    assert speaker["identified_name"] == "Bongile Gaka"
    assert speaker["selected_voice"] == "zu-ZA-ThembaNeural"
    assert speaker["resolution_status"] == "contextual_speaker_identity"
    assert speaker["voice_family_source"] == (
        "contextual_transcript_identity_gender"
    )
    assert json.loads(output.read_text())["unresolved_speakers"] == []
