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


def _transcript(*, include_outro_cue: bool = True) -> dict[str, object]:
    outro = (
        "You've just been listening to a briefing by the acting police "
        "minister, Feroz Kucalia."
        if include_outro_cue
        else "The acting police minister, Feroz Kucalia, was discussed earlier."
    )
    return {
        "segments": [
            _segment(
                "seg-1",
                "SPEAKER_00",
                "AZURE_1",
                0,
                10,
                "The minister is now speaking.",
            ),
            _segment(
                "seg-2",
                "SPEAKER_01",
                "AZURE_2",
                10.5,
                40,
                "It is very difficult as a police minister to visit the families.",
            ),
            _segment("seg-3", "SPEAKER_02", "AZURE_3", 41, 44, "One question."),
            _segment("seg-4", "SPEAKER_01", "AZURE_2", 45, 46, "Thank you."),
            _segment("seg-5", "SPEAKER_00", "AZURE_1", 46.2, 47, "Thank you."),
            _segment("seg-6", "SPEAKER_03", "AZURE_4", 48, 49, "Thanks."),
            _segment("seg-7", "SPEAKER_00", "AZURE_1", 50, 65, outro),
        ]
    }


def _registry() -> dict[str, object]:
    return {
        "schema_version": "mathula-contextual-speaker-identity-registry-v1",
        "identities": {
            "firoz cachalia": {
                "canonical_name": "Firoz Cachalia",
                "gender": "male",
                "confidence": 0.99,
                "evidence_count": 1,
                "evidence": [
                    {
                        "job_id": "earlier-job",
                        "speaker_id": "SPEAKER_01",
                        "identity_source": "explicit_broadcast_boundary_context",
                        "gender_source": "explicit_adjacent_transcript_pronoun",
                    }
                ],
            }
        },
    }


def _write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _resolve(
    tmp_path: Path,
    *,
    include_outro_cue: bool = True,
) -> dict[str, object]:
    registry_path = tmp_path / "contextual_speaker_identities" / "registry.json"
    _write(registry_path, _registry())
    return resolve_contextual_speaker_identities(
        job_id="ccc4add3e5844ba4b067f69ae3b103e8",
        transcript=_transcript(include_outro_cue=include_outro_cue),
        name_research={"accepted_corrections": [], "rejected_corrections": []},
        required_speaker_ids=["SPEAKER_01"],
        output_path=tmp_path / "contextual_speaker_identities.json",
        registry_path=registry_path,
    )


def test_audited_registry_rebinds_stt_damaged_name_across_outro_chatter(
    tmp_path: Path,
) -> None:
    result = _resolve(tmp_path)

    speaker = result["SPEAKER_01"]
    assert speaker["identified_name"] == "Firoz Cachalia"
    assert speaker["identity_confidence"] == pytest.approx(0.740741)
    assert speaker["identity_source"] == (
        "audited_registry_fuzzy_titled_boundary_context"
    )
    assert speaker["gender"] == "male"
    assert speaker["gender_source"] == "audited_contextual_identity_registry"
    proposal = speaker["proposals"][0]
    assert proposal["binding_mode"] == (
        "audited_registry_fuzzy_titled_presenter_outro"
    )
    assert proposal["registry_fuzzy_match"] == {
        "canonical_name": "Firoz Cachalia",
        "observed_transcript_name": "Feroz Kucalia",
        "preceding_title": "acting police minister",
        "overall_similarity": 0.740741,
        "token_similarities": [0.8, 0.666667],
        "minimum_name_similarity": 0.72,
        "minimum_token_similarity": 0.65,
    }
    assert proposal["speaker_role_evidence"] == ["as a police minister"]
    assert [
        item["segment_id"]
        for item in proposal["boundary_evidence"]["after_context"]
    ] == ["seg-5", "seg-6", "seg-7"]

    audit = json.loads(
        (tmp_path / "contextual_speaker_identities.json").read_text()
    )
    assert audit["policy_version"] == CONTEXTUAL_SPEAKER_POLICY_VERSION
    assert audit["safety_contract"][
        "registry_fuzzy_match_requires_unique_titled_handoff"
    ] is True
    assert audit["safety_contract"]["acoustic_threshold_lowered"] is False


def test_registry_fuzzy_name_does_not_bind_without_presenter_handoff(
    tmp_path: Path,
) -> None:
    assert _resolve(tmp_path, include_outro_cue=False) == {}


def test_direct_dub_uses_registry_context_instead_of_uncertain_acoustics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_id = "ccc4add3e5844ba4b067f69ae3b103e8"
    analysis = tmp_path / "jobs" / job_id / "analysis"
    _write(analysis / "transcript_en.json", _transcript())
    _write(
        analysis / "azure_diarization.json",
        {
            "turns": [
                {"speaker": "AZURE_1", "start": 0, "end": 10},
                {"speaker": "AZURE_2", "start": 10.5, "end": 40},
                {"speaker": "AZURE_3", "start": 41, "end": 44},
                {"speaker": "AZURE_2", "start": 45, "end": 46},
                {"speaker": "AZURE_1", "start": 46.2, "end": 47},
                {"speaker": "AZURE_4", "start": 48, "end": 49},
                {"speaker": "AZURE_1", "start": 50, "end": 65},
            ]
        },
    )
    _write(
        analysis / "acoustic_analysis.json",
        {
            "speakers": {
                "AZURE_2": {
                    "speaker_id": "AZURE_2",
                    "voice_family": "unknown",
                    "confidence": 0.0,
                    "probabilities": {
                        "masculine": 0.6624957919120789,
                        "feminine": 0.33750414848327637,
                    },
                }
            }
        },
    )
    _write(
        analysis / "autocorrection_name_research.json",
        {"accepted_corrections": [], "rejected_corrections": []},
    )
    _write(
        tmp_path / "contextual_speaker_identities" / "registry.json",
        _registry(),
    )
    monkeypatch.setattr(
        "mathula_tv.direct_azure_dub.ensure_canonical_voice_family_analysis",
        lambda **kwargs: pytest.fail(
            "registry-backed context must avoid an acoustic classifier rerun"
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
        required_speaker_ids=["SPEAKER_01"],
        output_path=output,
    )

    speaker = result["SPEAKER_01"]
    assert speaker["identified_name"] == "Firoz Cachalia"
    assert speaker["selected_voice"] == "zu-ZA-ThembaNeural"
    assert speaker["resolution_status"] == "contextual_speaker_identity"
    assert speaker["voice_family_source"] == (
        "contextual_transcript_identity_gender"
    )
    assert json.loads(output.read_text())["unresolved_speakers"] == []
