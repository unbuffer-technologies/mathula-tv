from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mathula_tv.contextual_speaker_identity import (
    resolve_contextual_speaker_identities,
)
from mathula_tv.direct_azure_dub import DirectAzureDubRenderer, DirectDubOptions


def _research(*names: tuple[str, float]) -> dict[str, object]:
    return {
        "accepted_corrections": [
            {
                "entity_id": f"person_{index}",
                "entity_type": "person",
                "canonical_text": name,
                "confidence": confidence,
                "grounded_source_urls": [f"https://example.test/{index}"],
            }
            for index, (name, confidence) in enumerate(names, start=1)
        ]
    }


def _segment(
    segment_id: str,
    speaker_id: str,
    start: float,
    end: float,
    text: str,
) -> dict[str, object]:
    return {
        "segment_id": segment_id,
        "speaker_id": speaker_id,
        "start": start,
        "end": end,
        "source_text": text,
    }


def _resolve(
    tmp_path: Path,
    transcript: dict[str, object],
    research: dict[str, object],
    *,
    registry_path: Path | None = None,
    voice_inference_provider_factory=None,
) -> dict[str, object]:
    return resolve_contextual_speaker_identities(
        job_id="job-context",
        transcript=transcript,
        name_research=research,
        required_speaker_ids=["SPEAKER_01"],
        output_path=tmp_path / "contextual_speaker_identities.json",
        registry_path=(
            registry_path
            or tmp_path / "contextual_speaker_identities" / "registry.json"
        ),
        voice_inference_provider_factory=voice_inference_provider_factory,
    )


class _Metadata:
    def __init__(self, server_tool_evidence=None) -> None:
        self.server_tool_evidence = list(server_tool_evidence or [])

    def to_dict(self) -> dict[str, object]:
        return {
            "provider": "test",
            "model_returned": "context-model",
            "server_tool_evidence": self.server_tool_evidence,
        }


class _ContextVoiceProvider:
    def __init__(
        self,
        speakers: list[dict[str, object]],
        *,
        server_tool_evidence=None,
    ) -> None:
        self.speakers = speakers
        self.server_tool_evidence = list(server_tool_evidence or [])
        self.requests = []

    def complete_structured(self, request):
        self.requests.append(request)
        return SimpleNamespace(
            data={
                "schema_version": "mathula-contextual-voice-family-inference-v1",
                "speakers": [
                    {
                        "identified_name": None,
                        "grounded_source_urls": [],
                        **speaker,
                    }
                    for speaker in self.speakers
                ],
            },
            metadata=_Metadata(self.server_tool_evidence),
        )


def test_two_sided_briefing_context_identifies_firoz_cachalia(tmp_path: Path) -> None:
    transcript = {
        "segments": [
            _segment(
                "seg-00001",
                "SPEAKER_00",
                0.11,
                16.19,
                "The acting police minister, Firoz Cachalia, he's briefing "
                "the media right now.",
            ),
            _segment(
                "seg-00002",
                "SPEAKER_01",
                24.17,
                260.11,
                "As far as I'm concerned, this is intolerable.",
            ),
            _segment(
                "seg-00003",
                "SPEAKER_00",
                260.43,
                294.75,
                "There you have it. That's the acting police minister, Firoz "
                "Cachalia. He's briefing members of the media.",
            ),
        ]
    }

    resolved = _resolve(
        tmp_path,
        transcript,
        _research(("Firoz Cachalia", 0.99)),
    )

    assert resolved["SPEAKER_01"]["identified_name"] == "Firoz Cachalia"
    assert resolved["SPEAKER_01"]["gender"] == "male"
    assert resolved["SPEAKER_01"]["voice_family"] == "masculine"
    assert resolved["SPEAKER_01"]["proposals"][0]["binding_mode"] == (
        "two_sided_boundary_reference"
    )
    registry = json.loads(
        (tmp_path / "contextual_speaker_identities" / "registry.json").read_text()
    )
    assert registry["identities"]["firoz cachalia"]["gender"] == "male"


def test_general_story_name_mention_does_not_bind_a_speaker(tmp_path: Path) -> None:
    transcript = {
        "segments": [
            _segment(
                "seg-1",
                "SPEAKER_00",
                0,
                5,
                "We will discuss Firoz Cachalia and General Mkhwanazi.",
            ),
            _segment("seg-2", "SPEAKER_01", 5, 20, "Crime remains a concern."),
        ]
    }

    resolved = _resolve(
        tmp_path,
        transcript,
        _research(("Firoz Cachalia", 0.99), ("Nhlanhla Mkhwanazi", 0.99)),
    )

    assert resolved == {}
    artifact = json.loads(
        (tmp_path / "contextual_speaker_identities.json").read_text()
    )
    assert artifact["speakers"][0]["reason"] == "no_explicit_contextual_identity"


def test_ai_infers_voice_family_from_transcript_without_identifying_person(
    tmp_path: Path,
) -> None:
    transcript = {
        "segments": [
            _segment(
                "seg-1",
                "SPEAKER_00",
                0,
                5,
                "Sir, could you explain what happened next?",
            ),
            _segment(
                "seg-2",
                "SPEAKER_01",
                5,
                20,
                "Yes. I arrived shortly after the meeting started.",
            ),
        ]
    }
    provider = _ContextVoiceProvider(
        [
            {
                "speaker_id": "SPEAKER_01",
                "voice_family": "masculine",
                "confidence": 0.94,
                "inference_basis": "explicit_title_or_address",
                "evidence_segment_ids": ["seg-1", "seg-2"],
                "evidence_summary": (
                    "The preceding speaker directly addresses SPEAKER_01 as sir."
                ),
            }
        ]
    )

    resolved = _resolve(
        tmp_path,
        transcript,
        {},
        voice_inference_provider_factory=lambda: provider,
    )

    record = resolved["SPEAKER_01"]
    assert record.get("identified_name") is None
    assert record["voice_family"] == "masculine"
    assert record["voice_family_source"] == "ai_transcript_context"
    assert record["reason"] == "voice_family_resolved_without_person_identity"
    assert provider.requests[0].operation == "contextual_voice_family_inference"
    artifact = json.loads(
        (tmp_path / "contextual_speaker_identities.json").read_text()
    )
    assert artifact["ai_voice_inference"]["status"] == "completed"
    assert artifact["safety_contract"]["name_based_gender_guessing_allowed"] is False


def test_ai_unknown_voice_family_remains_unresolved(tmp_path: Path) -> None:
    transcript = {
        "segments": [
            _segment("seg-1", "SPEAKER_01", 0, 10, "The meeting has started."),
        ]
    }
    provider = _ContextVoiceProvider(
        [
            {
                "speaker_id": "SPEAKER_01",
                "voice_family": "unknown",
                "confidence": 0.0,
                "inference_basis": "insufficient_context",
                "evidence_segment_ids": [],
                "evidence_summary": "The transcript provides no reliable evidence.",
            }
        ]
    )

    resolved = _resolve(
        tmp_path,
        transcript,
        {},
        voice_inference_provider_factory=lambda: provider,
    )

    assert resolved == {}
    artifact = json.loads(
        (tmp_path / "contextual_speaker_identities.json").read_text()
    )
    assert artifact["speakers"][0]["reason"] == (
        "ai_transcript_context_insufficient"
    )


def test_ai_can_ground_an_introduced_broadcast_identity_with_web_search(
    tmp_path: Path,
) -> None:
    source_url = "https://broadcaster.example.test/thembekile-mrototo"
    transcript = {
        "segments": [
            _segment(
                "seg-1",
                "SPEAKER_00",
                0,
                8,
                "Let's take you to SABC News anchor Thembekile Mrototo, "
                "joining us live. What is happening?",
            ),
            _segment(
                "seg-2",
                "SPEAKER_01",
                12,
                30,
                "Well, Nathi, good afternoon. The proceedings are under way.",
            ),
        ]
    }
    provider = _ContextVoiceProvider(
        [
            {
                "speaker_id": "SPEAKER_01",
                "voice_family": "masculine",
                "confidence": 0.97,
                "inference_basis": "grounded_identity_research",
                "identified_name": "Thembekile Mrototo",
                "grounded_source_urls": [source_url],
                "evidence_segment_ids": ["seg-1", "seg-2"],
                "evidence_summary": (
                    "The handoff identifies the remote anchor and a researched "
                    "broadcaster profile explicitly uses masculine pronouns."
                ),
            }
        ],
        server_tool_evidence=[{"url": source_url, "title": "Anchor profile"}],
    )

    resolved = _resolve(
        tmp_path,
        transcript,
        {},
        voice_inference_provider_factory=lambda: provider,
    )

    record = resolved["SPEAKER_01"]
    assert record["identified_name"] == "Thembekile Mrototo"
    assert record["identity_source"] == "grounded_broadcast_identity_research"
    assert record["voice_family"] == "masculine"
    assert record["grounded_source_urls"] == [source_url]
    request = provider.requests[0]
    assert request.server_tools == (
        {
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": 4,
        },
    )


def test_completed_ai_voice_inference_is_reused_from_job_cache(
    tmp_path: Path,
) -> None:
    transcript = {
        "segments": [
            _segment("seg-1", "SPEAKER_00", 0, 5, "Please continue, ma'am."),
            _segment("seg-2", "SPEAKER_01", 5, 10, "Thank you."),
        ]
    }
    provider = _ContextVoiceProvider(
        [
            {
                "speaker_id": "SPEAKER_01",
                "voice_family": "feminine",
                "confidence": 0.95,
                "inference_basis": "explicit_title_or_address",
                "evidence_segment_ids": ["seg-1"],
                "evidence_summary": "SPEAKER_01 is directly addressed as ma'am.",
            }
        ]
    )

    first = _resolve(
        tmp_path,
        transcript,
        {},
        voice_inference_provider_factory=lambda: provider,
    )
    second = _resolve(
        tmp_path,
        transcript,
        {},
        voice_inference_provider_factory=lambda: provider,
    )
    third = _resolve(
        tmp_path,
        transcript,
        {},
        voice_inference_provider_factory=lambda: provider,
    )

    assert first["SPEAKER_01"]["voice_family"] == "feminine"
    assert second["SPEAKER_01"]["voice_family"] == "feminine"
    assert third["SPEAKER_01"]["voice_family"] == "feminine"
    assert len(provider.requests) == 1
    artifact = json.loads(
        (tmp_path / "contextual_speaker_identities.json").read_text()
    )
    assert artifact["ai_voice_inference"]["status"] == "completed_cached"


def test_explicit_outro_can_resolve_a_feminine_voice(tmp_path: Path) -> None:
    transcript = {
        "segments": [
            _segment("seg-1", "SPEAKER_01", 0, 20, "That concludes my report."),
            _segment(
                "seg-2",
                "SPEAKER_00",
                20.2,
                28,
                "That was Naledi Molefe. She is reporting from Johannesburg.",
            ),
        ]
    }

    resolved = _resolve(
        tmp_path,
        transcript,
        _research(("Naledi Molefe", 0.98)),
    )

    assert resolved["SPEAKER_01"]["gender"] == "female"
    assert resolved["SPEAKER_01"]["voice_family"] == "feminine"
    assert resolved["SPEAKER_01"]["proposals"][0]["binding_mode"] == (
        "explicit_presenter_outro"
    )


def test_audited_registry_supplies_gender_only_after_identity_is_rebound(
    tmp_path: Path,
) -> None:
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "identities": {
                    "firoz cachalia": {
                        "canonical_name": "Firoz Cachalia",
                        "gender": "male",
                        "confidence": 0.99,
                    }
                }
            }
        )
    )
    transcript = {
        "segments": [
            _segment("seg-1", "SPEAKER_01", 0, 20, "The briefing continues."),
            _segment(
                "seg-2",
                "SPEAKER_00",
                20.1,
                28,
                "There you have it. That was Firoz Cachalia.",
            ),
        ]
    }

    resolved = _resolve(
        tmp_path,
        transcript,
        _research(("Firoz Cachalia", 0.99)),
        registry_path=registry_path,
    )

    assert resolved["SPEAKER_01"]["gender"] == "male"
    assert resolved["SPEAKER_01"]["gender_source"] == (
        "audited_contextual_identity_registry"
    )


def _write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def test_direct_dub_uses_context_instead_of_reclassifying_inconclusive_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_id = "job1"
    analysis = tmp_path / "jobs" / job_id / "analysis"
    transcript = {
        "segments": [
            {
                **_segment(
                    "seg-1",
                    "SPEAKER_00",
                    0,
                    10,
                    "Acting police minister Firoz Cachalia, he's briefing now.",
                ),
                "raw_speaker_id": "AZURE_1",
            },
            {
                **_segment("seg-2", "SPEAKER_01", 10, 30, "This is intolerable."),
                "raw_speaker_id": "AZURE_2",
            },
            {
                **_segment(
                    "seg-3",
                    "SPEAKER_00",
                    30,
                    40,
                    "There you have it. That's Firoz Cachalia. He's briefing.",
                ),
                "raw_speaker_id": "AZURE_1",
            },
        ]
    }
    _write(analysis / "transcript_en.json", transcript)
    _write(
        analysis / "azure_diarization.json",
        {
            "turns": [
                {"speaker": "AZURE_1", "start": 0, "end": 10},
                {"speaker": "AZURE_2", "start": 10, "end": 30},
                {"speaker": "AZURE_1", "start": 30, "end": 40},
            ]
        },
    )
    _write(
        analysis / "acoustic_analysis.json",
        {
            "speakers": {
                "AZURE_1": {
                    "speaker_id": "AZURE_1",
                    "voice_family": "masculine",
                    "confidence": 0.96,
                    "evidence": {"method": "ecapa_voice_family_classifier"},
                },
                "AZURE_2": {
                    "speaker_id": "AZURE_2",
                    "voice_family": "unknown",
                    "confidence": 0.0,
                    "probabilities": {"masculine": 0.522, "feminine": 0.478},
                    "evidence": {"method": "ecapa_voice_family_classifier"},
                },
            }
        },
    )
    _write(
        analysis / "autocorrection_name_research.json",
        _research(("Firoz Cachalia", 0.99)),
    )
    monkeypatch.setattr(
        "mathula_tv.direct_azure_dub.ensure_canonical_voice_family_analysis",
        lambda **kwargs: pytest.fail(
            "contextually resolved speaker must not rerun acoustic classification"
        ),
    )

    renderer = DirectAzureDubRenderer(
        SimpleNamespace(work_dir=tmp_path), SimpleNamespace()
    )
    output = tmp_path / "jobs" / job_id / "direct_dub" / "voice_resolution.json"
    result = renderer._voice_assignments(
        SimpleNamespace(job_id=job_id),
        None,
        DirectDubOptions(),
        required_speaker_ids=["SPEAKER_00", "SPEAKER_01"],
        output_path=output,
    )

    assert result["SPEAKER_01"]["identified_name"] == "Firoz Cachalia"
    assert result["SPEAKER_01"]["selected_voice"] == "zu-ZA-ThembaNeural"
    assert result["SPEAKER_01"]["voice_family_source"] == (
        "contextual_transcript_identity_gender"
    )
    assert result["SPEAKER_01"]["resolution_status"] == (
        "contextual_speaker_identity"
    )
    artifact = json.loads(output.read_text())
    assert artifact["unresolved_speakers"] == []
    assert artifact["unknown_defaults_to_masculine"] is False
