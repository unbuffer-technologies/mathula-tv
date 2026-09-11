from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mathula_tv.autocorrect_research import (
    extract_name_candidates,
    run_name_research_autocorrection,
)
from mathula_tv.manual_ai import ManualAIResponseRequired


class Metadata:
    server_tool_evidence = ()

    def to_dict(self):
        return {"provider": "test", "server_tool_evidence": []}


class EntityProvider:
    def complete_structured(self, request):
        candidates = extract_name_candidates(request.payload["transcript"])
        entities = [
            {
                "entity_id": f"entity-{index}",
                "canonical_name": "",
                "entity_type": "person",
                "confidence": 0.5,
                "needs_research": True,
                "reason": "test fixture marks name as uncertain",
                "surface_forms": [item["raw_text"]],
                "mentions": [
                    {
                        "segment_id": item["segment_id"],
                        "raw_text": item["raw_text"],
                    }
                ],
            }
            for index, item in enumerate(candidates, start=1)
        ]
        return SimpleNamespace(
            data={
                "schema_version": "mathula-autocorrect-entity-inventory-v1",
                "entities": entities,
            },
            metadata=Metadata(),
        )


def _transcript() -> dict:
    return {
        "schema_version": "transcript-v1",
        "complete_text": "Kanyaku said the charges against Khumalo were withdrawn.",
        "segments": [
            {
                "segment_id": "seg-1",
                "start": 0.0,
                "end": 4.0,
                "source_text": "Kanyaku said the charges against Khumalo were withdrawn.",
                "words": [],
            }
        ],
        "words": [],
        "autocorrect": {"schema_version": "autocorrect-v1"},
    }


def _job(job_id: str):
    return SimpleNamespace(
        job_id=job_id,
        source_filename="Khumalo charges report.mp4",
        local_source_path="/tmp/source.mp4",
        source_language="en-ZA",
    )


def test_manual_mode_exports_web_research_request_instead_of_azure(tmp_path: Path, monkeypatch) -> None:
    job_root = tmp_path / "working" / "jobs" / "job-manual-web"
    (job_root / "analysis").mkdir(parents=True)
    manual_root = job_root / "manual_ai"
    monkeypatch.setenv("MATHULA_TV_AI_PROVIDER", "manual-chatgpt")
    monkeypatch.setenv("MATHULA_TV_MANUAL_AI_DIR", str(manual_root))

    # If the dedicated Azure client is touched, this regression has returned.
    monkeypatch.setattr(
        "mathula_tv.autocorrect_research.AzureResponsesWebResearchProvider.from_environment",
        lambda: (_ for _ in ()).throw(AssertionError("Azure web research must not be created in manual mode")),
    )

    with pytest.raises(ManualAIResponseRequired):
        run_name_research_autocorrection(
            job_root=job_root,
            job=_job("job-manual-web"),
            transcript=_transcript(),
            provider=None,
            entity_provider=EntityProvider(),
            force=True,
        )

    requests = list((manual_root / "requests").glob("*.json"))
    assert len(requests) == 1
    envelope = json.loads(requests[0].read_text(encoding="utf-8"))
    assert envelope["operation"] == "autocorrect_name_research"
    assert envelope["server_tools"] == [
        {"type": "web_search_20250305", "name": "web_search", "max_uses": 8}
    ]


def test_manual_entity_inventory_handoff_is_not_swallowed_as_advisory(tmp_path: Path, monkeypatch) -> None:
    job_root = tmp_path / "working" / "jobs" / "job-manual-inventory"
    (job_root / "analysis").mkdir(parents=True)
    manual_root = job_root / "manual_ai"
    monkeypatch.setenv("MATHULA_TV_AI_PROVIDER", "manual-chatgpt")
    monkeypatch.setenv("MATHULA_TV_MANUAL_AI_DIR", str(manual_root))

    with pytest.raises(ManualAIResponseRequired):
        run_name_research_autocorrection(
            job_root=job_root,
            job=_job("job-manual-inventory"),
            transcript=_transcript(),
            provider=None,
            entity_provider=None,
            force=True,
        )

    requests = list((manual_root / "requests").glob("*.json"))
    assert len(requests) == 1
    envelope = json.loads(requests[0].read_text(encoding="utf-8"))
    assert envelope["operation"] == "autocorrect_entity_inventory"
