from pathlib import Path
from types import SimpleNamespace

import pytest

from mathula_tv.atomic_io import atomic_write_json
from mathula_tv.autocorrect_stage import (
    AutocorrectStageConfig,
    AutocorrectionReviewRequired,
    require_authoritative_transcript,
    run_autocorrection_first,
)
import mathula_tv.autocorrect_stage as autocorrect_stage
from mathula_tv.config import Settings
from mathula_tv.media import checksum
from mathula_tv.production_pipeline import ProductionDubbingPipeline


def settings(tmp_path: Path) -> Settings:
    return Settings(
        tmp_path,
        "bucket",
        "mathula-tv",
        tmp_path / "case",
        tmp_path / "politics",
        "",
        "eastus",
        "en-ZA",
        "2025-10-15",
        10,
        600,
        "",
        "",
        "preview",
        "model",
    )


def transcript() -> dict:
    return {
        "schema_version": "transcript-v1",
        "duration": 2.0,
        "complete_text": "First, we had a Madlanga ward.",
        "autocorrect": {
            "schema_version": "mathula-autocorrect-state-v1",
            "active_correction_count": 1,
            "overlays": [
                {
                    "azure_text": "malanga ward",
                    "corrected_text": "Madlanga ward",
                    "start_ms": 500,
                    "end_ms": 1800,
                }
            ],
        },
        "segments": [
            {
                "segment_id": "seg-00001",
                "speaker": "AZURE_1",
                "start": 0.0,
                "end": 2.0,
                "source_text": "First, we had a Madlanga ward.",
                "source_phrase_ids": ["seg-00001"],
                "source_word_ids": [
                    "word_000001",
                    "word_000002",
                    "word_000003",
                    "word_000004",
                    "word_000005",
                    "word_000006",
                ],
                "overlap": False,
            }
        ],
        "words": [
            {
                "word_id": f"word_{index:06d}",
                "source_phrase_id": "seg-00001",
                "text": text,
                "start": (index - 1) * 0.3,
                "end": index * 0.3,
                "speaker": "AZURE_1",
            }
            for index, text in enumerate(
                ["First", "we", "had", "a", "malanga", "ward"],
                1,
            )
        ],
    }


def write_authoritative_job(tmp_path: Path, *, review_count: int) -> tuple[ProductionDubbingPipeline, object]:
    app = ProductionDubbingPipeline(settings(tmp_path))
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    job = app.jobs.create(
        source_filename=source.name,
        local_source_path=str(source),
        source_checksum="a" * 64,
        source_duration=2.0,
    )
    root = app.jobs.job_dir(job.job_id)
    transcript_path = root / "analysis/transcript_en.json"
    raw_path = root / "analysis/transcript_en_raw.json"
    state_path = root / "analysis/autocorrection_state.json"
    atomic_write_json(transcript_path, transcript())
    atomic_write_json(raw_path, transcript())
    atomic_write_json(
        state_path,
        {
            "stage_schema_version": "autocorrection-first-stage-v1",
            "review_count": review_count,
            "active_count": 1,
            "ready_for_language_ai": review_count == 0,
            "raw_transcript_sha256": checksum(raw_path),
            "authoritative_transcript_sha256": checksum(transcript_path),
        },
    )
    return app, job


def test_autocorrection_review_blocks_all_downstream_text_work(tmp_path):
    app, job = write_authoritative_job(tmp_path, review_count=1)
    with pytest.raises(AutocorrectionReviewRequired, match="before classification"):
        app.build_units(job)


def test_units_record_authoritative_autocorrected_transcript_identity(tmp_path):
    app, job = write_authoritative_job(tmp_path, review_count=0)
    identity = require_authoritative_transcript(
        work_dir=tmp_path,
        job_id=job.job_id,
    )
    units = app.build_units(job)

    assert "Madlanga ward" in units["units"][0]["source_text"]
    assert "malanga ward" not in units["units"][0]["source_text"].casefold()
    assert units["source_transcript_sha256"] == identity["sha256"]
    assert units["source_autocorrection_state_sha256"] == identity["state_sha256"]


def test_ai_stt_ranking_runs_before_authoritative_render(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = transcript()
    source["autocorrect"] = {
        "schema_version": "mathula-autocorrect-state-v1",
        "overlays": [],
    }
    job = SimpleNamespace(job_id="job-ai-stt")
    applied = {}

    class FakeStore:
        def __init__(self, *_args, **_kwargs):
            pass

        def close(self):
            pass

    def fake_ensure(_store, *, paths, **_kwargs):
        atomic_write_json(paths.snapshot, source)

    def fake_apply(_store, **kwargs):
        applied.update(kwargs)
        return {
            "decision_count": 1,
            "auto_applied_count": 1,
            "suggested_count": 0,
            "dismissed_count": 0,
            "review_count": 0,
        }

    monkeypatch.setattr(autocorrect_stage, "Store", FakeStore)
    monkeypatch.setattr(autocorrect_stage, "ensure_job", fake_ensure)
    monkeypatch.setattr(
        autocorrect_stage, "detect_candidates", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        autocorrect_stage,
        "build_ai_request",
        lambda *_args, **_kwargs: {
            "schema_version": "mathula-autocorrect-ai-request-v1",
            "request_sha256": "request-hash",
            "spans": [{"correction_id": "correction-1"}],
        },
    )
    monkeypatch.setattr(autocorrect_stage, "apply_ai_advice", fake_apply)
    monkeypatch.setattr(
        autocorrect_stage,
        "render_effective_transcript",
        lambda *_args, **_kwargs: {
            "effective_sha256": "effective-hash",
            "active_count": 1,
            "review_count": 0,
            "corrections": [],
        },
    )
    monkeypatch.setattr(
        autocorrect_stage,
        "run_name_research_autocorrection",
        lambda **kwargs: {
            "transcript": kwargs["transcript"],
            "artifact": {
                "status": "no_candidates",
                "candidate_count": 0,
                "accepted_count": 0,
                "applied_count": 0,
                "unresolved_count": 0,
            },
        },
    )

    result = run_autocorrection_first(
        work_dir=tmp_path,
        job=job,
        reconciled_transcript=source,
        config=AutocorrectStageConfig(
            database_url="postgresql://unused",
            schema="test",
            hints_path=None,
            registry_path=None,
            user_id=None,
            project_id="mathula-tv",
            fuzzy_threshold=0.68,
            max_candidates=4,
            low_confidence_threshold=0.72,
            auto_confirmation_threshold=2,
        ),
        ai_ranker=lambda _request: (
            {
                "correction-1": {
                    "correction_id": "correction-1",
                    "decision": "ACCEPT_CANDIDATE",
                    "candidate_id": "entity:madlanga",
                    "confidence": 0.98,
                    "reason": "Known name in context",
                    "question": "",
                }
            },
            {"provider": "azure-openai-gpt"},
        ),
    )

    assert applied["auto_apply_confidence"] == 0.92
    assert applied["minimum_candidate_score"] == 0.75
    assert result["ai_stt_autocorrect_status"] == "completed"
    assert result["ai_stt_auto_applied_count"] == 1
    assert result["ready_for_language_ai"] is True
    assert (
        tmp_path
        / "jobs/job-ai-stt/analysis/autocorrection_ai_response.json"
    ).is_file()
