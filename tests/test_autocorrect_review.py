from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import pytest

from mathula_tv import autocorrect_review as review
from mathula_tv.autocorrect import sha256_json
from mathula_tv.autocorrect_stage import AutocorrectStageConfig
from mathula_tv.media import checksum


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def transcript() -> dict:
    return {
        "schema_version": "test",
        "segments": [
            {
                "segment_id": "seg-00004",
                "source_text": "Advocate Mhutibi reviewed the matters. NDPPs used the powers.",
                "words": [
                    {"text": "Advocate"},
                    {"text": "Mhutibi"},
                    {"text": "reviewed"},
                    {"text": "the"},
                    {"text": "matters"},
                ],
            }
        ],
        "complete_text": "Advocate Mhutibi reviewed the matters. NDPPs used the powers.",
        "autocorrect": {"source_sha256": "source", "name_research": {"applied_count": 5}},
    }


def research_artifact() -> dict:
    return {
        "schema_version": "mathula-autocorrect-name-research-artifact-v1",
        "status": "completed",
        "candidate_name_spans": [
            {
                "segment_id": "seg-00004",
                "raw_text": "Advocate Mhutibi",
                "source_text": "Advocate Mhutibi reviewed the matters.",
            },
            {
                "segment_id": "seg-00004",
                "raw_text": "NDPPs",
                "source_text": "NDPPs used the powers.",
            },
        ],
        "rejected_corrections": [
            {
                "segment_id": "seg-00004",
                "raw_text": "Advocate Mhutibi",
                "canonical_text": "Advocate Mothibi",
                "entity_type": "person",
                "confidence": 0.83,
                "reason": "Same-story evidence was weaker than threshold.",
                "validation_reason": "confidence_below_0.95",
                "source_urls": ["https://example.org/story"],
                "story_match_terms": ["IDAC", "Crime Intelligence"],
            }
        ],
        "unresolved_candidates": [
            {
                "segment_id": "seg-00004",
                "raw_text": "NDPPs",
                "reason": "Not a proper-name correction.",
            }
        ],
        "research_profile": {"name": "madlanga_commission"},
        "rejected_count": 1,
        "unresolved_count": 1,
        "applied_count": 5,
    }


def config() -> AutocorrectStageConfig:
    return AutocorrectStageConfig(
        database_url="postgresql://unused",
        schema="mathula_autocorrect",
        hints_path=None,
        registry_path=None,
        user_id="reviewer",
        project_id="mathula-tv",
        fuzzy_threshold=0.68,
        max_candidates=4,
        low_confidence_threshold=0.72,
        auto_confirmation_threshold=2,
    )


def test_build_review_queue_has_rejected_and_unresolved() -> None:
    queue = review.build_review_queue(research_artifact(), transcript())
    assert [item.source_status for item in queue] == ["rejected", "unresolved"]
    assert queue[0].suggested_text == "Advocate Mothibi"
    assert "Mhutibi" in queue[0].context


def test_build_review_queue_skips_final_decisions() -> None:
    artifact = research_artifact()
    key = review.decision_key("seg-00004", "NDPPs")
    queue = review.build_review_queue(
        artifact,
        transcript(),
        existing_decisions={key: {"action": "not_a_name_project"}},
    )
    assert [item.raw_text for item in queue] == ["Advocate Mhutibi"]


def test_project_not_name_suppression_is_persisted(tmp_path: Path) -> None:
    path = tmp_path / "not_name_suppressions.json"
    count = review._save_project_suppressions(
        path,
        [
            {
                "action": "not_a_name_project",
                "raw_text": "NDPPs",
                "reason": "Grammar, not a name",
                "reviewer": "Clerence",
                "decided_at": "2026-07-28T00:00:00Z",
            }
        ],
    )
    assert count == 1
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["suppressions"][0]["raw_norm"] == "ndpps"


class FakeResult:
    def __init__(self, row=None):
        self.row = row

    def fetchone(self):
        return self.row


class FakeDb:
    def __init__(self, row):
        self.row = row
        self.calls = []

    def execute(self, sql, params=()):
        self.calls.append((" ".join(str(sql).split()), params))
        if str(sql).lstrip().startswith("SELECT source_sha256"):
            return FakeResult(self.row)
        return FakeResult(None)


class FakeStore:
    instance = None

    def __init__(self, *args, **kwargs):
        self.db = FakeDb(FakeStore.row)
        FakeStore.instance = self

    @contextmanager
    def transaction(self):
        yield self.db

    def close(self):
        pass


def test_apply_review_updates_transcript_hash_and_audit(monkeypatch, tmp_path: Path) -> None:
    work = tmp_path / "working"
    paths = review.ReviewPaths.for_job(work, "job1")
    source = transcript()
    write_json(paths.transcript, source)
    write_json(paths.deterministic_transcript, source)
    write_json(paths.research, research_artifact())
    write_json(
        paths.autocorrection_state,
        {
            "ready_for_language_ai": True,
            "authoritative_transcript_sha256": checksum(paths.transcript),
        },
    )
    write_json(paths.job_root / "manifest.json", {"source_language": "en-ZA"})

    review_artifact = {
        "schema_version": review.MANUAL_REVIEW_SCHEMA,
        "job_id": "job1",
        "status": "draft",
        "decisions": [
            {
                "key": review.decision_key("seg-00004", "Advocate Mhutibi"),
                "segment_id": "seg-00004",
                "raw_text": "Advocate Mhutibi",
                "source_status": "rejected",
                "action": "approve",
                "replacement_text": "Advocate Mothibi",
                "reviewer": "Clerence",
                "reason": "Confirmed exact story",
                "learn_for_future_jobs": True,
                "suggested_text": "Advocate Mothibi",
                "entity_type": "person",
                "confidence": 0.83,
                "source_urls": ["https://example.org/story"],
                "story_match_terms": ["IDAC", "Crime Intelligence"],
                "decided_at": "2026-07-28T00:00:00Z",
            },
            {
                "key": review.decision_key("seg-00004", "NDPPs"),
                "segment_id": "seg-00004",
                "raw_text": "NDPPs",
                "source_status": "unresolved",
                "action": "not_a_name_project",
                "replacement_text": None,
                "reviewer": "Clerence",
                "reason": "Grammar, not a name",
                "learn_for_future_jobs": False,
                "decided_at": "2026-07-28T00:00:00Z",
            },
        ],
    }
    FakeStore.row = {
        "source_sha256": "not-current",
        "effective_sha256": sha256_json(source),
    }
    monkeypatch.setattr(review, "Store", FakeStore)
    monkeypatch.setattr(review, "_save_future_corrections", lambda **kwargs: 1)

    result = review.apply_review_decisions(
        paths=paths,
        config=config(),
        review_artifact=review_artifact,
        research_artifact=research_artifact(),
        reviewer="Clerence",
        dry_run=False,
    )

    effective = json.loads(paths.transcript.read_text(encoding="utf-8"))
    assert "Advocate Mothibi" in effective["segments"][0]["source_text"]
    assert "Mhutibi" not in effective["segments"][0]["source_text"]
    assert effective["autocorrect"]["name_research"] == {"applied_count": 5}
    assert effective["autocorrect"]["manual_review"]["applied_count"] == 1
    assert result["project_suppressions_saved"] == 1
    assert result["future_correction_cache_saved"] == 1
    assert paths.review.is_file()
    assert paths.project_not_names.is_file()
    state = json.loads(paths.autocorrection_state.read_text(encoding="utf-8"))
    assert state["authoritative_transcript_sha256"] == checksum(paths.transcript)
    assert state["effective_sha256"] == sha256_json(effective)
    assert state["manual_review_status"] == "completed"
    update_calls = [call for call in FakeStore.instance.db.calls if call[0].startswith("UPDATE jobs")]
    assert update_calls


def test_apply_review_dry_run_does_not_write(tmp_path: Path) -> None:
    paths = review.ReviewPaths.for_job(tmp_path / "working", "job1")
    source = transcript()
    write_json(paths.transcript, source)
    write_json(paths.deterministic_transcript, source)
    write_json(
        paths.autocorrection_state,
        {
            "ready_for_language_ai": True,
            "authoritative_transcript_sha256": checksum(paths.transcript),
        },
    )
    artifact = {
        "schema_version": review.MANUAL_REVIEW_SCHEMA,
        "job_id": "job1",
        "decisions": [
            {
                "segment_id": "seg-00004",
                "raw_text": "Advocate Mhutibi",
                "action": "edit",
                "replacement_text": "Advocate Mothibi",
                "entity_type": "person",
                "learn_for_future_jobs": False,
            }
        ],
    }
    result = review.apply_review_decisions(
        paths=paths,
        config=config(),
        review_artifact=artifact,
        research_artifact=research_artifact(),
        reviewer="Clerence",
        dry_run=True,
    )
    assert result["applied_count"] == 1
    assert json.loads(paths.transcript.read_text(encoding="utf-8")) == source
    assert not paths.review.exists()


def test_synchronize_completed_review_state_repairs_v1361_gap(monkeypatch, tmp_path: Path) -> None:
    paths = review.ReviewPaths.for_job(tmp_path / "working", "job1")
    source = transcript()
    source["segments"][0]["source_text"] = "Advocate Mothibi reviewed the matters."
    source["complete_text"] = "Advocate Mothibi reviewed the matters."
    write_json(paths.transcript, source)
    write_json(
        paths.autocorrection_state,
        {
            "ready_for_language_ai": True,
            "authoritative_transcript_sha256": "stale-file-hash",
            "effective_sha256": "stale-json-hash",
        },
    )
    artifact = {
        "status": "completed",
        "reviewer": "Clerence",
        "decisions": [
            {
                "segment_id": "seg-00004",
                "raw_text": "Advocate Mhutibi",
                "replacement_text": "Advocate Mothibi",
                "action": "approve",
            }
        ],
    }
    write_json(paths.review, artifact)
    FakeStore.row = {
        "source_sha256": "not-current",
        "effective_sha256": sha256_json(source),
    }
    monkeypatch.setattr(review, "Store", FakeStore)

    result = review.synchronize_completed_review_state(
        paths=paths,
        config=config(),
        review_artifact=artifact,
    )

    state = json.loads(paths.autocorrection_state.read_text(encoding="utf-8"))
    assert result["status"] == "state_synchronized"
    assert state["authoritative_transcript_sha256"] == checksum(paths.transcript)
    assert state["effective_sha256"] == sha256_json(source)
    assert state["manual_review_status"] == "completed"
