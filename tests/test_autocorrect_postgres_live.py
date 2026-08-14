import json
import os
import uuid

import pytest

from mathula_tv.autocorrect import (
    JobPaths,
    Store,
    change_correction,
    detect_candidates,
    ensure_job,
    list_corrections,
    load_json,
    moderate_public_term,
    platform_candidate_report,
    render_effective_transcript,
)

pytestmark = pytest.mark.live


@pytest.fixture
def live_store():
    database_url = os.environ.get("MATHULA_TV_TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("MATHULA_TV_TEST_DATABASE_URL is not configured")
    schema = f"mathula_test_{uuid.uuid4().hex[:12]}"
    store = Store(database_url, schema=schema)
    try:
        yield store
    finally:
        with store.pool.connection() as connection:
            connection.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
            connection.commit()
        store.close()


def test_postgres_new_term_public_workflow(tmp_path, live_store, monkeypatch):
    job_id = uuid.uuid4().hex
    paths = JobPaths.for_job(tmp_path, job_id)
    paths.effective.parent.mkdir(parents=True)
    paths.effective.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "segment_id": "s1",
                        "speaker": "SPEAKER_00",
                        "start": 0,
                        "end": 3,
                        "source_text": "That new slang is horse curry.",
                    }
                ],
                "complete_text": "That new slang is horse curry.",
            }
        ),
        encoding="utf-8",
    )
    hints = tmp_path / "hints.json"
    hints.write_text(
        json.dumps(
            {
                "rules": [
                    {
                        "pattern": r"\bhorse curry\b",
                        "replacement": "hoshkhari",
                        "flags": ["IGNORECASE"],
                    }
                ],
                "review_patterns": [],
            }
        ),
        encoding="utf-8",
    )
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"entities": []}), encoding="utf-8")

    ensure_job(
        live_store,
        paths=paths,
        job_id=job_id,
        user_id="user-1",
        project_id="project-1",
    )
    detect_candidates(
        live_store,
        job_id=job_id,
        user_id="user-1",
        project_id="project-1",
        source_document=load_json(paths.snapshot),
        hints_path=hints,
        registry_path=registry,
        fuzzy_threshold=0.68,
        max_candidates=4,
        low_confidence_threshold=0.72,
        auto_confirmation_threshold=2,
    )
    correction = list_corrections(live_store, job_id=job_id)[0]
    assert correction["status"] == "auto_applied"
    state = render_effective_transcript(
        live_store,
        paths=paths,
        job_id=job_id,
    )
    assert state["review_count"] == 0
    assert "hoshkhari" in load_json(paths.effective)["complete_text"]

    monkeypatch.setenv("MATHULA_TV_FEEDBACK_HASH_SALT", "test-secret")
    updated = change_correction(
        live_store,
        job_id=job_id,
        correction_id=correction["correction_id"],
        action="edit",
        actor_id="user-1",
        replacement_text="brand new kasi term",
        learn_scopes=["user", "project"],
        user_id="user-1",
        project_id="project-1",
        contribute_globally=True,
        term_category="slang",
        language="en-ZA",
        domain="street culture",
    )
    assert updated["status"] == "edited"
    assert updated["user_supplied"] is True

    state = render_effective_transcript(
        live_store,
        paths=paths,
        job_id=job_id,
    )
    assert state["active_count"] == 1
    assert "brand new kasi term" in load_json(paths.effective)["complete_text"]

    candidates = platform_candidate_report(
        live_store,
        minimum_users=1,
        minimum_jobs=1,
        status="candidate",
    )
    assert len(candidates) == 1
    approved = moderate_public_term(
        live_store,
        term_id=candidates[0]["term_id"],
        action="approve",
        moderator_id="moderator",
        minimum_users=1,
        minimum_jobs=1,
    )
    assert approved["status"] == "approved"
