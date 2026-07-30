from __future__ import annotations

import importlib.util
import json
from contextlib import contextmanager
from pathlib import Path

from mathula_tv import autocorrect_stage
from mathula_tv.autocorrect import sha256_json
from mathula_tv.autocorrect_stage import AutocorrectStageConfig


class FakeDatabase:
    def __init__(self, row: dict[str, object] | None = None) -> None:
        self.calls: list[tuple[str, tuple[object, ...]]] = []
        self.row = row

    def execute(self, sql: str, params: tuple[object, ...]):
        self.calls.append((" ".join(sql.split()), params))
        return self

    def fetchone(self):
        return self.row


class FakeStore:
    instances: list["FakeStore"] = []
    next_row: dict[str, object] | None = None

    def __init__(self, database_url: str, *, schema: str) -> None:
        self.database_url = database_url
        self.schema = schema
        self.database = FakeDatabase(self.next_row)
        self.closed = False
        self.instances.append(self)

    @contextmanager
    def transaction(self):
        yield self.database

    def close(self) -> None:
        self.closed = True


def stage_config(tmp_path: Path) -> AutocorrectStageConfig:
    return AutocorrectStageConfig(
        database_url="postgresql://example",
        schema="mathula_autocorrect",
        hints_path=None,
        registry_path=None,
        user_id=None,
        project_id=None,
        fuzzy_threshold=0.85,
        max_candidates=100,
        low_confidence_threshold=0.5,
        auto_confirmation_threshold=2,
    )


def test_researched_transcript_hash_is_registered(monkeypatch, tmp_path: Path) -> None:
    FakeStore.instances.clear()
    monkeypatch.setattr(autocorrect_stage, "Store", FakeStore)
    monkeypatch.setattr(autocorrect_stage, "utcnow", lambda: "2026-07-28T11:30:00Z")
    transcript = {
        "complete_text": "Kaizer Kganyago spoke.",
        "segments": [
            {"segment_id": "seg-1", "source_text": "Kaizer Kganyago spoke."}
        ],
    }

    result = autocorrect_stage._register_researched_effective_transcript(
        config=stage_config(tmp_path),
        job_id="job-1",
        transcript=transcript,
    )

    assert result == sha256_json(transcript)
    store = FakeStore.instances[-1]
    assert store.closed is True
    update_sql, update_params = store.database.calls[0]
    assert "UPDATE jobs SET effective_sha256" in update_sql
    assert update_params == (result, "2026-07-28T11:30:00Z", "job-1")
    event_sql, event_params = store.database.calls[1]
    assert "name_research_rendered" in event_sql
    assert event_params[0] == "job-1"
    assert json.loads(str(event_params[1])) == {"effective_sha256": result}


def load_rerun_script(repo_root: Path):
    path = repo_root / "scripts" / "rerun_autocorrect_name_research.py"
    spec = importlib.util.spec_from_file_location("rerun_autocorrect_name_research", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_explicit_rerun_keeps_database_authorized_effective(monkeypatch, tmp_path: Path) -> None:
    FakeStore.instances.clear()
    paths = type("Paths", (), {})()
    paths.effective = tmp_path / "analysis" / "transcript_en.json"
    paths.snapshot = tmp_path / "analysis" / "transcript_en.autocorrect_source.json"
    paths.corrected = tmp_path / "analysis" / "corrected_transcript.json"
    paths.effective.parent.mkdir(parents=True)

    current = {"complete_text": "Kaizer Kganyago spoke."}
    snapshot = {"complete_text": "Kaiser Kanyaku spoke."}
    corrected = {
        "complete_text": "Kaiser Kanyaku spoke.",
        "autocorrect": {"source_sha256": sha256_json(snapshot)},
    }
    paths.effective.write_text(json.dumps(current), encoding="utf-8")
    paths.snapshot.write_text(json.dumps(snapshot), encoding="utf-8")
    paths.corrected.write_text(json.dumps(corrected), encoding="utf-8")

    FakeStore.next_row = {
        "source_sha256": sha256_json(snapshot),
        "effective_sha256": sha256_json(current),
    }
    monkeypatch.setattr(autocorrect_stage, "Store", FakeStore)

    result = autocorrect_stage.prepare_explicit_name_research_rerun(
        config=stage_config(tmp_path),
        paths=paths,
        job_id="job-1",
    )

    assert result["status"] == "authorized"
    assert json.loads(paths.effective.read_text(encoding="utf-8")) == current
    assert len(FakeStore.instances[-1].database.calls) == 1


def test_explicit_rerun_recovers_legacy_hash_mismatch(monkeypatch, tmp_path: Path) -> None:
    FakeStore.instances.clear()
    paths = type("Paths", (), {})()
    paths.effective = tmp_path / "analysis" / "transcript_en.json"
    paths.snapshot = tmp_path / "analysis" / "transcript_en.autocorrect_source.json"
    paths.corrected = tmp_path / "analysis" / "corrected_transcript.json"
    paths.effective.parent.mkdir(parents=True)

    snapshot = {"complete_text": "Kaiser Kanyaku spoke."}
    current = {"complete_text": "Kaiser Kanyaku spoke.", "legacy": True}
    corrected = {
        "complete_text": "Kaiser Kanyaku spoke.",
        "autocorrect": {"source_sha256": sha256_json(snapshot)},
    }
    registered = {"complete_text": "Kaizer Kganyago spoke."}
    paths.effective.write_text(json.dumps(current), encoding="utf-8")
    paths.snapshot.write_text(json.dumps(snapshot), encoding="utf-8")
    paths.corrected.write_text(json.dumps(corrected), encoding="utf-8")

    FakeStore.next_row = {
        "source_sha256": sha256_json(snapshot),
        "effective_sha256": sha256_json(registered),
    }
    monkeypatch.setattr(autocorrect_stage, "Store", FakeStore)
    monkeypatch.setattr(autocorrect_stage, "utcnow", lambda: "2026-07-28T13:00:00Z")

    result = autocorrect_stage.prepare_explicit_name_research_rerun(
        config=stage_config(tmp_path),
        paths=paths,
        job_id="job-1",
    )

    assert result["status"] == "recovered"
    assert json.loads(paths.effective.read_text(encoding="utf-8")) == corrected
    calls = FakeStore.instances[-1].database.calls
    assert "UPDATE jobs SET effective_sha256" in calls[1][0]
    assert "name_research_rerun_integrity_recovered" in calls[2][0]
