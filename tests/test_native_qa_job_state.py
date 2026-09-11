from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import mathula_tv.cli as cli
import mathula_tv.native_dub as native_dub


def _job(job_id: str = "job-1") -> SimpleNamespace:
    return SimpleNamespace(
        job_id=job_id,
        state="analysis_running",
        completed_stages=["audio_preparation", "upload", "azure_transcription"],
        review_readiness="compatibility_pending",
        media={},
        providers={},
        updated_at=None,
    )


class _Jobs:
    def __init__(self, job, job_root: Path):
        self._job = job
        self._job_root = job_root
        self.saved: list = []

    def load(self, job_id):
        return self._job

    def job_dir(self, job_id):
        return self._job_root

    def save(self, job):
        self.saved.append(job)


class _App:
    def __init__(self, settings, jobs):
        self.jobs = jobs


def _qa_result(*, flagged_count: int) -> dict:
    return {
        "schema_version": "mathula-native-qa-back-translation-v1",
        "sentence_count": 5,
        "flagged_count": flagged_count,
        "requires_review": flagged_count > 0,
        "flagged": [
            {"ref": f"g{i}", "discrepancy_summary": "different question"} for i in range(flagged_count)
        ],
        "model": "grok-4.3",
        "usage": {"input_tokens": 100, "output_tokens": 50},
    }


def _wire_common(monkeypatch, tmp_path, job, qa_result):
    jobs = _Jobs(job, tmp_path)
    settings = SimpleNamespace(work_dir=tmp_path)
    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(cli, "Orchestrator", lambda settings: _App(settings, jobs))
    monkeypatch.setattr(
        cli,
        "create_native_grok_backend",
        lambda model, progress=None: SimpleNamespace(config=SimpleNamespace(deployment="grok-4.3")),
    )
    monkeypatch.setattr(native_dub, "run_native_qa_back_translation", lambda **kwargs: qa_result)
    return jobs


def test_native_qa_leaves_review_readiness_untouched_when_nothing_flagged(monkeypatch, tmp_path):
    job = _job()
    jobs = _wire_common(monkeypatch, tmp_path, job, _qa_result(flagged_count=0))

    assert cli.main(["native-qa", job.job_id, "--json"]) == 0

    saved = jobs.saved[-1]
    assert saved.review_readiness == "compatibility_pending"  # unchanged


def test_native_qa_marks_needs_translation_review_when_sentences_flagged(monkeypatch, tmp_path):
    job = _job()
    jobs = _wire_common(monkeypatch, tmp_path, job, _qa_result(flagged_count=2))

    assert cli.main(["native-qa", job.job_id, "--json"]) == 0

    saved = jobs.saved[-1]
    assert saved.review_readiness == "needs_translation_review"
