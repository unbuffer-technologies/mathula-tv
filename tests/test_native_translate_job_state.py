from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import mathula_tv.cli as cli
import mathula_tv.native_dub as native_dub


def _job(job_id: str = "job-1") -> SimpleNamespace:
    return SimpleNamespace(
        job_id=job_id,
        state="analysis_ready",
        completed_stages=["audio_preparation", "upload", "azure_transcription"],
        media={},
        providers={},
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


def _wire_common(monkeypatch, tmp_path, job, *, pass1_result=None, pass2_result=None, translation_result=None):
    jobs = _Jobs(job, tmp_path)
    settings = SimpleNamespace(work_dir=tmp_path)
    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(cli, "Orchestrator", lambda settings: _App(settings, jobs))
    monkeypatch.setattr(
        cli,
        "create_native_grok_backend",
        lambda model, progress=None: SimpleNamespace(config=SimpleNamespace(deployment="grok-4.3")),
    )
    monkeypatch.setattr(cli, "create_tts_backend", lambda settings: object())
    if pass1_result is not None:
        monkeypatch.setattr(native_dub, "run_native_pass1", lambda **kwargs: pass1_result)
    if pass2_result is not None:
        monkeypatch.setattr(native_dub, "run_native_pass2", lambda **kwargs: pass2_result)
    if translation_result is not None:
        monkeypatch.setattr(native_dub, "run_native_translation", lambda **kwargs: translation_result)
    return jobs


def test_native_translate_pass1_only_records_native_pass1_stage(monkeypatch, tmp_path):
    job = _job()
    jobs = _wire_common(
        monkeypatch,
        tmp_path,
        job,
        pass1_result={"segments": [], "corrections": [], "usage": {}},
    )

    assert cli.main(["native-translate", job.job_id, "--pass", "1", "--json"]) == 0

    saved = jobs.saved[-1]
    assert saved.completed_stages == [
        "audio_preparation",
        "upload",
        "azure_transcription",
        "native_pass1",
    ]


def test_native_translate_pass2_only_records_native_pass2_stage(monkeypatch, tmp_path):
    job = _job()
    jobs = _wire_common(
        monkeypatch,
        tmp_path,
        job,
        pass2_result={"translations": [], "usage": {}},
    )

    assert cli.main(["native-translate", job.job_id, "--pass", "2", "--json"]) == 0

    saved = jobs.saved[-1]
    assert saved.completed_stages == [
        "audio_preparation",
        "upload",
        "azure_transcription",
        "native_pass2",
    ]


def test_native_translate_all_records_both_pass_stages(monkeypatch, tmp_path):
    job = _job()
    jobs = _wire_common(
        monkeypatch,
        tmp_path,
        job,
        translation_result={"pass1_corrections": 0, "pass2_segments": 0, "pass1_path": "p1", "pass2_path": "p2"},
    )

    assert cli.main(["native-translate", job.job_id, "--json"]) == 0

    saved = jobs.saved[-1]
    assert "native_pass1" in saved.completed_stages
    assert "native_pass2" in saved.completed_stages
