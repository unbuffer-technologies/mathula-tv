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


def _wire(monkeypatch, tmp_path, job, captured_kwargs, *, translation_result=None, pass2_result=None):
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

    def fake_run_native_translation(**kwargs):
        captured_kwargs.update(kwargs)
        return translation_result or {
            "pass1_corrections": 0,
            "pass2_segments": 0,
            "pass1_path": "p1",
            "pass2_path": "p2",
            "referent_audit_summary": None,
        }

    def fake_run_native_pass2(**kwargs):
        captured_kwargs.update(kwargs)
        return pass2_result or {"translations": [], "usage": {}}

    monkeypatch.setattr(native_dub, "run_native_translation", fake_run_native_translation)
    monkeypatch.setattr(native_dub, "run_native_pass2", fake_run_native_pass2)
    return jobs


def test_native_translate_default_enables_referent_audit_with_defaults(monkeypatch, tmp_path):
    job = _job()
    captured: dict = {}
    _wire(monkeypatch, tmp_path, job, captured)

    assert cli.main(["native-translate", job.job_id, "--json"]) == 0

    assert captured["audit_referents"] is True
    assert captured["referent_audit_batch_size"] == 12
    assert captured["max_referent_repair_rounds"] == 1


def test_native_translate_no_audit_referents_disables_it(monkeypatch, tmp_path):
    job = _job()
    captured: dict = {}
    _wire(monkeypatch, tmp_path, job, captured)

    assert cli.main(["native-translate", job.job_id, "--no-audit-referents", "--json"]) == 0

    assert captured["audit_referents"] is False


def test_native_translate_pass2_only_forwards_referent_audit_flags(monkeypatch, tmp_path):
    job = _job()
    captured: dict = {}
    _wire(monkeypatch, tmp_path, job, captured)

    assert cli.main(
        [
            "native-translate",
            job.job_id,
            "--pass",
            "2",
            "--referent-audit-batch-size",
            "6",
            "--max-referent-repair-rounds",
            "2",
            "--json",
        ]
    ) == 0

    assert captured["audit_referents"] is True
    assert captured["referent_audit_batch_size"] == 6
    assert captured["max_referent_repair_rounds"] == 2


def test_native_translate_rejects_out_of_range_referent_audit_batch_size(monkeypatch, tmp_path, capsys):
    job = _job()
    captured: dict = {}
    _wire(monkeypatch, tmp_path, job, captured)

    exit_code = cli.main(["native-translate", job.job_id, "--referent-audit-batch-size", "0", "--json"])

    assert exit_code != 0
    assert "--referent-audit-batch-size" in capsys.readouterr().err
    assert "audit_referents" not in captured
