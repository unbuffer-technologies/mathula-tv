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


def _timing_plan(output_video: str, overlaps: list[dict]) -> dict:
    return {
        "schema_version": "native-dub-timing-plan-v1",
        "output_video": output_video,
        "phrase_group_count": 2,
        "timing_repairs": [],
        "video_retimed": False,
        "tts_rate_percent": 0,
        "preferred_raw_speed_percent": 6,
        "max_natural_speed_percent": 12,
        "mouth_close_target_tolerance_ms": 40,
        "timing_policy": "target_overflow_then_adaptive_speed_with_protected_source_gap",
        "hook_overlay_enabled": True,
        "hook_edit_enabled": True,
        "native_master_video": output_video,
        "dialogue": {"overlap_count": len(overlaps), "overlaps": overlaps},
    }


def _wire_common(monkeypatch, tmp_path, job, timing_plan):
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
    monkeypatch.setattr(native_dub, "render_native_dub", lambda **kwargs: timing_plan)
    return jobs


def test_native_dub_marks_review_ready_when_no_overlaps_need_review(monkeypatch, tmp_path):
    job = _job()
    timing_plan = _timing_plan(str(tmp_path / "native_dubbed_with_hook.mp4"), overlaps=[])
    jobs = _wire_common(monkeypatch, tmp_path, job, timing_plan)

    assert cli.main(["native-dub", job.job_id, "--json"]) == 0

    assert jobs.saved, "job.completed_stages/state must be persisted"
    saved = jobs.saved[-1]
    assert "native_dub" in saved.completed_stages
    assert saved.state == "review_ready"
    assert saved.review_readiness == "review_ready"


def test_native_dub_marks_needs_timing_review_when_overlap_flagged(monkeypatch, tmp_path):
    job = _job()
    overlaps = [
        {
            "left_unit_id": "native_phrase_0001",
            "right_unit_id": "native_phrase_0002",
            "duration_ms": 6650,
            "human_review_required": True,
        }
    ]
    timing_plan = _timing_plan(str(tmp_path / "native_dubbed_with_hook.mp4"), overlaps=overlaps)
    jobs = _wire_common(monkeypatch, tmp_path, job, timing_plan)

    assert cli.main(["native-dub", job.job_id, "--json"]) == 0

    saved = jobs.saved[-1]
    assert "native_dub" in saved.completed_stages
    assert saved.state == "review_ready"
    assert saved.review_readiness == "needs_timing_review"


def test_native_dub_marks_needs_timing_review_when_timing_quality_flagged(monkeypatch, tmp_path):
    # No dialogue overlaps at all -- the ONLY signal is the new timing-quality gate,
    # confirming it independently drives review_readiness rather than only ever
    # riding along with the pre-existing overlap check.
    job = _job()
    timing_plan = _timing_plan(str(tmp_path / "native_dubbed_with_hook.mp4"), overlaps=[])
    timing_plan["timing_quality_review_required"] = True
    timing_plan["timing_quality_flags"] = [
        {"group_id": "native_phrase_0039__isl12", "natural_overrun_ms": 35650, "raw_speed_percent": 990.3},
    ]
    jobs = _wire_common(monkeypatch, tmp_path, job, timing_plan)

    assert cli.main(["native-dub", job.job_id, "--json"]) == 0

    saved = jobs.saved[-1]
    assert saved.state == "review_ready"
    assert saved.review_readiness == "needs_timing_review"


def test_native_dub_does_not_duplicate_completed_stage_on_rerun(monkeypatch, tmp_path):
    job = _job()
    job.completed_stages.append("native_dub")
    timing_plan = _timing_plan(str(tmp_path / "native_dubbed_with_hook.mp4"), overlaps=[])
    jobs = _wire_common(monkeypatch, tmp_path, job, timing_plan)

    assert cli.main(["native-dub", job.job_id, "--json"]) == 0

    saved = jobs.saved[-1]
    assert saved.completed_stages.count("native_dub") == 1
