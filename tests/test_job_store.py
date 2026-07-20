import json
import pytest
from mathula_tv.atomic_io import atomic_write_json, read_json
from mathula_tv.job_store import JobStore


def make(store): return store.create(source_filename="x.mp4", local_source_path="/x.mp4", source_checksum="a"*64, source_duration=5)


def test_atomic_json_and_transitions(tmp_path):
    path = tmp_path / "a/b.json"; atomic_write_json(path, {"x": "✓"}); assert read_json(path) == {"x": "✓"}
    store = JobStore(tmp_path); job = make(store); job.transition("audio_prepared"); store.save(job); assert store.load(job.job_id).state == "audio_prepared"
    with pytest.raises(ValueError): job.transition("review_ready")


def test_completed_stage_is_idempotent(tmp_path):
    store = JobStore(tmp_path); job = make(store)
    assert store.complete_stage(job, "audio", "audio_prepared") is True
    assert store.complete_stage(job, "audio", "audio_prepared") is False
    assert store.load(job.job_id).completed_stages == ["audio"]

