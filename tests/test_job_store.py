import json
import pytest
from mathula_tv.atomic_io import atomic_write_json, read_json
from mathula_tv.gcs_store import object_name
from mathula_tv.job_store import JobStore
from mathula_tv.models import validate_job_id


def make(store): return store.create(source_filename="x.mp4", local_source_path="/x.mp4", source_checksum="a"*64, source_duration=5)


def test_atomic_json_and_transitions(tmp_path):
    path = tmp_path / "a/b.json"; atomic_write_json(path, {"x": "✓"}); assert read_json(path) == {"x": "✓"}
    store = JobStore(tmp_path); job = make(store); job.transition("audio_prepared"); store.save(job); assert store.load(job.job_id).state == "audio_prepared"
    with pytest.raises(ValueError): job.transition("review_ready")


@pytest.mark.parametrize("bad", ["../etc", "a/b", "..", "job/../..", "", "job id", "job\x00"])
def test_traversal_job_ids_are_rejected(tmp_path, bad):
    store = JobStore(tmp_path)
    with pytest.raises(ValueError):
        validate_job_id(bad)
    with pytest.raises(ValueError):
        store.job_dir(bad)
    with pytest.raises(ValueError):
        object_name("mathula-tv", bad, "status.json")


def test_valid_job_ids_are_accepted(tmp_path):
    store = JobStore(tmp_path)
    for good in ("19ba6d69f1b84132ba4f20599101834a", "colab-abc123", "job_1"):
        assert validate_job_id(good) == good
        assert store.job_dir(good) == store.root / good
        assert object_name("mathula-tv", good, "status.json") == f"mathula-tv/jobs/{good}/status.json"


def test_completed_stage_is_idempotent(tmp_path):
    store = JobStore(tmp_path); job = make(store)
    assert store.complete_stage(job, "audio", "audio_prepared") is True
    assert store.complete_stage(job, "audio", "audio_prepared") is False
    assert store.load(job.job_id).completed_stages == ["audio"]

