import pytest

from mathula_tv.atomic_io import atomic_write_json
from mathula_tv.job_store import JobStore


def legacy_job(store):
    job = store.create(
        source_filename="clip.mp4",
        local_source_path="/clip.mp4",
        source_checksum="a" * 64,
        source_duration=10,
    )
    job.state = "synthesis_queued"
    store.save(job)
    root = store.job_dir(job.job_id)
    atomic_write_json(root / "translation/transcript_zu.json", {"turns": [{"translated_text": "Sawubona"}]})
    atomic_write_json(root / "translation/youtube_zu.json", {"title": "Isihloko"})
    return job


def test_legacy_migration_preserves_translation_bytes(tmp_path):
    store = JobStore(tmp_path)
    job = legacy_job(store)
    translation = store.job_dir(job.job_id) / "translation/transcript_zu.json"
    before = translation.read_bytes()
    dry_run = store.migrate_synthesis_queued(job, dry_run=True)
    assert dry_run["changed"] and store.load(job.job_id).state == "synthesis_queued"
    result = store.migrate_synthesis_queued(store.load(job.job_id))
    migrated = store.load(job.job_id)
    assert result["translations_preserved"] and migrated.state == "azure_tts_queued"
    assert translation.read_bytes() == before
    assert migrated.compatibility_gate["state"] == "pending"


def test_invalid_legacy_migration_fails_without_state_change(tmp_path):
    store = JobStore(tmp_path)
    job = store.create(
        source_filename="clip.mp4",
        local_source_path="/clip.mp4",
        source_checksum="a" * 64,
        source_duration=10,
    )
    with pytest.raises(ValueError, match="Only legacy synthesis_queued"):
        store.migrate_synthesis_queued(job)
    assert store.load(job.job_id).state == "created"
