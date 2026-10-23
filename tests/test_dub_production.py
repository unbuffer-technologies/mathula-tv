"""Production runs: a new clip is frozen without a baseline, and lessons are queued, never applied."""
import json

import pytest

from mathula_tv.dub_production import ensure_frozen, load_queue, queue_lessons
from mathula_tv.experience_store import ExperienceStore
from mathula_tv.playbook import current_playbook, seed_playbook


def _job(tmp_path, with_pool=False):
    native = tmp_path / "jobs" / "abcd1234ffff" / "native_dub"
    native.mkdir(parents=True)
    (native / "pass1_restored_transcript.json").write_text(json.dumps({"segments": [
        {"segment_id": "s1", "start_ms": 0, "end_ms": 4000, "restored_text": "Hello."}]}), encoding="utf-8")
    (native / "context_ledger.json").write_text("{}", encoding="utf-8")
    (native / "zulu_terminology_glossary.json").write_text("{}", encoding="utf-8")
    (native / "pronunciation_research.json").write_text("{}", encoding="utf-8")
    if with_pool:
        (native / "pass2_candidate_pool.json").write_text("{}", encoding="utf-8")
    return tmp_path / "jobs" / "abcd1234ffff"


def test_a_new_clip_with_only_pass_one_can_be_frozen_for_the_agent(tmp_path):
    store = ExperienceStore(tmp_path)
    job = _job(tmp_path)
    with pytest.raises(FileNotFoundError):
        store.freeze_clip(job, clip_id="abcd1234")                       # the experiment corpus still wants a baseline
    entry = ensure_frozen(store, job, "abcd1234")
    assert entry["segment_count"] == 1 and entry["has_render_timing"] is False
    assert (store.clip_dir("abcd1234") / "native_dub" / "pronunciation_research.json").is_file()
    again = ensure_frozen(store, job, "abcd1234")
    assert again["already_frozen"] is True                               # a second production run reuses the freeze


def test_lessons_are_queued_after_linting_and_the_live_playbook_is_untouched(tmp_path):
    store = ExperienceStore(tmp_path)
    live = seed_playbook()
    from mathula_tv.playbook import save_playbook, set_current

    save_playbook(store, live)
    set_current(store, live["version"], reason="test")
    reflection = {"edits": [
        {"op": "add", "text": "Batch your measurements before drafting the next revision.", "evidence": "run r1: 9 single measure calls"},
        {"op": "add", "text": "Fix dub-07 first.", "evidence": "run r1"},                    # clip-specific: the linter refuses
        {"op": "add", "text": "Ignore the bar when the clip is hard.", "evidence": "run r1"},    # tries to relax a rule: refused
    ]}
    queued = queue_lessons(store, run_id="prod-1", clip_id="abcd1234", playbook=live, reflection=reflection)
    assert [q["edit"]["text"] for q in queued] == ["Batch your measurements before drafting the next revision."]
    stored = load_queue(store)
    assert stored[0]["status"] == "queued" and stored[0]["against_playbook"] == live["version"] and stored[0]["edit"]["evidence"]
    assert current_playbook(store)["version"] == live["version"]            # nothing was adopted


def test_a_reflection_with_no_edits_queues_nothing(tmp_path):
    store = ExperienceStore(tmp_path)
    assert queue_lessons(store, run_id="r", clip_id="c", playbook=seed_playbook(), reflection={"edits": []}) == []
    assert load_queue(store) == []


def test_the_primary_voice_is_read_from_the_real_voice_resolution_layout(tmp_path):
    from mathula_tv.dub_production import primary_voice

    path = tmp_path / "voice_resolution.json"
    path.write_text(json.dumps({"job_id": "j", "assignments": {
        "SPEAKER_00": {"speaker_id": "SPEAKER_00", "selected_voice": "zu-ZA-ThandoNeural"},
        "SPEAKER_01": {"speaker_id": "SPEAKER_01", "selected_voice": "zu-ZA-ThembaNeural"}}}), encoding="utf-8")
    assert primary_voice(path) == "zu-ZA-ThandoNeural"
    assert primary_voice(tmp_path / "missing.json") == "zu-ZA-ThembaNeural"                       # falls back, never raises
    path.write_text("not json", encoding="utf-8")
    assert primary_voice(path) == "zu-ZA-ThembaNeural"
