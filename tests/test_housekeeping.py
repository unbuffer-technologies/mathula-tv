import pytest

from mathula_tv import housekeeping
from mathula_tv.housekeeping import clean_measure, ensure_free_space


def make_run(root, run, clip="clipa", repeat="r1", audio_mb=1):
    run_dir = root / "runs" / run / clip / repeat
    (run_dir / "measure" / "m").mkdir(parents=True)
    (run_dir / "measure" / "m" / "a.wav").write_bytes(b"\0" * audio_mb * 1024 * 1024)
    (run_dir / "render" / "native_dub").mkdir(parents=True)
    (run_dir / "render" / "native_dub" / "video.mp4").write_bytes(b"video")
    (run_dir / "submission.json").write_text("{}")
    (run_dir / "scorecard.json").write_text("{}")
    return run_dir


def test_a_runs_scratch_audio_is_removed_and_every_record_and_the_render_stay(tmp_path):
    experience = tmp_path / "experience"
    run = make_run(experience, "prod-1")
    freed = clean_measure(experience, run)
    assert freed >= 1024 * 1024 and not (run / "measure").exists()
    assert (run / "submission.json").is_file() and (run / "scorecard.json").is_file() and (run / "render" / "native_dub" / "video.mp4").is_file()


def test_only_a_folder_named_measure_at_the_right_depth_inside_runs_is_ever_deleted(tmp_path):
    experience = tmp_path / "experience"
    run = make_run(experience, "prod-1")
    decoy = experience / "runs" / "prod-1" / "clipa" / "r1" / "render" / "measure"       # a 'measure' folder deeper inside a render
    decoy.mkdir(parents=True)
    (decoy / "keep.txt").write_text("keep")
    outside = tmp_path / "elsewhere" / "measure"
    outside.mkdir(parents=True)
    (outside / "keep.txt").write_text("keep")
    clean_measure(experience)
    assert not (run / "measure").exists()
    assert (decoy / "keep.txt").is_file()                                      # wrong depth: untouched
    clean_measure(experience, outside.parent)                                  # a run_dir outside the runs folder: refused
    assert (outside / "keep.txt").is_file()


def test_cleaning_every_run_spares_the_run_in_progress(tmp_path):
    experience = tmp_path / "experience"
    old, current = make_run(experience, "prod-1"), make_run(experience, "prod-2")
    clean_measure(experience, except_run_dir=current)
    assert not (old / "measure").exists() and (current / "measure").is_dir()


def test_a_low_disk_is_freed_from_other_runs_first_and_a_still_full_disk_stops_with_a_clear_message(tmp_path, monkeypatch):
    experience = tmp_path / "experience"
    old, current = make_run(experience, "prod-1"), make_run(experience, "prod-2")
    free = {"gb": 1.0}
    monkeypatch.setattr(housekeeping, "free_gb", lambda path: free["gb"])
    messages = []
    with pytest.raises(RuntimeError) as error:
        ensure_free_space(experience, need_gb=4, keep_run_dir=current, progress=messages.append)
    assert "only 1.0 GB free" in str(error.value) and not (old / "measure").exists() and (current / "measure").is_dir()
    free["gb"] = 9.0
    assert ensure_free_space(experience, need_gb=4, keep_run_dir=current) == 9.0           # enough room: nothing is touched
    assert (current / "measure").is_dir()


def test_the_agent_removes_its_own_scratch_audio_when_it_ends_even_if_it_crashes(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from mathula_tv import dub_agent

    experience = tmp_path / "experience"
    run = make_run(experience, "prod-1")
    other = make_run(experience, "prod-2")
    workbench = SimpleNamespace(store=SimpleNamespace(root=experience), episode={"dir": str(run)})

    def crash(client, wb, **kwargs):
        raise RuntimeError("model call failed")

    monkeypatch.setattr(dub_agent, "_run_agent", crash)
    with pytest.raises(RuntimeError):
        dub_agent.run_agent(object(), workbench, clip_id="clipa", run_id="prod-1")
    assert not (run / "measure").exists() and (other / "measure").is_dir()                  # only its own
    monkeypatch.setenv("MATHULA_DUB_KEEP_MEASURE", "1")
    kept = make_run(experience, "prod-3")
    wb = SimpleNamespace(store=SimpleNamespace(root=experience), episode={"dir": str(kept)})
    monkeypatch.setattr(dub_agent, "_run_agent", lambda client, w, **k: {"ok": True})
    assert dub_agent.run_agent(object(), wb, clip_id="clipa", run_id="prod-3") == {"ok": True} and (kept / "measure").is_dir()
