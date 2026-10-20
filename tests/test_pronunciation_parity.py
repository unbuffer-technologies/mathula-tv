"""Measurement hears the job's pronunciation dictionary, and a measurement/render gap is detected automatically."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from mathula_tv import dub_pronunciation, dub_workbench, native_dub
from mathula_tv.dub_fit import measure_text_fit
from mathula_tv.dub_pronunciation import register_clip_pronunciation
from mathula_tv.dub_smells import find_smells
from mathula_tv.dub_workbench import Workbench


def _store(tmp_path, with_research=True):
    root = tmp_path / "work" / "experience"
    (root / "corpus" / "clip").mkdir(parents=True)
    if with_research:
        native = tmp_path / "work" / "jobs" / "job1" / "native_dub"
        native.mkdir(parents=True)
        (native / "pronunciation_research.json").write_text(json.dumps({"accepted_corrections": [], "self_supervised_corrections": []}),
                                                           encoding="utf-8")
    return SimpleNamespace(root=root, clip_dir=lambda c: root / "corpus" / c, load_corpus=lambda: {"clips": {"clip": {"job_id": "job1"}}})


def test_the_clips_dictionary_is_registered_under_its_corpus_directory_and_frozen(tmp_path, monkeypatch):
    dub_pronunciation._REGISTERED.clear()
    store = _store(tmp_path)
    key = register_clip_pronunciation(store, "clip")
    assert key == tmp_path / "work" / "experience" / "corpus" / "clip"
    assert str(key) in native_dub._NATIVE_DUB_PRONUNCIATION_DICTIONARY_BY_JOB
    assert (key / "pronunciation_research.json").is_file()
    calls = []
    monkeypatch.setattr(native_dub, "_apply_native_pronunciation_research_to_job_cache", lambda k, r: calls.append(k))
    assert register_clip_pronunciation(store, "clip") == key and calls == []          # registered once per process


def test_a_clip_without_research_is_measured_with_the_default_dictionary(tmp_path):
    dub_pronunciation._REGISTERED.clear()
    assert register_clip_pronunciation(_store(tmp_path, with_research=False), "clip") is None
    assert register_clip_pronunciation(SimpleNamespace(), "clip") is None


def test_measure_text_fit_passes_the_dictionary_key_to_synthesis(tmp_path, monkeypatch):
    seen = {}

    def fake_measure(*, tts, group, voice_info, candidate, geometry, output_root, round_number, force, job_root=None):
        seen["job_root"] = job_root
        return {"measured_ms": 3000, "direction": None, "fit": True, "required_speed_percent": 0.0, "speed_fit_mode": "x",
                "rank": [0], "path": "p"}

    monkeypatch.setattr(native_dub, "_measure_temporal_candidate", fake_measure)
    tts = SimpleNamespace(default_voice="zu-ZA-ThandoNeural")
    measure_text_fit(tts, text="Sawubona", start_ms=0, end_ms=3000, output_root=tmp_path, pronunciation_root=Path("/k"))
    assert seen["job_root"] == Path("/k")
    measure_text_fit(tts, text="Sawubona", start_ms=0, end_ms=3000, output_root=tmp_path)
    assert seen["job_root"] is None


def _candidate_call(tmp_path, monkeypatch, job_root):
    captured = {}

    def fake_synthesize(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(duration_ms=2000)

    monkeypatch.setattr(native_dub, "_synthesize_group", fake_synthesize)
    geometry = {"preferred_raw_ms": 2000, "maximum_usable_raw_ms": 2200, "source_window_ms": 2000,
                "mouth_close_early_tolerance_ms": 40, "mouth_close_late_tolerance_ms": 40}
    group = {"group_id": "measure", "segment_ids": ["m"], "speaker_id": "S", "source_text": "x"}
    native_dub._measure_temporal_candidate(
        tts=None, group=group, voice_info={"selected_voice": "v", "base_prosody": {}},
        candidate={"candidate_id": "c", "spoken_text": "Sibiya"}, geometry=geometry, output_root=tmp_path,
        round_number=0, force=False, job_root=job_root)
    return captured


def test_the_dictionary_reaches_synthesis_and_changes_the_cache_key_only_when_present(tmp_path, monkeypatch):
    without = _candidate_call(tmp_path, monkeypatch, None)
    with_a = _candidate_call(tmp_path, monkeypatch, Path("/clipA"))
    with_b = _candidate_call(tmp_path, monkeypatch, Path("/clipB"))
    assert without["job_root"] is None and with_a["job_root"] == Path("/clipA")
    assert len({without["output_path"], with_a["output_path"], with_b["output_path"]}) == 3      # no stale audio is reused


def test_the_workbench_registers_the_dictionary_on_open_and_measures_with_it(tmp_path, monkeypatch):
    kwargs_seen = {}

    def fake_fit(tts, **kwargs):
        kwargs_seen.update(kwargs)
        return {"measured_ms": 1, "window_ms": 1, "direction": None, "fits_naturally": True, "required_speed_percent": 0.0,
                "audio_path": "x"}

    monkeypatch.setattr(dub_workbench, "measure_text_fit", fake_fit)
    wb = Workbench(SimpleNamespace(load_labels=lambda c: []), lambda: object(), measure_root=tmp_path)
    wb.pronunciation_root = Path("/clip")
    wb.episode = {"dir": str(tmp_path), "tts_budget": 5, "tts_used": 0, "speed_policy": "natural"}
    wb.measure_tts("a", 0, 1000)
    assert kwargs_seen["pronunciation_root"] == Path("/clip")


# ---- the automatic parity check ------------------------------------------------------------------------------------

def _run(tmp_path, scored, rendered):
    d = tmp_path / "r1"
    (d / "render" / "native_dub").mkdir(parents=True)
    (d / "scorecard.json").write_text(json.dumps({"groups": [{"group_id": g, "measured_ms": m} for g, m in scored.items()]}), encoding="utf-8")
    (d / "render" / "native_dub" / "timing_plan.json").write_text(
        json.dumps({"groups": [{"group_id": g, "natural_tts_duration_ms": n} for g, n in rendered.items()]}), encoding="utf-8")
    (d / "trace.jsonl").write_text("", encoding="utf-8")
    return {"clip": "c", "repeat": 1, "dir": d, "stop_reason": "agent_finished", "processed_tokens": 1, "tts_used": 1}


def test_a_unit_that_renders_longer_than_it_measured_is_reported_as_a_measurement_problem(tmp_path):
    run = _run(tmp_path, {"u1": 13835, "u4": 23787}, {"u1": 13800, "u4": 24657})        # u4: +870 ms, +3.7% (the real case)
    found = {f["smell"]: f for f in find_smells([run])}
    gap = found["measurement_render_gap"]
    assert gap["severity"] == "high" and gap["owner"] == "harness" and "u4" in gap["evidence"] and "+870 ms" in gap["evidence"]


def test_small_differences_and_unrendered_runs_are_not_reported(tmp_path):
    close = _run(tmp_path, {"u1": 10000, "u2": 5000}, {"u1": 10150, "u2": 5010})        # 1.5% / 0.2%: within tolerance
    assert "measurement_render_gap" not in {f["smell"] for f in find_smells([close])}
    unrendered = {"clip": "c", "repeat": 2, "dir": tmp_path / "nowhere", "stop_reason": "agent_finished", "processed_tokens": 1, "tts_used": 1}
    assert "measurement_render_gap" not in {f["smell"] for f in find_smells([unrendered])}
