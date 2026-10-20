"""Measure and score with each speaker's assigned voice, the one the renderer uses."""
from __future__ import annotations

import json
from types import SimpleNamespace

from mathula_tv import dub_submission, dub_workbench
from mathula_tv.dub_fit import measure_text_fit
from mathula_tv.dub_voices import load_clip_voices, pick_voice_info
from mathula_tv.dub_workbench import Workbench

VOICES = {
    "SPEAKER_00": {"speaker_id": "SPEAKER_00", "selected_voice": "zu-ZA-ThembaNeural", "base_prosody": {"pitch_percent": 0}},
    "SPEAKER_01": {"speaker_id": "SPEAKER_01", "selected_voice": "zu-ZA-ThandoNeural", "base_prosody": {"pitch_percent": 6}},
}
SEGMENTS = [{"segment_id": "s1", "speaker_id": "SPEAKER_00", "start_ms": 0, "end_ms": 5000},
            {"segment_id": "s2", "speaker_id": "SPEAKER_01", "start_ms": 5200, "end_ms": 9000}]


def test_a_named_known_speaker_gets_their_own_assignment():
    info = pick_voice_info(VOICES, "SPEAKER_01", 0, 1000, SEGMENTS)
    assert info["selected_voice"] == "zu-ZA-ThandoNeural" and info["base_prosody"]["pitch_percent"] == 6
    info["selected_voice"] = "changed"
    assert VOICES["SPEAKER_01"]["selected_voice"] == "zu-ZA-ThandoNeural"          # a copy, not the shared dict


def test_an_unknown_or_missing_speaker_is_inferred_from_the_window():
    assert pick_voice_info(VOICES, None, 5300, 8000, SEGMENTS)["selected_voice"] == "zu-ZA-ThandoNeural"
    assert pick_voice_info(VOICES, "SPEAKER_99", 100, 2000, SEGMENTS)["selected_voice"] == "zu-ZA-ThembaNeural"
    assert pick_voice_info(VOICES, None, 20_000, 21_000, SEGMENTS)["selected_voice"] == "zu-ZA-ThembaNeural"   # no overlap: first voice


def test_no_assignments_means_use_the_default_voice():
    assert pick_voice_info({}, "SPEAKER_00", 0, 1000, SEGMENTS) is None


def test_measure_text_fit_hands_the_assignment_to_the_synthesis_step(tmp_path, monkeypatch):
    captured = {}

    def fake_measure(*, tts, group, voice_info, candidate, geometry, output_root, round_number, force, job_root=None):
        captured["voice_info"] = voice_info
        return {"measured_ms": 3000, "direction": None, "fit": True, "required_speed_percent": 0.0,
                "speed_fit_mode": "x", "rank": [0], "path": "p"}

    monkeypatch.setattr("mathula_tv.native_dub._measure_temporal_candidate", fake_measure)
    tts = SimpleNamespace(default_voice="zu-ZA-ThandoNeural")
    out = measure_text_fit(tts, text="Sawubona", start_ms=0, end_ms=3000, output_root=tmp_path, voice_info=VOICES["SPEAKER_00"])
    assert captured["voice_info"]["selected_voice"] == "zu-ZA-ThembaNeural" and out["voice"] == "zu-ZA-ThembaNeural"
    measure_text_fit(tts, text="Sawubona", start_ms=0, end_ms=3000, output_root=tmp_path)
    assert captured["voice_info"] == {"selected_voice": "zu-ZA-ThandoNeural", "base_prosody": {}}   # unchanged fallback


def _store(tmp_path, with_job=True):
    root = tmp_path / "work" / "experience"
    (root / "corpus" / "clip").mkdir(parents=True)
    if with_job:
        direct = tmp_path / "work" / "jobs" / "job1" / "direct_dub"
        direct.mkdir(parents=True)
        (direct / "voice_resolution.json").write_text(json.dumps({"speakers": [dict(v, status="resolved") for v in VOICES.values()]}), encoding="utf-8")
    return SimpleNamespace(root=root, clip_dir=lambda clip: root / "corpus" / clip,
                           load_corpus=lambda: {"clips": {"clip": {"job_id": "job1"}}})


def test_clip_voices_are_loaded_from_the_job_and_frozen_into_the_corpus(tmp_path):
    store = _store(tmp_path)
    voices = load_clip_voices(store, "clip")
    assert voices["SPEAKER_00"]["selected_voice"] == "zu-ZA-ThembaNeural" and set(voices) == {"SPEAKER_00", "SPEAKER_01"}
    assert (tmp_path / "work" / "experience" / "corpus" / "clip" / "voice_resolution.json").is_file()
    (tmp_path / "work" / "jobs" / "job1" / "direct_dub" / "voice_resolution.json").unlink()      # job cleaned up later
    assert load_clip_voices(store, "clip")["SPEAKER_01"]["selected_voice"] == "zu-ZA-ThandoNeural"


def test_a_clip_with_no_voice_file_yields_nothing_instead_of_failing(tmp_path):
    assert load_clip_voices(_store(tmp_path, with_job=False), "clip") == {}
    assert load_clip_voices(SimpleNamespace(), "clip") == {}


def _workbench(tmp_path):
    wb = Workbench(SimpleNamespace(load_labels=lambda c: []), lambda: object(), measure_root=tmp_path, voices=VOICES)
    wb.segments = SEGMENTS
    wb.episode = {"dir": str(tmp_path), "tts_budget": 50, "tts_used": 0, "speed_policy": "natural"}
    return wb


def test_agent_measurements_use_the_speakers_voice_single_and_batch(tmp_path, monkeypatch):
    seen = []

    def fake_fit(tts, **kwargs):
        seen.append(kwargs["voice_info"]["selected_voice"])
        return {"measured_ms": 1, "window_ms": 1, "direction": None, "fits_naturally": True, "required_speed_percent": 0.0,
                "audio_path": "x"}

    monkeypatch.setattr(dub_workbench, "measure_text_fit", fake_fit)
    wb = _workbench(tmp_path)
    wb.measure_tts("a", 0, 4000, speaker_id="SPEAKER_00")
    wb.measure_tts("b", 5300, 8000, speaker_id="SPEAKER_01")
    wb.measure_tts_batch([{"text": "c", "start_ms": 0, "end_ms": 4000, "speaker_id": "SPEAKER_01"},
                          {"text": "d", "start_ms": 5300, "end_ms": 8000}])                   # speaker omitted: inferred
    assert seen == ["zu-ZA-ThembaNeural", "zu-ZA-ThandoNeural", "zu-ZA-ThandoNeural", "zu-ZA-ThandoNeural"]


def test_scoring_measures_each_unit_with_its_speakers_voice(tmp_path, monkeypatch):
    native = tmp_path / "native"
    native.mkdir()
    (native / "pass1_restored_transcript.json").write_text(json.dumps({"segments": SEGMENTS}), encoding="utf-8")
    seen = {}

    def fake_fit(tts, *, text, speaker_id, voice_info, **kwargs):
        seen[speaker_id] = voice_info["selected_voice"]
        return {"measured_ms": 1000, "direction": None, "required_speed_percent": 0.0, "fits_naturally": True,
                "free_gap_after_ms": 0, "tail_pad_ms": 0}

    monkeypatch.setattr(dub_submission, "measure_text_fit", fake_fit)
    monkeypatch.setattr(dub_submission, "score_items", lambda items, **kw: {"headline": {}, "groups": [], "turns": []})
    monkeypatch.setattr(dub_submission, "_load_context", lambda native_dir: ({}, {}))
    items = [{"item_id": "u1", "start_ms": 0, "end_ms": 5000, "speaker_id": "SPEAKER_00", "zulu_text": "a", "source_segment_ids": ["s1"]},
             {"item_id": "u2", "start_ms": 5200, "end_ms": 9000, "speaker_id": "SPEAKER_01", "zulu_text": "b", "source_segment_ids": ["s2"]}]
    dub_submission.score_submission_dir(native, {"items": items}, tts=None, measure_root=tmp_path, voices=VOICES)
    assert seen == {"SPEAKER_00": "zu-ZA-ThembaNeural", "SPEAKER_01": "zu-ZA-ThandoNeural"}
