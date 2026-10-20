"""The submission -> committed-translation adapter, checked against the renderer's own reader."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from mathula_tv.dub_render import (
    RenderAdapterError, _speed, assign_sentences, build_committed_translation, prepare_render_root,
)
from mathula_tv.native_dub import _temporal_mask_groups_for_render


def _sentence(sid, parent, speaker, start, end, text="english"):
    return {"segment_id": sid, "parent_segment_id": parent, "speaker_id": speaker, "start_ms": start, "end_ms": end,
            "restored_text": text}


# seg-1 has two sentences, seg-2 one, seg-3 one (another speaker)
SENTENCES = [
    _sentence("seg-1__s00", "seg-1", "A", 0, 3000, "One."),
    _sentence("seg-1__s01", "seg-1", "A", 3100, 6000, "Two."),
    _sentence("seg-2__s00", "seg-2", "A", 6200, 9000, "Three."),
    _sentence("seg-3__s00", "seg-3", "B", 9500, 13000, "Four."),
]


def _unit(uid, start, end, speaker, text, *segments):
    return {"item_id": uid, "start_ms": start, "end_ms": end, "speaker_id": speaker, "zulu_text": text,
            "source_segment_ids": list(segments)}


UNITS = [_unit("u1", 0, 6000, "A", "Kunye nokubili.", "seg-1"), _unit("u2", 6200, 9000, "A", "Kuthathu.", "seg-2"),
         _unit("u3", 9500, 13000, "B", "Kune.", "seg-3")]


def test_every_sentence_goes_to_the_unit_that_lists_its_parent_segment():
    owned = assign_sentences(UNITS, SENTENCES)
    assert owned == {"u1": ["seg-1__s00", "seg-1__s01"], "u2": ["seg-2__s00"], "u3": ["seg-3__s00"]}


def test_a_segment_the_agent_split_goes_by_window_overlap():
    split = [_unit("a", 0, 3050, "A", "x", "seg-1"), _unit("b", 3100, 6000, "A", "y", "seg-1"),
             UNITS[1], UNITS[2]]
    owned = assign_sentences(split, SENTENCES)
    assert owned["a"] == ["seg-1__s00"] and owned["b"] == ["seg-1__s01"]


def test_a_unit_that_owns_no_sentence_is_a_clear_error():
    merged = [_unit("u1", 0, 9000, "A", "x", "seg-1", "seg-2"), _unit("ghost", 9100, 9400, "A", "y", "seg-2"), UNITS[2]]
    with pytest.raises(RenderAdapterError) as error:
        assign_sentences(merged, SENTENCES)
    assert "ghost" in str(error.value)


def test_the_committed_translation_is_accepted_by_the_renderers_own_reader_and_keeps_windows_and_text():
    artifact = build_committed_translation({"items": UNITS, "approach_notes": "n"}, SENTENCES)
    groups, translations = _temporal_mask_groups_for_render(artifact, SENTENCES)       # the real reader: raises if wrong
    assert [g["group_id"] for g in groups] == ["u1", "u2", "u3"]
    assert [(g["start_ms"], g["source_end_ms"], g["source_span_ms"]) for g in groups] == [(0, 6000, 6000), (6200, 9000, 2800), (9500, 13000, 3500)]
    assert groups[0]["parts"][0]["text"] == "Kunye nokubili." and groups[0]["segment_ids"] == ["seg-1__s00", "seg-1__s01"]
    assert translations["seg-1__s00"] == translations["seg-1__s01"] == "Kunye nokubili."
    assert artifact["temporal_mask_mode"] and artifact["tts_rate_percent"] == 0 and artifact["agent_submission_committed"]


def test_turns_are_assigned_so_the_renderer_can_group_speakers():
    artifact = build_committed_translation({"items": UNITS}, SENTENCES)
    turns = {g["group_id"]: g["turn_id"] for g in artifact["temporal_mask_groups"]}
    assert turns["u1"] == turns["u2"] == "u1" and turns["u3"] == "u3"                  # same speaker, small gap -> one turn


def test_units_that_overlap_or_interleave_are_refused():
    crossed = [_unit("u1", 0, 9000, "A", "x", "seg-1", "seg-2"), _unit("u2", 3000, 6100, "A", "y", "seg-1"), UNITS[2]]
    with pytest.raises(RenderAdapterError):
        assign_sentences(crossed, SENTENCES)


def test_speed_up_is_read_from_the_groups_speed_fit():
    assert _speed({"speed_fit": {"speed_percent": 7.5}}) == 7.5 and _speed({"speed_percent": 3.0}) == 3.0 and _speed({}) == 0.0


# ---- preparing the render directory -----------------------------------------------------------------------------------

def _store(tmp_path):
    root = tmp_path / "experience"
    run_dir = root / "runs" / "r" / "clip" / "r1"
    frozen = root / "corpus" / "clip" / "native_dub"
    frozen.mkdir(parents=True)
    run_dir.mkdir(parents=True)
    segments = [{"segment_id": f"seg-{i}", "speaker_id": "A" if i < 3 else "B", "start_ms": s, "end_ms": e,
                 "restored_text": f"e{i}"} for i, (s, e) in enumerate([(0, 6000), (6200, 9000), (9500, 13000)], 1)]
    (frozen / "pass1_restored_transcript.json").write_text(json.dumps({"segments": segments}), encoding="utf-8")
    (frozen / "context_ledger.json").write_text("{}", encoding="utf-8")
    (frozen / "zulu_terminology_glossary.json").write_text("{}", encoding="utf-8")
    (run_dir / "submission.json").write_text(json.dumps({"items": UNITS, "approach_notes": ""}), encoding="utf-8")
    original = tmp_path / "work" / "jobs" / "job123"
    (original / "native_dub").mkdir(parents=True)
    (original / "native_dub" / "pass1_sentences.json").write_text(json.dumps({"records": SENTENCES}), encoding="utf-8")
    (original / "native_dub" / "speaker_seriousness_modes.json").write_text("{}", encoding="utf-8")
    (original / "direct_dub").mkdir()
    (original / "direct_dub" / "voice_resolution.json").write_text(json.dumps({"A": "zu-ZA-ThembaNeural", "B": "zu-ZA-ThandoNeural"}), encoding="utf-8")
    store = SimpleNamespace(
        load_corpus=lambda: {"clips": {"clip": {"job_id": "job123"}}},
        clip_dir=lambda clip: root / "corpus" / clip,
        repeat_dir=lambda run, clip, rep: root / "runs" / run / clip / f"r{rep}")
    return store, tmp_path / "work"


def test_prepare_builds_a_render_root_without_touching_the_original_job(tmp_path):
    store, work = _store(tmp_path)
    before = sorted(p.name for p in (work / "jobs" / "job123" / "native_dub").iterdir())
    root, info, voice = prepare_render_root(store, work, run_id="r", clip_id="clip", repeat=1)
    native = root / "native_dub"
    for name in ("pass1_restored_transcript.json", "context_ledger.json", "zulu_terminology_glossary.json",
                 "pass1_sentences.json", "pass2_natural_translation.json", "speaker_seriousness_modes.json"):
        assert (native / name).is_file(), name
    assert voice.is_file() and voice.parent.name == "direct_dub" and info == {"job_id": "job123", "groups": 3, "sentences": 4}
    assert sorted(p.name for p in (work / "jobs" / "job123" / "native_dub").iterdir()) == before


def test_prepare_refuses_a_run_without_an_accepted_submission(tmp_path):
    store, work = _store(tmp_path)
    (store.repeat_dir("r", "clip", 1) / "submission.json").unlink()
    with pytest.raises(RenderAdapterError):
        prepare_render_root(store, work, run_id="r", clip_id="clip", repeat=1)
