from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

import pytest


def _write_tone(path: Path, duration_ms: int, sample_rate: int = 24000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = round(sample_rate * duration_ms / 1000)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        payload = bytearray()
        for n in range(frames):
            sample = int(7000 * math.sin(2 * math.pi * 220 * n / sample_rate))
            payload += struct.pack("<h", sample)
        handle.writeframes(payload)


def _word(word_id: str, text: str, start: float, end: float) -> dict:
    return {"word_id": word_id, "text": text, "start": start, "end": end}


_WORDS = [
    _word("w1", "Good", 0.0, 0.4),
    _word("w2", "morning", 0.4, 0.9),
    _word("w3", "Ayanda.", 0.9, 1.3),
    _word("w4", "Why", 1.6, 1.8),
    _word("w5", "was", 1.8, 2.0),
    _word("w6", "it", 2.0, 2.1),
    _word("w7", "urgent?", 2.1, 2.6),
    _word("w8", "He", 3.0, 3.2),
    _word("w9", "laid", 3.2, 3.5),
    _word("w10", "charges.", 3.5, 4.0),
]
_WORDS_BY_ID = {w["word_id"]: w for w in _WORDS}
_SEGMENTS_BY_ID = {"seg-1": {"segment_id": "seg-1", "source_word_ids": [w["word_id"] for w in _WORDS]}}

_SPLITTABLE_GROUP = {
    "group_id": "native_phrase_0001",
    "speaker_id": "SPEAKER_00",
    "segment_ids": ["seg-1"],
    "start_ms": 0,
    "source_end_ms": 4000,
    "source_span_ms": 4000,
    "source_text": "Good morning Ayanda. Why was it urgent? He laid charges.",
    "parts": [{"type": "text", "segment_id": "seg-1", "text": "Sawubona Ayanda. Kungani kwakuphuthuma? Wafaka amacala."}],
}

_SINGLE_SENTENCE_GROUP = {
    "group_id": "native_phrase_0002",
    "speaker_id": "SPEAKER_01",
    "segment_ids": ["seg-2"],
    "start_ms": 4500,
    "source_end_ms": 5500,
    "source_span_ms": 1000,
    "source_text": "One more thing.",
    "parts": [{"type": "text", "segment_id": "seg-2", "text": "Okunye futhi."}],
}


def test_split_phrase_groups_with_islands_passthrough_when_no_word_index():
    from mathula_tv.native_dub import _split_phrase_groups_with_islands

    groups = [_SPLITTABLE_GROUP, _SINGLE_SENTENCE_GROUP]
    flattened, counts = _split_phrase_groups_with_islands(groups, None, progress=None)
    assert [g["group_id"] for g in flattened] == ["native_phrase_0001", "native_phrase_0002"]
    assert counts == {"native_phrase_0001": 1, "native_phrase_0002": 1}


def test_split_phrase_groups_with_islands_splits_only_the_multi_sentence_group():
    from mathula_tv.native_dub import _split_phrase_groups_with_islands

    groups = [_SPLITTABLE_GROUP, _SINGLE_SENTENCE_GROUP]
    flattened, counts = _split_phrase_groups_with_islands(
        groups, (_WORDS_BY_ID, _SEGMENTS_BY_ID), progress=None,
    )
    assert counts == {"native_phrase_0001": 3, "native_phrase_0002": 1}
    assert [g["group_id"] for g in flattened] == [
        "native_phrase_0001__isl00",
        "native_phrase_0001__isl01",
        "native_phrase_0001__isl02",
        "native_phrase_0002",
    ]
    assert all(g["parent_group_id"] == "native_phrase_0001" for g in flattened[:3])
    assert "parent_group_id" not in flattened[3]


def _island_measured_entry(
    *, index: int, path: Path, start_ms: int, source_span_ms: int, tts_duration_ms: int,
    parent_start_ms: int = 0, parent_source_end_ms: int = 4000, parent_source_span_ms: int = 4000,
    source_text: str = "", spoken_text: str = "", alignment_method: str = "word_exact",
    synthesis_method: str = "rest_per_island",
) -> dict:
    island_id = f"native_phrase_0001__isl{index:02d}"
    return {
        "group_id": island_id,
        "speaker_id": "SPEAKER_00",
        "segment_ids": [island_id],
        "island_index": index,
        "alignment_method": alignment_method,
        "synthesis_method": synthesis_method,
        "parent_group_id": "native_phrase_0001",
        "parent_speaker_id": "SPEAKER_00",
        "parent_segment_ids": ["seg-1"],
        "parent_start_ms": parent_start_ms,
        "parent_source_end_ms": parent_source_end_ms,
        "parent_source_span_ms": parent_source_span_ms,
        "start_ms": start_ms,
        "source_end_ms": start_ms + source_span_ms,
        "source_window_ms": source_span_ms,
        "source_text": source_text,
        "parts": [{"type": "text", "segment_id": island_id, "text": spoken_text}],
        "path": str(path),
        "tts_duration_ms": tts_duration_ms,
        "natural_tts_duration_ms": tts_duration_ms,
        "timing_repair_used": False,
        "speed_fit": {"method": "none"},
        "mouth_close_early_tolerance_ms": 40,
        "mouth_close_late_tolerance_ms": 40,
    }


def test_restitch_island_measurements_merges_into_one_entry_per_parent(tmp_path: Path):
    from mathula_tv.native_dub import _restitch_island_measurements

    a, b, c = tmp_path / "isl0.wav", tmp_path / "isl1.wav", tmp_path / "isl2.wav"
    _write_tone(a, 1200)
    _write_tone(b, 900)
    _write_tone(c, 950)
    measured = [
        _island_measured_entry(
            index=0, path=a, start_ms=0, source_span_ms=1300, tts_duration_ms=1200,
            source_text="Good morning Ayanda.", spoken_text="Sawubona Ayanda.",
        ),
        _island_measured_entry(
            index=1, path=b, start_ms=1600, source_span_ms=1000, tts_duration_ms=900,
            source_text="Why was it urgent?", spoken_text="Kungani kwakuphuthuma?",
        ),
        _island_measured_entry(
            index=2, path=c, start_ms=3000, source_span_ms=1000, tts_duration_ms=1200,
            source_text="He laid charges.", spoken_text="Wafaka amacala.",
            alignment_method="word_aligned_partial",
        ),
    ]

    restitched, records = _restitch_island_measurements(
        measured, {"native_phrase_0001": 3},
        audio_dir=tmp_path, max_natural_speed_percent=12, progress=None,
    )

    assert len(restitched) == 1
    entry = restitched[0]
    assert entry["group_id"] == "native_phrase_0001"
    assert entry["speaker_id"] == "SPEAKER_00"
    assert entry["segment_ids"] == ["seg-1"]
    assert entry["start_ms"] == 0
    assert entry["source_end_ms"] == 4000
    assert entry["tts_duration_ms"] == 1200 + 900 + 950
    assert entry["island_split_used"] is True
    assert entry["island_count"] == 3
    assert Path(entry["path"]).is_file()

    assert len(records) == 1
    assert records[0]["group_id"] == "native_phrase_0001"
    assert records[0]["island_count"] == 3
    islands = records[0]["islands"]
    assert [i["index"] for i in islands] == [0, 1, 2]
    # Each island carries both sides of the translation for island-for-island
    # comparison, plus how it was aligned and its real required rush.
    assert [i["source_text"] for i in islands] == [
        "Good morning Ayanda.", "Why was it urgent?", "He laid charges.",
    ]
    assert [i["spoken_text"] for i in islands] == [
        "Sawubona Ayanda.", "Kungani kwakuphuthuma?", "Wafaka amacala.",
    ]
    assert [i["alignment_method"] for i in islands] == [
        "word_exact", "word_exact", "word_aligned_partial",
    ]
    # Island 0: natural 1200ms against a 1300ms window -- already inside budget.
    assert islands[0]["required_rush_percent"] == 0.0
    # Island 1: natural 900ms against a 1000ms window -- also inside budget.
    assert islands[1]["required_rush_percent"] == 0.0
    # Island 2: natural 1200ms against a 1000ms window -- needs +20% rush.
    assert islands[2]["required_rush_percent"] == 20.0
    assert islands[2]["direction"] == "compress"
    # Default synthesis_method (no SDK pre-pass involved) is rest_per_island.
    assert [i["synthesis_method"] for i in islands] == ["rest_per_island"] * 3


def test_island_comparison_record_reports_which_synthesis_path_each_island_took(tmp_path: Path):
    # A restitched group can mix islands that were pre-synthesized by the SDK
    # bookmark pre-pass with islands that fell back to (or were later repaired
    # via) the per-island REST path -- speech_islands.json must show which path
    # each individual island actually took, not a single group-wide value.
    from mathula_tv.native_dub import _restitch_island_measurements

    a, b = tmp_path / "isl0.wav", tmp_path / "isl1.wav"
    _write_tone(a, 1000)
    _write_tone(b, 900)
    measured = [
        _island_measured_entry(
            index=0, path=a, start_ms=0, source_span_ms=1000, tts_duration_ms=1000,
            synthesis_method="sdk_bookmark_group",
        ),
        _island_measured_entry(
            index=1, path=b, start_ms=1000, source_span_ms=1000, tts_duration_ms=900,
            synthesis_method="rest_per_island",
        ),
    ]
    _, records = _restitch_island_measurements(
        measured, {"native_phrase_0001": 2}, audio_dir=tmp_path, max_natural_speed_percent=12, progress=None,
    )
    islands = records[0]["islands"]
    assert [i["synthesis_method"] for i in islands] == ["sdk_bookmark_group", "rest_per_island"]


def test_restitch_island_measurements_passes_through_unsplit_groups_unchanged(tmp_path: Path):
    from mathula_tv.native_dub import _restitch_island_measurements

    whole_group_entry = {
        "group_id": "native_phrase_0002", "speaker_id": "SPEAKER_01",
        "segment_ids": ["seg-2"], "start_ms": 4500, "source_end_ms": 5500,
        "source_window_ms": 1000, "source_text": "Good morning.",
        "parts": [{"type": "text", "segment_id": "seg-2", "text": "Sawubona."}],
        "path": str(tmp_path / "whole.wav"), "tts_duration_ms": 950,
    }
    restitched, records = _restitch_island_measurements(
        [whole_group_entry], {"native_phrase_0002": 1},
        audio_dir=tmp_path, max_natural_speed_percent=12, progress=None,
    )
    assert restitched == [whole_group_entry]
    # Unsplit groups still get a comparison record -- so a job's
    # speech_islands.json covers every real speech unit, not just the subset
    # that happened to split into multiple islands.
    assert len(records) == 1
    assert records[0]["group_id"] == "native_phrase_0002"
    assert records[0]["island_count"] == 1
    island = records[0]["islands"][0]
    assert island["source_text"] == "Good morning."
    assert island["spoken_text"] == "Sawubona."
    assert island["tts_duration_ms"] == 950


def test_restitch_island_measurements_applies_safety_speed_fit_when_stitched_total_overruns(tmp_path: Path):
    from mathula_tv.native_dub import _restitch_island_measurements

    a, b = tmp_path / "isl0.wav", tmp_path / "isl1.wav"
    _write_tone(a, 2000)
    _write_tone(b, 2000)
    # Stitched total (4000ms) badly overruns a 2000ms parent window -- must trigger
    # the same pitch-preserving safety-net speed fit the group-level path already uses.
    measured = [
        _island_measured_entry(
            index=0, path=a, start_ms=0, source_span_ms=1000, tts_duration_ms=2000,
            parent_source_end_ms=2000, parent_source_span_ms=2000,
        ),
        _island_measured_entry(
            index=1, path=b, start_ms=1000, source_span_ms=1000, tts_duration_ms=2000,
            parent_source_end_ms=2000, parent_source_span_ms=2000,
        ),
    ]

    restitched, _records = _restitch_island_measurements(
        measured, {"native_phrase_0001": 2},
        audio_dir=tmp_path, max_natural_speed_percent=12, progress=None,
    )

    entry = restitched[0]
    assert entry["speed_fit"]["method"] == "ffmpeg_atempo"
    assert entry["tts_duration_ms"] < 4000
    assert Path(entry["path"]).is_file()
