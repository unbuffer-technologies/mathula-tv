"""Tests for _pause_aware_speed_fit_wav: real internal near-silent gaps get
protected toward a floor instead of being uniformly crushed along with
speech during turn-level hard-sync compression (see native_dub.py's
_synthesize_speaker_turns_with_bookmarks compress branch).

Real production motivation (job 33cd7b46...): a genuine ~80ms natural pause
at a comma boundary survived at 0ms after the old uniform-atempo mechanism
(_speed_fit_overflow_wav) compressed a whole turn -- confirmed via real
Azure STT word timestamps. This file proves the replacement mechanism with
real synthesized WAVs and real ffmpeg atempo calls, not fakes.
"""
from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

from mathula_tv.native_dub import (
    _detect_internal_silence_runs,
    _pause_aware_speed_fit_wav,
    _remap_ms_through_segments,
    _speed_fit_overflow_wav,
)

FRAMERATE = 24_000


def _tone_frames(duration_ms: float, *, freq: float = 220.0, amplitude: int = 8000) -> bytes:
    n = int(round(duration_ms * FRAMERATE / 1000.0))
    return b"".join(
        struct.pack("<h", int(amplitude * math.sin(2 * math.pi * freq * i / FRAMERATE)))
        for i in range(n)
    )


def _silence_frames(duration_ms: float) -> bytes:
    return b"\x00\x00" * int(round(duration_ms * FRAMERATE / 1000.0))


def _write_wav(path: Path, frames: bytes) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setparams((1, 2, FRAMERATE, 0, "NONE", "not compressed"))
        handle.writeframes(frames)


def _wav_duration_ms(path: Path) -> int:
    with wave.open(str(path), "rb") as handle:
        return round(handle.getnframes() * 1000 / handle.getframerate())


def test_detect_internal_silence_runs_excludes_edges_and_short_gaps(tmp_path: Path):
    frames = (
        _silence_frames(500)  # leading edge -- never a real "internal" gap
        + _tone_frames(1000)
        + _silence_frames(20)  # too short to count (below min_run_ms)
        + _tone_frames(1000)
        + _silence_frames(100)  # a real, detectable internal gap
        + _tone_frames(1000)
        + _silence_frames(500)  # trailing edge -- never a real "internal" gap
    )
    runs = _detect_internal_silence_runs(
        frames, sample_width=2, channels=1, framerate=FRAMERATE, threshold=32, min_run_ms=50,
    )
    assert len(runs) == 1
    start_frame, end_frame = runs[0]
    duration_ms = (end_frame - start_frame) * 1000 / FRAMERATE
    assert 95 <= duration_ms <= 105


def test_no_internal_gaps_falls_back_to_plain_uniform_behavior(tmp_path: Path):
    frames = _tone_frames(6000)
    source = tmp_path / "source.wav"
    _write_wav(source, frames)

    pause_aware_output = tmp_path / "pause_aware.wav"
    pause_aware_result = _pause_aware_speed_fit_wav(
        source_path=source, output_path=pause_aware_output, target_duration_ms=4000, max_speed_percent=12,
    )
    uniform_output = tmp_path / "uniform.wav"
    uniform_result = _speed_fit_overflow_wav(
        source_path=source, output_path=uniform_output, target_duration_ms=4000, max_speed_percent=12,
    )
    assert pause_aware_result["method"] == "ffmpeg_atempo"
    assert pause_aware_result["final_duration_ms"] == uniform_result["final_duration_ms"]
    assert pause_aware_result["speed_factor"] == uniform_result["speed_factor"]
    assert pause_aware_result["segments"] == [
        {"old_start_ms": 0, "old_end_ms": 6000, "new_start_ms": 0, "new_end_ms": uniform_result["final_duration_ms"]}
    ]


def test_already_fits_returns_none_method_with_identity_segment(tmp_path: Path):
    frames = _tone_frames(1000) + _silence_frames(200) + _tone_frames(1000)
    source = tmp_path / "source.wav"
    _write_wav(source, frames)
    output = tmp_path / "out.wav"
    result = _pause_aware_speed_fit_wav(
        source_path=source, output_path=output, target_duration_ms=5000, max_speed_percent=12,
    )
    assert result["method"] == "none"
    before = result["before_duration_ms"]
    assert result["segments"] == [{"old_start_ms": 0, "old_end_ms": before, "new_start_ms": 0, "new_end_ms": before}]


def test_gap_scales_proportionally_when_floor_does_not_bind(tmp_path: Path):
    # Natural: 4000 + 300(gap) + 4000 + 200(gap) + 4000 = 12500ms. Target 10000ms
    # -> uniform_factor = 0.8, so naive proportional gaps would be 240ms/160ms,
    # both comfortably above the 30ms floor -- floor should never engage here.
    frames = (
        _tone_frames(4000) + _silence_frames(300) + _tone_frames(4000)
        + _silence_frames(200) + _tone_frames(4000)
    )
    source = tmp_path / "source.wav"
    _write_wav(source, frames)
    output = tmp_path / "out.wav"
    result = _pause_aware_speed_fit_wav(
        source_path=source, output_path=output, target_duration_ms=10_000, max_speed_percent=50,
    )
    assert result["method"] == "pause_aware_atempo"
    assert result["gap_count"] == 2
    # Floor not binding -> behaves identically to plain uniform scaling: the
    # speech-only factor should equal the naive uniform factor (1/0.8 = 1.25).
    assert abs(result["speed_percent"] - 25.0) < 1.0
    assert abs(result["gap_protected_ms_total"] - (240 + 160)) < 5
    assert abs(int(result["final_duration_ms"]) - 10_000) <= 50


def test_gap_clamped_to_floor_and_speech_compresses_harder_than_naive_uniform(tmp_path: Path):
    # Natural: 4000 + 80(gap) + 4000 = 8080ms. Target 1600ms -> naive uniform
    # gap would be 80 * (1600/8080) ~= 15.8ms, well below the 30ms floor.
    frames = _tone_frames(4000) + _silence_frames(80) + _tone_frames(4000)
    source = tmp_path / "source.wav"
    _write_wav(source, frames)
    output = tmp_path / "out.wav"
    result = _pause_aware_speed_fit_wav(
        source_path=source, output_path=output, target_duration_ms=1600, max_speed_percent=500,
    )
    assert result["method"] == "pause_aware_atempo"
    assert result["gap_count"] == 1
    assert abs(result["gap_protected_ms_total"] - 30.0) < 2  # clamped to the floor
    naive_uniform_speed_percent = (8080 / 1600 - 1.0) * 100.0
    assert result["speed_percent"] > naive_uniform_speed_percent  # speech pays the difference
    assert abs(int(result["final_duration_ms"]) - 1600) <= 20


def test_gap_never_expanded_beyond_its_own_natural_duration(tmp_path: Path):
    # A pathological floor higher than a real gap's own natural length must
    # never EXPAND that gap past what was actually there.
    frames = _tone_frames(2000) + _silence_frames(10) + _tone_frames(2000)
    source = tmp_path / "source.wav"
    _write_wav(source, frames)
    output = tmp_path / "out.wav"
    result = _pause_aware_speed_fit_wav(
        source_path=source, output_path=output, target_duration_ms=3000, max_speed_percent=50,
        min_gap_detect_ms=5, gap_floor_ms=200,  # floor far exceeds the real 10ms gap
    )
    if result["gap_count"]:
        assert result["gap_protected_ms_total"] <= 15  # never expanded toward the oversized floor


def test_bookmark_remap_is_piecewise_linear_through_segments(tmp_path: Path):
    frames = _tone_frames(4000) + _silence_frames(300) + _tone_frames(4000) + _silence_frames(200) + _tone_frames(4000)
    source = tmp_path / "source.wav"
    _write_wav(source, frames)
    output = tmp_path / "out.wav"
    result = _pause_aware_speed_fit_wav(
        source_path=source, output_path=output, target_duration_ms=10_000, max_speed_percent=50,
    )
    segments = result["segments"]
    # A bookmark sitting exactly at the start of the 2nd speech segment (right
    # after the first gap) must land at that segment's own new_start_ms.
    second_speech_start_old = 4000.0 + 300.0
    remapped = _remap_ms_through_segments(second_speech_start_old, segments)
    matching = next(s for s in segments if abs(s["old_start_ms"] - second_speech_start_old) < 1)
    assert abs(remapped - matching["new_start_ms"]) <= 2
    # Origin and final edge both clamp sensibly.
    assert _remap_ms_through_segments(0.0, segments) == 0
    assert abs(_remap_ms_through_segments(12_500.0, segments) - int(result["final_duration_ms"])) <= 5


def test_pause_aware_output_duration_matches_wav_file_on_disk(tmp_path: Path):
    frames = _tone_frames(4000) + _silence_frames(300) + _tone_frames(4000)
    source = tmp_path / "source.wav"
    _write_wav(source, frames)
    output = tmp_path / "out.wav"
    result = _pause_aware_speed_fit_wav(
        source_path=source, output_path=output, target_duration_ms=6000, max_speed_percent=50,
    )
    assert abs(_wav_duration_ms(output) - result["final_duration_ms"]) <= 1
