from __future__ import annotations

import math
import shutil
import struct
import wave
from pathlib import Path

import pytest

from mathula_tv.mixing import render_dialogue_tracks


def _write_tone(path: Path, *, sample_rate: int = 24_000, duration_ms: int = 1_000) -> None:
    frames = round(sample_rate * duration_ms / 1_000)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        samples = bytearray()
        for index in range(frames):
            value = round(8_000 * math.sin(2 * math.pi * 440 * index / sample_rate))
            samples.extend(struct.pack("<h", value))
        handle.writeframes(samples)


def _window_peak(path: Path, start_ms: int, end_ms: int) -> int:
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        handle.setpos(round(start_ms * rate / 1_000))
        raw = handle.readframes(round((end_ms - start_ms) * rate / 1_000))
    values = struct.iter_unpack("<h", raw)
    return max((abs(value[0]) for value in values), default=0)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required")
def test_delayed_dialogue_units_remain_audible(tmp_path: Path) -> None:
    clips = []
    for index, start_ms in enumerate((0, 2_000, 4_000), start=1):
        clip_path = tmp_path / f"unit_{index:04d}.wav"
        _write_tone(clip_path)
        clips.append(
            {
                "unit_id": f"unit_{index:04d}",
                "speaker_id": "SPEAKER_00",
                "start_ms": start_ms,
                "duration_ms": 1_000,
                "path": str(clip_path),
            }
        )

    dialogue = tmp_path / "dub_dialogue.wav"
    render_dialogue_tracks(
        clips,
        tmp_path / "speakers",
        dialogue,
        total_duration_ms=6_000,
    )

    assert _window_peak(dialogue, 200, 800) > 1_000
    assert _window_peak(dialogue, 2_200, 2_800) > 1_000
    assert _window_peak(dialogue, 4_200, 4_800) > 1_000
