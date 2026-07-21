"""Shared fixtures and real-media helpers for offline ffmpeg-backed tests."""

from __future__ import annotations

import math
import shutil
import subprocess
import wave
from pathlib import Path

import pytest

FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
requires_ffmpeg = pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg/ffprobe are required")


def write_wav(
    path: Path,
    duration_ms: int = 1000,
    *,
    rate: int = 24000,
    channels: int = 1,
    amplitude: int = 8000,
) -> Path:
    """Write a deterministic tone as PCM16 WAV without needing ffmpeg."""
    frames = round(rate * duration_ms / 1000)
    raw = bytearray()
    for index in range(frames):
        value = round(amplitude * math.sin(index / 8))
        for _ in range(channels):
            raw.extend(value.to_bytes(2, "little", signed=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setparams((channels, 2, rate, 0, "NONE", "not compressed"))
        handle.writeframes(raw)
    return path


def make_video(path: Path, *, duration: float = 1.0, size: str = "320x240", rate: int = 25) -> Path:
    """Render a short H.264 + AAC MP4 with a tone so probe reports real streams."""
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=duration={duration}:size={size}:rate={rate}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={duration}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ],
        check=True,
    )
    return path


@pytest.fixture
def wav_factory(tmp_path):
    def _factory(name: str = "audio.wav", **kwargs) -> Path:
        return write_wav(tmp_path / name, **kwargs)

    return _factory


@pytest.fixture
def video_factory(tmp_path):
    def _factory(name: str = "video.mp4", **kwargs) -> Path:
        return make_video(tmp_path / name, **kwargs)

    return _factory
