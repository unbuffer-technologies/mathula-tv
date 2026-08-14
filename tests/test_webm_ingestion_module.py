from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from mathula_tv.webm_ingestion import normalize_webm_to_mp4, probe_media


def test_webm_normalizes_to_h264_aac(tmp_path: Path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg tools unavailable")
    encoders = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if any(codec not in encoders for codec in ("libvpx-vp9", "libopus", "libx264", "aac")):
        pytest.skip("Required test codecs unavailable")

    source = tmp_path / "clip.webm"
    output = tmp_path / "source.mp4"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc=size=320x180:rate=25:duration=1.2",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=1.2",
            "-c:v", "libvpx-vp9", "-b:v", "300k", "-c:a", "libopus",
            "-shortest", str(source),
        ],
        check=True,
    )
    result = normalize_webm_to_mp4(source, output)
    info = probe_media(output)
    video = next(s for s in info["streams"] if s.get("codec_type") == "video")
    audio = next(s for s in info["streams"] if s.get("codec_type") == "audio")
    assert result["normalized"] is True
    assert video["codec_name"] == "h264"
    assert audio["codec_name"] == "aac"
    assert int(audio["sample_rate"]) == 48000
