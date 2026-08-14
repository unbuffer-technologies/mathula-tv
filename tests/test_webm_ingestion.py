from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from mathula_tv.config import Settings
from mathula_tv.errors import InvalidSourceMedia
from mathula_tv.media import (
    checksum,
    normalize_webm_to_mp4,
    probe,
    validate_video_input,
)
from mathula_tv.orchestrator import Orchestrator


def settings(tmp_path: Path) -> Settings:
    return Settings(
        tmp_path / "working",
        "bucket",
        "mathula-tv",
        tmp_path / "case.json",
        tmp_path / "politics.json",
        "",
        "eastus",
        "en-ZA",
        "2025-10-15",
        10,
        600,
        "",
        "",
        "preview",
        "model",
    )


def require_test_codecs() -> None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg tools are unavailable")
    encoders = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    required = ("libvpx-vp9", "libopus", "libx264", "aac")
    if any(codec not in encoders for codec in required):
        pytest.skip("Required WebM/MP4 test codecs are unavailable")


def make_webm(path: Path, duration: float = 1.2) -> Path:
    require_test_codecs()
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
            f"testsrc=size=320x180:rate=25:duration={duration}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:sample_rate=48000:duration={duration}",
            "-c:v",
            "libvpx-vp9",
            "-b:v",
            "300k",
            "-c:a",
            "libopus",
            "-shortest",
            str(path),
        ],
        check=True,
    )
    return path


def test_supported_input_extensions_are_explicit(tmp_path: Path):
    unsupported = tmp_path / "clip.mov"
    unsupported.write_bytes(b"not inspected")
    with pytest.raises(InvalidSourceMedia, match="Unsupported source video extension"):
        validate_video_input(unsupported)


def test_normalize_webm_produces_h264_aac_mp4(tmp_path: Path):
    source = make_webm(tmp_path / "source.webm")
    output = tmp_path / "source.mp4"

    manifest = normalize_webm_to_mp4(source, output)
    info = probe(output)
    video = next(item for item in info["streams"] if item.get("codec_type") == "video")
    audio = next(item for item in info["streams"] if item.get("codec_type") == "audio")

    assert manifest["normalized"] is True
    assert manifest["source_video_codec"] == "vp9"
    assert manifest["canonical_sha256"] == checksum(output)
    assert video["codec_name"] == "h264"
    assert audio["codec_name"] == "aac"
    assert int(audio["sample_rate"]) == 48_000
    assert abs(float(info["duration"]) - float(probe(source)["duration"])) <= 0.5


def test_submit_webm_preserves_original_and_uses_canonical_mp4(tmp_path: Path):
    source = make_webm(tmp_path / "incoming/clip.webm")
    app = Orchestrator(settings(tmp_path))

    job = app.submit(source)
    job_dir = app.jobs.job_dir(job.job_id)
    ingestion = job.media["source_ingestion"]

    assert job.source_filename == "clip.webm"
    assert Path(job.local_source_path) == job_dir / "input/source.mp4"
    assert Path(job.objects["original_media"]) == job_dir / "input/clip.webm"
    assert Path(job.objects["source_media"]) == Path(job.local_source_path)
    assert ingestion["normalized"] is True
    assert ingestion["original_sha256"] == checksum(source)
    assert job.source_checksum == checksum(Path(job.local_source_path))
    assert Path(job.objects["analysis_mono"]).is_file()
    assert Path(job.objects["mix_source_stereo"]).is_file()
    assert job.state == "audio_prepared"


def test_submit_mp4_remains_pass_through(tmp_path: Path):
    require_test_codecs()
    source = tmp_path / "incoming/clip.mp4"
    source.parent.mkdir(parents=True, exist_ok=True)
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
            "testsrc=size=320x180:rate=25:duration=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(source),
        ],
        check=True,
    )
    app = Orchestrator(settings(tmp_path))

    job = app.submit(source)
    ingestion = job.media["source_ingestion"]

    assert Path(job.local_source_path).name == "clip.mp4"
    assert ingestion["normalized"] is False
    assert ingestion["original_sha256"] == ingestion["canonical_sha256"]
    assert job.source_checksum == checksum(Path(job.local_source_path))
