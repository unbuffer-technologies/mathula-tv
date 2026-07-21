"""Unit tests for the ffmpeg/ffprobe media boundary in mathula_tv.media."""

from __future__ import annotations

import subprocess
import wave

import pytest

from conftest import make_video, requires_ffmpeg, write_wav

from mathula_tv import media
from mathula_tv.errors import InvalidSourceMedia, UnsupportedSourceDuration


def test_checksum_is_stable_and_content_sensitive(tmp_path):
    first = tmp_path / "a.bin"
    second = tmp_path / "b.bin"
    first.write_bytes(b"mathula" * 1000)
    second.write_bytes(b"mathula" * 1000)
    changed = tmp_path / "c.bin"
    changed.write_bytes(b"mathula" * 1000 + b"!")
    assert media.checksum(first) == media.checksum(second)
    assert media.checksum(first) != media.checksum(changed)


def test_require_media_tools_reports_missing(monkeypatch):
    monkeypatch.setattr(media.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="Missing required tools"):
        media.require_media_tools()


def test_require_media_tools_passes_when_present(monkeypatch):
    monkeypatch.setattr(media.shutil, "which", lambda name: f"/usr/bin/{name}")
    media.require_media_tools()


def test_probe_rejects_missing_or_empty(tmp_path):
    missing = tmp_path / "nope.wav"
    with pytest.raises(ValueError, match="Missing or empty media"):
        media.probe(missing)
    empty = tmp_path / "empty.wav"
    empty.write_bytes(b"")
    with pytest.raises(ValueError, match="Missing or empty media"):
        media.probe(empty)


@requires_ffmpeg
def test_probe_reads_streams_and_duration(tmp_path):
    source = write_wav(tmp_path / "tone.wav", 1500, rate=16000)
    info = media.probe(source)
    assert info["duration"] == pytest.approx(1.5, abs=0.05)
    assert any(stream["codec_type"] == "audio" for stream in info["streams"])


@requires_ffmpeg
def test_ffmpeg_version_returns_banner_line():
    banner = media.ffmpeg_version()
    assert "ffmpeg" in banner.lower()


@requires_ffmpeg
def test_prepare_audio_normalizes_to_mono_16k(tmp_path):
    source = write_wav(tmp_path / "stereo.wav", 1000, rate=48000, channels=2)
    result = media.prepare_audio(source, tmp_path / "out" / "analysis.wav")
    assert result["channels"] == 1
    assert result["sample_rate"] == 16000
    assert result["source_duration"] == pytest.approx(1.0, abs=0.05)
    assert result["audio_checksum"] == media.checksum(tmp_path / "out" / "analysis.wav")


@requires_ffmpeg
def test_prepare_source_derivatives_produces_two_formats(tmp_path):
    source = write_wav(tmp_path / "src.wav", 1200, rate=44100, channels=2)
    result = media.prepare_source_derivatives(
        source, tmp_path / "analysis_mono.wav", tmp_path / "mix_stereo.wav"
    )
    assert result["analysis_mono"]["channels"] == 1
    assert result["analysis_mono"]["sample_rate"] == 16000
    assert result["mix_source_stereo"]["channels"] == 2
    assert result["mix_source_stereo"]["sample_rate"] == 48000
    # Backward-compatible fields mirror the analysis derivative.
    assert result["channels"] == 1 and result["sample_rate"] == 16000


@requires_ffmpeg
def test_prepare_source_derivatives_rejects_long_source(tmp_path):
    source = write_wav(tmp_path / "src.wav", 1000)
    with pytest.raises(UnsupportedSourceDuration):
        media.prepare_source_derivatives(
            source,
            tmp_path / "a.wav",
            tmp_path / "m.wav",
            max_duration_seconds=0.1,
        )


@requires_ffmpeg
def test_prepare_source_derivatives_requires_audio_stream(tmp_path):
    silent_video = tmp_path / "silent.mp4"
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
            "testsrc=duration=1:size=160x120:rate=15",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(silent_video),
        ],
        check=True,
    )
    with pytest.raises(InvalidSourceMedia, match="no audio stream"):
        media.prepare_source_derivatives(
            silent_video, tmp_path / "a.wav", tmp_path / "m.wav"
        )


@requires_ffmpeg
def test_render_video_muxes_source_and_audio(tmp_path):
    source = make_video(tmp_path / "source.mp4", duration=1.0)
    audio = write_wav(tmp_path / "dub.wav", 1000, rate=48000)
    output = tmp_path / "out" / "review.mp4"
    media.render_video(source, audio, output)
    info = media.probe(output)
    assert any(stream["codec_type"] == "video" for stream in info["streams"])
    assert any(stream["codec_type"] == "audio" for stream in info["streams"])


@requires_ffmpeg
def test_validate_pcm_derivative_flags_format_mismatch(tmp_path):
    wrong = write_wav(tmp_path / "wrong.wav", 1000, rate=8000, channels=1)
    with pytest.raises(InvalidSourceMedia, match="format mismatch"):
        media._validate_pcm_derivative(wrong, channels=1, sample_rate=16000, expected_duration=1.0)


@requires_ffmpeg
def test_validate_pcm_derivative_flags_duration_mismatch(tmp_path):
    audio = write_wav(tmp_path / "short.wav", 1000, rate=16000, channels=1)
    with pytest.raises(InvalidSourceMedia, match="duration mismatch"):
        media._validate_pcm_derivative(audio, channels=1, sample_rate=16000, expected_duration=5.0)


def test_probe_zero_duration_is_rejected(tmp_path, monkeypatch):
    audio = tmp_path / "zero.wav"
    with wave.open(str(audio), "wb") as handle:
        handle.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        handle.writeframes(b"")
    monkeypatch.setattr(media, "require_media_tools", lambda: None)

    class _Result:
        stdout = '{"format": {"duration": "0"}, "streams": []}'

    monkeypatch.setattr(media.subprocess, "run", lambda *a, **k: _Result())
    with pytest.raises(ValueError, match="zero duration"):
        media.probe(audio)
