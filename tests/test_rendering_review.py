"""Unit tests for validated review MP4 rendering (no speech truncation)."""

from __future__ import annotations

import json

import pytest

from conftest import make_video, requires_ffmpeg, write_wav

from mathula_tv import rendering
from mathula_tv.errors import RenderFailure


@requires_ffmpeg
def test_render_review_mp4_stream_copies_video(tmp_path):
    source = make_video(tmp_path / "source.mp4", duration=2.0, size="320x240")
    audio = write_wav(tmp_path / "final.wav", 1800, rate=48000, channels=2)
    output = tmp_path / "out" / "review.mp4"
    manifest_path = tmp_path / "out" / "render.json"
    manifest = rendering.render_review_mp4(source, audio, output, manifest_path)
    assert output.is_file()
    assert manifest["video_stream_copy"] is True
    assert manifest["audio_codec"] == "aac"
    assert manifest["audio_sample_rate"] == 48000
    assert manifest["youtube_uploaded"] is False
    assert json.loads(manifest_path.read_text())["schema_version"] == "render-manifest-v1"


@requires_ffmpeg
def test_render_review_mp4_burns_subtitles(tmp_path):
    source = make_video(tmp_path / "source.mp4", duration=2.0)
    audio = write_wav(tmp_path / "final.wav", 1800, rate=48000, channels=2)
    subs = tmp_path / "subs.srt"
    subs.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nSawubona\n\n"
        "2\n00:00:01,000 --> 00:00:02,000\nUnjani\n",
        encoding="utf-8",
    )
    output = tmp_path / "review.mp4"
    manifest = rendering.render_review_mp4(source, audio, output, tmp_path / "m.json", subtitles=subs)
    assert manifest["burned_subtitles"] == str(subs)
    assert manifest["video_stream_copy"] is False
    assert output.is_file()


@requires_ffmpeg
def test_render_review_mp4_refuses_to_truncate_speech(tmp_path):
    source = make_video(tmp_path / "source.mp4", duration=1.0)
    # Audio noticeably longer than the source video must not be truncated.
    audio = write_wav(tmp_path / "final.wav", 3000, rate=48000, channels=2)
    with pytest.raises(RenderFailure, match="longer than the source"):
        rendering.render_review_mp4(source, audio, tmp_path / "out.mp4", tmp_path / "m.json")
