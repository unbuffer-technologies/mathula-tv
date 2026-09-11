"""native-dub's --lite mode: caps ffmpeg's own encode threads at half the
machine's logical CPU count and lowers the ffmpeg subprocess's OS scheduling
priority, so a background render doesn't peg every core on the user's
machine. There is no cross-platform, dependency-free way to enforce a
literal, guaranteed CPU-percentage ceiling (that needs OS-level throttling --
Windows Job Objects, complex and Windows-only) -- this is the practical,
stdlib-only approximation: thread-cap the actual CPU cost (ffmpeg/libx264)
and yield scheduling priority to foreground work.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from mathula_tv import native_dub, rendering


def test_lite_mode_cpu_threads_is_half_the_cpu_count(monkeypatch):
    monkeypatch.setattr(native_dub.os, "cpu_count", lambda: 8)
    assert native_dub._lite_mode_cpu_threads() == 4


def test_lite_mode_cpu_threads_floors_at_one(monkeypatch):
    monkeypatch.setattr(native_dub.os, "cpu_count", lambda: 1)
    assert native_dub._lite_mode_cpu_threads() == 1


def test_lite_mode_cpu_threads_falls_back_to_a_default_when_unknown(monkeypatch):
    # os.cpu_count() can genuinely return None on some platforms.
    monkeypatch.setattr(native_dub.os, "cpu_count", lambda: None)
    assert native_dub._lite_mode_cpu_threads() == 2  # half of the fallback default (4)


def test_lite_mode_subprocess_kwargs_lowers_priority_on_windows(monkeypatch):
    monkeypatch.setattr(native_dub.sys, "platform", "win32")
    kwargs = native_dub._lite_mode_subprocess_kwargs()
    assert kwargs == {"creationflags": subprocess.BELOW_NORMAL_PRIORITY_CLASS}


def test_lite_mode_subprocess_kwargs_uses_nice_on_posix(monkeypatch):
    monkeypatch.setattr(native_dub.sys, "platform", "linux")
    kwargs = native_dub._lite_mode_subprocess_kwargs()
    assert "preexec_fn" in kwargs
    assert callable(kwargs["preexec_fn"])


def test_lite_mode_popen_factory_is_a_popen_wrapper_carrying_the_platform_kwargs(monkeypatch):
    monkeypatch.setattr(native_dub.sys, "platform", "win32")
    factory = native_dub._lite_mode_popen_factory()
    assert factory.func is subprocess.Popen
    assert factory.keywords == {"creationflags": subprocess.BELOW_NORMAL_PRIORITY_CLASS}


def test_inject_ffmpeg_thread_limit_inserts_right_after_the_codec_name():
    command = ["ffmpeg", "-c:v", "libx264", "-preset", "medium", "out.mp4"]
    result = native_dub._inject_ffmpeg_thread_limit(command, codec="libx264", threads=4)
    codec_index = result.index("libx264")
    assert result[codec_index + 1] == "-threads"
    assert result[codec_index + 2] == "4"
    # Original list is untouched.
    assert command == ["ffmpeg", "-c:v", "libx264", "-preset", "medium", "out.mp4"]


def test_inject_ffmpeg_thread_limit_is_a_noop_when_the_codec_is_absent():
    command = ["ffmpeg", "-c:v", "copy", "out.mp4"]
    result = native_dub._inject_ffmpeg_thread_limit(command, codec="libx264", threads=4)
    assert result == command
    assert "-threads" not in result


def test_render_native_master_lite_mode_caps_threads_on_the_fallback_encode(tmp_path, monkeypatch):
    """The common path (stream-copy) never needs a thread cap -- only the
    libx264 fallback (triggered when the source codec can't be remuxed) does,
    since that's the actual CPU-heavy branch.
    """
    source_video = tmp_path / "source.mp4"
    dialogue_bus = tmp_path / "dialogue.wav"
    output_video = tmp_path / "master.mp4"
    source_video.write_bytes(b"source")
    dialogue_bus.write_bytes(b"dialogue")

    monkeypatch.setattr(native_dub, "probe", lambda path: {"duration": 12.0})

    calls: list[list[str]] = []

    def fake_run_ffmpeg_with_progress(command, *, expected_duration_seconds, progress_callback, popen_factory, stage):
        calls.append(list(command))
        if stage == "video_remux":
            raise subprocess.CalledProcessError(1, command)
        Path(command[-1]).write_bytes(b"encoded")

    monkeypatch.setattr(rendering, "run_ffmpeg_with_progress", fake_run_ffmpeg_with_progress)
    monkeypatch.setattr(native_dub.os, "cpu_count", lambda: 8)

    native_dub._render_native_master_with_ffmpeg_progress(
        source_video=source_video, dialogue_bus=dialogue_bus, output_video=output_video,
        progress=lambda message: None, lite=True,
    )

    assert len(calls) == 2
    remux_command, fallback_command = calls
    assert "-threads" not in remux_command
    codec_index = fallback_command.index("libx264")
    assert fallback_command[codec_index + 1] == "-threads"
    assert fallback_command[codec_index + 2] == "4"
    assert output_video.read_bytes() == b"encoded"


def test_render_native_master_without_lite_never_adds_threads(tmp_path, monkeypatch):
    source_video = tmp_path / "source.mp4"
    dialogue_bus = tmp_path / "dialogue.wav"
    output_video = tmp_path / "master.mp4"
    source_video.write_bytes(b"source")
    dialogue_bus.write_bytes(b"dialogue")

    monkeypatch.setattr(native_dub, "probe", lambda path: {"duration": 12.0})

    calls: list[list[str]] = []

    def fake_run_ffmpeg_with_progress(command, *, expected_duration_seconds, progress_callback, popen_factory, stage):
        calls.append(list(command))
        if stage == "video_remux":
            raise subprocess.CalledProcessError(1, command)
        Path(command[-1]).write_bytes(b"encoded")

    monkeypatch.setattr(rendering, "run_ffmpeg_with_progress", fake_run_ffmpeg_with_progress)

    native_dub._render_native_master_with_ffmpeg_progress(
        source_video=source_video, dialogue_bus=dialogue_bus, output_video=output_video,
        progress=lambda message: None, lite=False,
    )

    assert all("-threads" not in call for call in calls)
