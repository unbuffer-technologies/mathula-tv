from __future__ import annotations

import io
import subprocess

import pytest

from mathula_tv.tiktok_editor import (
    _ffmpeg_progress_seconds,
    _run_ffmpeg_with_progress,
)


class _Process:
    def __init__(self, command, *, lines: str, returncode: int = 0):
        self.command = list(command)
        self.stdout = io.StringIO(lines)
        self.stderr = io.StringIO("synthetic ffmpeg error" if returncode else "")
        self.returncode = returncode

    def wait(self):
        return self.returncode


def test_machine_readable_ffmpeg_progress_reports_percent_speed_and_eta() -> None:
    events = []
    processes = []
    times = iter((100.0, 102.0, 110.0, 120.0))

    def popen(command, **kwargs):
        assert kwargs["stdout"] is subprocess.PIPE
        assert kwargs["stderr"] is subprocess.PIPE
        assert kwargs["text"] is True
        process = _Process(
            command,
            lines=(
                "frame=300\n"
                "fps=60.0\n"
                "total_size=1000000\n"
                "out_time_us=10000000\n"
                "speed=2.0x\n"
                "progress=continue\n"
                "frame=1500\n"
                "fps=58.0\n"
                "total_size=5000000\n"
                "out_time_us=50000000\n"
                "speed=1.5x\n"
                "progress=continue\n"
                "frame=3000\n"
                "fps=55.0\n"
                "total_size=10000000\n"
                "out_time_us=100000000\n"
                "speed=1.25x\n"
                "progress=end\n"
            ),
        )
        processes.append(process)
        return process

    _run_ffmpeg_with_progress(
        ["ffmpeg", "-i", "input.mp4", "output.mp4"],
        expected_duration_seconds=100.0,
        progress_callback=events.append,
        popen_factory=popen,
        clock=lambda: next(times),
    )

    command = processes[0].command
    assert command[-1] == "output.mp4"
    assert command[command.index("-progress") + 1] == "pipe:1"
    assert command[command.index("-stats_period") + 1] == "1"
    assert "-nostats" in command

    assert events[0]["status"] == "started"
    running = [
        event for event in events if event["status"] == "running"
    ]
    assert [event["percent"] for event in running] == [10.0, 50.0]
    assert running[0]["speed"] == 2.0
    assert running[0]["eta_seconds"] == 45.0
    assert running[1]["frame"] == 1500
    assert running[1]["encoded_bytes"] == 5_000_000
    assert events[-1]["status"] == "completed"
    assert events[-1]["percent"] == 100.0
    assert events[-1]["eta_seconds"] == 0.0


def test_ffmpeg_progress_failure_preserves_stderr() -> None:
    def popen(command, **_kwargs):
        return _Process(
            command,
            lines="frame=0\nprogress=continue\n",
            returncode=9,
        )

    with pytest.raises(subprocess.CalledProcessError) as caught:
        _run_ffmpeg_with_progress(
            ["ffmpeg", "-i", "input.mp4", "output.mp4"],
            expected_duration_seconds=100.0,
            progress_callback=lambda _event: None,
            popen_factory=popen,
            clock=lambda: 0.0,
        )

    assert caught.value.returncode == 9
    assert caught.value.stderr == "synthetic ffmpeg error"


def test_ffmpeg_historical_out_time_ms_is_microseconds() -> None:
    assert _ffmpeg_progress_seconds({"out_time_ms": "12500000"}) == 12.5
