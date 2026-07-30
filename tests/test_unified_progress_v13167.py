from __future__ import annotations

import io

from mathula_tv.autocorrect_research import (
    _entity_inventory_progress_callback,
    _provider_progress_callback,
)
from mathula_tv.rendering import run_ffmpeg_with_progress


class _FakeProcess:
    def __init__(self) -> None:
        self.stdout = io.StringIO(
            "\n".join(
                (
                    "frame=125",
                    "fps=25.0",
                    "total_size=1048576",
                    "out_time_us=5000000",
                    "speed=0.50x",
                    "progress=continue",
                    "frame=250",
                    "fps=25.0",
                    "total_size=2097152",
                    "out_time_us=10000000",
                    "speed=0.50x",
                    "progress=end",
                )
            )
            + "\n"
        )
        self.stderr = io.StringIO("")

    def wait(self) -> int:
        return 0


def test_clean_master_ffmpeg_reports_percent_speed_size_and_eta() -> None:
    events: list[dict] = []
    commands: list[list[str]] = []
    times = iter((100.0, 105.0, 110.0))

    def popen(command, **_kwargs):
        commands.append(list(command))
        return _FakeProcess()

    run_ffmpeg_with_progress(
        ["ffmpeg", "-i", "source.mp4", "output.mp4"],
        expected_duration_seconds=10.0,
        progress_callback=events.append,
        popen_factory=popen,
        clock=lambda: next(times),
    )

    assert commands
    assert commands[0][-1] == "output.mp4"
    assert commands[0][commands[0].index("-progress") + 1] == "pipe:1"
    assert events[0]["status"] == "started"
    assert events[1]["percent"] == 50.0
    assert events[1]["speed"] == 0.5
    assert events[1]["encoded_bytes"] == 1_048_576
    assert events[1]["eta_seconds"] == 10.0
    assert events[-1]["status"] == "completed"
    assert events[-1]["percent"] == 100.0


def test_web_research_estimate_falls_back_to_request_bytes(capsys) -> None:
    callback = _provider_progress_callback(1, 2)

    callback(
        {
            "event": "request_attempt_started",
            "tool_type": "web_search",
            "attempt": 1,
            "max_retries": 2,
            "request_json_bytes": 10_000,
        }
    )

    captured = capsys.readouterr()
    assert "~2,500 input tokens estimated" in captured.err
    assert "~0 input tokens" not in captured.err


def test_entity_inventory_prints_request_and_actual_usage(capsys) -> None:
    _entity_inventory_progress_callback(
        {
            "event": "request_metrics",
            "request_body_bytes": 12_000,
            "payload_bytes": 8_000,
            "output_schema_bytes": 2_000,
            "max_output_tokens": 8_000,
        }
    )
    _entity_inventory_progress_callback(
        {
            "event": "response_usage",
            "input_tokens": 2_700,
            "output_tokens": 900,
            "thinking_tokens": 120,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "response_body_bytes": 4_096,
        }
    )

    captured = capsys.readouterr()
    assert "entity inventory AI: sending" in captured.err
    assert "~3,000 input tokens estimated" in captured.err
    assert "input=2,700" in captured.err
    assert "thinking=120" in captured.err
