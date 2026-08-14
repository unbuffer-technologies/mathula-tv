from __future__ import annotations

import mathula_tv.direct_azure_dub as direct
from mathula_tv.cli import _print_direct_dub_progress, parser


def test_progress_tracker_reports_percent_elapsed_and_eta(monkeypatch) -> None:
    clock = iter([100.0, 100.0, 104.0])
    monkeypatch.setattr(direct.time, "monotonic", lambda: next(clock))
    events: list[dict[str, object]] = []
    progress = direct._DirectDubProgress(events.append)

    progress.emit(
        "azure_baseline",
        "block_0001 synthesized",
        current=1,
        total=4,
        block_id="block_0001",
    )
    progress.emit(
        "azure_baseline",
        "block_0002 synthesized",
        current=2,
        total=4,
        block_id="block_0002",
    )

    assert events[0]["schema_version"] == "mathula-direct-dub-progress-v1"
    assert events[0]["percent"] == 25.0
    assert events[1]["elapsed_seconds"] == 4.0
    assert events[1]["percent"] == 50.0
    assert events[1]["eta_seconds"] == 4.0
    assert events[1]["block_id"] == "block_0002"


def test_progress_callback_failure_cannot_break_rendering() -> None:
    def broken_callback(event):
        raise RuntimeError("terminal closed")

    progress = direct._DirectDubProgress(broken_callback)
    progress.emit("prepare", "starting")


def test_cli_progress_is_on_stderr_and_stdout_stays_clean(capsys) -> None:
    _print_direct_dub_progress(
        {
            "stage": "azure_baseline",
            "status": "running",
            "message": "block_0003 synthesized and fitted",
            "elapsed_seconds": 12.0,
            "current": 3,
            "total": 12,
            "percent": 25.0,
            "eta_seconds": 36.0,
        }
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "[dub-azure 00:12]" in captured.err
    assert "azure-baseline 3/12 25%" in captured.err
    assert "ETA 00:36" in captured.err


def test_dub_azure_progress_can_be_disabled() -> None:
    default_args = parser().parse_args(["dub-azure", "job-1"])
    quiet_args = parser().parse_args(["dub-azure", "job-1", "--no-progress"])

    assert default_args.progress is True
    assert quiet_args.progress is False
