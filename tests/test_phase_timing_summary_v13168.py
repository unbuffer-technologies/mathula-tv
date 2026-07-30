from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest

from mathula_tv.cli import (
    _PhaseDurationTracker,
    _print_direct_dub_phase_summary,
)


def _clock(values: list[float]) -> tuple[Callable[[], float], Iterator[float]]:
    iterator = iter(values)
    return lambda: next(iterator), iterator


def test_phase_tracker_measures_nested_phases_against_wall_time() -> None:
    clock, _ = _clock([0, 0, 2, 2, 3, 8, 8, 18, 18, 20])
    tracker = _PhaseDurationTracker(clock=clock)

    tracker.observe({"stage": "prepare", "status": "started"})
    tracker.observe({"stage": "prepare", "status": "completed"})
    tracker.observe({"stage": "publication_render", "status": "started"})
    tracker.observe({"stage": "publication_ai", "status": "started"})
    tracker.observe({"stage": "publication_ai", "status": "completed"})
    tracker.observe({"stage": "publication_ffmpeg", "status": "started"})
    tracker.observe({"stage": "publication_ffmpeg", "status": "completed"})
    tracker.observe({"stage": "publication_render", "status": "completed"})

    summary = tracker.finish()
    phases = {
        phase["stage"]: phase
        for phase in summary["phases"]
    }

    assert summary["total_seconds"] == pytest.approx(20)
    assert phases["prepare"]["duration_seconds"] == pytest.approx(2)
    assert phases["publication_render"]["duration_seconds"] == pytest.approx(16)
    assert phases["publication_ai"]["duration_seconds"] == pytest.approx(5)
    assert phases["publication_ffmpeg"]["duration_seconds"] == pytest.approx(10)
    assert phases["publication_render"]["share_percent"] == pytest.approx(80)


def test_phase_summary_uses_stderr_and_explains_overlapping_rows(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _print_direct_dub_phase_summary(
        {
            "total_seconds": 20,
            "phases": [
                {
                    "stage": "prepare",
                    "duration_seconds": 2,
                    "share_percent": 10,
                },
                {
                    "stage": "publication_render",
                    "duration_seconds": 16,
                    "share_percent": 80,
                },
            ],
        }
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "[dub-azure summary] Phase durations" in captured.err
    assert "publication-render (aggregate)" in captured.err
    assert "TOTAL WALL TIME" in captured.err
    assert "00:20" in captured.err
    assert "nested phases overlap" in captured.err


def test_duplicate_terminal_event_does_not_reopen_a_phase() -> None:
    clock, _ = _clock([0, 0, 4, 9, 10])
    tracker = _PhaseDurationTracker(clock=clock)

    tracker.observe({"stage": "mix", "status": "started"})
    tracker.observe({"stage": "mix", "status": "completed"})
    tracker.observe({"stage": "mix", "status": "completed"})
    summary = tracker.finish()

    assert summary["total_seconds"] == pytest.approx(10)
    assert summary["phases"][0]["duration_seconds"] == pytest.approx(4)


def test_finish_closes_an_active_phase_after_failure() -> None:
    clock, _ = _clock([0, 2, 11])
    tracker = _PhaseDurationTracker(clock=clock)

    tracker.observe({"stage": "publication_ffmpeg", "status": "started"})
    summary = tracker.finish()

    assert summary["total_seconds"] == pytest.approx(11)
    assert summary["phases"][0]["duration_seconds"] == pytest.approx(9)
