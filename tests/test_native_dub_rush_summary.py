from __future__ import annotations

from mathula_tv.cli import _native_dub_rush_rows, _print_native_dub_rush_summary
from mathula_tv.native_dub import REVIEW_REQUIRED_MIN_OVERRUN_MS, REVIEW_REQUIRED_MIN_RUSH_PERCENT


def _group(
    group_id: str, *, speed_percent: float = 0.0, speed_fit_mode: str = "none",
    natural_overrun_ms: int = 0, raw_speed_percent: float = 0.0,
) -> dict:
    return {
        "group_id": group_id,
        "speed_fit": {"speed_percent": speed_percent, "speed_fit_mode": speed_fit_mode},
        "natural_overrun_ms": natural_overrun_ms,
        "raw_speed_percent": raw_speed_percent,
    }


def test_rush_rows_extracts_every_group_including_ones_needing_no_correction() -> None:
    groups = [
        _group("native_phrase_0001"),
        _group("native_phrase_0002", speed_percent=11.5, speed_fit_mode="sentence_end"),
    ]
    rows = _native_dub_rush_rows(groups)
    assert len(rows) == 2
    assert rows[0] == {
        "group_id": "native_phrase_0001", "speed_percent": 0.0,
        "speed_fit_mode": "none", "needs_review": False,
    }
    assert rows[1]["speed_percent"] == 11.5
    assert rows[1]["speed_fit_mode"] == "sentence_end"


def test_rush_rows_defaults_missing_speed_fit_to_zero_percent_none() -> None:
    rows = _native_dub_rush_rows([{"group_id": "native_phrase_0001"}])
    assert rows == [{
        "group_id": "native_phrase_0001", "speed_percent": 0.0,
        "speed_fit_mode": "none", "needs_review": False,
    }]


def test_rush_rows_empty_or_missing_groups_returns_empty_list() -> None:
    assert _native_dub_rush_rows([]) == []
    assert _native_dub_rush_rows(None) == []


# Real bug found on job 57b20602e76a46c08f0f91453e85de0d, 2026-09-10: the
# first version of this flagged mouth_close_sync_ok instead, which also trips
# on plain undershoot (natural speech ending early) -- ALL 22 turns in that
# real run showed "[OUT OF SYNC]" even though every single one fit its window
# comfortably (0 timing repairs, no NEEDS TIMING REVIEW line), because this
# pipeline treats undershoot as fine by design. natural_overrun_ms is 0 for
# any undershoot by construction, so a turn that merely ends early -- however
# much -- must never be flagged here.
def test_undershoot_however_large_is_never_flagged_for_review() -> None:
    groups = [_group("native_phrase_0001", natural_overrun_ms=0, raw_speed_percent=-19.4)]
    rows = _native_dub_rush_rows(groups)
    assert rows[0]["needs_review"] is False


def test_a_real_severe_overrun_is_flagged_for_review() -> None:
    groups = [_group(
        "native_phrase_0025",
        natural_overrun_ms=REVIEW_REQUIRED_MIN_OVERRUN_MS + 1,
        raw_speed_percent=REVIEW_REQUIRED_MIN_RUSH_PERCENT + 1,
    )]
    rows = _native_dub_rush_rows(groups)
    assert rows[0]["needs_review"] is True


def test_an_overrun_below_either_threshold_alone_is_not_flagged() -> None:
    # Mirrors _timing_quality_flags' own "BOTH absolute and relative" rule --
    # neither signal alone is reliable.
    only_absolute = _group(
        "native_phrase_0002",
        natural_overrun_ms=REVIEW_REQUIRED_MIN_OVERRUN_MS + 1,
        raw_speed_percent=1.0,
    )
    only_relative = _group(
        "native_phrase_0003",
        natural_overrun_ms=1,
        raw_speed_percent=REVIEW_REQUIRED_MIN_RUSH_PERCENT + 1,
    )
    rows = _native_dub_rush_rows([only_absolute, only_relative])
    assert rows[0]["needs_review"] is False
    assert rows[1]["needs_review"] is False


def test_print_summary_orders_worst_rush_first_and_covers_every_turn(capsys) -> None:
    groups = [
        _group("native_phrase_0001"),
        _group("native_phrase_0002", speed_percent=11.5, speed_fit_mode="sentence_end"),
        _group("native_phrase_0003", speed_percent=6.3, speed_fit_mode="gap_aware"),
    ]
    _print_native_dub_rush_summary(groups)
    output = capsys.readouterr().out
    lines = [line for line in output.splitlines() if line.strip()]
    assert lines[0] == "Rush by turn (3 total, worst first):"
    # Worst rush (11.5%) listed first, then 6.3%, then the 0.0% turn -- every
    # turn is present, not just the ones that needed correction.
    assert "native_phrase_0002" in lines[1] and "11.5%" in lines[1]
    assert "native_phrase_0003" in lines[2] and "6.3%" in lines[2]
    assert "native_phrase_0001" in lines[3] and "0.0%" in lines[3]


def test_print_summary_flags_a_genuine_severe_overrun(capsys) -> None:
    groups = [_group(
        "native_phrase_0054", speed_percent=6.3, speed_fit_mode="gap_aware",
        natural_overrun_ms=REVIEW_REQUIRED_MIN_OVERRUN_MS + 1,
        raw_speed_percent=REVIEW_REQUIRED_MIN_RUSH_PERCENT + 1,
    )]
    _print_native_dub_rush_summary(groups)
    output = capsys.readouterr().out
    assert "[NEEDS REVIEW]" in output


def test_print_summary_is_a_no_op_for_no_groups(capsys) -> None:
    _print_native_dub_rush_summary([])
    assert capsys.readouterr().out == ""
