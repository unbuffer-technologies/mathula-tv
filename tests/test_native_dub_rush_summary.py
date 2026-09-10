from __future__ import annotations

from mathula_tv.cli import _native_dub_rush_rows, _print_native_dub_rush_summary


def _group(group_id: str, *, speed_percent: float = 0.0, speed_fit_mode: str = "none", sync_ok: bool = True) -> dict:
    return {
        "group_id": group_id,
        "speed_fit": {"speed_percent": speed_percent, "speed_fit_mode": speed_fit_mode},
        "mouth_close_sync_ok": sync_ok,
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
        "speed_fit_mode": "none", "mouth_close_sync_ok": True,
    }
    assert rows[1]["speed_percent"] == 11.5
    assert rows[1]["speed_fit_mode"] == "sentence_end"


def test_rush_rows_defaults_missing_speed_fit_to_zero_percent_none() -> None:
    rows = _native_dub_rush_rows([{"group_id": "native_phrase_0001"}])
    assert rows == [{
        "group_id": "native_phrase_0001", "speed_percent": 0.0,
        "speed_fit_mode": "none", "mouth_close_sync_ok": True,
    }]


def test_rush_rows_empty_or_missing_groups_returns_empty_list() -> None:
    assert _native_dub_rush_rows([]) == []
    assert _native_dub_rush_rows(None) == []


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


def test_print_summary_flags_a_group_still_out_of_sync(capsys) -> None:
    groups = [_group("native_phrase_0054", speed_percent=6.3, speed_fit_mode="gap_aware", sync_ok=False)]
    _print_native_dub_rush_summary(groups)
    output = capsys.readouterr().out
    assert "[OUT OF SYNC]" in output


def test_print_summary_is_a_no_op_for_no_groups(capsys) -> None:
    _print_native_dub_rush_summary([])
    assert capsys.readouterr().out == ""
