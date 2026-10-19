"""Silence padding as the renderer applies it, mirrored as arithmetic."""
from __future__ import annotations

from mathula_tv.dub_padding import plan_turn_padding, tail_pad_ms
from mathula_tv.episode_scoring import _score_group, _turn_rollup
from mathula_tv.dub_workbench import Workbench


def test_tail_pad_closes_only_a_small_shortfall_that_fits_the_free_gap():
    assert tail_pad_ms(120, 500) == 120        # <=150 ms and room before the next speech
    assert tail_pad_ms(150, 150) == 150
    assert tail_pad_ms(151, 900) == 0          # too big for a tail pad
    assert tail_pad_ms(100, 60) == 0           # no room before the next speech
    assert tail_pad_ms(0, 900) == 0


def test_gap_widening_spreads_the_shortfall_over_real_gaps_in_proportion():
    members = [{"start_ms": 0, "end_ms": 4000}, {"start_ms": 4400, "end_ms": 8000}, {"start_ms": 8800, "end_ms": 12000}]
    plan = plan_turn_padding(members, shortfall_ms=1200, last_free_gap_ms=0)   # real gaps 400 + 800
    assert plan["mechanism"] == "gap_widening"
    assert plan["real_gaps_ms"] == [400, 800]
    assert plan["widened_gaps_ms"] == [400, 800]            # scale = 1200/1200 = 1
    assert plan["padded_ms"] == 1200 and plan["residual_ms"] == 0


def test_each_widened_gap_is_capped_so_a_big_shortfall_is_only_partly_absorbed():
    members = [{"start_ms": 0, "end_ms": 4000}, {"start_ms": 4500, "end_ms": 9000}]
    plan = plan_turn_padding(members, shortfall_ms=6000, last_free_gap_ms=0)   # one 500 ms gap, scale 12
    assert plan["widened_gaps_ms"] == [1500]                # capped at _TURN_SILENCE_GAP_CAP_MS
    assert plan["padded_ms"] == 1500 and plan["residual_ms"] == 4500


def test_back_to_back_units_and_single_units_get_no_widening():
    back_to_back = [{"start_ms": 0, "end_ms": 4000}, {"start_ms": 4000, "end_ms": 8000}]
    plan = plan_turn_padding(back_to_back, shortfall_ms=900, last_free_gap_ms=0)
    assert plan["mechanism"] == "none" and plan["padded_ms"] == 0 and plan["residual_ms"] == 900
    single = plan_turn_padding([{"start_ms": 0, "end_ms": 4000}], shortfall_ms=900, last_free_gap_ms=900)
    assert single["residual_ms"] == 900


def test_tail_pad_takes_priority_over_widening():
    members = [{"start_ms": 0, "end_ms": 4000}, {"start_ms": 4400, "end_ms": 8000}]
    plan = plan_turn_padding(members, shortfall_ms=100, last_free_gap_ms=300)
    assert plan["mechanism"] == "tail_pad" and plan["padded_ms"] == 100 and plan["residual_ms"] == 0


def _group(group_id, start, end, measured, turn, free_gap=0, tail_pad=0):
    shortfall = max(0, end - start - measured)
    return _score_group(
        {"group_id": group_id, "turn_id": turn, "start_ms": start, "end_ms": end, "measured_ms": measured,
         "direction": "expand" if shortfall else None, "source_text": "x", "spoken_text": "y",
         "free_gap_ms": free_gap, "tail_pad_ms": tail_pad, "variant_id": "agent"},
        known_tokens=set(), labels=[], rush_flag_pct=10.0,
    )


def test_a_tail_padded_shortfall_is_not_reported_as_dead_air():
    g = _group("u", 0, 10_000, 9_900, "t", free_gap=400, tail_pad=100)
    assert g["shortfall_ms"] == 100 and g["tail_pad_ms"] == 100
    # a large shortfall with no pad is still flagged
    assert "dead_air" in _group("v", 0, 10_000, 7_000, "t2", free_gap=400)["flags"]


def test_turn_rollup_reports_what_padding_absorbs_and_what_remains():
    groups = [_group("a", 0, 4000, 3000, "t"), _group("b", 4400, 8000, 3300, "t")]   # 1.7 s early overall
    turn = _turn_rollup(groups)[0]
    assert turn["shortfall_ms"] == 1700                      # window 8000 - measured 6300
    padding = turn["padding"]
    assert padding["mechanism"] == "gap_widening" and padding["real_gaps_ms"] == [400]
    assert padding["padded_ms"] == 1500 and padding["residual_ms"] == turn["shortfall_ms"] - 1500


def test_workbench_preview_turn_padding_is_free_arithmetic():
    wb = Workbench(store=None, tts_factory=lambda: None)
    plan = wb.preview_turn_padding([
        {"start_ms": 0, "end_ms": 4000, "measured_ms": 3500}, {"start_ms": 4400, "end_ms": 8000, "measured_ms": 3000},
    ])
    assert plan["window_ms"] == 8000 and plan["measured_ms"] == 6500
    assert plan["shortfall_ms"] == 1500 and plan["mechanism"] == "gap_widening"
