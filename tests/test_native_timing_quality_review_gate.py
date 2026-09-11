"""Tests for the timing-quality review gate: the fix for a real systemic gap where
job.review_readiness was driven ONLY by dialogue-overlap flags, so no automated
signal ever caught an extreme required-rush-percent unit (max_natural_speed_percent
is warning-only, never a hard ceiling). Confirmed in production: an 18-sentence
block's rush spiked to 990% (a sentence-correspondence bug -- see
speech_islands.py's _pairs_have_reliable_correspondence) and still reached
review_ready with zero signal, undetected for 3 days.

_timing_quality_flags is deliberately a pure function operating on the SAME
`measured` list _run_batch_adaptive_timing_controller already produces (one entry
per real synthesis unit -- per island where split, per whole group otherwise),
computed BEFORE island restitching so a severe unit buried inside an otherwise-fine
multi-island group is never diluted.
"""
from mathula_tv.native_dub import (
    REVIEW_REQUIRED_MIN_OVERRUN_MS,
    REVIEW_REQUIRED_MIN_RUSH_PERCENT,
    _timing_quality_flags,
)


def _unit(group_id: str, *, natural_overrun_ms: int, raw_speed_percent: float) -> dict:
    return {
        "group_id": group_id,
        "natural_overrun_ms": natural_overrun_ms,
        "raw_speed_percent": raw_speed_percent,
    }


def test_flags_the_real_production_catastrophe():
    # native_phrase_0039__isl12: 39250ms natural vs a 3600ms window.
    measured = [_unit("native_phrase_0039__isl12", natural_overrun_ms=35650, raw_speed_percent=990.3)]
    flags = _timing_quality_flags(
        measured, min_overrun_ms=REVIEW_REQUIRED_MIN_OVERRUN_MS, min_rush_percent=REVIEW_REQUIRED_MIN_RUSH_PERCENT
    )
    assert len(flags) == 1
    assert flags[0]["group_id"] == "native_phrase_0039__isl12"
    assert flags[0]["natural_overrun_ms"] == 35650


def test_does_not_flag_a_short_utterance_inflated_only_by_fixed_azure_overhead():
    # A real benign case from the same job: "No." against a 280ms window -- high
    # PERCENTAGE rush (243.9%) purely from Azure's fixed per-utterance overhead
    # dominating a tiny window, but a small absolute overrun (683ms).
    measured = [_unit("native_phrase_0052", natural_overrun_ms=683, raw_speed_percent=243.9)]
    flags = _timing_quality_flags(
        measured, min_overrun_ms=REVIEW_REQUIRED_MIN_OVERRUN_MS, min_rush_percent=REVIEW_REQUIRED_MIN_RUSH_PERCENT
    )
    assert flags == []


def test_does_not_flag_the_worst_known_benign_case_in_real_data():
    # native_phrase_0044__isl01: a docket number repeated twice, each incurring
    # Azure's fixed overhead against a ~1.1s window -- the worst genuinely benign
    # absolute overrun observed in real job data (4316ms), used to calibrate the
    # 8000ms threshold with wide margin.
    measured = [_unit("native_phrase_0044__isl01", natural_overrun_ms=4316, raw_speed_percent=385.4)]
    flags = _timing_quality_flags(
        measured, min_overrun_ms=REVIEW_REQUIRED_MIN_OVERRUN_MS, min_rush_percent=REVIEW_REQUIRED_MIN_RUSH_PERCENT
    )
    assert flags == []


def test_does_not_flag_a_long_group_with_large_absolute_overrun_but_modest_percentage():
    # A naturally verbose 20s group rendered at 50% over (10s absolute overrun) is
    # well within what this pipeline already tolerates as routine -- the percentage
    # condition exists specifically to avoid flagging this class of case even though
    # its absolute overrun alone would clear the ms threshold.
    measured = [_unit("native_phrase_0100", natural_overrun_ms=10_000, raw_speed_percent=50.0)]
    flags = _timing_quality_flags(
        measured, min_overrun_ms=REVIEW_REQUIRED_MIN_OVERRUN_MS, min_rush_percent=REVIEW_REQUIRED_MIN_RUSH_PERCENT
    )
    assert flags == []


def test_requires_both_conditions_not_either_alone():
    # High overrun, but at/under the rush threshold.
    only_overrun = [_unit("g1", natural_overrun_ms=9_000, raw_speed_percent=40.0)]
    # High rush, but at/under the overrun threshold.
    only_rush = [_unit("g2", natural_overrun_ms=2_000, raw_speed_percent=200.0)]
    assert _timing_quality_flags(only_overrun, min_overrun_ms=8_000, min_rush_percent=50.0) == []
    assert _timing_quality_flags(only_rush, min_overrun_ms=8_000, min_rush_percent=50.0) == []


def test_flags_multiple_units_and_reports_each():
    measured = [
        _unit("g1", natural_overrun_ms=683, raw_speed_percent=243.9),  # benign
        _unit("g2", natural_overrun_ms=35_650, raw_speed_percent=990.3),  # bad
        _unit("g3", natural_overrun_ms=12_000, raw_speed_percent=80.0),  # bad
    ]
    flags = _timing_quality_flags(measured, min_overrun_ms=8_000, min_rush_percent=50.0)
    assert {f["group_id"] for f in flags} == {"g2", "g3"}


def test_handles_missing_fields_defensively():
    assert _timing_quality_flags([{"group_id": "g1"}], min_overrun_ms=8_000, min_rush_percent=50.0) == []


def test_empty_measured_list_produces_no_flags():
    assert _timing_quality_flags([], min_overrun_ms=8_000, min_rush_percent=50.0) == []
