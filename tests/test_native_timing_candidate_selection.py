import pytest

from mathula_tv.native_dub import _timing_candidate_rank, _timing_recast_direction


def _rank(measured, window, target, early=40, late=40):
    return _timing_candidate_rank(
        measured,
        window,
        target,
        max_speed_percent=12,
        mouth_close_early_tolerance_ms=early,
        mouth_close_late_tolerance_ms=late,
    )


def test_safe_early_recast_beats_unrescuable_late_candidate():
    # Real native_phrase_0001 geometry from the production log:
    # canonical 49.250s is fatally late; round-1 recast 29.125s is early.
    assert _rank(29_125, 38_640, 40_958) < _rank(49_250, 38_640, 40_958)


def test_closer_early_candidate_beats_more_extreme_early_candidate():
    # Once a late phrase has crossed into the safe-early class, a later expand
    # pass must prefer the candidate closer to the source mouth-close.
    assert _rank(29_125, 38_640, 40_958) < _rank(23_886, 38_640, 40_958)


def test_zero_gap_safe_early_beats_unrescuable_late():
    # native_phrase_0002 has zero late tolerance. The early recast must still
    # outrank the original late candidate because early silence is containable.
    assert _rank(34_493, 37_760, 40_026, late=0) < _rank(42_969, 37_760, 40_026, late=0)


def test_feasible_speed_fit_beats_early_fallback():
    feasible = 40_000
    early = 34_000
    assert _timing_recast_direction(
        feasible, 38_640, max_speed_percent=12,
        mouth_close_early_tolerance_ms=40, mouth_close_late_tolerance_ms=40,
    ) is None
    assert _rank(feasible, 38_640, 40_958) < _rank(early, 38_640, 40_958)


# --- gap-aware compress ranking ------------------------------------------------------
# _timing_recast_direction already used window + late_tolerance internally to decide
# WHETHER a candidate needs compression at all. The rank's own required-rush value used
# to ignore that extension and always compute against the bare window -- meaning a real,
# already-safe gap before the next sentence reduced whether compression was needed, but
# never reduced the reported/ranked severity of the compression once it kicked in. This
# is the ranking half of the two speed-fit modes (the render half is
# _speed_fit_mode_label / the _speed_fit_overflow_wav call sites).

def test_compress_rank_uses_the_gap_aware_target_not_the_bare_window():
    no_gap = _rank(1500, 1000, 1000, late=0)
    with_gap = _rank(1500, 1000, 1000, late=100)
    assert no_gap[0] == 1 and with_gap[0] == 1  # both still classified "compress"
    assert with_gap[1] < no_gap[1]  # gap-aware ranking reports a smaller required rush
    assert with_gap[1] == pytest.approx(36_364, abs=5)
    assert no_gap[1] == 50_000


def test_compress_rank_with_zero_late_tolerance_matches_the_bare_window_exactly():
    # Regression guard: no usable gap must behave identically to before this feature --
    # window + 0 == window.
    from mathula_tv.native_dub import _temporal_required_rush_percent

    rank = _rank(1500, 1000, 1000, late=0)
    expected_milli = int(round(_temporal_required_rush_percent(1500, 1000) * 1000.0))
    assert rank[1] == expected_milli
