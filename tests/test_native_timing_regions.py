"""Phase 9/13 of the --candidate-pool architecture (see the approved plan at
C:\\Users\\mokgethwa\\.claude\\plans\\calm-wiggling-parasol.md): speaker turns
are the PRIMARY timing unit -- not a secondary rebalancing fallback on top of
per-sentence candidates (that was Phase 8's --region-rebalance, now removed).
A multi-member turn is judged on whether its total stitched Zulu duration
starts and ends in sync with its combined English span; a solo "bridge"
sentence (a short interjection like "All right.") is judged only on whether
it collides with its neighbors.

Phase 9/10's "discourse segment" grouping capped combined turn duration at 30s,
sentence count at 8, and force-broke around any bridge sentence even mid-turn
for the SAME speaker -- three artificial fragmentation rules unrelated to
speaker or timing continuity. Phase 13 removes all three: a speaker turn is
now genuinely UNBOUNDED, and a bridge sentence merges into a turn like any
other sentence when speaker continuity holds.

Phase 10's boundary-shift mechanism (moving whole ALREADY-TRANSLATED words
across adjacent member boundaries via a bookmarked Speech SDK call) had a
confirmed, unfixed defect: it could strand a word that needed a following
grammatical complement (e.g. a locative pulled onto the end of an unrelated
sentence with nothing to attach to). Phase 13 replaces it entirely with a
pure, text-invariant mechanism: resizing each member's window share
proportionally to its own real, already-measured content duration (see
_reallocate_turn_windows) -- no word ever moves, no new TTS/Speech-SDK call
is needed, and the confirmed grammar bug cannot recur by construction.
"""
from __future__ import annotations

import wave
from pathlib import Path

from mathula_tv.native_dub import (
    DEFAULT_MAX_NATURAL_SPEED_PERCENT,
    SPEAKER_TURN_MAX_GAP_MS,
    _BRIDGE_MAX_SOURCE_SPAN_MS,
    _BRIDGE_MAX_WORD_COUNT,
    _build_speaker_turns,
    _classify_discourse_unit_kind,
    _measure_speaker_turn_total,
    _reallocate_turn_windows,
    _rebalance_speaker_turns,
    _resync_turn_rush_aggregates,
    _score_speaker_turn,
    _speaker_turn_geometry,
    _temporal_required_rush_percent,
)


def _real_required_speed(measured_ms: int, *, span_ms: int = 3000, late_tolerance_ms: int = 40) -> float:
    """Compute the SAME required_speed_percent _measure_temporal_candidate would
    for a per-sentence winner fixture, so hand-built test fixtures stay
    internally consistent with what the real measurement path produces
    (avoids a hand-picked round number silently drifting from the real formula).
    """
    return round(_temporal_required_rush_percent(measured_ms, span_ms + late_tolerance_ms), 3)


def _group(group_id: str, *, speaker_id: str = "S0", start_ms: int, end_ms: int, source_text: str = "x") -> dict:
    return {
        "group_id": group_id, "speaker_id": speaker_id, "start_ms": start_ms,
        "source_end_ms": end_ms, "source_span_ms": end_ms - start_ms, "source_text": source_text,
    }


# --- _classify_discourse_unit_kind ----------------------------------------------------


def test_classify_bridge_when_short_words_and_short_span():
    group = _group("g1", start_ms=0, end_ms=900, source_text="All right.")
    assert _classify_discourse_unit_kind(group) == "bridge"


def test_classify_normal_when_words_exceed_the_bridge_threshold():
    group = _group("g1", start_ms=0, end_ms=900, source_text="This has more than four words in it.")
    assert _classify_discourse_unit_kind(group) == "normal"


def test_classify_normal_when_span_exceeds_the_bridge_threshold_even_with_few_words():
    group = _group("g1", start_ms=0, end_ms=_BRIDGE_MAX_SOURCE_SPAN_MS + 500, source_text="All right.")
    assert _classify_discourse_unit_kind(group) == "normal"


def test_classify_bridge_boundary_is_inclusive():
    group = _group(
        "g1", start_ms=0, end_ms=_BRIDGE_MAX_SOURCE_SPAN_MS,
        source_text=" ".join(["w"] * _BRIDGE_MAX_WORD_COUNT),
    )
    assert _classify_discourse_unit_kind(group) == "bridge"


# --- _build_speaker_turns ---------------------------------------------------------------


def test_adjacent_same_speaker_normal_sentences_with_small_gap_merge_into_one_turn():
    groups = [
        _group("g1", start_ms=0, end_ms=3000, source_text="This is a normal sentence here."),
        _group("g2", start_ms=3200, end_ms=6000, source_text="Another normal sentence follows."),
    ]
    turns = _build_speaker_turns(groups)
    assert len(turns) == 1
    assert turns[0]["member_group_ids"] == ["g1", "g2"]
    assert turns[0]["start_ms"] == 0
    assert turns[0]["source_end_ms"] == 6000


def test_speaker_change_breaks_the_turn():
    groups = [
        _group("g1", speaker_id="S0", start_ms=0, end_ms=3000, source_text="Normal sentence number one here."),
        _group("g2", speaker_id="S1", start_ms=3200, end_ms=6000, source_text="Normal sentence number two here."),
    ]
    turns = _build_speaker_turns(groups)
    assert [t["member_group_ids"] for t in turns] == [["g1"], ["g2"]]


def test_gap_over_the_threshold_breaks_the_turn():
    g2_start = 3000 + SPEAKER_TURN_MAX_GAP_MS + 1
    groups = [
        _group("g1", start_ms=0, end_ms=3000, source_text="Normal sentence number one here."),
        _group("g2", start_ms=g2_start, end_ms=g2_start + 3000, source_text="Normal sentence number two here."),
    ]
    turns = _build_speaker_turns(groups)
    assert [t["member_group_ids"] for t in turns] == [["g1"], ["g2"]]


def test_gap_at_exactly_the_threshold_still_merges():
    g2_start = 3000 + SPEAKER_TURN_MAX_GAP_MS
    groups = [
        _group("g1", start_ms=0, end_ms=3000, source_text="Normal sentence number one here."),
        _group("g2", start_ms=g2_start, end_ms=g2_start + 3000, source_text="Normal sentence number two here."),
    ]
    turns = _build_speaker_turns(groups)
    assert [t["member_group_ids"] for t in turns] == [["g1", "g2"]]


def test_a_lone_normal_sentence_with_no_eligible_neighbors_is_its_own_turn():
    groups = [_group("g1", start_ms=0, end_ms=3000, source_text="Normal sentence number one here.")]
    turns = _build_speaker_turns(groups)
    assert turns == [{"member_group_ids": ["g1"], "start_ms": 0, "source_end_ms": 3000}]


def test_a_bridge_sentence_now_merges_into_a_turn_with_the_same_speaker():
    """Confirmed real case (native_phrase_0060, "He served this country.") --
    a bridge-classified sentence embedded in a same-speaker emphatic run used
    to be force-isolated into its own segment purely by word count, even
    though nothing about speaker or timing continuity called for a break.
    Phase 13 drops that rule: a bridge merges exactly like any other sentence.
    """
    groups = [
        _group("g1", start_ms=0, end_ms=3000, source_text="Normal sentence number one right here."),
        _group("g2", start_ms=3100, end_ms=4000, source_text="All right."),  # bridge, same speaker
        _group("g3", start_ms=4100, end_ms=7000, source_text="Normal sentence number three right here."),
    ]
    turns = _build_speaker_turns(groups)
    assert [t["member_group_ids"] for t in turns] == [["g1", "g2", "g3"]]


def test_a_bridge_sentence_from_a_different_speaker_still_stands_alone():
    groups = [
        _group("g1", speaker_id="S0", start_ms=0, end_ms=3000, source_text="Normal sentence right here again."),
        _group("g2", speaker_id="S1", start_ms=3100, end_ms=4000, source_text="All right."),
    ]
    turns = _build_speaker_turns(groups)
    assert [t["member_group_ids"] for t in turns] == [["g1"], ["g2"]]


def test_every_input_group_is_covered_exactly_once_across_mixed_boundaries():
    groups = [
        _group("g1", speaker_id="A", start_ms=0, end_ms=3000, source_text="Normal sentence number one here."),
        _group("g2", speaker_id="A", start_ms=3100, end_ms=6000, source_text="Normal sentence number two here."),
        _group("g3", speaker_id="B", start_ms=6100, end_ms=9000, source_text="Normal sentence number three here."),
        _group("g4", speaker_id="A", start_ms=20000, end_ms=23000, source_text="Normal sentence number four here."),  # big gap from g3
    ]
    turns = _build_speaker_turns(groups)
    all_ids = [gid for t in turns for gid in t["member_group_ids"]]
    assert all_ids == ["g1", "g2", "g3", "g4"]
    assert [t["member_group_ids"] for t in turns] == [["g1", "g2"], ["g3"], ["g4"]]


def test_a_long_uninterrupted_same_speaker_run_is_one_turn_with_no_cap():
    """Confirmed real fragmentation this phase removes: the old discourse-
    segment grouping force-broke after 8 sentences or 30s combined duration
    even with zero gap between them. A genuine speaker turn has no such cap.
    """
    groups = [
        _group(f"g{i}", start_ms=i * 4000, end_ms=i * 4000 + 3900, source_text=f"Normal sentence number {i} here.")
        for i in range(12)
    ]
    turns = _build_speaker_turns(groups)
    assert len(turns) == 1
    assert len(turns[0]["member_group_ids"]) == 12


# --- _speaker_turn_geometry -------------------------------------------------------------


def test_speaker_turn_geometry_spans_from_first_start_to_last_end():
    groups = [
        _group("g1", start_ms=1000, end_ms=4000, source_text="Normal sentence number one here."),
        _group("g2", start_ms=4200, end_ms=7000, source_text="Normal sentence number two here."),
    ]
    groups_by_id = {g["group_id"]: g for g in groups}
    turn = {"member_group_ids": ["g1", "g2"], "start_ms": 1000, "source_end_ms": 7000}
    geometry = _speaker_turn_geometry(
        turn, groups_by_id=groups_by_id, next_source_start_ms=8000, preferred_raw_speed_percent=6,
    )
    assert geometry["preferred_start_ms"] == 1000
    assert geometry["preferred_end_ms"] == 7000
    assert geometry["source_window_ms"] == 6000


# --- _measure_speaker_turn_total ---------------------------------------------------------


def _write_wav(path: Path, *, seconds: float, framerate: int = 16000) -> None:
    nframes = int(seconds * framerate)
    with wave.open(str(path), "wb") as handle:
        handle.setparams((1, 2, framerate, 0, "NONE", "not compressed"))
        handle.writeframes(b"\x00\x01" * nframes)


def test_measure_speaker_turn_total_stitches_members_with_the_calibrated_azure_pause(tmp_path):
    # Real, repeatedly measured (2026-09-05): Azure's own bookmarked one-shot
    # synthesis pays its OWN sentence-boundary pause (AZURE_SENTENCE_BOUNDARY_PAUSE_MS),
    # not the original English speaker's real ASR gap -- confirmed to be a real,
    # ~1000-1050ms constant independent of the source's own gap length, which is
    # why this test's two members use REAL ASR gaps very different from 1000ms
    # (200ms here) yet the stitched result must still reflect the calibrated pause.
    from mathula_tv.native_dub import AZURE_SENTENCE_BOUNDARY_PAUSE_MS

    groups = [
        _group("g1", start_ms=0, end_ms=3000),
        _group("g2", start_ms=3200, end_ms=6000),  # 200ms real ASR gap -- irrelevant now
    ]
    wav1, wav2 = tmp_path / "g1.wav", tmp_path / "g2.wav"
    _write_wav(wav1, seconds=1.0)
    _write_wav(wav2, seconds=1.5)
    member_winners = {
        "g1": {"path": str(wav1)},
        "g2": {"path": str(wav2)},
    }
    turn = {"member_group_ids": ["g1", "g2"], "start_ms": 0, "source_end_ms": 6000}
    result = _measure_speaker_turn_total(
        segment=turn, member_winners=member_winners, output_root=tmp_path,
    )
    assert result["final_duration_ms"] == 1000 + AZURE_SENTENCE_BOUNDARY_PAUSE_MS + 1500


def test_measure_speaker_turn_total_keeps_the_output_filename_bounded_for_a_large_turn(tmp_path):
    """Confirmed real bug on job 33cd7b46...: joining every member id into the
    stitched-output filename (safe when turns were capped at 8 members) blew
    past Windows' 255-character filename component limit for a genuine,
    uncapped 15-member turn, crashing with [Errno 22] Invalid argument.
    """
    member_count = 20
    groups = [
        _group(f"native_phrase_{i:04d}", start_ms=i * 1000, end_ms=i * 1000 + 900)
        for i in range(member_count)
    ]
    member_winners = {}
    for group in groups:
        wav_path = tmp_path / f"{group['group_id']}.wav"
        _write_wav(wav_path, seconds=0.5)
        member_winners[group["group_id"]] = {"path": str(wav_path)}
    turn = {
        "member_group_ids": [g["group_id"] for g in groups],
        "start_ms": groups[0]["start_ms"], "source_end_ms": groups[-1]["source_end_ms"],
    }
    result = _measure_speaker_turn_total(
        segment=turn, member_winners=member_winners, output_root=tmp_path,
    )
    assert result["final_duration_ms"] > 0


def test_measure_speaker_turn_total_still_floors_at_the_minimum_protected_pause(tmp_path):
    # The gap is now a fixed max(DEFAULT_MIN_PROTECTED_PAUSE_MS, AZURE_SENTENCE_BOUNDARY_PAUSE_MS)
    # regardless of the members' real ASR timing (confirmed irrelevant now -- these two
    # groups overlap in real ASR time, which used to floor to DEFAULT_MIN_PROTECTED_PAUSE_MS
    # under the old model; the floor logic itself survives, just against a different pair
    # of constants).
    from mathula_tv.native_dub import AZURE_SENTENCE_BOUNDARY_PAUSE_MS, DEFAULT_MIN_PROTECTED_PAUSE_MS

    groups = [
        _group("g1", start_ms=0, end_ms=3000),
        _group("g2", start_ms=2900, end_ms=6000),  # overlapping ASR timestamps -- irrelevant now
    ]
    wav1, wav2 = tmp_path / "g1.wav", tmp_path / "g2.wav"
    _write_wav(wav1, seconds=1.0)
    _write_wav(wav2, seconds=1.0)
    member_winners = {"g1": {"path": str(wav1)}, "g2": {"path": str(wav2)}}
    turn = {"member_group_ids": ["g1", "g2"], "start_ms": 0, "source_end_ms": 6000}
    result = _measure_speaker_turn_total(
        segment=turn, member_winners=member_winners, output_root=tmp_path,
    )
    expected_gap = max(DEFAULT_MIN_PROTECTED_PAUSE_MS, AZURE_SENTENCE_BOUNDARY_PAUSE_MS)
    assert result["final_duration_ms"] == 1000 + expected_gap + 1000


# --- _reallocate_turn_windows -------------------------------------------------------------


def test_reallocate_gives_more_window_to_the_member_with_more_real_content():
    turn = {"member_group_ids": ["g1", "g2"]}
    groups_by_id = {
        "g1": _group("g1", start_ms=0, end_ms=3000),
        "g2": _group("g2", start_ms=3000, end_ms=6000),
    }
    winners = {
        "g1": {"measured_ms": 2000.0},  # less real content
        "g2": {"measured_ms": 4000.0},  # more real content
    }
    result = _reallocate_turn_windows(turn=turn, groups_by_id=groups_by_id, winners=winners)
    # No spoken_text field -- text is never touched by this mechanism.
    assert "spoken_text" not in result["g1"]
    assert "spoken_text" not in result["g2"]
    assert result["g1"]["source_span_ms"] + result["g2"]["source_span_ms"] == 6000
    assert result["g1"]["source_span_ms"] < result["g2"]["source_span_ms"]
    assert result["g1"]["start_ms"] == 0
    assert result["g2"]["source_end_ms"] == 6000


def test_reallocate_splits_evenly_when_members_have_equal_content():
    turn = {"member_group_ids": ["g1", "g2"]}
    groups_by_id = {
        "g1": _group("g1", start_ms=0, end_ms=3000),
        "g2": _group("g2", start_ms=3000, end_ms=6000),
    }
    winners = {"g1": {"measured_ms": 3000.0}, "g2": {"measured_ms": 3000.0}}
    result = _reallocate_turn_windows(turn=turn, groups_by_id=groups_by_id, winners=winners)
    assert result["g1"]["source_span_ms"] == result["g2"]["source_span_ms"] == 3000
    assert result["g1"]["start_ms"] == 0
    assert result["g1"]["source_end_ms"] == 3000
    assert result["g2"]["start_ms"] == 3000
    assert result["g2"]["source_end_ms"] == 6000


def test_reallocate_keeps_the_turns_fixed_outer_bounds_regardless_of_share_split(tmp_path):
    turn = {"member_group_ids": ["g1", "g2", "g3"]}
    groups_by_id = {
        "g1": _group("g1", start_ms=1000, end_ms=4000),
        "g2": _group("g2", start_ms=4000, end_ms=7000),
        "g3": _group("g3", start_ms=7000, end_ms=10000),
    }
    winners = {
        "g1": {"measured_ms": 1000.0},
        "g2": {"measured_ms": 5000.0},
        "g3": {"measured_ms": 500.0},
    }
    result = _reallocate_turn_windows(turn=turn, groups_by_id=groups_by_id, winners=winners)
    assert result["g1"]["start_ms"] == 1000
    assert result["g3"]["source_end_ms"] == 10000
    total = sum(result[gid]["source_span_ms"] for gid in ("g1", "g2", "g3"))
    assert total == 9000


# --- _rebalance_speaker_turns (full orchestration) ---------------------------------------


def _turn_group(group_id, *, speaker_id="S0", start_ms, span_ms=3000, source_text="Normal sentence right here."):
    return {
        "group_id": group_id, "speaker_id": speaker_id, "start_ms": start_ms,
        "source_end_ms": start_ms + span_ms, "source_span_ms": span_ms, "source_text": source_text,
        "segment_ids": [f"{group_id}_seg"],
    }


def _turn_geometry(group):
    return {
        "preferred_start_ms": group["start_ms"], "preferred_end_ms": group["source_end_ms"],
        "hard_earliest_start_ms": group["start_ms"], "hard_latest_end_ms": group["source_end_ms"] + 40,
        "source_window_ms": group["source_span_ms"], "preferred_raw_ms": int(group["source_span_ms"] * 1.06),
        "minimum_usable_raw_ms": max(1, group["source_span_ms"] - 40),
        "maximum_usable_raw_ms": int((group["source_span_ms"] + 40) * 1.12),
        "mouth_close_early_tolerance_ms": 40, "mouth_close_late_tolerance_ms": 40, "free_gap_ms": 0,
    }


def _write_real_wav(path: Path, *, ms: int, framerate: int = 16000) -> None:
    nframes = int(ms * framerate / 1000)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setparams((1, 2, framerate, 0, "NONE", "not compressed"))
        handle.writeframes(b"\x00\x01" * nframes)


def test_rebalance_skips_a_turn_that_already_fits_and_paces_evenly(tmp_path):
    # Durations chosen to land in the narrow band that is neither "expand"
    # (stitched total well under the combined window -- _timing_recast_direction
    # treats that as undersized, not "fit") nor "compress" (over the max natural
    # speed-up): close to fully using the combined window, same as a real natural
    # translation usually does. g2 starts 4000ms after g1 (not the members' own
    # real ASR gap, which no longer matters -- see AZURE_SENTENCE_BOUNDARY_PAUSE_MS)
    # so the turn's own fixed window (7100ms) has room for the calibrated 1000ms
    # inter-sentence pause _measure_speaker_turn_total now always uses.
    groups = [_turn_group("g1", start_ms=0), _turn_group("g2", start_ms=4000, span_ms=3100)]
    wav1, wav2 = tmp_path / "g1.wav", tmp_path / "g2.wav"
    _write_real_wav(wav1, ms=3000)
    _write_real_wav(wav2, ms=3100)
    winners = {
        "g1": {"candidate_id": "grok_natural", "spoken_text": "sawubona kakhulu", "measured_ms": 3000,
               "required_speed_percent": 0.0, "qa_penalty": 0, "path": str(wav1)},
        "g2": {"candidate_id": "grok_natural", "spoken_text": "sawubona futhi", "measured_ms": 3100,
               "required_speed_percent": 0.0, "qa_penalty": 0, "path": str(wav2)},
    }
    updated_winners, updated_geometries = _rebalance_speaker_turns(
        groups=groups, winners=winners,
        geometries={"g1": _turn_geometry(groups[0]), "g2": _turn_geometry(groups[1])},
        preferred_raw_speed_percent=6, measurement_audio_root=tmp_path,
    )
    # Text/candidate identity/window are never touched by this mechanism --
    # but required_speed_percent/speed_fit_mode/fit ARE always overwritten to
    # the turn's real, honest aggregate for a non-bridge turn (real user
    # feedback, 2026-09-07: "why are we aggregating segments rushes, the turn
    # is suppose to be the unit" -- each member's own individually-split
    # number never corresponded to what a turn-finalized render actually
    # produces, so it's corrected here even when nothing else changes).
    for group_id in ("g1", "g2"):
        assert updated_winners[group_id]["spoken_text"] == winners[group_id]["spoken_text"]
        assert updated_winners[group_id]["candidate_id"] == winners[group_id]["candidate_id"]
        assert "effective_start_ms" not in updated_winners[group_id]
    assert updated_winners["g1"]["required_speed_percent"] == updated_winners["g2"]["required_speed_percent"]
    assert updated_winners["g1"]["fit"] is True


def test_rebalance_skips_single_member_turns():
    groups = [_turn_group("g1", start_ms=0, source_text="All right.", span_ms=900)]
    winners = {"g1": {"candidate_id": "grok_natural", "spoken_text": "kulungile", "measured_ms": 900,
                       "required_speed_percent": 0.0, "qa_penalty": 0, "path": "/x.wav"}}
    updated_winners, updated_geometries = _rebalance_speaker_turns(
        groups=groups, winners=winners, geometries={"g1": _turn_geometry(groups[0])},
        preferred_raw_speed_percent=6, measurement_audio_root=Path("."),
    )
    assert updated_winners == winners


def test_rebalance_triggers_and_promotes_a_real_improvement_via_window_reallocation(tmp_path):
    """g1 has real slack while g2 is overloaded -- resizing each member's window
    share proportionally to its own real content resolves g2's overrun without
    any wording, translation, or candidate_id ever changing.
    """
    groups = [
        _turn_group("g1", start_ms=0, span_ms=3000, source_text="Normal sentence with slack right here."),
        _turn_group("g2", start_ms=3100, span_ms=3000, source_text="Normal sentence that is overloaded here."),
    ]
    wav1, wav2 = tmp_path / "g1.wav", tmp_path / "g2.wav"
    _write_real_wav(wav1, ms=2000)
    _write_real_wav(wav2, ms=4600)
    winners = {
        "g1": {"candidate_id": "grok_natural", "spoken_text": "sawubona bakithi manje", "measured_ms": 2000,
               "required_speed_percent": _real_required_speed(2000), "qa_penalty": 0, "path": str(wav1)},
        "g2": {"candidate_id": "grok_natural", "spoken_text": "kwenzekile kabi kakhulu", "measured_ms": 4600,
               "required_speed_percent": _real_required_speed(4600), "qa_penalty": 0, "path": str(wav2)},
    }
    updated_winners, updated_geometries = _rebalance_speaker_turns(
        groups=groups, winners=winners,
        geometries={"g1": _turn_geometry(groups[0]), "g2": _turn_geometry(groups[1])},
        preferred_raw_speed_percent=6, measurement_audio_root=tmp_path,
    )
    # Text and candidate identity are never touched by this mechanism.
    assert updated_winners["g1"]["spoken_text"] == "sawubona bakithi manje"
    assert updated_winners["g2"]["spoken_text"] == "kwenzekile kabi kakhulu"
    assert updated_winners["g1"]["candidate_id"] == "grok_natural"
    assert updated_winners["g2"]["candidate_id"] == "grok_natural"
    # g2 (more real content) ends up with a larger share of the fixed combined
    # window than g1 (less content), and the effective override is present.
    assert "effective_start_ms" in updated_winners["g1"]
    assert "effective_source_end_ms" in updated_winners["g2"]
    g1_span = updated_winners["g1"]["effective_source_end_ms"] - updated_winners["g1"]["effective_start_ms"]
    g2_span = updated_winners["g2"]["effective_source_end_ms"] - updated_winners["g2"]["effective_start_ms"]
    assert g2_span > g1_span
    # Real user feedback, 2026-09-07 ("why are we aggregating segments
    # rushes, the turn is suppose to be the unit"): window reallocation only
    # ever redistributes SPREAD between members -- the turn's real, stitched
    # aggregate required_speed_percent is invariant under it (same content,
    # same fixed outer window), so it is no longer refreshed per-member here;
    # both members report the SAME honest turn-level number, which is what a
    # turn-finalized render actually applies as one shared speed factor.
    assert updated_winners["g1"]["required_speed_percent"] == updated_winners["g2"]["required_speed_percent"]
    # The aggregate genuinely doesn't fit here (that's why rebalancing
    # triggered at all) -- reallocation only ever improves spread, never
    # turns a non-fitting aggregate into a fitting one.
    assert updated_winners["g2"]["fit"] is False


def test_rebalance_keeps_the_original_when_reallocation_does_not_improve(tmp_path):
    """Both members have equal real content and equal original windows, so
    reallocating reproduces the identical (already-even) split -- the turn
    doesn't fit overall (both genuinely overrun), and no redistribution of a
    fixed budget between two equally-loaded members can ever improve the
    SPREAD (already zero). Text, candidate identity, and window are left
    untouched -- but required_speed_percent is still corrected to the turn's
    real, honest aggregate (invariant under reallocation either way), which
    is what actually gets rendered for a non-bridge turn, even though this
    specific turn never triggers an actual reallocation.
    """
    groups = [
        _turn_group("g1", start_ms=0, span_ms=3000, source_text="Overloaded sentence number one right here."),
        _turn_group("g2", start_ms=3100, span_ms=3000, source_text="Overloaded sentence number two right here."),
    ]
    wav1, wav2 = tmp_path / "g1.wav", tmp_path / "g2.wav"
    _write_real_wav(wav1, ms=4000)
    _write_real_wav(wav2, ms=4000)
    winners = {
        "g1": {"candidate_id": "grok_natural", "spoken_text": "g1 zulu overloaded", "measured_ms": 4000,
               "required_speed_percent": _real_required_speed(4000), "qa_penalty": 0, "path": str(wav1)},
        "g2": {"candidate_id": "grok_natural", "spoken_text": "g2 zulu overloaded", "measured_ms": 4000,
               "required_speed_percent": _real_required_speed(4000), "qa_penalty": 0, "path": str(wav2)},
    }
    updated_winners, updated_geometries = _rebalance_speaker_turns(
        groups=groups, winners=winners,
        geometries={"g1": _turn_geometry(groups[0]), "g2": _turn_geometry(groups[1])},
        preferred_raw_speed_percent=6, measurement_audio_root=tmp_path,
    )
    for group_id in ("g1", "g2"):
        assert updated_winners[group_id]["spoken_text"] == winners[group_id]["spoken_text"]
        assert updated_winners[group_id]["candidate_id"] == winners[group_id]["candidate_id"]
        assert "effective_start_ms" not in updated_winners[group_id]
    # Both members share the same honest aggregate, which does not equal
    # either member's original (now-known-fictional) individual number.
    assert updated_winners["g1"]["required_speed_percent"] == updated_winners["g2"]["required_speed_percent"]
    assert updated_winners["g1"]["required_speed_percent"] != winners["g1"]["required_speed_percent"]


# --- _resync_turn_rush_aggregates -------------------------------------------------------
# Real gap found on job fb3d08b63fed4d90922b08f7e325b906, 2026-09-07: after
# _rebalance_speaker_turns sets a turn's honest shared aggregate on every
# member, a LATER stage (_trim_by_fact_priority) can shorten just ONE
# member's own content -- leaving that member with a fresh, lower individual
# number while its turn partner still carries the stale pre-trim aggregate.
# _resync_turn_rush_aggregates re-syncs both to the new, real aggregate.


def test_resync_recomputes_the_aggregate_after_one_members_content_shrinks(tmp_path):
    groups = [
        _turn_group("g1", start_ms=0, span_ms=3000, source_text="Normal sentence right here today."),
        _turn_group("g2", start_ms=3100, span_ms=3000, source_text="Normal sentence right there today."),
    ]
    wav1_before, wav2 = tmp_path / "g1_before.wav", tmp_path / "g2.wav"
    _write_real_wav(wav1_before, ms=4600)
    _write_real_wav(wav2, ms=4600)
    stale_winners = {
        "g1": {"candidate_id": "grok_natural", "spoken_text": "sawubona kakhulu manje namuhla",
               "measured_ms": 4600, "required_speed_percent": 40.0, "speed_fit_mode": "gap_aware",
               "fit": False, "qa_penalty": 0, "path": str(wav1_before)},
        "g2": {"candidate_id": "grok_natural", "spoken_text": "sawubona lapho manje namuhla",
               "measured_ms": 4600, "required_speed_percent": 40.0, "speed_fit_mode": "gap_aware",
               "fit": False, "qa_penalty": 0, "path": str(wav2)},
    }
    geometries = {"g1": _turn_geometry(groups[0]), "g2": _turn_geometry(groups[1])}

    # Simulate _trim_by_fact_priority promoting a genuinely shorter candidate
    # for g1 only -- g2 is untouched, exactly like a real trim that only
    # finds a safe droppable span in one of the two sentences.
    wav1_trimmed = tmp_path / "g1_trimmed.wav"
    _write_real_wav(wav1_trimmed, ms=2200)
    trimmed_winners = dict(stale_winners)
    trimmed_winners["g1"] = {
        **stale_winners["g1"], "spoken_text": "sawubona kakhulu",
        "measured_ms": 2200, "required_speed_percent": 12.0, "path": str(wav1_trimmed),
    }

    updated_winners = _resync_turn_rush_aggregates(
        groups=groups, winners=trimmed_winners, geometries=geometries,
        preferred_raw_speed_percent=6, measurement_audio_root=tmp_path,
    )

    # Both members now report the SAME real aggregate...
    assert updated_winners["g1"]["required_speed_percent"] == updated_winners["g2"]["required_speed_percent"]
    # ...and it reflects g1's real shrinkage: strictly lower than the stale
    # pre-trim aggregate both members previously shared.
    assert updated_winners["g1"]["required_speed_percent"] < 40.0
    assert updated_winners["g1"]["speed_fit_mode"] == updated_winners["g2"]["speed_fit_mode"]
    assert updated_winners["g1"]["fit"] == updated_winners["g2"]["fit"]
    # Text/candidate identity from the trim are preserved -- this function
    # only ever touches the rush-reporting fields.
    assert updated_winners["g1"]["spoken_text"] == "sawubona kakhulu"
    assert updated_winners["g1"]["measured_ms"] == 2200


def test_resync_leaves_a_bridge_containing_turn_untouched(tmp_path):
    groups = [
        _turn_group("g1", start_ms=0, span_ms=3000, source_text="Normal sentence right here today."),
        _turn_group("g2", start_ms=3100, span_ms=900, source_text="All right."),
    ]
    winners = {
        "g1": {"candidate_id": "grok_natural", "spoken_text": "sawubona", "measured_ms": 3000,
               "required_speed_percent": 15.0, "qa_penalty": 0, "path": "/g1.wav"},
        "g2": {"candidate_id": "grok_natural", "spoken_text": "yebo", "measured_ms": 900,
               "required_speed_percent": 3.0, "qa_penalty": 0, "path": "/g2.wav"},
    }
    geometries = {"g1": _turn_geometry(groups[0]), "g2": _turn_geometry(groups[1])}
    updated_winners = _resync_turn_rush_aggregates(
        groups=groups, winners=winners, geometries=geometries,
        preferred_raw_speed_percent=6, measurement_audio_root=tmp_path,
    )
    # g2 classifies as a bridge -- the whole turn is left exactly as-is,
    # since Phase 14 renders a bridge-containing turn per-sentence, not as
    # one shared-speed-factor unit.
    assert updated_winners == winners


def test_resync_skips_single_member_turns(tmp_path):
    groups = [_turn_group("g1", start_ms=0, source_text="All right.", span_ms=900)]
    winners = {"g1": {"candidate_id": "grok_natural", "spoken_text": "kulungile", "measured_ms": 900,
                       "required_speed_percent": 0.0, "qa_penalty": 0, "path": "/x.wav"}}
    updated_winners = _resync_turn_rush_aggregates(
        groups=groups, winners=winners, geometries={"g1": _turn_geometry(groups[0])},
        preferred_raw_speed_percent=6, measurement_audio_root=tmp_path,
    )
    assert updated_winners == winners


def test_rebalance_never_extends_the_turns_committed_end_past_the_real_mouth_close(tmp_path):
    """Regression test for a real, confirmed bug (job 33cd7b46..., 2026-08-31):
    an earlier version of this mechanism extended a turn's committed end into
    real trailing silence before the next turn, reasoning it could never
    collide with that next turn. That reasoning covered collision safety but
    missed that render's own, separate per-sentence gap-borrow mechanism
    (`_run_batch_adaptive_timing_controller`, unconditional, unrelated to this
    function) ALSO independently borrows from that same real gap on top of
    whatever `source_end_ms` it's handed -- the two compounded, and the
    rendered Zulu audio ran audibly past the true mouth-close point (confirmed
    directly by the user listening to native_dubbed.mp4, then by
    silencedetect). The turn's committed end must therefore always be the
    real, raw last member's `source_end_ms` -- exactly what render's own
    mechanism expects to independently borrow from -- even when a large real
    gap (5000ms here) sits unused right after it.
    """
    groups = [
        _turn_group("g1", start_ms=0, span_ms=3000, source_text="Overloaded sentence right here now."),
        _turn_group("g2", start_ms=3100, span_ms=3000, source_text="Sentence with real slack right here."),
        _turn_group("g3", speaker_id="S1", start_ms=11100, span_ms=800, source_text="A later, unrelated turn."),
    ]
    wav1, wav2 = tmp_path / "g1.wav", tmp_path / "g2.wav"
    _write_real_wav(wav1, ms=4200)
    _write_real_wav(wav2, ms=1800)
    winners = {
        "g1": {"candidate_id": "grok_natural", "spoken_text": "g1 zulu overloaded", "measured_ms": 4200,
               "required_speed_percent": _real_required_speed(4200), "qa_penalty": 0, "path": str(wav1)},
        "g2": {"candidate_id": "grok_natural", "spoken_text": "g2 zulu slack", "measured_ms": 1800,
               "required_speed_percent": 0.0, "qa_penalty": 0, "path": str(wav2)},
    }
    updated_winners, updated_geometries = _rebalance_speaker_turns(
        groups=groups, winners=winners,
        geometries={"g1": _turn_geometry(groups[0]), "g2": _turn_geometry(groups[1])},
        preferred_raw_speed_percent=6, measurement_audio_root=tmp_path,
    )
    # The turn's raw, real end is 6100 (g2's source_end_ms) -- the committed
    # result must stop exactly there, never extending into the real 5000ms
    # gap before g3, no matter how much real silence is available.
    assert updated_winners["g2"]["effective_source_end_ms"] == 6100
    assert updated_winners["g1"]["effective_start_ms"] == 0
    assert updated_winners["g1"]["spoken_text"] == "g1 zulu overloaded"
    assert updated_winners["g2"]["spoken_text"] == "g2 zulu slack"


def test_rebalance_speaker_turns_returns_updated_geometries_for_a_promoted_turn(tmp_path):
    """Phase 15 fix: a promoted turn's real, reallocated geometry must be
    returned so a caller measuring a NEW candidate against a rebalanced member
    afterward (turn compaction, or per-sentence compaction) sees the member's
    real, current effective window instead of the stale, pre-rebalance one.
    """
    groups = [
        _turn_group("g1", start_ms=0, span_ms=3000, source_text="Normal sentence with slack right here."),
        _turn_group("g2", start_ms=3100, span_ms=3000, source_text="Normal sentence that is overloaded here."),
    ]
    wav1, wav2 = tmp_path / "g1.wav", tmp_path / "g2.wav"
    _write_real_wav(wav1, ms=2000)
    _write_real_wav(wav2, ms=4600)
    winners = {
        "g1": {"candidate_id": "grok_natural", "spoken_text": "sawubona bakithi manje", "measured_ms": 2000,
               "required_speed_percent": _real_required_speed(2000), "qa_penalty": 0, "path": str(wav1)},
        "g2": {"candidate_id": "grok_natural", "spoken_text": "kwenzekile kabi kakhulu", "measured_ms": 4600,
               "required_speed_percent": _real_required_speed(4600), "qa_penalty": 0, "path": str(wav2)},
    }
    _, updated_geometries = _rebalance_speaker_turns(
        groups=groups, winners=winners,
        geometries={"g1": _turn_geometry(groups[0]), "g2": _turn_geometry(groups[1])},
        preferred_raw_speed_percent=6, measurement_audio_root=tmp_path,
    )
    assert set(updated_geometries) == {"g1", "g2"}
    # g2's real reallocated window is bigger than its original 3000ms span --
    # it earned a larger share of the fixed combined window for its content.
    assert updated_geometries["g2"]["source_window_ms"] > 3000


def test_rebalance_speaker_turns_returns_no_geometry_entry_for_an_untouched_turn(tmp_path):
    groups = [_turn_group("g1", start_ms=0), _turn_group("g2", start_ms=4000, span_ms=3100)]
    wav1, wav2 = tmp_path / "g1.wav", tmp_path / "g2.wav"
    _write_real_wav(wav1, ms=3000)
    _write_real_wav(wav2, ms=3100)
    winners = {
        "g1": {"candidate_id": "grok_natural", "spoken_text": "sawubona kakhulu", "measured_ms": 3000,
               "required_speed_percent": 0.0, "qa_penalty": 0, "path": str(wav1)},
        "g2": {"candidate_id": "grok_natural", "spoken_text": "sawubona futhi", "measured_ms": 3100,
               "required_speed_percent": 0.0, "qa_penalty": 0, "path": str(wav2)},
    }
    _, updated_geometries = _rebalance_speaker_turns(
        groups=groups, winners=winners,
        geometries={"g1": _turn_geometry(groups[0]), "g2": _turn_geometry(groups[1])},
        preferred_raw_speed_percent=6, measurement_audio_root=tmp_path,
    )
    assert updated_geometries == {}


# --- _score_speaker_turn: the "fits" signal vs. the coarse gap_aware/sentence_end label ---
# (moved here from the now-deleted tests/test_native_turn_compaction.py -- _score_speaker_turn
# itself is kept, reused by _rebalance_speaker_turns, even though the turn-aggregate
# compaction mechanism that originally motivated this specific test was removed.)


def test_a_turn_with_only_a_trivial_gap_can_still_report_fits_false(tmp_path):
    # Reproduces a real production geometry: a genuinely dense turn with only
    # the module's own 40ms floor tolerance before the next turn --
    # "gap_aware" by the coarse label, but nowhere near enough to absorb a
    # real ~50% overrun.
    g1 = _turn_group("g1", start_ms=0, span_ms=3000)
    g2 = _turn_group("g2", start_ms=3100, span_ms=3000)
    turn = {"member_group_ids": ["g1", "g2"], "start_ms": 0, "source_end_ms": 6100, "speaker_id": "S0"}
    turn_geometry = _speaker_turn_geometry(
        turn, groups_by_id={"g1": g1, "g2": g2}, next_source_start_ms=6100 + 40,
        preferred_raw_speed_percent=6,
    )
    assert turn_geometry["mouth_close_late_tolerance_ms"] > 0  # the coarse label would call this "gap_aware"

    wav1, wav2 = tmp_path / "g1.wav", tmp_path / "g2.wav"
    _write_real_wav(wav1, ms=3000)
    _write_real_wav(wav2, ms=8000)

    def _winner(measured_ms, path):
        return {
            "candidate_id": "grok_natural", "spoken_text": "izwi elikhulu kakhulu manje",
            "measured_ms": measured_ms, "required_speed_percent": 0.0, "direction": "compress",
            "qa_penalty": 0, "qa_record": None, "requires_review": False, "path": str(path),
            "translator": "grok", "variant_id": "natural",
        }

    _, info = _score_speaker_turn(
        turn=turn, turn_geometry=turn_geometry,
        member_winners={"g1": _winner(3000, wav1), "g2": _winner(8000, wav2)},
        member_geometries={"g1": _turn_geometry(g1), "g2": _turn_geometry(g2)},
        measurement_audio_root=tmp_path,
    )
    assert info["speed_fit_mode"] == "gap_aware"
    assert info["fits"] is False  # the real, fit-based signal correctly still says "doesn't fit"
    assert info["required_speed_percent"] > DEFAULT_MAX_NATURAL_SPEED_PERCENT
