from mathula_tv.native_dub import (
    _choose_sentence_candidate,
    _temporal_sentence_semantic_window,
    _timing_candidate_rank,
)


def test_tail_window_remains_four_blocks_with_locked_predecessor():
    groups = [
        {"group_id": f"g{i}", "speaker_id": "S", "source_text": f"E{i}"}
        for i in range(7)
    ]
    accepted = [
        {"group_id": f"g{i}", "spoken_text": f"Z{i}"}
        for i in range(4)
    ]
    window = _temporal_sentence_semantic_window(
        groups, mutable_start=4, mutable_count=3, accepted=accepted
    )
    assert [item["group_id"] for item in window["blocks"]] == ["g3", "g4", "g5", "g6"]
    assert window["blocks"][0]["locked"] is True
    assert window["blocks"][0]["zulu"] == "Z3"


# --- per-sentence independent selection ---------------------------------------------
# Superseded design correction: candidates used to be chosen as one shared "column"
# for the whole window (every sentence forced to the same letter), scored by the
# window's single WORST sentence. Confirmed real production bug: an undersized
# sentence gets MORE undersized at every more-compact letter (shortening an already-
# short rendering only widens the gap from its target), so any window containing even
# one naturally undersized sentence always scored worse at "b"/"c" than "a" for the
# WHOLE window -- vetoing compaction for every genuinely rushed sibling sentence in
# that same window, even though real jobs measure ~59% of sentences as rushed/oversized
# against only ~27% undersized. Selection is now per-sentence and independent -- see
# _choose_sentence_candidate.

def test_each_sentence_picks_its_own_best_candidate_independently():
    measured = {
        "g0": [
            {"candidate_id": "a", "semantic_guard_pass": True, "direction": "expand", "rank": (2, 100, 100)},
            {"candidate_id": "b", "semantic_guard_pass": True, "direction": "expand", "rank": (2, 300, 300)},
        ],
        "g1": [
            {"candidate_id": "a", "semantic_guard_pass": True, "direction": "compress", "rank": (1, 30000, 4000)},
            {"candidate_id": "b", "semantic_guard_pass": True, "direction": "compress", "rank": (1, 8000, 1000)},
        ],
    }
    g0_choice = _choose_sentence_candidate(measured["g0"])
    g1_choice = _choose_sentence_candidate(measured["g1"])
    # g0 is already undersized; compacting it only makes it worse, so it stays "a".
    assert g0_choice["candidate_id"] == "a"
    # g1 is genuinely rushed; compacting it to "b" is a real improvement.
    assert g1_choice["candidate_id"] == "b"


def test_undersized_candidate_is_scored_not_disqualified():
    measured = [
        {"candidate_id": "a", "semantic_guard_pass": True, "direction": "expand", "rank": (2, 500, 500)},
        {"candidate_id": "b", "semantic_guard_pass": True, "direction": None, "rank": (0, 10, 10)},
    ]
    # "b" fits normally and should still win over an undersized "a" ...
    assert _choose_sentence_candidate(measured)["candidate_id"] == "b"
    # ... but "a" alone (undersized, no better option) must still be selectable, not
    # disqualified outright.
    alone = _choose_sentence_candidate([measured[0]])
    assert alone["candidate_id"] == "a"
    assert alone["direction"] == "expand"


def test_candidate_failing_semantic_guard_is_excluded():
    measured = [
        {"candidate_id": "a", "semantic_guard_pass": False, "rank": (0, 1, 1)},
        {"candidate_id": "b", "semantic_guard_pass": True, "rank": (1, 100, 100)},
    ]
    assert _choose_sentence_candidate(measured)["candidate_id"] == "b"


def test_no_candidate_passing_the_semantic_guard_returns_none():
    measured = [{"candidate_id": "a", "semantic_guard_pass": False, "rank": (0, 1, 1)}]
    assert _choose_sentence_candidate(measured) is None


def test_unlimited_rush_is_preferred_to_undersized_speech():
    rush = _timing_candidate_rank(
        1500, 1000, 1060,
        max_speed_percent=12,
        mouth_close_early_tolerance_ms=40,
        mouth_close_late_tolerance_ms=40,
    )
    early = _timing_candidate_rank(
        700, 1000, 1060,
        max_speed_percent=12,
        mouth_close_early_tolerance_ms=40,
        mouth_close_late_tolerance_ms=40,
    )
    assert rush[0] == 1
    assert early[0] == 2
    assert rush < early
