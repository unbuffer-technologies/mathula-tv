from mathula_tv.native_dub import (
    _choose_temporal_candidate,
    _normalize_temporal_mask_response,
    _temporal_mask_groups_for_render,
)


def test_temporal_mask_response_reorders_masks_and_candidates_by_identity():
    groups = [{"group_id": "g1"}, {"group_id": "g2"}]
    data = {
        "masks": [
            {"group_id": "g2", "candidates": [
                {"candidate_id": "c", "spoken_text": "c2"},
                {"candidate_id": "a", "spoken_text": "a2"},
                {"candidate_id": "b", "spoken_text": "b2"},
            ]},
            {"group_id": "g1", "candidates": [
                {"candidate_id": "b", "spoken_text": "b1"},
                {"candidate_id": "c", "spoken_text": "c1"},
                {"candidate_id": "a", "spoken_text": "a1"},
            ]},
        ]
    }
    result = _normalize_temporal_mask_response(data, groups, candidate_count=3)
    assert [x["candidate_id"] for x in result["g1"]] == ["a", "b", "c"]
    assert [x["spoken_text"] for x in result["g1"]] == ["a1", "b1", "c1"]
    assert [x["candidate_id"] for x in result["g2"]] == ["a", "b", "c"]


def test_temporal_mask_prefers_provisional_a_when_a_is_feasible():
    measured = [
        {"candidate_id": "a", "fit": True, "rank": [0, 50, 50]},
        {"candidate_id": "b", "fit": True, "rank": [0, 1, 1]},
        {"candidate_id": "c", "fit": False, "rank": [2, 9, 9]},
    ]
    assert _choose_temporal_candidate(measured)["candidate_id"] == "a"


def test_temporal_mask_uses_alternative_when_a_is_not_feasible():
    measured = [
        {"candidate_id": "a", "fit": False, "rank": [2, 20, 20]},
        {"candidate_id": "b", "fit": True, "rank": [0, 5, 5]},
        {"candidate_id": "c", "fit": True, "rank": [0, 9, 9]},
    ]
    assert _choose_temporal_candidate(measured)["candidate_id"] == "b"


def test_temporal_mask_accepts_overlong_but_not_early_candidate():
    measured = [
        {"candidate_id": "a", "fit": False, "direction": "expand", "semantic_guard_pass": True, "measured_ms": 700, "rank": [2, 30, 30]},
        {"candidate_id": "b", "fit": False, "direction": "compress", "semantic_guard_pass": True, "measured_ms": 1400, "rank": [1, 10, 10]},
        {"candidate_id": "c", "fit": False, "direction": "expand", "semantic_guard_pass": True, "measured_ms": 600, "rank": [2, 50, 50]},
    ]
    assert _choose_temporal_candidate(measured)["candidate_id"] == "b"


def test_temporal_mask_render_reconstructs_exact_multi_segment_group():
    segments = [
        {"segment_id": "s1", "speaker_id": "A", "start_ms": 0, "end_ms": 500, "restored_text": "one"},
        {"segment_id": "s2", "speaker_id": "A", "start_ms": 450, "end_ms": 1000, "restored_text": "two"},
        {"segment_id": "s3", "speaker_id": "B", "start_ms": 1000, "end_ms": 1500, "restored_text": "three"},
    ]
    pass2 = {
        "temporal_mask_groups": [
            {"group_id": "g1", "speaker_id": "A", "segment_ids": ["s1", "s2"], "start_ms": 0, "source_end_ms": 1000, "source_span_ms": 1000, "spoken_text": "Zulu one two"},
            {"group_id": "g2", "speaker_id": "B", "segment_ids": ["s3"], "start_ms": 1000, "source_end_ms": 1500, "source_span_ms": 500, "spoken_text": "Zulu three"},
        ]
    }
    groups, translations = _temporal_mask_groups_for_render(pass2, segments)
    assert [g["segment_ids"] for g in groups] == [["s1", "s2"], ["s3"]]
    assert groups[0]["parts"] == [{"type": "text", "segment_id": "s1", "text": "Zulu one two"}]
    # Every member of a merged group gets the group's full text, not just the
    # first segment -- Pass 2 never produces a per-original-segment breakdown to
    # split it by, and leaving a real segment's translation empty silently
    # excludes it from any downstream consumer that treats "no text" as "not a
    # real block" (confirmed in production: a merged group's second segment
    # disappeared from TikTok hook evidence candidates entirely).
    assert translations == {"s1": "Zulu one two", "s2": "Zulu one two", "s3": "Zulu three"}


def test_temporal_mask_residual_fresh_labels_are_canonicalized_by_position():
    groups = [{"group_id": "native_phrase_0001"}]
    data = {
        "masks": [{
            "group_id": "native_phrase_0001",
            "candidates": [
                {"candidate_id": "d", "spoken_text": "first residual wording"},
                {"candidate_id": "e", "spoken_text": "second residual wording"},
                {"candidate_id": "f", "spoken_text": "third residual wording"},
            ],
        }]
    }
    result = _normalize_temporal_mask_response(data, groups, candidate_count=3)
    assert [item["candidate_id"] for item in result["native_phrase_0001"]] == ["a", "b", "c"]
    assert [item["spoken_text"] for item in result["native_phrase_0001"]] == [
        "first residual wording",
        "second residual wording",
        "third residual wording",
    ]


def test_temporal_mask_partial_identity_mix_is_rejected():
    groups = [{"group_id": "g1"}]
    data = {
        "masks": [{
            "group_id": "g1",
            "candidates": [
                {"candidate_id": "a", "spoken_text": "one"},
                {"candidate_id": "e", "spoken_text": "two"},
                {"candidate_id": "f", "spoken_text": "three"},
            ],
        }]
    }
    import pytest
    with pytest.raises(ValueError, match="candidate identity mismatch"):
        _normalize_temporal_mask_response(data, groups, candidate_count=3)


def test_temporal_mask_semantic_guard_can_reject_timing_fit_candidate():
    from mathula_tv.native_dub import _temporal_mask_high_risk_guard

    source = "less than 24 hours, then less than 24 hours later"
    target = "ngaphansi kwamahora angu-24, kwase kuba kamuva"
    passed, missing = _temporal_mask_high_risk_guard(source, target)
    assert passed is False
    assert missing == ["number:24 x1"]


def test_temporal_residual_wire_reports_semantic_failure_without_azure_ms():
    from mathula_tv.native_dub import _temporal_residual_candidate_wire

    wire = _temporal_residual_candidate_wire({
        "candidate_id": "a",
        "spoken_text": "candidate",
        "estimated_syllables": 42,
        "direction": None,
        "fit": True,
        "semantic_guard_pass": False,
        "semantic_guard_missing": ["number:24 x1"],
    })
    assert "measured_ms" not in wire
    assert "timing_fit" not in wire
    assert wire["estimated_syllables"] == 42
    assert wire["semantic_guard_pass"] is False
    assert wire["semantic_guard_missing"] == ["number:24 x1"]
    assert wire["rejection_reason"] == "semantic_guard"


def test_temporal_mask_semantic_guard_exposes_occurrence_level_contexts():
    from mathula_tv.native_dub import _temporal_mask_semantic_guard_details

    source = (
        "less than 24 hours after laying the complaints, then less than 24 hours "
        "you e-mail the minister"
    )
    target = "ngaphansi kwamahora angu-24 ngemva kokufaka izikhalazo, wabe esethumela i-imeyili"
    details = _temporal_mask_semantic_guard_details(source, target)
    assert len(details) == 1
    item = details[0]
    assert item["kind"] == "number"
    assert item["literal"] == "24"
    assert item["required_count"] == 2
    assert item["observed_count"] == 1
    assert item["missing_count"] == 1
    assert len(item["source_occurrences"]) == 2
    assert all("24" in occ["source_context"] for occ in item["source_occurrences"])


def test_temporal_number_guard_does_not_absorb_sentence_punctuation():
    from mathula_tv.native_dub import _temporal_mask_high_risk_guard

    assert _temporal_mask_high_risk_guard("24, then 24.", "24 bese 24") == (True, [])
    assert _temporal_mask_high_risk_guard("3,978 and 3.14", "3,978 kanye 3.14") == (True, [])


def test_bare_single_digit_number_naturally_worded_is_advisory_not_blocking():
    """Regression test for a real production false positive on a genuine content check.

    Confirmed in production: the same block correctly kept "27" (a section number) as a
    digit while "1" (a plain count -- "1 source") was not found as a literal digit,
    almost certainly because the model naturally worded it ("owodwa") instead of writing
    a bare digit into spoken isiZulu. A specific reference/date/percentage/multi-digit
    figure is a precise identifier worth a hard stop if missing; a bare 1-9 count is not.
    """
    from mathula_tv.native_dub import _temporal_mask_high_risk_guard, _temporal_mask_semantic_guard_details

    source = (
        "a diversion that emanates from his section 27 referral, and all you have is "
        "1 source who is now late and cannot corroborate that story"
    )
    target_naturally_worded = (
        "okuvela ekudluliselweni kwakhe kwesigaba 27, futhi konke onakho ngumthombo "
        "owodwa oseshonile futhi ongeke aqinisekise leyo ndaba"
    )
    details = _temporal_mask_semantic_guard_details(source, target_naturally_worded)
    assert len(details) == 1
    assert details[0]["literal"] == "1"
    assert details[0]["blocking"] is False

    passed, missing = _temporal_mask_high_risk_guard(source, target_naturally_worded)
    assert passed is True  # advisory-only findings never fail the guard
    assert missing == []


def test_decorated_or_multi_digit_numbers_still_hard_block_when_missing():
    from mathula_tv.native_dub import _temporal_mask_high_risk_guard

    # Multi-digit, both occurrences dropped -- still blocking.
    passed, missing = _temporal_mask_high_risk_guard(
        "less than 24 hours, then less than 24 hours later", "kamuva kancane"
    )
    assert passed is False
    assert missing == ["number:24 x2"]

    # A decorated single digit (percent, ordinal) is a precise value, not a natural
    # count word -- still blocking even though it is one character.
    passed, missing = _temporal_mask_high_risk_guard(
        "inflation rose by 5% last year", "ukukhuphuka kwentengo kwenzeke ngonyaka odlule"
    )
    assert passed is False
    assert missing == ["number:5% x1"]


def test_temporal_residual_best_prefers_fewer_semantic_deficits_before_timing():
    from mathula_tv.native_dub import _temporal_residual_best_rank

    timing_perfect_missing_two = {
        "semantic_guard_pass": False,
        "semantic_guard_requirements": [{"missing_count": 2}],
        "rank": [0, 1, 1],
    }
    timing_less_perfect_missing_one = {
        "semantic_guard_pass": False,
        "semantic_guard_requirements": [{"missing_count": 1}],
        "rank": [2, 500, 500],
    }
    assert _temporal_residual_best_rank(timing_less_perfect_missing_one) < _temporal_residual_best_rank(timing_perfect_missing_two)


def test_temporal_mask_schema_rejects_wrong_residual_group_id():
    from jsonschema import ValidationError, validate
    from mathula_tv.native_dub import _temporal_mask_schema

    schema = _temporal_mask_schema(3, expected_group_ids=["native_phrase_0005"])
    wrong = {
        "masks": [{
            "group_id": "native_phrase_0007",
            "candidates": [
                {"candidate_id": "a", "spoken_text": "one"},
                {"candidate_id": "b", "spoken_text": "two"},
                {"candidate_id": "c", "spoken_text": "three"},
            ],
        }]
    }
    import pytest
    with pytest.raises(ValidationError):
        validate(wrong, schema)


def test_temporal_mask_schema_allows_requested_compose_ids_in_any_order():
    from jsonschema import validate
    from mathula_tv.native_dub import _temporal_mask_schema

    schema = _temporal_mask_schema(1, expected_group_ids=["g1", "g2"])
    validate({"masks": [
        {"group_id": "g2", "candidates": [{"candidate_id": "a", "spoken_text": "two"}]},
        {"group_id": "g1", "candidates": [{"candidate_id": "a", "spoken_text": "one"}]},
    ]}, schema)



def test_four_block_window_tail_is_left_padded_with_locked_zulu():
    from mathula_tv.native_dub import _temporal_sentence_semantic_window

    groups = [
        {"group_id": f"g{i}", "speaker_id": "S", "source_text": f"English {i}"}
        for i in range(7)
    ]
    accepted = [
        {"group_id": f"g{i}", "spoken_text": f"Zulu {i}"}
        for i in range(4)
    ]
    window = _temporal_sentence_semantic_window(
        groups, mutable_start=4, mutable_count=3, accepted=accepted
    )
    assert [item["group_id"] for item in window["blocks"]] == ["g3", "g4", "g5", "g6"]
    assert window["blocks"][0]["locked"] is True
    assert window["blocks"][0]["zulu"] == "Zulu 3"


def test_sentence_candidate_selection_is_independent_per_sentence():
    # Superseded design correction: candidates used to be chosen as one shared
    # "column" for the whole window (every sentence forced to the same letter),
    # scored by the window's single worst sentence. Selection is now per-sentence and
    # independent -- see _choose_sentence_candidate and its module docstring for the
    # confirmed real production bug this fixes.
    from mathula_tv.native_dub import _choose_sentence_candidate

    measured = {
        "g0": [
            {"candidate_id": "a", "semantic_guard_pass": True, "direction": "compress", "rank": (1, 30000, 100)},
            {"candidate_id": "b", "semantic_guard_pass": True, "direction": None, "rank": (0, 10, 10)},
        ],
        "g1": [
            {"candidate_id": "a", "semantic_guard_pass": True, "direction": None, "rank": (0, 5, 5)},
            {"candidate_id": "b", "semantic_guard_pass": True, "direction": None, "rank": (0, 8, 8)},
        ],
    }
    assert _choose_sentence_candidate(measured["g0"])["candidate_id"] == "b"
    assert _choose_sentence_candidate(measured["g1"])["candidate_id"] == "a"
