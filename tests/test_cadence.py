from mathula_tv.cadence import (
    aggregate_cadence_qc,
    assess_perceptual_cadence,
    plan_phrase_cadence,
)


def test_phrase_plan_preserves_every_character_and_exact_window():
    text = "I-EFF iyafika. Kodwa i-ANC iseduze, namhlanje."
    plan = plan_phrase_cadence(
        block_id="block_0001",
        text=text,
        start_ms=1000,
        end_ms=9000,
    )
    assert plan.immutable_text == text
    assert plan.phrases[0].start_ms == 1000
    assert plan.phrases[-1].end_ms == 9000
    assert all(
        left.end_ms == right.start_ms
        for left, right in zip(plan.phrases, plan.phrases[1:])
    )
    assert [phrase.text for phrase in plan.phrases] == [
        "I-EFF iyafika.",
        "Kodwa i-ANC iseduze,",
        "namhlanje.",
    ]


def test_natural_rate_with_distributed_pauses_passes_cadence_qc():
    qc = assess_perceptual_cadence(
        selected_rate_percent=-3,
        base_rate_percent=-3,
        whole_voice_stretch_ratio=1.0,
        timeline_fill_action="distributed_pause_fill",
        pause_fill={"maximum_added_pause_ms": 1400},
    )
    assert qc["passed"] is True
    assert qc["issues"] == []


def test_rushed_dragged_and_oversized_pauses_require_review():
    qc = assess_perceptual_cadence(
        selected_rate_percent=15,
        base_rate_percent=0,
        whole_voice_stretch_ratio=0.72,
        timeline_fill_action="expanded_to_window",
        pause_fill={"maximum_added_pause_ms": 4500},
    )
    assert qc["passed"] is False
    assert {issue["code"] for issue in qc["issues"]} == {
        "whole_turn_rate_delta_excessive",
        "whole_turn_time_stretch_excessive",
        "inserted_pause_excessive",
    }
    assert aggregate_cadence_qc(
        [{"block_id": "block_0001", "cadence_qc": qc}]
    )["failed_block_ids"] == ["block_0001"]


def test_continuous_speech_cannot_dump_missing_rhythm_at_segment_end():
    qc = assess_perceptual_cadence(
        selected_rate_percent=3,
        base_rate_percent=3,
        whole_voice_stretch_ratio=1.0,
        timeline_fill_action="natural_speech_with_trailing_room",
        trailing_room_ms=8360,
    )
    assert qc["passed"] is False
    assert qc["issues"] == [
        {
            "code": "unexplained_trailing_room_excessive",
            "value": 8360,
            "limit": 750,
            "required_action": "align target breath groups to source phrase timestamps",
        }
    ]
