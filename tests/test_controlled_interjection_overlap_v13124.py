from mathula_tv.direct_azure_dub import (
    SpeechBlock,
    rebalance_sandwiched_interjection_timeline,
)


def _block(block_id: str, speaker: str, start: int, end: int) -> SpeechBlock:
    return SpeechBlock(
        block_id,
        speaker,
        start,
        end,
        (f"seg-{block_id}",),
        block_id,
        block_id,
        block_id,
    )


def test_production_overlap_closes_only_the_measured_shortfall() -> None:
    blocks = [
        _block("previous", "A", 0, 700),
        _block("interjection", "B", 700, 1300),
        _block("following", "A", 1300, 2200),
    ]

    adjusted, audits = rebalance_sandwiched_interjection_timeline(
        blocks,
        {0: 900, 1: 1000, 2: 900},
        max_total_overlap_ms=1000,
        max_boundary_overlap_ms=750,
        max_source_gap_ms=250,
    )

    assert len(audits) == 1
    audit = audits[0]
    assert audit["applied"] is True
    assert audit["solver_version"] == (
        "measured_triplet_span_solver_v3_controlled_overlap"
    )
    assert audit["cross_speaker_overlap"] is True
    assert audit["total_overlap_ms"] == 600
    assert audit["total_overlap_ms"] == (
        audit["overlap_before_ms"] + audit["overlap_after_ms"]
    )
    assert [item.duration_ms for item in adjusted] == [900, 1000, 900]
    # The resumed surrounding speaker must never overlap itself.
    assert adjusted[0].end_ms <= adjusted[2].start_ms
    assert (
        adjusted[0].end_ms > adjusted[1].start_ms
        or adjusted[1].end_ms > adjusted[2].start_ms
    )


def test_overlap_is_rejected_when_source_handoffs_are_not_interjection_like() -> None:
    blocks = [
        _block("previous", "A", 0, 500),
        _block("middle", "B", 1000, 1500),
        _block("following", "A", 2000, 2500),
    ]

    adjusted, audits = rebalance_sandwiched_interjection_timeline(
        blocks,
        {0: 900, 1: 1000, 2: 900},
        max_total_overlap_ms=1000,
        max_boundary_overlap_ms=750,
        max_source_gap_ms=250,
    )

    assert adjusted == blocks
    assert audits[0]["applied"] is False
    assert audits[0]["reason"] == "source_timeline_not_interjection_like"


def test_overlap_is_rejected_above_the_configured_budget() -> None:
    blocks = [
        _block("previous", "A", 0, 700),
        _block("middle", "B", 700, 1300),
        _block("following", "A", 1300, 2200),
    ]

    adjusted, audits = rebalance_sandwiched_interjection_timeline(
        blocks,
        {0: 1200, 1: 1200, 2: 1200},
        max_total_overlap_ms=500,
        max_boundary_overlap_ms=400,
        max_source_gap_ms=250,
    )

    assert adjusted == blocks
    assert audits[0]["applied"] is False
    assert audits[0]["reason"] == "interjection_overlap_budget_exceeded"


def test_default_helper_remains_strict_for_backward_compatibility() -> None:
    blocks = [
        _block("previous", "A", 0, 700),
        _block("middle", "B", 700, 1300),
        _block("following", "A", 1300, 2200),
    ]

    adjusted, audits = rebalance_sandwiched_interjection_timeline(
        blocks,
        {0: 900, 1: 1000, 2: 900},
    )

    assert adjusted == blocks
    assert audits[0]["applied"] is False
    assert audits[0]["solver_version"] == "measured_triplet_span_solver_v2"
    assert audits[0]["reason"] == "insufficient_measured_triplet_span"
