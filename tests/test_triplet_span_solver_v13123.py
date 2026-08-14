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


def test_fits_middle_when_previous_neighbour_is_also_below_requirement() -> None:
    blocks = [
        _block("previous", "A", 0, 800),
        _block("middle", "B", 800, 1450),
        _block("following", "A", 1450, 3000),
    ]

    adjusted, audits = rebalance_sandwiched_interjection_timeline(
        blocks,
        {0: 1000, 1: 1000, 2: 1000},
    )

    assert len(audits) == 1
    assert audits[0]["applied"] is True
    assert audits[0]["solver_version"] == "measured_triplet_span_solver_v2"
    assert [item.duration_ms for item in adjusted] == [1000, 1000, 1000]
    assert adjusted[0].end_ms <= adjusted[1].start_ms
    assert adjusted[1].end_ms <= adjusted[2].start_ms


def test_impossible_triplet_is_audited_without_internal_exception() -> None:
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
    assert len(audits) == 1
    assert audits[0]["applied"] is False
    assert audits[0]["reason"] == "insufficient_measured_triplet_span"
    assert audits[0]["shortfall_ms"] == 600


def test_repeated_sandwiches_preserve_every_measured_minimum() -> None:
    blocks = [
        _block("a0", "A", 0, 800),
        _block("b1", "B", 800, 1450),
        _block("a2", "A", 1450, 3000),
        _block("b3", "B", 3000, 3650),
        _block("a4", "A", 3650, 5000),
    ]
    requirements = {0: 1000, 1: 900, 2: 1000, 3: 900, 4: 1000}

    adjusted, audits = rebalance_sandwiched_interjection_timeline(
        blocks,
        requirements,
    )

    assert [audit["applied"] for audit in audits] == [True, True]
    for index, required in requirements.items():
        assert adjusted[index].duration_ms >= required
    for previous, following in zip(adjusted, adjusted[1:]):
        assert previous.end_ms <= following.start_ms
