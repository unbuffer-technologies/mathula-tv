from mathula_tv.direct_azure_dub import (
    SpeechBlock,
    _selected_azure_measurement_ms,
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


def test_global_fit_uses_measured_surplus_from_both_surrounding_turns() -> None:
    blocks = [
        _block("previous", "SPEAKER_01", 128000, 133150),
        _block("interjection", "SPEAKER_02", 133150, 133970),
        _block("resumed", "SPEAKER_01", 133970, 135270),
    ]

    adjusted, audits = rebalance_sandwiched_interjection_timeline(
        blocks,
        {0: 4150, 1: 1581, 2: 900},
    )

    assert len(audits) == 1
    assert audits[0]["applied"] is True
    assert adjusted[1].duration_ms == 1581
    assert adjusted[0].duration_ms >= 4150
    assert adjusted[2].duration_ms >= 900
    assert adjusted[0].end_ms == adjusted[1].start_ms
    assert adjusted[1].end_ms == adjusted[2].start_ms
    assert audits[0]["downstream_overflow_prevented"] is True


def test_global_fit_resolves_repeated_sandwiches_without_moving_failure_forward() -> None:
    blocks = [
        _block("a0", "A", 0, 4000),
        _block("b1", "B", 4000, 4700),
        _block("a2", "A", 4700, 8500),
        _block("b3", "B", 8500, 9200),
        _block("a4", "A", 9200, 13000),
    ]
    requirements = {0: 3300, 1: 1500, 2: 2500, 3: 1500, 4: 3300}

    adjusted, audits = rebalance_sandwiched_interjection_timeline(
        blocks,
        requirements,
    )

    assert [item["applied"] for item in audits] == [True, True]
    for index, required in requirements.items():
        assert adjusted[index].duration_ms >= required
    for previous, following in zip(adjusted, adjusted[1:]):
        assert previous.end_ms <= following.start_ms


def test_global_fit_fails_closed_when_triplet_has_no_measured_capacity() -> None:
    blocks = [
        _block("previous", "A", 0, 1000),
        _block("interjection", "B", 1000, 1800),
        _block("resumed", "A", 1800, 2800),
    ]

    adjusted, audits = rebalance_sandwiched_interjection_timeline(
        blocks,
        {0: 1000, 1: 1581, 2: 1000},
    )

    assert adjusted == blocks
    assert len(audits) == 1
    assert audits[0]["method"] == "global_sandwiched_interjection_timeline_fit"
    assert audits[0]["applied"] is False
    assert audits[0]["block_index"] == 1
    assert audits[0]["block_id"] == "interjection"
    assert audits[0]["reason"] == "insufficient_measured_triplet_span"
    assert audits[0]["shortfall_ms"] == 781
    assert audits[0]["text_immutable"] is True
    assert audits[0]["speaker_order_preserved"] is True
    assert audits[0]["cross_speaker_overlap"] is False


def test_selected_azure_measurement_uses_raw_selected_candidate() -> None:
    fitted = {
        "selected_azure_rate_percent": 20,
        "duration_ms": 1581,
        "azure_attempts": [
            {"rate_percent": 0, "duration_ms": 1900},
            {"rate_percent": 20, "duration_ms": 1628},
        ],
    }

    assert _selected_azure_measurement_ms(fitted) == 1628


def test_renderer_runs_global_fit_before_finalize_loop() -> None:
    import inspect

    from mathula_tv.direct_azure_dub import DirectAzureDubRenderer

    source = inspect.getsource(DirectAzureDubRenderer.render)
    assert "rebalance_sandwiched_interjection_timeline" in source
    assert source.index("rebalance_sandwiched_interjection_timeline") < source.index(
        '"finalize_blocks"'
    )
