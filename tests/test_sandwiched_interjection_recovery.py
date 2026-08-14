from mathula_tv.direct_azure_dub import SpeechBlock, rebalance_cross_speaker_handoff


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


def test_production_shaped_940ms_sandwiched_interjection_recovery() -> None:
    blocks = [
        _block("previous", "SPEAKER_01", 90000, 99350),
        _block("interjection", "SPEAKER_02", 99350, 99990),
        _block("resumed", "SPEAKER_01", 99990, 110000),
    ]

    expanded, shifted, audit = rebalance_cross_speaker_handoff(
        blocks,
        1,
        required_window_ms=1580,
        max_shift_ms=500,
    )

    assert shifted is not None
    assert expanded.end_ms == shifted.start_ms == 100930
    assert expanded.duration_ms == 1580
    assert shifted.duration_ms == 9070
    assert audit["boundary_shift_ms"] == 940
    assert audit["sandwiched_interjection"] is True
    assert audit["effective_maximum_boundary_shift_ms"] == 1000
    assert audit["surrounding_speaker_id"] == "SPEAKER_01"
    assert audit["text_immutable"] is True
    assert audit["cross_speaker_overlap"] is False


def test_generic_large_cross_speaker_shift_still_fails_closed() -> None:
    blocks = [
        _block("first", "A", 0, 640),
        _block("second", "B", 640, 5000),
    ]

    unchanged, shifted, audit = rebalance_cross_speaker_handoff(
        blocks,
        0,
        required_window_ms=1580,
        max_shift_ms=500,
    )

    assert unchanged == blocks[0]
    assert shifted is None
    assert audit["applied"] is False
    assert audit["sandwiched_interjection"] is False
    assert audit["effective_maximum_boundary_shift_ms"] == 500


def test_sandwiched_recovery_preserves_resumed_minimum_window() -> None:
    blocks = [
        _block("previous", "A", 0, 5000),
        _block("interjection", "B", 5000, 5640),
        _block("resumed", "A", 5640, 7300),
    ]

    unchanged, shifted, audit = rebalance_cross_speaker_handoff(
        blocks,
        1,
        required_window_ms=1580,
        max_shift_ms=500,
    )

    assert unchanged == blocks[1]
    assert shifted is None
    assert audit["applied"] is False
    assert audit["sandwiched_interjection"] is True
