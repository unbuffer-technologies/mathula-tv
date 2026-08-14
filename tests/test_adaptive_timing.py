from mathula_tv.adaptive_timing import (
    AdaptiveTimingConfig,
    TimingBlock,
    adaptive_same_speaker_group,
)


def test_expands_into_next_same_speaker_block():
    blocks = [
        TimingBlock("block_0001", "spk1", 0, 3000, "short"),
        TimingBlock("block_0002", "spk1", 3200, 7000, "longer"),
        TimingBlock("block_0003", "spk2", 7100, 9000, "other"),
    ]

    def measure(group):
        return 6000

    result = adaptive_same_speaker_group(
        blocks=blocks,
        failing_index=0,
        measure_tts_ms=measure,
        config=AdaptiveTimingConfig(
            max_same_speaker_gap_ms=1000,
            max_post_stretch_ratio=1.18,
            azure_rate_ratio=1.15,
        ),
        next_foreign_start_ms=7100,
    )

    assert result.fits is True
    assert [b.block_id for b in result.blocks] == ["block_0001", "block_0002"]


def test_fails_only_when_no_timeline_exists():
    blocks = [
        TimingBlock("block_0001", "spk1", 0, 1000, "very long"),
        TimingBlock("block_0002", "spk2", 1000, 2000, "other"),
    ]

    result = adaptive_same_speaker_group(
        blocks=blocks,
        failing_index=0,
        measure_tts_ms=lambda group: 5000,
        next_foreign_start_ms=1000,
    )

    assert result.fits is False
    assert result.reason == "timeline_collision_or_no_compatible_same_speaker_neighbour"
