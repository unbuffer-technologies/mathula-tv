from __future__ import annotations

from mathula_tv.direct_azure_dub import (
    ApprovedSegment,
    SpeechBlock,
    borrow_safe_silence_window,
    coalesce_same_speaker_segments,
    estimate_required_azure_rate,
)


def _segment(
    segment_id: str,
    speaker_id: str,
    start_ms: int,
    end_ms: int,
    text: str,
) -> ApprovedSegment:
    return ApprovedSegment(
        segment_id=segment_id,
        speaker_id=speaker_id,
        start_ms=start_ms,
        end_ms=end_ms,
        source_text=text,
        translated_text=text,
        tts_text=text,
    )


def test_coalesces_only_adjacent_same_speaker_fragments() -> None:
    segments = [
        _segment("seg-00001", "SPEAKER_00", 400, 23600, "umbhalo wokuqala"),
        _segment("seg-00002", "SPEAKER_00", 23600, 24080, "uyaqhubeka"),
        _segment("seg-00003", "SPEAKER_00", 24080, 29760, "lapha"),
        _segment("seg-00004", "SPEAKER_01", 34480, 107680, "omunye umuntu"),
    ]

    blocks = coalesce_same_speaker_segments(segments, max_gap_ms=1200)

    assert len(blocks) == 2
    assert blocks[0].segment_ids == ("seg-00001", "seg-00002", "seg-00003")
    assert blocks[0].start_ms == 400
    assert blocks[0].end_ms == 29760
    assert blocks[0].tts_text == "umbhalo wokuqala uyaqhubeka lapha"
    assert blocks[1].segment_ids == ("seg-00004",)


def test_does_not_merge_same_speaker_across_meaningful_silence() -> None:
    segments = [
        _segment("seg-00001", "SPEAKER_00", 0, 1000, "qala"),
        _segment("seg-00002", "SPEAKER_00", 3000, 4000, "qhubeka"),
    ]

    blocks = coalesce_same_speaker_segments(segments, max_gap_ms=1200)

    assert [block.segment_ids for block in blocks] == [
        ("seg-00001",),
        ("seg-00002",),
    ]


def test_estimates_absolute_azure_rate_from_measured_candidate() -> None:
    # 8242 ms at +2% must fit 6591 ms. The simple Azure-rate estimate is +28%.
    assert estimate_required_azure_rate(
        measured_duration_ms=8242,
        target_duration_ms=6591,
        measured_rate_percent=2,
    ) == 28


def test_candidate_cache_key_changes_with_voice(tmp_path) -> None:
    from mathula_tv.azure_tts import AzureTTSRequest
    from mathula_tv.direct_azure_dub import DirectDubArtifacts

    artifacts = DirectDubArtifacts(tmp_path)
    base = AzureTTSRequest(
        turn_id="block_0001",
        speaker_id="SPEAKER_00",
        text="sawubona",
        preferred_duration_ms=1000,
        maximum_duration_ms=1000,
        voice="zu-ZA-ThembaNeural",
    )
    female = AzureTTSRequest(
        turn_id="block_0001",
        speaker_id="SPEAKER_00",
        text="sawubona",
        preferred_duration_ms=1000,
        maximum_duration_ms=1000,
        voice="zu-ZA-ThandoNeural",
    )

    assert artifacts.candidate(base) != artifacts.candidate(female)


def test_direct_dub_defaults_to_dialogue_only() -> None:
    from mathula_tv.direct_azure_dub import DirectDubOptions

    assert DirectDubOptions().include_background is False


def test_borrows_only_silence_bounded_by_surrounding_speakers() -> None:
    blocks = [
        SpeechBlock("block_0001", "SPEAKER_00", 320, 85760, ("seg-1",), "a", "a", "a"),
        SpeechBlock(
            "block_0002",
            "SPEAKER_01",
            86560,
            87920,
            ("seg-2",),
            "hello",
            "Sawubona",
            "Sawubona",
        ),
        SpeechBlock("block_0003", "SPEAKER_00", 88160, 113760, ("seg-3",), "b", "b", "b"),
    ]

    expanded, adjustment = borrow_safe_silence_window(blocks, 1)

    assert expanded.block_id == "block_0002_adaptive"
    assert (expanded.start_ms, expanded.end_ms) == (85760, 88160)
    assert expanded.tts_text == blocks[1].tts_text
    assert adjustment["borrowed_before_ms"] == 800
    assert adjustment["borrowed_after_ms"] == 240
    assert adjustment["text_immutable"] is True
    assert adjustment["cross_speaker_overlap"] is False
