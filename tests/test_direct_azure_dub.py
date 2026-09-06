from __future__ import annotations

from mathula_tv.direct_azure_dub import (
    ApprovedSegment,
    SpeechBlock,
    borrow_safe_silence_window,
    borrow_trailing_transition_silence,
    coalesce_same_speaker_segments,
    estimate_required_azure_rate,
    load_approved_segments,
    normalize_tts_boundary_artifacts,
    plan_underfill_cadence_repair,
    rebalance_cross_speaker_handoff,
)
from mathula_tv.pronunciation import (
    PronunciationDictionary,
    with_default_organisation_initialisms,
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


def test_preserves_opening_unit_as_local_lipsync_endpoint() -> None:
    segments = [
        _segment("unit_0001", "SPEAKER_00", 110, 7445, "opening name anchor"),
        _segment("unit_0002", "SPEAKER_00", 7445, 13863, "next clause"),
        _segment("unit_0003", "SPEAKER_00", 13863, 21198, "third clause"),
    ]

    blocks = coalesce_same_speaker_segments(
        segments,
        max_gap_ms=1200,
        preserve_opening_segment=True,
    )

    assert [block.segment_ids for block in blocks] == [
        ("unit_0001",),
        ("unit_0002", "unit_0003"),
    ]
    assert blocks[0].end_ms == 7445


def test_borrows_only_bounded_trailing_transition_silence() -> None:
    blocks = [
        SpeechBlock("block_0001", "A", 0, 1000, ("s1",), "", "qala", "qala"),
        SpeechBlock("block_0002", "B", 6000, 7000, ("s2",), "", "landela", "landela"),
    ]
    adjusted = borrow_trailing_transition_silence(blocks, maximum_borrow_ms=3000)
    assert adjusted[0].end_ms == 4000
    assert adjusted[1] == blocks[1]


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


def test_direct_renderer_applies_pronunciation_without_changing_approved_text() -> None:
    dictionary = with_default_organisation_initialisms(
        PronunciationDictionary("v1", language="zu-ZA", job_id="job-123")
    )
    segments = load_approved_segments(
        {
            "segments": [
                {
                    "segment_id": "seg-00001",
                    "speaker_id": "SPEAKER_00",
                    "start": 0.0,
                    "end": 2.0,
                    "source_text": "The ANC, EFF and NPA.",
                    "translated_text": "I-ANC, i-EFF ne-NPA.",
                }
            ]
        },
        source_duration_ms=2_000,
        pronunciation_dictionary=dictionary,
    )

    assert segments[0].translated_text == "I-ANC, i-EFF ne-NPA."
    assert segments[0].tts_text == "I-ANC, i-Ee Eff Eff ne-En Pee Ey."


def test_direct_renderer_pronounces_reviewed_word_acronyms_as_words() -> None:
    dictionary = with_default_organisation_initialisms(
        PronunciationDictionary("v1", language="zu-ZA", job_id="job-123")
    )
    segments = load_approved_segments(
        {
            "segments": [
                {
                    "segment_id": "seg-00001",
                    "speaker_id": "SPEAKER_00",
                    "start": 0.0,
                    "end": 2.0,
                    "source_text": "IDAC works with IPID and SAPS.",
                    "translated_text": "I-IDAC isebenza ne-IPID kanye ne-SAPS.",
                }
            ]
        },
        source_duration_ms=2_000,
        pronunciation_dictionary=dictionary,
    )

    assert segments[0].translated_text == "I-IDAC isebenza ne-IPID kanye ne-SAPS."
    assert segments[0].tts_text == "I-Ay-dak isebenza ne-Ay-pid kanye ne-Saps."


def test_tts_boundary_ellipses_become_one_natural_pause() -> None:
    text = "Futhi... ...ngifuna ukubona. Njengoba sibona... abantu bayangena."
    assert normalize_tts_boundary_artifacts(text) == (
        "Futhi, ngifuna ukubona. Njengoba sibona, abantu bayangena."
    )


def test_tts_boundary_cleanup_does_not_change_approved_translation() -> None:
    segments = load_approved_segments(
        {
            "segments": [
                {
                    "segment_id": "seg-00001",
                    "speaker_id": "SPEAKER_02",
                    "start": 0.0,
                    "end": 2.0,
                    "source_text": "And... ...I want",
                    "translated_text": "Futhi... ...ngifuna",
                }
            ]
        },
        source_duration_ms=2_000,
    )
    assert segments[0].translated_text == "Futhi... ...ngifuna"
    assert segments[0].tts_text == "Futhi, ngifuna"


def test_atempo_supports_bounded_slowdown_for_timeline_fill() -> None:
    from mathula_tv.direct_azure_dub import _atempo_filters

    # 250.729 seconds of speech expanded over a 300.960-second source turn.
    ratio = 250_729 / 300_960
    assert _atempo_filters(ratio) == [f"atempo={ratio:.8f}"]


def test_large_underfill_becomes_bounded_cadence_repair() -> None:
    repair = plan_underfill_cadence_repair(
        measured_duration_ms=53_968,
        window_duration_ms=57_040,
    )

    assert repair["applied"] is True
    assert repair["target_duration_ms"] == 56_540
    assert repair["transition_room_after_ms"] == 500
    assert repair["required_expansion_ratio"] == 1.047658


def test_underfill_above_safe_expansion_remains_unresolved() -> None:
    repair = plan_underfill_cadence_repair(
        measured_duration_ms=46_070,
        window_duration_ms=57_040,
    )

    assert repair["applied"] is False
    assert repair["required_expansion_ratio"] > 1.06


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


def test_rebalances_a_small_cross_speaker_handoff_without_overlap() -> None:
    blocks = [
        SpeechBlock(
            "block_0014",
            "SPEAKER_03",
            652360,
            653000,
            ("seg-20",),
            "Thank you very much.",
            "Ngiyabonga kakhulu.",
            "Ngiyabonga kakhulu.",
        ),
        SpeechBlock(
            "block_0015",
            "SPEAKER_02",
            653000,
            691640,
            ("seg-21",),
            "Next report",
            "Umbiko olandelayo",
            "Umbiko olandelayo",
        ),
    ]

    expanded, shifted, adjustment = rebalance_cross_speaker_handoff(
        blocks,
        0,
        required_window_ms=936,
        max_shift_ms=500,
    )

    assert shifted is not None
    assert expanded.end_ms == shifted.start_ms == 653296
    assert expanded.tts_text == blocks[0].tts_text
    assert shifted.tts_text == blocks[1].tts_text
    assert adjustment["boundary_shift_ms"] == 296
    assert adjustment["cross_speaker_overlap"] is False
    assert adjustment["text_immutable"] is True


def test_rejects_cross_speaker_handoff_shift_above_bound() -> None:
    blocks = [
        SpeechBlock("block_1", "A", 0, 640, ("s1",), "a", "a", "a"),
        SpeechBlock("block_2", "B", 640, 5000, ("s2",), "b", "b", "b"),
    ]

    unchanged, shifted, adjustment = rebalance_cross_speaker_handoff(
        blocks,
        0,
        required_window_ms=1200,
        max_shift_ms=500,
    )

    assert unchanged == blocks[0]
    assert shifted is None
    assert adjustment["applied"] is False
