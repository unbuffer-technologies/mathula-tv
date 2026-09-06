from __future__ import annotations

from types import SimpleNamespace

from mathula_tv.direct_azure_dub import (
    SpeechBlock,
    _select_azure_timing_attempt,
    attach_source_pause_anchors,
    build_source_pause_ssml_parts,
    enforce_source_word_speaker_alignment,
)
from mathula_tv.dub_mastering import build_mastering_override_plan
from mathula_tv.tts_ssml import (
    BreakPart,
    SafeSSMLBuilder,
    SSMLBounds,
    TextPart,
)


def _block(block_id: str, speaker: str, start: int, end: int) -> SpeechBlock:
    return SpeechBlock(
        block_id=block_id,
        speaker_id=speaker,
        start_ms=start,
        end_ms=end,
        segment_ids=(block_id,),
        source_text="source",
        translated_text="inkulumo",
        tts_text="inkulumo",
    )


def test_mastering_does_not_guess_rank_of_ai_finalized_variant() -> None:
    audit = {
        "round": 1,
        "blocks": [
            {
                "block_index": 59,
                "block_id": "block_0060_variant_ai_balanced_r1",
                "selected_variant_id": "ai_balanced_r1",
                "available_variant_ids": ["natural", "concise", "compact"],
                "production_gate_passed": False,
                "production_blockers": ["same_speaker_tempo_jump"],
                "effective_tempo_factor": 1.05,
                "same_speaker_tempo_jump": 0.19,
            }
        ],
    }

    plan = build_mastering_override_plan(job_id="job", audit=audit)

    assert plan["new_override_count"] == 0
    assert plan["new_overrides"] == []


def test_fitting_base_rate_beats_large_slowdown_for_trailing_room() -> None:
    attempts = [
        SimpleNamespace(rate_percent=-2, duration_ms=3754),
        SimpleNamespace(rate_percent=-12, duration_ms=4204),
    ]

    selected = _select_azure_timing_attempt(
        attempts,
        window_duration_ms=11790,
        base_rate_percent=-2,
    )

    assert selected.rate_percent == -2
    assert selected.duration_ms == 3754


def test_small_bounded_slowdown_can_still_reduce_trailing_room() -> None:
    attempts = [
        SimpleNamespace(rate_percent=-3, duration_ms=4362),
        SimpleNamespace(rate_percent=-7, duration_ms=4720),
    ]

    selected = _select_azure_timing_attempt(
        attempts,
        window_duration_ms=5460,
        base_rate_percent=-3,
    )

    assert selected.rate_percent == -7


def test_slow_source_word_gap_becomes_audited_pause_anchor() -> None:
    block = SpeechBlock(
        block_id="block_0008_variant_natural",
        speaker_id="SPEAKER_00",
        start_ms=77870,
        end_ms=81570,
        segment_ids=("unit_0013",),
        source_text="When God is about to bless you,",
        translated_text="Uma uNkulunkulu esezokubusisa",
        tts_text="Uma uNkulunkulu esezokubusisa",
    )
    source_words = [
        ("When", 77870, 78190),
        ("God", 78190, 78430),
        ("is", 78430, 78590),
        ("about", 78590, 79150),
        ("to", 80270, 80390),
        ("bless", 80430, 80830),
        ("you,", 80830, 81070),
    ]
    transcript = {
        "segments": [
            {
                "speaker_id": "SPEAKER_00",
                "source_word_ids": [
                    f"word_{index:04d}"
                    for index in range(1, len(source_words) + 1)
                ],
            }
        ],
        "words": [
            {
                "word_id": f"word_{index:04d}",
                "text": text,
                "start": start_ms / 1000,
                "end": end_ms / 1000,
            }
            for index, (text, start_ms, end_ms) in enumerate(
                source_words,
                start=1,
            )
        ],
    }

    adjusted, audit = attach_source_pause_anchors([block], transcript)

    assert len(audit) == 1
    assert len(adjusted[0].source_pause_anchors) == 1
    assert adjusted[0].source_speech_start_ms == 77870
    assert adjusted[0].source_speech_end_ms == 81070
    anchor = adjusted[0].source_pause_anchors[0]
    assert anchor["source_gap_ms"] == 1120
    assert anchor["requested_pause_ms"] == 1020
    assert anchor["source_progress"] == 0.571428571
    assert anchor["left_word"]["text"] == "about"
    assert anchor["right_word"]["text"] == "to"


def test_source_pause_uses_underfill_inside_target_instead_of_at_end() -> None:
    text = "Uma uNkulunkulu esezokubusisa"
    parts, audit = build_source_pause_ssml_parts(
        text,
        [
            {
                "source_gap_ms": 1120,
                "requested_pause_ms": 1020,
                "source_progress": 4 / 7,
                "left_word": {"text": "about"},
                "right_word": {"text": "to"},
            }
        ],
        pause_budget_ms=825,
    )

    assert audit["applied"] is True
    assert audit["total_added_pause_ms"] == 825
    assert audit["maximum_added_pause_ms"] == 825
    assert audit["approved_text_changed"] is False
    assert [type(part) for part in parts] == [TextPart, BreakPart, TextPart]
    assert parts[0].text == "Uma uNkulunkulu "
    assert parts[1].duration_ms == 825
    assert parts[2].text == "esezokubusisa"
    assert "".join(
        part.text for part in parts if isinstance(part, TextPart)
    ) == text


def test_source_pause_budget_never_overflows_with_multiple_gaps() -> None:
    parts, audit = build_source_pause_ssml_parts(
        "okokuqala okwesibili okwesithathu okwesine",
        [
            {"source_progress": 0.25, "requested_pause_ms": 1000},
            {"source_progress": 0.50, "requested_pause_ms": 1000},
            {"source_progress": 0.75, "requested_pause_ms": 1000},
        ],
        pause_budget_ms=501,
    )

    pauses = [part.duration_ms for part in parts if isinstance(part, BreakPart)]
    assert len(pauses) == 2
    assert all(duration_ms >= 250 for duration_ms in pauses)
    assert sum(pauses) == audit["total_added_pause_ms"]
    assert audit["total_added_pause_ms"] <= 501


def test_long_source_pause_is_split_into_azure_safe_breaks() -> None:
    parts, audit = build_source_pause_ssml_parts(
        "inkulumo yokuqala inkulumo yesibili",
        [
            {
                "source_progress": 0.5,
                "source_gap_ms": 2360,
                "requested_pause_ms": 2260,
                "pause_kind": "island_boundary",
            }
        ],
        pause_budget_ms=2260,
        pause_max_ms=2000,
    )

    pauses = [
        part.duration_ms for part in parts if isinstance(part, BreakPart)
    ]
    assert pauses == [2000, 260]
    assert sum(pauses) == 2260
    assert audit["total_added_pause_ms"] == 2260
    assert audit["maximum_added_pause_ms"] == 2260
    assert audit["maximum_ssml_break_ms"] == 2000
    assert audit["insertions"][0]["ssml_break_durations_ms"] == [2000, 260]

    document = SafeSSMLBuilder(
        allowed_voices=("zu-ZA-ThandoNeural",),
        bounds=SSMLBounds(total_pause_max_ms=10_000),
    ).build(
        parts,
        voice="zu-ZA-ThandoNeural",
        language="zu-ZA",
    )
    assert '<break time="2000ms" />' in document.xml
    assert '<break time="260ms" />' in document.xml


def test_terminal_source_pause_does_not_double_azure_sentence_pause() -> None:
    parts, audit = build_source_pause_ssml_parts(
        "UZibi uhola le nhlamba. Ngakho siyaqhubeka.",
        [
            {
                "source_progress": 0.5,
                "source_gap_ms": 1740,
                "requested_pause_ms": 1640,
                "pause_kind": "island_boundary",
            }
        ],
        pause_budget_ms=1640,
    )

    pauses = [part.duration_ms for part in parts if isinstance(part, BreakPart)]
    assert pauses == [900]
    assert audit["total_added_pause_ms"] == 900
    assert audit["insertions"][0]["uncapped_duration_ms"] == 1640
    assert audit["insertions"][0]["terminal_punctuation_cap_applied"] is True


def test_source_pause_snaps_to_nearby_punctuation_boundary() -> None:
    _parts, audit = build_source_pause_ssml_parts(
        "i-Dee Ey, vumelani i-Dee Ey, gxilani",
        [{"source_progress": 0.22, "requested_pause_ms": 900}],
        pause_budget_ms=700,
    )

    insertion = audit["insertions"][0]
    assert insertion["target_left_text"] == "Ey,"
    assert insertion["target_right_text"] == "vumelani"


def test_word_alignment_preserves_safe_measured_overlap_capacity() -> None:
    source = [
        _block("block_0060", "A", 797810, 803200),
        _block("block_0061", "B", 803200, 804180),
        _block("block_0062", "A", 804180, 805320),
    ]
    fitted = [
        _block("block_0060", "A", 797810, 803390),
        _block("block_0061", "B", 803102, 804932),
        _block("block_0062", "A", 803932, 805320),
    ]
    anchors = [
        {
            "boundary_index": 0,
            "compact_handoff": True,
            "handoff_min_ms": 803140,
            "handoff_max_ms": 803200,
            "tolerance_ms": 250,
        },
        {
            "boundary_index": 1,
            "compact_handoff": False,
            "left_end_min_ms": 803620,
            "left_end_max_ms": 804120,
            "right_start_min_ms": 804180,
            "right_start_max_ms": 804680,
            "tolerance_ms": 250,
        },
    ]

    adjusted, corrections = enforce_source_word_speaker_alignment(
        fitted,
        source,
        anchors,
        required_windows_ms={0: 5580, 1: 1830, 2: 1388},
    )

    assert adjusted == fitted
    assert [item["boundary_overlap_ms"] for item in corrections] == [288, 1000]
    assert all(
        item["reason"] == "measured_capacity_fit_takes_precedence"
        for item in corrections
    )
