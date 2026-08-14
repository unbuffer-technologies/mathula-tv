from __future__ import annotations

from mathula_tv.direct_azure_dub import (
    PreparedTranslationVariant,
    SpeechBlock,
    rebalance_edge_timeline_with_neighbour_surplus,
)


def _block(
    block_id: str,
    speaker_id: str,
    start_ms: int,
    end_ms: int,
    text: str,
) -> SpeechBlock:
    variant = PreparedTranslationVariant(
        variant_id="compact",
        spoken_text=text,
        tts_text=text,
        meaning_preserved=True,
    )
    return SpeechBlock(
        block_id=block_id,
        speaker_id=speaker_id,
        start_ms=start_ms,
        end_ms=end_ms,
        segment_ids=(block_id,),
        source_text=text,
        translated_text=text,
        tts_text=text,
        variants=(variant,),
        selected_variant_id="compact",
    )


def test_supplied_final_block_uses_previous_measured_surplus_without_ai() -> None:
    previous_text = (
        "Kuyingxenye yesixwayiso kuRam Sami, ukuthi uGabinde "
        "wayengathembekile, uma ngikhumbula kahle."
    )
    blocks = [
        _block(
            "block_0008_variant_compact",
            "SPEAKER_02",
            417_070,
            443_170,
            previous_text,
        ),
        _block(
            "block_0009_variant_compact",
            "SPEAKER_01",
            443_350,
            444_440,
            "Siyaqhubeka.",
        ),
    ]
    # Exact Azure evidence from job 88c279a25bd64e069cbb0e58d01a5e2c:
    # block 8 measured 22,768ms; block 9 measured 1,444ms. At the 1.03
    # bounded ratio their required windows are 22,105ms and 1,402ms.
    adjusted, audits = rebalance_edge_timeline_with_neighbour_surplus(
        blocks,
        {0: 22_105, 1: 1_402},
        {1},
        max_boundary_shift_ms=500,
    )

    assert len(audits) == 1
    assert audits[0]["applied"] is True
    assert audits[0]["deficit_resolved_ms"] == 312
    assert audits[0]["source_silence_consumed_ms"] == 180
    assert audits[0]["neighbour_boundary_contribution_ms"] == 132
    assert audits[0]["neighbour_measured_surplus_ms"] == 3_995
    assert audits[0]["ai_called"] is False
    assert adjusted[0].end_ms == 443_038
    assert adjusted[1].start_ms == 443_038
    assert adjusted[0].duration_ms == 25_968
    assert adjusted[1].duration_ms == 1_402
    assert adjusted[0].translated_text == previous_text
    assert adjusted[1].translated_text == "Siyaqhubeka."


def test_edge_fit_rejects_when_neighbour_cannot_spare_the_boundary() -> None:
    blocks = [
        _block("block_0001", "SPEAKER_00", 0, 1_000, "Okokuqala."),
        _block("block_0002", "SPEAKER_01", 1_000, 2_000, "Okwesibili."),
    ]

    adjusted, audits = rebalance_edge_timeline_with_neighbour_surplus(
        blocks,
        {0: 1_000, 1: 1_300},
        {1},
        max_boundary_shift_ms=500,
    )

    assert adjusted == blocks
    assert audits[0]["applied"] is False
    assert "neighbour_below_required_window" in audits[0]["invariant_failures"]


def test_first_block_can_use_following_measured_surplus_symmetrically() -> None:
    blocks = [
        _block("block_0001", "SPEAKER_00", 0, 1_000, "Okokuqala."),
        _block("block_0002", "SPEAKER_01", 1_150, 3_000, "Okwesibili."),
    ]

    adjusted, audits = rebalance_edge_timeline_with_neighbour_surplus(
        blocks,
        {0: 1_300, 1: 1_400},
        {0},
        max_boundary_shift_ms=500,
    )

    assert audits[0]["applied"] is True
    assert audits[0]["source_silence_consumed_ms"] == 150
    assert audits[0]["neighbour_boundary_contribution_ms"] == 150
    assert adjusted[0].end_ms == 1_300
    assert adjusted[1].start_ms == 1_300
    assert adjusted[1].end_ms == 3_000


def test_uploaded_final_collision_fits_with_compact_edge_policy() -> None:
    blocks = [
        _block(
            "block_0014_variant_compact",
            "SPEAKER_01",
            405_550,
            439_630,
            "Umbiko wangaphambilini.",
        ),
        _block(
            "block_0015_variant_compact",
            "SPEAKER_00",
            439_630,
            445_057,
            "Barry Bateman, we-AfriForum Private Prosecution Unit. Siyabonga ngokusijoyina.",
        ),
    ]
    # Exact measurements from de4edf2950444f5ca0b277b5a6954e27.
    # The compact final block needs ceil(6289/1.08)=5824ms, while its
    # neighbour has ample measured surplus (33562ms inside a 34080ms window).
    adjusted, audits = rebalance_edge_timeline_with_neighbour_surplus(
        blocks,
        {0: 32_585, 1: 5_824},
        {1},
        max_boundary_shift_ms=500,
    )

    assert audits[0]["applied"] is True
    assert audits[0]["deficit_resolved_ms"] == 397
    assert adjusted[0].end_ms == 439_233
    assert adjusted[1].start_ms == 439_233
    assert adjusted[0].duration_ms == 33_683
    assert adjusted[1].duration_ms == 5_824
