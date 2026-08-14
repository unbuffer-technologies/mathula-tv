from __future__ import annotations

from mathula_tv.direct_azure_dub import (
    PreparedTranslationVariant,
    SpeechBlock,
    _collision_allowed_strategies,
)
from mathula_tv.timeline_collision_panel import render_timeline_collision_panel


def _block(block_id: str, start_ms: int, end_ms: int) -> dict[str, object]:
    return {
        "block_id": block_id,
        "speaker_id": "SPEAKER_01",
        "start_ms": start_ms,
        "end_ms": end_ms,
        "source_text": f"Source {block_id}",
        "translated_text": f"Selected {block_id}",
        "variants": [
            {
                "variant_id": "compact",
                "spoken_text": f"Compact {block_id}",
                "estimated_duration_ms": end_ms - start_ms,
            }
        ],
    }


def _review(triplet: list[dict[str, object]]) -> dict[str, object]:
    return {
        "job_id": "job-boundary",
        "cases": [
            {
                "collision_id": "block_0015",
                "case_token": "job-boundary:block_0015:r1:test",
                "status": "open",
                "reason": "genuine_cross_speaker_timeline_collision",
                "shortfall_ms": 250,
                "available_duration_ms": 1000,
                "measured_duration_ms": 1250,
                "allowed_strategies": ["manual_text", "select_variants"],
                "triplet": triplet,
            }
        ],
    }


def test_final_block_panel_shows_end_boundary_instead_of_unknown() -> None:
    page = render_timeline_collision_panel(
        _review(
            [
                {"role": "previous", "block": _block("block_0014", 0, 1000)},
                {"role": "current", "block": _block("block_0015", 1000, 2000)},
            ]
        )
    )

    assert "Following: end of timeline" in page
    assert "There is no following speech block after this collision." in page
    assert "unknown" not in page.casefold()
    assert 'name="previous_variant_id"' in page
    assert 'name="current_variant_id"' in page
    assert 'name="following_variant_id"' not in page
    assert 'value="allow_overlap"' not in page
    assert "force prepared variants across the available blocks" in page


def test_first_block_panel_preserves_zero_and_shows_start_boundary() -> None:
    page = render_timeline_collision_panel(
        _review(
            [
                {"role": "current", "block": _block("block_0001", 0, 1000)},
                {"role": "following", "block": _block("block_0002", 1000, 2000)},
            ]
        )
    )

    assert "Previous: start of timeline" in page
    assert "Window:</strong> 0–1000 ms" in page
    assert "unknown" not in page.casefold()
    assert 'name="previous_variant_id"' not in page
    assert 'name="current_variant_id"' in page
    assert 'name="following_variant_id"' in page


def test_missing_current_metadata_is_explicit_not_fabricated() -> None:
    page = render_timeline_collision_panel(_review([]))

    assert "Current: block data unavailable" in page
    assert "unknown" not in page.casefold()


def _speech_block(block_id: str, start_ms: int, end_ms: int) -> SpeechBlock:
    variant = PreparedTranslationVariant(
        variant_id="compact",
        spoken_text=block_id,
        tts_text=block_id,
    )
    return SpeechBlock(
        block_id=block_id,
        speaker_id="SPEAKER_01",
        start_ms=start_ms,
        end_ms=end_ms,
        segment_ids=(block_id,),
        source_text=block_id,
        translated_text=block_id,
        tts_text=block_id,
        variants=(variant,),
        selected_variant_id="compact",
    )


def test_renderer_does_not_offer_impossible_boundary_overlap() -> None:
    blocks = [
        _speech_block("block_0001", 0, 1000),
        _speech_block("block_0002", 1000, 2000),
        _speech_block("block_0003", 2000, 3000),
    ]

    assert _collision_allowed_strategies(blocks, 0) == [
        "manual_text",
        "select_variants",
    ]
    assert _collision_allowed_strategies(blocks, 1) == [
        "allow_overlap",
        "manual_text",
        "select_variants",
    ]
    assert _collision_allowed_strategies(blocks, 2) == [
        "manual_text",
        "select_variants",
    ]
