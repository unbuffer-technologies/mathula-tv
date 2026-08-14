from pathlib import Path

from mathula_tv.direct_azure_dub import (
    PreparedTranslationVariant,
    SpeechBlock,
    apply_timeline_collision_resolutions,
    rebalance_sandwiched_interjection_timeline,
)
from mathula_tv.timeline_collision_panel import (
    load_timeline_collision_review,
    save_timeline_collision_resolution,
    upsert_timeline_collision_case,
)


def variant(variant_id: str, text: str, ms: int) -> PreparedTranslationVariant:
    return PreparedTranslationVariant(
        variant_id=variant_id,
        spoken_text=text,
        tts_text=text,
        target_duration_ms=ms,
        estimated_duration_ms=ms,
    )


def block(block_id: str, speaker: str, start: int, end: int) -> SpeechBlock:
    variants = (
        variant("natural", f"{block_id} natural", end - start),
        variant("concise", f"{block_id} concise", max(1, end - start - 100)),
        variant("compact", f"{block_id} compact", max(1, end - start - 200)),
    )
    return SpeechBlock(
        block_id=block_id,
        speaker_id=speaker,
        start_ms=start,
        end_ms=end,
        segment_ids=(block_id,),
        source_text=f"source {block_id}",
        translated_text=variants[0].spoken_text,
        tts_text=variants[0].tts_text,
        variants=variants,
        selected_variant_id="natural",
    )


def test_manual_text_is_direct_dub_overlay_only() -> None:
    blocks = [block("block_0001", "A", 0, 1000)]
    adjusted, audits = apply_timeline_collision_resolutions(
        blocks,
        {
            "block_0001": {
                "strategy": "manual_text",
                "manual_text": "Umbhalo omfushane.",
                "reviewed_by": "Reviewer",
            }
        },
    )
    assert blocks[0].translated_text == "block_0001 natural"
    assert adjusted[0].translated_text == "Umbhalo omfushane."
    assert adjusted[0].selected_variant_id == "manual"
    assert audits[0]["authoritative_translation_changed"] is False


def test_force_prepared_variants_across_triplet() -> None:
    blocks = [
        block("block_0001", "A", 0, 1000),
        block("block_0002", "B", 1000, 1800),
        block("block_0003", "A", 1800, 3000),
    ]
    adjusted, _ = apply_timeline_collision_resolutions(
        blocks,
        {
            "block_0002": {
                "strategy": "select_variants",
                "variants": {
                    "previous": "concise",
                    "current": "compact",
                    "following": "concise",
                },
            }
        },
    )
    assert [value.selected_variant_id for value in adjusted] == [
        "concise",
        "compact",
        "concise",
    ]
    assert all(len(value.variants) == 1 for value in adjusted)


def test_human_overlap_resolution_expands_local_budget() -> None:
    blocks = [
        block("block_0001", "A", 0, 1000),
        block("block_0002", "B", 1000, 2000),
        block("block_0003", "A", 2000, 3000),
    ]
    adjusted, audits = rebalance_sandwiched_interjection_timeline(
        blocks,
        {0: 1000, 1: 1500, 2: 1000},
        max_total_overlap_ms=0,
        max_boundary_overlap_ms=0,
        manual_overlap_resolutions={
            "block_0002": {
                "strategy": "allow_overlap",
                "overlap_before_ms": 250,
                "overlap_after_ms": 250,
                "max_total_overlap_ms": 500,
                "reviewed_by": "Reviewer",
            }
        },
    )
    assert audits[0]["applied"] is True
    assert audits[0]["overlap_policy"] == "human_approved_commission_interjection_overlap"
    assert audits[0]["total_overlap_ms"] == 500
    assert adjusted[1].duration_ms == 1500


def test_review_panel_saves_resolution(tmp_path: Path) -> None:
    review = tmp_path / "review.json"
    panel = tmp_path / "panel.html"
    resolutions = tmp_path / "resolutions.json"
    upsert_timeline_collision_case(
        review_path=review,
        panel_path=panel,
        resolutions_path=resolutions,
        job_id="job-1",
        case={
            "collision_id": "block_0055",
            "reason": "collision",
            "shortfall_ms": 456,
            "triplet": [],
        },
    )
    saved = save_timeline_collision_resolution(
        resolutions_path=resolutions,
        review_path=review,
        panel_path=panel,
        job_id="job-1",
        collision_id="block_0055",
        strategy="allow_overlap",
        reviewed_by="Clerence",
        overlap_before_ms=228,
        overlap_after_ms=228,
    )
    assert saved["max_total_overlap_ms"] == 456
    payload = load_timeline_collision_review(review)
    assert payload["cases"][0]["status"] == "resolution_saved"
    assert "Option 1" in panel.read_text(encoding="utf-8")
