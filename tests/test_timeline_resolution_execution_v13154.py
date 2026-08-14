from __future__ import annotations

from pathlib import Path

import pytest

from mathula_tv.direct_azure_dub import (
    PreparedTranslationVariant,
    SpeechBlock,
    apply_reviewed_triplet_silence_expansion,
)
from mathula_tv.timeline_collision_panel import (
    save_timeline_collision_resolution,
    upsert_timeline_collision_case,
)


def _block(block_id: str, speaker: str, start: int, end: int, text: str = "x") -> SpeechBlock:
    variants = tuple(
        PreparedTranslationVariant(
            variant_id=variant_id,
            spoken_text=f"{text}-{variant_id}",
            tts_text=f"{text}-{variant_id}",
            target_duration_ms=end - start,
            estimated_duration_ms=end - start,
            meaning_preserved=True,
        )
        for variant_id in ("natural", "concise", "compact")
    )
    return SpeechBlock(
        block_id=block_id,
        speaker_id=speaker,
        start_ms=start,
        end_ms=end,
        source_text=text,
        translated_text=f"{text}-natural",
        tts_text=f"{text}-natural",
        segment_ids=(block_id,),
        variants=variants,
        selected_variant_id="natural",
    )


def test_reviewed_fit_borrows_adjacent_silence_for_outer_speaker_conflict() -> None:
    blocks = [
        _block("block_0001", "S0", 0, 1000),
        _block("block_0002", "A", 1500, 3500),
        _block("block_0003", "B", 3500, 4000),
        _block("block_0004", "A", 4000, 6000),
        _block("block_0005", "S5", 6800, 7800),
    ]
    # The surrounding A turns alone require 4.6 seconds but their fixed outer
    # span is only 4.5 seconds. The old overlap-only solver cannot resolve this.
    required = {1: 2300, 2: 900, 3: 2300}
    unresolved = [{"block_index": 2, "applied": False, "reason": "surrounding_same_speaker_turns_would_overlap"}]
    resolutions = {
        "block_0003": {
            "strategy": "allow_overlap",
            "overlap_before_ms": 300,
            "overlap_after_ms": 300,
            "case_token": "token",
        }
    }
    adjusted, outcomes = apply_reviewed_triplet_silence_expansion(
        blocks, required, unresolved, resolutions
    )
    assert outcomes[0]["applied"] is True
    assert outcomes[0]["left_silence_borrow_ms"] + outcomes[0]["right_silence_borrow_ms"] > 0
    assert adjusted[1].start_ms >= blocks[0].end_ms
    assert adjusted[3].end_ms <= blocks[4].start_ms
    assert adjusted[1].duration_ms >= required[1]
    assert adjusted[2].duration_ms >= required[2]
    assert adjusted[3].duration_ms >= required[3]
    assert adjusted[1].end_ms <= adjusted[3].start_ms


def test_reviewed_fit_reports_insufficient_adjacent_silence() -> None:
    blocks = [
        _block("block_0001", "S0", 0, 1400),
        _block("block_0002", "A", 1500, 3500),
        _block("block_0003", "B", 3500, 4000),
        _block("block_0004", "A", 4000, 6000),
        _block("block_0005", "S5", 6100, 7100),
    ]
    required = {1: 3000, 2: 1000, 3: 3000}
    unresolved = [{"block_index": 2, "applied": False, "reason": "surrounding_same_speaker_turns_would_overlap"}]
    resolutions = {"block_0003": {"strategy": "manual_text", "manual_text": "short", "case_token": "token"}}
    _, outcomes = apply_reviewed_triplet_silence_expansion(
        blocks, required, unresolved, resolutions
    )
    assert outcomes[0]["applied"] is False
    assert outcomes[0]["reason"] == "reviewed_adjacent_silence_insufficient"


def _review_case() -> dict:
    return {
        "collision_id": "block_0003",
        "reason": "test",
        "shortfall_ms": 400,
        "triplet": [
            {"role": "previous", "block": {"block_id": "block_0002", "selected_variant_id": "natural", "translated_text": "prev"}},
            {"role": "current", "block": {"block_id": "block_0003", "selected_variant_id": "compact", "translated_text": "Umbhalo omude lapha"}},
            {"role": "following", "block": {"block_id": "block_0004", "selected_variant_id": "natural", "translated_text": "next"}},
        ],
    }


def _paths(tmp_path: Path):
    root = tmp_path / "direct_dub"
    return root / "review.json", root / "panel.html", root / "resolutions.json"


def test_manual_text_must_change(tmp_path: Path) -> None:
    review_path, panel_path, resolutions_path = _paths(tmp_path)
    review = upsert_timeline_collision_case(
        review_path=review_path,
        panel_path=panel_path,
        resolutions_path=resolutions_path,
        job_id="job123",
        case=_review_case(),
    )
    token = review["cases"][0]["case_token"]
    with pytest.raises(ValueError, match="unchanged"):
        save_timeline_collision_resolution(
            resolutions_path=resolutions_path,
            review_path=review_path,
            panel_path=panel_path,
            job_id="job123",
            collision_id="block_0003",
            strategy="manual_text",
            reviewed_by="Reviewer",
            case_token=token,
            manual_text="Umbhalo omude lapha",
        )


def test_variant_choice_must_change(tmp_path: Path) -> None:
    review_path, panel_path, resolutions_path = _paths(tmp_path)
    review = upsert_timeline_collision_case(
        review_path=review_path,
        panel_path=panel_path,
        resolutions_path=resolutions_path,
        job_id="job123",
        case=_review_case(),
    )
    token = review["cases"][0]["case_token"]
    with pytest.raises(ValueError, match="already active"):
        save_timeline_collision_resolution(
            resolutions_path=resolutions_path,
            review_path=review_path,
            panel_path=panel_path,
            job_id="job123",
            collision_id="block_0003",
            strategy="select_variants",
            reviewed_by="Reviewer",
            case_token=token,
            current_variant_id="compact",
        )
