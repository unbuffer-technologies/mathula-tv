from __future__ import annotations

from types import SimpleNamespace

import pytest

from mathula_tv import tiktok_editor


class _LateBoundaryProvider:
    def complete_structured(self, request):
        scores = {
            "visual_impact": 8,
            "curiosity": 8,
            "specificity": 8,
            "stakes": 8,
            "immediacy": 8,
            "audience_relevance": 8,
        }
        return SimpleNamespace(
            data={
                "opening_boundary_block_id": "block_0003",
                "opening_payoff_block_id": "block_0003",
                "opening_relationship": "standalone",
                "opening_boundary_rationale": "The third block contains the payoff.",
                "candidates": [
                    {
                        "candidate_id": f"candidate_{index}",
                        "selected_block_id": "block_0003",
                        "hook_text": f"Ubufakazi obusha buveza ukuphikisana {index}",
                        "rationale": "Grounded in the retained third block.",
                        "confidence": 0.9,
                        "grounded": True,
                        "preserves_attribution": True,
                        "no_sensationalism": True,
                        "scores": scores,
                        "human_review_flags": [],
                    }
                    for index in range(1, 4)
                ],
            },
            metadata=SimpleNamespace(to_dict=lambda: {"provider": "fake"}),
        )


def _blocks() -> list[dict[str, object]]:
    return [
        {
            "block_id": "block_0001",
            "speaker_id": "HOST",
            "start_ms": 1_000,
            "end_ms": 5_000,
            "source_text": "Welcome to the programme.",
            "translated_text": "Siyanamukela ohlelweni.",
        },
        {
            "block_id": "block_0002",
            "speaker_id": "HOST",
            "start_ms": 8_000,
            "end_ms": 12_000,
            "source_text": "The record now turns to the central allegation.",
            "translated_text": "Umbhalo usuphendukela ezinsolweni eziyinhloko.",
        },
        {
            "block_id": "block_0003",
            "speaker_id": "WITNESS",
            "start_ms": 30_000,
            "end_ms": 40_000,
            "source_text": "The document contradicts the earlier account.",
            "translated_text": "Umbhalo uyayiphikisa incazelo yangaphambilini.",
        },
    ]


def test_real_late_opening_is_moved_to_nearest_preceding_candidate() -> None:
    result = tiktok_editor.select_tiktok_hook(
        provider=_LateBoundaryProvider(),
        job_id="late-opening-boundary",
        target_language="zu-ZA",
        blocks=_blocks(),
        seo={},
        source_duration_seconds=120.0,
        maximum_intro_cut_seconds=10.0,
    )

    assert result["ai_requested_opening_block_id"] == "block_0003"
    assert result["opening_anchor_block_id"] == "block_0002"
    assert result["opening_payoff_block_id"] == "block_0003"
    assert result["cut_start_seconds"] == pytest.approx(8.0)
    guard = result["opening_candidate_boundary_guard"]
    assert guard["recovery_applied"] is True
    assert guard["effective_candidate_opening_block_id"] == "block_0002"
    assert guard["retained_extra_block_count"] == 1
    assert (
        "opening_boundary_remapped_to_preceding_eligible_candidate"
        in result["human_review_flags"]
    )
    assert (
        "opening_boundary_remapped_to_preceding_eligible_candidate"
        in result["editorial_seo"]["human_review_flags"]
    )


def test_source_block_aliases_resolve_to_the_single_rendered_variant(
    tmp_path,
) -> None:
    rendered_blocks = [
        {
            **block,
            "block_id": f"{block['block_id']}_variant_natural",
        }
        for block in _blocks()
    ]

    provider = _LateBoundaryProvider()
    checkpoint = tmp_path / "editorial_response_checkpoint.json"
    selection_args = dict(
        provider=provider,
        job_id="rendered-variant-alias",
        target_language="zu-ZA",
        blocks=rendered_blocks,
        seo={},
        source_duration_seconds=120.0,
        maximum_intro_cut_seconds=50.0,
        response_checkpoint_path=checkpoint,
    )
    result = tiktok_editor.select_tiktok_hook(**selection_args)

    assert result["ai_requested_opening_block_id"] == "block_0003"
    assert result["ai_requested_payoff_block_id"] == "block_0003"
    assert result["opening_anchor_block_id"] == "block_0003_variant_natural"
    assert result["opening_payoff_block_id"] == "block_0003_variant_natural"
    assert result["selected_block_id"] == "block_0003_variant_natural"
    assert result["opening_payoff_candidate"]["block_id"] == (
        "block_0003_variant_natural"
    )
    canonicalization = result["rendered_block_id_canonicalization"]
    assert canonicalization["applied"] is True
    assert canonicalization["canonicalization_count"] == 8
    assert {
        entry["rendered_block_id"] for entry in canonicalization["entries"]
    } == {"block_0003_variant_natural"}
    ranked = result["engagement_ranking"]["ranked_candidates"]
    assert all(
        candidate["selected_block_id"] == "block_0003_variant_natural"
        for candidate in ranked
    )
    assert all(
        candidate["evidence_block_ids"] == ["block_0003_variant_natural"]
        for candidate in ranked
    )

    def should_not_call_again(_request):
        raise AssertionError("saved editorial response should have been reused")

    provider.complete_structured = should_not_call_again
    reused = tiktok_editor.select_tiktok_hook(**selection_args)
    assert reused["opening_anchor_block_id"] == "block_0003_variant_natural"
    assert reused["editorial_response_checkpoint"]["reused"] is True


def test_invented_opening_boundary_still_fails_closed() -> None:
    provider = _LateBoundaryProvider()
    original = provider.complete_structured

    def _invalid(request):
        response = original(request)
        response.data["opening_boundary_block_id"] = "block_9999"
        return response

    provider.complete_structured = _invalid

    with pytest.raises(ValueError, match="does not exist in the rendered transcript"):
        tiktok_editor.select_tiktok_hook(
            provider=provider,
            job_id="invented-opening-boundary",
            target_language="zu-ZA",
            blocks=_blocks(),
            seo={},
            source_duration_seconds=120.0,
            maximum_intro_cut_seconds=10.0,
        )
