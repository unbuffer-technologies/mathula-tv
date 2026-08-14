from __future__ import annotations

from mathula_tv.tiktok_editor import (
    _canonicalize_editorial_rendered_block_references,
)


def _editorial_result(block_id: str) -> dict:
    return {
        "opening_boundary_block_id": block_id,
        "opening_payoff_block_id": block_id,
        "candidates": [
            {
                "selected_block_id": block_id,
                "evidence_block_ids": [block_id],
            }
        ],
    }


def test_block_0005_resolves_to_its_only_rendered_variant() -> None:
    result = _editorial_result("block_0005")

    audits = _canonicalize_editorial_rendered_block_references(
        result,
        full_rendered_transcript=[
            {"block_id": "block_0004_variant_compact"},
            {"block_id": "block_0005_variant_natural"},
            {"block_id": "block_0006_variant_concise"},
        ],
    )

    expected = "block_0005_variant_natural"
    assert result["opening_boundary_block_id"] == expected
    assert result["opening_payoff_block_id"] == expected
    assert result["candidates"][0]["selected_block_id"] == expected
    assert result["candidates"][0]["evidence_block_ids"] == [expected]
    assert len(audits) == 4
    assert all(item["reason"] == "single_rendered_variant_alias" for item in audits)


def test_invented_block_id_is_not_silently_accepted() -> None:
    result = _editorial_result("block_9999")

    audits = _canonicalize_editorial_rendered_block_references(
        result,
        full_rendered_transcript=[
            {"block_id": "block_0005_variant_natural"},
        ],
    )

    assert audits == []
    assert result["opening_boundary_block_id"] == "block_9999"


def test_ambiguous_rendered_variants_are_not_guessed() -> None:
    result = _editorial_result("block_0005")

    audits = _canonicalize_editorial_rendered_block_references(
        result,
        full_rendered_transcript=[
            {"block_id": "block_0005_variant_natural"},
            {"block_id": "block_0005_variant_compact"},
        ],
    )

    assert audits == []
    assert result["opening_boundary_block_id"] == "block_0005"
