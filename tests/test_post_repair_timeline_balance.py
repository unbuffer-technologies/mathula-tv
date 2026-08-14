from __future__ import annotations

import inspect

import mathula_tv.direct_azure_dub as direct_dub


def test_post_claude_repair_merge_layer_is_removed() -> None:
    assert not hasattr(
        direct_dub, "merge_timing_repair_candidates_for_timeline_balance"
    )
    assert not hasattr(direct_dub, "load_cached_best_failed_timing_candidate")


def test_timeline_strategy_uses_measured_three_sentence_variants() -> None:
    source = inspect.getsource(direct_dub.DirectAzureDubRenderer.render)
    assert "azure_measured_three_sentence_variants" in source
    assert "choose_three_block_variant_balance" in source
    assert "rebalance_sandwiched_interjection_timeline" in source
    assert "repair_turn" not in source
