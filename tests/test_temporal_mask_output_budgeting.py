from __future__ import annotations

import pytest

from mathula_tv.foundry_grok import GrokOutputTruncated
from mathula_tv.native_dub import (
    TEMPORAL_MASK_MAX_WINDOW_ITEMS,
    _estimate_pass2_output_tokens,
    _estimate_segment_output_tokens,
    _estimate_temporal_compact_chunk_tokens,
    _estimate_temporal_translate_window_tokens,
    _estimate_text_output_tokens,
    _retry_batch_call,
    _shrink_batch_to_budget,
    _TEMPORAL_MASK_OUTPUT_CALIBRATION,
)


# --- _estimate_text_output_tokens ---------------------------------------------------

def test_estimate_text_output_tokens_scales_with_content_length():
    small = _estimate_text_output_tokens("Yebo.")
    large = _estimate_text_output_tokens("Yebo. " * 400)
    assert large > small * 10


def test_estimate_text_output_tokens_has_a_floor_for_empty_or_tiny_text():
    assert _estimate_text_output_tokens("") == 18
    assert _estimate_text_output_tokens("Hi") >= 18


def test_estimate_segment_output_tokens_is_a_thin_wrapper():
    # Backward-compatible: the older (non-temporal-mask) Pass 2 path reads source_text
    # off a segment mapping directly -- this must keep working unchanged.
    item = {"source_text": "A reasonably long English sentence for estimation."}
    assert _estimate_segment_output_tokens(item) == _estimate_text_output_tokens(item["source_text"])


def test_estimate_pass2_output_tokens_sums_segments_plus_overhead():
    segments = [{"source_text": "One."}, {"source_text": "Two."}]
    expected = 80 + sum(_estimate_segment_output_tokens(item) for item in segments)
    assert _estimate_pass2_output_tokens(segments) == expected


def test_estimate_temporal_translate_window_tokens_sums_group_source_text():
    # Regression test for a real production gap: the raw character-based estimate alone
    # undershot a real observed 157 tokens/sentence by ~4x (see
    # _TEMPORAL_MASK_OUTPUT_CALIBRATION's definition), which let one real window balloon
    # to 154 of 155 sentences before the shrink logic ever thought it was near budget.
    # The temporal-mask estimators apply that calibration; the raw per-item estimate does not.
    batch = [{"source_text": "Short."}, {"source_text": "A somewhat longer sentence here."}]
    expected = 80 + sum(
        int(_estimate_text_output_tokens(g["source_text"]) * _TEMPORAL_MASK_OUTPUT_CALIBRATION) for g in batch
    )
    assert _estimate_temporal_translate_window_tokens(batch) == expected
    # And it must be meaningfully larger than the uncalibrated raw sum -- a calibration
    # factor of 1.0 would silently defeat the whole point of this fix.
    raw = 80 + sum(_estimate_text_output_tokens(g["source_text"]) for g in batch)
    assert _estimate_temporal_translate_window_tokens(batch) > raw


def test_estimate_temporal_compact_chunk_tokens_budgets_off_current_zulu_not_english():
    repairs = [
        {"english": "A very long English source sentence " * 10, "current_zulu": "Short."},
    ]
    # Must NOT scale with the long English text -- only the current (already-translated,
    # roughly-final-length) Zulu text is what a compaction attempt needs to reproduce.
    expected = 60 + int(_estimate_text_output_tokens("Short.") * _TEMPORAL_MASK_OUTPUT_CALIBRATION)
    assert _estimate_temporal_compact_chunk_tokens(repairs) == expected


# --- _shrink_batch_to_budget ---------------------------------------------------------
# No fixed sentence-count ceiling exists any more (the old TEMPORAL_MASK_SEMANTIC_
# WINDOW_SIZE=8 constant was removed): window/chunk size is purely a function of the
# real estimated token cost against the dynamic budget. Confirmed real production gap
# this closes: an 8-sentence window used only ~1,258 of ~6,553 available budget tokens
# (~157 tokens/sentence) -- capping at a small fixed count regardless of content left
# most of a call's real capacity unused every time.

def test_shrink_batch_grows_well_past_the_old_fixed_window_size_when_budget_allows():
    candidates = [{"source_text": f"Short sentence number {i}."} for i in range(30)]
    result = _shrink_batch_to_budget(
        candidates, budget=100_000, base_overhead=80,
        estimate_fn=lambda item: _estimate_text_output_tokens(item["source_text"]),
    )
    assert len(result) == 30


def test_temporal_mask_max_window_items_bounds_the_candidate_pool_even_with_huge_budget():
    # Mirrors the real Phase A/B call sites: the candidate pool passed to
    # _shrink_batch_to_budget is pre-sliced to TEMPORAL_MASK_MAX_WINDOW_ITEMS BEFORE the
    # estimate ever runs, so even a wildly generous (or wrongly-calibrated) budget can
    # never grow a window past this ceiling -- the real backstop for the production
    # failure where a miscalibrated estimate let one window swallow 154 of 155 sentences.
    candidates = [{"source_text": "Short."} for _ in range(200)]
    pool = candidates[:TEMPORAL_MASK_MAX_WINDOW_ITEMS]
    result = _shrink_batch_to_budget(
        pool, budget=10_000_000, base_overhead=80,
        estimate_fn=lambda item: _estimate_text_output_tokens(item["source_text"]),
    )
    assert len(result) == TEMPORAL_MASK_MAX_WINDOW_ITEMS


def test_shrink_batch_passes_through_unchanged_when_everything_fits():
    candidates = [{"source_text": "Short one."}, {"source_text": "Short two."}, {"source_text": "Short three."}]
    result = _shrink_batch_to_budget(
        candidates, budget=100_000, base_overhead=80,
        estimate_fn=lambda item: _estimate_text_output_tokens(item["source_text"]),
    )
    assert result == candidates


def test_shrink_batch_splits_off_an_unusually_long_item():
    long_text = "This is a much longer, denser testimony-style sentence. " * 20
    candidates = [{"source_text": "Short."}, {"source_text": long_text}, {"source_text": "Also short."}]
    tight_budget = 80 + _estimate_text_output_tokens("Short.") + 5  # room for the first item only
    result = _shrink_batch_to_budget(
        candidates, budget=tight_budget, base_overhead=80,
        estimate_fn=lambda item: _estimate_text_output_tokens(item["source_text"]),
    )
    assert result == [candidates[0]]


def test_shrink_batch_always_includes_at_least_the_first_candidate():
    # Even a budget too small for even ONE item must not return an empty batch --
    # progress must never stall; the truncation-retry safety net handles the fallout.
    huge_text = "Extremely long testimony passage. " * 200
    candidates = [{"source_text": huge_text}, {"source_text": "Short."}]
    result = _shrink_batch_to_budget(
        candidates, budget=50, base_overhead=80,
        estimate_fn=lambda item: _estimate_text_output_tokens(item["source_text"]),
    )
    assert result == [candidates[0]]


# --- _retry_batch_call ---------------------------------------------------------------

def test_retry_batch_call_returns_directly_when_nothing_truncates():
    def call(items):
        return {item: f"ok:{item}" for item in items}, {"output_tokens": 10}

    result, usage = _retry_batch_call(call, ["a", "b", "c"], operation="test_op", progress=None)
    assert result == {"a": "ok:a", "b": "ok:b", "c": "ok:c"}
    assert usage == {"output_tokens": 10}


def test_retry_batch_call_halves_and_merges_on_truncation():
    calls_made: list[list[str]] = []

    def call(items):
        calls_made.append(list(items))
        if len(items) > 1:
            raise GrokOutputTruncated("hit max_tokens", model="grok-test", input_tokens=10, output_tokens=999, attempts=1)
        item = items[0]
        return {item: f"fixed:{item}"}, {"output_tokens": 5, "attempts": 1}

    items = ["a", "b", "c", "d"]
    result, usage = _retry_batch_call(call, items, operation="test_op", progress=None)

    assert result == {"a": "fixed:a", "b": "fixed:b", "c": "fixed:c", "d": "fixed:d"}
    assert usage == {"output_tokens": 20, "attempts": 4}
    # First attempt is the full batch (and truncates); the rest are the recursive
    # halving down to single items that then succeed.
    assert calls_made[0] == items
    assert calls_made.count(["a"]) + calls_made.count(["b"]) + calls_made.count(["c"]) + calls_made.count(["d"]) == 4


def test_retry_batch_call_propagates_when_a_single_item_still_truncates():
    def call(items):
        raise GrokOutputTruncated("hit max_tokens", model="grok-test", input_tokens=10, output_tokens=999, attempts=1)

    with pytest.raises(GrokOutputTruncated):
        _retry_batch_call(call, ["only-one"], operation="test_op", progress=None)
