from mathula_tv.native_dub import (
    _build_rolling_syllable_batches,
    _estimate_zulu_syllables,
    _normalize_pass2_by_id,
    _rolling_refinement_payload,
    _rolling_syllable_budget,
    _annotate_rolling_syllables,
)


def _seg(i: int, text: str, start: int, end: int):
    return {
        "segment_id": f"seg-{i:05d}",
        "speaker_id": "SPEAKER_00",
        "source_text": text,
        "start_ms": start,
        "end_ms": end,
    }


def test_affine_syllable_budget_marks_sub_overhead_window_impossible():
    budget = _rolling_syllable_budget(
        _seg(1, "Paragraph four.", 0, 560),
        preferred_raw_speed_percent=6,
        content_syllables_per_second=5.3,
        azure_utterance_overhead_ms=900,
        tolerance_percent=12,
    )
    assert budget["target_raw_ms"] == 594
    assert budget["budget_impossible"] is True
    assert budget["target_syllables"] == 1


def test_long_window_gets_duration_derived_not_english_count_budget():
    budget = _rolling_syllable_budget(
        _seg(1, "Any English wording can be here.", 0, 6720),
        preferred_raw_speed_percent=6,
        content_syllables_per_second=5.3,
        azure_utterance_overhead_ms=900,
        tolerance_percent=12,
    )
    assert budget["target_raw_ms"] == 7123
    assert 30 <= budget["target_syllables"] <= 35
    assert budget["min_syllables"] < budget["target_syllables"] < budget["max_syllables"]


def test_estimator_counts_numeric_and_acronym_tokens_as_spoken_content():
    plain = _estimate_zulu_syllables("Ngikhuluma naye.")
    enriched = _estimate_zulu_syllables("Ngikhuluma naye ngo-2024 e-SAPS.")
    assert enriched > plain


def test_rolling_batches_are_ordered_and_bounded():
    segments = [_seg(i, "short source", i * 1000, i * 1000 + 900) for i in range(1, 20)]
    batches = _build_rolling_syllable_batches(
        segments,
        max_segments=8,
        max_output_tokens=8192,
    )
    assert all(len(batch) <= 8 for batch in batches)
    flattened = [item["segment_id"] for batch in batches for item in batch]
    assert flattened == [item["segment_id"] for item in segments]


def test_rolling_response_is_reconstructed_by_segment_id_not_model_order():
    segments = [_seg(1, "one", 0, 1000), _seg(2, "two", 1000, 2000)]
    data = {
        "translations": [
            {"segment_id": "seg-00002", "spoken_text": "Okwesibili."},
            {"segment_id": "seg-00001", "spoken_text": "Okokuqala."},
        ]
    }
    normalized = _normalize_pass2_by_id(data, segments)
    assert [item["segment_id"] for item in normalized] == ["seg-00001", "seg-00002"]


def test_refinement_only_targets_severe_deterministic_misses():
    source = [_seg(1, "one", 0, 4000), _seg(2, "two", 4000, 8000)]
    first = [
        {"segment_id": "seg-00001", "source_text": "one", "spoken_text": "Yebo."},
        {"segment_id": "seg-00002", "source_text": "two", "spoken_text": "Lena yinkulumo ende kakhulu kakhulu kakhulu kakhulu kakhulu."},
    ]
    annotated = _annotate_rolling_syllables(
        first,
        source,
        preferred_raw_speed_percent=6,
        content_syllables_per_second=5.3,
        azure_utterance_overhead_ms=900,
        tolerance_percent=12,
    )
    rows, severe = _rolling_refinement_payload(annotated)
    assert severe >= 1
    assert any(row["revise"] for row in rows)
