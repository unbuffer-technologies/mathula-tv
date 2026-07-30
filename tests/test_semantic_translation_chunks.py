import json

from mathula_tv.semantic_translation_chunks import (
    build_semantic_batch_requests,
    plan_semantic_ranges,
)


def _units(count=18):
    result = []
    cursor = 0
    for index in range(count):
        text = (f"Unit {index} contains a complete statement about the public story. " * 8).strip()
        result.append(
            {
                "unit_id": f"unit_{index:04d}",
                "speaker_id": "speaker_1" if index < count // 2 else "speaker_2",
                "start_ms": cursor,
                "end_ms": cursor + 4000,
                "source_text": text,
                "protected_spans": ["Mathula TV"] if index == 5 else [],
            }
        )
        cursor += 4200
    return result


def _request(count=18):
    return {
        "request_sha256": "f" * 64,
        "source_transcript_file_sha256": "a" * 64,
        "source_timeline_file_sha256": "b" * 64,
        "target_locale": "zu-ZA",
        "global_context": {"context": {"summary": "A public-affairs discussion"}},
        "global_protected_spans": ["Mathula TV"],
        "units": _units(count),
    }


def test_semantic_ranges_cover_every_unit_once_in_order():
    units = _units()
    ranges = plan_semantic_ranges(units, target_tokens=900, hard_tokens=1300)
    covered = [i for item in ranges for i in range(item.start, item.stop)]
    assert covered == list(range(len(units)))
    assert len(ranges) > 1


def test_every_batch_has_complete_bounded_context_spine_but_only_target_units():
    request = _request()
    batches, plan = build_semantic_batch_requests(
        request,
        target_tokens=900,
        hard_tokens=1300,
        context_units=2,
        context_spine_tokens=1200,
    )
    spine = plan["context_spine"]
    assert spine["complete_timeline_coverage"] is True
    assert spine["covered_unit_count"] == len(request["units"])
    assert sum(section["covered_unit_count"] for section in spine["timeline_sections"]) == len(
        request["units"]
    )
    requested = [unit["unit_id"] for batch in batches for unit in batch["units"]]
    assert requested == [unit["unit_id"] for unit in request["units"]]
    assert all(batch["translation_batch"]["max_output_tokens"] < 100000 for batch in batches)
    assert all("global_context" not in batch for batch in batches)


def test_multi_hour_context_spine_stays_bounded_while_covering_all_units():
    request = _request(1200)
    _, plan = build_semantic_batch_requests(
        request,
        target_tokens=3200,
        hard_tokens=4600,
        context_spine_tokens=4500,
    )
    spine = plan["context_spine"]
    assert spine["covered_unit_count"] == 1200
    assert len(spine["timeline_sections"]) < 100
    # Token estimation is conservative and JSON overhead varies; this guard catches
    # accidental per-unit expansion rather than enforcing an exact provider count.
    assert len(json.dumps(spine, ensure_ascii=False)) < 30000


def test_multivariant_output_budget_is_unit_aware_and_chunks_are_capped():
    request = _request(37)
    batches, _ = build_semantic_batch_requests(
        request,
        target_tokens=3200,
        hard_tokens=4600,
        context_units=2,
        context_spine_tokens=1200,
        max_units=18,
        max_output_tokens=32000,
    )
    sizes = [len(batch["units"]) for batch in batches]
    assert sum(sizes) == 37
    assert max(sizes) <= 18
    assert len(sizes) >= 3
    assert all(
        batch["translation_batch"]["max_output_tokens"] >= 12000
        for batch in batches
    )
    first = batches[0]["translation_batch"]
    expected = (
        3500
        + int(first["estimated_source_tokens"] * 5.0)
        + len(batches[0]["units"]) * 180
    )
    assert first["max_output_tokens"] == min(32000, max(12000, expected))
