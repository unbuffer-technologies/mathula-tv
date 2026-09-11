import json
from pathlib import Path

import pytest

from mathula_tv.ai_provider import create_production_ai_provider
from mathula_tv.manual_ai import ManualAIResponseRequired, pending_manual_requests
from mathula_tv.semantic_translation_chunks import build_single_handoff_batch_request, sha256_json


def _full_request(unit_count: int = 100):
    units = []
    cursor = 0
    for index in range(1, unit_count + 1):
        start = cursor
        end = start + 5000
        cursor = end
        units.append(
            {
                "unit_id": f"unit_{index:04d}",
                "speaker_id": "SPEAKER_00",
                "start_ms": start,
                "end_ms": end,
                "available_duration_ms": end - start,
                "overlap_group": None,
                "source_text": (
                    "This is a representative source sentence for the complete "
                    f"manual translation handoff number {index}."
                ),
                "protected_spans": [],
                "duration_targets": {
                    "natural_target_duration_ms": 4500,
                    "concise_target_duration_ms": 3900,
                    "compact_target_duration_ms": 3300,
                },
            }
        )
    request = {
        "schema_version": "mathula.translation.request.v1",
        "prompt_version": "mathula-multivariant-translation-prompt-v6-natural-code-switch-grammar",
        "job_id": "manual-single-handoff-test",
        "source_language": "en-ZA",
        "target_language": "isiZulu",
        "target_locale": "zu-ZA",
        "translation_strategy": "contextual_code_switching",
        "source_transcript_sha256": "a" * 64,
        "source_transcript_file_sha256": "b" * 64,
        "source_timeline_kind": "dubbing_units",
        "source_timeline_sha256": "c" * 64,
        "source_timeline_file_sha256": "d" * 64,
        "global_context": {},
        "global_protected_spans": [],
        "units": units,
    }
    request["request_sha256"] = sha256_json(request)
    return request


def test_single_handoff_contains_every_unit_once_without_context_duplication():
    request = _full_request(100)
    batches, plan = build_single_handoff_batch_request(
        request,
        context_spine_tokens=2000,
        max_output_tokens=100_000,
    )

    assert len(batches) == 1
    batch = batches[0]
    assert [unit["unit_id"] for unit in batch["units"]] == [
        unit["unit_id"] for unit in request["units"]
    ]
    translation_batch = batch["translation_batch"]
    assert translation_batch["chunk_index"] == 1
    assert translation_batch["chunk_count"] == 1
    assert translation_batch["unit_count"] == 100
    assert translation_batch["context_units"] == []
    assert translation_batch["manual_single_handoff"] is True
    assert translation_batch["max_output_tokens"] > 12_000
    assert plan["mode"] == "manual_single_handoff"
    assert plan["chunk_count"] == 1


def test_manual_provider_does_not_reapply_cloud_12k_output_cap(tmp_path):
    request = _full_request(100)
    batch = build_single_handoff_batch_request(
        request,
        context_spine_tokens=2000,
        max_output_tokens=100_000,
    )[0][0]

    root = tmp_path / "manual_ai"
    provider = create_production_ai_provider(
        {
            "MATHULA_TV_AI_PROVIDER": "manual-chatgpt",
            "MATHULA_TV_MANUAL_AI_DIR": str(root),
        }
    )
    with pytest.raises(ManualAIResponseRequired):
        provider.translate_multivariant(batch)

    pending = pending_manual_requests(root)
    assert len(pending) == 1
    exported = json.loads(Path(pending[0]["request_path"]).read_text(encoding="utf-8"))
    assert exported["payload"]["translation_batch"]["manual_single_handoff"] is True
    assert exported["generation_hints"]["max_output_tokens"] > 12_000
    assert len(exported["payload"]["units"]) == 100
    assert exported["payload"]["translation_batch"]["context_units"] == []
