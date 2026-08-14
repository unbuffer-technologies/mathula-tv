from __future__ import annotations

import json

import jsonschema
import pytest

from mathula_tv import ai_provider
from mathula_tv.atomic_io import atomic_write_json
from mathula_tv.multivariant_translation import build_translation_request
from mathula_tv.one_call_translation import _recover_failed_candidate
from mathula_tv.semantic_translation_chunks import build_semantic_batch_requests


def _variant(text: str, milliseconds: int) -> dict[str, object]:
    return {"t": text, "ms": milliseconds, "p": [], "o": [], "ok": True}


def _wire(natural: dict[str, object]) -> dict[str, object]:
    return {
        "units": [
            {
                "id": "unit_0191",
                "f": ["fact"],
                "o": [],
                "n": natural,
                "c": _variant("Okufushane", 1500),
                "k": _variant("Mfushane", 1000),
            }
        ],
        "terms": [],
        "warnings": [],
    }


def test_leading_comma_ms_key_is_normalized_and_audited() -> None:
    natural = _variant("Umbhalo wemvelo", 1900)
    natural[",ms"] = natural.pop("ms")

    result = ai_provider._parse_compact_multivariant_json(
        json.dumps(_wire(natural), ensure_ascii=False)
    )

    assert result["units"][0]["n"]["ms"] == 1900
    assert ",ms" not in result["units"][0]["n"]
    assert any(
        "unit_0191/natural: normalized punctuated compact field ',ms' as ms"
        in warning
        for warning in result["warnings"]
    )
    jsonschema.validate(result, ai_provider.COMPACT_MULTIVARIANT_WIRE_SCHEMA)


def test_identical_canonical_and_punctuated_keys_are_deduplicated() -> None:
    natural = _variant("Umbhalo wemvelo", 1900)
    natural["ms,"] = 1900

    result = ai_provider._canonicalize_compact_multivariant_wire(_wire(natural))

    assert result["units"][0]["n"]["ms"] == 1900
    assert "ms," not in result["units"][0]["n"]
    assert any("ignored redundant punctuated compact field" in item for item in result["warnings"])


def test_conflicting_punctuated_alias_still_fails_closed() -> None:
    natural = _variant("Umbhalo wemvelo", 1900)
    natural[",ms"] = 9999

    with pytest.raises(ValueError, match="conflicting compact field aliases"):
        ai_provider._canonicalize_compact_multivariant_wire(_wire(natural))


def test_unknown_structural_field_still_fails_closed() -> None:
    natural = _variant("Umbhalo wemvelo", 1900)
    natural["invented_translation"] = "do not discard"

    with pytest.raises(ValueError, match="unexpected fields: invented_translation"):
        ai_provider._canonicalize_compact_multivariant_wire(_wire(natural))


def test_saved_paid_chunk_is_recovered_without_another_provider_call(
    tmp_path,
) -> None:
    full_request = build_translation_request(
        transcript={
            "language": "en-ZA",
            "segments": [
                {
                    "segment_id": "unit_0191",
                    "speaker": "SPEAKER_00",
                    "start": 0,
                    "end": 3,
                    "source_text": "Hello.",
                }
            ],
        },
        target_locale="zu-ZA",
        job_id="job-punctuated-key-recovery",
    )
    requests, _plan = build_semantic_batch_requests(
        full_request,
        target_tokens=10_000,
        hard_tokens=12_000,
        context_spine_tokens=1_600,
        max_units=1,
    )
    request = requests[0]
    natural = _variant("Sawubona kakhulu.", 1900)
    natural[",ms"] = natural.pop("ms")
    raw_output = json.dumps(_wire(natural), ensure_ascii=False)
    atomic_write_json(
        tmp_path / "failure.json",
        {
            "request_sha256": request["request_sha256"],
            "error": {
                "details": {
                    "provider": "azure-foundry-claude",
                    "validation_stage": "parse",
                    "raw_output": raw_output,
                    "token_usage": {
                        "input_tokens": 12_000,
                        "output_tokens": 4_000,
                    },
                }
            },
        },
    )

    recovered = _recover_failed_candidate(chunk_dir=tmp_path, request=request)

    assert recovered is not None
    response, metadata, validation = recovered
    assert response["units"][0]["unit_id"] == "unit_0191"
    assert metadata["recovered_from_failed_candidate"] is True
    assert metadata["token_usage"]["input_tokens"] == 12_000
    assert validation["valid"] is True
