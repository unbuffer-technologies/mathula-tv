from __future__ import annotations

import json

import jsonschema
import pytest

from mathula_tv.ai_provider import (
    COMPACT_MULTIVARIANT_WIRE_SCHEMA,
    _parse_compact_multivariant_json,
)
from mathula_tv.atomic_io import atomic_write_json
from mathula_tv.multivariant_translation import build_translation_request
from mathula_tv.one_call_translation import _recover_failed_candidate


def _request() -> dict:
    request = build_translation_request(
        transcript={
            "language": "en-ZA",
            "segments": [
                {
                    "segment_id": "unit_0018",
                    "speaker": "SPEAKER_01",
                    "start": 0,
                    "end": 5,
                    "source_text": "The witness answered the question.",
                    "required_facts": ["The witness answered"],
                    "optional_details": ["the question"],
                    "protected_spans": [],
                }
            ],
        },
        target_locale="zu-ZA",
        job_id="compact-metadata-recovery",
    )
    request["units"][0]["required_facts"] = ["The witness answered"]
    request["units"][0]["optional_details"] = ["the question"]
    return request


def _compact_with_missing_ledgers() -> dict:
    return {
        "units": [
            {
                "id": "unit_0018",
                "f": ["The witness answered"],
                "n": {
                    "t": "Ufakazi uwuphendulile umbuzo.",
                    "ms": 2_600,
                    "ok": True,
                },
                "c": {
                    "t": "Ufakazi uphendulile.",
                    "ms": 2_100,
                    "p": [],
                    "ok": True,
                },
                "k": {
                    "t": "Uphendulile.",
                    "ms": 1_600,
                    "p": [],
                    "o": ["the question"],
                    "ok": True,
                },
            }
        ],
        "terms": [],
        "warnings": [],
    }


def test_source_bound_parser_restores_only_missing_metadata_ledgers() -> None:
    request = _request()

    restored = _parse_compact_multivariant_json(
        json.dumps(_compact_with_missing_ledgers()),
        source_units=request["units"],
    )

    jsonschema.validate(restored, COMPACT_MULTIVARIANT_WIRE_SCHEMA)
    unit = restored["units"][0]
    assert unit["o"] == request["units"][0]["optional_details"]
    assert unit["n"]["p"] == []
    assert unit["n"]["o"] == []
    assert unit["c"]["o"] == []
    assert any(
        "restored missing optional-details ledger from request" in warning
        for warning in restored["warnings"]
    )


def test_saved_paid_candidate_missing_unit_o_recovers_without_provider_call(
    tmp_path,
) -> None:
    request = _request()
    chunk_dir = tmp_path / "chunk_0003"
    chunk_dir.mkdir()
    atomic_write_json(
        chunk_dir / "failure.json",
        {
            "request_sha256": request["request_sha256"],
            "error": {
                "details": {
                    "provider": "azure-foundry-claude",
                    "candidate_output": _compact_with_missing_ledgers(),
                    "provider_attempts": 1,
                    "token_usage": {
                        "input_tokens": 9_637,
                        "output_tokens": 5_137,
                        "cache_read_input_tokens": 9_353,
                    },
                }
            },
        },
    )

    recovered = _recover_failed_candidate(
        chunk_dir=chunk_dir,
        request=request,
    )

    assert recovered is not None
    response, metadata, validation = recovered
    assert response["units"][0]["unit_id"] == "unit_0018"
    assert (
        response["units"][0]["optional_details"]
        == request["units"][0]["optional_details"]
    )
    assert metadata["recovered_from_failed_candidate"] is True
    assert metadata["token_usage"]["output_tokens"] == 5_137
    assert validation["valid"] is True


def test_missing_core_translation_text_still_fails_closed() -> None:
    request = _request()
    compact = _compact_with_missing_ledgers()
    del compact["units"][0]["n"]["t"]

    restored = _parse_compact_multivariant_json(
        json.dumps(compact),
        source_units=request["units"],
    )

    with pytest.raises(jsonschema.ValidationError, match="'t' is a required property"):
        jsonschema.validate(restored, COMPACT_MULTIVARIANT_WIRE_SCHEMA)
