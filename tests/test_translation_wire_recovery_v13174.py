from __future__ import annotations

import json

from mathula_tv.ai_provider import (
    ClaudeInvalidStructuredOutput,
    parse_compact_multivariant_text,
)
from mathula_tv.atomic_io import atomic_write_json
from mathula_tv.multivariant_translation import (
    build_translation_request,
    normalize_multivariant_response,
)
from mathula_tv.one_call_translation import (
    _attempt_chunk_split_recovery,
    _chunk_split_recovery_reason,
    _load_saved_split_failure,
    _recover_failed_candidate,
)
from mathula_tv.semantic_translation_chunks import build_semantic_batch_requests


def _batch_request(unit_count: int = 4) -> dict:
    request = build_translation_request(
        transcript={
            "language": "en-ZA",
            "segments": [
                {
                    "segment_id": f"unit_{index:04d}",
                    "speaker": "SPEAKER_00",
                    "start": (index - 1) * 2,
                    "end": (index - 1) * 2 + 1.8,
                    "source_text": "Hello.",
                }
                for index in range(1, unit_count + 1)
            ],
        },
        target_locale="zu-ZA",
        job_id="job-wire-recovery",
    )
    batches, _plan = build_semantic_batch_requests(
        request,
        target_tokens=10_000,
        hard_tokens=12_000,
        context_spine_tokens=1_600,
        max_units=unit_count,
    )
    return batches[0]


def _normalized_response(request: dict) -> dict:
    compact = {
        "units": [
            {
                "unit_id": unit["unit_id"],
                "variants": [
                    {
                        "variant_id": variant_id,
                        "spoken_text": spoken_text,
                        "estimated_duration_ms": duration,
                        "preserved_english_spans": [],
                        "omitted_optional_details": [],
                        "meaning_preserved": True,
                    }
                    for variant_id, spoken_text, duration in (
                        ("natural", "Sawubona kakhulu.", 1_500),
                        ("concise", "Sawubona.", 1_100),
                        ("compact", "Yebo.", 700),
                    )
                ],
            }
            for unit in request["units"]
        ],
        "global_terminology": [],
        "warnings": [],
    }
    return normalize_multivariant_response(compact, request=request)


def _compact_unit(unit_id: str) -> dict:
    variants = {
        "n": {"t": "Sawubona kakhulu.", "ms": 1_500},
        "c": {"t": "Sawubona.", "ms": 1_100},
        "k": {"t": "Yebo.", "ms": 700},
    }
    return {
        "id": unit_id,
        "f": [],
        "o": [],
        **{
            key: {
                **value,
                "p": [],
                "o": [],
                "ok": True,
            }
            for key, value in variants.items()
        },
    }


def test_nested_full_envelope_is_converted_to_compact_wire_locally() -> None:
    request = _batch_request(unit_count=1)
    full = _normalized_response(request)
    wrapped = {
        "data": {
            "response": {
                "translation": full,
            }
        }
    }

    recovered = parse_compact_multivariant_text(json.dumps(wrapped))

    assert [item["unit_id"] for item in recovered["units"]] == [
        "unit_0001"
    ]
    assert [
        item["variant_id"] for item in recovered["units"][0]["variants"]
    ] == ["natural", "concise", "compact"]


def test_top_level_compact_unit_list_is_recovered() -> None:
    recovered = parse_compact_multivariant_text(
        json.dumps([_compact_unit("unit_0001")])
    )

    assert recovered["units"][0]["unit_id"] == "unit_0001"
    assert recovered["units"][0]["variants"][2]["spoken_text"] == "Yebo."


def test_saved_raw_full_response_is_checkpoint_recoverable_without_ai(
    tmp_path,
) -> None:
    request = _batch_request(unit_count=2)
    full = _normalized_response(request)
    atomic_write_json(
        tmp_path / "failure.json",
        {
            "request_sha256": request["request_sha256"],
            "error": {
                "details": {
                    "provider": "azure-foundry-claude",
                    "validation_stage": "parse",
                    "raw_output": json.dumps(
                        {"output": {"translation": full}}
                    ),
                    "token_usage": {
                        "input_tokens": 9_553,
                        "output_tokens": 1,
                        "cache_read_input_tokens": 9_189,
                    },
                }
            },
        },
    )

    recovered = _recover_failed_candidate(
        chunk_dir=tmp_path,
        request=request,
    )

    assert recovered is not None
    response, metadata, validation = recovered
    assert [item["unit_id"] for item in response["units"]] == [
        item["unit_id"] for item in request["units"]
    ]
    assert metadata["recovered_from_failed_candidate"] is True
    assert metadata["token_usage"]["input_tokens"] == 9_553
    assert validation["valid"] is True


class _Metadata:
    def to_dict(self):
        return {
            "provider": "azure-foundry-claude",
            "model_returned": "claude-opus-4-8",
            "prompt_version": "test",
            "provider_attempts": 1,
            "token_usage": {"input_tokens": 20, "output_tokens": 10},
        }


class _ChildProvider:
    def __init__(self):
        self.requests = []

    def translate_multivariant(self, request):
        self.requests.append(request)
        return type(
            "ProviderResponse",
            (),
            {
                "data": _normalized_response(request),
                "metadata": _Metadata(),
            },
        )()


def test_degenerate_compact_completion_splits_only_failed_chunk(
    tmp_path,
) -> None:
    request = _batch_request(unit_count=4)
    original_error = ClaudeInvalidStructuredOutput(
        "Anthropic structured output failed validation after 0 repairs: "
        "ValueError: structured output did not contain compact translation units",
        details={
            "validation_stage": "parse",
            "validation_error": (
                "ValueError: structured output did not contain compact "
                "translation units"
            ),
            "provider_attempts": 1,
            "token_usage": {
                "input_tokens": 9_553,
                "output_tokens": 1,
                "cache_read_input_tokens": 9_189,
            },
        },
    )
    provider = _ChildProvider()

    assert (
        _chunk_split_recovery_reason(original_error)
        == "invalid_translation_wire_shape"
    )
    recovered = _attempt_chunk_split_recovery(
        provider=provider,
        chunk_dir=tmp_path,
        batch_request=request,
        original_error=original_error,
        emit=lambda *_args, **_kwargs: None,
        chunk_index=4,
        chunk_count=15,
    )

    assert recovered is not None
    response, metadata, validation = recovered
    assert [len(item["units"]) for item in provider.requests] == [2, 2]
    assert [item["unit_id"] for item in response["units"]] == [
        item["unit_id"] for item in request["units"]
    ]
    assert metadata["split_recovery_reason"] == (
        "invalid_translation_wire_shape"
    )
    assert metadata["token_usage"]["input_tokens"] == 9_593
    assert validation["valid"] is True


def test_saved_degenerate_failure_resumes_at_split_without_parent_retry(
    tmp_path,
) -> None:
    request = _batch_request(unit_count=4)
    atomic_write_json(
        tmp_path / "failure.json",
        {
            "request_sha256": request["request_sha256"],
            "error": {
                "message": (
                    "Anthropic structured output failed validation after 0 "
                    "repairs: ValueError: structured output did not contain "
                    "compact translation units"
                ),
                "details": {
                    "validation_stage": "parse",
                    "validation_error": (
                        "ValueError: structured output did not contain compact "
                        "translation units"
                    ),
                    "raw_output": "{}",
                    "provider_attempts": 1,
                    "token_usage": {
                        "input_tokens": 9_553,
                        "output_tokens": 1,
                        "cache_read_input_tokens": 9_189,
                    },
                },
            },
        },
    )

    saved = _load_saved_split_failure(
        chunk_dir=tmp_path,
        request=request,
    )

    assert saved is not None
    assert (
        _chunk_split_recovery_reason(saved)
        == "invalid_translation_wire_shape"
    )
    assert saved.details["loaded_from_saved_chunk_failure"] is True
