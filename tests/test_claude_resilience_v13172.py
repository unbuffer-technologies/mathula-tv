from __future__ import annotations

import json

import pytest
import requests

from mathula_tv.ai_provider import (
    AZURE_FOUNDRY_CLAUDE_PROVIDER,
    AIProviderRequestRejected,
    AnthropicClaudeProvider,
    AnthropicConfig,
    ClaudeInvalidStructuredOutput,
    StructuredAIRequest,
)
from mathula_tv.multivariant_translation import (
    build_translation_request,
    normalize_multivariant_response,
)
from mathula_tv.one_call_translation import _attempt_chunk_split_recovery
from mathula_tv.semantic_translation_chunks import build_semantic_batch_requests


SIMPLE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["ok"],
    "properties": {"ok": {"type": "boolean"}},
}


class Response:
    def __init__(self, status_code=200, body=None, headers=None):
        self.status_code = status_code
        self.body = body
        self.headers = headers or {}
        self.text = json.dumps(body or {})

    def json(self):
        return self.body

    def close(self):
        return None


class Session:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        value = self.responses.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value


def _message(text='{"ok":true}', *, stop_reason="end_turn"):
    return {
        "id": "msg_test",
        "model": "claude-opus-4-8",
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop_reason,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


def _request(**updates):
    values = {
        "operation": "resilience_test",
        "payload": {"value": 1},
        "output_schema": SIMPLE_SCHEMA,
        "prompt_version": "resilience-v1",
        "system_prompt": "Return the requested JSON.",
        "response_schema_version": "resilience-result-v1",
    }
    values.update(updates)
    return StructuredAIRequest(**values)


def _foundry_provider(session, **updates):
    config = AnthropicConfig(
        api_key="test-key",
        provider=AZURE_FOUNDRY_CLAUDE_PROVIDER,
        model="claude-opus-4-8",
        base_url="https://mathula-test.services.ai.azure.com/anthropic",
        max_retries=1,
        max_repairs=0,
        **updates,
    )
    return AnthropicClaudeProvider(config, session=session, sleep=lambda _delay: None)


def test_foundry_auto_falls_back_once_and_remembers_capability() -> None:
    unsupported = Response(
        400,
        {
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": "output_config.format is not supported by this deployment",
            },
        },
    )
    session = Session(
        [
            unsupported,
            Response(body=_message()),
            Response(body=_message()),
        ]
    )
    provider = _foundry_provider(session)

    assert provider.complete_structured(_request()).data == {"ok": True}
    assert provider.complete_structured(_request()).data == {"ok": True}

    first = session.calls[0][1]["json"]
    second = session.calls[1][1]["json"]
    third = session.calls[2][1]["json"]
    assert first["output_config"]["format"]["type"] == "json_schema"
    assert "format" not in second["output_config"]
    assert "format" not in third["output_config"]
    second_payload = json.loads(second["messages"][0]["content"])
    assert second_payload["required_output_schema"] == SIMPLE_SCHEMA


def test_foundry_auto_recognises_underscored_structured_outputs_error() -> None:
    session = Session(
        [
            Response(
                400,
                {
                    "error": {
                        "type": "invalid_request_error",
                        "message": "structured_outputs not supported in your workspace.",
                    }
                },
            ),
            Response(body=_message()),
        ]
    )
    provider = _foundry_provider(session)

    assert provider.complete_structured(_request()).data == {"ok": True}
    assert len(session.calls) == 2
    assert "format" in session.calls[0][1]["json"]["output_config"]
    assert "format" not in session.calls[1][1]["json"]["output_config"]


def test_foundry_enabled_mode_fails_closed_when_format_is_unsupported() -> None:
    session = Session(
        [
            Response(
                400,
                {
                    "error": {
                        "type": "invalid_request_error",
                        "message": "Structured outputs are not supported",
                    }
                },
            )
        ]
    )
    provider = _foundry_provider(
        session,
        foundry_structured_outputs="enabled",
    )
    with pytest.raises(AIProviderRequestRejected):
        provider.complete_structured(_request())
    assert len(session.calls) == 1


def test_foundry_removes_only_unsupported_effort_control() -> None:
    session = Session(
        [
            Response(
                400,
                {
                    "error": {
                        "type": "invalid_request_error",
                        "message": "output_config.effort is not supported by this model",
                    }
                },
            ),
            Response(body=_message()),
        ]
    )
    provider = _foundry_provider(session)
    assert provider.complete_structured(_request()).data == {"ok": True}
    first = session.calls[0][1]["json"]
    second = session.calls[1][1]["json"]
    assert first["output_config"]["effort"] == "high"
    assert second["output_config"]["format"]["type"] == "json_schema"
    assert "effort" not in second["output_config"]


def test_open_object_schema_keeps_prompt_and_local_validation_path() -> None:
    provider = _foundry_provider(Session([]))
    request = _request(
        output_schema={"type": "object", "additionalProperties": True}
    )

    body = provider.build_request(request)

    assert "format" not in body["output_config"]
    payload = json.loads(body["messages"][0]["content"])
    assert payload["required_output_schema"] == request.output_schema


def test_closed_schema_drops_only_undeclared_properties_without_repair() -> None:
    events = []
    provider = _foundry_provider(
        Session([Response(body=_message('{"ok":true,"explanation":"extra"}'))]),
        foundry_structured_outputs="disabled",
    ).with_request_options(event_callback=events.append)

    response = provider.complete_structured(_request())

    assert response.data == {"ok": True}
    normalized = [
        event
        for event in events
        if event.get("event") == "structured_output_normalized"
    ]
    assert normalized == [
        {
            "event": "structured_output_normalized",
            "operation": "resilience_test",
            "dropped_property_count": 1,
            "dropped_property_paths": ["$.explanation"],
        }
    ]
    assert (
        response.metadata.request_summary["schema_normalization"]
        ["dropped_property_paths"]
        == ["$.explanation"]
    )


def test_unknown_property_cleanup_does_not_replace_missing_required_data() -> None:
    provider = _foundry_provider(
        Session([Response(body=_message('{"explanation":"extra"}'))]),
        foundry_structured_outputs="disabled",
    )

    with pytest.raises(ClaudeInvalidStructuredOutput) as caught:
        provider.complete_structured(_request())

    assert "'ok' is a required property" in str(caught.value)


def test_schema_complexity_is_rejected_before_http() -> None:
    properties = {
        f"optional_{index:02d}": {"type": "string"} for index in range(25)
    }
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
    }
    session = Session([])
    provider = _foundry_provider(session)
    with pytest.raises(AIProviderRequestRejected, match="optional-parameter"):
        provider.complete_structured(_request(output_schema=schema))
    assert session.calls == []


def test_529_retries_and_honours_long_retry_after() -> None:
    delays = []
    session = Session(
        [
            Response(529, {"error": {"type": "overloaded_error"}}, {"retry-after": "120"}),
            Response(body=_message()),
        ]
    )
    provider = AnthropicClaudeProvider(
        AnthropicConfig(api_key="test", max_retries=2, max_repairs=0),
        session=session,
        sleep=delays.append,
    )
    assert provider.complete_structured(_request()).data == {"ok": True}
    assert delays == [120.0]


def test_enum_case_is_canonicalised_before_local_validation() -> None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["status"],
        "properties": {
            "status": {
                "type": "string",
                "enum": ["Needs Human Review"],
            }
        },
    }
    session = Session(
        [Response(body=_message('{"status":"Needs human review"}'))]
    )
    provider = AnthropicClaudeProvider(
        AnthropicConfig(api_key="test", max_retries=1, max_repairs=0),
        session=session,
        sleep=lambda _delay: None,
    )
    assert provider.complete_structured(
        _request(output_schema=schema)
    ).data == {"status": "Needs Human Review"}


class InterruptedAfterJSONResponse:
    status_code = 200
    headers = {}

    def iter_lines(self, decode_unicode=True):
        events = [
            (
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_interrupted",
                        "type": "message",
                        "role": "assistant",
                        "model": "claude-opus-4-8",
                        "content": [],
                        "usage": {"input_tokens": 10, "output_tokens": 0},
                    },
                },
            ),
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": '{"ok":true}'},
                },
            ),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ]
        for event_name, data in events:
            yield f"event: {event_name}"
            yield "data: " + json.dumps(data)
            yield ""
        raise requests.ConnectionError("connection closed after completed JSON")

    def close(self):
        return None


def test_complete_json_is_salvaged_from_interrupted_sse() -> None:
    session = Session([InterruptedAfterJSONResponse()])
    provider = _foundry_provider(
        session,
        foundry_structured_outputs="disabled",
    )
    result = provider.complete_structured(_request(stream_response=True))
    assert result.data == {"ok": True}
    assert result.metadata.stop_reason == "stream_interrupted_recovered"


def _translation_request() -> dict:
    return build_translation_request(
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
                for index in range(1, 5)
            ],
        },
        target_locale="zu-ZA",
        job_id="job-split",
    )


def _valid_child_response(request):
    compact = {
        "units": [
            {
                "unit_id": unit["unit_id"],
                "variants": [
                    {
                        "variant_id": variant,
                        "spoken_text": text,
                        "estimated_duration_ms": duration,
                        "preserved_english_spans": [],
                        "omitted_optional_details": [],
                        "meaning_preserved": True,
                    }
                    for variant, text, duration in (
                        ("natural", "Sawubona kakhulu.", 1500),
                        ("concise", "Sawubona.", 1100),
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


class Metadata:
    def to_dict(self):
        return {
            "provider": "azure-foundry-claude",
            "model_returned": "claude-opus-4-8",
            "prompt_version": "test",
            "provider_attempts": 1,
            "token_usage": {"input_tokens": 20, "output_tokens": 10},
        }


class ChildProvider:
    def __init__(self):
        self.requests = []

    def translate_multivariant(self, request):
        self.requests.append(request)
        return type(
            "ProviderResponse",
            (),
            {"data": _valid_child_response(request), "metadata": Metadata()},
        )()


def test_truncated_chunk_is_split_serially_and_merged(tmp_path) -> None:
    full = _translation_request()
    batches, _plan = build_semantic_batch_requests(
        full,
        target_tokens=10_000,
        hard_tokens=12_000,
        context_spine_tokens=1_600,
        max_units=4,
    )
    batch = batches[0]
    original_error = ClaudeInvalidStructuredOutput(
        "max tokens",
        details={
            "stop_reason": "max_tokens",
            "provider_attempts": 1,
            "token_usage": {"input_tokens": 100, "output_tokens": 500},
        },
    )
    provider = ChildProvider()

    repaired = _attempt_chunk_split_recovery(
        provider=provider,
        chunk_dir=tmp_path,
        batch_request=batch,
        original_error=original_error,
        emit=lambda *_args, **_kwargs: None,
        chunk_index=1,
        chunk_count=1,
    )

    assert repaired is not None
    response, metadata, validation = repaired
    assert [item["unit_id"] for item in response["units"]] == [
        item["unit_id"] for item in batch["units"]
    ]
    assert [len(item["units"]) for item in provider.requests] == [2, 2]
    assert metadata["split_chunk_recovery_count"] == 1
    assert metadata["token_usage"]["output_tokens"] == 520
    assert validation["valid"] is True
    assert (tmp_path / "split_recovery/depth_01/completed.json").is_file()
