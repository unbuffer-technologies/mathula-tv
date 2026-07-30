from __future__ import annotations

import json

import pytest
import requests

from mathula_tv.ai_provider import (
    AZURE_FOUNDRY_CLAUDE_PROVIDER,
    AnthropicClaudeProvider,
    AnthropicConfig,
    ClaudeInvalidStructuredOutput,
    ClaudeTimeout,
)
from mathula_tv.multivariant_translation import (
    build_translation_request,
    render_system_prompt,
    render_user_prompt,
)


class Response:
    status_code = 200
    headers = {}

    def __init__(self, body):
        self.body = body
        self.closed = False

    def json(self):
        return self.body

    def iter_lines(self, decode_unicode=True):
        text = "".join(
            str(block.get("text") or "")
            for block in self.body.get("content", [])
            if isinstance(block, dict) and block.get("type") == "text"
        )
        events = [
            (
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_test",
                        "type": "message",
                        "role": "assistant",
                        "model": self.body.get("model"),
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {
                            "input_tokens": self.body.get("usage", {}).get("input_tokens", 0),
                            "output_tokens": 0,
                        },
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
        ]
        midpoint = max(1, len(text) // 2)
        for part in (text[:midpoint], text[midpoint:]):
            if part:
                events.append(
                    (
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {"type": "text_delta", "text": part},
                        },
                    )
                )
        events.extend(
            [
                (
                    "content_block_stop",
                    {"type": "content_block_stop", "index": 0},
                ),
                (
                    "message_delta",
                    {
                        "type": "message_delta",
                        "delta": {
                            "stop_reason": self.body.get("stop_reason"),
                            "stop_sequence": None,
                        },
                        "usage": {
                            "output_tokens": self.body.get("usage", {}).get("output_tokens", 0)
                        },
                    },
                ),
                ("message_stop", {"type": "message_stop"}),
            ]
        )
        for event_name, data in events:
            yield f"event: {event_name}"
            yield "data: " + json.dumps(data)
            yield ""

    def close(self):
        self.closed = True


class Session:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return Response(
            {
                "model": "claude-test",
                "content": [{"type": "text", "text": json.dumps(self.result)}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 20},
            }
        )


def _request() -> dict:
    return build_translation_request(
        transcript={
            "language": "en-ZA",
            "segments": [
                {
                    "segment_id": "seg-1",
                    "speaker": "SPEAKER_00",
                    "start": 0,
                    "end": 2,
                    "source_text": "The EFF speaks.",
                    "protected_spans": ["EFF"],
                }
            ],
        },
        target_locale="zu-ZA",
        job_id="job-1",
    )


def _valid_response(request: dict) -> dict:
    source = request["units"][0]
    targets = source["duration_targets"]
    return {
        "schema_version": "mathula.translation.multivariant.v1",
        "source_language": "en-ZA",
        "target_language": "isiZulu",
        "target_locale": "zu-ZA",
        "translation_strategy": "contextual_code_switching",
        "units": [
            {
                "unit_id": source["unit_id"],
                "speaker_id": source["speaker_id"],
                "start_ms": source["start_ms"],
                "end_ms": source["end_ms"],
                "available_duration_ms": source["available_duration_ms"],
                "overlap_group": source.get("overlap_group"),
                "source_text": source["source_text"],
                "protected_spans": ["EFF"],
                "required_facts": [source["source_text"]],
                "optional_details": [],
                "variants": [
                    {
                        "variant_id": variant,
                        "compression_level": variant,
                        "target_duration_ms": targets[f"{variant}_target_duration_ms"],
                        "spoken_text": text,
                        "estimated_duration_ms": duration,
                        "preserved_english_spans": ["EFF"],
                        "omitted_optional_details": [],
                        "meaning_preserved": True,
                    }
                    for variant, text, duration in (
                        ("natural", "I-EFF iyakhuluma namhlanje.", 1800),
                        ("concise", "I-EFF iyakhuluma.", 1400),
                        ("compact", "I-EFF iyasho.", 1000),
                    )
                ],
                "selection_status": "unselected",
                "selected_variant_id": None,
                "warnings": [],
            }
        ],
        "global_terminology": [],
        "warnings": [],
    }


def test_cloud_request_uses_exact_exported_system_and_user_prompts() -> None:
    request = _request()
    session = Session(_valid_response(request))
    events = []
    provider = AnthropicClaudeProvider(
        AnthropicConfig(
            api_key="test",
            provider=AZURE_FOUNDRY_CLAUDE_PROVIDER,
            model="claude-opus-4-8",
            base_url="https://mathula-test.services.ai.azure.com/anthropic",
            max_retries=1,
            max_repairs=4,
        ),
        session=session,
        sleep=lambda _delay: None,
    ).with_request_options(event_callback=events.append)

    provider.translate_multivariant(request)

    assert len(session.calls) == 1
    body = session.calls[0][1]["json"]
    assert session.calls[0][0] == (
        "https://mathula-test.services.ai.azure.com/anthropic/v1/messages"
    )
    assert session.calls[0][1]["headers"]["x-api-key"] == "test"
    assert body["system"] == render_system_prompt("zu-ZA")
    user_content = body["messages"][0]["content"]
    assert user_content == render_user_prompt(request).rstrip()
    assert "Required output schema" not in user_content
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert body["stream"] is True
    assert body["thinking"] == {"type": "disabled"}
    assert body["output_config"]["effort"] == "low"
    assert body["max_tokens"] == 100_000
    assert session.calls[0][1]["stream"] is True
    assert session.calls[0][1]["headers"]["accept"] == "text/event-stream"
    assert session.calls[0][1]["timeout"] == (30.0, 900.0)
    assert "api.anthropic.com" not in session.calls[0][0]
    event_names = [event["event"] for event in events]
    assert event_names[0] == "request_metrics"
    assert event_names[1] == "request_attempt"
    assert "stream_opened" in event_names
    assert "stream_progress" in event_names
    assert event_names[-2] == "stream_completed"
    assert event_names[-1] == "response_usage"


def test_invalid_output_does_not_trigger_a_second_model_repair_call() -> None:
    request = _request()
    session = Session({})
    provider = AnthropicClaudeProvider(
        AnthropicConfig(
            api_key="test",
            provider=AZURE_FOUNDRY_CLAUDE_PROVIDER,
            model="claude-opus-4-8",
            base_url="https://mathula-test.services.ai.azure.com/anthropic",
            max_retries=1,
            max_repairs=4,
        ),
        session=session,
        sleep=lambda _delay: None,
    )

    with pytest.raises(ClaudeInvalidStructuredOutput):
        provider.translate_multivariant(request)

    assert len(session.calls) == 1


class TimeoutResponse(Response):
    def iter_lines(self, decode_unicode=True):
        yield "event: message_start"
        yield "data: " + json.dumps(
            {
                "type": "message_start",
                "message": {
                    "id": "msg_timeout",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-opus-4-8",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 10, "output_tokens": 0},
                },
            }
        )
        yield ""
        yield "event: ping"
        yield 'data: {"type":"ping"}'
        yield ""
        raise requests.Timeout("simulated idle stream timeout")


class TimeoutSession(Session):
    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return TimeoutResponse(
            {
                "model": "claude-opus-4-8",
                "content": [],
                "stop_reason": None,
                "usage": {"input_tokens": 10, "output_tokens": 0},
            }
        )


def test_open_stream_timeout_is_not_retried() -> None:
    request = _request()
    session = TimeoutSession({})
    events = []
    provider = AnthropicClaudeProvider(
        AnthropicConfig(
            api_key="test",
            provider=AZURE_FOUNDRY_CLAUDE_PROVIDER,
            model="claude-opus-4-8",
            base_url="https://mathula-test.services.ai.azure.com/anthropic",
            max_retries=3,
            max_repairs=4,
        ),
        session=session,
        sleep=lambda _delay: None,
    ).with_request_options(
        max_retries=1,
        timeout_seconds=17,
        event_callback=events.append,
    )

    with pytest.raises(ClaudeTimeout, match="read-idle timeout"):
        provider.translate_multivariant(request)

    assert len(session.calls) == 1
    assert session.calls[0][1]["timeout"] == (17.0, 17.0)
    assert any(event["event"] == "stream_timeout" for event in events)
    assert not any(event["event"] == "retry_backoff" for event in events)

def test_foundry_stream_restores_deterministic_envelope_without_second_call() -> None:
    request = _request()
    response = _valid_response(request)
    unit = response["units"][0]
    # Reproduce the real Foundry failure: valid units but omitted top-level
    # schema metadata. Also prove immutable source/timing fields are restored
    # locally rather than trusted from generated output.
    compact = {
        "units": [
            {
                **unit,
                "speaker_id": "WRONG",
                "start_ms": 999,
                "end_ms": 1000,
                "available_duration_ms": 1,
                "source_text": "WRONG",
                "selection_status": "selected",
                "selected_variant_id": "natural",
                "variants": [
                    {
                        **variant,
                        "compression_level": "wrong",
                        "target_duration_ms": 1,
                    }
                    for variant in unit["variants"]
                ],
            }
        ]
    }
    session = Session(compact)
    provider = AnthropicClaudeProvider(
        AnthropicConfig(
            api_key="test",
            provider=AZURE_FOUNDRY_CLAUDE_PROVIDER,
            model="claude-opus-4-8",
            base_url="https://mathula-test.services.ai.azure.com/anthropic",
            max_retries=1,
            max_repairs=4,
        ),
        session=session,
        sleep=lambda _delay: None,
    )

    result = provider.translate_multivariant(request)

    assert len(session.calls) == 1
    assert result.data["schema_version"] == "mathula.translation.multivariant.v1"
    assert result.data["source_language"] == "en-ZA"
    assert result.data["target_language"] == "isiZulu"
    assert result.data["target_locale"] == "zu-ZA"
    assert result.data["translation_strategy"] == "contextual_code_switching"
    restored = result.data["units"][0]
    source = request["units"][0]
    assert restored["speaker_id"] == source["speaker_id"]
    assert restored["start_ms"] == source["start_ms"]
    assert restored["end_ms"] == source["end_ms"]
    assert restored["available_duration_ms"] == source["available_duration_ms"]
    assert restored["source_text"] == source["source_text"]
    assert restored["selection_status"] == "unselected"
    assert restored["selected_variant_id"] is None
    for variant in restored["variants"]:
        variant_id = variant["variant_id"]
        assert variant["compression_level"] == variant_id
        assert variant["target_duration_ms"] == source["duration_targets"][
            f"{variant_id}_target_duration_ms"
        ]


def test_parser_selects_unit_array_after_glossary_fragment() -> None:
    from mathula_tv.ai_provider import _parse_multivariant_structured_json
    from mathula_tv.multivariant_translation import (
        normalize_multivariant_response,
        validate_multivariant_response,
    )

    request = _request()
    valid = _valid_response(request)
    raw_unit = dict(valid["units"][0])
    raw_unit["debug_note"] = "must be discarded locally"
    raw_unit["variants"] = [
        {**variant, "extra_model_field": "discard"}
        for variant in raw_unit["variants"]
    ]
    glossary = [
        {
            "source_term": "NDPP",
            "target_rendering": "i-NDPP",
            "preserve_english": True,
            "reason": "Acronym",
        }
    ]
    # This deliberately is not one valid top-level JSON object. The glossary
    # appears first, matching the real Foundry response that exposed the bug.
    text = (
        '"global_terminology": '
        + json.dumps(glossary)
        + '\n"units": '
        + json.dumps([raw_unit])
        + '\n"warnings": []'
    )

    parsed = _parse_multivariant_structured_json(text)
    assert parsed["units"][0]["unit_id"] == request["units"][0]["unit_id"]
    assert parsed["global_terminology"][0]["source_term"] == "NDPP"

    normalized = normalize_multivariant_response(parsed, request=request)
    assert "debug_note" not in normalized["units"][0]
    assert "extra_model_field" not in normalized["units"][0]["variants"][0]
    validate_multivariant_response(normalized, request=request)


def test_parser_never_wraps_glossary_as_units() -> None:
    from mathula_tv.ai_provider import _parse_multivariant_structured_json

    glossary_only = json.dumps(
        [
            {
                "source_term": "NDPP",
                "target_rendering": "i-NDPP",
                "preserve_english": True,
                "reason": "Acronym",
            }
        ]
    )
    with pytest.raises(ValueError, match="unit-shaped"):
        _parse_multivariant_structured_json(glossary_only)
