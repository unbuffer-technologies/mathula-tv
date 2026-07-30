from __future__ import annotations

import json
import math

from mathula_tv.ai_provider import (
    AnthropicClaudeProvider,
    AnthropicConfig,
    StructuredAIRequest,
)
from mathula_tv.cli import _publication_ai_progress_payload
from mathula_tv.azure_web_research import (
    _request_consumption_metrics,
    _response_consumption_metrics,
)


class _Response:
    status_code = 200
    headers: dict[str, str] = {}

    def json(self):
        return {
            "model": "claude-opus-4-8",
            "content": [{"type": "text", "text": '{"ok":true}'}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 321,
                "output_tokens": 45,
                "cache_creation_input_tokens": 12,
                "cache_read_input_tokens": 34,
                "output_tokens_details": {"thinking_tokens": 7},
            },
        }


class _Session:
    def __init__(self):
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return _Response()


def _request() -> StructuredAIRequest:
    return StructuredAIRequest(
        operation="telemetry_test",
        payload={
            "transcript": "This text is counted but never included in telemetry.",
            "segments": [{"segment_id": "seg-1", "source_text": "Example"}],
        },
        output_schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["ok"],
            "properties": {"ok": {"type": "boolean"}},
        },
        prompt_version="telemetry-v1",
        system_prompt="Return the requested JSON shape.",
        response_schema_version="telemetry-result-v1",
        max_output_tokens=2000,
    )


def test_provider_emits_exact_wire_size_estimate_and_actual_usage() -> None:
    events: list[dict] = []
    session = _Session()
    provider = AnthropicClaudeProvider(
        AnthropicConfig(
            api_key="secret",
            max_retries=1,
            max_repairs=0,
        ),
        session=session,
        sleep=lambda _delay: None,
        event_callback=events.append,
    )
    request = _request()

    response = provider.complete_structured(request)

    request_event = next(
        event for event in events if event["event"] == "request_metrics"
    )
    posted_body = session.calls[0][1]["json"]
    wire_text = json.dumps(
        posted_body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    assert request_event["request_body_bytes"] == len(
        wire_text.encode("utf-8")
    )
    assert request_event["request_body_characters"] == len(wire_text)
    assert request_event["estimated_input_tokens"] == math.ceil(
        len(wire_text) / 4
    )
    assert request_event["max_output_tokens"] == 2000
    assert "transcript" not in json.dumps(request_event)

    usage_event = next(
        event for event in events if event["event"] == "response_usage"
    )
    assert usage_event["input_tokens"] == 321
    assert usage_event["output_tokens"] == 45
    assert usage_event["thinking_tokens"] == 7
    assert usage_event["cache_creation_input_tokens"] == 12
    assert usage_event["cache_read_input_tokens"] == 34
    assert usage_event["total_tokens"] == 366
    assert usage_event["response_body_bytes"] > 0

    persisted = response.metadata.request_summary["wire_requests"]
    assert persisted == [
        {
            key: value
            for key, value in request_event.items()
            if key != "event"
        }
    ]


def test_publication_progress_formats_request_and_actual_usage() -> None:
    request_payload = _publication_ai_progress_payload(
        {
            "event": "request_metrics",
            "request_body_bytes": 102400,
            "payload_bytes": 81920,
            "output_schema_bytes": 10240,
            "estimated_input_tokens": 25600,
            "max_output_tokens": 32000,
        }
    )
    assert request_payload is not None
    assert request_payload["stage"] == "publication_ai"
    assert "100.0 KiB JSON" in request_payload["message"]
    assert "~25,600 input tokens estimated" in request_payload["message"]

    usage_payload = _publication_ai_progress_payload(
        {
            "event": "response_usage",
            "input_tokens": 24000,
            "output_tokens": 1200,
            "thinking_tokens": 300,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 8000,
            "response_body_bytes": 16384,
        }
    )
    assert usage_payload is not None
    assert usage_payload["status"] == "completed"
    assert "input=24,000" in usage_payload["message"]
    assert "cache-read=8,000" in usage_payload["message"]
    assert "response=16.0 KiB" in usage_payload["message"]


def test_publication_progress_suppresses_raw_sse_delta_events() -> None:
    assert (
        _publication_ai_progress_payload(
            {"event": "stream_event", "data": {"private": "content"}}
        )
        is None
    )


def test_web_research_reports_request_estimate_and_responses_usage() -> None:
    body = {
        "model": "gpt-test",
        "input": "x" * 4096,
        "tools": [{"type": "web_search"}],
    }
    request_metrics = _request_consumption_metrics(body)
    serialized = json.dumps(
        body,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    assert request_metrics["request_json_bytes"] == len(
        serialized.encode("utf-8")
    )
    assert request_metrics["estimated_input_tokens"] == math.ceil(
        len(serialized) / 4
    )

    usage = _response_consumption_metrics(
        {
            "usage": {
                "input_tokens": 1400,
                "output_tokens": 250,
                "total_tokens": 1650,
                "input_tokens_details": {"cached_tokens": 400},
                "output_tokens_details": {"reasoning_tokens": 75},
            },
            "output": [],
        },
        output_text="result",
    )
    assert usage["input_tokens"] == 1400
    assert usage["output_tokens"] == 250
    assert usage["total_tokens"] == 1650
    assert usage["cache_read_input_tokens"] == 400
    assert usage["thinking_tokens"] == 75
    assert usage["response_text_characters"] == 6
