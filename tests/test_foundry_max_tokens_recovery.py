from __future__ import annotations

import json

import pytest

from mathula_tv.ai_provider import (
    AZURE_FOUNDRY_CLAUDE_PROVIDER,
    AnthropicClaudeProvider,
    AnthropicConfig,
    ClaudeInvalidStructuredOutput,
)
from mathula_tv.multivariant_translation import build_translation_request
from mathula_tv.foundry_max_tokens_recovery import (
    _continuation_request,
    extract_complete_translation_units,
)


def _source_request() -> dict:
    return build_translation_request(
        transcript={
            "language": "en-ZA",
            "segments": [
                {
                    "segment_id": f"unit_{index:04d}",
                    "speaker": "SPEAKER_00",
                    "start": index * 2,
                    "end": index * 2 + 1.5,
                    "source_text": f"Source sentence {index}.",
                }
                for index in range(1, 5)
            ],
        },
        target_locale="zu-ZA",
        job_id="job-test",
    )


def _raw_unit(unit_id: str) -> dict:
    return {
        "unit_id": unit_id,
        "variants": [
            {
                "variant_id": variant_id,
                "spoken_text": f"{variant_id} translation",
                "estimated_duration_ms": 1000,
                "preserved_english_spans": [],
                "omitted_optional_details": [],
                "meaning_preserved": True,
            }
            for variant_id in ("natural", "concise", "compact")
        ],
    }


def test_extracts_only_fully_closed_units_from_truncated_json() -> None:
    first = _raw_unit("unit_0001")
    second = _raw_unit("unit_0002")
    text = (
        '{"schema_version":"mathula.translation.multivariant.v1",'
        '"global_terminology":[{"source_term":"IDAC"}],'
        '"units":['
        + json.dumps(first)
        + ","
        + json.dumps(second)
        + ',{"unit_id":"unit_0003","variants":[{"variant_id":"natural","spoken_text":"cut'
    )

    units = extract_complete_translation_units(text)

    assert [unit["unit_id"] for unit in units] == ["unit_0001", "unit_0002"]


def test_continuation_request_returns_only_missing_units_but_keeps_full_context() -> None:
    request = _source_request()
    completed = request["units"][:2]
    missing = request["units"][2:]

    continuation = _continuation_request(
        request,
        completed_units=completed,
        missing_units=missing,
    )

    assert [unit["unit_id"] for unit in continuation["units"]] == [
        unit["unit_id"] for unit in missing
    ]
    recovery = continuation["global_context"]["max_tokens_recovery"]
    assert len(recovery["full_english_timeline"]) == len(request["units"])
    assert recovery["completed_unit_ids"] == [unit["unit_id"] for unit in completed]
    assert recovery["requested_unit_ids"] == [unit["unit_id"] for unit in missing]


class MaxTokensResponse:
    status_code = 200
    headers = {}

    def json(self):
        raise AssertionError("streaming response must not call json()")

    def iter_lines(self, decode_unicode=True):
        events = [
            (
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_max",
                        "type": "message",
                        "role": "assistant",
                        "model": "claude-opus-4-8",
                        "content": [],
                        "stop_reason": None,
                        "usage": {"input_tokens": 100, "output_tokens": 0},
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
                    "delta": {"type": "text_delta", "text": '{"units":['},
                },
            ),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            (
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "max_tokens", "stop_sequence": None},
                    "usage": {
                        "output_tokens": 50_000,
                        "output_tokens_details": {"thinking_tokens": 13_886},
                    },
                },
            ),
            ("message_stop", {"type": "message_stop"}),
        ]
        for event_name, data in events:
            yield f"event: {event_name}"
            yield "data: " + json.dumps(data)
            yield ""

    def close(self):
        return None


class MaxTokensSession:
    def __init__(self):
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return MaxTokensResponse()


def test_max_tokens_stream_is_reported_as_truncated_without_retry() -> None:
    request = _source_request()
    session = MaxTokensSession()
    events = []
    provider = AnthropicClaudeProvider(
        AnthropicConfig(
            api_key="test",
            provider=AZURE_FOUNDRY_CLAUDE_PROVIDER,
            model="claude-opus-4-8",
            base_url="https://mathula-test.services.ai.azure.com/anthropic",
            max_retries=4,
            max_repairs=4,
        ),
        session=session,
        sleep=lambda _delay: None,
    ).with_request_options(max_retries=1, event_callback=events.append)

    with pytest.raises(ClaudeInvalidStructuredOutput, match="reached max_tokens") as raised:
        provider.translate_multivariant(request)

    assert len(session.calls) == 1
    assert raised.value.details["stop_reason"] == "max_tokens"
    assert raised.value.details["thinking_tokens"] == 13_886
    assert any(event["event"] == "stream_truncated" for event in events)
    assert not any(event["event"] == "stream_completed" for event in events)
