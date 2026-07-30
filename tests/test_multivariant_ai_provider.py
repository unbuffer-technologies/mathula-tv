from __future__ import annotations

import json

import pytest

from mathula_tv.ai_provider import AnthropicClaudeProvider, AnthropicConfig
from mathula_tv.multivariant_translation import build_translation_request


class Response:
    status_code = 200
    headers = {}

    def __init__(self, body):
        self.body = body

    def json(self):
        return self.body


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


def test_provider_uses_multivariant_prompt_and_source_bound_validator() -> None:
    transcript = {
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
    }
    request = build_translation_request(
        transcript=transcript,
        target_locale="zu-ZA",
        job_id="job-1",
    )
    source = request["units"][0]
    targets = source["duration_targets"]
    result = {
        "schema_version": "mathula.translation.multivariant.v1",
        "source_language": "en-ZA",
        "target_language": "isiZulu",
        "target_locale": "zu-ZA",
        "translation_strategy": "contextual_code_switching",
        "units": [
            {
                "unit_id": "seg-1",
                "speaker_id": "SPEAKER_00",
                "start_ms": 0,
                "end_ms": 2000,
                "available_duration_ms": 2000,
                "overlap_group": None,
                "source_text": "The EFF speaks.",
                "protected_spans": ["EFF"],
                "required_facts": ["The EFF speaks."],
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
    session = Session(result)
    provider = AnthropicClaudeProvider(
        AnthropicConfig(api_key="test", max_retries=1, max_repairs=0),
        session=session,
        sleep=lambda _delay: None,
    )
    response = provider.translate_multivariant(request)
    assert response.data["units"][0]["variants"][2]["variant_id"] == "compact"
    body = session.calls[0][1]["json"]
    assert "three usable spoken variations" in body["system"]
    assert body["output_config"]["format"]["type"] == "json_schema"


def test_provider_request_options_apply_hard_timeout_and_emit_lifecycle_events() -> None:
    transcript = {
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
    }
    request = build_translation_request(
        transcript=transcript,
        target_locale="zu-ZA",
        job_id="job-1",
    )
    source = request["units"][0]
    targets = source["duration_targets"]
    result = {
        "schema_version": "mathula.translation.multivariant.v1",
        "source_language": "en-ZA",
        "target_language": "isiZulu",
        "target_locale": "zu-ZA",
        "translation_strategy": "contextual_code_switching",
        "units": [
            {
                "unit_id": "seg-1",
                "speaker_id": "SPEAKER_00",
                "start_ms": 0,
                "end_ms": 2000,
                "available_duration_ms": 2000,
                "overlap_group": None,
                "source_text": "The EFF speaks.",
                "protected_spans": ["EFF"],
                "required_facts": ["The EFF speaks."],
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
    session = Session(result)
    events = []
    provider = AnthropicClaudeProvider(
        AnthropicConfig(api_key="test", max_retries=3, max_repairs=0),
        session=session,
        sleep=lambda _delay: None,
    ).with_request_options(
        timeout_seconds=17,
        max_retries=1,
        event_callback=events.append,
    )
    provider.translate_multivariant(request)
    assert session.calls[0][1]["timeout"] == 17
    assert events[0]["event"] == "request_metrics"
    assert events[1]["event"] == "request_attempt"
    assert {"event": "http_response", "attempt": 1, "status_code": 200} in events
    assert events[-1]["event"] == "response_usage"


def test_request_guard_runs_before_paid_http_request() -> None:
    request = build_translation_request(
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
    session = Session({})
    calls = []

    def deny(event):
        calls.append(event)
        raise RuntimeError("budget exhausted")

    provider = AnthropicClaudeProvider(
        AnthropicConfig(api_key="test", max_retries=1, max_repairs=0),
        session=session,
    ).with_request_options(request_guard=deny)

    with pytest.raises(RuntimeError, match="budget exhausted"):
        provider.translate_multivariant(request)

    assert calls[0]["event"] == "request_attempt"
    assert session.calls == []
