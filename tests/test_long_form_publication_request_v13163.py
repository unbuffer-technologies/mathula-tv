from __future__ import annotations

import json

import pytest

from mathula_tv.ai_provider import (
    AIProviderRequestRejected,
    AnthropicClaudeProvider,
    AnthropicConfig,
    StructuredAIRequest,
)
from mathula_tv.tiktok_editor import _compact_editorial_source_transcript


class _Response:
    status_code = 400
    headers: dict[str, str] = {}

    def json(self):
        return {
            "error": {
                "type": "invalid_request_error",
                "message": "prompt is too long for this model",
            },
            "api_key": "must-not-leak",
        }

    def close(self):
        pass


class _Session:
    def post(self, *_args, **_kwargs):
        return _Response()


def _request(*, stream_response: bool) -> StructuredAIRequest:
    return StructuredAIRequest(
        operation="editorial_package",
        payload={"input": "safe"},
        output_schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["ok"],
            "properties": {"ok": {"type": "boolean"}},
        },
        prompt_version="test-long-form-publication-v1",
        system_prompt="Return JSON.",
        response_schema_version="test-v1",
        stream_response=stream_response,
    )


def test_long_form_transcript_keeps_all_segments_without_word_ledger() -> None:
    transcript = {
        "schema_version": "transcript-v1",
        "language": "en-ZA",
        "segments": [
            {
                "segment_id": "seg-00001",
                "speaker_id": "SPEAKER_00",
                "start": 0.0,
                "end": 12.5,
                "source_text": "The complete authoritative first segment.",
                "source_word_ids": ["word_000001", "word_000002"],
            },
            {
                "segment_id": "seg-00002",
                "speaker": "SPEAKER_01",
                "start": 12.5,
                "end": 25.0,
                "source_text": "The complete authoritative second segment.",
            },
        ],
        "words": [
            {
                "word_id": f"word_{index:06d}",
                "text": "duplicate",
                "speaker_assignment_confidence": 0.99,
            }
            for index in range(1, 6001)
        ],
    }

    compact = _compact_editorial_source_transcript(transcript)

    assert "words" not in compact
    assert compact["complete_segment_count"] == 2
    assert compact["omitted_duplicate_word_timing_records"] == 6000
    assert [item["source_text"] for item in compact["segments"]] == [
        "The complete authoritative first segment.",
        "The complete authoritative second segment.",
    ]
    assert compact["segments"][0] == {
        "segment_id": "seg-00001",
        "speaker_id": "SPEAKER_00",
        "start_seconds": 0.0,
        "end_seconds": 12.5,
        "source_text": "The complete authoritative first segment.",
    }


@pytest.mark.parametrize("stream_response", [False, True])
def test_http_400_preserves_safe_azure_reason(stream_response: bool) -> None:
    provider = AnthropicClaudeProvider(
        AnthropicConfig(
            api_key="top-secret",
            max_retries=1,
            max_repairs=0,
        ),
        session=_Session(),
        sleep=lambda _delay: None,
    )

    with pytest.raises(AIProviderRequestRejected) as caught:
        provider.complete_structured(
            _request(stream_response=stream_response)
        )

    rendered = json.dumps(caught.value.to_dict())
    assert "invalid_request_error" in rendered
    assert "prompt is too long for this model" in rendered
    assert "must-not-leak" not in rendered
    assert "<redacted>" in rendered
