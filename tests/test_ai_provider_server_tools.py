from __future__ import annotations

from datetime import datetime, timezone

from mathula_tv.ai_provider import (
    AnthropicClaudeProvider,
    AnthropicConfig,
    StructuredAIRequest,
)


class _Response:
    status_code = 200
    headers: dict[str, str] = {}
    text = ""

    def __init__(self, data):
        self._data = data

    def json(self):
        return self._data


class _Session:
    def __init__(self) -> None:
        self.requests = []
        self.responses = [
            {
                "id": "message-1",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-4-8",
                "stop_reason": "pause_turn",
                "usage": {"input_tokens": 3, "output_tokens": 2},
                "content": [
                    {
                        "type": "server_tool_use",
                        "id": "srvtoolu_1",
                        "name": "web_search",
                        "input": {"query": "name story"},
                    },
                    {
                        "type": "web_search_tool_result",
                        "tool_use_id": "srvtoolu_1",
                        "content": [
                            {
                                "type": "web_search_result",
                                "url": "https://news.example/story",
                                "title": "Verified story",
                            }
                        ],
                    },
                ],
            },
            {
                "id": "message-2",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-4-8",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 4, "output_tokens": 3},
                "content": [
                    {
                        "type": "text",
                        "text": '{"schema_version":"test-v1","ok":true}',
                    }
                ],
            },
        ]

    def post(self, _url, **kwargs):
        self.requests.append(kwargs["json"])
        return _Response(self.responses.pop(0))


def test_server_web_search_pause_turn_is_resumed_and_evidence_is_retained() -> None:
    session = _Session()
    provider = AnthropicClaudeProvider(
        config=AnthropicConfig(api_key="test-key", max_retries=1),
        session=session,
        sleep=lambda _seconds: None,
        clock=lambda: datetime.now(timezone.utc),
    )
    request = StructuredAIRequest(
        operation="publication_person_resolution",
        payload={"name": "Kanyaku"},
        output_schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["schema_version", "ok"],
            "properties": {
                "schema_version": {"const": "test-v1"},
                "ok": {"const": True},
            },
        },
        prompt_version="test-web-resolution-v1",
        system_prompt="Search and return JSON.",
        response_schema_version="test-v1",
        provider_json_schema=False,
        max_repairs=0,
        server_tools=(
            {
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 2,
            },
        ),
    )

    result = provider.complete_structured(request)

    assert result.data == {"schema_version": "test-v1", "ok": True}
    assert len(session.requests) == 2
    assert session.requests[1]["messages"][-1]["role"] == "assistant"
    assert result.metadata.token_usage.input_tokens == 7
    assert result.metadata.token_usage.output_tokens == 5
    assert result.metadata.server_tool_evidence == (
        {
            "type": "web_search_result",
            "url": "https://news.example/story",
            "title": "Verified story",
            "page_age": "",
        },
    )
