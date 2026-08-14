from __future__ import annotations

import json

from mathula_tv.ai_provider import (
    AZURE_FOUNDRY_CLAUDE_PROVIDER,
    COMPACT_MULTIVARIANT_WIRE_SCHEMA,
    AnthropicClaudeProvider,
    AnthropicConfig,
    _parse_compact_multivariant_json,
)
from mathula_tv.multivariant_translation import build_translation_request


def _provider() -> AnthropicClaudeProvider:
    class CaptureProvider(AnthropicClaudeProvider):
        def complete_structured(self, request):
            return request

    return CaptureProvider(
        AnthropicConfig(
            api_key="test",
            provider=AZURE_FOUNDRY_CLAUDE_PROVIDER,
            model="claude-opus-4-8",
            base_url="https://mathula-test.services.ai.azure.com/anthropic",
            translation_max_output_tokens=100_000,
        )
    )


def _request():
    source = {
        "language": "en-ZA",
        "segments": [
            {
                "segment_id": "unit_0001",
                "speaker": "SPEAKER_01",
                "start": 0,
                "end": 4,
                "source_text": (
                    "The Madlanga Commission of Inquiry heard the evidence."
                ),
                "protected_spans": ["Madlanga Commission of Inquiry"],
            }
        ],
    }
    request = build_translation_request(
        transcript=source,
        target_locale="zu-ZA",
        job_id="job-compact",
    )
    request["translation_batch"] = {
        "max_output_tokens": 14_000,
        "cacheable_context": {"complete_timeline_coverage": True},
    }
    return request


def _compact_response():
    return {
        "units": [
            {
                "id": "unit_0001",
                "f": ["the commission heard evidence"],
                "o": [],
                "n": {
                    "t": "IKhomishini izwe ubufakazi.",
                    "ms": 2500,
                    "p": [],
                    "o": [],
                    "ok": True,
                },
                "c": {
                    "t": "IKhomishini izwe ubufakazi.",
                    "ms": 2200,
                    "p": [],
                    "o": [],
                    "ok": True,
                },
                "k": {
                    "t": "IKhomishini izwe.",
                    "ms": 1800,
                    "p": [],
                    "o": [],
                    "ok": True,
                },
            }
        ],
        "terms": [],
        "warnings": [],
    }


def test_chunk_requests_use_compact_wire_schema_and_safe_output_cap():
    provider = _provider()
    request = provider.translate_multivariant(_request())

    assert request.provider_output_schema is COMPACT_MULTIVARIANT_WIRE_SCHEMA
    assert request.max_output_tokens == 6000
    assert "COMPACT WIRE FORMAT" in request.system_prompt
    body = provider.build_request(request)
    dynamic_text = body["messages"][0]["content"][1]["text"]
    assert '"id"' in dynamic_text
    assert '"n"' in dynamic_text


def test_compact_wire_response_expands_to_full_validated_artifact():
    request = _provider().translate_multivariant(_request())
    compact = _compact_response()
    full = request.normalizer(compact)
    request.validator(full)

    unit = full["units"][0]
    assert unit["speaker_id"] == "SPEAKER_01"
    assert unit["source_text"].startswith("The Madlanga")
    assert all(
        "kaMadlanga" in variant["spoken_text"]
        for variant in unit["variants"]
    )
    assert len(json.dumps(compact, ensure_ascii=False)) < (
        len(json.dumps(full, ensure_ascii=False)) * 0.45
    )


def test_compact_parser_tolerates_streamed_control_characters():
    raw = json.dumps(_compact_response(), ensure_ascii=False)
    raw = raw.replace("IKhomishini izwe ubufakazi.", "IKhomishini\\nizwe ubufakazi.", 1)
    parsed = _parse_compact_multivariant_json("provider prefix\n" + raw)
    assert parsed["units"][0]["n"]["t"] == "IKhomishini izwe ubufakazi."
