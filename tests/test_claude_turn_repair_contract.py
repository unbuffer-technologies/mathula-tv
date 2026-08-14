from __future__ import annotations

import jsonschema
import pytest

from mathula_tv.ai_provider import (
    AnthropicClaudeProvider,
    AnthropicConfig,
    TURN_REPAIR_SCHEMA,
    TURN_TIMING_REPAIR_OPTIONS_SCHEMA,
    _normalise_turn_repair_result,
)


PAYLOAD = {
    "source_unit": {
        "unit_id": "unit_0002",
        "source_text": "That is a big W.",
        "maximum_duration_ms": 2200,
    },
    "constraints": {
        "protected_verbatim_phrases_must_be_preserved": ["big W"],
    },
}

REPAIRED_UNIT = {
    "unit_id": "unit_0002",
    "faithful_translation": "Leyo yi-big W.",
    "spoken_text": "Leyo yi-big W.",
    "tts_text": "Leyo yi-big double-you.",
    "language_features": [
        {
            "source_text": "big W",
            "policy": "spell_out",
            "display_text": "big W",
            "tts_text": "big double-you",
            "reason": "Preserve the code-switched expression.",
        }
    ],
    "human_review_flags": [],
}


@pytest.mark.parametrize(
    "raw_model_result",
    [
        {"schema_version": "claude-turn-repair-v1", "unit": REPAIRED_UNIT},
        {"schema_version": "claude-turn-repair-v1", "units": [REPAIRED_UNIT]},
    ],
)
def test_turn_repair_accepts_unit_and_single_item_units_envelopes(monkeypatch, raw_model_result):
    provider = AnthropicClaudeProvider(AnthropicConfig(api_key="test-key"))

    def complete_without_network(request):
        value = request.normalizer(raw_model_result)
        jsonschema.validate(value, dict(request.output_schema))
        request.validator(value)
        return value

    monkeypatch.setattr(provider, "complete_structured", complete_without_network)

    repaired = provider.repair_turn(PAYLOAD, output_schema=TURN_REPAIR_SCHEMA)

    assert repaired["schema_version"] == "claude-turn-repair-v1"
    assert repaired["unit"]["unit_id"] == "unit_0002"
    assert repaired["unit"]["spoken_text"] == "Leyo yi-big W."


def test_turn_repair_rejects_multiple_units():
    with pytest.raises(ValueError, match="exactly one unit"):
        _normalise_turn_repair_result(
            PAYLOAD,
            {"units": [REPAIRED_UNIT, {**REPAIRED_UNIT, "unit_id": "unit_0003"}]},
        )


def test_turn_timing_repair_options_contract(monkeypatch):
    provider = AnthropicClaudeProvider(AnthropicConfig(api_key="test-key"))
    result = {
        "schema_version": "claude-turn-timing-repair-options-v1",
        "candidates": [
            {
                "candidate_id": "balanced",
                "estimated_duration_ms": 2050,
                "compression_strategy": "balanced",
                "unit": REPAIRED_UNIT,
            },
            {
                "candidate_id": "safe",
                "estimated_duration_ms": 1850,
                "compression_strategy": "extra headroom",
                "unit": {
                    **REPAIRED_UNIT,
                    "faithful_translation": "Yi-big W.",
                    "spoken_text": "Yi-big W.",
                    "tts_text": "Yi-big double-you.",
                },
            },
            {
                "candidate_id": "maximum_compression",
                "estimated_duration_ms": 1650,
                "compression_strategy": "maximum faithful compression",
                "unit": {
                    **REPAIRED_UNIT,
                    "faithful_translation": "Big W.",
                    "spoken_text": "Big W.",
                    "tts_text": "Big double-you.",
                },
            },
        ],
    }

    def complete_without_network(request):
        value = request.normalizer(result)
        jsonschema.validate(value, dict(request.output_schema))
        request.validator(value)
        return value

    monkeypatch.setattr(provider, "complete_structured", complete_without_network)

    repaired = provider.repair_turn(
        PAYLOAD,
        output_schema=TURN_TIMING_REPAIR_OPTIONS_SCHEMA,
    )

    assert repaired["schema_version"] == "claude-turn-timing-repair-options-v1"
    assert [item["candidate_id"] for item in repaired["candidates"]] == [
        "balanced",
        "safe",
        "maximum_compression",
    ]
    assert repaired["candidates"][1]["unit"]["spoken_text"] == "Yi-big W."


def test_turn_timing_repair_batch_contract(monkeypatch):
    from mathula_tv.ai_provider import TURN_TIMING_REPAIR_BATCH_SCHEMA

    provider = AnthropicClaudeProvider(AnthropicConfig(api_key="test-key"))
    second_payload = {
        **PAYLOAD,
        "unit_id": "unit_0003",
        "source_unit": {
            **PAYLOAD["source_unit"],
            "unit_id": "unit_0003",
            "source_text": "That is another big W.",
        },
    }
    batch_payload = {
        "schema_version": "turn-timing-repair-batch-request-v1",
        "target_language": "zu-ZA",
        "repairs": [{"unit_id": "unit_0002", **PAYLOAD}, second_payload],
    }
    result = {
        "schema_version": "claude-turn-timing-repair-batch-v1",
        "repairs": [
            {
                "unit_id": "unit_0002",
                "candidates": [
                    {
                        "candidate_id": "balanced",
                        "estimated_duration_ms": 2050,
                        "compression_strategy": "balanced",
                        "unit": REPAIRED_UNIT,
                    },
                    {
                        "candidate_id": "safe",
                        "estimated_duration_ms": 1850,
                        "compression_strategy": "safe",
                        "unit": {**REPAIRED_UNIT, "spoken_text": "Yi-big W.", "faithful_translation": "Yi-big W.", "tts_text": "Yi-big double-you."},
                    },
                    {
                        "candidate_id": "maximum_compression",
                        "estimated_duration_ms": 1650,
                        "compression_strategy": "maximum",
                        "unit": {**REPAIRED_UNIT, "spoken_text": "Big W.", "faithful_translation": "Big W.", "tts_text": "Big double-you."},
                    },
                ],
            },
            {
                "unit_id": "unit_0003",
                "candidates": [
                    {
                        "candidate_id": "balanced",
                        "estimated_duration_ms": 2100,
                        "compression_strategy": "balanced",
                        "unit": {**REPAIRED_UNIT, "unit_id": "unit_0003", "spoken_text": "Lena yi-big W.", "faithful_translation": "Lena yi-big W.", "tts_text": "Lena yi-big double-you."},
                    },
                    {
                        "candidate_id": "safe",
                        "estimated_duration_ms": 1900,
                        "compression_strategy": "safe",
                        "unit": {**REPAIRED_UNIT, "unit_id": "unit_0003", "spoken_text": "Yi-big W futhi.", "faithful_translation": "Yi-big W futhi.", "tts_text": "Yi-big double-you futhi."},
                    },
                    {
                        "candidate_id": "maximum_compression",
                        "estimated_duration_ms": 1700,
                        "compression_strategy": "maximum",
                        "unit": {**REPAIRED_UNIT, "unit_id": "unit_0003", "spoken_text": "Big W futhi.", "faithful_translation": "Big W futhi.", "tts_text": "Big double-you futhi."},
                    },
                ],
            },
        ],
    }

    def complete_without_network(request):
        value = request.normalizer(result)
        jsonschema.validate(value, dict(request.output_schema))
        request.validator(value)
        return value

    monkeypatch.setattr(provider, "complete_structured", complete_without_network)
    repaired = provider.repair_turn_batch(
        batch_payload,
        output_schema=TURN_TIMING_REPAIR_BATCH_SCHEMA,
    )

    assert [item["unit_id"] for item in repaired["repairs"]] == [
        "unit_0002",
        "unit_0003",
    ]
    assert repaired["repairs"][1]["candidates"][0]["unit"]["unit_id"] == "unit_0003"
