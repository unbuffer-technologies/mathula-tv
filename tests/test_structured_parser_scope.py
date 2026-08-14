from __future__ import annotations

import json

import pytest

from mathula_tv.ai_provider import (
    StructuredAIRequest,
    _parse_multivariant_structured_json,
    _parse_structured_json,
)


def test_generic_parser_accepts_hook_panel_object_without_units() -> None:
    payload = {
        "caption": "Umbiko omfishane",
        "cover_hook": "Kucelwe ukukhululwa ngokushesha",
        "hashtags": ["#eNingizimuAfrika"],
    }

    assert _parse_structured_json(json.dumps(payload)) == payload


def test_generic_parser_accepts_fenced_publication_json() -> None:
    text = '''```json
{"hook_text":"Kwenzenjani manje?","panel_duration_ms":2400}
```'''

    assert _parse_structured_json(text) == {
        "hook_text": "Kwenzenjani manje?",
        "panel_duration_ms": 2400,
    }


def test_translation_parser_selects_units_after_glossary() -> None:
    text = '''
{"global_terminology":[{"source_term":"NDPP","target_rendering":"i-NDPP","preserve_english":true,"reason":"acronym"}]}
{"units":[{"unit_id":"unit_0001","variants":[{"variant_id":"natural","spoken_text":"Sawubona"}]}]}
'''

    parsed = _parse_multivariant_structured_json(text)

    assert parsed["units"][0]["unit_id"] == "unit_0001"
    assert parsed["global_terminology"][0]["source_term"] == "NDPP"


def test_translation_parser_rejects_glossary_only_response() -> None:
    text = json.dumps(
        [
            {
                "source_term": "NDPP",
                "target_rendering": "i-NDPP",
                "preserve_english": True,
                "reason": "acronym",
            }
        ]
    )

    with pytest.raises(ValueError, match="unit-shaped translation array"):
        _parse_multivariant_structured_json(text)


def test_structured_request_defaults_to_generic_parser() -> None:
    request = StructuredAIRequest(
        operation="seo_generation",
        payload={},
        output_schema={"type": "object"},
        prompt_version="test-v1",
        system_prompt="Return JSON.",
        response_schema_version="test-response-v1",
    )

    assert request.parser is None


def test_translation_parser_recovers_literal_newline_inside_spoken_text() -> None:
    # Azure Foundry can stream an unescaped line break inside a JSON string.
    # The response is otherwise a complete 42-unit-style translation object.
    text = (
        '{"schema_version":"mathula.translation.multivariant.v1",'
        '"units":[{"unit_id":"unit_0001","source_text":"Source",'
        '"variants":[{"variant_id":"natural",'
        '"spoken_text":"Lokhu kuphenywa kuvez\n ukungqubuzana okuningi."}]}]}'
    )

    parsed = _parse_multivariant_structured_json(text)

    assert parsed["units"][0]["unit_id"] == "unit_0001"
    assert parsed["units"][0]["variants"][0]["spoken_text"] == (
        "Lokhu kuphenywa kuvez ukungqubuzana okuningi."
    )


def test_translation_parser_normalizes_other_raw_control_characters() -> None:
    text = (
        '{"units":[{"unit_id":"unit_0001",'
        '"variants":[{"variant_id":"natural",'
        '"spoken_text":"Izwi\t elilodwa\r manje"}]}]}'
    )

    parsed = _parse_multivariant_structured_json(text)

    assert parsed["units"][0]["variants"][0]["spoken_text"] == (
        "Izwi elilodwa manje"
    )
