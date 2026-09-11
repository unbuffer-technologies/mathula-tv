import json

import jsonschema

from mathula_tv.ai_provider import (
    COMPACT_MULTIVARIANT_WIRE_SCHEMA,
    COMPACT_MULTIVARIANT_WIRE_SCHEMA_VERSION,
    _expand_compact_multivariant_result,
    _parse_compact_multivariant_json,
)


def test_compact_v3_omits_source_owned_ledgers_and_restores_them_locally():
    assert COMPACT_MULTIVARIANT_WIRE_SCHEMA_VERSION.endswith(".v3")
    unit_schema = COMPACT_MULTIVARIANT_WIRE_SCHEMA["properties"]["units"]["items"]
    assert unit_schema["required"] == ["id", "n", "c", "k"]
    assert "f" not in unit_schema["properties"]
    assert "o" not in unit_schema["properties"]
    variant_schema = unit_schema["properties"]["n"]
    assert "p" not in variant_schema["required"]
    assert "p" not in variant_schema["properties"]

    source_units = [
        {
            "unit_id": "unit_0001",
            "required_facts": ["The witness testified."],
            "optional_details": ["in the morning"],
        }
    ]
    compact = {
        "units": [
            {
                "id": "unit_0001",
                "n": {"t": "Ufakazi wethule ubufakazi.", "ms": 2100, "o": [], "ok": True},
                "c": {"t": "Ufakazi ufakazile.", "ms": 1700, "o": [], "ok": True},
                "k": {"t": "Ufakazi ufakazile.", "ms": 1600, "o": [], "ok": True},
            }
        ],
        "terms": [],
        "warnings": [],
    }

    expanded = _expand_compact_multivariant_result(
        compact,
        source_units=source_units,
    )
    unit = expanded["units"][0]
    assert unit["required_facts"] == ["The witness testified."]
    assert unit["optional_details"] == ["in the morning"]
    assert all(variant["preserved_english_spans"] == [] for variant in unit["variants"])


def test_compact_v3_accepts_v2_wire_but_keeps_source_ledgers_authoritative():
    source_units = [
        {
            "unit_id": "unit_0001",
            "required_facts": ["authoritative fact"],
            "optional_details": ["authoritative detail"],
            "protected_spans": ["EFF"],
        }
    ]
    legacy_v2 = {
        "units": [
            {
                "id": "unit_0001",
                "f": ["model-supplied fact"],
                "o": ["model-supplied detail"],
                "n": {
                    "t": "I-EFF iyakhuluma.",
                    "ms": 1800,
                    "p": ["EFF"],
                    "o": [],
                    "ok": True,
                },
                "c": {
                    "t": "I-EFF iyasho.",
                    "ms": 1400,
                    "p": ["EFF"],
                    "o": [],
                    "ok": True,
                },
                "k": {
                    "t": "I-EFF.",
                    "ms": 1000,
                    "p": ["EFF"],
                    "o": [],
                    "ok": True,
                },
            }
        ],
        "terms": [],
        "warnings": [],
    }

    canonical = _parse_compact_multivariant_json(
        json.dumps(legacy_v2),
        source_units=source_units,
    )
    jsonschema.validate(canonical, COMPACT_MULTIVARIANT_WIRE_SCHEMA)
    assert "f" not in canonical["units"][0]
    assert "o" not in canonical["units"][0]
    assert "p" not in canonical["units"][0]["n"]

    expanded = _expand_compact_multivariant_result(
        canonical,
        source_units=source_units,
    )
    assert expanded["units"][0]["required_facts"] == ["authoritative fact"]
    assert expanded["units"][0]["optional_details"] == ["authoritative detail"]
    assert expanded["units"][0]["variants"][0][
        "preserved_english_spans"
    ] == ["EFF"]
