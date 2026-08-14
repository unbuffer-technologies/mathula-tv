from __future__ import annotations

import json

import jsonschema
import pytest

from mathula_tv import ai_provider


def _variant(text: str, ms: int) -> dict:
    return {"t": text, "ms": ms, "p": [], "o": [], "ok": True}


def _wire(**unit_extras) -> dict:
    return {
        "units": [
            {
                "id": "unit_0006",
                "f": ["fact"],
                "o": [],
                "n": _variant("Umbhalo wemvelo", 1900),
                "c": _variant("Umbhalo omfushane", 1500),
                "k": _variant("Mfushane", 1000),
                **unit_extras,
            }
        ],
        "terms": [],
        "warnings": [],
    }


def test_unit_level_t_is_stripped_and_audited() -> None:
    result = ai_provider._canonicalize_compact_multivariant_wire(_wire(t="Mfushane"))
    unit = result["units"][0]
    assert "t" not in unit
    assert unit["n"]["t"] == "Umbhalo wemvelo"
    assert any("unit-level compact variant field t" in item for item in result["warnings"])
    jsonschema.validate(result, ai_provider.COMPACT_MULTIVARIANT_WIRE_SCHEMA)


def test_all_known_unit_level_variant_leaks_are_stripped() -> None:
    result = ai_provider._canonicalize_compact_multivariant_wire(
        _wire(t="Mfushane", ms=1000, p=[], ok=True)
    )
    unit = result["units"][0]
    assert not {"t", "ms", "p", "ok"}.intersection(unit)
    assert len(result["warnings"]) == 4


def test_unknown_unit_structure_still_fails_closed() -> None:
    with pytest.raises(ValueError, match="unexpected fields: source_text"):
        ai_provider._canonicalize_compact_multivariant_wire(
            _wire(source_text="must not be silently discarded")
        )


def test_leaked_field_without_three_variant_objects_is_rejected() -> None:
    value = _wire(t="Mfushane")
    value["units"][0]["k"] = None
    with pytest.raises(ValueError, match="without three complete variant objects"):
        ai_provider._canonicalize_compact_multivariant_wire(value)


def test_tolerant_compact_parser_recovers_paid_response_shape() -> None:
    parsed = ai_provider._parse_compact_multivariant_json(
        json.dumps(_wire(t="Mfushane"), ensure_ascii=False)
    )
    assert parsed["units"][0]["k"]["t"] == "Mfushane"
    assert "t" not in parsed["units"][0]
