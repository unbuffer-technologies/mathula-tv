import json
import jsonschema
import pytest
from mathula_tv.ai_provider import (
    COMPACT_MULTIVARIANT_WIRE_SCHEMA,
    _parse_compact_multivariant_json,
    expand_compact_multivariant_result,
)

def compact(extra=None):
    n = {"t":"Sawubona.","ms":1000,"p":[],"o":[],"ok":True}
    if extra: n.update(extra)
    return {"units":[{"id":"unit_1","f":[],"o":[],"n":n,"c":{"t":"Yebo.","ms":900,"p":[],"o":[],"ok":True},"k":{"t":"Yebo.","ms":900,"p":[],"o":[],"ok":True}}],"terms":[],"warnings":[]}

def test_nested_warnings_are_hoisted_before_strict_schema():
    value = _parse_compact_multivariant_json(json.dumps(compact({"warnings":["review phrase"]})))
    jsonschema.validate(value, COMPACT_MULTIVARIANT_WIRE_SCHEMA)
    assert value["warnings"] == ["unit_1/natural: review phrase"]
    assert "warnings" not in value["units"][0]["n"]
    expanded = expand_compact_multivariant_result(value)
    assert expanded["warnings"] == ["unit_1/natural: review phrase"]

def test_unknown_structural_field_still_fails_closed():
    with pytest.raises(ValueError, match="unexpected fields"):
        _parse_compact_multivariant_json(json.dumps(compact({"invented_translation":"bad"})))
