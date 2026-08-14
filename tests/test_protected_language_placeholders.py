import jsonschema
from mathula_tv.ai_provider import (
    _prepare_protected_translation_payload,
    _normalise_protected_full_clip_result,
    _prepare_protected_turn_repair_payload,
    _normalise_protected_turn_repair_result,
    _translation_unit_validator,
    build_full_clip_payload,
    FULL_CLIP_TRANSLATION_SCHEMA,
    TURN_REPAIR_SCHEMA,
)


def source_unit(unit_id, text):
    return {"unit_id": unit_id, "source_text": text, "maximum_duration_ms": 5000}


def raw_unit(unit_id, text):
    return {
        "unit_id": unit_id,
        "faithful_translation": text,
        "spoken_text": text,
        "tts_text": text,
        "language_features": [],
        "human_review_flags": [],
    }


def test_single_unit_phrase_is_masked_and_restored_with_tts_alias():
    payload = build_full_clip_payload(
        transcript={"complete_text": "Welcome to The Big W West Side."},
        dubbing_units={"units": [source_unit("unit_0009", "Welcome to The Big W West Side.")]},
        semantic_annotations={"spans": [{
            "span_id": "span_0001",
            "source_text": "The Big W West Side",
            "source_unit_ids": ["unit_0009"],
            "output_unit_id": "unit_0009",
            "policy": "spell_out",
            "display_text": "The Big W West Side",
            "tts_text": "The Big double-you West Side",
        }]},
    )
    assert payload["semantic_annotations"]["spans"]
    masked, placeholders = _prepare_protected_translation_payload(payload)
    token = placeholders[0].token
    assert "The Big W West Side" not in str(masked)
    assert token in masked["dubbing_units"]["units"][0]["source_text"]
    raw = {"units": [raw_unit("unit_0009", f"Siyakwamukela e-{token}.")], "seo": {}}
    restored = _normalise_protected_full_clip_result(masked, placeholders, raw)
    unit = restored["units"][0]
    assert "The Big W West Side" in unit["spoken_text"]
    assert "The Big double-you West Side" in unit["tts_text"]
    assert token not in str(restored)
    jsonschema.validate(restored, FULL_CLIP_TRANSLATION_SCHEMA)


def test_missing_token_is_rejected_before_persistence():
    payload = build_full_clip_payload(
        transcript={"complete_text": "The Big W West Side."},
        dubbing_units={"units": [source_unit("unit_0009", "The Big W West Side.")]},
        semantic_annotations={"spans": [{
            "source_text": "The Big W West Side",
            "source_unit_ids": ["unit_0009"],
            "output_unit_id": "unit_0009",
            "policy": "preserve_verbatim",
        }]},
    )
    masked, placeholders = _prepare_protected_translation_payload(payload)
    raw = {"units": [raw_unit("unit_0009", "Uhlangothi olusentshonalanga.")], "seo": {}}
    try:
        _normalise_protected_full_clip_result(masked, placeholders, raw)
    except ValueError as exc:
        assert "protected placeholder" in str(exc)
    else:
        raise AssertionError("missing immutable placeholder was accepted")


def test_cross_unit_phrase_uses_same_token_and_restores_fragments():
    payload = build_full_clip_payload(
        transcript={"complete_text": "That is a Big W for us."},
        dubbing_units={"units": [
            source_unit("unit_0001", "That is a Big"),
            source_unit("unit_0002", "W for us."),
        ]},
        semantic_annotations={"spans": [{
            "source_text": "Big W",
            "source_unit_ids": ["unit_0001", "unit_0002"],
            "output_unit_id": "unit_0001",
            "policy": "spell_out",
            "display_text": "Big W",
            "tts_text": "Big double-you",
        }]},
    )
    masked, placeholders = _prepare_protected_translation_payload(payload)
    assert len(placeholders) == 2
    token = placeholders[0].token
    assert all(item.token == token for item in placeholders)
    raw = {"units": [
        raw_unit("unit_0001", f"Lokho kuyi-{token}"),
        raw_unit("unit_0002", f"{token} kithi."),
    ], "seo": {}}
    restored = _normalise_protected_full_clip_result(masked, placeholders, raw)
    assert "Big" in restored["units"][0]["spoken_text"]
    assert "W" in restored["units"][1]["spoken_text"]
    assert "double-you" in restored["units"][1]["tts_text"]


def test_turn_repair_masks_and_restores_existing_protected_feature():
    payload = {
        "unit_id": "unit_0009",
        "source_unit": source_unit("unit_0009", "The Big W West Side."),
        "current_translation": {
            **raw_unit("unit_0009", "The Big W West Side."),
            "language_features": [{
                "source_text": "The Big W West Side",
                "policy": "preserve_verbatim",
                "display_text": "The Big W West Side",
                "tts_text": "The Big W West Side",
                "reason": "brand",
            }],
        },
        "constraints": {"protected_verbatim_phrases_must_be_preserved": []},
    }
    masked, placeholders = _prepare_protected_turn_repair_payload(payload)
    token = placeholders[0].token
    raw = {"unit": raw_unit("unit_0009", f"Nansi i-{token}.")}
    restored = _normalise_protected_turn_repair_result(
        masked, placeholders, raw
    )
    assert "The Big W West Side" in restored["unit"]["spoken_text"]
    jsonschema.validate(restored, TURN_REPAIR_SCHEMA)


def test_overbroad_model_feature_is_narrowed_to_protected_placeholder():
    payload = build_full_clip_payload(
        transcript={"complete_text": "He came from the pool of the Big W."},
        dubbing_units={
            "units": [source_unit("unit_0008", "He came from the pool of the Big W.")]
        },
        semantic_annotations={"spans": [{
            "span_id": "span_0001",
            "source_text": "Big W",
            "source_unit_ids": ["unit_0008"],
            "output_unit_id": "unit_0008",
            "policy": "spell_out",
            "display_text": "Big W",
            "tts_text": "Big double-you",
        }]},
    )
    masked, placeholders = _prepare_protected_translation_payload(payload)
    token = placeholders[0].token
    raw = {
        "units": [{
            **raw_unit("unit_0008", f"Uvela echibini le-{token}."),
            "language_features": [{
                "source_text": f"the pool of the {token}",
                "policy": "preserve_verbatim",
                "display_text": f"the pool of the {token}",
                "tts_text": f"the pool of the {token}",
                "reason": "Contains a protected English name",
            }],
        }],
        "seo": {},
    }

    restored = _normalise_protected_full_clip_result(masked, placeholders, raw)
    unit = restored["units"][0]

    assert unit["spoken_text"] == "Uvela echibini le-Big W."
    assert unit["language_features"] == [{
        "source_text": "Big W",
        "policy": "preserve_verbatim",
        "display_text": "Big W",
        "tts_text": "Big double-you",
        "reason": "Contains a protected English name",
    }]
    assert any(
        "Narrowed over-broad protected language feature" in flag
        for flag in unit["human_review_flags"]
    )
    _translation_unit_validator(payload)(restored)
    jsonschema.validate(restored, FULL_CLIP_TRANSLATION_SCHEMA)


def test_overbroad_model_feature_without_placeholder_narrows_to_surviving_code_switch():
    """Regression for the live unit_0008 validation failure.

    There may be no semantic placeholder at all. Claude can still return an
    over-broad language feature while correctly translating the ordinary frame.
    The server must retain the actual code-switched core instead of requiring
    the entire English phrase verbatim.
    """

    payload = build_full_clip_payload(
        transcript={"complete_text": "He came from the pool of the Big W."},
        dubbing_units={
            "units": [source_unit("unit_0008", "He came from the pool of the Big W.")]
        },
        semantic_annotations={"spans": []},
    )
    masked, placeholders = _prepare_protected_translation_payload(payload)
    assert placeholders == []

    raw = {
        "units": [{
            **raw_unit("unit_0008", "Uvela echibini le-Big W."),
            "language_features": [{
                "source_text": "the pool of the Big W",
                "policy": "preserve_verbatim",
                "display_text": "the pool of the Big W",
                "tts_text": "the pool of the Big W",
                "reason": "Contains a protected English name",
            }],
        }],
        "seo": {},
    }

    restored = _normalise_protected_full_clip_result(masked, placeholders, raw)
    unit = restored["units"][0]

    assert unit["spoken_text"] == "Uvela echibini le-Big W."
    assert unit["language_features"] == [{
        "source_text": "Big W",
        "policy": "preserve_verbatim",
        "display_text": "Big W",
        "tts_text": "Big W",
        "reason": "Contains a protected English name",
    }]
    assert any(
        "surviving code-switch span 'Big W'" in flag
        for flag in unit["human_review_flags"]
    )
    _translation_unit_validator(payload)(restored)
    jsonschema.validate(restored, FULL_CLIP_TRANSLATION_SCHEMA)


def test_overbroad_lowercase_feature_does_not_weaken_validation():
    """Ordinary lowercase overlap is not enough to bypass protection checks."""

    payload = build_full_clip_payload(
        transcript={"complete_text": "He came from the pool near us."},
        dubbing_units={
            "units": [source_unit("unit_0008", "He came from the pool near us.")]
        },
        semantic_annotations={"spans": []},
    )
    masked, placeholders = _prepare_protected_translation_payload(payload)
    raw = {
        "units": [{
            **raw_unit("unit_0008", "Uvela pool eduze kwethu."),
            "language_features": [{
                "source_text": "the pool near us",
                "policy": "preserve_verbatim",
                "display_text": "the pool near us",
                "tts_text": "the pool near us",
                "reason": "incorrect broad feature",
            }],
        }],
        "seo": {},
    }

    restored = _normalise_protected_full_clip_result(masked, placeholders, raw)
    try:
        _translation_unit_validator(payload)(restored)
    except ValueError as exc:
        assert "the pool near us" in str(exc)
    else:
        raise AssertionError("ordinary lowercase overlap incorrectly weakened validation")
