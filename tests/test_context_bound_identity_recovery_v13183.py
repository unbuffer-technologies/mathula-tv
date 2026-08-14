from __future__ import annotations

import json
from pathlib import Path

import pytest

from mathula_tv import multivariant_translation as translation
from mathula_tv.entity_registry import EntityRegistry
from mathula_tv.multivariant_translation import MultivariantTranslationError
from mathula_tv.one_call_translation import (
    _apply_all_local_identity_repairs,
    _targeted_identity_repair_request,
)


def _source_unit(
    unit_id: str,
    source_text: str,
    protected_spans: list[str],
) -> dict:
    return {
        "unit_id": unit_id,
        "speaker_id": "SPEAKER_00",
        "start_ms": 0,
        "end_ms": 6000,
        "available_duration_ms": 6000,
        "overlap_group": None,
        "source_text": source_text,
        "protected_spans": protected_spans,
        "duration_targets": {
            "natural_target_duration_ms": 5700,
            "concise_target_duration_ms": 4920,
            "compact_target_duration_ms": 4080,
        },
    }


def _raw_unit(source: dict, texts: tuple[str, str, str]) -> dict:
    estimates = (5200, 4400, 3500)
    return {
        "unit_id": source["unit_id"],
        "required_facts": [],
        "optional_details": [],
        "variants": [
            {
                "variant_id": variant_id,
                "compression_level": variant_id,
                "spoken_text": text,
                "estimated_duration_ms": estimate,
                "omitted_optional_details": [],
                "preserved_english_spans": [],
                "meaning_preserved": True,
            }
            for variant_id, text, estimate in zip(
                ("natural", "concise", "compact"),
                texts,
                estimates,
                strict=True,
            )
        ],
        "warnings": [],
    }


def _request(units: list[dict]) -> dict:
    return {
        "schema_version": "mathula.translation.request.v1",
        "prompt_version": translation.PROMPT_VERSION,
        "job_id": "production-regression",
        "source_language": "en-ZA",
        "target_language": "isiZulu",
        "target_locale": "zu-ZA",
        "translation_strategy": "contextual_code_switching",
        "units": units,
        "translation_batch": {
            "schema_version": "mathula.translation.semantic-batch.v1",
            "unit_count": len(units),
            "requested_unit_ids": [item["unit_id"] for item in units],
            "cacheable_context": {"sections": ["large complete story context"]},
            "context_spine_sha256": "a" * 64,
            "context_units": [{"unit_id": "context-only"}],
            "prior_translated_context": [{"unit_id": "prior"}],
            "prior_global_terminology": [{"source_term": "term"}],
            "instructions": [],
            "estimated_source_tokens": 1600,
            "max_output_tokens": 14000,
        },
    }


def _normalized(request: dict, raw_units: list[dict]) -> dict:
    return translation.normalize_multivariant_response(
        {"units": raw_units},
        request=request,
    )


def test_context_binding_absent_from_source_is_advisory_not_injected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aliases = {
        "Witness K": ("Witness K", "Witness-K"),
        "dRK Tactical": ("dRK Tactical", "dRK"),
    }
    monkeypatch.setattr(
        translation,
        "_approved_protected_forms",
        lambda span: aliases.get(span, (span,)),
    )
    source = _source_unit(
        "unit_0008",
        (
            "who had also been arrested alongside Mkhwanazi and a woman "
            "identified as Witness K."
        ),
        ["Witness K", "dRK Tactical"],
    )
    request = _request([source])
    response = _normalized(
        request,
        [
            _raw_unit(
                source,
                (
                    "owayeboshwe noMkhwanazi nowesifazane uWitness K.",
                    "owayeboshwe noMkhwanazi noWitness K.",
                    "uMkhwanazi noWitness K baboshwa naye.",
                ),
            )
        ],
    )

    validation = translation.validate_multivariant_response(
        response,
        request=request,
    )

    reviews = validation["units"][0]["contextual_identity_reviews"]
    assert validation["valid"] is True
    assert validation["contextual_identity_review_count"] == 3
    assert {item["required_span"] for item in reviews} == {"dRK Tactical"}
    assert {item["review_policy"] for item in reviews} == {
        "context_binding_not_lexically_grounded_in_source_unit"
    }
    assert all(
        "dRK" not in variant["spoken_text"]
        for variant in response["units"][0]["variants"]
    )


def test_single_letter_registry_alias_cannot_bind_different_entity(
    tmp_path: Path,
) -> None:
    registry_path = tmp_path / "entity_registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": "mathula-entity-registry-v1",
                "entities": [
                    {
                        "entity_id": "organisation_drk_tactical",
                        "entity_type": "organisation",
                        "canonical_text": "dRK Tactical",
                        "display_text": "dRK Tactical",
                        "aliases": ["dRK", "K"],
                        "domains": ["news"],
                        "roles": [],
                        "source_status": "reviewed",
                        "active": True,
                        "must_preserve": True,
                        "translation_policy": "protected_token",
                        "stt_phrases": ["dRK Tactical", "K"],
                        "spoken_forms": {},
                        "do_not_confuse_with": [],
                        "sources": [],
                    }
                ],
                "validation": {"alias_collisions": []},
            }
        ),
        encoding="utf-8",
    )
    registry = EntityRegistry(registry_path)

    assert registry.match_entities("a woman identified as Witness K") == []
    matches = registry.match_entities("The director of dRK Tactical testified")
    assert [item.entity_id for item in matches] == [
        "organisation_drk_tactical"
    ]


def test_successive_missing_identities_are_repaired_locally_in_one_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aliases = {
        "Ekurhuleni Metropolitan Police Department": (
            "Ekurhuleni Metropolitan Police Department",
            "EMPD",
            "Ekurhuleni Metro Police",
        ),
        "Investigating Directorate Against Corruption": (
            "Investigating Directorate Against Corruption",
            "IDAC",
        ),
    }
    monkeypatch.setattr(
        translation,
        "_approved_protected_forms",
        lambda span: aliases.get(span, (span,)),
    )
    empd = _source_unit(
        "unit_0013",
        "The Ekurhuleni Metro Police Deputy Chief will appear.",
        ["Ekurhuleni Metropolitan Police Department"],
    )
    idac = _source_unit(
        "unit_0014",
        "The Investigating Directorate Against Corruption will investigate.",
        ["Investigating Directorate Against Corruption"],
    )
    request = _request([empd, idac])
    response = _normalized(
        request,
        [
            _raw_unit(
                empd,
                (
                    "IPhini leNhloko lizovela.",
                    "IPhini lizovela.",
                    "Lizovela.",
                ),
            ),
            _raw_unit(
                idac,
                (
                    "Lolu phiko luzophenya.",
                    "Luzophenya lolu daba.",
                    "Luzophenya.",
                ),
            ),
        ],
    )
    first_error = (
        "MultivariantTranslationError: Unit unit_0013 variant natural is "
        "missing protected spans or approved identity aliases: "
        "['Ekurhuleni Metropolitan Police Department (accepted forms: "
        "Ekurhuleni Metropolitan Police Department, EMPD, Ekurhuleni Metro Police)']"
    )

    normalized, validation, repairs, issues = _apply_all_local_identity_repairs(
        candidate=response,
        batch_request=request,
        first_issue={
            "unit_id": "unit_0013",
            "variant_id": "natural",
            "validation_error": first_error,
        },
    )

    assert validation["valid"] is True
    assert [item["unit_id"] for item in issues] == ["unit_0013", "unit_0014"]
    assert {item["inserted_accepted_form"] for item in repairs} == {
        "EMPD",
        "IDAC",
    }
    assert all(
        variant["spoken_text"].startswith("EMPD,")
        for variant in normalized["units"][0]["variants"]
    )
    assert all(
        variant["spoken_text"].startswith("IDAC,")
        for variant in normalized["units"][1]["variants"]
    )


def test_identity_provider_fallback_drops_full_chunk_context() -> None:
    source = _source_unit(
        "unit_0013",
        "The Ekurhuleni Metro Police Deputy Chief will appear.",
        ["Ekurhuleni Metropolitan Police Department"],
    )
    request = _request([source])
    candidate = _normalized(
        request,
        [_raw_unit(source, ("Uzovela manje.", "Uzovela.", "Uyeza."))],
    )
    issue = {
        "unit_id": "unit_0013",
        "variant_id": "natural",
        "validation_error": (
            "Unit unit_0013 variant natural is missing protected spans or "
            "approved identity aliases: ['Ekurhuleni Metropolitan Police "
            "Department (accepted forms: EMPD)']"
        ),
    }

    repair = _targeted_identity_repair_request(
        batch_request=request,
        candidate=candidate,
        issue=issue,
    )
    batch = repair["translation_batch"]

    assert "cacheable_context" not in batch
    assert "context_spine_sha256" not in batch
    assert batch["context_units"] == []
    assert batch["prior_translated_context"] == []
    assert batch["prior_global_terminology"] == []
    assert batch["max_output_tokens"] == 6000
    assert batch["repair_context_policy"] == "single_unit_identity_contract_only"

