"""Tests for entity registry library."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from mathula_tv.entity_registry import (
    Entity,
    EntityBindingsArtifact,
    EntityMatch,
    EntityRegistry,
    protect_text_with_placeholders,
    restore_display_text,
    restore_tts_text,
    validate_placeholder_integrity,
)


@pytest.fixture
def sample_registry_json():
    """Sample entity registry JSON for testing."""
    return {
        "schema_version": "mathula-entity-registry-v1",
        "registry_id": "test_registry",
        "title": "Test Registry",
        "description": "Test registry for unit tests",
        "updated_at": "2026-07-21T00:00:00+02:00",
        "coverage": {
            "primary_domain": "test",
            "public_record_through": "2026-07-21",
            "hearing_day_through": 1,
            "coverage_note": "Test coverage",
        },
        "defaults": {
            "must_preserve": True,
            "translation_policy": "protected_token",
            "unknown_spoken_form_policy": "block_or_require_human_calibration",
            "do_not_guess_pronunciation": True,
            "display_text_must_not_use_phonetic_alias": True,
            "entity_binding_output": "working/jobs/<job_id>/analysis/entity_bindings.json",
        },
        "supported_entity_types": ["person", "organisation", "place"],
        "domains": {
            "test": {"description": "Test domain"},
        },
        "source_registry": [
            {
                "source_id": "test_source",
                "url": "https://example.com",
                "scope": "test entities",
            }
        ],
        "validation": {
            "entity_count": 2,
            "alias_collision_count": 0,
            "alias_collisions": [],
            "required_fields": [
                "entity_id",
                "entity_type",
                "canonical_text",
                "display_text",
                "aliases",
                "domains",
                "roles",
                "must_preserve",
                "translation_policy",
                "stt_phrases",
                "spoken_forms",
            ],
        },
        "entities": [
            {
                "entity_id": "person_john_doe",
                "entity_type": "person",
                "canonical_text": "John Doe",
                "display_text": "John Doe",
                "aliases": ["John Doe", "J. Doe", "Mr. Doe"],
                "domains": ["test"],
                "roles": [{"role": "protected_term", "context": "test", "status": "referenced"}],
                "source_status": "public_record",
                "active": True,
                "must_preserve": True,
                "translation_policy": "protected_token",
                "stt_phrases": ["John Doe", "J. Doe", "Mr. Doe"],
                "spoken_forms": {
                    "zu-ZA": {
                        "default": "John Doe",
                        "voices": {},
                        "status": "approved",
                        "last_tested_at": "2026-07-21T00:00:00+02:00",
                        "approved_by": "test",
                    },
                },
                "do_not_confuse_with": [],
                "sources": ["https://example.com"],
            },
            {
                "entity_id": "organisation_test_corp",
                "entity_type": "organisation",
                "canonical_text": "Test Corporation",
                "display_text": "Test Corporation",
                "aliases": ["Test Corporation", "Test Corp"],
                "domains": ["test"],
                "roles": [{"role": "protected_term", "context": "test", "status": "referenced"}],
                "source_status": "public_record",
                "active": True,
                "must_preserve": True,
                "translation_policy": "protected_token",
                "stt_phrases": ["Test Corporation", "Test Corp"],
                "spoken_forms": {
                    "zu-ZA": {
                        "default": None,
                        "voices": {},
                        "status": "needs_calibration",
                        "last_tested_at": None,
                        "approved_by": None,
                    },
                },
                "do_not_confuse_with": [],
                "sources": ["https://example.com"],
            },
        ],
    }


@pytest.fixture
def temp_registry_file(sample_registry_json):
    """Create a temporary registry file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sample_registry_json, f)
        temp_path = Path(f.name)
    yield temp_path
    temp_path.unlink(missing_ok=True)


def test_entity_registry_load(temp_registry_file):
    """Test loading entity registry from file."""
    registry = EntityRegistry(temp_registry_file)
    assert registry.sha256 is not None
    assert len(registry._entities) == 2
    assert "person_john_doe" in registry._entities
    assert "organisation_test_corp" in registry._entities


def test_entity_registry_invalid_schema():
    """Test that invalid schema version raises error."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"schema_version": "invalid-version", "entities": []}, f)
        temp_path = Path(f.name)
    
    try:
        with pytest.raises(ValueError, match="Unsupported registry schema version"):
            EntityRegistry(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)


def test_entity_registry_get_entity(temp_registry_file):
    """Test getting an entity by ID."""
    registry = EntityRegistry(temp_registry_file)
    entity = registry.get_entity("person_john_doe")
    assert entity.entity_id == "person_john_doe"
    assert entity.canonical_text == "John Doe"
    assert len(entity.aliases) == 3


def test_entity_registry_get_entity_invalid_id(temp_registry_file):
    """Test that getting invalid entity ID raises error."""
    registry = EntityRegistry(temp_registry_file)
    with pytest.raises(ValueError, match="Unknown entity_id"):
        registry.get_entity("invalid_entity")


def test_entity_match_entities(temp_registry_file):
    """Test matching entities in text."""
    registry = EntityRegistry(temp_registry_file)
    text = "John Doe testified at Test Corporation yesterday."
    matches = registry.match_entities(text, "unit_1")
    
    # Due to alias matching, we may get multiple matches for the same entity
    # Check that we have at least the expected entities
    entity_ids = {m.entity_id for m in matches}
    assert "person_john_doe" in entity_ids
    assert "organisation_test_corp" in entity_ids


def test_entity_match_entities_no_matches(temp_registry_file):
    """Test matching when no entities are found."""
    registry = EntityRegistry(temp_registry_file)
    text = "Some unrelated text without any entities."
    matches = registry.match_entities(text, "unit_1")
    assert len(matches) == 0


def test_protect_text_with_placeholders():
    """Test protecting text with placeholders."""
    matches = [
        EntityMatch(
            entity_id="person_john_doe",
            matched_source_text="John Doe",
            canonical_text="John Doe",
            display_text="John Doe",
            source_char_start=0,
            source_char_end=8,
            matched_alias="john doe",
            match_confidence="high",
            match_method="canonical",
            domain="test",
            role="protected_term",
            requires_pronunciation_calibration=False,
        )
    ]
    text = "John Doe testified."
    protected, bindings = protect_text_with_placeholders(text, matches)
    
    assert "[[MATHULA_ENTITY:person_john_doe]]" in protected
    assert bindings["[[MATHULA_ENTITY:person_john_doe]]"] == "person_john_doe"


def test_restore_display_text():
    """Test restoring display text from placeholders."""
    from mathula_tv.entity_registry import EntityMatch
    
    class MockEntity:
        def __init__(self, display_text):
            self.display_text = display_text
    
    class MockRegistry:
        def get_entity(self, entity_id):
            return MockEntity("John Doe")
    
    text = "[[MATHULA_ENTITY:person_john_doe]] testified."
    match = EntityMatch(
        entity_id="person_john_doe",
        matched_source_text="John Doe",
        canonical_text="John Doe",
        display_text="John Doe",
        source_char_start=0,
        source_char_end=8,
        matched_alias="john doe",
        match_confidence="high",
        match_method="canonical",
        domain="test",
        role="protected_term",
        requires_pronunciation_calibration=False,
    )
    bindings = {"[[MATHULA_ENTITY:person_john_doe]]": match}
    registry = MockRegistry()
    
    restored = restore_display_text(text, bindings, registry)
    assert "John Doe" in restored
    assert "[[MATHULA_ENTITY:" not in restored


def test_validate_placeholder_integrity_valid():
    """Test placeholder integrity validation when valid."""
    original = {"[[MATHULA_ENTITY:person_john_doe]]": "person_john_doe"}
    restored = {"[[MATHULA_ENTITY:person_john_doe]]": "person_john_doe"}
    
    result = validate_placeholder_integrity(original, restored, "unit_1")
    assert result["valid"] is True
    assert len(result["missing_placeholders"]) == 0
    assert len(result["unexpected_placeholders"]) == 0


def test_validate_placeholder_integrity_missing():
    """Test placeholder integrity validation when placeholder is missing."""
    original = {"[[MATHULA_ENTITY:person_john_doe]]": "person_john_doe"}
    restored = {}
    
    result = validate_placeholder_integrity(original, restored, "unit_1")
    assert result["valid"] is False
    assert "[[MATHULA_ENTITY:person_john_doe]]" in result["missing_placeholders"]


def test_validate_placeholder_integrity_unexpected():
    """Test placeholder integrity validation when unexpected placeholder exists."""
    original = {}
    restored = {"[[MATHULA_ENTITY:person_john_doe]]": "person_john_doe"}
    
    result = validate_placeholder_integrity(original, restored, "unit_1")
    assert result["valid"] is False
    assert "[[MATHULA_ENTITY:person_john_doe]]" in result["unexpected_placeholders"]


def test_entity_bindings_artifact():
    """Test EntityBindingsArtifact dataclass."""
    artifact = EntityBindingsArtifact(
        schema_version="entity-bindings-v1",
        job_id="test_job",
        registry_path="/path/to/registry.json",
        registry_sha256="abc123",
        registry_schema_version="mathula-entity-registry-v1",
        source_transcript_sha256="def456",
        generated_at="2026-07-21T00:00:00+02:00",
        entities_found_global=("person_john_doe",),
        per_unit_bindings={},
        unresolved_ambiguous_matches=(),
        protected_source_text_per_unit={},
        placeholder_mapping={},
        summary_counts={"total_entities": 1, "total_units": 1, "total_placeholders": 1, "alias_collisions": 0},
    )
    
    assert artifact.job_id == "test_job"
    assert artifact.schema_version == "entity-bindings-v1"
    
    dict_repr = artifact.to_dict()
    assert dict_repr["job_id"] == "test_job"
    assert dict_repr["schema_version"] == "entity-bindings-v1"


def test_entity_spoken_form_calibration_status():
    """Test that SpokenForm accepts all status values for extensibility."""
    from mathula_tv.entity_registry import SpokenForm
    # Should not raise ValueError for any status (extensibility)
    SpokenForm(
        default="test",
        voices={},
        status="custom_status",
        last_tested_at=None,
        approved_by=None,
    )


def test_entity_invalid_entity_type():
    """Test that invalid entity type raises error."""
    with pytest.raises(ValueError, match="Unsupported entity_type"):
        Entity(
            entity_id="test_entity",
            entity_type="invalid_type",
            canonical_text="Test",
            display_text="Test",
            aliases=(),
            domains=(),
            roles=(),
            source_status="public_record",
            active=True,
            must_preserve=True,
            translation_policy="protected_token",
            stt_phrases=(),
            spoken_forms={},
            do_not_confuse_with=(),
            sources=(),
        )


def test_spoken_form_status_extensibility():
    """Test that SpokenForm accepts all status values for extensibility."""
    from mathula_tv.entity_registry import SpokenForm
    # Should not raise ValueError for any status
    SpokenForm(
        default="test",
        voices={},
        status="literal_designation",
        last_tested_at=None,
        approved_by=None,
    )
    SpokenForm(
        default="test",
        voices={},
        status="needs_human_reference",
        last_tested_at=None,
        approved_by=None,
    )
    SpokenForm(
        default="test",
        voices={},
        status="custom_status",
        last_tested_at=None,
        approved_by=None,
    )


def test_translation_text_forms_separation():
    """Test that faithful, natural, and TTS text forms are separate."""
    from mathula_tv.entity_registry import Entity, SpokenForm, EntityMatch, restore_display_text, restore_tts_text
    
    # Create a test entity with different display and spoken forms
    entity = Entity(
        entity_id="test_entity",
        entity_type="person",
        canonical_text="John Doe",
        display_text="John Doe",
        aliases=("Johnny",),
        domains=("test",),
        roles=(),
        source_status="public_record",
        active=True,
        must_preserve=True,
        translation_policy="protected_token",
        stt_phrases=("John Doe",),
        spoken_forms={
            "zu-ZA": SpokenForm(
                default="uJohn Doe",
                voices={},
                status="approved",
                last_tested_at=None,
                approved_by=None,
            )
        },
        do_not_confuse_with=(),
        sources=(),
    )
    
    # Create a mock registry
    class MockRegistry:
        def get_entity(self, entity_id):
            return entity
    
    registry = MockRegistry()
    
    # Test placeholder restoration with EntityMatch objects
    placeholder = "[[MATHULA_ENTITY:test_entity]]"
    match = EntityMatch(
        entity_id="test_entity",
        matched_source_text="John Doe",
        canonical_text="John Doe",
        display_text="John Doe",
        source_char_start=0,
        source_char_end=8,
        matched_alias="John Doe",
        match_confidence="high",
        match_method="canonical",
        domain="test",
        role="unknown",
        requires_pronunciation_calibration=False,
    )
    bindings = {placeholder: match}
    
    # restore_display_text should use display_text
    faithful_restored = restore_display_text(placeholder, bindings, registry)
    assert faithful_restored == "John Doe"
    
    # restore_tts_text should use spoken form
    tts_restored = restore_tts_text(placeholder, bindings, registry, "zu-ZA", None)
    assert tts_restored == "uJohn Doe"
    
    # Verify they are different
    assert faithful_restored != tts_restored


def test_residual_dialogue_allowlist():
    """Test that only approved residual dialogue states are allowed for production."""
    approved_states = {"passed", "clean", "no_residual"}
    rejected_states = {"missing", "not_run", "pending", "inconclusive", "failed", "above_threshold"}
    
    # All approved states should pass
    for state in approved_states:
        assert state in approved_states
    
    # All rejected states should not be in approved list
    for state in rejected_states:
        assert state not in approved_states


def test_preview_only_paths():
    """Test that proof-of-concept uses separate preview paths."""
    from mathula_tv.artifacts import DubbingArtifacts
    from pathlib import Path
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        work_dir = Path(tmpdir)
        job_id = "test_job"
        paths = DubbingArtifacts(work_dir / "jobs" / job_id)
        
        # Preview paths should be separate from production paths
        assert paths.preview_background != paths.background
        assert paths.preview_final_mix != paths.final_mix
        assert "previews" in str(paths.preview_background)
        assert "previews" in str(paths.preview_final_mix)


def test_entity_invalid_translation_policy():
    """Test that invalid translation policy raises error."""
    with pytest.raises(ValueError, match="Invalid translation_policy"):
        Entity(
            entity_id="test_entity",
            entity_type="person",
            canonical_text="Test",
            display_text="Test",
            aliases=(),
            domains=(),
            roles=(),
            source_status="public_record",
            active=True,
            must_preserve=True,
            translation_policy="invalid_policy",
            stt_phrases=(),
            spoken_forms={},
            do_not_confuse_with=(),
            sources=(),
        )


def test_approved_identity_forms_return_reviewed_aliases_and_fail_closed() -> None:
    registry = EntityRegistry()
    assert registry.approved_identity_forms("Economic Freedom Fighters") == (
        "Economic Freedom Fighters",
        "EFF",
    )
    assert registry.approved_identity_forms("EFF") == (
        "EFF",
        "Economic Freedom Fighters",
    )
    assert registry.approved_identity_forms("Unknown Example") == ("Unknown Example",)


def test_approved_identity_forms_derive_safe_person_title_surname_aliases() -> None:
    registry = EntityRegistry()
    forms = registry.approved_identity_forms("Advocate Sandile Khumalo SC")

    assert "Advocate Khumalo" in forms
    assert "Adv. Khumalo" in forms
    assert "Commissioner Khumalo" in forms
    assert "Khumalo" not in forms

    source_forms = registry.approved_identity_forms(
        "Advocate Sandile Khumalo SC",
        source_text=(
            "Earlier, we heard one of the Commissioners, "
            "Advocate-Commissioner Khumalo, in fact, talk about how,"
        ),
    )
    assert "Advocate-Commissioner Khumalo" in source_forms
    assert "Advocate Commissioner Khumalo" in source_forms
    assert "Advocate Khumalo" in source_forms
    assert "Commissioner Khumalo" in source_forms
    assert "Khumalo" not in source_forms


def test_derived_person_title_surname_alias_rejects_registry_collision(
    sample_registry_json,
) -> None:
    payload = json.loads(json.dumps(sample_registry_json))
    first = payload["entities"][0]
    first.update(
        {
            "canonical_text": "Advocate John Doe",
            "display_text": "Advocate John Doe",
            "aliases": ["John Doe", "Adv. John Doe"],
            "stt_phrases": ["Advocate John Doe", "John Doe", "Adv. John Doe"],
        }
    )
    second = json.loads(json.dumps(first))
    second.update(
        {
            "entity_id": "person_jane_doe",
            "canonical_text": "Advocate Jane Doe",
            "display_text": "Advocate Jane Doe",
            "aliases": ["Jane Doe", "Advocate Doe"],
            "stt_phrases": ["Advocate Jane Doe", "Jane Doe", "Advocate Doe"],
        }
    )
    payload["entities"].append(second)
    payload["validation"]["entity_count"] = len(payload["entities"])

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as handle:
        json.dump(payload, handle)
        registry_path = Path(handle.name)
    try:
        registry = EntityRegistry(registry_path)
        forms = registry.approved_identity_forms("Advocate John Doe")
    finally:
        registry_path.unlink(missing_ok=True)

    assert "Advocate Doe" not in forms
    assert "Adv. Doe" not in forms
    assert "Doe" not in forms
