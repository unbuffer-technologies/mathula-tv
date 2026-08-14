from mathula_tv.entity_registry import (
    Entity,
    EntityMatch,
    SpokenForm,
    protect_text_with_placeholders,
    restore_display_text,
    restore_tts_text,
)


class FakeRegistry:
    def __init__(self, entity: Entity):
        self.entity = entity
        self.requested_ids: list[str] = []

    def get_entity(self, entity_id: str) -> Entity:
        self.requested_ids.append(entity_id)
        assert entity_id == self.entity.entity_id
        return self.entity


def _entity() -> Entity:
    return Entity(
        entity_id="organisation_big_w",
        entity_type="organisation",
        canonical_text="Big W",
        display_text="Big W",
        aliases=("Big W",),
        domains=("general",),
        roles=(),
        source_status="reviewed",
        active=True,
        must_preserve=True,
        translation_policy="protected_token",
        stt_phrases=("Big W",),
        spoken_forms={
            "zu-ZA": SpokenForm(
                default="Big double-you",
                voices={},
                status="approved",
                last_tested_at=None,
                approved_by=None,
            )
        },
        do_not_confuse_with=(),
        sources=(),
    )


def _match() -> EntityMatch:
    return EntityMatch(
        entity_id="organisation_big_w",
        matched_source_text="Big W",
        canonical_text="Big W",
        display_text="Big W",
        source_char_start=29,
        source_char_end=34,
        matched_alias="big w",
        match_confidence="high",
        match_method="canonical",
        domain="general",
        role="unknown",
        requires_pronunciation_calibration=False,
    )


def test_persisted_string_binding_restores_display_and_tts_text():
    entity = _entity()
    registry = FakeRegistry(entity)
    placeholder = "[[MATHULA_ENTITY:organisation_big_w]]"
    bindings = {placeholder: "organisation_big_w"}

    assert restore_display_text(f"echibini le-{placeholder}", bindings, registry) == "echibini le-Big W"
    assert restore_tts_text(f"echibini le-{placeholder}", bindings, registry) == "echibini le-Big double-you"
    assert registry.requested_ids == ["organisation_big_w", "organisation_big_w"]


def test_placeholder_protection_persists_entity_id_strings():
    protected, bindings = protect_text_with_placeholders(
        "He came from the pool of the Big W.",
        [_match()],
    )

    placeholder = "[[MATHULA_ENTITY:organisation_big_w]]"
    assert protected == f"He came from the pool of the {placeholder}."
    assert bindings == {placeholder: "organisation_big_w"}


def test_legacy_object_and_mapping_bindings_remain_supported():
    entity = _entity()
    placeholder = "[[MATHULA_ENTITY:organisation_big_w]]"

    assert restore_display_text(placeholder, {placeholder: _match()}, FakeRegistry(entity)) == "Big W"
    assert restore_display_text(
        placeholder,
        {placeholder: {"entity_id": "organisation_big_w"}},
        FakeRegistry(entity),
    ) == "Big W"
