from mathula_tv.pronunciation import (
    PronunciationDictionary,
    get_zulu_public_affairs_acronym_registry,
    with_default_organisation_initialisms,
)
from mathula_tv.target_text import build_target_text


def _dictionary() -> PronunciationDictionary:
    return with_default_organisation_initialisms(
        PronunciationDictionary("v1", language="zu-ZA", job_id="job-123")
    )


def test_registry_contains_exactly_100_unique_curated_terms():
    registry = get_zulu_public_affairs_acronym_registry()
    assert len(registry) == 100
    assert len({item["token"].casefold() for item in registry}) == 100
    assert {item["kind"] for item in registry} == {"initials", "acronym"}


def test_pktt_is_spelled_with_letter_names_and_supports_zulu_prefixes():
    result = _dictionary().apply("I-PKTT ixhumana ne-PKTT kanye ne-CRTT.")
    assert result.tts_text == (
        "I-Pee Kay Tee Tee ixhumana ne-Pee Kay Tee Tee kanye ne-See Ar Tee Tee."
    )
    assert all(item.kind == "initials" for item in result.substitutions)


def test_major_word_acronyms_are_not_spelled_letter_by_letter():
    result = _dictionary().apply(
        "I-IDAC ne-IPID basebenza ne-SAPS, SARS, SASSA, SANRAL ne-NATJOINTS."
    )
    assert result.tts_text == (
        "I-Aydak ne-Ay-pid basebenza ne-Saps, Sars, Sassa, San-ral ne-Nat-joints."
    )
    assert all(item.kind == "acronym" for item in result.substitutions)


def test_public_service_and_economic_terms_use_reviewed_readings():
    result = _dictionary().apply(
        "I-NSFAS, TVET, SAQA, VAT, PAYE, BEE, B-BBEE, POPIA ne-PAIA."
    )
    assert result.tts_text == (
        "I-En-sfas, Tee-vet, Sah-kwa, Vat, Pay-ee, Bee, "
        "Triple Bee Ee Ee, Pop-ee-ah ne-Pie-ah."
    )


def test_bbbbee_alias_without_hyphen_uses_same_pronunciation():
    result = _dictionary().apply("B-BBEE kanye ne-BBBEE")
    assert result.tts_text == "Triple Bee Ee Ee kanye ne-Triple Bee Ee Ee"
    assert [item.before for item in result.substitutions] == ["B-BBEE", "BBBEE"]


def test_unknown_uppercase_term_is_preserved_for_future_review():
    result = _dictionary().apply("I-XYZQ ayikabuyekezwa.")
    assert result.tts_text == "I-XYZQ ayikabuyekezwa."
    assert result.substitutions == ()


def test_visible_translation_remains_immutable_while_tts_text_changes():
    target = build_target_text(
        "I-PKTT isebenza ne-SAPS.",
        dictionary=_dictionary(),
        protected_terms=("PKTT", "SAPS"),
    )
    assert target.faithful_translation == target.spoken_text == "I-PKTT isebenza ne-SAPS."
    assert target.tts_text == "I-Pee Kay Tee Tee isebenza ne-Saps."
