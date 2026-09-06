from __future__ import annotations

from mathula_tv.direct_azure_dub import (
    DIRECT_DUB_SCHEMA_VERSION,
    coalesce_same_speaker_segments,
    load_approved_segments,
)
from mathula_tv.pronunciation import (
    PronunciationDictionary,
    PronunciationEntry,
    ZU_NATIVE_HONORIFIC_PRONUNCIATION_VERSION,
    with_default_organisation_initialisms,
)
from mathula_tv.target_text import build_target_text, subtitle_text


def _dictionary(language: str = "zu-ZA") -> PronunciationDictionary:
    return with_default_organisation_initialisms(
        PronunciationDictionary("v13.18.23", language=language, job_id="job")
    )


def test_exact_chaskalson_phrase_expands_fused_mister_without_visible_change() -> None:
    source = (
        "kuka-9. Ngakho sikhulumile noMnu. Adams neqembu lakhe lezomthetho. "
        "UMnu. Adams uyatholakala."
    )

    result = _dictionary().apply(source)

    assert result.spoken_text == source
    assert result.tts_text == (
        "kuka-9. Ngakho sikhulumile noMnumzane Adams neqembu lakhe lezomthetho. "
        "UMnumzane Adams uyatholakala."
    )
    assert [(item.before, item.after) for item in result.substitutions] == [
        ("noMnu.", "noMnumzane"),
        ("UMnu.", "UMnumzane"),
    ]


def test_common_fused_zulu_honorific_forms_are_expanded_for_tts() -> None:
    source = (
        "Mnu. Dlamini, uMnu. Adams, kuMnu. Cele, ngoMnu. Mkhize, "
        "uNkk. Molefe, noNksz. Khumalo, noDkt. Matebula."
    )

    assert _dictionary().apply(source).tts_text == (
        "Mnumzane Dlamini, uMnumzane Adams, kuMnumzane Cele, "
        "ngoMnumzane Mkhize, uNkosikazi Molefe, noNkosazana Khumalo, "
        "noDokotela Matebula."
    )


def test_associative_lika_prefixed_honorific_is_expanded_for_tts() -> None:
    # Confirmed real defect: "Ubusekhaya likaMnu. Lincoln." synthesized with an
    # ~880ms mid-utterance silent gap (Azure treating the abbreviation's period
    # as a real sentence boundary) because the "lika-" associative prefix
    # ("of Mr. ...") wasn't in the fused-prefix alternation -- only bare "ka-"
    # was. A sibling sentence using bare "kaMnu." already expanded correctly,
    # which is what exposed the gap.
    result = _dictionary().apply("Ubusekhaya likaMnu. Lincoln.")

    # "Lincoln" -> "Linken" is a separate, later-added pronunciation
    # correction (real en-ZA STT round-trip confirmed bare "Lincoln" was
    # misread as "Lingon" by this voice) that necessarily also fires on this
    # same input -- this test's own real subject is the "lika-" associative
    # prefix expansion, asserted via the first substitution below.
    assert result.tts_text == "Ubusekhaya likaMnumzane Linken."
    assert [(item.before, item.after) for item in result.substitutions] == [
        ("likaMnu.", "likaMnumzane"),
        ("Lincoln", "Linken"),
    ]


def test_honorific_expansion_is_tts_only_in_target_text() -> None:
    source = "Ngakho sikhulumile noMnu. Adams."
    target = build_target_text(
        faithful_translation=source,
        spoken_text=source,
        dictionary=_dictionary(),
    )

    assert subtitle_text(target) == source
    assert target.tts_text == "Ngakho sikhulumile noMnumzane Adams."
    assert target.changes[0].layer == "spoken_to_tts"


def test_honorific_expansion_reaches_coalesced_direct_dub_block() -> None:
    source = "Ngakho sikhulumile noMnu. Adams."
    segments = load_approved_segments(
        {
            "units": [
                {
                    "unit_id": "unit_0022",
                    "speaker_id": "SPEAKER_01",
                    "start_ms": 120_070,
                    "end_ms": 127_299,
                    "source_text": "So we've spoken with Mr. Adams.",
                    "variants": [
                        {
                            "variant_id": variant_id,
                            "spoken_text": source,
                            "meaning_preserved": True,
                        }
                        for variant_id in ("natural", "concise", "compact")
                    ],
                }
            ]
        },
        source_duration_ms=807_428,
        pronunciation_dictionary=_dictionary(),
    )

    block = coalesce_same_speaker_segments(segments)[0]
    assert block.translated_text == source
    assert block.tts_text == "Ngakho sikhulumile noMnumzane Adams."
    assert all("noMnumzane Adams" in item.tts_text for item in block.variants)


def test_explicit_job_override_remains_authoritative() -> None:
    dictionary = _dictionary().with_job_overrides(
        "job",
        [
            PronunciationEntry(
                display_text="noMnu.",
                spoken_text="noMnu.",
                tts_text="no Mnumzane",
                language="zu-ZA",
                source="human_review",
                kind="other",
            )
        ],
    )

    assert dictionary.apply("Sikhulume noMnu. Adams.").tts_text == (
        "Sikhulume no Mnumzane Adams."
    )


def test_honorific_fallback_does_not_change_other_locales() -> None:
    source = "We spoke to uMnu. Adams."
    assert _dictionary("en-ZA").apply(source).tts_text == source


def test_version_markers_force_new_tts_cache_identity() -> None:
    assert ZU_NATIVE_HONORIFIC_PRONUNCIATION_VERSION.endswith("v13.18.23")
    assert DIRECT_DUB_SCHEMA_VERSION.endswith("v13.18.56")
