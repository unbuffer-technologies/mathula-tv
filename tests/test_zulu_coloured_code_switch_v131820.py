from __future__ import annotations

from mathula_tv.direct_azure_dub import load_approved_segments
from mathula_tv.pronunciation import (
    PronunciationDictionary,
    ZU_CODE_SWITCH_PRONUNCIATION_VERSION,
    with_default_organisation_initialisms,
)


def _zulu_dictionary() -> PronunciationDictionary:
    return with_default_organisation_initialisms(
        PronunciationDictionary("v1", language="zu-ZA", job_id="job-123")
    )


def test_coloured_keeps_protected_spelling_but_uses_reviewed_tts_reading() -> None:
    result = _zulu_dictionary().apply(
        "Umholi we-National Coloured Congress uFadiel Adams ukhulume."
    )

    assert result.spoken_text == (
        "Umholi we-National Coloured Congress uFadiel Adams ukhulume."
    )
    assert result.tts_text == (
        "Umholi we-National Khalad Congress uFadiel Adams ukhulume."
    )
    assert len(result.substitutions) == 1
    substitution = result.substitutions[0]
    assert substitution.before == "Coloured"
    assert substitution.after == "Khalad"
    assert substitution.kind == "english_code_switch"


def test_coloured_matching_is_case_insensitive_and_accepts_us_alias() -> None:
    result = _zulu_dictionary().apply("coloured, COLOURED, colored")

    assert result.spoken_text == "coloured, COLOURED, colored"
    assert result.tts_text == "Khalad, Khalad, Khalad"
    assert all(
        item.kind == "english_code_switch" for item in result.substitutions
    )


def test_saved_job_dictionary_is_augmented_without_manual_edit() -> None:
    saved = PronunciationDictionary(
        "v1",
        language="zu-ZA",
        job_id="existing-job",
    )
    upgraded = with_default_organisation_initialisms(saved)
    coloured = next(
        entry for entry in upgraded.entries if entry.display_text == "coloured"
    )

    assert coloured.tts_text == "Khalad"
    assert coloured.aliases == ("colored",)
    assert ZU_CODE_SWITCH_PRONUNCIATION_VERSION in coloured.notes


def test_direct_dub_regenerates_hidden_tts_without_changing_translation() -> None:
    segments = load_approved_segments(
        {
            "segments": [
                {
                    "segment_id": "seg-00001",
                    "speaker_id": "SPEAKER_00",
                    "start": 0.0,
                    "end": 4.0,
                    "source_text": "The National Coloured Congress responded.",
                    "translated_text": (
                        "I-National Coloured Congress iphendulile."
                    ),
                    "protected_spans": ["National Coloured Congress"],
                }
            ]
        },
        source_duration_ms=4_000,
        pronunciation_dictionary=_zulu_dictionary(),
    )

    assert segments[0].translated_text == (
        "I-National Coloured Congress iphendulile."
    )
    assert segments[0].tts_text == "I-National Khalad Congress iphendulile."
    assert segments[0].protected_entities == ("National Coloured Congress",)


def test_english_code_switch_default_does_not_leak_into_other_locales() -> None:
    xhosa = PronunciationDictionary("v1", language="xh-ZA", job_id="job-123")
    augmented = with_default_organisation_initialisms(xhosa)

    assert augmented.apply("National Coloured Congress").tts_text == (
        "National Coloured Congress"
    )

