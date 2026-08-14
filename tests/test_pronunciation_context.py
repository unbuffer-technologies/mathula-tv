from __future__ import annotations

from mathula_tv.direct_azure_dub import DirectAzureDubRenderer
from mathula_tv.pronunciation import (
    PronunciationDictionary,
    PronunciationEntry,
    with_default_organisation_initialisms,
)


def _dictionary() -> PronunciationDictionary:
    return PronunciationDictionary(
        dictionary_version="test-v1",
        language="zu-ZA",
        job_id="job-1",
        job_overrides=(
            PronunciationEntry(
                display_text="noC Mackenzie",
                spoken_text="noC Mackenzie",
                tts_text="no Makenzi",
                language="zu-ZA",
                source="job_override",
                kind="personal_name",
            ),
        ),
    )


def test_reviewed_noc_override_derives_other_zulu_name_contexts() -> None:
    dictionary = with_default_organisation_initialisms(_dictionary())

    assert dictionary.apply("uC Mackenzie").tts_text == "u Makenzi"
    assert dictionary.apply("uMackenzie").tts_text == "u Makenzi"
    assert dictionary.apply("noC Mackenzie").tts_text == "no Makenzi"


def test_semantic_rewrite_reapplies_application_pronunciation() -> None:
    dictionary = with_default_organisation_initialisms(_dictionary())

    tts_text, substitutions = DirectAzureDubRenderer._apply_candidate_pronunciation(
        "U-Adams noC Mackenzie sebeyingozi.",
        "U-Adams noC Mackenzie sebeyingozi.",
        dictionary,
    )

    assert tts_text == "U-Adams no Makenzi sebeyingozi."
    assert substitutions[0]["before"] == "noC Mackenzie"


def test_zulu_prefixed_cardinals_are_spelled_out_only_for_tts() -> None:
    dictionary = with_default_organisation_initialisms(_dictionary())
    spoken = (
        "Ngasebenzisa izigidi ezingu-31 zamaRandi, "
        "kwase kundiza abantu abangu-80."
    )

    result = dictionary.apply(spoken)

    assert result.spoken_text == spoken
    assert result.tts_text == (
        "Ngasebenzisa izigidi ezingamashumi amathathu nanye zamaRandi, "
        "kwase kundiza abantu abangamashumi ayisishiyagalombili."
    )
    assert [value.before for value in result.substitutions] == [
        "ezingu-31",
        "abangu-80",
    ]
    assert all(value.kind == "number" for value in result.substitutions)


def test_unreviewed_number_morphology_is_not_guessed() -> None:
    dictionary = with_default_organisation_initialisms(_dictionary())

    assert dictionary.apply("izigidi ezingu-19").tts_text == (
        "izigidi ezingu-19"
    )
