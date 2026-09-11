import pytest

from mathula_tv.pronunciation import (
    PronunciationDictionary,
    PronunciationEntry,
    build_initialism_ssml_parts,
    with_web_researched_organisation_pronunciations,
    expand_initials_for_tts,
    normalise_dates_for_tts,
    normalise_numbers_for_tts,
    with_default_organisation_initialisms,
)
from mathula_tv.tts_ssml import CharacterPart, TextPart
from mathula_tv.target_text import (
    TargetText,
    build_target_text,
    qc_reference_text,
    record_full_text_change,
    subtitle_text,
    upgrade_legacy_translation,
)


def entry(display, tts, kind, **updates):
    return PronunciationEntry(
        display_text=display,
        spoken_text=updates.pop("spoken_text", display),
        tts_text=tts,
        kind=kind,
        **updates,
    )


def reviewed_dictionary():
    return PronunciationDictionary(
        "1.2.0",
        entries=(
            entry("Feroz Khan", "Feh-rohz Kaan", "personal_name", confidence=0.9),
            entry("20%", "amaphesenti angamashumi amabili", "percentage"),
            entry("21 July 2026", "umhla zingamashumi amabili nanye kuNtulikazi wonyaka ka-2026", "date"),
            entry("S.A.", "Ess Ay", "initials"),
        ),
    )


def test_versioned_dictionary_round_trip_and_hash_verification():
    dictionary = reviewed_dictionary()
    value = dictionary.to_dict()
    assert value["schema_version"] == "pronunciation-dictionary-v1"
    assert len(value["dictionary_sha256"]) == 64
    assert PronunciationDictionary.from_dict(value).to_dict() == value
    value["dictionary_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        PronunciationDictionary.from_dict(value)


def test_global_entries_cover_names_numbers_dates_and_initials():
    dictionary = reviewed_dictionary()
    text = "Feroz Khan uthole 20% ngo-21 July 2026 e-S.A."
    result = dictionary.apply(text)
    assert "Feh-rohz Kaan" in result.tts_text
    assert "amaphesenti angamashumi amabili" in result.tts_text
    assert "umhla zingamashumi amabili nanye" in result.tts_text
    assert "Ess Ay" in result.tts_text
    assert {item.kind for item in result.substitutions} == {
        "personal_name",
        "percentage",
        "date",
        "initials",
    }
    assert normalise_numbers_for_tts(text, dictionary).tts_text.count("amaphesenti") == 1
    assert normalise_dates_for_tts(text, dictionary).tts_text.count("Ntulikazi") == 1
    assert expand_initials_for_tts(text, dictionary).tts_text.count("Ess Ay") == 1


def test_reviewed_organisation_initialisms_are_spelled_for_zulu_tts_only():
    dictionary = with_default_organisation_initialisms(
        PronunciationDictionary("v1", language="zu-ZA", job_id="job-123")
    )
    target = build_target_text(
        "I-ANC iphikisana ne-EFF.",
        dictionary=dictionary,
        protected_terms=("ANC", "EFF"),
    )
    assert target.faithful_translation == target.spoken_text == "I-ANC iphikisana ne-EFF."
    assert target.tts_text == "I-ANC iphikisana ne-Ee Eff Eff."
    assert [item.before for item in target.pronunciation_substitutions] == ["ANC", "EFF"]
    assert all(item.kind == "initials" for item in target.pronunciation_substitutions)


def test_npa_uses_reviewed_zulu_letter_names_with_attached_prefixes():
    dictionary = with_default_organisation_initialisms(
        PronunciationDictionary("v1", language="zu-ZA", job_id="job-123")
    )
    target = build_target_text(
        "I-NPA ikhuluma ngecala le-NPA ne-NPA.",
        dictionary=dictionary,
        protected_terms=("NPA",),
    )

    assert target.spoken_text == "I-NPA ikhuluma ngecala le-NPA ne-NPA."
    assert target.tts_text == ("I-En Pee Ey ikhuluma ngecala le-En Pee Ey ne-En Pee Ey.")
    assert [item.before for item in target.pronunciation_substitutions] == [
        "NPA",
        "NPA",
        "NPA",
    ]
    assert all(item.tts_text == "En Pee Ey" for item in target.pronunciation_substitutions)


def test_zulu_word_acronyms_are_pronounced_as_words_not_letter_names():
    dictionary = with_default_organisation_initialisms(
        PronunciationDictionary("v1", language="zu-ZA", job_id="job-123")
    )
    target = build_target_text(
        "I-IDAC ibambisene ne-IPID kanye ne-SAPS.",
        dictionary=dictionary,
        protected_terms=("IDAC", "IPID", "SAPS"),
    )

    assert target.spoken_text == "I-IDAC ibambisene ne-IPID kanye ne-SAPS."
    assert target.tts_text == "I-Aydak ibambisene ne-Ay-pid kanye ne-Saps."
    assert [item.before for item in target.pronunciation_substitutions] == [
        "IDAC",
        "IPID",
        "SAPS",
    ]
    assert all(item.kind == "acronym" for item in target.pronunciation_substitutions)


def test_other_reviewed_sa_party_initialisms_are_spelled_letter_by_letter():
    dictionary = with_default_organisation_initialisms(
        PronunciationDictionary("v1", language="zu-ZA", job_id="job-123")
    )
    result = dictionary.apply("MK, MKP, DA, IFP ne-UDM")
    assert result.tts_text == ("Em Kay, Em Kay Pee, Dee Ey, Eye Eff Pee ne-You Dee Em")


@pytest.mark.parametrize(
    "language",
    ("nso-ZA", "st-ZA", "tn-ZA", "ve-ZA", "xh-ZA", "ts-ZA", "ss-ZA", "nr-ZA"),
)
def test_organisation_initialisms_support_all_sa_language_accounts(language):
    dictionary = with_default_organisation_initialisms(
        PronunciationDictionary("v1", language=language, job_id="job-123")
    )
    assert dictionary.apply("ANC EFF MKP").tts_text == ("Ay En See EFF Em Kay Pee")


def test_zulu_initialism_profile_uses_human_selected_aliases_and_character_mode():
    dictionary = with_default_organisation_initialisms(
        PronunciationDictionary("v1", language="zu-ZA", job_id="job-123")
    )
    text = dictionary.apply("I-ANC, EFF, MK ne-DA").tts_text
    assert text == "I-ANC, Ee Eff Eff, Em Kay ne-Dee Ey"
    assert build_initialism_ssml_parts(text, language="zu-ZA") == (
        TextPart("I-"),
        CharacterPart("ANC"),
        TextPart(", Ee Eff Eff, Em Kay ne-Dee Ey"),
    )
    assert build_initialism_ssml_parts(text, language="xh-ZA") == ()


def test_kzn_character_mode_preserves_fused_zulu_prefixes():
    dictionary = with_default_organisation_initialisms(
        PronunciationDictionary("v1", language="zu-ZA", job_id="job-123")
    )
    text = "eKZN, aseKZN, baseKZN nase-KZN; KZN."
    result = dictionary.apply(text)

    assert result.spoken_text == result.tts_text == text
    assert [item.before for item in result.substitutions] == [
        "KZN",
        "KZN",
        "KZN",
        "KZN",
        "KZN",
    ]
    assert build_initialism_ssml_parts(result.tts_text, language="zu-ZA") == (
        TextPart("e"),
        CharacterPart("KZN"),
        TextPart(", "),
        TextPart("ase"),
        CharacterPart("KZN"),
        TextPart(", "),
        TextPart("base"),
        CharacterPart("KZN"),
        TextPart(" nase-"),
        CharacterPart("KZN"),
        TextPart("; "),
        CharacterPart("KZN"),
        TextPart("."),
    )


def test_kzn_character_mode_does_not_touch_words_containing_same_letters():
    text = "eKZNology AKZN KZName inkzn"
    assert build_initialism_ssml_parts(text, language="zu-ZA") == ()


@pytest.mark.parametrize(
    ("text", "prefix"),
    (
        ("SA First Forum", ""),
        ("i-SA First Forum", "i-"),
        ("ye-SA First Forum", "ye-"),
    ),
)
def test_sa_first_forum_spells_only_reviewed_initials(text, prefix):
    parts = build_initialism_ssml_parts(text, language="zu-ZA")

    expected_prefix = (TextPart(prefix),) if prefix else ()
    assert parts == (
        *expected_prefix,
        CharacterPart("SA"),
        TextPart(" First Forum"),
    )


def test_sa_character_mode_is_scoped_to_first_forum_name():
    text = "SA iseningizimu, SASA, nese-SA Reserve"
    assert build_initialism_ssml_parts(text, language="zu-ZA") == ()


def test_hawks_uses_verified_ama_plural_code_switch():
    dictionary = with_default_organisation_initialisms(
        PronunciationDictionary("v1", language="zu-ZA", job_id="job-123")
    )
    target = build_target_text(
        "Ama-Hawks athi azophenya.",
        dictionary=dictionary,
        protected_terms=("Hawks",),
    )

    assert target.spoken_text == "Ama-Hawks athi azophenya."
    assert target.tts_text == "Ama-Horks athi azophenya."
    assert target.pronunciation_substitutions[0].before == "Hawks"
    assert target.pronunciation_substitutions[0].after == "Horks"


def test_hawks_calibration_is_not_masked_by_job_research():
    stale_web_override = PronunciationEntry(
        display_text="The Hawks",
        spoken_text="The Hawks",
        tts_text="The Hawks",
        language="zu-ZA",
        source="web_research",
        confidence=0.94,
        kind="organisation_name",
    )
    dictionary = with_web_researched_organisation_pronunciations(
        with_default_organisation_initialisms(
            PronunciationDictionary(
                "v1",
                language="zu-ZA",
                job_id="job-123",
                job_overrides=(stale_web_override,),
            )
        ),
        {
            "accepted_corrections": [
                {
                    "entity_type": "organisation",
                    "canonical_text": "The Hawks",
                    "confidence": 0.94,
                    "pronunciation_mode": "word_name",
                    "pronunciation_tts_text": "The Hawks",
                    "pronunciation_confidence": 0.94,
                    "grounded_source_urls": ["https://example.test/hawks"],
                    "grounded_pronunciation_evidence_urls": ["https://example.test/hawks"],
                }
            ]
        },
    )

    assert dictionary.apply("Ama-Hawks athi").tts_text == "Ama-Horks athi"
    assert not any(
        entry.display_text == "The Hawks" and entry.source == "web_research" for entry in dictionary.job_overrides
    )


def test_justice_college_uses_verified_code_switch():
    """Live TTS+STT calibration (job 33cd7b46...): unmodified "Justice College"
    embedded in isiZulu was recovered by en-ZA Azure STT as a garbled
    "Chastique Koleke"; the hidden alias "Jastis Koleji" recovered a clean,
    literal "Justice College". Captions must stay unaffected -- this is a
    TTS-only alias.
    """
    dictionary = with_default_organisation_initialisms(
        PronunciationDictionary("v1", language="zu-ZA", job_id="job-123")
    )
    target = build_target_text(
        "waseBritish Rabanda Justice College usejoyina manje.",
        dictionary=dictionary,
        protected_terms=("Justice College",),
    )

    assert target.spoken_text == "waseBritish Rabanda Justice College usejoyina manje."
    assert target.tts_text == "waseBritish Rabanda Jastis Koleji usejoyina manje."
    assert target.pronunciation_substitutions[0].before == "Justice College"
    assert target.pronunciation_substitutions[0].after == "Jastis Koleji"


@pytest.mark.parametrize("letter", tuple("ABCDEFGHIJK"))
def test_anonymous_witness_designation_uses_character_mode(letter):
    assert build_initialism_ssml_parts(f"u-Witness {letter} uyafakaza", language="zu-ZA") == (
        TextPart("u-Witness "),
        CharacterPart(letter),
        TextPart(" uyafakaza"),
    )


def test_web_researched_organisation_pronunciation_is_phrase_scoped():
    dictionary = with_web_researched_organisation_pronunciations(
        with_default_organisation_initialisms(PronunciationDictionary("v1", language="zu-ZA", job_id="job-123")),
        {
            "accepted_corrections": [
                {
                    "entity_type": "organisation",
                    "canonical_text": "SA First Forum",
                    "representative_text": "SA press forum",
                    "aliases": ["SA press forum"],
                    "confidence": 0.99,
                    "grounded_source_urls": ["https://example.test/report"],
                }
            ]
        },
    )

    result = dictionary.apply("ye-SA First Forum ne-SA Reserve")
    assert result.spoken_text == "ye-SA First Forum ne-SA Reserve"
    assert result.tts_text == "ye-Ess Ey Ferst Forum ne-SA Reserve"
    assert not dictionary.job_overrides
    assert any(
        entry.display_text == "SA First Forum" and entry.source == "application_default" for entry in dictionary.entries
    )


def test_grounded_big_five_rendering_changes_only_hidden_tts_text():
    pronunciation_url = "https://broadcaster.example/video/big-five"
    dictionary = with_web_researched_organisation_pronunciations(
        PronunciationDictionary("v1", language="zu-ZA", job_id="job-123"),
        {
            "accepted_corrections": [
                {
                    "entity_type": "organisation",
                    "canonical_text": "The Big Five cartel",
                    "confidence": 1.0,
                    "grounded_source_urls": [pronunciation_url],
                    "pronunciation_mode": "word_name",
                    "pronunciation_tts_text": "The Big Faiv cartel",
                    "pronunciation_confidence": 0.97,
                    "pronunciation_language": "en-ZA",
                    "pronunciation_ipa": "/ðə bɪɡ faɪv kɑːtəl/",
                    "grounded_pronunciation_evidence_urls": [pronunciation_url],
                }
            ]
        },
    )

    result = dictionary.apply("Ama-WhatsApp omholi we-The Big Five cartel, uJothan Msibi.")
    assert result.spoken_text == ("Ama-WhatsApp omholi we-The Big Five cartel, uJothan Msibi.")
    assert result.tts_text == ("Ama-WhatsApp omholi we-The Big Faiv cartel, uJothan Msibi.")
    assert dictionary.job_overrides[-1].display_text == "The Big Five cartel"
    assert "pronunciation_language=en-ZA" in dictionary.job_overrides[-1].notes
    assert "pronunciation_ipa=/ðə bɪɡ faɪv kɑːtəl/" in dictionary.job_overrides[-1].notes


def test_web_researched_pronunciation_rejects_ungrounded_or_uncorroborated_names():
    base = PronunciationDictionary("v1", language="zu-ZA", job_id="job-123")
    research = {
        "accepted_corrections": [
            {
                "entity_type": "organisation",
                "canonical_text": "SA First Forum",
                "representative_text": "First Forum",
                "confidence": 0.99,
                "grounded_source_urls": ["https://example.test/report"],
            },
            {
                "entity_type": "organisation",
                "canonical_text": "AB News",
                "representative_text": "AB News",
                "confidence": 0.99,
                "grounded_source_urls": [],
            },
        ]
    }

    assert with_web_researched_organisation_pronunciations(base, research) == base


def test_web_researched_pronunciation_is_idempotent_and_retires_stale_override():
    base = PronunciationDictionary("v1", language="zu-ZA", job_id="job-123")
    research = {
        "accepted_corrections": [
            {
                "entity_type": "organisation",
                "canonical_text": "SA First Forum",
                "representative_text": "SA press forum",
                "confidence": 0.99,
                "grounded_source_urls": ["https://example.test/report"],
            }
        ]
    }
    researched = with_web_researched_organisation_pronunciations(base, research)

    assert with_web_researched_organisation_pronunciations(researched, research) == researched
    assert with_web_researched_organisation_pronunciations(researched, None) == base


def test_long_researched_initialism_requires_direct_pronunciation_evidence():
    base = PronunciationDictionary("v1", language="zu-ZA", job_id="job-123")
    item = {
        "entity_type": "organisation",
        "canonical_text": "ABC Forum",
        "representative_text": "ABC Forum",
        "confidence": 0.99,
        "grounded_source_urls": ["https://example.test/identity"],
    }

    assert with_web_researched_organisation_pronunciations(base, {"accepted_corrections": [item]}) == base
    researched = with_web_researched_organisation_pronunciations(
        base,
        {
            "accepted_corrections": [
                {
                    **item,
                    "pronunciation_mode": "initialism",
                    "pronunciation_evidence_urls": ["https://example.test/pronunciation"],
                }
            ]
        },
    )
    assert researched.apply("ABC Forum").tts_text == "Ey Bee See Forum"


def test_job_override_remains_authoritative_for_fused_kzn():
    dictionary = with_default_organisation_initialisms(
        PronunciationDictionary(
            "v1",
            language="zu-ZA",
            job_id="job-123",
            job_overrides=(
                PronunciationEntry(
                    display_text="KZN",
                    spoken_text="KZN",
                    tts_text="reviewed regional reading",
                    language="zu-ZA",
                    kind="initials",
                    source="human_review",
                ),
            ),
        )
    )

    assert dictionary.apply("eKZN aseKZN").tts_text == ("ereviewed regional reading asereviewed regional reading")


def test_south_african_police_service_uses_reviewed_hidden_code_switch():
    dictionary = with_default_organisation_initialisms(
        PronunciationDictionary("v1", language="zu-ZA", job_id="job-123")
    )
    target = build_target_text(
        "I-South African Police Service iphendulile.",
        dictionary=dictionary,
        protected_terms=("South African Police Service",),
    )

    assert target.spoken_text == "I-South African Police Service iphendulile."
    assert target.tts_text == "I-South African Police Sir-vis iphendulile."
    assert target.pronunciation_substitutions[0].kind == "organisation_name"


def test_zulu_code_switch_place_names_keep_display_text_and_use_reviewed_tts_aliases():
    dictionary = with_default_organisation_initialisms(
        PronunciationDictionary("v1", language="zu-ZA", job_id="job-123")
    )
    result = dictionary.apply("Namhlanje iTshwane ifana neLondon; bantu baseBuffalo City.")

    assert result.spoken_text == ("Namhlanje iTshwane ifana neLondon; bantu baseBuffalo City.")
    assert result.tts_text == ("Namhlanje i Tšhwane ifana ne Landen; bantu base Baffalo Siti.")
    assert {item.kind for item in result.substitutions} == {"place_name"}


def test_talisha_naidoo_uses_reviewed_hidden_code_switch():
    # Real user-reported defect, job b15075e7268049b491ee9e2222e5811f, four
    # rounds. Round 4's real, confirmed lesson: raw measured phone duration
    # (phoneme_recognizer.py's recognize_with_timing) is NOT a reliable proxy
    # for perceived naturalness -- picking "Taleesha Nai-dooooo" for
    # measuring a longer final vowel than "Taleesha Nai-dooo" got direct,
    # blunt feedback: "the drag is worse than no drag". Reverted to round 3's
    # "Taleesha Nai-dooo" -- the last version never reported as sounding
    # wrong -- rather than chase more measured duration further.
    dictionary = with_default_organisation_initialisms(
        PronunciationDictionary("v1", language="zu-ZA", job_id="job-123")
    )
    result = dictionary.apply("Manje sidlulisela kuTalisha Naidoo we-SABC News.")

    assert result.spoken_text == "Manje sidlulisela kuTalisha Naidoo we-SABC News."
    assert "Taleesha Nai-dooo" in result.tts_text
    assert "Talisha Naidoo" not in result.tts_text
    assert {item.kind for item in result.substitutions} & {"personal_name"}


def test_organisation_defaults_do_not_leak_into_unsupported_locales():
    dictionary = PronunciationDictionary("v1", language="en-US", job_id="job-123")
    assert with_default_organisation_initialisms(dictionary) is dictionary


def test_job_pronunciation_override_wins_over_default_initialism():
    dictionary = PronunciationDictionary(
        "v1",
        language="zu-ZA",
        job_id="job-123",
        job_overrides=(entry("ANC", "custom ANC", "initials", source="job_override"),),
    )
    augmented = with_default_organisation_initialisms(dictionary)
    assert augmented.apply("ANC ne-EFF").tts_text == "custom ANC ne-Ee Eff Eff"


def test_updated_application_default_replaces_stale_saved_default():
    dictionary = PronunciationDictionary(
        "v1",
        language="zu-ZA",
        entries=(entry("EFF", "EFF", "initials", source="application_default"),),
    )
    augmented = with_default_organisation_initialisms(dictionary)
    assert augmented.apply("EFF").tts_text == "Ee Eff Eff"


def test_job_override_wins_without_mutating_global_dictionary():
    global_dictionary = reviewed_dictionary()
    job_dictionary = global_dictionary.with_job_overrides(
        "job-123",
        [entry("Feroz Khan", "Feh-rohz Khahn", "personal_name", source="job_override")],
    )
    assert global_dictionary.apply("Feroz Khan").tts_text == "Feh-rohz Kaan"
    result = job_dictionary.apply("Feroz Khan")
    assert result.tts_text == "Feh-rohz Khahn"
    assert result.substitutions[0].source == "job_override"
    assert job_dictionary.to_dict()["job_id"] == "job-123"


def test_dictionary_rejects_xml_injection_and_invalid_confidence():
    with pytest.raises(ValueError, match="not XML"):
        entry("Name", '<audio src="https://example.invalid">', "personal_name")
    with pytest.raises(ValueError, match="confidence"):
        entry("Name", "Naym", "personal_name", confidence=1.1)


def test_three_text_representation_records_every_tts_difference():
    target = build_target_text(
        "Feroz Khan uthole 20%.",
        dictionary=reviewed_dictionary(),
        protected_terms=("Feroz Khan", "20%"),
    )
    assert target.faithful_translation == target.spoken_text == "Feroz Khan uthole 20%."
    assert "Feh-rohz Kaan" in target.tts_text
    assert "amaphesenti" in target.tts_text
    assert len(target.changes) == len(target.pronunciation_substitutions) == 2
    assert subtitle_text(target) == "Feroz Khan uthole 20%."
    assert qc_reference_text(target) == "Feroz Khan uthole 20%."
    assert "Feh-rohz" not in subtitle_text(target)
    assert TargetText.from_dict(target.to_dict()) == target


def test_unrecorded_spoken_or_tts_rewrite_is_rejected():
    with pytest.raises(ValueError, match="omission, contraction"):
        TargetText("Incazelo ephelele", "Incazelo", "Incazelo")
    with pytest.raises(ValueError, match="Every difference"):
        TargetText("Igama", "Igama", "Ee-gah-mah")


def test_reviewed_semantic_change_can_record_timing_paraphrase():
    faithful = "Incazelo ende ephelele."
    spoken = "Incazelo ephelele."
    change = record_full_text_change(
        faithful,
        spoken,
        layer="faithful_to_spoken",
        change_type="contraction",
        reason="Reviewed timing contraction; no fact removed",
    )
    target = build_target_text(faithful, spoken_text=spoken, changes=[change])
    assert target.spoken_text == spoken and target.tts_text == spoken
    assert target.changes[0].change_type == "contraction"


def test_protected_term_cannot_disappear_silently():
    faithful = "UFeroz Khan ukhulume."
    spoken = "Ukhulume."
    unreviewed = record_full_text_change(
        faithful,
        spoken,
        layer="faithful_to_spoken",
        change_type="omission",
        reason="Timing",
    )
    with pytest.raises(ValueError, match="Protected term removed"):
        build_target_text(faithful, spoken_text=spoken, changes=[unreviewed], protected_terms=["Feroz Khan"])

    reviewed = record_full_text_change(
        faithful,
        spoken,
        layer="faithful_to_spoken",
        change_type="omission",
        reason="Editor explicitly requires human review",
        protected_terms=["Feroz Khan"],
        human_review_required=True,
    )
    assert (
        build_target_text(
            faithful,
            spoken_text=spoken,
            changes=[reviewed],
            protected_terms=["Feroz Khan"],
        ).spoken_text
        == spoken
    )


def test_legacy_upgrade_preserves_approved_wording_in_all_three_forms():
    value = upgrade_legacy_translation("Umbhalo ogunyaziwe")
    assert value["faithful_translation"] == value["spoken_text"] == value["tts_text"]
    assert value["text_changes"] == []
