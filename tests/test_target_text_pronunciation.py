import pytest

from mathula_tv.pronunciation import (
    PronunciationDictionary,
    PronunciationEntry,
    expand_initials_for_tts,
    normalise_dates_for_tts,
    normalise_numbers_for_tts,
)
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
    assert build_target_text(
        faithful,
        spoken_text=spoken,
        changes=[reviewed],
        protected_terms=["Feroz Khan"],
    ).spoken_text == spoken


def test_legacy_upgrade_preserves_approved_wording_in_all_three_forms():
    value = upgrade_legacy_translation("Umbhalo ogunyaziwe")
    assert value["faithful_translation"] == value["spoken_text"] == value["tts_text"]
    assert value["text_changes"] == []
