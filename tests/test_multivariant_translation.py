from __future__ import annotations

from pathlib import Path

import pytest

from mathula_tv.atomic_io import atomic_write_json, read_json
from mathula_tv.multivariant_translation import (
    INSTALLED_SCHEMA_VERSION,
    MultivariantTranslationError,
    SYSTEM_PROMPT,
    build_translation_request,
    export_translation_package,
    install_manual_translation,
    normalize_multivariant_response,
    validate_multivariant_response,
)


def test_prompt_forbids_doubled_determiners_for_code_switched_names() -> None:
    prompt = " ".join(SYSTEM_PROMPT.split())
    assert 'do not produce doubled determiners such as "I-The Hawks"' in prompt
    assert '"Ama-Hawks athi"' in prompt


def _source() -> dict:
    return {
        "schema_version": "transcript-v1",
        "segments": [
            {
                "segment_id": "seg-00001",
                "speaker": "SPEAKER_00",
                "start": 0.32,
                "end": 16.48,
                "source_text": "The EFF event in Thohoyandou features Dr Levy Ndou.",
                "protected_spans": ["EFF", "Thohoyandou", "Dr Levy Ndou"],
            }
        ],
    }


def _response(request: dict) -> dict:
    source = request["units"][0]
    targets = source["duration_targets"]
    return {
        "schema_version": "mathula.translation.multivariant.v1",
        "source_language": "en-ZA",
        "target_language": "isiZulu",
        "target_locale": "zu-ZA",
        "translation_strategy": "contextual_code_switching",
        "units": [
            {
                "unit_id": source["unit_id"],
                "speaker_id": source["speaker_id"],
                "start_ms": source["start_ms"],
                "end_ms": source["end_ms"],
                "available_duration_ms": source["available_duration_ms"],
                "overlap_group": None,
                "source_text": source["source_text"],
                "protected_spans": source["protected_spans"],
                "required_facts": [
                    "The event is associated with the EFF.",
                    "It is in Thohoyandou.",
                    "Dr Levy Ndou is involved.",
                ],
                "optional_details": ["Extended introduction"],
                "variants": [
                    {
                        "variant_id": "natural",
                        "compression_level": "natural",
                        "target_duration_ms": targets["natural_target_duration_ms"],
                        "spoken_text": "Umcimbi we-EFF eThohoyandou uhlanganisa uDr Levy Ndou namhlanje.",
                        "estimated_duration_ms": 14000,
                        "preserved_english_spans": ["EFF", "Dr Levy Ndou"],
                        "omitted_optional_details": [],
                        "meaning_preserved": True,
                    },
                    {
                        "variant_id": "concise",
                        "compression_level": "concise",
                        "target_duration_ms": targets["concise_target_duration_ms"],
                        "spoken_text": "EThohoyandou, umcimbi we-EFF unoDr Levy Ndou.",
                        "estimated_duration_ms": 11000,
                        "preserved_english_spans": ["EFF", "Dr Levy Ndou"],
                        "omitted_optional_details": ["Extended introduction"],
                        "meaning_preserved": True,
                    },
                    {
                        "variant_id": "compact",
                        "compression_level": "compact",
                        "target_duration_ms": targets["compact_target_duration_ms"],
                        "spoken_text": "I-EFF iseThohoyandou noDr Levy Ndou.",
                        "estimated_duration_ms": 8500,
                        "preserved_english_spans": ["EFF", "Dr Levy Ndou"],
                        "omitted_optional_details": ["Extended introduction"],
                        "meaning_preserved": True,
                    },
                ],
                "selection_status": "unselected",
                "selected_variant_id": None,
                "warnings": [],
            }
        ],
        "global_terminology": [],
        "warnings": [],
    }


def test_validates_source_bound_multivariant_response() -> None:
    request = build_translation_request(
        transcript=_source(),
        target_locale="zu-ZA",
        job_id="job-1",
    )
    result = validate_multivariant_response(_response(request), request=request)
    assert result["valid"] is True
    assert result["unit_count"] == 1


def test_rejects_large_local_named_anchor_drift() -> None:
    request = build_translation_request(
        transcript=_source(),
        target_locale="zu-ZA",
        job_id="job-local-anchor-drift",
    )
    response = _response(request)
    response["units"][0]["variants"][0]["spoken_text"] = (
        "UDr Levy Ndou usemcimbini we-EFF eThohoyandou namhlanje."
    )

    with pytest.raises(
        MultivariantTranslationError,
        match=r"moves local named anchor 'Dr Levy Ndou'.*preserve its local source progression",
    ):
        validate_multivariant_response(response, request=request)


def test_entity_binding_preserves_source_reference_specificity() -> None:
    transcript = {
        "language": "en-ZA",
        "segments": [
            {
                "segment_id": "unit_0098",
                "speaker": "SPEAKER_00",
                "start": 0.0,
                "end": 5.0,
                "source_text": "Matlala was supposed to testify.",
            }
        ],
    }
    bindings = {
        "per_unit_bindings": {
            "unit_0098": {
                "matches": [
                    {
                        "matched_source_text": "Matlala",
                        "display_text": "Vusimuzi “Cat” Matlala",
                        "canonical_text": "Vusimuzi Cat Matlala",
                    }
                ]
            }
        }
    }

    request = build_translation_request(
        transcript=transcript,
        target_locale="zu-ZA",
        job_id="job-reference-specificity",
        entity_bindings=bindings,
    )

    assert request["units"][0]["protected_spans"] == ["Matlala"]


def test_rejects_substring_name_corruption() -> None:
    request = build_translation_request(
        transcript=_source(),
        target_locale="zu-ZA",
        job_id="job-1",
    )
    response = _response(request)
    response["units"][0]["variants"][2]["spoken_text"] = (
        "I-EFF iseThohoyandou noDr Levy Ando."
    )
    with pytest.raises(MultivariantTranslationError, match="Dr Levy Ndou"):
        validate_multivariant_response(response, request=request)


def test_export_and_atomic_manual_install(tmp_path: Path) -> None:
    job_root = tmp_path / "working" / "jobs" / "job-1"
    atomic_write_json(job_root / "analysis" / "transcript_en.json", _source())
    package = export_translation_package(
        job_root=job_root,
        job_id="job-1",
        target_locale="zu-ZA",
        output_dir=job_root / "translation" / "prompt_package",
    )
    request_path = Path(package["request_path"])
    request = read_json(request_path)
    response_path = job_root / "translation" / "manual_response.json"
    atomic_write_json(response_path, _response(request))

    dry = install_manual_translation(
        job_root=job_root,
        job_id="job-1",
        target_locale="zu-ZA",
        response_path=response_path,
        reviewed_by="Clerence Mapaya",
        request_path=request_path,
        dry_run=True,
    )
    assert dry["status"] == "validated"
    assert not (job_root / "translation" / "transcript_zu.json").exists()

    installed = install_manual_translation(
        job_root=job_root,
        job_id="job-1",
        target_locale="zu-ZA",
        response_path=response_path,
        reviewed_by="Clerence Mapaya",
        request_path=request_path,
    )
    translation = read_json(job_root / "translation" / "transcript_zu.json")
    compatibility = read_json(job_root / "dubbing" / "translation_zu.json")
    assert installed["status"] == "installed"
    assert translation["schema_version"] == INSTALLED_SCHEMA_VERSION
    assert translation["installation"]["reviewed_by"] == "Clerence Mapaya"
    assert compatibility["schema_version"] == "three-text-translation-v2-multivariant"
    assert len(compatibility["units"][0]["translation_variants"]) == 3


def test_rejects_non_decreasing_duration_estimates() -> None:
    request = build_translation_request(
        transcript=_source(),
        target_locale="zu-ZA",
        job_id="job-1",
    )
    response = _response(request)
    response["units"][0]["variants"][1]["estimated_duration_ms"] = 14500
    with pytest.raises(MultivariantTranslationError, match="strictly decrease"):
        validate_multivariant_response(response, request=request)


def test_manual_install_without_explicit_request_prefers_dubbing_units(tmp_path: Path) -> None:
    job_root = tmp_path / "working" / "jobs" / "job-1"
    atomic_write_json(job_root / "analysis" / "transcript_en.json", _source())
    units = {
        "schema_version": "dubbing-units-v1",
        "units": [
            {
                "unit_id": "unit-1",
                "speaker_id": "SPEAKER_00",
                "start_ms": 320,
                "end_ms": 16480,
                "source_text": "The EFF event in Thohoyandou features Dr Levy Ndou.",
                "protected_spans": ["EFF", "Thohoyandou", "Dr Levy Ndou"],
            }
        ],
    }
    atomic_write_json(job_root / "dubbing" / "dubbing_units.json", units)
    request = build_translation_request(
        transcript=units,
        target_locale="zu-ZA",
        job_id="job-1",
    )
    response_path = job_root / "translation" / "manual_response.json"
    atomic_write_json(response_path, _response(request))
    result = install_manual_translation(
        job_root=job_root,
        job_id="job-1",
        target_locale="zu-ZA",
        response_path=response_path,
        reviewed_by="Reviewer",
        dry_run=True,
    )
    assert result["status"] == "validated"


def test_sepedi_artifact_paths_are_language_specific(tmp_path: Path) -> None:
    from mathula_tv.artifacts import DubbingArtifacts

    paths = DubbingArtifacts(tmp_path, target_locale="nso-ZA")
    assert paths.raw_multivariant_translation.name == "transcript_nso.json"
    assert paths.three_text_translation.name == "translation_nso.json"


def test_translation_request_accepts_production_dubbing_unit_timing() -> None:
    units = {
        "schema_version": "dubbing-units-v1",
        "units": [
            {
                "unit_id": "unit_0001",
                "speaker_id": "SPEAKER_00",
                "source_text": "The EFF event is today.",
                "start_ms": 320,
                "preferred_end_ms": 14_480,
                "hard_end_ms": 16_480,
                "preferred_duration_ms": 14_160,
                "maximum_duration_ms": 16_160,
                "overlap_group_id": "overlap_01",
            }
        ],
    }

    request = build_translation_request(
        transcript=units,
        target_locale="zu-ZA",
        job_id="job-1",
    )

    unit = request["units"][0]
    assert unit["start_ms"] == 320
    assert unit["end_ms"] == 16_480
    assert unit["available_duration_ms"] == 16_160
    assert unit["overlap_group"] == "overlap_01"
    assert unit["duration_targets"] == {
        "natural_target_duration_ms": 15_352,
        "concise_target_duration_ms": 13_251,
        "compact_target_duration_ms": 10_989,
    }


def test_translation_request_can_derive_end_from_maximum_duration() -> None:
    units = {
        "units": [
            {
                "unit_id": "unit_0001",
                "speaker_id": "SPEAKER_00",
                "source_text": "Hello.",
                "start_ms": 500,
                "maximum_duration_ms": 2_500,
            }
        ]
    }

    request = build_translation_request(
        transcript=units,
        target_locale="nso-ZA",
        job_id="job-1",
    )

    assert request["units"][0]["end_ms"] == 3_000
    assert request["units"][0]["available_duration_ms"] == 2_500


def test_accepts_reviewed_entity_alias_for_protected_identity() -> None:
    source = _source()
    segment = source["segments"][0]
    segment["source_text"] = "The Economic Freedom Fighters event is in Thohoyandou."
    segment["protected_spans"] = ["Economic Freedom Fighters", "Thohoyandou"]
    request = build_translation_request(
        transcript=source,
        target_locale="zu-ZA",
        job_id="job-1",
    )
    response = _response(request)
    # All three variants use the entity registry's reviewed, timing-safe alias.
    for variant in response["units"][0]["variants"]:
        variant["preserved_english_spans"] = ["EFF"]
    validation = validate_multivariant_response(response, request=request)
    evidence = validation["units"][0]["variants"][0]["protected_span_evidence"]
    assert evidence[0]["required_span"] == "Economic Freedom Fighters"
    assert evidence[0]["matched_form"] == "EFF"



def _khumalo_identity_case() -> tuple[dict, dict]:
    source_text = (
        "Earlier, we heard one of the Commissioners, "
        "Advocate-Commissioner Khumalo, in fact, talk about how,"
    )
    transcript = {
        "source_language": "en-ZA",
        "segments": [
            {
                "unit_id": "unit_0023",
                "speaker_id": "SPEAKER_01",
                "start_ms": 140115,
                "end_ms": 145949,
                "source_text": source_text,
                "protected_spans": ["Advocate Sandile Khumalo SC"],
            }
        ],
    }
    request = build_translation_request(
        transcript=transcript,
        target_locale="zu-ZA",
        job_id="7c32c0745a0a432987af35f5ef91ec43",
    )
    response = _response(request)
    unit = response["units"][0]
    unit["required_facts"] = ["Commissioner Khumalo talked"]
    unit["optional_details"] = []
    spoken = [
        (
            "Ekuqaleni, sizwe omunye wamaKhomishana, "
            "u-Advocate Khumalo, empeleni, ekhuluma ngokuthi,"
        ),
        "Ekuqaleni sizwe u-Advocate Khumalo, iKhomishana, ekhuluma ngokuthi,",
        "Sizwe u-Advocate Khumalo ekhuluma ngokuthi,",
    ]
    estimates = [5480, 4720, 3900]
    for variant, spoken_text, estimate in zip(
        unit["variants"], spoken, estimates, strict=True
    ):
        variant["spoken_text"] = spoken_text
        variant["estimated_duration_ms"] = estimate
        variant["preserved_english_spans"] = ["Advocate"]
        variant["omitted_optional_details"] = []
        variant["meaning_preserved"] = True
    return request, response


def test_accepts_source_supported_advocate_khumalo_identity_alias() -> None:
    request, response = _khumalo_identity_case()

    validation = validate_multivariant_response(response, request=request)

    evidence = [
        variant["protected_span_evidence"][0]
        for variant in validation["units"][0]["variants"]
    ]
    assert [item["matched_form"] for item in evidence] == [
        "Advocate Khumalo",
        "Advocate Khumalo",
        "Advocate Khumalo",
    ]
    assert all("Advocate Khumalo" in item["approved_forms"] for item in evidence)
    assert all("Khumalo" not in item["approved_forms"] for item in evidence)


def test_bare_khumalo_does_not_satisfy_full_protected_identity() -> None:
    request, response = _khumalo_identity_case()
    for variant in response["units"][0]["variants"]:
        variant["spoken_text"] = variant["spoken_text"].replace(
            "u-Advocate Khumalo", "uKhumalo"
        )
        variant["preserved_english_spans"] = []

    with pytest.raises(MultivariantTranslationError, match="Advocate Sandile Khumalo SC"):
        validate_multivariant_response(response, request=request)

def test_does_not_accept_invented_person_initials_as_identity_alias() -> None:
    request = build_translation_request(
        transcript=_source(),
        target_locale="zu-ZA",
        job_id="job-1",
    )
    response = _response(request)
    for variant in response["units"][0]["variants"]:
        variant["spoken_text"] = variant["spoken_text"].replace("Dr Levy Ndou", "LN")
        variant["preserved_english_spans"] = ["EFF"]
    with pytest.raises(MultivariantTranslationError, match="Dr Levy Ndou"):
        validate_multivariant_response(response, request=request)


def test_accepts_adjacent_equivalent_variants_at_compression_floor() -> None:
    request = build_translation_request(
        transcript=_source(),
        target_locale="zu-ZA",
        job_id="job-1",
    )
    response = _response(request)
    natural = response["units"][0]["variants"][0]
    for variant in response["units"][0]["variants"][1:]:
        variant["spoken_text"] = natural["spoken_text"]
        variant["preserved_english_spans"] = list(
            natural["preserved_english_spans"]
        )

    validation = validate_multivariant_response(response, request=request)

    unit = validation["units"][0]
    assert unit["compression_floor_reached"] is True
    assert unit["distinct_spoken_variant_count"] == 1
    assert unit["equivalent_variant_groups"] == [
        ["natural", "concise", "compact"]
    ]


def test_rejects_non_adjacent_equivalent_variants() -> None:
    request = build_translation_request(
        transcript=_source(),
        target_locale="zu-ZA",
        job_id="job-1",
    )
    response = _response(request)
    natural = response["units"][0]["variants"][0]
    compact = response["units"][0]["variants"][2]
    compact["spoken_text"] = natural["spoken_text"]
    compact["preserved_english_spans"] = list(natural["preserved_english_spans"])

    with pytest.raises(MultivariantTranslationError, match="non-adjacent"):
        validate_multivariant_response(response, request=request)


def test_normalizer_discards_false_preserved_english_span_claim() -> None:
    request = build_translation_request(
        transcript=_source(),
        target_locale="zu-ZA",
        job_id="job-false-preservation-metadata",
    )
    response = _response(request)
    source_spoken = []
    for variant in response["units"][0]["variants"]:
        source_spoken.append(variant["spoken_text"])
        variant["preserved_english_spans"] = [
            *variant["preserved_english_spans"],
            "Witness",
        ]

    normalized = normalize_multivariant_response(response, request=request)
    unit = normalized["units"][0]

    assert [item["spoken_text"] for item in unit["variants"]] == source_spoken
    assert all(
        "Witness" not in item["preserved_english_spans"]
        for item in unit["variants"]
    )
    assert all(
        "EFF" in item["preserved_english_spans"]
        for item in unit["variants"]
    )
    assert unit["warnings"][-3:] == [
        {
            "code": "discarded_false_preserved_english_span_claim",
            "variant_id": variant_id,
            "span": "Witness",
        }
        for variant_id in ("natural", "concise", "compact")
    ]

    validation = validate_multivariant_response(normalized, request=request)
    assert validation["unit_count"] == 1


def test_direct_validator_remains_strict_for_false_preservation_claims() -> None:
    request = build_translation_request(
        transcript=_source(),
        target_locale="zu-ZA",
        job_id="job-strict-direct-validation",
    )
    response = _response(request)
    response["units"][0]["variants"][0]["preserved_english_spans"].append(
        "Witness"
    )

    with pytest.raises(
        MultivariantTranslationError,
        match="claims preserved English spans",
    ):
        validate_multivariant_response(response, request=request)


def _madlanga_translation_case(source_text: str) -> tuple[dict, dict]:
    transcript = {
        "schema_version": "transcript-v1",
        "segments": [
            {
                "segment_id": "seg-madlanga-1",
                "speaker": "SPEAKER_00",
                "start": 0.0,
                "end": 12.0,
                "source_text": source_text,
                "protected_spans": ["Madlanga Commission of Inquiry"],
            }
        ],
    }
    request = build_translation_request(
        transcript=transcript,
        target_locale="zu-ZA",
        job_id="job-madlanga-alias",
    )
    response = _response(request)
    response["units"][0]["required_facts"] = ["The Commission remains identified."]
    response["units"][0]["optional_details"] = []
    for variant in response["units"][0]["variants"]:
        variant["preserved_english_spans"] = []
        variant["omitted_optional_details"] = []
    return request, response


def test_accepts_reviewed_isizulu_alias_for_named_madlanga_commission() -> None:
    request, response = _madlanga_translation_case(
        "The Madlanga Commission of Inquiry will consider the matter."
    )
    spoken = [
        "Udaba luzobhekwa eKhomishini kaMadlanga ngokugcwele.",
        "Udaba luzobhekwa eKhomishini kaMadlanga.",
        "EKhomishini kaMadlanga kuzobhekwa udaba.",
    ]
    for variant, text in zip(response["units"][0]["variants"], spoken, strict=True):
        variant["spoken_text"] = text

    validation = validate_multivariant_response(response, request=request)

    evidence = validation["units"][0]["variants"][0]["protected_span_evidence"][0]
    assert evidence["matched_form"] == "Khomishini kaMadlanga"
    assert "Khomishini kaMadlanga" in evidence["approved_forms"]


def test_accepts_generic_isizulu_commission_only_for_generic_source_alias() -> None:
    request, response = _madlanga_translation_case(
        "The Commission will consider the matter."
    )
    spoken = [
        "IKhomishini izolubheka ngokugcwele lolu daba.",
        "IKhomishini izolubheka lolu daba.",
        "Luzobhekwa yiKhomishini.",
    ]
    for variant, text in zip(response["units"][0]["variants"], spoken, strict=True):
        variant["spoken_text"] = text

    validation = validate_multivariant_response(response, request=request)

    evidence = validation["units"][0]["variants"][0]["protected_span_evidence"][0]
    assert evidence["matched_form"] == "Khomishini"
    assert "Khomishini" in evidence["approved_forms"]


def test_named_madlanga_commission_cannot_collapse_to_generic_commission() -> None:
    request, response = _madlanga_translation_case(
        "The Madlanga Commission of Inquiry will consider the matter."
    )
    spoken = [
        "IKhomishini izolubheka ngokugcwele lolu daba.",
        "IKhomishini izolubheka lolu daba.",
        "Luzobhekwa yiKhomishini.",
    ]
    for variant, text in zip(response["units"][0]["variants"], spoken, strict=True):
        variant["spoken_text"] = text

    with pytest.raises(MultivariantTranslationError, match="Madlanga Commission"):
        validate_multivariant_response(response, request=request)


def test_accepts_isizulu_locative_suffix_for_named_madlanga_commission() -> None:
    request, response = _madlanga_translation_case(
        "The Madlanga Commission of Inquiry will consider the matter."
    )
    spoken = [
        "Udaba luzobhekwa eKhomishinini kaMadlanga ngokugcwele.",
        "Udaba luzobhekwa eKhomishinini kaMadlanga.",
        "EKhomishinini kaMadlanga kuzobhekwa udaba.",
    ]
    for variant, text in zip(response["units"][0]["variants"], spoken, strict=True):
        variant["spoken_text"] = text

    validation = validate_multivariant_response(response, request=request)

    evidence = validation["units"][0]["variants"][0]["protected_span_evidence"][0]
    assert evidence["matched_form"] == "Khomishini … Madlanga"


def test_accepts_noncontiguous_isizulu_identity_anchors() -> None:
    request, response = _madlanga_translation_case(
        "The Madlanga Commission of Inquiry will consider the matter."
    )
    spoken = [
        "Udaba luzobhekwa yiKhomishini Yophenyo eholwa uJustice Madlanga ngokugcwele.",
        "IKhomishini Yophenyo eholwa uMadlanga izolubheka udaba.",
        "Luzobhekwa eKhomishini eholwa uMadlanga.",
    ]
    for variant, text in zip(response["units"][0]["variants"], spoken, strict=True):
        variant["spoken_text"] = text

    validation = validate_multivariant_response(response, request=request)

    evidence = validation["units"][0]["variants"][0]["protected_span_evidence"][0]
    assert evidence["matched_form"] == "Khomishini … Madlanga"


def test_named_madlanga_identity_token_group_has_bounded_gap() -> None:
    request, response = _madlanga_translation_case(
        "The Madlanga Commission of Inquiry will consider the matter."
    )
    spoken = (
        "IKhomishini izobheka udaba olude kakhulu olunezinto eziningi kakhulu "
        "ezihlukene ngaphambi kokuba kukhulunywe ngoMadlanga."
    )
    for variant in response["units"][0]["variants"]:
        variant["spoken_text"] = spoken

    with pytest.raises(MultivariantTranslationError, match="Madlanga Commission"):
        validate_multivariant_response(response, request=request)


def test_accepts_reviewed_ikhomishana_for_generic_commission_source() -> None:
    request, response = _madlanga_translation_case(
        "The Commission will consider the matter."
    )
    spoken = [
        "IKhomishana izolubheka ngokugcwele lolu daba.",
        "IKhomishana izolubheka lolu daba.",
        "Luzobhekwa yiKhomishana.",
    ]
    for variant, text in zip(response["units"][0]["variants"], spoken, strict=True):
        variant["spoken_text"] = text

    validation = validate_multivariant_response(response, request=request)

    evidence = validation["units"][0]["variants"][0]["protected_span_evidence"][0]
    assert evidence["matched_form"] == "Khomishana"
    assert "Khomishana" in evidence["approved_forms"]


def test_accepts_reviewed_ikhomishana_for_named_madlanga_commission() -> None:
    request, response = _madlanga_translation_case(
        "The Madlanga Commission of Inquiry will consider the matter."
    )
    spoken = [
        "Udaba luzobhekwa eKhomishaneni Yophenyo kaMadlanga ngokugcwele.",
        "Udaba luzobhekwa yiKhomishana kaMadlanga.",
        "IKhomishana eholwa uJustice Madlanga izolubheka udaba.",
    ]
    for variant, text in zip(response["units"][0]["variants"], spoken, strict=True):
        variant["spoken_text"] = text

    validation = validate_multivariant_response(response, request=request)

    evidence = validation["units"][0]["variants"][0]["protected_span_evidence"][0]
    assert evidence["matched_form"] in {
        "Khomishana Yophenyo kaMadlanga",
        "Khomishana kaMadlanga",
        "Khomishana … Madlanga",
    }


def test_named_madlanga_commission_cannot_collapse_to_generic_ikhomishana() -> None:
    request, response = _madlanga_translation_case(
        "The Madlanga Commission of Inquiry will consider the matter."
    )
    spoken = [
        "IKhomishana izolubheka ngokugcwele lolu daba.",
        "IKhomishana izolubheka lolu daba.",
        "Luzobhekwa yiKhomishana.",
    ]
    for variant, text in zip(response["units"][0]["variants"], spoken, strict=True):
        variant["spoken_text"] = text

    with pytest.raises(MultivariantTranslationError, match="Madlanga Commission"):
        validate_multivariant_response(response, request=request)



def test_accepts_source_attested_madlanga_shorthand_as_person_name_anchor() -> None:
    request, response = _madlanga_translation_case(
        "These are matters she already admitted before Madlanga."
    )
    spoken = [
        "Lezi yizinto asezivumile phambi kukaJustice Mbuyiseli Madlanga.",
        "Usezivumile lezi zinto phambi kukaJustice Madlanga.",
        "Wazivuma phambi kukaMadlanga.",
    ]
    for variant, text in zip(response["units"][0]["variants"], spoken, strict=True):
        variant["spoken_text"] = text

    validation = validate_multivariant_response(response, request=request)

    evidence = validation["units"][0]["variants"][0]["protected_span_evidence"][0]
    assert evidence["matched_form"] == "Madlanga"
    assert "Madlanga" in evidence["approved_forms"]


def test_full_madlanga_commission_name_cannot_collapse_to_person_name_anchor() -> None:
    request, response = _madlanga_translation_case(
        "The Madlanga Commission of Inquiry will consider the matter."
    )
    for variant in response["units"][0]["variants"]:
        variant["spoken_text"] = "Udaba luzobhekwa nguJustice Mbuyiseli Madlanga."

    with pytest.raises(MultivariantTranslationError, match="Madlanga Commission"):
        validate_multivariant_response(response, request=request)


def test_accepts_source_attested_english_commission_with_isizulu_prefix() -> None:
    request, response = _madlanga_translation_case(
        "let's put it that way, in the Crime Intelligence division and what the Commission must make of that alleged improper conduct."
    )
    spoken = [
        "ake sisho kanjalo, ophikweni lwe-Crime Intelligence, nokuthi i-Commission kufanele yenzeni ngalokho kuziphatha okubi okusolwayo.",
        "ophikweni lwe-Crime Intelligence, nokuthi iCommission yenzeni ngalokho kuziphatha okubi okusolwayo.",
        "ophikweni lwe-Crime Intelligence, nokuthi yi-Commission okufanele ithathe isinqumo.",
    ]
    for variant, text in zip(response["units"][0]["variants"], spoken, strict=True):
        variant["spoken_text"] = text

    validation = validate_multivariant_response(response, request=request)

    evidence = validation["units"][0]["variants"][0]["protected_span_evidence"][0]
    assert evidence["matched_form"] == "Commission"
    assert "Commission" in evidence["approved_forms"]


def test_named_madlanga_commission_cannot_collapse_to_prefixed_english_commission() -> None:
    request, response = _madlanga_translation_case(
        "The Madlanga Commission of Inquiry will consider the matter."
    )
    for variant in response["units"][0]["variants"]:
        variant["spoken_text"] = "I-Commission izolubheka lolu daba."

    with pytest.raises(MultivariantTranslationError, match="Madlanga Commission"):
        validate_multivariant_response(response, request=request)



def test_accepts_reviewed_ikhomishani_morphology_for_generic_commission() -> None:
    request, response = _madlanga_translation_case(
        "The Commission will consider the matter."
    )
    spoken = [
        "Usihlalo weKhomishani uzolubheka lolu daba.",
        "Udaba luzobhekwa eKhomishanini.",
        "IKhomishani izolubheka.",
    ]
    for variant, text in zip(response["units"][0]["variants"], spoken, strict=True):
        variant["spoken_text"] = text

    validation = validate_multivariant_response(response, request=request)

    evidence = validation["units"][0]["variants"]
    assert [item["protected_span_evidence"][0]["matched_form"] for item in evidence] == [
        "Khomishani",
        "Khomishani",
        "Khomishani",
    ]


def test_identity_alias_sequence_accepts_attached_isizulu_morphology() -> None:
    transcript = {
        "source_language": "en-ZA",
        "segments": [
            {
                "speaker_id": "speaker_1",
                "start_ms": 0,
                "end_ms": 6000,
                "text": "Witness F referred to the Mokwele docket.",
                "protected_spans": ["Witness F", "Brigadier Dineo Mokwele"],
            }
        ],
    }
    request = build_translation_request(
        transcript=transcript,
        target_locale="zu-ZA",
        job_id="job-attached-isizulu-morphology",
    )
    response = _response(request)
    response["units"][0]["required_facts"] = ["Witness F", "Mokwele docket"]
    response["units"][0]["optional_details"] = []
    spoken = [
        "KwakunguWitness F ekhuluma ngedokethi likaMokwele.",
        "NjengoWitness F, wakhuluma ngelikaMokwele.",
        "NjengoWitness F, ngelikaMokwele.",
    ]
    for variant, text in zip(response["units"][0]["variants"], spoken, strict=True):
        variant["spoken_text"] = text
        variant["preserved_english_spans"] = []
        variant["omitted_optional_details"] = []

    validation = validate_multivariant_response(response, request=request)

    for variant in validation["units"][0]["variants"]:
        matched = {
            item["required_span"]: item["matched_form"]
            for item in variant["protected_span_evidence"]
        }
        assert matched["Witness F"] == "Witness F"
        assert matched["Brigadier Dineo Mokwele"] == "Mokwele"


def test_compact_may_omit_generic_identity_when_longer_variants_preserve_it() -> None:
    request, response = _madlanga_translation_case(
        "A witness was in camera before the Commission."
    )
    spoken = [
        "Ufakazi ubengasese phambi kweKhomishani.",
        "Ufakazi ubengasese eKhomishanini.",
        "Ufakazi ubengasese.",
    ]
    for variant, text in zip(response["units"][0]["variants"], spoken, strict=True):
        variant["spoken_text"] = text

    validation = validate_multivariant_response(response, request=request)

    compact_evidence = validation["units"][0]["variants"][2][
        "protected_span_evidence"
    ][0]
    assert compact_evidence["matched_form"] is None
    assert compact_evidence["omission_policy"] == (
        "compact_generic_identity_omission_with_longer_variants_preserved"
    )


def test_compact_cannot_omit_named_identity_even_when_longer_variants_preserve_it() -> None:
    request, response = _madlanga_translation_case(
        "A witness appeared before the Madlanga Commission of Inquiry."
    )
    spoken = [
        "Ufakazi wavela phambi kweKhomishani kaMadlanga.",
        "Ufakazi wavela eKhomishini kaMadlanga.",
        "Ufakazi wavela.",
    ]
    for variant, text in zip(response["units"][0]["variants"], spoken, strict=True):
        variant["spoken_text"] = text

    with pytest.raises(MultivariantTranslationError, match="Madlanga Commission"):
        validate_multivariant_response(response, request=request)


def test_generic_identity_omission_requires_both_longer_variants_to_preserve() -> None:
    request, response = _madlanga_translation_case(
        "A witness was in camera before the Commission."
    )
    spoken = [
        "Ufakazi ubengasese phambi kweKhomishani.",
        "Ufakazi ubengasese.",
        "Ufakazi ubengasese.",
    ]
    for variant, text in zip(response["units"][0]["variants"], spoken, strict=True):
        variant["spoken_text"] = text

    with pytest.raises(MultivariantTranslationError, match="variant concise"):
        validate_multivariant_response(response, request=request)


def _npa_direct_ndpp_case() -> tuple[dict, dict]:
    source_text = (
        "the lawyers of the accused then sent representation to the NPA, "
        "directly to the NDPP,"
    )
    transcript = {
        "source_language": "en-ZA",
        "segments": [
            {
                "speaker_id": "SPEAKER_01",
                "start_ms": 0,
                "end_ms": 6183,
                "text": source_text,
                "protected_spans": ["National Prosecuting Authority"],
            }
        ],
    }
    request = build_translation_request(
        transcript=transcript,
        target_locale="zu-ZA",
        job_id="job-npa-direct-ndpp",
    )
    response = _response(request)
    unit = response["units"][0]
    unit["required_facts"] = [
        "lawyers sent representations",
        "the representations went to the NPA directly through the NDPP",
    ]
    unit["optional_details"] = []
    spoken = [
        "abameli babasolwa base bethumela izicelo e-NPA, ngqo ku-NDPP,",
        "abameli babasolwa bathumela izicelo e-NPA, ngqo ku-NDPP,",
        "abameli bathumela izicelo ngqo ku-NDPP,",
    ]
    preserved = [["NPA", "NDPP"], ["NPA", "NDPP"], ["NDPP"]]
    for variant, text, spans in zip(
        unit["variants"], spoken, preserved, strict=True
    ):
        variant["spoken_text"] = text
        variant["preserved_english_spans"] = spans
        variant["omitted_optional_details"] = []
    return request, response


def test_compact_may_use_reviewed_ndpp_substitute_for_direct_npa_recipient() -> None:
    request, response = _npa_direct_ndpp_case()

    validation = validate_multivariant_response(response, request=request)

    evidence = validation["units"][0]["variants"][2][
        "protected_span_evidence"
    ][0]
    assert evidence["matched_form"] is None
    assert evidence["substitute_form"] == "NDPP"
    assert evidence["omission_policy"] == (
        "compact_reviewed_identity_substitution_with_longer_variants_preserved"
    )


def test_compact_ndpp_substitution_requires_ndpp_in_same_source_unit() -> None:
    request, response = _npa_direct_ndpp_case()
    request["units"][0]["source_text"] = (
        "the lawyers of the accused sent representations to the NPA."
    )
    response["units"][0]["source_text"] = request["units"][0]["source_text"]

    with pytest.raises(MultivariantTranslationError, match="National Prosecuting"):
        validate_multivariant_response(response, request=request)


def test_compact_ndpp_substitution_does_not_accept_unreviewed_recipient() -> None:
    request, response = _npa_direct_ndpp_case()
    response["units"][0]["variants"][2]["spoken_text"] = (
        "abameli bathumela izicelo ngqo kuMongameli."
    )
    response["units"][0]["variants"][2]["preserved_english_spans"] = []

    with pytest.raises(MultivariantTranslationError, match="National Prosecuting"):
        validate_multivariant_response(response, request=request)


def test_accepts_reviewed_khomishane_and_nge_prefix_for_madlanga_commission() -> None:
    request, response = _madlanga_translation_case(
        "There is concern about developments at the Madlanga Commission."
    )
    spoken = [
        "Kusekhona ukukhathazeka ngokuthuthuka kweKhomishane kaMadlanga.",
        "Kukhona ukukhathazeka kweKhomishane kaMadlanga.",
        "Kukhona ukukhathazeka ngeKhomishane kaMadlanga.",
    ]
    for variant, text in zip(response["units"][0]["variants"], spoken, strict=True):
        variant["spoken_text"] = text

    validation = validate_multivariant_response(response, request=request)

    evidence = validation["units"][0]["variants"]
    assert all(item["protected_span_evidence"][0]["matched_form"] for item in evidence)


def _undeclared_compact_detail_case(
    *,
    source_text: str,
    detail: str,
    spoken: list[str],
    protected_spans: list[str] | None = None,
) -> tuple[dict, dict]:
    transcript = {
        "source_language": "en-ZA",
        "segments": [
            {
                "speaker_id": "SPEAKER_00",
                "start_ms": 0,
                "end_ms": 9000,
                "text": source_text,
                "protected_spans": protected_spans or [],
            }
        ],
    }
    request = build_translation_request(
        transcript=transcript,
        target_locale="zu-ZA",
        job_id="job-compact-optional-detail-recovery",
    )
    response = _response(request)
    unit = response["units"][0]
    unit["required_facts"] = []
    unit["optional_details"] = []
    for variant, text in zip(unit["variants"], spoken, strict=True):
        variant["spoken_text"] = text
        variant["preserved_english_spans"] = []
        variant["omitted_optional_details"] = []
    unit["variants"][2]["omitted_optional_details"] = [detail]
    return request, response


def test_normalization_promotes_source_grounded_compact_optional_detail() -> None:
    detail = "Department of Police Practice"
    request, response = _undeclared_compact_detail_case(
        source_text=(
            "Professor Dumisani Mabunda is a security analyst and lecturer at "
            "Unisa's Department of Police Practice."
        ),
        detail=detail,
        spoken=[
            "UProfessor Dumisani Mabunda uyingcweti kwezokuphepha futhi "
            "ungumfundisi eUnisa eMnyangweni wePolice Practice.",
            "UProfessor Dumisani Mabunda uyingcweti kwezokuphepha nomfundisi "
            "eUnisa kwaPolice Practice.",
            "UProfessor Dumisani Mabunda uyingcweti yezokuphepha eUnisa.",
        ],
    )

    normalized = normalize_multivariant_response(response, request=request)
    validation = validate_multivariant_response(normalized, request=request)

    unit = normalized["units"][0]
    assert unit["optional_details"] == [detail]
    assert {
        "code": "promoted_source_grounded_compact_optional_detail",
        "variant_id": "compact",
        "detail": detail,
    } in unit["warnings"]
    assert validation["units"][0]["variants"][2]["optional_detail_count"] == 1


def test_normalization_promotes_source_grounded_compact_acronym_detail() -> None:
    request, response = _undeclared_compact_detail_case(
        source_text=(
            "with the testimony of the former NDPP, Shamila Batohi, "
            "at the Nugent Inquiry"
        ),
        detail="NDPP",
        spoken=[
            "nobufakazi bowayeyi-NDPP, uShamila Batohi, eNugent Inquiry,",
            "nobufakazi bowayeyi-NDPP, uShamila Batohi, eNugent Inquiry,",
            "nobufakazi buka-Shamila Batohi, eNugent Inquiry,",
        ],
    )

    normalized = normalize_multivariant_response(response, request=request)
    validate_multivariant_response(normalized, request=request)

    assert normalized["units"][0]["optional_details"] == ["NDPP"]


def test_normalization_does_not_promote_detail_absent_from_source() -> None:
    request, response = _undeclared_compact_detail_case(
        source_text="Professor Mabunda is a security analyst at Unisa.",
        detail="Department of Police Practice",
        spoken=[
            "UProfessor Mabunda ungumhlaziyi eUnisa kwaPolice Practice.",
            "UProfessor Mabunda ungumhlaziyi kwaPolice Practice.",
            "UProfessor Mabunda ungumhlaziyi eUnisa.",
        ],
    )

    normalized = normalize_multivariant_response(response, request=request)

    assert normalized["units"][0]["optional_details"] == []
    with pytest.raises(MultivariantTranslationError):
        validate_multivariant_response(normalized, request=request)


def test_normalization_never_promotes_protected_compact_omission() -> None:
    detail = "Department of Police Practice"
    request, response = _undeclared_compact_detail_case(
        source_text=(
            "Professor Mabunda lectures at the Department of Police Practice."
        ),
        detail=detail,
        protected_spans=[detail],
        spoken=[
            "UProfessor Mabunda ufundisa kwaPolice Practice.",
            "UMabunda ufundisa kwaPolice Practice.",
            "UMabunda uyafundisa.",
        ],
    )

    normalized = normalize_multivariant_response(response, request=request)

    assert normalized["units"][0]["optional_details"] == []
    with pytest.raises(MultivariantTranslationError):
        validate_multivariant_response(normalized, request=request)


def test_validation_boundary_recovers_source_grounded_compact_detail_without_normalizer() -> None:
    detail = "Department of Police Practice"
    request, response = _undeclared_compact_detail_case(
        source_text=(
            "Professor Dumisani Mabunda is a security analyst and lecturer at "
            "Unisa's Department of Police Practice."
        ),
        detail=detail,
        spoken=[
            "UProfessor Dumisani Mabunda uyingcweti kwezokuphepha futhi "
            "ungumfundisi eUnisa eMnyangweni wePolice Practice.",
            "UProfessor Dumisani Mabunda uyingcweti kwezokuphepha nomfundisi "
            "eUnisa kwaPolice Practice.",
            "UProfessor Dumisani Mabunda uyingcweti yezokuphepha eUnisa.",
        ],
    )

    validation = validate_multivariant_response(response, request=request)

    compact = validation["units"][0]["variants"][2]
    assert compact["optional_detail_count"] == 1
    assert compact["validation_promoted_optional_details"] == [detail]


def test_validation_boundary_does_not_recover_protected_compact_detail() -> None:
    detail = "Department of Police Practice"
    request, response = _undeclared_compact_detail_case(
        source_text=(
            "Professor Mabunda lectures at the Department of Police Practice."
        ),
        detail=detail,
        protected_spans=[detail],
        spoken=[
            "UProfessor Mabunda ufundisa kwaPolice Practice.",
            "UMabunda ufundisa kwaPolice Practice.",
            "UMabunda uyafundisa.",
        ],
    )

    with pytest.raises(MultivariantTranslationError):
        validate_multivariant_response(response, request=request)


def test_normalizer_recovers_missing_natural_identity_from_valid_concise_sibling() -> None:
    request, response = _khumalo_identity_case()
    unit = response["units"][0]
    natural = unit["variants"][0]
    concise = unit["variants"][1]
    natural_original = natural["spoken_text"]
    natural["spoken_text"] = (
        "Ekuqaleni, sizwe omunye wamaKhomishana, empeleni, ekhuluma ngokuthi,"
    )
    natural["preserved_english_spans"] = []

    normalized = normalize_multivariant_response(response, request=request)
    normalized_unit = normalized["units"][0]
    normalized_natural = normalized_unit["variants"][0]

    assert normalized_natural["spoken_text"] == concise["spoken_text"]
    assert normalized_natural["estimated_duration_ms"] == concise["estimated_duration_ms"]
    assert natural_original != normalized_natural["spoken_text"]
    assert {
        "code": "protected_identity_compression_floor_fallback",
        "missing_required_spans": ["Advocate Sandile Khumalo SC"],
        "from_variant_id": "concise",
        "to_variant_id": "natural",
        "discarded_spoken_text": (
            "Ekuqaleni, sizwe omunye wamaKhomishana, empeleni, ekhuluma ngokuthi,"
        ),
        "replacement_policy": (
            "reuse_nearest_sibling_preserving_all_protected_identities"
        ),
    } in normalized_unit["warnings"]

    validation = validate_multivariant_response(normalized, request=request)
    assert validation["valid"] is True
    assert validation["units"][0]["compression_floor_reached"] is True


def test_normalizer_fails_closed_when_every_sibling_omits_protected_identity() -> None:
    request, response = _khumalo_identity_case()
    for variant in response["units"][0]["variants"]:
        variant["spoken_text"] = "Sizwe iKhomishana ikhuluma ngalolu daba."
        variant["preserved_english_spans"] = []

    normalized = normalize_multivariant_response(response, request=request)

    assert not any(
        warning.get("code") == "protected_identity_compression_floor_fallback"
        for warning in normalized["units"][0]["warnings"]
        if isinstance(warning, dict)
    )
    with pytest.raises(MultivariantTranslationError, match="Advocate Sandile Khumalo SC"):
        validate_multivariant_response(normalized, request=request)


def test_identity_sibling_donor_must_preserve_every_protected_identity() -> None:
    request, response = _khumalo_identity_case()
    request["units"][0]["protected_spans"] = [
        "Advocate Sandile Khumalo SC",
        "EFF",
    ]
    unit = response["units"][0]
    unit["protected_spans"] = list(request["units"][0]["protected_spans"])
    unit["variants"][0]["spoken_text"] = "I-EFF ikhuluma ngalolu daba."
    unit["variants"][0]["preserved_english_spans"] = ["EFF"]

    normalized = normalize_multivariant_response(response, request=request)

    assert normalized["units"][0]["variants"][0]["spoken_text"] == (
        "I-EFF ikhuluma ngalolu daba."
    )
    with pytest.raises(MultivariantTranslationError):
        validate_multivariant_response(normalized, request=request)


def _commissioner_khumalo_split_identity_case() -> tuple[dict, dict]:
    source_text = (
        "What Commissioner Khumalo said earlier regarding this emerging pattern "
        "of IDAC going beyond the scope of its work and how that may, on his reading,"
    )
    transcript = {
        "source_language": "en-ZA",
        "segments": [
            {
                "unit_id": "unit_0029",
                "speaker_id": "SPEAKER_01",
                "start_ms": 174011,
                "end_ms": 181513,
                "source_text": source_text,
                "protected_spans": [
                    "Advocate Sandile Khumalo SC",
                    "Investigating Directorate Against Corruption",
                ],
            }
        ],
    }
    request = build_translation_request(
        transcript=transcript,
        target_locale="zu-ZA",
        job_id="7c32c0745a0a432987af35f5ef91ec43",
    )
    response = _response(request)
    unit = response["units"][0]
    unit["required_facts"] = [
        "Commissioner Khumalo said",
        "pattern of IDAC going beyond scope",
        "on his reading",
    ]
    unit["optional_details"] = []
    spoken = [
        (
            "Lokho okushiwo yiKhomishana uKhumalo ngaphambilini mayelana nalo "
            "mkhuba ovelayo we-IDAC wokweqa umkhawulo womsebenzi wayo nokuthi "
            "lokho kungaba, ngokubuka kwakhe,"
        ),
        (
            "Lokho okushiwo yiKhomishana uKhumalo ngalo mkhuba we-IDAC wokweqa "
            "umkhawulo womsebenzi, nokuthi ngokubuka kwakhe,"
        ),
        "Okushiwo uKhumalo ngomkhuba we-IDAC wokweqa umkhawulo, nokuthi ngokubuka kwakhe,",
    ]
    estimates = [7090, 6110, 5060]
    for variant, spoken_text, estimate in zip(
        unit["variants"], spoken, estimates, strict=True
    ):
        variant["spoken_text"] = spoken_text
        variant["estimated_duration_ms"] = estimate
        variant["preserved_english_spans"] = ["IDAC"]
        variant["omitted_optional_details"] = []
        variant["meaning_preserved"] = True
    return request, response


def test_accepts_split_commissioner_title_and_compact_source_attested_surname() -> None:
    request, response = _commissioner_khumalo_split_identity_case()

    validation = validate_multivariant_response(response, request=request)

    variants = validation["units"][0]["variants"]
    khumalo_evidence = [
        next(
            item
            for item in variant["protected_span_evidence"]
            if item["required_span"] == "Advocate Sandile Khumalo SC"
        )
        for variant in variants
    ]
    assert [item["matched_form"] for item in khumalo_evidence[:2]] == [
        "Khomishana … Khumalo",
        "Khomishana … Khumalo",
    ]
    assert khumalo_evidence[2]["matched_form"] is None
    assert khumalo_evidence[2]["omission_policy"] == (
        "compact_reviewed_identity_substitution_with_longer_variants_preserved"
    )
    assert khumalo_evidence[2]["substitute_form"] == "Khumalo"


def test_compact_bare_khumalo_requires_richer_variants_to_preserve_title() -> None:
    request, response = _commissioner_khumalo_split_identity_case()
    for variant in response["units"][0]["variants"][:2]:
        variant["spoken_text"] = variant["spoken_text"].replace(
            "yiKhomishana uKhumalo", "uKhumalo"
        )

    with pytest.raises(MultivariantTranslationError, match="Advocate Sandile Khumalo SC"):
        validate_multivariant_response(response, request=request)
