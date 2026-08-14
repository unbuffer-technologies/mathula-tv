from mathula_tv.professional_dub import (
    ProfessionalDubPolicy,
    TranslationRevision,
    auto_approve_low_risk_cadence_revision,
    build_translation_revision_request,
    validate_translation_revision,
)


def test_low_risk_zulu_acknowledgement_can_be_auto_approved():
    decision = auto_approve_low_risk_cadence_revision(
        block_id="block_0013",
        text="Siyabonga kakhulu ngesikhathi sakho.",
        target_language="zu-ZA",
    )
    assert decision is not None
    assert decision["decision"] == "auto_approved"
    assert decision["revision"]["after"] == "Siyabonga."
    assert decision["validation"]["accepted"] is True


def test_substantive_dialogue_is_never_auto_approved():
    decision = auto_approve_low_risk_cadence_revision(
        block_id="block_0002",
        text="I-EFF ithole amavoti angu-20%.",
        target_language="zu-ZA",
    )
    assert decision is None


def test_wording_can_change_when_hard_editorial_invariants_are_preserved():
    revision = TranslationRevision(
        revision_id="rev-1",
        block_id="block_0003",
        phrase_id="block_0003_phrase_002",
        before="Kusolwa ukuthi i-EFF ithole amavoti angu-20%.",
        after="I-EFF kusolwa ukuthi ithole u-20% wamavoti.",
        reason="Natural phrase timing",
        semantic_equivalence_confirmed=True,
        protected_terms=("EFF",),
    )
    result = validate_translation_revision(revision)
    assert result["accepted"] is True
    assert result["wording_changed"] is True


def test_revision_rejects_lost_entity_number_or_uncertainty():
    revision = TranslationRevision(
        revision_id="rev-2",
        block_id="block_0003",
        phrase_id=None,
        before="Kusolwa ukuthi i-EFF ithole amavoti angu-20%.",
        after="Iqembu lithole amavoti angu-30%.",
        reason="Shorten delivery",
        semantic_equivalence_confirmed=True,
        protected_terms=("EFF",),
    )
    result = validate_translation_revision(revision)
    assert result["accepted"] is False
    assert {issue["code"] for issue in result["issues"]} == {
        "protected_terms_removed",
        "numbers_changed",
        "attribution_or_uncertainty_removed",
    }


def test_failed_cadence_creates_targeted_revision_request():
    artifact = build_translation_revision_request(
        job_id="job-1",
        translation_path="/tmp/transcript_zu_job-1.json",
        translation_sha256="a" * 64,
        policy=ProfessionalDubPolicy(),
        blocks=[
            {
                "block_id": "block_0001",
                "segment_ids": ["seg-00001"],
                "speaker_id": "SPEAKER_00",
                "translated_text": "Umbhalo.",
                "tts_text": "Umbhalo.",
                "cadence_qc": {
                    "passed": False,
                    "issues": [{"code": "whole_turn_rate_delta_excessive"}],
                },
                "cadence_plan": {
                    "phrases": [
                        {
                            "phrase_id": "block_0001_phrase_001",
                            "text": "Umbhalo.",
                            "start_ms": 0,
                            "end_ms": 1000,
                        }
                    ]
                },
            }
        ],
    )
    assert artifact["status"] == "revision_required"
    assert artifact["request_count"] == 1
    assert artifact["requests"][0]["segment_ids"] == ["seg-00001"]
    assert artifact["requests"][0]["renderer_rewrite_allowed"] is False
