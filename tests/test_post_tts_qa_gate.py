from __future__ import annotations

import inspect

from mathula_tv import cli
from mathula_tv.atomic_io import atomic_write_json
from mathula_tv.dub_mastering import (
    _contains_identity_surface,
    apply_english_entity_pronunciation_gate,
    build_post_tts_quality_gate,
)


def _mastering_report(*, ready: bool, blockers: int = 0) -> dict:
    return {
        "job_id": "job-qa",
        "state": "production_ready" if ready else "best_effort_completed",
        "stopped_reason": (
            "production_threshold_reached" if ready else "round_budget_exhausted"
        ),
        "best_round": 0,
        "best_production_score": 91.5 if ready else 74.0,
        "thresholds": {
            "minimum_production_score": 82.0,
            "maximum_aggregate_wer": 0.40,
        },
        "rounds": [
            {
                "round": 0,
                "production_ready": ready,
                "aggregate_word_error_rate": 0.12 if ready else 0.48,
                "production_blocker_count": blockers,
                "quality_dimensions": {
                    "intelligibility": 88.0 if ready else 52.0,
                    "protected_identity_recognition": 100.0,
                },
            }
        ],
    }


def test_post_tts_gate_authorizes_only_a_clean_production_round(tmp_path) -> None:
    report_path = tmp_path / "report.json"
    atomic_write_json(report_path, _mastering_report(ready=True))

    gate = build_post_tts_quality_gate(
        _mastering_report(ready=True),
        report_path=report_path,
    )

    assert gate["state"] == "passed"
    assert gate["publication_authorized"] is True
    assert gate["production_blocker_count"] == 0
    assert gate["mastering_report_sha256"]


def test_post_tts_gate_flags_but_does_not_block_on_contextual_grammar_qa_issues(tmp_path) -> None:
    # Explicit user direction: contextual grammar QA should only flag issues,
    # never terminate the process -- publication proceeds on a clean acoustic
    # round even when grammar QA reported blocking issues.
    report_path = tmp_path / "report.json"
    report = _mastering_report(ready=True)
    atomic_write_json(report_path, report)

    gate = build_post_tts_quality_gate(
        report,
        report_path=report_path,
        grammar_gate={
            "state": "flagged",
            "publication_authorized": True,
            "blocking_issue_count": 1,
            "warning_issue_count": 0,
        },
    )

    assert gate["publication_authorized"] is True
    assert gate["production_blocker_count"] == 1
    assert gate["grammar_blocker_count"] == 1
    assert "contextual_grammar_qa_flagged" in gate["reasons"]


def test_post_tts_gate_fails_closed_on_quality_blockers(tmp_path) -> None:
    report_path = tmp_path / "report.json"
    report = _mastering_report(ready=False, blockers=2)
    atomic_write_json(report_path, report)

    gate = build_post_tts_quality_gate(report, report_path=report_path)

    assert gate["state"] == "failed"
    assert gate["publication_authorized"] is False
    assert "production_threshold_not_reached" in gate["reasons"]
    assert "production_blockers_present" in gate["reasons"]


def test_dub_azure_enables_stt_qa_by_default(monkeypatch) -> None:
    monkeypatch.delenv("MATHULA_TV_POST_TTS_QA_ENABLED", raising=False)
    args = cli.parser().parse_args(["dub-azure", "job-qa"])

    assert args.qa_stt is True
    assert args.qa_stt_max_rounds == 1
    assert args.qa_minimum_production_score == 82.0
    assert args.qa_maximum_aggregate_wer == 0.40


def test_dub_azure_runs_stt_gate_before_publication() -> None:
    source = inspect.getsource(cli.main)
    branch = source.split('elif args.command == "dub-azure":', 1)[1].split(
        'elif args.command == "optimize-dub":', 1
    )[0]

    assert branch.index("qa_loop.run(") < branch.index("edit_tiktok_job(")
    assert "publication stopped" in branch


def test_dub_azure_runs_contextual_grammar_gate_before_synthesis() -> None:
    source = inspect.getsource(cli.main)
    branch = source.split('elif args.command == "dub-azure":', 1)[1].split(
        'elif args.command == "optimize-dub":', 1
    )[0]

    assert branch.index("ensure_job_grammar_quality(") < branch.index(
        "renderer.render("
    )
    assert "grammar_gate=grammar_gate" in branch


def test_english_stt_entity_gate_blocks_first_as_fast() -> None:
    audit = {
        "production_ready": True,
        "production_score": 100.0,
        "failed_block_count": 0,
        "production_blocker_count": 0,
        "quality_dimensions": {"protected_identity_recognition": 100.0},
        "blocks": [
            {
                "protected_entities": ["SA First Forum"],
                "reason_codes": [],
                "production_blockers": [],
                "passed": True,
                "production_gate_passed": True,
            }
        ],
    }

    result = apply_english_entity_pronunciation_gate(
        audit,
        protected_entities=("SA First Forum",),
        raw_english_stt={
            "combinedPhrases": [{"text": "lokho kuyingxenye ye-SA fast forum"}]
        },
    )

    assert result["passed"] is False
    assert result["failed_entities"] == ["SA First Forum"]
    assert audit["production_ready"] is False
    assert audit["production_blocker_count"] == 1
    assert audit["blocks"][0]["production_blockers"] == [
        "english_code_switch_pronunciation_failure"
    ]


def test_english_stt_entity_gate_accepts_exact_code_switch() -> None:
    audit = {
        "production_ready": True,
        "production_score": 90.0,
        "failed_block_count": 0,
        "production_blocker_count": 0,
        "quality_dimensions": {"protected_identity_recognition": 100.0},
        "blocks": [],
    }

    result = apply_english_entity_pronunciation_gate(
        audit,
        protected_entities=("SA First Forum",),
        raw_english_stt={
            "combinedPhrases": [{"text": "lokho kuyingxenye ye-SA First Forum"}]
        },
    )

    assert result["passed"] is True
    assert audit["production_ready"] is True


def test_english_stt_gate_skips_unspoken_entity_and_trusts_local_titled_name() -> None:
    audit = {
        "production_ready": True,
        "production_score": 100.0,
        "failed_block_count": 0,
        "production_blocker_count": 0,
        "quality_dimensions": {"protected_identity_recognition": 100.0},
        "blocks": [
            {
                "protected_entities": [
                    "Witness F",
                    "Major-General Lesetja Senona",
                ],
                "protected_entities_assessed": [
                    {
                        "entity": "Major-General Lesetja Senona",
                        "recognized": True,
                    }
                ],
                "reason_codes": [],
                "production_blockers": [],
                "passed": True,
                "production_gate_passed": True,
            }
        ],
    }

    result = apply_english_entity_pronunciation_gate(
        audit,
        protected_entities=("Witness F", "Major-General Lesetja Senona"),
        raw_english_stt={
            "combinedPhrases": [
                {"text": "umechocheneral le secha senona ayesaba ukumqamba"}
            ]
        },
    )

    assert result["passed"] is True
    assert result["failed_entities"] == []
    assert result["protected_entity_count"] == 1
    assert result["unassessable_entity_count"] == 1


def test_english_stt_gate_arbitrates_one_target_locale_code_switch_error() -> None:
    audit = {
        "production_ready": False,
        "production_score": 91.3,
        "aggregate_word_error_rate": 0.034,
        "failed_block_count": 1,
        "production_blocker_count": 1,
        "thresholds": {
            "minimum_production_score": 82.0,
            "maximum_aggregate_wer": 0.40,
        },
        "quality_dimensions": {"protected_identity_recognition": 50.0},
        "blocks": [
            {
                "protected_entities": ["The Hawks"],
                "protected_entities_assessed": [
                    {"entity": "The Hawks", "recognized": False}
                ],
                "protected_entities_missing": ["The Hawks"],
                "reason_codes": ["protected_entity_not_recognized"],
                "production_blockers": ["protected_entity_not_recognized"],
                "passed": False,
                "production_gate_passed": False,
            }
        ],
    }

    result = apply_english_entity_pronunciation_gate(
        audit,
        protected_entities=("The Hawks",),
        raw_english_stt={
            "combinedPhrases": [{"text": "Ngokwe-dha hawks izophenya"}]
        },
    )

    assert result["passed"] is True
    assert audit["production_ready"] is True
    assert audit["production_blocker_count"] == 0
    assert audit["quality_dimensions"]["protected_identity_recognition"] == 100.0
    assert audit["blocks"][0]["protected_entities_missing"] == []
    assert audit["blocks"][0]["protected_entities_assessed"][0][
        "match_policy"
    ] == "dual-locale-en-za-arbitrated-v1"


def test_english_stt_gate_accepts_reviewed_zulu_noun_class_surface() -> None:
    audit = {
        "production_ready": False,
        "production_score": 91.0,
        "aggregate_word_error_rate": 0.05,
        "failed_block_count": 1,
        "production_blocker_count": 1,
        "thresholds": {
            "minimum_production_score": 82.0,
            "maximum_aggregate_wer": 0.40,
        },
        "quality_dimensions": {"protected_identity_recognition": 50.0},
        "blocks": [
            {
                "protected_entities": ["The Hawks"],
                "protected_entities_assessed": [
                    {
                        "entity": "The Hawks",
                        "expected_surface_tokens": ["hawks"],
                        "recognized": False,
                    }
                ],
                "protected_entities_missing": ["The Hawks"],
                "reason_codes": ["protected_entity_not_recognized"],
                "production_blockers": ["protected_entity_not_recognized"],
                "passed": False,
                "production_gate_passed": False,
            }
        ],
    }

    result = apply_english_entity_pronunciation_gate(
        audit,
        protected_entities=("The Hawks",),
        raw_english_stt={
            "combinedPhrases": [{"text": "Amahauks ati umbiko uzosiza"}]
        },
    )

    assert result["passed"] is True
    assert result["failed_entities"] == []
    assert audit["production_ready"] is True
    assert audit["blocks"][0]["protected_entities_missing"] == []


def test_english_stt_gate_does_not_overrule_local_untitled_name_pass() -> None:
    audit = {
        "production_ready": True,
        "production_score": 100.0,
        "failed_block_count": 0,
        "production_blocker_count": 0,
        "quality_dimensions": {"protected_identity_recognition": 100.0},
        "blocks": [
            {
                "protected_entities": ["Cat Matlala"],
                "protected_entities_assessed": [
                    {"entity": "Cat Matlala", "recognized": True}
                ],
                "reason_codes": [],
                "production_blockers": [],
                "passed": True,
                "production_gate_passed": True,
            }
        ],
    }

    result = apply_english_entity_pronunciation_gate(
        audit,
        protected_entities=("Cat Matlala",),
        raw_english_stt={"combinedPhrases": [{"text": "uvusimuzi ket matlala"}]},
    )

    assert result["passed"] is True
    assert result["failed_entities"] == []
    assert audit["blocks"][0]["production_blockers"] == []


def test_local_entity_match_accepts_non_rhotic_and_fused_zulu_surfaces() -> None:
    assert _contains_identity_surface(
        "usagent faninyenkosi",
        ["sergeant", "fannie", "nkosi"],
    )
    assert _contains_identity_surface(
        "usuleiman karimganye",
        ["suleiman", "carrim"],
    )
    assert _contains_identity_surface(
        "uvusimuzi kats matlala",
        ["cat", "matlala"],
    )
