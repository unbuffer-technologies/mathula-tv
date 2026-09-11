from __future__ import annotations

from types import SimpleNamespace

import pytest

from mathula_tv.grammar_quality import (
    _apply_canonical_name_repairs,
    apply_deterministic_grammar_repairs,
    deterministic_grammar_issues,
    run_contextual_grammar_pass,
)


def _request() -> dict:
    return {
        "target_locale": "zu-ZA",
        "units": [
            {
                "unit_id": "unit_0001",
                "speaker_id": "SPEAKER_00",
                "start_ms": 0,
                "end_ms": 4000,
                "source_text": "The court ordered treatment in prison.",
                "protected_spans": [],
            },
            {
                "unit_id": "unit_0002",
                "speaker_id": "SPEAKER_00",
                "start_ms": 4000,
                "end_ms": 8000,
                "source_text": "The Hawks say the report will clarify the matter.",
                "protected_spans": ["The Hawks"],
            },
        ],
    }


def _response() -> dict:
    return {
        "global_terminology": [],
        "units": [
            {
                **_request()["units"][0],
                "variants": [
                    {"variant_id": variant_id, "spoken_text": "Inkantolo iyalele ukwelashwa."}
                    for variant_id in ("natural", "concise", "compact")
                ],
            },
            {
                **_request()["units"][1],
                "variants": [
                    {
                        "variant_id": variant_id,
                        "spoken_text": "Ama-Hawks athi umbiko uzocacisa udaba.",
                    }
                    for variant_id in ("natural", "concise", "compact")
                ],
            },
        ],
    }


class _Metadata:
    def __init__(self, operation: str) -> None:
        self.operation = operation

    def to_dict(self) -> dict:
        return {"provider": "test", "operation": self.operation}


class _PassingProvider:
    def __init__(self) -> None:
        self.operations: list[str] = []

    def complete_structured(self, request):
        self.operations.append(request.operation)
        context = request.payload["complete_translated_timeline"]
        assert len(context) == 2
        hawks = context[1]["variants"][0]["spoken_text"]
        assert hawks.startswith("Ama-Hawks athi")
        groups = request.payload["connected_sentence_groups"]
        assert groups[1]["joined_translation_variants"][0]["spoken_text"].startswith(
            "Ama-Hawks athi"
        )
        whole = request.payload["comprehensive_translation_variants"]
        assert whole[0]["variant_id"] == "natural"
        assert whole[0]["spoken_text"] == (
            "Inkantolo iyalele ukwelashwa. "
            "Ama-Hawks athi umbiko uzocacisa udaba."
        )
        assert "[[unit_0001]]" in whole[0]["annotated_spoken_text"]
        contract = request.payload["speech_island_contract"]
        assert [item["unit_id"] for item in contract] == [
            "unit_0001",
            "unit_0002",
        ]
        assert all(
            item["content_may_move_to_another_island"] is False
            for item in contract
        )
        if request.operation == "contextual_grammar_editor":
            data = {"schema_version": "mathula-contextual-grammar-editor-v1", "edits": []}
        else:
            data = {
                "schema_version": "mathula-contextual-grammar-qa-v1",
                "passed": True,
                "issues": [],
            }
        request.validator(data)
        return SimpleNamespace(data=data, metadata=_Metadata(request.operation))


def test_contextual_editor_removes_prefix_from_complete_english_island() -> None:
    response = _response()
    for variant in response["units"][1]["variants"]:
        variant["spoken_text"] = "I-The Hawks ithi umbiko uzocacisa udaba."
    assert deterministic_grammar_issues(response)[0]["code"] == (
        "prefixed_complete_english_island"
    )

    edited, repairs = apply_deterministic_grammar_repairs(response)

    assert len(repairs) == 3
    assert deterministic_grammar_issues(edited) == []
    assert edited["units"][1]["variants"][0]["spoken_text"] == (
        "Ama-Hawks athi umbiko uzocacisa udaba."
    )


def test_deterministic_editor_repairs_unnatural_purpose_calque() -> None:
    response = _response()
    response["units"][0]["variants"][0]["spoken_text"] = (
        "Ngokusho kombuso, lokhu kwenzelwa izizathu zokuphepha."
    )

    assert deterministic_grammar_issues(response)[0]["code"] == (
        "unnatural_purpose_calque"
    )
    edited, repairs = apply_deterministic_grammar_repairs(response)

    assert len(repairs) == 1
    assert edited["units"][0]["variants"][0]["spoken_text"] == (
        "Ngokusho kombuso, lokhu kwenzelwa ukuphepha."
    )
    assert deterministic_grammar_issues(edited) == []


def test_canonical_name_continuity_repairs_later_stt_alias() -> None:
    response = _response()
    response["units"][0]["variants"][0]["spoken_text"] = (
        "UReagan Collis uvele enkantolo, kwathi kamuva kwakhulunywa ngoCollins."
    )

    edited, repairs = _apply_canonical_name_repairs(
        response,
        [
            {
                "entity_type": "person",
                "canonical_name": "Reagan Collis",
                "confidence": 0.93,
                "surface_forms": ["Reagan Collis", "Collis", "Collins"],
            }
        ],
    )

    assert len(repairs) == 1
    assert "ngoCollis" in edited["units"][0]["variants"][0]["spoken_text"]
    assert "Collins" not in edited["units"][0]["variants"][0]["spoken_text"]


def test_editor_and_independent_qa_receive_full_context_in_order() -> None:
    provider = _PassingProvider()

    result = run_contextual_grammar_pass(
        provider=provider,
        request=_request(),
        response=_response(),
    )

    assert result["passed"] is True
    assert result["blocking_issue_count"] == 0
    assert provider.operations == [
        "contextual_grammar_editor",
        "contextual_grammar_qa",
    ]


def test_default_is_exactly_one_editor_and_qa_call_even_when_qa_flags_an_issue() -> None:
    # Explicit user direction: QA should run once and report its verdict, not
    # loop spending further editor/QA calls trying to auto-fix what it flags.
    class OnceOnlyFailingProvider(_PassingProvider):
        def complete_structured(self, request):
            if request.operation == "contextual_grammar_editor":
                return super().complete_structured(request)
            self.operations.append(request.operation)
            data = {
                "schema_version": "mathula-contextual-grammar-qa-v1",
                "passed": False,
                "issues": [
                    {
                        "unit_id": "unit_0002",
                        "variant_id": "natural",
                        "severity": "blocking",
                        "code": "agreement",
                        "evidence": "remaining error",
                        "explanation": "Subject agreement is incorrect.",
                    }
                ],
            }
            request.validator(data)
            return SimpleNamespace(data=data, metadata=_Metadata(request.operation))

    provider = OnceOnlyFailingProvider()
    result = run_contextual_grammar_pass(
        provider=provider,
        request=_request(),
        response=_response(),
    )

    assert provider.operations == ["contextual_grammar_editor", "contextual_grammar_qa"]
    assert result["editor_result"]["round_count"] == 1
    assert result["passed"] is False
    assert result["blocking_issue_count"] == 1


def test_independent_qa_cannot_pass_with_a_blocking_issue() -> None:
    class InvalidProvider(_PassingProvider):
        def complete_structured(self, request):
            if request.operation == "contextual_grammar_editor":
                return super().complete_structured(request)
            data = {
                "schema_version": "mathula-contextual-grammar-qa-v1",
                "passed": True,
                "issues": [
                    {
                        "unit_id": "unit_0002",
                        "variant_id": "natural",
                        "severity": "blocking",
                        "code": "agreement",
                        "evidence": "remaining error",
                        "explanation": "Subject agreement is incorrect.",
                    }
                ],
            }
            request.validator(data)
            raise AssertionError("validator should reject the contradictory QA result")

    with pytest.raises(ValueError, match="blocking issues"):
        run_contextual_grammar_pass(
            provider=InvalidProvider(),
            request=_request(),
            response=_response(),
        )
