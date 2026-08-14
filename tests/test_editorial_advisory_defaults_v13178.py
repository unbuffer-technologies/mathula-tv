from __future__ import annotations

import json

import jsonschema
import pytest

from mathula_tv.ai_provider import (
    AZURE_FOUNDRY_CLAUDE_PROVIDER,
    AnthropicClaudeProvider,
    AnthropicConfig,
    StructuredAIRequest,
)
from mathula_tv.cli import _publication_ai_progress_payload
from mathula_tv.tiktok_editor import (
    EDITORIAL_PACKAGE_SCHEMA,
    _normalize_editorial_package_wire,
)


class Response:
    status_code = 200
    headers: dict[str, str] = {}

    def __init__(self, body: dict[str, object]):
        self.body = body
        self.text = json.dumps(body)

    def json(self) -> dict[str, object]:
        return self.body

    def close(self) -> None:
        return None


class Session:
    def __init__(self, response: Response):
        self.response = response
        self.calls: list[tuple[str, dict[str, object]]] = []

    def post(self, url: str, **kwargs: object) -> Response:
        self.calls.append((url, kwargs))
        return self.response


def _candidate(index: int) -> dict[str, object]:
    return {
        "candidate_id": f"candidate_{index}",
        "selected_block_id": f"block_{index:04d}",
        "evidence_block_ids": [f"block_{index:04d}"],
        "editorial_angle": "direct_quote",
        "hook_strategy": "direct_consequence",
        "withheld_answer": "",
        "hook_text": f"Verified hook number {index}",
        "accessible_hook_text": f"Verified hook number {index}",
        "title_lead": {"type": "topic", "text": "Verified evidence"},
        "rationale": "The title is supported by the selected evidence.",
        "confidence": 0.9,
        "grounded": True,
        "preserves_attribution": True,
        "no_sensationalism": True,
        "scores": {
            "visual_impact": 8,
            "curiosity": 8,
            "specificity": 8,
            "stakes": 8,
            "immediacy": 8,
            "audience_relevance": 8,
        },
        "human_review_flags": [],
    }


def _package() -> dict[str, object]:
    return {
        "opening_boundary_block_id": "block_0001",
        "opening_payoff_block_id": "block_0001",
        "opening_relationship": "standalone",
        "opening_boundary_rationale": "The first block contains the verified lead.",
        "candidates": [_candidate(1), _candidate(2), _candidate(3)],
        "primary_story_angle": "Verified evidence changes the decision.",
        "caption": "Buka ubufakazi obuqinisekisiwe.",
        "search_keywords": ["verified evidence"],
        "topic_hashtags": ["#Ubufakazi"],
        "human_review_flags": [],
    }


def _provider(session: Session) -> AnthropicClaudeProvider:
    return AnthropicClaudeProvider(
        AnthropicConfig(
            api_key="test-key",
            provider=AZURE_FOUNDRY_CLAUDE_PROVIDER,
            model="claude-opus-4-8",
            base_url="https://mathula-test.services.ai.azure.com/anthropic",
            max_retries=1,
            max_repairs=0,
            foundry_structured_outputs="disabled",
        ),
        session=session,
        sleep=lambda _delay: None,
    )


def test_missing_candidate_review_flags_are_restored_before_schema_validation() -> None:
    package = _package()
    del package["human_review_flags"]
    del package["candidates"][1]["human_review_flags"]  # type: ignore[index]
    response_body = {
        "id": "msg_editorial",
        "model": "claude-opus-4-8",
        "content": [{"type": "text", "text": json.dumps(package)}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 100, "output_tokens": 50},
    }
    session = Session(Response(response_body))

    response = _provider(session).complete_structured(
        StructuredAIRequest(
            operation="editorial_package",
            payload={"job_id": "job_test"},
            output_schema=EDITORIAL_PACKAGE_SCHEMA,
            prompt_version="editorial-test-v1",
            system_prompt="Return the requested editorial JSON.",
            response_schema_version="mathula-editorial-package-v1",
            max_repairs=0,
            normalizer=_normalize_editorial_package_wire,
        )
    )

    assert len(session.calls) == 1
    assert response.data["candidates"][1]["human_review_flags"] == []
    assert "wire_defaulted:human_review_flags" in response.data["human_review_flags"]
    assert (
        "wire_defaulted:candidates[1].human_review_flags"
        in response.data["human_review_flags"]
    )
    assert response.metadata.repair_count == 0
    assert (
        response.metadata.request_summary["schema_normalization"]
        ["application_normalizer_applied"]
        is True
    )


def test_editorial_wire_defaults_do_not_replace_required_scores() -> None:
    package = _package()
    del package["candidates"][0]["scores"]  # type: ignore[index]

    with pytest.raises(jsonschema.ValidationError, match="'scores' is a required property"):
        jsonschema.validate(
            _normalize_editorial_package_wire(package),
            EDITORIAL_PACKAGE_SCHEMA,
        )


def test_invalid_present_review_flags_still_fail_closed() -> None:
    package = _package()
    package["candidates"][0]["human_review_flags"] = None  # type: ignore[index]

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(
            _normalize_editorial_package_wire(package),
            EDITORIAL_PACKAGE_SCHEMA,
        )


def test_publication_progress_reports_local_metadata_recovery() -> None:
    progress = _publication_ai_progress_payload(
        {
            "event": "structured_output_normalized",
            "operation": "editorial_package",
            "dropped_property_count": 0,
            "dropped_property_paths": [],
            "application_normalizer_applied": True,
        }
    )

    assert progress is not None
    assert progress["stage"] == "publication_ai"
    assert progress["message"] == (
        "Editorial AI restored safely derivable omitted metadata; "
        "strict local validation continues"
    )
    assert progress["status"] == "warning"
    assert progress["provider_event"] == "structured_output_normalized"
