from __future__ import annotations

import json

from mathula_tv.ai_provider import (
    AZURE_FOUNDRY_CLAUDE_PROVIDER,
    AnthropicClaudeProvider,
    AnthropicConfig,
    StructuredAIRequest,
)
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
        self.calls = 0

    def post(self, _url: str, **_kwargs: object) -> Response:
        self.calls += 1
        return self.response


def _candidate(index: int, *, hook_strategy: object = "direct_quote") -> dict[str, object]:
    return {
        "candidate_id": f"candidate_{index}",
        "selected_block_id": f"block_{index:04d}",
        "evidence_block_ids": [f"block_{index:04d}"],
        "editorial_angle": "direct_quote",
        "hook_strategy": hook_strategy,
        "withheld_answer": "",
        "hook_text": f"Verified hook {index}",
        "accessible_hook_text": f"Verified hook {index}",
        "title_lead": {"type": "topic", "text": "Verified evidence"},
        "rationale": "The title is supported by the cited evidence.",
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
        "candidates": [
            _candidate(1, hook_strategy="direct_quote"),
            _candidate(2, hook_strategy="Decision Consequence"),
            _candidate(3, hook_strategy="new_model_label"),
        ],
        "primary_story_angle": "Verified testimony changes the decision.",
        "caption": "Verified testimony changes the decision.",
        "featured_person_names": [],
        "search_keywords": ["verified testimony"],
        "topic_hashtags": ["#Testimony", "#Evidence", "#Hearing", "#Decision"],
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


def test_advisory_hook_strategy_mismatches_are_recovered_before_validation() -> None:
    package = _package()
    response_body = {
        "id": "msg_editorial_enum_recovery",
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

    assert session.calls == 1
    assert [item["hook_strategy"] for item in response.data["candidates"]] == [
        "other",
        "direct_consequence",
        "other",
    ]
    assert all(
        item["editorial_angle"] == "direct_quote"
        for item in response.data["candidates"]
    )
    flags = response.data["human_review_flags"]
    assert "wire_normalized:candidates[0].hook_strategy:direct_quote->other" in flags
    assert (
        "wire_normalized:candidates[1].hook_strategy:Decision Consequence->direct_consequence"
        in flags
    )
    assert (
        "wire_normalized:candidates[2].hook_strategy:new_model_label->other" in flags
    )
    assert response.metadata.repair_count == 0


def test_missing_advisory_classifications_default_to_other() -> None:
    package = _package()
    del package["candidates"][0]["hook_strategy"]  # type: ignore[index]
    del package["candidates"][0]["editorial_angle"]  # type: ignore[index]

    normalized = _normalize_editorial_package_wire(package)

    assert normalized["candidates"][0]["hook_strategy"] == "other"
    assert normalized["candidates"][0]["editorial_angle"] == "other"
    assert (
        "wire_normalized:candidates[0].hook_strategy:missing->other"
        in normalized["human_review_flags"]
    )
    assert (
        "wire_normalized:candidates[0].editorial_angle:missing->other"
        in normalized["human_review_flags"]
    )
