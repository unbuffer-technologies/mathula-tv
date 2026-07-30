from __future__ import annotations

import json

import pytest

from mathula_tv.azure_web_research import (
    AzureResponsesWebResearchProvider,
    AzureWebResearchConfig,
    AzureWebResearchError,
)


SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "corrections", "unresolved_candidates"],
    "properties": {
        "schema_version": {"const": "mathula-autocorrect-name-research-v1"},
        "corrections": {"type": "array"},
        "unresolved_candidates": {"type": "array"},
    },
}


class Response:
    headers = {}

    def __init__(self, status_code: int, body: dict | str):
        self.status_code = status_code
        self.body = body
        self.text = body if isinstance(body, str) else json.dumps(body)

    def json(self):
        if isinstance(self.body, str):
            raise ValueError("not json")
        return self.body


class Session:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.responses.pop(0)


def successful_envelope() -> dict:
    data = {
        "schema_version": "mathula-autocorrect-name-research-v1",
        "corrections": [
            {
                "segment_id": "seg-1",
                "raw_text": "Kanyaku",
                "canonical_text": "Kganyago",
                "entity_type": "person",
                "confidence": 0.99,
                "source_urls": [
                    "https://www.npa.gov.za/media/kganyago-khumalo",
                    "https://www.sabcnews.com/sabcnews/kganyago-khumalo/",
                ],
                "story_match_terms": ["Khumalo", "charges withdrawn"],
                "reason": "same story",
            }
        ],
        "unresolved_candidates": [],
    }
    return {
        "id": "resp-1",
        "status": "completed",
        "model": "gpt-test",
        "usage": {"input_tokens": 10, "output_tokens": 20},
        "output": [
            {
                "type": "web_search_call",
                "status": "completed",
                "action": {
                    "type": "search",
                    "query": "Kanyaku Khumalo charges",
                    "sources": [
                        {
                            "type": "url",
                            "url": "https://www.npa.gov.za/media/kganyago-khumalo",
                            "title": "NPA",
                        }
                    ],
                },
            },
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": json.dumps(data),
                        "annotations": [
                            {
                                "type": "url_citation",
                                "url": "https://www.sabcnews.com/sabcnews/kganyago-khumalo/",
                                "title": "SABC",
                            }
                        ],
                    }
                ],
            },
        ],
    }


def config(tool: str = "auto") -> AzureWebResearchConfig:
    return AzureWebResearchConfig(
        endpoint="https://resource.services.ai.azure.com",
        api_key="secret",
        deployment="gpt-test",
        timeout_seconds=30,
        max_retries=1,
        web_search_tool=tool,
    )


def test_uses_azure_responses_web_search_and_collects_sources() -> None:
    session = Session([Response(200, successful_envelope())])
    events = []
    provider = AzureResponsesWebResearchProvider(
        config(), session=session, progress_callback=events.append
    )

    result = provider.research_json(
        system_prompt="Return JSON.",
        payload={"candidate_name_spans": [{"raw_text": "Kanyaku"}]},
        output_schema=SCHEMA,
    )

    assert result.data["corrections"][0]["canonical_text"] == "Kganyago"
    assert result.metadata.web_search_tool == "web_search"
    assert [item["url"] for item in result.metadata.server_tool_evidence] == [
        "https://www.npa.gov.za/media/kganyago-khumalo",
        "https://www.sabcnews.com/sabcnews/kganyago-khumalo/",
    ]
    url, kwargs = session.calls[0]
    assert url == "https://resource.services.ai.azure.com/openai/v1/responses"
    assert "params" not in kwargs
    assert kwargs["json"]["tools"] == [{"type": "web_search"}]
    assert "tool_choice" not in kwargs["json"]
    assert "text" not in kwargs["json"]
    assert kwargs["json"]["include"] == ["web_search_call.action.sources"]
    assert "instructions" not in kwargs["json"]
    assert "max_output_tokens" not in kwargs["json"]
    assert set(kwargs["json"]) == {"model", "tools", "include", "input"}
    assert kwargs["headers"]["api-key"] == "secret"
    assert events[0]["event"] == "request_attempt_started"
    assert events[-1]["event"] == "request_completed"


def test_auto_falls_back_to_preview_tool_after_http_400() -> None:
    session = Session(
        [
            Response(
                400,
                {
                    "error": {
                        "type": "invalid_request_error",
                        "message": "Unsupported tool web_search",
                    }
                },
            ),
            Response(200, successful_envelope()),
        ]
    )
    provider = AzureResponsesWebResearchProvider(config(), session=session)

    result = provider.research_json(
        system_prompt="Return JSON.", payload={}, output_schema=SCHEMA
    )

    assert result.metadata.web_search_tool == "web_search_preview"
    assert session.calls[0][1]["json"]["tools"] == [{"type": "web_search"}]
    assert session.calls[1][1]["json"]["tools"] == [
        {"type": "web_search_preview"}
    ]


def test_failure_exposes_safe_azure_error_body() -> None:
    session = Session(
        [
            Response(400, {"error": {"message": "Web search is disabled"}}),
            Response(400, {"error": {"message": "Preview is unavailable"}}),
        ]
    )
    events = []
    provider = AzureResponsesWebResearchProvider(
        config(), session=session, progress_callback=events.append
    )

    with pytest.raises(AzureWebResearchError) as raised:
        provider.research_json(
            system_prompt="Return JSON.", payload={}, output_schema=SCHEMA
        )

    message = str(raised.value)
    assert "Web search is disabled" in message
    assert "Preview is unavailable" in message
    assert "secret" not in message
    failed_events = [
        event for event in events if event.get("event") == "request_attempt_failed"
    ]
    assert len(failed_events) == 2
    assert "Web search is disabled" in failed_events[0]["error"]
    assert events[-1]["event"] == "request_failed"


def test_environment_uses_existing_mathula_azure_variables() -> None:
    value = AzureWebResearchConfig.from_environment(
        {
            "AZURE_AI_ENDPOINT": "https://resource.services.ai.azure.com/",
            "AZURE_AI_KEY": "key",
            "AZURE_AI_DEPLOYMENT": "gpt-5.4",
        }
    )
    assert value.endpoint == "https://resource.services.ai.azure.com"
    assert value.deployment == "gpt-5.4"
    assert value.web_search_tool == "auto"


def envelope_with_output(data: dict) -> dict:
    envelope = successful_envelope()
    envelope["output"][-1]["content"][0]["text"] = json.dumps(data)
    return envelope


def test_normalizes_observed_azure_wrapper_fields_before_validation() -> None:
    canonical = json.loads(
        successful_envelope()["output"][-1]["content"][0]["text"]
    )
    wrapped = {
        **canonical,
        "resolved_corrections": canonical["corrections"],
        "sources": [{"url": "https://model-output.example/not-trusted"}],
        "batch": {"index": 1, "count": 7},
        "request_contract": "echoed-by-model",
    }
    events = []
    provider = AzureResponsesWebResearchProvider(
        config(),
        session=Session([Response(200, envelope_with_output(wrapped))]),
        progress_callback=events.append,
    )

    result = provider.research_json(
        system_prompt="Return JSON.", payload={}, output_schema=SCHEMA
    )

    assert set(result.data) == {
        "schema_version",
        "corrections",
        "unresolved_candidates",
    }
    assert result.data["corrections"][0]["canonical_text"] == "Kganyago"
    normalized = [event for event in events if event.get("event") == "response_normalized"]
    assert len(normalized) == 1
    assert set(normalized[0]["dropped_fields"]) == {
        "batch",
        "request_contract",
        "resolved_corrections",
        "sources",
    }


def test_maps_resolved_corrections_alias_and_supplies_transport_defaults() -> None:
    canonical = json.loads(
        successful_envelope()["output"][-1]["content"][0]["text"]
    )
    alias_only = {
        "resolved_corrections": canonical["corrections"],
        "sources": ["https://www.npa.gov.za/media/kganyago-khumalo"],
    }
    provider = AzureResponsesWebResearchProvider(
        config(), session=Session([Response(200, envelope_with_output(alias_only))])
    )

    result = provider.research_json(
        system_prompt="Return JSON.", payload={}, output_schema=SCHEMA
    )

    assert result.data == {
        "schema_version": "mathula-autocorrect-name-research-v1",
        "corrections": canonical["corrections"],
        "unresolved_candidates": [],
    }


def test_canonicalizes_model_supplied_wrong_schema_version() -> None:
    canonical = json.loads(
        successful_envelope()["output"][-1]["content"][0]["text"]
    )
    canonical["schema_version"] = "mathula-autocorrect-name-research-batch-v1"
    events = []
    session = Session([Response(200, envelope_with_output(canonical))])
    provider = AzureResponsesWebResearchProvider(
        config(), session=session, progress_callback=events.append
    )

    result = provider.research_json(
        system_prompt="Return JSON.", payload={}, output_schema=SCHEMA
    )

    assert result.data["schema_version"] == "mathula-autocorrect-name-research-v1"
    normalized = [event for event in events if event.get("event") == "response_normalized"]
    assert len(normalized) == 1
    assert normalized[0]["canonicalized_fields"] == ["schema_version"]
    assert len(session.calls) == 1


def test_does_not_repeat_search_with_preview_after_2xx_schema_failure() -> None:
    invalid = {
        "schema_version": "mathula-autocorrect-name-research-v1",
        "corrections": "not-an-array",
        "unresolved_candidates": [],
    }
    bad_repair = envelope_with_output(invalid)
    session = Session(
        [
            Response(200, envelope_with_output(invalid)),
            Response(200, bad_repair),
        ]
    )
    provider = AzureResponsesWebResearchProvider(config(), session=session)

    with pytest.raises(AzureWebResearchError):
        provider.research_json(
            system_prompt="Return JSON.", payload={}, output_schema=SCHEMA
        )

    assert len(session.calls) == 2
    assert session.calls[0][1]["json"]["tools"] == [{"type": "web_search"}]
    assert "tools" not in session.calls[1][1]["json"]

DETAILED_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "corrections", "unresolved_candidates"],
    "properties": {
        "schema_version": {"const": "mathula-autocorrect-name-research-v1"},
        "corrections": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "segment_id",
                    "raw_text",
                    "canonical_text",
                    "entity_type",
                    "confidence",
                    "source_urls",
                    "story_match_terms",
                    "reason",
                ],
                "properties": {
                    "segment_id": {"type": "string"},
                    "raw_text": {"type": "string"},
                    "canonical_text": {"type": "string"},
                    "entity_type": {
                        "enum": ["person", "organisation", "place", "case", "other_name"]
                    },
                    "confidence": {"type": "number"},
                    "source_urls": {"type": "array", "items": {"type": "string"}},
                    "story_match_terms": {"type": "array", "items": {"type": "string"}},
                    "reason": {"type": "string"},
                },
            },
        },
        "unresolved_candidates": {"type": "array"},
    },
}


def test_normalizes_common_correction_field_aliases() -> None:
    alias_data = {
        "schema_version": "wrong-transport-label",
        "resolved_corrections": [
            {
                "segment": "seg-1",
                "heard_text": "Kaiser Kanyaku",
                "canonical_replacement": "Kaizer Kganyago",
                "type": "Person",
                "confidence_score": "99%",
                "evidence": [
                    {"url": "https://www.npa.gov.za/media/kganyago-khumalo"}
                ],
                "same_story_clues": "Khumalo charges",
                "explanation": "Official NPA spokesperson in the same story",
            }
        ],
    }
    provider = AzureResponsesWebResearchProvider(
        config(), session=Session([Response(200, envelope_with_output(alias_data))])
    )

    result = provider.research_json(
        system_prompt="Return JSON.",
        payload={
            "candidate_name_spans": [
                {"segment_id": "seg-1", "raw_text": "Kaiser Kanyaku"}
            ]
        },
        output_schema=DETAILED_SCHEMA,
    )

    correction = result.data["corrections"][0]
    assert correction == {
        "segment_id": "seg-1",
        "raw_text": "Kaiser Kanyaku",
        "canonical_text": "Kaizer Kganyago",
        "entity_type": "person",
        "confidence": 0.99,
        "source_urls": ["https://www.npa.gov.za/media/kganyago-khumalo"],
        "story_match_terms": ["Khumalo charges"],
        "reason": "Official NPA spokesperson in the same story",
    }


def test_persists_failed_response_for_diagnosis_without_reusing_it(tmp_path) -> None:
    invalid = {
        "schema_version": "mathula-autocorrect-name-research-v1",
        "corrections": [
            {
                "segment_id": "seg-1",
                "raw_text": "Kanyaku",
                "entity_type": "person",
                "confidence": 0.99,
                "source_urls": [],
                "story_match_terms": ["Khumalo"],
                "reason": "Missing canonical text on purpose",
            }
        ],
        "unresolved_candidates": [],
    }
    events = []
    session = Session(
        [
            Response(200, envelope_with_output(invalid)),
            Response(200, envelope_with_output(invalid)),
        ]
    )
    provider = AzureResponsesWebResearchProvider(
        config(), session=session, progress_callback=events.append
    )

    with pytest.raises(AzureWebResearchError) as raised:
        provider.research_json(
            system_prompt="Return JSON.",
            payload={
                "candidate_name_spans": [
                    {"segment_id": "seg-1", "raw_text": "Kanyaku"}
                ]
            },
            output_schema=DETAILED_SCHEMA,
            diagnostic_dir=tmp_path,
            diagnostic_context={"job_id": "job-1", "batch_index": 1},
        )

    assert len(session.calls) == 2
    assert len(raised.value.diagnostic_paths) >= 1
    paths = sorted(tmp_path.glob("*.json"))
    assert len(paths) >= 2
    path = paths[0]
    assert path.is_file()
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["outcome"] == "output_schema_validation_repair_required"
    assert record["reuse_policy"] == "diagnostic_only_never_used_as_research_result"
    assert record["diagnostic_context"] == {"job_id": "job-1", "batch_index": 1}
    assert record["parsed_output"]["corrections"][0]["raw_text"] == "Kanyaku"
    assert record["normalized_output"]["corrections"][0].get("canonical_text") is None
    assert record["validation_error"]["path"] == ["corrections", 0]
    assert "secret" not in path.read_text(encoding="utf-8")
    rejected = [event for event in events if event.get("event") == "response_validation_failed"]
    assert rejected[0]["diagnostic_path"] == str(path)



def compact_proposal_envelope() -> dict:
    envelope = successful_envelope()
    proposal = {
        "schema_version": "1.0",
        "corrections": [
            {
                "segment_id": "seg-1",
                "raw_text": "Kaiser Kanyaku",
                "canonical_replacement": "Kaizer Kganyago",
                "evidence": [
                    "https://www.npa.gov.za/media/kganyago-khumalo",
                    "https://www.sabcnews.com/sabcnews/kganyago-khumalo/",
                ],
            }
        ],
        "unresolved_candidates": [],
    }
    envelope["output"][-1]["content"][0]["text"] = json.dumps(proposal)
    return envelope


def repaired_envelope() -> dict:
    envelope = successful_envelope()
    envelope["id"] = "resp-repair-1"
    envelope["output"] = [envelope["output"][-1]]
    return envelope


def test_repairs_compact_grounded_proposal_without_repeating_web_search(tmp_path) -> None:
    events = []
    session = Session(
        [
            Response(200, compact_proposal_envelope()),
            Response(200, repaired_envelope()),
        ]
    )
    provider = AzureResponsesWebResearchProvider(
        config(), session=session, progress_callback=events.append
    )

    result = provider.research_json(
        system_prompt="Return JSON.",
        payload={
            "candidate_name_spans": [
                {
                    "segment_id": "seg-1",
                    "raw_text": "Kaiser Kanyaku",
                    "source_text": "NPA spokesperson Kaiser Kanyaku joins us.",
                }
            ],
            "source_metadata": {"title": "Kaizer Kganyago explains"},
            "research_profile": {
                "name": "madlanga_commission",
                "authoritative_domains": ["npa.gov.za"],
            },
        },
        output_schema=DETAILED_SCHEMA,
        diagnostic_dir=tmp_path,
    )

    assert result.data["corrections"][0]["canonical_text"] == "Kganyago"
    assert result.metadata.schema_repair_used is True
    assert result.metadata.schema_repair_response_id == "resp-repair-1"
    assert len(session.calls) == 2
    assert session.calls[0][1]["json"]["tools"] == [{"type": "web_search"}]
    assert session.calls[0][1]["json"]["include"] == [
        "web_search_call.action.sources"
    ]
    repair_body = session.calls[1][1]["json"]
    assert "tools" not in repair_body
    assert repair_body["text"]["format"]["type"] == "json_schema"
    assert repair_body["text"]["format"]["strict"] is True
    schema = repair_body["text"]["format"]["schema"]
    assert schema["properties"]["schema_version"]["enum"] == [
        "mathula-autocorrect-name-research-v1"
    ]
    schema_text = json.dumps(schema)
    assert "minLength" not in schema_text
    assert "maxItems" not in schema_text
    assert "minimum" not in schema_text
    assert "maximum" not in schema_text
    assert any(event["event"] == "response_schema_repair_required" for event in events)
    assert any(event["event"] == "schema_repair_started" for event in events)
    assert any(event["event"] == "schema_repair_completed" for event in events)
    diagnostics = list(tmp_path.glob("*.json"))
    assert diagnostics
    record = json.loads(diagnostics[0].read_text(encoding="utf-8"))
    assert record["parsed_output"]["corrections"][0]["canonical_replacement"] == "Kaizer Kganyago"
    assert record["reuse_policy"] == "diagnostic_only_never_used_as_research_result"


def test_schema_repair_failure_does_not_repeat_web_search(tmp_path) -> None:
    bad_repair = repaired_envelope()
    bad_repair["output"][0]["content"][0]["text"] = json.dumps(
        {
            "schema_version": "mathula-autocorrect-name-research-v1",
            "corrections": "invalid",
            "unresolved_candidates": [],
        }
    )
    session = Session(
        [
            Response(200, compact_proposal_envelope()),
            Response(200, bad_repair),
        ]
    )
    provider = AzureResponsesWebResearchProvider(config(), session=session)

    with pytest.raises(AzureWebResearchError):
        provider.research_json(
            system_prompt="Return JSON.",
            payload={
                "candidate_name_spans": [
                    {"segment_id": "seg-1", "raw_text": "Kaiser Kanyaku"}
                ]
            },
            output_schema=DETAILED_SCHEMA,
            diagnostic_dir=tmp_path,
        )

    assert len(session.calls) == 2
    assert session.calls[0][1]["json"]["tools"] == [{"type": "web_search"}]
    assert "tools" not in session.calls[1][1]["json"]
