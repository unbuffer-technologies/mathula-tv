from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from mathula_tv.azure_web_research import AzureResponsesWebResearchProvider, AzureWebResearchError
from mathula_tv.gemini_web_research import GeminiWebResearchProvider, create_web_research_provider, grounding_evidence, resolve_redirect

SCHEMA = {"type": "object", "properties": {"schema_version": {"const": "research-v1"}, "corrections": {"type": "array"}, "unresolved_candidates": {"type": "array"}},
          "required": ["schema_version", "corrections", "unresolved_candidates"], "additionalProperties": False}
REDIRECT = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/abc"
PAGE = "https://example.org/madlanga-commission"


def grounded(text, usage=(900, 120, 50)):
    return {"candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP",
                            "groundingMetadata": {"webSearchQueries": ["madlanga commission"], "groundingChunks": [{"web": {"uri": REDIRECT, "title": "example.org"}}, {"web": {"uri": "https://b.example/x", "title": "b"}}]}}],
            "usageMetadata": {"promptTokenCount": usage[0], "candidatesTokenCount": usage[1], "cachedContentTokenCount": usage[2]}}


class FakeClient:
    def __init__(self, answers):
        self.answers, self.calls = list(answers), []

    def complete_raw(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        return self.answers.pop(0)


class FakeSession:
    def get(self, url, **kwargs):
        return SimpleNamespace(url=PAGE if "grounding-api-redirect" in url else url, close=lambda: None)


def provider(client):
    return GeminiWebResearchProvider(client, model="gemini-test", session=FakeSession())


VALID = json.dumps({"schema_version": "research-v1", "corrections": [], "unresolved_candidates": []})


def test_a_grounded_answer_returns_checked_data_with_the_pages_it_was_grounded_on():
    client = FakeClient([grounded("```json\n" + VALID + "\n```")])
    response = provider(client).research_json(system_prompt="Research the names.", payload={"candidate_entities": [{"name": "Madlanga"}]}, output_schema=SCHEMA)
    assert response.data["schema_version"] == "research-v1"
    meta = response.metadata
    assert meta.provider == "gemini-web-research" and meta.request_attempts == 1 and not meta.schema_repair_used and meta.web_search_tool == "googleSearch"
    assert meta.usage == {"input_tokens": 900, "output_tokens": 120, "cached_tokens": 50}
    urls = [e["url"] for e in meta.server_tool_evidence]
    assert PAGE in urls and REDIRECT in urls and "https://b.example/x" in urls                          # the page the redirect leads to counts as evidence, and so does the link
    call = client.calls[0]
    assert call["web_search"] is True and "Return exactly one JSON object" in call["messages"][0]["content"] and "research-v1" in call["messages"][0]["content"]


def test_an_answer_that_breaks_the_schema_gets_one_json_only_repair_without_a_new_search():
    client = FakeClient([grounded('{"schema_version": "research-v1", "corrections": "oops", "unresolved_candidates": []}'), grounded(VALID, usage=(300, 40, 0))])
    response = provider(client).research_json(system_prompt="s", payload={}, output_schema=SCHEMA)
    assert response.metadata.schema_repair_used and response.metadata.request_attempts == 2 and response.metadata.usage["input_tokens"] == 1200
    assert client.calls[1].get("web_search") in (None, False) and client.calls[1]["json_mode"] is True


def test_a_repair_that_still_fails_is_an_error_and_there_is_no_third_call():
    client = FakeClient([grounded("not json"), grounded("still not json")])
    with pytest.raises(AzureWebResearchError):
        provider(client).research_json(system_prompt="s", payload={}, output_schema=SCHEMA)
    assert len(client.calls) == 2


def test_a_failed_request_is_an_error_the_callers_can_degrade_on():
    class Broken:
        def complete_raw(self, *a, **k):
            raise RuntimeError("503")

    with pytest.raises(AzureWebResearchError, match="grounded research failed"):
        provider(Broken()).research_json(system_prompt="s", payload={}, output_schema=SCHEMA)


def test_redirect_links_are_followed_and_other_links_and_failures_are_left_alone():
    assert resolve_redirect(REDIRECT, session=FakeSession()) == PAGE
    assert resolve_redirect("https://example.org/a", session=FakeSession()) == "https://example.org/a"
    assert resolve_redirect(REDIRECT, session=SimpleNamespace(get=lambda *a, **k: (_ for _ in ()).throw(OSError("down")))) == REDIRECT
    assert grounding_evidence({"candidates": [{}]}) == ()


def test_the_factory_gives_gemini_only_when_asked_to():
    chosen = create_web_research_provider({"MATHULA_AI_PROVIDER": "gemini", "GEMINI_API_KEY": "k", "GEMINI_MODEL": "gemini-test"})
    assert isinstance(chosen, GeminiWebResearchProvider) and chosen.model == "gemini-test"
    azure = create_web_research_provider({"MATHULA_AI_PROVIDER": "azure", "AZURE_AI_KEY": "k", "AZURE_AI_ENDPOINT": "https://x.openai.azure.com", "AZURE_AI_DEPLOYMENT": "gpt"})
    assert isinstance(azure, AzureResponsesWebResearchProvider)
