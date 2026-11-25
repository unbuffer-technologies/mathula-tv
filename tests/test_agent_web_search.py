from types import SimpleNamespace

from mathula_tv import dub_agent
from mathula_tv.dub_agent import WEB_PER_TURN, ResponsesClient, build_system_prompt

CONFIG = SimpleNamespace(endpoint="https://example", deployment="d", api_key="k", reasoning_effort=None, timeout_seconds=30)

SEARCH = {"type": "web_search_call", "action": {"type": "search", "queries": ["Kenya v South Africa 2026 line-up", "Bafana v Harambee Stars squad"], "query": "Kenya v South Africa 2026 line-up"}}
OPEN = {"type": "web_search_call", "action": {"type": "open_page", "url": "https://example.org/match"}}
ANSWER = {"type": "message", "content": [{"type": "output_text", "text": "Ahoya", "annotations": [{"type": "url_citation", "title": "Match report", "url": "https://example.org/match"}]}]}


def _client(budget, output, monkeypatch):
    seen = []
    monkeypatch.setattr(dub_agent, "post_with_retry", lambda url, headers, body, **kw: (seen.append(body), {"output": output, "usage": {}})[1])
    client = ResponsesClient(CONFIG)
    client.web_budget = budget
    return client, seen


def test_the_web_tool_is_offered_only_with_a_budget_and_each_turn_is_capped(monkeypatch):
    client, seen = _client(0, [ANSWER], monkeypatch)
    client.complete([{"role": "user", "content": "hi"}], tools=[])
    assert not any(t.get("type") == "web_search" for t in seen[0].get("tools", [])) and "max_tool_calls" not in seen[0]
    client, seen = _client(10, [ANSWER], monkeypatch)
    client.complete([{"role": "user", "content": "hi"}], tools=[])
    assert {"type": "web_search"} in seen[0]["tools"] and seen[0]["max_tool_calls"] == WEB_PER_TURN
    client, seen = _client(2, [ANSWER], monkeypatch)
    client.complete([{"role": "user", "content": "hi"}], tools=[])
    assert seen[0]["max_tool_calls"] == 2                                              # less budget left than the per-turn cap


def test_lookups_are_counted_against_the_runs_budget_and_the_tool_is_withdrawn_when_it_is_used_up(monkeypatch):
    client, seen = _client(3, [SEARCH, OPEN, ANSWER], monkeypatch)
    client.complete([{"role": "user", "content": "hi"}], tools=[])
    assert client.web_used == 2
    client.complete([{"role": "user", "content": "again"}], tools=[])
    assert seen[1]["max_tool_calls"] == 1 and client.web_used == 4
    client.complete([{"role": "user", "content": "more"}], tools=[])
    assert not any(t.get("type") == "web_search" for t in seen[2].get("tools", []))      # budget used up (a turn may overshoot by what it already started): no more offered


def test_what_the_model_looked_up_and_cited_reaches_the_trace_and_not_the_replayed_history(monkeypatch):
    client, _ = _client(5, [SEARCH, OPEN, ANSWER], monkeypatch)
    message = client.complete([{"role": "user", "content": "hi"}], tools=[])["choices"][0]["message"]
    assert message["_web"][0] == {"action": "search", "queries": ["Kenya v South Africa 2026 line-up", "Bafana v Harambee Stars squad"]}
    assert message["_web"][1] == {"action": "open_page", "url": "https://example.org/match"} and message["_web"][2] == {"cites": [{"title": "Match report", "url": "https://example.org/match"}]}
    assert "_web" not in {k for k in ("role", "content", "tool_calls", "_raw") if k in message}          # the loop keeps only these keys of a message when it replays the history


def test_the_prompt_tells_the_agent_when_to_search_and_that_the_result_must_be_about_this_event():
    assert "WEB SEARCH" not in build_system_prompt(mode="training", budget="free")
    prompt = build_system_prompt(mode="training", budget="free", web=True)
    assert "WEB SEARCH" in prompt and "THIS event" in prompt and "another year is not this match" in prompt and "never for facts to add" in prompt
