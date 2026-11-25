import dataclasses

import pytest

from mathula_tv import dub_agent, gemini_client as gc
from mathula_tv.cache_watch import CacheWatch, fingerprint

TOOLS = [{"type": "function", "function": {"name": "a", "description": "d", "parameters": {"type": "object", "properties": {}}}}]


class FakeProvider:
    """A stand-in for Vertex: makes caches, serves them, and can be told to refuse."""

    def __init__(self, cache_tokens=10_000, base_prompt=12_000, grow=1_000):
        self.cache_tokens, self.base_prompt, self.grow = cache_tokens, base_prompt, grow
        self.creates, self.generates, self.deleted = [], [], []
        self.fail_create = False
        self.refuse_cache = False
        self.serve = True

    def post(self, url, headers, body, *, timeout, retries, sleep):
        if url.endswith("/cachedContents"):
            if self.fail_create:
                raise RuntimeError("model HTTP 400: cache too small")
            self.creates.append(body)
            return {"name": f"projects/1/locations/global/cachedContents/{len(self.creates)}", "usageMetadata": {"totalTokenCount": self.cache_tokens}}
        self.generates.append(body)
        cached = body.get("cachedContent")
        if cached and self.refuse_cache:
            raise RuntimeError("model HTTP 404: cached content not found")
        prompt = self.base_prompt + self.grow * len(self.generates)
        usage = {"promptTokenCount": prompt, "candidatesTokenCount": 5}
        if cached and self.serve:
            usage["cachedContentTokenCount"] = self.cache_tokens
        return {"candidates": [{"content": {"parts": [{"text": "ok"}]}}], "usageMetadata": usage}


class DeleteSpy:
    def __init__(self, provider):
        self.provider = provider

    def __call__(self, url, headers=None, timeout=None):
        self.provider.deleted.append(url.rsplit("/", 1)[-1])


def _client(monkeypatch, provider, **config):
    import requests

    monkeypatch.setattr(dub_agent, "post_with_retry", provider.post)
    monkeypatch.setattr(requests, "delete", DeleteSpy(provider))
    monkeypatch.setattr(gc._Token, "get", lambda self: "token")
    base = gc.GeminiConfig(project="proj", explicit_cache=True, cache_min_tokens=4096, **config)
    return gc.GeminiClient(base, sleep=lambda s: None)


def _history(turns):
    messages = [{"role": "system", "content": "You dub news."}, {"role": "user", "content": "Dub the clip."}]
    for i in range(turns):
        messages.append({"role": "assistant", "content": None, "tool_calls": [{"id": f"c{i}", "type": "function", "function": {"name": "a", "arguments": "{}"}}]})
        messages.append({"role": "tool", "tool_call_id": f"c{i}", "content": f"result {i}"})
    return messages


def test_the_cache_is_made_on_the_first_call_after_the_prompt_is_known_to_be_large_and_then_reused(monkeypatch):
    provider = FakeProvider()
    client = _client(monkeypatch, provider)
    client.complete(_history(1), tools=TOOLS)                                  # the first call: no prompt size known yet, so nothing is cached
    assert provider.creates == [] and "cachedContent" not in provider.generates[0] and client.last_explicit == {"state": "none"}
    client.complete(_history(2), tools=TOOLS)
    assert len(provider.creates) == 1
    made = provider.creates[0]
    assert made["systemInstruction"] and made["tools"] and made["ttl"] == "1800s" and made["model"].endswith("publishers/google/models/gemini-3.8-flash")
    assert len(made["contents"]) == 4                                          # everything but the newest turn
    second = provider.generates[1]
    assert second["cachedContent"] == "projects/1/locations/global/cachedContents/1" and "tools" not in second and "systemInstruction" not in second and len(second["contents"]) == 1
    client.complete(_history(3), tools=TOOLS)                                  # one more exchange: same cache, the turns after it are sent
    assert len(provider.creates) == 1 and provider.generates[2]["cachedContent"] and len(provider.generates[2]["contents"]) == 3
    assert client.cache.report()["used_calls"] == 2 and client.last_explicit["state"] == "used"


def test_what_the_provider_sees_is_still_the_whole_request_for_the_cache_watch(monkeypatch):
    provider = FakeProvider()
    client = _client(monkeypatch, provider)
    client.complete(_history(1), tools=TOOLS)
    client.complete(_history(2), tools=TOOLS)
    assert client.last_request["tools"] and client.last_request["system"] and len(client.last_request["contents"]) == 5
    watch = CacheWatch()
    row = watch.observe(2, fingerprint(client.last_request["contents"], client.last_request["tools"], client.last_request["system"]),
                        {"prompt_tokens": 14_000, "cached_tokens": 10_000}, client.last_explicit)
    watch.previous = None
    assert row["explicit"] == "used" and row["cause"] in (None, "first_request")
    served_nothing = CacheWatch()
    served_nothing.observe(1, fingerprint([], [], ""), {"prompt_tokens": 14_000})
    miss = served_nothing.observe(2, fingerprint([], [], ""), {"prompt_tokens": 14_000, "cached_tokens": 1_000}, {"state": "used", "cache_tokens": 10_000})
    assert miss["cause"] == "provider_miss" and miss["lost_tokens_est"] > 0


def test_a_compaction_changes_the_front_so_the_old_cache_is_deleted_and_a_new_one_is_made(monkeypatch):
    provider = FakeProvider()
    client = _client(monkeypatch, provider)
    client.complete(_history(1), tools=TOOLS)
    client.complete(_history(2), tools=TOOLS)
    compacted = _history(3)
    compacted[1] = {"role": "user", "content": "Dub the clip (older steps summarised)."}
    client.complete(compacted, tools=TOOLS)
    assert provider.deleted == ["1"] and len(provider.creates) == 2 and provider.generates[-1]["cachedContent"].endswith("/2")
    assert client.cache.report()["created"] == 2


def test_a_new_cache_is_bought_only_when_the_rent_paid_reaches_its_price(monkeypatch):
    provider = FakeProvider(cache_tokens=10_000, base_prompt=11_000, grow=2_000)
    client = _client(monkeypatch, provider)
    for turns in range(1, 12):
        client.complete(_history(turns), tools=TOOLS)
    created_at = [i for i, g in enumerate(provider.generates) if "cachedContent" in g and i > 0]
    assert 2 <= len(provider.creates) <= 4                                     # not on every turn (that would pay full price each time), and not never (the rent keeps growing)
    assert client.cache.report()["created"] == len(provider.creates) and created_at


def test_a_cache_that_cannot_be_made_falls_back_to_the_plain_request_and_turns_itself_off_after_two_failures(monkeypatch):
    provider = FakeProvider()
    provider.fail_create = True
    client = _client(monkeypatch, provider)
    for turns in range(1, 6):
        reply = client.complete(_history(turns), tools=TOOLS)
        assert reply["choices"][0]["message"]["content"] == "ok"
    assert all("cachedContent" not in g and g.get("tools") for g in provider.generates)
    assert client.cache.report()["disabled"] is True and client.cache.failures == 2


def test_a_cache_the_provider_refuses_is_dropped_and_the_same_call_is_sent_plain(monkeypatch):
    provider = FakeProvider()
    client = _client(monkeypatch, provider)
    client.complete(_history(1), tools=TOOLS)
    client.complete(_history(2), tools=TOOLS)
    provider.refuse_cache = True
    reply = client.complete(_history(3), tools=TOOLS)
    assert reply["choices"][0]["message"]["content"] == "ok" and "cachedContent" not in provider.generates[-1] and provider.generates[-1]["tools"]
    assert client.last_explicit == {"state": "fallback"} and client.cache.name is None and client.cache.report()["fallbacks"] == 1 and provider.deleted == ["1"]


def test_a_cache_the_provider_does_not_serve_is_given_up(monkeypatch):
    provider = FakeProvider()
    provider.serve = False
    client = _client(monkeypatch, provider)
    client.complete(_history(1), tools=TOOLS)
    client.complete(_history(2), tools=TOOLS)
    assert client.cache.name is None and client.cache.failures == 1


def test_calls_without_tools_and_clients_without_the_flag_never_use_a_cache(monkeypatch):
    provider = FakeProvider()
    client = _client(monkeypatch, provider)
    client.complete(_history(1), tools=TOOLS)
    client.complete([{"role": "user", "content": "just text"}])
    assert provider.creates == [] and "cachedContent" not in provider.generates[-1]
    plain = gc.GeminiClient(gc.GeminiConfig(api_key="k"), sleep=lambda s: None)
    assert plain.cache is None and plain.cache_report() is None and plain.drain_cache_events() == []


def test_explicit_caching_uses_the_project_path_and_a_bearer_token_even_when_a_key_is_set():
    keyed = gc.GeminiConfig(api_key="k", project="proj", explicit_cache=True)
    assert keyed.uses_project and "/projects/proj/locations/global/publishers/google/models/" in keyed.url()
    assert dataclasses.replace(keyed, explicit_cache=False).uses_project is False
    config = gc.gemini_config_from_environment({"GEMINI_API_KEY": "k", "GEMINI_EXPLICIT_CACHE": "1", "GOOGLE_CLOUD_PROJECT": "proj"})
    assert config.explicit_cache and config.uses_project
    assert gc.gemini_config_from_environment({"GEMINI_API_KEY": "k", "GEMINI_EXPLICIT_CACHE": "1"}).explicit_cache is False            # no project: nothing to cache in
    assert gc.gemini_config_from_environment({"GEMINI_API_KEY": "k", "GOOGLE_CLOUD_PROJECT": "proj"}).explicit_cache is False             # not asked for


def test_closing_the_client_releases_the_cache(monkeypatch):
    provider = FakeProvider()
    client = _client(monkeypatch, provider)
    client.complete(_history(1), tools=TOOLS)
    client.complete(_history(2), tools=TOOLS)
    client.close()
    assert provider.deleted == ["1"] and [e["event"] for e in client.drain_cache_events()][-1] == "dropped"


def test_a_cache_never_ends_on_a_model_turn_that_is_only_text(monkeypatch):
    provider = FakeProvider()
    client = _client(monkeypatch, provider)
    client.complete(_history(1), tools=TOOLS)
    history = _history(2) + [{"role": "assistant", "content": "I am thinking aloud."}, {"role": "user", "content": "Note from the harness."}]
    client.complete(history, tools=TOOLS)
    made = provider.creates[0]["contents"]
    assert made[-1]["role"] != "model" or any("functionCall" in p for p in made[-1]["parts"])
    assert len(made) == 5 and client.cache.report()["failures"] == 0                               # cut before the text-only model turn
    assert len(provider.generates[-1]["contents"]) == 2 and provider.generates[-1]["cachedContent"]
