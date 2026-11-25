"""An explicit context cache for the agent's growing conversation on Gemini (Vertex AI).

Why: Gemini's implicit caching is best-effort and, on the global endpoint, it is served from whichever backend answers.  On the same 31k-token prefix sent 12 times it gave 1 hit (prod-117/118 lost
a fifth of their input to it, and one run had ten misses in a row).  An explicit ``cachedContent`` is a reference the provider must honour: 12 of 12 in the same probe.

What it does, behind ``GeminiClient.complete`` and invisible to the agent loop:

* the cache holds the tools, the system prompt and the history so far WITHOUT the newest turn; a request then sends only what came after it (``cachedContent`` + the later turns);
* it is reused while the request still begins with exactly what was cached (the loop only appends, except at a compaction, which changes the front and makes the cache useless);
* creating one costs the whole prefix once at the normal input price, and every use after it is billed at the cached rate, so a new one is made only when it pays: the tokens paid beyond the
  cache each turn (the history grown since) add up as "rent", and when the rent paid (at the price difference between a cached and an uncached token) reaches the price of a fresh
  cache, it is bought.  That is the ski-rental rule: it never pays more than twice the best choice, whatever the run turns out to be, and it needs no tuning to the prompt size;
* any failure (the cache cannot be made, has expired, the model refuses it) falls back to the plain request for that call and counts as a failure; two failures turn the cache off for the run,
  so it can only ever cost one extra call, never stop a run.

``CACHE_READ_RATIO`` is the cached price as a share of the normal input price (Gemini 3 Flash: a tenth); it only sets when a new cache is bought, never whether the run works.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Mapping

CACHE_READ_RATIO = 0.1
MAX_FAILURES = 2                      # failures of any kind before the cache is turned off for the run
MIN_USES_BEFORE_REBUY = 2             # a cache is used at least this many times before a replacement is considered


def _digest(value: Any) -> str:
    return hashlib.sha1(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()[:16]


class ExplicitCache:
    def __init__(self, *, model_path: str, create: Callable[[dict[str, Any]], dict[str, Any]], delete: Callable[[str], None], ttl_seconds: int = 1800, min_tokens: int = 8192) -> None:
        self.model_path, self._create, self._delete = model_path, create, delete
        self.ttl_seconds, self.min_tokens = int(ttl_seconds), int(min_tokens)
        self.name: str | None = None
        self.tokens = 0                         # what the cache holds (the provider's count)
        self.header = ""                        # digest of tools + system instruction the cache was made with
        self.digests: list[str] = []            # one per cached content
        self.uses = 0
        self.baseline = 0                       # the uncached tokens of the first use: the newest turn, unavoidable whatever is cached
        self.rent = 0.0                         # tokens paid beyond the cache, at the price difference, since it was made
        self.failures = 0
        self.disabled = False
        self.created = 0
        self.creation_tokens = 0                # billed once each, at the normal input price (not part of the run's token ceiling)
        self.used_calls = 0
        self.fallbacks = 0
        self.events: list[dict[str, Any]] = []

    # -- the request ---------------------------------------------------------------------------------------------------------------------------------------
    def prepare(self, body: dict[str, Any], last_prompt_tokens: int) -> tuple[dict[str, Any], dict[str, Any]]:
        """The body to send for this request and a description of what happened (``state``: used | created | none)."""
        contents = body.get("contents") or []
        if self.disabled or len(contents) < 2:
            return body, {"state": "none"}
        header = _digest([body.get("tools"), body.get("systemInstruction")])
        every = [_digest(c) for c in contents[:-1]]                # the cacheable front of this request, one digest per content
        valid = self.name is not None and header == self.header and every[:len(self.digests)] == self.digests and len(every) >= len(self.digests)
        if valid and self.uses >= MIN_USES_BEFORE_REBUY and self.rent >= int(last_prompt_tokens):
            self._drop("rebought")                     # the rent paid has reached the price of a new cache
            valid = False
        elif not valid and self.name is not None:
            self._drop("front_changed")                # a compaction or an edited note: this cache can no longer be used
        if not valid:
            if int(last_prompt_tokens) < self.min_tokens:
                return body, {"state": "none"}
            cover = self._cover(contents)
            if cover < 1 or not self._make(body, header, contents[:cover], every[:cover]):
                return body, {"state": "none"}
        return self._request(body, len(self.digests)), {"state": "used", "cache_tokens": self.tokens, "name": self.name, "created_now": self.uses == 0}

    @staticmethod
    def _cover(contents: list[dict[str, Any]]) -> int:
        """How many leading contents the cache takes: all but the newest, and never ending on a model turn that is only text (the provider refuses a request that ends with one; a model turn
        that ends in a function call is fine, its result is the next turn)."""
        cover = len(contents) - 1
        while cover > 0 and contents[cover - 1].get("role") == "model" and not any("functionCall" in part for part in contents[cover - 1].get("parts") or []):
            cover -= 1
        return cover

    def _request(self, body: dict[str, Any], covered: int) -> dict[str, Any]:
        sent = {k: v for k, v in body.items() if k not in ("tools", "systemInstruction", "contents")}
        sent["cachedContent"] = self.name
        sent["contents"] = body["contents"][covered:]
        return sent

    def _make(self, body: dict[str, Any], header: str, prefix: list[dict[str, Any]], digests: list[str]) -> bool:
        payload: dict[str, Any] = {"model": self.model_path, "contents": prefix, "ttl": f"{self.ttl_seconds}s"}
        if body.get("systemInstruction"):
            payload["systemInstruction"] = body["systemInstruction"]
        if body.get("tools"):
            payload["tools"] = body["tools"]
        try:
            made = self._create(payload)
            name = str(made["name"])
            tokens = int((made.get("usageMetadata") or {}).get("totalTokenCount") or 0)
        except Exception as exc:                   # noqa: BLE001 - a cache that cannot be made must never stop the run
            self._fail(f"create: {type(exc).__name__}: {str(exc)[:240]}")
            return False
        self.name, self.tokens, self.header, self.digests = name, tokens, header, digests
        self.uses, self.baseline, self.rent = 0, 0, 0.0
        self.created += 1
        self.creation_tokens += tokens
        self.events.append({"event": "created", "tokens": tokens, "covers_contents": len(digests)})
        return True

    # -- what came back ------------------------------------------------------------------------------------------------------------------------------------
    def observe(self, usage: Mapping[str, Any]) -> dict[str, Any]:
        """After a call that used the cache: how much of the prompt it served, and the rent the rest costs."""
        prompt, cached = int(usage.get("prompt_tokens") or 0), int(usage.get("cached_tokens") or 0)
        self.used_calls += 1
        if cached < 0.5 * self.tokens:                # the provider did not serve what the cache holds: count it, and give it up after repeats
            self._fail(f"the cache served {cached} of its {self.tokens} tokens")
            self._drop("not_served")
            return {"served": cached}
        beyond = max(0, prompt - cached)
        if self.uses == 0:
            self.baseline = beyond
        self.uses += 1
        self.rent += max(0, beyond - self.baseline) * (1.0 - CACHE_READ_RATIO)
        return {"served": cached, "beyond": beyond, "rent": int(self.rent)}

    def request_failed(self, error: Exception) -> None:
        """A call that used the cache was refused (expired, not found, not accepted): forget it; the caller sends the plain request."""
        self.fallbacks += 1
        self._fail(f"use: {type(error).__name__}: {str(error)[:240]}")
        self._drop("refused")

    # -- bookkeeping ---------------------------------------------------------------------------------------------------------------------------------------
    def _fail(self, reason: str) -> None:
        self.failures += 1
        self.events.append({"event": "failed", "reason": reason, "failures": self.failures})
        if self.failures >= MAX_FAILURES:
            self.disabled = True
            self.events.append({"event": "disabled"})

    def _drop(self, why: str) -> None:
        name, self.name = self.name, None
        self.tokens, self.digests, self.header, self.uses, self.rent, self.baseline = 0, [], "", 0, 0.0, 0
        self.events.append({"event": "dropped", "why": why})
        if name:
            try:
                self._delete(name)
            except Exception:                      # noqa: BLE001 - the ttl removes it anyway
                pass

    def close(self) -> None:
        if self.name:
            self._drop("run_ended")

    def drain_events(self) -> list[dict[str, Any]]:
        events, self.events = self.events, []
        return events

    def report(self) -> dict[str, Any]:
        return {"created": self.created, "creation_tokens": self.creation_tokens, "used_calls": self.used_calls, "fallbacks": self.fallbacks, "failures": self.failures, "disabled": self.disabled}
