"""Why a turn's input was or was not served from the provider's cache.

A provider cache is a byte-exact prefix match: tool declarations and system prompt first, then the messages in order, and every token after the first changed one is billed in full.  The agent loop
pays for the whole conversation again each turn, so what decides the cost is how much of the previous turn's request is still the front of this turn's.  This module fingerprints every request the
loop sends (tools, system prompt, each message), compares it with the one before, and joins that with the cached-token count the provider reports, so a turn that missed says WHY:

* ``tools_changed`` / ``system_changed``: the header differs, so nothing after it could be reused;
* ``history_rewritten``: an earlier message was removed or edited (compaction, a note placed before the work); only the messages before the first change could be reused;
* ``provider_miss``: the front of the request was identical and the provider did not serve it anyway (its caching is best-effort), which no change here can fix.

It records and decides nothing.  The numbers go to the trace (``cache`` events) and the run summary, and are what a header or compaction change is measured against.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

MISS_SHARE = 0.5          # served less than this fraction of what an unchanged front should have given: a provider miss
MIN_CACHEABLE_TOKENS = 4096      # below this a provider does not cache at all (the strictest published minimum), so a low hit there is not a miss


def _digest(value: Any) -> str:
    return hashlib.sha1(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()[:10]


def fingerprint(items: Sequence[Mapping[str, Any]], tools: Any, system: Any) -> dict[str, Any]:
    """The tools, the system prompt and every message of one request, each reduced to a short hash and a size."""
    rows = [json.dumps(m, sort_keys=True, ensure_ascii=False, default=str) for m in items]
    return {"tools": _digest(tools or []), "system": _digest(system or ""),
            "messages": [hashlib.sha1(r.encode("utf-8")).hexdigest()[:10] for r in rows], "sizes": [len(r) for r in rows]}


def request_fingerprint(client: Any, messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]] | None) -> dict[str, Any]:
    """The fingerprint of the request the client actually sent (what the provider sees, tool additions the client makes included); the chat messages when the client does not say."""
    sent = getattr(client, "last_request", None)
    if isinstance(sent, Mapping):
        return fingerprint(sent.get("contents") or [], sent.get("tools"), sent.get("system"))
    return fingerprint(list(messages[1:]), tools, messages[0] if messages else "")


class CacheWatch:
    def __init__(self) -> None:
        self.previous: dict[str, Any] | None = None
        self.rows: list[dict[str, Any]] = []

    def observe(self, turn: int, current: Mapping[str, Any], usage: Mapping[str, Any] | None, explicit: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Join this turn's request fingerprint (taken BEFORE the call) with the usage the provider reported for it.  ``explicit``: what the client's own cache did this turn
        (see gemini_cache): a request served from one is expected to be served exactly what the cache holds, so a miss is judged against that, not against the unchanged front."""
        usage = usage or {}
        prompt, cached = int(usage.get("prompt_tokens") or 0), int(usage.get("cached_tokens") or 0)
        row: dict[str, Any] = {"turn": turn, "prompt_tokens": prompt, "cached_tokens": cached, "hit_pct": min(100.0, round(100.0 * cached / prompt, 1)) if prompt else 0.0,          # an explicit cache can hold more than the prompt counts (it counts the model's own thinking parts)
                               "thought_tokens": int(usage.get("thoughts_tokens") or 0), "tools": current["tools"], "system": current["system"], "messages": len(current["messages"])}
        before = self.previous
        cause: str | None = None
        expected = 0.0
        if before is None:
            cause = "first_request"
        else:
            shared = 0
            for a, b in zip(before["messages"], current["messages"]):
                if a != b:
                    break
                shared += 1
            total_chars = sum(current["sizes"]) or 1
            expected = sum(current["sizes"][:shared]) / total_chars          # the share of this request that is the front of the last one
            row["kept_messages"], row["kept_pct"] = shared, round(100.0 * expected, 1)
            if current["tools"] != before["tools"]:
                cause, expected = "tools_changed", 0.0
            elif current["system"] != before["system"]:
                cause, expected = "system_changed", 0.0
            elif shared < len(before["messages"]):
                cause = "history_rewritten"
                row["first_changed_message"] = shared
            elif prompt >= MIN_CACHEABLE_TOKENS and cached < MISS_SHARE * expected * prompt:
                cause = "provider_miss"
        if explicit and explicit.get("state") in ("used", "fallback"):
            row["explicit"] = explicit["state"]
            if explicit["state"] == "used" and prompt:
                held = int(explicit.get("cache_tokens") or 0)
                expected = held / prompt
                cause = "provider_miss" if cached < MISS_SHARE * held else (cause if cause in ("tools_changed", "system_changed", "history_rewritten") else None)
        row["expected_hit_pct"] = round(100.0 * expected, 1)
        row["lost_tokens_est"] = max(0, int(expected * prompt) - cached) if cause not in (None, "first_request") else 0
        row["cause"] = cause
        self.previous = {k: current[k] for k in ("tools", "system", "messages", "sizes")}
        self.rows.append(row)
        return row

    def report(self) -> dict[str, Any]:
        prompt, cached = sum(r["prompt_tokens"] for r in self.rows), sum(r["cached_tokens"] for r in self.rows)
        causes: dict[str, int] = {}
        for r in self.rows:
            if r["cause"]:
                causes[r["cause"]] = causes.get(r["cause"], 0) + 1
        return {"turns": len(self.rows), "prompt_tokens": prompt, "cached_tokens": cached, "hit_pct": round(100.0 * cached / prompt, 1) if prompt else 0.0,
                "thought_tokens": sum(r["thought_tokens"] for r in self.rows), "misses_by_cause": causes,
                "lost_tokens_est": sum(r["lost_tokens_est"] for r in self.rows),
                "header_hashes": sorted({r["tools"] + "/" + r["system"] for r in self.rows})}

    def to_dict(self) -> dict[str, Any]:
        return {"previous": self.previous, "rows": self.rows}

    def load(self, state: Mapping[str, Any] | None) -> None:
        if state:
            self.previous = state.get("previous")
            self.rows = list(state.get("rows") or [])
