"""The harness's own record of the agent's working drafts, so the conversation does not have to be the record.

Until its first ``submit_dub`` the agent's drafts live only in its tool calls (the texts it measures, estimates and lays out), so the conversation could not be shortened
without losing them, and a conversation that cannot be shortened grows with every turn (prod-113: 51 turns, no submission, prompts of 155k tokens).  This ledger is built
from those tool calls by the harness, with nothing asked of the agent: the latest text tried for each time window, with its last measured or estimated length.  Compaction
can then drop the old steps and put the ledger in their place, whatever the agent does and whenever it submits.
"""
from __future__ import annotations

import json
from typing import Any, Mapping

LEDGER_NOTE_PREFIX = "[Working drafts so far"


class DraftLedger:
    def __init__(self, segments: list[Mapping[str, Any]] | None = None) -> None:
        self.entries: dict[str, dict[str, Any]] = {}
        self._windows = {str(s["segment_id"]): (int(s["start_ms"]), int(s["end_ms"]), str(s.get("speaker_id") or "")) for s in segments or []}
        self.submitted = False          # a full submission is the new record: drafts before it are no longer the state
        self._clock = 0                 # counts every draft written, so overlapping drafts can be told apart by which is newer

    def set_segments(self, segments: list[Mapping[str, Any]] | None) -> None:
        """The clip's segments once it is open (a timeline_fit item names a segment, and its window comes from here)."""
        self._windows = {str(s["segment_id"]): (int(s["start_ms"]), int(s["end_ms"]), str(s.get("speaker_id") or "")) for s in segments or []}

    # -- recording ------------------------------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _key(start: int, end: int) -> str:
        return f"{int(start)}-{int(end)}"

    def _put(self, start: Any, end: Any, text: Any, *, speaker: Any = None, **facts: Any) -> None:
        text = str(text or "").strip()
        if not text or start is None or end is None:
            return
        key = self._key(int(start), int(end))
        entry = self.entries.get(key)
        if entry is None or entry["text"] != text:          # a new text for a window forgets what was known about the old one
            entry = {"start_ms": int(start), "end_ms": int(end), "speaker": str(speaker or ""), "text": text}
            self.entries[key] = entry
        elif speaker:
            entry["speaker"] = str(speaker)
        self._clock += 1
        entry["seq"] = self._clock
        entry.update({k: v for k, v in facts.items() if v is not None})

    def as_submission(self, segments: list[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
        """The drafts as a submission: (items, uncovered segment ids).  Where drafts overlap (a sentence tried inside a unit, say) the newer one wins; a segment is covered when a
        kept draft's window holds it.  Only a complete cover can be submitted, so the caller submits when ``uncovered`` is empty."""
        ordered = sorted(segments, key=lambda s: (int(s["start_ms"]), str(s["segment_id"])))
        kept: list[dict[str, Any]] = []
        for entry in sorted(self.entries.values(), key=lambda e: -int(e.get("seq", 0))):
            if all(entry["end_ms"] <= k["start_ms"] or entry["start_ms"] >= k["end_ms"] for k in kept):
                kept.append(entry)
        items, covered = [], set()
        for number, entry in enumerate(sorted(kept, key=lambda e: e["start_ms"]), 1):
            inside = [s for s in ordered if int(s["start_ms"]) >= entry["start_ms"] - 50 and int(s["end_ms"]) <= entry["end_ms"] + 50]
            if not inside:
                continue
            covered.update(str(s["segment_id"]) for s in inside)
            items.append({"item_id": f"d{number}", "start_ms": entry["start_ms"], "end_ms": entry["end_ms"], "zulu_text": entry["text"],
                          "speaker_id": entry.get("speaker") or str(inside[0].get("speaker_id") or "SPEAKER_00"), "source_segment_ids": [str(s["segment_id"]) for s in inside]})
        return items, [str(s["segment_id"]) for s in ordered if str(s["segment_id"]) not in covered]

    def observe(self, tool: str, arguments: str, result: str) -> None:
        """One tool call and what it returned.  Never raises: a ledger that cannot read a call just misses it."""
        try:
            args = json.loads(arguments or "{}")
            out = json.loads(result or "{}")
        except (ValueError, TypeError):
            return
        try:
            if tool == "measure_tts":
                self._measured(args, out)
            elif tool == "measure_tts_batch":
                results = out.get("results") or []
                for position, request in enumerate(args.get("requests") or []):          # by position, so a batch that failed whole (no results) still keeps every draft it carried
                    self._measured(request, results[position] if position < len(results) else None)
            elif tool == "estimate_fit":
                results = out.get("results") or []
                for position, request in enumerate(args.get("requests") or []):
                    answer = results[position] if position < len(results) else None
                    estimated = int(answer["estimated_ms"]) if isinstance(answer, dict) and answer.get("estimated_ms") else None
                    self._put(request["start_ms"], request["end_ms"], request.get("text"), speaker=request.get("speaker_id"), estimated_ms=estimated, window_ms=int(request["end_ms"]) - int(request["start_ms"]))
            elif tool == "timeline_fit":
                estimated = {str(r.get("segment_id")): r for r in out.get("sentences") or [] if isinstance(r, dict)}
                for item in args.get("items") or []:
                    if item.get("start_ms") is not None and item.get("end_ms") is not None:          # one of the agent's own units
                        label = str(item.get("item_id") or f"{int(item['start_ms'])}-{int(item['end_ms'])}")
                        self._put(item["start_ms"], item["end_ms"], item.get("text"), speaker=item.get("speaker_id"), estimated_ms=(estimated.get(label) or {}).get("estimated_ms"),
                                  window_ms=int(item["end_ms"]) - int(item["start_ms"]))
                        continue
                    window = self._windows.get(str(item.get("segment_id")))
                    if window:
                        row = estimated.get(str(item.get("segment_id"))) or {}
                        self._put(window[0], window[1], item.get("text"), speaker=window[2], estimated_ms=row.get("estimated_ms"), window_ms=window[1] - window[0])
            elif tool == "submit_dub":
                if not args.get("partial"):
                    self.entries.clear()
                    self.submitted = True
                for item in args.get("items") or []:
                    self._put(item.get("start_ms"), item.get("end_ms"), item.get("zulu_text"), speaker=item.get("speaker_id"), status="submitted")
        except (KeyError, TypeError, ValueError):
            return

    def replay(self, messages: list[Mapping[str, Any]]) -> None:
        """Rebuild the ledger from a saved conversation: every tool call in it, with the result that answered it.  For a checkpoint that was saved before the ledger existed."""
        results = {str(m.get("tool_call_id")): str(m.get("content") or "") for m in messages if m.get("role") == "tool"}
        for message in messages:
            if message.get("role") == "assistant":
                for call in message.get("tool_calls") or []:
                    self.observe(call["function"]["name"], call["function"].get("arguments") or "{}", results.get(str(call.get("id")), ""))

    def _measured(self, request: Mapping[str, Any], answer: Any) -> None:
        if not isinstance(answer, dict) or not answer.get("measured_ms"):
            self._put(request["start_ms"], request["end_ms"], request.get("text"), speaker=request.get("speaker_id"))          # a call that failed (budget, validation) still carried a draft the agent wrote
            return
        self._put(request["start_ms"], request["end_ms"], request.get("text"), speaker=request.get("speaker_id"), measured_ms=int(answer["measured_ms"]),
                  window_ms=int(answer.get("window_ms") or int(request["end_ms"]) - int(request["start_ms"])), fits=bool(answer.get("fits_naturally")), direction=answer.get("direction"))

    # -- showing it -----------------------------------------------------------------------------------------------------------------------------------------
    def render(self) -> str:
        lines = []
        for entry in sorted(self.entries.values(), key=lambda e: (e["start_ms"], e["end_ms"])):
            window = entry.get("window_ms") or (entry["end_ms"] - entry["start_ms"])
            if entry.get("measured_ms"):
                gap = (entry["measured_ms"] - window) / max(window, 1) * 100.0
                state = f"measured {entry['measured_ms'] / 1000:.1f}s in a {window / 1000:.1f}s window ({'fits' if entry.get('fits') else ('over by' if gap > 0 else 'short by')}{'' if entry.get('fits') else f' {abs(gap):.0f}%'})"
            elif entry.get("estimated_ms"):
                state = f"only estimated: {entry['estimated_ms'] / 1000:.1f}s in a {window / 1000:.1f}s window, not measured"
            else:
                state = "not measured"
            if entry.get("status") == "submitted":
                state += ", submitted"
            lines.append(f"{entry['start_ms'] / 1000:.1f}-{entry['end_ms'] / 1000:.1f}s {entry['speaker'] or ''}: {entry['text']}  [{state}]")
        return "\n".join(lines)

    # -- the checkpoint -------------------------------------------------------------------------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {"entries": self.entries, "submitted": self.submitted, "clock": self._clock}

    def load(self, state: Mapping[str, Any] | None) -> None:
        if state:
            self.entries = {str(k): dict(v) for k, v in (state.get("entries") or {}).items()}
            self.submitted = bool(state.get("submitted"))
            self._clock = int(state.get("clock") or max((int(e.get("seq", 0)) for e in self.entries.values()), default=0))
