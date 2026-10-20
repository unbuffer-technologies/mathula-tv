"""The reflector: turn a batch of runs into proposed playbook edits.

One model call reads a compact digest of each run (what the agent said and did, every submission's
score, every rejection, every pushback) beside the current playbook, contrasts runs that met the bar
with runs that did not, and proposes at most a few small edits with the evidence for each. It must
separate three causes -- a *strategy* lesson (the playbook can fix it), an *environment* trap (a harness
design problem the agent could not have avoided; reported, not "fixed" with a lesson) and plain *noise*.

Everything around the model call is deterministic code: building the digest, validating the reply,
and (in ``playbook.apply_edits``) applying edits. The model never writes the playbook directly.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

DIGEST_MAX_CHARS = 5500
_ALWAYS_KEEP = ("REJECTED", "PUSHBACK", "NUDGE", "SAYS-NO-TOOLS", "END")

REFLECTOR_SYSTEM_PROMPT = """\
You improve the strategy notes (the "playbook") of an AI agent that dubs English news clips into isiZulu.
The agent works inside fixed rules enforced by a harness (a scorer, a bar for "done", budgets, a timing lock).
Rules are not yours to change. You only improve the playbook: short strategy notes about how to work inside
the rules.

You are given the current playbook and a digest of several runs of the same agent on the same clip. Some met
the bar, some did not. Work like this:
1. For each run, say what happened and the root cause: "strategy" (a different way of working would have
   avoided it), "environment" (a harness trap the agent could not reasonably have avoided -- report it, do not
   write a lesson for it), or "noise" (luck; nothing to learn).
2. Contrast runs that met the bar with runs that did not: what did the good runs do that the bad ones did not?
3. Propose 0 to 3 edits to the playbook. Prefer revising or removing a note that caused a failure over adding a
   new one. Every edit must cite specific evidence from the digest (which run, what happened).

Rules for edits:
- Each note is one imperative sentence of at most 240 characters, about method, general enough to hold for ANY
  clip. Never mention segment, unit or clip identifiers or specific numbers from these runs.
- Never tell the agent to ignore, skip or relax a harness rule (the bar, budgets, the timing lock, the scorer).
- If the runs show no strategy lesson, propose no edits. Do not invent lessons to look useful.

Reply with JSON only:
{"diagnosis":[{"run":"r1","what_happened":"...","root_cause":"strategy|environment|noise","evidence":"..."}],
 "contrast":"what the successful runs did differently (or empty)",
 "edits":[{"op":"add|revise|remove","id":"p01 (revise/remove only)","text":"the note (add/revise only)",
           "evidence":"run rN: ..."}]}
"""


def _clip(text: Any, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _headline_line(headline: Mapping[str, Any]) -> str:
    return (f"residual_dead_air={headline.get('residual_dead_air_pct')}% overruns={headline.get('groups_rushed')} "
            f"collisions={headline.get('collisions')}/{headline.get('cross_speaker_collisions')}cross "
            f"gates={headline.get('gate_failures')} meaning_loss={headline.get('meaning_loss_units', 'n/a')}")


def digest_run(run_dir: Path, *, max_chars: int = DIGEST_MAX_CHARS) -> str:
    """A compact, deterministic account of one run, built from its trace and summary."""
    run_dir = Path(run_dir)
    summary_path = run_dir / "agent_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    card_path = run_dir / "scorecard.json"
    final = json.loads(card_path.read_text(encoding="utf-8"))["headline"] if card_path.is_file() else {}
    tokens = summary.get("tokens") or {}
    header = (
        f"OUTCOME: stop={summary.get('stop_reason')} met_bar={summary.get('stop_reason') == 'agent_finished'} "
        f"turns={summary.get('turns')} tts={summary.get('tts_used')} accepted_submissions={summary.get('submissions')} "
        f"pushbacks={summary.get('pushbacks')} tokens={(tokens.get('in') or 0) + (tokens.get('out') or 0):,} "
        f"| final: {_headline_line(final)} | unmet_at_end={summary.get('unmet_at_end')}"
    )
    lines: list[tuple[bool, str]] = []
    trace_path = run_dir / "trace.jsonl"
    records = [json.loads(l) for l in trace_path.read_text(encoding="utf-8").splitlines() if l.strip()] if trace_path.is_file() else []
    for record in records:
        kind = record["kind"]
        t = f"[{record.get('t')}s]"
        if kind == "model":
            calls = {}
            for call in record.get("tool_calls") or []:
                calls[call["name"]] = calls.get(call["name"], 0) + 1
            said = _clip(record.get("text"), 200)
            if said and not calls:
                lines.append((True, f"{t} SAYS-NO-TOOLS (turn {record['turn']}): {said}"))
            elif said:
                lines.append((False, f"{t} turn {record['turn']} says: {said} | calls {calls}"))
            else:
                lines.append((False, f"{t} turn {record['turn']} calls {calls}"))
        elif kind == "tool":
            name, result = record["name"], record.get("result", "")
            if name == "submit_dub":
                try:
                    data = json.loads(result)
                except ValueError:
                    data = {}
                if data.get("accepted"):
                    lines.append((False, f"{t} submit accepted: {_headline_line(data.get('headline', {}))} lock={data.get('timing_lock')}"))
                else:
                    lines.append((True, f"{t} submit REJECTED: {_clip(data.get('errors') or result, 220)} "
                                        f"this={data.get('this_submission')} previous={data.get('previous_submission')}"))
            elif not record.get("ok", True):
                lines.append((False, f"{t} {name} FAILED: {_clip(result, 120)}"))
        elif kind == "pushback":
            lines.append((True, f"{t} PUSHBACK {record.get('number')}: unmet={record.get('unmet')}"))
        elif kind == "nudge":
            lines.append((True, f"{t} NUDGE: {_clip(record.get('text'), 120)}"))
        elif kind == "compact":
            lines.append((False, f"{t} history compacted ({record.get('dropped_steps')} steps dropped)"))
        elif kind == "end":
            lines.append((True, f"{t} END stop={record.get('stop_reason')}"))

    body = [text for _, text in lines]
    budget = max_chars - len(header)
    if sum(len(x) + 1 for x in body) > budget:
        kept = [(important, text) for important, text in lines]
        # drop unimportant lines from the middle outward until it fits
        middle = [i for i, (important, _) in enumerate(kept) if not important]
        while middle and sum(len(t) + 1 for _, t in kept) > budget:
            drop = middle.pop(len(middle) // 2)
            kept[drop] = (False, "")
        body = [text for _, text in kept if text]
        if sum(len(x) + 1 for x in body) > budget:
            body = [_clip(x, 160) for x in body]
    return header + "\n" + "\n".join(body)


def build_reflection_messages(playbook: Mapping[str, Any], digests: Mapping[str, str], *, clip_note: str = "") -> list[dict[str, str]]:
    entries = "\n".join(f"{e['id']}: {e['text']}" for e in playbook["entries"]) or "(empty)"
    runs = "\n\n".join(f"=== RUN {name} ===\n{text}" for name, text in digests.items())
    user = (
        f"CURRENT PLAYBOOK (version {playbook['version']}):\n{entries}\n\n"
        f"{('CONTEXT: ' + clip_note + chr(10) + chr(10)) if clip_note else ''}"
        f"RUNS OF THE SAME AGENT WITH THIS PLAYBOOK:\n\n{runs}\n\nReply with the JSON only."
    )
    return [{"role": "system", "content": REFLECTOR_SYSTEM_PROMPT}, {"role": "user", "content": user}]


def parse_reflection(text: str) -> dict[str, Any]:
    """Tolerant JSON extraction; never raises on a malformed reply (returns an empty reflection with the error)."""
    try:
        start, end = text.index("{"), text.rindex("}") + 1
        data = json.loads(text[start:end])
    except (ValueError, TypeError) as exc:
        return {"diagnosis": [], "contrast": "", "edits": [], "parse_error": str(exc)[:120]}
    edits = [e for e in data.get("edits") or [] if isinstance(e, dict)]
    return {"diagnosis": [d for d in data.get("diagnosis") or [] if isinstance(d, dict)],
            "contrast": str(data.get("contrast") or ""), "edits": edits}


def reflect(client: Any, playbook: Mapping[str, Any], digests: Mapping[str, str], *, clip_note: str = "") -> dict[str, Any]:
    """One model call. Returns the parsed reflection plus the token usage it cost."""
    response = client.complete(build_reflection_messages(playbook, digests, clip_note=clip_note), json_mode=True,
                               max_output_tokens=6000)
    message = (response.get("choices") or [{}])[0].get("message") or {}
    result = parse_reflection(message.get("content") or "")
    result["usage"] = response.get("usage") or {}
    return result
