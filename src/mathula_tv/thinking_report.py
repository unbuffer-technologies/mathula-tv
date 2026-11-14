"""The training-mode report of a run: how the agent says it decided, in English, unit by unit.

A training run (``--mode training``) asks the agent to record, with each submit_dub, its choice points (where it weighed alternatives or went against evidence) and one line of strategy for
each unit it built, and it asks the model for a summary of its own reasoning each turn.  The records are in ``thinking.jsonl`` (written by the workbench as the run goes) and the summaries
in ``trace.jsonl``; this module reads both and writes ``thinking_report.md`` beside them for someone who does not read Zulu.

What the report is not: proof of why the model produced a word.  A choice point is the agent's own account, written alongside the text; the harness only checks that the form it names is in the
unit (``chosen_in_text``).  Study it to find where tools, prompts and evidence help or are set aside; confirm any change on live runs with the measured checks.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path
from typing import Any

REASONING_CHARS = 1200
EVIDENCE_ROWS = 15


def _lines(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    except OSError:
        pass
    return rows


def load(directory: Path) -> dict[str, Any]:
    """The choice points (the latest record for each unit and English chunk), the unit strategies (the latest per unit) and the per-turn reasoning summaries of a run."""
    directory = Path(directory)
    choices: dict[tuple[str, str], dict[str, Any]] = {}
    strategies: dict[str, dict[str, Any]] = {}
    missing: dict[tuple[str, str], dict[str, Any]] = {}
    for row in _lines(directory / "thinking.jsonl"):
        if row.get("kind") == "missing_evidence":
            missing[(str(row.get("unit")), str(row.get("wanted") or "").casefold()[:80])] = row
        if row.get("kind") == "choice":
            choices[(str(row.get("unit")), str(row.get("english") or "").casefold())] = row
        elif row.get("kind") == "strategy":
            strategies[str(row.get("unit"))] = row
    reasoning = []
    for row in _lines(directory / "trace.jsonl"):
        if row.get("kind") == "model" and row.get("reasoning"):
            reasoning.append({"turn": row.get("turn"), "text": str(row["reasoning"]), "tools": [c.get("name") for c in row.get("tool_calls") or []]})
    try:
        summary = json.loads((directory / "agent_summary.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        summary = {}
    return {"choices": list(choices.values()), "strategies": strategies, "reasoning": reasoning, "summary": summary, "missing": list(missing.values())}


def _cell(value: Any) -> str:
    return " ".join(str(value or "").split()).replace("|", "/")


def _unit_key(unit: str) -> tuple[int, str]:
    digits = "".join(ch for ch in unit if ch.isdigit())
    return (int(digits) if digits else 10 ** 6, unit)


def build_report(directory: Path) -> str:
    data = load(directory)
    choices, strategies, reasoning, summary = data["choices"], data["strategies"], data["reasoning"], data["summary"]
    out = [f"# How the agent decided: {summary.get('run_id', Path(directory).name)}", ""]
    tokens = summary.get("tokens") or {}
    out.append(f"Mode {summary.get('agent_mode', 'training')}, model {summary.get('model', '?')}, {summary.get('turns', '?')} turns, stop reason {summary.get('stop_reason', '?')}; "
               f"tokens in {tokens.get('in', '?')}, out {tokens.get('out', '?')}.")
    out.append("")
    out.append("The records below are the agent's own account, written with its text. The harness checked only that the form it names is in the unit. Use them to see where evidence was used or "
               "set aside; confirm any change on live runs with the measured checks.")
    out.append("")
    adrift = [c for c in choices if not c.get("chosen_in_text")]
    unknown = [c for c in choices if not c.get("unit_known", True)]
    units = sorted({str(c.get("unit")) for c in choices} | set(strategies), key=_unit_key)
    out.append(f"**{len(choices)} choice points** in {len(units)} units; **{len(adrift)}** name a form that is not in the final unit text (the agent changed its mind or the record is wrong); "
               f"{len(unknown)} name a unit that does not exist; **{len(reasoning)}** turns have a reasoning summary.")
    out.append("")
    wanted = data["missing"]
    if wanted:
        out.append("## What the agent said it wanted and was not given")
        out.append("")
        out.append("This is the list to act on: each line is information the agent says would have helped its decision.")
        out.append("")
        out.append("| unit | wanted | why |")
        out.append("|---|---|---|")
        for m in wanted:
            out.append(f"| {_cell(m.get('unit')) or 'whole clip'} | {_cell(m.get('wanted'))} | {_cell(m.get('why'))} |")
        out.append("")
    if not choices and not strategies:
        out.append("No choice points were recorded (the agent sent none, or this was a live run).")
        out.append("")
    tally = collections.Counter(_cell(c.get("evidence")).casefold()[:70] or "(none given)" for c in choices)
    if tally:
        out.append("## Evidence the choices rest on")
        out.append("")
        out.append("| times | evidence |")
        out.append("|---|---|")
        for text, count in tally.most_common(EVIDENCE_ROWS):
            out.append(f"| {count} | {text} |")
        set_aside = sum(1 for c in choices if "set aside" in str(c.get("evidence") or "").casefold() or "ignored" in str(c.get("evidence") or "").casefold())
        out.append("")
        out.append(f"Choices that say they set evidence aside: **{set_aside}**.")
        out.append("")
    if units:
        out.append("## Unit by unit")
        out.append("")
    for unit in units:
        out.append(f"### {unit}")
        if unit in strategies:
            out.append(f"Strategy: {_cell(strategies[unit].get('strategy'))}")
        rows = [c for c in choices if str(c.get("unit")) == unit]
        if rows:
            out.append("")
            out.append("| English | Zulu chosen | alternatives | why | evidence | in text |")
            out.append("|---|---|---|---|---|---|")
            for c in rows:
                out.append(f"| {_cell(c.get('english'))} | {_cell(c.get('zulu'))} | {_cell('; '.join(c.get('alternatives') or []))} | {_cell(c.get('why'))} | {_cell(c.get('evidence'))} | "
                           f"{'yes' if c.get('chosen_in_text') else 'NO'} |")
        out.append("")
    if reasoning:
        out.append("## The model's own reasoning summaries, by turn")
        out.append("")
        for r in reasoning:
            text = r["text"].strip()
            out.append(f"**Turn {r['turn']}** ({', '.join(t for t in r['tools'] if t) or 'no tool call'})")
            out.append("")
            out.append(text if len(text) <= REASONING_CHARS else text[:REASONING_CHARS].rstrip() + " ...")
            out.append("")
    return "\n".join(out).rstrip() + "\n"


def write_report(directory: Path) -> Path:
    path = Path(directory) / "thinking_report.md"
    path.write_text(build_report(directory), encoding="utf-8")
    return path
