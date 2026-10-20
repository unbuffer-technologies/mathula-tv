"""Harness smells: deterministic detectors for problems in the ENVIRONMENT, not the agent's strategy.

A failed run has two kinds of cause. Some are strategy -- a different way of working would have avoided
them -- and belong in the playbook. Others are traps in the harness itself: a tool that rejects reasonable
input, a rule that freezes a bad draft, a unit whose window no wording can fit. An agent cannot learn its way
out of those, and it must not be allowed to edit its own rules, so they are reported to a human. These
detectors find the patterns that used to take a person reading traces: they read the traces and summaries of
a batch and return findings with evidence and a suggested owner.

Every detector is plain code over the trace; none calls a model.
"""
from __future__ import annotations

import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

REPEATED_ERROR_MIN_TOTAL = 3
TOKEN_BLOWUP_FACTOR = 2.0
TOKEN_BLOWUP_FLOOR = 600_000
LONG_RUN_TURNS = 15


def _trace(run: Mapping[str, Any]) -> list[dict[str, Any]]:
    path = Path(run["dir"]) / "trace.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _summary(run: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(run["dir"]) / "agent_summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _label(run: Mapping[str, Any]) -> str:
    return f"{run.get('clip', '')}:r{run['repeat']}".lstrip(":")


def _normalise_error(text: str) -> str:
    return re.sub(r"\d+", "N", re.sub(r"\s+", " ", text)).strip()[:140]


def _finding(smell: str, severity: str, evidence: str, runs: Sequence[str], suggestion: str) -> dict[str, Any]:
    return {"smell": smell, "severity": severity, "evidence": evidence, "runs": sorted(set(runs)),
            "owner": "harness", "suggestion": suggestion}


def repeated_tool_errors(runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str]] = Counter()
    where: dict[tuple[str, str], list[str]] = defaultdict(list)
    for run in runs:
        for record in _trace(run):
            if record["kind"] == "tool" and not record.get("ok", True):
                try:
                    detail = json.loads(record["result"])
                    message = f"{detail.get('error')}: {detail.get('detail', '')}"
                except ValueError:
                    message = record["result"]
                key = (record["name"], _normalise_error(message))
                counts[key] += 1
                where[key].append(_label(run))
    findings = []
    for (tool, message), total in counts.items():
        if total >= REPEATED_ERROR_MIN_TOTAL or len(set(where[(tool, message)])) >= 2:
            findings.append(_finding(
                "repeated_tool_error", "high" if len(set(where[(tool, message)])) >= 2 else "medium",
                f"{tool} failed {total}x with: {message}", where[(tool, message)],
                "A tool that rejects reasonable input wastes turns in every run; relax or fix the tool rather than "
                "teach the agent around it.",
            ))
    return findings


def rejection_trap(runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    findings = []
    for run in runs:
        rejected = [r for r in _trace(run) if r["kind"] == "tool" and r["name"] == "submit_dub"
                    and '"accepted": false' in r.get("result", "").lower()]
        if len(rejected) >= 2 and run["stop_reason"] != "agent_finished":
            reasons = Counter(_normalise_error(r["result"])[:90] for r in rejected)
            findings.append(_finding(
                "rejection_trap", "high",
                f"{_label(run)} had {len(rejected)} rejected submissions and ended {run['stop_reason']}; "
                f"most common: {reasons.most_common(1)[0][0]}",
                [_label(run)], "A rule that blocks repeated attempts can freeze a bad state; check whether the "
                               "rejected submissions were net improvements."))
    return findings


def declined_to_submit(runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    flagged = [_label(r) for r in runs if any(x["kind"] == "nudge" for x in _trace(r))]
    if not flagged:
        return []
    return [_finding(
        "declined_to_submit", "high" if len(flagged) > 1 else "medium",
        f"{len(flagged)} run(s) stopped without any submission and had to be nudged",
        flagged, "The agent withheld work; look at what the prompt or playbook implies about when to submit.")]


def budget_exhaustion(runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    findings = []
    for run in runs:
        episode = Path(run["dir"]) / "episode.json"
        budget = json.loads(episode.read_text(encoding="utf-8")).get("tts_budget") if episode.is_file() else None
        used = run.get("tts_used") or 0
        hit = any(r["kind"] == "tool" and "budget_exhausted" in r.get("result", "") for r in _trace(run))
        if hit or (budget and used >= 0.95 * budget):
            findings.append(_finding(
                "budget_exhausted", "high", f"{_label(run)} used {used} of {budget} TTS measurements",
                [_label(run)], "The measurement budget ended the run, not the agent's judgement; raise it or make "
                               "measurements cheaper."))
    return findings


def ceiling_stops(runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    stopped = [(_label(r), r["stop_reason"]) for r in runs if r["stop_reason"] in ("max_turns", "token_budget")]
    if not stopped:
        return []
    return [_finding("ceiling_stop", "medium", "; ".join(f"{l} hit {s}" for l, s in stopped), [l for l, _ in stopped],
                     "A hard ceiling ended the run before the agent chose to; check the ceiling against the work.")]


def token_blowup(runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    tokens = {_label(r): r["processed_tokens"] for r in runs}
    if len(tokens) < 2:
        return []
    median = statistics.median(tokens.values())
    outliers = {l: t for l, t in tokens.items() if t >= TOKEN_BLOWUP_FLOOR and t >= TOKEN_BLOWUP_FACTOR * median}
    if not outliers:
        return []
    return [_finding("token_blowup", "medium",
                     "; ".join(f"{l} used {t:,} tokens vs a median of {median:,.0f}" for l, t in outliers.items()),
                     list(outliers), "One run cost far more than its siblings; inspect it for a loop or runaway context.")]


def compaction_missing(runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    flagged = []
    for run in runs:
        summary = _summary(run)
        if (summary.get("turns") or 0) >= LONG_RUN_TURNS and not summary.get("compactions"):
            flagged.append(_label(run))
    if not flagged:
        return []
    return [_finding("compaction_missing", "medium", f"{len(flagged)} long run(s) never compacted their history",
                     flagged, "History compaction did not trigger; check the threshold and the submit-step logic.")]


def unreachable_windows(runs: Sequence[Mapping[str, Any]], *, tolerance_ms: int = 60) -> list[dict[str, Any]]:
    """A window flagged for the same timing problem in EVERY run suggests no wording can fit it."""
    if len(runs) < 2:
        return []
    per_run: list[dict[int, set[str]]] = []
    for run in runs:
        path = Path(run["dir"]) / "scorecard.json"
        if not path.is_file():
            return []
        card = json.loads(path.read_text(encoding="utf-8"))
        flagged: dict[int, set[str]] = {}
        for group in card.get("groups", []):
            problems = {f for f in group["flags"] if f in ("overrun", "dead_air", "rushed")}
            if group.get("collision_ms"):
                problems.add("collision")
            if problems:
                flagged[int(group["start_ms"])] = problems
        per_run.append(flagged)
    findings = []
    for start in per_run[0]:
        matches = [next((p for s, p in other.items() if abs(s - start) <= tolerance_ms), None) for other in per_run]
        if all(matches):
            common = set.intersection(*matches)
            if common:
                findings.append(_finding(
                    "unreachable_window", "medium",
                    f"the unit starting at {start} ms is flagged {sorted(common)} in all {len(runs)} runs",
                    [_label(r) for r in runs], "Every run fails here: check whether the window is physically fillable "
                                               "(very short or very long) before blaming the agent."))
    return findings


DETECTORS = (repeated_tool_errors, rejection_trap, declined_to_submit, budget_exhaustion, ceiling_stops,
             token_blowup, compaction_missing, unreachable_windows)


def find_smells(runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """All findings for a batch of runs (each a ``dub_improve.collect_batch`` row, optionally with ``clip``)."""
    findings: list[dict[str, Any]] = []
    for detector in DETECTORS:
        findings.extend(detector(runs))
    order = {"high": 0, "medium": 1, "low": 2}
    return sorted(findings, key=lambda f: order[f["severity"]])


def render(findings: Sequence[Mapping[str, Any]]) -> str:
    if not findings:
        return "no harness problems detected"
    return "\n".join(f"[{f['severity']}] {f['smell']}: {f['evidence']} (runs: {', '.join(f['runs'])}). {f['suggestion']}"
                     for f in findings)
