"""Compact the experience ledger into the few lessons the agent should read before it writes.

The ledger (``experience_ledger``) keeps every choice, strategy and missing-evidence note the agent recorded in a training run, and what became of each choice in the final text.  Nothing read
it back: the agent's next run started as ignorant as the last.  This turns the accumulated record into a short list of lessons, in one bounded step:

* ONE model call (never a loop) reads a bounded selection of the ledger: choices the run itself dropped or set evidence aside for, a sample of the kept ones, the open and answered
  missing-evidence notes, and the latest strategies;
* the model proposes at most ``MAX_LESSONS`` lessons, each citing the ledger rows it rests on;
* a deterministic check, not the model, decides what stays: every cited row must exist, and the rows must come from at least ``MIN_JOBS`` different videos (a habit seen over eleven runs of one
  video is one observation).  A lesson that is about one name, one amount or one sentence of one video is the model's to keep out; the prompt says so and the check on distinct videos enforces it
  for what it can.

A lesson is a hypothesis the agent reads, not a rule: ``source`` is ``inferred`` and the agent is told a counter-example outweighs it.  Lessons are stored in ``working/experience_lessons.json``
and reach the agent through ``ear_lessons.for_agent`` (``recall_lessons``).  A stored lesson spreads to every run, so ``retire`` takes one out with the reason kept, and the default run only
prints what it would store: ``--apply`` stores it.

    python -m mathula_tv.experience_compact              # show the lessons the ledger supports
    python -m mathula_tv.experience_compact --apply     # store them
    python -m mathula_tv.experience_compact retire --id <id> --reason "..."
    python -m mathula_tv.experience_compact reinstate --id <id>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

FILE = "experience_lessons.json"
MAX_LESSONS = 12
MIN_JOBS = 2                  # a lesson must rest on rows from this many different videos
MIN_ROWS = 3                  # ... and on this many rows
SELECT_CHARS = 60000          # the most ledger text one call reads
KEPT_SAMPLE = 40              # kept choices shown (the dropped ones all are)
STRATEGY_SAMPLE = 30
FIELD_CHARS = 280

PROMPT = """You compact what a dubbing agent has learned across many runs into a few lessons for its next run.

The agent translates English news speech into isiZulu for a social-media audience. Its own ledger of past runs is below: CHOICES (a place where it weighed two ways of saying
something, what it picked, why, and whether the pick survived into the run's final text), STRATEGIES (how it approached a unit) and MISSING EVIDENCE notes (what it could not find out).

Write at most %(max)d lessons. A lesson is a habit worth keeping or a trap worth avoiding that you can see in SEVERAL rows from DIFFERENT videos (each row carries its video as "job").
Prefer: choices that did not survive into the final text and why; evidence the agent set aside and what came of it; the same kind of thing the agent could not find out again and again;
a strategy that shows up across videos. Do NOT write: a lesson about one name, number, sentence or video; advice that is only the agent's existing instructions restated; anything you
cannot support with the cited rows. Each lesson says what situation it applies to, what to do, what to avoid, and why (what the rows show). Cite the ids of the rows it rests on.

Reply with JSON only: {"lessons":[{"item_class":"short label","situation":"...","do":"...","avoid":"...","because":"...","ledger_ids":["id", ...]}]}
"""


def path_of(work_dir: Path) -> Path:
    return Path(work_dir) / FILE


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _cut(value: Any, width: int = FIELD_CHARS) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= width else text[:width - 1] + "…"


def select(rows: Sequence[Mapping[str, Any]], *, kept_sample: int = KEPT_SAMPLE, strategy_sample: int = STRATEGY_SAMPLE, budget: int = SELECT_CHARS) -> list[dict[str, Any]]:
    """The rows the model reads: every choice that did not survive or set evidence aside, a spread of the kept ones, every missing-evidence note, the latest strategies; each cut short,
    and the whole kept within ``budget`` characters."""
    def view(row: Mapping[str, Any]) -> dict[str, Any]:
        base = {"id": row["id"], "job": str(row.get("job") or "")[:8], "run": row.get("run"), "kind": row["kind"], "unit": row.get("unit")}
        if row["kind"] == "choice":
            return {**base, "english": _cut(row.get("english"), 160), "zulu": _cut(row.get("zulu"), 160), "alternatives": [_cut(a, 100) for a in (row.get("alternatives") or [])[:3]],
                    "why": _cut(row.get("why")), "evidence": _cut(row.get("evidence"), 200), "set_aside_evidence": bool(row.get("set_aside_evidence")), "outcome": row.get("outcome")}
        if row["kind"] == "missing_evidence":
            return {**base, "wanted": _cut(row.get("wanted")), "why": _cut(row.get("why")), "outcome": row.get("outcome")}
        return {**base, "strategy": _cut(row.get("strategy"), 360)}

    choices = [r for r in rows if r.get("kind") == "choice"]
    dropped = [r for r in choices if r.get("outcome") != "kept" or r.get("set_aside_evidence")]
    kept = [r for r in choices if r not in dropped]
    step = max(1, len(kept) // max(1, kept_sample))
    chosen = [*dropped, *kept[::step][:kept_sample], *[r for r in rows if r.get("kind") == "missing_evidence"], *[r for r in rows if r.get("kind") == "strategy"][-strategy_sample:]]
    out: list[dict[str, Any]] = []
    used = 0
    for row in chosen:
        item = view(row)
        size = len(json.dumps(item, ensure_ascii=False))
        if used + size > budget:
            continue
        out.append(item)
        used += size
    return out


def check(proposed: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]], *, min_jobs: int = MIN_JOBS, min_rows: int = MIN_ROWS) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(kept, dropped): a lesson stays when it says what to do, cites ledger rows that exist, and those rows span ``min_jobs`` videos and number ``min_rows``.  The dropped carry the reason."""
    by_id = {str(r["id"]): r for r in rows}
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for lesson in list(proposed)[:MAX_LESSONS * 2]:
        cited = [str(i) for i in (lesson.get("ledger_ids") or []) if str(i) in by_id]
        text = {k: _cut(lesson.get(k), 400) for k in ("item_class", "situation", "do", "avoid", "because")}
        if not text["do"] or not text["situation"]:
            dropped.append({**text, "why_dropped": "no situation or no action"})
        elif len(set(cited)) < min_rows:
            dropped.append({**text, "why_dropped": f"cites {len(set(cited))} existing row(s), needs {min_rows}"})
        elif len({str(by_id[i].get("job")) for i in cited}) < min_jobs:
            dropped.append({**text, "why_dropped": f"rests on {len({str(by_id[i].get('job')) for i in cited})} video(s), needs {min_jobs}"})
        else:
            digest = hashlib.sha256((text["situation"] + "|" + text["do"]).encode("utf-8")).hexdigest()[:12]
            kept.append({"id": f"exp-{digest}", **text, "ledger_ids": sorted(set(cited)), "videos": len({str(by_id[i].get("job")) for i in cited}), "source": "inferred", "status": "active"})
    return kept[:MAX_LESSONS], dropped


def propose(rows: Sequence[Mapping[str, Any]], complete: Callable[[list[dict[str, Any]]], str]) -> list[dict[str, Any]]:
    """One model call over the selected rows; the lessons it proposes, unchecked."""
    shown = select(rows)
    reply = complete([{"role": "system", "content": PROMPT % {"max": MAX_LESSONS}}, {"role": "user", "content": json.dumps({"ledger": shown}, ensure_ascii=False)}])
    try:
        data = json.loads(reply)
    except ValueError:
        return []
    return [x for x in (data.get("lessons") or []) if isinstance(x, dict)]


def _read(work_dir: Path) -> dict[str, Any]:
    try:
        data = json.loads(path_of(work_dir).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def store(work_dir: Path, lessons: Sequence[Mapping[str, Any]], *, rows: int) -> None:
    """Replace the active lessons with these (retired ones stay, with their reasons); the file is rewritten whole."""
    old = _read(work_dir)
    retired = [x for x in (old.get("lessons") or []) if x.get("status") == "retired"]
    retired_ids = {x["id"] for x in retired}
    path = path_of(work_dir)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"built": _now(), "from_ledger_rows": rows, "lessons": [*[dict(x) for x in lessons if x["id"] not in retired_ids], *retired]}, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def retire(work_dir: Path, lesson_id: str, reason: str) -> bool:
    data = _read(work_dir)
    found = False
    for lesson in data.get("lessons") or []:
        if lesson.get("id") == lesson_id:
            lesson["status"], lesson["retired_because"], lesson["retired_at"] = "retired", reason, _now()
            found = True
    if found:
        path_of(work_dir).write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    return found


def reinstate(work_dir: Path, lesson_id: str) -> bool:
    """Bring a retired lesson back (its retirement reason is kept as history)."""
    data = _read(work_dir)
    found = False
    for lesson in data.get("lessons") or []:
        if lesson.get("id") == lesson_id and lesson.get("status") == "retired":
            lesson["status"], lesson["reinstated_at"] = "active", _now()
            lesson.setdefault("history", []).append({"retired_because": lesson.pop("retired_because", ""), "retired_at": lesson.pop("retired_at", "")})
            found = True
    if found:
        path_of(work_dir).write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    return found


def for_agent(work_dir: Path) -> list[dict[str, Any]]:
    """The active lessons as the agent reads them (without the ledger bookkeeping)."""
    return [{"item_class": x.get("item_class"), "situation": x.get("situation"), "do": x.get("do"), "avoid": x.get("avoid"), "because": x.get("because"), "source": "inferred",
             "videos": x.get("videos")} for x in _read(work_dir).get("lessons") or [] if x.get("status") == "active"]


def _complete_with_model(messages: list[dict[str, Any]]) -> str:
    from .dub_agent import ResponsesClient
    from .foundry_grok import native_gpt_config_from_environment

    client = ResponsesClient(native_gpt_config_from_environment(dict(os.environ)))
    reply = client.complete(messages, json_mode=True, max_output_tokens=6000)
    return str(reply["choices"][0]["message"].get("content") or "")


def main(argv: list[str] | None = None) -> int:
    from .config import load_settings
    from .experience_ledger import load

    parser = argparse.ArgumentParser(prog="mathula_tv.experience_compact")
    parser.add_argument("command", nargs="?", default="compact", choices=("compact", "retire", "reinstate"))
    parser.add_argument("--apply", action="store_true", help="store the lessons (without it they are only shown)")
    parser.add_argument("--id")
    parser.add_argument("--reason")
    args = parser.parse_args(argv)
    work_dir = Path(load_settings().work_dir)
    if args.command == "reinstate":
        if not args.id:
            parser.error("reinstate needs --id")
        print("reinstated" if reinstate(work_dir, args.id) else "no such retired lesson")
        return 0
    if args.command == "retire":
        if not args.id or not args.reason:
            parser.error("retire needs --id and --reason")
        print("retired" if retire(work_dir, args.id, args.reason) else "no such lesson")
        return 0
    rows = load(work_dir)
    proposed = propose(rows, _complete_with_model)
    kept, dropped = check(proposed, rows)
    print(f"{len(rows)} ledger rows read {len(select(rows))} shown; the model proposed {len(proposed)}; {len(kept)} stand, {len(dropped)} dropped")
    for lesson in kept:
        print(f"\n[{lesson['id']}] {lesson['item_class']}  ({lesson['videos']} videos, {len(lesson['ledger_ids'])} rows)\n  when: {lesson['situation']}\n  do: {lesson['do']}\n  avoid: {lesson['avoid']}\n  because: {lesson['because']}")
    for lesson in dropped:
        print(f"\n(dropped: {lesson['why_dropped']}) {lesson.get('item_class')}: {lesson.get('do')}")
    if args.apply:
        store(work_dir, kept, rows=len(rows))
        print(f"\nstored {len(kept)} lesson(s) in {path_of(work_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
