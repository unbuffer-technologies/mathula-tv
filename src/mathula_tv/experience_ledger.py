"""What the agent recorded about its own decisions, kept across jobs.

In training mode the agent writes, with every submit, the choice points it weighed (English chunk, the Zulu chosen, the alternatives, why, the evidence), a strategy per unit, and what it
wanted to know and was not given (``missing_evidence``).  Those lines used to stay in the run's own folder (``thinking.jsonl``) and nothing read them again.  This gathers them, run
after run, in one file the later jobs can learn from::

    <work_dir>/experience/agent_ledger.jsonl        one JSON object per line, written atomically as a whole

An entry is the agent's own account, so it is labelled ``source: "agent_inferred"``: a hypothesis, never a rule.  What turns an account into experience is what happened to it afterwards, kept
in ``outcome``:

* a choice: ``kept`` (the form is in the run's final unit text) or ``not_in_final`` (the agent changed its mind after writing the record);
* a missing-evidence note: ``open`` until the listener answers the console's question for it, then ``answered`` with what they said (``outcome_note``);
* a strategy: ``recorded`` (there is nothing to check it against yet);
* any entry: ``contradicted`` when ``mark`` is told the listener or a measurement overturned it.

Nothing here calls a model, and nothing here decides anything.  Compacting the ledger into a short, bounded set of lessons is a later step that works from this file.

    python -m mathula_tv.experience_ledger --job <job id> --run <run id>       # add one run (a run ends with this by itself in training mode)
    python -m mathula_tv.experience_ledger --all                               # every run of every job that has a thinking.jsonl (idempotent)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

LEDGER_NAME = "agent_ledger.jsonl"
SOURCE = "agent_inferred"
OUTCOMES = ("kept", "not_in_final", "open", "answered", "recorded", "contradicted")
_SET_ASIDE = re.compile(r"\bset aside\b|\bset (?:the |this |that )?\w+ aside\b|\boverrode\b|\bignored\b", re.IGNORECASE)          # the choice says it set a flag or a piece of evidence aside


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ledger_path(work_dir: Path) -> Path:
    return Path(work_dir) / "experience" / LEDGER_NAME


def _digest(*parts: Any) -> str:
    return hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:16]


_QUOTED = re.compile(r"""['‘’"“”]([^'‘’"“”]{3,80})['‘’"“”]""")


def evidence_key(job_id: str, unit: str, wanted: str) -> str:
    """The same missing-evidence note in two runs of one job is one question for the listener.  Units are cut afresh in every run and the agent words the note differently ("the garbled
    words immediately after 'third party'" in prod-101, "the phrase between 'third party' and 'business bank account'" in prod-102), so the key is the job and the first stretch of the
    source the note quotes; a note that quotes nothing falls back to the unit and the start of what was wanted."""
    quoted = _QUOTED.search(str(wanted or ""))
    if quoted:
        return _digest("evidence", job_id, re.sub(r"[^a-z0-9 ]", "", quoted.group(1).casefold()).strip())
    return _digest("evidence", job_id, unit, " ".join(str(wanted or "").casefold().split())[:80])


def load(work_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for line in ledger_path(work_dir).read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and row.get("id"):
                if row.get("kind") == "missing_evidence":          # the key is computed from the note, so a change in how it is computed re-keys the old entries too
                    row["key"] = evidence_key(str(row.get("job") or ""), str(row.get("unit") or ""), str(row.get("wanted") or ""))
                rows.append(row)
    except OSError:
        pass
    return rows


def _save(work_dir: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path = ledger_path(work_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part")
    temporary.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    temporary.replace(path)


def _final_texts(run_dir: Path) -> dict[str, str]:
    try:
        items = json.loads((Path(run_dir) / "submission.json").read_text(encoding="utf-8")).get("items") or []
    except (OSError, ValueError):
        return {}
    return {str(i.get("item_id")): str(i.get("zulu_text") or "") for i in items if isinstance(i, Mapping)}


def _in_text(form: str, text: str) -> bool:
    """Whether the Zulu form the agent named is in the unit's final text, by stem (ZulMorph when it can be reached, the plain forms otherwise)."""
    from . import zulu_stems

    if not form or not text:
        return False
    try:
        analyses = zulu_stems.analyse(zulu_stems.words_of(form) + zulu_stems.words_of(text))
    except Exception:                                             # noqa: BLE001 - an optional analyser must never stop the ledger
        analyses = {}
    return zulu_stems.holds(form, text, analyses)


def entries_of_run(run_dir: Path, *, job_id: str, run_id: str, clip_id: str) -> list[dict[str, Any]]:
    """The ledger entries of one finished run: the latest record of each choice, strategy and missing-evidence note in its ``thinking.jsonl``, with the outcome the run's final text shows."""
    from . import thinking_report

    run_dir = Path(run_dir)
    data = thinking_report.load(run_dir)
    texts = _final_texts(run_dir)
    summary = data.get("summary") or {}
    base = {"job": job_id, "run": run_id, "clip": clip_id, "at": _now(), "source": SOURCE, "model": str(summary.get("model") or ""), "turns": summary.get("turns")}
    rows: list[dict[str, Any]] = []
    for c in data["choices"]:
        unit, english, zulu = str(c.get("unit") or ""), str(c.get("english") or ""), str(c.get("zulu") or "")
        kept = _in_text(zulu, texts.get(unit, ""))
        why, evidence = str(c.get("why") or ""), str(c.get("evidence") or "")
        rows.append({**base, "id": _digest("choice", job_id, run_id, unit, english.casefold()), "kind": "choice", "unit": unit, "english": english, "zulu": zulu,
                     "alternatives": list(c.get("alternatives") or []), "why": why, "evidence": evidence, "set_aside_evidence": bool(_SET_ASIDE.search(why + " " + evidence)),
                     "outcome": "kept" if kept else "not_in_final", "outcome_note": ""})
    for unit, s in sorted(data["strategies"].items()):
        rows.append({**base, "id": _digest("strategy", job_id, run_id, unit), "kind": "strategy", "unit": str(unit), "strategy": str(s.get("strategy") or ""),
                     "outcome": "recorded", "outcome_note": ""})
    for m in data["missing"]:
        unit, wanted = str(m.get("unit") or ""), str(m.get("wanted") or "")
        rows.append({**base, "id": _digest("missing", job_id, run_id, unit, wanted.casefold()[:80]), "key": evidence_key(job_id, unit, wanted), "kind": "missing_evidence",
                     "unit": unit, "wanted": wanted, "why": str(m.get("why") or ""), "outcome": "open", "outcome_note": ""})
    return rows


def ingest_run(work_dir: Path, *, job_id: str, run_id: str, clip_id: str, repeat: int = 1) -> dict[str, int]:
    """Add one run's entries to the ledger; running it again adds nothing (an entry has one id for ever).  A missing-evidence note the listener already answered for this job starts ``answered``."""
    from .experience_store import ExperienceStore

    run_dir = ExperienceStore(Path(work_dir)).repeat_dir(run_id, clip_id, repeat)
    new = entries_of_run(run_dir, job_id=job_id, run_id=run_id, clip_id=clip_id)
    rows = load(work_dir)
    have = {r["id"] for r in rows}
    answers = _answers_of(work_dir, job_id)
    added = 0
    for row in new:
        if row["id"] in have:
            continue
        if row["kind"] == "missing_evidence" and row.get("key") in answers:
            row["outcome"], row["outcome_note"] = "answered", str(answers[row["key"]].get("text") or answers[row["key"]].get("comment") or "")[:600]
        rows.append(row)
        added += 1
    if added:
        _save(work_dir, rows)
    return {"entries_in_run": len(new), "added": added, "ledger": len(rows)}


def _answers_of(work_dir: Path, job_id: str) -> dict[str, Any]:
    from . import language_questions

    return language_questions.load_answers(Path(work_dir) / "jobs" / job_id).get("evidence") or {}


def mark_answered(work_dir: Path, *, job_id: str, key: str, text: str) -> int:
    """The listener answered the console's question for this missing-evidence note: every entry of the job with that key becomes ``answered``.  Returns how many changed."""
    rows = load(work_dir)
    changed = 0
    for row in rows:
        if row.get("kind") == "missing_evidence" and row.get("job") == job_id and row.get("key") == key and row.get("outcome") in ("open", "answered"):
            row["outcome"], row["outcome_note"] = "answered", " ".join(str(text or "").split())[:600]
            changed += 1
    if changed:
        _save(work_dir, rows)
    return changed


def mark(work_dir: Path, entry_id: str, outcome: str, note: str = "") -> bool:
    """Record what happened to one entry afterwards (``contradicted``: the listener or a measurement overturned it).  False when there is no such entry."""
    if outcome not in OUTCOMES:
        raise ValueError(f"outcome must be one of {', '.join(OUTCOMES)}")
    rows = load(work_dir)
    for row in rows:
        if row["id"] == entry_id:
            row["outcome"], row["outcome_note"] = outcome, " ".join(str(note or "").split())[:600]
            _save(work_dir, rows)
            return True
    return False


def runs_with_thinking(work_dir: Path) -> list[tuple[str, str, str]]:
    """``(run id, clip id, repeat)`` of every run folder that has a thinking.jsonl, oldest first."""
    found = []
    for path in sorted((Path(work_dir) / "experience" / "runs").glob("*/*/r*/thinking.jsonl")):
        repeat = path.parent.name
        if repeat[1:].isdigit():
            found.append((path.parents[2].name, path.parents[1].name, repeat[1:]))
    return found


def _job_of_clip(work_dir: Path, clip_id: str) -> str | None:
    matches = [p.name for p in (Path(work_dir) / "jobs").glob(f"{clip_id}*") if p.is_dir()]
    return matches[0] if len(matches) == 1 else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mathula_tv.experience_ledger")
    parser.add_argument("--job")
    parser.add_argument("--run")
    parser.add_argument("--clip", help="default: the first 8 characters of the job id")
    parser.add_argument("--all", action="store_true", help="every run that has a thinking.jsonl")
    args = parser.parse_args(argv)
    from .config import load_settings

    work_dir = Path(load_settings().work_dir)
    if args.all:
        for run_id, clip_id, repeat in runs_with_thinking(work_dir):
            job_id = _job_of_clip(work_dir, clip_id)
            if job_id is None:
                print(f"{run_id}/{clip_id}: no single job for this clip, skipped")
                continue
            print(f"{run_id}/{clip_id}: {ingest_run(work_dir, job_id=job_id, run_id=run_id, clip_id=clip_id, repeat=int(repeat))}")
        return 0
    if not (args.job and args.run):
        parser.error("give --job and --run, or --all")
    print(ingest_run(work_dir, job_id=args.job, run_id=args.run, clip_id=args.clip or args.job[:8]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
