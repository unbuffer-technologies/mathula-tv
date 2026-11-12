"""What the listener has decided, replayed against the rules: which open choice of the rules, if changed, would have written more of the listener's own spellings without
unwriting the ones the listener approved.

The pronunciation book holds two kinds of decision per word: "approved" (the rules' spelling was right) and "custom" (the listener wrote another).  Those decisions are the
only ground truth the rules have, and they are the listener's ear, not a model's: nothing here listens, scores audio or asks a model.  For each decision the word is spelled again
by the rules as they stand and compared with what the listener decided; then each open choice of the rules (``zulu_rules.RULE_LABELS``) is tried at each of its other options, one at
a time, and the result is counted:

    fixed   a custom spelling the rules now write            (the rules learned a word)
    broken  an approved spelling the rules no longer write   (the rules unlearned one)

A change is worth proposing when it fixes more than it breaks and the listener hears the words it changes and agrees.  This module only measures and lists; changing a default is the
listener's decision on the console's Rules page (``zulu_rules.write_rule_default``).

    python -m mathula_tv.rule_lessons            # the replay as the rules stand, and every single-choice change
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from . import pronunciation_book as book
from . import zulu_rules

VOICE = "zu-ZA-ThandoNeural"          # the voices are spelled alike by the rules; a decision for one voice only is compared on this one


def decisions(entries: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One row per word the listener decided: the word, whether the rules' spelling was approved or another was written, the spelling, and the jobs it was heard in."""
    rows = []
    for key, entry in entries.items():
        decided = book.spelling_for(entries, entry.get("word") or key, VOICE)
        if decided and not re.fullmatch(r"\d+(?:st|nd|rd|th)", key):          # a day (4th) is spelled by the number rule, not by the interpreter this replays
            rows.append({"word": entry.get("word") or key, "kind": entry.get("kind"), "decided": decided, "jobs": list(entry.get("jobs") or []), "rule_then": entry.get("rule_spelling") or ""})
    return sorted(rows, key=lambda r: (r["kind"] != "custom", r["word"].casefold()))


def judge(rows: Sequence[Mapping[str, Any]], spell: Callable[[str, Sequence[str]], str | None]) -> list[dict[str, Any]]:
    """Each decision with the spelling the rules write now (``spell(word, jobs)``; None when the rules have no reading) and whether it is the listener's."""
    out = []
    for row in rows:
        ruled = spell(row["word"], row["jobs"])
        out.append({**row, "rules_now": ruled, "agrees": ruled is not None and ruled.casefold() == str(row["decided"]).casefold()})
    return out


def tally(judged: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    approved = [r for r in judged if r["kind"] == "approved"]
    custom = [r for r in judged if r["kind"] == "custom"]
    return {"approved": len(approved), "approved_kept": sum(1 for r in approved if r["agrees"]), "custom": len(custom), "custom_written": sum(1 for r in custom if r["agrees"])}


def sweep(run: Callable[[Mapping[str, Any]], list[dict[str, Any]]], labels: Mapping[str, tuple[str, Mapping[Any, str]]] = zulu_rules.RULE_LABELS,
          current: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """Try each open choice at each option other than the present default, one change at a time.  ``run(defaults)`` judges every decision with those defaults.  Returns the changes
    that alter anything, best first: by words fixed less words broken, then by fewer words changed."""
    current = dict(current or {})
    base = {r["word"]: r for r in run(current)}
    found = []
    for name, (label, options) in labels.items():
        now = current.get(name, next(iter(options)))
        for option, means in options.items():
            if option == now:
                continue
            after = {r["word"]: r for r in run({**current, name: option})}
            fixed = [w for w, r in after.items() if r["agrees"] and not base[w]["agrees"] and r["kind"] == "custom"]
            broken = [w for w, r in after.items() if not r["agrees"] and base[w]["agrees"] and r["kind"] == "approved"]
            moved = [w for w, r in after.items() if r["rules_now"] != base[w]["rules_now"]]
            if moved:
                found.append({"rule": name, "label": label, "option": option, "means": means, "fixes": fixed, "breaks": broken, "changes": len(moved), "net": len(fixed) - len(broken),
                              "examples": {w: f"{base[w]['rules_now']} → {after[w]['rules_now']} (you: {after[w]['decided']})" for w in moved[:8]}})
    return sorted(found, key=lambda c: (-c["net"], c["changes"], c["rule"]))


def _runner(work_dir: Path, rows: list[dict[str, Any]]) -> Callable[[Mapping[str, Any]], list[dict[str, Any]]]:
    """Judge ``rows`` with the real interpreter, each word in the job it was heard in (a job's web research can change a name's sounds), under a scratch rules config."""
    from .job_pronunciations import Prepared

    prepared: dict[str, Any] = {}

    def interpreter(jobs: Sequence[str]) -> Any:
        for job in [*jobs, *sorted(p.name for p in (work_dir / "jobs").iterdir() if (p / "native_dub").is_dir())]:
            root = work_dir / "jobs" / job
            if root.is_dir():
                if job not in prepared:
                    prepared[job] = Prepared(root).interpreter
                return prepared[job]
        raise RuntimeError("no job to take the dictionary from")

    def run(defaults: Mapping[str, Any]) -> list[dict[str, Any]]:
        with tempfile.TemporaryDirectory() as scratch:
            config = Path(scratch) / "rules_config.json"
            config.write_text(json.dumps({"defaults": dict(defaults)}), encoding="utf-8")
            before = os.environ.get(zulu_rules.RULES_CONFIG_ENV)
            os.environ[zulu_rules.RULES_CONFIG_ENV] = str(config)
            try:
                def spell(word: str, jobs: Sequence[str]) -> str | None:
                    found = interpreter(jobs).rules_spelling(word)
                    return found["spelling"] if found else None

                return judge(rows, spell)
            finally:
                if before is None:
                    os.environ.pop(zulu_rules.RULES_CONFIG_ENV, None)
                else:
                    os.environ[zulu_rules.RULES_CONFIG_ENV] = before
                zulu_rules._config_seen.update(path=None, mtime=None)

    return run


def report(work_dir: Path) -> dict[str, Any]:
    """The replay as the rules stand (with the listener's present defaults) and every single-choice change, from the book of ``work_dir``."""
    work_dir = Path(work_dir)
    rows = decisions(book.load(work_dir / "pronunciation_book.json")["entries"])
    run = _runner(work_dir, rows)
    current = zulu_rules.rule_defaults()
    judged = run(current)
    return {"defaults": current, "tally": tally(judged), "decisions": judged, "changes": sweep(run, current=current)}


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    from .config import load_settings

    parser = argparse.ArgumentParser(prog="mathula_tv.rule_lessons")
    parser.add_argument("--json", help="also write the whole report here")
    args = parser.parse_args(argv)
    result = report(Path(load_settings().work_dir))
    t = result["tally"]
    print(f"defaults: {result['defaults'] or 'the rules as written'}")
    print(f"approved spellings the rules still write: {t['approved_kept']}/{t['approved']}   custom spellings the rules now write: {t['custom_written']}/{t['custom']}")
    print("\nthe listener's words the rules do not write:")
    for r in result["decisions"]:
        if not r["agrees"]:
            print(f"  {r['kind']:8s} {r['word']:14s} rules {str(r['rules_now']):22s} you {r['decided']}")
    print("\nsingle-choice changes (fixes - breaks):")
    for c in result["changes"]:
        print(f"  {c['net']:+d}  {c['rule']}={c['option']!r} ({c['means']}): fixes {c['fixes']}, breaks {c['breaks']}, changes {c['changes']} word(s)")
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
