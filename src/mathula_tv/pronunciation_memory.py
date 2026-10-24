"""Pronunciation memory shared by every job.

What one job learns about how the voice says a word is kept here and applied to the next job:

* a swallowed word that a repair brought back (``source="repair"``), stored only after the real render's check confirmed it;
* a name or word whose reading a person approved (``source="person"``), stored with the CLI.

Each entry is for one voice: a respelling verified on Themba says nothing about Thando. Each job that uses an entry is a
test of it: when the post-render check finds the word swallowed again, the entry counts a failure, and two failures switch
it off. A person's entry outranks a repair, and a repair outranks nothing else here.

The file is ``<work_dir>/experience/pronunciation_memory.json``. Jobs receive its entries through their own
``native_dub/pronunciation_names.json`` lexicon, so no renderer code needs to know about it.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA = "mathula-pronunciation-memory-v1"
MEMORY_SOURCE = "memory"
MAX_FAILURES = 2
_PRIORITY = {"person": 2, "repair": 1}


def memory_path(work_dir: Path) -> Path:
    return Path(work_dir) / "experience" / "pronunciation_memory.json"


def _key(display_text: str, voice: str | None) -> str:
    return f"{display_text.casefold()}|{voice or '*'}"


def load(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        return {"schema": SCHEMA, "entries": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("entries", {})
    return data


def _save(path: Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(data, indent=1, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def remember(path: Path, display_text: str, tts_text: str, *, source: str, voice: str | None, job_id: str,
             kind: str = "other", note: str = "") -> dict[str, Any]:
    """Record a verified reading. Same reading again = one more confirming job; a different one replaces it only from an
    equal or higher-priority source."""
    if source not in _PRIORITY:
        raise ValueError(f"source must be one of {sorted(_PRIORITY)}")
    data = load(path)
    key = _key(display_text, voice)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    entry = data["entries"].get(key)
    if entry and entry["tts_text"] == tts_text:
        entry["jobs"] = sorted({*entry["jobs"], job_id})
        entry["source"] = max(entry["source"], source, key=_PRIORITY.get)
        entry["disabled"] = False if source == "person" else entry.get("disabled", False)
    elif entry and _PRIORITY[source] < _PRIORITY[entry["source"]] and not entry.get("disabled"):
        return entry                                          # a person's reading is not overwritten by a repair
    else:
        entry = {"display_text": display_text, "tts_text": tts_text, "voice": voice, "kind": kind, "source": source,
                 "jobs": [job_id], "failures": 0, "disabled": False, "first_seen": now, "note": note}
    entry["last_confirmed"] = now
    data["entries"][key] = entry
    _save(path, data)
    return entry


def entries_for(path: Path, voice: str | None) -> list[dict[str, Any]]:
    """The enabled entries that apply to ``voice`` (entries with no voice apply to all)."""
    return [e for e in load(path)["entries"].values()
            if not e.get("disabled") and (e.get("voice") in (None, voice))]


def apply_to_lexicon(lexicon_path: Path, memory_file: Path, voice: str | None) -> list[str]:
    """Add the memory's entries to a job's names lexicon (replacing an automatic reading of the same text). Returns the texts added."""
    lexicon_path = Path(lexicon_path)
    lexicon = json.loads(lexicon_path.read_text(encoding="utf-8")) if lexicon_path.is_file() else {"schema": "mathula-names-lexicon-v1", "entries": []}
    wanted = entries_for(memory_file, voice)
    taken = {e["display_text"].casefold() for e in wanted}
    kept = [e for e in lexicon.get("entries", []) if e["display_text"].casefold() not in taken]
    added = [{"display_text": e["display_text"], "tts_text": e["tts_text"], "kind": e.get("kind", "other"), "confidence": 0.9,
              "source": MEMORY_SOURCE, "memory_source": e["source"]} for e in wanted]
    lexicon["entries"] = [*kept, *added]
    lexicon_path.write_text(json.dumps(lexicon, indent=1, ensure_ascii=False), encoding="utf-8")
    return [e["display_text"] for e in added]


def record_outcomes(lexicon_path: Path, memory_file: Path, voice: str | None, dropped: Sequence[Mapping[str, Any]]) -> list[str]:
    """After a job's post-render check: an applied memory entry whose word was swallowed anyway counts a failure.
    Returns the entries that were switched off."""
    lexicon_path = Path(lexicon_path)
    if not lexicon_path.is_file() or not dropped:
        return []
    applied = {e["display_text"].casefold() for e in json.loads(lexicon_path.read_text(encoding="utf-8")).get("entries", [])
               if e.get("source") == MEMORY_SOURCE}
    swallowed = {re.sub(r"[^a-z]", "", w.casefold()) for item in dropped for w in item.get("words", [])}
    data = load(memory_file)
    switched_off: list[str] = []
    for key, entry in data["entries"].items():
        respelt = re.sub(r"[^a-z]", "", entry["tts_text"].casefold())
        if entry["display_text"].casefold() in applied and entry.get("voice") in (None, voice) and (
                respelt in swallowed or re.sub(r"[^a-z]", "", entry["display_text"].casefold()) in swallowed):
            entry["failures"] = entry.get("failures", 0) + 1
            if entry["failures"] >= MAX_FAILURES and entry["source"] != "person":
                entry["disabled"] = True
                switched_off.append(entry["display_text"])
    _save(memory_file, data)
    return switched_off


def apply_for_job(job_root: Path) -> list[str]:
    """Give a job the memory's readings for its voice (called at the end of the names stage)."""
    lexicon = Path(job_root) / "native_dub" / "pronunciation_names.json"
    voice = json.loads(lexicon.read_text(encoding="utf-8")).get("voice") if lexicon.is_file() else None
    return apply_to_lexicon(lexicon, memory_path(Path(job_root).parents[1]), voice)


def record_job_outcomes(job_root: Path, check_report: Mapping[str, Any]) -> list[str]:
    """Feed a job's post-render intelligibility report back into the memory entries that job used."""
    lexicon = Path(job_root) / "native_dub" / "pronunciation_names.json"
    if not lexicon.is_file():
        return []
    voice = json.loads(lexicon.read_text(encoding="utf-8")).get("voice")
    dropped = [item for unit in check_report.get("units") or [] for item in unit.get("dropped") or []]
    return record_outcomes(lexicon, memory_path(Path(job_root).parents[1]), voice, dropped)


def main(argv: list[str] | None = None) -> int:
    import argparse

    from .config import load_settings

    parser = argparse.ArgumentParser(prog="mathula_tv.pronunciation_memory")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    add = sub.add_parser("remember", help="record a reading a person approved")
    add.add_argument("display_text")
    add.add_argument("tts_text")
    add.add_argument("--voice", default="zu-ZA-ThembaNeural")
    add.add_argument("--kind", default="personal_name")
    forget = sub.add_parser("forget")
    forget.add_argument("display_text")
    forget.add_argument("--voice", default="zu-ZA-ThembaNeural")
    args = parser.parse_args(argv)
    path = memory_path(load_settings().work_dir)
    if args.command == "list":
        for entry in load(path)["entries"].values():
            state = "OFF" if entry.get("disabled") else "on"
            print(f"{state:>3} {entry['source']:<6} {entry['display_text']!r} -> {entry['tts_text']!r} [{entry.get('voice')}] "
                  f"jobs={len(entry['jobs'])} failures={entry.get('failures', 0)}")
    elif args.command == "remember":
        remember(path, args.display_text, args.tts_text, source="person", voice=args.voice, job_id="manual", kind=args.kind)
        print("remembered")
    else:
        data = load(path)
        removed = data["entries"].pop(_key(args.display_text, args.voice), None)
        _save(path, data)
        print("forgotten" if removed else "no such entry")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
