"""Pronunciation memory shared by every job.

What one job learns about how the voice says a word is kept here and applied to the next job:

* a spelling the dubbing agent chose with the pronunciation lab (``source="agent"``, lowest priority: a person's reading and a verified
  repair outrank it);
* a swallowed word that a repair brought back (``source="repair"``), stored only after the real render's check confirmed it.

No person puts a reading here: the agent has to learn each word from the audio, and every entry is rechecked by later jobs.  Each entry is
for one voice: a respelling verified on Themba says nothing about Thando. Each job that uses an entry is a test of it: when the post-render
check finds the word swallowed again, the entry counts a failure, and two failures switch it off. A verified repair outranks an agent's choice.

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
_PRIORITY = {"repair": 1, "agent": 0}
CLOSENESS_MARGIN = 0.02        # a reading that measures this much closer to the speaker is not replaced by one that measures less close


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


def saved(path: Path, display_text: str, voice: str | None) -> dict[str, Any] | None:
    """The memory's enabled entry for this word and voice, or None."""
    entry = load(path)["entries"].get(_key(display_text, voice))
    return entry if entry and not entry.get("disabled") else None


def beaten_by_saved(entry: Mapping[str, Any] | None, tts_text: str, closeness: float | None) -> bool:
    """Whether the saved reading measured closer to the speaker than this one: then the saved one stays (a run that ran out of turns once
    replaced a closeness 1.0 reading of "Fighters" with a 0.94 one that lost the final s)."""
    if not entry or closeness is None or entry.get("closeness") is None or str(entry["tts_text"]).casefold() == tts_text.casefold():
        return False
    return float(closeness) < float(entry["closeness"]) - CLOSENESS_MARGIN


def remember(path: Path, display_text: str, tts_text: str, *, source: str, voice: str | None, job_id: str,
             kind: str = "other", note: str = "", closeness: float | None = None) -> dict[str, Any]:
    """Record a verified reading. Same reading again = one more confirming job; a different one replaces it only from an
    equal or higher-priority source, and not when the saved one measured closer to the speaker (``closeness``).  Only the agent's
    choices and verified repairs can be recorded: a person's reading is refused."""
    if source not in _PRIORITY:
        raise ValueError(f"source must be one of {sorted(_PRIORITY)}")
    data = load(path)
    key = _key(display_text, voice)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    entry = data["entries"].get(key)
    if entry and entry["tts_text"] == tts_text:
        entry["jobs"] = sorted({*entry["jobs"], job_id})
        entry["source"] = max(entry["source"], source, key=lambda name: _PRIORITY.get(name, -1))
        if closeness is not None:
            entry["closeness"] = float(closeness)
    elif entry and _PRIORITY[source] < _PRIORITY[entry["source"]] and not entry.get("disabled"):
        return entry                                          # a verified repair is not overwritten by an agent's choice
    elif entry and not entry.get("disabled") and _PRIORITY[source] <= _PRIORITY[entry["source"]] and beaten_by_saved(entry, tts_text, closeness):
        return entry                                          # a reading that measured closer stays
    else:
        entry = {"display_text": display_text, "tts_text": tts_text, "voice": voice, "kind": kind, "source": source,
                 "jobs": [job_id], "failures": 0, "disabled": False, "first_seen": now, "note": note,
                 **({"closeness": float(closeness)} if closeness is not None else {})}
    entry["last_confirmed"] = now
    data["entries"][key] = entry
    _save(path, data)
    return entry


def entries_for(path: Path, voice: str | None) -> list[dict[str, Any]]:
    """The enabled entries that apply to ``voice`` (entries with no voice apply to all)."""
    return [e for e in load(path)["entries"].values()
            if not e.get("disabled") and (e.get("voice") in (None, voice))]


def apply_to_lexicon(lexicon_path: Path, memory_file: Path, voice: str | None) -> list[str]:
    """Make a job's names lexicon carry the memory's entries (replacing an automatic reading of the same text). Returns the texts added.

    Readings an earlier call copied from the memory are dropped first: an entry that has since left the memory (cleared, or switched off)
    must not go on being spoken from a job's stale copy."""
    lexicon_path = Path(lexicon_path)
    lexicon = json.loads(lexicon_path.read_text(encoding="utf-8")) if lexicon_path.is_file() else {"schema": "mathula-names-lexicon-v1", "entries": []}
    wanted = entries_for(memory_file, voice)
    taken = {e["display_text"].casefold() for e in wanted}
    kept = [e for e in lexicon.get("entries", []) if e["display_text"].casefold() not in taken and e.get("source") != MEMORY_SOURCE]
    from .pronunciation_lab import written_prefix

    def spoken(entry: Mapping[str, Any]) -> str:
        """The Zulu prefix glued on the word in the text ("we" of "weScience") stays on its spoken form: a reading respells the English word, not
        the grammar (an entry saved with only the English part would drop the prefix and the sentence would lose its "of")."""
        prefix, text = written_prefix(str(entry["display_text"])), str(entry["tts_text"])
        return prefix + text if prefix and entry.get("kind") == "english_code_switch" and not text.casefold().startswith(prefix.casefold()) else text

    added = [{"display_text": e["display_text"], "tts_text": spoken(e), "kind": e.get("kind", "other"), "confidence": 0.9,
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
            if entry["failures"] >= MAX_FAILURES:
                entry["disabled"] = True
                switched_off.append(entry["display_text"])
    _save(memory_file, data)
    return switched_off


def restore_written_words(items: Sequence[Mapping[str, Any]], entries: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Put the word back where a unit's text carries its respelling instead.

    A reading is applied when the voice speaks the word; the text keeps the word itself (it is what the captions show, and the only form the
    reading can find again).  An agent that wrote the chosen spelling into the text to hear it ("Fayithaz" for "maFighter") gets the word back,
    with a Zulu prefix glued on kept as it was written.  Returns the units (copies) and a plain sentence for each change."""
    from .pronunciation_lab import without_prefix, written_prefix

    respelt = [(str(e["display_text"]), str(e["tts_text"])) for e in entries
               if e.get("kind") == "english_code_switch" and not e.get("disabled") and len(str(e.get("tts_text") or "")) >= 4
               and str(e["tts_text"]).casefold() != str(e["display_text"]).casefold()]
    notes: list[str] = []
    restored: list[dict[str, Any]] = []
    for item in items:
        item = dict(item)
        text = str(item.get("zulu_text") or "")
        for display, spelling in respelt:
            pattern = re.compile(r"(?<![A-Za-z])((?:[a-z]{1,5}(?=[A-Z]))?)((?i:" + re.escape(spelling) + r"))(?![A-Za-z])")

            def put_back(match: re.Match[str], display: str = display) -> str:
                word = without_prefix(display, written_prefix(display)) if match.group(1) else display
                return match.group(1) + (word[:1].upper() + word[1:] if match.group(2)[:1].isupper() else word)

            changed = pattern.sub(put_back, text)
            if changed != text:
                notes.append(f"{item.get('item_id')}: the text had the respelling {spelling!r}; it keeps the word {display!r} (the voice respells it when speaking)")
                text = changed
        item["zulu_text"] = text
        restored.append(item)
    return restored, notes


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
    forget = sub.add_parser("forget", help="remove an entry (there is no way to add one by hand: the agent learns the readings)")
    forget.add_argument("display_text")
    forget.add_argument("--voice", default="zu-ZA-ThembaNeural")
    args = parser.parse_args(argv)
    path = memory_path(load_settings().work_dir)
    if args.command == "list":
        for entry in load(path)["entries"].values():
            state = "OFF" if entry.get("disabled") else "on"
            print(f"{state:>3} {entry['source']:<6} {entry['display_text']!r} -> {entry['tts_text']!r} [{entry.get('voice')}] "
                  f"jobs={len(entry['jobs'])} failures={entry.get('failures', 0)}")
    else:
        data = load(path)
        removed = data["entries"].pop(_key(args.display_text, args.voice), None)
        _save(path, data)
        print("forgotten" if removed else "no such entry")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
