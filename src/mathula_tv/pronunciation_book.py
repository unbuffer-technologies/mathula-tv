"""The pronunciation book: what the listener has decided, and why.

    spelling = dictionary sounds -> rules          (nothing here scores anything)
    the listener hears it and commits a spelling:
        the rules' own spelling         -> "approved": every job uses exactly it from then on
        any other spelling (an alternative of the rules, or the listener's own)
                                        -> "custom": every job uses it too, AND it is kept as feedback for the rules: the sounds the dictionary heard, how the rules
                                           built the spelling, what the listener wrote instead and how the two differ.  There is no "exception" that is set apart
                                           from the rules: a custom spelling is a lesson, and a pattern across them is what changes a rule.

The book is one file (``pronunciation_book.json``).  ``job_pronunciations`` reads it when it writes a job's lexicon, so a committed word reaches the voice before
anything the rules would write.
"""
from __future__ import annotations

import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

SCHEMA = "mathula-pronunciation-book-v1"
ALL_VOICES = "*"
_LOCK = threading.Lock()


def key_of(word: str) -> str:
    return re.sub(r"\s+", " ", str(word).strip()).casefold()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


KINDS = ("approved", "custom")


def _kind(kind: str) -> str:
    return "custom" if kind == "exception" else kind          # the book written before there was no such thing as an exception


def clean_spelling(spelling: Any) -> str:
    """A spelling is what the voice is asked to say for a word, and a comma or a full stop in it makes the voice pause or end the sentence ("khoonggres," came from a rule
    reading with a comma in it): they are dropped, and the spaces they leave are closed."""
    return re.sub(r"\s+", " ", re.sub(r"[,.]", "", str(spelling))).strip()


def load(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    entries = {key: {**entry, "kind": _kind(str(entry.get("kind"))), "spelling": {voice: clean_spelling(text) for voice, text in dict(entry.get("spelling") or {}).items()},
                     **({"rule_spelling": clean_spelling(entry["rule_spelling"])} if entry.get("rule_spelling") else {})}          # a book written before spellings were cleaned
               for key, entry in dict(data.get("entries") or {}).items()}
    return {"schema": SCHEMA, "entries": entries, "feedback": list(data.get("feedback") or []), "log": list(data.get("log") or [])}


def clean(path: Path) -> int:
    """Write the book back with every spelling cleaned (``clean_spelling``); returns how many spellings changed."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8")).get("entries") or {}
    except (OSError, ValueError):
        return 0
    with _LOCK:
        data = load(path)
        changed = sum(1 for key, entry in raw.items() for voice, text in dict(entry.get("spelling") or {}).items() if data["entries"][key]["spelling"][voice] != text)
        if changed:
            data["log"].append({"at": _now(), "action": "cleaned", "note": f"{changed} spelling(s) had a comma or a full stop, which the voice reads as a pause"})
            _save(path, data)
        return changed


def _save(path: Path, data: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({**data, "log": list(data["log"])[-1000:]}, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _valid_spelling(spelling: str) -> str:
    spelling = clean_spelling(spelling)
    if not spelling or len(spelling) > 120 or not re.fullmatch(r"[A-Za-z0-9'’ \-]+", spelling):
        raise ValueError("a spelling is letters, digits, hyphens and spaces, up to 120 characters")
    return spelling


def decide(path: Path, word: str, spelling: str, *, kind: str, voice: str = ALL_VOICES, rule_spelling: str = "", job: str = "", note: str = "", replace: bool = False) -> dict[str, Any]:
    """Commit a spelling for a word (``replace``: drop the spellings it already had for other voices).  ``kind`` is "approved" (it is what the rules wrote) or "custom"
    (anything else the listener chose).  Per voice when ``voice`` is given, else for both."""
    kind = _kind(kind)
    if kind not in KINDS:
        raise ValueError("kind is approved or custom")
    spelling = _valid_spelling(spelling)
    if not word.strip() or len(word) > 80:
        raise ValueError("a word is required")
    with _LOCK:
        data = load(path)
        entry = data["entries"].get(key_of(word)) or {"word": word.strip(), "spelling": {}, "jobs": []}
        entry.update(word=word.strip(), kind=kind, rule_spelling=rule_spelling or entry.get("rule_spelling", ""), decided_at=_now(), note=note)
        entry["spelling"] = {voice: spelling} if voice == ALL_VOICES or replace else {**entry.get("spelling", {}), voice: spelling}          # one spelling for both voices replaces any per-voice ones, which would still win over it
        if job and job not in entry["jobs"]:
            entry["jobs"].append(job)
        data["entries"][key_of(word)] = entry
        data["log"].append({"at": _now(), "action": kind, "word": word.strip(), "voice": voice, "spelling": spelling, "job": job, "note": note})
        _save(path, data)
        return entry


def revoke(path: Path, word: str) -> bool:
    with _LOCK:
        data = load(path)
        gone = data["entries"].pop(key_of(word), None)
        if gone:
            data["log"].append({"at": _now(), "action": "revoked", "word": gone.get("word"), "spelling": gone.get("spelling")})
            _save(path, data)
        return gone is not None


def spelling_for(entries: Mapping[str, Any], word: str, voice: str) -> str | None:
    """The committed spelling of ``word`` for ``voice`` (its own, else the one for both voices), or None."""
    entry = entries.get(key_of(word))
    if not entry:
        return None
    by_voice = entry.get("spelling") or {}
    return by_voice.get(voice) or by_voice.get(ALL_VOICES)


def shape(token: str) -> str:
    """A syllable's letters as runs: a run of vowels becomes ``V<length>``, consonants stay as written (``aaar`` -> ``V3r``, ``aarr`` -> ``V2rr``)."""
    return "".join(f"V{len(run)}" if run[0] in "aeiou" else run for run in re.findall(r"[aeiou]+|[^aeiou]+", token.lower()))


def letter_edits(a: str, b: str) -> str:
    """The letters that changed between two spellings of one syllable or word, as written (``ey→a, ∅→e``), and where the first change sits."""
    import difflib

    steps = [(tag, a[i1:i2], b[j1:j2], i1, i2) for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if tag != "equal"]
    if not steps:
        return ""
    _, _, _, i1, i2 = steps[0]
    where = "at the start" if i1 == 0 else "at the end" if i2 == len(a) and len(steps) == 1 else "inside"
    return ", ".join(f"{old or '∅'}→{new or '∅'}" for _, old, new, _, _ in steps) + f" ({where})"


def spelling_diff(ruled: str, chosen: str) -> list[dict[str, Any]]:
    """How the chosen spelling differs from the rules' (both written in syllables joined by hyphens): one row per syllable that changed, with the ``pattern`` that
    names the change so that the same change in different words is seen as one.  A spelling in syllables (an initialism, a number) is compared syllable by syllable
    and the pattern is the syllable's shape before and after (``V3r → V2rr``, last syllable); a word written as one piece is compared letter by letter (``ey→a``).
    Spellings with a different number of syllables are one change."""
    a, b = ruled.lower().split("-"), chosen.lower().split("-")
    if len(a) != len(b):
        return [{"from": ruled, "to": chosen, "shape": f"{len(a)} syllables → {len(b)} syllables", "where": "whole word", "pattern": f"{len(a)} syllables → {len(b)} syllables"}]
    rows = []
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            where = "only syllable" if len(a) == 1 else "first syllable" if i == 0 else "last syllable" if i == len(a) - 1 else "middle syllable"
            shapes = f"{shape(x)} → {shape(y)}"
            rows.append({"from": x, "to": y, "shape": shapes, "where": where, "pattern": letter_edits(x, y) if len(a) == 1 else f"{shapes} ({where})"})
    return rows


def record_feedback(path: Path, *, word: str, rule_spelling: str, chosen: str, differences: Iterable[Mapping[str, Any]] = (), job: str = "", note: str = "",
                    english: str = "", reading: str = "", derivation: Iterable[str] = ()) -> dict[str, Any]:
    """Keep a custom spelling as a lesson for the rules: the word, the sounds the dictionary heard (``english``), how the rules built their spelling
    (``derivation``), what the listener wrote instead, the rules' open choices on which it departs (``differences``, when it is one of the rules' alternatives)
    and the syllable-level ``diff`` that is there for every spelling.  The same word and spelling again updates its row."""
    chosen = _valid_spelling(chosen)
    row = {"at": _now(), "word": word.strip(), "job": job, "rule_spelling": rule_spelling, "chosen": chosen, "note": note, "english": english, "reading": reading,
           "derivation": [str(d) for d in derivation], "diff": spelling_diff(rule_spelling, chosen) if rule_spelling else [],
           "differences": [{k: d.get(k) for k in ("rule", "label", "main", "other", "main_means", "other_means")} for d in differences]}
    with _LOCK:
        data = load(path)
        data["feedback"] = [r for r in data["feedback"] if not (key_of(r.get("word", "")) == key_of(word) and r.get("chosen") == chosen and r.get("rule_spelling") == rule_spelling)]
        data["feedback"].append(row)
        data["log"].append({"at": row["at"], "action": "custom spelling kept for the rules", "word": row["word"], "spelling": chosen, "job": job, "note": note})
        _save(path, data)
    return row


def patterns(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The changes the listener made, grouped by what changed (the syllable's shape before and after, and where in the word): a pattern is a change seen in
    more than one word.  Most frequent first."""
    found: dict[str, dict[str, Any]] = {}
    for row in data.get("feedback", []):
        for change in row.get("diff") or []:
            key = change.get("pattern") or f"{change['shape']} ({change['where']})"
            entry = found.setdefault(key, {"pattern": key, "words": [], "examples": []})
            if row["word"] not in entry["words"]:
                entry["words"].append(row["word"])
                entry["examples"].append(f"{row['word']}: {row['rule_spelling']} → {row['chosen']}")
    return sorted(found.values(), key=lambda e: (-len(e["words"]), e["pattern"]))


def rule_tally(data: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """For each rule: how many committed words the rules got right as they stood, and how often the listener's chosen spelling needed another option.  Only
    counts of the listener's own decisions; nothing is measured on audio."""
    tally: dict[str, dict[str, Any]] = {}
    for row in data.get("feedback", []):
        for difference in row.get("differences") or []:
            rule = tally.setdefault(str(difference["rule"]), {"label": difference.get("label"), "wanted": {}, "words": {}})
            option = str(difference.get("other"))
            rule["wanted"][option] = rule["wanted"].get(option, 0) + 1
            rule["words"].setdefault(option, [])
            if row["word"] not in rule["words"][option]:
                rule["words"][option].append(row["word"])
    approved_as_ruled = sum(1 for e in data.get("entries", {}).values() if e.get("kind") == "approved" and e.get("spelling", {}).get(ALL_VOICES) == e.get("rule_spelling"))
    return {"rules": tally, "approved_as_ruled": approved_as_ruled, "custom": sum(1 for e in data.get("entries", {}).values() if e.get("kind") == "custom")}
