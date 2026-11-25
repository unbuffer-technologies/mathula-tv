"""What the listener has heard wrong in earlier dubs, as lessons the agent reads before it writes.

The pronunciation book keeps the listener's decisions word by word, and the rules take them, but the agent that wrote the text never learned from them: the same kind of defect ("Dkt" read raw,
digits read in Zulu-accented English, a name run together, a figure with no breath) came back run after run and was answered with a new rule.  A lesson closes the loop from the listener's ear to the
agent's next draft: what was written, what the listener heard, what fixed it, and which item class it belongs to.

Three sources, all merged when the agent calls ``recall_lessons`` (which it reliably does first; an optional tool was skipped):

* ``SEED``: the concepts the listener taught in words, classified by item class;
* ``working/ear_lessons.json``: lessons added since (``add``, or ``python -m mathula_tv.ear_lessons add ...`` after a job);
* the book: every spelling the listener changed by ear is a (rule's spelling -> chosen spelling) pair; the latest are shown as examples of how this listener respells.

Nothing here decides a spelling: it shows the agent what it has been told, and the ear stays the gate.

Every lesson carries where it came from: ``listener`` (said or decided by the listener), ``measured`` (measured on the voices) or ``inferred`` (the assistant's reading of several corrections: a
hypothesis).  Research on agents that keep lessons across runs warns that a stored lesson can be wrong and, once stored, spreads to every run; so a lesson has a source the agent can weigh, and a
status: ``retire`` takes one out of what the agent sees, with the reason kept.  A complaint can also add its line to the standing hard-text test (``tests/fixtures/hard_text.txt``).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FILE = "ear_lessons.json"
HARD_TEXT = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "hard_text.txt"          # one line per complaint: after the rules, nothing may be left as written
SOURCES = ("listener", "measured", "inferred")

SEED: list[dict[str, str]] = [
    {"id": "seed-fused-name-run", "source": "listener", "item_class": "speakability", "written": "u-Ayanda Nyathi nentatheli ephezulu u-Canny Maphanga", "heard": "the names and 'na-' fused into one rushed phrase; nentatheli rushed",
     "fix": "a run of names followed directly by a conjunction and another noun is said in one breath and rushed: give it room (a comma after the names, a different order, or a clause that carries the title)"},
    {"id": "seed-modifier-chain", "source": "listener", "item_class": "speakability", "written": "ukungaziphathi kahle okubucayi kwamaphoyisa", "heard": "the voice took the chain for one long compound and clipped its middle: kahle lost",
     "fix": "we are not writing for an expert Zulu speaker: the reader is a TTS model and the audience is general listeners, so write plain speakable Zulu, one idea to a clause. Do not stack a long verb-noun, an adverb and an adjective in one phrase: split it into two plain phrases or move the adverb; the listener hears the clipped word although nothing counted on the audio shows it, so only reading the construction catches it"},
    {"id": "seed-stacked-nouns", "source": "listener", "item_class": "grammar", "written": "bobabili ekolishi, u-Brigitte Mabandla Justice College", "heard": "terrible grammar: the college named twice and nothing saying they come from it",
     "fix": "say how the things relate with a verb or a relative clause (they come from, they work at) instead of stacking nouns and names, and name a place once for one mention; read the text for this"},
    {"id": "seed-title-raw", "source": "listener", "item_class": "title", "written": "sinoDkt Arthur Shopola", "heard": "the raw abbreviation, 'section what'",
     "fix": "write a title in full (uDokotela, Mnumzane) or with its full stop; the renderer now expands Mnu/Nkk/Nksz/Dkt with or without the stop, but a title it does not know is read raw"},
    {"id": "seed-digits-accent", "source": "listener", "item_class": "figure", "written": "31%, 'eSection 79', 'kuka-31'", "heard": "Zulu-accented English: 'titty one', 'seventhy nine'",
     "fix": "a figure the renderer does not convert reaches the voice as digits and is read badly; write figures with the Zulu prefix the sentence needs (they are said in English after it) and check say_it"},
    {"id": "seed-decimal-exhale", "source": "listener", "item_class": "figure", "written": "ku-44.05%", "heard": "each digit after the point said as its own falling phrase, an exhale",
     "fix": "the listener respells the digit chain by hand (foty-foo-phoynt-zeeroh-faaaivv); avoid decimals that add nothing (round when the speaker rounded)"},
    {"id": "seed-final-consonant", "source": "inferred", "item_class": "spelling", "written": "eyt, fayiv, pakt, Moonshot, Economic", "heard": "a hard word-final consonant (t, v, k, h) released with a breath",
     "fix": "the listener's fixes give the end a vowel or length: eyt -> eyit, fayiv -> faaaivv, pakt -> phekt, muunshot -> muunshooth, ekonoomik -> ekonoomiki"},
    {"id": "seed-name-flow", "source": "listener", "item_class": "name", "written": "eNorth West University (spaces between the words)", "heard": "three separate phrases, not flowing",
     "fix": "the words of one English name are said as one hyphenated chain (enooth-west-yuni-vesithi); write a name once, whole, with its capitals"},
    {"id": "seed-unit-twice", "source": "listener", "item_class": "figure", "written": "ngamaphesenti angaphezu kuka-31%", "heard": "'amaphesenti' twice",
     "fix": "the unit word is said once: after 'amaphesenti' a figure with % needs no unit of its own"},
    {"id": "seed-figures-run-together", "source": "listener", "item_class": "pacing", "written": "ku-75.8% ngo-2006 laya ku-44.05%", "heard": "rushing through the numbers, no time to digest",
     "fix": "a comma after each percentage or figure that carries the point; never two figures with no breath between them"},
    {"id": "seed-year-zero", "source": "listener", "item_class": "year", "written": "ngo-2006", "heard": "the 0 in 06 badly pronounced ('oh')", "fix": "the listener chose twenty-ooo-six"},
    {"id": "seed-name-spelling", "source": "listener", "item_class": "name", "written": "Thabiso Mos (pass 1 took the caption's spelling over the recogniser's Moss)", "heard": "not a real name",
     "fix": "a name spelling that two sources disagree on is checked on the web; do not trust the YouTube auto-caption's spelling of a name over the recogniser's"},
    {"id": "seed-vowel-clash", "source": "listener", "item_class": "figure", "written": "abangu-eleven", "heard": "u before e read as one clash",
     "fix": "kept as written by the listener: the corpus has abangu-11 and abayi-11 equally; do not elide the English word's vowel (no 'leven')"},
    {"id": "seed-names-of-address", "source": "listener", "item_class": "pacing", "written": "Impela, Ayanda, sethemba sizozithola izimpendulo (a comma on each side of the name)",
     "heard": "an awkward gap: people talking to each other do not stop round a name. The source speaker said 'Indeed, Ayanda, hopefully' with no pause before or after Ayanda (measured: 0.0 s)",
     "fix": "a name of address is said in the same breath as the words round it: no comma before it, usually none after it; a pause only where the speaker's thought changes (the phrasing in open_clip shows where): write commas where the speaker breathed, not from habit"},
    {"id": "seed-spacing-silence", "source": "measured", "item_class": "pacing", "written": "spaces or hyphens inside a number", "heard": "no difference in pauses (measured)",
     "fix": "punctuation sets the pause (comma-class ~0.15 s, sentence end ~0.9 s); a hyphen joins pieces into one word but adds no silence"},
]


def _read(work_dir: Path) -> list[dict[str, Any]]:
    try:
        return list(json.loads((Path(work_dir) / FILE).read_text(encoding="utf-8")).get("lessons") or [])
    except (OSError, ValueError):
        return []


def add_hard_text(line: str, path: Path | None = None) -> bool:
    """Append a complaint's line to the standing hard-text test (once).  Returns whether it was added."""
    path = Path(path or HARD_TEXT)
    line = " ".join(str(line).split())
    existing = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    if not line or line in existing:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([*existing, line]) + "\n", encoding="utf-8")
    return True


def add(work_dir: Path, *, item_class: str, written: str, heard: str, fix: str, job: str = "", source: str = "listener", hard_text: str = "") -> dict[str, Any]:
    """Record one lesson (idempotent on class + written + heard).  ``source`` is where it came from (listener / measured / inferred); ``hard_text``, when given, is the line that went
    wrong and joins the standing test.  Returns the stored lesson."""
    if source not in SOURCES:
        raise ValueError(f"source must be one of {', '.join(SOURCES)}")
    if hard_text:
        add_hard_text(hard_text)
    lessons = _read(work_dir)
    lesson = {"id": f"{item_class}-{abs(hash((item_class, written, heard))) % 10**8:08d}", "item_class": item_class, "written": written.strip(), "heard": heard.strip(), "fix": fix.strip(),
              "job": job, "source": source, "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    for existing in lessons:
        if (existing.get("item_class"), existing.get("written"), existing.get("heard")) == (lesson["item_class"], lesson["written"], lesson["heard"]):
            return existing
    lessons.append(lesson)
    path = Path(work_dir) / FILE
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"schema": "mathula-ear-lessons-v1", "lessons": lessons}, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return lesson


def retire(work_dir: Path, lesson_id: str, reason: str) -> None:
    """Take a lesson out of what the agent sees (a seed or an added one), keeping the reason."""
    path = Path(work_dir) / FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {"schema": "mathula-ear-lessons-v1", "lessons": []}
    data.setdefault("retired", {})[lesson_id] = {"reason": reason.strip(), "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _retired(work_dir: Path) -> dict[str, Any]:
    try:
        return dict(json.loads((Path(work_dir) / FILE).read_text(encoding="utf-8")).get("retired") or {})
    except (OSError, ValueError):
        return {}


def spelling_decisions(book_entries: dict[str, Any], limit: int = 25) -> list[dict[str, str]]:
    """The listener's own respellings, newest first: what the rules wrote and what was chosen instead (only where they differ)."""
    rows = []
    for key, entry in book_entries.items():
        if entry.get("kind") != "custom":
            continue
        chosen = entry.get("spelling")
        chosen = (chosen.get("*") or next(iter(chosen.values()), "")) if isinstance(chosen, dict) else str(chosen or "")
        rule = str(entry.get("rule_spelling") or "")
        if chosen and rule and chosen != rule:
            rows.append((str(entry.get("decided_at") or ""), {"word": str(entry.get("word") or key), "rules_wrote": rule, "listener_chose": chosen}))
    rows.sort(key=lambda row: row[0], reverse=True)
    return [row for _, row in rows[:limit]]


def _experience(work_dir: Path) -> dict[str, Any]:
    """Lessons compacted from the agent's own earlier runs (``experience_compact``): hypotheses from several videos, never rules."""
    try:
        from . import experience_compact

        found = experience_compact.for_agent(work_dir)
    except Exception:                                                  # noqa: BLE001 - lessons are advice: they must not break the first call of a run
        return {}
    return {"experience_lessons": found, "experience_note": ("What your own earlier runs showed across several videos (compacted from your recorded choices, strategies and missing-evidence notes). "
                                                              "They are hypotheses: use them where the situation matches, and a counter-example in this clip outweighs one.")} if found else {}


def for_agent(work_dir: Path, book_path: Path | None = None, *, limit_decisions: int = 25) -> dict[str, Any]:
    """Everything the agent should know before it writes: the concept lessons (seed + added) and the listener's latest respellings."""
    from .pronunciation_book import load as load_book

    work_dir = Path(work_dir)
    seen = {lesson["id"] for lesson in SEED}
    gone = _retired(work_dir)
    concepts = [lesson for lesson in [*SEED, *[lesson for lesson in _read(work_dir) if lesson.get("id") not in seen]] if lesson.get("id") not in gone]
    notes: list[dict[str, str]] = []
    try:
        book = load_book(book_path or work_dir / "pronunciation_book.json")
        decisions = spelling_decisions(book["entries"], limit_decisions)
        notes = [{"word": str(f.get("word") or ""), "note": str(f.get("note")).strip()} for f in reversed(book.get("feedback") or []) if str(f.get("note") or "").strip()][:10]
    except Exception:                                                  # noqa: BLE001 - lessons are advice: a missing book must not break the first call of a run
        decisions = []
    return {
        "note": ("What the LISTENER has heard wrong in earlier dubs, by item class: what was written, what was heard, what fixed it. They come from the listener's own corrections. Write your first "
                 "draft so these do not recur, and check the spoken form with say_it. 'listener_respellings' are spellings the listener chose over the rules' (the rules already apply them: "
                 "they show how this listener hears). 'source' says where a lesson came from: listener = said or decided by the listener; measured = measured on the voices; inferred = "
                 "an assistant's reading of several corrections, a hypothesis: use it, but a counter-example outweighs it."),
        "concepts": [{k: v for k, v in lesson.items() if k in ("item_class", "source", "written", "heard", "fix")} for lesson in concepts],
        "listener_respellings": decisions,
        **({"listener_notes": notes} if notes else {}),          # the listener's own words on a spelling, newest first
        **_experience(work_dir),
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Add a lesson about what the listener heard wrong.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add")
    for name in ("item-class", "written", "heard", "fix"):
        a.add_argument(f"--{name}", required=True)
    a.add_argument("--job", default="")
    a.add_argument("--source", default="listener", choices=SOURCES)
    a.add_argument("--hard-text", default="", help="the line that went wrong: it joins the standing hard-text test")
    r = sub.add_parser("retire")
    r.add_argument("--id", required=True)
    r.add_argument("--reason", required=True)
    r.add_argument("--work-dir", default=str(Path.home() / "mathula-tv" / "working"))
    a.add_argument("--work-dir", default=str(Path.home() / "mathula-tv" / "working"))
    sub.add_parser("show").add_argument("--work-dir", default=str(Path.home() / "mathula-tv" / "working"))
    args = ap.parse_args(argv)
    if args.cmd == "add":
        print(json.dumps(add(Path(args.work_dir), item_class=args.item_class, written=args.written, heard=args.heard, fix=args.fix, job=args.job, source=args.source,
                             hard_text=args.hard_text), ensure_ascii=False, indent=1))
    elif args.cmd == "retire":
        retire(Path(args.work_dir), args.id, args.reason)
        print(f"retired {args.id}")
    else:
        print(json.dumps(for_agent(Path(args.work_dir)), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
