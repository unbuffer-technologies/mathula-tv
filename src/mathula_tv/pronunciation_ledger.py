"""The pronunciation ledger: for every word the dub's pronunciation could be about, what the agent was shown and what it did with it.

The harness records the facts (a word listed, a word measured with something named wrong, a spelling tried, a reading used) and the agent adds its own
reason for a word it checked and left or changed (``note_pronunciation``).  When a word sounds wrong in the finished dub, the ledger answers the
questions that used to need a hand-read trace: was the word ever put in front of the agent, with what measurement, and what did it decide and why?
A word that was shown with something named wrong and has no try, no reading and no note is ``unexamined``: a gap in the agent's attention or in
what the tools showed, to be found here and not guessed at.

It is written to ``<run>/pronunciation_ledger.json`` and summarised in the production report.
"""
from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

DECISIONS = ("kept_as_is", "changed", "not_a_problem")
MIN_REASON_CHARS = 8
FAR_CLOSENESS = 0.75           # a word the voice reads as written this far from the reference sounds is not "fine as it is" until a spelling has been heard: at 0.7 "High"
                               # (the voice says "heh" for the English "hi") sat exactly on the line and was kept for being part of a title
MAX_FAR_LISTED = 16
MAX_UNEXAMINED_LISTED = 24     # prod-63 and prod-64 listed the first 8: Heidi, Party, SG, MK, October, SABC, Jacob and Litchfield were cut from the one request and never asked about


def problems_of(row: Mapping[str, Any]) -> list[str]:
    """What a lab measurement names as wrong: differences from the speaker, an ending that differs, sounds lost in the sentence, and, for a reading
    saved by an earlier run, an English listener who does not write the word when the voice says it (prod-64 kept a saved "Eyai" for AI that a listener
    heard as "Ed.": the failure was a suggestion the agent could ignore, and now it is a named problem the harness asks about)."""
    problems = [str(d) for d in row.get("differences") or []]
    if row.get("ending"):
        problems.append(f"ending: {row['ending']}")
    if row.get("lost_in_its_sentence"):
        problems.append("in its sentence the voice loses " + " ".join(str(s) for s in row["lost_in_its_sentence"]))
    if row.get("spelling") and row.get("english_listener_understands") is False and row.get("english_hears") is not None:
        problems.append(f"an English listener writes {row['english_hears']!r} when the voice says the saved {row['spelling']!r}, not the word")
    return problems


def _key(word: str) -> str:
    from .pronunciation_lab import without_prefix, written_prefix

    english = word.split("-")[-1]
    return re.sub(r"[^a-z]", "", without_prefix(english, written_prefix(english)).casefold())


class Ledger:
    def __init__(self) -> None:
        self._words: dict[str, dict[str, Any]] = {}

    def _entry(self, word: str) -> dict[str, Any]:
        return self._words.setdefault(_key(word), {"word": word, "shown": [], "tried": [], "used": None, "note": None})

    def shown(self, word: str, where: str, *, closeness: float | None = None, problems: Sequence[str] = ()) -> None:
        entry = self._entry(word)
        row = {"where": where, **({"closeness": closeness} if closeness is not None else {}), "problems": list(problems)}
        if row not in entry["shown"]:
            entry["shown"].append(row)

    def tried(self, word: str, spellings: Sequence[str]) -> None:
        self._entry(word)["tried"].append(list(spellings))

    def used(self, word: str, spelling: str, closeness: float | None = None) -> None:
        self._entry(word)["used"] = {"spelling": spelling, **({"closeness": closeness} if closeness is not None else {})}

    def note(self, word: str, decision: str, why: str) -> dict[str, Any]:
        if decision not in DECISIONS:
            raise ValueError(f"decision must be one of {', '.join(DECISIONS)}")
        if len(why.strip()) < MIN_REASON_CHARS:
            raise ValueError("say why in a few words: a decision with no reason cannot be learned from")
        self._entry(word)["note"] = {"decision": decision, "why": why.strip()}
        return {"ok": True, "word": word, "decision": decision}

    def unexamined(self) -> list[dict[str, Any]]:
        """Words shown with something named wrong that the agent neither tried, used nor wrote a note about."""
        found = []
        for entry in self._words.values():
            problems = [p for row in entry["shown"] for p in row["problems"]]
            if problems and not entry["tried"] and not entry["used"] and not entry["note"]:
                found.append({"word": entry["word"], "problems": list(dict.fromkeys(problems))[:3]})
        return found

    def untried_far(self) -> list[dict[str, Any]]:
        """Words that measured under FAR_CLOSENESS as written and that no spelling was tried or used for, whatever reason was written for leaving them.
        prod-48 left "High" (0.47), "Party" (0.48) and "OK" (0.63) as they were because they were part of a title or a familiar word, while the speaker's
        sounds, not the kind of word, are what a reading is matched to.  Weakest first."""
        found = []
        for entry in self._words.values():
            levels = [row["closeness"] for row in entry["shown"] if row.get("closeness") is not None]
            if levels and min(levels) < FAR_CLOSENESS and not entry["tried"] and not entry["used"]:
                found.append({"word": entry["word"], "closeness": min(levels)})
        return sorted(found, key=lambda item: item["closeness"])

    def to_dict(self) -> dict[str, Any]:
        return {"words": sorted(self._words.values(), key=lambda e: e["word"].casefold()), "unexamined": self.unexamined(), "untried_far": self.untried_far()}

    def summary(self) -> dict[str, Any]:
        entries = list(self._words.values())
        return {"words": len(entries), "tried": sum(1 for e in entries if e["tried"]), "used": sum(1 for e in entries if e["used"]),
                "noted": sum(1 for e in entries if e["note"]), "unexamined": self.unexamined()}


def far_request_text(items: Sequence[Mapping[str, Any]]) -> str:
    """The one request about words that measure far from the speaker and were not tried."""
    listed = "; ".join(f"{item['word']} (closeness {item['closeness']})" for item in items[:MAX_FAR_LISTED])
    return ("One more thing before you finish. These words measure far from the speaker's own sounds as you wrote them, and no spelling was heard "
            f"for them: {listed}. Being part of a title, a familiar word or a name is not a reason to leave one: the speaker's sounds in the source "
            "audio are what the voice is matched to. For each, call try_pronunciation (or suggest_spellings) once and use what you hear. If no "
            "spelling is closer, keep the word and say so with note_pronunciation, citing what you tried. A word that is a real isiZulu word is not "
            "in this list, so there is no Zulu spelling to protect.")


def request_text(items: Sequence[Mapping[str, Any]]) -> str:
    """The one request, sent when the agent says it is done and words are still unexamined."""
    listed = "; ".join(f"{item['word']} ({'; '.join(item['problems'][:2])})" for item in items[:MAX_UNEXAMINED_LISTED])
    return ("One more thing before you finish. These words were put in front of you with something named wrong, and nothing was tried, changed or "
            f"recorded for them: {listed}. For each, either check it with try_pronunciation, or call note_pronunciation(word, decision, why) with what "
            "you decided and why. Leaving a word as it is is allowed; it needs a reason, so that the next run and the people reading the ledger can see it.")
