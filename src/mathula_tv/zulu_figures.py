"""How professional translators wrote a figure in isiZulu: the prefix, the unit word, or the figure spelled out.

The dub agent writes a figure with the isiZulu prefix the sentence needs ("kuka-31%", "abantu abangu-11", "ngo-2006", "eminyakeni engu-20") and the renderer reads it in English, so the
choices that are the writer's are the prefix and the unit word.  Those are learnt from the aligned English-isiZulu pairs (``human_translations``), not from rules: the English phrase is turned
into a shape ("31 percent" -> two digits, then "percent"), the pairs whose English has that shape are found, and each pair's isiZulu is read for the same number: what stands in front of the
digits ("abangu-", "ngo-", nothing), whether "%" is kept, and how often the number is spelled out in words instead.  Evidence for the agent to weigh, never an answer chosen for it; free,
no model.  (A corpus audit of 69,875 sentences showed no fixed list of prefixes holds: "angu-" and "angama-" both follow 20-99, "ngo-" is the commonest before a year, "wamaphesenti"
follows a figure, "ya-" is another construction.)
"""
from __future__ import annotations

import re
import sqlite3
from collections import Counter
from typing import Any

from .human_translations import RATIO_HIGH, RATIO_LOW

_NUMBER = re.compile(r"\d[\d.,]*\d|\d")
_ZU_FIGURE = re.compile(r"(?<![A-Za-z0-9])(?P<pre>[A-Za-z]{1,12}-\s?)?(?P<n>\d[\d.,]*\d|\d)(?P<pct>\s?%)?")
MAX_PAIRS = 2500
MAX_SHAPES = 6


def _norm(number: str) -> str:
    return re.sub(r"[ ,]", "", number).rstrip(".")


def _size(number: str) -> int:
    """The class of a figure whose isiZulu writing changes with its size: 1-9, 10-19, 20-99, and anything else (a decimal, a year, hundreds) on its own."""
    if not re.fullmatch(r"\d{1,2}", number):
        return 3
    value = int(number)
    return 0 if value < 10 else 1 if value < 20 else 2


def _shape(phrase: str) -> tuple[str, re.Pattern[str], str] | None:
    """(GLOB pattern, regex that captures the English number, the figure as given) for an English phrase that holds one figure, or None."""
    text = " ".join(phrase.lower().replace("’", "'").split())
    found = _NUMBER.search(text)
    if not found:
        return None
    figure = found.group(0)
    before, after = text[:found.start()], text[found.end():]
    glob_digits = "".join("[0-9]" if c.isdigit() else ("[.]" if c == "." else ("[,]" if c == "," else c)) for c in figure)
    glob = "*" + before.replace("[", "[[]").replace("*", "[*]").replace("?", "[?]") + glob_digits + after.replace("[", "[[]").replace("*", "[*]").replace("?", "[?]") + "*"
    regex = re.compile(re.escape(before) + r"(?P<n>\d[\d.,]*\d|\d)" + re.escape(after), re.IGNORECASE)
    return glob, regex, figure


def figure_style(db: sqlite3.Connection | None, phrase: str, limit: int = MAX_SHAPES) -> dict[str, Any]:
    """How the isiZulu pairs write the figure of an English phrase such as "31 percent", "20 years", "in 2006", "11 municipalities", "at 7am", "R500"."""
    shape = _shape(str(phrase or ""))
    if shape is None:
        return {"phrase": phrase, "error": "give the English phrase with its figure in digits, e.g. \"31 percent\", \"20 years\", \"in 2006\", \"11 municipalities\""}
    if db is None:
        return {"phrase": phrase, "error": "the human-translation pairs are not installed on this machine"}
    glob, regex, figure = shape
    rows = db.execute("SELECT en, zu, en_words, zu_words FROM pairs WHERE lower(en) GLOB ? LIMIT ?", (glob, MAX_PAIRS)).fetchall()
    shapes: Counter[str] = Counter()
    example: dict[str, tuple[int, str]] = {}
    in_words = used = 0
    for en, zu, ew, zw in rows:
        if not (RATIO_LOW <= zw / max(ew, 1) <= RATIO_HIGH) or en.isupper():
            continue                                                    # a pair shifted by a line, or a heading in capitals
        match = regex.search(en.lower().replace("’", "'"))
        if not match:
            continue
        wanted = _norm(match.group("n"))
        if _size(wanted) != _size(_norm(figure)):
            continue                                                    # "11 municipalities" is compared with 10-19, not with 63
        used += 1
        hit = next((m for m in _ZU_FIGURE.finditer(zu) if _norm(m.group("n")) == wanted), None)
        if hit is None:
            in_words += 1                                               # the number is not written as digits in the isiZulu (spelled out, or left out)
            continue
        prefix = (hit.group("pre") or "").lower().replace(" ", "")
        key = (prefix or "(no prefix)") + "<figure>" + ("%" if hit.group("pct") else "")
        shapes[key] += 1
        start = max(0, hit.start() - 28)
        snippet = zu[start:hit.end() + 22].replace("\n", " ")
        if key not in example or len(zu) < example[key][0]:
            example[key] = (len(zu), ("..." if start else "") + snippet + "...")
    total = sum(shapes.values())
    if not used:
        return {"phrase": phrase, "pairs_checked": 0, "note": "no pair has this phrase with a figure; try the unit word with another figure or fewer words"}
    sparse = ({"few_pairs": f"only {used} pairs have this phrase with a figure of this size: ask again with a commoner unit (people, years, percent) to see more usage"} if used < 8 else {})
    return {
        "phrase": phrase,
        "pairs_checked": used,
        **sparse,
        "figure_written_as_digits": total,
        "figure_spelled_out_or_left_out": in_words,
        "ways_the_figure_is_written": [{"written": key, "pairs": n, "share_of_digit_pairs": round(n / total, 2), "example": example[key][1]} for key, n in shapes.most_common(limit)] if total else [],
        "note": ("The voice says a figure of one or two digits in English (after its prefix), so what is yours to choose is the prefix and the unit word: use what translators wrote for "
                 "this kind of figure and the agreement your sentence needs. Different prefixes can both be right (angu-/angama-): this shows usage, not a rule."),
    }
