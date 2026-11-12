"""English weekday names inside a unit of isiZulu, shown to the agent as a flag in ``check_text``: evidence to weigh, never a gate and never a rewrite.

Found by a blind read of five dub runs (2026-11-10): ``ngoMonday`` and ``uma sekunguMonday`` where Zulu news writes ``ngoMsombuluko``.  Counted in 8.4 million words of Zulu news and
web text: the English day names are almost never written (monday 7, tuesday 8, wednesday 7, thursday 2, saturday 12; friday 36 and sunday 88 come from titles such as Sunday Times),
while the Zulu forms are written thousands of times (ngomsombuluko 3,008, ngolwesibili 3,314, ngolwesithathu 3,157, ngolwesine 2,420, ngolwesihlanu 2,922, ngomgqibelo 4,088,
ngesonto 5,678).  The voice reads the Zulu name as Zulu, so the Zulu day is the one that sounds like the rest of the line.

Two checks that were tried and are NOT here, because the numbers did not support them: months (the English forms stay common, july 347 against julayi 81), and "a form real text never writes
while a form one edit away is common" (69 flags on 191 units, nearly all valid inflections such as nizoshintsha and nibatshelile; it missed the slips it was built for, enaniba and
nibezwile, and the ZulMorph analyser parses those two as well, so a morphological test does not separate them either).  Rarity is not an error signal in Zulu.
"""
from __future__ import annotations

import re

WEEKDAYS = {
    "monday": ("uMsombuluko", "ngoMsombuluko"), "tuesday": ("uLwesibili", "ngoLwesibili"), "wednesday": ("uLwesithathu", "ngoLwesithathu"),
    "thursday": ("uLwesine", "ngoLwesine"), "friday": ("uLwesihlanu", "ngoLwesihlanu"), "saturday": ("uMgqibelo", "ngoMgqibelo"), "sunday": ("iSonto", "ngeSonto"),
}
_NAMES = "|".join(day.capitalize() for day in WEEKDAYS)
# a day name glued to a lowercase Zulu prefix (ngoMonday, uMonday, nguMonday), or the bare word; not a title (Sunday Times, Friday Night)
_GLUED = re.compile(r"(?<![A-Za-z])[a-z]{1,8}(" + _NAMES + r")(?![A-Za-z])(?!\s+[A-Z][a-z])")
_BARE = re.compile(r"(?<![A-Za-z])(" + "|".join(WEEKDAYS) + r")(?![A-Za-z])(?!\s+[A-Z][a-z])", re.IGNORECASE)
_PAUSE = re.compile(r"~pause\d+")


def weekday_notes(zulu_text: str) -> list[str]:
    text = _PAUSE.sub(" ", zulu_text or "")
    found: dict[str, str] = {}
    for pattern in (_GLUED, _BARE):
        for match in pattern.finditer(text):
            day = match.group(1).lower()
            found.setdefault(day, match.group(0))
    return [f"{written}: Zulu news almost never writes the English day name (about 40 times in 8.4 million words) and writes the Zulu one constantly "
            f"({WEEKDAYS[day][0]}; {WEEKDAYS[day][1]} for 'on {day.capitalize()}'); the voice reads the Zulu name as Zulu"
            for day, written in found.items()]
