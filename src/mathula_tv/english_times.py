"""A time of day in the English transcript is written H:MM, whichever way the recogniser wrote it.

The English recogniser writes a time both ways in one clip ("5:00 o'clock" and "9.30", job 37bae6ec).  A point can be read as a decimal, by the Zulu voice, by a gate that counts the English
literal in the Zulu, and by the agent copying it; a colon is a time and nothing else.  Pass 1 only restores what was misheard and is told never to touch numbers, so nothing normalised
this.  This is the deterministic rule, applied where pass 1 assembles the restored text: a number shaped like a time (H.MM or H:MM, hour 0-23, minutes 00-59) is rewritten H:MM only when the
text itself shows it is a time: am / pm / o'clock / a part of the day within a short distance, or a preposition of time right before it (at, around, by, until, till, from, before, after,
since, about).  An amount (R9.30, $9.30), a figure with a unit (9.30 million, 9.30%), a longer decimal (9.305) and a version or ratio (3.1.30) are never changed.
"""
from __future__ import annotations

import re

_TIME = re.compile(r"(?<![\w.:/$£€-])(?P<h>[01]?\d|2[0-3])\.(?P<m>[0-5]\d)(?![\d.]|%)(?!\s*(?:million|billion|trillion|thousand|percent|per cent|per-cent|km|kg|rand|dollars?|euros?|pounds?|cents?)\b)", re.IGNORECASE)
_EVIDENCE_NEAR = re.compile(r"(?<![A-Za-z])(?:a\.?m\.?|p\.?m\.?|o['’]clock|morning|afternoon|evening|tonight|noon|midnight|hours?|hrs)(?![A-Za-z])", re.IGNORECASE)          # "8am": a digit may come right before
_PREPOSITION_BEFORE = re.compile(r"\b(?:at|around|by|until|till|from|before|after|since|about|past|to)\s+(?:about\s+|around\s+|roughly\s+|exactly\s+)?$", re.IGNORECASE)
WINDOW = 70


def normalise(text: str) -> str:
    """``text`` with each time written with a point (9.30) rewritten with a colon (9:30), where the text shows it is a time; everything else exactly as it was."""
    value = str(text or "")

    def swap(match: re.Match[str]) -> str:
        around = value[max(0, match.start() - WINDOW):match.end() + WINDOW]
        before = value[max(0, match.start() - 24):match.start()]
        if _EVIDENCE_NEAR.search(around) or _PREPOSITION_BEFORE.search(before):
            return f"{match.group('h')}:{match.group('m')}"
        return match.group(0)

    return _TIME.sub(swap, value)
