"""What the voice will actually be given for a line of Zulu text: the spoken form, change by change, and what is still unprocessed.

The agent writes isiZulu with digits, abbreviations and English words; a dictionary of rules turns that into the text the voice reads (figures and years in English, titles expanded,
names respelled).  The agent used to see none of it, so a defect ("Dkt" left raw, "Section 79" as digits) was found by a listener's ear and answered with a new rule.  ``say_it`` shows the
agent its own line as the voice gets it, so it can notice and fix what it wrote.  It runs the same dictionary the measurements use; free, no speech, no model.
"""
from __future__ import annotations

import re
from typing import Any

from .abbreviations import expand as expand_abbreviations

_DIGITS = re.compile(r"(?<![A-Za-z])\d[\d.,:/%-]*\d%?|\d")
_ABBREVIATION = re.compile(r"(?<![A-Za-z])(?:Mnu|Nkk|Nksz|Dkt|Dr|Prof|Mr|Mrs|Ms|Mnr|Mev|Adv|Hon|Cllr|Rev|Gen|Col|Capt|Lt|St|Jr|Sr|No)\.(?=\s|$)")
_ALL_CAPS = re.compile(r"(?<![A-Za-z-])[A-Z]{2,6}(?![A-Za-z])")


def say_it(dictionary: Any, text: str, voice: str | None = None) -> dict[str, Any]:
    """``dictionary`` is a PronunciationDictionary; ``text`` the agent's Zulu line (``~pauseN`` markers are not speech and are left out)."""
    plain = re.sub(r"\s*~pause\d+\s*", " ", str(text or "")).strip()
    plain, written_out = expand_abbreviations(plain)          # the abbreviations the system writes out itself, shown like any other change
    result = dictionary.apply(plain, voice=voice)
    spoken = result.tts_text
    changes = [{"was": c["was"], "now": c["now"], "kind": "abbreviation"} for c in written_out] + [{"was": s.before, "now": s.after, "kind": s.kind} for s in result.substitutions if s.before != s.after]
    left_over: list[str] = []
    for label, pattern in (("digits the voice will read by itself", _DIGITS), ("an abbreviation the voice will read letter by letter or skip", _ABBREVIATION),
                           ("capitals the voice will spell or guess at", _ALL_CAPS)):
        found = sorted({m.group(0) for m in pattern.finditer(spoken)})
        if found:
            left_over.append(f"{label}: {', '.join(found)}")
    return {
        "spoken": spoken,
        "changes": changes[:40],
        "still_as_written": left_over,
        "note": ("This is exactly what the voice reads (the same dictionary as measure_tts). Figures and years are said in English after their Zulu prefix, titles and names are respelled "
                 "by the rules. 'still_as_written' lists what no rule converted: if one is wrong for the ear, write it another way or say so in approach_notes. Free: no try, no speech."),
    }
