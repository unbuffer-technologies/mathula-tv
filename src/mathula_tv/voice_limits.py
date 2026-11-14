"""Words the Azure isiZulu voice cannot say well, and what to write instead.

A listener heard a stutter on "emininingwaneni" (job 1c6f3f5a, 2026-11-12): the word is real, common isiZulu ("imininingwane", details/information, 1,814 uses in the written corpus) but
its repeated -ni-ni- syllables blur in this voice, and re-spelling it (hyphens, a longer vowel) is a gamble the listener has to judge by ear each time.  The listener's decision was to
swap the word.  Note the meanings: "ulwazi" is INFORMATION (3,694 uses) and is easy for the voice, but the written corpus's word for DETAILS is imininingwane (515 of the 901 sentences
with "details"), so ulwazi is no swap for "details".  The listener asked for the English word instead ("ama-details"): the written parallel corpus never keeps "details" in English (0 of 901
sentences) but everyday speech does (ama-details 7, details 5 in 482 h; imininingwane 489), and the voice says an English word well.  The agent chose imininingwane for "details" and ulwazi for
"information" in the same unit, so it was following the corpus; this list is what overrides the corpus for the one word the voice cannot say.

Evidence for the agent, never a gate: it sees the line where it can still act (the prompt, check_text, every submit_dub, the wrap-up).  To add a word, add a row: the pattern of every
form of the word, what to write instead, and why.
"""
from __future__ import annotations

import re
from typing import Any, Iterable, Mapping

VOICE_LIMIT_WORDS: tuple[dict[str, Any], ...] = (
    {"pattern": re.compile(r"\b\w*mininingwan\w*", re.IGNORECASE | re.UNICODE), "word": "imininingwane",
     "instead": "ama-details when the English says details (the English word kept, as everyday speakers say it: ama-details is in the 482 h spoken corpus), ulwazi when it says information",
     "why": "the voice blurs its repeated -ni-ni- syllables into a stutter"},
    {"pattern": re.compile(r"(?<![\w-])(?:mm?-?h?m+|hm+m*|mh?m+)(?![\w-])", re.IGNORECASE | re.UNICODE), "word": "Mm-hmm",
     "instead": "Yebo (or Cha for a no) when the speaker is agreeing, or nothing when it is only the other person's murmur under the speaker; never the sound itself",
     "why": "a hummed sound written in letters is read letter by letter (\"mhim\", \"Mm-himi\"): a listener heard it at the end of a turn"},
)


# A murmur in the ENGLISH ("Mm-hmm", "Mhm", "Uh-huh"): usually the other person listening, caught inside a speaker's unit.  Shown with the unit when the clip opens so the agent decides about
# it before drafting (the YouTube captions had "Mm-hmm." as the interviewer's murmur at the end of an analyst's answer; the agent copied it and the voice read it letter by letter).
MURMUR = re.compile(r"(?<![\w-])(?:mm?-?hm+|mhm+|uh-?huh|um-?hm+|mmm+)(?![\w-])", re.IGNORECASE | re.UNICODE)
MURMUR_NOTE = ("Some segments hold a murmur in the English (murmur_in_the_english: Mm-hmm, Mhm, Uh-huh): usually the other person listening, caught inside this speaker's segment. It is a sound, "
               "not speech, and the voice reads a written hum letter by letter. Leave it out; if the speaker is plainly answering with it, write Yebo (or Cha).")


def murmurs(text: str) -> list[str]:
    seen: dict[str, str] = {}
    for m in MURMUR.finditer(str(text or "")):
        seen.setdefault(m.group(0).casefold(), m.group(0))          # once each, the first spelling, in order
    return list(seen.values())


def find(text: str) -> list[str]:
    """One short line for each limit word found in ``text``."""
    lines = []
    for limit in VOICE_LIMIT_WORDS:
        found = list(dict.fromkeys(m.group(0) for m in limit["pattern"].finditer(str(text or ""))))
        if found:
            lines.append(f"“{', '.join(found)}”: {limit['why']}; write {limit['instead']}")
    return lines


def for_items(items: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    found = []
    for item in items:
        lines = find(str(item.get("zulu_text") or ""))
        if lines:
            found.append({"item_id": str(item.get("item_id")), "words": lines})
    return found


def prompt_line() -> str:
    return "; ".join(f"never write {l['word']} (or a form of it): {l['why']}; write {l['instead']}" for l in VOICE_LIMIT_WORDS)
