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
)


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
