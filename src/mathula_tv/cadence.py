"""Is this unit spoken in sentences a person would use?  The cadence of a Zulu unit against real spoken Zulu, for the agent's submit feedback.

Every full stop leaves about 0.9 s of silence in these voices, so a unit chopped into very short sentences does not only read choppy: it is mostly pauses.  The agent writes about 45 more
sentences than the English speakers said over a 15-minute clip (job 37bae6ec: 168 Zulu sentences for 123 English, about 45 s of added silence).  Some splitting is right (Zulu builds long words and
needs fewer of them), so the test is not "more sentences than the English" but "sentences shorter than almost any real speaker's".

The reference is measured, not assumed: across 23,345 African Next Voices isiZulu transcripts of three sentences or more, the mean words per sentence is 5.4 at the 5th percentile, 6.3 at the
10th, 8.0 at the 25th and 10.2 at the median (Zulu news text: median 12.7).  A unit is flagged only when it has three or more sentences, at least one and a half times the English sentence
count, and a mean below the 10th percentile: only about one real speaker in ten talks in sentences that short.  Evidence for the agent to weigh, never a gate: a unit may need the silence.
"""
from __future__ import annotations

import re
from typing import Any, Iterable, Mapping

from .naturalness import sentences

SPOKEN_P10_WORDS = 6.3
SPOKEN_MEDIAN_WORDS = 10.2
MIN_SENTENCES = 3
MIN_RATIO = 1.5
FULL_STOP_S = 0.9
_ENGLISH_SENTENCES = re.compile(r"(?<=[.!?])\s+")


def english_sentence_count(text: str) -> int:
    return len([p for p in _ENGLISH_SENTENCES.split(str(text or "").strip()) if p.strip()]) or 1


def cadence_line(item: Mapping[str, Any]) -> str | None:
    """One sentence about a unit chopped into sentences shorter than almost any real speaker's, or None."""
    zulu = sentences(str(item.get("zulu_text") or ""))
    if len(zulu) < MIN_SENTENCES:
        return None
    mean = sum(len(s) for s in zulu) / len(zulu)
    english = english_sentence_count(str(item.get("source_text") or ""))
    if mean >= SPOKEN_P10_WORDS or len(zulu) < MIN_RATIO * english:
        return None
    return (f"{len(zulu)} Zulu sentences for {english} English, {mean:.1f} words each: only about one real speaker in ten talks in sentences this short (the median is {SPOKEN_MEDIAN_WORDS} words), "
            f"and every full stop leaves about {FULL_STOP_S} s of silence. Join them with commas where the speaker kept going, unless the unit needs that silence.")


def cadence_lines(items: Iterable[Mapping[str, Any]]) -> list[dict[str, str]]:
    return [{"item_id": str(i.get("item_id")), "cadence": line} for i in items if (line := cadence_line(i))]
