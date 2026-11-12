"""How much a line asks of the listener: measurements only, for the agent to weigh.

A viewer hears a line once and cannot re-read it.  Two things make a line hard to follow and can be measured from the text alone, using what the dub voices leave for each mark
(measured 2026-11-11, working/punctuation_calibration.json: a comma-class mark ~0.15 s, a sentence end ~0.9 s, nothing between words ~0):

* figures run together: two figures with no breath between them (a figure is what the viewer must take in);
* a long run: more than about six seconds of speech between breaths.

The renderer adds a comma after a percentage that is followed by words, so a "%" counts as a breath here.  Nothing is rewritten: a flag carries evidence and the agent decides.
"""
from __future__ import annotations

import re

CHARS_PER_SECOND = 12.0                 # the dub voices' natural pace, as the agent is told
MAX_RUN_SECONDS = 8.0                 # Zulu is dense (an agglutinative sentence of 8 words is ~7 s): a flag at 6 s fired on half the units of a clip
CLUSTER_CHARS = 45                      # two figures closer than this with no breath between them are run together
_FIGURE = re.compile(r"(?<![A-Za-z0-9])(\d[\d.,]*\d|\d)%?")
_PAUSE = re.compile(r"~pause\d+")
_BREATH = re.compile(r"[,;:—–]|\s-\s|[.?!](?=\s|$)|~pause\d+|%(?=\s+\w)")


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def figures_run_together(text: str) -> list[str]:
    """Pairs of figures with no breath mark between them and little text in between: "ngo-2006 laya ku-44.05%"."""
    text = _PAUSE.sub(lambda m: "," * 1, text)               # a marked pause is a breath, and its digits are not a figure
    found: list[str] = []
    figures = list(_FIGURE.finditer(text))
    for first, second in zip(figures, figures[1:]):
        between = text[first.end():second.start()]
        if len(between) <= CLUSTER_CHARS and not _BREATH.search(first.group(0) + between):
            start, end = max(0, first.start() - 12), min(len(text), second.end() + 8)
            found.append(f"'{_clean(text[start:end])}' - nothing separates {first.group(0)} from {second.group(0)}")
    return found


def long_runs(text: str, limit: int = 2) -> list[str]:
    """The longest stretches of speech with no breath mark, when they run past MAX_RUN_SECONDS (estimated from the characters)."""
    pieces = [p for p in _BREATH.split(_clean(text)) if p and p.strip()]
    longest = sorted(((len(p.strip()), p.strip()) for p in pieces), reverse=True)
    out = []
    for size, piece in longest[:limit]:
        seconds = size / CHARS_PER_SECOND
        if seconds > MAX_RUN_SECONDS:
            out.append(f"about {seconds:.0f} s with no breath: '{piece[:60]}...{piece[-30:]}'" if size > 100 else f"about {seconds:.0f} s with no breath: '{piece}'")
    return out
