"""Where a long unit is cut into separate generations of the voice: at the places where the SOURCE speaker paused.

OmniVoice speaks a unit span by span (a span is one generation, about 15-20 s at most), and every join between two spans is a cold start of the voice after a gap.  The cut used to fall wherever
the next sentence would push a span past 280 characters, which in prod-119 was between "Sanibonani ekuseni kuwe Ayanda." and "Ngesonto eledlule ...": the source speaker runs straight from
one into the other, and the dub stopped there, then started the next sentence with a pitch spike.

So the cut is chosen, not forced: for every boundary between two Zulu sentences, find how long the speaker paused at the matching place in the English (the word alignment of ``word_alignment``,
or the position alone when it is not installed), and cut where the speaker paused; a span may run to HARD_CHARS to reach such a place.  Where the source pauses nowhere in reach, the cut falls at the
nearest sentence end as before.  Nothing here changes a word of the text.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

REAL_PAUSE_S = 0.2            # a gap this long in the source is a pause the speaker took (comma_check uses 0.25 for a comma; a join is cheaper to place, so a little less)
SOFT_CHARS = 280              # about 15-20 s of speech: what one generation was judged to hold comfortably
HARD_CHARS = 420              # the most a span may run to reach a pause of the source
MIN_SPAN_CHARS = 70           # a span shorter than this is a cold start for very little speech
CUT_AT_PAUSE = 0.15           # the price of a cut where the speaker paused ...
CUT_AT_NO_PAUSE = 1.0         # ... and where nobody did
SHORT_SPAN = 0.5
LONG_SPAN = 0.6               # the price of a span running from SOFT_CHARS to HARD_CHARS

PAD_MS = 200                  # the silence the model itself puts at the ends of a generation (100 ms each side): a join is measured from it
MAX_JOIN_PAUSE_S = 0.9


def join_gap_ms(pause_s: float) -> int:
    """The gap to ask for after a cut, on top of the model's own 200 ms of padding.  Where the speaker paused, the join is as long as the pause (the padding counts toward it); where the speaker
    ran straight on, the join is as short as it can be (-PAD_MS: the padding itself is dropped, so the next generation follows at once).  A negative gap is only ever read by this code."""
    if pause_s >= REAL_PAUSE_S:
        return int(min(pause_s, MAX_JOIN_PAUSE_S) * 1000) - PAD_MS
    return -PAD_MS


SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-ZÀ-ɏ\"'(])")
PAUSE_MARK = re.compile(r"\s*~pause\d+\s*")


@lru_cache(maxsize=4)
def get_aligner(work_dir: str) -> Callable[..., Any] | None:
    """The corpus-based word aligner (the same the submit checks use), or None when the corpus is not installed."""
    try:
        from . import attestation, word_alignment

        resources = attestation.load(Path(work_dir))
        return word_alignment.make_aligner(resources.translation_forms) if resources is not None else None
    except Exception:                      # noqa: BLE001 - an optional resource must never stop a synthesis
        return None


def sentences_of(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_END.split(PAUSE_MARK.sub(" ", str(text or "")).strip()) if s.strip()]


def boundary_gaps(text: str, inside: Sequence[Mapping[str, Any]], aligner: Callable[..., Any] | None = None) -> list[float] | None:
    """For each boundary between two sentences of the Zulu ``text``, the longest pause the speaker took at the matching place of the English words ``inside`` (0.0 for none); None when the
    text has fewer than two sentences or there are too few timed words to say anything."""
    from .comma_check import NEAR_CHARS, NEAR_FRACTION, _gap_after, _place_in_zulu

    sentences = sentences_of(text)
    if len(sentences) < 2 or len(inside) < 4:
        return None
    flat: list[dict[str, Any]] = []
    boundaries: list[int] = []
    cursor = 0
    for sentence in sentences:
        for token in sentence.split():
            cursor += len(token) + 1
            flat.append({"word": token, "end": cursor})
        boundaries.append(cursor)
    boundaries = boundaries[:-1]
    english = list(inside)
    lengths = [len(str(w["text"])) + 1 for w in english]
    cumulative = [sum(lengths[:i + 1]) for i in range(len(lengths))]
    total_z = flat[-1]["end"]
    pairs = []
    if aligner is not None:
        try:
            pairs = aligner([str(w["text"]) for w in english], [w["word"] for w in flat]) or []
        except Exception:                  # noqa: BLE001
            pairs = []
    gaps = [0.0] * len(boundaries)
    for k in range(len(english) - 1):
        gap = _gap_after(english, k)
        if gap is None or gap < REAL_PAUSE_S:
            continue
        if pairs:
            target, window = _place_in_zulu(k, pairs, english, flat, cumulative, total_z)
        else:
            target, window = total_z * cumulative[k] / cumulative[-1], max(NEAR_CHARS, int(NEAR_FRACTION * total_z))
        nearest = min(range(len(boundaries)), key=lambda i: abs(boundaries[i] - target))
        if abs(boundaries[nearest] - target) <= window:
            gaps[nearest] = max(gaps[nearest], gap)
    return gaps


def choose_cuts(lengths: Sequence[int], gaps: Sequence[float], *, soft: int = SOFT_CHARS, hard: int = HARD_CHARS, minimum: int = MIN_SPAN_CHARS) -> list[int]:
    """Which sentence boundaries to cut at (``i`` = after sentence ``i``, 0-based), so that spans stay within ``hard`` characters and the cuts fall where the speaker paused; the cheapest set
    by dynamic programming over (cut price + a price for a long or very short span)."""
    n = len(lengths)
    prefix = [0]
    for length in lengths:
        prefix.append(prefix[-1] + length + 1)

    best = [float("inf")] * (n + 1)
    previous = [0] * (n + 1)
    best[0] = 0.0
    for j in range(1, n + 1):
        for i in range(j):
            size = prefix[j] - prefix[i]
            if size > hard and j - i > 1:
                continue                                                      # a single sentence longer than the hard limit is allowed to stand alone
            cost = best[i]
            if size > soft:
                cost += LONG_SPAN * min(1.0, (size - soft) / max(1, hard - soft))
            if size < minimum and n > 1:
                cost += SHORT_SPAN
            if j < n:
                cost += CUT_AT_PAUSE if gaps[j - 1] >= REAL_PAUSE_S else CUT_AT_NO_PAUSE
            if cost < best[j]:
                best[j], previous[j] = cost, i
    cuts: list[int] = []
    j = n
    while j > 0:
        i = previous[j]
        if i > 0:
            cuts.append(i - 1)
        j = i
    return sorted(cuts)
