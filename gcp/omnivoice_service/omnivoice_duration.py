"""A free estimate of how long OmniVoice speaks a text at its natural pace: no GPU, no synthesis.

OmniVoice fixes the length of its output before it generates a sound: a rule estimator weighs the characters of the text (Latin letter 1.0, space 0.2,
punctuation 0.5, digit 3.5, combining mark 0) and scales them by the reference clip's own pace (reference seconds per unit of reference text weight),
divided by ``speed``.  Very short outputs are lifted by a power curve (under 50 tokens), and the post-processing pads 100 ms at each end.
So the pace of a cloned voice is the pace of its reference clip, and the natural length of a turn is a function of the Zulu text alone: the agent can
try wordings against a window without synthesising any of them.

Checked on six real runs (one unit of 24-40 s, two of about 2 s): -9% to +5% of the length OmniVoice actually gave.  The frame rate (25 fps) and the
curve's exponent (1/3) are assumptions taken from reading the source, not fitted to a large set; treat the error band as unproven until the whole-dub
report's predicted-vs-spoken column has been read on all units.
"""
from __future__ import annotations

import unicodedata

FPS = 25.0                 # the audio codec's frame rate (assumed)
LOW_TOKENS = 50.0          # below this many tokens the estimator lifts the length
ALPHA = 1.0 / 3.0          # exponent of that lift (assumed from 'boost_strength=3')
EDGE_PAD_S = 0.2           # post-processing keeps 100 ms of silence at each end of every generation


def weight(text: str) -> float:
    """OmniVoice's rule weight of a text: how much speaking time it counts for, relative to a Latin letter."""
    total = 0.0
    for ch in text:
        category = unicodedata.category(ch)
        if ch.isascii() and ch.isalpha():
            total += 1.0
        elif ch == " ":
            total += 0.2
        elif category.startswith("M"):
            continue
        elif category[0] in "PS":
            total += 0.5
        elif category[0] == "Z":
            total += 0.2
        elif category[0] == "N":
            total += 3.5
        elif ch.isalpha():
            total += 1.0
    return total


def seconds_per_weight(ref_text: str, ref_seconds: float) -> float:
    """The reference clip's pace: seconds of speech per unit of text weight."""
    reference = weight(ref_text)
    if reference <= 0 or ref_seconds <= 0:
        raise ValueError("a reference needs text and a positive duration")
    return ref_seconds / reference


def span_seconds(text: str, pace: float, speed: float = 1.0) -> float:
    """One generation's length: the estimator's tokens (with the short-output lift) as seconds, plus the 200 ms of edge padding."""
    tokens = weight(text) * pace / (speed if speed > 0 else 1.0) * FPS
    if 0 < tokens < LOW_TOKENS:
        tokens = LOW_TOKENS * (tokens / LOW_TOKENS) ** ALPHA
    return tokens / FPS + EDGE_PAD_S


def estimate_seconds(spans: list[str], gaps_ms: list[int], ref_text: str, ref_seconds: float, speed: float = 1.0) -> float:
    """A unit spoken as these spans (each its own generation) with these silences between them: its length in seconds at the reference's natural pace."""
    pace = seconds_per_weight(ref_text, ref_seconds)
    return sum(span_seconds(span, pace, speed) for span in spans) + sum(gaps_ms) / 1000.0


def speed_to_fit(spans: list[str], gaps_ms: list[int], ref_text: str, ref_seconds: float, window_s: float) -> float:
    """The speed setting that would make this unit fill window_s (1.0 means it already does).  Lengths are not linear in speed because of the edge padding
    and the short-output lift, so this is solved by bisection; the result is clamped to 0.5-2.0."""
    low, high = 0.5, 2.0
    if estimate_seconds(spans, gaps_ms, ref_text, ref_seconds, high) > window_s:
        return high
    if estimate_seconds(spans, gaps_ms, ref_text, ref_seconds, low) < window_s:
        return low
    for _ in range(40):
        mid = (low + high) / 2
        if estimate_seconds(spans, gaps_ms, ref_text, ref_seconds, mid) > window_s:
            low = mid
        else:
            high = mid
    return round((low + high) / 2, 3)
