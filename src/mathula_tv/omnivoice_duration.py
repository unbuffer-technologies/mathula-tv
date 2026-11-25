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


SPEECH_FACTOR = 0.93       # the dub's speech is this much of what the formula says: fitted on 11 real runs of the 7 units of one job (0.931), each span cut out of a longer generation (its lead sentence and the next span's first word removed)
NATURAL_GAP_S = 0.28       # the silence the voice itself leaves between the last word of a span and the first word of the next (0.2-0.4 s measured, 0.28 on average): a join is never shorter than this
TYPICAL_WEIGHT_PER_CHAR = 0.9          # measured on 21 real Zulu units (12,868 characters): 0.898 on average, 0.874-0.925 per unit; spaces and punctuation weigh less than a letter


def chars_to_fill(window_s: float, ref_text: str, ref_seconds: float, speed: float = 1.0) -> int:
    """How many characters of typical Zulu text fill a window at the pace of a reference clip: the inverse of ``estimate_seconds`` for a first draft's length target."""
    pace = seconds_per_weight(ref_text, ref_seconds)
    speaking = max(0.3, window_s - EDGE_PAD_S)                    # the model keeps 200 ms of padding in every generation: it is no speech, and matters for a window of a second or two
    return max(1, int(round(speaking * (speed if speed > 0 else 1.0) / (pace * TYPICAL_WEIGHT_PER_CHAR * SPEECH_FACTOR))))


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
    """A unit as the dub makes it: the spans' speech at the reference's natural pace (scaled by SPEECH_FACTOR, since each span is cut out of a longer generation), the joins between them, and the 200 ms
    of padding at the two ends of the unit.  A join is the model's own 200 ms of padding plus the planned gap (the speaker's own pause, or none where they ran on), or the voice's own gap
    (NATURAL_GAP_S) when that is longer: what the dub sets between two spans from their word times.  Fitted on 11 real runs: mean error 2.2%, worst 3.5% (the formula alone ran 6% high)."""
    pace = seconds_per_weight(ref_text, ref_seconds)
    speech = sum(span_seconds(span, pace, speed) - EDGE_PAD_S for span in spans)
    joins = sum(max(EDGE_PAD_S + gap / 1000.0, NATURAL_GAP_S) for gap in gaps_ms[:max(0, len(spans) - 1)])
    return SPEECH_FACTOR * speech + joins + EDGE_PAD_S


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
