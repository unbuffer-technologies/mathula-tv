"""How fluent and how clean a reference clip is, measured from its audio.

The voice copies its reference: its pace, its pauses, its hesitations, its noise (OmniVoice deep dive, section 5: "the reference decides fluency").  The bank chooses a speaker's one clip by what the
recogniser says (confidence, pace against the speaker's usual, gaps between words).  The recogniser's word times run on without gaps (96% of them have none), so what a clip really does is in its audio.
These are the measures of the audio itself, per clip, none of which needs a model but the forced aligner (the same one that cuts the primers):

- ``pause_s`` / ``pauses``: seconds of quiet inside the clip (25 dB under its speech, at least 0.12 s, not counting the first and last 0.15 s) and how many: hesitation, not phrasing.
- ``rate_cv``: how much the speed of the words varies (letters a second word by word: spread over mean): a clip that rushes and drags is not a steady voice.
- ``fidelity``: how well the aligner hears the clip's own transcript in its audio (mean confidence of its letters): a clip whose text does not match its sound, or is mumbled, scores low.
- ``snr_db``: the speech level over the noise floor (95th over 5th percentile of the 10 ms levels): a clip with a bed of noise or music under it teaches that.
- ``clipped``: the share of samples at full scale.

``fluency`` is what is left of 1 after the pauses and the unsteadiness are taken off; it ranks clips by hesitation and steadiness only: it never says whose voice or what accent is better, and the ear
has the last word (``audition`` lists the clips to listen to).
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

QUIET_DB = 25.0
PAUSE_MIN_S = 0.12
EDGE_S = 0.15


def _levels(audio: np.ndarray, rate: int) -> np.ndarray:
    frame = max(1, int(0.01 * rate))
    count = len(audio) // frame
    if count == 0:
        return np.zeros(0)
    return 20.0 * np.log10(np.sqrt((audio[: count * frame].reshape(count, frame) ** 2).mean(axis=1)) + 1e-9)


def inner_pauses(audio: np.ndarray, rate: int) -> tuple[float, int]:
    """(seconds of quiet inside the clip, how many such stretches): the first and last EDGE_S are left out, a quiet of at least PAUSE_MIN_S counts."""
    db = _levels(audio, rate)
    if len(db) < 30:
        return 0.0, 0
    quiet = db < np.percentile(db, 95) - QUIET_DB
    edge = int(EDGE_S / 0.01)
    total, count, i = 0.0, 0, edge
    while i < len(quiet) - edge:
        if quiet[i]:
            j = i
            while j < len(quiet) - edge and quiet[j]:
                j += 1
            if (j - i) * 0.01 >= PAUSE_MIN_S:
                total, count = total + (j - i) * 0.01, count + 1
            i = j
        else:
            i += 1
    return round(total, 2), count


def rate_spread(words: Sequence[Mapping[str, Any]]) -> float | None:
    """The spread over the mean of the speed of the words (letters a second, each word on its own: ``{text, start, end}``), None when there are under five words to say anything."""
    rates = [sum(ch.isalpha() for ch in str(w["text"])) / max(float(w["end"]) - float(w["start"]), 0.05) for w in words if sum(ch.isalpha() for ch in str(w["text"])) >= 3]
    if len(rates) < 5:
        return None
    rates = np.array(rates)
    return round(float(rates.std() / rates.mean()), 3)


def noise_ratio_db(audio: np.ndarray, rate: int) -> float:
    db = _levels(audio, rate)
    return round(float(np.percentile(db, 95) - np.percentile(db, 5)), 1) if len(db) else 0.0


def clipped_share(audio: np.ndarray) -> float:
    return round(float((np.abs(audio) >= 0.99).mean()), 5) if len(audio) else 0.0


def fidelity(aligned_words: Sequence[Mapping[str, Any] | None]) -> float | None:
    """The mean confidence of the letters the forced aligner finds in the clip's own text (``word_align`` words with their ``letters``), or None when it cannot tell."""
    sure = [float(item[2]) for w in aligned_words if w for item in (w.get("letters") or []) if len(item) > 2]
    return round(sum(sure) / len(sure), 3) if sure else None


def fluency(pause_s: float, pauses: int, seconds: float, spread: float | None) -> float:
    """1 less the share of the clip that is hesitation and less the unsteadiness of its speed: 1.0 is a clip with no pause inside it and a steady pace."""
    share = pause_s / max(seconds, 0.1)
    return round(max(0.0, 1.0 - 2.0 * share - 0.05 * max(0, pauses - 1) - 0.5 * max(0.0, (spread or 0.0) - 0.25)), 3)


def measure(audio: np.ndarray, rate: int, words: Sequence[Mapping[str, Any]], aligned: Sequence[Mapping[str, Any] | None] = ()) -> dict[str, Any]:
    """Every measure of one clip.  ``words`` are the clip's words with seconds relative to the clip; ``aligned`` the aligner's words for its text (optional)."""
    audio = np.asarray(audio, dtype=np.float32)
    seconds = len(audio) / rate
    pause_s, pauses = inner_pauses(audio, rate)
    spread = rate_spread(words)
    return {"seconds": round(seconds, 2), "pause_s": pause_s, "pauses": pauses, "rate_cv": spread, "fidelity": fidelity(aligned), "snr_db": noise_ratio_db(audio, rate), "clipped": clipped_share(audio),
            "fluency": fluency(pause_s, pauses, seconds, spread)}
