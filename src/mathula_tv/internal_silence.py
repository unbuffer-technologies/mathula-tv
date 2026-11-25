"""How much of a stretch of generated speech is the voice's own silence between phrases (detection only: nothing is cut, stretched or pinned).

OmniVoice takes its pace, and its pauses, from the reference clip it clones.  A reference that is slow or hesitant (prod-119, unit-03: a 6 s clip at 2.2 words a second ending in "I suppose",
the slowest of that speaker's clips) makes the model speak at that pace and leave phrase-sized gaps: a slow fade-out, a flat floor about 60 dB down and an abrupt restart, heard as badly joined
clips.  A fluent reference of the same speaker gave the same text with no gaps and half the length.  So the remedy is in choosing the reference (``ReferenceBank``), and this measurement is how a
bad choice is noticed: a synthesis with a lot of internal silence is repeated once with the next-best reference.
"""
from __future__ import annotations

import numpy as np

FRAME_S = 0.01
MARGIN_DB = 20.0              # "silent" = this far below the speech level
MIN_RUN_S = 0.25              # a silence shorter than this is ordinary phrasing


def _levels(audio: np.ndarray, frame: int) -> np.ndarray:
    count = len(audio) // frame
    return 20.0 * np.log10(np.sqrt((audio[: count * frame].reshape(count, frame) ** 2).mean(axis=1)) + 1e-9)


def silent_runs(audio: np.ndarray, rate: int, *, min_run_s: float = MIN_RUN_S, margin_db: float = MARGIN_DB) -> list[tuple[int, int]]:
    """The runs of FRAME_S frames (first, last+1) inside the speech that are margin_db below its level and at least min_run_s long; the lead-in and the tail are not runs."""
    levels = _levels(np.asarray(audio, dtype=np.float32), max(1, int(FRAME_S * rate)))
    if len(levels) < 10:
        return []
    quiet = levels < float(np.median(levels[levels > np.percentile(levels, 30)])) - margin_db
    runs: list[tuple[int, int]] = []
    index = 0
    while index < len(quiet):
        if quiet[index]:
            end = index
            while end < len(quiet) and quiet[end]:
                end += 1
            if index > 0 and end < len(quiet) and (end - index) * FRAME_S >= min_run_s:
                runs.append((index, end))
            index = end
        else:
            index += 1
    return runs


def internal_silence_s(audio: np.ndarray, rate: int, *, min_run_s: float = MIN_RUN_S) -> float:
    """How many seconds of this stretch are the voice's own gaps inside the speech."""
    return round(sum((last - first) * FRAME_S for first, last in silent_runs(audio, rate, min_run_s=min_run_s)), 3)
