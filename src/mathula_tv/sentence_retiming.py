"""Put each Zulu sentence of a turn where the source speaker's sentence begins, by moving only the SILENCE between sentences.

A dub unit is one long synthesis (a whole speaker turn).  Only its start and its end are tied to the picture; inside it every sentence begins where the text and the voice's
fixed silences put it (0.92 s after each full stop, 0.15 s after a comma, whatever a ~pause marker adds).  The source speaker's pauses are mostly shorter and differently placed,
so a sentence several seconds into a turn arrives late against the speaker's gesture (job 1c6f3f5a, 0:17: the reporter turned to the interviewee at 0:17.4, the dub's next sentence
began at 0:18.1 after 1.73 s of silence where the speaker had paused 0.72 s).

What this does, after the voice has spoken and before the audio is mixed:
  * finds the sentence ends in the unit's audio (a silence of 0.8 s or more),
  * lines the unit's sentences up with the source speaker's sentences (the words of the source audio end a sentence with . ? !) by how far into the speech each one is,
  * moves the silence between sentences so each Zulu sentence starts as near as it can to the matching source onset.
The speech is never touched (no speed change, no cut).  The unit is never shorter than it was; when it ends before its source window, the shortfall (``extra_ms``) is spent as
silence at the boundaries where the source speaker paused, instead of sitting as dead air at the end of the unit, and the unit is never longer than its window.  Each boundary
stays between MIN_GAP_S and MAX_GAP_S and moves by at most MAX_CHANGE_S, so a sentence that is matched to the wrong one can only do limited harm.
A gap is never made LONGER than the voice's own or than the speaker's own pause before the matched sentence (whichever is more): a Zulu sentence that begins before the speaker's
English one is left early, because lengthening the silence to reach the speaker's mouth only makes dead air (prod-86: a 1.0 s gap grew to 1.9 s and the Zulu stood silent after the
reporter had turned).  Silence taken out of the gaps goes to the end of the unit, which keeps its exact length.
"""
from __future__ import annotations

import wave
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

BOUNDARY_MIN_S = 0.8          # a silence this long inside a unit is the end of a sentence (a full stop leaves 0.92 s; a comma 0.15 s)
EDGE_S = 0.3                  # silence this close to the start or the end of the unit is the unit's own edge, not a boundary
MIN_GAP_S = 0.45              # a sentence end is never made shorter than this
MAX_GAP_S = 2.0               # ... nor longer than this
MAX_CHANGE_S = 0.9            # a boundary moves by at most this much
MAX_MATCH_FRACTION = 0.08     # a Zulu sentence is matched to a source sentence only when they sit this close (share of the turn's speech)
MAX_ONSET_ERROR_S = 2.5       # ... and only when its start is within this many seconds of the source's: a larger gap is far more likely a different sentence than real drift, and is left alone
MIN_SHIFT_S = 0.08            # nothing is rewritten for less than this
_STOPS = (".", "?", "!")


def source_onsets(words: Sequence[Mapping[str, Any]], start_ms: int, end_ms: int) -> tuple[list[float], float]:
    """Seconds from the unit's start at which each source sentence after the first begins, and the speech seconds before each (for matching).  A sentence ends on a word
    that carries . ? or ! in the recogniser's text."""
    inside = [w for w in words if float(w["start"]) >= start_ms / 1000 - 0.05 and float(w["end"]) <= end_ms / 1000 + 0.05]
    inside.sort(key=lambda w: float(w["start"]))
    onsets: list[float] = []
    speech_before: list[float] = []
    spoken = 0.0
    for index, word in enumerate(inside):
        if index and str(inside[index - 1].get("text", "")).rstrip("\"'”’)").endswith(_STOPS):
            onsets.append(float(word["start"]) - start_ms / 1000)
            speech_before.append(spoken)
        spoken += max(0.0, float(word["end"]) - float(word["start"]))
    total = max(spoken, 1e-6)
    return onsets, [s / total for s in speech_before]


def source_gaps(words: Sequence[Mapping[str, Any]], start_ms: int, end_ms: int) -> list[float]:
    """The silence the source speaker left before each sentence after the first (parallel to ``source_onsets``): from the last word of one sentence to the first of the next."""
    inside = [w for w in words if float(w["start"]) >= start_ms / 1000 - 0.05 and float(w["end"]) <= end_ms / 1000 + 0.05]
    inside.sort(key=lambda w: float(w["start"]))
    return [max(0.0, float(inside[i]["start"]) - float(inside[i - 1]["end"])) for i in range(1, len(inside))
            if str(inside[i - 1].get("text", "")).rstrip("\"'”’)").endswith(_STOPS)]


def _boundaries(x: np.ndarray, sr: int, min_s: float = BOUNDARY_MIN_S) -> list[tuple[int, int]]:
    """The silences of at least ``min_s`` inside the audio, as (first sample, one past the last)."""
    win = max(1, int(0.02 * sr))
    energy = np.sqrt(np.convolve(x.astype(float) ** 2, np.ones(win) / win, mode="same"))
    quiet = energy < max(150.0, 0.03 * float(energy.max() or 1.0))
    runs: list[tuple[int, int]] = []
    i, n = 0, len(quiet)
    while i < n:
        if quiet[i]:
            j = i
            while j < n and quiet[j]:
                j += 1
            if (j - i) / sr >= min_s and i / sr > EDGE_S and (n - j) / sr > EDGE_S:
                runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def _match(zulu_fractions: Sequence[float], source_fractions: Sequence[float]) -> list[int | None]:
    """For each Zulu sentence start, the index of the source sentence start nearest in how far into the speech it is (in order, each source start used once), or None."""
    matched: list[int | None] = []
    last = -1
    for fraction in zulu_fractions:
        best, best_distance = None, MAX_MATCH_FRACTION
        for j in range(last + 1, len(source_fractions)):
            distance = abs(source_fractions[j] - fraction)
            if distance <= best_distance:
                best, best_distance = j, distance
            elif source_fractions[j] > fraction:
                break
        matched.append(best)
        if best is not None:
            last = best
    return matched


def _solve_sum(onsets: np.ndarray, gaps: np.ndarray, targets: Sequence[float | None], total: float | None, ceiling: np.ndarray | None = None) -> np.ndarray:
    """The change to each gap (seconds) that brings the matched onsets nearest to their targets inside the bounds, summing to ``total`` (None: whatever is best), an active-set
    least squares.  ``ceiling`` is the longest each gap may become (default MAX_GAP_S)."""
    m = len(gaps)
    if ceiling is None:
        ceiling = np.full(m, MAX_GAP_S)
    rows = [k for k, t in enumerate(targets) if t is not None]
    if not rows:
        return np.zeros(m)
    c = np.zeros((len(rows), m))
    for r, k in enumerate(rows):
        c[r, : k + 1] = 1.0                           # onset k moves by the sum of the changes of the gaps up to it
    error = np.array([float(targets[k]) - onsets[k] for k in rows])
    lo = np.maximum(MIN_GAP_S - gaps, -MAX_CHANGE_S)
    hi = np.minimum(ceiling - gaps, MAX_CHANGE_S)
    delta = np.zeros(m)
    free = np.ones(m, dtype=bool)
    fixed = np.zeros(m)
    ridge = 0.3                                       # a gap with no target of its own moves only as much as the others need
    for _ in range(m + 2):
        index = np.where(free)[0]
        if not len(index):
            break
        a = c[:, index].T @ c[:, index] + ridge * np.eye(len(index))
        b = c[:, index].T @ (error - c @ fixed)
        try:
            if total is None:
                solution = np.linalg.solve(a, b)
            else:
                kkt = np.block([[a, np.ones((len(index), 1))], [np.ones((1, len(index))), np.zeros((1, 1))]])
                solution = np.linalg.solve(kkt, np.concatenate([b, [total - fixed.sum()]]))[: len(index)]
        except np.linalg.LinAlgError:
            return np.zeros(m)
        delta = fixed.copy()
        delta[index] = solution
        over = (delta < lo - 1e-9) | (delta > hi + 1e-9)
        if not over[index].any():
            break
        worst = index[np.argmax(np.maximum(lo[index] - delta[index], delta[index] - hi[index]))]
        fixed[worst] = np.clip(delta[worst], lo[worst], hi[worst])
        free[worst] = False
    delta = np.clip(delta, lo, hi)
    if total is not None:
        for _ in range(m + 2):                        # the clipping can leave a little to put back: give it to the gaps that still have room
            residual = float(delta.sum()) - total
            if abs(residual) < 1e-6:
                break
            room = np.where(residual > 0, delta - lo, hi - delta)
            if float(room.sum()) <= 1e-9:
                return np.zeros(m)
            delta = delta - np.sign(residual) * np.minimum(room, abs(residual) * room / room.sum())
    return delta


def _solve(onsets: np.ndarray, gaps: np.ndarray, targets: Sequence[float | None], extra_s: float = 0.0, ceiling: np.ndarray | None = None) -> np.ndarray:
    """The best change to the gaps; the unit may get ``extra_s`` seconds longer in all (the silence it was short of its window by, spent where the source speaker paused) but never
    shorter."""
    delta = _solve_sum(onsets, gaps, targets, None, ceiling)
    total = float(delta.sum())
    if total > extra_s:                                           # (a net shortening is allowed: the silence taken out goes to the end of the unit, which keeps its length)
        return _solve_sum(onsets, gaps, targets, max(0.0, extra_s), ceiling)
    return delta


def retime(source_path: Path, output_path: Path, words: Sequence[Mapping[str, Any]], start_ms: int, end_ms: int, extra_ms: int = 0) -> dict[str, Any]:
    """Write ``output_path``: the unit's audio with its sentence gaps moved toward the source speaker's sentence onsets.  Returns a report; ``applied`` is False (and nothing is
    written) when there is nothing to do or the sentences cannot be matched with confidence."""
    with wave.open(str(source_path), "rb") as src:
        params = src.getparams()
        frames = src.readframes(params.nframes)
    if params.sampwidth != 2:
        return {"applied": False, "why": "not 16-bit audio"}
    sr, channels = params.framerate, params.nchannels
    x = np.frombuffer(frames, dtype=np.int16).reshape(-1, channels)
    mono = x[:, 0]
    runs = _boundaries(mono, sr)
    english, fractions = source_onsets(words, start_ms, end_ms)
    report: dict[str, Any] = {"zulu_sentence_ends": len(runs), "source_sentence_starts": len(english), "applied": False}
    if not runs or not english:
        report["why"] = "a single sentence on one side"
        return report
    # where each Zulu sentence (after a boundary) starts now, and how much speech came before it
    onsets = np.array([b / sr for _, b in runs])
    gaps = np.array([(b - a) / sr for a, b in runs])
    silent = sum(b - a for a, b in runs) / sr
    total_speech = max(len(mono) / sr - silent, 1e-6)
    speech_before = []
    previous_end = 0
    spoken = 0.0
    for a, b in runs:
        spoken += (a - previous_end) / sr
        previous_end = b
        speech_before.append(spoken / total_speech)
    matches = _match(speech_before, fractions)
    targets = [english[j] if j is not None and abs(english[j] - onsets[k]) <= MAX_ONSET_ERROR_S else None for k, j in enumerate(matches)]
    report["matched"] = sum(1 for t in targets if t is not None)
    if not report["matched"]:
        report["why"] = "no Zulu sentence sits near a source sentence"
        return report
    # A gap is never made longer than the voice's own or the speaker's own pause before the matched sentence, whichever is more: pushing a sentence later toward the speaker's
    # mouth only makes dead air (prod-86: a 1.0 s gap was lengthened to 1.9 s to reach the English "Speak to us", and the Zulu stood silent while the reporter had already turned).
    spoken_gaps = source_gaps(words, start_ms, end_ms)
    ceiling = np.array([min(MAX_GAP_S, max(float(gaps[k]), float(spoken_gaps[j]) if j is not None and targets[k] is not None and j < len(spoken_gaps) else 0.0))
                        for k, j in enumerate(matches)])
    delta = _solve(onsets, gaps, targets, extra_ms / 1000.0, ceiling)
    if float(np.abs(delta).max()) < MIN_SHIFT_S:
        report["why"] = "already where the source speaker's sentences begin"
        return report
    shifts = np.cumsum(delta)
    new_onsets = onsets + shifts
    wanted = [(k, t) for k, t in enumerate(targets) if t is not None]
    report["onset_error_s"] = {"before": round(float(np.sqrt(np.mean([(onsets[k] - t) ** 2 for k, t in wanted]))), 2),
                               "after": round(float(np.sqrt(np.mean([(new_onsets[k] - t) ** 2 for k, t in wanted]))), 2)}
    report["gap_changes_s"] = [round(float(d), 2) for d in delta]
    report["sentences"] = [{"zulu_starts_s": round(float(onsets[k]), 2), "source_starts_s": None if targets[k] is None else round(float(targets[k]), 2),
                            "zulu_starts_after_s": round(float(new_onsets[k]), 2)} for k in range(len(onsets))]
    if report["onset_error_s"]["after"] >= report["onset_error_s"]["before"]:
        report["why"] = "the move would not bring the sentences nearer"
        return report
    # rebuild: speech untouched, each boundary's silence made the new length by cutting from, or adding to, its middle; the total length is kept to the sample
    pieces: list[np.ndarray] = []
    cursor = 0
    changes = [int(round(d * sr)) for d in delta]
    allowed = int(round(max(0.0, extra_ms / 1000.0) * sr))
    if sum(changes) > allowed:
        changes[-1] -= sum(changes) - allowed                      # rounding never makes the unit longer than the room it has
    tail = -sum(changes) if sum(changes) < 0 else 0                  # silence taken out of the gaps goes to the end: the unit is never shorter than it was
    for (a, b), change in zip(runs, changes):
        pieces.append(x[cursor:a])
        silence = x[a:b]
        if change < 0:
            keep = max(0, len(silence) + change)
            pieces.append(np.concatenate([silence[: keep // 2], silence[len(silence) - (keep - keep // 2):]]) if keep else silence[:0])
        elif change > 0:
            pieces.append(np.concatenate([silence[: len(silence) // 2], np.zeros((change, channels), dtype=x.dtype), silence[len(silence) // 2:]]))
        else:
            pieces.append(silence)
        cursor = b
    pieces.append(x[cursor:])
    if tail:
        pieces.append(np.zeros((tail, channels), dtype=x.dtype))
    out = np.concatenate(pieces)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as dst:
        dst.setparams((channels, params.sampwidth, sr, 0, "NONE", "not compressed"))
        dst.writeframes(out.astype(np.int16).tobytes())
    report.update(applied=True, samples_before=int(len(x)), samples_after=int(len(out)), duration_after_ms=int(round(len(out) * 1000 / sr)))
    return report
