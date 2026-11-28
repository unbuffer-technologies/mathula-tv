"""Take the smoothest of several syntheses of each span, and join the spans over room tone.

OmniVoice's output varies from one run to the next on the same text, voice and reference: a generation sometimes starts with a pitch spike (prod-119 at 0:27: 348 Hz against a speaker who stays
between 130 and 290, falling to 135 Hz within a quarter of a second) or leaves quiet, half-voiced stretches inside the speech.  A listener hears it as jumping into the middle of a sentence.  It is
not the text (the same sentence regenerated four times had it in two) so the remedy is to generate again and keep the better take.  TTS is cheap, so naturalness is not traded for a call.

Every span is scored on what a listener would notice and nothing else (the number only ranks takes of the SAME span, it never judges a dub): ``onset_ratio`` (the highest pitch in the first 0.3 s
of speech against the span's median pitch; the speaker's own 95th percentile is about 1.6x) and the seconds of quiet stretches inside the speech.  A unit is synthesised up to MAX_TAKES times,
the best take of each span is kept (spans are separate generations), and the spans are joined with the room tone of the source recording (``room_tone``) where the model leaves exact zeros.
Nothing is cut, stretched or pinned.
"""
from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

import numpy as np

from .internal_silence import FRAME_S, silent_runs
from . import room_tone

MAX_TAKES = 3
ONSET_WINDOW_S = 0.3
ONSET_OK_RATIO = 1.6          # the speaker's 95th percentile over their median
ONSET_WEIGHT = 4.0            # a spike costs as much as this many seconds of quiet stretch per unit of ratio over the limit
FALSE_START_WINDOW_S = 0.8
FALSE_START_WEIGHT = 3.0
QUIET_MARGIN_DB = 14.0
QUIET_MIN_RUN_S = 0.2
F0_MIN, F0_MAX = 70.0, 400.0
VOICED = 0.45


def f0_track(audio: np.ndarray, rate: int, *, hop_s: float = 0.02, win_s: float = 0.04) -> np.ndarray:
    """Pitch in Hz per hop (NaN where unvoiced or silent), by the autocorrelation of each 40 ms window (numpy only)."""
    n, hop = int(win_s * rate), int(hop_s * rate)
    if len(audio) < n + hop:
        return np.zeros(0)
    starts = np.arange(0, len(audio) - n, hop)
    frames = np.stack([audio[s:s + n] for s in starts]) * np.hanning(n)
    size = 1 << int(np.ceil(np.log2(2 * n)))
    spectrum = np.fft.rfft(frames, n=size, axis=1)
    ac = np.fft.irfft(np.abs(spectrum) ** 2, axis=1)[:, :n]
    ac = ac / (ac[:, :1] + 1e-12)
    lo, hi = int(rate / F0_MAX), int(rate / F0_MIN)
    peak = lo + np.argmax(ac[:, lo:hi], axis=1)
    strength = ac[np.arange(len(ac)), peak]
    loud = np.sqrt((frames ** 2).mean(axis=1)) > 1e-3
    return np.where(loud & (strength > VOICED), rate / peak, np.nan)


def onset_ratio(core: np.ndarray, rate: int) -> float:
    """The highest pitch in the first ONSET_WINDOW_S of speech over the span's median pitch (1.0 when there is nothing to compare)."""
    track = f0_track(core, rate)
    voiced = track[~np.isnan(track)]
    if len(voiced) < 10:
        return 1.0
    head = track[: max(1, int(ONSET_WINDOW_S / 0.02))]
    head = head[~np.isnan(head)]
    return float(head.max() / np.median(voiced)) if len(head) else 1.0


def quiet_seconds(core: np.ndarray, rate: int, min_run_s: float | None = None) -> float:
    """Seconds of quiet stretch inside the speech: the runs of at least ``min_run_s`` (QUIET_MIN_RUN_S by default).  A caller that knows how long the speaker's own pauses are passes that, so that
    the pauses of natural phrasing are not counted against a take: the take with the fewest pauses is the most breathless one, and was the one kept (prod-129 0:39, 'a crazy rush')."""
    return round(sum((last - first) * FRAME_S for first, last in silent_runs(core, rate, min_run_s=max(QUIET_MIN_RUN_S, min_run_s or 0.0), margin_db=QUIET_MARGIN_DB)), 3)


def false_start_s(core: np.ndarray, rate: int) -> float:
    """How long a quiet dip (12 dB under the speech, at least 0.1 s) starting in the first FALSE_START_WINDOW_S of speech lasts: a stray sound, then a pause, then the words."""
    runs = [(a, b) for a, b in silent_runs(core, rate, min_run_s=0.1, margin_db=12.0) if a * FRAME_S < FALSE_START_WINDOW_S]
    return round(sum((b - a) * FRAME_S for a, b in runs), 3)


DROPOUT_MIN_S = 0.02          # exact digital zero this long inside speech is not a pause, it is the model dropping out (prod-120 at 0:18: 96 ms of zeros, then a click as the next word came in)
DROPOUT_COST = 2.0            # a take with one costs as much as this many seconds of quiet stretch, so any clean take of the same span is kept instead


def dropouts(audio: np.ndarray, rate: int, *, min_s: float = DROPOUT_MIN_S) -> list[tuple[int, int]]:
    """(first sample, last sample + 1) of every run of exact digital zero at least ``min_s`` long that has sound on both sides: the ends of a span are left alone (they are the model's padding)."""
    zero = np.abs(audio) < 1e-6
    edges = np.diff(np.concatenate(([0], zero.astype(np.int8), [0])))
    runs = list(zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]))
    return [(int(a), int(b)) for a, b in runs if a > 0 and b < len(audio) and (b - a) >= int(min_s * rate)]


STRAY_MAX_S = 0.12            # a sound this short between a span's end and a dropout is not a word: prod-120 at 0:18, 96 ms of zeros and then a 40 ms click as the very last thing the model made


def strip_stray_ends(core: np.ndarray, rate: int, *, max_s: float = STRAY_MAX_S) -> tuple[np.ndarray, int, int]:
    """(the span without a stray fragment at either end, samples removed from the head, samples removed from the tail).  A fragment is what is left of the audio on the far side of a dropout when it is
    shorter than ``max_s``: the model's own end-of-generation click, never speech.  The caller adds what was removed to the span's pad so the timeline does not move."""
    holes = dropouts(core, rate)
    head = tail = 0
    if holes and holes[0][0] < int(max_s * rate):
        head = holes[0][1]
    if holes and len(core) - holes[-1][1] < int(max_s * rate) and holes[-1][0] > head:
        tail = len(core) - holes[-1][0]
    if head or tail:
        return core[head: len(core) - tail], head, tail
    return core, 0, 0


TAIL_QUIET_DB = 25.0          # quiet is this far under the span's speech level (its 95th percentile): the voice's pauses sit 25-45 dB under it
TAIL_MIN_PAUSE_S = 0.08       # a pause between the last word and the one after it is at least this long to be a place to cut
TAIL_WINDOW_BEFORE_S = 0.10   # looked for from this long after the aligner's start of the last word ...
TAIL_WINDOW_AFTER_S = 0.40    # ... to this long after its start of the next: the aligner's word edges are out by up to half a second at a pause (prod-y1 0:56: 0.5 s)


def locate_tail(core: np.ndarray, rate: int, last_word_start_s: float, next_word_start_s: float) -> dict[str, float] | None:
    """Where the last word's sound really ends and the next word's really begins, from the signal: {speech_end_s, next_onset_s, pause_s}, or None when there is no pause between them to cut in.
    The aligner's word edges say roughly where to look and no more: it counts the pause after a word as part of that word (prod-y1 0:56: the year 'ended' where the next word began, 0.5 s late, so
    the cut kept 90 ms of the next word, and the pause the voice left was recorded as 0).  The pause is the longest quiet (TAIL_QUIET_DB under the speech, at least TAIL_MIN_PAUSE_S) that starts
    between the last word and a little after the next one and has sound after it; the cut is made in it.  None means continuous speech into the next word: the caller falls back to the aligner's edges."""
    frame = int(0.01 * rate)
    count = len(core) // frame
    if count < 30:
        return None
    db = 20.0 * np.log10(np.sqrt((core[: count * frame].reshape(count, frame) ** 2).mean(axis=1)) + 1e-9)
    quiet = db < np.percentile(db, 95) - TAIL_QUIET_DB
    lo, hi = max(0, int((last_word_start_s + TAIL_WINDOW_BEFORE_S) * 100)), min(count, int((next_word_start_s + TAIL_WINDOW_AFTER_S) * 100))
    found, i = [], 0
    while i < count:
        if quiet[i]:
            j = i
            while j < count and quiet[j]:
                j += 1
            if lo <= i < hi and j < count and (j - i) * 0.01 >= TAIL_MIN_PAUSE_S and i > 0:          # sound before it, in the window, with sound after it
                found.append((i, j))
            i = j
        else:
            i += 1
    if not found:
        return None
    a, b = max(found, key=lambda run: ((run[1] - run[0]), -abs(run[1] * 0.01 - next_word_start_s)))
    return {"speech_end_s": a * 0.01, "next_onset_s": b * 0.01, "pause_s": (b - a) * 0.01}


def verify_cut_end(core: np.ndarray, rate: int) -> str | None:
    """What is wrong with the end of a cut span, or None: ``cut_in_sound`` (it ends while the voice is still loud: the cut is inside a word) or ``leaked_onset`` (a short burst after a real quiet:
    the first sound of the word that should have been cut away).  The seam audit (cut_audit) finds the same thing in the finished dub; this finds it where the cut is made, so it can be corrected there."""
    frame = int(0.01 * rate)
    count = len(core) // frame
    if count < 30:
        return None
    db = 20.0 * np.log10(np.sqrt((core[: count * frame].reshape(count, frame) ** 2).mean(axis=1)) + 1e-9)
    if strip_leaked_onset(core, rate)[1]:
        return "leaked_onset"
    return "cut_in_sound" if db[-3:].max() > np.percentile(db, 95) - 12.0 else None


STRAY_SOUND_MARGIN_S = 0.20   # a sound this far from every word of the text (the aligner's word edges are out by up to half a second at a pause, but not by this much at the front of a word) ...
STRAY_SOUND_MIN_S = 0.08      # ... that lasts this long (shorter is a click, found elsewhere) ...
STRAY_SOUND_MAX_S = 0.60      # ... and no longer (a longer sound is a word the aligner lost, not an extra one)
STRAY_ISOLATION_S = 0.06      # ... with this much quiet either side, is a sound the voice made that no word of the text accounts for (prod-y1 3:05.8: 300 ms at speech level, 0.7 s after "phansi")


def find_stray_sounds(audio: np.ndarray, rate: int, word_spans_s: Sequence[tuple[float, float]], *, margin_s: float = STRAY_SOUND_MARGIN_S) -> list[tuple[int, int]]:
    """(first sample, one past the last) of every sound in ``audio`` that no word accounts for: ``word_spans_s`` are the (start, end) seconds the forced aligner gives the words of the text the voice was
    asked to say.  The voice sometimes adds a sound of its own between two sentences (a breath, an 'uh', half a syllable); it sits in the gap between two words, apart from both, and it is what the ear
    hears as a hitch.  Never a sound inside or beside a word."""
    frame = int(0.01 * rate)
    count = len(audio) // frame
    if count < 30 or not word_spans_s:
        return []
    db = 20.0 * np.log10(np.sqrt((audio[: count * frame].reshape(count, frame) ** 2).mean(axis=1)) + 1e-9)
    sound = db > np.percentile(db, 95) - LEAK_QUIET_DB
    found, i = [], 0
    while i < count:
        if sound[i]:
            j = i
            while j < count and sound[j]:
                j += 1
            start_s, end_s = i * 0.01, j * 0.01
            quiet_before = 0
            while i - 1 - quiet_before >= 0 and not sound[i - 1 - quiet_before]:
                quiet_before += 1
            quiet_after = 0
            while j + quiet_after < count and not sound[j + quiet_after]:
                quiet_after += 1
            apart = all(end_s < ws - margin_s or start_s > we + margin_s for ws, we in word_spans_s)
            if STRAY_SOUND_MIN_S <= end_s - start_s <= STRAY_SOUND_MAX_S and apart and quiet_before * 0.01 >= STRAY_ISOLATION_S and quiet_after * 0.01 >= STRAY_ISOLATION_S and (quiet_before < count and quiet_after < count):
                found.append((i * frame, j * frame))
            i = j
        else:
            i += 1
    return found


def quiet_over(audio: np.ndarray, rate: int, spans: Sequence[tuple[int, int]]) -> np.ndarray:
    """``audio`` with each span (a stray sound) replaced by the quiet around it, faded in and out over 10 ms: noise at the level of the quiet either side, so the pause stays a pause with the room in it."""
    out = np.array(audio, dtype=np.float32, copy=True)
    rng = np.random.default_rng(len(audio))
    fade = int(0.01 * rate)
    for a, b in spans:
        reach = int(0.4 * rate)
        around = np.concatenate([out[max(0, a - reach): a], out[b: b + reach]])
        floor = float(np.sqrt(np.mean(np.sort(around ** 2)[: max(1, len(around) // 2)]))) if len(around) else 0.0
        lo, hi = max(0, a - fade), min(len(out), b + fade)
        fill = rng.normal(0.0, floor, hi - lo).astype(np.float32)
        weight = np.ones(hi - lo, dtype=np.float32)
        ramp = np.linspace(0.0, 1.0, min(fade, (hi - lo) // 2), dtype=np.float32)
        weight[: len(ramp)], weight[len(weight) - len(ramp):] = ramp, ramp[::-1]
        out[lo:hi] = out[lo:hi] * (1.0 - weight) + fill * weight
    return out


LEAK_MAX_S = 0.30             # what is left of the next word at the end of a cut span is shorter than this ...
LEAK_MIN_QUIET_S = 0.12       # ... and comes after at least this much quiet (a word is never only a fragment of itself after a real pause: the span's own words end before the quiet)
LEAK_QUIET_DB = 25.0          # quiet is this far under the span's loud (95th percentile) level; the voice's pauses are 25-45 dB under it (prod-y1 0:56: -50 to -70 dB against -22)


def strip_leaked_onset(core: np.ndarray, rate: int, *, not_before_s: float = 0.0, max_s: float = LEAK_MAX_S, min_quiet_s: float = LEAK_MIN_QUIET_S, quiet_db: float = LEAK_QUIET_DB) -> tuple[np.ndarray, int]:
    """(the span without the onset of the next word that its cut let through, samples removed from the tail).  The cut at the end of a span is made at the aligner's word edges, which are not exact
    (prod-y1 0:56, "ngo nineteen ninety-four." and then the follow word "Nge-": the edges said the year ended where the next word began, 0.5 s later than it did), so when the span's last
    sound is a short burst after a real quiet, it is the first sound of the word that should have been cut away: the cut moves into the middle of the quiet.  ``not_before_s`` is the earliest it may fall (the
    last word's own start and a little more): a last word that follows a pause is never taken for a leak.  The silence taken off is added to the span's pad by the caller, so the timeline does not move."""
    frame = int(0.01 * rate)
    count = len(core) // frame
    if count < 20:
        return core, 0
    db = 20.0 * np.log10(np.sqrt((core[: count * frame].reshape(count, frame) ** 2).mean(axis=1)) + 1e-9)
    loud = db > np.percentile(db, 95) - quiet_db
    i = count
    while i > 0 and loud[i - 1]:
        i -= 1
    if i == count or (count - i) * 0.01 > max_s:                    # the span ends in quiet (the usual case), or in something longer than a fragment
        return core, 0
    j = i
    while j > 0 and not loud[j - 1]:
        j -= 1
    if j == 0 or (i - j) * 0.01 < min_quiet_s:
        return core, 0
    cut = (j + (i - j) // 2) * frame
    if cut < int(not_before_s * rate):
        return core, 0
    return core[:cut], len(core) - cut


def score_span(core: np.ndarray, rate: int, pause_ok_s: float | None = None) -> dict[str, Any]:
    ratio = onset_ratio(core, rate)
    quiet = quiet_seconds(core, rate, pause_ok_s)
    dip = false_start_s(core, rate)
    holes = len(dropouts(core, rate))
    return {"score": round(quiet + ONSET_WEIGHT * max(0.0, ratio - ONSET_OK_RATIO) + FALSE_START_WEIGHT * dip + DROPOUT_COST * holes, 3), "onset_ratio": round(ratio, 2), "quiet_s": quiet, "false_start_s": dip,
            "dropouts": holes}


LEAD_MAX_CHARS = 220          # the sentence that leads a following span: a longer one is not repeated
CUT_WINDOW_BEFORE_S = 0.6
CUT_WINDOW_AFTER_S = 0.7
CUT_QUIET_DB = 14.0
CUT_MIN_QUIET_S = 0.04
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def lead_sentence(previous_span: str) -> str | None:
    """The last sentence of the span before: what the following span repeats at its start so that it does not begin cold (None when there is none or it is too long to repeat)."""
    sentences = [s.strip() for s in SENTENCE_END.split(str(previous_span or "").strip()) if s.strip()]
    return sentences[-1] if sentences and len(sentences[-1]) <= LEAD_MAX_CHARS else None


def active_extent(core: np.ndarray, rate: int, *, below_peak_db: float = 35.0) -> tuple[float, float]:
    """(when the speech starts, when it ends) in a generation, seconds: the first and last 10 ms frames within ``below_peak_db`` of its loudest."""
    frame = int(0.01 * rate)
    count = len(core) // frame
    if count < 3:
        return 0.0, len(core) / rate
    db = 20.0 * np.log10(np.sqrt((core[: count * frame].reshape(count, frame) ** 2).mean(axis=1)) + 1e-9)
    live = np.nonzero(db > db.max() - below_peak_db)[0]
    return float(live[0] * 0.01), float((live[-1] + 1) * 0.01)


MEL_BANDS = 24
DTW_HOP_S = 0.02
DTW_WIN_S = 0.04
DTW_FREE_START_S = 0.6        # the lead may start this late in the generation
CUT_WINDOW_ALIGNED_S = 0.25   # when the end of the lead is aligned (not guessed from a length) the cut is looked for this close to it


def _mel_features(audio: np.ndarray, rate: int) -> np.ndarray:
    """Log mel energies per 20 ms hop with each frame's own mean removed (so loudness does not matter): (frames, MEL_BANDS)."""
    n, hop = int(DTW_WIN_S * rate), int(DTW_HOP_S * rate)
    if len(audio) < n + hop:
        return np.zeros((0, MEL_BANDS))
    starts = np.arange(0, len(audio) - n, hop)
    frames = np.stack([audio[s:s + n] for s in starts]) * np.hanning(n)
    power = np.abs(np.fft.rfft(frames, n=1 << int(np.ceil(np.log2(n))), axis=1)) ** 2
    bins = power.shape[1]
    mel = lambda hz: 2595.0 * np.log10(1.0 + hz / 700.0)
    inverse = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    edges = inverse(np.linspace(mel(60.0), mel(min(8000.0, rate / 2.0)), MEL_BANDS + 2))
    at = np.clip(np.round(edges / (rate / 2.0) * (bins - 1)).astype(int), 0, bins - 1)
    bank = np.zeros((bins, MEL_BANDS))
    for b in range(MEL_BANDS):
        lo, mid, hi = at[b], max(at[b + 1], at[b] + 1), max(at[b + 2], at[b] + 2)
        bank[lo:mid, b] = np.linspace(0.0, 1.0, mid - lo, endpoint=False)
        bank[mid:min(hi, bins), b] = np.linspace(1.0, 0.0, min(hi, bins) - mid, endpoint=False)
    logmel = np.log(power @ bank + 1e-8)
    return logmel - logmel.mean(axis=1, keepdims=True)


def align_end(generation: np.ndarray, spoken_alone: np.ndarray, rate: int) -> float | None:
    """Where in ``generation`` the speech of ``spoken_alone`` ends, seconds.  The generation begins with the same sentence (spoken by the same voice from the same reference, at a pace of its own) and
    carries on with new text; the sentence spoken alone is warped onto the start of the generation (dynamic time warping, the generation's end left free) and the frame it finishes on is the answer.
    None when either is too short to align.  This replaces guessing from the sentence's length alone, which could land on a quiet inside the last word (the p of "phoyisa") and leave half of it in."""
    a, g = _mel_features(spoken_alone, rate), _mel_features(generation, rate)
    if len(a) < 5 or len(g) < len(a):
        return None
    g = g[: 2 * len(a) + int(DTW_FREE_START_S / DTW_HOP_S) + 10]                     # the lead cannot take more than twice as long as it did alone
    cost = np.sqrt(((a[:, None, :] - g[None, :, :]) ** 2).mean(axis=2))              # (alone frames, generation frames)
    n, m = cost.shape
    big = 1e9
    previous = np.full(m, big)
    free = min(m, int(DTW_FREE_START_S / DTW_HOP_S) + 1)
    previous[:free] = cost[0, :free]
    for i in range(1, n):
        stay = previous
        one = np.concatenate(([big], previous[:-1]))
        two = np.concatenate(([big, big], previous[:-2]))
        previous = cost[i] + np.minimum(stay, np.minimum(one, two))
    end = int(np.argmin(previous))
    return round(end * DTW_HOP_S + DTW_WIN_S, 3)


TAIL_KEEP_S = 0.12            # after the end of a span's last word the cut keeps at most this much (the word's own decay) of the quiet that follows
TAIL_MIN_KEEP_S = 0.03        # and at least this much: the aligner's word end can sit a little inside the sound


HEAD_KEEP_S = 0.08            # before the first new word the cut keeps this much of the quiet, so its onset is not clipped (the aligner's word start can sit a little late)
HEAD_MIN_KEEP_S = 0.03
PAD_S = 0.2                   # the model's own padding at both ends of a generation, which a planned gap is measured from (span_cuts.PAD_MS)


def edge_keep_s(silence_s: float, *, share: float, most_s: float, least_s: float) -> float:
    """How much quiet one side of a join keeps next to its word: its share of the silence the join is to have, between a little (the word's own edge) and ``most_s``."""
    return min(most_s, max(least_s, share * max(0.0, silence_s)))


def join_target_s(planned_gap_ms: float, natural_gap_s: float | None) -> float:
    """The silence between the last word of one span and the first word of the next: what the voice itself left there (the first word of the next span spoken after this one, its gap from the last word:
    natural_gap_s) or the planned one (the model's padding plus the planned gap, which is the speaker's own pause where they paused), whichever is longer.  A plan never squeezes the voice's own spacing, and a
    longer held pause is kept."""
    return max(0.0, PAD_S + planned_gap_ms / 1000.0, natural_gap_s or 0.0)


def head_cut(quiet_mid_s: float, lead_end_s: float, new_start_s: float, keep_s: float = HEAD_KEEP_S) -> float:
    """Where to cut a span after its repeated lead: ``keep_s`` before the first new word (never nearer than HEAD_MIN_KEEP_S, never back inside the lead), the middle of the quiet when the quiet is shorter."""
    wanted = min(max(quiet_mid_s, new_start_s - keep_s), new_start_s - HEAD_MIN_KEEP_S)
    return max(wanted, lead_end_s + 0.02) if new_start_s - lead_end_s > 0.05 else (lead_end_s + new_start_s) / 2.0


def join_fill_s(target_s: float, kept_tail_s: float, kept_head_s: float) -> float:
    """The room tone to put between two cut spans so that the silence between the last word of the first and the first word of the second is ``target_s``, less the quiet each cut already kept."""
    return max(0.0, target_s - kept_tail_s - kept_head_s)


STRAY_MIN_S = 0.15            # sound this long (or longer) before the first word of a span or after its last is looked at
STRAY_DB = 25.0               # and is a stray sound when its loudest 10 ms is within this many dB of the span's speech


def _loud_between(core: np.ndarray, rate: int, start_s: float, end_s: float) -> bool:
    frame = int(0.01 * rate)
    count = len(core) // frame
    if count < 10 or end_s <= start_s:
        return False
    db = 20.0 * np.log10(np.sqrt((core[: count * frame].reshape(count, frame) ** 2).mean(axis=1)) + 1e-9)
    speech = float(np.median(db[db > np.percentile(db, 30)]))
    lo, hi = max(0, int(start_s * 100)), min(count, int(end_s * 100))
    return hi > lo and float(db[lo:hi].max()) > speech - STRAY_DB


def stray_before(core: np.ndarray, rate: int, first_word_s: float) -> float | None:
    """Where to cut the front of a span that begins cold, or None.  The service's word times say where its first word starts; sound well before that, as loud as the speech, is not in the text (prod-124 at
    1:16: 0.4 s of voiced sound, 'out', before the first word of a span).  The cut keeps HEAD_KEEP_S in front of the word, as a cut after a lead does."""
    if first_word_s - 0.0 < STRAY_MIN_S + HEAD_KEEP_S or not _loud_between(core, rate, 0.0, first_word_s - HEAD_KEEP_S):
        return None
    return round(first_word_s - HEAD_KEEP_S, 3)


def stray_after(core: np.ndarray, rate: int, last_word_end_s: float) -> float | None:
    """Where to cut the end of a span that nothing follows, or None: sound after its last word's end, as loud as the speech, is not in the text."""
    length = len(core) / rate
    if length - last_word_end_s < STRAY_MIN_S + TAIL_KEEP_S or not _loud_between(core, rate, last_word_end_s + TAIL_KEEP_S, length):
        return None
    return round(last_word_end_s + TAIL_KEEP_S, 3)


def fade_out(core: np.ndarray, rate: int, seconds: float = 0.012) -> np.ndarray:
    """The audio with its last few milliseconds faded out, so a cut made while a word is still dying away does not click."""
    n = min(len(core), int(seconds * rate))
    if n < 2:
        return core
    out = core.copy()
    out[-n:] *= np.linspace(1.0, 0.0, n, dtype=out.dtype)
    return out


def find_cut(core: np.ndarray, rate: int, expected_s: float, *, before_s: float = CUT_WINDOW_BEFORE_S, after_s: float = CUT_WINDOW_AFTER_S) -> tuple[float, bool]:
    """Where the repeated lead sentence ends in a generation: (seconds, found).  The lead is spoken first, so its end is near ``expected_s`` (how long it took on its own, from where the speech
    starts); the cut is the middle of the quietest stretch (at least CUT_MIN_QUIET_S, CUT_QUIET_DB under the speech) closest to that, and when there is none the quietest frame there, not found."""
    frame = int(0.01 * rate)
    count = len(core) // frame
    db = 20.0 * np.log10(np.sqrt((core[: count * frame].reshape(count, frame) ** 2).mean(axis=1)) + 1e-9)
    speech = float(np.median(db[db > np.percentile(db, 30)]))
    lo, hi = max(0, int((expected_s - before_s) * 100)), min(count - 1, int((expected_s + after_s) * 100))
    if hi <= lo:
        return expected_s, False
    quiet = db < speech - CUT_QUIET_DB
    runs, i = [], lo
    while i <= hi:
        if quiet[i]:
            j = i
            while j <= hi and quiet[j]:
                j += 1
            if (j - i) * 0.01 >= CUT_MIN_QUIET_S:
                runs.append(((i + j) / 2 * 0.01, (j - i) * 0.01))
            i = j
        else:
            i += 1
    if runs:
        return min(runs, key=lambda run: abs(run[0] - expected_s))[0], True
    return (lo + int(np.argmin(db[lo:hi + 1]))) * 0.01, False


def fade_in(core: np.ndarray, rate: int, seconds: float = 0.008) -> np.ndarray:
    """The audio with its first few milliseconds faded in, so a cut made inside a sound does not click."""
    n = min(len(core), int(seconds * rate))
    if n < 2:
        return core
    out = core.copy()
    out[:n] *= np.linspace(0.0, 1.0, n, dtype=out.dtype)
    return out


def trim_zero_pads(segment: np.ndarray) -> tuple[np.ndarray, int, int]:
    """(the span without the exact-zero padding the model puts at both ends, lead samples, tail samples)."""
    live = np.nonzero(np.abs(segment) > 1e-6)[0]
    if len(live) == 0:
        return segment, 0, 0
    return segment[live[0]: live[-1] + 1], int(live[0]), int(len(segment) - 1 - live[-1])


def split_spans(answer: Mapping[str, Any], audio: np.ndarray, rate: int) -> list[np.ndarray]:
    return [audio[int(float(s) * rate): int((float(s) + ms / 1000.0) * rate)] for s, ms in zip(answer.get("span_starts_s") or [], answer.get("span_ms") or [])]


def fill_dropouts(audio: np.ndarray, rate: int, tone: np.ndarray, delta_db: float, speech_db: float) -> np.ndarray:
    """The audio with every digital-zero run inside it (``dropouts``) replaced by room tone, so a stretch the model left at exact zero does not play as a hole with the next word clicking in out of it."""
    holes = dropouts(audio, rate)
    if not holes:
        return audio
    out = audio.copy()
    for a, b in holes:
        out[a:b] = room_tone.fill(b - a, tone, delta_db, speech_db)
    return out


def assemble(cores: Sequence[tuple[np.ndarray, int, int]], gaps_ms: Sequence[int], rate: int, tone: np.ndarray, delta_db: float) -> tuple[np.ndarray, list[float]]:
    """The spans one after another, each join as long as the model's two pads and the planned gap would have made it (shorter for a negative gap, none at all when they cancel), filled with room tone.  (audio, the start of each span in it)."""
    speech = np.concatenate([c[0] for c in cores]) if cores else np.zeros(0, dtype=np.float32)
    live = np.abs(speech) > 1e-6
    speech_db = 20.0 * float(np.log10(np.sqrt((speech[live] ** 2).mean()) + 1e-9)) if live.any() else -30.0
    pieces: list[np.ndarray] = []
    starts: list[float] = []
    position = 0
    for index, (core, lead, tail) in enumerate(cores):
        if index == 0 and lead:
            pieces.append(np.zeros(lead, dtype=np.float32))
            position += lead
        starts.append(round((position - lead) / rate, 3))          # a span starts at its own lead pad, as the service reports it
        pieces.append(core)
        position += len(core)
        if index + 1 < len(cores):
            join = max(0, tail + int(((gaps_ms[index] if index < len(gaps_ms) else 0) / 1000.0) * rate) + cores[index + 1][1])          # a negative gap closes the model's own pads: the speaker ran on
            if join:
                pieces.append(room_tone.fill(join, tone, delta_db, speech_db))
            position += join
        elif tail:
            pieces.append(np.zeros(tail, dtype=np.float32))
            position += tail
    return fill_dropouts(np.concatenate(pieces).astype(np.float32), rate, tone, delta_db, speech_db), starts
