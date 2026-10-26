"""Find speech the English transcript missed.

In job 13eacc6f the speaker said "Rupert is going to pay for water and electricity". The transcript has "Rupert is" at 422 s
and then nothing until "We don't care" at 426 s: four seconds of speech were never written down, so the dub said a dangling
"uRupert uzo—" and the meaning of that sentence was lost before the agent saw it.

For every pause in the transcript of at least ``MIN_GAP_SECONDS`` this measures the audio in the gap. Where it is clearly louder
than the clip's quiet floor (speech, not silence), the window is transcribed again, and the words that fall inside the gap are
added to the segment they continue. Applause and crowd noise are loud too, so a window that yields fewer than two words is ignored.

The result is a list of corrections in the form ``source_suspects`` uses, so they reach the frozen transcript, the agent and the
meaning check the same way.
"""
from __future__ import annotations

import re
import wave
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

MIN_GAP_SECONDS = 1.2
MAX_GAPS = 25
MIN_WORDS = 2
_FRAME = 0.5
END_TAIL_SECONDS = 0.15              # a recovered sentence's window ends a little after its last word
NEXT_WORD_MARGIN_SECONDS = 0.1       # ... but never within this of where the next speech starts


def _end(word: Mapping[str, Any]) -> float:
    return float(word.get("end") or float(word.get("start") or 0.0) + 0.2)


def find_gaps(words: Sequence[Mapping[str, Any]], min_gap: float = MIN_GAP_SECONDS) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    ordered = sorted(words, key=lambda w: float(w.get("start") or 0.0))
    return [(a, b) for a, b in zip(ordered, ordered[1:]) if float(b.get("start") or 0.0) - _end(a) >= min_gap]


CHUNK_SECONDS = 30.0
CHUNK_OVERLAP_SECONDS = 4.0           # each chunk keeps its middle, so no word is judged at the edge where it may be cut in half
COVERAGE_TOLERANCE_SECONDS = 0.3      # a second-pass word this close to a first-pass word is the same word
STRETCH_JOIN_SECONDS = 1.0            # second-pass words closer than this belong to one stretch of speech
STRETCH_MIN_SECONDS = 0.6


def second_pass_words(
    duration_s: float, transcribe: Callable[[Path, float, float], Sequence[Mapping[str, Any]]], audio: Path, *,
    chunk: float = CHUNK_SECONDS, overlap: float = CHUNK_OVERLAP_SECONDS, workers: int = 4,
) -> list[dict[str, Any]]:
    """Every word an independent pass over the whole audio hears, in chunks, with absolute times (seconds).

    The first pass is one call over the whole recording, and it can skip short bursts of speech (a speaker talking over a crowd).
    Chunks are transcribed on their own; each chunk contributes only its middle, so the two ends of a cut word never count.
    """
    from concurrent.futures import ThreadPoolExecutor

    step = chunk - overlap
    starts = [i * step for i in range(max(1, int(duration_s // step) + (1 if duration_s % step > overlap else 0)))]

    def listen(start: float) -> list[dict[str, Any]]:
        end = min(duration_s, start + chunk)
        keep_from = 0.0 if start == 0 else start + overlap / 2
        keep_to = duration_s if end >= duration_s else end - overlap / 2
        try:
            heard = transcribe(audio, start, end)
        except Exception:                                  # a chunk that cannot be heard (silence, a failed call) adds nothing
            return []
        words = []
        for w in heard:
            first = start + float(w.get("start") or 0.0)
            last = start + float(w.get("end") or float(w.get("start") or 0.0) + 0.2)
            if keep_from <= (first + last) / 2 < keep_to:
                words.append({**w, "start": first, "end": last})
        return words

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return sorted((w for part in pool.map(listen, starts) for w in part), key=lambda w: w["start"])


def uncovered_stretches(
    heard: Sequence[Mapping[str, Any]], words: Sequence[Mapping[str, Any]], *,
    tolerance: float = COVERAGE_TOLERANCE_SECONDS, join: float = STRETCH_JOIN_SECONDS, min_words: int = MIN_WORDS,
) -> list[tuple[float, float]]:
    """Stretches (start, end seconds) where the second pass hears at least ``min_words`` words and the first pass has none there."""
    first_pass = sorted((float(w.get("start") or 0.0), _end(w)) for w in words)
    starts = [s for s, _ in first_pass]
    import bisect

    def covered(word: Mapping[str, Any]) -> bool:
        a, b = float(word["start"]) - tolerance, _end(word) + tolerance
        index = bisect.bisect_right(starts, b)
        return any(first_pass[i][1] >= a for i in range(max(0, index - 6), index))

    stretches: list[list[Mapping[str, Any]]] = []
    for word in heard:
        if covered(word):
            continue
        if stretches and float(word["start"]) - _end(stretches[-1][-1]) <= join:
            stretches[-1].append(word)
        else:
            stretches.append([word])
    return [(float(group[0]["start"]), _end(group[-1])) for group in stretches
            if len(group) >= min_words and _end(group[-1]) - float(group[0]["start"]) >= STRETCH_MIN_SECONDS]


def bracketing_pair(words: Sequence[Mapping[str, Any]], start: float, end: float) -> tuple[Mapping[str, Any], Mapping[str, Any]] | None:
    """The first-pass words either side of a stretch (None at the very start or end of the recording, where no segment continues)."""
    ordered = sorted(words, key=lambda w: float(w.get("start") or 0.0))
    before = [w for w in ordered if _end(w) <= start + 0.05]
    after = [w for w in ordered if float(w.get("start") or 0.0) >= end - 0.05]
    return (before[-1], after[0]) if before and after else None


def frame_levels(wav_path: Path, frame: float = _FRAME) -> tuple[list[float], float]:
    """Loudness (dB) of each half-second of a wav, and the frame length in seconds."""
    import numpy as np

    with wave.open(str(wav_path), "rb") as reader:
        rate, width, channels = reader.getframerate(), reader.getsampwidth(), reader.getnchannels()
        data = np.frombuffer(reader.readframes(reader.getnframes()), dtype={1: np.int8, 2: np.int16, 4: np.int32}[width]).astype(np.float64)
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    size = max(1, int(rate * frame))
    count = len(data) // size
    rms = np.sqrt((data[: count * size].reshape(count, size) ** 2).mean(axis=1)) + 1e-9
    return list(20 * np.log10(rms / float(2 ** (8 * width - 1)))), frame


def has_speech(levels: Sequence[float], frame: float, start: float, end: float) -> bool:
    """Is the audio in [start, end] clearly above the clip's quiet floor? (floor = 10th percentile, speech = 75th)."""
    import numpy as np

    floor, loud = np.percentile(levels, 10), np.percentile(levels, 75)
    threshold = floor + 0.4 * (loud - floor)
    inside = [lv for i, lv in enumerate(levels) if start <= (i + 0.5) * frame <= end]
    return bool(inside) and sum(lv >= threshold for lv in inside) / len(inside) >= 0.4


def insertion(segment_text: str, after_word: str, inserted: str) -> dict[str, str] | None:
    """The correction (token -> replacement) that puts ``inserted`` right after the last ``after_word`` of the segment text."""
    matches = list(re.finditer(rf"(?<![A-Za-z']){re.escape(after_word.strip('.,!?'))}[.,!?—–\-]*", segment_text, re.IGNORECASE))
    if not matches:
        return None
    last = matches[-1]
    token = last.group(0)
    kept = token.rstrip(".,!?—–-")
    inserted = inserted.strip()
    cut_off = token.endswith(("—", "–", "-"))
    closes_sentence = token.endswith((".", "!", "?")) and not cut_off
    if cut_off or (closes_sentence and inserted[:1].islower()):
        # the transcript broke the sentence off (or ended it too early): the missing words continue it
        return {"token": token, "replacement": f"{kept} {inserted}" + ("" if inserted.endswith((".", "!", "?")) else ".")}
    return {"token": token, "replacement": f"{token} {inserted}"}            # the missing words are a sentence of their own


def find_missing_speech(
    segments: Sequence[Mapping[str, Any]], words: Sequence[Mapping[str, Any]], audio: Path, transcribe: Callable[[Path, float, float], Sequence[Mapping[str, Any]]],
    *, progress: Callable[[str], None] = print, second_pass: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Corrections that add words the transcript missed. ``transcribe(audio, start, end)`` returns word dicts with times relative to ``start``.

    Candidates are the pauses of the first-pass transcript, and -- when ``second_pass`` (the words of an independent chunked pass, see
    ``second_pass_words``) is given -- every stretch where that pass hears two or more words the first pass has none for.  Both are
    listened to again in a tight window before anything is added."""
    levels, frame = frame_levels(audio)
    found: list[dict[str, Any]] = []
    candidates: list[tuple[Mapping[str, Any], Mapping[str, Any], bool]] = [(a, b, False) for a, b in find_gaps(words)[:MAX_GAPS * 4]]
    seen = {(float(a.get("start") or 0.0), float(b.get("start") or 0.0)) for a, b, _ in candidates}
    for start, end in uncovered_stretches(second_pass or (), words):
        pair = bracketing_pair(words, start, end)
        key = (float(pair[0].get("start") or 0.0), float(pair[1].get("start") or 0.0)) if pair else None
        if pair is None or key in seen:
            continue
        seen.add(key)
        candidates.append((pair[0], pair[1], True))              # a second-pass word here is evidence of speech: no loudness test needed
    for before, after, heard_by_second_pass in candidates:
        if len(found) >= MAX_GAPS:
            break
        gap_start, gap_end = _end(before), float(after.get("start") or 0.0)
        if not heard_by_second_pass and not has_speech(levels, frame, gap_start, gap_end):
            continue
        window_start = max(0.0, gap_start - 1.0)
        heard = transcribe(audio, window_start, gap_end + 1.0)
        inside = [w for w in heard if gap_start - 0.05 <= window_start + float(w.get("start") or 0.0) + 0.1 <= gap_end + 0.05]
        text = " ".join(str(w.get("text")) for w in inside).strip()
        if len(inside) < MIN_WORDS:
            continue
        segment = next((s for s in segments if int(s.get("start_ms") or 0) / 1000 - 0.5 <= float(before.get("start") or 0.0) <= int(s.get("end_ms") or 0) / 1000 + 0.5), None)
        if segment is None:
            continue
        change = insertion(str(segment.get("restored_text") or segment.get("source_text") or ""), str(before.get("text")), text)
        if change is None:
            continue
        spoken_until = max(window_start + float(w.get("end") or w.get("start") or 0.0) for w in inside)
        found.append({"segment_id": segment["segment_id"], **change, "reason": f"{gap_end - gap_start:.1f} s of speech the transcript missed, heard again: {text!r}",
                      "sentence": "", "audio_phones": None, "kind": "missing speech",
                      # the words are spoken in the gap, so the segment's window has to reach them: the dub then says them there
                      "extends_to_ms": int(min(spoken_until + END_TAIL_SECONDS, gap_end - NEXT_WORD_MARGIN_SECONDS) * 1000)})
        progress(f"[source] speech missed by the transcript at {gap_start:.0f}-{gap_end:.0f} s: {text!r}")
    return found
