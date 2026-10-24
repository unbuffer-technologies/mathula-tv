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


def _end(word: Mapping[str, Any]) -> float:
    return float(word.get("end") or float(word.get("start") or 0.0) + 0.2)


def find_gaps(words: Sequence[Mapping[str, Any]], min_gap: float = MIN_GAP_SECONDS) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    ordered = sorted(words, key=lambda w: float(w.get("start") or 0.0))
    return [(a, b) for a, b in zip(ordered, ordered[1:]) if float(b.get("start") or 0.0) - _end(a) >= min_gap]


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
    tail = token[len(kept):]
    ending = "." if tail.strip() in ("", "—", "–", "-") and token.endswith(("—", "–", "-")) else tail
    return {"token": token, "replacement": f"{kept} {inserted.strip()}{ending}"}


def find_missing_speech(
    segments: Sequence[Mapping[str, Any]], words: Sequence[Mapping[str, Any]], audio: Path, transcribe: Callable[[Path, float, float], Sequence[Mapping[str, Any]]],
    *, progress: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    """Corrections that add words the transcript missed. ``transcribe(audio, start, end)`` returns word dicts with times relative to ``start``."""
    levels, frame = frame_levels(audio)
    found: list[dict[str, Any]] = []
    for before, after in find_gaps(words)[:MAX_GAPS * 4]:
        if len(found) >= MAX_GAPS:
            break
        gap_start, gap_end = _end(before), float(after.get("start") or 0.0)
        if not has_speech(levels, frame, gap_start, gap_end):
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
        found.append({"segment_id": segment["segment_id"], **change, "reason": f"{gap_end - gap_start:.1f} s of speech the transcript missed, heard again: {text!r}",
                      "sentence": "", "audio_phones": None, "kind": "missing speech"})
        progress(f"[source] speech missed by the transcript at {gap_start:.0f}-{gap_end:.0f} s: {text!r}")
    return found
