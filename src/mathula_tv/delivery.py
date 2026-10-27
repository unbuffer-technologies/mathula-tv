"""How each speaker delivers: pace, pauses, pitch, loudness and the words they lean on, measured from the source audio.

The dub should sound like the person people know, not like a neutral newsreader.  The isiZulu voices take no style, so what can
be carried over is the *shape* of the speech -- where the speaker speeds up, stops for effect, repeats, stresses a word -- which
the agent reproduces in its wording and in the pauses it holds.  Pace and pitch are never bent: Azure sounds unnatural when they are.

Everything here is relative to the speaker's own habits (their median, their usual swing), never to an absolute scale, and nothing
is scored: the sheet reports measurements, the agent and the renderer use them.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import wave
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

SCHEMA_VERSION = "mathula-delivery-sheet-v1"
SHEET_NAME = "delivery_sheet.json"
HOP_S = 0.01
WINDOW_S = 0.04
F0_MIN_HZ = 70.0
F0_MAX_HZ = 400.0
MIN_PAUSE_S = 0.3                    # a stop between words that a listener hears as a pause
ARTICULATION_GAP_S = 0.25            # shorter gaps are part of speaking; longer ones are pauses and do not count toward pace
STRESSED_WORDS = 3
_SMALL_WORDS = {"the", "and", "that", "this", "with", "from", "have", "will", "your", "they", "them", "what", "were", "been", "for",
                "are", "you", "not", "but", "its", "our", "was", "has", "had", "can", "all", "who", "how", "why"}


# -- pause marker ------------------------------------------------------------------------------------------------------------
# The voice leaves ~0.55 s after a sentence; a speaker who holds a pause for effect (this one up to 2 s) cannot be matched by wording.
# The agent writes ``~pause800`` at the spot; it is never spoken, becomes a silence of that many milliseconds, and counts toward the
# unit's length like any other part of the speech.
PAUSE_MARKER = re.compile(r"\s*~pause(\d{3,4})\b\s*", re.IGNORECASE)
MIN_MARKED_PAUSE_MS = 200
MAX_MARKED_PAUSE_MS = 2000
MAX_MARKED_PAUSE_TOTAL_MS = 4500              # the SSML builder refuses more silence than this in one document


def split_pauses(text: str) -> list[str | int]:
    """The text as speech pieces with the marked pauses (milliseconds) between them; each pause clamped, the total capped."""
    pieces: list[str | int] = []
    position, total = 0, 0
    for match in PAUSE_MARKER.finditer(text):
        before = text[position:match.start()]
        if before.strip():
            pieces.append(before)
        pause = max(MIN_MARKED_PAUSE_MS, min(MAX_MARKED_PAUSE_MS, int(match.group(1))))
        pause = min(pause, MAX_MARKED_PAUSE_TOTAL_MS - total)
        if pause >= MIN_MARKED_PAUSE_MS and pieces and not isinstance(pieces[-1], int):       # none at the very start, none doubled
            pieces.append(pause)
            total += pause
        position = match.end()
    rest = text[position:]
    if rest.strip() or not pieces:
        pieces.append(rest)
    while pieces and isinstance(pieces[-1], int):                                              # a pause at the very end would be trimmed anyway
        pieces.pop()
    return pieces


def strip_pauses(text: str) -> str:
    """The text without its pause markers (what a reader or the meaning check should see)."""
    return PAUSE_MARKER.sub(" ", text).strip()


# -- signal ------------------------------------------------------------------------------------------------------------------

def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as reader:
        rate, width, channels = reader.getframerate(), reader.getsampwidth(), reader.getnchannels()
        if width != 2:
            raise ValueError("delivery analysis needs 16-bit audio")
        data = np.frombuffer(reader.readframes(reader.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    return (data.reshape(-1, channels).mean(axis=1) if channels > 1 else data), rate


def track(samples: np.ndarray, rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Per 10 ms frame: pitch in semitones above 100 Hz (NaN where unvoiced) and loudness in dB."""
    hop, window = int(round(rate * HOP_S)), int(round(rate * WINDOW_S))
    if len(samples) < window + hop:
        return np.zeros(0), np.zeros(0)
    count = 1 + (len(samples) - window) // hop
    lo, hi = int(rate / F0_MAX_HZ), int(rate / F0_MIN_HZ)
    taper = np.hanning(window).astype(np.float32)
    size = 1 << int(math.ceil(math.log2(2 * window)))
    shape = np.fft.irfft(np.abs(np.fft.rfft(taper, size)) ** 2)[: hi + 3]
    shape = shape / shape[0]                                      # the window's own autocorrelation: divides out its roll-off
    semitones = np.full(count, np.nan, dtype=np.float64)
    loudness = np.empty(count, dtype=np.float64)
    for first in range(0, count, 2048):
        last = min(count, first + 2048)
        frames = samples[(np.arange(first, last)[:, None] * hop) + np.arange(window)[None, :]]
        loudness[first:last] = 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-9)
        spectrum = np.fft.rfft((frames - frames.mean(axis=1, keepdims=True)) * taper, size, axis=1)
        auto = np.fft.irfft(spectrum * np.conj(spectrum), axis=1)[:, : hi + 3]
        auto = auto / (auto[:, :1] + 1e-12) / shape[None, :]
        rows = np.arange(auto.shape[0])
        lag = auto[:, lo:hi + 1].argmax(axis=1) + lo
        peak = auto[rows, lag]
        half = lag // 2                                            # a peak at twice the period is as high: take the shorter period
        near = np.stack([auto[rows, np.clip(half + d, lo, hi)] for d in (-1, 0, 1)])
        half_peak, half_shift = near.max(axis=0), near.argmax(axis=0) - 1
        use_half = (half >= lo) & (half_peak >= 0.85 * peak)
        lag = np.where(use_half, np.clip(half + half_shift, lo, hi), lag)
        peak = np.where(use_half, half_peak, peak)
        before, here, after = auto[rows, lag - 1], auto[rows, lag], auto[rows, np.minimum(lag + 1, auto.shape[1] - 1)]
        denominator = before - 2 * here + after
        usable = np.abs(denominator) > 1e-9
        shift = np.where(usable, 0.5 * (before - after) / np.where(usable, denominator, 1.0), 0.0)
        hz = rate / (lag + np.clip(shift, -1, 1))
        voiced = peak >= 0.5
        semitones[first:last] = np.where(voiced, 12 * np.log2(hz / 100.0), np.nan)
    floor = float(np.percentile(loudness, 10))
    semitones[loudness < floor + 8] = np.nan                       # near the noise floor a pitch is not trustworthy
    return _drop_octave_jumps(semitones), loudness


def _drop_octave_jumps(semitones: np.ndarray, radius: int = 5, limit: float = 4.0) -> np.ndarray:
    padded = np.pad(semitones, radius, constant_values=np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * radius + 1)
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            nearby = np.nanmedian(windows, axis=1)
    result = semitones.copy()
    result[np.abs(semitones - nearby) > limit] = np.nan
    return result


# -- measurements ------------------------------------------------------------------------------------------------------------

def _median(values: Sequence[float]) -> float | None:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.median(clean)) if clean else None


def _quantile(values: Sequence[float], q: float) -> float | None:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.percentile(clean, q)) if clean else None


def _frames(start_s: float, end_s: float, count: int) -> slice:
    return slice(max(0, int(start_s / HOP_S)), min(count, max(int(end_s / HOP_S), int(start_s / HOP_S) + 1)))


def _round(value: float | None, digits: int = 1) -> float | None:
    return None if value is None or not math.isfinite(value) else round(float(value), digits)


def _segment_words(segment: Mapping[str, Any], words: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    start, end = int(segment["start_ms"]) / 1000.0, int(segment["end_ms"]) / 1000.0
    return [w for w in words if start - 0.05 <= (float(w["start"]) + float(w["end"])) / 2 <= end + 0.05]


def _pauses(words: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    found = []
    for before, after in zip(words, words[1:]):
        gap = float(after["start"]) - float(before["end"])
        if gap >= MIN_PAUSE_S:
            found.append({"ms": int(round(gap * 1000)), "after": str(before.get("text", "")).strip(".,!?"),
                          "before": str(after.get("text", "")).strip(".,!?")})
    return found


def _pace(words: Sequence[Mapping[str, Any]]) -> float | None:
    """Words per second of speaking, with every stop longer than ``ARTICULATION_GAP_S`` taken out."""
    if len(words) < 3:
        return None
    spoken = sum(float(w["end"]) - float(w["start"]) for w in words)
    spoken += sum(min(max(0.0, float(b["start"]) - float(a["end"])), ARTICULATION_GAP_S) for a, b in zip(words, words[1:]))
    return len(words) / spoken if spoken > 0 else None


def repeated_phrases(text: str, limit: int = 3) -> list[str]:
    """Phrases the speaker says more than once in one breath: the repetition a politician builds a point with."""
    tokens = re.findall(r"[A-Za-z']+", text.lower())
    found: list[str] = []
    for size in (4, 3, 2):
        seen: dict[tuple[str, ...], int] = {}
        for index in range(len(tokens) - size + 1):
            gram = tuple(tokens[index:index + size])
            if any(len(t) >= 4 and t not in _SMALL_WORDS for t in gram):
                seen[gram] = seen.get(gram, 0) + 1
        for gram, times in seen.items():
            phrase = " ".join(gram)
            if times >= 2 and not any(phrase in longer for longer in found):
                found.append(phrase)
    return found[:limit]


def build_sheet(audio_path: Path, segments: Sequence[Mapping[str, Any]], words: Sequence[Mapping[str, Any]], *, pass1_sha256: str = "") -> dict[str, Any]:
    """Measure every segment against its speaker's own habits.  ``words`` carry ``text``, ``start`` and ``end`` in seconds."""
    samples, rate = read_wav(audio_path)
    semitones, loudness = track(samples, rate)
    count = len(semitones)
    ordered = sorted(segments, key=lambda s: int(s["start_ms"]))
    raw: list[dict[str, Any]] = []
    for index, segment in enumerate(ordered):
        start, end = int(segment["start_ms"]) / 1000.0, int(segment["end_ms"]) / 1000.0
        inside = _segment_words(segment, words)
        frame = _frames(start, end, count)
        voiced = semitones[frame][~np.isnan(semitones[frame])]
        gate = np.percentile(loudness, 10) + 8
        loud = loudness[frame][loudness[frame] > gate]
        word_rows = []
        for word in inside:
            span = _frames(float(word["start"]), float(word["end"]), count)
            pitch = semitones[span][~np.isnan(semitones[span])]
            level = loudness[span]
            word_rows.append({"text": str(word.get("text", "")).strip(".,!?\"'"), "st": float(np.median(pitch)) if len(pitch) >= 3 else None,
                              "db": float(np.percentile(level, 90)) if len(level) else None})
        tail = semitones[_frames(max(start, end - 1.0), end, count)]
        tail = tail[~np.isnan(tail)]
        ends_by = float(np.median(tail[-max(3, len(tail) // 3):]) - np.median(tail[:max(3, len(tail) // 3)])) if len(tail) >= 10 else None
        raw.append({
            "segment_id": str(segment["segment_id"]), "speaker_id": str(segment.get("speaker_id") or ""), "start_ms": int(segment["start_ms"]),
            "end_ms": int(segment["end_ms"]), "words": len(inside), "pace_wps": _pace(inside), "pauses": _pauses(inside),
            "pause_before_ms": max(0, int(segment["start_ms"]) - int(ordered[index - 1]["end_ms"])) if index else None,
            "st": float(np.median(voiced)) if len(voiced) >= 10 else None,
            "spread_st": float(np.percentile(voiced, 90) - np.percentile(voiced, 10)) if len(voiced) >= 10 else None,
            "db": float(np.percentile(loud, 75)) if len(loud) >= 10 else None, "ends_by_st": ends_by, "word_rows": word_rows,
            "repeats": repeated_phrases(str(segment.get("restored_text") or segment.get("source_text") or "")),
        })

    speakers: dict[str, dict[str, Any]] = {}
    for speaker in sorted({r["speaker_id"] for r in raw if r["speaker_id"]}):
        mine = [r for r in raw if r["speaker_id"] == speaker]
        pooled_st = np.concatenate([semitones[_frames(r["start_ms"] / 1000, r["end_ms"] / 1000, count)] for r in mine]) if mine else np.zeros(0)
        pooled_st = pooled_st[~np.isnan(pooled_st)]
        word_rows = [w for r in mine for w in r["word_rows"]]
        db_values = [w["db"] for w in word_rows if w["db"] is not None]
        st_values = [w["st"] for w in word_rows if w["st"] is not None]
        speaking_s = sum(r["end_ms"] - r["start_ms"] for r in mine) / 1000.0
        pause_ms = [p["ms"] for r in mine for p in r["pauses"]]
        enders = [r["ends_by_st"] for r in mine if r["ends_by_st"] is not None]
        speakers[speaker] = {
            "segments": len(mine), "speaking_seconds": _round(speaking_s),
            "pace_wps": {"usual": _round(_median([r["pace_wps"] for r in mine]), 2), "quick": _round(_quantile([r["pace_wps"] for r in mine], 90), 2),
                         "slow": _round(_quantile([r["pace_wps"] for r in mine], 10), 2)},
            "pitch_hz_usual": _round(100 * 2 ** (float(np.median(pooled_st)) / 12), 0) if len(pooled_st) else None,
            "pitch_swing_st": _round(float(np.percentile(pooled_st, 90) - np.percentile(pooled_st, 10))) if len(pooled_st) else None,
            "loudness_swing_db": _round((_quantile([r["db"] for r in mine], 90) or 0) - (_quantile([r["db"] for r in mine], 10) or 0)),
            "pauses_per_minute": _round(len(pause_ms) / (speaking_s / 60.0)) if speaking_s else None,
            "pause_ms_usual": int(np.median(pause_ms)) if pause_ms else None, "pause_ms_longest": max(pause_ms) if pause_ms else None,
            "segments_ending_lower_pct": int(round(100 * sum(1 for e in enders if e < -1.0) / len(enders))) if enders else None,
            "segments_with_repetition": sum(1 for r in mine if r["repeats"]),
            "_base": {"st": float(np.median(pooled_st)) if len(pooled_st) else None, "pace": _median([r["pace_wps"] for r in mine]),
                      "db_seg": _median([r["db"] for r in mine]), "spread": _median([r["spread_st"] for r in mine]),
                      "word_db": _median(db_values), "word_db_sd": float(np.std(db_values)) if len(db_values) > 5 else None,
                      "word_st": _median(st_values), "word_st_sd": float(np.std(st_values)) if len(st_values) > 5 else None},
        }

    rows: dict[str, dict[str, Any]] = {}
    for r in raw:
        base = speakers.get(r["speaker_id"], {}).get("_base", {})
        stressed = []
        for w in r["word_rows"]:
            if len(w["text"]) < 4 or w["text"].lower() in _SMALL_WORDS or w["db"] is None or not base.get("word_db_sd"):
                continue
            z = (w["db"] - base["word_db"]) / base["word_db_sd"]
            if w["st"] is not None and base.get("word_st_sd"):
                z += (w["st"] - base["word_st"]) / base["word_st_sd"]
            stressed.append((z, w["text"]))
        stressed.sort(reverse=True)
        rows[r["segment_id"]] = {
            "speaker_id": r["speaker_id"], "start_ms": r["start_ms"], "end_ms": r["end_ms"], "words": r["words"],
            "pace_wps": _round(r["pace_wps"], 2),
            "pace_vs_usual_pct": _round(100 * (r["pace_wps"] / base["pace"] - 1), 0) if r["pace_wps"] and base.get("pace") else None,
            "pitch_vs_usual_st": _round(r["st"] - base["st"]) if r["st"] is not None and base.get("st") is not None else None,
            "spread_vs_usual_st": _round(r["spread_st"] - base["spread"]) if r["spread_st"] is not None and base.get("spread") is not None else None,
            "loudness_vs_usual_db": _round(r["db"] - base["db_seg"]) if r["db"] is not None and base.get("db_seg") is not None else None,
            "ends_by_st": _round(r["ends_by_st"]), "pause_before_ms": r["pause_before_ms"], "pauses": sorted(r["pauses"], key=lambda p: -p["ms"])[:4],
            "stressed_words": [word for _, word in stressed[:STRESSED_WORDS]], "repeats": r["repeats"],
        }
    for entry in speakers.values():
        entry.pop("_base", None)
    return {"schema_version": SCHEMA_VERSION, "pass1_sha256": pass1_sha256, "frame_seconds": HOP_S, "speakers": speakers, "segments": rows}


# -- files -------------------------------------------------------------------------------------------------------------------

def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure_sheet(job_root: Path, *, progress: Any = print) -> Path | None:
    """``native_dub/delivery_sheet.json`` for a job, built from its analysis audio and word timings (cached by the transcript's checksum)."""
    job_root = Path(job_root)
    native, audio = job_root / "native_dub", job_root / "audio" / "analysis_mono.wav"
    pass1_path, words_path = native / "pass1_restored_transcript.json", job_root / "analysis" / "azure_diarization.json"
    if not (audio.is_file() and pass1_path.is_file() and words_path.is_file()):
        progress("[delivery] no audio or word timings for this job; delivery sheet skipped")
        return None
    sha = _file_sha(pass1_path)
    target = native / SHEET_NAME
    if target.is_file():
        try:
            if json.loads(target.read_text(encoding="utf-8")).get("pass1_sha256") == sha:
                return target
        except ValueError:
            pass
    pass1 = json.loads(pass1_path.read_text(encoding="utf-8"))
    words = json.loads(words_path.read_text(encoding="utf-8")).get("words") or []
    sheet = build_sheet(audio, pass1.get("segments") or [], words, pass1_sha256=sha)
    target.write_text(json.dumps(sheet, ensure_ascii=False, indent=1), encoding="utf-8")
    for speaker, entry in sheet["speakers"].items():
        progress(f"[delivery] {speaker}: {entry['pace_wps']['usual']} words/s (quick {entry['pace_wps']['quick']}, slow {entry['pace_wps']['slow']}), "
                 f"pitch ~{entry['pitch_hz_usual']} Hz, {entry['pauses_per_minute']} pauses/min")
    return target


def load_sheet(path: Path) -> dict[str, Any] | None:
    try:
        sheet = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return sheet if sheet.get("schema_version") == SCHEMA_VERSION else None


# -- for the agent -----------------------------------------------------------------------------------------------------------

AGENT_NOTE = (
    "How each speaker actually delivers, measured from the audio against their OWN usual (not an absolute scale). People who know "
    "these speakers should recognise them in the dub, so carry their delivery into your isiZulu: keep a speaker's rhythm (a quick "
    "run of short phrases stays quick and short; a slow, weighted line stays weighted), keep a pause where they pause for effect "
    "(end the sentence there, so the voice stops there), keep the phrases they repeat repeated, and put the words they lean on "
    "('stressed_words') where isiZulu word order lets them land at the end of a phrase or right after a pause. 'ends_by_st' < 0 "
    "means the sentence falls at the end (a statement), > 0 that it lifts (a question or a build-up). Never stretch a unit past its "
    "window for delivery's sake, and never invent a pause the speaker did not take."
)


def for_agent(sheet: Mapping[str, Any], segment_ids: Sequence[str] | None = None) -> dict[str, Any]:
    wanted = set(segment_ids) if segment_ids is not None else None
    rows = []
    for segment_id, row in (sheet.get("segments") or {}).items():
        if wanted is not None and segment_id not in wanted:
            continue
        compact: dict[str, Any] = {"segment_id": segment_id}
        for key in ("pace_vs_usual_pct", "pitch_vs_usual_st", "ends_by_st"):
            if row.get(key) is not None:
                compact[key] = row[key]
        if abs(row.get("loudness_vs_usual_db") or 0) >= 1.5:          # broadcast audio is compressed: small differences are not delivery
            compact["loudness_vs_usual_db"] = row["loudness_vs_usual_db"]
        pauses = [f"{p['ms']} ms after '{p['after']}'" for p in (row.get("pauses") or [])[:3]]
        if pauses:
            compact["pauses_inside"] = pauses
        for key in ("stressed_words", "repeats"):
            if row.get(key):
                compact[key] = row[key]
        rows.append(compact)
    return {"note": AGENT_NOTE, "speakers": sheet.get("speakers") or {}, "segments": rows}
