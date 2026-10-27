"""Post-render scan of the dubbed audio for the faults a listener hears as a hiccup.

The pipeline's other checks read the audio as words (missing words, English in the pauses); none of them listens for a break in
the sound itself.  This scan reads the signal and reports, with timestamps, four kinds of fault:

* ``dropout``   -- the sound falls to almost nothing for 60-400 ms *inside* speech (loud on both sides within 150 ms).
* ``dead_gap``  -- an exact-zero stretch of 250 ms or more.  Sentence pauses are 380-460 ms of exact zero by design; they are
                   counted separately (``sentence_pause``) so only the unexpected lengths show up here.
* ``stutter``   -- several short exact-zero blocks in a row inside speech (the voice gating on and off).
* ``click``     -- an isolated burst of high-frequency energy of a few milliseconds.

It reports measurements, nothing is scored, and it never changes the audio.  Words next to each event come from an optional
speech-to-text pass (``--words``) so a listener can find the spot and so patterns (a click consonant, a number) show up.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

SAMPLE_RATE = 48000
FRAME_S = 0.005
SPEECH_DB = -34.0                     # context level (dBFS) that counts as speech on both sides of a dropout
CONTEXT_FRAMES = 30                   # 150 ms either side
DROPOUT_DEPTH_DB = 20.0               # how far under the surrounding speech the sound falls
DROPOUT_MIN_MS, DROPOUT_MAX_MS = 60, 400
DROPOUT_FLOOR_DB = -55.0              # and how quiet it gets: an ordinary stop closure in this voice stays above this
ZERO_LEVEL = 2                        # |sample| at or under this is digital silence
DEAD_GAP_MIN_MS = 250
SENTENCE_PAUSE_MS = (340, 480)        # the voice's own 400 ms boundary (+ trim) -- by design
STUTTER_BLOCK_MS = (15, 60)
STUTTER_MIN_BLOCKS = 4
CLICK_RATIO_DB = 18.0                 # burst in the 8 kHz+ band over its neighbours
CLICK_MIN_DB = -45.0


def read_audio(path: Path, *, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Mono 16-bit samples at ``sample_rate`` from any media file ffmpeg reads."""
    with tempfile.TemporaryDirectory() as directory:
        wav = Path(directory) / "audio.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sample_rate), str(wav)], check=True)
        with wave.open(str(wav)) as handle:
            return np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16).copy()


def frame_db(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    hop = int(sample_rate * FRAME_S)
    count = len(samples) // hop
    frames = samples[: count * hop].astype(np.float64).reshape(count, hop)
    return 20.0 * np.log10((np.sqrt((frames ** 2).mean(axis=1)) + 1.0) / 32768.0)


def _zero_runs(samples: np.ndarray, sample_rate: int, min_ms: float, max_ms: float | None = None) -> list[tuple[float, float]]:
    """(start s, length ms) of each stretch of digital silence."""
    quiet = np.abs(samples) <= ZERO_LEVEL
    edges = np.diff(np.concatenate(([0], quiet.astype(np.int8), [0])))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    runs = []
    for a, b in zip(starts, ends):
        length = (b - a) * 1000.0 / sample_rate
        if length >= min_ms and (max_ms is None or length <= max_ms):
            runs.append((a / sample_rate, length))
    return runs


def find_dropouts(db: np.ndarray) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    i = CONTEXT_FRAMES
    while i < len(db) - CONTEXT_FRAMES:
        left = float(np.median(db[i - CONTEXT_FRAMES:i - 2]))
        if left < SPEECH_DB or db[i] >= left - DROPOUT_DEPTH_DB:
            i += 1
            continue
        j = i
        while j < len(db) and db[j] < left - DROPOUT_DEPTH_DB * 0.6:
            j += 1
        right = float(np.median(db[j + 2:j + CONTEXT_FRAMES])) if j + 2 < len(db) else -120.0
        duration_ms = (j - i) * FRAME_S * 1000.0
        floor = float(db[i:j].min())
        if right >= SPEECH_DB and DROPOUT_MIN_MS <= duration_ms <= DROPOUT_MAX_MS and floor <= DROPOUT_FLOOR_DB:
            events.append({"type": "dropout", "start_s": round(i * FRAME_S, 3), "ms": round(duration_ms), "floor_db": round(floor),
                           "around_db": round(min(left, right))})
        i = max(j, i + 1)
    return events


def find_dead_gaps(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> tuple[list[dict[str, Any]], int]:
    gaps, sentence_pauses = [], 0
    for start, ms in _zero_runs(samples, sample_rate, DEAD_GAP_MIN_MS):
        if SENTENCE_PAUSE_MS[0] <= ms <= SENTENCE_PAUSE_MS[1]:
            sentence_pauses += 1
            continue
        gaps.append({"type": "dead_gap", "start_s": round(start, 3), "ms": round(ms)})
    return gaps, sentence_pauses


def find_stutters(samples: np.ndarray, db: np.ndarray, sample_rate: int = SAMPLE_RATE) -> list[dict[str, Any]]:
    blocks = _zero_runs(samples, sample_rate, STUTTER_BLOCK_MS[0], STUTTER_BLOCK_MS[1])
    events, cluster = [], []

    def flush() -> None:
        if len(cluster) >= STUTTER_MIN_BLOCKS:
            frame = int(cluster[0][0] / FRAME_S)
            around = float(np.median(db[max(0, frame - 20):frame + 40]))
            if around > SPEECH_DB:                                   # inside speech, not the tail of a word
                events.append({"type": "stutter", "start_s": round(cluster[0][0], 3), "blocks": len(cluster),
                               "ms": round((cluster[-1][0] - cluster[0][0]) * 1000 + cluster[-1][1])})
    for block in blocks:
        if cluster and block[0] - cluster[-1][0] > 0.12:
            flush()
            cluster = []
        cluster.append(block)
    flush()
    return events


def find_clicks(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> list[dict[str, Any]]:
    """A burst in the band above 8 kHz that stands well over the frames around it."""
    hop = int(sample_rate * 0.002)
    count = len(samples) // hop
    x = samples[: count * hop].astype(np.float64)
    spectrum = np.abs(np.fft.rfft(x.reshape(count, hop), axis=1)) ** 2
    freqs = np.fft.rfftfreq(hop, 1.0 / sample_rate)
    band = 10.0 * np.log10(spectrum[:, freqs >= 8000].sum(axis=1) / hop + 1e-3) - 90.3
    events, i = [], 12
    while i < count - 12:
        around = max(float(np.median(band[i - 12:i - 3])), float(np.median(band[i + 4:i + 13])))      # over BOTH sides: a word onset is not a click
        if band[i] > CLICK_MIN_DB and band[i] - around > CLICK_RATIO_DB and band[i] == band[i - 2:i + 3].max():
            events.append({"type": "click", "start_s": round(i * 0.002, 3), "over_db": round(float(band[i] - around))})
            i += 25
        else:
            i += 1
    return events


def scan(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> dict[str, Any]:
    db = frame_db(samples, sample_rate)
    dead_gaps, sentence_pauses = find_dead_gaps(samples, sample_rate)
    events = find_dropouts(db) + dead_gaps + find_stutters(samples, db, sample_rate) + find_clicks(samples, sample_rate)
    events.sort(key=lambda event: event["start_s"])
    return {"duration_s": round(len(samples) / sample_rate, 1), "sentence_pauses": sentence_pauses, "events": events,
            "counts": {kind: sum(1 for event in events if event["type"] == kind) for kind in ("dropout", "dead_gap", "stutter", "click")}}


def locate(events: Sequence[dict[str, Any]], groups: Sequence[Mapping[str, Any]], cut_s: float) -> None:
    """Add the unit and the second inside it to each event (final-video time + cut = master time)."""
    for event in events:
        master_ms = (event["start_s"] + cut_s) * 1000.0
        for group in groups:
            start = float(group["start_ms"])
            if start <= master_ms <= start + float(group.get("tts_duration_ms") or 0):
                event["unit"], event["unit_s"] = str(group["group_id"]), round((master_ms - start) / 1000.0, 2)
                break


def add_words(events: Sequence[dict[str, Any]], media: Path, *, window_s: float = 1.6, limit: int = 60) -> None:
    """The words the speech-to-text hears around each event (zu-ZA): the spot in the script, and a hint of its cause."""
    from .azure_stt import normalize
    from .cli import create_speech_backend
    from .config import load_settings

    backend = create_speech_backend(load_settings())
    with tempfile.TemporaryDirectory() as directory:
        for event in list(events)[:limit]:
            begin = max(0.0, event["start_s"] - window_s)
            wav = Path(directory) / f"w{int(begin * 1000)}.wav"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{begin:.3f}", "-t", f"{2 * window_s:.3f}", "-i", str(media), "-vn",
                            "-ac", "1", "-ar", "16000", str(wav)], check=True)
            try:
                words = normalize(backend.transcribe(wav, locale="zu-ZA")).get("words") or []
            except Exception:                                          # an empty transcription is not a failure of the scan
                words = []
            event["heard"] = " ".join(str(word.get("text")) for word in words)


def format_time(seconds: float) -> str:
    return f"{int(seconds // 60)}:{seconds % 60:05.2f}"


def report(result: Mapping[str, Any]) -> str:
    lines = [f"{result['duration_s']} s scanned; {result['sentence_pauses']} sentence pauses (by design); "
             + ", ".join(f"{count} {kind}" for kind, count in result["counts"].items())]
    for event in result["events"]:
        detail = {key: value for key, value in event.items() if key not in {"type", "start_s", "heard", "unit", "unit_s"}}
        where = f" [{event['unit']} +{event['unit_s']}s]" if "unit" in event else ""
        lines.append(f"  {format_time(event['start_s'])} {event['type']:<8} {detail}{where} {event.get('heard', '')}")
    times = [event["start_s"] for event in result["events"]]
    if len(times) > 2:
        gaps = np.diff(times)
        lines.append(f"events are a median {float(np.median(gaps)):.0f} s apart (shortest {float(gaps.min()):.1f} s, longest {float(gaps.max()):.0f} s)")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="mathula_tv.audio_glitches", description=__doc__)
    parser.add_argument("media", type=Path)
    parser.add_argument("--timing-plan", type=Path, help="timing_plan.json of the render, to name the unit each event falls in")
    parser.add_argument("--cut", type=float, default=0.0, help="seconds cut from the start of the master (the manifest's applied_cut_start_seconds)")
    parser.add_argument("--words", action="store_true", help="add the words heard around each event (Azure speech-to-text)")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    result = scan(read_audio(args.media))
    if args.timing_plan:
        locate(result["events"], json.loads(args.timing_plan.read_text(encoding="utf-8"))["groups"], args.cut)
    if args.words:
        add_words(result["events"], args.media)
    print(report(result))
    if args.json:
        args.json.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
