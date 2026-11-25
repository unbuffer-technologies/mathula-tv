"""The source recording's own room tone, to fill the silence between two generations of the voice.

OmniVoice is run span by span and each span is padded with exact digital zero, so between two spans of one unit the audio does not pause, it drops out: a hard black bar in the spectrogram
(prod-119: twelve of them, 0.35-0.43 s each, one at 0:27).  A real recording never goes to zero: there is always a floor of room sound.  This takes that floor from the job's own source audio (its
quietest second, not digital silence) and fills the joins with it at the level the source had it relative to its speech.  Only silence is replaced; no speech is touched.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

RATE = 24000
TONE_S = 1.0                 # the length of the sample of floor that is looped
FRAME_S = 0.05
MIN_DELTA_DB = 30.0          # the floor is at least this far under the speech (a noisier source is not copied hiss for hiss)
MAX_DELTA_DB = 60.0          # and never so far under that it is just zero again
FADE_S = 0.03
STEADY_WINDOWS_S = (1.0, 0.6, 0.4)   # a steady stretch this long is looped; a source that is never steady for a whole second may still have a steady 0.6 s (the flat floor after a breath)
STEADY_DB = 6.0              # the floor must be steady: no frame of it more than this far from the quietest frame of the same second (a breath, a click or a word's tail is not a floor)
VERSION = 3                  # 1 took the quietest second whatever it held: on the Mbalula clip it began with 0.15 s of breath-like hiss that every join then repeated (a breath after every sentence)


def _resample(mono: np.ndarray, rate: int) -> np.ndarray:
    if rate == RATE:
        return mono.astype(np.float32)
    if rate % RATE == 0:
        factor = rate // RATE
        return mono[: len(mono) // factor * factor].reshape(-1, factor).mean(axis=1).astype(np.float32)
    return np.interp(np.linspace(0, len(mono) - 1, int(len(mono) * RATE / rate)), np.arange(len(mono)), mono).astype(np.float32)


def find_room_tone(audio_path: Path | str) -> tuple[np.ndarray, float]:
    """(a second of the source's quietest non-silent stretch at 24 kHz with its mean removed, how many dB it is below the source's speech).  A source with no floor gives a silent tone."""
    info = sf.info(str(audio_path))
    audio, rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
    mono = _resample(audio.mean(axis=1), int(info.samplerate))
    frame = int(FRAME_S * RATE)
    count = len(mono) // frame
    if count < int(TONE_S / FRAME_S) + 2:
        return np.zeros(int(TONE_S * RATE), dtype=np.float32), MAX_DELTA_DB
    db = 20.0 * np.log10(np.sqrt((mono[: count * frame].reshape(count, frame) ** 2).mean(axis=1)) + 1e-9)
    speech = float(np.median(db[db > np.percentile(db, 40)]))
    live = db > -100.0                                           # not digital zero
    best, window = None, int(TONE_S / FRAME_S)
    for seconds in STEADY_WINDOWS_S:                             # the longest steady stretch there is, then shorter ones
        size = int(round(seconds / FRAME_S))
        if count < size + 2:
            continue
        means = np.convolve(db, np.ones(size) / size, mode="valid")
        ok = np.convolve(live.astype(float), np.ones(size), mode="valid") >= size
        spread = np.array([db[i: i + size].max() - db[i: i + size].min() for i in range(len(means))])        # how much the stretch moves: a steady floor moves little
        steady = ok & (spread <= STEADY_DB)
        if steady.any():
            best, window = int(np.argmin(np.where(steady, means, np.inf))), size
            break
    if best is None:                                             # nothing is steady: the quietest second, as before
        window = int(TONE_S / FRAME_S)
        means = np.convolve(db, np.ones(window) / window, mode="valid")
        ok = np.convolve(live.astype(float), np.ones(window), mode="valid") >= window
        if not ok.any():
            return np.zeros(int(TONE_S * RATE), dtype=np.float32), MAX_DELTA_DB
        best = int(np.argmin(np.where(ok, means, np.inf)))
    means = np.convolve(db, np.ones(window) / window, mode="valid")
    tone = mono[best * frame: best * frame + window * frame].copy()
    tone -= float(tone.mean())
    delta = float(np.clip(speech - float(means[best]), MIN_DELTA_DB, MAX_DELTA_DB))
    return tone, delta


def tile(tone: np.ndarray, length: int) -> np.ndarray:
    """The tone looped to ``length`` samples, each seam crossfaded so the loop has no click."""
    if length <= 0:
        return np.zeros(0, dtype=np.float32)
    if len(tone) == 0 or not np.any(tone):
        return np.zeros(length, dtype=np.float32)
    fade = min(len(tone) // 4, int(0.1 * RATE))
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    out = tone.copy()
    while len(out) < length:
        out = np.concatenate([out[:-fade], out[-fade:] * (1.0 - ramp) + tone[:fade] * ramp, tone[fade:]])
    return out[:length]


def fill(length_samples: int, tone: np.ndarray, delta_db: float, speech_db: float) -> np.ndarray:
    """``length_samples`` of room tone, level ``delta_db`` under ``speech_db`` (the dub's own speech level), with short fades at both ends so it does not click on and off."""
    body = tile(tone, length_samples)
    rms = float(np.sqrt((body ** 2).mean())) if len(body) else 0.0
    if rms <= 0.0:
        return body
    target = 10.0 ** ((speech_db - delta_db) / 20.0)
    body = body * (target / rms)
    fade = min(len(body) // 2, int(FADE_S * RATE))
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        body[:fade] *= ramp
        body[-fade:] *= ramp[::-1]
    return body.astype(np.float32)


def describe(tone: np.ndarray, delta_db: float) -> dict[str, Any]:
    return {"tone_rms_db": round(20.0 * float(np.log10(np.sqrt((tone ** 2).mean()) + 1e-9)), 1), "below_speech_db": round(delta_db, 1)}
