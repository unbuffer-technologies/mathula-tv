"""What a sound is made of, measured from the audio, so the agent can see why two spellings differ when the phone recogniser calls them alike.

The phone recogniser gives each sound a label ("b").  Two different sounds can share a label: Zulu's plain b is an implosive (a soft
release, voiced through the closure), while an English b is a plosive (a clear release burst after a short closure); the recogniser says
"b" for both, and so does the dictionary.  This module measures the properties that tell them apart, on the speaker's own audio and on the
voice's audio, and reports the differences in plain words.  It reports; it does not choose a spelling and it scores nothing.

Measured per stop (p b t d k g and affricates): the release (how sharply the sound rises when the lips or tongue let go), the closure
before it, whether the vocal folds vibrate during that closure, the delay until voicing starts after the release (aspiration), and the
hiss (energy above 4 kHz) just after the release.  Measured per vowel: the first formant (F1) at its start and at its end, which is how open
the mouth is (open "a" about 700-1000 Hz, closed "i" about 300-400 Hz): "science" has an open "ai", and an "ei" spelling starts nearly shut.
For every sound: duration.
"""
from __future__ import annotations

import difflib
import subprocess
import tempfile
import wave
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

SAMPLE_RATE = 16_000
HOP = 32                                   # 2 ms
WINDOW = 64                                # 4 ms energy window
VOICING_WINDOW = 480                       # 30 ms, enough for two periods of a low male voice
F0_LAGS = (SAMPLE_RATE // 300, SAMPLE_RATE // 70)
STOPS = {"p", "b", "t", "d", "k", "ɡ", "g", "c", "ɟ", "q", "ʔ", "tʃ", "dʒ", "ts", "dz", "pʰ", "tʰ", "kʰ", "bʱ", "ɓ", "ɗ", "ʄ", "ɠ", "p'", "t'", "k'"}
RELEASE_DIFFERENCE_DB = 6.0                # a release this much sharper or softer is audible
CLOSURE_DIFFERENCE_MS = 40.0
VOICING_DIFFERENCE = 0.4
VOT_DIFFERENCE_MS = 20.0
HISS_DIFFERENCE = 0.25
FINAL_HISS_WINDOW_S = 0.30                # the last hiss counts as the word's last consonant only if it is in the word's last 300 ms
FINAL_VOWEL_BELOW_PEAK_DB = 15.0          # a final vowel is voiced, hiss-free and within this many dB of the word's loudest part
FINAL_VOWEL_MIN_MS = 20
VOWEL_F1_DIFFERENCE_HZ = 150.0            # vocal tracts differ by about 15%: a smaller difference in mouth opening is not the spelling's doing
VOWEL_LETTERS = set("aeiouɑɒɔəɛɜɪʊʌæɐøɨ")
FORMANT_ORDER = 12


def load(path: Path) -> np.ndarray:
    """Mono float samples at 16 kHz from any audio file."""
    with tempfile.TemporaryDirectory() as directory:
        wav = Path(directory) / "a.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(path), "-ac", "1", "-ar", str(SAMPLE_RATE), str(wav)], check=True)
        with wave.open(str(wav)) as handle:
            return np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16).astype(np.float64) / 32768.0


def _energy_db(samples: np.ndarray) -> np.ndarray:
    count = max(0, (len(samples) - WINDOW) // HOP)
    if not count:
        return np.zeros(0)
    index = np.arange(count)[:, None] * HOP + np.arange(WINDOW)[None, :]
    return 10.0 * np.log10((samples[index] ** 2).mean(axis=1) + 1e-10)


def _periodicity(samples: np.ndarray, center: int) -> float:
    """How periodic the sound is around ``center`` (0 = noise, 1 = a clean voice); a vibrating larynx scores high even when it is quiet."""
    a, b = max(0, center - VOICING_WINDOW // 2), min(len(samples), center + VOICING_WINDOW // 2)
    frame = samples[a:b] - samples[a:b].mean()
    if len(frame) < F0_LAGS[1] + 20 or float(np.dot(frame, frame)) < 1e-9:
        return 0.0
    full = np.correlate(frame, frame, mode="full")[len(frame) - 1:]
    return float(max(0.0, full[F0_LAGS[0]:F0_LAGS[1]].max() / full[0]))


def _hiss(samples: np.ndarray, start: int, length: int) -> float:
    chunk = samples[start:start + length]
    if len(chunk) < 64:
        return 0.0
    spectrum = np.abs(np.fft.rfft(chunk * np.hanning(len(chunk)))) ** 2
    freqs = np.fft.rfftfreq(len(chunk), 1.0 / SAMPLE_RATE)
    return float(spectrum[freqs >= 4000].sum() / (spectrum.sum() + 1e-12))


def is_vowel(token: str) -> bool:
    return any(c in VOWEL_LETTERS for c in token)


def first_formant(frame: np.ndarray) -> float | None:
    """F1 of a short frame by linear prediction (autocorrelation method), or None when no clear resonance is found."""
    if len(frame) < 300 or float(np.dot(frame, frame)) < 1e-9:
        return None
    x = np.append(frame[0], frame[1:] - 0.97 * frame[:-1]) * np.hamming(len(frame))
    r = np.correlate(x, x, "full")[len(x) - 1:len(x) + FORMANT_ORDER]
    if len(r) <= FORMANT_ORDER or r[0] <= 0:
        return None
    matrix = np.array([[r[abs(i - j)] for j in range(FORMANT_ORDER)] for i in range(FORMANT_ORDER)]) + np.eye(FORMANT_ORDER) * 1e-6 * r[0]
    try:
        coefficients = np.linalg.solve(matrix, r[1:FORMANT_ORDER + 1])
        roots = np.roots(np.concatenate(([1.0], -coefficients)))
    except np.linalg.LinAlgError:
        return None
    roots = roots[np.imag(roots) > 0.01]
    frequencies = np.angle(roots) * SAMPLE_RATE / (2 * np.pi)
    widths = -np.log(np.abs(roots)) * SAMPLE_RATE / np.pi
    clear = sorted(float(f) for f, w in zip(frequencies, widths) if 200 < f < 1300 and w < 350)
    return clear[0] if clear else None


def vowel_features(samples: np.ndarray, start_s: float, end_s: float) -> dict[str, float]:
    """How open the mouth is at the start and at the end of a vowel (F1 in Hz); empty when no clear resonance was found there.

    The span runs to the next sound's first frame, in 25 ms frames every 20 ms: the start is the median of the first frames, the end the median
    of the last ones (a vowel that does not move has the same value at both ends, a diphthong like "ai" falls from open to closed)."""
    end_s = min(end_s, start_s + 0.30)
    if end_s - start_s < 0.05:
        return {}
    track: list[tuple[float, float]] = []
    t = start_s
    while t + 0.025 <= end_s + 1e-9:
        a = int(t * SAMPLE_RATE)
        value = first_formant(samples[a:a + int(0.025 * SAMPLE_RATE)])
        if value is not None:
            track.append((t - start_s, value))
        t += 0.02
    first = [v for offset, v in track if offset <= 0.05]
    last = [v for offset, v in track if offset >= (end_s - start_s) - 0.08]
    if not first:
        return {}
    features = {"onset_f1": round(float(np.median(first)))}
    if last:
        features["offset_f1"] = round(float(np.median(last)))
    return features


def phone_features(samples: np.ndarray, token: str, start_s: float, end_s: float) -> dict[str, Any]:
    """The measurable properties of one sound, from its start and end in ``samples``."""
    features: dict[str, Any] = {"token": token, "duration_ms": round((end_s - start_s) * 1000)}
    if token not in STOPS and is_vowel(token):
        features.update(vowel_features(samples, start_s, end_s))
    if token not in STOPS:
        return features
    lo, hi = max(0, int((start_s - 0.04) * SAMPLE_RATE)), min(len(samples), int((end_s + 0.06) * SAMPLE_RATE))
    energy = _energy_db(samples[lo:hi])
    if len(energy) < 8:
        return features
    rise = energy[3:] - energy[:-3]                                  # dB gained over 6 ms
    release = int(rise.argmax())
    features["release_rise_db"] = round(float(rise[release]), 1)
    peak = float(energy[release + 3:release + 24].max())            # the level the sound reaches within about 45 ms of the release
    before = release
    while before > 0 and energy[before - 1] < peak - 15:             # frames still quiet before the release: the closure
        before -= 1
    features["closure_ms"] = round((release - before) * HOP * 1000 / SAMPLE_RATE)
    release_sample = lo + release * HOP
    closure_centres = [lo + i * HOP + WINDOW // 2 for i in range(before, release)]
    voiced = [_periodicity(samples, c) > 0.45 for c in closure_centres[::3]] or [False]
    features["voiced_closure"] = round(sum(voiced) / len(voiced), 2)
    onset = None
    for step in range(0, 40):                                        # up to 80 ms after the release
        centre = release_sample + WINDOW // 2 + step * HOP
        if centre >= len(samples):
            break
        if _periodicity(samples, centre) > 0.6:
            onset = step
            break
    features["voicing_delay_ms"] = round((onset if onset is not None else 40) * HOP * 1000 / SAMPLE_RATE)
    features["hiss_after_release"] = round(_hiss(samples, release_sample, 480), 2)
    return features


def timed_features(samples: np.ndarray, timed: Sequence[tuple[str, float, float]]) -> list[dict[str, Any]]:
    out = []
    for index, (token, start, end) in enumerate(timed):
        if token not in STOPS and is_vowel(token):             # the recogniser's frame for a vowel is one 20 ms step: its span runs to the next sound
            end = timed[index + 1][1] if index + 1 < len(timed) else end + 0.12
        out.append(phone_features(samples, token, start, end))
    return out


def _same(a: float, b: float, limit: float) -> bool:
    return abs(a - b) < limit


_PLACE = {"p": "lips", "b": "lips", "ɓ": "lips", "pʰ": "lips", "bʱ": "lips", "p'": "lips", "t": "tongue-tip", "d": "tongue-tip", "ɗ": "tongue-tip",
          "tʰ": "tongue-tip", "t'": "tongue-tip", "k": "tongue-back", "ɡ": "tongue-back", "g": "tongue-back", "ɠ": "tongue-back", "kʰ": "tongue-back",
          "k'": "tongue-back", "q": "tongue-back", "c": "tongue-back", "ɟ": "tongue-back"}


# Affricates are paired by where they are made too, so a recogniser that calls the same half-voiced sound tʃ in one cut and dʒ in the next
# still has its voicing compared with the speaker's (the labels differ; the voicing is what is measured).
_PLACE.update({"tʃ": "postalveolar-affricate", "dʒ": "postalveolar-affricate", "ts": "alveolar-affricate", "dz": "alveolar-affricate"})


def _place(token: str) -> str:
    """Stops made in the same place pair up whatever their labels (p, b and the recogniser's other b's are all 'lips'); vowels pair in order."""
    if token not in STOPS and is_vowel(token):
        return "V"
    return _PLACE.get(token, token)


def _measured(feature: dict[str, Any]) -> bool:
    return ("release_rise_db" in feature and feature["token"] in STOPS) or ("onset_f1" in feature and is_vowel(feature["token"]) and feature["token"] not in STOPS)


def _vowel_note(index: int, voice: dict[str, Any], group: list[dict[str, Any]], voice_open_f1: float | None = None) -> str | None:
    onset = _median([g["onset_f1"] for g in group])
    found = []
    if abs(voice["onset_f1"] - onset) >= VOWEL_F1_DIFFERENCE_HZ:
        reference = f"; this voice's own plain 'a' reaches about {round(voice_open_f1)} Hz" if voice_open_f1 else ""
        found.append(f"it starts {'more closed' if voice['onset_f1'] < onset else 'more open'}: the first formant is {voice['onset_f1']} Hz "
                     f"against the speaker's {round(onset)} Hz (the higher, the wider the mouth is open{reference})")
    ends = [g["offset_f1"] for g in group if "offset_f1" in g]
    if ends and "offset_f1" in voice:
        speaker_fall, voice_fall = onset - _median(ends), voice["onset_f1"] - voice["offset_f1"]
        if abs(voice_fall - speaker_fall) >= 2 * VOWEL_F1_DIFFERENCE_HZ:
            found.append(f"it {'closes more' if voice_fall > speaker_fall else 'closes less'} through the vowel (F1 falls {round(voice_fall)} Hz "
                         f"against the speaker's {round(speaker_fall)} Hz)")
    return (f"the voice's vowel number {index + 1} ({voice['token']}), against {len(group)} of the speaker's occurrence(s): " + "; ".join(found)) if found else None


def _median(values: Sequence[float]) -> float:
    return float(np.median(values))


def differences(speaker_occurrences: Sequence[Sequence[dict[str, Any]]], voice: Sequence[dict[str, Any]], *, voice_open_f1: float | None = None) -> list[str]:
    """Plain-words differences between the stops of the voice's word and the speaker's, paired by where in the mouth they are made.

    Each stop of the voice is compared with the same stop in every occurrence of the speaker's word (the speaker's measures are the median
    over the occurrences that have it: one clipped or mis-timed occurrence does not decide)."""
    paired: dict[int, list[dict[str, Any]]] = {}
    for occurrence in speaker_occurrences:
        matcher = difflib.SequenceMatcher(a=[_place(f["token"]) for f in occurrence], b=[_place(f["token"]) for f in voice], autojunk=False)
        for block in matcher.get_matching_blocks():
            for k in range(block.size):
                if _measured(occurrence[block.a + k]):
                    paired.setdefault(block.b + k, []).append(occurrence[block.a + k])
    notes: list[str] = []
    for index in sorted(paired):
        v = voice[index]
        if not _measured(v):
            continue
        group = [g for g in paired[index] if g["token"] in STOPS] if v["token"] in STOPS else [g for g in paired[index] if "onset_f1" in g]
        if not group:
            continue
        if v["token"] not in STOPS:
            note = _vowel_note(index, v, group, voice_open_f1)
            if note:
                notes.append(note)
            continue
        s = {key: _median([g[key] for g in group]) for key in ("release_rise_db", "closure_ms", "voiced_closure", "voicing_delay_ms", "hiss_after_release")}
        found = []
        if not _same(s["release_rise_db"], v["release_rise_db"], RELEASE_DIFFERENCE_DB):
            found.append(f"its release is {'softer' if v['release_rise_db'] < s['release_rise_db'] else 'sharper'} "
                         f"({v['release_rise_db']} dB rise against the speaker's {round(s['release_rise_db'], 1)})")
        if not _same(s["closure_ms"], v["closure_ms"], CLOSURE_DIFFERENCE_MS):
            found.append(f"the closure before it lasts {v['closure_ms']} ms against the speaker's {round(s['closure_ms'])}")
        if not _same(s["voiced_closure"], v["voiced_closure"], VOICING_DIFFERENCE):
            found.append(f"the vocal folds vibrate through {round(v['voiced_closure'] * 100)}% of the closure against the speaker's "
                         f"{round(s['voiced_closure'] * 100)}%")
        if not _same(s["voicing_delay_ms"], v["voicing_delay_ms"], VOT_DIFFERENCE_MS):
            found.append(f"voicing starts {v['voicing_delay_ms']} ms after the release against the speaker's {round(s['voicing_delay_ms'])}")
        if not _same(s["hiss_after_release"], v["hiss_after_release"], HISS_DIFFERENCE):
            found.append(f"the hiss after the release is {'more' if v['hiss_after_release'] > s['hiss_after_release'] else 'less'} than the speaker's")
        if found:
            notes.append(f"the voice's stop number {index + 1} ({v['token']}), against {len(group)} of the speaker's occurrence(s): " + "; ".join(found))
    return notes


_STAT_KEYS = ("release_rise_db", "closure_ms", "voiced_closure", "voicing_delay_ms", "hiss_after_release")


def differences_over_cuts(speaker_occurrences: Sequence[Sequence[dict[str, Any]]], cuts: Sequence[Sequence[dict[str, Any]]],
                          *, voice_open_f1: float | None = None) -> list[str]:
    """``differences`` for a voice word measured in several cuts of the same audio.

    The audio is identical in every cut, but where the recogniser puts a sound's edges moves with the amount of audio around it, and the
    measures follow (one half-voiced affricate measured 33% and 67% voiced in two cuts of the same samples).  Each of the speaker's stops is
    paired, cut by cut, with the voice's stop made in the same place, and the median of each measure over the cuts is what is compared; a stop
    found in fewer than half the cuts is left out."""
    occurrences = [occurrence for occurrence in speaker_occurrences if occurrence]
    if not occurrences or not cuts:
        return []
    base = occurrences[0]
    found: dict[int, list[dict[str, Any]]] = {}
    for voice in cuts:
        matcher = difflib.SequenceMatcher(a=[_place(f["token"]) for f in base], b=[_place(f["token"]) for f in voice], autojunk=False)
        for block in matcher.get_matching_blocks():
            for k in range(block.size):
                i, j = block.a + k, block.b + k
                if _measured(base[i]) and _measured(voice[j]) and (base[i]["token"] in STOPS) == (voice[j]["token"] in STOPS):
                    found.setdefault(i, []).append(voice[j])
    voice_word: list[dict[str, Any]] = []
    for i, feature in enumerate(base):
        group = found.get(i, [])
        if _measured(feature) and group and 2 * len(group) >= len(cuts):
            token = Counter(g["token"] for g in group).most_common(1)[0][0]
            keys = _STAT_KEYS if feature["token"] in STOPS else tuple(key for key in ("onset_f1", "offset_f1") if all(key in g for g in group))
            voice_word.append({"token": token, "duration_ms": feature.get("duration_ms", 0), **{key: _median([g[key] for g in group]) for key in keys}})
        else:
            voice_word.append({"token": f"·{i}"})                       # nothing paired here: a placeholder that matches no sound
    return differences(occurrences, voice_word, voice_open_f1=voice_open_f1)


def final_vowel(samples: np.ndarray, start_s: float, end_s: float) -> dict[str, Any] | None:
    """Whether the voice's word ends with a vowel after its last s / ts, measured on the signal itself.

    The phone recogniser is a poor listener at the edge of a word: it missed 5 of 12 final vowels the voice really speaks ("Saensa" was heard as
    "saens").  The signal is clear: after the hiss of the last s / ts comes a voiced stretch nearly as loud as the word's vowels, or silence.
    Returns {"present": bool, "ms": int}, or None when the word does not end in a hiss to anchor on (the check says nothing then)."""
    frames: list[tuple[float, float, float]] = []
    t = start_s
    while t < end_s:
        centre = int(t * SAMPLE_RATE)
        window = samples[max(0, centre - 160):centre + 160]
        if len(window) < 160:
            break
        energy = float(10 * np.log10(np.mean(window ** 2) + 1e-10))
        frames.append((t, energy, _hiss(samples, max(0, centre - 240), 480)))
        t += 0.01
    if not frames:
        return None
    peak = max(energy for _, energy, _ in frames)
    hiss = [i for i, (when, energy, share) in enumerate(frames) if share > 0.5 and energy > peak - 22 and when >= end_s - FINAL_HISS_WINDOW_S]
    if not hiss:
        return None
    run = best = 0
    for _, energy, share in frames[max(hiss) + 1:]:
        if share < 0.3 and energy > peak - FINAL_VOWEL_BELOW_PEAK_DB:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return {"present": best * 10 >= FINAL_VOWEL_MIN_MS, "ms": best * 10}
