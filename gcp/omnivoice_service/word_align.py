"""Where every word of a span falls in the audio the voice made for it (forced alignment with Meta's MMS aligner, a wav2vec2 trained on 1,100+ languages).

The text is known exactly, so nothing is recognised: the model only says where each known word sits.  The aligner works on the 26 letters and the apostrophe of a romanised transcript; Zulu, Xhosa and Sepedi are
written in Latin letters, so romanising is just lower-casing and dropping the marks (ê ô š become e o s; the click letters c q x are letters like any other).  An acronym is the exception: its capitals are replaced by the names of the letters, as it is said.  Words are the text split on whitespace, the same way
the caller splits it, and a word with no letter in it (a figure, a dash) comes back as None, so the caller can count words in the text and read the time of the Nth.

Each word also carries "letters": [[letter, seconds, confidence]] for its letters, the middle of each letter's span (the CTC spike of the letter: about 20 ms resolution) and how sure the aligner is that the letter was
said (0-1: a swallowed letter scores near 0), when the aligner gave one span per letter.

Boundaries come from a CTC model, which tends to put a word's edges a little inside the sound (the spike of its first and last letter), up to a few hundredths of a second: callers cut between the end of one word
and the start of the next, never exactly on either.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Callable, Sequence

SAMPLE_RATE = 16000
_KEEP = re.compile(r"[^a-z']")


# An acronym is said as the names of its letters ("ANC" -> "ay en see"), and the aligner is given the sound, not the spelling: with "anc" it hears a word and puts the next word's start 0.3 s early.
LETTER_NAMES = {"a": "ei", "b": "bi", "c": "si", "d": "di", "e": "i", "f": "ef", "g": "ji", "h": "eich", "i": "ai", "j": "jei", "k": "kei", "l": "el", "m": "em", "n": "en", "o": "ou", "p": "pi", "q": "kiu", "r": "ar",
                "s": "es", "t": "ti", "u": "iu", "v": "vi", "w": "dabliu", "x": "eks", "y": "wai", "z": "zed"}
_ACRONYM = re.compile(r"(?<![A-Z])[A-Z]{2,6}(?![A-Za-z])")          # two to six capitals together, not the tail of a longer run of capitals ("ANC", the "SA" of "ActionSA")


def _spelled(match: "re.Match[str]") -> str:
    return "".join(LETTER_NAMES[c.lower()] for c in match.group(0))


def normalise_word(word: str) -> str:
    """The word as the aligner reads it: lower case a-z and apostrophes only, accents dropped, an acronym (two or more capitals) as the names of its letters; empty when nothing is left."""
    stripped = "".join(c for c in unicodedata.normalize("NFKD", str(word).replace("’", "'").replace("‘", "'")) if not unicodedata.combining(c))
    return _KEEP.sub("", _ACRONYM.sub(_spelled, stripped).lower())


def words_of(text: str) -> list[str]:
    return str(text).split()


def align_words(text: str, audio: Any, rate: int, *, model_run: Callable[[Any], Any], tokenize: Callable[[Sequence[str]], Any], align: Callable[[Any, Any], Any],
                resample: Callable[[Any, int, int], Any]) -> list[dict[str, float] | None]:
    """[{"start", "end", "score"} or None] for each whitespace-separated word of ``text``, seconds from the start of ``audio`` (a mono float array at ``rate``).

    The four callables are the model's own (see ``load``); keeping them out of this function lets it be tested without the model."""
    originals = words_of(text)
    normal = [normalise_word(w) for w in originals]
    kept = [i for i, w in enumerate(normal) if w]
    result: list[dict[str, float] | None] = [None] * len(originals)
    if not kept:
        return result
    waveform = resample(audio, rate, SAMPLE_RATE)
    emission = model_run(waveform)                                          # (frames, tokens)
    frames = int(emission.shape[0])
    seconds_per_frame = float(waveform.shape[-1]) / frames / SAMPLE_RATE
    spans = align(emission, tokenize([normal[i] for i in kept]))
    for index, tokens in zip(kept, spans):
        if not tokens:
            continue
        result[index] = {"start": round(float(tokens[0].start) * seconds_per_frame, 3), "end": round(float(tokens[-1].end) * seconds_per_frame, 3),
                         "score": round(sum(float(t.score) for t in tokens) / len(tokens), 3)}
        if len(tokens) == len(normal[index]):                                      # one span per letter: where each letter of the word sounds (the middle of its span), so a caller can name a point inside a word by its letter
            result[index]["letters"] = [[letter, round((float(t.start) + float(t.end)) / 2.0 * seconds_per_frame, 3), round(float(t.score), 2)] for letter, t in zip(normal[index], tokens)]
    return result


def load(device: str = "cuda"):
    """The MMS aligner on ``device``: (align function of (text, mono float32 numpy array, rate) -> words).  Weights come from the torch hub cache (baked into the image)."""
    import numpy as np
    import torch
    import torchaudio

    bundle = torchaudio.pipelines.MMS_FA
    model = bundle.get_model(with_star=False).to(device).eval()
    tokenizer, aligner = bundle.get_tokenizer(), bundle.get_aligner()

    def model_run(waveform):
        with torch.inference_mode():
            emission, _ = model(waveform.to(device))
        return emission[0].float().cpu()

    def resample(audio, rate, target):
        wave = torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0)
        return torchaudio.functional.resample(wave, rate, target) if rate != target else wave

    def run(text: str, audio, rate: int):
        return align_words(text, audio, rate, model_run=model_run, tokenize=tokenizer, align=lambda emission, tokens: aligner(emission, tokens), resample=resample)

    return run
