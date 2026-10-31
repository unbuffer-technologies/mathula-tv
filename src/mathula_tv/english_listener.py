"""A second listener for the speaker's audio: an English letter recogniser (CTC), used for what the phone recogniser cannot tell us.

The pronunciation lab compares the voice with "the speaker's sounds", which are one phone recogniser's best guess for the stretch of audio that the
speech-to-text service timed as the word.  Both can be wrong: the service's word times are coarse (a stretch timed as "High" decoded as "p aː k", the
sounds of another word), and one recogniser's guess is not a measurement.  This module adds two independent views from a different model that reads letters:

* ``align``: where the words really are.  The known words around the one of interest are force-aligned to the audio with a CTC Viterbi, which gives each
  word its own start and end and a confidence, in place of the service's coarse times.
* ``word_score``: how well the audio is the word.  The log-probability of the word's letters given the audio, against the best the model can find in the
  same stretch, as a share between 0 and 1.  A graded version of "did the English listener write the word", for the speaker's clip (is the reference the
  word at all?) and for the voice's clip of a spelling (how recognisable is it?).

The model is ``facebook/wav2vec2-base-960h``; it runs on the CPU and is loaded once, on first use.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np

MODEL = "facebook/wav2vec2-base-960h"
FRAME_SECONDS = 0.02            # one emission per 20 ms
ALIGN_CONTEXT_WORDS = 2         # words either side of the target that are aligned with it, to pin it down
ALIGN_MARGIN_SECONDS = 0.4
MIN_WORD_CONFIDENCE = -2.5      # mean log-probability per frame below which an alignment is not trusted


def letters(word: str) -> str:
    """The word as the model reads it: capital letters and the apostrophe."""
    return re.sub(r"[^A-Z']", "", str(word or "").upper())


class EnglishListener:
    def __init__(self, model_name: str = MODEL) -> None:
        self.model_name = model_name
        self._model = None
        self._vocab: dict[str, int] = {}
        self._blank = 0

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch  # noqa: F401
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        processor = Wav2Vec2Processor.from_pretrained(self.model_name)
        self._model = Wav2Vec2ForCTC.from_pretrained(self.model_name).eval()
        self._vocab = dict(processor.tokenizer.get_vocab())
        self._blank = int(self._vocab.get("<pad>", 0))

    def log_probs(self, wav: Path, start: float | None = None, end: float | None = None) -> tuple[np.ndarray, float]:
        """(frames x symbols log-probabilities, the stretch's start in seconds) for part of an audio file."""
        import soundfile
        import torch
        import torchaudio

        self._load()
        info = soundfile.info(str(wav))
        first = max(0, int((start or 0.0) * info.samplerate))
        last = info.frames if end is None else min(info.frames, int(end * info.samplerate))
        audio, rate = soundfile.read(str(wav), start=first, stop=last, dtype="float32", always_2d=True)
        signal = torch.from_numpy(audio.mean(axis=1))
        if rate != 16_000:
            signal = torchaudio.functional.resample(signal, rate, 16_000)
        signal = (signal - signal.mean()) / torch.sqrt(signal.var() + 1e-7)
        with torch.no_grad():
            logits = self._model(signal.unsqueeze(0)).logits[0]
        return torch.log_softmax(logits, dim=-1).numpy(), first / info.samplerate

    def transcribe(self, wav: Path, start: float | None = None, end: float | None = None) -> str:
        """The model's own best reading of the stretch (greedy), in lower case."""
        lp, _ = self.log_probs(wav, start, end)
        ids = lp.argmax(axis=-1)
        inverse = {i: s for s, i in self._vocab.items()}
        out, previous = [], -1
        for i in ids:
            if i != previous and i != self._blank:
                out.append(inverse.get(int(i), ""))
            previous = i
        return "".join(out).replace("|", " ").strip().lower()

    def word_score(self, wav: Path, word: str, start: float | None = None, end: float | None = None) -> float | None:
        """How much of the best reading of the stretch the word's letters keep: exp(log P(word) - log P(best path)), 0 to 1.  None when the word has no letters."""
        import torch

        text = letters(word)
        if not text:
            return None
        lp, _ = self.log_probs(wav, start, end)
        targets = torch.tensor([[self._vocab[c] for c in text if c in self._vocab]], dtype=torch.long)
        if targets.shape[1] == 0 or lp.shape[0] < targets.shape[1]:
            return 0.0
        log_p = torch.from_numpy(lp)[:, None, :]
        nll = torch.nn.functional.ctc_loss(log_p, targets, torch.tensor([lp.shape[0]]), torch.tensor([targets.shape[1]]), blank=self._blank,
                                           reduction="sum", zero_infinity=True)
        best = float(lp.max(axis=-1).sum())
        return float(np.exp(min(0.0, -float(nll) - best)))

    def align(self, wav: Path, words: Sequence[str], start: float, end: float) -> list[dict[str, Any]]:
        """Force-align ``words`` (in order) to the stretch ``start``..``end``: each word's start and end in the file's seconds, and its confidence."""
        self._load()
        lp, offset = self.log_probs(wav, start, end)
        symbols: list[int] = []
        owner: list[int] = []
        delimiter = self._vocab.get("|")
        for index, word in enumerate(words):
            for c in letters(word):
                if c in self._vocab:
                    symbols.append(self._vocab[c])
                    owner.append(index)
            if delimiter is not None and index < len(words) - 1:
                symbols.append(delimiter)
                owner.append(-1)
        if not symbols or lp.shape[0] < len(symbols):
            return []
        path = ctc_viterbi(lp, symbols, self._blank)
        rows: list[dict[str, Any]] = []
        for index, word in enumerate(words):
            frames = [t for t, state in enumerate(path) if state is not None and owner[state] == index]
            if not frames:
                rows.append({"word": word, "start": None, "end": None, "confidence": None})
                continue
            confidence = float(np.mean([lp[t, symbols[path[t]]] for t in frames]))
            rows.append({"word": word, "start": offset + min(frames) * FRAME_SECONDS, "end": offset + (max(frames) + 1) * FRAME_SECONDS,
                         "confidence": round(confidence, 3)})
        return rows


def ctc_viterbi(lp: np.ndarray, symbols: Sequence[int], blank: int) -> list[int | None]:
    """The most likely CTC path: for each frame, the index into ``symbols`` that it emits, or None for a blank."""
    frames, labels = lp.shape[0], len(symbols)
    states = 2 * labels + 1
    extended = [blank if s % 2 == 0 else symbols[s // 2] for s in range(states)]
    score = np.full((frames, states), -np.inf)
    back = np.zeros((frames, states), dtype=np.int16)
    score[0, 0] = lp[0, blank]
    if states > 1:
        score[0, 1] = lp[0, extended[1]]
    for t in range(1, frames):
        for s in range(states):
            options = [(score[t - 1, s], 0)]
            if s >= 1:
                options.append((score[t - 1, s - 1], 1))
            if s >= 2 and extended[s] != blank and extended[s] != extended[s - 2]:
                options.append((score[t - 1, s - 2], 2))
            best, step = max(options, key=lambda o: o[0])
            score[t, s] = best + lp[t, extended[s]]
            back[t, s] = step
    state = states - 1 if states == 1 or score[-1, states - 1] >= score[-1, states - 2] else states - 2
    path: list[int | None] = [None] * frames
    for t in range(frames - 1, -1, -1):
        path[t] = None if extended[state] == blank else state // 2
        state -= int(back[t, state])
    return path
