"""A second phone recogniser, from a different family, to say when the first one cannot be trusted.

The lab's "speaker's sounds" are one recogniser's best guess (wav2vec 2.0 XLSR-53 with eSpeak phones).  A second, independent recogniser (ZIPA, a Zipformer trained
on 17,000 hours of 88 languages) reads the same clip; where the two decodings of a word differ a lot, neither is a reference to match a spelling to.  Agreement
is not trust: both read the stretch timed as "High" in "the Western Cape High Court" as "pa", because that is what the audio holds.  The check is one of
several, beside the plausibility of the sounds against the word's letters and the English-voice reference.

The model is the small CTC one (``anyspeech/zipa-small-crctc-ns-700k``, 64M parameters, ONNX), which runs on the CPU in a fraction of a second per clip.
The extra ``second-listener`` (onnxruntime) must be installed; without it ``available()`` is False and nothing here runs.
"""
from __future__ import annotations

import difflib
import re
import unicodedata
from pathlib import Path
from typing import Any, Sequence

REPO = "anyspeech/zipa-small-crctc-ns-700k"
FRAME_SECONDS = 0.02            # the CTC head emits one frame per two 10 ms feature frames
AGREEMENT_FLOOR = 0.5           # below this share the two decodings of a word are different words' worth of sounds

_COARSE = {                     # vowels to five classes, consonants to their base: a comparison of what was said, not of how the recogniser spells it
    **{c: "a" for c in "aɑæʌɐəɚɜ"}, **{c: "e" for c in "eɛ"}, **{c: "i" for c in "iɪɨɯ"}, **{c: "o" for c in "oɔɒɵø"}, **{c: "u" for c in "uʊʉ"},
    "ɹ": "r", "ɾ": "r", "ɻ": "r", "ʁ": "r", "ɡ": "g", "ɫ": "l", "ʃ": "S", "ʒ": "S", "ɕ": "S", "ʂ": "S", "ʐ": "S", "ʈ": "t", "ɖ": "d", "ŋ": "n", "ɲ": "n", "θ": "t", "ð": "d",
}


def coarse(phones: Sequence[str]) -> str:
    """The phones as a short string of base sounds: modifiers and length marks dropped, vowels in five classes, repeats merged."""
    text = unicodedata.normalize("NFD", "".join(str(p) for p in phones))
    text = "".join(c for c in text if not unicodedata.combining(c) and c not in "ːˑʰʲʷ˞▁ ʔ")
    text = "".join(_COARSE.get(c, c) for c in text)
    return re.sub(r"(.)\1+", r"\1", text)


def agreement(first: Sequence[str], second: Sequence[str]) -> float | None:
    """How alike two decodings of the same word are, 0 to 1; None when either is empty."""
    a, b = coarse(first), coarse(second)
    if not a or not b:
        return None
    return round(difflib.SequenceMatcher(None, a, b).ratio(), 3)


class ZipaListener:
    def __init__(self, repo: str = REPO) -> None:
        self.repo = repo
        self._session: Any = None
        self._vocab: dict[int, str] = {}

    @staticmethod
    def available() -> bool:
        try:
            import onnxruntime  # noqa: F401
            import torchaudio  # noqa: F401
            return True
        except Exception:
            return False

    def _load(self) -> None:
        if self._session is not None:
            return
        import onnxruntime as ort
        from huggingface_hub import snapshot_download

        root = Path(snapshot_download(self.repo, allow_patterns=["*.onnx", "*tokens*", "*.txt", "*.json"]))
        model = sorted((p for p in root.rglob("*.onnx") if "int8" not in p.name and "fp16" not in p.name), key=lambda p: p.name)[0]
        for line in next(root.rglob("tokens.txt")).read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if parts:
                self._vocab[int(parts[1])] = parts[0]
        self._session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])

    def timed_phones(self, wav: Path, start: float | None = None, end: float | None = None) -> list[tuple[str, float, float]]:
        """The phones heard in the stretch with their start and end seconds from the start of the stretch."""
        import numpy as np
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
        features = torchaudio.compliance.kaldi.fbank(signal[None], num_mel_bins=80, sample_frequency=16_000, dither=0.0, snip_edges=False)
        log_probs = self._session.run(None, {"x": features[None].numpy().astype(np.float32), "x_lens": np.array([features.shape[0]], dtype=np.int64)})[0][0]
        ids = log_probs.argmax(axis=-1)
        out: list[tuple[str, float, float]] = []
        t = 0
        while t < len(ids):
            if ids[t] == 0:
                t += 1
                continue
            j = t
            while j + 1 < len(ids) and ids[j + 1] == ids[t]:
                j += 1
            token = self._vocab.get(int(ids[t]), "")
            if token and token != "▁":
                out.append((token, t * FRAME_SECONDS, (j + 1) * FRAME_SECONDS))
            t = j + 1
        return out

    def phones_between(self, wav: Path, low: float, high: float) -> list[str]:
        """The phones whose middle falls between ``low`` and ``high`` seconds of the clip (the same window the first recogniser's decoding was cut to)."""
        return [token for token, begin, finish in self.timed_phones(wav) if low <= (begin + finish) / 2 <= high]
