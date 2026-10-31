"""Scores for how well a spelling makes the Zulu voice say an English word, built from signals the phone-label comparison throws away.

The lab's closeness compares the phone labels a recogniser writes for the voice with the labels it wrote for the speaker.  It ties spellings that sound different
(Heidi: "Haidi", "Aydie" and "Haydi" all 0.88, which the English listener wrote as "Heidi", "I did" and "Heidi"), rewards readings that lose the word ("Dyi" 1.0,
"Phakh" 0.97), and prices a gross vowel error at a few hundredths (the "heh" for High scored 0.70).  Two independent signals, tried on the same spellings:

* ``listener_score``: how likely an English listener hears the word.  Azure's short-audio recogniser returns its alternative hearings with a confidence each
  (``format=detailed``); the score is the confidence of the best alternative that is the word, homophones counting ("Hi" for High).  Zero when none is.
* ``feature_similarity``: how alike the voice's clip and a clean English-voice clip of the same word are in the internal features of a speech model, compared
  with dynamic time warping (the method of SpeechBERTScore).  It sees vowel quality that a phone label cannot.

Both are scored per spelling of one word and compared across the spellings, never read as absolute numbers.
"""
from __future__ import annotations

import math
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

LISTENER_LANGUAGE = "en-ZA"


def graded_understanding(nbest: Sequence[Mapping[str, Any]], word: str, understood: Callable[[str, str], bool]) -> float:
    """The confidence of the best alternative hearing that is the word (0 when none of them is)."""
    best = 0.0
    for alternative in nbest:
        text = str(alternative.get("Display") or alternative.get("Lexical") or "")
        if text and understood(text, word):
            best = max(best, float(alternative.get("Confidence") or 0.0))
    return best


def listener_nbest(wav: Path, key: str, region: str, language: str = LISTENER_LANGUAGE) -> list[dict[str, Any]]:
    """Azure short-audio recognition with the alternative hearings; the clip is converted to 16 kHz mono first."""
    import requests

    with tempfile.TemporaryDirectory() as directory:
        converted = Path(directory) / "clip.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(wav), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(converted)], check=True)
        response = requests.post(
            f"https://{region}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1",
            params={"language": language, "format": "detailed"}, timeout=60, data=converted.read_bytes(),
            headers={"Ocp-Apim-Subscription-Key": key, "Content-Type": "audio/wav; codecs=audio/pcm; samplerate=16000", "Accept": "application/json"})
    response.raise_for_status()
    return list(response.json().get("NBest") or [])


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance along the best monotonic path between two feature sequences (frames x dimensions), divided by the two sequences' lengths together."""
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    cost = 1.0 - a @ b.T
    rows, cols = cost.shape
    total = np.full((rows + 1, cols + 1), np.inf)
    total[0, 0] = 0.0
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            total[i, j] = cost[i - 1, j - 1] + min(total[i - 1, j - 1], total[i - 1, j], total[i, j - 1])
    return float(total[rows, cols] / (rows + cols))


def similarity(distance: float) -> float:
    """A distance as a similarity between 0 and 1 (identical sequences are 1)."""
    return math.exp(-3.0 * max(0.0, distance))


class FeatureExtractor:
    """Internal features of a speech model (wav2vec 2.0 base, a middle layer where the content of what is said is strongest and the speaker least), 50 frames a second."""

    LAYER = 8

    def __init__(self, model_name: str = "facebook/wav2vec2-base-960h") -> None:
        self.model_name, self._model = model_name, None

    def features(self, wav: Path, start: float | None = None, end: float | None = None) -> np.ndarray:
        import soundfile
        import torch
        import torchaudio
        from transformers import Wav2Vec2Model

        if self._model is None:
            self._model = Wav2Vec2Model.from_pretrained(self.model_name).eval()
        info = soundfile.info(str(wav))
        first = max(0, int((start or 0.0) * info.samplerate))
        last = info.frames if end is None else min(info.frames, int(end * info.samplerate))
        audio, rate = soundfile.read(str(wav), start=first, stop=last, dtype="float32", always_2d=True)
        signal = torch.from_numpy(audio.mean(axis=1))
        if rate != 16_000:
            signal = torchaudio.functional.resample(signal, rate, 16_000)
        signal = (signal - signal.mean()) / torch.sqrt(signal.var() + 1e-7)
        with torch.no_grad():
            hidden = self._model(signal.unsqueeze(0), output_hidden_states=True).hidden_states[self.LAYER][0]
        return hidden.numpy()


def feature_similarity(extractor: FeatureExtractor, voice_clip: Path, reference_clip: Path, reference_span: tuple[float, float] | None = None) -> float:
    reference = extractor.features(reference_clip, *(reference_span or (None, None)))
    return similarity(dtw_distance(extractor.features(voice_clip), reference))
