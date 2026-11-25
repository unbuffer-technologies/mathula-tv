import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gcp" / "omnivoice_service"))
import word_align as wa  # noqa: E402


def test_words_are_read_as_the_aligner_reads_them_letters_only_accents_dropped():
    assert wa.normalise_word("Ngesonto") == "ngesonto"
    assert wa.normalise_word("u-Ayanda") == "uayanda"
    assert wa.normalise_word("Sepedi:") == "sepedi"
    assert wa.normalise_word("šê") == "se"                       # Sepedi marks
    assert wa.normalise_word("ma'am,") == "ma'am"
    assert wa.normalise_word("31") == "" and wa.normalise_word("—") == ""


def _fake_model(frames: int):
    class Emission:
        shape = (frames, 29)

    return lambda waveform: Emission()


def test_word_times_come_back_for_every_whitespace_word_and_none_for_a_word_without_letters():
    seen = {}

    def tokenize(words):
        seen["words"] = list(words)
        return [[1] * len(w) for w in words]

    def align(emission, tokens):
        # one span per word, 10 frames each, starting at frame 5
        return [[SimpleNamespace(start=5 + 10 * i, end=15 + 10 * i, score=0.9)] for i in range(len(tokens))]

    class Wave:
        shape = (1, 16000)                                       # one second at 16 kHz

    out = wa.align_words("Sanibonani 31 Ayanda.", np.zeros(24000, dtype=np.float32), 24000, model_run=_fake_model(50), tokenize=tokenize, align=align,
                         resample=lambda audio, rate, target: Wave())
    assert seen["words"] == ["sanibonani", "ayanda"]
    assert out[1] is None and len(out) == 3
    assert out[0] == {"start": round(5 * 1 / 50, 3), "end": round(15 / 50, 3), "score": 0.9}          # 50 frames over 1 s: 0.02 s per frame
    assert out[2]["start"] == round(15 / 50, 3)


def test_a_text_with_no_letters_is_not_aligned_at_all():
    out = wa.align_words("12 — 3", np.zeros(100, dtype=np.float32), 24000, model_run=lambda w: 1 / 0, tokenize=lambda w: 1 / 0, align=lambda e, t: 1 / 0, resample=lambda a, r, t: 1 / 0)
    assert out == [None, None, None]
