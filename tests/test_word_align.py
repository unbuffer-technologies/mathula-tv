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


def test_a_word_carries_the_time_of_each_of_its_letters_when_the_aligner_gave_one_span_per_letter():
    def tokenize(words):
        return [[1] * len(w) for w in words]

    def align(emission, tokens):
        return [[SimpleNamespace(start=10 * i + 4 * k, end=10 * i + 4 * k + 2, score=0.8) for k in range(len(word))] for i, word in enumerate(tokens)]

    class Wave:
        shape = (1, 16000)

    out = wa.align_words("Ama-details okugcina", np.zeros(24000, dtype=np.float32), 24000, model_run=_fake_model(50), tokenize=tokenize, align=align, resample=lambda audio, rate, target: Wave())
    assert [letter for letter, *_ in out[0]["letters"]] == list("amadetails") and [letter for letter, *_ in out[1]["letters"]] == list("okugcina")
    assert out[1]["letters"][0] == ["o", round((10 + 1) * 1 / 50, 3), 0.8]                               # the middle of frames 10-12, 0.02 s a frame, and the aligner's confidence in the letter
    assert out[1]["letters"][3][0] == "g" and out[1]["letters"][3][1] > out[1]["letters"][2][1]


def test_an_acronym_is_read_as_the_names_of_its_letters_and_other_words_are_not():
    assert wa.normalise_word("ANC") == "eiensi" and wa.normalise_word("I-ANC") == "ieiensi" and wa.normalise_word("yi-ANC.") == "yieiensi"
    assert wa.normalise_word("be-ActionSA,") == "beactionesei" and wa.normalise_word("se-MMC") == "seememsi"
    assert wa.normalise_word("Ayanda") == "ayanda" and wa.normalise_word("I") == "i" and wa.normalise_word("uMkhulu") == "umkhulu"          # one capital is a name's first letter, not an acronym
    assert wa.normalise_word("ABCDEFGHIJ") == "abcdefghij"                                                                                  # a long run of capitals is a shouted word
