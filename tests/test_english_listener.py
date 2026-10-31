import numpy as np

from mathula_tv.english_listener import ctc_viterbi, letters


def _emissions(frames, symbols, blank=0, vocab=5):
    """Frames x vocab log-probabilities where each frame strongly prefers the symbol given for it (None = blank)."""
    lp = np.full((len(frames), vocab), np.log(0.01))
    for t, symbol in enumerate(frames):
        lp[t, blank if symbol is None else symbol] = np.log(0.96)
    return lp


def test_letters_are_what_the_model_reads():
    assert letters("Nonvalo,") == "NONVALO" and letters("don't") == "DON'T" and letters("22nd") == "ND" and letters("") == ""


def test_the_viterbi_path_puts_each_symbol_on_the_frames_that_prefer_it():
    # symbols 1, 2, 2 (a repeated letter needs a blank between its two frames) and 3
    lp = _emissions([None, 1, 1, None, 2, None, 2, 3, 3, None], [1, 2, 2, 3])
    path = ctc_viterbi(lp, [1, 2, 2, 3], blank=0)
    assert path == [None, 0, 0, None, 1, None, 2, 3, 3, None]


def test_a_symbol_the_audio_never_prefers_still_gets_a_frame_and_the_path_is_complete():
    lp = _emissions([1, 1, None, 3, 3], [1, 2, 3])
    path = ctc_viterbi(lp, [1, 2, 3], blank=0)
    assert len(path) == 5 and {p for p in path if p is not None} == {0, 1, 2}
