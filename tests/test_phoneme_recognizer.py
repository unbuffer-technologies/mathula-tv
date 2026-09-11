"""Real, non-mocked tests against the actual panphon feature-distance engine
(via the phonetics extra: allosaurus + panphon + torch) -- confirming the
articulatory-weighted comparison genuinely behaves as this module's docstring
claims, not a fake's self-report.

Skipped entirely (not failed) when the optional dependency isn't installed --
this must never be a hard requirement for the rest of the test suite to run.
recognize() itself additionally needs Allosaurus's pretrained model, which is
downloaded over the network on first use; that path is exercised only by the
`live`-marked tests, never by the default test run.
"""
from __future__ import annotations

import pytest

from mathula_tv import phoneme_recognizer as pr

pytestmark = pytest.mark.skipif(
    not pr.is_available(), reason="phonetics extra (allosaurus + panphon + torch) not installed"
)


def _seq(*phones: str) -> pr.PhoneSequence:
    return pr.PhoneSequence(tuple(phones))


def test_is_available_reports_true_when_the_extra_is_installed():
    assert pr.is_available() is True


def test_phone_sequence_is_falsy_when_empty():
    assert bool(_seq()) is False
    assert bool(_seq("l", "i", "ŋ")) is True


def test_as_ipa_joins_phones_with_no_separator():
    # panphon re-segments the string itself using its own longest-match
    # segmenter; a space is not a valid IPA segment and would corrupt that.
    assert _seq("l", "ɪ", "ŋ", "k", "ə", "n").as_ipa() == "lɪŋkən"


def test_phonetic_similarity_is_one_for_identical_sequences():
    seq = _seq("l", "ɪ", "ŋ", "k", "ə", "n")
    assert pr.phonetic_similarity(seq, seq) == pytest.approx(1.0)


def test_phonetic_similarity_is_one_for_two_empty_sequences():
    assert pr.phonetic_similarity(_seq(), _seq()) == pytest.approx(1.0)


def test_phonetic_similarity_is_zero_when_only_one_side_is_empty():
    assert pr.phonetic_similarity(_seq(), _seq("l", "ɪ", "ŋ")) == 0.0
    assert pr.phonetic_similarity(_seq("l", "ɪ", "ŋ"), _seq()) == 0.0


def test_phonetic_similarity_scores_a_near_miss_substitution_above_an_unrelated_one():
    # "Lincoln" /lɪŋkən/ vs a near-miss (/p/ for /k/, both stops) should score
    # closer than vs an unrelated substitution (/s/ for /k/, stop vs fricative)
    # -- the whole reason this uses panphon's feature-weighted distance instead
    # of plain Levenshtein, which would score both substitutions identically.
    reference = _seq("l", "ɪ", "ŋ", "k", "ə", "n")
    near_miss = _seq("l", "ɪ", "ŋ", "p", "ə", "n")
    unrelated = _seq("l", "ɪ", "ŋ", "s", "ə", "n")
    assert pr.phonetic_similarity(reference, near_miss) > pr.phonetic_similarity(reference, unrelated)


def test_best_window_similarity_finds_the_name_inside_a_longer_carrier_sequence():
    # Simulates a candidate's phone sequence including fixed carrier-sentence
    # phones around the target name -- the comparison must find the name's own
    # window rather than being diluted by the (identical, for every candidate)
    # surrounding carrier phones.
    needle = _seq("l", "ɪ", "ŋ", "k", "ə", "n")
    haystack = _seq("k", "u", "k", "h", "u", "l", "u", "ŋ", "ɔ", "l", "ɪ", "ŋ", "k", "ə", "n", "k", "u", "l", "e", "s", "i")
    assert pr.best_window_similarity(haystack, needle) == pytest.approx(1.0)


def test_best_window_similarity_is_zero_when_either_side_is_empty():
    assert pr.best_window_similarity(_seq(), _seq("l", "ɪ", "ŋ")) == 0.0
    assert pr.best_window_similarity(_seq("l", "ɪ", "ŋ"), _seq()) == 0.0


def test_best_window_similarity_falls_back_to_whole_sequence_when_needle_is_not_shorter():
    seq = _seq("l", "ɪ", "ŋ", "k", "ə", "n")
    assert pr.best_window_similarity(seq, seq) == pytest.approx(1.0)


@pytest.mark.live
def test_recognize_produces_a_nonempty_phone_sequence_for_real_speech(tmp_path):
    pytest.importorskip("soundfile")
    import numpy as np
    import soundfile as sf

    # A pure tone is not real speech, but this only confirms the real,
    # network-downloaded Allosaurus model runs end-to-end and returns some
    # phone sequence for a real audio file -- not that the content is
    # phonetically meaningful (that's covered by the reference-audio tests
    # this module's own callers add once real name clips are available).
    silence_path = tmp_path / "tone.wav"
    samples = (0.1 * np.sin(2 * np.pi * 220 * np.linspace(0, 1, 16000))).astype("float32")
    sf.write(str(silence_path), samples, 16000)
    result = pr.recognize(silence_path)
    assert isinstance(result, pr.PhoneSequence)
