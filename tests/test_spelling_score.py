import numpy as np

from mathula_tv.spelling_score import dtw_distance, graded_understanding, similarity


def test_the_best_alternative_hearing_that_is_the_word_gives_the_score():
    nbest = [{"Display": "Head.", "Confidence": 0.9}, {"Display": "Hi.", "Confidence": 0.4}, {"Display": "High", "Confidence": 0.3}]
    same = lambda heard, word: heard.strip(".").lower() in {"hi", "high"} and word.lower() == "high"
    assert graded_understanding(nbest, "High", same) == 0.4                                   # the top hearing is wrong, a lower one is the word
    assert graded_understanding(nbest[:1], "High", same) == 0.0 and graded_understanding([], "High", same) == 0.0


def test_feature_sequences_of_the_same_sound_are_closer_than_different_ones_and_speed_does_not_matter_much():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(20, 16)); other = rng.normal(size=(20, 16))
    slow = np.repeat(a, 2, axis=0)                                                            # the same sounds, said at half the speed
    assert dtw_distance(a, a) < 1e-6
    assert dtw_distance(a, slow) < dtw_distance(a, other)
    assert similarity(0.0) == 1.0 and similarity(0.5) < similarity(0.1) < 1.0
