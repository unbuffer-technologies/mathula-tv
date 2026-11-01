from mathula_tv.phone_features import describe, distance, voicing_partners
from mathula_tv.pronunciation_lab import align, substitution_cost


def test_neighbours_on_the_chart_are_closer_than_sounds_far_apart():
    assert distance("i", "ɪ") < distance("i", "e") < distance("i", "u")          # a near vowel, the next height, the far corner
    assert distance("a", "ʌ") < distance("a", "u")
    assert distance("s", "ʃ") < distance("s", "k")                               # the next place against another place and the same manner
    assert distance("n", "m") < distance("n", "s")                               # the same manner against another manner
    assert distance("s", "z") < distance("s", "t")                               # voicing alone against a different manner
    assert distance("tʃ", "ʃ") < distance("tʃ", "k")                             # an affricate is nearer its fricative


def test_a_diphthong_is_compared_by_where_it_glides():
    assert distance("aɪ", "eɪ") < distance("aɪ", "u")
    assert distance("oʊ", "o") < distance("oʊ", "ɪ")
    assert distance("ɛɹ", "ɛ") < distance("ɛɹ", "i")                             # the r-colour alone is a small step


def test_a_sound_the_table_does_not_describe_has_no_distance_and_the_older_rule_applies():
    assert describe("a2") is None and distance("a2", "a") is None
    assert distance("a", "s") is None                                            # a vowel against a consonant is the caller's rule
    assert substitution_cost("a", "s") == 1.0 and substitution_cost("a2", "e") == 0.6


def test_the_lab_ruler_now_tells_a_near_sound_from_a_far_one():
    assert substitution_cost("i", "ɪ") < substitution_cost("i", "u")
    assert substitution_cost("p", "b") < substitution_cost("p", "k")
    assert substitution_cost("e", "e") == 0.0 and substitution_cost("e", "eː") == 0.1
    assert substitution_cost("w", "u") < substitution_cost("w", "k")             # a glide against the vowel it shortens
    assert align(["w", "ɛ", "s", "t"], ["w", "ɪ", "s", "t"])["closeness"] > align(["w", "ɛ", "s", "t"], ["w", "u", "s", "t"])["closeness"]


def test_voicing_partners():
    assert voicing_partners("s", "z") and voicing_partners("t", "d") and not voicing_partners("t", "s") and not voicing_partners("a", "e")
