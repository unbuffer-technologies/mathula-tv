from mathula_tv import comma_check as cc
from mathula_tv import word_alignment as wa

FORMS = {
    "speaker": [("somlomo", 0.47), ("usomlomo", 0.35)],
    "signature": [("ukusayina", 0.23), ("isiginesha", 0.11)],
    "catch": [("isihlahla", 0.5)],
    "lawyer": [("ummeli", 0.27), ("kummeli", 0.23)],
}


def forms_of(word):
    return FORMS.get(word, [])


def _pairs(english, zulu):
    return [(english[k], zulu[j]) for k, j in wa.make_aligner(forms_of)(english, zulu)]


def test_a_name_a_number_and_a_hyphenated_loanword_cross_into_the_zulu_with_grammar_round_them():
    assert ("Matlala", "likaMatlala") in _pairs(["Matlala", "said"], ["likaMatlala", "wathi"])
    assert ("2017", "ngo-2017") in _pairs(["in", "2017"], ["ngo-2017"])
    assert ("initials", "ama-initials") in _pairs(["his", "initials"], ["ama-initials", "akhe"])


def test_a_short_english_word_inside_a_longer_zulu_word_is_chance_not_a_copy():
    assert _pairs(["that", "is"], ["wesithathu", "uyena"]) == []
    assert _pairs(["signed", "there"], ["nesignesha", "lapho"]) == []              # a respelling is the corpus's to say (translation_forms), not a letter run


def test_a_word_the_corpus_translates_is_found_in_any_prefixed_form_and_not_in_an_unrelated_word():
    assert ("speaker", "likasomlomo") in _pairs(["the", "speaker", "said"], ["isinqumo", "likasomlomo", "esitshelwe"])
    assert ("lawyer", "kummeli") in _pairs(["his", "lawyer"], ["kummeli"])
    assert _pairs(["his", "lawyer"], ["ngumuntu", "wesifunda"]) == []


def test_pairs_that_cross_each_other_are_dropped_so_only_the_order_the_rest_keep_stays():
    pairs = wa.chain([(0, 3, 1.0), (1, 1, 0.5), (2, 2, 0.5), (3, 4, 1.0)])
    assert pairs == [(1, 1), (2, 2), (3, 4)] or pairs == [(0, 3), (3, 4)]
    assert all(a[0] < b[0] and a[1] < b[1] for a, b in zip(pairs, pairs[1:]))


def _heard(english, held_after, gap=1.4):
    spec, t = [], 0.0
    for n, word in enumerate(english.split()):
        spec.append({"text": word, "start": t, "end": t + 0.3})
        t += 0.3 + (gap if n == held_after else 0.0)
    return spec


ENGLISH = "we know Dumisani Khumalo and Nhlanhla Mkhwanazi who was the big catch as well as Masemola."
HELD_AFTER = ENGLISH.split().index("catch")
# the Zulu retells the sentence: the catch comes right after the first name and the pause belongs there, far from the sentence's proportional place
RETOLD = ("Ubale uMkhwanazi isihlahla {marker}esikhulu kakhulu futhi kudlule isikhathi eside kakhulu ngoba abantu abaningi bamazi kahle ngokwaneleyo noMasemola.")


def _check(text, aligner):
    words = _heard(ENGLISH, HELD_AFTER)
    return cc.comma_check(text, words, 0, int(words[-1]["end"] * 1000) + 50, aligner)


def test_the_pause_is_looked_for_where_the_aligned_words_are_said_not_at_the_same_fraction_of_the_sentence():
    aligner = wa.make_aligner(forms_of)
    at_the_word = RETOLD.format(marker="~pause1400 ")
    assert _check(at_the_word, None)                                              # by position alone the marker is far from the sentence's 0.8 and the pause is called missing
    assert _check(at_the_word, aligner) is None                                   # by the words it sits right after the catch


def test_a_marker_that_is_not_at_the_aligned_place_is_still_reported():
    aligner = wa.make_aligner(forms_of)
    elsewhere = RETOLD.format(marker="").replace("abaningi ", "abaningi ~pause1400 ")
    found = _check(elsewhere, aligner)
    assert found and "catch" in " ".join(found["pauses_the_speaker_held_that_the_dub_does_not"])


def test_nothing_aligned_means_the_position_only_check_runs():
    aligner = wa.make_aligner(lambda word: [])
    text = "Yebo kuyiqiniso futhi abantu abaningi bayazi ngalokho."
    assert _check(text, aligner) == _check(text, None)


def test_an_aligner_whose_corpus_lookup_fails_answers_nothing_and_the_check_still_runs():
    def broken(word):
        raise RuntimeError("no corpus")
    safe = wa.make_aligner(broken)
    assert safe(["catch", "Matlala"], ["isihlahla", "likaMatlala"]) == [(1, 1)]          # the copy still aligns; the translation lookup failing only loses the translations
    assert _check(RETOLD.format(marker=""), safe)
