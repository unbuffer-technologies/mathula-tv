from mathula_tv.phonology import NOTHING, Phonology, learn, per


def _pairs():
    # English speakers' phones vs Zulu speakers': the schwa becomes a or i, the GOAT vowel o, a vowel is added after a final consonant
    return [(["h", "ə", "l", "əʊ"], ["h", "a", "l", "ɔ"]),
            (["b", "ə", "n", "əʊ"], ["b", "a", "n", "ɔ"]),
            (["t", "ə", "m", "əʊ"], ["t", "i", "m", "ɔ"]),
            (["s", "t", "ɒ", "p"], ["s", "i", "t", "ɔ", "p", "i"]),
            (["k", "ɒ", "p"], ["k", "ɔ", "p", "i"])]


def test_learn_counts_what_zulu_speakers_do_with_each_english_phone():
    knowledge = learn(_pairs())
    schwa = {o["zulu"]: o["share"] for o in knowledge["map"]["ə"]}
    assert schwa["a"] > schwa["i"] > 0
    assert knowledge["map"]["əʊ"][0]["zulu"] == "ɔ" and knowledge["map"]["əʊ"][0]["share"] == 1.0


def test_learn_records_where_a_vowel_is_added():
    insertions = learn(_pairs())["insertions"]
    assert insertions["C_#"]["per_chance"] > 0 and insertions["C_#"]["vowels"][0][0] == "i"


def test_adapt_uses_the_most_usual_phone_and_leaves_unseen_phones_alone():
    model = Phonology(learn(_pairs()))
    assert model.adapt(["h", "ə", "l", "əʊ"]) == ["h", "a", "l", "ɔ"]
    assert model.options("ʔ") == [{"zulu": "ʔ", "share": 1.0, "n": 0}]


def test_per_is_edits_over_the_reference_length():
    assert per(["a", "b"], ["a", "b"]) == 0 and per(["a"], ["a", "b"]) == 0.5 and per([], []) == 0
    assert NOTHING == ""
