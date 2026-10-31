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


def test_human_cost_is_above_an_exact_match_and_none_for_what_speakers_do_not_do():
    model = Phonology(learn(_pairs()))
    assert 0.15 <= model.human_cost("ə", "a") < model.human_cost("ə", "i") <= 0.3
    assert model.human_cost("əʊ", "ɔ") == 0.15                                  # what most do is the cheapest way
    assert model.human_cost("əʊ", "f") is None
    assert model.human_cost("ʊ", "u") is None or model.human_cost("ʊ", "u") >= 0.15      # a recogniser's ʊ is read as the corpus's u


def test_what_speakers_do_lists_only_the_sounds_that_get_changed():
    knowledge = {"schema": "mathula-phonology-v1", "map": {
        "ə": [{"zulu": "ə", "n": 40, "share": 0.4}, {"zulu": "a", "n": 21, "share": 0.21}, {"zulu": "", "n": 6, "share": 0.06}],
        "s": [{"zulu": "s", "n": 92, "share": 0.92}],
        "ʃ": [{"zulu": "ʃ", "n": 5, "share": 1.0}]}}
    said = Phonology(knowledge).what_speakers_do(["s", "ə", "ʃ", "ə"])
    assert [s["speaker_sound"] for s in said] == ["ə"]                           # s is kept by most, ʃ too rare to say anything, ə listed once
    assert said[0]["zulu_speakers_say"].startswith("ə 40%, a 21%")
