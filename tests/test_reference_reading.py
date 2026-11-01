from mathula_tv import reference_reading as rr


def test_the_variants_start_with_the_dictionary_and_add_how_a_speaker_reduces_it():
    western = ["w", "ɛ", "s", "t", "ɚ", "n"]
    forms = rr.variants(western)
    assert forms[0] == western
    assert ["w", "ɛ", "s", "ɚ", "n"] in forms                      # the t of Western is not said
    assert ["w", "ɛ", "s", "t", "ə", "n"] in forms                 # the r-coloured schwa as a plain one
    assert len(forms) <= rr.MAX_VARIANTS and len({tuple(f) for f in forms}) == len(forms)
    assert rr.variants(["k", "ɔː", "t"]) == [["k", "ɔː", "t"]]     # nothing to reduce


VOCAB = {"<pad>": 0, "w": 1, "ɛ": 2, "s": 3, "t": 4, "ɚ": 5, "n": 6, "ə": 7, "ɪ": 8, "k": 9, "o": 10, "a": 11, "e": 12, "m": 13, "i": 14, "h": 15}


def fake_cost(spoken):
    """A stand-in for the CTC score: the edit distance between a reading's ids and the sounds the speaker 'said'."""
    def gop_excess(_clip, ids, _blank):
        target = [VOCAB[t] for t in spoken]
        previous = list(range(len(target) + 1))
        for i, x in enumerate(ids, 1):
            row = [i]
            for j, y in enumerate(target, 1):
                row.append(min(previous[j] + 1, row[-1] + 1, previous[j - 1] + (x != y)))
            previous = row
        return float(previous[-1])
    return gop_excess


WESTERN = [{"how": "word", "accent": "rhotic", "phones": ["w", "ɛ", "s", "t", "ɚ", "n"]}, {"how": "word", "accent": "non_rhotic", "phones": ["w", "ɛ", "s", "t", "ə", "n"]}]
DECOYS = [["m", "m", "m", "m", "m"], ["k", "o", "k", "o", "k"], ["a", "a", "a", "a", "a"]]


def test_the_audio_picks_the_variant_it_fits_and_a_cut_that_brings_extra_sounds_does_not_change_the_choice(monkeypatch):
    import mathula_tv.pronunciation_assess as assess

    monkeypatch.setattr(assess, "gop_excess", fake_cost(["w", "ɛ", "s", "ə", "n"]))                       # the speaker drops the t and reduces the vowel
    picked = rr.choose(["clip"], WESTERN, ["w", "ɪ", "s", "ə", "n"], VOCAB, 0, DECOYS)
    assert picked["phones"] == ["w", "ɛ", "s", "ə", "n"] and picked["excess"] == 0.0
    assert picked["accepted"] is True and picked["beaten_by_decoys"] == 0 and picked["separation"] >= rr.SEPARATION
    # extra sounds from a neighbouring word cost every reading and every decoy, so the winner is the same and the word is still found
    monkeypatch.setattr(assess, "gop_excess", fake_cost(["w", "ɛ", "s", "ə", "n", "k", "o"]))
    contaminated = rr.choose(["clip"], WESTERN, ["w", "ɛ", "s", "ə", "n", "k", "o"], VOCAB, 0, DECOYS)
    assert contaminated["phones"] == ["w", "ɛ", "s", "ə", "n"] and contaminated["accepted"] is True


def test_a_window_that_holds_other_speech_is_not_taken(monkeypatch):
    import mathula_tv.pronunciation_assess as assess

    monkeypatch.setattr(assess, "gop_excess", fake_cost(["m", "m", "m", "m", "m"]))                       # the audio is not this word: a decoy fits it better
    picked = rr.choose(["clip"], WESTERN, ["m", "m", "m", "m", "m"], VOCAB, 0, DECOYS)
    assert picked["accepted"] is False and picked["beaten_by_decoys"] >= 1
    assert rr.choose([], WESTERN, [], VOCAB, 0) is None and rr.choose(["clip"], [], [], VOCAB, 0) is None


def test_another_reading_replaces_the_first_only_when_it_is_clearly_better(monkeypatch):
    import mathula_tv.pronunciation_assess as assess

    first = {"how": "word", "accent": "rhotic", "phones": ["w", "ɛ", "s"]}
    rare = {"how": "word", "accent": "rhotic", "phones": ["h", "w", "ɛ", "s"]}                             # a dictionary's rare "hw" variant
    monkeypatch.setattr(assess, "gop_excess", lambda _clip, ids, _blank: 3.0 if len(ids) == 3 else 2.9)       # better by 0.1: noise
    assert rr.choose(["clip"], [first, rare], [], VOCAB, 0)["phones"] == ["w", "ɛ", "s"]
    monkeypatch.setattr(assess, "gop_excess", lambda _clip, ids, _blank: 3.0 if len(ids) == 3 else 2.0)       # better by 1.0: the audio means it
    assert rr.choose(["clip"], [first, rare], [], VOCAB, 0)["phones"] == ["h", "w", "ɛ", "s"]


def test_decoys_are_other_words_of_about_the_same_length_and_the_same_every_time():
    words = {"western": [["W", "EH1", "S", "T", "ER0", "N"]], "bolster": [["B", "OW1", "L", "S", "T", "ER0"]], "monster": [["M", "AA1", "N", "S", "T", "ER0"]],
             "cat": [["K", "AE1", "T"]], "listen": [["L", "IH1", "S", "AH0", "N"]]}
    vocab = {"b": 1, "oʊ": 2, "l": 3, "s": 4, "t": 5, "ɚ": 6, "m": 7, "ɑː": 8, "n": 9, "ɪ": 10, "ʌ": 11, "ə": 12, "k": 13, "æ": 14}
    one = rr.decoy_tokens(words, "Western", 6, vocab)
    assert one == rr.decoy_tokens(words, "Western", 6, vocab) and ["k", "æ", "t"] not in one            # deterministic, and a three-sound word is no decoy for six
    assert all(len(tokens) in {5, 6, 7} for tokens in one) and len(one) == 3                              # western itself is left out
