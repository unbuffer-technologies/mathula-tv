from mathula_tv.english_sounds import EnglishSounds, is_initialism, non_rhotic, to_ipa
from mathula_tv.pronunciation_lab import align

DICTIONARY = """\
ai AY1
court K AO1 R T
heidi HH AY1 D IY0
news N UW1 Z
party P AA1 R T IY0
western W EH1 S T ER0 N
the DH AH0
the(2) DH IY0
about AH0 B AW1 T  # a comment
"""


def sounds(tmp_path):
    path = tmp_path / "cmudict.dict"
    path.write_text(DICTIONARY, encoding="utf-8")
    return EnglishSounds.load(path)


def test_a_dictionary_word_is_read_in_the_recognisers_symbols(tmp_path):
    english = sounds(tmp_path)
    assert english.word("Heidi") == [["h", "aɪ", "d", "iː"]]
    assert english.word("western") == [["w", "ɛ", "s", "t", "ɚ", "n"]]            # the unstressed ER is the r-coloured schwa
    assert english.word("about") == [["ə", "b", "aʊ", "t"]]                       # the unstressed AH is the schwa and the comment is dropped
    assert english.word("the") == [["ð", "ə"], ["ð", "iː"]]                       # both of the dictionary's readings
    assert english.word("Nkandla") == [] and EnglishSounds.load(tmp_path / "missing") is None


def test_capitals_are_said_by_the_names_of_their_letters(tmp_path):
    english = sounds(tmp_path)
    readings = english.say("ANC")
    assert readings[0]["how"] == "letters" and readings[0]["phones"] == ["eɪ", "ɛ", "n", "s", "iː"]
    assert english.letters("AI")[0] == ["eɪ", "aɪ"]                                # the capital I is the name of the letter, aɪ
    assert english.letters("MK")[0] == ["ɛ", "m", "k", "eɪ"]
    assert ["z", "iː"] == english.letters("Z")[1]                                   # the second name of Z
    assert is_initialism("AI") and is_initialism("SABC") and not is_initialism("Heidi") and not is_initialism("A") and not is_initialism("COVID19")


def test_a_run_of_capitals_the_dictionary_also_knows_is_read_both_ways_letters_first(tmp_path):
    readings = sounds(tmp_path).say("NEWS")
    assert [r["how"] for r in readings][0] == "letters" and "word" in {r["how"] for r in readings}
    assert any(r["phones"] == ["n", "uː", "z"] for r in readings)
    ai = sounds(tmp_path).say("AI")
    assert ai[0]["phones"] == ["eɪ", "aɪ"] and any(r["how"] == "word" for r in ai)       # "ai" is also a word in the dictionary


def test_a_speaker_who_does_not_say_r_after_a_vowel_has_their_own_reading(tmp_path):
    english = sounds(tmp_path)
    readings = english.say("court")
    assert [r["accent"] for r in readings] == ["rhotic", "non_rhotic"]
    assert readings[1]["phones"] == ["k", "ɔː", "t"]
    assert non_rhotic(to_ipa(["P", "AA1", "R", "T", "IY0"])) == ["p", "ɑː", "t", "iː"]
    assert non_rhotic(["ɹ", "ɛ", "d"]) == ["ɹ", "ɛ", "d"]                         # r before a vowel stays
    assert english.say("Nkandla") == []


def test_english_stresses_the_last_letter_of_a_word_said_letter_by_letter(tmp_path):
    english = sounds(tmp_path)
    ai = english.say("AI")[0]
    assert ai["phones"] == ["eɪ", "aɪ"] and ai["stress"] == [2, 1]                  # A-I: the weight is on the I
    hiv = english.say("HIV")[0]
    assert hiv["phones"] == ["eɪ", "tʃ", "aɪ", "v", "iː"] and hiv["stress"] == [2, None, 2, None, 1]      # H-I-V: the I is not the last letter, the V is
    heidi = english.say("Heidi")[0]
    assert heidi["stress"] == [None, 1, None, 0]                                    # a word keeps the dictionary's own stress
    court = english.say("court")
    assert court[1]["accent"] == "non_rhotic" and court[1]["phones"] == ["k", "ɔː", "t"] and court[1]["stress"] == [None, 1, None]


def test_the_dictionary_reading_shows_what_the_audio_added(tmp_path):
    english = sounds(tmp_path)
    # the speaker's window for "News" carried the next word's sounds (n uː z m ɪ): against the dictionary they are two extra sounds
    heard = ["n", "uː", "z", "m", "ɪ"]
    best = max(english.say("News"), key=lambda r: align(r["phones"], heard)["closeness"])
    assert best["phones"] == ["n", "uː", "z"] and align(best["phones"], heard)["closeness"] < 1.0
