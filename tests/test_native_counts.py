import re

import pytest

from mathula_tv import pronunciation as p
from mathula_tv.pronunciation import PronunciationDictionary, native_count


def _count(text):
    match = p._ZU_PREFIXED_TWO_DIGIT_CARDINAL.search(text)
    return native_count(match, text) if match else None


@pytest.mark.parametrize("text, said", [
    ("iqembu elineminyaka engu-6, silethe", "emiyisithupha"),          # 4: the relative concord e- of a plural class 4 noun (iminyaka)
    ("abantu abangu-5 bafile", "abahlanu"),
    ("amavikini angu-4 edlule", "amane"),
    ("izinsuku ezingu-3", "ezintathu"),                                 # izin-: class 10
    ("ezinsukwini ezi-3 ezedlule", "ezintathu"),                        # a locative noun is still class 10
    ("izikhulu ezingu-4", "ezine"),                                     # izi- before anything but n or m: class 8
    ("abantu aba-2", "ababili"),
    ("iminyaka emi-9", "emiyisishiyagalolunye"),
])
def test_a_count_of_two_to_nine_after_its_prefix_is_the_counting_word_that_agrees_with_the_noun(text, said):
    assert _count(text) == said


@pytest.mark.parametrize("text", [
    "ngo-9 ekuseni", "kuka-5", "ku-3 kuya ku-6",             # a time or a range: spoken as English as before
    "abantu abangu-11", "engu-20", "engu-2.5", "ngu-50%",      # 10 and over, a decimal, a percent
    "iminyaka engu-1", "ezingu-3",                              # 1 is the noun, not a count; "ezingu-3" with no noun to say its class
    "ngowesi-4", "owesi-2",                                     # an ordinal
])
def test_everything_else_keeps_the_english_code_switch(text):
    assert _count(text) is None


def test_a_figure_the_listener_spelled_by_ear_is_said_as_spelled_and_a_native_count_is_not_listed_as_a_figure():
    text = "abantu abangu-5 bafile"
    match = p._ZU_PREFIXED_TWO_DIGIT_CARDINAL.search(text)
    assert native_count(match, text, {"5": "fayiv"}) is None
    assert p.prefixed_figures(["abantu abangu-5 bafile", "abantu abangu-11"]) == [("11", "", False)]          # the 11 is still a figure to hear; the 5 is a word
    assert p.prefixed_figures(["ngo-5 ekuseni"]) == [("5", "", False)]


def test_the_dictionary_says_a_small_count_in_isizulu_and_a_larger_figure_in_english():
    dictionary = PronunciationDictionary("1.2.0", entries=())
    out = dictionary.apply("Uzobusa iminyaka emihlanu. Abantu abangu-5 nezinsuku ezingu-3 kanye neminyaka engu-34.").tts_text
    assert "abahlanu" in out and "ezintathu" in out and "engu-" in out and "abangu-5" not in out and "ezingu-3" not in out
    assert re.search(r"engu-thethi", out)                                                                      # 34 stays English
