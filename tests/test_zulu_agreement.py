import pytest

from mathula_tv.zulu_agreement import CLASSES, by_class, locative, merge_vowels, render


@pytest.mark.parametrize("prefix,word,expected", [
    ("wa", "umuntu", "womuntu"),                 # a + u -> o
    ("ya", "indlu", "yendlu"),                   # a + i -> e
    ("ba", "amazwe", "bamazwe"),                 # a + a -> a (corpus: bamazwe)
    ("na", "izinto", "nezinto"),
    ("nga", "umthetho", "ngomthetho"),
    ("kwa", "ukuba", "kokuba"),                  # corpus: kokuba  (the w of kwa goes)
    ("lwa", "umuntu", "lomuntu"),                # corpus: lomuntu
    ("kwa", "izinye", "kwezinye"),               # a + i keeps the w (corpus: kwezinye)
    ("nga", "okwami", "ngokwami"),               # a + o -> o
    ("ngi", "umuntu", "ngiumuntu"),              # a prefix that does not end in -a is left alone
    ("wa", "bantu", "wabantu"),                  # a consonant-initial word is left alone
])
def test_vowel_merge_follows_the_annotated_corpus(prefix, word, expected):
    assert merge_vowels(prefix, word) == expected


@pytest.mark.parametrize("noun,expected", [
    ("imiphakathi", "emiphakathini"),            # all four of these are locatives written in the annotated corpus
    ("indlela", "endleleni"),
    ("umkhakha", "emkhakheni"),
    ("uhlelo", "ohlelweni"),
    ("ukuphela", "ekupheleni"),
    ("indawo", "endaweni"),
    ("izindawo", "ezindaweni"),
    ("ulimi", "olimini"),
])
def test_locative_reproduces_corpus_locatives(noun, expected):
    assert locative(noun) == expected


def test_the_class_table_is_complete_and_consistent():
    assert len({c.cls for c in CLASSES}) == len(CLASSES)
    assert [c.cls for c in CLASSES] == ["1", "1a", "2", "2a", "3", "4", "5", "6", "7", "8", "9", "10", "11", "14", "15"]
    for c in CLASSES:
        for field in ("prefix", "sc", "oc", "poss", "rel", "adj", "quant", "dem_near", "dem_far", "example"):
            assert getattr(c, field), (c.cls, field)
    assert by_class("7").sc == "si-" and by_class("7").poss == "sa" and by_class("15").dem_near == "lokhu"
    # the possessive concord ends in -a for every class, which is what the vowel merge works on
    assert all(c.poss.endswith("a") for c in CLASSES)


def test_the_rendered_module_carries_the_table_and_the_rules():
    text = render()
    assert text.count("\n") > 20
    for needle in ("class | noun prefix", "a+i -> e", "a+u -> o", "womuntu", "kokuba", "yinkantolo", "emiphakathini", "ekupheleni", "kuhulumeni"):
        assert needle in text
    # every class row is in it
    for c in CLASSES:
        assert f"\n{c.cls} | {c.prefix} |" in text
