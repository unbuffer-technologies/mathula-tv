import pytest

from mathula_tv.zulu_object import (ABSOLUTE_PRONOUNS, CLASS_OBJECT, PERSON_OBJECT, join_concord, object_concord, render, with_object)
from mathula_tv.zulu_tam import conjugate


@pytest.mark.parametrize("concord,root,expected", [
    ("ku", "az", "kwaz"),                    # kwazi (news, 7,282)
    ("ku", "enz", "kwenz"),                  # kwenza (5,960)
    ("lu", "az", "lwaz"),                    # lwazi (2,317)
    ("ngi", "az", "ngaz"),                   # ngazi (487)
    ("ngi", "enz", "ngenz"),                 # ngenza (276)
    ("si", "enz", "senz"),                   # senza (106)
    ("ba", "enz", "benz"),                   # benza (162)
    ("yi", "az", "yaz"),                     # yazi (3,230)
    ("li", "enz", "lenz"),                   # lenza (49)
    ("wa", "enz", "wenz"),                   # wenza (329)
    ("zi", "az", "zaz"),                     # zazi (265): also the reflexive zi + azi
    ("zi", "enz", "zenz"),                   # zenza (295)
    ("wu", "enz", "wenz"),                   # class 3: the w is already in wu
    ("ku", "thand", "kuthand"),              # a consonant root: no change
    ("ngi", "bon", "ngibon"),
])
def test_a_concord_merges_with_a_vowel_initial_root(concord, root, expected):
    assert join_concord(concord, root) == expected


def test_class_1_is_m_except_before_a_root_with_no_vowel():
    assert object_concord("1", "thand") == "m" and object_concord("1", "az") == "m" and object_concord("1", "bon") == "m"
    assert object_concord("1", "dl") == "mu" and object_concord("1", "ph") == "mu"
    assert [with_object(r, "1") for r in ("thand", "bon", "az", "enz", "dl", "ph")] == ["mthand", "mbon", "maz", "menz", "mudl", "muph"]


@pytest.mark.parametrize("root,target,subject,form,expected", [
    ("thand", "1", "1ps", "present_long", "ngiyamthanda"),            # ngiyamthanda (news)
    ("bon", "1", "1ps", "present_long", "ngiyambona"),                # ngiyambona
    ("dl", "1", "1ps", "present_long", "ngiyamudla"),                 # mudla (28 against 2)
    ("enz", "2ps", "1ps", "present_long", "ngiyakwenza"),
    ("az", "2ps", "1pp", "negative_present", "asikwazi"),             # asikwazi (corpus shape: 5,355 news words)
    ("az", "2ps", "2ps", "negative_present", "awukwazi"),             # awukwazi
    ("bon", "2ps", "1ps", "perfect", "ngikubonile"),
    ("bon", "2", "1pp", "future", "sizobabona"),
    ("qheny", "refl", "1pp", "present_long", "siyaziqhenya"),         # siyaziqhenya (news, 6,116 words of this shape)
    ("qheny", "refl", "2ps", "present_long", "uyaziqhenya"),          # uyaziqhenya
    ("phath", "refl", "1ps", "present_short", "ngiziphatha"),
])
def test_object_concord_fits_every_form_of_the_tense_module(root, target, subject, form, expected):
    # "az" is the verb azi (-i): the negative forms asikwazi / awukwazi take the ending -i, which negative_present supplies
    assert conjugate(with_object(root, target), subject, form) == expected


def test_person_and_class_tables_and_pronouns():
    assert PERSON_OBJECT == {"1ps": "ngi", "2ps": "ku", "1pp": "si", "2pp": "ni"}
    assert CLASS_OBJECT["2"] == "ba" and CLASS_OBJECT["3"] == "wu" and CLASS_OBJECT["9"] == "yi" and CLASS_OBJECT["15"] == "ku" and "1a" not in CLASS_OBJECT
    assert ABSOLUTE_PRONOUNS["1ps"] == "mina" and ABSOLUTE_PRONOUNS["9"] == "yona" and ABSOLUTE_PRONOUNS["15"] == "khona"
    assert object_concord("refl") == "zi"


def test_the_rendered_module_states_the_rules_with_their_counts():
    text = render()
    for needle in ("m- 229, mu- 10", "mudla (28 against 2)", "kwazi (7,282", "ukuzethemba", "5,355", "6,116", "Reflexive", "mina"):
        assert needle in text
