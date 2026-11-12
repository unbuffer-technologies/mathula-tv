import pytest

from mathula_tv.zulu_relative import PERSON_RELATIVE, RELATIVE_CONCORD, SHAPES, rc, relative, render


@pytest.mark.parametrize("root,cls,form,expected", [
    ("sebenz", "2", "present", "abasebenza"),             # abasebenza (corpus)
    ("buy", "2", "present", "ababuya"),                    # ababuya (corpus)
    ("ding", "10", "present", "ezidinga"),                 # ezidinga (corpus)
    ("fun", "1", "present", "ofuna"),                      # ofuna (corpus)
    ("hlal", "1", "present", "ohlala"),                    # ohlala (corpus)
    ("lalel", "9", "present", "elalela"),                  # elalela (corpus)
    ("chaz", "7", "present", "esichaza"),                  # esichaza (corpus)
    ("bek", "15", "present", "okubeka"),                   # okubeka (corpus)
    ("landel", "9", "present_yo", "elandelayo"),           # elandelayo (corpus)
    ("landel", "15", "present_yo", "okulandelayo"),        # okulandelayo (corpus)
    ("landel", "10", "present_yo", "ezilandelayo"),        # ezilandelayo (corpus)
    ("landel", "7", "present_yo", "esilandelayo"),         # esilandelayo (corpus)
    ("landel", "2", "present_yo", "abalandelayo"),         # abalandelayo (corpus)
    ("sekel", "6", "present_yo", "asekelayo"),             # asekelayo (corpus)
    ("lung", "1", "perfect", "olungile"),                  # olungile (corpus)
    ("balulek", "9", "perfect", "ebalulekile"),            # ebalulekile (corpus)
    ("qal", "1", "future", "ozoqala"),                     # ozoqala (news)
    ("bhekan", "2", "future", "abazobhekana"),             # abazobhekana (news)
    ("sebenz", "2", "negative", "abangasebenzi"),          # abangasebenzi (news)
    ("sebenz", "10", "negative", "ezingasebenzi"),         # ezingasebenzi (news)
    ("hamb", "1", "negative", "ongahambi"),                # ongahambi (news)
    ("khul", "2", "still", "abasakhula"),                  # abasakhula (news)
    ("nqob", "1", "past", "owanqoba"),                     # owanqoba (corpus)
    ("hlab", "1", "past", "owahlaba"),                     # owahlaba (corpus)
])
def test_relative_builder_reproduces_attested_words(root, cls, form, expected):
    assert relative(root, cls, form) == expected


def test_a_clause_with_its_own_person_subject_fuses_the_relative_concord():
    assert relative("fun", "9", "present_yo", subject="1ps") == "engifunayo"                   # checked with ZulMorph
    assert relative("fun", "9", "present_yo", subject="1ps", oc="yi") == "engiyifunayo"
    assert relative("bon", "7", "present_yo", subject="1pp", oc="ba") == "esibabonayo"
    assert relative("khulum", "9", "perfect", subject="1ps") == "engikhulumile"
    assert PERSON_RELATIVE == {"1ps": "engi", "1pp": "esi", "2pp": "eni"} and rc("9", "2pp") == "eni"


def test_the_class_table_gives_one_relative_concord_per_class():
    assert RELATIVE_CONCORD["1"] == "o" and RELATIVE_CONCORD["2"] == "aba" and RELATIVE_CONCORD["15"] == "oku" and RELATIVE_CONCORD["9"] == "e"
    assert "1a" not in RELATIVE_CONCORD


def test_forms_without_a_checked_pattern_are_refused():
    with pytest.raises(ValueError):
        relative("hamb", "2", "past")
    with pytest.raises(ValueError):
        relative("hamb", "9", "past", subject="1ps")
    with pytest.raises(ValueError):
        relative("hamb", "9", "no_such_form")


def test_the_rendered_module_states_the_shapes_and_the_yo_rule():
    text = render()
    for s in SHAPES:
        assert s.meaning in text and s.pattern in text
    for needle in ("engiyifunayo", "-yo:", "ezilula", "ten times", "29%", "abakhulu", "checked for classes 1, 4, 9"):
        assert needle in text
