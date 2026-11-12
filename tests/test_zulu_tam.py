import pytest

from mathula_tv.zulu_tam import FORMS, conjugate, negative_sc, past_sc, render


@pytest.mark.parametrize("root,subject,form,expected", [
    ("bon", "1pp", "present_long", "siyabona"),             # siyabona (corpus)
    ("hamb", "1ps", "present_short", "ngihamba"),
    ("landel", "5", "still", "lisalandela"),                 # lisalandela (corpus)
    ("qhubek", "15", "still", "kusaqhubeka"),               # kusaqhubeka (corpus)
    ("qal", "1pp", "future", "sizoqala"),                   # sizoqala (corpus)
    ("bhal", "1", "future", "uzobhala"),                    # uzobhala (corpus)
    ("qhubek", "15", "future", "kuzoqhubeka"),              # kuzoqhubeka (corpus)
    ("khokh", "2ps", "perfect", "ukhokhile"),               # ukhokhile (corpus)
    ("shiy", "1", "remote_past", "washiya"),                 # washiya (corpus)
    ("bek", "1", "potential", "angabeka"),                   # angabeka (corpus: class 1 takes a- after nga)
    ("khiph", "9", "potential", "ingakhipha"),              # ingakhipha (corpus)
    ("khulum", "6", "participial", "ekhuluma"),
    ("khulum", "2", "participial", "bekhuluma"),
    ("fak", "9", "negative_present", "ayifaki"),            # ayifaki (corpus)
    ("gqok", "9", "negative_present", "ayigqoki"),          # ayigqoki (corpus)
    ("hamb", "1ps", "negative_present", "angihambi"),
    ("hamb", "1", "negative_present", "akahambi"),
    ("phel", "11", "not_yet", "alukapheli"),                 # alukapheli (corpus)
    ("hamb", "1ps", "already", "sengihambile"),             # checked with ZulMorph
    ("hamb", "1", "already", "usehambile"),
    ("hamb", "2", "already", "sebehambile"),
    ("hamb", "15", "already", "sekuhambile"),
    ("hamb", "1ps", "ngeke", "ngeke ngihambe"),
    ("phind", "1", "ngeke", "ngeke aphinde"),               # corpus text: ngeke aphinde aphumelele
    ("hamb", "1ps", "negative_future", "angizuhamba"),
    ("hamb", "1pp", "subjunctive", "sihambe"),
    ("hamb", "1", "subjunctive", "ahambe"),
])
def test_conjugate_reproduces_attested_and_checked_forms(root, subject, form, expected):
    assert conjugate(root, subject, form) == expected


def test_remote_past_concords_fuse_with_a():
    assert {s: past_sc(s) for s in ("1ps", "1pp", "2pp", "2ps", "1", "2", "7", "9", "11", "15")} == {
        "1ps": "nga", "1pp": "sa", "2pp": "na", "2ps": "wa", "1": "wa", "2": "ba", "7": "sa", "9": "ya", "11": "lwa", "15": "kwa"}


def test_negative_concords():
    assert [negative_sc(s) for s in ("1ps", "2ps", "1", "3", "4", "9", "6", "2", "11")] == ["angi", "awu", "aka", "awu", "ayi", "ayi", "awa", "aba", "alu"]


def test_a_combination_without_a_checked_form_is_refused_not_guessed():
    with pytest.raises(ValueError):
        conjugate("hamb", "9", "already")
    with pytest.raises(ValueError):
        conjugate("hamb", "1ps", "no_such_form")


def test_every_form_has_a_builder_and_the_module_renders_them_all():
    for f in FORMS:
        conjugate("hamb", "1pp", f.key) if f.key != "already" else conjugate("hamb", "1pp", f.key)
    text = render()
    for f in FORMS:
        assert f.meaning in text and f.pattern in text
    assert "158 k" in text and "ngeke" in text
