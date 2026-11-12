import pytest

from mathula_tv.zulu_extensions import EXTENSIONS, FORMS, STACKS, extend, render, soften


@pytest.mark.parametrize("root,exts,expected", [
    ("hamb", ["caus"], "hambisa"),
    ("hamb", ["appl"], "hambela"),
    ("hamb", ["recip"], "hambana"),
    ("hamb", ["neut"], "hambeka"),
    ("bon", ["neut"], "bonakala"),                           # kubonakala (news, 883): -akal-, not boneka (0)
    ("thol", ["neut"], "tholakala"),
    ("enz", ["neut"], "enzeka"),                             # kwenzeka (17,585): -ek- is the default
    ("qhub", ["neut"], "qhubeka"),
    ("ph", ["pass"], "phiwa"),                               # a root with no vowel: -iw-, no softening (baphiwe, uphiwe in the news)
    ("dl", ["pass"], "dliwa"),                               # kudliwa (74), ukudliwa (91)
    ("sebenz", ["caus"], "sebenzisa"),                       # sebenzisa (corpus)
    ("hamb", ["pass"], "hanjwa"),
    ("bamb", ["pass"], "banjwa"),                            # ubanjwa (news)
    ("khulum", ["pass"], "khulunywa"),                       # kukhulunywa (news, 830)
    ("bop", ["pass"], "boshwa"),                             # kuboshwa (news)
    ("gweb", ["pass"], "gwetshwa"),                          # ugwetshwe (news)
    ("gubh", ["pass"], "gujwa"),                             # kugujwa (news)
    ("thum", ["appl", "pass"], "thunyelwa"),                 # sithunyelwe (news, 487)
    ("vum", ["appl", "pass"], "vunyelwa"),                   # kuvunyelwe (news)
    ("hamb", ["caus", "pass"], "hanjiswa"),                  # ukuhanjiswa (news)
    ("khomb", ["caus", "pass"], "khonjiswa"),                # ukukhonjiswa (news)
    ("bamb", ["appl", "pass"], "banjelwa"),                  # obubanjelwe (news)
    ("xhum", ["recip", "caus", "pass"], "xhunyaniswa"),      # ukuxhunyaniswa (corpus)
    ("fak", ["appl", "pass"], "fakelwa"),                    # kufakelwa (corpus)
    ("enz", ["pass"], "enziwa"),                             # kwenziwa (news)
    ("akh", ["pass"], "akhiwa"),                             # kwakhiwa (corpus)
    ("az", ["pass"], "aziwa"),                               # akwaziwa (corpus)
    ("fund", ["caus", "appl"], "fundisela"),
    ("hamb", ["caus", "recip"], "hambisana"),                # ukuhambisana (corpus)
])
def test_extend_reproduces_attested_words(root, exts, expected):
    assert extend(root, exts) == expected


def test_the_reciprocal_perfect_ends_in_ene():
    assert extend("ahluk", ["recip"], ending="ile") == "ahlukene"      # ahlukene (corpus)
    assert extend("bhek", ["recip"], ending="ile") == "bhekene"        # ibhekene (corpus)


def test_softening_only_touches_a_final_labial():
    assert [soften(r) for r in ("bop", "khulum", "bamb", "gweb", "gubh", "fund", "bon", "thand")] == ["bosh", "khuluny", "banj", "gwetsh", "guj", "fund", "bon", "thand"]
    assert soften("hleph") == "hlesh"


def test_an_unknown_extension_is_refused():
    with pytest.raises(ValueError):
        extend("hamb", ["no_such"])


def test_the_rendered_module_carries_the_forms_the_order_and_the_sound_changes():
    text = render()
    for e in EXTENSIONS:
        assert e.meaning in text and e.english in text
    for needle in ("caus + pass", "appl + pass", "vunyelwa", "khulunywa", "2,697", "ahlukene", "enziwa", "length lever"):
        assert needle in text
    assert set(FORMS) == {"caus", "appl", "recip", "neut", "intens"} and len(STACKS) >= 6
