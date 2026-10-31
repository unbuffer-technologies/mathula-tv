from mathula_tv.terminology import Terminology, parse_text

ELECTION = """
<<<PAGE 17>>>
absentee voter
Afrikaans afwesige kieser
IsiZulu umvoti othumele ivoti
IsiXhosa x
candidate-centered ballot {directly
elected}
Afrikaans kandidaatsentriese
 stemming
IsiZulu ukuvotela ummelikhetho
 oyedwa ohlwini
Constitutional Court <RSA>
IsiZulu Inkantolo Yomthethosisekelo
"""

PARLIAMENT = """
<<<PAGE 20>>>
act n. <see also law; rule> {promulgated by parliament}
Afrikaans wet
IsiZulu umthetho {wephalamende}
precedent <see also create a precedent; set a
precedent>
IsiZulu inqubo ezolandelwa
budget v.
IsiZulu -bekela imali
Member of Parliament (MP)
IsiZulu Ilungu Lephalamende
"""


def test_election_layout_joins_wrapped_lines():
    entries = {e["english"]: e for e in parse_text(ELECTION, "election")}
    assert entries["absentee voter"]["zulu"] == ["umvoti othumele ivoti"]
    assert entries["candidate-centered ballot"]["context"] == ["directly elected"]
    assert entries["candidate-centered ballot"]["zulu"] == ["ukuvotela ummelikhetho oyedwa ohlwini"]
    assert entries["Constitutional Court"]["context"] == ["RSA"]


def test_parliament_layout_drops_notes_from_the_headword():
    entries = {e["english"]: e for e in parse_text(PARLIAMENT, "parliament")}
    assert entries["act"]["pos"] == ["n"] and entries["act"]["zulu"] == ["umthetho"]
    assert entries["precedent"]["zulu"] == ["inqubo ezolandelwa"]
    assert entries["budget"]["pos"] == ["v"]
    assert entries["Member of Parliament"]["english_synonyms"] == ["MP"]


def test_front_matter_is_not_an_entry():
    text = "<<<PAGE 3>>>\n5. Mabasa, N.A., Ms. National Language Service\nIsiZulu collaborators for the working meeting\n"
    assert parse_text(text, "election") == []


def test_lookup_and_find():
    terms = Terminology(parse_text(ELECTION + PARLIAMENT, "x"))
    assert terms.lookup("Absentee Voters")[0]["zulu"] == ["umvoti othumele ivoti"]
    assert terms.lookup("MP")[0]["english"] == "Member of Parliament"
    found = terms.find("The absentee voter asked the Constitutional Court about the act.")
    assert {f["term"] for f in found} >= {"absentee voter", "constitutional court"}


def _entry(english, zulu, **extra):
    return {"english": english, "english_synonyms": [], "zulu": zulu, "zulu_alternatives": [], "source": "x", "page": 1, **extra}


def test_assess_keeps_specific_terms_and_drops_common_words_the_translators_render_differently():
    from mathula_tv.terminology import assess

    pairs = [("The state of the economy is bad.", "Isimo komnotho sibi."),
             ("The state spoke.", "Uhulumeni ukhulumile."),
             ("A state visit.", "Ukuvakasha kukahulumeni."),
             ("State power", "Amandla kahulumeni"),
             ("The councillor spoke.", "Ikhansela likhulumile."),
             ("Councillors met.", "Amakhansela ahlangene.")]
    entries = [_entry("state", ["uMnyango Wezobulungiswa Nokuthuthukiswa Komthethosisekelo"]),
               _entry("state", ["isifunda"]),
               _entry("councillor", ["ikhansela"]),
               _entry("absentee voter", ["umvoti othumele ivoti"]),
               _entry("MP", ["Ilungu Lephalamende"]),
               _entry("president", ["umongameli"])]
    by = {(e["english"], e["zulu"][0]): e for e in assess(entries, pairs)}
    assert not by[("state", "uMnyango Wezobulungiswa Nokuthuthukiswa Komthethosisekelo")]["reliable"]   # a long official name for a one-word term
    assert not by[("state", "isifunda")]["reliable"]                                                      # common, and the translators wrote something else
    assert by[("councillor", "ikhansela")]["reliable"] and by[("councillor", "ikhansela")]["support"] == 2
    assert by[("absentee voter", "umvoti othumele ivoti")]["reliable"] and by[("MP", "Ilungu Lephalamende")]["reliable"]
    assert by[("president", "umongameli")]["reliable"]                                                    # never seen: rare, so a specialist word


def test_terminology_ignores_unreliable_entries_unless_asked():
    entries = [_entry("state", ["isifunda"], reliable=False), _entry("councillor", ["ikhansela"], reliable=True)]
    assert Terminology(entries).lookup("state") == []
    assert Terminology(entries, reliable_only=False).lookup("state")
    assert Terminology(entries).lookup("councillor")


def test_assess_does_not_trust_verb_stems_or_short_stems_inside_other_words():
    from mathula_tv.terminology import assess

    pairs = [("They create jobs.", "Badala imisebenzi."), ("We create peace.", "Sidala ukuthula."), ("Create more.", "Dala okuningi.")]
    entries = [_entry("create", ["-dala", "-akha"]), _entry("make", ["dala"])]
    by = {e["english"]: e for e in assess(entries, pairs)}
    assert not by["create"]["reliable"] and "verb stem" in by["create"]["why"]
    assert by["make"]["support"] == 0 or by["make"]["agreement"] in (None, 0.0)       # "dala" inside "badala" is not the word dala
