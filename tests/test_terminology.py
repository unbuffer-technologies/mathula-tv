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
