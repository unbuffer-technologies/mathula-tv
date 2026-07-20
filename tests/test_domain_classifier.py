import pytest
from mathula_tv.domain_classifier import classify


@pytest.mark.parametrize(("text","expected"), [
    ("The witness testified at the Madlanga Commission hearing", "commission"),
    ("The party launched its campaign in Parliament", "politics"),
    ("Ward 12 candidates prepare for the municipal election", "local_elections"),
    ("Heavy rain caused traffic and a school closure", "general_news"),
    ("At the Commission, a party discussed a municipal election", "mixed"),
])
def test_domains(text, expected): assert classify(text)["primary_domain"] == expected


def test_politician_in_nonpolitical_story_is_not_automatically_political():
    assert classify("The minister attended a school sports event")["primary_domain"] == "general_news"


def test_party_names_preserved():
    assert classify("The ANC party began an election campaign")["entities"]["political_parties"] == ["ANC"]

