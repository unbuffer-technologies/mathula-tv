import pytest

from mathula_tv import zulu_morphology, zulu_stems


def test_stems_are_read_from_the_analyses_and_two_forms_of_one_word_share_one():
    analyses = {"isu": ["i[NPrePre][5]li[BPre][5]su.5-6[NStem]"], "amasu": ["a[NPrePre][6]ma[BPre][6]su.5-6[NStem]", "a[NegPre]a[SCPT][6]masu[RelStem]"]}
    assert zulu_stems.extract_stems(analyses["amasu"]) == {"su", "masu"}
    assert zulu_stems.holds("isu", "nangamasu akhe", analyses | {"nangamasu": ["na[NPre]nga[AdvPre]a[NPrePre][6]ma[BPre][6]su.5-6[NStem]"]})
    assert not zulu_stems.holds("isu", "ngisuka lapho", analyses)          # the letters of another word are not the word


def test_a_phrase_needs_every_word_and_an_unparsed_word_is_its_own_stem():
    assert zulu_stems.holds("lelo su", "ephikisa lelo su", {})
    assert not zulu_stems.holds("lelo su", "ephikisa lelo", {})
    assert not zulu_stems.holds("", "anything", {})
    assert zulu_stems.words_of("ama-details, Ngo-2017") == ["ama", "details", "ngo"]


@pytest.mark.skipif(not zulu_morphology.is_available(), reason="ZulMorph is not installed here")
def test_the_real_analyser_gives_isu_amasu_and_ngamasu_one_stem():
    found = zulu_stems.analyse(["isu", "amasu", "ngamasu", "ilungelo"])
    assert all("su" in zulu_stems.extract_stems(found[w]) for w in ("isu", "amasu", "ngamasu"))
    assert "lungelo" in zulu_stems.extract_stems(found["ilungelo"])


def test_status_says_how_the_analyser_is_reached_and_that_it_is_not(monkeypatch):
    monkeypatch.setattr(zulu_stems, "_in_process", lambda words: {"isu": ["x[NStem]"]})
    assert zulu_stems.status() == {"available": True, "via": "in_process"}
    monkeypatch.setattr(zulu_stems, "_in_process", lambda words: None)
    monkeypatch.setattr(zulu_stems, "_through_helper", lambda words: {"isu": ["i[NPrePre][5]li[BPre][5]su.5-6[NStem]"]})
    assert zulu_stems.status() == {"available": True, "via": "helper"}
    monkeypatch.setattr(zulu_stems, "_through_helper", lambda words: None)
    assert zulu_stems.status() == {"available": False, "via": None}
