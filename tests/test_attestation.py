from mathula_tv.attestation import Attestation, lookup, tokens


def small():
    english = ["The computer is new .", "She bought a computer today .", "Two computers were sold .", "The bus is late .", "The bus stops here .",
               "He took the bus to town .", "We want free water .", "The Covid rules changed .", "Covid is over .", "Covid cases fell ."]
    zulu = ["Ikhompyutha intsha .", "Wathenga ikhompyutha namuhla .", "Amakhompyutha amabili athengiswe .", "Ibhasi sekwephuzile .", "Ibhasi iyama lapha .",
            "Wenyuka ibhasi waya edolobheni .", "Sifuna amanzi mahhala .", "Imithetho ye-Covid iguqukile .", "I-Covid iphelile .", "Amacala e-Covid ehlile ."]
    return Attestation(zulu, english, ["imali", "abantu"])


def test_a_form_is_attested_when_real_text_writes_it_or_the_lexicon_has_it():
    a = small()
    assert a.status("Ibhasi") == {"token": "Ibhasi", "corpus_count": 3, "in_lexicon": False}
    assert a.attested("ibhasi") and a.attested("IMALI") and not a.attested("maFighter") and a.status("amafighter")["corpus_count"] == 0
    assert tokens("i-Tshela Thupa, iCovid-19") == ["i", "tshela", "thupa", "icovid"]                     # a hyphen splits: the particle is counted apart


def test_the_zulu_forms_that_share_the_most_sentences_with_an_english_word_are_its_equivalents():
    a = small()
    bus = a.equivalents("bus")
    assert bus["sentences"] == 3 and bus["forms"][0]["form"] == "ibhasi" and bus["forms"][0]["shared_sentences"] == 3 and bus["kept_in_english"] == 0
    computer = a.equivalents("computer")                                                              # the plural line counts as a sentence with the word
    assert computer["sentences"] == 3 and computer["forms"] == []                                      # but ikhompyutha is in only two of them: under MIN_PAIRS, not an equivalent
    assert a.equivalents("unheard") == {"english": "unheard", "sentences": 0, "forms": [], "kept_in_english": 0}


def test_an_english_word_that_stays_english_in_the_zulu_is_reported_as_code_switched():
    covid = small().equivalents("covid")
    assert covid["sentences"] == 3 and covid["kept_in_english"] == 3                                   # i-Covid, I-Covid, e-Covid: the word is there every time


def test_the_answer_for_the_agent_reads_in_plain_words_for_each_case():
    a = small()
    assert "ibhasi" in lookup(a, "bus", ["ibhasi", "maBus"])["reading"] and [c["corpus_count"] for c in lookup(a, "bus", ["ibhasi", "maBus"])["checked_forms"]] == [3, 0]
    assert "code-switching" in lookup(a, "covid")["reading"] or "stays English" in lookup(a, "covid")["reading"]
    assert "keep it as the speaker says it" in lookup(a, "fighters")["reading"]


def test_the_two_sides_must_be_aligned():
    import pytest

    with pytest.raises(ValueError):
        Attestation(["a"], ["a", "b"])
