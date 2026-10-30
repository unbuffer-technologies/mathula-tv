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
    assert a.equivalents("unheard") == {"english": "unheard", "sentences": 0, "forms": [], "kept_in_english": 0, "with_particle": []}


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


def test_a_prefixed_place_and_an_everyday_word_are_ordinary_zulu_but_a_rare_capitalised_name_is_not():
    zulu = (["Siya eMpumalanga namuhla ."] * 25 + ["Imali iyanda ."] * 2 + ["Sinemali : Imali yethu ."] * 3 + ["Ngitshela abantu ."] * 120
            + ["Imali ayibonwa ."] * 60 + ["Wathenga imali futhi ."] * 130 + ["Ngitshela ukuthi iTshela Thupa ifikile ."] * 2)
    english = ["x ."] * len(zulu)
    a = Attestation(zulu, english)
    assert a.written_count("eMpumalanga") == 25 and a.ordinary_zulu("eMpumalanga") and not a.ordinary_zulu("eGauteng")
    assert a.written_count("Imali") == 3 and a.written_count("imali") >= 130                              # a capital that starts a line proves nothing
    assert a.ordinary_zulu("Imali")                                                                       # an everyday word capitalised by its place in the sentence
    assert not a.ordinary_zulu("Tshela") and not a.ordinary_zulu("Fighters")                             # a name real text does not write that way


def test_a_phrase_and_a_form_with_a_hyphenated_particle_are_counted_exactly_and_word_by_word():
    zulu = (["Wathola isikhwama sempesheni namuhla ."] * 4 + ["Sebenzisa isikhwama futhi ."] * 3 + ["Ibhalwe ngu-pension kanye se-provident ."] * 2
            + ["Sithenga sempesheni ."] * 2)
    a = Attestation(zulu, ["x ."] * len(zulu), ["umhlomulo"])
    phrase = a.status("isikhwama sempesheni")
    assert phrase["corpus_count"] == 4 and [w["corpus_count"] for w in phrase["words"]] == [7, 6] and phrase["all_words_attested"] is True
    assert a.attested("isikhwama sempesheni") and a.attested("Isikhwama  Sempesheni")                    # spacing and capitals do not matter
    new = a.status("isikhwama sepension")                                                                 # a new combination: the parts are real, the phrase is not
    assert new["corpus_count"] == 0 and new["all_words_attested"] is False and a.attested("isikhwama sepension") is False
    particle = a.status("se-provident")
    assert particle["corpus_count"] == 2 and [w["token"] for w in particle["words"]] == ["se", "provident"] and a.attested("se-provident")
    assert a.status("sempesheni")["corpus_count"] == 6 and "words" not in a.status("sempesheni")          # a single form is as before
    assert a.attested("umhlomulo") and not a.attested("")


def test_the_lookup_shows_how_real_writers_attach_a_particle_to_an_english_word_and_cautions_about_a_small_corpus():
    zulu = ["Behlangene kwi-rally kusizwe .", "Beya kwi-rally ngoLwesihlanu .", "I-rally yeqembu ifikile .", "Babuthane embuthanweni ."]
    english = ["They met at the rally today .", "They went to the rally on Friday .", "The party rally has arrived .", "They gathered at the meeting ."]
    a = Attestation(zulu, english)
    answer = lookup(a, "rally", ["irally", "kwi-rally"])
    assert answer["sentences"] == 3 and answer["kept_in_english"] == 3
    assert answer["with_particle"][0] == {"form": "kwi-rally", "count": 2}
    assert "kwi-rally (2)" in answer["reading"] and "weak evidence" in answer["caution"] and "only 3 sentences" in answer["caution"]
    assert [c["corpus_count"] for c in answer["checked_forms"]] == [0, 2]                              # irally is not written, kwi-rally is
