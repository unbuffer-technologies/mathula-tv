from mathula_tv.attestation import Attestation
from mathula_tv.episode_scoring import find_fabricated_entities


def corpus():
    zulu, english = [], []
    for n in range(12):                                   # ikhomishana is how this corpus writes "commission"
        zulu.append(f"Ikhomishana iphenya icala {n} ngaphambi kokuba kuphume umbiko")
        english.append(f"The commission investigates the case {n} before the report is released")
    for n in range(10):                                   # isingisi: the language
        zulu.append(f"Lo mbhalo ubhalwe ngesingisi {n} nje")
        english.append(f"This document is written in English {n} only")
    for n in range(8):                                   # a real name that shares sentences with a frequent word
        zulu.append(f"URamaphosa ukhulume nabantu {n} namuhla")
        english.append(f"President Ramaphosa spoke to the people {n} today")
    for n in range(400):                                  # "president" and "the" are frequent everywhere
        zulu.append(f"Umongameli uthe okuthile {n}")
        english.append(f"The president said something {n}")
    return Attestation(zulu, english)


def test_a_capitalised_word_that_translates_a_word_of_the_english_is_not_a_fabricated_name():
    att = corpus()
    source = "The Commission heard that the statement was written in English."
    explains = lambda token: att.translates(token, source)                        # noqa: E731
    flagged = find_fabricated_entities("Wathi iKhomishana ithole isitatimende esibhalwe isiNgisi", {"wathi"}, explains)
    assert flagged == []                                                          # iKhomishana ~ Commission, isiNgisi ~ English
    assert set(find_fabricated_entities("Wathi iKhomishana ithole isitatimende esibhalwe isiNgisi", {"wathi"})) == {"iKhomishana", "isiNgisi"}          # before: both flagged


def test_an_invented_name_is_still_flagged_even_when_a_frequent_english_word_shares_its_sentences():
    att = corpus()
    explains = lambda token: att.translates(token, "The president said that the commission was ready")          # noqa: E731
    assert find_fabricated_entities("Wathi uRamaphosa kunjalo", {"wathi"}, explains) == ["uRamaphosa"]          # a real name, but the English never said it: still a fabrication
    assert find_fabricated_entities("Wathi uZwelakhe kunjalo", {"wathi"}, explains) == ["uZwelakhe"]            # unseen name


def test_translates_needs_enough_shared_sentences_and_a_long_enough_core():
    att = corpus()
    assert att.translates("iKhomishana", "The Commission") and att.translates("isiNgisi", "in English") and not att.translates("iKhomishana", "The weather")
    assert not att.translates("uMa", "The Commission")                            # a core under five letters is never looked up
    assert att.english_associates("khomishana").get("commission", 0) >= 0.25


def test_a_capital_after_a_pause_marker_is_a_sentence_start_and_not_a_name_once_the_markers_are_stripped():
    from mathula_tv.delivery import strip_pauses

    raw = "Bamkhiphe ejele. ~pause920 Ngoba ubengakwazi ukuphuma."
    assert find_fabricated_entities(raw, set()) == ["Ngoba"]                                     # the marker hides the full stop: what check_text used to hand the gate
    assert find_fabricated_entities(strip_pauses(raw), set()) == []                              # what the scorer hands it, and what check_text does now


def test_a_name_that_the_english_side_writes_too_is_never_excused_as_a_translation():
    att = corpus()
    assert not att.translates("uRamaphosa", "The president said that")                  # ramaphosa is on both sides of the corpus: a name, judged by the known tokens
    assert "ramaphosa" in att.english_associates("ramaphosa")


def test_a_locative_or_plural_form_is_excused_through_its_shortened_stem():
    att = corpus()
    assert att.translates("eKhomishaneni", "The Commission heard") and att.translates("kuKhomishana", "The Commission")
    assert not att.translates("eKhomishaneni", "The weather today")


def test_a_zulu_form_the_glossary_holds_counts_by_its_core_after_its_prefix():
    from mathula_tv.episode_scoring import _known_entity_tokens

    glossary = {"terms": [{"english": "Witness C", "zulu": "uFakazi C"}, {"english": "Commission", "zulu": "iKhomishana"}]}
    known = _known_entity_tokens("Witness C said", glossary, {})
    assert {"fakazi", "khomishana", "ufakazi"} <= known
    assert find_fabricated_entities("Wathi uFakazi C ukuthi iKhomishana ithole", known | {"wathi"}) == []
    assert find_fabricated_entities("Wathi uMpofu C", known | {"wathi"}) == ["uMpofu"]                    # a name the glossary does not hold is still caught
