from mathula_tv.english_prefix import find_prefixed_english, fix_glossary, prefixed_words


def test_a_noun_prefix_on_an_english_word_the_speaker_says_is_found_and_ordinary_zulu_is_not():
    english = "Fighters, we want free water. The medical aid and the benefits are for the Zulu people, said Malema."
    hits = find_prefixed_english("MaFighter, sifuna amanzi. Ne-medical aid, amaBenefit, isiZulu, uMalema, eLeopard, weScience.", english)
    assert [h.split(":")[0] for h in hits] == ["MaFighter", "amaBenefit"]
    assert "write it as spoken" in hits[0] and "'Fighters'" in hits[0]
    assert prefixed_words("maFighters!", english) == [("maFighters", "Fighters")]
    assert find_prefixed_english("MaFighter", "We want water") == []                       # the speaker did not say that word


def test_the_glossary_term_and_the_register_instruction_lose_the_prefixed_form():
    glossary = {"terms": [{"english": "Fighters", "kind": "role", "zulu": "maFighter"}, {"english": "rally", "kind": "other", "zulu": "irali"}],
                "do_not_translate": ["EFF"],
                "register": {"address": "usebenzise maFighter lapho ethi Fighters, nina lapho ethi you", "formality": "MaFighter kuphela"}}
    fixed, changes = fix_glossary(glossary)
    assert fixed["terms"][0]["zulu"] == "Fighters" and fixed["terms"][1]["zulu"] == "irali"
    assert fixed["register"]["address"] == "usebenzise Fighters lapho ethi Fighters, nina lapho ethi you" and fixed["register"]["formality"] == "Fighters kuphela"
    assert len(changes) == 1 and "maFighter" in changes[0] and glossary["terms"][0]["zulu"] == "maFighter"      # the input is not changed
    clean, none = fix_glossary({"terms": [{"english": "grant", "kind": "other", "zulu": "igranti"}], "register": {}})
    assert none == [] and clean["terms"][0]["zulu"] == "igranti"
