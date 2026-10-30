from mathula_tv.english_prefix import find_prefixed_english, find_soft_prefixed_english, fix_glossary, prefixed_words


def test_the_short_ma_on_an_english_word_the_speaker_says_is_a_hard_failure_and_ordinary_zulu_is_not():
    english = "Fighters, we want free water. The medical aid and the benefits are for the Zulu people, said Malema."
    zulu = "MaFighter, sifuna amanzi. Ne-medical aid, amaBenefits, isiZulu, uMalema, eLeopard, weScience."
    hard = find_prefixed_english(zulu, english)
    assert [h.split(":")[0] for h in hard] == ["MaFighter"]
    assert "short prefix ma-" in hard[0] and "'Fighters'" in hard[0] and "without the Zulu noun prefix" in hard[0]
    assert find_prefixed_english("MaFighter", "We want water") == []                       # the speaker did not say that word
    assert [h["token"] for h in prefixed_words("maFighters!", english)] == ["maFighters"]


def test_a_prefix_whose_number_contradicts_the_english_the_speaker_says_is_hard_and_one_that_agrees_is_only_a_flag():
    # real isiZulu text has iCovid, amaTechnikons, amaWorkshops, amaCouncilor: a prefix on an English noun is not wrong in itself
    plural, singular = "Fighters, we are with you.", "The fighter is here and Covid is over."
    assert [h.split(":")[0] for h in find_prefixed_english("iFighters", plural)] == ["iFighters"]        # a singular prefix on his plural
    assert [h.split(":")[0] for h in find_prefixed_english("amaFighter", singular)] == ["amaFighter"]     # a plural prefix on his singular
    assert find_prefixed_english("amaFighters", plural) == [] and find_prefixed_english("iFighter", singular) == []
    assert find_prefixed_english("iCovid", singular) == [] and find_prefixed_english("amaTechnikons", "The Technikons started") == []
    assert [s.split(":")[0] for s in find_soft_prefixed_english("amaFighters", plural)] == ["amaFighters"]
    assert find_soft_prefixed_english("MaFighter", plural) == []                       # a hard case is not also a flag
    both = "A fighter and the fighters"                                                # he says both numbers: nothing to contradict
    assert find_prefixed_english("iFighter amaFighters", both) == []


def test_a_singular_word_that_ends_in_s_is_not_taken_for_a_plural():
    assert find_prefixed_english("iBusiness", "The business is open.") == []
    assert [h.split(":")[0] for h in find_prefixed_english("iServices", "The services are open.")] == ["iServices"]


def test_the_glossary_term_and_the_register_instruction_lose_the_hard_form_only():
    glossary = {"terms": [{"english": "Fighters", "kind": "role", "zulu": "maFighter"}, {"english": "rally", "kind": "other", "zulu": "irali"},
                          {"english": "Technikons", "kind": "other", "zulu": "amaTechnikons"}],
                "do_not_translate": ["EFF"],
                "register": {"address": "usebenzise maFighter lapho ethi Fighters, nina lapho ethi you", "formality": "MaFighter kuphela"}}
    fixed, changes = fix_glossary(glossary)
    assert fixed["terms"][0]["zulu"] == "Fighters" and fixed["terms"][1]["zulu"] == "irali" and fixed["terms"][2]["zulu"] == "amaTechnikons"
    assert fixed["register"]["address"] == "usebenzise Fighters lapho ethi Fighters, nina lapho ethi you" and fixed["register"]["formality"] == "Fighters kuphela"
    assert len(changes) == 1 and "maFighter" in changes[0] and "ma-" in changes[0] and glossary["terms"][0]["zulu"] == "maFighter"      # the input is not changed
    clean, none = fix_glossary({"terms": [{"english": "grant", "kind": "other", "zulu": "igranti"}], "register": {}})
    assert none == [] and clean["terms"][0]["zulu"] == "igranti"
