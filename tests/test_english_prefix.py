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


def _attestation_with_hawks():
    from mathula_tv.attestation import Attestation, WebText

    counts = {"ama-hawks": (6, 8), "amahawks": (5, 2), "wamahawks": (3, 0), "kumahawks": (1, 0), "tomahawks": (0, 35), "hawks": (18, 41)}
    return Attestation(["x"], ["y"], [], WebText(counts))


def test_a_prefix_form_real_zulu_text_never_writes_is_flagged_with_the_forms_it_does_write():
    from mathula_tv.english_prefix import find_unattested_prefixed

    att = _attestation_with_hawks()
    flags = find_unattested_prefixed("wayengaphansi kwabe-Hawks, owanikwa umsebenzi", "he was working under the Hawks", att)
    assert len(flags) == 1 and flags[0].startswith("Hawks: you wrote kwabe-Hawks") and "ama-Hawks (11 news, 10 other)" in flags[0] and "wama-Hawks" in flags[0]
    assert "can still be right" in flags[0]                                         # a reminder to compare, not a verdict
    two = find_unattested_prefixed("kwabe-Hawks nabe-Hawks", "the Hawks said", att)
    assert len(two) == 1 and "kwabe-Hawks, nabe-Hawks" in two[0]                       # one line per word, however many forms of it
    assert "toma" not in flags[0]                                                   # a compound that merely ends in hawks, seen only outside the news, is not what Zulu glues on


def test_an_attested_form_a_word_the_speaker_does_not_say_and_a_word_with_nothing_to_compare_are_left_alone():
    from mathula_tv.english_prefix import find_unattested_prefixed

    att = _attestation_with_hawks()
    assert find_unattested_prefixed("bonke ama-Hawks nabo wamaHawks", "the Hawks said", att) == []          # both forms are written (hyphen or not)
    assert find_unattested_prefixed("kwabe-Hawks", "the police said", att) == []                       # the speaker never says Hawks
    assert find_unattested_prefixed("u-Phineas", "Phineas Manyama said", att) == []                    # the corpora never write Phineas with a prefix: nothing to compare with
    assert find_unattested_prefixed("kwabe-Hawks", "the Hawks said", None) == []                        # no corpus installed
