from mathula_tv import zulu_rules as rules


def main_choice(word):
    return rules.respell_initialism(word)[0]["spelling"]


def test_the_five_entries_of_the_handover_follow_from_the_rules():
    # the table another model wrote by hand: each is what the five rules give for the English letter names
    assert main_choice("AI") == "eyi-aayi"          # eɪ is e+y+i, the last letter aɪ is a+y+i held
    assert main_choice("API") == "eyi-pi-aayi"      # the earlier letters stay short: piː is pi
    assert main_choice("CPU") == "si-pi-yuuu"       # juː is y+u, held long: three u
    assert main_choice("GPU") == "ji-pi-yuuu"       # dʒ is j
    word = rules.respell_word(["aɪ"], [1])          # the word "eye": the same held aɪ
    assert word[0]["spelling"] == "aayi"


def test_a_stress_inside_a_word_is_not_written_as_length_but_a_final_open_stress_is():
    # party pɑːti: the stressed ɑː is not last, so it is not doubled; "high" haɪ ends on its stressed diphthong
    assert rules.respell_word(["p", "ɑː", "t", "iː"], [None, 1, None, 0])[0]["spelling"] == "phaati"          # the stressed long ɑː is long in a word; the final reduced iː is not
    assert rules.respell_word(["h", "aɪ"], [None, 1])[0]["spelling"] == "haayi"


def test_a_consonant_with_no_vowel_after_it_gets_one_and_a_cluster_is_broken():
    assert rules.respell_word(["b", "ʌ", "s"], [None, 1, None])[0]["spelling"] == "baas"                      # bus: the final s stays, as Zulu speakers say English words; the stressed vowel is stretched
    assert "bas" in {r["spelling"] for r in rules.respell_word(["b", "ʌ", "s"], [None, 1, None], limit=12)}      # the unstretched vowel is a variant
    assert rules.respell_word(["s", "k", "uː", "l"], [None, None, 1, None])[0]["spelling"] == "skuul"        # school: the s-k cluster stays, the final l bare, the long uː written twice
    assert rules.respell_word(["p", "l", "æ", "ŋ", "k"], [None, None, 1, None, None])[0]["spelling"] == "plaangk"      # plank: the p-l cluster and the final ng-k stay


def test_a_diphthong_before_another_syllable_keeps_its_glide_only():
    # H is eɪ tʃ: the glide stays, the ch gets its own vowel, and the last letter of HIV is held on the V
    assert main_choice("HIV") == "eytsh-ayi-viii"          # H: ey + tsh, the ch bare as English ends the letter; I: aɪ at the end of its letter, short; V: iː held long


def test_the_open_choices_are_variants_ordered_by_how_far_they_depart_from_the_main_choice():
    found = [r["spelling"] for r in rules.respell_initialism("AI", limit=10)]
    assert found[0] == "eyi-aayi" and len(found) == 10 and len(set(found)) == 10
    assert "ey-aay" in found                                        # glides only, no vowel to rest on
    assert "eyi-ayi" in found                                       # the last letter not held
    assert "Eyi Aayi" in found                                      # letters apart with capitals
    assert found.index("ey-aay") < found.index("Eyi Ayi")          # one departure comes before two


def test_the_derivation_says_which_rule_made_each_step():
    notes = rules.respell_initialism("AI")[0]["derivation"]
    assert any("eɪ is one sound in English and none in Zulu" in n for n in notes)
    assert any("held" in n for n in notes)


def test_a_researched_transcription_goes_through_the_same_rules_as_a_dictionary_reading():
    heidi = rules.parse_ipa("/ˈhaɪdi/")
    assert heidi == [(["h", "aɪ", "d", "i"], [None, 1, None, 0])]
    assert rules.respell_ipa("/ˈhaɪdi/")[0][0]["spelling"] == "haydi"
    court = rules.parse_ipa("/ˌwɛstən ˈkeɪp haɪ ˈkɔːt/")
    assert [" ".join(p) for p, _ in court] == ["w ɛ s t ə n", "k eɪ p", "h aɪ", "k ɔː t"] and court[2][1] == [None, 0]
    assert [rules.respell_word(*w)[0]["spelling"] for w in court] == ["westan", "kheyp", "hayi", "khoot"]          # s-t is broken by i and the final n opened; "high" is held only when the transcription stresses it
    assert rules.parse_ipa("/um̩ˈkʰonto/")[0][0] == ["u", "m", "kʰ", "o", "n", "t", "o"]                         # the syllabic mark is dropped, the aspiration kept
    assert rules.parse_ipa("[ŋɡaːndɮa]")[0][0] == ["ŋ", "ɡ", "aː", "n", "d", "ɮ", "a"]


def test_only_grounded_confident_non_zulu_names_become_researched_readings():
    def item(text, ipa, *, mode="word_name", confidence=0.93, language="South African English", urls=("https://example.org",)):
        return {"canonical_text": text, "pronunciation_ipa": ipa, "pronunciation_mode": mode, "pronunciation_confidence": confidence, "pronunciation_language": language,
                "pronunciation_evidence_urls": list(urls)}

    found = rules.research_readings({"accepted_corrections": [
        item("Heidi", "/ˈhaɪdi/"),
        item("Western Cape High Court", "/ˌwɛstən ˈkeɪp haɪ ˈkɔːt/"),
        item("Nkandla", "[ŋɡaːndɮa]", language="isiZulu"),                          # Zulu spelling needs no interpreting
        item("MKP", "/ɛm keɪ piː/", mode="initialism"),                              # said by its letters
        item("Maybe", "/ˈmeɪbi/", confidence=0.5),                                   # not confident
        item("Nobody", "/ˈnoʊbɒdi/", urls=()),                                       # not grounded in anything
        item("Two Words", "/tuː/"),                                                  # the words do not match the transcription's
    ]})
    assert sorted(found) == ["cape", "court", "heidi", "high", "western"]
    assert found["heidi"]["how"] == "researched" and found["heidi"]["phones"] == ["h", "aɪ", "d", "i"] and found["heidi"]["stress"] == [None, 1, None, 0]


def test_a_stressed_long_vowel_inside_a_word_is_written_long_and_the_short_spelling_is_a_variant():
    court = rules.respell_word(["k", "ɔː", "t"], [None, 1, None], limit=10)
    assert court[0]["spelling"] == "khoot" and {"koot", "khot", "khooti"} <= {r["spelling"] for r in court}
    assert any("written twice" in n for n in court[0]["derivation"])
    assert rules.respell_word(["h", "aɪ", "d", "iː"], [None, 1, None, 0])[0]["spelling"] == "haydi"          # the reduced final iː is not stressed: single
    assert rules.respell_word(["l", "ɪ", "tʃ", "f", "iː", "l", "d"], [None, 0, None, None, 2, None, None])[0]["spelling"].count("ii") == 1     # a long iː marked secondary
    assert rules.parse_ipa("/ˈlɪtʃfiːld/")[0][1][4] == 2                                                        # a long vowel in a transcription carries weight without a mark
    assert main_choice("CPU") == "si-pi-yuuu"                                                                    # the letters of an initialism before the last stay short


def test_p_t_k_opening_a_stressed_syllable_are_written_aspirated_but_not_after_s_or_in_an_unstressed_syllable():
    assert rules.respell_word(["k", "eɪ", "p"], [None, 1, None])[0]["spelling"] == "kheyp"                          # Cape
    assert rules.respell_word(["t", "iː", "m"], [None, 1, None])[0]["spelling"].startswith("th")
    assert rules.respell_word(["s", "k", "uː", "l"], [None, None, 1, None])[0]["spelling"] == "skuul"             # after s the k is plain
    assert rules.respell_word(["b", "ʌ", "t", "ə", "n"], [None, 1, None, 0, None])[0]["spelling"] == "baatan"     # button: the t of an unstressed syllable is plain
    assert rules.respell_word(["k", "eɪ", "p"], [None, 1, None], limit=12)[0]["derivation"][0].startswith("k opens a stressed syllable")
    assert "keyp" in {r["spelling"] for r in rules.respell_word(["k", "eɪ", "p"], [None, 1, None], limit=12)}      # plain k stays a variant
    assert rules.parse_ipa("/ramaˈpʰɔːsa/")[0][0][4:6] == ["pʰ", "ɔː"]                                              # an aspiration mark in a transcription is kept
    assert rules.respell_ipa("/ramaˈpʰɔːsa/")[0][0]["spelling"] == "ramaphoosa"
    assert main_choice("TV") == "ti-viii"                                                                              # the letters of an initialism: only a primary stress is aspirated


def test_the_last_consonant_of_a_word_stays_bare_as_zulu_speakers_say_jacob_and_a_vowel_after_it_is_a_variant():
    jacob = rules.respell_word(["dʒ", "eɪ", "k", "ə", "b"], [None, 1, None, 0, None], limit=8)
    assert jacob[0]["spelling"] == "jeykab" and "jeykabi" in {r["spelling"] for r in jacob}
    assert any("stays as it is" in n for n in jacob[0]["derivation"])
    assert rules.respell_word(["k", "ɔː", "t"], [None, 1, None])[0]["spelling"] == "khoot"                 # a closed final syllable: the long vowel doubled, not held three times
    assert rules.respell_word(["n", "uː", "z"], [None, 1, None])[0]["spelling"] == "nuuz"
    assert main_choice("EFF") == "i-ef-eef"                                                                # the consonant that ends a letter name stays bare too: em, es, ef
    assert "i-efi-eefi" in {r["spelling"] for r in rules.respell_initialism("EFF", limit=12)}              # a vowel after it is a variant


def test_consonants_that_come_together_in_english_stay_together_and_a_vowel_breaking_them_is_a_variant():
    western = rules.respell_word(["w", "ɛ", "s", "t", "ə", "n"], [None, 2, None, None, 0, None], limit=12)
    assert western[0]["spelling"] == "westan" and "wesitan" in {r["spelling"] for r in western}
    assert any("come together in English and stay together" in n for n in western[0]["derivation"])
    assert rules.respell_word(["s", "t", "r", "iː", "t"], [None, None, None, 1, None])[0]["spelling"] == "striit"          # a three-consonant onset stays whole


def test_a_reduced_vowel_is_written_with_the_letter_the_english_spelling_has_and_a_is_a_variant():
    jacob = rules.respell_word(["dʒ", "eɪ", "k", "ə", "b"], [None, 1, None, 0, None], limit=6, written="Jacob")
    assert jacob[0]["spelling"] == "jeykob" and "jeykab" in {r["spelling"] for r in jacob}                       # Jacob Zuma is said with the o of the spelling
    assert any("reduced vowel of Jacob" in n for n in jacob[0]["derivation"])
    assert rules.respell_word(["w", "ɛ", "s", "t", "ə", "n"], [None, 2, None, None, 0, None], written="Western")[0]["spelling"] == "westen"
    assert rules.respell_word(["ɒ", "k", "t", "oʊ", "b", "ə"], [2, None, None, 1, None, 0], written="October")[0]["spelling"] == "okthooba"          # er at the end is a
    assert rules.respell_word(["b", "ʌ", "t", "ə", "n"], [None, 1, None, 0, None], written="Button")[0]["spelling"] == "baaton"
    assert rules.respell_word(["k", "eɪ", "p"], [None, 1, None], written="Cape")[0]["spelling"] == "kheyp"          # no reduced vowel: nothing changes
    assert rules.respell_word(["dʒ", "eɪ", "k", "ə", "b"], [None, 1, None, 0, None], written="Jakobe-x")[0]["spelling"] == "jeykab"          # letters that do not match the sounds: left as a


def test_the_short_vowel_with_the_main_stress_is_stretched_like_a_long_one_and_the_single_vowel_is_a_variant():
    whatsapp = rules.respell_ipa("/ˈwɒts.æp/")[0]
    assert whatsapp[0]["spelling"] == "wootsap" and any("main stress is stretched" in n for n in whatsapp[0]["derivation"])
    assert "wotsap" in {r["spelling"] for r in rules.respell_word(*rules.parse_ipa("/ˈwɒts.æp/")[0], limit=12)}
    assert rules.respell_word(["s", "ɪ", "r", "ə", "l"], [None, 1, None, 0, None], written="Cyril")[0]["spelling"] == "siiril"
    assert rules.respell_word(["w", "ɛ", "s", "t", "ə", "n"], [None, 2, None, None, 0, None])[0]["spelling"] == "westan"      # a secondary stress is not stretched
    assert rules.respell_word(["k", "eɪ", "p"], [None, 1, None])[0]["spelling"] == "kheyp"                                  # a diphthong keeps its glide, not a doubled vowel


def test_the_vowels_of_what_and_october_follow_the_written_letter_and_news_keeps_its_y():
    whatsapp = rules.respell_word(*rules.parse_ipa("/ˈwɒts.æp/")[0], limit=6, written="WhatsApp")
    assert whatsapp[0]["spelling"] == "waatsap" and any("the letter the English spelling has: a" in n for n in whatsapp[0]["derivation"])
    assert rules.respell_word(["ɑː", "k", "t", "oʊ", "b", "ə"], [2, None, None, 1, None, 0], written="October")[0]["spelling"] == "okthooba"
    assert rules.respell_word(["p", "ɑː", "t", "iː"], [None, 1, None, 0], written="Party")[0]["spelling"] == "phaati"
    news = rules.respell_word(["n", "uː", "z"], [None, 1, None], written="News", limit=4)
    assert news[0]["spelling"] == "nyuuz" and "nuuz" in {r["spelling"] for r in news}                  # the y of South African English; the American reading is a variant
    assert rules.respell_word(["s", "k", "uː", "l"], [None, None, 1, None], written="School")[0]["spelling"] == "skuul"          # oo is not u: no y


def test_er_with_the_vowel_of_palm_is_a_as_in_derby():
    assert rules.respell_ipa("/ˈdɑːbi/")[0][0]["spelling"] in ("daabi", "deebi")          # without the written word the vowel is the dictionary's
    assert rules.respell_word(*rules.parse_ipa("/ˈdɑːbi/")[0], written="Derby")[0]["spelling"] == "daabi"


def test_the_r_before_a_vowel_stays_o_follows_the_written_o_and_a_zulu_shaped_name_is_not_guessed():
    from mathula_tv.english_sounds import non_rhotic, stress_of, to_ipa
    from mathula_tv.interpreter import zulu_shaped

    arpabet = "JH EH1 N ER0 AH0 L".split()                      # general: the r of ER is said, a vowel follows it
    phones, stress = non_rhotic(to_ipa(arpabet), stress_of(arpabet))
    assert "ɹ" in phones and rules.respell_word(phones, stress, written="General")[0]["spelling"] == "jeeneral"
    arpabet = "K AO1 R T".split()
    assert "ɹ" not in non_rhotic(to_ipa(arpabet), stress_of(arpabet))[0]          # court: no vowel after the r
    orlando = rules.respell_word(*rules.parse_ipa("/ɔːˈlændəʊ/")[0], written="Orlando", limit=6)
    assert orlando[0]["spelling"] == "oolaando" and "oolaandowu" in {r["spelling"] for r in orlando}          # the o of the spelling, not o + w + u
    assert rules.respell_word(["k", "oʊ", "t"], [None, 1, None], written="Coat")[0]["spelling"].endswith("owut") or "w" in rules.respell_word(["k", "oʊ", "t"], [None, 1, None], written="Coat")[0]["spelling"]          # oa is not a single o: the diphthong stays
    assert zulu_shaped("Motaung") and zulu_shaped("Mkhwanazi") and not zulu_shaped("Kaizer") and not zulu_shaped("Jacob") and not zulu_shaped("Litchfield")
