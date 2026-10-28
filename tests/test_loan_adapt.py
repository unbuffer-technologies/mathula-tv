from mathula_tv import loan_adapt as la


def stems(phones, limit=40):
    return [item["stem"] for item in la.adapt(phones, limit=limit)]


def test_a_word_ending_in_a_consonant_gets_a_vowel_the_way_zulu_does_bus_becomes_ibhasi():
    out = la.adapt(["b", "ʌ", "s"])
    assert out[0]["stem"] == "bhasi" and out[0]["with_i"] == "ibhasi" and out[0]["syllables"] == 2


def test_a_round_vowel_before_the_end_gives_a_round_final_vowel_spoon_becomes_isipunu():
    found = stems(["s", "p", "uː", "n"])
    assert "sipunu" in found and "siphunu" in found                       # i after s (a cluster Zulu does not have), the final vowel copies the u
    assert la.adapt(["s", "p", "uː", "n"])[0]["with_i"] == "isiphunu"


def test_a_vowel_is_inserted_after_a_labial_as_u_and_a_final_g_takes_i_flag_becomes_ifulegi():
    found = stems(["f", "l", "æ", "ɡ"])
    assert "fulegi" in found and "fulagi" in found


def test_a_glide_is_enough_before_another_vowel_science_becomes_isayensi():
    found = stems(["s", "aɪ", "ə", "n", "s"])
    assert "sayensi" in found and "sayansi" in found
    assert not any("sayiensi" in stem or "sayiansi" in stem for stem in found)


def test_a_consonant_cluster_is_broken_by_a_vowel_but_a_nasal_and_a_homorganic_consonant_is_a_zulu_unit_technology_becomes_ithekhinoloji():
    found = stems(["t", "e", "k", "n", "o", "l", "o", "dʒ"])                # the speaker's own sounds for the word, as the phone recogniser reads them
    assert "thekhinoloji" in found and "thekhunoloji" in found            # k + n is not a Zulu unit: the vowel goes between, u first after a velar
    assert all(not stem.endswith(("k", "n", "l", "j")) for stem in found)  # nothing ends on a consonant
    assert stems(["m", "ɛ", "n", "t"])[0] == "menthi" and "menti" in stems(["m", "ɛ", "n", "t"])          # nt stays together; the word cannot end on t


def test_variants_say_what_they_change_and_the_ranking_puts_the_typical_letters_first():
    out = la.adapt(["p", "ɪ", "n"], limit=6)
    assert out[0]["stem"] == "phini" and out[0]["choices"] == []
    plain = next(item for item in out if item["stem"] == "pini")
    assert plain["choices"] == ["p written p"]


def test_nothing_is_made_from_sounds_zulu_has_no_letters_for_and_tones_and_length_marks_are_ignored():
    assert la.adapt([]) == [] and la.adapt(["?"]) == []
    assert stems(["b", "ʌ5", "sː"])[0] == "bhasi"
