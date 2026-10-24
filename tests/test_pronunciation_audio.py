from mathula_tv.pronunciation_audio import respell


def test_the_speakers_vowels_replace_the_written_ones_when_the_counts_match():
    assert respell("Moses", ["m", "oː", "z", "ə", "z"]) == "Moses"                     # o heard, weak vowel keeps the written e
    assert respell("Jacob", ["dʒ", "eɪ", "k", "o", "b"]) == "Jeykhob"                  # a -> ey, k aspirated, o heard
    assert respell("Party", ["p", "ɑː", "t", "i"]) == "Phaathi"                        # ar -> aa (no sounded r), final y -> i, p and t aspirated


def test_a_word_whose_heard_vowels_do_not_match_its_written_vowels_keeps_the_page():
    assert respell("District", ["iː", "s", "t", "iː", "k", "t", "ɛ"]) == "Distrikt"     # three heard for two written: page, with c -> k


def test_consonants_follow_isizulu_spelling():
    assert respell("Municipality", []) == "Munisiphalithi"       # soft c -> s, p and t aspirated, final y -> i
    assert respell("Moses", []) == "Moses"
    assert respell("Floyd", []) == "Floyd"                       # a multi-letter vowel spelling is never replaced
    assert respell("Matthew", []) == "Mathew"                    # doubled letters collapse
