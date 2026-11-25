from mathula_tv import hook_spelling as hs

DUB = ("Uma sesiqedile, sizobiza isithangami sabezindaba sinazise ngalokho okuvunyelwene ngakho. Konke sekulungisiwe manje. "
       "Sesilungele ukuqhubeka. Sihlangene namaqembu ezepolitiki. Amakhansela asunguliwe, futhi lokho kumele kwenzeke ezinsukwini eziyi-4.")


def test_a_hook_word_that_is_a_near_miss_of_a_dub_word_is_written_the_way_the_dub_writes_it():
    fixed, fixes = hs.correct("I-ANC ithi izobiza isithangazi sabezindaba: Kuthathwe ziphi izinqumo?", [DUB])
    assert fixed == "I-ANC ithi izobiza isithangami sabezindaba: Kuthathwe ziphi izinqumo?"
    assert [(f["hook"], f["dub"]) for f in fixes] == [("isithangazi", "isithangami")]


def test_a_different_word_is_never_turned_into_another_even_when_the_two_are_close():
    # izobiza (it will convene) and sizobiza (we will convene) are two words that differ at the front: not a spelling
    fixed, fixes = hs.correct("I-ANC izobiza isithangami", [DUB])
    assert fixed == "I-ANC izobiza isithangami" and fixes == []
    assert hs.correct("Amaqembu azohlangana", [DUB])[1] == []                       # nothing near it in the dub


def test_capitals_are_kept_words_the_dub_says_are_left_alone_and_short_words_and_acronyms_are_not_touched():
    fixed, fixes = hs.correct("Isithangazi sabezindaba i-ANC ithi", [DUB])
    assert fixed == "Isithangami sabezindaba i-ANC ithi" and fixes[0]["dub"] == "Isithangami"
    assert hs.correct("isithangami sabezindaba", [DUB]) == ("isithangami sabezindaba", [])
    assert hs.correct("sithi ithi", [DUB]) == ("sithi ithi", [])


def test_the_words_of_the_settled_terms_count_as_the_dubs_words():
    terms = [{"english": "press briefing", "zulu": "isikhulumo sabezindaba"}]
    fixed, fixes = hs.correct("Isikhulumi sabezindaba", ["Sawubona"], terms)
    assert fixed == "Isikhulumo sabezindaba" and fixes[0]["dub"] == "Isikhulumo"       # the settled term's word is the dub's word, and the near miss is written as it is
    assert hs.vocabulary(["Sawubona baba"], terms) >= {"sawubona", "baba", "isikhulumo", "sabezindaba"}
