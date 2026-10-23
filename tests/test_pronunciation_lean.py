from mathula_tv.pronunciation_lean import lean_reading, verdict
from mathula_tv.pronunciation_search import looks_isizulu

TOKENS = {"dʒeɪkob": ["dʒ", "eɪ", "k", "o", "b"], "zuma": ["z", "u", "m", "a"], "məʊzəz": ["m", "əʊ", "z", "ə", "z"]}


def test_the_lean_reading_leaves_isizulu_words_and_acronyms_alone_and_spells_the_rest_by_rule():
    reading = lean_reading("Jacob Zuma", ["dʒeɪkob", "zuma"], ["Jeyikab", "Zuma"], lambda ipa: TOKENS[ipa], looks_isizulu=looks_isizulu)
    assert [w["lean"] for w in reading] == ["Jeykhob", "Zuma"]
    assert reading[1]["why"].startswith("isiZulu shape")
    acronym = lean_reading("SABC", ["ɛs"], ["Es Ey Bii Sii"], lambda ipa: ["ɛ", "s"], looks_isizulu=looks_isizulu)
    assert acronym[0]["lean"] == "Es Ey Bii Sii" and acronym[0]["why"].startswith("acronym")


def test_the_lean_reading_does_not_use_the_research_respelling_for_an_english_word():
    reading = lean_reading("Moses", ["məʊzəz"], ["Mowziz"], lambda ipa: TOKENS[ipa], looks_isizulu=looks_isizulu)
    assert reading[0]["lean"] != "Mowziz" and reading[0]["why"] == "rule from the speaker's sounds"


def test_a_verdict_needs_a_clear_gap():
    assert verdict(0.3, 0.5) == "lean better" and verdict(0.5, 0.3) == "current better"
    assert verdict(0.40, 0.45) == "about equal"
