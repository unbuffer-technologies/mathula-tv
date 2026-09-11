"""Real, non-mocked tests against the actual bundled Autshumato English-isiZulu
lexicon (src/mathula_tv/data/autshumato_english_isizulu.txt) -- confirmed
2026-08-29 to be a real, freely-licensed (CC BY 2.5 South Africa), 6,241-entry
word list. Not mocked: the whole point is confirming the real bundled data file
loads and looks the way this module assumes it does.
"""
from __future__ import annotations

from mathula_tv import zulu_lexicon as zx


def test_lookup_finds_a_real_entry():
    assert "ungqongqoshe" in zx.lookup("minister")


def test_lookup_is_case_insensitive():
    assert zx.lookup("Minister") == zx.lookup("minister")


def test_lookup_returns_empty_for_a_word_not_in_the_lexicon():
    # Confirmed real gap: our own carefully-researched "ukuvimba" for
    # "intercepted" isn't in this general-purpose lexicon at all.
    assert zx.lookup("intercepted") == []


def test_lookup_returns_multiple_senses_when_the_lexicon_has_them():
    # "minister" has both a government and a religious sense in this data --
    # confirms the module doesn't silently collapse multiple real translations.
    assert len(zx.lookup("minister")) > 1


def test_relevant_entries_extracts_only_words_with_lexicon_hits():
    entries = zx.relevant_entries("The minister was told the dockets have been intercepted.")
    assert "minister" in entries
    assert "intercepted" not in entries  # confirmed not in this lexicon
    # 2026-08-30: closed-class function words are now skipped entirely, not a
    # loose "no requirement either way" -- see the stopword tests below for
    # the real defect this closes.
    assert "the" not in entries
    assert "was" not in entries
    assert "have" not in entries
    assert "been" not in entries


def test_relevant_entries_skips_stopwords_even_when_they_have_lexicon_hits():
    # Confirmed real defect 2026-08-30: a real production sentence's entire
    # 12-slot cap filled with "but"/"before"/"we"/"in"/"who"/"also"/"what"/
    # "has"/"said"/"here"/"it" -- all of which DO have Autshumato entries --
    # before ever reaching "interesting", the one word that actually mattered
    # (and IS in the lexicon, as "okunakisayo"). Mistranslated as a
    # positive-emotion verb ("-jabulisa") instead, for lack of this hint.
    entries = zx.relevant_entries(
        "But before we bring in Kenny Mapango, who's also been listening "
        "into what has been said here, it was interesting to hear."
    )
    assert "interesting" in entries
    assert entries["interesting"] == ["okunakisayo"]
    for stopword in ("but", "before", "we", "in", "who", "also", "what", "has", "here", "it", "was"):
        assert stopword not in entries


def test_relevant_entries_includes_supplementary_entries_not_in_base_lexicon():
    # Confirmed real gap 2026-08-30: the base Autshumato lexicon has no entry
    # for "compare"/"contrast" at all -- a real sentence ("...and he
    # contrasted it with...") was mistranslated as a broken English/isiZulu
    # code-switch ("waz compare") for lack of any vocabulary hint.
    assert zx.lookup("contrasted") == ["-qhathanisa"]
    entries = zx.relevant_entries("He contrasted the two reports carefully.")
    assert entries.get("contrasted") == ["-qhathanisa"]


def test_relevant_entries_caps_at_max_entries():
    text = " ".join(["minister"] * 3 + ["told"] * 3 + ["have"] * 3 + ["was"] * 3 + ["been"] * 3)
    entries = zx.relevant_entries(text, max_entries=2)
    assert len(entries) <= 2


def test_relevant_entries_returns_empty_dict_for_no_matches():
    assert zx.relevant_entries("Xyzabc qwerty123.") == {}
