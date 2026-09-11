"""Real, non-mocked tests against the actual ZulMorph FST (via the zulu-morphology
extra: foma-with-lib + zulmorph) -- confirmed 2026-08-29 to load and analyse
correctly on Windows, after finding the plain "foma" PyPI package has no Windows
binary at all. These tests exercise the real finite-state analyser, not a fake,
since the whole point of this module is a genuine mechanical check independent
of any LLM or mock's self-report.

Skipped entirely (not failed) when the optional dependency isn't installed --
this must never be a hard requirement for the rest of the test suite to run.
"""
from __future__ import annotations

import pytest

from mathula_tv import zulu_morphology as zm

pytestmark = pytest.mark.skipif(
    not zm.is_available(), reason="zulu-morphology extra (foma-with-lib + zulmorph) not installed"
)


def test_is_available_reports_true_when_the_extra_is_installed():
    assert zm.is_available() is True


def test_word_parses_confirms_a_real_verb_form():
    # "ngikwenze" = ngi[SC][1ps] + ku[OC][15] + enz[VRoot] + e[VTPerf] -- the exact
    # subject-concord + object-concord + verb-stem + perfective-"e" construction
    # confirmed against Halpert's ngabe counterfactual paper this same session.
    assert zm.word_parses("ngikwenze") is True


def test_word_parses_confirms_the_concord_bearing_focused_echo_case():
    # native_phrase_0126's real fix: "Sesine." (class-7 concord form of "four",
    # agreeing with "isigaba") is a genuine, well-formed isiZulu word.
    assert zm.word_parses("sesine") is True


def test_word_parses_is_case_insensitive():
    # Confirmed empirically: the FST's lexicon is lowercase-only, so a
    # sentence-initial capitalized word would otherwise produce a false negative.
    assert zm.word_parses("Sesine") is True
    assert zm.word_parses("SESINE") is True


def test_word_parses_confirms_the_new_glossary_term():
    # The "intercepted" -> "ukuvimba" glossary fix -- confirm it's a real word.
    assert zm.word_parses("ukuvimba") is True


def test_word_parses_flags_a_genuinely_invalid_string():
    assert zm.word_parses("gibberishxyz") is False


def test_word_parses_allows_a_known_unparsed_interjection():
    # Confirmed real gap in ZulMorph's own lexicon: pure interjections like "yebo"
    # (yes) don't parse at all, despite being unambiguously correct, common isiZulu
    # -- confirmed backchannel output in this session's own real production job.
    assert zm.word_parses("yebo") is True
    assert "yebo" in zm.KNOWN_UNPARSED_INTERJECTIONS


def test_word_parses_treats_empty_string_as_trivially_valid():
    assert zm.word_parses("") is True
    assert zm.word_parses("   ") is True


def test_unparsed_words_extracts_only_the_bad_word_from_real_text():
    # "Sine." (the real wrong-referent production bug) is a DIFFERENT real word
    # (parses as an ideophone/verb-object-concord fragment) -- confirming this
    # module cannot and does not claim to catch valid-but-wrong-meaning errors,
    # only outright invalid ones.
    assert zm.word_parses("sine") is True
    assert zm.unparsed_words("Sine.") == []
    assert zm.unparsed_words("Kwenzeka gibberishxyz manje.") == ["gibberishxyz"]


def test_unparsed_words_deduplicates_in_first_seen_order():
    text = "gibberishxyz sawubona gibberishxyz anothernonword"
    assert zm.unparsed_words(text) == ["gibberishxyz", "anothernonword"]


def test_unparsed_words_returns_empty_for_a_fully_valid_sentence():
    assert zm.unparsed_words("Ngabe ngikwenze kangcono.") == []


def test_unparsed_words_excludes_a_known_proper_noun_glued_to_a_zulu_prefix():
    # Real 2026-08-29 finding: "Mogotsi" is a real surname, not a Zulu word, and
    # will never parse -- checking it against a Zulu morphology lexicon was never
    # going to succeed. Confirm the exact glued form from a real job run.
    assert zm.word_parses("mogotsi") is False
    assert zm.unparsed_words("noMnu. Mogotsi", known_proper_nouns=["mnu", "mogotsi"]) == []


def test_unparsed_words_still_reports_a_word_not_covered_by_known_proper_nouns():
    assert zm.unparsed_words(
        "Kwenzeka gibberishxyz manje.", known_proper_nouns=["mogotsi", "lincoln"],
    ) == ["gibberishxyz"]


def test_unparsed_words_skips_a_word_fused_to_a_digit_with_no_separator():
    # Real bug found 2026-08-29 running the check on a real job's every sentence
    # for the first time: a letter-only tokenizer split "ka2024" at the digit
    # boundary into a bare "ka" fragment that can never parse alone -- a false
    # positive unrelated to word-formedness. The fused token is now captured
    # whole and skipped entirely rather than checked as a fragment.
    assert zm.unparsed_words("kwakunguNovemba ka2024.") == []


def test_unparsed_words_skips_a_word_hyphenated_to_a_name_or_number():
    # Same real bug, hyphenated variants: "kwe-T[...]" and "angu-24" from the
    # same job run.
    assert zm.unparsed_words("kokuhlehliswa kwe-T umbuzo.") == []
    assert zm.unparsed_words("kwamahora angu-24.") == []


def test_unparsed_words_still_flags_a_genuine_bad_word_next_to_a_fused_token():
    # The digit/hyphen skip must not swallow an unrelated, genuinely invalid
    # word elsewhere in the same sentence.
    assert zm.unparsed_words("ka2024 gibberishxyz manje.") == ["gibberishxyz"]
