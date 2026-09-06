"""A real, general-purpose English->isiZulu word lexicon, used as SUPPLEMENTARY
reference vocabulary for generation -- distinct from every other context
mechanism in this pipeline. Confirmed real gap, 2026-08-29: ZULU_GRAMMAR_FOUNDATION
(native_dub.py) gives the model grammatical RULES, zulu_terminology_glossary locks
down a per-job handful of recurring proper nouns/institutional terms, but nothing
gives the model a general VOCABULARY reference the way a human translator would
consult a dictionary for an ordinary word they're unsure of.

Source: Autshumato Multilingual Word and Phrase Translations (CTexT, North-West
University / South African Department of Arts and Culture), 6,241 English->isiZulu
word entries, Creative Commons Attribution 2.5 South Africa licence -- see
data/AUTSHUMATO_LICENSE.txt. Government-domain-sourced, not hand-verified by this
project, and confirmed to contain real quality issues on spot-check (e.g. its
"interception" entries don't match this job's own carefully-researched
"ukuvimba"). Treated accordingly everywhere it's used: a HINT the model may take
or leave, never authoritative, and never a substitute for zulu_terminology_glossary
(job-specific and verified) when the two conflict.
"""
from __future__ import annotations

import re
from functools import lru_cache
from importlib import resources

_WORD_PATTERN = re.compile(r"[A-Za-z]+")
_MAX_ENTRIES_PER_SENTENCE = 12

# Confirmed real defect 2026-08-30: relevant_entries() used to take the first
# N lexicon-matching words in sentence order with no content-word priority --
# a general bilingual word list has entries for closed-class function words
# too, so a real production sentence ("But before we bring in Kenny Mapango,
# who's also been listening into what has been said here, it was interesting
# to hear.") filled its entire 12-slot cap with "but", "before", "we", "in",
# "who", "also", "what", "has", "said", "here", "it" -- and "interesting", the
# one word that actually mattered (and IS in the lexicon, as "okunakisayo"),
# never reached the model at all. It was then mistranslated using a
# positive-emotion verb ("-jabulisa", to please) instead of the neutral
# "-naka"-based form the lexicon would have suggested. Skipping stopwords
# entirely (not just deprioritizing them) is deliberate: a hint slot spent on
# "we"/"it" is never useful, so there is no tradeoff to weigh here.
_STOPWORDS = frozenset({
    "a", "about", "after", "all", "also", "an", "and", "any", "are", "as", "at",
    "be", "been", "being", "before", "but", "by", "can", "could", "did", "do",
    "does", "doing", "down", "during", "each", "few", "for", "from", "further",
    "had", "has", "have", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its",
    "itself", "just", "me", "more", "most", "my", "myself", "no", "nor",
    "not", "now", "of", "off", "on", "once", "only", "or", "other", "our",
    "ours", "ourselves", "out", "over", "own", "s", "same", "shall", "she",
    "should", "so", "some", "such", "t", "than", "that", "the", "their",
    "theirs", "them", "themselves", "then", "there", "these", "they", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "we", "were", "what", "when", "where", "which", "while", "who", "whom",
    "why", "will", "with", "would", "you", "your", "yours", "yourself",
    "yourselves",
})


# Small, hand-verified additions for real, confirmed gaps in the bundled
# Autshumato data (each checked against zulu_morphology.py's real FST before
# being added here -- see CONFIRMED_TRANSLATION_PITFALLS in native_dub.py for
# the production case each entry closes). Not a general-purpose supplement --
# only words a real job actually needed and the base lexicon genuinely lacks.
# Takes priority over the base lexicon on any future key collision, since
# these are specifically verified by this project rather than government-
# domain-sourced and unreviewed.
_SUPPLEMENTARY_ENTRIES: dict[str, list[str]] = {
    "compare": ["-qhathanisa"],
    "contrast": ["-qhathanisa"],
    "contrasted": ["-qhathanisa"],
    "comparing": ["-qhathanisa"],
}


@lru_cache(maxsize=1)
def _load_lexicon() -> dict[str, list[str]]:
    lexicon: dict[str, list[str]] = {}
    raw = resources.files("mathula_tv.data").joinpath("autshumato_english_isizulu.txt").read_text(encoding="utf-8")
    for line in raw.splitlines():
        if not line.strip() or "\t" not in line:
            continue
        english, _, zulu_field = line.partition("\t")
        english = english.strip().lower()
        translations = [t.strip() for t in zulu_field.split(";") if t.strip()]
        if english and translations:
            lexicon[english] = translations
    lexicon.update(_SUPPLEMENTARY_ENTRIES)
    return lexicon


def lookup(word: str) -> list[str]:
    """Return the lexicon's isiZulu translation(s) for `word` (case-insensitive,
    exact match only), or [] if not found.
    """
    return _load_lexicon().get(word.strip().lower(), [])


def relevant_entries(english_text: str, *, max_entries: int = _MAX_ENTRIES_PER_SENTENCE) -> dict[str, list[str]]:
    """Return {english_word: [zulu_options]} for every DISTINCT CONTENT word in
    `english_text` that has a lexicon entry, in first-seen order, capped at
    `max_entries` (a long sentence should not balloon the payload). Closed-class
    function words (see _STOPWORDS) are skipped entirely, not merely
    deprioritized -- confirmed real defect: without this, a real sentence's
    entire cap filled with "but"/"we"/"in"/"who"/"it" before ever reaching the
    one word ("interesting") that actually mattered.
    """
    result: dict[str, list[str]] = {}
    for word in _WORD_PATTERN.findall(english_text):
        if len(result) >= max_entries:
            break
        key = word.lower()
        if key in result or key in _STOPWORDS:
            continue
        translations = lookup(key)
        if translations:
            result[key] = translations
    return result
