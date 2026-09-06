"""Real, mechanical isiZulu morphological validity checking via Pretorius/Bosch's
ZulMorph finite-state analyser (see pyproject.toml's ``zulu-morphology`` extra).

This is a SUPPLEMENTARY signal, not a replacement for anything else in this
codebase's QA machinery: it confirms whether a word is a well-formed isiZulu
word at all (real morphology, not garbled/hallucinated), never whether it is
the semantically CORRECT word for its context. A wrong-but-valid word (the real
production case: "Sine." meaning "we have", generated in place of the correct
"Sesine." meaning "four") parses cleanly -- confirmed directly against this
FST -- so this can never catch that class of error. Its actual, honest value is
narrower: catching outright invented/malformed non-words, which nothing else in
this pipeline checks mechanically (every other check is an LLM's own judgment).

Requires the optional ``zulu-morphology`` extra (``foma-with-lib`` + ``zulmorph``).
The plain PyPI ``foma`` package has no Windows binary at all (``ctypes.util.
find_library('foma')`` finds nothing); ``foma-with-lib`` bundles a working
``libfoma.dll`` and was confirmed to load and analyse correctly on Windows
2026-08-29. Every public function here degrades gracefully (returns an empty/
permissive result) when the extra isn't installed -- this must never be a hard
requirement for the pipeline to run.
"""
from __future__ import annotations

import re
from functools import lru_cache

# Confirmed empirically 2026-08-29 by testing real words from a real production
# transcript against this exact FST: ZulMorph's lexicon has solid coverage for
# verbs, nouns, and adjectives (including correctly parsing "sesine", "kulungile",
# "kuhle", "ukuvimba", and the ngabe-companion verb "ngikwenze"), but is missing at
# least these common, unambiguously-real, pure interjections. Extend this list
# only after confirming a new real false positive the same way -- never
# speculatively -- mirroring CONFIRMED_TRANSLATION_PITFALLS's own discipline.
KNOWN_UNPARSED_INTERJECTIONS = frozenset({"yebo", "hhayi"})

# Confirmed real bug 2026-08-29 (found by running the morphology check on
# EVERY sentence in a real job for the first time, not just compaction-touched
# ones): a letter-only pattern splits a word fused directly to a number or a
# hyphen-attached name/abbreviation ("ka2024", "kwe-T[...]", "no-uAndre",
# "angu-24") at that boundary, leaving a bare fragment ("ka", "kwe", "no",
# "angu") that can never parse on its own -- a false positive unrelated to
# word-formedness. This pattern instead captures the whole fused token, and
# unparsed_words() below skips checking any token containing a digit or a
# hyphen entirely, since it's a number/abbreviation/name compound the
# analyser was never going to verify as a standalone Zulu word.
_WORD_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9-]*")


class ZuluMorphologyUnavailable(RuntimeError):
    """Raised internally when the optional zulu-morphology extra isn't installed."""


@lru_cache(maxsize=1)
def _load_fst():
    try:
        from foma_with_lib.foma import FST
    except ImportError as exc:
        raise ZuluMorphologyUnavailable(
            "foma-with-lib is not installed; install the 'zulu-morphology' extra"
        ) from exc
    try:
        import importlib.resources as resources
        fst_path = resources.files("zulmorph.res").joinpath("zul.fom")
    except ImportError as exc:
        raise ZuluMorphologyUnavailable(
            "zulmorph is not installed; install the 'zulu-morphology' extra"
        ) from exc
    return FST.load(str(fst_path))


def is_available() -> bool:
    """Whether the optional zulu-morphology extra is installed and the FST loads."""
    try:
        _load_fst()
        return True
    except ZuluMorphologyUnavailable:
        return False


def word_parses(word: str) -> bool:
    """True if `word` is a well-formed isiZulu word per ZulMorph's finite-state
    analyser, or is a known interjection its lexicon doesn't cover (see
    KNOWN_UNPARSED_INTERJECTIONS). Case-insensitive -- the FST's own lexicon is
    lowercase-only, confirmed empirically (a capitalized sentence-initial word
    otherwise produces a false "doesn't parse" result).

    Only ever a NEGATIVE signal worth acting on: False means "this exact surface
    form doesn't parse and isn't a known exception" -- worth surfacing for human
    review, never grounds to silently discard a candidate outright, since the
    analyser's real lexicon coverage is solid but not exhaustive.
    """
    normalized = word.strip().lower()
    if not normalized:
        return True
    if normalized in KNOWN_UNPARSED_INTERJECTIONS:
        return True
    fst = _load_fst()
    try:
        next(fst.apply_up(normalized))
        return True
    except StopIteration:
        return False


def unparsed_words(text: str, *, known_proper_nouns=()) -> list[str]:
    """Return every DISTINCT word in `text` (tokenized on runs of letters, so
    punctuation/digits are never themselves flagged) that fails word_parses(),
    in first-seen order. Returns [] when the optional dependency isn't installed
    (never raises) -- this is a supplementary signal, never a hard requirement.

    `known_proper_nouns`: lowercase tokens the CALLER already knows will never
    parse as complete isiZulu word forms for a reason unrelated to whether the
    candidate is well-formed (e.g. glossary name/role terms, capitalized words
    from the English source). Confirmed real need 2026-08-29: a flagged word
    CONTAINING one of these as a substring (case-insensitive) is either a
    foreign name glued to a Zulu prefix/concord ("noShadrack", "waseBritish"),
    OR a locked ABBREVIATION of a real Zulu word ("kaMnu" -- "Mnu." is short for
    "Mnumzana", confirmed: the full word parses, the abbreviation alone does
    not, exactly like "Mr." for "Mister" in English). Neither case says anything
    about whether the candidate is well-formed, so checking either against a
    Zulu morphology lexicon was never going to succeed -- doing so anyway
    drowned this signal's one genuine catch (native_phrase_0039/0040's locative
    construction) in a real job run's worth of these false positives.
    """
    if not is_available():
        return []
    proper_lower = [str(p).lower() for p in known_proper_nouns if p]
    seen: dict[str, None] = {}
    for word in _WORD_PATTERN.findall(text):
        if any(ch.isdigit() for ch in word) or "-" in word:
            continue  # a number/abbreviation/name compound -- never checkable, not a word-formedness signal
        if word_parses(word):
            continue
        word_lower = word.lower()
        if any(proper in word_lower for proper in proper_lower):
            continue
        seen.setdefault(word, None)
    return list(seen)
