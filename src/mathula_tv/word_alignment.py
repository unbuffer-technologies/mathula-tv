"""Which word of the Zulu says what a word of the English said.

The dub is a retelling, so the speaker's pause "after the word X" cannot be placed in the Zulu by counting letters: a sentence that is re-ordered or re-worded moves X.  This lines the two
sides up from evidence, not from position:

* a word that crosses unchanged (a name, a number, a loanword: Matlala / likaMatlala, 105A, initials / ama-initials) is an anchor;
* a word the parallel corpus translates (``Attestation.translation_forms``: the Zulu forms that share the aligned sentences of an English word, scored by Dice: lawyer / ummeli, speaker /
  somlomo) is an anchor when a word of the Zulu is that form or shares its stem (ZulMorph's stems are patchy, so the stem test is a plain one: a run of letters most of the shorter
  word long, which is what likasomlomo and somlomo, or ummeli and kummeli, share).

The anchors that keep the words in the same order (the longest weighted chain; a re-ordered pair is dropped) are the alignment.  Between two anchors a position is found by how far it falls
between them, so the error is the distance between two anchors and not the length of the sentence.  Nothing here decides anything: it only says where in the Zulu to look for the silence.
"""
from __future__ import annotations

import difflib
import re
from functools import lru_cache
from typing import Callable, Sequence

LITERAL_MIN_LETTERS = 5         # a copied word that is not the whole Zulu word is at least this long (Matlala in likaMatlala; "that" in wesithathu is chance)
LITERAL_EXTRA = 6               # and the Zulu word is at most this many letters longer (grammar round the copy)
STEM_MIN_COMMON = 5             # a stem shared by a corpus form and a word of the Zulu: at least this many letters,
STEM_SHARE = 0.6                # and this share of the shorter word
TRANSLATION_WEIGHT = 0.8        # an inflected form is weaker evidence than the form itself
MIN_ENGLISH_LETTERS = 4         # a shorter English word is a function word or a particle: only an exact copy counts
GRAM = 4

FormsOf = Callable[[str], Sequence[tuple[str, float]]]
Aligner = Callable[[Sequence[str], Sequence[str]], list[tuple[int, int]]]


def letters(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


@lru_cache(maxsize=50000)
def _grams(word: str) -> frozenset[str]:
    return frozenset(word[i:i + GRAM] for i in range(len(word) - GRAM + 1)) if len(word) >= GRAM else frozenset()


@lru_cache(maxsize=200000)
def _common(a: str, b: str) -> int:
    """The length of the longest run of letters the two words share (cheap 4-gram test first: most pairs share nothing)."""
    if not _grams(a) & _grams(b):
        return 0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).find_longest_match(0, len(a), 0, len(b)).size


def _parts(zulu_token: str) -> list[str]:
    """The letters of a Zulu token and of each hyphen-separated part of it (ne-Section, ama-initials, ngo-2017)."""
    parts = [letters(p) for p in re.split(r"[-–—]", str(zulu_token))]
    whole = letters(zulu_token)
    return [p for p in dict.fromkeys(parts + [whole]) if p]


def _literal(english: str, part: str) -> float:
    """A word that crosses into the Zulu as it is: equal, or the Zulu word is the English word with grammar before it (likaMatlala, nesignesha is not: that is a respelling) or after it
    (a locative).  A short English word inside a longer Zulu one is chance (that in wesithathu), so the copy has to be this long."""
    if english == part:
        return 1.0
    if len(english) < LITERAL_MIN_LETTERS or english.isdigit() or part.isdigit():
        return 0.0
    if len(part) <= len(english) + LITERAL_EXTRA and (part.endswith(english) or part.startswith(english)):
        return 0.9
    return 0.0


def _same_stem(form: str, part: str) -> bool:
    if form == part:
        return True
    shorter = min(len(form), len(part))
    return shorter >= STEM_MIN_COMMON and _common(form, part) >= max(STEM_MIN_COMMON, -(-STEM_SHARE * shorter // 1))


def anchors(english: Sequence[str], zulu: Sequence[str], forms_of: FormsOf) -> list[tuple[int, int, float]]:
    """Every (English word, Zulu word, weight) pair that has evidence: a copy, or a corpus translation."""
    zulu_parts = [_parts(z) for z in zulu]
    found: list[tuple[int, int, float]] = []
    for k, raw in enumerate(english):
        word = letters(raw)
        if len(word) < 3:
            continue
        forms: list[tuple[str, float]] = []
        translatable = len(word) >= MIN_ENGLISH_LETTERS and not any(c.isdigit() for c in word)
        if translatable:
            try:
                forms = [(letters(f), float(s)) for f, s in forms_of(word)]
            except Exception:                                             # noqa: BLE001 - an advisory view must never fail a check
                forms = []
        for j, parts in enumerate(zulu_parts):
            best = 0.0
            for part in parts:
                best = max(best, _literal(word, part))
                if best >= 1.0:
                    break
                for form, score in forms:
                    if score * TRANSLATION_WEIGHT > best and _same_stem(form, part):
                        best = score * TRANSLATION_WEIGHT
            if best > 0:
                found.append((k, j, best))
    return found


def chain(pairs: Sequence[tuple[int, int, float]]) -> list[tuple[int, int]]:
    """The heaviest set of pairs in which both words only move forward (a longest increasing chain over the anchors; a re-ordered pair loses to the order the rest keep)."""
    ordered = sorted(pairs)
    if not ordered:
        return []
    best = [p[2] for p in ordered]
    back = [-1] * len(ordered)
    for i, (k, j, weight) in enumerate(ordered):
        for h in range(i):
            kh, jh, _ = ordered[h]
            if kh < k and jh < j and best[h] + weight > best[i]:
                best[i], back[i] = best[h] + weight, h
    at = max(range(len(ordered)), key=best.__getitem__)
    out: list[tuple[int, int]] = []
    while at >= 0:
        out.append((ordered[at][0], ordered[at][1]))
        at = back[at]
    return out[::-1]


def make_aligner(forms_of: FormsOf) -> Aligner:
    """An aligner over the corpus's translations: ``aligner(english_words, zulu_words) -> [(english index, zulu index), ...]`` in order."""
    def align(english: Sequence[str], zulu: Sequence[str]) -> list[tuple[int, int]]:
        try:
            return chain(anchors(english, zulu, forms_of))
        except Exception:                                             # noqa: BLE001 - no alignment is the position-only answer, never an error
            return []
    return align
