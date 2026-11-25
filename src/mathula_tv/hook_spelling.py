"""The footer hook is written in isiZulu by a separate editorial call, and it can spell a word of the dub another way (the dub said "isithangami sabezindaba", the hook "isithangazi sabezindaba").
The approved dub is immutable and is the authority on how its words are spelled, so a hook word that is not a word of the dub, but is a near miss of one that is, is written the way the dub writes it.

A near miss is deliberately narrow: the same first four letters, a length within one letter and a close match (a Zulu word changes at its front when its concord changes: "izobiza" and "sizobiza" are two
different words, and must never be turned into each other).  Every correction is returned, so the publication manifest says what was changed and why.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Iterable, Mapping, Sequence

MIN_LETTERS = 6          # shorter words are too easy to confuse with another word
MIN_RATIO = 0.88         # "isithangazi" / "isithangami" is 0.91
SAME_START = 4           # the front of a Zulu word carries its concords: a word that differs there is another word, not a spelling


def _words(text: str) -> list[str]:
    return [w for w in re.findall(r"[A-Za-z][A-Za-z'’-]*", str(text or "")) if w]


def vocabulary(dub_texts: Iterable[str], settled_terms: Sequence[Mapping[str, Any]] = ()) -> set[str]:
    """The words the dub says, lower case, with the words of the terms the agent settled."""
    words: set[str] = set()
    for text in dub_texts:
        words.update(w.lower() for w in _words(text))
    for term in settled_terms:
        words.update(w.lower() for w in _words(str(term.get("zulu") or "")))
    return words


def _near(word: str, known: set[str]) -> tuple[str, float] | None:
    best: tuple[str, float] | None = None
    for candidate in known:
        if len(candidate) < MIN_LETTERS or candidate[:SAME_START] != word[:SAME_START] or abs(len(candidate) - len(word)) > 1:
            continue
        ratio = SequenceMatcher(None, word, candidate).ratio()
        if ratio >= MIN_RATIO and (best is None or ratio > best[1]):
            best = (candidate, ratio)
    return best


def correct(hook_text: str, dub_texts: Iterable[str], settled_terms: Sequence[Mapping[str, Any]] = ()) -> tuple[str, list[dict[str, Any]]]:
    """(the hook with its near-miss words written as the dub writes them, the corrections made as [{hook, dub, ratio}])."""
    known = vocabulary(dub_texts, settled_terms)
    fixes: list[dict[str, Any]] = []

    def swap(match: re.Match[str]) -> str:
        token = match.group(0)
        word = token.lower()
        if "-" in word or len(word) < MIN_LETTERS or word in known:
            return token
        near = _near(word, known)
        if near is None:
            return token
        replacement = near[0].capitalize() if token[:1].isupper() else near[0]
        fixes.append({"hook": token, "dub": replacement, "ratio": round(near[1], 2)})
        return replacement

    return re.sub(r"[A-Za-z][A-Za-z'’-]*", swap, str(hook_text or "")), fixes
