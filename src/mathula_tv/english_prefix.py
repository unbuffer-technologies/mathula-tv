"""An English word the speaker says as it is takes no Zulu noun-class prefix: "MaFighter" cannot be written.

The glossary once made "Fighters" into "maFighter": the plural was changed to a singular and a Zulu plural was then built on the singular, a form
the speaker never said and nobody speaks, and the agent followed the glossary.  Rules in a prompt did not stop it, so this is checked in code:

* ``find_prefixed_english`` is a hard gate on the dub (a unit with such a word fails the gate, and the evidence says what to write);
* ``fix_glossary`` puts the English word back wherever the glossary has the prefixed form, so the form can no longer reach an agent or a translation call.

Only noun-class prefixes glued straight onto an English-shaped capitalised word that the speaker's own English has are caught.  A hyphenated particle
(ne-, ye-), a locative e-, the u- / o- of a personal name, and Nguni roots ("isiZulu", "amaXhosa", which end in a vowel like a Zulu word) are not.
"""
from __future__ import annotations

import copy
import re
from typing import Any, Mapping

_CLASS_PREFIXED = re.compile(r"(?<![A-Za-z'\-])((?i:ma|ama|isi|izi|aba|abe|imi|izin|i))(?=[A-Z][a-z]{3,})([A-Za-z]+)")


def _stem(word: str) -> str:
    word = re.sub(r"[^a-z]", "", word.casefold())
    return word[:-1] if word.endswith("s") and len(word) > 4 else word


def _english_shaped(word: str) -> bool:
    from .pronunciation_search import looks_isizulu

    return not looks_isizulu(word)


def prefixed_words(text: str, english_text: str) -> list[tuple[str, str]]:
    """(the prefixed token, the English word as the speaker says it) for each noun-class prefix glued on an English-shaped word that
    ``english_text`` has."""
    spoken = {_stem(w): re.sub(r"[^A-Za-z']", "", w) for w in re.findall(r"[A-Za-z']+", english_text)}
    found: list[tuple[str, str]] = []
    for match in _CLASS_PREFIXED.finditer(text):
        stem = _stem(match.group(2))
        if len(stem) >= 4 and stem in spoken and _english_shaped(match.group(2)) and match.group(0) not in [token for token, _ in found]:
            found.append((match.group(0), spoken[stem]))
    return found


def find_prefixed_english(spoken_zulu: str, source_english: str) -> list[str]:
    """Gate evidence: each such word, with what to write instead."""
    return [f"{token}: the speaker says the English word {english!r} as it is; write it as spoken, without the Zulu noun prefix"
            for token, english in prefixed_words(spoken_zulu, source_english)]


def _replace_everywhere(value: Any, old: str, new: str) -> Any:
    if isinstance(value, str):
        return value.replace(old, new).replace(old[:1].swapcase() + old[1:], new)
    if isinstance(value, list):
        return [_replace_everywhere(v, old, new) for v in value]
    if isinstance(value, dict):
        return {k: _replace_everywhere(v, old, new) for k, v in value.items()}
    return value


def fix_glossary(glossary: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """The glossary with every term whose Zulu is a noun prefix on its own English word ("Fighters" -> "maFighter") set back to the English word as
    the term gives it, and the old form replaced wherever else the glossary writes it (the register's address instructions).  Returns it with a plain
    sentence for each change."""
    fixed: dict[str, Any] = copy.deepcopy(dict(glossary))
    changes: list[str] = []
    replacements: dict[str, str] = {}
    for term in fixed.get("terms") or []:
        if isinstance(term, dict):
            for token, english in prefixed_words(str(term.get("zulu") or ""), str(term.get("english") or "")):
                replacements[token] = english
                changes.append(f"glossary: {term.get('english')!r} was rendered {token!r}, a Zulu noun prefix on an English word the speaker says as it is; now {english!r}")
    for token, english in replacements.items():
        fixed = _replace_everywhere(fixed, token, english)
    return fixed, changes
