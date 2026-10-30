"""An English word the speaker says as it is: which Zulu noun-class prefixes on it are wrong.

The glossary once made "Fighters" into "maFighter": the plural was changed to a singular and a Zulu plural was then built on the singular, a form
the speaker never said and nobody writes.  The agent followed the glossary.  Rules in a prompt did not stop it, so it is checked in code.

What real Zulu does (the Autshumato English-isiZulu corpus, 233,691 aligned sentence pairs, scanned for a noun-class prefix glued on a capitalised
word that is English): i- on an English noun is common (iCovid 119 times, iDurban, iGrain), ama- occurs (amaWorkshops, amaTechnikons, amaCouncilor), abe- and
isi- are rare, and the short ma- without its a never occurs.  So a blanket ban on prefixes would block ordinary Zulu, and the checks are narrow:

* a HARD failure (``find_prefixed_english``, the ``english_with_class_prefix`` gate) for the short ``ma-`` on an English word the speaker says, and for a
  prefix whose number contradicts the English the speaker says (a singular prefix on his plural: iFighter for "Fighters"; a plural prefix on his singular);
* a soft flag (``find_soft_prefixed_english``) for every other noun prefix on an English word: attested, but the voice reads the English letters with Zulu
  rules and the word is better kept as the speaker says it;
* ``fix_glossary`` sets the English word back wherever the glossary has a hard case, so that form cannot reach an agent or a translation call.

Only prefixes glued straight onto an English-shaped capitalised word that the speaker's own English has are looked at.  A hyphenated particle (ne-, ye-),
a locative e-, the u- / o- of a personal name, and Nguni roots ("isiZulu", "amaXhosa", which end in a vowel like a Zulu word) are not.
"""
from __future__ import annotations

import copy
import re
from typing import Any, Mapping

_CLASS_PREFIXED = re.compile(r"(?<![A-Za-z'\-])((?i:ma|ama|isi|izi|aba|abe|imi|izin|i))(?=[A-Z][a-z]{3,})([A-Za-z]+)")
_SINGULAR_PREFIXES = {"i", "isi"}
_PLURAL_PREFIXES = {"ama", "aba", "abe", "imi", "izi", "izin"}       # "ma" is neither: it is the short form that never occurs


def _stem(word: str) -> str:
    word = re.sub(r"[^a-z]", "", word.casefold())
    return word[:-1] if word.endswith("s") and len(word) > 4 else word


def _english_shaped(word: str) -> bool:
    from .pronunciation_search import looks_isizulu

    return not looks_isizulu(word)


def _english_forms(english_text: str) -> dict[str, list[str]]:
    forms: dict[str, list[str]] = {}
    for word in re.findall(r"[A-Za-z']+", english_text):
        forms.setdefault(_stem(word), []).append(re.sub(r"[^A-Za-z']", "", word))
    return forms


def _numbers(forms: list[str], stem: str) -> set[str]:
    """Which numbers the speaker's English has for this word: "plural" for a form that adds s to the stem, "singular" for the bare stem."""
    found = set()
    for form in forms:
        letters = re.sub(r"[^a-z]", "", form.casefold())
        plural = letters == stem + "s" and not letters.endswith(("ss", "us", "is"))          # "business", "bonus" and "analysis" are singular
        found.add("plural" if plural else "singular")
    return found


def prefixed_words(text: str, english_text: str) -> list[dict[str, Any]]:
    """Each noun-class prefix glued on an English-shaped word that ``english_text`` has: the token, the prefix, the English word as the speaker says
    it, and whether it is a hard case (``hard``) with the reason."""
    forms = _english_forms(english_text)
    found: list[dict[str, Any]] = []
    for match in _CLASS_PREFIXED.finditer(text):
        stem = _stem(match.group(2))
        if len(stem) < 4 or stem not in forms or not _english_shaped(match.group(2)) or match.group(0) in [f["token"] for f in found]:
            continue
        prefix = match.group(1).casefold()
        spoken = forms[stem][0]
        numbers = _numbers(forms[stem], stem)
        reason = ""
        if prefix == "ma":
            reason = "the short prefix ma- is not used on English words (it never occurs in real isiZulu text)"
        elif len(numbers) == 1 and prefix in _SINGULAR_PREFIXES and "plural" in numbers:
            reason = f"a singular prefix on the plural {spoken!r} the speaker says"
        elif len(numbers) == 1 and prefix in _PLURAL_PREFIXES and "singular" in numbers:
            reason = f"a plural prefix on the singular {spoken!r} the speaker says"
        found.append({"token": match.group(0), "prefix": prefix, "english": spoken, "hard": bool(reason), "reason": reason})
    return found


def find_prefixed_english(spoken_zulu: str, source_english: str) -> list[str]:
    """Gate evidence: the hard cases, with what to write instead."""
    return [f"{item['token']}: {item['reason']}; write the English word as spoken ({item['english']!r}) without the Zulu noun prefix"
            for item in prefixed_words(spoken_zulu, source_english) if item["hard"]]


def find_soft_prefixed_english(spoken_zulu: str, source_english: str) -> list[str]:
    """The other noun prefixes on an English word the speaker says (a flag, not a failure)."""
    return [f"{item['token']}: a Zulu noun prefix on the English word {item['english']!r}; the speaker says it as it is"
            for item in prefixed_words(spoken_zulu, source_english) if not item["hard"]]


def _replace_everywhere(value: Any, old: str, new: str) -> Any:
    if isinstance(value, str):
        return value.replace(old, new).replace(old[:1].swapcase() + old[1:], new)
    if isinstance(value, list):
        return [_replace_everywhere(v, old, new) for v in value]
    if isinstance(value, dict):
        return {k: _replace_everywhere(v, old, new) for k, v in value.items()}
    return value


def fix_glossary(glossary: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """The glossary with every term whose Zulu is a hard case ("Fighters" -> "maFighter") set back to the English word as the term gives it, and the
    old form replaced wherever else the glossary writes it (the register's address instructions).  Returns it with a plain sentence for each change."""
    fixed: dict[str, Any] = copy.deepcopy(dict(glossary))
    changes: list[str] = []
    replacements: dict[str, str] = {}
    for term in fixed.get("terms") or []:
        if isinstance(term, dict):
            for item in prefixed_words(str(term.get("zulu") or ""), str(term.get("english") or "")):
                if item["hard"]:
                    replacements[item["token"]] = item["english"]
                    changes.append(f"glossary: {term.get('english')!r} was rendered {item['token']!r} ({item['reason']}); now {item['english']!r}")
    for token, english in replacements.items():
        fixed = _replace_everywhere(fixed, token, english)
    return fixed, changes
