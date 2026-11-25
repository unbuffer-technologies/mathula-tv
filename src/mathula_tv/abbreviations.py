"""Abbreviations the voice would read wrongly, written out before it is given the text.

The agent writes the way a newsroom writes: "Mnu. Mashaba", "Dr. Smith", "Prof. Mbeki", "20 km", "etc.".  The voice reads an abbreviation as the letters it sees or skips it, so a title came out as "mnu".  The
rule is the one a listener's ear has always asked for: every abbreviation that has a spoken form is written out, by the system, before the text reaches the voice, whatever the agent wrote and whatever
name follows it.  Only the hidden spoken text changes; the approved translation and the captions keep their spelling.

* The isiZulu honorifics (Mnu, Nkk, Nksz, Dkt: pronunciation.py's table, with a Zulu prefix glued on or not) say the Zulu word.
* The English and Afrikaans titles say the word the way it is said aloud in a Zulu news bulletin (Dr -> Doctor, Mnr -> Meneer): English as spoken is the default for what has no Zulu word the corpora attest.
* "e.g.", "i.e.", "etc." and a few units after a figure (km, kg, ...) say what they stand for.

A title is only taken where it cannot be an ordinary word: a full stop after it, or a capital letter after it (a name follows a title).  The abbreviations that are also words or names (Gen, Col, Rev, Hon, Lt, Capt, Maj,
Sgt, Jr, Sr) need their full stop.  ``expand`` is idempotent: what it wrote has no abbreviation left in it.
"""
from __future__ import annotations

import re
from typing import Any, Iterable

from .pronunciation import _ZU_FUSED_HONORIFIC, _ZU_HONORIFIC_EXPANSIONS, _zulu_native_honorific_tts

# said before a name, spoken as the word they stand for; the capital after them is what tells a title from a word
TITLES_BY_CAPITAL = {"dr": "Doctor", "prof": "Professor", "adv": "Advocate", "mr": "Mister", "mrs": "Missus", "ms": "Miss", "mnr": "Meneer", "mev": "Mevrou", "cllr": "Councillor"}
# also ordinary words or names without their stop ("Gen Z"), so these need it
TITLES_WITH_STOP = {"hon": "Honourable", "rev": "Reverend", "gen": "General", "col": "Colonel", "capt": "Captain", "lt": "Lieutenant", "maj": "Major", "sgt": "Sergeant", "jr": "Junior", "sr": "Senior"}
UNITS = {"km/h": "kilometres per hour", "km": "kilometres", "kg": "kilograms", "cm": "centimetres", "mm": "millimetres", "ml": "millilitres"}

_PREFIX = r"(?P<prefix>kwa|ngo|lika|ku|no|ne|na|ka|u|sino|nino)?"
_BY_CAPITAL = re.compile(r"(?<![\w.])" + _PREFIX + r"(?P<abbr>" + "|".join(sorted(TITLES_BY_CAPITAL, key=len, reverse=True)) + r")(?:\.(?=\s|$)|(?=\s+(?-i:[A-Z])))", re.IGNORECASE | re.UNICODE)
_WITH_STOP = re.compile(r"(?<![\w.])" + _PREFIX + r"(?P<abbr>" + "|".join(sorted(TITLES_WITH_STOP, key=len, reverse=True)) + r")\.(?=\s|$)", re.IGNORECASE | re.UNICODE)
_LATIN = ((re.compile(r"(?<![\w.])e\.g\.", re.IGNORECASE), "for example"), (re.compile(r"(?<![\w.])i\.e\.", re.IGNORECASE), "that is"), (re.compile(r"(?<![\w.])etc\.", re.IGNORECASE), "et cetera."))
_NUMBER = re.compile(r"(?<![\w.])No\.(?=\s*\d)\s*")
_UNIT = re.compile(r"(?<=\d)\s?(?P<unit>km/h|km|kg|cm|mm|ml)(?![\w/])")


# Zulu class, locative and linking prefixes that an English word is kept after ("ama-details", "ezingu-forty", "kwa-workshops"); two letters or more, so "e-mail" and the like are left alone
CLASS_PREFIXES = frozenset({"ama", "aba", "abe", "abo", "izi", "izin", "izim", "imi", "isi", "ezi", "ezin", "ezim", "ezingu", "ezama", "ezaba", "eyama", "nama", "nezi", "nezin", "neza", "nge", "ngama", "ngezi", "ngezin",
                            "ngo", "nga", "kuma", "kuzi", "kwama", "kwezi", "kwa", "ne", "na", "no", "ku", "ka", "wa", "ya", "za", "la", "ba", "lo", "so", "uku", "we", "ye", "ze", "le", "se", "ngu", "ngi", "sino", "nino"})
_GLUED_TOKEN = re.compile(r"(?P<lead>[^\w]*)(?P<prefix>[a-z]{2,6})-(?P<stem>[a-z]{4,})(?P<tail>[^\w-]*)")


def english_stem(prefix: str, stem: str) -> bool:
    """Whether ``stem`` is an English word kept in English after the Zulu class prefix ``prefix``: a known prefix, a lower-case stem of four letters or more that the Zulu spelling rules do not make."""
    from .pronunciation_search import looks_isizulu

    return prefix in CLASS_PREFIXES and stem.isalpha() and stem.islower() and len(stem) >= 4 and not looks_isizulu(stem)


def title_words() -> frozenset[str]:
    """Every title, abbreviated or written out, lower case: none of them is part of a name (omnivoice_tts.ReferenceBank.dub_name_runs)."""
    words = set(_ZU_HONORIFIC_EXPANSIONS) | set(TITLES_BY_CAPITAL) | set(TITLES_WITH_STOP)
    words |= {w.lower() for w in _ZU_HONORIFIC_EXPANSIONS.values()} | {w.lower() for w in TITLES_BY_CAPITAL.values()} | {w.lower() for w in TITLES_WITH_STOP.values()}
    words |= {"mnumzane", "umnumzane", "umnumzana", "solwazi"}
    return frozenset(words)


def _cased(word: str, like: str) -> str:
    """``word`` with the first letter in the case the abbreviation had ("dr." -> "doctor", "Dr." -> "Doctor")."""
    return (word[:1].lower() + word[1:]) if like[:1].islower() else word


def expand(text: str) -> tuple[str, list[dict[str, str]]]:
    """(the text with its abbreviations written out as they are said, what was changed as [{was, now}])."""
    changes: list[dict[str, str]] = []

    def record(was: str, now: str) -> str:
        if was != now:
            changes.append({"was": was, "now": now})
        return now

    def zulu(match: re.Match[str]) -> str:
        return record(match.group(0), _zulu_native_honorific_tts(match))

    def english(table: dict[str, str]):
        def swap(match: re.Match[str]) -> str:
            abbr = match.group("abbr")
            return record(match.group(0), (match.group("prefix") or "") + _cased(table[abbr.casefold()], abbr))
        return swap

    out = _ZU_FUSED_HONORIFIC.sub(zulu, str(text or ""))
    out = _BY_CAPITAL.sub(english(TITLES_BY_CAPITAL), out)
    out = _WITH_STOP.sub(english(TITLES_WITH_STOP), out)
    for pattern, spoken in _LATIN:
        out = pattern.sub(lambda m, spoken=spoken: record(m.group(0), spoken), out)
    out = _NUMBER.sub(lambda m: record(m.group(0), "Number "), out)
    out = _UNIT.sub(lambda m: record(m.group(0), " " + UNITS[m.group("unit").lower()]), out)
    return out, changes


def separate_stems(text: str, keep: Iterable[str] = ()) -> tuple[str, list[int]]:
    """(the text with each English word kept after a glued Zulu class prefix written as two words ("ama-details" -> "ama details"), how many words each word of the original became).

    Only for a sentence the voice speaks and whose audio is heard as it is, or whose word is taken from the speaker's own audio (omnivoice_tts speak): the voice reads the glued form as one Zulu word and
    swallows the stem's consonants (prod-x1: the t of "details" in 5 of 6 draws; with a space, 3 of 3 seeds said it).  It is NOT written into the dub text: in the sentence the word is cut out of, the glued
    form leaves the word after it whole (the "oku" of "okugcina": 11 of 12 draws glued, 4 of 8 spaced), and a word in ``keep`` (lower case: the words that will be cut out) stays glued.  The caller folds the voice's
    word times back to the original words with ``fold_words``."""
    kept = {str(k).lower() for k in keep}
    out: list[str] = []
    groups: list[int] = []
    changed = False
    for token in str(text or "").split():
        m = _GLUED_TOKEN.fullmatch(token)
        if m and english_stem(m.group("prefix"), m.group("stem")) and m.group("stem").lower() not in kept:
            out.extend((m.group("lead") + m.group("prefix"), m.group("stem") + m.group("tail")))
            groups.append(2)
            changed = True
        else:
            out.append(token)
            groups.append(1)
    return (" ".join(out) if changed else str(text or "")), groups


def fold_words(words: Any, groups: list[int]) -> Any:
    """The word times of a text written with ``separate_stems``, as the words of the original text: a word that became two takes the start of the first and the end of the last (None when either has no time).
    A list that does not match the groups is returned as it is."""
    if not isinstance(words, list) or sum(groups) != len(words) or all(g == 1 for g in groups):
        return words
    folded: list[Any] = []
    at = 0
    for g in groups:
        chunk = words[at:at + g]
        at += g
        if g == 1:
            folded.append(chunk[0])
        elif all(chunk):
            folded.append({**chunk[0], "start": chunk[0]["start"], "end": chunk[-1]["end"]})
        else:
            folded.append(None)
    return folded
