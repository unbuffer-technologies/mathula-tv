"""isiZulu object concords and the reflexive: "I see HIM", "tell US", "he cut HIMSELF" (module 5 of the grammar foundation).

The concord stands directly before the verb root, after the subject concord and any tense marker, so it fits every form of module 2: ngi-ya-ku-bon-a, ngi-zo-ku-bon-a,
ngi-ku-bon-ile, a-si-ku-bon-i.  The forms are the annotated-corpus table of module 1 plus these facts counted in 8.4 million words of isiZulu news
(``scripts/grammar_mine_oc.py`` and the counts quoted below); ``scripts/grammar_verify_object.sh`` builds words with ``with_object`` and checks them with ZulMorph.
"""
from __future__ import annotations

import re

from .zulu_agreement import CLASSES

PERSON_OBJECT = {"1ps": "ngi", "2ps": "ku", "1pp": "si", "2pp": "ni"}
CLASS_OBJECT = {c.cls: c.oc.split("/")[0].strip("-") for c in CLASSES if c.cls not in ("1a", "2a")}
CLASS_OBJECT["1"] = "m"                       # class 1: m- (mu- only before a root with no vowel, see below)
REFLEXIVE = "zi"                               # himself / herself / themselves / myself ...: the same zi- for every person and class


def join_concord(concord: str, root: str) -> str:
    """Attach a concord (object concord, reflexive zi-, and the same holds for subject concords) to a root.
    Before a vowel-initial root the concord loses its final vowel, and u becomes w: ku + az -> kwaz, lu + az -> lwaz, ngi + enz -> ngenz, si + enz -> senz, ba + enz -> benz,
    yi + az -> yaz, zi + enz -> zenz.  The unmerged forms (kuazi, ngienza) are not written: 0 of 14,000 news words."""
    if root and root[0] in "aeiou" and concord not in ("m", "mu"):
        if concord.endswith("u"):
            return concord[:-1] + ("" if concord.endswith("wu") else "w") + root           # ku -> kw, lu -> lw, but wu (class 3) already holds the w: wu + enz -> wenz
        return concord[:-1] + root
    return concord + root


def object_concord(target: str, root: str = "") -> str:
    """The concord for a person (1ps 2ps 1pp 2pp), a noun class ("2", "9"...) or "refl".  Class 1 is m-, and mu- before a root that has no vowel (mudla, mupha)."""
    if target == "refl":
        return REFLEXIVE
    if target in PERSON_OBJECT:
        return PERSON_OBJECT[target]
    if target == "1" and root and not re.search(r"[aeiou]", root):
        return "mu"
    return CLASS_OBJECT[target]


def with_object(root: str, target: str) -> str:
    """The root with its object concord attached, ready to be used as the root of ``zulu_tam.conjugate``: with_object("thand", "1") -> "mthand"."""
    return join_concord(object_concord(target, root), root)


ABSOLUTE_PRONOUNS = {"1ps": "mina", "2ps": "wena", "1pp": "thina", "2pp": "nina", "1": "yena", "2": "bona", "3": "wona", "4": "yona", "5": "lona", "6": "wona",
                     "7": "sona", "8": "zona", "9": "yona", "10": "zona", "11": "lona", "14": "bona", "15": "khona"}


def render() -> str:
    people = ", ".join(f"{k}: {v}-" for k, v in PERSON_OBJECT.items())
    classes = ", ".join(f"{k}: {v}-" for k, v in CLASS_OBJECT.items())
    pron = ", ".join(f"{k}: {v}" for k, v in ABSOLUTE_PRONOUNS.items())
    return f"""ISIZULU OBJECT CONCORDS AND THE REFLEXIVE -- "I see HIM", "tell US", "he cut HIMSELF": one short syllable on the verb replaces an English pronoun, so it saves words.
Object concord by person: {people}. By the noun's class: {classes} (1: mu- before a root with no vowel: mudla, mupha). Reflexive (himself, themselves, myself): zi- for every person.
Place. Directly before the root, after the subject concord and after any tense marker: ngi-ya-ku-bona, ngi-zo-ku-bona, ngi-ku-bonile, a-si-ku-bon-i (the negative shape
a + subject + object + root: 5,355 news words, asikwazi, awukwazi), si-ya-zi-qhenya (the shape subject + ya + concord + root: 6,116 news words, siyaziqhenya, uyaziqhenya).
The same place in every form of the tense module: build the verb with the concord attached to the root.
Class 1. m- before nearly every root, long or short, vowel-initial or not: ngiyamthanda (m- 229, mu- 10), ngiyambona (418 against 7), ngiyamazi (258 against 0), ngiyamenza.
mu- only before a root with no vowel in it: mudla (28 against 2), mupha (10 against 2).
Vowel-initial roots. The concord loses its final vowel, and u becomes w, so the two never stand side by side: ku + azi -> kwazi (7,282 news words, 0 for kuazi), ku + enza -> kwenza,
lu + azi -> lwazi, ngi + azi -> ngazi, ngi + enza -> ngenza, si + enza -> senza, ba + enza -> benza, yi + azi -> yazi, li + enza -> lenza, wa + enza -> wenza, zi + azi -> zazi.
The subject concord does the same before a vowel-initial root (ngenza, senza, benza, wenza, yenza, kwenza).
Reflexive: zi + root (ziphatha, zibona, zivikela, zihlola, zisebenzisa; zi + azi -> zazi, zi + enza -> zenza). As a noun: ukuziphatha "behaviour", ukuzethemba "confidence".
Emphatic (absolute) pronoun, "he himself", "us too": {pron}."""
