"""How alike two readings of the same word are, as phone sequences from different recognisers.

The recognisers spell the same sounds differently (``aɪ`` or ``a ɪ``, ``ː`` and ``ʰ`` marks, five vowel qualities that are one in running speech), so both readings
are reduced to a short string of base sounds first.  Used to check that a word cut out of one recogniser's reading of a phrase is the word the English voice says
(``reference_tts``, ``pronunciation_lab``).
"""
from __future__ import annotations

import difflib
import re
import unicodedata
from typing import Sequence

_COARSE = {                     # vowels to five classes, consonants to their base: a comparison of what was said, not of how the recogniser spells it
    **{c: "a" for c in "aɑæʌɐəɚɜ"}, **{c: "e" for c in "eɛ"}, **{c: "i" for c in "iɪɨɯ"}, **{c: "o" for c in "oɔɒɵø"}, **{c: "u" for c in "uʊʉ"},
    "ɹ": "r", "ɾ": "r", "ɻ": "r", "ʁ": "r", "ɡ": "g", "ɫ": "l", "ʃ": "S", "ʒ": "S", "ɕ": "S", "ʂ": "S", "ʐ": "S", "ʈ": "t", "ɖ": "d", "ŋ": "n", "ɲ": "n", "θ": "t", "ð": "d",
}


def coarse(phones: Sequence[str]) -> str:
    """The phones as a short string of base sounds: modifiers and length marks dropped, vowels in five classes, repeats merged."""
    text = unicodedata.normalize("NFD", "".join(str(p) for p in phones))
    text = "".join(c for c in text if not unicodedata.combining(c) and c not in "ːˑʰʲʷ˞▁ ʔ")
    text = "".join(_COARSE.get(c, c) for c in text)
    return re.sub(r"(.)\1+", r"\1", text)


def agreement(first: Sequence[str], second: Sequence[str]) -> float | None:
    """How alike two decodings of the same word are, 0 to 1; None when either is empty."""
    a, b = coarse(first), coarse(second)
    if not a or not b:
        return None
    return round(difflib.SequenceMatcher(None, a, b).ratio(), 3)
