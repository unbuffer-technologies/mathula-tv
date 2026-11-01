"""How far apart two sounds are, from how they are made.

The pronunciation lab compares the speaker's sounds with the voice's. Its first ruler put every vowel pair at 0.6 and every consonant
pair at 0.7, so "i" against "ɪ" (nearly one sound) cost the same as "i" against "u", and "s" against "ʃ" the same as "s" against "k".
A comparison made with that ruler cannot say which of two spellings is closer, only whether the sounds are identical.

Here a consonant is a place of articulation, a manner and a voicing, and a vowel is a height, a backness and a rounding, as in the IPA
charts. The distance grows with each step apart on those axes, so neighbours on the chart are cheap and opposite corners are dear.
A sound this table does not describe (a tone-marked syllable, a click) gives ``None`` and the caller keeps its older rule.
"""
from __future__ import annotations

from functools import lru_cache

_STRIP = "ːˑ̩̯̥̪̞̃̊ʰʲʷˤᵝ.^[]ˈˌ"
_ASCII_ALIAS = {"tS": "tʃ", "dZ": "dʒ", "S": "ʃ", "N": "ŋ", "X": "x", "ts.": "ts"}

# place of articulation, front to back: 0 bilabial .. 9 glottal; manner; voiced
_PLACE = {"bilabial": 0.0, "labiodental": 1.0, "dental": 2.0, "alveolar": 3.0, "retroflex": 3.5, "postalveolar": 4.0, "alveolopalatal": 4.5,
          "palatal": 5.0, "velar": 6.0, "uvular": 7.0, "pharyngeal": 8.0, "glottal": 9.0}
_CONSONANTS: dict[str, tuple[str, str, bool]] = {}


def _add(symbols: str, place: str, manner: str, voiced: bool) -> None:
    for symbol in symbols.split():
        _CONSONANTS[symbol] = (place, manner, voiced)


_add("p", "bilabial", "plosive", False); _add("b", "bilabial", "plosive", True)
_add("t", "alveolar", "plosive", False); _add("d", "alveolar", "plosive", True)
_add("ʈ", "retroflex", "plosive", False); _add("ɖ", "retroflex", "plosive", True)
_add("c", "palatal", "plosive", False); _add("ɟ", "palatal", "plosive", True)
_add("k", "velar", "plosive", False); _add("ɡ g", "velar", "plosive", True)
_add("q", "uvular", "plosive", False); _add("ʔ", "glottal", "plosive", False)
_add("m", "bilabial", "nasal", True); _add("n", "alveolar", "nasal", True); _add("ɳ", "retroflex", "nasal", True)
_add("ɲ", "palatal", "nasal", True); _add("ŋ", "velar", "nasal", True); _add("ɴ", "uvular", "nasal", True)
_add("ɸ", "bilabial", "fricative", False); _add("β", "bilabial", "fricative", True)
_add("f", "labiodental", "fricative", False); _add("v", "labiodental", "fricative", True)
_add("θ", "dental", "fricative", False); _add("ð", "dental", "fricative", True)
_add("s", "alveolar", "fricative", False); _add("z", "alveolar", "fricative", True)
_add("ʃ", "postalveolar", "fricative", False); _add("ʒ", "postalveolar", "fricative", True)
_add("ʂ", "retroflex", "fricative", False); _add("ʐ", "retroflex", "fricative", True)
_add("ɕ", "alveolopalatal", "fricative", False); _add("ʑ", "alveolopalatal", "fricative", True)
_add("ç", "palatal", "fricative", False); _add("ʝ", "palatal", "fricative", True)
_add("x", "velar", "fricative", False); _add("ɣ", "velar", "fricative", True)
_add("χ", "uvular", "fricative", False); _add("ʁ", "uvular", "fricative", True)
_add("ħ", "pharyngeal", "fricative", False); _add("ʕ", "pharyngeal", "fricative", True)
_add("h", "glottal", "fricative", False); _add("ɦ", "glottal", "fricative", True)
_add("ɬ", "alveolar", "lateral_fricative", False)
_add("tʃ", "postalveolar", "affricate", False); _add("dʒ", "postalveolar", "affricate", True)
_add("ts", "alveolar", "affricate", False); _add("dz", "alveolar", "affricate", True)
_add("tɕ", "alveolopalatal", "affricate", False); _add("dʑ", "alveolopalatal", "affricate", True)
_add("pf", "labiodental", "affricate", False)
_add("ɹ", "alveolar", "approximant", True); _add("ɻ", "retroflex", "approximant", True)
_add("j", "palatal", "approximant", True); _add("w", "bilabial", "approximant", True); _add("ʋ", "labiodental", "approximant", True)
_add("l ɫ", "alveolar", "lateral", True); _add("ɭ", "retroflex", "lateral", True); _add("ʎ", "palatal", "lateral", True)
_add("ɾ r ɽ", "alveolar", "tap", True)

_MANNER = {("plosive", "affricate"): 0.3, ("affricate", "fricative"): 0.3, ("plosive", "fricative"): 0.6, ("plosive", "nasal"): 0.6,
           ("fricative", "approximant"): 0.6, ("lateral", "approximant"): 0.4, ("lateral", "tap"): 0.5, ("tap", "approximant"): 0.5,
           ("plosive", "tap"): 0.8, ("fricative", "tap"): 0.8, ("lateral", "nasal"): 0.7, ("nasal", "approximant"): 0.7,
           ("lateral_fricative", "fricative"): 0.4, ("lateral_fricative", "lateral"): 0.4, ("lateral_fricative", "affricate"): 0.6,
           ("lateral_fricative", "plosive"): 0.8}
_PLACE_STEPS_FOR_FULL = 3.0       # three places apart on the chart is as far as a place can be
_WEIGHT = {"place": 0.6, "manner": 0.5, "voice": 0.25}
_MOST_CONSONANT = 0.95
_MOST_VOWEL = 0.9

# height 0 close .. 6 open, backness 0 front .. 2 back, rounded
_VOWELS: dict[str, tuple[float, float, bool]] = {
    "i": (0, 0, False), "y": (0, 0, True), "ɨ": (0, 1, False), "ʉ": (0, 1, True), "ɯ": (0, 2, False), "u": (0, 2, True),
    "ɪ": (1, 0.3, False), "ʏ": (1, 0.3, True), "ᵻ": (1, 1, False), "ʊ": (1, 1.7, True),
    "e": (2, 0, False), "ø": (2, 0, True), "ɘ": (2, 1, False), "ɵ": (2, 1, True), "ɤ": (2, 2, False), "o": (2, 2, True),
    "ə": (3, 1, False), "ɚ": (3, 1, False),
    "ɛ": (4, 0, False), "œ": (4, 0, True), "ɜ": (4, 1, False), "ɞ": (4, 1, True), "ʌ": (4, 2, False), "ɔ": (4, 2, True),
    "æ": (5, 0, False), "ɐ": (5, 1, False),
    "a": (6, 0.5, False), "ä": (6, 1, False), "ɶ": (6, 0, True), "ɑ": (6, 2, False), "ɒ": (6, 2, True),
}
_HEIGHT_FOR_FULL, _BACK_FOR_FULL, _ROUND_WEIGHT, _VOWEL_SCALE = 3.0, 1.5, 0.5, 0.55
_RHOTIC = 0.15                    # a vowel with the r-colour of "ɛɹ" against one without


@lru_cache(maxsize=4096)
def _clean(phone: str) -> str:
    phone = _ASCII_ALIAS.get(phone, phone)
    return "".join(c for c in phone if c not in _STRIP)


@lru_cache(maxsize=4096)
def describe(phone: str) -> tuple | None:
    """("consonant", place, manner, voiced) or ("vowel", ((height, back, rounded), ...), rhotic); ``None`` when not described."""
    token = _clean(phone)
    if not token or any(c.isdigit() or c in "?" for c in token):
        return None
    if token in _CONSONANTS:
        return ("consonant",) + _CONSONANTS[token]
    rhotic = "ɚ" in token or (token.endswith("ɹ") and len(token) > 1)
    body = token[:-1] if token.endswith("ɹ") and len(token) > 1 else token
    targets = []
    for c in body:
        if c in _VOWELS:
            targets.append(_VOWELS[c])
        elif c == "ɹ":
            rhotic = True
        else:
            return None
    return ("vowel", tuple(targets), rhotic) if targets else None


def _vowel_distance(a: tuple[float, float, bool], b: tuple[float, float, bool]) -> float:
    dh, db = (a[0] - b[0]) / _HEIGHT_FOR_FULL, (a[1] - b[1]) / _BACK_FOR_FULL
    dr = _ROUND_WEIGHT * (a[2] != b[2])
    return min(_MOST_VOWEL, _VOWEL_SCALE * (dh * dh + db * db + dr * dr) ** 0.5)


def voicing_partners(a: str, b: str) -> bool:
    """Two consonants made in the same place and manner that differ only in voicing (s and z, t and d)."""
    da, db = describe(a), describe(b)
    return bool(da and db and da[0] == db[0] == "consonant" and da[1:3] == db[1:3])


def distance(a: str, b: str) -> float | None:
    """0 for the same sound up to about 0.9 for opposite corners of the chart; ``None`` when either sound is not described."""
    da, db = describe(a), describe(b)
    if da is None or db is None:
        return None
    if da[0] != db[0]:
        return None                                  # a vowel against a consonant is the caller's rule (glides are its business)
    if da[0] == "consonant":
        (_, pa, ma, va), (_, pb, mb, vb) = da, db
        place = min(1.0, abs(_PLACE[pa] - _PLACE[pb]) / _PLACE_STEPS_FOR_FULL)
        manner = 0.0 if ma == mb else _MANNER.get((ma, mb), _MANNER.get((mb, ma), 1.0))
        return min(_MOST_CONSONANT, _WEIGHT["place"] * place + _WEIGHT["manner"] * manner + _WEIGHT["voice"] * (va != vb))
    ta, tb = da[1], db[1]
    size = max(len(ta), len(tb))
    ta, tb = ta + (ta[-1],) * (size - len(ta)), tb + (tb[-1],) * (size - len(tb))     # a shorter vowel is held while the other glides
    glide = sum(_vowel_distance(x, y) for x, y in zip(ta, tb)) / size
    return min(_MOST_VOWEL, glide + _RHOTIC * (da[2] != db[2]) + (0.05 if len(da[1]) != len(db[1]) else 0.0))
