"""Creating a Zulu word for an English word that has none: loanword adaptation.

When a word comes into isiZulu it is not imitated, it is rebuilt in Zulu syllables by Zulu rules, and the result is a Zulu word that a Zulu
voice reads natively and a Zulu listener hears as Zulu.  That is how "science" became isayensi, "bus" ibhasi, "spoon" isipunu, "flag" ifulegi.
This module does the same for any English word, from the sounds the speaker made (the phone recogniser's reading of his own audio), in four steps:

1. each English sound is mapped to the Zulu letters that write the nearest Zulu sound (English p, t, k, b are written ph, th, kh, bh: the
   aspirated and plosive sounds Zulu hears in them; the five-vowel system collapses English vowels into a, e, i, o, u);
2. Zulu syllables are open (consonant plus vowel), so a consonant cluster Zulu does not have, or a consonant at the end of the word, gets a vowel
   inserted: ``u`` after a labial or velar consonant, ``i`` after the others; a final vowel copies a round vowel before it (isipunu) and is ``i``
   after the others (ibhasi);
3. homorganic nasal clusters (nd, nt, mb, ng, ns, nts) and consonant plus y / w are Zulu units already and stay as they are;
4. every choice that has an alternative (ph or p, a or e for a reduced vowel, u or i for an inserted vowel) is kept as a variant, most typical
   first.

It returns spellings of the word itself.  The noun prefix the sentence needs (i-, isi-, ama-, u-) is the translator's, though ``with_i`` shows the
common class 5 / 9 form.  Names, acronyms and titles are not adapted: they keep their original sound.  The candidates are only candidates:
the lab speaks and measures them before the agent chooses.
"""
from __future__ import annotations

import re
from typing import Any, Sequence

# English consonant -> Zulu letters, the most typical first
CONSONANTS: dict[str, list[str]] = {
    "p": ["ph", "p"], "b": ["bh", "b"], "t": ["th", "t"], "d": ["d"], "k": ["kh", "k"], "g": ["g"],
    "tʃ": ["tsh", "ch"], "dʒ": ["j"], "f": ["f"], "v": ["v"], "θ": ["th", "s", "t"], "ð": ["d", "z"], "s": ["s"], "z": ["z"],
    "ʃ": ["sh"], "ʒ": ["j", "sh"], "h": ["h"], "m": ["m"], "n": ["n"], "ŋ": ["ng"], "l": ["l"], "r": ["r"], "j": ["y"], "w": ["w"],
    "ts": ["ts"], "dz": ["dz", "z"], "ɬ": ["hl"], "ç": ["hl"], "ɕ": ["dl"], "x": ["kh", "h"],
}
# English vowel -> Zulu letters, the most typical first
VOWELS: dict[str, list[str]] = {
    "a": ["a"], "ɑ": ["a"], "ɐ": ["a"], "ʌ": ["a"], "æ": ["a", "e"], "ɛ": ["e"], "e": ["e"], "ɜ": ["e", "a"], "ə": ["a", "e", "i"],
    "ɪ": ["i"], "i": ["i"], "ɔ": ["o"], "o": ["o"], "ɒ": ["o", "a"], "ʊ": ["u"], "u": ["u"],
    "aɪ": ["ayi", "ai"], "eɪ": ["e", "ei"], "aʊ": ["au", "awu"], "oʊ": ["o", "ou"], "əʊ": ["o", "ou"], "ɔɪ": ["oyi", "oi"], "əɪ": ["ayi", "ai"],
}
_ALIASES = {"ɡ": "g", "ɹ": "r", "ɾ": "r", "ɻ": "r", "ɚ": "ə", "ɝ": "ɜ", "ɨ": "i", "ɵ": "ə"}
_MARKS = "ːˑʰʲʷ̩̯̃'ˈˌ"
_LABIAL_OR_VELAR = {"p", "b", "m", "f", "v", "k", "g", "w", "ŋ"}
_ROUND = {"u", "ʊ", "o", "ɔ", "ɒ", "oʊ", "əʊ"}
_NASAL_PARTNERS = {"m": {"p", "b", "f", "v"}, "n": {"t", "d", "s", "z", "ts", "dz", "tʃ", "dʒ", "k", "g", "ʃ", "θ"}, "ŋ": {"k", "g"}}
BEAM = 80


def _normal(phone: str) -> str:
    phone = re.sub(r"[0-9]", "", str(phone))
    phone = "".join(c for c in phone if c not in _MARKS)
    return _ALIASES.get(phone, phone)


def _legal_cluster(first: str, second: str) -> bool:
    """Whether Zulu already has this two-consonant unit: a nasal and a homorganic consonant (nd, mb, ng, ns, nts), or a consonant and y / w."""
    return second in _NASAL_PARTNERS.get(first, set()) or second in {"j", "w"}


def _sequence(phones: Sequence[str]) -> list[tuple[str, Any]]:
    """The word as a list of ("C" | "V" | "E" | "F", options): sounds with their Zulu letters, and the vowels Zulu syllables need inserted."""
    sounds = [_normal(p) for p in phones]
    sounds = [p for p in sounds if p in CONSONANTS or p in VOWELS]
    sequence: list[tuple[str, Any]] = []
    last_vowel = ""
    for index, sound in enumerate(sounds):
        if sound in VOWELS:
            sequence.append(("V", (sound, VOWELS[sound])))
            last_vowel = sound
            continue
        sequence.append(("C", (sound, CONSONANTS[sound])))
        following = sounds[index + 1] if index + 1 < len(sounds) else None
        if following is None:
            if sound in {"j", "w"}:
                continue
            sequence.append(("F", (sound, ["u", "i"] if last_vowel in _ROUND else ["i", "u"])))              # a word cannot end on a consonant
        elif following in CONSONANTS and not _legal_cluster(sound, following):
            sequence.append(("E", (sound, ["u", "i"] if sound in _LABIAL_OR_VELAR else ["i", "u"])))        # a cluster Zulu does not have
    return sequence


def _syllables(spelling: str) -> int:
    return len(re.findall(r"[aeiou]+", spelling))


def adapt(phones: Sequence[str], *, limit: int = 12) -> list[dict[str, Any]]:
    """Zulu spellings of the word whose sounds are ``phones``, most typical first: {"stem", "with_i", "syllables", "choices"}.

    ``choices`` names each place where the spelling took an alternative to the most typical letters, so the agent can see what a variant changes."""
    sequence = _sequence(phones)
    if not sequence:
        return []
    beam: list[tuple[float, str, list[str]]] = [(0.0, "", [])]
    for kind, (sound, options) in sequence:
        grown: list[tuple[float, str, list[str]]] = []
        for cost, text, notes in beam:
            for rank, letters in enumerate(options):
                extra = [f"{sound} written {letters}" if kind in ("C", "V") else f"vowel {letters} {'inserted' if kind == 'E' else 'at the end'} after {sound}"] if rank else []
                grown.append((cost + rank, text + letters, notes + extra))
        beam = sorted(grown, key=lambda item: (item[0], len(item[1]), item[1]))[:BEAM]
    found: dict[str, dict[str, Any]] = {}
    for cost, text, notes in beam:
        text = re.sub(r"(ay|oy)i(?=[aeiou])", r"\1", text)                                    # a glide is enough before another vowel (sayensi, not sayiensi)
        found.setdefault(text, {"stem": text, "with_i": "i" + text, "syllables": _syllables(text), "choices": notes, "_cost": cost})
    ranked = sorted(found.values(), key=lambda item: (item["_cost"], len(item["stem"]), item["stem"]))[:limit]
    for item in ranked:
        item.pop("_cost")
    return ranked
