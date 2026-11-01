"""The sounds of an English word, from a dictionary and not from a recording.

The pronunciation lab used to take the English side from the speaker's audio alone: the recogniser's reading of a word cut out of running
speech.  That reading is noisy at the edges (the "ko" after ANC and the "mɪ" after News were the next words' sounds) and for a letter spelled
out it can fail altogether (AI decoded as one sound for two letters).  This is the other half: what the word's sounds are in the dictionary
(CMUdict, 135,000 words), what a run of capitals is when said letter by letter, and the accents a speaker may have, so that the audio
only has to choose among readings instead of inventing one.

* ``EnglishSounds.say(written)`` gives the readings of a written word: ``how`` is ``"word"`` or ``"letters"``, ``phones`` the sounds.  A run of 2 to 5
  capitals is read as letters first (AI is A-I; "ai" the word is a sloth), and as a word as well when the dictionary has it in lower case.
* A reading comes with a variant for a speaker who does not say r after a vowel, which is how South African news readers say "court" and "party".

The phones are written the way the recogniser writes them (ɪ, ə, iː, ɑː, ɹ) so they compare with a decode by ``pronunciation_lab.align``.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Mapping

ARPABET_TO_IPA: dict[str, str] = {
    "AA": "ɑː", "AE": "æ", "AH": "ʌ", "AO": "ɔː", "AW": "aʊ", "AY": "aɪ", "B": "b", "CH": "tʃ", "D": "d", "DH": "ð", "EH": "ɛ", "ER": "ɜː",
    "EY": "eɪ", "F": "f", "G": "ɡ", "HH": "h", "IH": "ɪ", "IY": "iː", "JH": "dʒ", "K": "k", "L": "l", "M": "m", "N": "n", "NG": "ŋ",
    "OW": "oʊ", "OY": "ɔɪ", "P": "p", "R": "ɹ", "S": "s", "SH": "ʃ", "T": "t", "TH": "θ", "UH": "ʊ", "UW": "uː", "V": "v", "W": "w",
    "Y": "j", "Z": "z", "ZH": "ʒ",
}
_UNSTRESSED = {"AH": "ə", "ER": "ɚ"}          # the reduced vowels: AH0 is the schwa, ER0 the r-coloured schwa

# the names of the letters, as a speaker says them one by one ("ANC" is eɪ ɛn siː); W is "double-u" and Z is "zed" or "zee"
_LETTER_NAMES = {
    "A": "EY", "B": "B IY", "C": "S IY", "D": "D IY", "E": "IY", "F": "EH F", "G": "JH IY", "H": "EY CH", "I": "AY", "J": "JH EY",
    "K": "K EY", "L": "EH L", "M": "EH M", "N": "EH N", "O": "OW", "P": "P IY", "Q": "K Y UW", "R": "AA R", "S": "EH S", "T": "T IY",
    "U": "Y UW", "V": "V IY", "W": "D AH B AH L Y UW", "X": "EH K S", "Y": "W AY", "Z": "Z EH D",
}
LETTER_NAMES: dict[str, list[str]] = {letter: names.split() for letter, names in _LETTER_NAMES.items()}
LETTER_VARIANTS: dict[str, list[list[str]]] = {"Z": [["Z", "IY"]]}          # the other way a letter is named
MAX_INITIALISM = 5


VOWELS_ARPABET = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}


def to_ipa(arpabet: list[str]) -> list[str]:
    """CMUdict's stress-marked sounds as the recogniser's symbols; an unstressed AH or ER is the schwa."""
    phones: list[str] = []
    for token in arpabet:
        base, stress = token.rstrip("012"), token[len(token.rstrip("012")):]
        phones.append(_UNSTRESSED[base] if stress == "0" and base in _UNSTRESSED else ARPABET_TO_IPA[base])
    return phones


def stress_of(arpabet: list[str]) -> list[int | None]:
    """CMUdict's stress for each sound: 1 primary, 2 secondary, 0 none for a vowel; None for a consonant."""
    return [int(token[-1]) if token[-1:] in "012" else None for token in arpabet]


def non_rhotic(phones: list[str], stress: list[int | None] | None = None):
    """The same word for a speaker who says r only before a vowel: "court" kɔːt, "party" pɑːti, "letter" lɛtə.  With ``stress`` (one entry per
    sound) it returns the sounds and their stress, the stress following each sound that stays."""
    out: list[str] = []
    kept: list[int | None] = []
    for index, phone in enumerate(phones):
        if phone == "ɹ" and not (index + 1 < len(phones) and _starts_with_vowel(phones[index + 1])):
            continue
        out.append("ə" if phone == "ɚ" else phone)
        kept.append(stress[index] if stress is not None else None)
    return out if stress is None else (out, kept)


def _starts_with_vowel(phone: str) -> bool:
    return phone[:1] in "aeiouɑɒɔəɛɜɪʊʌæɐøɨ"


def is_initialism(token: str) -> bool:
    clean = re.sub(r"[^\w']", "", token)
    return 2 <= len(clean) <= MAX_INITIALISM and clean.isalpha() and clean.isupper()


def letter_name_sounds(written: str) -> set[tuple[str, ...]]:
    """How a word said letter by letter sounds, as ARPAbet without stress ("AI" is EY AY; Z has two names, so Z gives two); empty for anything else."""
    if not is_initialism(written):
        return set()
    spoken: list[tuple[str, ...]] = [()]
    for letter in (c for c in written.upper() if c.isalpha()):
        if letter not in LETTER_NAMES:
            return set()
        spoken = [done + tuple(name) for done in spoken for name in (LETTER_NAMES[letter], *LETTER_VARIANTS.get(letter, []))]
    return set(spoken)


_guesser: Callable[[str], list[str]] | None | str = "unset"


def g2p_guesser() -> Callable[[str], list[str]] | None:
    """A model's guess at the sounds of a word the dictionary lacks (g2p_en, trained on CMUdict: it reads Litchfield as L IH1 CH F IY0 L D); None when
    g2p_en is not installed.  Loaded once per process."""
    global _guesser
    if _guesser == "unset":
        try:
            from g2p_en import G2p

            _guesser = G2p()
        except Exception:                                   # not installed, or its data cannot be fetched: no guesses, never a failure
            _guesser = None
    return _guesser                                         # type: ignore[return-value]


class EnglishSounds:
    def __init__(self, words: Mapping[str, list[list[str]]], guesser: Callable[[str], list[str]] | None = None) -> None:
        self.words = words                       # lower-case word -> every dictionary reading (CMUdict's own variants), as ARPAbet
        self.guesser = guesser                   # what to ask about a word the dictionary lacks; the dictionary lookups themselves never use it

    @classmethod
    def load(cls, path: Path) -> "EnglishSounds | None":
        """Read CMUdict's ``cmudict.dict`` (``word PH PH ...``, a second reading as ``word(2) ...``, comments after ``#``); None when it is missing."""
        try:
            lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return None
        words: dict[str, list[list[str]]] = {}
        for line in lines:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            head, *phones = line.split()
            words.setdefault(re.sub(r"\(\d+\)$", "", head.lower()), []).append(phones)
        return cls(words)

    def word(self, written: str) -> list[list[str]]:
        """The dictionary readings of one word as recogniser symbols; [] when the dictionary does not have it."""
        key = re.sub(r"^[^\w']+|[^\w']+$", "", written).lower()
        return [to_ipa(reading) for reading in self.words.get(key, [])]

    def letters(self, written: str) -> list[list[str]]:
        """The word said letter by letter: one reading, and another where a letter has a second name (Z is zed or zee)."""
        letters = [c for c in written.upper() if c.isalpha()]
        readings: list[list[str]] = [[]]
        for letter in letters:
            names = [LETTER_NAMES[letter], *LETTER_VARIANTS.get(letter, [])] if letter in LETTER_NAMES else []
            readings = [done + name for done in readings for name in names]
        return [to_ipa(reading) for reading in readings if reading]

    def _letters_arpabet(self, written: str) -> list[list[str]]:
        """The letter names of a word, as ARPAbet with a stress digit on each vowel: English stresses the LAST letter of a word said letter by letter
        (A-I has its weight on the I, H-I-V on the V) and the letters before it carry secondary stress."""
        letters = [c for c in written.upper() if c.isalpha() and c in LETTER_NAMES]
        readings: list[list[list[str]]] = [[]]
        for position, letter in enumerate(letters):
            digit = "1" if position == len(letters) - 1 else "2"
            names = [LETTER_NAMES[letter], *LETTER_VARIANTS.get(letter, [])]
            marked = [[t + digit if t in VOWELS_ARPABET else t for t in name] for name in names]
            readings = [done + [name] for done in readings for name in marked]
        return [[token for name in reading for token in name] for reading in readings if reading]

    def guess(self, written: str) -> list[list[str]]:
        """A model's guess at the sounds (ARPAbet) of a word the dictionary does not have; [] when there is no guesser or it fails."""
        key = re.sub(r"[^a-z]", "", written.lower())
        if self.guesser is None or len(key) < 2:
            return []
        try:
            tokens = [t for t in self.guesser(key) if t.rstrip("012") in ARPABET_TO_IPA]
        except Exception:
            return []
        return [tokens] if tokens else []

    def say(self, written: str, *, guess: bool = False) -> list[dict[str, Any]]:
        """Every reading worth choosing between for a written word, as ``{"how", "phones", "stress", "accent"}``, the likeliest first; ``stress``
        has one entry per sound (1 primary, 2 secondary, 0 none for a vowel, None for a consonant).

        A run of 2 to 5 capitals is said by its letters; when the dictionary also knows it as a word ("NEWS", "COVID") that reading follows.  With
        ``guess`` a word the dictionary lacks gets the guesser's reading, labelled ``"guessed"``: the caller decides which words may be guessed (a Zulu
        name read by an English model comes out English)."""
        key = re.sub(r"^[^\w']+|[^\w']+$", "", written).lower()
        spelled = [("letters", reading) for reading in self._letters_arpabet(written)] if is_initialism(written) else []
        spoken = [("word", reading) for reading in self.words.get(key, [])]
        if guess and not spelled and not spoken:
            spoken = [("guessed", reading) for reading in self.guess(written)]
        readings: list[dict[str, Any]] = []
        for how, arpabet in spelled + spoken:
            phones, stress = to_ipa(arpabet), stress_of(arpabet)
            plain, plain_stress = non_rhotic(phones, stress)
            for accent, variant, marks in (("rhotic", phones, stress), ("non_rhotic", plain, plain_stress)):
                if not any(r["phones"] == variant for r in readings):
                    readings.append({"how": how, "phones": variant, "stress": marks, "accent": accent})
        return readings
