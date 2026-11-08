"""English sounds to isiZulu spelling by rule: the English-to-Zulu half of the interpreter that does not depend on a voice.

``interpreter`` finds letters that make a voice say a sound (measured on the voice).  This is what a person who knows both languages does first: the rules by which
a Zulu speaker, or a Zulu spelling, takes in an English sound.  They are few, and the spelling "eyi-aayi" for AI is just their result:

1. ``sounds``: AI is the names of its letters, eɪ and aɪ (a word comes from the dictionary as sounds, with stress).
2. No diphthongs: eɪ is a vowel and a glide and, at the end of a syllable, a vowel to rest on, so e + y + i = "eyi"; aɪ is a + y + i, aʊ is a + w + u.
3. Open syllables: every consonant is followed by a vowel.  A consonant with none (the f of "eff", a cluster's first consonant) gets one, i by default, u after a
   rounded sound.
4. Stress is length: English stresses the last letter of an initialism (the I of AI, the V of HIV) and the last syllable of a word that ends in a stressed
   vowel, and Zulu spelling shows length by doubling.  The vowel of a stressed long sound (iː, uː) is written three times ("yuuu"), any other twice ("aayi");
   the syllables before it stay short ("pi", "si").  Inside a word a long vowel is long too: the ɔː of "court" is written twice ("kooti"), the ɑː of "party"
   ("paati"); a reduced vowel, a short one and the letters of an initialism before its last stay single.
5. Letters are separate syllables: the letters of an initialism are written apart (a hyphen), so that Zulu spelling does not run them together.

Where a rule has more than one reasonable result (the epenthetic vowel, how long to hold, a glide or a full ending) the results are ``variants``, ordered by how
many departures from the main choice they make; the caller speaks them in a voice and measures, because only the voice says which one it reads right.
"""
from __future__ import annotations

import itertools
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

CONSONANTS: dict[str, list[str]] = {
    "p": ["p"], "b": ["b"], "t": ["t"], "d": ["d"], "k": ["k"], "ɡ": ["g"], "g": ["g"], "tʃ": ["tsh", "ch"], "dʒ": ["j"], "f": ["f"], "v": ["v"], "θ": ["th", "s"],
    "ð": ["d", "dh"], "s": ["s"], "z": ["z"], "ʃ": ["sh"], "ʒ": ["zh", "j"], "h": ["h"], "m": ["m"], "n": ["n"], "ŋ": ["ng"], "l": ["l"], "ɹ": ["r"], "r": ["r"],
    "w": ["w"], "j": ["y"], "ɓ": ["b"], "ɾ": ["r"], "ɬ": ["hl"], "ɮ": ["dl"], "x": ["kh"], "ç": ["h"], "ɲ": ["ny"], "ʋ": ["v"], "ɦ": ["h"],
    "pʰ": ["ph", "p"], "tʰ": ["th", "t"], "kʰ": ["kh", "k"]}
ASPIRATED = {"p": "pʰ", "t": "tʰ", "k": "kʰ"}          # English p, t, k open a stressed syllable with a puff of air; Zulu spelling writes that ph, th, kh
VOWELS: dict[str, list[str]] = {
    "iː": ["i"], "ɪ": ["i"], "ɛ": ["e"], "e": ["e"], "æ": ["a", "e"], "ʌ": ["a"], "ɑː": ["a"], "ɑ": ["a"], "ɒ": ["o"], "ɔː": ["o"], "ʊ": ["u"], "uː": ["u"],
    "ə": ["a", "e"], "ɚ": ["a", "e"], "ɜː": ["e", "a"], "a": ["a"], "i": ["i"], "o": ["o"], "u": ["u"], "ɔ": ["o"], "ɐ": ["a"], "ɜ": ["e", "a"], "eː": ["e"],
    "oː": ["o"], "aː": ["a"], "ɛː": ["e"], "y": ["i"], "ø": ["e"]}
DIPHTHONGS: dict[str, tuple[str, str, str]] = {          # vowel, glide, vowel to rest on
    "eɪ": ("e", "y", "i"), "aɪ": ("a", "y", "i"), "ɔɪ": ("o", "y", "i"), "aʊ": ("a", "w", "u"), "oʊ": ("o", "w", "u")}
LONG = frozenset({"iː", "uː", "ɑː", "ɔː", "ɜː", "eː", "oː", "aː", "ɛː"})
IPA_UNITS = ("tʃ", "dʒ", "eɪ", "aɪ", "ɔɪ", "aʊ", "əʊ", "oʊ")             # sounds written with two symbols
IPA_DROPPED = set("ʰʷʲˠʼ̩̥̯̃̊ˑ.‿͜͡ʔ/[]()")                                     # aspiration, syllabic marks, syllable breaks, brackets: no sound of their own here
IPA_ALIASES = {"əʊ": "oʊ", "ɡ": "ɡ", "g": "ɡ", "ɝ": "ɚ", "ɫ": "l", "ɹ": "ɹ", "ɻ": "ɹ", "ʍ": "w"}
ROUND = frozenset({"u", "o"})
LABIAL = frozenset({"p", "b", "f", "v", "m", "w"})
GLIDE_ENDING = {"i": "yi", "u": "wu"}          # a held long i or u as a vowel and a glide: "siyi", "yuwu"


@dataclass(frozen=True)
class Choice:
    """One way through the rules' open choices; the defaults are the main choice."""
    epen: str = "i"                 # the vowel added to a consonant with none: "i", or "u" after a rounded sound
    diph: str = "full"              # a diphthong at the end: "full" (eyi) or "glide" (ey)
    hold: int = 1                   # 1 as the rule says, 0 not held, 2 one letter longer
    hold_form: str = "repeat"       # a held iː or uː: the vowel repeated (siii) or a glide added (siyi)
    cons: int = 0                   # 0 the usual letters for a consonant, 1 its other spelling (ch for tsh)
    aspirate: int = 1               # 1 p, t, k opening a stressed syllable are written ph, th, kh; 0 plain
    final_vowel: int = 0            # 0 a word's last consonant stays bare (Jacob: jeykab), as Zulu speakers say the English words their text keeps in English spelling;
                                    # 1 it gets a vowel after it (jeykabi), as a rebuilt Zulu loanword would
    cluster_vowel: int = 0          # 0 consonants that come together stay together (Western: westan); 1 a vowel breaks them (wesitan)
    letter_vowel: int = 1           # 1 a reduced vowel is written with the letter the English spelling has (Jacob: jeykob); 0 as a (jeykab)
    yod: int = 1                    # 1 uː after n, t, d keeps the y South African English says (nyuuz); 0 as the American dictionary has it (nuuz)
    short_stress: int = 1          # 1 the short vowel that carries the word's main stress is written twice too (WhatsApp: wootsap); 0 once
    length: int = 1                 # 1 a stressed long vowel inside a word is written twice (kooti), 0 once (koti)
    join: str = "-"                 # between the letters of an initialism: "-" or " " (the pieces then begin with a capital)


@dataclass
class Syl:
    onset: str
    vowel: str
    glide: str = ""
    ending: str = ""
    coda: str = ""                  # consonants kept bare after the vowel at the end of a word
    long: bool = False
    stress: int | None = None
    epenthetic: bool = False


def _consonant(phone: str, choice: Choice) -> str:
    options = CONSONANTS.get(phone, [phone])
    return options[min(choice.cons, len(options) - 1)]


def _units(phones: Sequence[str], stress: Sequence[int | None], choice: Choice, notes: list[str] | None, *, bare_final: bool = False, keep_clusters: bool = False) -> list[Syl]:
    """The sounds of one stretch (a word or one letter name) as Zulu syllables, each a consonant (or none) and a vowel.  With ``bare_final`` the consonants that end
    a word are kept as they are, and with ``keep_clusters`` so are the consonants that come together inside it: the English word is said as English, as Zulu speakers say
    the names and words their text keeps in English spelling."""
    units: list[Syl] = []
    pending: list[str] = []
    previous = ""

    def epenthetic(consonant: str) -> str:
        rounded = choice.epen == "u" and (consonant in LABIAL or previous in ROUND)
        if notes is not None:
            notes.append(f"{consonant} has no vowel after it: {'u' if rounded else 'i'} is added (Zulu syllables are open)")
        return "u" if rounded else "i"

    for phone, weight in zip(phones, stress):
        if phone in VOWELS or phone in DIPHTHONGS:
            joined = ""
            if keep_clusters:
                joined = "".join(_consonant(consonant, choice) for consonant in pending[:-1])          # the earlier consonants of a cluster stay in front of the last
            else:
                for consonant in pending[:-1]:
                    units.append(Syl(_consonant(consonant, choice), epenthetic(consonant), epenthetic=True))
            first = pending[-1] if pending else ""
            if first in ASPIRATED and weight == 1 and choice.aspirate and not (len(pending) >= 2 and pending[-2] == "s"):          # not after s: "school" has a plain k
                if notes is not None:
                    notes.append(f"{first} opens a stressed syllable, so it has a puff of air in English: Zulu writes it {_consonant(ASPIRATED[first], choice)}")
                first = ASPIRATED[first]
            onset = joined + (_consonant(first, choice) if first else "")
            if joined and notes is not None:
                notes.append(f"{' '.join(pending)} come together in English and stay together")
            if phone in DIPHTHONGS:
                vowel, glide, ending = DIPHTHONGS[phone]
                units.append(Syl(onset, vowel, glide, ending, stress=weight))
                if notes is not None:
                    notes.append(f"{phone} is one sound in English and none in Zulu: {vowel} + {glide}" + (f" + {ending}" if ending else ""))
            else:
                units.append(Syl(onset, VOWELS[phone][min(choice.cons, len(VOWELS[phone]) - 1)] if phone in ("æ", "ə", "ɚ", "ɜː") else VOWELS[phone][0],
                                 long=phone in LONG, stress=weight))
            previous, pending = units[-1].vowel, []
        else:
            pending.append(phone)
    if pending and bare_final and units:
        units[-1].coda = "".join(_consonant(consonant, choice) for consonant in pending)
        if notes is not None:
            notes.append(f"{' '.join(pending)} ends the word and stays as it is: Zulu speakers say English words and names with their last consonant")
    else:
        for consonant in pending:
            units.append(Syl(_consonant(consonant, choice), epenthetic(consonant), epenthetic=True))
            previous = units[-1].vowel
    # a diphthong with a syllable after it keeps its glide only: the next onset is the break; at the end it also gets the vowel to rest on
    for index, unit in enumerate(units):
        if unit.glide and (index < len(units) - 1 or choice.diph == "glide" or unit.coda) and unit.ending:
            unit.ending = ""
    return units


def _write(units: list[Syl], choice: Choice, *, held: bool, final_only: bool, notes: list[str] | None, double_long: bool = False) -> str:
    """The units as letters; the stressed vowel of the last syllable carrying primary stress is held when ``held`` (``final_only``: only when that syllable ends
    the stretch, as in "eye": a word's stress inside the word is not written as length)."""
    target = max((i for i, u in enumerate(units) if u.stress == 1 and not u.epenthetic), default=None) if held else None
    if final_only and target is not None and (target != len(units) - 1 or units[target].coda):          # a closed final syllable ("court") is not held, only a long vowel doubled
        target = None
    text = ""
    for index, unit in enumerate(units):
        reps = 1
        if double_long and choice.length and unit.long and unit.stress in (1, 2) and index != target:
            reps = 2
            if notes is not None:
                notes.append(f"the stressed long vowel is long in Zulu spelling: {unit.vowel} is written twice")
        elif double_long and choice.short_stress and not unit.long and not unit.glide and unit.stress == 1 and not unit.epenthetic and index != target:
            reps = 2
            if notes is not None:
                notes.append(f"the vowel with the main stress is stretched in Zulu speech: {unit.vowel} is written twice")
        if index == target and choice.hold:
            reps = (3 if unit.long else 2) + (1 if choice.hold == 2 else 0)
            if notes is not None:
                notes.append(f"stressed and last, so held: the vowel {unit.vowel} is written {reps} times" if choice.hold_form == "repeat" else "stressed and last, so held with a glide")
        if index == target and choice.hold and choice.hold_form == "glide" and unit.long and unit.vowel in GLIDE_ENDING:
            text += unit.onset + unit.vowel + GLIDE_ENDING[unit.vowel] + unit.coda
        else:
            text += unit.onset + unit.vowel * reps + unit.glide + unit.ending + unit.coda
    return text


# What each open choice of the rules means, in words a listener can read in the console: its label, and what each of its options does to a spelling.
RULE_LABELS: dict[str, tuple[str, dict[Any, str]]] = {
    "epen": ("Vowel added after a consonant that has none", {"i": "i", "u": "u after a rounded sound"}),
    "diph": ("A diphthong at the end of a letter or word", {"full": "vowel + glide + vowel to rest on (eyi)", "glide": "vowel + glide only (ey)"}),
    "hold": ("How long the stressed last letter of an initialism is held", {1: "as the rule says (aayi)", 0: "not held (ayi)", 2: "one letter longer"}),
    "hold_form": ("How a held long i or u is held", {"repeat": "the vowel repeated (siii)", "glide": "a glide added (siyi)"}),
    "cons": ("A consonant with two spellings", {0: "the usual one (tsh, kh)", 1: "the other one (ch, k)"}),
    "aspirate": ("p, t, k that open a stressed syllable", {1: "written ph, th, kh", 0: "written plain p, t, k"}),
    "final_vowel": ("The last consonant of a word", {0: "stays bare (jeykob)", 1: "gets a vowel after it (jeykobi)"}),
    "cluster_vowel": ("Consonants that come together", {0: "stay together (westen)", 1: "a vowel breaks them (wesiten)"}),
    "letter_vowel": ("A reduced vowel and the vowels of lot and palm", {1: "follow the letter of the English spelling (jeykob)", 0: "are the English sound (jeykab)"}),
    "yod": ("The y of news, Tuesday, student", {1: "kept, as South African English says it (nyuuz)", 0: "dropped (nuuz)"}),
    "short_stress": ("A short vowel with the main stress", {1: "written twice (waatsap)", 0: "written once (watsap)"}),
    "length": ("A stressed long vowel inside a word", {1: "written twice (khoot)", 0: "written once (khot)"}),
    "join": ("Between the letters of an initialism", {"-": "a hyphen (eyi-aayi)", " ": "a space, each letter capitalised (Eyi Aayi)"}),
}
RULES_CONFIG_ENV = "MATHULA_TV_RULES_CONFIG"
_config_seen: dict[str, Any] = {"path": None, "mtime": None, "defaults": {}}


def rules_config_path() -> Path:
    """Where the listener's choices of default are kept: the console writes it, the rules read it (``working/rules_config.json`` next to the code unless the
    environment says otherwise)."""
    return Path(os.environ.get(RULES_CONFIG_ENV) or Path(__file__).resolve().parents[2] / "working" / "rules_config.json")


def rule_defaults() -> dict[str, Any]:
    """The default picked for each open choice by the listener ({} = the rules as written); re-read when the file changes."""
    path = rules_config_path()
    try:
        stat = path.stat()
        mtime = (stat.st_mtime_ns, stat.st_size)           # a timestamp alone can repeat between two quick writes on a coarse clock (Windows): the size helps, and a write here forgets the cache
    except OSError:
        mtime = None
    if _config_seen["path"] != str(path) or _config_seen["mtime"] != mtime:
        defaults: dict[str, Any] = {}
        if mtime is not None:
            try:
                value = json.loads(path.read_text(encoding="utf-8")).get("defaults") or {}
                defaults = {name: pick for name, pick in value.items() if name in RULE_LABELS and pick in RULE_LABELS[name][1]}
            except (OSError, ValueError, AttributeError):
                defaults = {}
        _config_seen.update(path=str(path), mtime=mtime, defaults=defaults)
    return dict(_config_seen["defaults"])


def write_rule_default(name: str, pick: Any, *, path: Path | None = None, note: str = "") -> dict[str, Any]:
    """Make ``pick`` the default of one open choice (``pick=None`` puts it back to the rule as written).  Keeps the history of changes with the reason."""
    if name not in RULE_LABELS:
        raise ValueError(f"unknown rule: {name}")
    if pick is not None and pick not in RULE_LABELS[name][1]:
        raise ValueError(f"{name} has no option {pick!r}")
    path = Path(path) if path else rules_config_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    defaults = dict(data.get("defaults") or {})
    before = defaults.get(name)
    if pick is None:
        defaults.pop(name, None)
    else:
        defaults[name] = pick
    history = list(data.get("history") or [])
    history.append({"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "rule": name, "from": before, "to": pick, "note": note})
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"defaults": defaults, "history": history[-200:]}, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    _config_seen.update(path=None, mtime=None)             # this process re-reads the file on its next question, whatever the file's timestamp says
    return defaults


def choice_differences(main: Choice, other: Choice) -> list[dict[str, Any]]:
    """The open choices on which ``other`` departs from ``main``, with words for each."""
    found = []
    for name, (label, options) in RULE_LABELS.items():
        a, b = getattr(main, name), getattr(other, name)
        if a != b:
            found.append({"rule": name, "label": label, "main": a, "other": b, "main_means": options.get(a, str(a)), "other_means": options.get(b, str(b))})
    return found


def _choices(letters: bool, held: bool) -> list[Choice]:
    """Every combination of the open choices, the main choice first and then those that depart from it least.  The main choice is the rules as written,
    except where the listener has picked another default for a choice (``rule_defaults``)."""
    dims = [("epen", ["i", "u"]), ("diph", ["full", "glide"]), ("hold", [1, 0, 2] if held else [1]), ("hold_form", ["repeat", "glide"] if held else ["repeat"]),
            ("cons", [0, 1]), ("length", [1] if letters else [1, 0]), ("aspirate", [1] if letters else [1, 0]), ("final_vowel", [0, 1]), ("cluster_vowel", [0] if letters else [0, 1]), ("letter_vowel", [0] if letters else [1, 0]), ("short_stress", [0] if letters else [1, 0]), ("yod", [0] if letters else [1, 0]),
            ("join", ["-", " "] if letters else ["-"])]
    chosen = rule_defaults()
    dims = [(name, [chosen[name], *[o for o in options if o != chosen[name]]] if chosen.get(name) in options else options) for name, options in dims]
    combos = itertools.product(*[range(len(options)) for _, options in dims])
    ordered = sorted(combos, key=lambda picks: (sum(1 for p in picks if p), picks))
    return [Choice(**{name: options[pick] for (name, options), pick in zip(dims, picks)}) for picks in ordered]


REDUCED = ("ə", "ɚ")
LETTERED = ("ɒ", "ɑ", "ɑː", "oʊ")        # the vowels of "what" and "October": a Zulu speaker reads the letter, a (waatsap) and o (okthowba), not the English vowel


def _letter_vowels(phones: Sequence[str], written: str | None) -> tuple[list[str], list[str]]:
    """A reduced vowel (schwa) has no letter of its own in English, and a Zulu speaker who has the word in front of them says the vowel the spelling shows:
    Jacob is jeykob, not jeykab.  The vowel letters of the written word are matched to the vowels of the sounds in order; when their numbers differ
    (a silent final e is dropped first) nothing is matched and the schwa stays as it was.  er at the end is a (October: oktoba), as English without r says it."""
    phones = list(phones)
    if not written or not any(p in REDUCED or p in LETTERED for p in phones):
        return phones, []
    word = re.sub(r"[^a-z]", "", written.lower())
    spans = [m for m in re.finditer(r"(?<!^)[aeiouy]+|^[aeiou]+", word)]
    nuclei = [i for i, p in enumerate(phones) if p in VOWELS or p in DIPHTHONGS]
    if len(spans) == len(nuclei) + 1 and spans[-1].group() == "e" and spans[-1].end() == len(word):
        spans = spans[:-1]
    if len(spans) != len(nuclei):
        return phones, []
    notes: list[str] = []
    for index, span in zip(nuclei, spans):
        if phones[index] not in REDUCED and not (phones[index] in LETTERED and len(span.group()) == 1):          # the lot and palm vowels follow a single written letter only
            continue
        letter = span.group()[0]
        letter = "i" if letter == "y" else letter
        if letter == "e" and word[span.end():span.end() + 1] == "r" and (span.end() + 1 == len(word) or phones[index] in LETTERED):          # er: October's last syllable, and Derby's (the vowel of "palm")
            letter = "a"
        notes.append(f"the {'reduced vowel' if phones[index] in REDUCED else 'vowel ' + phones[index]} of {written} is written with the letter the English spelling has: {letter} (a person reading the word says it)")
        phones[index] = letter
    return phones, notes


def _yod(phones: Sequence[str], stress: Sequence[int | None], written: str | None) -> tuple[list[str], list[int | None], list[str]]:
    """South African English keeps the y of "news", "Tuesday", "student": uː after n, t, d or th that the spelling writes ew, eu, ue or u is nyuuz, not nuuz (the American
    dictionary leaves it out).  The vowels of the written word are matched to the sounds as for the reduced vowel; nothing is changed when their numbers differ."""
    phones, stress = list(phones), list(stress)
    if not written or "uː" not in phones:
        return phones, stress, []
    word = re.sub(r"[^a-z]", "", written.lower())
    spans = list(re.finditer(r"(?<!^)[aeiouy]+|^[aeiou]+", word))
    nuclei = [i for i, p in enumerate(phones) if p in VOWELS or p in DIPHTHONGS]
    if len(spans) == len(nuclei) + 1 and spans[-1].group() == "e" and spans[-1].end() == len(word):
        spans = spans[:-1]
    if len(spans) != len(nuclei):
        return phones, stress, []
    notes: list[str] = []
    for index, span in reversed(list(zip(nuclei, spans))):
        written_vowel = span.group() + ("w" if word[span.end():span.end() + 1] == "w" else "")          # ew is the vowel letter e and the w after it
        if phones[index] == "uː" and index > 0 and phones[index - 1] in ("n", "t", "d", "θ") and written_vowel in ("ew", "eu", "ue", "u"):
            phones.insert(index, "j")
            stress.insert(index, None)
            notes.append(f"South African English says the y of {written}: {phones[index - 1]} + y + uː, not {phones[index - 1]} + uː")
    return phones, stress, notes


def respell_word(phones: Sequence[str], stress: Sequence[int | None], *, limit: int = 8, written: str | None = None) -> list[dict]:
    """Spellings of a word from its English sounds (and their stress), main choice first, each with the rules it applied.  With ``written`` the word as it is spelled in
    English, a reduced vowel takes the letter of that spelling."""
    found: dict[str, dict] = {}
    yodded, yod_stress, yod_notes = _yod(phones, stress, written)
    versions = {}            # the sounds and stress for each way through the two spelling-based rules
    for yod in (1, 0):
        base, marks = (yodded, yod_stress) if yod else (list(phones), list(stress))
        lettered, lettered_notes = _letter_vowels(base, written)
        versions[(yod, 1)] = (lettered, marks, [*(yod_notes if yod else []), *lettered_notes])
        versions[(yod, 0)] = (base, marks, list(yod_notes) if yod else [])
    for choice in _choices(letters=False, held=True):
        use, use_stress, rule_notes = versions[(choice.yod, choice.letter_vowel)]
        notes: list[str] = list(rule_notes)
        spelling = _write(_units(use, use_stress, choice, notes, bare_final=choice.final_vowel == 0, keep_clusters=choice.cluster_vowel == 0), choice, held=True, final_only=True,
                          notes=notes, double_long=True)
        if spelling and spelling not in found:
            found[spelling] = {"spelling": spelling, "derivation": list(dict.fromkeys(notes)), "choice": choice}
        if len(found) >= limit:
            break
    return list(found.values())


def respell_initialism(word: str, *, limit: int = 8) -> list[dict]:
    """Spellings of a word said letter by letter (AI, API, CPU): each letter's name by the rules, the last held, the letters apart."""
    from .english_sounds import LETTER_NAMES, VOWELS_ARPABET, plural_initialism, plural_sound, to_ipa

    plural = plural_initialism(word)                 # MPs: the letters of MP, then the plural said after the last
    names = [LETTER_NAMES[c] for c in (plural or word).upper() if c in LETTER_NAMES]
    if len(names) < 1:
        return []
    if plural:
        names[-1] = [*names[-1], plural_sound(names[-1])]
    found: dict[str, dict] = {}
    for choice in _choices(letters=True, held=True):
        pieces: list[str] = []
        notes: list[str] = []
        for position, name in enumerate(names):
            last = position == len(names) - 1
            phones = to_ipa([t + ("1" if last else "2") if t in VOWELS_ARPABET else t for t in name])
            weights = [(1 if last else 2) if t in VOWELS_ARPABET else None for t in name]
            units = _units(phones, weights, choice, notes, bare_final=choice.final_vowel == 0)          # M is em, not emi: the consonant English ends the letter with stays bare
            closed = bool(plural) and last                  # the plural closes the last letter: its long vowel is doubled, as in a word, not held
            piece = _write(units, choice, held=last, final_only=closed, notes=notes, double_long=closed)
            pieces.append(piece.capitalize() if choice.join == " " else piece)
        spelling = choice.join.join(pieces)
        if spelling not in found:
            found[spelling] = {"spelling": spelling, "derivation": list(dict.fromkeys(notes)), "choice": choice}
        if len(found) >= limit:
            break
    return list(found.values())


def parse_ipa(ipa: str) -> list[tuple[list[str], list[int | None]]]:
    """A transcription (``/ˈhaɪdi/``, ``/ˌwɛstən ˈkeɪp haɪ ˈkɔːt/``) as the sounds and stress of each of its words, in the symbols ``respell_word`` takes.  A mark ˈ
    gives the next vowel primary stress and ˌ secondary; a vowel with neither has 0 and a consonant None.  Aspiration, syllabic marks and syllable breaks are
    dropped: they are not sounds the Zulu spelling writes."""
    words: list[tuple[list[str], list[int | None]]] = []
    for chunk in ipa.replace("​", " ").split():
        phones: list[str] = []
        stress: list[int | None] = []
        pending = 0
        index = 0
        while index < len(chunk):
            two = chunk[index:index + 2]
            if chunk[index] in "ˈˌ":
                pending = 1 if chunk[index] == "ˈ" else 2
                index += 1
                continue
            if chunk[index] in IPA_DROPPED:
                index += 1
                continue
            if two in IPA_UNITS:
                phone, index = IPA_ALIASES.get(two, two), index + 2
            else:
                phone, index = IPA_ALIASES.get(chunk[index], chunk[index]), index + 1
            if phone in ASPIRATED and index < len(chunk) and chunk[index] == "ʰ":          # a transcription that marks the puff of air keeps it
                phone, index = ASPIRATED[phone], index + 1
            if index < len(chunk) and chunk[index] == "ː":
                phone, index = phone + "ː", index + 1
            vowel = phone in VOWELS or phone in DIPHTHONGS
            phones.append(phone)
            stress.append(((pending or (2 if phone in LONG else 0)) if vowel else None))          # a transcription marks length only where the vowel is long
            if vowel:
                pending = 0
        if phones:
            words.append((phones, stress))
    return words


def respell_ipa(ipa: str, *, limit: int = 8) -> list[list[dict]]:
    """``respell_word`` for each word of a transcription: a list of spelling lists, one per word."""
    return [respell_word(phones, stress, limit=limit) for phones, stress in parse_ipa(ipa)]


MIN_RESEARCH_CONFIDENCE = 0.9      # a researched transcription below this is not used as the sounds of a name


def research_readings(research: dict) -> dict[str, dict]:
    """The sounds of the words of the names a job's web research found a transcription for (``pronunciation_research.json``), as ``{lowercase word: reading}`` in the
    shape ``Interpreter.readings`` returns.  Only a grounded, confident transcription of a name or word that is not isiZulu (Zulu spelling needs no interpreting) and
    whose words match the transcription's one for one; an initialism is said by its letters, which the dictionary rules already know."""
    found: dict[str, dict] = {}
    for item in research.get("accepted_corrections") or []:
        language = str(item.get("pronunciation_language") or "")
        ipa = str(item.get("pronunciation_ipa") or "")
        if (item.get("pronunciation_mode") != "word_name" or not ipa or float(item.get("pronunciation_confidence") or 0) < MIN_RESEARCH_CONFIDENCE
                or not item.get("pronunciation_evidence_urls") or ("zulu" in language.lower() and "english" not in language.lower())):
            continue
        words = [w for w in re.sub(r"[().,;:]", " ", str(item.get("canonical_text") or "")).split() if w]          # "Democratic Alliance (DA)": the brackets are not a word
        parsed = parse_ipa(re.sub(r"[,;:!?()\[\]]", " ", ipa))          # research writes "əˈlaɪəns, ˌdiːˈeɪ": the comma between the name and its initialism is not a sound
        if len(words) != len(parsed):
            continue
        for word, (phones, stress) in zip(words, parsed):
            found.setdefault(word.lower(), {"how": "researched", "accent": "researched", "phones": phones, "stress": stress,
                                            "confidence": float(item["pronunciation_confidence"]), "language": language})
    return found
