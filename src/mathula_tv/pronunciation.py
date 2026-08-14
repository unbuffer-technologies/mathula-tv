"""Versioned, application-controlled pronunciation dictionaries.

Entries are literal substitutions.  They are never interpreted as XML and
cannot inject SSML; an SSML builder must escape the resulting text later.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from typing import Any, Iterable, Mapping

from .tts_ssml import CharacterPart, SSMLPart, TextPart


PRONUNCIATION_DICTIONARY_SCHEMA_VERSION = "pronunciation-dictionary-v1"
PRONUNCIATION_ENTRY_SCHEMA_VERSION = "pronunciation-entry-v1"
ZU_CODE_SWITCH_PRONUNCIATION_VERSION = (
    "mathula-zu-code-switch-v2-coloured-v13.18.20"
)
ZU_NATIVE_DATE_PRONUNCIATION_VERSION = (
    "mathula-zu-native-calendar-date-v1-v13.18.22"
)
ZU_NATIVE_HONORIFIC_PRONUNCIATION_VERSION = (
    "mathula-zu-native-honorific-flow-v1-v13.18.23"
)
ZU_CONTEXTUAL_NAME_PREFIX_VERSION = (
    "mathula-zu-contextual-name-prefix-v1-v13.18.42"
)
ZU_PREFIXED_CARDINAL_VERSION = (
    "mathula-zu-prefixed-cardinal-v2-currency-safe-v13.18.62"
)
ZU_RAND_MILLION_CODE_SWITCH_VERSION = (
    "mathula-zu-rand-million-code-switch-v1-v13.18.62"
)
PRONUNCIATION_KINDS = {
    "personal_name",
    "place_name",
    "organisation_name",
    "initials",
    "acronym",
    "number",
    "date",
    "percentage",
    "currency",
    "english_code_switch",
    "commission_term",
    "politics_term",
    "other",
}
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Reviewed letter-name organisations.  These aliases are deliberately explicit:
# an uppercase token is not sufficient evidence that it should be spelled out
# (for example, SARS is normally pronounced as a word).
_SA_ORGANISATION_INITIALISMS = {
    "ANC": "Ay En See",
    "DA": "Dee Ay",
    # Preserve EFF as a voice-native code-switched token. Do not force English
    # letter names or SSML character spelling into South African language voices.
    "EFF": "EFF",
    "IFP": "Eye Eff Pee",
    "MK": "Em Kay",
    "MKP": "Em Kay Pee",
    "NPA": "En Pee Ay",
    "UDM": "You Dee Em",
}

# Azure's isiZulu voices do not consistently infer alphabet names from an
# uppercase token. Keep the alphabet-to-TTS mapping application-owned, then
# register only terms that have been reviewed as true initialisms. This avoids
# spelling word-acronyms such as SARS or SANRAL letter by letter.
_ZU_AZURE_LETTER_NAMES = {
    "A": "Ey",
    "B": "Bee",
    "C": "See",
    "D": "Dee",
    "E": "Ee",
    "F": "Eff",
    "G": "Gee",
    "H": "Aitch",
    "I": "Eye",
    "J": "Jay",
    "K": "Kay",
    "L": "El",
    "M": "Em",
    "N": "En",
    "O": "Oh",
    "P": "Pee",
    "Q": "Cue",
    "R": "Ar",
    "S": "Ess",
    "T": "Tee",
    "U": "You",
    "V": "Vee",
    "W": "Double-you",
    "X": "Ex",
    "Y": "Why",
    "Z": "Zed",
}


def _spell_zu_initialism(value: str) -> str:
    token = str(value).strip().upper()
    if not token or not token.isalpha():
        raise ValueError("Reviewed isiZulu initialism must contain letters only")
    try:
        return " ".join(_ZU_AZURE_LETTER_NAMES[letter] for letter in token)
    except KeyError as exc:  # defensive guard for future non-Latin entries
        raise ValueError(f"Unsupported isiZulu initialism letter: {exc.args[0]}") from exc


@dataclass(frozen=True)
class _ReviewedAcronymSpec:
    token: str
    kind: str
    tts_text: str | None = None
    aliases: tuple[str, ...] = ()
    character_mode: bool = False

    def resolved_tts_text(self) -> str:
        if self.character_mode:
            return self.token
        if self.tts_text is not None:
            return self.tts_text
        if self.kind == "initials":
            return _spell_zu_initialism(self.token)
        raise ValueError(f"Word acronym {self.token} requires reviewed TTS text")


# Curated for South African news, politics, justice, public services, labour and
# economic reporting. This is a practical high-frequency registry, not a claim
# of an official national usage ranking. Unknown uppercase terms still remain
# unchanged and require explicit review before the application forces a reading.
_ZU_SA_PUBLIC_AFFAIRS_ACRONYMS = (
    _ReviewedAcronymSpec("ANC", "initials", character_mode=True),
    _ReviewedAcronymSpec("DA", "initials"),
    _ReviewedAcronymSpec("EFF", "initials"),
    _ReviewedAcronymSpec("MK", "initials"),
    _ReviewedAcronymSpec("MKP", "initials"),
    _ReviewedAcronymSpec("IFP", "initials"),
    _ReviewedAcronymSpec("UDM", "initials"),
    _ReviewedAcronymSpec("ACDP", "initials"),
    _ReviewedAcronymSpec("ATM", "initials"),
    _ReviewedAcronymSpec("BOSA", "acronym", "Bo-sa"),
    _ReviewedAcronymSpec("COPE", "acronym", "Cope"),
    _ReviewedAcronymSpec("PAC", "initials"),
    _ReviewedAcronymSpec("PA", "initials"),
    _ReviewedAcronymSpec("FF+", "initials", "Eff Eff Plus"),
    _ReviewedAcronymSpec("GOOD", "acronym", "Good"),
    _ReviewedAcronymSpec("NPA", "initials"),
    _ReviewedAcronymSpec("NDPP", "initials"),
    _ReviewedAcronymSpec("DPP", "initials"),
    _ReviewedAcronymSpec("IDAC", "acronym", "Ay-dak"),
    _ReviewedAcronymSpec("IPID", "acronym", "Ay-pid"),
    _ReviewedAcronymSpec("SAPS", "acronym", "Saps"),
    _ReviewedAcronymSpec("PKTT", "initials"),
    _ReviewedAcronymSpec("SIU", "initials"),
    _ReviewedAcronymSpec("CRTT", "initials"),
    _ReviewedAcronymSpec("SSA", "initials"),
    _ReviewedAcronymSpec("SANDF", "initials"),
    _ReviewedAcronymSpec("DPCI", "initials"),
    _ReviewedAcronymSpec("NATJOINTS", "acronym", "Nat-joints"),
    _ReviewedAcronymSpec("JSC", "initials"),
    _ReviewedAcronymSpec("SCA", "initials"),
    _ReviewedAcronymSpec("OCJ", "initials"),
    _ReviewedAcronymSpec("SAHRC", "initials"),
    _ReviewedAcronymSpec("IEC", "initials"),
    _ReviewedAcronymSpec("ICASA", "acronym", "Eye-casa"),
    _ReviewedAcronymSpec("PANSALB", "acronym", "Pan-salb"),
    _ReviewedAcronymSpec("AGSA", "acronym", "Ag-sa"),
    _ReviewedAcronymSpec("GCIS", "initials"),
    _ReviewedAcronymSpec("DIRCO", "acronym", "Dir-co"),
    _ReviewedAcronymSpec("COGTA", "acronym", "Cog-ta"),
    _ReviewedAcronymSpec("DPME", "initials"),
    _ReviewedAcronymSpec("DPSA", "initials"),
    _ReviewedAcronymSpec("DFFE", "initials"),
    _ReviewedAcronymSpec("DBE", "initials"),
    _ReviewedAcronymSpec("DHET", "initials"),
    _ReviewedAcronymSpec("DHA", "initials"),
    _ReviewedAcronymSpec("DCS", "initials"),
    _ReviewedAcronymSpec("DSD", "initials"),
    _ReviewedAcronymSpec("DTIC", "acronym", "Dee-tick"),
    _ReviewedAcronymSpec("DPWI", "initials"),
    _ReviewedAcronymSpec("DWS", "initials"),
    _ReviewedAcronymSpec("DMPR", "initials"),
    _ReviewedAcronymSpec("DCDT", "initials"),
    _ReviewedAcronymSpec("SARS", "acronym", "Sars"),
    _ReviewedAcronymSpec("SASSA", "acronym", "Sassa"),
    _ReviewedAcronymSpec("UIF", "initials"),
    _ReviewedAcronymSpec("NSFAS", "acronym", "En-sfas"),
    _ReviewedAcronymSpec("PRASA", "acronym", "Pra-sa"),
    _ReviewedAcronymSpec("ACSA", "acronym", "Ack-sa"),
    _ReviewedAcronymSpec("SANRAL", "acronym", "San-ral"),
    _ReviewedAcronymSpec("RAF", "initials"),
    _ReviewedAcronymSpec("RTMC", "initials"),
    _ReviewedAcronymSpec("RTIA", "initials"),
    _ReviewedAcronymSpec("SABC", "initials"),
    _ReviewedAcronymSpec("SENTECH", "acronym", "Sen-tech"),
    _ReviewedAcronymSpec("ESKOM", "acronym", "Es-kom"),
    _ReviewedAcronymSpec("TRANSNET", "acronym", "Trans-net"),
    _ReviewedAcronymSpec("TELKOM", "acronym", "Tel-kom"),
    _ReviewedAcronymSpec("PIC", "initials"),
    _ReviewedAcronymSpec("GEPF", "initials"),
    _ReviewedAcronymSpec("GPAA", "initials"),
    _ReviewedAcronymSpec("IDC", "initials"),
    _ReviewedAcronymSpec("DBSA", "initials"),
    _ReviewedAcronymSpec("NEF", "initials"),
    _ReviewedAcronymSpec("NYDA", "initials"),
    _ReviewedAcronymSpec("SETA", "acronym", "See-ta"),
    _ReviewedAcronymSpec("TVET", "acronym", "Tee-vet"),
    _ReviewedAcronymSpec("SAQA", "acronym", "Sah-kwa"),
    _ReviewedAcronymSpec("COSATU", "acronym", "Co-sa-too"),
    _ReviewedAcronymSpec("NUMSA", "acronym", "Num-sa"),
    _ReviewedAcronymSpec("NEHAWU", "acronym", "Neh-ha-woo"),
    _ReviewedAcronymSpec("SADTU", "acronym", "Sad-too"),
    _ReviewedAcronymSpec("DENOSA", "acronym", "De-no-sa"),
    _ReviewedAcronymSpec("SAMWU", "acronym", "Sam-woo"),
    _ReviewedAcronymSpec("PSA", "initials"),
    _ReviewedAcronymSpec("SACP", "initials"),
    _ReviewedAcronymSpec("SANCO", "acronym", "San-co"),
    _ReviewedAcronymSpec("OUTA", "acronym", "Ow-ta"),
    _ReviewedAcronymSpec("SARB", "initials"),
    _ReviewedAcronymSpec("JSE", "initials"),
    _ReviewedAcronymSpec("GDP", "initials"),
    _ReviewedAcronymSpec("CPI", "initials"),
    _ReviewedAcronymSpec("VAT", "acronym", "Vat"),
    _ReviewedAcronymSpec("PAYE", "acronym", "Pay-ee"),
    _ReviewedAcronymSpec("BEE", "acronym", "Bee"),
    _ReviewedAcronymSpec("B-BBEE", "initials", "Triple Bee Ee Ee", ("BBBEE",)),
    _ReviewedAcronymSpec("NHI", "initials"),
    _ReviewedAcronymSpec("POPIA", "acronym", "Pop-ee-ah"),
    _ReviewedAcronymSpec("PAIA", "acronym", "Pie-ah"),
    _ReviewedAcronymSpec("PFMA", "initials"),
    _ReviewedAcronymSpec("MFMA", "initials"),
)

if len(_ZU_SA_PUBLIC_AFFAIRS_ACRONYMS) != 100:
    raise RuntimeError("The South African public-affairs acronym registry must contain 100 entries")
if len({item.token.casefold() for item in _ZU_SA_PUBLIC_AFFAIRS_ACRONYMS}) != 100:
    raise RuntimeError("The South African public-affairs acronym registry contains duplicates")

_ZU_CHARACTER_INITIALISMS = frozenset(
    item.token for item in _ZU_SA_PUBLIC_AFFAIRS_ACRONYMS if item.character_mode
)
_ZU_REVIEWED_CODE_SWITCH_PRONUNCIATIONS = {
    # Contextual forms are explicit because isiZulu prefixes are normally
    # attached in writing, while Azure needs the reviewed name isolated.
    "Tshwane": ("Tšhwane", "place_name", ()),
    "iTshwane": ("i Tšhwane", "place_name", ()),
    "eTshwane": ("e Tšhwane", "place_name", ()),
    "iseTshwane": ("ise Tšhwane", "place_name", ()),
    "London": ("Landen", "place_name", ()),
    "neLondon": ("ne Landen", "place_name", ()),
    "Buffalo City": ("Baffalo Siti", "place_name", ()),
    "baseBuffalo City": ("base Baffalo Siti", "place_name", ()),
    # The standard zu-ZA voices are not multilingual and Azure does not support
    # <lang xml:lang="en-ZA"> for them. Keep the protected/display spelling,
    # but give the voice a calibrated South African-English approximation instead
    # of letting it infer isiZulu phonetics for the English social descriptor.
    "coloured": ("Khalad", "english_code_switch", ("colored",)),
}
_SUPPORTED_SA_LANGUAGE_LOCALES = {
    "nr-za",   # isiNdebele
    "nso-za",  # Sepedi
    "ss-za",   # siSwati
    "st-za",   # Sesotho
    "tn-za",   # Setswana
    "ts-za",   # Xitsonga
    "ve-za",   # Tshivenda
    "xh-za",   # isiXhosa
    "zu-za",   # isiZulu
}

# Azure's isiZulu voices can split a fused grammatical prefix from an
# abbreviated honorific, then treat the abbreviation's full stop as a sentence
# boundary.  For example, ``noMnu. Adams`` has been heard as ``no`` followed by
# a conspicuous pause.  Expand only the hidden TTS form; the approved translation
# and captions retain their editorial spelling.
_ZU_HONORIFIC_EXPANSIONS = {
    "mnu": "Mnumzane",
    "nkk": "Nkosikazi",
    "nksz": "Nkosazana",
    "dkt": "Dokotela",
}
_ZU_FUSED_HONORIFIC = re.compile(
    r"(?<!\w)"
    r"(?P<prefix>kwa|ngo|ku|no|ne|na|ka|u)?"
    r"(?P<title>Mnu|Nkk|Nksz|Dkt)\."
    r"(?=\s|$)",
    re.IGNORECASE | re.UNICODE,
)


def _zulu_native_honorific_tts(match: re.Match[str]) -> str:
    prefix = str(match.group("prefix") or "")
    title = _ZU_HONORIFIC_EXPANSIONS[match.group("title").casefold()]
    return prefix + title


def _zulu_dynamic_honorific_candidates(
    text: str,
    *,
    language: str,
) -> list[tuple[int, int, PronunciationEntry]]:
    if language.casefold() != "zu-za":
        return []
    candidates: list[tuple[int, int, PronunciationEntry]] = []
    for match in _ZU_FUSED_HONORIFIC.finditer(text):
        before = match.group(0)
        after = _zulu_native_honorific_tts(match)
        candidates.append(
            (
                match.start(),
                match.end(),
                PronunciationEntry(
                    display_text=before,
                    spoken_text=before,
                    tts_text=after,
                    language=language,
                    source="application_default",
                    confidence=1.0,
                    kind="other",
                    notes=(
                        "Application-controlled expansion of an isiZulu honorific abbreviation",
                        ZU_NATIVE_HONORIFIC_PRONUNCIATION_VERSION,
                        "Prevents Azure from inserting a false sentence pause after the fused prefix",
                    ),
                ),
            )
        )
    return candidates

# Government isiZulu uses indigenous month names and a day concord, for
# example ``mhla ziyi-19 kuNcwaba``. Translation models often emit hybrids such
# as ``ngomhla ka-19 Agasti``. Azure's locale-independent SSML date hint is not
# supported for isiZulu, so the hidden TTS layer owns this deterministic
# normalization while the approved/display text remains immutable.
_ZU_CALENDAR_MONTHS = (
    ("Masingana", ("January", "Januwari", "Masingana")),
    ("Nhlolanja", ("February", "Febhuwari", "Nhlolanja")),
    ("Ndasa", ("March", "Mashi", "Ndasa")),
    ("Mbasa", ("April", "Ephreli", "Mbasa")),
    ("Nhlaba", ("May", "Meyi", "Nhlaba")),
    ("Nhlangulana", ("June", "Juni", "Nhlangulana")),
    ("Ntulikazi", ("July", "Julayi", "Ntulikazi")),
    ("Ncwaba", ("August", "Agasti", "Ncwaba")),
    ("Mandulo", ("September", "Septhemba", "Mandulo")),
    ("Mfumfu", ("October", "Okthoba", "Mfumfu")),
    ("Lwezi", ("November", "Novemba", "Lwezi")),
    ("Zibandlela", ("December", "Disemba", "Zibandlela")),
)


def _zulu_month_aliases() -> dict[str, str]:
    result: dict[str, str] = {}
    for canonical, aliases in _ZU_CALENDAR_MONTHS:
        for alias in aliases:
            for value in (
                alias,
                f"ku{alias}",
                f"u{alias}",
                f"ngo{alias}",
                f"ngo-{alias}",
            ):
                result[value.casefold()] = canonical
    return result


_ZU_MONTH_BY_ALIAS = _zulu_month_aliases()
_ZU_MONTH_PATTERN = "|".join(
    re.escape(value)
    for value in sorted(_ZU_MONTH_BY_ALIAS, key=lambda item: (-len(item), item))
)
_ZU_CALENDAR_DATE = re.compile(
    rf"(?<!\w)"
    rf"(?P<prefix>"
    rf"(?:(?:ngumhla|ngomhla|umhla|mhla)\s+)"
    rf"(?:(?:ka|ziyi|zingama|zi|lu)\s*-?\s*)?"
    rf")?"
    rf"(?P<day>0?[1-9]|[12][0-9]|3[01])"
    rf"(?:st|nd|rd|th)?"
    rf"(?:\s+|\s*[-/.]\s*)"
    rf"(?P<month>{_ZU_MONTH_PATTERN})"
    rf"(?P<year>\s*,?\s*(?:19|20)[0-9]{{2}})?"
    rf"(?!\w)",
    re.IGNORECASE | re.UNICODE,
)

_ZU_DATE_COMPOUND_UNITS = {
    1: "nanye",
    2: "nambili",
    3: "nantathu",
    4: "nane",
    5: "nesihlanu",
    6: "nesithupha",
    7: "nesikhombisa",
    8: "nesishiyagalombili",
    9: "nesishiyagalolunye",
}

# Translation text often preserves an exact numeral after an isiZulu concord,
# for example ``izigidi ezingu-31`` or ``abantu abangu-80``.  Azure's zu-ZA
# voices do not reliably read that mixed written form.  Normalize the hidden
# TTS layer, not the approved/display text.  This first reviewed grammar covers
# 20--99, including the monetary and head-count constructions that exposed the
# defect in job d724a4c9.  Values outside the reviewed range remain untouched
# instead of guessing at noun-class morphology.
_ZU_PREFIXED_TWO_DIGIT_CARDINAL = re.compile(
    r"(?<!\w)"
    r"(?P<prefix>ezingu|abangu|zingu|engu)\s*-\s*"
    r"(?P<number>[2-9][0-9])"
    r"(?![0-9])",
    re.IGNORECASE | re.UNICODE,
)
_ZU_CARDINAL_TENS = {
    2: "amabili",
    3: "amathathu",
    4: "amane",
    5: "amahlanu",
    6: "ayisithupha",
    7: "ayisikhombisa",
    8: "ayisishiyagalombili",
    9: "ayisishiyagalolunye",
}
_ZU_CARDINAL_CONCORD = {
    "ezingu": "ezinga",
    "abangu": "abanga",
    "zingu": "zinga",
    "engu": "enga",
}


def _zulu_prefixed_two_digit_cardinal(match: re.Match[str]) -> str:
    number = int(match.group("number"))
    tens, unit = divmod(number, 10)
    prefix = _ZU_CARDINAL_CONCORD[match.group("prefix").casefold()]
    value = f"{prefix}mashumi {_ZU_CARDINAL_TENS[tens]}"
    if unit:
        value += " " + _ZU_DATE_COMPOUND_UNITS[unit]
    return value


_ZU_CODE_SWITCH_RAND_MILLION = re.compile(
    r"(?<!\w)(?P<prefix>u-)?R\s*(?P<number>[2-9][0-9])\s*million(?!\w)",
    re.IGNORECASE | re.UNICODE,
)


def _zulu_class10_two_digit_cardinal(number: int) -> str:
    if not 20 <= number <= 99:
        raise ValueError("Reviewed isiZulu monetary cardinal range is 20--99")
    tens, unit = divmod(number, 10)
    value = f"ezingamashumi {_ZU_CARDINAL_TENS[tens]}"
    if unit:
        value += " " + _ZU_DATE_COMPOUND_UNITS[unit]
    return value


def _zulu_dynamic_currency_candidates(
    text: str,
    *,
    language: str,
) -> list[tuple[int, int, PronunciationEntry]]:
    """Normalize compact rand-million code switches in the hidden TTS layer.

    Semantic timing repair can legitimately shorten a native isiZulu amount to
    a compact written form such as ``u-R31 million``.  The zu-ZA neural voices
    do not reliably pronounce that mixed token.  Keep the approved/display text
    untouched, but synthesize the amount as native isiZulu.  Only the reviewed
    20--99 million range is handled here; unknown morphology remains unchanged.
    """

    if language.casefold() != "zu-za":
        return []
    candidates: list[tuple[int, int, PronunciationEntry]] = []
    for match in _ZU_CODE_SWITCH_RAND_MILLION.finditer(text):
        number = int(match.group("number"))
        before = match.group(0)
        after = (
            "izigidi "
            + _zulu_class10_two_digit_cardinal(number)
            + " zamaRandi"
        )
        candidates.append(
            (
                match.start(),
                match.end(),
                PronunciationEntry(
                    display_text=before,
                    spoken_text=before,
                    tts_text=after,
                    language=language,
                    source="application_default",
                    confidence=1.0,
                    kind="currency",
                    notes=(
                        "Application-controlled isiZulu rand-million reading",
                        ZU_RAND_MILLION_CODE_SWITCH_VERSION,
                        "Approved/display code switch remains immutable",
                    ),
                ),
            )
        )
    return candidates


def _zulu_dynamic_number_candidates(
    text: str,
    *,
    language: str,
) -> list[tuple[int, int, PronunciationEntry]]:
    if language.casefold() != "zu-za":
        return []
    candidates: list[tuple[int, int, PronunciationEntry]] = []
    for match in _ZU_PREFIXED_TWO_DIGIT_CARDINAL.finditer(text):
        before = match.group(0)
        after = _zulu_prefixed_two_digit_cardinal(match)
        candidates.append(
            (
                match.start(),
                match.end(),
                PronunciationEntry(
                    display_text=before,
                    spoken_text=before,
                    tts_text=after,
                    language=language,
                    source="application_default",
                    confidence=1.0,
                    kind="number",
                    notes=(
                        "Application-controlled isiZulu concorded cardinal reading",
                        ZU_PREFIXED_CARDINAL_VERSION,
                        "Approved/display digits remain immutable",
                    ),
                ),
            )
        )
    return candidates


def _zulu_date_subject_day(day: int) -> str:
    """Return the fully inflected day after ``mhla`` for days 1 through 31."""

    if day == 1:
        return "lulunye"
    if day == 2:
        return "zimbili"
    if day == 3:
        return "zintathu"
    if day == 4:
        return "zine"
    if day == 5:
        return "zinhlanu"
    if 6 <= day <= 9:
        return "ziyi" + _ZU_DATE_COMPOUND_UNITS[day].removeprefix("ne")
    if day == 10:
        return "ziyishumi"
    if 11 <= day <= 19:
        return "ziyishumi " + _ZU_DATE_COMPOUND_UNITS[day - 10]
    tens, unit = divmod(day, 10)
    tens_word = {2: "amabili", 3: "amathathu"}[tens]
    value = f"zingamashumi {tens_word}"
    if unit:
        value += " " + _ZU_DATE_COMPOUND_UNITS[unit]
    return value


def _zulu_date_ordinal_day(day: int) -> str:
    """Return a day modifying the noun ``umhla`` for days 1 through 31."""

    first_nine = {
        1: "wokuqala",
        2: "wesibili",
        3: "wesithathu",
        4: "wesine",
        5: "wesihlanu",
        6: "wesithupha",
        7: "wesikhombisa",
        8: "wesishiyagalombili",
        9: "wesishiyagalolunye",
    }
    if day in first_nine:
        return first_nine[day]
    if day == 10:
        return "weshumi"
    if 11 <= day <= 19:
        return "weshumi " + _ZU_DATE_COMPOUND_UNITS[day - 10]
    tens, unit = divmod(day, 10)
    tens_word = {2: "amabili", 3: "amathathu"}[tens]
    value = f"wamashumi {tens_word}"
    if unit:
        value += " " + _ZU_DATE_COMPOUND_UNITS[unit]
    return value


def _zulu_native_date_tts(match: re.Match[str]) -> str:
    day = int(match.group("day"))
    month = _ZU_MONTH_BY_ALIAS[match.group("month").casefold()]
    prefix = str(match.group("prefix") or "").strip().casefold()
    year = str(match.group("year") or "")
    if prefix.startswith("ngumhla"):
        date = f"ngumhla {_zulu_date_ordinal_day(day)} ku{month}"
    elif prefix.startswith("umhla"):
        date = f"umhla {_zulu_date_ordinal_day(day)} ku{month}"
    else:
        date = f"mhla {_zulu_date_subject_day(day)} ku{month}"
    return date + year


def _zulu_dynamic_date_candidates(
    text: str,
    *,
    language: str,
) -> list[tuple[int, int, PronunciationEntry]]:
    if language.casefold() != "zu-za":
        return []
    candidates: list[tuple[int, int, PronunciationEntry]] = []
    for match in _ZU_CALENDAR_DATE.finditer(text):
        before = match.group(0)
        after = _zulu_native_date_tts(match)
        if before.casefold() == after.casefold():
            continue
        candidates.append(
            (
                match.start(),
                match.end(),
                PronunciationEntry(
                    display_text=before,
                    spoken_text=before,
                    tts_text=after,
                    language=language,
                    source="application_default",
                    confidence=1.0,
                    kind="date",
                    notes=(
                        "Application-controlled native isiZulu calendar reading",
                        ZU_NATIVE_DATE_PRONUNCIATION_VERSION,
                    ),
                ),
            )
        )
    return candidates


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _safe_literal(value: str, label: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{label} is required")
    if "<" in text or ">" in text or _CONTROL.search(text):
        raise ValueError(f"{label} must be plain text, not XML or control data")
    return text


def _validate_reviewed_at(value: str | None) -> None:
    if value is None:
        return
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError("reviewed_at must be an ISO-8601 timestamp") from exc


@dataclass(frozen=True)
class PronunciationEntry:
    display_text: str
    spoken_text: str
    tts_text: str
    language: str = "zu-ZA"
    source: str = "human_review"
    reviewed_by: str | None = None
    reviewed_at: str | None = None
    confidence: float = 1.0
    notes: tuple[str, ...] = ()
    kind: str = "other"
    aliases: tuple[str, ...] = ()
    entry_id: str | None = None
    schema_version: str = PRONUNCIATION_ENTRY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PRONUNCIATION_ENTRY_SCHEMA_VERSION:
            raise ValueError("Unsupported pronunciation entry schema")
        object.__setattr__(self, "display_text", _safe_literal(self.display_text, "display_text"))
        object.__setattr__(self, "spoken_text", _safe_literal(self.spoken_text, "spoken_text"))
        object.__setattr__(self, "tts_text", _safe_literal(self.tts_text, "tts_text"))
        if not re.fullmatch(r"[A-Za-z]{2,3}(?:-[A-Za-z]{2,4})?", self.language):
            raise ValueError("language must be a locale such as zu-ZA")
        if not self.source.strip():
            raise ValueError("source is required")
        if self.kind not in PRONUNCIATION_KINDS:
            raise ValueError(f"Unsupported pronunciation kind: {self.kind}")
        if not 0 <= float(self.confidence) <= 1:
            raise ValueError("confidence must be between 0 and 1")
        _validate_reviewed_at(self.reviewed_at)
        notes = tuple(str(note) for note in self.notes)
        aliases = tuple(_safe_literal(alias, "alias") for alias in self.aliases)
        object.__setattr__(self, "notes", notes)
        object.__setattr__(self, "aliases", aliases)
        if self.entry_id is None:
            identity = {
                "display_text": self.display_text,
                "spoken_text": self.spoken_text,
                "tts_text": self.tts_text,
                "language": self.language,
                "kind": self.kind,
                "source": self.source,
            }
            object.__setattr__(self, "entry_id", "pron_" + hashlib.sha256(_canonical_json(identity).encode()).hexdigest()[:16])
        elif not re.fullmatch(r"[A-Za-z0-9_.:-]+", self.entry_id):
            raise ValueError("entry_id contains unsupported characters")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PronunciationEntry":
        allowed = {item.name for item in fields(cls)}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"Unknown pronunciation entry fields: {sorted(unknown)}")
        return cls(
            **{
                **dict(value),
                "notes": tuple(value.get("notes") or ()),
                "aliases": tuple(value.get("aliases") or ()),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["notes"] = list(self.notes)
        value["aliases"] = list(self.aliases)
        return value

    def match_texts(self) -> tuple[str, ...]:
        seen: set[str] = set()
        values: list[str] = []
        for value in (self.spoken_text, self.display_text, *self.aliases):
            folded = value.casefold()
            if folded not in seen:
                seen.add(folded)
                values.append(value)
        return tuple(values)


@dataclass(frozen=True)
class PronunciationSubstitution:
    entry_id: str
    kind: str
    display_text: str
    spoken_text: str
    tts_text: str
    before: str
    after: str
    source_start: int
    source_end: int
    source: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PronunciationResult:
    spoken_text: str
    tts_text: str
    substitutions: tuple[PronunciationSubstitution, ...]
    dictionary_version: str
    dictionary_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "spoken_text": self.spoken_text,
            "tts_text": self.tts_text,
            "substitutions": [item.to_dict() for item in self.substitutions],
            "dictionary_version": self.dictionary_version,
            "dictionary_sha256": self.dictionary_sha256,
        }


@dataclass(frozen=True)
class PronunciationDictionary:
    dictionary_version: str
    entries: tuple[PronunciationEntry, ...] = ()
    job_overrides: tuple[PronunciationEntry, ...] = ()
    language: str = "zu-ZA"
    job_id: str | None = None
    schema_version: str = PRONUNCIATION_DICTIONARY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PRONUNCIATION_DICTIONARY_SCHEMA_VERSION:
            raise ValueError("Unsupported pronunciation dictionary schema")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]*", self.dictionary_version):
            raise ValueError("dictionary_version must be a safe version label, for example 1.0.0 or v1")
        entries = tuple(
            entry if isinstance(entry, PronunciationEntry) else PronunciationEntry.from_dict(entry)
            for entry in self.entries
        )
        overrides = tuple(
            entry if isinstance(entry, PronunciationEntry) else PronunciationEntry.from_dict(entry)
            for entry in self.job_overrides
        )
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "job_overrides", overrides)
        for entry in (*entries, *overrides):
            if entry.language.casefold() != self.language.casefold():
                raise ValueError("Pronunciation entry language does not match dictionary language")
        for label, scoped in (("global", entries), ("job override", overrides)):
            entry_ids = [entry.entry_id for entry in scoped]
            if len(entry_ids) != len(set(entry_ids)):
                raise ValueError(f"Pronunciation entry IDs must be unique within {label} scope")
        if overrides and not self.job_id:
            raise ValueError("job_overrides require a job-local job_id")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PronunciationDictionary":
        if value.get("schema_version") != PRONUNCIATION_DICTIONARY_SCHEMA_VERSION:
            raise ValueError("Unsupported pronunciation dictionary schema")
        allowed = {
            "schema_version",
            "dictionary_version",
            "language",
            "job_id",
            "entries",
            "job_overrides",
            "dictionary_sha256",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"Unknown pronunciation dictionary fields: {sorted(unknown)}")
        dictionary = cls(
            dictionary_version=str(value["dictionary_version"]),
            language=str(value.get("language", "zu-ZA")),
            job_id=value.get("job_id"),
            entries=tuple(PronunciationEntry.from_dict(item) for item in value.get("entries") or []),
            job_overrides=tuple(PronunciationEntry.from_dict(item) for item in value.get("job_overrides") or []),
        )
        expected = value.get("dictionary_sha256")
        if expected is not None and expected != dictionary.sha256:
            raise ValueError("Pronunciation dictionary hash mismatch")
        return dictionary

    def _without_hash(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dictionary_version": self.dictionary_version,
            "language": self.language,
            "job_id": self.job_id,
            "entries": [entry.to_dict() for entry in self.entries],
            "job_overrides": [entry.to_dict() for entry in self.job_overrides],
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self._without_hash()).encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {**self._without_hash(), "dictionary_sha256": self.sha256}

    def with_job_overrides(
        self,
        job_id: str,
        overrides: Iterable[PronunciationEntry | Mapping[str, Any]],
    ) -> "PronunciationDictionary":
        if not str(job_id).strip():
            raise ValueError("job_id is required for job pronunciation overrides")
        return PronunciationDictionary(
            dictionary_version=self.dictionary_version,
            entries=self.entries,
            job_overrides=tuple(
                item if isinstance(item, PronunciationEntry) else PronunciationEntry.from_dict(item)
                for item in overrides
            ),
            language=self.language,
            job_id=str(job_id),
        )

    def apply(
        self,
        spoken_text: str,
        *,
        kinds: set[str] | None = None,
    ) -> PronunciationResult:
        text = _safe_literal(spoken_text, "spoken_text")
        candidates: list[tuple[int, int, int, int, str, PronunciationEntry]] = []
        scoped_entries = [(0, entry) for entry in self.job_overrides] + [(1, entry) for entry in self.entries]
        for scope_priority, entry in scoped_entries:
            if kinds is not None and entry.kind not in kinds:
                continue
            for match_text in entry.match_texts():
                pattern = re.compile(rf"(?<!\w){re.escape(match_text)}(?!\w)", re.IGNORECASE | re.UNICODE)
                for match in pattern.finditer(text):
                    candidates.append(
                        (
                            match.start(),
                            match.end(),
                            scope_priority,
                            -(match.end() - match.start()),
                            str(entry.entry_id),
                            entry,
                        )
                    )
        if kinds is None or "date" in kinds:
            for start, end, entry in _zulu_dynamic_date_candidates(
                text,
                language=self.language,
            ):
                # Explicit job/global entries retain authority over the
                # application fallback when both begin at the same span.
                candidates.append(
                    (start, end, 2, -(end - start), str(entry.entry_id), entry)
                )
        if kinds is None or "currency" in kinds:
            for start, end, entry in _zulu_dynamic_currency_candidates(
                text,
                language=self.language,
            ):
                candidates.append(
                    (start, end, 2, -(end - start), str(entry.entry_id), entry)
                )
        if kinds is None or "number" in kinds or "currency" in kinds:
            for start, end, entry in _zulu_dynamic_number_candidates(
                text,
                language=self.language,
            ):
                candidates.append(
                    (start, end, 2, -(end - start), str(entry.entry_id), entry)
                )
        if kinds is None or "other" in kinds:
            for start, end, entry in _zulu_dynamic_honorific_candidates(
                text,
                language=self.language,
            ):
                # Explicit job/global entries remain authoritative over the
                # deterministic application fallback.
                candidates.append(
                    (start, end, 2, -(end - start), str(entry.entry_id), entry)
                )
        candidates.sort(key=lambda item: (item[0], item[2], item[3], item[4]))
        selected: list[tuple[int, int, PronunciationEntry]] = []
        occupied_until = -1
        for start, end, _scope, _length, _entry_id, entry in candidates:
            if start < occupied_until:
                continue
            selected.append((start, end, entry))
            occupied_until = end

        output: list[str] = []
        substitutions: list[PronunciationSubstitution] = []
        cursor = 0
        for start, end, entry in selected:
            output.append(text[cursor:start])
            output.append(entry.tts_text)
            before = text[start:end]
            substitutions.append(
                PronunciationSubstitution(
                    entry_id=str(entry.entry_id),
                    kind=entry.kind,
                    display_text=entry.display_text,
                    spoken_text=entry.spoken_text,
                    tts_text=entry.tts_text,
                    before=before,
                    after=entry.tts_text,
                    source_start=start,
                    source_end=end,
                    source=entry.source,
                    confidence=float(entry.confidence),
                )
            )
            cursor = end
        output.append(text[cursor:])
        return PronunciationResult(
            spoken_text=text,
            tts_text="".join(output),
            substitutions=tuple(substitutions),
            dictionary_version=self.dictionary_version,
            dictionary_sha256=self.sha256,
        )


def apply_pronunciation_dictionary(
    spoken_text: str,
    dictionary: PronunciationDictionary,
) -> PronunciationResult:
    return dictionary.apply(spoken_text)


def normalise_numbers_for_tts(spoken_text: str, dictionary: PronunciationDictionary) -> PronunciationResult:
    return dictionary.apply(spoken_text, kinds={"number", "percentage", "currency"})


def normalize_numbers_for_tts(spoken_text: str, dictionary: PronunciationDictionary) -> PronunciationResult:
    return normalise_numbers_for_tts(spoken_text, dictionary)


def normalise_dates_for_tts(spoken_text: str, dictionary: PronunciationDictionary) -> PronunciationResult:
    return dictionary.apply(spoken_text, kinds={"date"})


def normalize_dates_for_tts(spoken_text: str, dictionary: PronunciationDictionary) -> PronunciationResult:
    return normalise_dates_for_tts(spoken_text, dictionary)


def expand_initials_for_tts(spoken_text: str, dictionary: PronunciationDictionary) -> PronunciationResult:
    return dictionary.apply(spoken_text, kinds={"initials", "acronym"})


def get_zulu_public_affairs_acronym_registry() -> tuple[dict[str, Any], ...]:
    """Return the application-owned 100-term South African pronunciation registry."""

    return tuple(
        {
            "token": item.token,
            "kind": item.kind,
            "tts_text": item.resolved_tts_text(),
            "aliases": list(item.aliases),
            "character_mode": item.character_mode,
        }
        for item in _ZU_SA_PUBLIC_AFFAIRS_ACRONYMS
    )


def with_default_organisation_initialisms(
    dictionary: PronunciationDictionary,
) -> PronunciationDictionary:
    """Add reviewed locale pronunciations without replacing job overrides."""

    locale = dictionary.language.casefold()
    if locale not in _SUPPORTED_SA_LANGUAGE_LOCALES:
        return dictionary

    zulu_specs = _ZU_SA_PUBLIC_AFFAIRS_ACRONYMS if locale == "zu-za" else ()
    zulu_matches = {
        value.casefold()
        for item in zulu_specs
        for value in (item.token, *item.aliases)
    }
    pronunciations = {
        token: tts_text
        for token, tts_text in _SA_ORGANISATION_INITIALISMS.items()
        if token.casefold() not in zulu_matches
    }
    code_switches = (
        _ZU_REVIEWED_CODE_SWITCH_PRONUNCIATIONS
        if locale == "zu-za"
        else {}
    )
    code_switch_matches = {
        value.casefold()
        for display_text, (_tts_text, _kind, aliases) in code_switches.items()
        for value in (display_text, *aliases)
    }
    default_pronunciations = {
        value.casefold()
        for value in (
            *pronunciations,
            *zulu_matches,
            *code_switch_matches,
        )
    }
    retained_entries = tuple(
        entry
        for entry in dictionary.entries
        if not (
            entry.source == "application_default"
            and any(
                match.casefold() in default_pronunciations
                for match in entry.match_texts()
            )
        )
    )
    known = {
        match.casefold()
        for entry in (*dictionary.job_overrides, *retained_entries)
        for match in entry.match_texts()
    }
    defaults = tuple(
        PronunciationEntry(
            display_text=initialism,
            spoken_text=initialism,
            tts_text=letter_names,
            language=dictionary.language,
            source="application_default",
            confidence=1.0,
            kind="initials",
            notes=("Reviewed South African organisation initialism",),
        )
        for initialism, letter_names in pronunciations.items()
        if initialism.casefold() not in known
    )
    zulu_defaults = tuple(
        PronunciationEntry(
            display_text=item.token,
            spoken_text=item.token,
            tts_text=item.resolved_tts_text(),
            language=dictionary.language,
            source="application_default",
            confidence=1.0 if item.kind == "initials" else 0.95,
            kind=item.kind,
            aliases=item.aliases,
            notes=(
                "Curated South African public-affairs pronunciation",
                "Registry v2026.07; practical high-frequency list, not an official ranking",
            ),
        )
        for item in zulu_specs
        if not any(
            value.casefold() in known
            for value in (item.token, *item.aliases)
        )
    )
    code_switch_defaults = tuple(
        PronunciationEntry(
            display_text=display_text,
            spoken_text=display_text,
            tts_text=tts_text,
            language=dictionary.language,
            source="application_default",
            confidence=1.0,
            kind=kind,
            aliases=aliases,
            notes=(
                "Application-controlled hidden TTS code-switch pronunciation",
                ZU_CODE_SWITCH_PRONUNCIATION_VERSION,
                (
                    "South African English pronunciation through a standard "
                    "isiZulu Azure voice"
                    if kind == "english_code_switch"
                    else "Reviewed isiZulu pronunciation of a foreign place name"
                ),
            ),
        )
        for display_text, (tts_text, kind, aliases) in code_switches.items()
        if not any(
            value.casefold() in known
            for value in (display_text, *aliases)
        )
    )
    contextual_job_overrides = list(dictionary.job_overrides)
    known_job_matches = {
        value.casefold()
        for entry in contextual_job_overrides
        for value in entry.match_texts()
    }
    if locale == "zu-za":
        # A reviewed job override such as "noC Mackenzie" -> "no Makenzi"
        # declares that Azure/STT's isolated C is not part of the person's
        # name. Translation variants can legitimately change the isiZulu
        # prefix to uC or attach u directly to the name. Derive those contexts
        # from the reviewed override so every selected variant receives the
        # same pronunciation without hard-coding a particular person.
        for entry in dictionary.job_overrides:
            contextual_match = re.fullmatch(
                r"noC\s+(.+)",
                entry.display_text,
                re.IGNORECASE | re.UNICODE,
            )
            tts_match = re.fullmatch(
                r"no\s+(.+)",
                entry.tts_text,
                re.IGNORECASE | re.UNICODE,
            )
            if not contextual_match or not tts_match:
                continue
            display_name = contextual_match.group(1).strip()
            tts_name = tts_match.group(1).strip()
            for display_text in (
                f"uC {display_name}",
                f"u{display_name}",
            ):
                if display_text.casefold() in known_job_matches:
                    continue
                contextual_job_overrides.append(
                    PronunciationEntry(
                        display_text=display_text,
                        spoken_text=display_text,
                        tts_text=f"u {tts_name}",
                        language=dictionary.language,
                        source="job_override_context",
                        confidence=entry.confidence,
                        kind=entry.kind,
                        notes=(
                            "Derived from reviewed job-local noC name override",
                            ZU_CONTEXTUAL_NAME_PREFIX_VERSION,
                            f"source_entry_id={entry.entry_id}",
                        ),
                    )
                )
                known_job_matches.add(display_text.casefold())

    if (
        not defaults
        and not zulu_defaults
        and not code_switch_defaults
        and retained_entries == dictionary.entries
        and tuple(contextual_job_overrides) == dictionary.job_overrides
    ):
        return dictionary
    return PronunciationDictionary(
        dictionary_version=dictionary.dictionary_version,
        entries=(
            *retained_entries,
            *defaults,
            *zulu_defaults,
            *code_switch_defaults,
        ),
        job_overrides=tuple(contextual_job_overrides),
        language=dictionary.language,
        job_id=dictionary.job_id,
    )


def build_initialism_ssml_parts(
    tts_text: str,
    *,
    language: str,
) -> tuple[SSMLPart, ...]:
    """Render human-calibrated initialisms without changing visible text."""

    if language.casefold() != "zu-za":
        return ()
    alternatives = "|".join(map(re.escape, sorted(_ZU_CHARACTER_INITIALISMS)))
    pattern = re.compile(
        rf"(?<!\w)({alternatives})(?!\w)",
        re.IGNORECASE | re.UNICODE,
    )
    # Avoid emitting typed parts when plain text is sufficient.
    matches = list(pattern.finditer(tts_text))
    if not matches:
        return ()
    parts: list[SSMLPart] = []
    cursor = 0
    for match in matches:
        if match.start() > cursor:
            parts.append(TextPart(tts_text[cursor:match.start()]))
        parts.append(CharacterPart(match.group(0).upper()))
        cursor = match.end()
    if cursor < len(tts_text):
        parts.append(TextPart(tts_text[cursor:]))
    return tuple(parts)


__all__ = [
    "PRONUNCIATION_DICTIONARY_SCHEMA_VERSION",
    "PRONUNCIATION_ENTRY_SCHEMA_VERSION",
    "PRONUNCIATION_KINDS",
    "ZU_CODE_SWITCH_PRONUNCIATION_VERSION",
    "ZU_NATIVE_DATE_PRONUNCIATION_VERSION",
    "ZU_NATIVE_HONORIFIC_PRONUNCIATION_VERSION",
    "ZU_PREFIXED_CARDINAL_VERSION",
    "PronunciationDictionary",
    "PronunciationEntry",
    "PronunciationResult",
    "PronunciationSubstitution",
    "apply_pronunciation_dictionary",
    "build_initialism_ssml_parts",
    "expand_initials_for_tts",
    "get_zulu_public_affairs_acronym_registry",
    "normalise_dates_for_tts",
    "normalise_numbers_for_tts",
    "normalize_dates_for_tts",
    "normalize_numbers_for_tts",
    "with_default_organisation_initialisms",
]
