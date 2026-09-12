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
ZU_CODE_SWITCH_PRONUNCIATION_VERSION = "mathula-zu-code-switch-v2-coloured-v13.18.20"
ZU_NATIVE_DATE_PRONUNCIATION_VERSION = "mathula-zu-native-calendar-date-v2-loanword-month-v13.19.29"
ZU_NATIVE_YEAR_PRONUNCIATION_VERSION = "mathula-zu-native-year-reading-v3-faif-v13.19.28"
ZU_NATIVE_HONORIFIC_PRONUNCIATION_VERSION = "mathula-zu-native-honorific-flow-v1-v13.18.23"
ZU_CONTEXTUAL_NAME_PREFIX_VERSION = "mathula-zu-contextual-name-prefix-v1-v13.18.42"
ZU_REGION_INITIALISM_PRONUNCIATION_VERSION = "mathula-zu-region-initialism-v1-kzn-v13.18.63"
ZU_ENGLISH_SERVICE_CODE_SWITCH_VERSION = "mathula-zu-english-service-code-switch-v1-v13.19.12"
ZU_ENGLISH_ENTITY_CODE_SWITCH_VERSION = "mathula-zu-english-entity-code-switch-v9-brigitte-v13.19.25"
ZU_WEB_RESEARCHED_ORGANISATION_PRONUNCIATION_VERSION = "mathula-zu-web-researched-entity-pronunciation-v2-v13.19.14"
ZU_PREFIXED_CARDINAL_VERSION = "mathula-zu-prefixed-cardinal-v2-currency-safe-v13.18.62"
ZU_RAND_MILLION_CODE_SWITCH_VERSION = "mathula-zu-rand-million-code-switch-v1-v13.18.62"
ZU_SPACE_GROUPED_THOUSAND_CODE_SWITCH_VERSION = "mathula-zu-space-grouped-thousand-code-switch-v1"
ZU_ENGLISH_REFERENCE_NUMBER_CODE_SWITCH_VERSION = "mathula-zu-english-reference-number-code-switch-v1-v13.19.20"
ZU_VOICE_SPECIFIC_PRONUNCIATION_VERSION = "mathula-zu-voice-specific-pronunciation-v1"
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

# isiZulu glues short grammatical prefixes (subject "u-", associative "no-"/"na-",
# locative "ku-"/"e-", possessive/relative "ka-"/"wase-"/"nase-" etc.) directly onto
# a code-switched proper noun with no space -- "uBrown Mogotsi", "kuFadiel Adams",
# "noBrown Mogotsi", "enoFadiel Adams" are all real, ordinary isiZulu, not typos.
# A plain \b-style start boundary rejects every one of these, since the character
# immediately before the match is a word character. Only relax the start boundary
# when the match text itself begins with an uppercase letter (i.e. is genuinely a
# code-switched proper noun, never ordinary lowercase vocabulary), and only for a
# short (<=5 char) glued prefix that itself starts at a real word boundary -- so an
# unrelated, longer isiZulu word can never masquerade as a "prefix" of the noun.
_GLUED_PREFIX_LOOKBEHIND = "|".join(rf"(?<=\b[a-z]{{{n}}})" for n in range(1, 6))

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
    # Real user-reported production defect (job 8372120960474ef6b1d75af7de51a605):
    # the same real occurrence, glued to the isiZulu "i-"/"I-" prefix twice in
    # one job, sounded inconsistent between the two spots ("two different
    # pronunciations... back to back"). Root cause, confirmed live: the
    # previous hyphenated alias "Ay-dak" produces a double-hyphen sequence
    # once glued ("i-Ay-dak"), which Azure's zu-ZA-ThembaNeural phonemizer
    # rendered differently depending on sentence position. A fused,
    # non-hyphenated alias removes the ambiguity -- calibrated live against
    # both real sentences from this job; the user picked the fused form as
    # sounding consistent in both.
    _ReviewedAcronymSpec("IDAC", "acronym", "Aydak"),
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

# Regional abbreviations are kept separate from the fixed 100-term organisation
# registry: KZN names a province, not an organisation.  Azure's isiZulu voice
# has been observed trying to pronounce it as a word, so character-mode is the
# safest hidden TTS representation.
_ZU_REGION_CHARACTER_INITIALISMS = frozenset({"KZN"})
_ZU_CHARACTER_INITIALISMS = (
    frozenset(item.token for item in _ZU_SA_PUBLIC_AFFAIRS_ACRONYMS if item.character_mode)
    | _ZU_REGION_CHARACTER_INITIALISMS
)

# ``SA`` is ambiguous in general (it commonly abbreviates South Africa), so it
# must not be added to the global character-mode registry.  In this reviewed
# organisation name, however, it is explicitly the spoken initials S-A.  Match
# only the token itself so attached isiZulu forms such as ``ye-SA First Forum``
# keep their grammatical prefix outside <say-as>.
_ZU_REVIEWED_CHARACTER_PHRASE_INITIALISMS = re.compile(
    r"(?<!\w)(?P<initialism>SA)(?=\s+First\s+Forum(?!\w))",
    re.IGNORECASE | re.UNICODE,
)

# Anonymous-witness designations are letter names, never isiZulu syllables.
# Keep "Witness" as text and put only the reviewed A--K designation in Azure's
# typed character mode. This prevents B/P, C/Si, G/Ki and I/E identity swaps.
_ZU_REVIEWED_WITNESS_DESIGNATIONS = re.compile(
    r"(?<=\bWitness\s)(?P<initialism>[A-K])(?!\w)",
    re.IGNORECASE | re.UNICODE,
)

# IsiZulu locative and concord prefixes are commonly fused to an abbreviation
# in running text (for example eKZN, aseKZN and baseKZN).  They must remain
# outside <say-as>; only the initialism itself is spelled letter by letter.
_ZU_FUSED_INITIALISM_PREFIXES = (
    "kwase",
    "ngase",
    "base",
    "kase",
    "lase",
    "wase",
    "yase",
    "zase",
    "ase",
    "kwa",
    "nge",
    "e",
    "i",
    "u",
)
_ZU_FUSED_KZN = re.compile(
    rf"(?<!\w)(?:{'|'.join(_ZU_FUSED_INITIALISM_PREFIXES)})"
    r"(?P<token>KZN)(?!\w)",
    re.IGNORECASE | re.UNICODE,
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
    # ThembaNeural renders the English word "Service" as "civic/sewic" in
    # this protected organisation name.  The hidden alias was verified by a
    # zu-ZA Azure STT round-trip as "South African Police Service".
    "South African Police Service": (
        "South African Police Sir-vis",
        "organisation_name",
        (),
    ),
    # Both production voices were tested at Azure rate +5% followed by the
    # renderer's 1.03x atempo stage. en-ZA STT recovered the exact protected
    # identity from this hidden form; "Fierst" instead produced "fiast".
    "SA First Forum": (
        "Ess Ey Ferst Forum",
        "organisation_name",
        (),
    ),
    # Keep the approved English noun visible, but give the standard isiZulu
    # voice the non-rhotic South African-English vowel. In the complete
    # production sentence literal ``Ama-Hawks`` was heard/transcribed as
    # ``Ama-hauks``; the hidden ``Ama-Horks`` surface was recovered as
    # ``Ama-hoks``. This is a TTS-only alias and never changes captions.
    "Hawks": ("Horks", "english_code_switch", ()),
    # Live full-phrase calibration on both Thando and Themba recovered literal
    # ``The Hawks`` exactly with en-ZA STT.  Do not attach an isiZulu ``I-`` in
    # translation and do not rewrite the article as ``Dha``: together those
    # older rules produced the audible failure ``I da Hawks``.
    "The Hawks": ("The Hawks", "organisation_name", ()),
    # Real English institution name embedded in isiZulu (e.g. "...waseBritish
    # Rabanda Justice College usejoyina..."). Unmodified, ThandoNeural's en-ZA
    # STT round-trip recovered a garbled "Chastique Koleke" rather than
    # recognizable English. The hidden alias "Jastis Koleji" was calibrated
    # live (job 33cd7b46...) and recovered a clean, literal "Justice College"
    # on en-ZA STT round-trip. This is a TTS-only alias and never changes
    # captions.
    "Justice College": ("Jastis Koleji", "organisation_name", ()),
    # Real person (Madlanga Commission witness), spelling confirmed unchanged
    # against public reporting (Wikipedia "Brown Mogotsi", News24, Daily
    # Maverick etc.) -- do not alter the visible/spoken name. The web-researched
    # alias "Braun Mogotsi" fixed "Brown" but left the Setswana-origin surname
    # unmodified; user feedback on a real render reported the surname itself
    # still sounding wrong and specifically asked for a "kg"-style articulation
    # ("Mokgotsi", not "Mogotsi"). This hand-curated entry supersedes the
    # web-researched one (matches its canonical text) so both halves apply.
    "Brown Mogotsi": ("Braun Mokgotsi", "personal_name", ()),
    # Real person (Madlanga Commission evidence leader, "Advocate Lee Segeels
    # Ncube"). Unmodified, ThandoNeural read "ge" as an English soft-g/j sound
    # -- real en-ZA STT round-trip on job 33cd7b46... recovered "Lee Segeels"
    # as "Lisa Jills" (user-reported live: "Lee Seegels" heard as "Lee
    # Sejels"). Live calibration testing several respellings found the "gu"
    # digraph (as in "guest"/"guide") is what actually forces a hard-g
    # articulation from this voice's English-loanword phonemizer -- plain
    # "gh"/doubled-"g"/capitalization variants all still recovered a soft
    # consonant. "Seguels" recovered "girls"/"sequels" on repeated real
    # round-trips (hard g, no more soft-g/j), while the plain spelling never
    # did across every variant tried. This is a TTS-only alias and never
    # changes captions.
    "Segeels": ("Seguels", "personal_name", ()),
    # Real person (Andre Lincoln, referred to as "Mr. Lincoln"/"General Lincoln"
    # throughout this transcript). Unmodified, ThembaNeural's real en-ZA STT
    # round-trip recovered bare "Lincoln" as "Lingon" -- the "c" was read as a
    # nasalized "ng", losing the hard "k" sound entirely (user-reported live on
    # job 33cd7b46...). This also broke badly in the glued possessive form used
    # elsewhere in this transcript: raw "endlini kaLincoln" round-tripped as
    # "Engine calling all". Live calibration testing confirmed "Linken" (and
    # "Linkon") both recover a clean "Lincoln" in isolation, in the "General
    # Lincoln"/"Mr. Lincoln" context, AND in the glued "ka-" possessive form --
    # "Linken" gave the cleanest full-context round-trip ("General Lincoln Y."
    # vs. the unmodified "General Ling on"). This is a TTS-only alias and never
    # changes captions.
    "Lincoln": ("Linken", "personal_name", ()),
    # Real person (Brigitte Mabandla, former Minister of Justice; a training
    # college is named after her, "Brigitte Mabandla Justice College" -- job
    # 8372120960474ef6b1d75af7de51a605). Unmodified, ThandoNeural's real en-ZA
    # STT round-trip recovered "Brigitte" as garbled "pregite" (user-reported
    # live: "bridget is not pronounced correctly"). Live calibration testing
    # several respellings found "Bridjit" recovers a clean "Brigitte" on
    # repeated en-ZA STT round-trips (matching the real person's own
    # anglicized "Bridget"-style pronunciation); plain "Bridget" instead
    # recovered as unrelated "project". This is a TTS-only alias and never
    # changes captions.
    "Brigitte": ("Bridjit", "personal_name", ()),
    # The standard zu-ZA voices are not multilingual and Azure does not support
    # <lang xml:lang="en-ZA"> for them. Keep the protected/display spelling,
    # but give the voice a calibrated South African-English approximation instead
    # of letting it infer isiZulu phonetics for the English social descriptor.
    "coloured": ("Khalad", "english_code_switch", ("colored",)),
    # Real person (EFF Mbombela mayoral candidate, job
    # fb3d08b63fed4d90922b08f7e325b906). Unmodified, ThembaNeural's real
    # zu-ZA STT round-trip badly mangled the surname (heard variously as
    # "keet"/"kithi"/"kade" across repeated real attempts -- genuine call-to-
    # call TTS/STT variance for this name, confirmed directly: the RAW
    # spelling itself scored a clean 1.0 on one real round-trip and badly
    # mangled on others, too noisy for the automated round-trip verifier to
    # settle on its own). The correct pronunciation was found directly from
    # this job's own source audio: Pass 1's raw (pre-spelling-correction)
    # ASR transcript already read "Godrej Gade" for this exact name, before
    # a later autocorrect step normalised it to the tidier "Godfrey Gidi" --
    # real, job-specific audio evidence, not a guess. The self-supervised
    # pronunciation pipeline (native_dub.py's raw_asr_hint mechanism)
    # correctly proposed this exact respelling from that evidence on its
    # own; a human confirmed it by ear once automated round-trip scoring
    # proved too noisy to settle the case unattended. This is a TTS-only
    # alias and never changes captions.
    "Godfrey Gidi": ("Godrej Gade", "personal_name", ()),
    # Real person (SABC News senior reporter, correctly spelled "Taliesha
    # Naidoo" -- confirmed via real web research, job
    # b15075e7268049b491ee9e2222e5811f's own ASR transcript simplified it to
    # "Talisha"; captions keep the ASR spelling unchanged, only the hidden TTS
    # text is corrected here). User-reported live across FOUR rounds: (1)
    # "we are not pronouncing Taliesha Naidoo correctly", (2) after a fix
    # confirmed via STT round-trip alone ("Talisha Naydoo") -- "the 'o' in
    # Naydoo suppose to be dragged", (3) after a second fix ("Taleesha
    # Nai-dooo", confirmed via STT round-trip + real web research into the
    # surname's standard "NAI-doo" pronunciation) -- "the o is still not
    # dragging", followed by "I need the system to find a way of being able
    # to measure" -- built phoneme_recognizer.py's real per-phone duration
    # measurement (recognize_with_timing/phone_durations_ms) specifically for
    # this. (4) Applied that measurement mechanically -- picked "Taleesha
    # Nai-dooooo" for measuring a 50% longer final vowel (90ms vs. 60ms) at
    # the same phone-identity match quality -- and got real, direct
    # correction: "the drag is worse than no drag". Real, confirmed lesson:
    # raw measured phone duration is NOT a reliable proxy for perceived
    # naturalness -- every text-based elongation attempt tried (extra
    # repeated vowels, hyphens, commas) measured as "more dragged" while
    # sounding WORSE by ear, a consistent pattern across multiple real
    # attempts, not one bad guess. Reverted to round 3's "Taleesha Nai-dooo"
    # -- the last version never reported as sounding wrong, only as not yet
    # dragged enough -- and stopped trying to force further elongation via
    # spelling; genuine prosody control (SSML on this one word, not
    # currently supported by this file's SSML builder) would be the correct
    # tool if more drag is wanted, not another respelling guess. This is a
    # TTS-only alias and never changes captions.
    "Talisha Naidoo": ("Taleesha Nai-dooo", "personal_name", ()),
    # Generic English-origin code-switched word ("i-racism"/"ne-racism"), not a
    # name -- confirmed via real Azure TTS+STT round-trip testing (2026-09-12,
    # job fb3d08b63fed4d90922b08f7e325b906, user-reported live: "second turn
    # does not pronounce racism well"). The raw spelling is kept as the shared
    # default (it already recovers reasonably, 0.833, on zu-ZA-ThembaNeural),
    # but on zu-ZA-ThandoNeural it garbles ("iresism"/"irathism", 0.667) -- a
    # genuinely per-voice defect, not a universal one: see
    # _ZU_VOICE_SPECIFIC_CODE_SWITCH_PRONUNCIATIONS below for the
    # ThandoNeural-only override, which was the confirmed motivating case for
    # adding per-voice branching to PronunciationEntry at all.
    "racism": ("racism", "english_code_switch", ()),
}
# A handful of code-switch pronunciations measurably help on one configured
# Azure zu-ZA voice but measurably HURT on another (confirmed via real TTS+STT
# round-trip testing, not assumed) -- a genuine per-voice trade-off, unlike
# every other entry above which shares one tts_text across every voice. Any
# voice not listed here for a given display_text falls back to that entry's
# ordinary shared tts_text, so adding a voice-specific branch can never affect
# a voice (including any future one added to the roster) it wasn't measured
# against.
_ZU_VOICE_SPECIFIC_CODE_SWITCH_PRONUNCIATIONS: dict[str, dict[str, str]] = {
    "racism": {"zu-ZA-ThandoNeural": "raysizim"},
}
_SUPPORTED_SA_LANGUAGE_LOCALES = {
    "nr-za",  # isiNdebele
    "nso-za",  # Sepedi
    "ss-za",  # siSwati
    "st-za",  # Sesotho
    "tn-za",  # Setswana
    "ts-za",  # Xitsonga
    "ve-za",  # Tshivenda
    "xh-za",  # isiXhosa
    "zu-za",  # isiZulu
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
    r"(?P<prefix>kwa|ngo|lika|ku|no|ne|na|ka|u)?"
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


def _zulu_dynamic_prefixed_initialism_candidates(
    text: str,
    *,
    language: str,
    reviewed_entry: PronunciationEntry | None = None,
) -> list[tuple[int, int, PronunciationEntry]]:
    """Identify KZN inside a fused isiZulu prefix without consuming the prefix."""

    if language.casefold() != "zu-za":
        return []
    candidates: list[tuple[int, int, PronunciationEntry]] = []
    for match in _ZU_FUSED_KZN.finditer(text):
        before = match.group("token")
        entry = reviewed_entry or PronunciationEntry(
            display_text=before,
            spoken_text=before,
            # Character-mode keeps the literal token in the hidden text;
            # build_initialism_ssml_parts supplies the safe <say-as>.
            tts_text=before.upper(),
            language=language,
            source="application_default",
            confidence=1.0,
            kind="initials",
            notes=(
                "Reviewed KwaZulu-Natal regional initialism",
                ZU_REGION_INITIALISM_PRONUNCIATION_VERSION,
                "Preserves the fused isiZulu prefix outside character spelling",
            ),
        )
        candidates.append(
            (
                match.start("token"),
                match.end("token"),
                entry,
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


def _zulu_month_loanword_aliases() -> dict[str, str]:
    """Same alias-matching surface as _ZU_MONTH_BY_ALIAS, but mapping to the
    established English-borrowed isiZulu month name (_ZU_CALENDAR_MONTHS'
    aliases[1], e.g. "Novemba") instead of the traditional/canonical name
    (aliases[2]/the tuple's own first element, e.g. "Lwezi").

    Real user feedback, 2026-09-07: "probably only 10% of people who
    understand zulu know what 'lwamhla lu-1 kuLwezi' means, let's only use
    the borrowed english words" -- confirms this session's own earlier,
    separately-established finding that June/August/December/March's
    borrowed forms (Juni/Agasti/Disemba/Mashi) are the real, everyday isiZulu
    vocabulary, not the traditional calendar names. _zulu_native_date_tts
    previously normalized an already-correct loanword month spelling (e.g.
    someone wrote "Mashi") to the obscure traditional name ("Ndasa") in its
    TTS output -- exactly backwards from what real speakers use.
    """
    result: dict[str, str] = {}
    for _canonical, aliases in _ZU_CALENDAR_MONTHS:
        loanword = aliases[1]
        for alias in aliases:
            for prefix in ("", "ku", "u", "ngo", "ngo-"):
                result[f"{prefix}{alias}".casefold()] = loanword
    return result


_ZU_MONTH_LOANWORD_BY_ALIAS = _zulu_month_loanword_aliases()


def _zulu_month_english_code_switch_aliases() -> dict[str, str]:
    """Same alias-matching surface as _ZU_MONTH_BY_ALIAS (every prefix
    variant of every month spelling), but mapping to the REBUILT form with
    the month itself code-switched to plain English -- e.g. "ngoNovemba" ->
    "ngoNovember" -- keeping whatever Zulu grammatical prefix (locative
    "ngo-", associative "ku-"/"u-") was already glued on, since that is
    sentence grammar, not part of the date itself. Real user feedback,
    2026-09-07: "most native speakers in South Africa code switch dates to
    english instead of the native pronunciation" -- confirmed live for the
    "November" case (see _ZU_MONTH_BARE_DAY below); used generally here on
    that direction, not individually re-verified for all 12 months.
    """
    result: dict[str, str] = {}
    for canonical, aliases in _ZU_CALENDAR_MONTHS:
        english_name = aliases[0]
        for alias in aliases:
            for prefix in ("", "ku", "u", "ngo", "ngo-"):
                result[f"{prefix}{alias}".casefold()] = f"{prefix}{english_name}"
    return result


_ZU_MONTH_ENGLISH_CODE_SWITCH_BY_ALIAS = _zulu_month_english_code_switch_aliases()
_ZU_MONTH_PATTERN = "|".join(
    re.escape(value) for value in sorted(_ZU_MONTH_BY_ALIAS, key=lambda item: (-len(item), item))
)
_ZU_CALENDAR_DATE = re.compile(
    # Real production case (job fb3d08b63fed4d90922b08f7e325b906, 2026-09-07):
    # a Zulu class-11 relative concord ("lwa-") glued directly onto "mhla"
    # with no space ("lwamhla lu-1 kuLwezi") -- the plain `(?<!\w)` boundary
    # rejects this outright, since "mhla" doesn't start at a real word
    # boundary. Reuses _GLUED_PREFIX_LOOKBEHIND (already proven for the same
    # shape on code-switched proper nouns and _ZU_MONTH_BARE_DAY) so this
    # falls through to the raw/unhandled bare-digit reading instead of the
    # intended native construction whenever grammar glues a prefix on.
    rf"(?:(?<!\w)|{_GLUED_PREFIX_LOOKBEHIND})"
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

# Real bug, confirmed live (job b15075e7268049b491ee9e2222e5811f, user-
# reported: "bad pronunciation of 10000"): a South African-formatted amount
# using a SPACE as the thousands separator (e.g. "amaRandi ayi-10 000" for
# R10,000) has no reviewed reading at all -- unlike _ZU_CODE_SWITCH_RAND_
# MILLION above, nothing here ever handled thousands. The leading group
# ("10") is left as a bare digit (read fine by the voice's own native number
# handling), but the trailing "000" group, with a SPACE separating it from
# "10" rather than being part of one attached digit string, independently
# matches _ZU_BARE_REFERENCE_NUMBER's "3+ bare digits" reference-number
# pattern below and gets digit-spelled literally as "zeeroh zeeroh zeeroh"
# instead of being read as a magnitude -- confirmed directly from the real
# SSML sent to Azure: "amaRandi ayi-10 zeeroh zeeroh zeeroh ngemibandela...".
# Fixed per the user's own direction ("we should code switch '10 thousand'",
# "the model should be able to pronounce money like a normal person"): code-
# switch the whole space-grouped amount as English "<N> thousand" rather than
# reading it as isiZulu digits at all -- this is deliberately NOT restricted
# to a currency-word/"R" prefix immediately before it (unlike the million
# case above): an exact "<1-3 digits> 000" shape is essentially always a
# rounded quantity or currency amount in real transcript text, never a
# coincidental reference/docket number (those are not round multiples of
# 1000), so this is safe to apply generally.
_ZU_SPACE_GROUPED_THOUSAND = re.compile(
    r"(?<!\w)(?P<number>[1-9][0-9]{0,2})[  ](?P<thousands>000)(?!\w)",
    re.UNICODE,
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
        after = "izigidi " + _zulu_class10_two_digit_cardinal(number) + " zamaRandi"
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
    for match in _ZU_SPACE_GROUPED_THOUSAND.finditer(text):
        before = match.group(0)
        after = f"{match.group('number')} thousand"
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
                        "Application-controlled English thousand-magnitude code switch",
                        ZU_SPACE_GROUPED_THOUSAND_CODE_SWITCH_VERSION,
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


# Digit-by-digit English readings, calibrated by live Azure zu-ZA-ThembaNeural
# synthesis + en-US Azure STT round-trip confidence (the same technique already
# used for _ZU_REVIEWED_CODE_SWITCH_PRONUNCIATIONS above). Real dictionary IPA
# via <phoneme alphabet="ipa"> and <lang xml:lang="en-US"> were both tried first
# and both failed: <lang> is silently ignored by this voice, and <phoneme> IPA
# collapsed to byte-identical output as plain English spelling (this voice
# already applies its own implicit English grapheme-to-phoneme handling to
# embedded Latin-script words, so external phoneme guidance has no effect).
# Real English spelling alone is unreliable per digit ("four" scored 0.71,
# "three" scored 0.05) -- these hand-tuned respellings consistently outperform
# plain spelling. Validated on multiple real multi-digit numbers (STT-recovered
# the exact digit string every time, confidence 0.51-0.78), not just one case.
_ZU_ENGLISH_DIGIT_WORDS = {
    "0": "zeeroh",
    "1": "wan",
    "2": "tuu",
    "3": "triiy",
    "4": "four",
    "5": "fayiv",
    "6": "seeks",
    "7": "seven",
    "8": "eyt",
    "9": "nayn",
}
# A reference/identifier number (docket number, case number, phone number) --
# three or more bare digits with no isiZulu grammatical prefix attached and no
# currency marker -- reads as a full isiZulu cardinal-number expansion that can
# take several seconds for a single short number (confirmed real case: a docket
# number transcribed as "3978." measured 4930ms of synthesized audio against a
# 1120ms source window). This does NOT match an isiZulu-prefixed cardinal
# (handled naturally by _ZU_PREFIXED_TWO_DIGIT_CARDINAL above) or a rand amount
# (handled by _ZU_CODE_SWITCH_RAND_MILLION above) -- only a number with no
# surrounding isiZulu grammar, which is the reference/identifier case, not an
# ordinary quantity naturally embedded in isiZulu speech. A year-shaped 4-digit
# token immediately following a month name or a "ka-" year-reference glue is
# claimed first by _zulu_dynamic_date_candidates's month/ka-year mechanism
# (longer, earlier-starting match wins under apply()'s greedy leftmost
# selection) -- this mechanism is deliberately left otherwise unrestricted so
# an unrelated 4-digit reference/docket number (which may coincidentally be
# year-shaped, e.g. "Idokethi 2015") still reads digit-by-digit as before.
_ZU_BARE_REFERENCE_NUMBER = re.compile(
    r"(?<![\w-])(?<!R)(?<!R\s)(?P<number>[0-9]{3,})(?!\w)",
    re.UNICODE,
)


def _zulu_dynamic_reference_number_candidates(
    text: str,
    *,
    language: str,
) -> list[tuple[int, int, PronunciationEntry]]:
    if language.casefold() != "zu-za":
        return []
    candidates: list[tuple[int, int, PronunciationEntry]] = []
    for match in _ZU_BARE_REFERENCE_NUMBER.finditer(text):
        before = match.group(0)
        after = " ".join(_ZU_ENGLISH_DIGIT_WORDS[digit] for digit in before)
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
                    kind="english_code_switch",
                    notes=(
                        "Application-controlled English digit-by-digit code switch "
                        "for a reference/identifier number",
                        ZU_ENGLISH_REFERENCE_NUMBER_CODE_SWITCH_VERSION,
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


# Real user-reported production defect (job 8372120960474ef6b1d75af7de51a605):
# a bare 4-digit year embedded in isiZulu speech ("ngoDisemba 2024", "uDisemba
# ka-2024") was NOT being read as a year at all. _ZU_CALENDAR_DATE requires a
# day-of-month digit to match, so a "Month YYYY" or Zulu-possessive "ka-YYYY"
# form (both real, common ways of referring to a year with no day attached)
# fell through entirely -- either caught by the unrelated bare-reference-number
# mechanism (digit-by-digit English, "two zero two four") when space-separated,
# or, when hyphen-glued ("ka-2024"), excluded from THAT mechanism too and left
# to Azure's own raw locale-default number reading, which the user correctly
# described as sounding like "an amount" rather than a date. English speakers
# read a year as two two-digit groups ("twenty twenty-four"), not digit-by-
# digit and not a full cardinal quantity -- calibrated live via real Azure
# zu-ZA-ThandoNeural synthesis + en-ZA STT round-trip (both plain English
# spelling and a phonetic respelling recovered the digits "2024" cleanly; the
# user picked plain spelling by ear as sounding the most natural).
#
# Deliberately scoped to a year with real date-context evidence immediately
# next to it (a month name, or the "ka-" year-reference glue) rather than ANY
# bare 4-digit 19xx/20xx-shaped token -- a first, broader attempt at this fix
# was caught by the existing test suite regressing a real, different case: a
# genuinely unrelated reference/docket number that happens to be year-shaped
# ("Idokethi 2015" -- a docket number, not a year) must keep its established
# digit-by-digit reading, since there is no month/"ka-" context to distinguish
# it from an actual year.
# Real user-reported production defect (job 8372120960474ef6b1d75af7de51a605,
# "twenty-five" heard as clearly Zulu-accented, unlike its sibling
# "twenty-four" which already sounds like natural English): plain "five" has
# no English-multilingual voice to fall back on here (see the "coloured"
# entry's comment above -- these zu-ZA voices are not multilingual at all),
# so this locale's own G2P applies full isiZulu vowel realization to the
# trailing "-ve", instead of the single-syllable English /faɪv/. Live
# calibration (six respellings synthesized and sent for a real listen, since
# Azure STT round-trips every candidate to the correct digit regardless and
# cannot distinguish accent quality) confirmed "faif" recovers a natural
# English "five" ("fayf" also worked equally well; "faif" was kept as the
# more standard-looking English digraph).
#
# Real user-reported production defect (job fb3d08b63fed4d90922b08f7e325b906,
# 2026-09-07): "1" was ALSO still broken, discovered while diagnosing a
# separate "November 1" bare-day-number defect (see _ZU_MONTH_BARE_DAY
# below) -- plain "1" and spelled-out "one" both round-tripped via real
# Azure STT as "ON", not "1"/"one". Confirms this exact mechanism's "1"
# entry was never actually fixed alongside "5" -- it silently affects every
# year ending in 1 too (2001/2011/2021/2031), not just this one bare-day
# case. "wani" round-tripped cleanly as "1" on the same real voice.
_ZU_YEAR_ONES = {
    1: "wani", 2: "two", 3: "three", 4: "four", 5: "faif",
    6: "six", 7: "seven", 8: "eight", 9: "nine",
}
_ZU_YEAR_TEENS = {
    10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen",
    15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen",
}
_ZU_YEAR_TENS = {
    2: "twenty", 3: "thirty", 4: "forty", 5: "fifty",
    6: "sixty", 7: "seventy", 8: "eighty", 9: "ninety",
}


def _zulu_two_digit_year_words(value: int) -> str:
    """Render 0-99 the way an English speaker reads a year's last two digits
    ("05" -> "oh five", "24" -> "twenty-four", "00" -> "hundred")."""
    if value == 0:
        return "hundred"
    if value < 10:
        return f"oh {_ZU_YEAR_ONES[value]}"
    if value < 20:
        return _ZU_YEAR_TEENS[value]
    tens, ones = divmod(value, 10)
    word = _ZU_YEAR_TENS[tens]
    if ones:
        word += f"-{_ZU_YEAR_ONES[ones]}"
    return word


def _zulu_year_tts(year: int) -> str:
    """Render a 1900-2099 year as two spoken two-digit groups, e.g. 2024 ->
    "twenty twenty-four", matching how South African English broadcast speech
    actually reads a year -- never digit-by-digit, never a bare cardinal."""
    head, tail = divmod(year, 100)
    head_word = {19: "nineteen", 20: "twenty"}[head]
    return f"{head_word} {_zulu_two_digit_year_words(tail)}"


def _zulu_native_date_tts(match: re.Match[str]) -> str:
    day = int(match.group("day"))
    month = _ZU_MONTH_LOANWORD_BY_ALIAS[match.group("month").casefold()]
    prefix = str(match.group("prefix") or "").strip().casefold()
    year = str(match.group("year") or "")
    if prefix.startswith("ngumhla"):
        date = f"ngumhla {_zulu_date_ordinal_day(day)} ku{month}"
    elif prefix.startswith("umhla"):
        date = f"umhla {_zulu_date_ordinal_day(day)} ku{month}"
    else:
        date = f"mhla {_zulu_date_subject_day(day)} ku{month}"
    # A day+month+year date's own trailing year is deliberately left as raw
    # digits, matching this function's established, tested behavior (see
    # test_zulu_native_dates_v131822.py) -- the real, user-reported defect
    # this session fixes is only the DAY-LESS "Month YYYY"/"ka-YYYY" case
    # (_ZU_BARE_YEAR below), which never reaches this function at all.
    return date + year


# A year with no day attached, immediately preceded by a month name -- e.g.
# "ngoDisemba 2024" (_ZU_MONTH_PATTERN already includes the "ngo"-prefixed
# form as one of its alternatives, matching _ZU_CALENDAR_DATE's own month
# handling). Matches the whole "month year" span so it wins over the
# unrelated bare-reference-number mechanism under apply()'s greedy
# leftmost/longest selection (this match starts earlier, at the month, than
# a bare-number match starting at the year digits alone).
# Reuses _GLUED_PREFIX_LOOKBEHIND (defined near the top of this file for the
# exact same "short Zulu prefix glued directly onto a capitalized token, no
# space" shape already proven for code-switched proper nouns, e.g.
# "uBrown Mogotsi") rather than enumerating specific prefix forms one at a
# time -- real production text glues arbitrary concord prefixes onto a month
# name depending on grammatical agreement (e.g. "lwangoNovemba", class-11
# "lwa-" + the already-recognized "ngo-" month alias), and a fixed prefix
# list would need a new entry for every noun class this could ever agree
# with. See _ZU_MONTH_BARE_DAY below for the real defect this fixes.
_ZU_MONTH_START_BOUNDARY = rf"(?:(?<!\w)|{_GLUED_PREFIX_LOOKBEHIND})"
_ZU_MONTH_YEAR_ONLY = re.compile(
    rf"{_ZU_MONTH_START_BOUNDARY}(?P<month>{_ZU_MONTH_PATTERN})(?P<year>\s*,?\s*(?:19|20)[0-9]{{2}})(?!\w)",
    re.IGNORECASE | re.UNICODE,
)
# A bare day-of-month number with no ordinal suffix, immediately following a
# month name -- e.g. "Novemba 1" (mirroring English "November 1" word order,
# as opposed to the day-before-month "mhla ka-3 Mashi" order _ZU_CALENDAR_DATE
# already handles). The digit-count cap (1-2 digits, 1-31) means this can
# never collide with a 4-digit year match above; `(?!\w)` excludes an
# ordinal-suffixed form like "1st"/"4th", which is a separate, already-solved
# concern (the turn-block literal-preservation check, not TTS pronunciation).
#
# Real user-reported production defect (job fb3d08b63fed4d90922b08f7e325b906,
# 2026-09-07): the committed Zulu translation of "ahead of the November 1
# local government elections" reads a bare "1" with no date-reading treatment
# at all, and Azure's own raw number reading came out sounding like "on", not
# "1". First fixed as a Zulu-month + Zulu-phonetic-cardinal hybrid
# ("Novemba wani") -- confirmed working via real TTS+STT round trip -- but
# real user feedback redirected this: "we should be code switching to
# 'November first'... most native speakers in South Africa code switch dates
# to english instead of the native pronunciation." Re-verified live: plain
# "November first" (full English spelling, English ordinal) round-trips
# cleanly; the SAME day-number in a Zulu-spelled-month sentence ("Novemba
# first") mispronounces the ordinal as "fast" -- the month's own spelling
# measurably affects how the following word gets read, not just the digit
# itself. See _ZU_MONTH_ENGLISH_CODE_SWITCH_BY_ALIAS above and
# _english_ordinal_day_word below.
_ZU_MONTH_BARE_DAY = re.compile(
    rf"{_ZU_MONTH_START_BOUNDARY}(?P<month>{_ZU_MONTH_PATTERN})(?P<sep>\s*,?\s*)(?P<day>[1-9]|[12][0-9]|3[01])(?!\w)",
    re.IGNORECASE | re.UNICODE,
)

_ZU_ENGLISH_ORDINAL_ONES = {
    1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth",
    6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth",
}
_ZU_ENGLISH_ORDINAL_TEENS = {
    10: "tenth", 11: "eleventh", 12: "twelfth", 13: "thirteenth", 14: "fourteenth",
    15: "fifteenth", 16: "sixteenth", 17: "seventeenth", 18: "eighteenth", 19: "nineteenth",
}
_ZU_ENGLISH_ORDINAL_TENS = {
    2: "twentieth", 3: "thirtieth",
}


def _english_ordinal_day_word(value: int) -> str:
    """Render 1-31 as a plain English ordinal word ("1" -> "first", "21" ->
    "twenty-first") -- only "1" (the real reported case) has been directly
    live-verified round-tripping cleanly after an English month name; the
    rest follow standard English ordinal formation and are shipped on the
    same "code switch dates to English" direction the user confirmed
    generally applies for South African isiZulu speech, not individually
    re-verified for every day.
    """
    if value < 10:
        return _ZU_ENGLISH_ORDINAL_ONES[value]
    if value < 20:
        return _ZU_ENGLISH_ORDINAL_TEENS[value]
    tens, ones = divmod(value, 10)
    if not ones:
        return _ZU_ENGLISH_ORDINAL_TENS[tens]
    return f"{_ZU_YEAR_TENS[tens]}-{_ZU_ENGLISH_ORDINAL_ONES[ones]}"
# A year with no day or month attached, referenced via the Zulu possessive
# "ka-" glue -- e.g. "ka-2024" ("of 2024"), a real, common standalone way of
# naming a year in isiZulu (also seen as "ngonyaka ka-2024", "the year of
# 2024"). Excluded from a longer digit run so a genuine multi-digit
# reference number glued the same way is never mistaken for a year.
#
# Real user-reported production defect (job 8372120960474ef6b1d75af7de51a605,
# "kuka-2025" pronounced as a raw cardinal "two thousand and twenty-five"
# instead of a year): the locative-infinitive prefix "ku-" glued directly
# onto the possessive "ka-" ("kuka-2024"/"kuka-2025", e.g. "ngasekupheleni
# kuka-2024" -- "towards the end of 2024") is a real, common isiZulu
# construction, but the original `(?<!\w)` lookbehind required NO word
# character immediately before "ka-", so "kuka-" never matched at all --
# both years fell through to Azure's own raw locale-default number reading.
# The optional leading "ku" is matched as part of `prefix` (so the rebuilt
# text keeps it, e.g. "kuka-twenty twenty-five") and the boundary check
# still applies before whichever prefix actually starts, so an unrelated
# word merely ending in "u" immediately before "ka-YYYY" is still excluded
# exactly as before -- only the specific "ku"+"ka-" glue is now recognized.
_ZU_KA_YEAR = re.compile(
    r"(?<!\w)(?P<prefix>(?:ku)?ka-)(?P<year>(?:19|20)[0-9]{2})(?!\w)",
    re.IGNORECASE | re.UNICODE,
)


def _zulu_dynamic_date_candidates(
    text: str,
    *,
    language: str,
) -> list[tuple[int, int, PronunciationEntry]]:
    if language.casefold() != "zu-za":
        return []
    candidates: list[tuple[int, int, PronunciationEntry]] = []
    consumed: list[tuple[int, int]] = []
    for match in _ZU_CALENDAR_DATE.finditer(text):
        before = match.group(0)
        after = _zulu_native_date_tts(match)
        consumed.append((match.start(), match.end()))
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
    for match in _ZU_MONTH_YEAR_ONLY.finditer(text):
        # A month+year already covered by a full day+month+year date match
        # above (the month is inside that larger match) must not also get a
        # second, overlapping candidate here.
        if any(start <= match.start() < end for start, end in consumed):
            continue
        before = match.group(0)
        year_raw = match.group("year")
        digits = re.search(r"(?:19|20)[0-9]{2}", year_raw)
        if digits is None:
            continue
        separator = ", " if "," in year_raw else " "
        after = match.group("month") + separator + _zulu_year_tts(int(digits.group()))
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
                        "Application-controlled English year-style reading "
                        "for a month-only year with no day attached",
                        ZU_NATIVE_YEAR_PRONUNCIATION_VERSION,
                        "Approved/display digits remain immutable",
                    ),
                ),
            )
        )
    for match in _ZU_MONTH_BARE_DAY.finditer(text):
        # A month+year already covered by a full calendar-date match or a
        # month-year-only match above must not also get a second, overlapping
        # candidate here (defensive -- the digit-count cap already prevents a
        # 4-digit year from ever matching this pattern in the first place).
        if any(start <= match.start() < end for start, end in consumed):
            continue
        before = match.group(0)
        english_month = _ZU_MONTH_ENGLISH_CODE_SWITCH_BY_ALIAS.get(
            match.group("month").casefold(), match.group("month"),
        )
        after = english_month + match.group("sep") + _english_ordinal_day_word(int(match.group("day")))
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
                        "Application-controlled English code-switch reading "
                        "for a bare day-of-month with no ordinal suffix",
                        ZU_NATIVE_YEAR_PRONUNCIATION_VERSION,
                        "Approved/display digits remain immutable",
                    ),
                ),
            )
        )
    for match in _ZU_KA_YEAR.finditer(text):
        before = match.group(0)
        after = match.group("prefix") + _zulu_year_tts(int(match.group("year")))
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
                        "Application-controlled English year-style reading "
                        "for a Zulu-possessive 'ka-' year reference",
                        ZU_NATIVE_YEAR_PRONUNCIATION_VERSION,
                        "Approved/display digits remain immutable",
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
    # Optional per-voice tts_text overrides, e.g. {"zu-ZA-ThandoNeural": "raysizim"}.
    # A voice absent from this mapping (including any voice added to the roster
    # after this entry was written) always falls back to the shared `tts_text`
    # above -- this can only ever narrow a substitution's effect to a specific,
    # already-measured voice, never silently broaden it. Stored as a tuple of
    # (voice, text) pairs for hashability/immutability, matching this
    # dataclass's own `notes`/`aliases` convention; accepts a plain mapping too.
    tts_text_by_voice: tuple[tuple[str, str], ...] | Mapping[str, str] = ()

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
        raw_voice_map = self.tts_text_by_voice
        pairs = raw_voice_map.items() if isinstance(raw_voice_map, Mapping) else raw_voice_map
        voice_map: list[tuple[str, str]] = []
        seen_voices: set[str] = set()
        for voice_name, voice_text in pairs:
            voice_name = str(voice_name).strip()
            if not voice_name:
                raise ValueError("tts_text_by_voice keys must be non-empty voice names")
            if voice_name in seen_voices:
                raise ValueError(f"Duplicate tts_text_by_voice entry for voice {voice_name!r}")
            seen_voices.add(voice_name)
            voice_map.append((voice_name, _safe_literal(voice_text, f"tts_text_by_voice[{voice_name}]")))
        object.__setattr__(self, "tts_text_by_voice", tuple(sorted(voice_map)))
        if self.entry_id is None:
            identity = {
                "display_text": self.display_text,
                "spoken_text": self.spoken_text,
                "tts_text": self.tts_text,
                "language": self.language,
                "kind": self.kind,
                "source": self.source,
            }
            # Only included when non-empty, so an entry that never used this
            # field keeps computing the exact same entry_id it always has.
            if self.tts_text_by_voice:
                identity["tts_text_by_voice"] = [list(pair) for pair in self.tts_text_by_voice]
            object.__setattr__(
                self, "entry_id", "pron_" + hashlib.sha256(_canonical_json(identity).encode()).hexdigest()[:16]
            )
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
                "tts_text_by_voice": dict(value.get("tts_text_by_voice") or {}),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["notes"] = list(self.notes)
        value["aliases"] = list(self.aliases)
        value["tts_text_by_voice"] = dict(self.tts_text_by_voice)
        return value

    def tts_text_for_voice(self, voice: str | None) -> str:
        """The TTS-application text for a specific Azure voice, falling back to
        the shared ``tts_text`` when no voice is given or this entry carries no
        override for it."""
        if voice:
            for voice_name, voice_text in self.tts_text_by_voice:
                if voice_name == voice:
                    return voice_text
        return self.tts_text

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
        voice: str | None = None,
    ) -> PronunciationResult:
        """``voice``, when given, selects a matched entry's per-voice
        `tts_text_by_voice` override (falling back to its shared `tts_text`
        for any voice not listed) -- lets the same source text render
        differently for a voice with a confirmed, measured pronunciation
        defect without touching every other configured voice's own,
        already-working default."""
        text = _safe_literal(spoken_text, "spoken_text")
        candidates: list[tuple[int, int, int, int, str, PronunciationEntry]] = []
        scoped_entries = [(0, entry) for entry in self.job_overrides] + [(1, entry) for entry in self.entries]
        for scope_priority, entry in scoped_entries:
            if kinds is not None and entry.kind not in kinds:
                continue
            for match_text in entry.match_texts():
                # Short (<4 char) all-caps entries -- initialisms/acronyms like "DA",
                # "MK", "ANC" -- are exactly the ones prone to an accidental substring
                # collision under IGNORECASE (e.g. "da" inside "Rabanda"). Only relax
                # the start boundary for a genuinely long-enough proper-noun match.
                if match_text[:1].isupper() and len(match_text) >= 4 and entry.kind not in {"initials", "acronym"}:
                    start_boundary = rf"(?:(?<!\w)|{_GLUED_PREFIX_LOOKBEHIND})"
                else:
                    start_boundary = r"(?<!\w)"
                pattern = re.compile(
                    rf"{start_boundary}{re.escape(match_text)}(?!\w)", re.IGNORECASE | re.UNICODE
                )
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
                candidates.append((start, end, 2, -(end - start), str(entry.entry_id), entry))
        if kinds is None or "currency" in kinds:
            for start, end, entry in _zulu_dynamic_currency_candidates(
                text,
                language=self.language,
            ):
                candidates.append((start, end, 2, -(end - start), str(entry.entry_id), entry))
        if kinds is None or "number" in kinds or "currency" in kinds:
            for start, end, entry in _zulu_dynamic_number_candidates(
                text,
                language=self.language,
            ):
                candidates.append((start, end, 2, -(end - start), str(entry.entry_id), entry))
        if kinds is None or "english_code_switch" in kinds or "number" in kinds:
            for start, end, entry in _zulu_dynamic_reference_number_candidates(
                text,
                language=self.language,
            ):
                candidates.append((start, end, 2, -(end - start), str(entry.entry_id), entry))
        if kinds is None or "other" in kinds:
            for start, end, entry in _zulu_dynamic_honorific_candidates(
                text,
                language=self.language,
            ):
                # Explicit job/global entries remain authoritative over the
                # deterministic application fallback.
                candidates.append((start, end, 2, -(end - start), str(entry.entry_id), entry))
        if kinds is None or "initials" in kinds:
            reviewed_kzn_entry = next(
                (
                    entry
                    for _scope_priority, entry in scoped_entries
                    if entry.kind == "initials" and any(value.casefold() == "kzn" for value in entry.match_texts())
                ),
                None,
            )
            for start, end, entry in _zulu_dynamic_prefixed_initialism_candidates(
                text,
                language=self.language,
                reviewed_entry=reviewed_kzn_entry,
            ):
                candidates.append((start, end, 2, -(end - start), str(entry.entry_id), entry))
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
            applied_tts_text = entry.tts_text_for_voice(voice)
            output.append(text[cursor:start])
            output.append(applied_tts_text)
            before = text[start:end]
            substitutions.append(
                PronunciationSubstitution(
                    entry_id=str(entry.entry_id),
                    kind=entry.kind,
                    display_text=entry.display_text,
                    spoken_text=entry.spoken_text,
                    tts_text=applied_tts_text,
                    before=before,
                    after=applied_tts_text,
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
    region_initialisms = _ZU_REGION_CHARACTER_INITIALISMS if locale == "zu-za" else ()
    zulu_matches = {value.casefold() for item in zulu_specs for value in (item.token, *item.aliases)}
    region_matches = {value.casefold() for value in region_initialisms}
    pronunciations = {
        token: tts_text
        for token, tts_text in _SA_ORGANISATION_INITIALISMS.items()
        if token.casefold() not in zulu_matches
    }
    code_switches = _ZU_REVIEWED_CODE_SWITCH_PRONUNCIATIONS if locale == "zu-za" else {}
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
            *region_matches,
            *code_switch_matches,
        )
    }
    retained_entries = tuple(
        entry
        for entry in dictionary.entries
        if not (
            entry.source == "application_default"
            and any(match.casefold() in default_pronunciations for match in entry.match_texts())
        )
    )
    known = {
        match.casefold() for entry in (*dictionary.job_overrides, *retained_entries) for match in entry.match_texts()
    }
    # Automated web research may preserve the visible spelling as its hidden
    # rendering. Application-reviewed code switches retain priority, including
    # literal full-entity surfaces such as ``The Hawks``. Human/job-authored
    # overrides retain priority.
    code_switch_authoritative_known = {
        match.casefold()
        for entry in (*dictionary.job_overrides, *retained_entries)
        if entry.source != "web_research"
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
        if not any(value.casefold() in known for value in (item.token, *item.aliases))
    )
    region_defaults = tuple(
        PronunciationEntry(
            display_text=initialism,
            spoken_text=initialism,
            tts_text=initialism,
            language=dictionary.language,
            source="application_default",
            confidence=1.0,
            kind="initials",
            notes=(
                "Reviewed South African regional initialism",
                ZU_REGION_INITIALISM_PRONUNCIATION_VERSION,
                "Rendered with Azure character-mode SSML",
            ),
        )
        for initialism in sorted(region_initialisms)
        if initialism.casefold() not in known
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
            tts_text_by_voice=_ZU_VOICE_SPECIFIC_CODE_SWITCH_PRONUNCIATIONS.get(display_text, {}),
            notes=(
                "Application-controlled hidden TTS code-switch pronunciation",
                ZU_CODE_SWITCH_PRONUNCIATION_VERSION,
                (
                    "South African English pronunciation through a standard isiZulu Azure voice"
                    if kind in {"english_code_switch", "organisation_name"}
                    else "Reviewed isiZulu pronunciation of a foreign place name"
                ),
                *((ZU_ENGLISH_SERVICE_CODE_SWITCH_VERSION,) if display_text == "South African Police Service" else ()),
                *(
                    (ZU_ENGLISH_ENTITY_CODE_SWITCH_VERSION,)
                    if display_text in {"SA First Forum", "Hawks", "Justice College", "Brown Mogotsi", "Segeels", "Lincoln", "Brigitte"} else ()
                ),
                *(
                    (ZU_VOICE_SPECIFIC_PRONUNCIATION_VERSION,)
                    if display_text in _ZU_VOICE_SPECIFIC_CODE_SWITCH_PRONUNCIATIONS else ()
                ),
            ),
        )
        for display_text, (tts_text, kind, aliases) in code_switches.items()
        if not any(value.casefold() in code_switch_authoritative_known for value in (display_text, *aliases))
    )
    contextual_job_overrides = list(dictionary.job_overrides)
    known_job_matches = {value.casefold() for entry in contextual_job_overrides for value in entry.match_texts()}
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
        and not region_defaults
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
            *region_defaults,
            *code_switch_defaults,
        ),
        job_overrides=tuple(contextual_job_overrides),
        language=dictionary.language,
        job_id=dictionary.job_id,
    )


_RESEARCHED_ORGANISATION_LEADING_INITIALISM = re.compile(
    r"^(?P<initialism>[A-Z]{2,5})(?P<tail>\s+\S.*)$",
    re.UNICODE,
)


def with_web_researched_organisation_pronunciations(
    dictionary: PronunciationDictionary,
    research: Mapping[str, Any] | None,
) -> PronunciationDictionary:
    """Promote grounded entity pronunciations into scoped hidden TTS aliases.

    The historical function name is retained for API compatibility. New
    research can supply a conservative ``zu-ZA`` TTS rendering for any entity;
    older artifacts still receive the reviewed leading-initialism fallback.
    Visible and spoken text always retain canonical editorial spelling.
    """

    if dictionary.language.casefold() != "zu-za":
        return dictionary
    retained_overrides = tuple(entry for entry in dictionary.job_overrides if entry.source != "web_research")
    base = dictionary.with_job_overrides(dictionary.job_id, retained_overrides)
    if not isinstance(research, Mapping):
        return base
    known = {value.casefold() for entry in (*base.entries, *base.job_overrides) for value in entry.match_texts()}
    generated: list[PronunciationEntry] = []
    for raw_item in research.get("accepted_corrections") or ():
        if not isinstance(raw_item, Mapping):
            continue
        entity_type = str(raw_item.get("entity_type") or "other_name").casefold()
        try:
            confidence = float(raw_item.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        source_urls = tuple(
            str(value).strip()
            for value in (raw_item.get("grounded_source_urls") or raw_item.get("source_urls") or ())
            if str(value).strip()
        )
        canonical = str(raw_item.get("canonical_text") or "").strip()
        if not canonical or canonical.casefold() in known:
            continue
        pronunciation_mode = str(raw_item.get("pronunciation_mode") or "unknown").casefold()
        pronunciation_urls = tuple(
            str(value).strip()
            for value in (raw_item.get("grounded_pronunciation_evidence_urls") or ())
            if str(value).strip()
        )
        pronunciation_tts_text = str(raw_item.get("pronunciation_tts_text") or "").strip()
        try:
            pronunciation_confidence = float(raw_item.get("pronunciation_confidence") or 0.0)
        except (TypeError, ValueError):
            pronunciation_confidence = 0.0
        # Phase 18: a real TTS+STT round trip (native_dub.py's
        # _verify_pronunciation_corrections_round_trip) may have already
        # checked this exact respelling against actual Azure audio and found
        # it does NOT recover better than the raw spelling -- an explicit
        # False here means "we tried and it didn't work," not "unknown," and
        # must block promotion the same way a missing pronunciation_urls does.
        # A missing field (older artifacts, or verification never ran because
        # no Speech credentials were configured) defaults to True so every
        # cached artifact from before this feature existed keeps working.
        round_trip_verified = raw_item.get("round_trip_verified", True) is not False
        # Phase 24: a self-supervised guess (native_dub.py's
        # _request_self_supervised_pronunciation_candidates, used when web
        # research found no pronunciation evidence at all -- a real,
        # confirmed limitation for local/lesser-known names, not a bug) has
        # no pronunciation_urls by construction. An EXPLICIT, CONFIRMED
        # round-trip pass is real, direct acoustic evidence -- arguably
        # stronger than a URL, since it tests the actual claim rather than
        # inferring it from text -- so it is accepted as an alternative
        # evidence source. Gated on the round-trip having ACTUALLY run and
        # ACTUALLY passed (not just "verification never ran," which defaults
        # true above for backward compatibility) so an unverified guess can
        # never slip through this path.
        has_confirmed_round_trip = (
            raw_item.get("round_trip_verified") is True
            and raw_item.get("round_trip_candidate_score") is not None
        )
        if (
            pronunciation_tts_text
            and pronunciation_confidence >= 0.9
            and pronunciation_mode in {"initialism", "acronym", "word_name"}
            and (pronunciation_urls or has_confirmed_round_trip)
            and round_trip_verified
        ):
            kind = {
                "person": "personal_name",
                "place": "place_name",
                "organisation": "organisation_name",
                "organization": "organisation_name",
                "brand": "organisation_name",
            }.get(entity_type, "other")
            generated.append(
                PronunciationEntry(
                    display_text=canonical,
                    spoken_text=canonical,
                    tts_text=pronunciation_tts_text,
                    language=dictionary.language,
                    source="web_research",
                    confidence=pronunciation_confidence,
                    kind=kind,
                    notes=(
                        "Self-supervised, round-trip-verified entity pronunciation "
                        "(no web evidence available)"
                        if str(raw_item.get("correction_mode") or "") == "self_supervised_no_evidence"
                        else "Web-grounded target-locale entity pronunciation",
                        ZU_WEB_RESEARCHED_ORGANISATION_PRONUNCIATION_VERSION,
                        f"pronunciation_mode={pronunciation_mode}",
                        *(
                            (f"pronunciation_language={str(raw_item.get('pronunciation_language')).strip()}",)
                            if str(raw_item.get("pronunciation_language") or "").strip()
                            else ()
                        ),
                        *(
                            (f"pronunciation_ipa={str(raw_item.get('pronunciation_ipa')).strip()}",)
                            if str(raw_item.get("pronunciation_ipa") or "").strip()
                            else ()
                        ),
                        *(f"source_url={value}" for value in source_urls),
                        *(f"pronunciation_url={value}" for value in pronunciation_urls),
                    ),
                )
            )
            known.add(canonical.casefold())
            continue

        # Backward compatibility for pre-v12 research artifacts, which only
        # recorded canonical organisation styling and optional mode rather
        # than a grounded hidden rendering.
        if entity_type not in {"organisation", "organization"}:
            continue
        match = _RESEARCHED_ORGANISATION_LEADING_INITIALISM.fullmatch(canonical)
        if confidence < 0.9 or not source_urls or match is None:
            continue
        initialism = match.group("initialism")
        legacy_pronunciation_urls = tuple(
            str(value).strip() for value in raw_item.get("pronunciation_evidence_urls") or () if str(value).strip()
        )
        if pronunciation_mode in {"acronym", "word_name"}:
            continue
        if len(initialism) > 2 and not (pronunciation_mode == "initialism" and legacy_pronunciation_urls):
            continue
        source_forms = (
            str(raw_item.get("representative_text") or ""),
            *(str(value) for value in raw_item.get("aliases") or ()),
            *(str(item.get("raw_text") or "") for item in raw_item.get("mentions") or () if isinstance(item, Mapping)),
        )
        if not any(re.match(rf"^{re.escape(initialism)}(?:\b|[-\s])", value.strip()) for value in source_forms):
            continue
        generated.append(
            PronunciationEntry(
                display_text=canonical,
                spoken_text=canonical,
                tts_text=f"{_spell_zu_initialism(initialism)}{match.group('tail')}",
                language=dictionary.language,
                source="web_research",
                confidence=confidence,
                kind="organisation_name",
                notes=(
                    "Web-grounded canonical organisation styling",
                    ZU_WEB_RESEARCHED_ORGANISATION_PRONUNCIATION_VERSION,
                    f"character_mode_initialism={initialism}",
                    *(f"source_url={value}" for value in source_urls),
                    *(f"pronunciation_url={value}" for value in legacy_pronunciation_urls),
                ),
            )
        )
        known.add(canonical.casefold())
    if not generated:
        return base
    return base.with_job_overrides(
        base.job_id,
        (*base.job_overrides, *generated),
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
    fused_prefixes = "|".join(map(re.escape, sorted(_ZU_FUSED_INITIALISM_PREFIXES, key=len, reverse=True)))
    pattern = re.compile(
        rf"(?<!\w)(?P<prefix>{fused_prefixes})?"
        rf"(?P<initialism>{alternatives})(?!\w)",
        re.IGNORECASE | re.UNICODE,
    )
    # Avoid emitting typed parts when plain text is sufficient.
    matches = sorted(
        (
            *pattern.finditer(tts_text),
            *_ZU_REVIEWED_CHARACTER_PHRASE_INITIALISMS.finditer(tts_text),
            *_ZU_REVIEWED_WITNESS_DESIGNATIONS.finditer(tts_text),
        ),
        key=lambda item: item.start(),
    )
    if not matches:
        return ()
    parts: list[SSMLPart] = []
    cursor = 0
    for match in matches:
        if match.start() > cursor:
            parts.append(TextPart(tts_text[cursor : match.start()]))
        prefix = match.groupdict().get("prefix")
        if prefix:
            parts.append(TextPart(prefix))
        parts.append(CharacterPart(match.group("initialism").upper()))
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
    "with_web_researched_organisation_pronunciations",
    "expand_initials_for_tts",
    "get_zulu_public_affairs_acronym_registry",
    "normalise_dates_for_tts",
    "normalise_numbers_for_tts",
    "normalize_dates_for_tts",
    "normalize_numbers_for_tts",
    "with_default_organisation_initialisms",
]
