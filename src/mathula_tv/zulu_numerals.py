"""isiZulu numerals: counting words that agree with the noun, and how a figure is written in news (module 6 of the grammar foundation).

The cardinal patterns and the examples they must reproduce come from Mahlaza, Magwenzi, Keet and Khumalo, "Automatically Generating IsiZulu Words From Indo-Arabic
Numerals" (INLG 2024; two isiZulu grammarians judged the output, 90-100% valid in six of seven categories).  Which spelling real writers use is counted in 8.4 million
words of isiZulu news (``scripts/grammar_verify_numerals.py``).  ``cardinal`` covers 1-9,999 for the plural classes 2, 4, 6, 7, 8, 10 and for no agreement.
"""
from __future__ import annotations

from .zulu_agreement import by_class

STEMS = {1: "nye", 2: "bili", 3: "thathu", 4: "ne", 5: "hlanu"}
NOUN_NUMBERS = {6: "thupha", 7: "khombisa", 8: "shiyagalombili", 9: "shiyagalolunye"}          # 6-9 are nouns, isithupha ...: the stem is the noun without isi-
# the concord that counting words take is the adjective prefix of the noun's class (module 1); a nasal class loses its n before a vowel-initial word
AGREEMENT_CLASSES = ("2", "4", "6", "8", "10")        # counting happens in the plural classes; 5 and 7 are singular (elibili, esibili: 0 and 10 words in the news)
_CONCORD = {"2": "aba", "4": "emi", "6": "ama", "8": "ezi", "10": "ezin"}


def _nasal(concord: str, stem: str) -> str:
    """ezin + bili -> ezimbili, ezin + thathu -> ezintathu, ezin + hlanu -> ezinhlanu, ezin + ne -> ezine, ezin + nye -> ezinye."""
    if not concord.endswith("n"):
        return concord + stem
    base = concord[:-1]
    if stem.startswith("b"):
        return base + "m" + stem
    if stem.startswith("th"):
        return base + "nt" + stem[2:]
    if stem.startswith("hl"):
        return base + "nhl" + stem[2:]
    if stem.startswith("n"):
        return base + stem
    return concord + stem


def small(d: int, cls: str | None) -> str:
    """1-9 as a counting word. With agreement: abahlanu, ezimbili, abayisithupha; without: the number as a noun (isibili, isithupha)."""
    if cls is None:
        return ("kunye" if d == 1 else "isi" + STEMS[d]) if d <= 5 else "isi" + NOUN_NUMBERS[d]
    concord = _CONCORD[cls]
    if d <= 5:
        return _nasal(concord, STEMS[d])
    if concord == "ama":
        concord = "a"                                               # ayisithupha (news 724), not amayisithupha
    elif concord.endswith("n"):
        concord = concord[:-1]                                      # eziyisithupha
    return concord + "yisi" + NOUN_NUMBERS[d]                       # abayisithupha


def _merge(prefix: str, word: str) -> str:
    """The vowels of two joined morphemes: na + i- -> ne-, na + a- -> na-, ama + yi -> a-yi, aba + ngama -> abangama."""
    if prefix == "na":
        if word[0] == "i":
            return "ne" + word[1:]
        if word[0] == "a":
            return "na" + word[1:]
        if word[0] == "u":
            return "no" + word[1:]
        return "na" + word
    return prefix + word


def _first(cls: str | None, multiple: bool, unit: str, opening: bool = True) -> str:
    """The word that names the unit (shumi / khulu / nkulungwane) with its agreement: abayishumi, abangamashumi, ezingamakhulu, ayishumi.
    After 'na' (not the first unit of the number) the plain forms are used, as with no agreement: namakhulu, namashumi, nekhulu, neshumi.  The paper's pattern 1d has (yi|ngama) there,
    but in 8.4 million words of news it is namakhulu 38 against nangamakhulu 1, namashumi 32 against nangamashumi 0 (and the renderer already reads it that way); the model's own
    answers in the grammar test agreed (the item's gold followed the paper and was wrong)."""
    if cls is None or not opening:
        if unit == "nkulungwane":
            return "izinkulungwane" if multiple else "inkulungwane"
        return "ama" + unit if multiple else "i" + unit
    if unit == "nkulungwane":
        link = "yizi" if multiple else "yi"
        link += "nkulungwane"
    else:
        link = ("ngama" if multiple else "yi") + unit
    concord = _CONCORD[cls]
    if concord.endswith("n"):
        concord = concord[:-1]                                      # ezin -> ezi
    if concord == "ama":
        concord = "a"                                               # ama + yi- -> ayi-, ama + ngama- -> angama-
    return concord + link


def _count(d: int, unit: str) -> str:
    """How many tens / hundreds / thousands: amabili ... ayisithupha for 10s and 100s; ezimbili, ezintathu, ezine, ezinhlanu, eziyisithupha for 1000s."""
    if unit == "nkulungwane":
        return {2: "ezimbili", 3: "ezintathu", 4: "ezine", 5: "ezinhlanu"}.get(d) or "eziyi" + "si" + NOUN_NUMBERS[d]
    return "ama" + STEMS[d] if d <= 5 else "ayisi" + NOUN_NUMBERS[d]


def _remainder_small(d: int) -> str:
    """After 'na': nanye, nambili, nantathu, nane, nanhlanu, nesithupha ..."""
    if d <= 5:
        return _merge("na", {1: "nye", 2: "mbili", 3: "ntathu", 4: "ne", 5: "nhlanu"}[d])
    return _merge("na", "isi" + NOUN_NUMBERS[d])


def render() -> str:
    rows = ["number | class 2 (people) | class 4 (imi-) | class 6 (ama-) | class 10 (izin-: things, books)"]
    for n in (2, 3, 4, 5, 6, 7, 8, 9, 10):
        rows.append(f"{n} | " + " | ".join(cardinal(n, c) for c in ("2", "4", "6", "10")))
    table = "\n".join(rows)
    return f"""ISIZULU NUMBERS -- how real news writes them (8.4 million words) and how a counting word agrees with the noun. The digits themselves are spoken for you
(see Figures); this is for choosing between words and digits, and for the prefix a digit takes.
Words or digits. Small counts are written as WORDS that agree with the counted noun (two: amabili 4,669, ezimbili 3,174, emibili 2,659, ababili 2,339 words; three, four and
five likewise); anything larger, years, percentages and money are written as DIGITS. In the news: 148,000 bare digit strings and 82,000 digits with a prefix, against 37,000 number words.
Counting words (1-5 = concord + stem: -bili 2, -thathu 3, -ne 4, -hlanu 5; 6-9 are nouns, isithupha, isikhombisa, isishiyagalombili, isishiyagalolunye, with -yisi- when they agree; 10 = -yishumi):
{table}
The concord is the counted noun's adjective prefix; class 10 (izin-: izinto, izincwadi, izindaba) is the common plural for things, and its n changes before the stem: ezin + bili -> ezimbili,
+ thathu -> ezintathu, + hlanu -> ezinhlanu, + ne -> ezine. Singular classes (5, 7) are not counted: "one" is just the noun, or kunye when counting.
Bigger numbers. 11-99: the unit word, the number of tens, then na + the remainder. One ten: yi- (abayishumi, emiyishumi, eziyishumi, ayishumi); two or more: -ngama- (abangamashumi amabili 20,
ezingamashumi amathathu 30, angamashumi amane 40; 2-5 tens = ama-bili/-thathu/-ne/-hlanu, 6-9 = ayisithupha ...). The remainder: nanye 1, nambili 2, nantathu 3, nane 4, nanhlanu 5, nesithupha 6,
nesikhombisa 7, nesishiyagalombili 8, nesishiyagalolunye 9: abangamashumi amabili nanhlanu = 25 (class 2), emiyishumi nane = 14 (class 4). 100: -yikhulu (abayikhulu), 200+: -ngamakhulu
amabili; 1,000: -yinkulungwane, 2,000+: -yizinkulungwane ezimbili. Without agreement: amashumi amabili 20, ikhulu nanhlanu 105, amakhulu amahlanu 500, inkulungwane 1,000, izinkulungwane ezine
namakhulu amahlanu 4,500 (the thousands count: ezimbili 2, ezintathu 3, ezine 4, ezinhlanu 5, eziyisithupha 6 ...). In the plural, writers use -ngama- (ezingamashumi ama- 56 words, eziyishumi ama- 0).
Only the FIRST unit word carries the agreement; after na the hundreds and tens are the plain forms, with or without a counted noun: eziyizinkulungwane ezintathu namakhulu amathathu namashumi
amathathu nantathu = 3,333 (namakhulu 38 words, nangamakhulu 1; namashumi 32, nangamashumi 0; nekhulu, neshumi for a single one).
Digits with a prefix. A counted figure takes a concord + ngu- (or ngama-): abantu abangu-500 "500 people", iminyaka engu-34 "34 years" (the commonest: engu- 5,608, angu- 3,773, abangu- 1,636,
ezingu- 1,362), with ngu- almost always below 20 and about as often -ngama- as ngu- from 20 to 99 (engama-62, angama-27, zingama-23).
Years, dates and times. ngo- is the time preposition: kusukela ngo-2007 "from 2007" (ngo-YEAR 6,558 words, ngowe-YEAR 1,907), and with a small number a clock time: ngomgqibelo ngo-9 "Saturday at 9".
After a noun the "of" form is used: ngonyaka ka-2012 / ngonyaka we-2013 "in the year 2013", ngesizini ka-2012 "in the 2012 season", ngomhla ka-25 / mhla ka-15 "on the 25th / 15th",
ngomzuzu ka-90 "in the 90th minute" (1,137 words). ku- is "to / at / by": kuya ku-3 "up to 3", abalinganiselwa ku-30 "estimated at 30".
Percentages are mostly bare (17%, 59.3%: 3,617 words; ama-20%: 90 words). Money is bare (R3 000: 8,166 words; amaRandi for the word)."""


def cardinal(n: int, cls: str | None = None) -> str:
    """The cardinal number 1-9,999 as isiZulu words, agreeing with a noun of class ``cls`` ("2", "4", "6", "8", "10") or standing alone (None)."""
    if not 1 <= n <= 9999:
        raise ValueError("1-9,999 only")
    if cls is not None and cls not in AGREEMENT_CLASSES:
        raise ValueError(f"no checked agreement for class {cls}")
    if n < 10:
        return small(n, cls)
    parts: list[str] = []
    thousands, rest = divmod(n, 1000)
    hundreds, rest = divmod(rest, 100)
    tens, units = divmod(rest, 10)
    first = True

    def unit_word(multiple: bool, unit: str) -> str:
        nonlocal first
        word = _first(cls, multiple, unit, opening=first)
        out = word if first else _merge("na", word)
        first = False
        return out

    if thousands:
        parts.append(unit_word(thousands > 1, "nkulungwane"))
        if thousands > 1:
            parts.append(_count(thousands, "nkulungwane"))
    if hundreds:
        parts.append(unit_word(hundreds > 1, "khulu"))
        if hundreds > 1:
            parts.append(_count(hundreds, "khulu"))
    if tens:
        parts.append(unit_word(tens > 1, "shumi"))
        if tens > 1:
            parts.append(_count(tens, "shumi"))
    if units:
        parts.append(_remainder_small(units) if not first else small(units, cls))
    return " ".join(parts)
