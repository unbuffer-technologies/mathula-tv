"""isiZulu relative clauses: "the people who work", "the book that I want", "a decision which was taken" (module 3 of the grammar foundation).

A relative clause is a verb that begins with the relative concord of the noun it describes.  The shapes below are counted in 264 k words of real isiZulu news
(ZulMorph analysis of 113 k word forms, ``scripts/zulmorph_census.py``) and in the SADiLaR annotated corpus (``scripts/grammar_mine_tam.py``); the builder is
checked with ZulMorph by ``scripts/grammar_verify_relative.sh``.  A root that begins with a vowel changes the concord and is left to the agent.
"""
from __future__ import annotations

from dataclasses import dataclass

from .zulu_agreement import CLASSES
from .zulu_tam import past_sc

# relative concord by the noun it describes (module 1 table); class 1 also takes e- (54 of 375 in the corpus)
RELATIVE_CONCORD = {c.cls: c.rel.split("/")[0].strip("-") for c in CLASSES if c.cls not in ("1a", "2a")}
# when the clause has its own person subject ("that I want"), the relative concord fuses with that subject
PERSON_RELATIVE = {"1ps": "engi", "1pp": "esi", "2pp": "eni"}
ORDER = ("present", "present_yo", "perfect", "future", "still", "negative", "past")


def rc(antecedent: str, subject: str | None = None) -> str:
    """The relative concord: of the described noun's class, or fused with a person subject (engi-, esi-, eni-)."""
    return PERSON_RELATIVE[subject] if subject in PERSON_RELATIVE else RELATIVE_CONCORD[antecedent]


def relative(root: str, antecedent: str, form: str, *, subject: str | None = None, oc: str = "") -> str:
    """Build a relative verb. ``root`` is the stem without -a (sebenz-); ``antecedent`` the noun's class ("2", "9"...); ``subject`` a person (1ps, 1pp, 2pp)
    when the clause has its own subject (object-type relative); ``oc`` an object concord ("yi", "ba") placed after the relative concord."""
    head = rc(antecedent, subject) + oc
    if form == "present":
        return f"{head}{root}a"
    if form == "present_yo":
        return f"{head}{root}ayo"
    if form == "perfect":
        return f"{head}{root}ile"
    if form == "future":
        return f"{head}zo{root}a"
    if form == "still":
        return f"{head}sa{root}a"
    if form == "negative":
        return f"{head}nga{root}i"
    if form == "past":
        if subject is not None or antecedent not in PAST_CLASSES:
            raise ValueError("no checked past relative for this subject / class")
        return rc(antecedent) + past_sc(antecedent) + oc + root + "a"          # o + wa + hamba -> owahamba
    raise ValueError(form)


PAST_CLASSES = ("1", "4", "9")         # owanqoba (1), eyacikelwe (4) are in the corpus; the rest are not checked


@dataclass(frozen=True)
class Shape:
    key: str
    meaning: str
    pattern: str
    example: str
    news: int


SHAPES: tuple[Shape, ...] = (
    Shape("present", "who/which does (the common one)", "RC + root + a", "abasebenza, ezidinga, ofuna, okubeka", 83667 + 39875),
    Shape("present_yo", "who/which does (nothing follows the verb in its clause)", "RC + root + ayo", "elandelayo, okulandelayo, ezilandelayo, esebenzayo", 18869 + 9708),
    Shape("perfect", "who/which has done / is (stative)", "RC + root + ile (or -e)", "olungile, ebalulekile, ohlukile, ephelele", 56929 + 38432),
    Shape("future", "who/which will do", "RC + zo + root + a", "ozoqala, abazobhekana, okuzokwenza", 5000 + 2086),
    Shape("still", "who/which still does", "RC + sa + root + a", "abasakhula, abasaqhubeka, osanda", 1606 + 106),
    Shape("negative", "who/which does not do", "RC + nga + root + i", "abangasebenzi, ezingasebenzi, ongahambi", 4189 + 915),
    Shape("past", "who/which did (narrative; checked for classes 1, 4, 9)", "RC + the remote-past concord (owa-, eya-) + root + a", "owanqoba, owahlaba, owaqokwa", 0),
)

NEWS_NOTE = ("counts: relative verbs in 264 k words of isiZulu news: present 152 k, perfect 95 k, future 7 k, negative 6 k, still 2 k; in the present, the form "
             "with -yo ends its clause about ten times as often as the form without (29% against 3% followed by a punctuation mark).")


def render() -> str:
    rows = ["meaning | pattern | corpus example | news words"]
    for s in SHAPES:
        rows.append(f"{s.meaning} | {s.pattern} | {s.example} | {s.news or '-'}")
    table = "\n".join(rows)
    rc_table = ", ".join(f"{k}: {v}-" for k, v in RELATIVE_CONCORD.items())
    return f"""ISIZULU RELATIVE CLAUSES ("the people who work", "the book that I want") -- shapes from 264 k words of real news and the annotated corpus.
A relative clause is ONE verb that starts with the relative concord of the noun it describes, so it is short: use it where English needs "who / which / that".
Relative concord by the noun's class: {rc_table}
{table}
Subject of the clause is the described noun ("who works"): RC + verb. When the clause has its own subject ("that I want", "which we saw"), the relative
concord fuses with that subject: engi- (I), esi- (we), eni- (you pl); an object concord follows: engiyifunayo "that I want it", esibabonayo "that we see them",
engikhulumile "that I spoke".
-yo: a present relative verb takes -yo (elandelayo, ezidingayo, okwenzekayo) or not (abasebenza, ezidinga). In the news the -yo forms are followed by a
punctuation mark 29% of the time and the forms without -yo 3% of the time: use -yo for a relative verb that closes its clause or phrase (isigaba esilandelayo),
and the bare form when an object or adverb follows it (abasebenza kakhulu). After the perfect -yo is rare (olungile). Passive and neuter extensions go before
the ending as usual (ebhaliwe, ezibizwa, okudingekayo).
Relative stems. A few words are relatives themselves, with no verb: RC + stem. The ones in the corpus: -thile and -thize "certain" (ezithile), -nzima "heavy",
-ngcono "better" (engcono), -lula "easy" (ezilula), -mnyama "dark", -mpofu "poor", -nzulu "deep" (olunzulu). Adjectives such as -khulu "big" and -hle "good"
take the adjective prefix instead (abakhulu, abahle; module 1).
{NEWS_NOTE}"""
