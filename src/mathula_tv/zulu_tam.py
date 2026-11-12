"""isiZulu tense, aspect, mood and negation: how a verb is built, with the subject concord changes each form needs (module 2 of the grammar foundation).

Forms and counts come from two places, not from memory: the SADiLaR linguist-annotated corpus (``scripts/grammar_mine_tam.py``) and a ZulMorph analysis of
113 k word forms of real isiZulu news (``scripts/zulmorph_census.py``).  ``scripts/grammar_verify_tam.py`` builds words with ``conjugate`` for a set of verbs and
subjects and checks with ZulMorph that each is a well-formed word carrying the tag the form claims.  The builder covers verbs whose root begins with a
consonant; a vowel-initial root changes the concord (ngi + enza -> ngenza) and is left to the agent.
"""
from __future__ import annotations

from dataclasses import dataclass

from .zulu_agreement import CLASSES

PERSONS = {"1ps": "ngi", "2ps": "u", "1pp": "si", "2pp": "ni"}
# subject concord, indicative (module 1); class 1 takes a- after nga- and in the subjunctive and negative (the "secondary" concord)
SC = {**PERSONS, **{c.cls: c.sc.strip("-") for c in CLASSES if c.cls not in ("1a", "2a")}}
SECONDARY = {**SC, "1": "a"}
PARTICIPIAL = {**SC, "1": "e", "2": "be", "6": "e", "9": "e"}          # the classes the corpus and ZulMorph agree on; the others keep the primary concord

# negative concord: a + the secondary concord, with the irregular ones
_NEGATIVE = {"1ps": "angi", "2ps": "awu", "1": "aka", "3": "awu", "4": "ayi", "6": "awa", "9": "ayi"}


def negative_sc(subject: str) -> str:
    return _NEGATIVE.get(subject) or "a" + SECONDARY[subject]


def past_sc(subject: str) -> str:
    """The concord that carries the remote past: the concord fused with a- (ngi -> nga, si -> sa, u -> wa, i -> ya, lu -> lwa, ku -> kwa, ba -> ba)."""
    sc = SC[subject]
    if sc == "i":
        return "ya"
    if sc.endswith("i"):
        return sc[:-1] + "a"
    if sc.endswith("u"):
        return sc[:-1] + "wa"
    return sc if sc.endswith("a") else sc + "a"


def conjugate(root: str, subject: str, form: str) -> str:
    """Build a verb. ``root`` is the stem without -a (hamb-, khulum-); ``subject`` a person (1ps, 2ps, 1pp, 2pp) or noun class ("1", "7"...)."""
    if form == "present_long":
        return f"{SC[subject]}ya{root}a"
    if form == "present_short":
        return f"{SC[subject]}{root}a"
    if form == "still":
        return f"{SC[subject]}sa{root}a"
    if form == "already":
        sc = SC[subject]
        if subject == "2":
            return f"sebe{root}ile"                  # ba + se + be: sebehambile
        if sc in ("u", "a"):
            return f"{sc}se{root}ile"                # usehambile, asehambile (the concord stays in front of se-)
        if sc == "i":
            raise ValueError("no checked form for the i- concord (classes 4 and 9)")
        return f"se{sc}{root}ile"                    # sengihambile, sesihambile, selihambile, sekuhambile
    if form == "future":
        return f"{SC[subject]}zo{root}a"
    if form == "perfect":
        return f"{SC[subject]}{root}ile"
    if form == "remote_past":
        return f"{past_sc(subject)}{root}a"
    if form == "potential":
        return f"{SECONDARY[subject]}nga{root}a"
    if form == "participial":
        return f"{PARTICIPIAL[subject]}{root}a"
    if form == "negative_present":
        return f"{negative_sc(subject)}{root}i"
    if form == "negative_perfect":
        return f"{negative_sc(subject)}{root}anga"
    if form == "negative_future":
        return f"{negative_sc(subject)}zu{root}a"
    if form == "ngeke":
        return f"ngeke {SECONDARY[subject]}{root}e"            # the verb after ngeke is the subjunctive (-e): 71% of 8,271 news occurrences, 84% in Isolezwe
    if form == "not_yet":
        return f"{negative_sc(subject)}ka{root}i"
    if form == "subjunctive":
        return f"{SECONDARY[subject]}{root}e"
    raise ValueError(form)


@dataclass(frozen=True)
class Form:
    key: str           # the `conjugate` form
    meaning: str       # what English says
    pattern: str
    example: str       # attested in the corpus
    news: int          # ZulMorph tag count in the 113 k-form news census (0 = not counted separately)
    note: str = ""


FORMS: tuple[Form, ...] = (
    Form("present_long", "does / is doing (the verb ends the clause)", "SC + ya + root + a", "siyabona, bayahlanganyela", 7864,
         "ya- is dropped when a complement follows: ngihamba kakhulu, usho lokhu"),
    Form("present_short", "does / is doing (a complement follows)", "SC + root + a", "babandakanya, ihlinzeka, lenza", 0),
    Form("still", "is still doing", "SC + sa + root + a", "lisalandela, kusaqhubeka, asathuthuka", 4866),
    Form("already", "has already done / is now done", "se + SC + root + -ile/-e (u-, a- concords: use-, ase-; class 2: sebe-)",
         "sekuhlolwe, sesidlulile, sekwanele, sengihambile", 19620,
         "no checked form for the i- concord (classes 4, 9): rephrase; se- also stands before ya-forms: sesiyaqhubeka"),
    Form("future", "will do", "SC + zo + root + a", "sizoqala, uzobhala, kuzoqhubeka, izobheka", 38624,
         "zaku- and zoku- are the formal long future; yo-/yoku- is 'will go and'"),
    Form("perfect", "has done / did (recently)", "SC + root + -ile (or -e)", "ukhokhile, bashadile, iqukethe", 158367),
    Form("remote_past", "did (a while ago, narrative)", "SC fused with a- (wa-, ba-, sa-, nga-...) + root + a", "washiya, wathathelwa, lwabhalwa", 0),
    Form("potential", "can / may / could do", "SC + nga + root + a (class 1: a-)", "ungenza, angabeka, kungalungiswa, ingakhipha", 6577),
    Form("subjunctive", "so that X do / let X do", "SC + root + e (class 1: a-)", "uwethule, ubanike, ahlale", 2107),
    Form("participial", "while / when doing; the 'situative'", "SC (class 1: e-, class 2: be-) + root + a", "ekhuluma, ebuya, ebhala", 139061,
         "the largest verb group in news: the clause that says how, when or why something happened"),
    Form("negative_present", "does not do", "a + SC + root + i", "ayifaki, ayigqoki, angahambi", 14565,
         "class 1 aka-, you awu-, I angi-; -i is also the ending after nga: ukungasebenzi 'not working'"),
    Form("negative_perfect", "did not do", "a + SC + root + anga", "kungaphumelelekanga, kungadonswanga", 0),
    Form("ngeke", "will not / cannot do (the common form)", "ngeke + SC + root + e (the subjunctive)",
         "ngeke kube lula, ngeke aphinde aphumelele, angeke avume", 0,
         "8,271 occurrences in the news text; the verb after it ends in -e in 71% (84% in Isolezwe), -a in 13%"),
    Form("negative_future", "will not do (the long form)", "a + SC + zu + root + a", "angizuhamba", 227),
    Form("not_yet", "has not yet done", "a + SC + ka + root + i (after ukuthi, when, if: SC + nga + ka + root + i)",
         "alukapheli, akakaqali; kungakapheli, ungakapheli", 1516),
)

NEWS_NOTE = ("counts: ZulMorph tags in 113,565 word forms of isiZulu news (perfect 158 k, participial and situative about 139 k, future 39 k, negative 35 k, "
             "auxiliaries 20 k). The corpus examples are formal written Zulu.")


def render() -> str:
    rows = ["meaning | pattern | corpus example | news count"]
    for f in FORMS:
        note = f"  [{f.note}]" if f.note else ""
        rows.append(f"{f.meaning} | {f.pattern} | {f.example} | {f.news or '-'}{note}")
    table = "\n".join(rows)
    sc = ", ".join(f"{k}: {v}" for k, v in SC.items())
    return f"""ISIZULU TENSE, ASPECT, MOOD AND NEGATION -- forms from the SADiLaR annotated corpus and a ZulMorph analysis of real news. Choose the form for what the
English MEANS (still, already, will, can, has done), not for a word-by-word match; this is also how the unit is made shorter or longer without adding words.
{table}
Subject concords (indicative): {sc}. Secondary (after nga-, in the subjunctive and negatives) differs only for class 1: a-. Participial: class 1 e-, class 2 be-,
class 6 e-, class 9 e-, the rest as the indicative. Negative: a + concord (angi-, awu-, aka- for class 1, awa- for class 6, ayi- for classes 4 and 9, aba-, asi-, ani-...).
Remote past: the concord joins -a: ngi -> nga, si -> sa, ni -> na, u -> wa, i -> ya, li -> la, zi -> za, ba -> ba, lu -> lwa, ku -> kwa.
A root that begins with a vowel changes its concord (ngi + enza -> ngenza, u + enza -> wenza); build such forms from words you know.
Auxiliary ba/be ("was", "to be"): kube, babe, ibe, ube; kuze "until"; ngeke "will not"; sekufanele "it is now necessary". Imperative: the bare root + a (hamba), yi- + root
before a one-syllable root (yiya), plural -ni; negative imperative: musa ukuhamba. {NEWS_NOTE}"""
