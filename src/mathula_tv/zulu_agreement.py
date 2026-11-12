"""isiZulu noun classes and agreement: the table and the vowel-merge rule that the dub agent's grammar foundation (module 1) states.

Every form here is checked against the SADiLaR morphologically annotated isiZulu corpus (45,933 tokens, linguist-checked; the tag protocol is part of that
release): ``scripts/grammar_mine_agreement.py`` counts the forms by noun class and ``scripts/grammar_verify_agreement.py`` fails when a form below is not
found in the corpus for its class, or when the vowel-merge rule disagrees with what the annotators wrote.  Nothing here comes from a model's memory.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NounClass:
    cls: str
    prefix: str            # augment + basic prefix of the noun, as written in the corpus tags (NPrePre + BPre)
    partner: str           # the class that is usually its plural (or singular)
    sc: str                # subject concord, indicative
    oc: str                # object concord
    poss: str              # possessive concord (the -a- of "of"); merges with the following noun
    rel: str               # relative concord
    adj: str               # adjective prefix
    quant: str             # quantitative concord (-onke "all", -odwa "only")
    dem_near: str          # demonstrative, near the speaker ("this")
    dem_far: str           # demonstrative, away from the speaker ("that")
    example: str           # singular / plural, both found in the corpus


CLASSES: tuple[NounClass, ...] = (
    NounClass("1",  "umu-/um-",   "2",     "u-",  "mu-/m-",  "wa",   "o-/e-",  "omu-",   "wo-", "lo",    "lowo",  "umuntu / abantu"),
    NounClass("1a", "u-",         "2a",    "u-",  "mu-/m-",  "wa",   "o-/e-",  "omu-",   "wo-", "lo",    "lowo",  "uhulumeni"),
    NounClass("2",  "aba-",       "1",     "ba-", "ba-",     "ba",   "aba-",   "aba-",   "bo-", "laba",  "labo",  "abantu / umuntu"),
    NounClass("2a", "o-",         "1a",    "ba-", "ba-",     "ba",   "aba-",   "aba-",   "bo-", "laba",  "labo",  "obaba / ubaba"),
    NounClass("3",  "umu-/um-",   "4",     "u-",  "wu-",     "wa",   "o-",     "omu-",   "wo-", "lo",    "lowo",  "umsebenzi / imisebenzi"),
    NounClass("4",  "imi-",       "3",     "i-",  "yi-",     "ya",   "e-",     "emi-",   "yo-", "le",    "leyo",  "imisebenzi / umsebenzi"),
    NounClass("5",  "i-/ili-",    "6",     "li-", "li-",     "la",   "eli-",   "eli-",   "lo-", "leli",  "lelo",  "ilizwe / amazwe"),
    NounClass("6",  "ama-",       "5",     "a-",  "wa-",     "a",    "a-",     "ama-",   "o-",  "la",    "lawo",  "amazwe / ilizwe"),
    NounClass("7",  "isi-",       "8",     "si-", "si-",     "sa",   "esi-",   "esi-",   "so-", "lesi",  "leso",  "isicelo / izicelo"),
    NounClass("8",  "izi-",       "7",     "zi-", "zi-",     "za",   "ezi-",   "ezi(n)-", "zo-", "lezi",  "lezo",  "izicelo / isicelo"),
    NounClass("9",  "in-/im-/i-", "10",    "i-",  "yi-",     "ya",   "e-",     "en-/em-", "yo-", "le",    "leyo",  "indlela / izindlela"),
    NounClass("10", "izin-/izim-", "9, 11", "zi-", "zi-",     "za",   "ezi-",   "ezin-",  "zo-", "lezi",  "lezo",  "izindlela / indlela"),
    NounClass("11", "ulu-/u-",    "10",    "lu-", "lu-",     "lwa",  "olu-",   "olu-",   "lo-", "lolu",  "lolo",  "ulwazi"),
    NounClass("14", "ubu-",       "-",     "bu-", "bu-",     "ba",   "obu-",   "obu-",   "bo-", "lobu",  "lobo",  "ubufakazi"),
    NounClass("15", "uku-",       "-",     "ku-", "ku-",     "kwa",  "oku-",   "oku-",   "ko-", "lokhu", "lokho", "ukudla"),
)

# What the annotators wrote when the last vowel of a prefix (possessive concord, na-, nga-, ka-, njenga-) meets the first vowel of the word after it.
# Counts in the corpus (PossConc / AdvPre): a+i -> e 1909 / 1005, a+u -> o 1291 / 991, a+a -> a 425 / 341, a+o -> o 29 / 79, a+e -> e 34 / 9.
VOWEL_MERGE = {"a": "a", "e": "e", "i": "e", "o": "o", "u": "o"}


def merge_vowels(prefix: str, word: str) -> str:
    """Join a prefix ending in -a (wa, ya, za, kwa; na, nga, ka, njenga) to a word that begins with a vowel the way isiZulu writes it:
    wa + umuntu -> womuntu, ya + indlu -> yendlu, na + izinto -> nezinto, nga + amandla -> ngamandla."""
    if prefix.endswith("a") and word and word[0] in VOWEL_MERGE:
        merged = VOWEL_MERGE[word[0]]
        if merged == "o" and prefix.endswith("wa") and len(prefix) > 2:      # kwa + uku -> koku, lwa + umuntu -> lomuntu: the glide goes (wa + umuntu -> womuntu keeps it)
            return prefix[:-2] + "o" + word[1:]
        return prefix[:-1] + merged + word[1:]
    return prefix + word


_LOCATIVE_END = {"a": "eni", "e": "eni", "i": "ini", "o": "weni", "u": "wini"}


def locative(noun: str) -> str:
    """"In / at / on" for a common noun written with its augment: imiphakathi -> emiphakathini, indlela -> endleleni, umkhakha -> emkhakheni, uhlelo -> ohlelweni.
    The first vowel: i- and a- become e-, um- becomes em-, any other u- becomes o-.  The last vowel: -a and -e become -eni, -i -ini, -o -weni, -u -wini."""
    if len(noun) < 3 or noun[0] not in "aiu":
        return noun
    if noun[0] in "ia":
        head = "e" + noun[1:]
    elif noun.startswith("uku"):
        head = "e" + noun[1:]                # ukuphela -> ekuphela (class 15 takes e-, not o-)
    elif noun.startswith("umu") and noun[3] not in "aeiou":
        head = "em" + noun[3:]               # umuzi -> emzi
    elif noun.startswith("um"):
        head = "e" + noun[1:]                # umkhakha -> emkhakha
    else:
        head = "o" + noun[1:]                # ulimi -> olimi
    last = head[-1]
    if last not in _LOCATIVE_END:
        return head + "ini"
    ending = _LOCATIVE_END[last]
    if last in "ou" and len(head) > 1 and head[-2] == "w":     # indawo -> endaweni (the w is already there)
        ending = ending[1:]
    return head[:-1] + ending


def by_class(cls: str) -> NounClass:
    return next(c for c in CLASSES if c.cls == cls)


def render() -> str:
    """The module as the dub agent reads it."""
    rows = ["class | noun prefix | pair | subj | obj | of (poss.) | rel | adj | all/only | this | that | example"]
    for c in CLASSES:
        rows.append(f"{c.cls} | {c.prefix} | {c.partner} | {c.sc} | {c.oc} | {c.poss} | {c.rel} | {c.adj} | {c.quant} | {c.dem_near} | {c.dem_far} | {c.example}")
    table = "\n".join(rows)
    return f"""ISIZULU NOUN CLASSES AND AGREEMENT -- forms checked against the SADiLaR linguist-annotated corpus (45,933 words). Apply them to any noun.
Every word that describes or refers to a noun takes the form of that noun's class, and so does the verb's subject (and object). Find the noun's class from
its prefix, then read across: subj/obj = the verb's concords (indicative; negative, subjunctive and participial forms belong to the verb rules), of (poss.) =
the possessive concord, rel = relative, adj = adjective prefix.
{table}
Vowel merge. A prefix ending in -a (the possessive concord, na- "with/and", nga- "by/in", ka-, njenga- "like") joins a following noun that starts with
a vowel: a+a -> a, a+i -> e, a+u -> o, a+e -> e, a+o -> o. wa + umuntu -> womuntu; ya + indlu -> yendlu; ba + amazwe -> bamazwe; na + izinto -> nezinto;
nga + umthetho -> ngomthetho. The possessive concord is chosen by the class of the POSSESSED noun, not the owner: umsebenzi (3) -> wa-, izindlela (10) -> za-.
After -kwa and -lwa the w goes in this merge: kwa + ukuba -> kokuba, lwa + umuntu -> lomuntu (wa + umuntu -> womuntu keeps it).
Owners that are a person's name or a class 1a noun take -ka- and lose their own u: izinto zikahulumeni, kahulumeni (ka + the name for a class 1 owner).
is / are (copulative). Before a noun that starts with i-: yi- (yinkantolo, yisicelo). Before u-, a-, o-: ng- (nguhulumeni, ngabasebenzisi, ngomthetho);
in classes 3, 11 and 15 wu- is also written (wukuphungula "it is to reduce").
Locative ("in, at, on"). e-/o- before the noun and -ini after it. First vowel: i- and a- become e- (imiphakathi -> emiphakathini, indlela -> endleleni),
um- becomes em- (umkhakha -> emkhakheni), uku- becomes eku- (ukuphela -> ekupheleni), other u- become o- (uhlelo -> ohlelweni, ulimi -> olimini). Last vowel:
-a and -e become -eni, -i becomes -ini, -o becomes -weni (after w: indawo -> endaweni). A stem ending in -mbo or -m can change its consonant before the ending
(imithambo -> emithanjeni, ukuxhumaniswa -> ekuxhunyanisweni): when unsure of the stem, use a different construction. Names, class 1a nouns and pronouns take
ku- and no -ini (kuhulumeni, kubo). The rules above reproduced 99.7% of the vowel merges and 84% of the locatives in the annotated corpus."""
