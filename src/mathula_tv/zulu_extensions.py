"""isiZulu verb extensions: one suffix does the work of an English phrase (module 4 of the grammar foundation).

    -is-  causative   "make / cause X to"       hambisa   -el-  applicative "for / to / at / with"   hambela
    -w-   passive     "be done (by)"            hanjwa    -an-  reciprocal  "each other"             hambana
    -ek-  neuter      "can be done / happens"   hambeka   -elel-/-isis- intensive "thoroughly"

The forms, their order when they stack and the sound changes come from the SADiLaR annotated corpus (``scripts/grammar_mine_tam.py``) and from a ZulMorph
analysis of real news (113 k word forms); ``scripts/grammar_verify_extensions.sh`` builds words with ``extend`` and checks them with ZulMorph, and checks that the
wrong, unpalatalised passives (khulumwa, bambwa) are refused.
"""
from __future__ import annotations

from dataclasses import dataclass

FORMS = {"caus": "is", "appl": "el", "recip": "an", "neut": "ek", "intens": "elel"}
# roots whose neuter is -akal- and not -ek- (news: kubonakala 883, kuboneka 0; tholakala, zwakala, kholakala ...); -ek- is the default (enzeka 17,585 against enzakala 442)
AKAL_ROOTS = ("bon", "thol", "zw", "on", "khol", "sol", "siz", "qond", "bez", "khohl", "swel", "wez", "sez", "yon", "bebez", "biz")
# before the passive -w- the last consonant of the stem, if it is a labial, changes (counts: news words; examples are written words)
PALATALISE = (("mb", "nj"), ("bh", "j"), ("ph", "sh"), ("p", "sh"), ("m", "ny"), ("b", "tsh"))
VOWELS = "aeiou"


def soften(root: str) -> str:
    """The root's last consonant in a passive verb: a labial softens (bop -> bosh, khulum -> khuluny, bamb -> banj, gweb -> gwetsh, gubh -> guj).
    9 verbs in 2.7 k news words: 2,697 softened passives (kukhulunywa, sithunyelwe, kuboshwe, ugwetshwe, ukukhonjiswa), 1 unsoftened."""
    for plain, soft in PALATALISE:
        if root.endswith(plain):
            return root[: -len(plain)] + soft
    return root


def extend(root: str, extensions: list[str], *, ending: str = "a") -> str:
    """Build a verb stem from a root (hamb-) and extensions in the order given ("caus", "appl", "recip", "neut", "intens", "pass"), then the ending (-a).
    Any passive softens the root's final labial, whether -w- follows it directly (banjwa) or after -is-, -el-, -an- (hanjiswa, vunyelwa, xhunyaniswa).
    A vowel-initial root takes -iw- when the passive follows it directly (enziwa), otherwise the passive is -w-."""
    no_vowel = not any(ch in VOWELS for ch in root)                                  # dl-a, ph-a: a root with no vowel
    stem = soften(root) if ("pass" in extensions and not no_vowel) else root
    for i, e in enumerate(extensions):
        if e == "pass":
            stem += "iw" if ((root[0] in VOWELS or no_vowel) and i == 0) else "w"    # enziwa, akhiwa, aziwa, dliwa, phiwa; hambiswa, vunyelwa, boshwa
        elif e == "neut":
            stem += "akal" if root in AKAL_ROOTS else "ek"                           # bonakala, tholakala; but kwenzeka, kuqhubeka
        elif e in FORMS:
            stem += FORMS[e]
        else:
            raise ValueError(e)
    if extensions and extensions[-1] == "recip" and ending == "ile":
        return stem[:-2] + "ene"                                # the reciprocal in the perfect: -ana -> -ene (ahlukene, bhekene)
    return stem + ending


@dataclass(frozen=True)
class Extension:
    key: str
    form: str
    meaning: str
    english: str
    example: str
    news: int


EXTENSIONS: tuple[Extension, ...] = (
    Extension("pass", "-w- (-iw- after a vowel-initial root)", "passive", "was / is done (by)", "kuthiwa, kubikwa, kwenziwa, kuboshwa", 156653),
    Extension("appl", "-el-", "applicative", "for, to, at, with, on behalf of, towards", "okhulumela, ukuthumela, ukusebenzela, ukuvakashela", 98787),
    Extension("neut", "-ek- (-akal- after some roots)", "neuter", "can be done / is able to be / happens / becomes", "kwenzeka, kubonakala, kuyadingeka, kuthandeka", 86137),
    Extension("caus", "-is-", "causative", "make or let X do / cause / use (X) to", "ukusebenzisa, ukuhlanganisa, ukulungisa, ukuhambisa", 59231),
    Extension("recip", "-an-", "reciprocal", "each other / together / with one another", "ukuxhumana, ukubhekana, ukusebenzisana, ukuvumelana", 47484),
    Extension("intens", "-elel-, -isis-", "intensive / completive", "thoroughly, fully, to the end", "ukuphumelela, ukuqondisisa, ukwelulekisisa", 6822),
)

# the order the extensions come in when they stack (corpus counts; news counts in brackets)
STACKS = (
    ("caus + pass", "ukusetshenziswa, kukhonjiswe, ukuxhunyaniswa", "348 (news 15,960)"),
    ("appl + pass", "kufakelwa, ezivakashelwa, kuzobanjelwa", "294 (17,776)"),
    ("caus + recip", "ukuhambisana, ukusebenzisana, ukubambisana", "160 (12,378)"),
    ("appl + recip", "ukuvumelana, ehambelana, ephathelene", "84 (3,363)"),
    ("caus + appl", "akusebenzisele, uzingenisela, njengokuthengisela", "30 (1,872)"),
    ("caus + appl + pass", "lulungiselwe, lwadluliselwa, lusetshenziselwe", "31"),
    ("appl + neut", "kulindeleke, kuvumelekile, okufinyeleleka", "29 (6,767)"),
)


def render() -> str:
    rows = ["extension | form | English it replaces | corpus examples | news words"]
    for e in EXTENSIONS:
        rows.append(f"{e.meaning} | {e.form} | {e.english} | {e.example} | {e.news}")
    stacks = "\n".join(f"  {s} -- {ex}  [{n}]" for s, ex, n in STACKS)
    return f"""ISIZULU VERB EXTENSIONS -- one suffix on the verb does the work of an English phrase, so they are the main way to say it in fewer words (and the
agent's best length lever: 'make them work' is one verb, sebenzisa; 'was done' is one verb, kwenziwa). Forms and counts are from the annotated corpus and real news.
{chr(10).join(rows)}
Stacking. Several can follow the root. The usual order is causative or applicative first, then reciprocal, then passive, as in these (corpus count, news count):
{stacks}
The neuter comes before the causative in qinisekisa (qiniseka + is) and after the applicative in kulindeleke (linda + el + ek).
Neuter: -ek- is the default (kwenzeka, kuqhubeka, kudingeka, kubukeka), but some roots take -akal-: bonakala (kubonakala 883 words, kuboneka 0), tholakala, zwakala, kholakala, solakala,
sizakala, qondakala. Passive sound changes. -w- joins a consonant-final stem (bonwa, thengwa, kuthiwa); a vowel-initial root takes -iw- (enziwa, akhiwa, aziwa) and so does a root with no
vowel in it, without any softening: dla -> dliwa (kudliwa 74, ukudliwa 91), pha -> phiwa (not shwa). In ANY passive the
last consonant of the root, if it is a labial, softens: p -> sh (bopa -> boshwa), m -> ny (khuluma -> khulunywa), mb -> nj (bamba -> banjwa), b -> tsh (gweba ->
gwetshwa), bh -> j (gubha -> gujwa). This holds with an extension in between: vuma -> vunyelwa (not vumelwa), thuma -> thunyelwa, hamba -> hanjiswa, khomba ->
khonjiswa, xhuma -> xhunyaniswa. In 9 verbs the news has 2,697 softened passives and 1 unsoftened (kuvumelwe once; kukhulumwa and bambwa never).
Perfect of the reciprocal: -ana becomes -ene (ahlukene, bhekene, ezehlukene); the neuter -akal- becomes -akele (bonakele).
Choose the extension by what the English says: "for / on behalf of" -> -el-; "make / cause" -> -is-; "by / was done" -> -w-; "each other / together" -> -an-; "can be /
happens" -> -ek-. The combination is one word, so it often saves a whole phrase."""
