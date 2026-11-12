"""A grammar competence test for the model that writes the Zulu: can it produce the forms, without the noise of a whole dub?

Every item gives all the vocabulary and asks for ONE isiZulu form (a verb with its concords, a joined prefix, a relative verb, a passive stem, a number in words).  The gold answer
comes from the module builders, whose forms are counted in a linguist-annotated corpus and in 8.4 million words of news (see zulu_agreement, zulu_tam, zulu_relative, zulu_extensions,
zulu_object, zulu_numerals).  ``scripts/grammar_test_run.py`` asks the model in batches, with and without the grammar reference, and scores the answers; an item that every attempt
gets "wrong" the same way is a candidate gold error, not a model error, and is listed for checking.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .zulu_agreement import merge_vowels
from .zulu_extensions import extend
from .zulu_numerals import cardinal
from .zulu_object import with_object
from .zulu_relative import relative
from .zulu_tam import conjugate

CATEGORIES = ("merge", "tense", "relative", "extension", "object", "numeral")


@dataclass(frozen=True)
class Item:
    id: str
    category: str
    prompt: str
    accept: tuple[str, ...]          # normalised answers that are right; the first is the gold form


def normalise(answer: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[.,;:!?\"'’‘“”()\[\]-]", "", (answer or "").lower())).strip()


def correct(item: Item, answer: str) -> bool:
    return normalise(answer) in {normalise(a) for a in item.accept}


# ---- vocabulary shown to the model ---------------------------------------------------------------------------------------------------------------------------
VERBS = {"bon": ("ukubona", "see"), "thand": ("ukuthanda", "love"), "fund": ("ukufunda", "read"), "sebenz": ("ukusebenza", "work"), "bhek": ("ukubheka", "look at"),
         "buy": ("ukubuya", "come back"), "phind": ("ukuphinda", "repeat"), "nik": ("ukunika", "give"), "khulum": ("ukukhuluma", "speak"), "lung": ("ukulunga", "be right"),
         "hamb": ("ukuhamba", "go"), "shay": ("ukushaya", "hit"), "enz": ("ukwenza", "do"), "akh": ("ukwakha", "build"), "and": ("ukwanda", "increase"),
         "dl": ("ukudla", "eat")}
SUBJECT = {"1ps": "I", "2ps": "you (one person)", "1pp": "we", "2pp": "you (several people)",
           "1": "he or she (class 1, e.g. umuntu)", "2": "they, people (class 2, e.g. abantu)", "3": "it (class 3, e.g. umsebenzi)", "4": "they (class 4, e.g. imisebenzi)",
           "5": "it (class 5, e.g. ilizwe)", "6": "they (class 6, e.g. amazwe)", "7": "it (class 7, e.g. isicelo)", "8": "they (class 8, e.g. izicelo)",
           "9": "it (class 9, e.g. indlela)", "10": "they (class 10, e.g. izindlela)", "11": "it (class 11, e.g. ulwazi)", "15": "it (class 15, e.g. ukudla)"}
OBJECT = {"1ps": "me", "2ps": "you (one person)", "1pp": "us", "2pp": "you (several people)", "1": "him or her (class 1)", "2": "them, people (class 2)", "3": "it (class 3, e.g. umsebenzi)",
          "4": "them (class 4, e.g. imisebenzi)", "5": "it (class 5, e.g. ilizwe)", "6": "them (class 6, e.g. amazwe)", "7": "it (class 7, e.g. isicelo)",
          "8": "them (class 8, e.g. izicelo)", "9": "it (class 9, e.g. indlela)", "10": "them (class 10, e.g. izindlela)", "11": "it (class 11, e.g. ulwazi)", "15": "it (class 15, e.g. ukudla)",
          "refl": "himself / herself / themselves / myself (reflexive)"}
FORM_LABEL = {"future": "future ('will ...')", "perfect": "perfect ('has/have ...')", "still": "still ('is/are still ...ing')", "potential": "potential ('can/may ...')",
              "negative_present": "negative present ('does/do not ...')", "remote_past": "remote past, narrative ('... did, long ago')", "ngeke": "'will not / cannot', the common form with the word ngeke",
              "not_yet": "'has/have not yet ...'", "already": "already ('has/have already ...')", "present_long": "present with -ya- ('... is happening', verb closing the clause)"}
NOUN = {"2": ("abantu", "people"), "4": ("imisebenzi", "jobs"), "6": ("amazwe", "countries"), "10": ("izincwadi", "books"), "1": ("umuntu", "person"), "9": ("indlela", "road"),
        "7": ("isicelo", "request"), "15": ("ukudla", "food")}


def _desc(verb: str, subject: str, form: str, obj: str | None = None) -> str:
    inf, gloss = VERBS[verb]
    line = f"Verb: {inf} (to {gloss}). Subject: {SUBJECT[subject]}. "
    if obj:
        line += f"Object: {OBJECT[obj]}. "
    return line + f"Form: {FORM_LABEL[form]}. Give the single isiZulu word (two words for ngeke)."


def build_items() -> list[Item]:
    items: list[Item] = []

    def add(category: str, prompt: str, *accept: str) -> None:
        items.append(Item(f"{category[:3]}{sum(1 for i in items if i.category == category) + 1:02d}", category, prompt, tuple(accept)))

    # 1. prefixes that merge with the next word (possessive concords, na-, nga-)
    for prefix, gloss, noun, ng in [("wa", "of", "umuntu", "person"), ("ya", "of", "indlu", "house"), ("ba", "of", "amazwe", "countries"), ("za", "of", "izinto", "things"),
                                    ("kwa", "of", "ukuba", "being"), ("lwa", "of", "umuntu", "person"), ("sa", "of", "isikhathi", "time"), ("la", "of", "ilizwe", "country"),
                                    ("na", "with / and", "izinto", "things"), ("na", "with / and", "abantu", "people"), ("na", "with / and", "umfundi", "student"),
                                    ("nga", "by / in", "umthetho", "law"), ("nga", "by / in", "ingxoxo", "talk"), ("nga", "by / in", "amandla", "power"), ("kwa", "of", "izinye", "others"),
                                    ("wa", "of", "ukuqala", "starting"), ("ya", "of", "imali", "money"), ("za", "of", "amaphoyisa", "police"), ("na", "with / and", "ukudla", "food"),
                                    ("nga", "by / in", "okuhle", "good")]:
        add("merge", f"Join the prefix '{prefix}' ({gloss}) and the noun '{noun}' ({ng}) into ONE correctly written isiZulu word.", merge_vowels(prefix, noun))

    # 2. tense, aspect, mood, negation
    plan = [("future", s, v) for s, v in [("1ps", "bon"), ("2", "sebenz"), ("1pp", "khulum"), ("5", "buy"), ("10", "thand"), ("15", "lung"), ("1", "fund"), ("2pp", "hamb"), ("9", "nik")]]
    plan += [("perfect", s, v) for s, v in [("2ps", "khulum"), ("2", "hamb"), ("1pp", "bon"), ("1ps", "fund"), ("7", "lung"), ("10", "buy"), ("3", "phind")]]
    plan += [("still", s, v) for s, v in [("5", "sebenz"), ("15", "hamb"), ("6", "khulum"), ("1pp", "fund"), ("2", "buy"), ("1ps", "thand")]]
    plan += [("potential", s, v) for s, v in [("2", "bon"), ("1", "bhek"), ("9", "nik"), ("2ps", "hamb"), ("15", "lung"), ("1pp", "sebenz")]]
    plan += [("negative_present", s, v) for s, v in [("1ps", "hamb"), ("1", "sebenz"), ("2", "bon"), ("9", "thand"), ("1pp", "khulum"), ("6", "buy"), ("2ps", "fund")]]
    plan += [("remote_past", s, v) for s, v in [("1", "hamb"), ("2", "bon"), ("1pp", "buy"), ("9", "bhek"), ("10", "sebenz")]]
    plan += [("ngeke", s, v) for s, v in [("1ps", "hamb"), ("2", "sebenz"), ("1pp", "bon"), ("5", "buy")]]
    plan += [("already", s, v) for s, v in [("1ps", "hamb"), ("1pp", "bon"), ("15", "hamb"), ("2", "sebenz"), ("1", "fund")]]
    for form, subject, verb in plan:
        gold = conjugate(verb, subject, form)
        add("tense", _desc(verb, subject, form), gold)

    # 3. relative verbs
    rel_plan = [("2", "sebenz", "present"), ("1", "fun", "present"), ("10", "ding", "present"), ("9", "hlal", "present"), ("15", "bek", "present"), ("2", "landel", "present_yo"),
                ("7", "landel", "present_yo"), ("1", "lung", "perfect"), ("2", "hamb", "perfect"), ("1", "qal", "future"), ("2", "bhekan", "future"), ("2", "sebenz", "negative"),
                ("10", "sebenz", "negative"), ("2", "khul", "still"), ("6", "hamb", "present")]
    labels = {"present": "who/which does", "present_yo": "who/which does (nothing follows it in its clause)", "perfect": "who/which has done", "future": "who/which will do",
              "negative": "who/which does not do", "still": "who/which still does"}
    for cls, root, form in rel_plan:
        noun, ng = NOUN[cls]
        accept = {relative(root, cls, "present"), relative(root, cls, "present_yo")} if form in ("present", "present_yo") else {relative(root, cls, form)}
        gold = relative(root, cls, form)
        inf = "uku" + root + "a"
        add("relative", f"Noun: {noun} ({ng}). Verb stem: -{root}a (as in {inf}). Give the single isiZulu relative verb meaning '{labels[form]}' that describes {noun}.", gold, *sorted(accept - {gold}))
    for person, cls, root, gloss in [("1ps", "10", "fun", "want"), ("1pp", "2", "bon", "see")]:
        noun, ng = NOUN[cls]
        add("relative", f"Noun: {noun} ({ng}). Verb: -{root}a (to {gloss}). Give the single isiZulu relative verb for '{ng} that {SUBJECT[person]} {gloss}', with {SUBJECT[person]} as the subject of the clause "
                        f"(the word that follows {noun}).", relative(root, cls, "present", subject=person), relative(root, cls, "present_yo", subject=person),
            relative(root, cls, "present", subject=person, oc={"10": "zi", "2": "ba"}[cls]), relative(root, cls, "present_yo", subject=person, oc={"10": "zi", "2": "ba"}[cls]))

    # 4. verb extensions (the stem, ending in -a)
    ext_plan = [("bamb", ["pass"], "hold", "be held"), ("khulum", ["pass"], "speak", "be spoken"), ("bop", ["pass"], "tie", "be tied"), ("gweb", ["pass"], "cut", "be cut"),
                ("thum", ["appl", "pass"], "send", "be sent to someone"), ("vum", ["appl", "pass"], "agree", "be agreed to / permitted"), ("hamb", ["caus", "pass"], "go", "be made to go / transported"),
                ("khomb", ["caus", "pass"], "point", "be pointed out"), ("enz", ["pass"], "do", "be done"), ("akh", ["pass"], "build", "be built"), ("sebenz", ["caus"], "work", "make work / use"),
                ("hamb", ["appl"], "go", "go for / to"), ("hamb", ["recip"], "go", "go with one another"), ("hamb", ["caus", "recip"], "go", "go together / be linked"), ("fak", ["appl", "pass"], "put", "be put in for"),
                ("fund", ["caus"], "learn", "teach"), ("bon", ["neut"], "see", "be visible"), ("lung", ["caus", "pass"], "be right", "be made right / repaired"), ("xhum", ["recip", "caus", "pass"], "connect", "be connected together"),
                ("ph", ["pass"], "give", "be given")]
    for root, exts, gloss, meaning in ext_plan:
        kinds = " + ".join({"caus": "causative", "appl": "applicative", "recip": "reciprocal", "neut": "neuter", "pass": "passive"}[e] for e in exts)
        add("extension", f"Verb: -{root}a (to {gloss}). Give the verb STEM with these extensions in this order: {kinds} (meaning roughly '{meaning}'), ending in -a, with no subject or tense prefix.", extend(root, exts))

    # 5. object concords and the reflexive
    obj_plan = [("future", "1ps", "2ps", "bon"), ("perfect", "2", "1", "thand"), ("present_long", "1pp", "2", "bhek"), ("future", "2", "1ps", "nik"), ("perfect", "1", "1pp", "fund"),
                ("negative_present", "1pp", "2ps", "bon"), ("future", "1ps", "9", "enz"), ("perfect", "2", "7", "akh"), ("present_long", "1ps", "refl", "bon"), ("future", "1", "refl", "thand"),
                ("perfect", "1pp", "10", "enz"), ("negative_present", "2", "15", "enz"), ("present_long", "2ps", "2pp", "bon"), ("future", "1ps", "11", "akh"), ("perfect", "9", "5", "thand"),
                ("future", "2pp", "6", "enz"), ("present_long", "1", "3", "bhek"), ("perfect", "1ps", "1", "dl"), ("future", "2", "4", "and"), ("present_long", "1pp", "refl", "hamb")]
    for form, subject, obj, verb in obj_plan:
        add("object", _desc(verb, subject, form, obj), conjugate(with_object(verb, obj), subject, form))

    # 6. numerals in words
    num_plan = [(2, "2"), (3, "4"), (4, "10"), (5, "6"), (5, "2"), (6, "2"), (7, "10"), (9, "6"), (10, "2"), (10, "10"), (11, None), (14, "4"), (15, None), (20, "10"), (20, "6"), (25, "2"), (26, "2"),
                (30, "2"), (45, "10"), (60, "2"), (84, None), (100, None), (105, None), (200, None), (500, None), (1000, None), (2000, "10"), (4500, None), (1005, None), (3333, "10")]
    for n, cls in num_plan:
        if cls is None:
            add("numeral", f"Write the number {n} in isiZulu words, counting on its own (no noun it must agree with), as in 'the amount is ...'.", cardinal(n, None))
        else:
            noun, ng = NOUN[cls]
            add("numeral", f"Write the number {n} in isiZulu words so that it agrees with the noun '{noun}' ({ng}, class {cls}), as in '{noun} ...'. Give the number words only.", cardinal(n, cls))
    return items
