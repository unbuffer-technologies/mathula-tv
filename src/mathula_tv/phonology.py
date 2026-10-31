"""What isiZulu speakers do with English sounds, learned from people instead of assumed.

The Multipron corpus (Giwa, Davel and Barnard 2011; CC licence, SADiLaR) has first-language English and isiZulu speakers each saying the same ~650 South African
names, every utterance with a checked phoneme transcription.  Aligning a Zulu speaker's phones with an English speaker's for the same name gives, measured:

* ``map``: for every English phone, the phones Zulu speakers actually use for it and how often (the schwa is a 40%, a 21%, i 14%...; the GOAT vowel is o 59%),
  including "nothing" (the sound was left out);
* ``insertions``: where Zulu speakers add a vowel (between consonants, after a final consonant) and which one.

The first-language speakers' own phones are the starting point (audio first); this is what a Zulu speaker does to them.  ``Phonology.adapt`` gives the most likely
Zulu-speaker phones for a sequence of English phones, ``Phonology.options`` the alternatives for one phone, and ``evaluate`` checks the map on name lists it was
not learned from.  Phones are written in IPA, the way the recognisers here write them.
"""
from __future__ import annotations

import collections
import glob
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

SCHEMA = "mathula-phonology-v1"
NOTHING = ""                                # an English phone the Zulu speaker left out

# The corpus writes phones in SAMPA; the recognisers here write IPA.  ``@i`` is the South African English PRICE vowel, ``@u`` GOAT, ``au`` MOUTH.
SAMPA_TO_IPA = {
    "@": "ə", "{": "æ", "E": "ɛ", "O": "ɔ", "A:": "ɑː", "3:": "ɜː", "Q": "ɒ", "9": "œ", "2:": "øː", "y": "y", "@i": "aɪ", "@u": "əʊ", "au": "aʊ", "i@": "ɪə", "u@": "ʊə", "e@": "ɛə",
    "9y": "œy", "r\\": "ɹ", "T": "θ", "D": "ð", "S": "ʃ", "Z": "ʒ", "N": "ŋ", "J": "ɲ", "K": "ɬ", "K\\": "ɮ", "tl_>": "tɬʼ", "|\\": "ǀ", "!\\": "ǃ", "|\\|\\": "ǁ", "l\\": "l", "kx": "kx",
    "g": "ɡ",
}
VOWELS = set("aeiouyɛɔəæɑɒœøɜɪʊ")
UNUSABLE = ("*", "(", "<")                  # the corpus's markers for noise, repeats and empty utterances


def _ipa(phone: str) -> str:
    return SAMPA_TO_IPA.get(phone, phone)


# The recognisers write some English vowels with symbols the corpus does not use (it writes South African English with i, u, a and the schwa).
RECOGNISER_TO_CORPUS = {"ɪ": "i", "ʊ": "u", "ʌ": "a", "ɐ": "a", "ɚ": "ə", "ɝ": "ɜː", "ɑ": "ɑː", "ɜ": "ɜː", "iː": "i", "uː": "u",
                        "oʊ": "əʊ", "ɹ": "ɹ", "ɾ": "ɹ", "ɡ": "ɡ"}


def corpus_symbol(phone: str) -> str:
    phone = phone.replace("͡", "").replace("͜", "")
    return RECOGNISER_TO_CORPUS.get(phone, phone)


def is_vowel(phone: str) -> bool:
    return bool(phone) and phone[0] in VOWELS


def _kind(phone: str | None) -> str:
    return "#" if phone is None else ("V" if is_vowel(phone) else "C")


def load_corpus(root: Path) -> dict[str, dict[tuple[str, str], list[list[str]]]]:
    """``{language: {(list, name): [phones of each speaker's utterance]}}`` for the English and isiZulu speakers, phones in IPA."""
    root = Path(root)
    data: dict[str, dict[tuple[str, str], list[list[str]]]] = {"english": collections.defaultdict(list), "isizulu": collections.defaultdict(list)}
    for lst in ("list_A", "list_B", "list_C"):
        for language in data:
            for phn in glob.glob(str(root / lst / language / "*" / "transcription" / "*.phn")):
                try:
                    name = " ".join(Path(phn[:-4] + ".txt").read_text(encoding="utf-8", errors="replace").split()).lower()
                    phones = [_ipa(p) for p in Path(phn).read_text(encoding="utf-8", errors="replace").split() if not p.startswith(UNUSABLE)]
                except OSError:
                    continue
                if phones:
                    data[language][(lst, name)].append(phones)
    return data


def _align(english: Sequence[str], zulu: Sequence[str], cost) -> list[tuple[str | None, str | None]]:
    """The cheapest pairing of the two phone strings (None = no counterpart on that side)."""
    n, m = len(english), len(zulu)
    best = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        best[i][0] = best[i - 1][0] + cost(english[i - 1], None)
    for j in range(1, m + 1):
        best[0][j] = best[0][j - 1] + cost(None, zulu[j - 1])
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best[i][j] = min(best[i - 1][j - 1] + cost(english[i - 1], zulu[j - 1]), best[i - 1][j] + cost(english[i - 1], None), best[i][j - 1] + cost(None, zulu[j - 1]))
    i, j, out = n, m, []
    while i or j:
        if i and j and abs(best[i][j] - (best[i - 1][j - 1] + cost(english[i - 1], zulu[j - 1]))) < 1e-9:
            out.append((english[i - 1], zulu[j - 1])); i -= 1; j -= 1
        elif i and abs(best[i][j] - (best[i - 1][j] + cost(english[i - 1], None))) < 1e-9:
            out.append((english[i - 1], None)); i -= 1
        else:
            out.append((None, zulu[j - 1])); j -= 1
    return out[::-1]


def _class_cost(a: str | None, b: str | None) -> float:
    if a is None or b is None:
        return 1.0
    if a == b:
        return 0.0
    return 0.6 if is_vowel(a) == is_vowel(b) else 1.6


def _learned_cost(counts: dict[str, collections.Counter]):
    """Costs from what has been seen: a pairing is cheap when Zulu speakers often do it."""
    totals = {a: sum(c.values()) for a, c in counts.items()}

    def cost(a: str | None, b: str | None) -> float:
        if a is None:
            return 1.6
        seen = counts.get(a, {}).get(NOTHING if b is None else b, 0)
        return min(2.0, -math.log((seen + 0.2) / (totals.get(a, 0) + 2.0))) if totals.get(a) else _class_cost(a, b)

    return cost


def pairs_of(data: dict, lists: Iterable[str] | None = None) -> list[tuple[list[str], list[str]]]:
    """(English speaker's phones, a Zulu speaker's phones) for every name both said, optionally limited to some of the corpus's lists."""
    keep = set(lists) if lists else None
    out = []
    for key, english in data["english"].items():
        if keep and key[0] not in keep:
            continue
        for zulu in data["isizulu"].get(key, []):
            out.append((english[0], zulu))
    return out


def learn(pairs: Sequence[tuple[Sequence[str], Sequence[str]]], *, rounds: int = 2) -> dict[str, Any]:
    """The map and the insertion habits from aligned pairs; the alignment itself is refined by what the first round learned."""
    cost = _class_cost
    for _ in range(rounds):
        counts: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        for english, zulu in pairs:
            for a, b in _align(english, zulu, cost):
                if a is not None:
                    counts[a][NOTHING if b is None else b] += 1
        cost = _learned_cost(counts)
    inserted: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    chances: collections.Counter = collections.Counter()
    for english, zulu in pairs:
        alignment = _align(english, zulu, cost)
        before: str | None = None
        first = next((x for x, _ in alignment if x is not None), None)
        chances[f"#_{_kind(first)}"] += 1                                  # the chance to add something before the word's first sound
        for index, (a, b) in enumerate(alignment):
            following = next((x for x, _ in alignment[index + 1:] if x is not None), None)
            if a is None:
                inserted[f"{_kind(before)}_{_kind(following)}"][b] += 1
            else:
                if index + 1 < len(alignment):
                    chances[f"{_kind(a)}_{_kind(following)}"] += 1
                before = a
        chances[f"{_kind(before)}_#"] += 1
    return {"schema": SCHEMA, "pairs": len(pairs),
            "map": {a: [{"zulu": z, "n": n, "share": round(n / sum(c.values()), 3)} for z, n in c.most_common()] for a, c in sorted(counts.items())},
            "insertions": {ctx: {"per_chance": round(sum(c.values()) / max(1, chances[ctx]), 3), "chances": chances[ctx],
                                 "vowels": [[p, n] for p, n in c.most_common() if is_vowel(p)][:4], "glides_and_consonants": [[p, n] for p, n in c.most_common() if not is_vowel(p)][:4]}
                           for ctx, c in sorted(inserted.items()) if chances[ctx]}}


class Phonology:
    def __init__(self, knowledge: dict[str, Any]) -> None:
        self.knowledge = knowledge
        self._map = knowledge.get("map", {})

    @classmethod
    def load(cls, path: Path) -> "Phonology | None":
        try:
            knowledge = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return cls(knowledge) if knowledge.get("schema") == SCHEMA else None

    def options(self, english_phone: str, *, minimum_share: float = 0.03) -> list[dict[str, Any]]:
        """What Zulu speakers do with an English phone: [{"zulu": phone or "" for left out, "share": 0-1, "n": count}], most usual first; a phone never seen is kept as it is."""
        found = [o for o in self._map.get(english_phone, []) if o["share"] >= minimum_share]
        return found or [{"zulu": english_phone, "share": 1.0, "n": 0}]

    def human_cost(self, target: str, made: str, *, minimum_share: float = 0.08) -> float | None:
        """What it costs to reach the English phone ``target`` with a letter group whose sound is ``made``, when Zulu speakers do that: between 0.15 (what most do)
        and 0.3 (rarely done); None when they do not.  Above what the spelling prior charges for an exact match, so an exact match stays cheaper."""
        options = [o for o in self._map.get(corpus_symbol(target), []) if o["zulu"] and o["share"] >= minimum_share]
        wanted = corpus_symbol(made)
        for option in options:
            if option["zulu"] == wanted:
                return round(0.15 + 0.15 * (1.0 - option["share"] / options[0]["share"]), 3)
        return None

    def what_speakers_do(self, english_phones: Sequence[str], *, limit: int = 6, minimum_seen: int = 15) -> list[dict[str, Any]]:
        """For the sounds of a word that Zulu speakers change (the top choice is not the sound itself, or no choice is clearly the most usual), what they say instead."""
        out, seen = [], set()
        for phone in english_phones:
            symbol = corpus_symbol(phone)
            options = self._map.get(symbol, [])
            if symbol in seen or not options or sum(o["n"] for o in options) < minimum_seen:
                continue
            seen.add(symbol)
            if options[0]["zulu"] == symbol and options[0]["share"] >= 0.8:
                continue
            shown = [o for o in options if o["share"] >= 0.05][:4]
            out.append({"speaker_sound": phone, "zulu_speakers_say": ", ".join(f"{o['zulu'] or '(leave out)'} {round(100 * o['share'])}%" for o in shown),
                        "seen": sum(o["n"] for o in options)})
        return out[:limit]

    def adapt(self, english_phones: Sequence[str]) -> list[str]:
        """The most usual Zulu-speaker phones for the English phones (no vowels inserted)."""
        return [o for o in (self.options(p)[0]["zulu"] for p in english_phones) if o]


def per(predicted: Sequence[str], actual: Sequence[str]) -> float:
    """Phone error rate: edits needed to turn ``predicted`` into ``actual`` over the length of ``actual``."""
    n, m = len(predicted), len(actual)
    row = list(range(m + 1))
    for i in range(1, n + 1):
        previous, row[0] = row[0], i
        for j in range(1, m + 1):
            previous, row[j] = row[j], min(previous + (predicted[i - 1] != actual[j - 1]), row[j] + 1, row[j - 1] + 1)
    return row[m] / max(1, m)


def evaluate(data: dict) -> dict[str, Any]:
    """Learn from two of the corpus's lists, predict the third (every way round); report the phone error rate against what Zulu speakers said, for the learned map and
    for the baseline of keeping the English speaker's phones as they are.  The best of a name's Zulu speakers counts, since speakers differ."""
    lists = ["list_A", "list_B", "list_C"]
    learned, kept, total = [], [], 0
    for held in lists:
        model = Phonology(learn(pairs_of(data, [x for x in lists if x != held])))
        for key, english in data["english"].items():
            if key[0] != held:
                continue
            zulus = data["isizulu"].get(key, [])
            if not zulus:
                continue
            total += 1
            learned.append(min(per(model.adapt(english[0]), z) for z in zulus))
            kept.append(min(per(english[0], z) for z in zulus))
    mean = lambda values: round(sum(values) / max(1, len(values)), 4)
    return {"names": total, "phone_error_rate_keeping_english_phones": mean(kept), "phone_error_rate_learned_map": mean(learned)}


def build(corpus_root: Path, out_path: Path) -> dict[str, Any]:
    data = load_corpus(corpus_root)
    knowledge = learn(pairs_of(data))
    knowledge["held_out_check"] = evaluate(data)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(knowledge, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"pairs": knowledge["pairs"], **knowledge["held_out_check"]}
