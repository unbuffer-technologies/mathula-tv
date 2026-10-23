"""The names stage: decide, word by word, which parts of a name keep the voice's own reading and which get a respelling.

A name in a South African news clip is often half English and half isiZulu ("Jacob Zuma"). The isiZulu voice already
says the isiZulu half correctly and says the English half wrongly, so the stage

  1. protects every word with isiZulu shape (never touched),
  2. works out how the speaker in the source audio said each remaining word (or, if nobody says it, how a South African
     English voice reads it),
  3. converts those sounds to this voice's spelling with rules that every by-ear approval so far has obeyed
     (an English /k/, /p/, /t/ before a vowel is "kh", "ph", "th"; /eɪ/ is "ey"; /aɪ/ is "ay"; long vowels are doubled),
  4. keeps the respelling only if the word's own audio scores clearly better than the plain spelling, and
  5. never overrides a name a person has already approved.

Its output is a per-job lexicon, ``native_dub/pronunciation_names.json``, which the job's pronunciation dictionary reads
so that measurement and rendering use it identically. A person samples it afterwards (``audit_takes``); nothing in it
needs a person before use.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

NAMES_FILE = "pronunciation_names.json"
SCHEMA = "mathula-pronunciation-names-v1"

_STOPS = {"k": "kh", "p": "ph", "t": "th"}          # validated by ear: Jeykhob, Khey, Phaathi
_VOWELS = {
    "eɪ": "ey", "aɪ": "ay", "ɔɪ": "oy", "oʊ": "ow", "əʊ": "ow", "aʊ": "aw",            # ey, ay: validated by ear
    "ɑː": "aa", "uː": "uu", "iː": "ii", "ɜː": "ur", "ɔː": "o",                          # aa: validated by ear
    "ɑ": "a", "i": "i", "ɪ": "i", "ɛ": "e", "e": "e", "æ": "a", "a": "a", "ʌ": "a", "ə": "a",
    "o": "o", "ɔ": "o", "ɒ": "o", "u": "u", "ʊ": "u",
}
_CONSONANTS = {"dʒ": "j", "tʃ": "tsh", "ʃ": "sh", "ʒ": "zh", "θ": "th", "ð": "dh", "ŋ": "ng", "j": "y", "ɹ": "r", "ɾ": "r",
               "ɡ": "g", "ɫ": "l", "ʔ": ""}
_PLAIN = set("bdfhlmnrsvwz")
VALIDATED_BY_EAR = {"kh", "ph", "th", "ey", "ay", "aa", "j", "o", "i", "e", "a", "b", "m"}


def phones_to_spelling(tokens: Sequence[str], like: str = "") -> tuple[str, list[str]]:
    """Spell English phones the way this voice needs them. Returns (spelling, rules applied that no listener has yet confirmed).

    An English stop is aspirated before a vowel (onsets) and plain after /s/ and at the end of a word.
    """
    out: list[str] = []
    unconfirmed: list[str] = []
    for i, token in enumerate(tokens):
        previous = tokens[i - 1] if i else ""
        following = tokens[i + 1] if i + 1 < len(tokens) else ""
        if token in _STOPS:
            aspirate = following in _VOWELS and previous != "s"
            piece = _STOPS[token] if aspirate else token
        elif token in _VOWELS:
            piece = _VOWELS[token]
        elif token in _CONSONANTS:
            piece = _CONSONANTS[token]
        elif len(token) == 1 and token in _PLAIN:
            piece = token
        else:
            piece = token if token.isascii() else ""
        if piece and piece not in VALIDATED_BY_EAR and piece != token:
            unconfirmed.append(f"{token}->{piece}")
        out.append(piece)
    spelling = "".join(out)
    if like[:1].isupper():
        spelling = spelling[:1].upper() + spelling[1:]
    return spelling, unconfirmed


def plan_name(name: str, word_ipas: Sequence[str], scorer: Any, tokenise: Callable[[str], Sequence[str]], *,
              looks_isizulu: Callable[[str], bool], margin: float = 0.15,
              current: Sequence[str] | None = None) -> dict[str, Any]:
    """Decide each word of ``name``: protected, respelt by rule (and confirmed by its own audio), or left as it is.

    ``current`` is how the existing dictionary already speaks each word. The stage only acts on a word that is still
    read as written: an acronym, an isiZulu-shaped word or a word the pipeline has already respelt is left exactly as
    it is, and a respelling must beat the current reading, not merely the raw spelling.
    """
    words = name.split()
    spoken = list(current) if current is not None else list(words)
    decisions: list[dict[str, Any]] = []
    for i, word in enumerate(words):
        if looks_isizulu(word):
            decisions.append({"word": word, "decision": spoken[i], "changed": False, "why": "isiZulu shape: left to the voice"})
            continue
        if len(word) > 1 and word.isupper():
            decisions.append({"word": word, "decision": spoken[i], "changed": False, "why": "acronym: left to the initialism rules"})
            continue
        if spoken[i] != word:
            decisions.append({"word": word, "decision": spoken[i], "changed": False, "why": "already respelt by the pipeline: kept"})
            continue
        rule, unconfirmed = phones_to_spelling(tokenise(word_ipas[i]), like=word)
        plain = scorer.word_scores(" ".join(spoken))[i]
        trial = scorer.word_scores(" ".join([*spoken[:i], rule, *spoken[i + 1:]]))[i] if rule and rule != word else plain
        changed = bool(rule) and rule != word and trial >= plain + margin
        decisions.append({"word": word, "decision": rule if changed else spoken[i], "changed": changed, "rule": rule,
                          "plain_score": round(plain, 3), "rule_score": round(trial, 3), "unconfirmed_rules": unconfirmed,
                          "why": "respelt: beats the current reading" if changed else "kept: no clear gain over the current reading"})
    result = " ".join(d["decision"] for d in decisions)
    return {"name": name, "result": result, "changed": result != " ".join(spoken), "decisions": decisions}


def approved_names(dictionary: Any) -> set[str]:
    """Display texts (and aliases) that a person has already approved; the automatic stage must never override them."""
    approved: set[str] = set()
    for entry in getattr(dictionary, "entries", ()):
        approved.add(entry.display_text.casefold())
        approved.update(alias.casefold() for alias in entry.aliases)
    return approved


def load_names_entries(path: Path, approved: set[str] | None = None) -> list[Any]:
    """The lexicon as pronunciation entries, minus any name a person has approved."""
    from .pronunciation import PRONUNCIATION_KINDS, PronunciationEntry

    if not Path(path).is_file():
        return []
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = []
    for item in data.get("entries") or []:
        if approved and item["display_text"].casefold() in approved:
            continue
        kind = item.get("kind") if item.get("kind") in PRONUNCIATION_KINDS else "personal_name"
        entries.append(PronunciationEntry(
            display_text=item["display_text"], spoken_text=item["display_text"], tts_text=item["tts_text"], source="names_stage",
            reviewed_by="automatic", confidence=float(item.get("confidence", 0.8)), kind=kind,
            notes=("Automatic names stage: isiZulu words untouched, English words respelt by rule and confirmed by audio",)))
    return entries


def merge_into_dictionary(dictionary: Any, names_path: Path, job_id: str) -> Any:
    """Add the lexicon as job overrides, above the built-in entries, but never over an approved name or a researched one."""
    entries = load_names_entries(names_path, approved_names(dictionary))
    if not entries:
        return dictionary
    taken = {e.display_text.casefold() for e in entries}
    kept = [o for o in dictionary.job_overrides if o.display_text.casefold() not in taken]
    return dictionary.with_job_overrides(job_id, [*kept, *entries])


def write_lexicon(path: Path, plans: Sequence[Mapping[str, Any]], *, voice: str, kinds: Mapping[str, str]) -> Path:
    entries = [{"display_text": p["name"], "tts_text": p["result"], "kind": kinds.get(p["name"], "personal_name"),
                "confidence": 0.8, "decisions": p["decisions"]} for p in plans if p["changed"]]
    Path(path).write_text(json.dumps({"schema": SCHEMA, "voice": voice, "entries": entries,
                                      "examined": [{"name": p["name"], "changed": p["changed"]} for p in plans]},
                                     indent=1, ensure_ascii=False), encoding="utf-8")
    return Path(path)
