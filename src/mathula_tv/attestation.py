"""Attestation: what real written isiZulu does with a word.

A model produces fluent Zulu, and it also produces forms nobody writes (the glossary made "Fighters" into "maFighter"); nothing inside the model says "I
have never seen this".  This module is the check the model cannot give itself: a word is attested when real text has it.  The ladder it serves is the
user's rule for a word the system does not know what to do with: use an attested Zulu word or loan if there is one, otherwise code-switch (the English
word as the speaker says it).  "Knowing" is attestation, not a model's confidence.

Resources (all downloaded from the SADiLaR repository, repo.sadilar.org, licences CC BY 2.5 ZA and CC BY 4.0; credit CTexT, North-West University, and the
Department of Arts and Culture):

* the Autshumato English-isiZulu parallel corpus: 233,691 aligned sentence pairs, 2.87 million Zulu tokens;
* the NCHLT isiZulu lexicon files (nouns, verbs, lemma list): 43,178 entries.

They are plain text in ``<work_dir>/resources/attestation/``: ``autshumato.zu.txt`` and ``autshumato.en.txt`` (one sentence per line, the two files
aligned line by line) and ``lexicon/*.txt`` (one entry per line, or token<TAB>lemma).  Nothing here calls a model.
"""
from __future__ import annotations

import bisect
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

_TOKEN = re.compile(r"[A-Za-z][A-Za-z']*")
MIN_PAIRS = 3                    # a Zulu form must share at least this many sentences with the English word to count as its equivalent


def resources_dir(work_dir: Path) -> Path:
    return Path(work_dir) / "resources" / "attestation"


def tokens(line: str) -> list[str]:
    """Lowercase word tokens; a hyphen splits ("i-Tshela" -> i, tshela), so a particle before an English word is counted apart from it."""
    return [t.lower() for t in _TOKEN.findall(line)]


class Attestation:
    def __init__(self, zulu_lines: Sequence[str], english_lines: Sequence[str], lexicon: Iterable[str] = ()) -> None:
        if len(zulu_lines) != len(english_lines):
            raise ValueError(f"the two sides must be aligned line by line ({len(zulu_lines)} against {len(english_lines)})")
        self.zulu_lines = list(zulu_lines)
        self.frequency: Counter[str] = Counter()
        self.document_frequency: Counter[str] = Counter()
        for line in self.zulu_lines:
            found = tokens(line)
            self.frequency.update(found)
            self.document_frequency.update(set(found))
        lowered = [line.lower() for line in english_lines]
        self._english = "\n".join(lowered)
        self._starts: list[int] = []
        position = 0
        for line in lowered:
            self._starts.append(position)
            position += len(line) + 1
        self.lexicon = {entry.lower() for entry in lexicon if entry}

    @classmethod
    def from_directory(cls, directory: Path) -> "Attestation":
        directory = Path(directory)
        zulu = (directory / "autshumato.zu.txt").read_text(encoding="utf-8").split("\n")
        english = (directory / "autshumato.en.txt").read_text(encoding="utf-8").split("\n")
        size = min(len(zulu), len(english))
        entries: list[str] = []
        for path in sorted((directory / "lexicon").glob("*.txt")) if (directory / "lexicon").is_dir() else []:
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                entries.extend(part.strip() for part in line.split("\t") if part.strip())
        return cls(zulu[:size], english[:size], entries)

    # -- a Zulu form -----------------------------------------------------------------
    def status(self, token: str) -> dict[str, Any]:
        """How often real text writes this exact form, and whether the lexicon knows it."""
        key = token.lower()
        return {"token": token, "corpus_count": self.frequency.get(key, 0), "in_lexicon": key in self.lexicon}

    def attested(self, token: str) -> bool:
        key = token.lower()
        return self.frequency.get(key, 0) > 0 or key in self.lexicon

    # -- an English word ---------------------------------------------------------------
    def _lines_with(self, word: str) -> list[int]:
        pattern = re.compile(r"(?<![a-z'])" + re.escape(word.lower()) + r"(?:s|es)?(?![a-z'])")
        found: list[int] = []
        last = -1
        for match in pattern.finditer(self._english):
            index = bisect.bisect_right(self._starts, match.start()) - 1
            if index != last:
                found.append(index)
                last = index
        return found

    def equivalents(self, english_word: str, *, top: int = 8) -> dict[str, Any]:
        """What the Zulu side of the corpus writes where the English side has this word: the Zulu forms that share the most sentences with it (a Dice
        score, so common function words do not win), and how often the English word itself stays in the Zulu sentence (code-switching)."""
        word = english_word.lower().strip()
        lines = self._lines_with(word)
        result: dict[str, Any] = {"english": english_word, "sentences": len(lines), "forms": [], "kept_in_english": 0}
        if not lines:
            return result
        together: Counter[str] = Counter()
        kept = 0
        word_forms = {word, word + "s", word + "es"}
        for index in lines:
            found = set(tokens(self.zulu_lines[index]))
            if found & word_forms:
                kept += 1
            together.update(found)
        scored = []
        for form, pairs in together.items():
            if pairs < MIN_PAIRS or form in word_forms:
                continue
            scored.append((2 * pairs / (len(lines) + self.document_frequency[form]), form, pairs))
        scored.sort(reverse=True)
        result["kept_in_english"] = kept
        result["forms"] = [{"form": form, "shared_sentences": pairs, "score": round(score, 3), "corpus_count": self.frequency[form]}
                           for score, form, pairs in scored[:top]]
        return result


def lookup(attestation: Attestation, english_word: str, zulu_forms: Sequence[str] = ()) -> dict[str, Any]:
    """The answer for one English word, shaped for the agent: the Zulu forms real text pairs with it, whether it stays English there, and the status
    of any form the agent is thinking of writing."""
    answer = attestation.equivalents(english_word)
    answer["checked_forms"] = [attestation.status(form) for form in zulu_forms]
    sentences, kept = answer["sentences"], answer["kept_in_english"]
    strong = [f for f in answer["forms"] if f["score"] >= 0.1][:3]
    if not sentences:
        answer["reading"] = "the English word does not occur in the corpus: nothing is attested; keep it as the speaker says it"
    elif strong:
        answer["reading"] = ("real text pairs it with " + ", ".join(f"{f['form']!r} ({f['shared_sentences']} of {sentences} sentences)" for f in strong)
                             + (f"; the English word itself stays in {kept} of them" if kept else ""))
    elif kept and kept * 5 >= sentences:
        answer["reading"] = f"no Zulu form clearly pairs with it, and it stays English in {kept} of {sentences} sentences: real text code-switches it"
    else:
        answer["reading"] = (f"no Zulu form clearly pairs with it and it is rarely kept in English ({kept} of {sentences} sentences): nothing is attested; "
                             "keep it as the speaker says it")
    return answer


_cache: dict[str, Attestation] = {}


def load(work_dir: Path) -> Attestation | None:
    """The resources for this work directory, loaded once per process; None when they have not been put there."""
    directory = resources_dir(work_dir)
    key = str(directory)
    if key not in _cache:
        if not (directory / "autshumato.zu.txt").is_file():
            return None
        _cache[key] = Attestation.from_directory(directory)
    return _cache[key]


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    from .config import load_settings

    parser = argparse.ArgumentParser(prog="mathula_tv.attestation")
    sub = parser.add_subparsers(dest="command", required=True)
    form = sub.add_parser("form", help="how often real text writes an isiZulu form")
    form.add_argument("token", nargs="+")
    english = sub.add_parser("english", help="what real text writes where the English side has this word")
    english.add_argument("word")
    english.add_argument("--zulu", nargs="*", default=[], help="forms to check as well")
    args = parser.parse_args(argv)
    resources = load(load_settings().work_dir)
    if resources is None:
        print("no resources in", resources_dir(load_settings().work_dir))
        return 1
    if args.command == "form":
        for token in args.token:
            print(json.dumps(resources.status(token), ensure_ascii=False))
    else:
        print(json.dumps(lookup(resources, args.word, args.zulu), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
