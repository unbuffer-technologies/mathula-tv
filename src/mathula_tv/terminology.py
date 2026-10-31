"""Official English-to-isiZulu political terminology, from the National Language Service's two multilingual lists.

* the Multilingual Election Terminology List (Department of Arts and Culture, 2014): about 560 English terms;
* the Multilingual Parliamentary / Political Terminology List (Parliament and the provincial legislatures, built on a 2003 project): about 500 terms.

Both are printed as one entry per English term, followed by the equivalents in the ten other official languages.  The election list gives each language one block, the
parliamentary list one line per equivalent and synonym; this module keeps the isiZulu side: the preferred equivalent first, then the alternatives the lists offer.
The PDFs are not redistributed; ``build`` reads them (needs ``pypdf``) and writes ``political_terms.json``, which ``Terminology`` loads.  The lists' own note says to
contact the National Language Service for usage conditions.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

LANGUAGES = ("Afrikaans", "IsiZulu", "IsiXhosa", "Siswati", "IsiNdebele", "Setswana", "Sepedi", "Sesotho", "Tshivenḓa", "Xitsonga")
_LANGUAGE_LINE = re.compile(r"^(" + "|".join(re.escape(l) for l in LANGUAGES) + r")\s+(.*)$", re.IGNORECASE)
_PAGE_MARKER = re.compile(r"^<<<PAGE \d+>>>$")
_PAGE_NUMBER = re.compile(r"^(?:\d{1,3}|[ivxlc]{1,6})$", re.IGNORECASE)
_TAGS = re.compile(r"\s+(?:equivalent synonym|equivalent|abbreviation|acronym|explanation|context indication)\b.*$", re.IGNORECASE)
_PARTS_OF_SPEECH = re.compile(r"\s*<([a-z .,/]+)>")
_CROSS_REFERENCE = re.compile(r"\s*<(?:see also|see)[^>]*>", re.IGNORECASE)
_BARE_PART_OF_SPEECH = re.compile(r"\s+(n|v|adj|adv|prep|conj|interj|pron|num)\.(?=\s|$)(?:\s+part of speech)?", re.IGNORECASE)
_NOT_AN_ENTRY = re.compile(r"headword|special note|equivalent|�|^\d+\.|collaborator|linguist|terminologist", re.IGNORECASE)
_CONTEXT = re.compile(r"\s*\{([^}]*)\}")
_ARROW = re.compile(r"\s(?:g|ð|→)\s+", re.IGNORECASE)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _split_parentheses(text: str) -> tuple[str, list[str]]:
    """``main (alternative; another)`` -> ("main", ["alternative", "another"]); nested or unbalanced brackets are left in the main part."""
    main, alternatives, depth, buffer = [], [], 0, []
    for ch in text:
        if ch == "(":
            depth += 1
            if depth == 1:
                continue
        elif ch == ")" and depth > 0:
            depth -= 1
            if depth == 0:
                alternatives.append("".join(buffer)); buffer = []
                continue
        (buffer if depth else main).append(ch)
    if depth:                                   # an unclosed bracket: keep the text as it was
        return _clean(text), []
    flat = [_clean(p) for a in alternatives for p in re.split(r"[;,]", a)]
    return _clean("".join(main)), [p for p in flat if p]


def parse_text(text: str, source: str) -> list[dict[str, Any]]:
    """Entries with an isiZulu side from the extracted text of one of the lists (pages marked ``<<<PAGE n>>>``)."""
    entries: list[dict[str, Any]] = []
    head: list[str] = []
    synonyms: list[str] = []
    zulu: list[list[str]] = []
    language: str | None = None
    page = 0

    def finish() -> None:
        nonlocal head, synonyms, zulu, language
        if head and zulu:
            entry = _entry(head, synonyms, zulu, source, page)
            if entry:
                entries.append(entry)
        head, synonyms, zulu, language = [], [], [], None

    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        if _PAGE_MARKER.match(line.strip()):
            page = int(re.findall(r"\d+", line)[0])
            continue
        if _PAGE_NUMBER.match(line.strip()):
            continue
        match = _LANGUAGE_LINE.match(line.strip())
        if match:
            language = match.group(1).title()
            if language == "Isizulu":
                zulu.append([match.group(2)])
            continue
        if raw.startswith(" ") and language is not None:
            if language == "Isizulu" and zulu:
                zulu[-1].append(line.strip())
            continue
        if raw.startswith(" ") and head and language is None:
            head.append(line.strip())
            continue
        if line.strip().lower().startswith("english synonym "):
            synonyms.append(_TAGS.sub("", line.strip()[len("english synonym "):]).strip())
            continue
        if head and not zulu and _unclosed(" ".join(head)):       # a note in {} or <> that wrapped onto the next line
            head.append(line.strip())
            continue
        finish()
        head = [line.strip()]
    finish()
    return entries


def _unclosed(text: str) -> bool:
    return text.count("{") > text.count("}") or text.count("<") > text.count(">")


def _entry(head: Sequence[str], synonyms: Sequence[str], zulu: Sequence[Sequence[str]], source: str, page: int) -> dict[str, Any] | None:
    heading = _clean(" ".join(head))
    heading = _CROSS_REFERENCE.sub("", heading)
    if heading.count("<") != heading.count(">"):
        return None
    pos = [m.group(1) for m in _PARTS_OF_SPEECH.finditer(heading)] + [m.group(1).lower() for m in _BARE_PART_OF_SPEECH.finditer(heading)]
    context = [m.group(1).strip() for m in _CONTEXT.finditer(heading)] + re.findall(r"<([A-Z]{2,6})>", heading)
    heading = re.sub(r"\s*<[A-Z]{2,6}>", "", heading)
    heading = _BARE_PART_OF_SPEECH.sub("", _CONTEXT.sub("", _PARTS_OF_SPEECH.sub("", heading)))
    term, english_synonyms = _split_parentheses(_TAGS.sub("", heading))
    english_synonyms += [x for x in synonyms if x and x not in english_synonyms]
    if not term or len(term) > 90 or _NOT_AN_ENTRY.search(term) or any(_NOT_AN_ENTRY.search(" ".join(f)) for f in zulu):
        return None                      # the lists' front matter (collaborators, notes on how entries are written) parses like entries; it is not one
    equivalents: list[str] = []
    alternatives: list[str] = []
    for fragments in zulu:
        main, extra = _split_parentheses(_TAGS.sub("", _CONTEXT.sub("", _clean(" ".join(fragments)))))
        if main and main not in equivalents:
            equivalents.append(main)
        alternatives += [e for e in extra if e not in alternatives]
    return {"english": term, "english_synonyms": english_synonyms, "pos": pos, "context": context, "zulu": equivalents,
            "zulu_alternatives": [a for a in alternatives if a not in equivalents], "source": source, "page": page}


def build(election_pdf: Path | None, parliament_pdf: Path | None, out_dir: Path) -> dict[str, Any]:
    """Read the PDFs (``pypdf``), parse them and write ``political_terms.json`` into ``out_dir``."""
    import logging

    from pypdf import PdfReader

    logging.disable(logging.CRITICAL)
    entries: list[dict[str, Any]] = []
    for path, source in ((election_pdf, "election"), (parliament_pdf, "parliament")):
        if not path:
            continue
        reader = PdfReader(str(path))
        text = "".join(f"\n<<<PAGE {i + 1}>>>\n{page.extract_text() or ''}" for i, page in enumerate(reader.pages))
        entries += parse_text(text, source)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "political_terms.json").write_text(json.dumps({"schema": "mathula-political-terms-v1", "entries": entries}, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"entries": len(entries), "election": sum(e["source"] == "election" for e in entries), "parliament": sum(e["source"] == "parliament" for e in entries)}


def _key(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", str(text).casefold()).strip()


class Terminology:
    def __init__(self, entries: Iterable[dict[str, Any]]) -> None:
        self.entries = list(entries)
        self._index: dict[str, list[dict[str, Any]]] = {}
        for entry in self.entries:
            for name in [entry["english"], *entry.get("english_synonyms", [])]:
                if _key(name):
                    self._index.setdefault(_key(name), []).append(entry)

    @classmethod
    def load(cls, directory: Path) -> "Terminology | None":
        path = Path(directory) / "political_terms.json"
        return cls(json.loads(path.read_text(encoding="utf-8"))["entries"]) if path.is_file() else None

    def lookup(self, term: str) -> list[dict[str, Any]]:
        """Entries whose English term or synonym is ``term`` (case, punctuation and a plural -s do not matter)."""
        key = _key(term)
        return self._index.get(key) or self._index.get(key.rstrip("s"), [])

    def find(self, english_text: str, *, longest_first: bool = True) -> list[dict[str, Any]]:
        """The terms of the lists that occur in an English text (whole words, longest term first, each place used once)."""
        text = " " + _key(english_text) + " "
        found, taken = [], []
        for name in sorted(self._index, key=len, reverse=True):
            if len(name) < 4:
                continue
            for hit in re.finditer(r"(?<=\s)" + re.escape(name) + r"s?(?=\s)", text):
                if not any(a < hit.end() and hit.start() < b for a, b in taken):
                    taken.append((hit.start(), hit.end()))
                    found.append({"term": name, "entries": self._index[name]})
        return found
