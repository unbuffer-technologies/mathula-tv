"""What spoken isiZulu does with a word: evidence from human-transcribed speech, for the dub agent's ``attested_zulu`` lookup.

Sources: the soap-opera dialogue (Rhythm City and Generations, English-isiZulu utterances that switch language, 5 hours, ``scripts/build_spoken_sources.py``) and
African Next Voices, isiZulu (dsfsi-anv/za-african-next-voices, gated; 56,479 recordings, 478 hours, about 2.0 million words, speakers talking for about a minute on health,
agriculture, transport, finance, telecommunication, sport, culture and general topics).  The transcripts are written by people, with the speaker's code-switching kept ("i-credit score",
"nama-microphone", "one thousand two hundred").  ``scripts/fetch_anv_text.py`` reads only the transcript columns (25 MB, no audio) into ``resources/spoken/anv_zul.jsonl``.

Why it exists: the translated and news text the lookup used before is formal.  Government translators wrote ``umbhobho`` for a microphone in 4 of 14 pairs; speakers say ``i-microphone`` /
``mayikhrofoni`` and use ``umbhobho`` for a public-address loudspeaker.  What people say aloud is the register of the product (a social-media dub heard once), so it is shown to the
agent beside the written evidence.  Evidence to weigh, never an answer picked for the agent.

Three things are reported for an English word: the word itself kept in the Zulu (bare or behind a hyphenated particle: i-microphone, nama-microphone); Zulu respellings of it, found by
matching consonant skeletons after dropping vowels and the Zulu digraph h (microphone ~ mayikhrofoni ~ makrofoni); and, for each Zulu form the agent is thinking of, how often speakers
use it, with example sentences so the meaning shows (umbhobho: a loudspeaker, a pipe).
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

DEFAULT_PATH = Path.home() / "mathula-tv" / "working" / "resources" / "spoken" / "anv_zul.jsonl"
MIN_SKELETON = 5            # measured on real transcripts: at 4 consonants coalition matched ezikweletini (in the debts), election matched esidlakathini, grant matched granite
EXAMPLES = 3
EXAMPLE_CHARS = 90          # characters of context on each side of the form
_TOKEN = re.compile(r"[a-z][a-z'\-]*")
_PARTICLES = frozenset("i ama ne nge ku kwa e ye we le se ka sa wa ya la nga na no njenge njengama nama nesi".split())


def skeleton_zulu(token: str) -> str:
    """Consonants of a written Zulu form: aspirates and digraphs folded (kh th ph bh -> k t p b), vowels and the glides y w h dropped, repeats collapsed."""
    s = re.sub(r"[^a-z]", "", token.lower())
    for a, b in (("kh", "k"), ("th", "t"), ("ph", "p"), ("bh", "b"), ("sh", "s"), ("ch", "c")):
        s = s.replace(a, b)
    s = re.sub(r"[aeiouywh]", "", s)
    return re.sub(r"(.)\1+", r"\1", s)


def skeleton_english(word: str) -> str:
    """The same skeleton for an English spelling: ph is f, hard c and q are k, x is ks."""
    s = re.sub(r"[^a-z]", "", word.lower()).replace("ph", "f").replace("ck", "k")
    s = re.sub(r"c(?=[eiy])", "s", s).replace("c", "k").replace("q", "k").replace("x", "ks")
    s = re.sub(r"[aeiouywh]", "", s)
    return re.sub(r"(.)\1+", r"\1", s)


def _last_part(token: str) -> str:
    return token.rsplit("-", 1)[-1]


class SpokenZulu:
    def __init__(self, rows: Sequence[dict[str, Any]]) -> None:
        self.rows = [(str(r.get("domain") or ""), str(r.get("transcript") or "")) for r in rows if r.get("transcript")]
        self.hours = sum(float(r.get("duration") or 0) for r in rows) / 3600
        self.counts: Counter[str] = Counter()
        for _domain, text in self.rows:
            self.counts.update(_TOKEN.findall(text.lower()))
        self.words = sum(self.counts.values())
        self._skeletons: dict[str, str] = {}

    @classmethod
    def from_file(cls, path: Path = DEFAULT_PATH) -> "SpokenZulu | None":
        path = Path(path)
        if not path.is_file():
            return None
        return cls([json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()])

    @classmethod
    def from_directory(cls, directory: Path = DEFAULT_PATH.parent) -> "SpokenZulu | None":
        """Every spoken source in the folder (``*.jsonl`` with domain / transcript / duration in seconds): African Next Voices monologues and the soap-opera dialogue."""
        directory = Path(directory)
        rows: list[dict[str, Any]] = []
        for path in sorted(directory.glob("*.jsonl")) if directory.is_dir() else []:
            rows += [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return cls(rows) if rows else None

    def _skeleton_of(self, token: str) -> str:
        if token not in self._skeletons:
            self._skeletons[token] = skeleton_zulu(_last_part(token))
        return self._skeletons[token]

    def _examples(self, forms: set[str], limit: int = EXAMPLES) -> list[dict[str, str]]:
        """Sentences with one of the forms, spread over domains (one per domain first) so the meanings do not all come from one topic."""
        pattern = re.compile(r"(?<![A-Za-z'\-])(" + "|".join(re.escape(f) for f in sorted(forms, key=len, reverse=True)) + r")(?![A-Za-z'])", re.IGNORECASE)
        out: list[dict[str, str]] = []
        seen_domains: set[str] = set()
        spare: list[dict[str, str]] = []
        for domain, text in self.rows:
            match = pattern.search(text)
            if not match:
                continue
            item = {"domain": domain, "text": ("..." if match.start() > EXAMPLE_CHARS else "") + text[max(0, match.start() - EXAMPLE_CHARS):match.end() + EXAMPLE_CHARS] + "..."}
            (out if domain not in seen_domains else spare).append(item)
            seen_domains.add(domain)
            if len(out) >= limit:
                break
        return (out + spare)[:limit]

    def lookup(self, english_word: str, zulu_forms: Sequence[str] = ()) -> dict[str, Any]:
        word = re.sub(r"[^a-z]", "", english_word.lower())
        answer: dict[str, Any] = {"source": f"spoken isiZulu, human-transcribed: {self.hours:.0f} hours, {self.words:,} words, speakers on everyday topics"}
        if not word:
            return answer
        english_forms = {word, word + "s", word + "es"}
        kept = Counter({t: c for t, c in self.counts.items() if _last_part(t) in english_forms})
        answer["english_word_kept"] = {"total": sum(kept.values()), "written_as": [{"form": f, "count": c} for f, c in kept.most_common(6)]}
        target = skeleton_english(word)
        adapted: Counter[str] = Counter()
        if len(target) >= MIN_SKELETON:
            for token, count in self.counts.items():
                if token in kept or _last_part(token) in english_forms or len(token) < len(word) - 3:
                    continue
                if self._skeleton_of(token).endswith(target):
                    adapted[token] = count
        answer["respelled_in_zulu"] = {"total": sum(adapted.values()), "written_as": [{"form": f, "count": c} for f, c in adapted.most_common(6)]}
        used = set(kept) | set(adapted)
        if used:
            answer["examples"] = self._examples({_last_part(f) if "-" in f else f for f in used} | used)
        checked = []
        for form in zulu_forms:
            key = re.sub(r"[^a-z'\-]", "", str(form).lower())
            counted = sum(c for t, c in self.counts.items() if t == key or _last_part(t) == key)
            entry: dict[str, Any] = {"form": form, "spoken_count": counted}
            if counted:
                entry["examples"] = self._examples({key})
            checked.append(entry)
        if checked:
            answer["checked_forms"] = checked
        parts = []
        if kept:
            parts.append("speakers keep the English word: " + ", ".join(f"{f['form']} ({f['count']})" for f in answer["english_word_kept"]["written_as"][:4]))
        if adapted:
            parts.append("speakers respell it in Zulu: " + ", ".join(f"{f['form']} ({f['count']})" for f in answer["respelled_in_zulu"]["written_as"][:4]))
        if not parts:
            parts.append("speakers neither keep nor respell this word in the transcripts (it may simply not come up in them)")
        answer["reading"] = "; ".join(parts) + ("; read the examples to see what each form means in speech" if (used or checked) else "")
        return answer
