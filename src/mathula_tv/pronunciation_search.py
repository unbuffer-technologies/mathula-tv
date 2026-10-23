"""Search for the respelling whose synthesized phones sit closest to the English speaker's own pronunciation.

Analysis-by-synthesis, with no person in the loop:

    propose a spelling -> synthesize it in its real sentence -> recognise the phones -> compare with the phones of the
    English speaker saying the name in the source audio -> keep the closest -> change one thing -> repeat

Azure's isiZulu voices accept no IPA, phoneme tag or lexicon, so respelling is the only control there is; this finds
respellings automatically. It is only as good as the score, so ``agreement`` measures how often the score prefers
what a native listener chose over what they rejected. A search is trustworthy only where that agreement is high.

Synthesis of identical text is deterministic, so every candidate has one fixed score and is cached.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

Synthesize = Callable[[str, str, Path], None]          # (inner SSML text, pitch, output wav) -> None
Recognise = Callable[[Path], Sequence[str]]            # wav -> phone sequence
Similarity = Callable[[Sequence[str], Sequence[str]], float]   # (haystack phones, needle phones) -> 0..1

SEPARATORS = (" ", ", ")
_VOWELS = "aeiou"

# Single edits a spelling can undergo. Each is (pattern, replacement); every match position is tried once.
_EDITS: tuple[tuple[str, str], ...] = (
    (r"(?<=[ktpKTP])(?!h)", "h"),          # k -> kh, t -> th, p -> ph (aspiration)
    (r"(?<=[ktpKTP])h", ""),               # kh -> k
    (r"ay", "ey"), (r"ey", "ay"), (r"ey", "ei"), (r"ay", "ai"), (r"ei", "ey"),
    (r"a(?!a)", "aa"), (r"aa", "a"), (r"e(?!e)", "ee"), (r"ee", "e"), (r"o(?!o)", "oo"), (r"oo", "o"),
    (r"i(?!i)", "ii"), (r"ii", "i"), (r"u(?!u)", "uu"), (r"uu", "u"),
    (r"i\b", "ee"), (r"ee\b", "i"), (r"i\b", "y"), (r"y\b", "i"), (r"a\b", "ah"), (r"ah\b", "a"),
    (r"o", "oh"), (r"oh", "o"), (r"o", "ow"), (r"ow", "o"), (r"u", "oo"), (r"oo", "u"),
    (r"c", "k"), (r"k", "c"), (r"j", "dj"), (r"dj", "j"), (r"j", "dzh"), (r"s", "z"), (r"z", "s"),
    (r"b\b", "bh"), (r"bh\b", "b"), (r"b\b", "p"), (r"th", "t"), (r"t(?!h)", "th"),
    (r"y", "ey"), (r"ey", "y"),
)
_ALLOWED = re.compile(r"^[A-Za-z][A-Za-z -]{0,40}$")


@dataclass(frozen=True)
class Candidate:
    spelling: str
    sep: str = " "          # what follows the name: a space, or a comma (separates a final vowel from the next word)

    def label(self) -> str:
        return f"{self.spelling}{'+comma' if self.sep.strip() else ''}"


@dataclass(frozen=True)
class Context:
    """A real sentence the name occurs in. ``after`` follows the name; a leading comma there is already natural."""
    before: str
    after: str
    pitch: str = "-6%"

    def render(self, candidate: Candidate) -> str:
        """``before`` carries its own trailing space (or glue such as "i-" / "u"); punctuation in ``after`` needs no separator."""
        sep = "" if self.after[:1] in ",.;:!?" else candidate.sep
        return f"{self.before}{candidate.spelling}{sep}{self.after}".strip()


def neighbours(spelling: str) -> list[str]:
    """Every distinct spelling one edit away (and within a sane length), most conservative edits first."""
    seen: dict[str, None] = {}
    for pattern, replacement in _EDITS:
        for match in re.finditer(pattern, spelling):
            out = spelling[:match.start()] + replacement + spelling[match.end():]
            if out != spelling and _ALLOWED.match(out) and abs(len(out) - len(spelling)) <= 4:
                seen.setdefault(out, None)
    return list(seen)


@dataclass
class Scorer:
    synthesize: Synthesize
    recognise: Recognise
    similarity: Similarity
    reference: Sequence[str]
    contexts: Sequence[Context]
    work_dir: Path
    evaluations: int = 0
    _cache: dict[Candidate, float] = field(default_factory=dict)
    _phones: dict[Candidate, tuple[str, ...]] = field(default_factory=dict)

    def score(self, candidate: Candidate) -> float:
        """Mean over the contexts of how closely the best-matching stretch of the sentence's phones matches the reference."""
        if candidate in self._cache:
            return self._cache[candidate]
        self.evaluations += 1
        scores = []
        for index, context in enumerate(self.contexts):
            key = hashlib.sha1(f"{context.pitch}|{context.render(candidate)}".encode("utf-8")).hexdigest()[:16]
            wav = self.work_dir / f"{key}.wav"
            if not wav.is_file():
                self.synthesize(context.render(candidate), context.pitch, wav)
            phones = tuple(self.recognise(wav))
            if index == 0:
                self._phones[candidate] = phones
            scores.append(self.similarity(phones, self.reference))
        self._cache[candidate] = sum(scores) / len(scores)
        return self._cache[candidate]


def search(
    scorer: Scorer, seeds: Iterable[str], *, width: int = 3, depth: int = 4, budget: int = 80,
) -> list[tuple[float, Candidate]]:
    """Beam search over one-edit moves (and the comma toggle). Stops at ``budget`` evaluations. Best first."""
    frontier = [Candidate(seed, sep) for seed in seeds for sep in SEPARATORS]
    for candidate in frontier:
        scorer.score(candidate)
    for _ in range(depth):
        ranked = sorted(frontier, key=lambda c: -scorer.score(c))[:width]
        fresh: list[Candidate] = []
        for parent in ranked:
            moves = [Candidate(n, parent.sep) for n in neighbours(parent.spelling)]
            moves.append(Candidate(parent.spelling, ", " if parent.sep == " " else " "))
            for move in moves:
                if move in scorer._cache:
                    continue
                if scorer.evaluations >= budget:
                    break
                scorer.score(move)
                fresh.append(move)
        if not fresh:
            break
        frontier = ranked + fresh
    return sorted(((score, c) for c, score in scorer._cache.items()), key=lambda item: -item[0])


def agreement(scorer: Scorer, pairs: Sequence[tuple[Candidate, Candidate]]) -> dict[str, Any]:
    """How often the score prefers the listener's choice (first of each pair) over what they rejected (second)."""
    rows = []
    for better, worse in pairs:
        a, b = scorer.score(better), scorer.score(worse)
        rows.append({"chosen": better.label(), "rejected": worse.label(), "chosen_score": round(a, 3),
                     "rejected_score": round(b, 3), "agrees": a > b})
    return {"pairs": rows, "agreement": (sum(r["agrees"] for r in rows) / len(rows)) if rows else None}


# ---- command line: run the search for entities taken from a real job, against the English speaker's own audio ----------

def _candidate(raw: Mapping[str, Any]) -> Candidate:
    return Candidate(str(raw["spelling"]), str(raw.get("sep", " ")))


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json
    import os

    import requests

    parser = argparse.ArgumentParser(prog="mathula_tv.pronunciation_search")
    parser.add_argument("--config", required=True, help="JSON: {job, voice, entities:[{name, seeds, contexts, pairs, budget}]}")
    parser.add_argument("--out", default="pronunciation_search")
    args = parser.parse_args(argv)

    from . import phoneme_recognizer
    from .config import load_settings
    from .native_dub import _extract_entity_english_source_clip

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    settings = load_settings()
    job_root = Path(settings.work_dir) / "jobs" / config["job"]
    voice = config.get("voice", "zu-ZA-ThembaNeural")
    region = os.environ["AZURE_SPEECH_REGION"].strip().lower()
    key = os.environ["AZURE_SPEECH_KEY"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def synthesize(text: str, pitch: str, wav: Path) -> None:
        ssml = (f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="zu-ZA"><voice name="{voice}">'
                f'<prosody rate="+0%" pitch="{pitch}" volume="+0%">{text}</prosody></voice></speak>')
        response = requests.post(
            f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1", timeout=60, data=ssml.encode("utf-8"),
            headers={"Ocp-Apim-Subscription-Key": key, "Content-Type": "application/ssml+xml",
                     "X-Microsoft-OutputFormat": "riff-24khz-16bit-mono-pcm", "User-Agent": "mathula-pronunciation-search"})
        response.raise_for_status()
        wav.write_bytes(response.content)

    def recognise(wav: Path) -> Sequence[str]:
        return phoneme_recognizer.recognize(wav).phones

    def similarity(haystack: Sequence[str], needle: Sequence[str]) -> float:
        return phoneme_recognizer.best_window_similarity(
            phoneme_recognizer.PhoneSequence(tuple(haystack)), phoneme_recognizer.PhoneSequence(tuple(needle)))

    report: dict[str, Any] = {}
    for entity in config["entities"]:
        name = entity["name"]
        slug = re.sub(r"\W+", "_", name.lower())
        clip = _extract_entity_english_source_clip(
            job_root=job_root, entity_id=slug, canonical_text=name, audio_dir=out / "reference")
        if clip is None:
            print(f"{name}: no English reference clip found in the source audio; skipped")
            continue
        reference = tuple(phoneme_recognizer.recognize(clip).phones)
        scorer = Scorer(synthesize, recognise, similarity, reference,
                        [Context(**c) for c in entity["contexts"]], out / slug)
        scorer.work_dir.mkdir(parents=True, exist_ok=True)
        pairs = [(_candidate(p["better"]), _candidate(p["worse"])) for p in entity.get("pairs", [])]
        checked = agreement(scorer, pairs)
        ranked = search(scorer, entity["seeds"], width=3, depth=4, budget=int(entity.get("budget", 80)))
        report[name] = {
            "reference_phones": "".join(reference), "evaluations": scorer.evaluations, "agreement": checked,
            "top": [{"candidate": c.label(), "score": round(s, 3), "phones": "".join(scorer._phones.get(c, ()))}
                    for s, c in ranked[:10]],
        }
        print(f"\n== {name}  (English reference phones: {''.join(reference)}; {scorer.evaluations} evaluations)")
        print(f"   agreement with the listener's choices: {checked['agreement']}")
        for row in checked["pairs"]:
            verdict = "OK " if row["agrees"] else "NO "
            print(f"     {verdict} chose {row['chosen']!r} ({row['chosen_score']}) over {row['rejected']!r} ({row['rejected_score']})")
        print("   search, best first:")
        for row in report[name]["top"][:6]:
            print(f"     {row['score']:.3f}  {row['candidate']!r}  {row['phones'][:60]}")
    (out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
