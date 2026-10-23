"""Search for the respelling whose synthesized phones sit closest to the English pronunciation of the name.

Analysis-by-synthesis, with no person in the loop:

    propose a spelling -> synthesize it in its real sentence -> cut out exactly the name's audio (bookmarks) ->
    recognise its phones -> compare with the English reference -> keep the closest -> change one thing -> repeat

Azure's isiZulu voices accept no IPA, phoneme tag or lexicon, so respelling is the only control there is; this finds
respellings automatically. It is only as good as the score, so ``agreement`` measures how often the score prefers what
a native listener chose over what they rejected, and a search is trustworthy only where that agreement is high.

Version 1 compared the best-matching stretch of the WHOLE sentence with the reference. It agreed with a listener in
only 4 of 10 pairs: the winning stretch was often other words ("nje kokuba" beat "Jeykhob Zuma" as a match for
"Jacob Zuma"). This version scores only the name's own audio window, found with SSML bookmarks, and ignores the
phone-level modifiers (aspiration, palatalisation, length) that the recogniser and a listener weigh very differently.

Synthesis of identical text is deterministic, so every candidate has one fixed score and is cached.
"""
from __future__ import annotations

import hashlib
import json
import re
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

# (SSML inner text containing <bookmark mark="n0"/> ... <bookmark mark="n1"/>, pitch, output wav) -> {mark: offset in ms}
SynthesizeMarked = Callable[[str, str, Path], Mapping[str, float]]
Recognise = Callable[[Path], Sequence[str]]            # wav -> phone sequence
Similarity = Callable[[Sequence[str], Sequence[str]], float]   # (name phones, reference phones) -> 0..1

SEPARATORS = (" ", ", ")
LEAD_MS, TRAIL_MS = 40, 120          # a little audio around the markers, so a clipped onset or a swallowed ending shows
_MODIFIERS = re.compile("[ʰʲʷˠˤːˑ̪̟̠̞̝̥̤̰̩̯̃̈̚͜͡ʼ]")      # aspiration, palatalisation, length, tie bars and other diacritics

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


def normalise_phones(phones: Iterable[str]) -> list[str]:
    """Drop aspiration, palatalisation, length and other modifiers: the recogniser and a listener disagree about them."""
    out = []
    for phone in phones:
        base = _MODIFIERS.sub("", phone)
        if base:
            out.append(base)
    return out


@dataclass(frozen=True)
class Candidate:
    spelling: str
    sep: str = " "          # what follows the name: a space, or a comma (separates a final vowel from the next word)

    def label(self) -> str:
        return f"{self.spelling}{'+comma' if self.sep.strip() else ''}"


@dataclass(frozen=True)
class Context:
    """A real sentence the name occurs in. ``before`` carries its own trailing space (or glue such as "i-" / "u")."""
    before: str
    after: str
    pitch: str = "-6%"

    def _sep(self, candidate: Candidate) -> str:
        return "" if self.after[:1] in ",.;:!?" else candidate.sep

    def render(self, candidate: Candidate) -> str:
        return f"{self.before}{candidate.spelling}{self._sep(candidate)}{self.after}".strip()

    def render_marked(self, candidate: Candidate) -> str:
        """The same sentence with a bookmark just before and just after the name (before any glued prefix such as "u")."""
        if self.before == "" or self.before.endswith(" "):
            head, glue = self.before, ""
        else:
            head, _, glue = self.before.rpartition(" ")
            head = f"{head} " if head else ""
        return (f'{head}<bookmark mark="n0"/>{glue}{candidate.spelling}<bookmark mark="n1"/>'
                f"{self._sep(candidate)}{self.after}").strip()


def neighbours(spelling: str) -> list[str]:
    """Every distinct spelling one edit away (and within a sane length), most conservative edits first."""
    seen: dict[str, None] = {}
    for pattern, replacement in _EDITS:
        for match in re.finditer(pattern, spelling):
            out = spelling[:match.start()] + replacement + spelling[match.end():]
            if out != spelling and _ALLOWED.match(out) and abs(len(out) - len(spelling)) <= 4:
                seen.setdefault(out, None)
    return list(seen)


def slice_wav(source: Path, start_ms: float, end_ms: float, target: Path) -> Path:
    with wave.open(str(source), "rb") as reader:
        rate, width, channels = reader.getframerate(), reader.getsampwidth(), reader.getnchannels()
        first = max(0, int(rate * start_ms / 1000))
        last = min(reader.getnframes(), int(rate * end_ms / 1000))
        reader.setpos(first)
        frames = reader.readframes(max(0, last - first))
    with wave.open(str(target), "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(width)
        writer.setframerate(rate)
        writer.writeframes(frames)
    return target


@dataclass
class NameWindows:
    """Synthesizes a candidate in a context and returns the recognised phones of just the name's own audio."""
    synthesize: SynthesizeMarked
    recognise: Recognise
    work_dir: Path
    syntheses: int = 0
    _memo: dict[tuple[Candidate, Context], tuple[str, ...]] = field(default_factory=dict)

    def phones(self, context: Context, candidate: Candidate) -> tuple[str, ...]:
        memo_key = (candidate, context)
        if memo_key in self._memo:
            return self._memo[memo_key]
        key = hashlib.sha1(f"{context.pitch}|{context.render_marked(candidate)}".encode("utf-8")).hexdigest()[:16]
        wav, marks_file = self.work_dir / f"{key}.wav", self.work_dir / f"{key}.json"
        if wav.is_file() and marks_file.is_file():
            marks = json.loads(marks_file.read_text(encoding="utf-8"))
        else:
            self.work_dir.mkdir(parents=True, exist_ok=True)
            marks = dict(self.synthesize(context.render_marked(candidate), context.pitch, wav))
            marks_file.write_text(json.dumps(marks), encoding="utf-8")
            self.syntheses += 1
        piece = slice_wav(wav, marks["n0"] - LEAD_MS, marks["n1"] + TRAIL_MS, self.work_dir / f"{key}.name.wav")
        self._memo[memo_key] = tuple(self.recognise(piece))
        return self._memo[memo_key]


@dataclass
class Scorer:
    windows: NameWindows
    similarity: Similarity
    references: Sequence[Sequence[str]]
    contexts: Sequence[Context]
    evaluations: int = 0
    _cache: dict[Candidate, float] = field(default_factory=dict)

    def score(self, candidate: Candidate) -> float:
        """Mean over contexts and references of the phonetic similarity between the name's audio and the reference."""
        if candidate in self._cache:
            return self._cache[candidate]
        self.evaluations += 1
        scores = []
        for context in self.contexts:
            name = normalise_phones(self.windows.phones(context, candidate))
            scores.extend(self.similarity(name, normalise_phones(reference)) for reference in self.references)
        self._cache[candidate] = sum(scores) / len(scores)
        return self._cache[candidate]

    def name_phones(self, candidate: Candidate) -> str:
        """The first context's recognised phones for the name, for display."""
        return "".join(self.windows.phones(self.contexts[0], candidate))


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


# ---- command line: run the search for entities taken from a real job -------------------------------------------------

def _candidate(raw: Mapping[str, Any]) -> Candidate:
    return Candidate(str(raw["spelling"]), str(raw.get("sep", " ")))


def main(argv: list[str] | None = None) -> int:
    import argparse
    import os

    import requests

    parser = argparse.ArgumentParser(prog="mathula_tv.pronunciation_search")
    parser.add_argument("--config", required=True, help="JSON: {job, voice, entities:[{name, seeds, contexts, pairs, budget}]}")
    parser.add_argument("--out", default="pronunciation_search")
    parser.add_argument("--english-voice", default="en-ZA-LukeNeural")
    parser.add_argument("--validate-only", action="store_true", help="score the labelled pairs, skip the search")
    args = parser.parse_args(argv)

    import azure.cognitiveservices.speech as speechsdk

    from . import phoneme_recognizer
    from .config import load_settings
    from .native_dub import _extract_entity_english_source_clip

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    settings = load_settings()
    job_root = Path(settings.work_dir) / "jobs" / config["job"]
    voice = config.get("voice", "zu-ZA-ThembaNeural")
    key, region = os.environ["AZURE_SPEECH_KEY"], os.environ["AZURE_SPEECH_REGION"].strip().lower()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
    speech_config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm)

    def synthesize(marked: str, pitch: str, wav: Path) -> Mapping[str, float]:
        ssml = (f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="zu-ZA"><voice name="{voice}">'
                f'<prosody rate="+0%" pitch="{pitch}" volume="+0%">{marked}</prosody></voice></speak>')
        synth = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
        marks: dict[str, float] = {}
        synth.bookmark_reached.connect(lambda event: marks.__setitem__(event.text, event.audio_offset / 10_000))
        result = synth.speak_ssml_async(ssml).get()
        if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted or not {"n0", "n1"} <= set(marks):
            raise RuntimeError(f"synthesis failed or returned no bookmarks: {result.reason}")
        wav.write_bytes(result.audio_data)
        return marks

    def english_reference(name: str) -> Sequence[str]:
        """The name read by a South African English voice: a clean reference, unlike a clip cut from a news broadcast."""
        ssml = (f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-ZA">'
                f'<voice name="{args.english_voice}">{name}.</voice></speak>')
        response = requests.post(
            f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1", timeout=60, data=ssml.encode("utf-8"),
            headers={"Ocp-Apim-Subscription-Key": key, "Content-Type": "application/ssml+xml",
                     "X-Microsoft-OutputFormat": "riff-24khz-16bit-mono-pcm", "User-Agent": "mathula-pronunciation-search"})
        response.raise_for_status()
        path = out / "reference" / f"{re.sub(r'[^a-z0-9]+', '_', name.lower())}_en_tts.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
        return phoneme_recognizer.recognize(path).phones

    def recognise(wav: Path) -> Sequence[str]:
        return phoneme_recognizer.recognize(wav).phones

    def similarity(name: Sequence[str], reference: Sequence[str]) -> float:
        if not name or not reference:
            return 0.0
        return phoneme_recognizer.phonetic_similarity(
            phoneme_recognizer.PhoneSequence(tuple(name)), phoneme_recognizer.PhoneSequence(tuple(reference)))

    report: dict[str, Any] = {}
    for entity in config["entities"]:
        name = entity["name"]
        slug = re.sub(r"\W+", "_", name.lower())
        references: dict[str, Sequence[str]] = {"english_tts": english_reference(name)}
        clip = _extract_entity_english_source_clip(
            job_root=job_root, entity_id=slug, canonical_text=name, audio_dir=out / "reference")
        if clip is not None:
            references["source_audio"] = phoneme_recognizer.recognize(clip).phones
        contexts = [Context(**c) for c in entity["contexts"]]
        windows = NameWindows(synthesize, recognise, out / slug)
        pairs = [(_candidate(p["better"]), _candidate(p["worse"])) for p in entity.get("pairs", [])]
        variants = {"english_tts": ["english_tts"], "source_audio": ["source_audio"], "both": list(references)}
        report[name] = {"references": {k: "".join(v) for k, v in references.items()}, "variants": {}}
        print(f"\n== {name}")
        for k, v in references.items():
            print(f"   reference {k}: {''.join(v)}")
        best_variant, best_agreement = None, -1.0
        for variant, names in variants.items():
            if not all(n in references for n in names):
                continue
            scorer = Scorer(windows, similarity, [references[n] for n in names], contexts)
            checked = agreement(scorer, pairs)
            report[name]["variants"][variant] = checked
            print(f"   [{variant}] agreement with the listener: {checked['agreement']}")
            for row in checked["pairs"]:
                print(f"       {'OK ' if row['agrees'] else 'NO '} {row['chosen']!r} ({row['chosen_score']}) vs {row['rejected']!r} ({row['rejected_score']})")
            if checked["agreement"] is not None and checked["agreement"] > best_agreement:
                best_variant, best_agreement = variant, checked["agreement"]
        if args.validate_only or best_variant is None:
            continue
        scorer = Scorer(windows, similarity, [references[n] for n in variants[best_variant]], contexts)
        ranked = search(scorer, entity["seeds"], width=3, depth=4, budget=int(entity.get("budget", 60)))
        report[name]["search"] = {
            "reference": best_variant, "evaluations": scorer.evaluations,
            "top": [{"candidate": c.label(), "score": round(s, 3), "phones": scorer.name_phones(c)} for s, c in ranked[:10]]}
        print(f"   search with the [{best_variant}] reference, best first ({scorer.evaluations} evaluations):")
        for row in report[name]["search"]["top"][:6]:
            print(f"       {row['score']:.3f}  {row['candidate']!r}  {row['phones'][:50]}")
    (out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
