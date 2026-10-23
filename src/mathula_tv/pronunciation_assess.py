"""Pronunciation assessment: does this audio pronounce the word the way the reference says it?

Two assessors, both scoring the AUDIO of a name against an expected English pronunciation (higher is better):

* ``azure_assessor`` -- Azure Speech pronunciation assessment (a service built for learner speech). It supports en-GB,
  en-US, en-AU, en-CA and en-IN but not en-ZA or zu-ZA, so a British or American model judges South African English.
* ``LocalGop`` -- goodness of pronunciation computed locally with a wav2vec 2.0 phoneme recogniser
  (``facebook/wav2vec2-xlsr-53-espeak-cv-ft``): how much less likely the EXPECTED phone sequence is than the
  recogniser's own best path, per expected phone. Only one side is recognised and the expected side comes from a
  dictionary, unlike the phone-string comparison that failed validation twice.

``evaluate`` measures each assessor the same way as the earlier scorers: how often it prefers the spelling a native
listener chose over the ones they rejected. Nothing here is trusted until that agreement is high.
"""
from __future__ import annotations

import json
import math
import re
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


_SKIPPABLE_MODIFIERS = frozenset("ːˑʰʲʷˠˤ̩̯̃̈̚")


def phone_tokens(ipa: str, vocab: Mapping[str, int]) -> list[str]:
    """Greedy longest-match of an IPA string into the recogniser's phone vocabulary (spaces are ignored)."""
    text = re.sub(r"\s+", "", ipa)
    longest = max((len(token) for token in vocab), default=1)
    tokens: list[str] = []
    i = 0
    while i < len(text):
        for width in range(min(longest, len(text) - i), 0, -1):
            token = text[i:i + width]
            if token in vocab and not token.startswith("<"):
                tokens.append(token)
                i += width
                break
        else:
            if text[i] in _SKIPPABLE_MODIFIERS:        # a length or aspiration mark the model has no symbol for on its own
                i += 1
                continue
            raise ValueError(f"phone {text[i]!r} (position {i} of {ipa!r}) is not in the recogniser's vocabulary")
    return tokens


def phones_to_ids(ipa: str, vocab: Mapping[str, int]) -> list[int]:
    return [vocab[token] for token in phone_tokens(ipa, vocab)]


# A dictionary says "Jacob" ends in a reduced vowel (schwa); a South African speaker says a full "o", and a scorer told
# to expect a schwa accepts the "a" of a wrong spelling. Reduced vowels are therefore tried as every full vowel, and the
# speaker's own audio decides which one they said.
REDUCED_VOWELS = frozenset({"ə", "ɐ", "ɪ"})
# Provisional calibration, three words: the audio clearly decided the final "o" of Jacob (11% lower cost than the
# dictionary form) but not "Moses" (6%, one short clip) or "Municipality" (2%).
MIN_RELATIVE_GAIN = 0.08
FULL_VOWELS = ("a", "e", "i", "o", "u", "ɔ", "ɛ", "ʌ", "ɑ", "ɪ", "ə")


def ipa_variants(tokens: Sequence[str], limit: int = 600) -> list[list[str]]:
    """Every variant of ``tokens`` with each reduced vowel replaced by each full vowel (the original comes first)."""
    positions = [i for i, token in enumerate(tokens) if token in REDUCED_VOWELS or (i == len(tokens) - 1 and token in {"a", "ə"})]
    variants: list[list[str]] = [list(tokens)]
    for position in positions:
        grown = []
        for variant in variants:
            for vowel in FULL_VOWELS:
                if vowel != variant[position]:
                    grown.append([*variant[:position], vowel, *variant[position + 1:]])
        variants = (variants + grown)[: limit * 4]
    seen: dict[tuple[str, ...], None] = {}
    for variant in variants:
        seen.setdefault(tuple(variant), None)
    return [list(v) for v in list(seen)[:limit]]


def choose_expected(log_probs: Any, base_tokens: Sequence[str], vocab: Mapping[str, int], blank: int,
                    limit: int = 600) -> dict[str, Any]:
    """The pronunciation variant that the speaker's audio (its log-probabilities) supports best, with the base for comparison.

    ``log_probs`` is one clip's tensor, or a list of tensors (several occurrences of the same word): each variant is then
    scored by its mean excess over the clips, so one badly recorded occurrence cannot decide the vowel alone.
    """
    clips = list(log_probs) if isinstance(log_probs, (list, tuple)) else [log_probs]
    scored = []
    for variant in (v for v in ipa_variants(base_tokens, limit) if all(t in vocab for t in v)):
        ids = [vocab[t] for t in variant]
        excess = sum(gop_excess(clip, ids, blank) for clip in clips) / len(clips)
        scored.append((excess, "".join(variant)))
    scored.sort()
    base = "".join(base_tokens)
    base_excess = next(e for e, v in scored if v == base)
    reduced_best = next(v for v in ipa_variants(base_tokens, limit) if "".join(v) == scored[0][1])
    refined_tokens, refined_excess = refine_vowels(clips, reduced_best, vocab, blank)
    # the audio overrules the dictionary only when it clearly prefers another vowel: a gain within noise (a short clip,
    # a vowel the recogniser cannot tell apart) keeps the dictionary form
    if base_excess - refined_excess >= MIN_RELATIVE_GAIN * abs(base_excess):
        chosen_tokens, chosen_excess = refined_tokens, refined_excess
    else:
        chosen_tokens, chosen_excess = list(base_tokens), base_excess
    return {"chosen": "".join(chosen_tokens), "chosen_tokens": chosen_tokens, "chosen_excess": chosen_excess, "base": base,
            "base_excess": base_excess, "best_variant": "".join(refined_tokens),
            "runners_up": [(round(e, 3), v) for e, v in scored[1:4]]}


# A dictionary's English diphthongs and vowel qualities are a hypothesis too: a Zulu speaker says "Moses" with a long
# "oː", not the English "əʊ". After the reduced vowels are settled, every vowel may be replaced, one position at a time.
_DIPHTHONGS = frozenset({"eɪ", "aɪ", "ɔɪ", "əʊ", "oʊ", "aʊ"})
_LONG_VOWELS = frozenset({"iː", "uː", "ɑː", "ɔː", "ɜː", "oː", "eː", "aː"})
VOWEL_ALTERNATIVES = ("a", "e", "i", "o", "u", "ɔ", "ɛ", "ʌ", "ɑ", "ɪ", "ə", "oː", "eː", "aː", "iː", "uː")


def refine_vowels(clips: Sequence[Any], tokens: Sequence[str], vocab: Mapping[str, int], blank: int,
                  sweeps: int = 2) -> tuple[list[str], float]:
    """Coordinate descent over the vowel positions: each is replaced by whichever vowel the clips support best."""
    current = list(tokens)

    def cost(candidate: Sequence[str]) -> float:
        ids = [vocab[t] for t in candidate]
        return sum(gop_excess(clip, ids, blank) for clip in clips) / len(clips)

    best = cost(current)
    positions = [i for i, t in enumerate(current) if t in FULL_VOWELS or t in _DIPHTHONGS or t in _LONG_VOWELS]
    for _ in range(sweeps):
        improved = False
        for position in positions:
            for vowel in VOWEL_ALTERNATIVES:
                if vowel == current[position] or vowel not in vocab:
                    continue
                trial = [*current[:position], vowel, *current[position + 1:]]
                trial_cost = cost(trial)
                if trial_cost < best - 1e-9:
                    best, current, improved = trial_cost, trial, True
        if not improved:
            break
    return current, best


def gop_excess(log_probs: Any, target_ids: Sequence[int], blank: int) -> float:
    """Per-phone excess cost of the expected sequence over the recogniser's own best path (0 = as good as it gets).

    ``log_probs`` is a (frames, vocabulary) tensor of log-probabilities. The expected cost is the CTC negative
    log-likelihood of ``target_ids``, summed over every alignment; the free cost is the best path with no constraint.
    """
    import torch

    frames = log_probs.shape[0]
    targets = torch.tensor([list(target_ids)], dtype=torch.long)
    expected = torch.nn.functional.ctc_loss(
        log_probs.unsqueeze(1), targets, torch.tensor([frames]), torch.tensor([len(target_ids)]),
        blank=blank, reduction="sum", zero_infinity=True)
    free = -log_probs.max(dim=-1).values.sum()
    return float((expected - free) / max(1, len(target_ids)))


class LocalGop:
    """Goodness of pronunciation from a wav2vec 2.0 phoneme model, running on CPU (about 1 s per second of audio)."""

    MODEL = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or self.MODEL
        self._model: Any = None
        self._vocab: dict[str, int] = {}

    def _load(self) -> None:
        if self._model is not None:
            return
        from huggingface_hub import hf_hub_download
        from transformers import Wav2Vec2ForCTC

        self._vocab = json.loads(Path(hf_hub_download(self.model_name, "vocab.json")).read_text(encoding="utf-8"))
        self._model = Wav2Vec2ForCTC.from_pretrained(self.model_name).eval()

    def log_probs(self, wav: Path) -> Any:
        import soundfile
        import torch
        import torchaudio

        self._load()
        audio, rate = soundfile.read(str(wav), dtype="float32", always_2d=True)
        signal = torch.from_numpy(audio.mean(axis=1))
        if rate != 16_000:
            signal = torchaudio.functional.resample(signal, rate, 16_000)
        signal = (signal - signal.mean()) / torch.sqrt(signal.var() + 1e-7)
        with torch.no_grad():
            logits = self._model(signal.unsqueeze(0)).logits[0]
        return torch.log_softmax(logits, dim=-1)

    def decode(self, wav: Path) -> list[str]:
        """The phones the recogniser hears in ``wav``, with no expectation: its best path with repeats and blanks removed."""
        inverse = {index: token for token, index in self._vocab.items()}
        blank = self._vocab["<pad>"]
        heard: list[str] = []
        previous = None
        for index in self.log_probs(wav).argmax(dim=-1).tolist():
            if index != previous and index != blank:
                heard.append(inverse[index])
            previous = index
        return [token for token in heard if not token.startswith("<")]

    def choose_expected(self, wav: "Path | Sequence[Path]", base_ipa: str) -> dict[str, Any]:
        """Decide how the speaker in ``wav`` (or in several clips of the same word) pronounced it, from the dictionary form
        and its reduced-vowel variants."""
        self._load()
        wavs = [wav] if isinstance(wav, (str, Path)) else list(wav)
        return choose_expected([self.log_probs(w) for w in wavs], phone_tokens(base_ipa, self._vocab), self._vocab,
                               self._vocab["<pad>"])

    def assess(self, wav: Path, ipa: str) -> dict[str, float]:
        self._load()
        excess = gop_excess(self.log_probs(wav), phones_to_ids(ipa, self._vocab), blank=self._vocab["<pad>"])
        return {"score": -excess, "excess": excess}


def azure_assessor(key: str, region: str, locale: str, reference_text: str) -> Callable[[Path], dict[str, Any]]:
    """Azure pronunciation assessment of a wav against ``reference_text`` (score = accuracy / 100; 0 if nothing is recognised)."""
    import azure.cognitiveservices.speech as speechsdk

    def assess(wav: Path) -> dict[str, Any]:
        config = speechsdk.SpeechConfig(subscription=key, region=region)
        config.speech_recognition_language = locale
        recogniser = speechsdk.SpeechRecognizer(speech_config=config, audio_config=speechsdk.audio.AudioConfig(filename=str(wav)))
        speechsdk.PronunciationAssessmentConfig(
            reference_text=reference_text, grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
            granularity=speechsdk.PronunciationAssessmentGranularity.Phoneme, enable_miscue=False).apply_to(recogniser)
        result = recogniser.recognize_once_async().get()
        if result.reason != speechsdk.ResultReason.RecognizedSpeech:
            return {"score": 0.0, "recognised": False, "reason": str(result.reason)}
        assessed = speechsdk.PronunciationAssessmentResult(result)
        return {"score": assessed.accuracy_score / 100.0, "recognised": True, "text": result.text,
                "accuracy": assessed.accuracy_score, "pronunciation": assessed.pronunciation_score,
                "completeness": assessed.completeness_score}

    return assess


def english_reading(name: str, voice: str, key: str, region: str, path: Path) -> Path:
    """The name read by a South African English voice: a sanity reference that any sensible assessor should score highly."""
    import requests

    ssml = (f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-ZA">'
            f'<voice name="{voice}">{name}.</voice></speak>')
    response = requests.post(
        f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1", timeout=60, data=ssml.encode("utf-8"),
        headers={"Ocp-Apim-Subscription-Key": key, "Content-Type": "application/ssml+xml",
                 "X-Microsoft-OutputFormat": "riff-24khz-16bit-mono-pcm", "User-Agent": "mathula-pronunciation-assess"})
    response.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)
    return path


def cached(name: str, assess: Callable[[Path], Mapping[str, Any]], store: Path) -> Callable[[Path], float]:
    """Wrap an assessor so each wav is assessed once; details are kept on disk beside the audio for inspection."""
    store.mkdir(parents=True, exist_ok=True)

    def run(wav: Path) -> float:
        path = store / f"{wav.stem}.{name}.json"
        if not path.is_file():
            path.write_text(json.dumps(dict(assess(wav)), ensure_ascii=False), encoding="utf-8")
        return float(json.loads(path.read_text(encoding="utf-8"))["score"])

    return run


def azure_goodness(score: float) -> float:
    """Azure accuracy (0-1) spread over its useful range: below 0.8 is poor, and 0.93-0.99 is where the choices lie."""
    return round(max(0.0, min(1.0, (score - 0.8) / 0.2)), 6)


def gop_goodness(score: float) -> float:
    """GOP score (minus the per-phone excess cost) as a 0-1 likelihood ratio: 1 means as good as the recogniser's own path."""
    return math.exp(min(0.0, score))


@dataclass
class CombinedScorer:
    """A spelling is only as good as its worst assessor: it must look right to every one of them.

    The two assessors agreed with the listener 9 times in 11 when they agreed with each other, and were unreliable
    when they did not, so the minimum (not the mean) is the objective: a candidate that fools one assessor does not rise.
    """
    parts: Sequence[tuple[Any, Callable[[float], float]]]          # (scorer, transform to a 0-1 goodness)
    evaluations: int = 0
    _cache: dict = field(default_factory=dict)
    detail: dict = field(default_factory=dict)

    def score(self, candidate: Any) -> float:
        if candidate not in self._cache:
            self.evaluations += 1
            values = [transform(scorer.score(candidate)) for scorer, transform in self.parts]
            self.detail[candidate] = values
            self._cache[candidate] = min(values)
        return self._cache[candidate]


def concatenate_takes(wavs: Sequence[Path], target: Path, gap_s: float = 1.0) -> Path:
    """Join wavs of one format into one file with a pause after each, for a listener to compare back to back."""
    data, params = b"", None
    for path in wavs:
        with wave.open(str(path), "rb") as reader:
            params = params or (reader.getnchannels(), reader.getsampwidth(), reader.getframerate())
            data += reader.readframes(reader.getnframes())
            data += b"\x00" * (params[1] * params[0] * int(params[2] * gap_s))
    assert params is not None, "no takes to join"
    with wave.open(str(target), "wb") as writer:
        writer.setnchannels(params[0])
        writer.setsampwidth(params[1])
        writer.setframerate(params[2])
        writer.writeframes(data)
    return target


def main(argv: list[str] | None = None) -> int:
    import argparse
    import os

    from . import pronunciation_search as ps
    from .config import load_settings
    from .native_dub import _extract_entity_english_source_clip

    parser = argparse.ArgumentParser(prog="mathula_tv.pronunciation_assess")
    parser.add_argument("--config", required=True, help="JSON like the search config, plus per entity: reference_text, ipa")
    parser.add_argument("--out", default="pronunciation_assess")
    parser.add_argument("--assessors", nargs="*", default=["azure-en-GB", "azure-en-US", "local-gop"])
    parser.add_argument("--search", action="store_true", help="search with Azure en-US + GOP combined, write takes for listening")
    parser.add_argument("--budget", type=int, default=50, help="candidates the search may score per entity")
    parser.add_argument("--words", action="store_true", help="respell word by word, leaving words untouched unless a change clearly helps")
    args = parser.parse_args(argv)
    if args.search:
        args.assessors = ["azure-en-US", "local-gop"]

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    settings = load_settings()
    job_root = Path(settings.work_dir) / "jobs" / config["job"]
    key, region = os.environ["AZURE_SPEECH_KEY"], os.environ["AZURE_SPEECH_REGION"].strip().lower()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    synthesize = ps.bookmark_synthesizer(config.get("voice", "zu-ZA-ThembaNeural"), key, region)
    gop = LocalGop() if ("local-gop" in args.assessors or args.words) else None
    if args.words:
        return _run_words(args, config, ps, gop, synthesize, job_root, out)

    if args.search:
        _derive_expected(config, ps, gop, job_root, out)
        return _run_search(args, config, ps, gop, synthesize, key, region, out)

    _derive_expected(config, ps, gop, job_root, out)

    summary: dict[str, Any] = {}
    for entity in config["entities"]:
        name = entity["name"]
        slug = re.sub(r"\W+", "_", name.lower())
        contexts = [ps.Context(**c) for c in entity["contexts"]]
        pairs = [(ps._candidate(p["better"]), ps._candidate(p["worse"])) for p in entity.get("pairs", [])]
        windows = ps.NameWindows(synthesize, lambda wav: (), out / slug)
        clip = _extract_entity_english_source_clip(job_root=job_root, entity_id=slug, canonical_text=name, audio_dir=out / "reference")
        references = {"english_voice": english_reading(entity["reference_text"], "en-ZA-LukeNeural", key, region,
                                                       out / "reference" / f"{slug}_en_tts.wav")}
        if clip is not None:
            references["source_clip"] = clip
        summary[name] = {}
        print(f"\n== {name}   (reference text {entity['reference_text']!r}, expected phones {entity['ipa']!r})")
        for assessor in args.assessors:
            if assessor == "local-gop":
                assess_one = cached(assessor, lambda wav, g=gop: g.assess(wav, entity["ipa"]), out / slug / "assess")
            else:
                locale = assessor.removeprefix("azure-")
                assess_one = cached(assessor, azure_assessor(key, region, locale, entity["reference_text"]), out / slug / "assess")
            scorer = ps.AssessmentScorer(windows, assess_one, contexts)
            checked = ps.agreement(scorer, pairs)
            baseline = {label: round(assess_one(path), 3) for label, path in references.items()}
            summary[name][assessor] = {"agreement": checked, "reference_audio_scores": baseline}
            print(f"  [{assessor}] agreement {checked['agreement']}   (source clip scores {baseline})")
            for row in checked["pairs"]:
                print(f"      {'OK ' if row['agrees'] else 'NO '} {row['chosen']!r} {row['chosen_score']}  vs  {row['rejected']!r} {row['rejected_score']}")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    totals = {a: [0, 0] for a in args.assessors}
    for per_entity in summary.values():
        for assessor, result in per_entity.items():
            for row in result["agreement"]["pairs"]:
                totals[assessor][0] += int(row["agrees"])
                totals[assessor][1] += 1
    print("\n== overall agreement with the listener")
    for assessor, (agree, total) in totals.items():
        print(f"   {assessor:<12} {agree}/{total}" + (f" = {agree / total:.0%}" if total else ""))
    return 0


def sanitize_ipa(ipa: str, vocab: Mapping[str, int]) -> str:
    """Strip the stress marks, slashes and syllable dots a pronunciation note carries, and drop symbols the model lacks."""
    text = re.sub(r"[/\[\]ˈˌ.]", "", ipa)
    known = {ch for token in vocab for ch in token}
    return " ".join("".join(ch for ch in word if ch in known) for word in text.split())


def strip_length(ipa: str) -> str:
    """Remove vowel-length marks. isiZulu lengthens the second-to-last syllable of a phrase-final word ("Maseko" said as
    "Maseːko" before a pause) but not mid-sentence, so a clip taken at a pause must not teach that length."""
    return ipa.replace("ː", "").replace("ˑ", "")


def regroup(tokens: Sequence[str], base_ipa: str, vocab: Mapping[str, int]) -> list[str]:
    """Split a chosen token sequence back into words using the word lengths of ``base_ipa`` (variants swap 1:1)."""
    lengths = [len(phone_tokens(word, vocab)) for word in base_ipa.split()]
    words, at = [], 0
    for n in lengths:
        words.append("".join(tokens[at:at + n]))
        at += n
    return words


def _run_words(args: Any, config: Mapping[str, Any], ps: Any, gop: "LocalGop", synthesize: Any, job_root: Path, out: Path) -> int:
    """Word-by-word respelling of the names in ``word_entities``: untouched unless a change clearly helps."""
    import os

    from .native_dub import _extract_entity_english_source_clip

    research = json.loads((job_root / "native_dub" / "pronunciation_research.json").read_text(encoding="utf-8"))
    by_name = {c["canonical_text"]: c for c in research.get("accepted_corrections", [])}
    report: dict[str, Any] = {}
    gop._load()
    for entity in config["word_entities"]:
        name = entity["name"]
        slug = re.sub(r"\W+", "_", name.lower())
        base = entity.get("ipa") or sanitize_ipa(by_name[name]["pronunciation_ipa"], gop._vocab)
        clip = _extract_entity_english_source_clip(job_root=job_root, entity_id=slug, canonical_text=name, audio_dir=out / "reference")
        source = "the speaker's audio"
        if clip is None:
            # nobody says it in the clip: a South African English voice reading the name is the next best reference. Only the
            # English-shaped words are ever changed, so its mangling of an isiZulu word cannot reach the result.
            clip = english_reading(name, "en-ZA-LukeNeural", os.environ["AZURE_SPEECH_KEY"],
                                   os.environ["AZURE_SPEECH_REGION"].strip().lower(), out / "reference" / f"{slug}_en_tts.wav")
            source = "an English voice reading the name (no speaker said it)"
        picked = gop.choose_expected(clip, base)
        words = regroup(picked["chosen_tokens"], base, gop._vocab)
        name_words = name.split()
        if len(words) != len(name_words):
            print(f"\n== {name}: {len(words)} sound groups for {len(name_words)} words; skipped")
            continue
        contexts = [ps.Context(**c) for c in entity.get("contexts", config["default_contexts"])]
        windows = ps.NameWindows(synthesize, lambda wav: (), out / slug)
        memo: dict[tuple[str, str], float] = {}

        def goodness(wav: Path, ipa: str) -> float:
            if (wav.name, ipa) not in memo:
                memo[(wav.name, ipa)] = gop_goodness(gop.assess(wav, ipa)["score"])
            return memo[(wav.name, ipa)]

        words = [strip_length(w) if ps.looks_isizulu(word) else w for w, word in zip(words, name_words)]
        scorer = ps.WordScorer(windows, goodness, words, contexts)
        decisions = ps.respell_words(scorer, name, margin=float(config.get("margin", 0.15)), budget=int(config.get("word_budget", 24)))
        final = " ".join(d["decision"] for d in decisions)
        print(f"\n== {name}  (expected sounds from {source}: {' | '.join(words)};  base from research: {base!r})")
        for d in decisions:
            verdict = f"CHANGED to {d['decision']!r}" if d["changed"] else "kept untouched"
            print(f"   {d['word']:<14} plain {d['plain']:.3f}   best found {d['best']!r} {d['best_score']:.3f}   -> {verdict}")
        print(f"   result: {final!r}   ({scorer.evaluations} spellings scored)")
        report[name] = {"expected_words": words, "decisions": decisions, "result": final}
    (out / "word_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


def _derive_expected(config: Mapping[str, Any], ps: Any, gop: "LocalGop | None", job_root: Path, out: Path) -> None:
    """Replace each entity's dictionary IPA with the variant the real speaker's audio supports best (when a clip exists)."""
    from .native_dub import _extract_entity_english_source_clip

    if gop is None or config.get("_expected_derived"):
        return
    for entity in config["entities"]:
        slug = re.sub(r"\W+", "_", entity["name"].lower())
        clip = _extract_entity_english_source_clip(job_root=job_root, entity_id=slug, canonical_text=entity["name"],
                                                   audio_dir=out / "reference")
        if clip is None:
            continue
        picked = gop.choose_expected(clip, entity["ipa"])
        print(f"expected pronunciation for {entity['name']}: dictionary {picked['base']!r} (excess {picked['base_excess']:.3f}) "
              f"-> the speaker's audio supports {picked['chosen']!r} (excess {picked['chosen_excess']:.3f}); next best {picked['runners_up'][:2]}")
        entity["ipa"] = picked["chosen"]
    config["_expected_derived"] = True  # type: ignore[index]


def _run_search(args: Any, config: Mapping[str, Any], ps: Any, gop: "LocalGop | None", synthesize: Any,
                key: str, region: str, out: Path) -> int:
    report: dict[str, Any] = {}
    for entity in config["entities"]:
        name = entity["name"]
        slug = re.sub(r"\W+", "_", name.lower())
        contexts = [ps.Context(**c) for c in entity["contexts"]]
        windows = ps.NameWindows(synthesize, lambda wav: (), out / slug)
        store = out / slug / "assess"
        azure = ps.AssessmentScorer(
            windows, cached("azure-en-US", azure_assessor(key, region, "en-US", entity["reference_text"]), store), contexts)
        local = ps.AssessmentScorer(
            windows, cached("local-gop", lambda wav: gop.assess(wav, entity["ipa"]), store), contexts)
        combined = CombinedScorer([(azure, azure_goodness), (local, gop_goodness)])
        human = [ps._candidate(p["better"]) for p in entity.get("pairs", [])][:1]
        ranked = ps.search(combined, entity["seeds"], width=3, depth=5, budget=args.budget)
        top = [c for _, c in ranked[:5]]
        for candidate in human:
            combined.score(candidate)
        print(f"\n== {name}: search with the combined score ({combined.evaluations} candidates scored)")
        print(f"   {'combined':>8} {'azure':>7} {'gop':>7}   candidate")
        for score, candidate in ranked[:8]:
            a, g = combined.detail[candidate]
            print(f"   {score:8.3f} {a:7.3f} {g:7.3f}   {candidate.label()!r}")
        for candidate in human:
            a, g = combined.detail[candidate]
            print(f"   {combined.score(candidate):8.3f} {a:7.3f} {g:7.3f}   {candidate.label()!r}  <- your choice")
        order = [*human, *[c for c in top if c not in human]][:6]
        files = []
        for candidate in order:
            for context in contexts[:2]:
                name_wav = windows.audio(context, candidate)
                files.append(name_wav.with_name(name_wav.name.replace(".name.wav", ".wav")))
        takes = concatenate_takes(files, out / f"takes_{slug}.wav")
        print(f"   takes file {takes}: " + "; ".join(f"{i}: {c.label()}" for i, c in enumerate(order, 1))
              + "   (two sentences per candidate)")
        report[name] = {
            "ranked": [{"candidate": c.label(), "combined": round(s, 3), "azure": round(combined.detail[c][0], 3),
                        "gop": round(combined.detail[c][1], 3)} for s, c in ranked[:15]],
            "takes": str(takes), "takes_order": [c.label() for c in order]}
    (out / "search_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
