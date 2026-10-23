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
import re
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


def phones_to_ids(ipa: str, vocab: Mapping[str, int]) -> list[int]:
    """Greedy longest-match of an IPA string into the recogniser's phone vocabulary (spaces are ignored)."""
    text = re.sub(r"\s+", "", ipa)
    longest = max((len(token) for token in vocab), default=1)
    ids: list[int] = []
    i = 0
    while i < len(text):
        for width in range(min(longest, len(text) - i), 0, -1):
            token = text[i:i + width]
            if token in vocab and not token.startswith("<"):
                ids.append(vocab[token])
                i += width
                break
        else:
            raise ValueError(f"phone {text[i]!r} (position {i} of {ipa!r}) is not in the recogniser's vocabulary")
    return ids


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
    args = parser.parse_args(argv)

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    settings = load_settings()
    job_root = Path(settings.work_dir) / "jobs" / config["job"]
    key, region = os.environ["AZURE_SPEECH_KEY"], os.environ["AZURE_SPEECH_REGION"].strip().lower()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    synthesize = ps.bookmark_synthesizer(config.get("voice", "zu-ZA-ThembaNeural"), key, region)
    gop = LocalGop() if "local-gop" in args.assessors else None

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


if __name__ == "__main__":
    raise SystemExit(main())
