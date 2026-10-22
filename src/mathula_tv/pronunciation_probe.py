"""Loop candidate pronunciations through TTS and STT, to find which ones a recogniser hears as the intended word.

For code-switched English number words ("ngowesi-seven", "ngo-twenty-two") the question is whether the isiZulu
voice produces something an English listener recognises as that number. A probe synthesises each candidate several
times (Azure synthesis varies from call to call), transcribes it with the English recogniser, and scores the share of
samples in which the intended number comes back.

What this can and cannot say: a recogniser is lenient, so a pass means "intelligible as that word", not "sounds
right to a native listener", and it cannot tell a good accent from a mediocre one. A FAIL is much more informative
than a pass. Use it to reject bad respellings and to rank candidates, then let a person confirm the ones that matter.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

CARRIERS = {
    "ordinal": "Angaba ngowesi-{x} osusiwe.",
    "date": "Isitatimende sikhishwe ngo-{x} October.",
}
_UNITS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen "
    "eighteen nineteen".split())}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}


def numbers_in(text: str) -> list[int]:
    """Every number a transcript contains, whether written as digits or as English words ("twenty two", "twenty-two")."""
    found: list[int] = []
    tokens = re.findall(r"[a-z]+|\d+", str(text or "").lower())
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.isdigit():
            found.append(int(token))
        elif token in _TENS:
            value = _TENS[token]
            if i + 1 < len(tokens) and tokens[i + 1] in _UNITS and 1 <= _UNITS[tokens[i + 1]] <= 9:
                value += _UNITS[tokens[i + 1]]
                i += 1
            found.append(value)
        elif token in _UNITS:
            found.append(_UNITS[token])
        i += 1
    return found


def probe_candidate(
    synthesize: Callable[[str, Path], None], transcribe: Callable[[Path], str], *, number: int, candidate: str,
    carrier: str, audio_dir: Path, repeats: int = 3, tag: str = "",
) -> dict[str, Any]:
    """Synthesise ``carrier`` with ``candidate`` ``repeats`` times; score how often ``number`` is heard back."""
    audio_dir.mkdir(parents=True, exist_ok=True)
    text = carrier.format(x=candidate)
    samples = []
    for repeat in range(repeats):
        path = audio_dir / f"{tag}n{number:02d}_{re.sub(r'[^a-z0-9]+', '-', candidate.lower())}_{repeat}.wav"
        synthesize(text, path)
        heard = transcribe(path)
        samples.append({"heard": heard, "ok": number in numbers_in(heard)})
    return {"number": number, "candidate": candidate, "text": text,
            "score": sum(s["ok"] for s in samples) / len(samples), "samples": samples}


def probe_numbers(
    synthesize: Callable[[str, Path], None], transcribe: Callable[[Path], str], *, numbers: Sequence[int],
    word_for: Callable[[int], str], alternatives: Mapping[int, Sequence[str]] | None = None, contexts: Sequence[str] = ("ordinal", "date"),
    audio_dir: Path, repeats: int = 3, try_alternatives_below: float = 1.0,
) -> list[dict[str, Any]]:
    """Probe the shipped word for every number; try the alternatives only where the shipped word is not heard every time."""
    rows = []
    for number in numbers:
        for context in contexts:
            carrier = CARRIERS[context]
            shipped = probe_candidate(synthesize, transcribe, number=number, candidate=word_for(number), carrier=carrier,
                                      audio_dir=audio_dir, repeats=repeats, tag=f"{context}_")
            rows.append({**shipped, "context": context, "role": "shipped"})
            if shipped["score"] < try_alternatives_below:
                for alt in (alternatives or {}).get(number, ()):
                    result = probe_candidate(synthesize, transcribe, number=number, candidate=alt, carrier=carrier,
                                             audio_dir=audio_dir, repeats=repeats, tag=f"{context}_")
                    rows.append({**result, "context": context, "role": "alternative"})
    return rows


def best_per_number(rows: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    """For each number: the candidate with the best mean score over contexts, preferring the shipped word on a tie."""
    by_candidate: dict[tuple[int, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        by_candidate.setdefault((row["number"], row["candidate"]), []).append(row)
    best: dict[int, dict[str, Any]] = {}
    for (number, candidate), group in by_candidate.items():
        mean = sum(r["score"] for r in group) / len(group)
        shipped = any(r["role"] == "shipped" for r in group)
        current = best.get(number)
        if current is None or (mean, shipped) > (current["score"], current["shipped"]):
            best[number] = {"candidate": candidate, "score": round(mean, 3), "shipped": shipped}
    return best


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(prog="mathula_tv.pronunciation_probe")
    parser.add_argument("--voice", default="zu-ZA-ThembaNeural")
    parser.add_argument("--from", dest="first", type=int, default=1)
    parser.add_argument("--to", dest="last", type=int, default=31)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", default="pronunciation_probe")
    args = parser.parse_args(argv)

    from .azure_stt import normalize as normalize_stt_response
    from .azure_tts import AzureTTSRequest
    from .cli import create_speech_backend, create_tts_backend
    from .config import load_settings
    from .pronunciation import _ZU_ENGLISH_DIGIT_WORDS, _english_cardinal_word

    settings = load_settings()
    tts, stt = create_tts_backend(settings), create_speech_backend(settings)
    counter = [0]

    def synthesize(text: str, path: Path) -> None:
        counter[0] += 1
        tts.synthesize(AzureTTSRequest(turn_id=f"probe_{counter[0]}", speaker_id="probe", text=text,
                                       preferred_duration_ms=6000, maximum_duration_ms=15000, voice=args.voice),
                       path, force=True)

    def transcribe(path: Path) -> str:
        data = normalize_stt_response(stt.transcribe(path, locale="en-ZA"))
        return " ".join(str(w.get("text") or "") for w in data.get("words") or ())

    alternatives = {n: [_ZU_ENGLISH_DIGIT_WORDS[str(n)]] for n in range(1, 10)}      # the pipeline's hand-tuned digit words
    out = Path(args.out)
    rows = probe_numbers(synthesize, transcribe, numbers=range(args.first, args.last + 1), word_for=_english_cardinal_word,
                         alternatives=alternatives, audio_dir=out / "audio", repeats=args.repeats)
    (out / "rows.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    for number, pick in sorted(best_per_number(rows).items()):
        shipped = [r for r in rows if r["number"] == number and r["role"] == "shipped"]
        heard = "; ".join(f"{r['context']}: {s['heard']!r}" for r in shipped for s in r["samples"] if not s["ok"])
        print(f"{number:>2}  shipped={_english_cardinal_word(number)!r:<16} best={pick['candidate']!r:<16} "
              f"score={pick['score']}  {'(shipped)' if pick['shipped'] else '(ALTERNATIVE WINS)'}  misheard: {heard or '-'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
