"""After the render: did the voice actually say every word?

The TTS voice sometimes swallows a short word (a final "hlu" before a pause is typical). Nobody finds out until a listener
does. This check transcribes each unit's rendered speech and compares the transcript with the text that was spoken. A word is
reported as dropped only when it is missing from the transcript AND the audio has a gap where it should be: a missing word
with no gap is usually the recogniser merging two words, not a word the voice swallowed.

It is a check, not a repair: it reports where, and never changes the dub or blocks it.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Mapping, Sequence

MIN_GAP_SECONDS = 0.5           # the audio gap that one swallowed word leaves (it grows by PER_EXTRA_WORD for each further word)
PER_EXTRA_WORD_SECONDS = 0.3


def _norm(text: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text or "").casefold())


def dropped_words(spoken_text: str, heard: Sequence[Mapping[str, Any]], *, min_gap: float = MIN_GAP_SECONDS) -> list[dict[str, Any]]:
    """Words of ``spoken_text`` that the voice swallowed, from the transcript ``heard`` ({text, start, end} in seconds)."""
    expected_raw = [t for t in spoken_text.split() if _norm(t)]
    expected = [_norm(t) for t in expected_raw]
    transcript = [(_norm(w.get("text")), float(w.get("start") or 0.0), float(w.get("end") or w.get("start") or 0.0))
                  for w in heard if _norm(w.get("text"))]
    matcher = SequenceMatcher(None, expected, [t for t, _, _ in transcript], autojunk=False)
    found: list[dict[str, Any]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "delete":                 # a substituted or merged word is a recognition difference, not a swallowed word
            continue
        count = i2 - i1
        before = transcript[j1 - 1][2] if j1 > 0 else None
        after = transcript[j1][1] if j1 < len(transcript) else None
        if before is None or after is None:
            continue                        # at the very start or end of the unit there is no gap to measure
        gap = after - before
        if gap >= min_gap + PER_EXTRA_WORD_SECONDS * (count - 1):
            found.append({"words": expected_raw[i1:i2], "after": expected_raw[i1 - 1] if i1 else None,
                          "before": expected_raw[i2] if i2 < len(expected_raw) else None,
                          "gap_s": round(gap, 2), "at_s": round(before + gap / 2, 2)})
    return found


CHECK_WORKERS = 6
RECHECK_MARGIN_S = 0.8                 # seconds of the unit either side of a flagged gap that are transcribed again


def _letters(word: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(word).casefold())


def _recheck(wav: Path, dropped: list[dict[str, Any]], backend: Any, normalize: Any, locale: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Transcribe only the stretch around each flagged gap.  Words that are heard there were spoken (the whole-unit pass skipped them): they are
    not reported.  Words still not heard stay reported.  A failure to recheck leaves the item reported, as before."""
    import subprocess
    import tempfile
    from difflib import SequenceMatcher

    kept, cleared = [], []
    with tempfile.TemporaryDirectory() as directory:
        for index, item in enumerate(dropped):
            start, end = max(0.0, float(item["at_s"]) - RECHECK_MARGIN_S), float(item["at_s"]) + float(item.get("gap_s") or 0) + RECHECK_MARGIN_S
            clip = Path(directory) / f"gap{index}.wav"
            try:
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", str(wav), "-ac", "1", "-ar", "16000",
                                str(clip)], check=True)
                heard = {_letters(w.get("text", "")) for w in normalize(backend.transcribe(clip, locale=locale)).get("words") or []} - {""}
            except Exception:
                kept.append(item)
                continue
            missing = [w for w in item["words"] if not any(SequenceMatcher(None, _letters(w), h).ratio() >= 0.8 for h in heard)]
            if missing:
                kept.append({**item, "words": missing})
            else:
                cleared.append({"words": item["words"], "at_s": item["at_s"]})
    return kept, cleared


def check_run(run_dir: Path, backend: Any, normalize: Any, *, locale: str = "zu-ZA") -> dict[str, Any]:
    """Check every unit of a rendered run; writes ``intelligibility.json`` beside the scorecard."""
    run_dir = Path(run_dir)
    scorecard = json.loads((run_dir / "scorecard.json").read_text(encoding="utf-8"))
    audio_dir = run_dir / "render" / "native_dub" / "audio" / "phrases"
    checks = []
    for group in scorecard.get("groups") or []:
        wav = audio_dir / f"{group.get('group_id')}.wav"
        if wav.is_file() and group.get("spoken_text"):
            checks.append((group, wav))

    def check(pair: tuple[dict[str, Any], Path]) -> dict[str, Any]:
        group, wav = pair
        heard = normalize(backend.transcribe(wav, locale=locale)).get("words") or []
        offset = float(group.get("start_ms") or 0) / 1000
        dropped = dropped_words(str(group["spoken_text"]), heard)
        spoken_after_all: list[dict[str, Any]] = []
        if dropped:                  # one pass over a whole unit can skip a short phrase: listen again to just the stretch before reporting it
            dropped, spoken_after_all = _recheck(wav, dropped, backend, normalize, locale)
        for item in dropped:
            item["at_video_s"] = round(offset + item["at_s"], 2)
        return {"unit": str(group.get("group_id")), "speaker": group.get("speaker_id"), "dropped": dropped,
                **({"heard_on_rechecking": spoken_after_all} if spoken_after_all else {})}

    # The units are independent requests to the speech service: side by side (the order of the units is kept), not one after another.
    with ThreadPoolExecutor(max_workers=max(1, min(CHECK_WORKERS, len(checks)))) as pool:
        units: list[dict[str, Any]] = list(pool.map(check, checks))
    try:        # quantities that went to the voice digit by digit (see number_check)
        from .number_check import check_numbers

        number_flags = check_numbers(run_dir)
    except Exception as exc:
        number_flags = [{"problem": f"number check failed: {type(exc).__name__}: {str(exc)[:100]}"}]
    report = {"units_checked": len(units), "dropped_total": sum(len(u["dropped"]) for u in units),
              "units": [u for u in units if u["dropped"]], "number_flags": number_flags}
    (run_dir / "intelligibility.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    import argparse

    from .azure_stt import normalize
    from .cli import create_speech_backend
    from .config import load_settings

    parser = argparse.ArgumentParser(prog="mathula_tv.dub_intelligibility")
    parser.add_argument("--run-dir", required=True, help="a rendered run directory (holds scorecard.json and render/)")
    args = parser.parse_args(argv)
    report = check_run(Path(args.run_dir), create_speech_backend(load_settings()), normalize)
    print(f"{report['units_checked']} units checked, {report['dropped_total']} dropped word(s), "
          f"{len(report['number_flags'])} number problem(s)")
    for flag in report["number_flags"]:
        print(f"  number {flag.get('number')} in {flag.get('unit')} ({flag.get('speaker')}): {flag['problem']}")
    for unit in report["units"]:
        for item in unit["dropped"]:
            print(f"  {unit['unit']} ({unit['speaker']}) at {item['at_video_s']}s: {' '.join(item['words'])!r} "
                  f"between {item['after']!r} and {item['before']!r} (gap {item['gap_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
