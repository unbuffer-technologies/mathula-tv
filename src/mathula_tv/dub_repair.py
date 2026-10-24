"""Repair the words the voice swallowed (see ``dub_intelligibility``), once, at a bounded cost.

For each dropped word:
  1. build up to ``MAX_CANDIDATES`` respellings by rule (a longer final vowel, a doubled one, a leading h);
  2. test each in the unit's own SSML (same voice, prosody and surrounding text, so the failure is reproduced), by
     synthesising the unit with only that word changed and transcribing the result;
  3. keep the first respelling after which the word is no longer dropped.
If any word was fixed, the respellings go into the job's lexicon and the clip is rendered ONCE more, then checked once more.
Nothing loops: a word that no candidate fixes is reported as unfixed and left as it was.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Mapping

from .dub_intelligibility import check_run, dropped_words

MAX_CANDIDATES = 3
REPAIR_SOURCE = "swallowed-word repair"
_VOWELS = "aeiou"


def candidates(word: str) -> list[str]:
    """Respellings that give a swallowed word more voiced material, most conservative first."""
    base = re.sub(r"[^A-Za-z]", "", word)
    if not base:
        return []
    last = next((c for c in reversed(base.lower()) if c in _VOWELS), "")
    options = [base + last, base + last * 2, "h" + base] if last else ["h" + base]
    unique: list[str] = []
    for option in options:
        if option.lower() != base.lower() and option not in unique:
            unique.append(option)
    return unique[:MAX_CANDIDATES]


def with_candidate(ssml: str, after: str | None, word: str, candidate: str) -> str | None:
    """The unit's SSML with the dropped occurrence of ``word`` (the one after ``after``) replaced; None if it cannot be placed."""
    bare = re.sub(r"[^A-Za-z]", "", word)
    anchor = re.escape(re.sub(r"[^A-Za-z]", "", after or "")) if after else ""
    pattern = re.compile((rf"({anchor}\W+)" if anchor else r"(>\s*)") + rf"{re.escape(bare)}(?![A-Za-z])", re.IGNORECASE)
    found = pattern.search(ssml)
    if not found:
        return None
    return ssml[:found.start()] + found.group(1) + candidate + ssml[found.end():]


def synthesize_ssml(ssml: str, key: str, region: str, path: Path) -> Path:
    import requests

    response = requests.post(
        f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1", timeout=180, data=ssml.encode("utf-8"),
        headers={"Ocp-Apim-Subscription-Key": key, "Content-Type": "application/ssml+xml",
                 "X-Microsoft-OutputFormat": "riff-24khz-16bit-mono-pcm", "User-Agent": "mathula-dub-repair"})
    response.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)
    return path


def _same_spot(item: Mapping[str, Any], other: Mapping[str, Any]) -> bool:
    return item["words"] == other["words"] and item.get("after") == other.get("after") and item.get("before") == other.get("before")


def find_respelling(item: Mapping[str, Any], unit_text: str, ssml: str, transcribe: Callable[[Path], list[dict[str, Any]]],
                    synthesize: Callable[[str, Path], Path], work: Path) -> dict[str, Any]:
    """The first candidate that removes this dropped word from the unit, with what was tried."""
    word = item["words"][0]
    tried: list[dict[str, Any]] = []
    if len(item["words"]) != 1:
        return {"word": " ".join(item["words"]), "fixed": False, "tried": tried, "why": "several consecutive words dropped"}
    for index, candidate in enumerate(candidates(word)):
        trial = with_candidate(ssml, item.get("after"), word, candidate)
        if trial is None:
            return {"word": word, "fixed": False, "tried": tried, "why": "the word could not be located in the unit's SSML"}
        heard = transcribe(synthesize(trial, work / f"candidate_{index}.wav"))
        still = any(_same_spot(item, d) for d in dropped_words(unit_text, heard))
        tried.append({"candidate": candidate, "still_dropped": still})
        if not still:
            return {"word": word, "respelling": candidate, "fixed": True, "tried": tried}
    return {"word": word, "fixed": False, "tried": tried, "why": "no candidate brought the word back"}


def add_to_lexicon(path: Path, fixes: list[dict[str, Any]]) -> None:
    data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"schema": "mathula-names-lexicon-v1", "entries": []}
    taken = {e["display_text"].casefold() for e in data.get("entries", []) if e.get("source") != REPAIR_SOURCE}
    data["entries"] = [e for e in data.get("entries", []) if e.get("source") != REPAIR_SOURCE or e["display_text"] not in
                       {f["word"] for f in fixes}]
    for fix in fixes:
        bare = re.sub(r"[^A-Za-z]", "", fix["word"])
        if bare.casefold() in taken:
            continue
        data["entries"].append({"display_text": bare, "tts_text": fix["respelling"], "kind": "other", "confidence": 0.8,
                                "source": REPAIR_SOURCE})
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")


def remember_verified_fixes(work_dir: Path, job_id: str, fixes: list[dict[str, Any]], after_check: Mapping[str, Any]) -> list[str]:
    """Put a repair in the shared memory only once the real render's check no longer finds that word swallowed."""
    from .pronunciation_memory import memory_path, record_job_outcomes, remember

    still = {re.sub(r"[^a-z]", "", w.casefold()) for unit in after_check.get("units") or [] for d in unit["dropped"] for w in d["words"]}
    saved: list[str] = []
    for fix in fixes:
        bare = re.sub(r"[^A-Za-z]", "", fix["word"])
        if fix.get("fixed") and bare.casefold() not in still:
            remember(memory_path(work_dir), bare, fix["respelling"], source="repair", voice=fix.get("voice"), job_id=job_id,
                     note=f"swallowed in unit {fix.get('unit')}; verified by the post-render check")
            saved.append(bare)
    record_job_outcomes(work_dir / "jobs" / job_id, after_check)
    return saved


def repair_run(settings: Any, *, job_id: str, run_id: str, clip_id: str | None = None, hook: bool = True,
               progress: Callable[[str], None] = print) -> dict[str, Any]:
    import os

    from .azure_stt import normalize
    from .cli import create_speech_backend
    from .experience_store import ExperienceStore

    clip_id = clip_id or job_id[:8]
    store = ExperienceStore(settings.work_dir)
    run_dir = store.repeat_dir(run_id, clip_id, 1)
    backend = create_speech_backend(settings)
    key, region = os.environ["AZURE_SPEECH_KEY"], os.environ["AZURE_SPEECH_REGION"].strip().lower()
    before = check_run(run_dir, backend, normalize)
    report: dict[str, Any] = {"dropped_before": before["dropped_total"], "fixes": []}
    if not before["dropped_total"]:
        progress("[repair] nothing was swallowed; nothing to repair")
        return report
    scorecard = json.loads((run_dir / "scorecard.json").read_text(encoding="utf-8"))
    texts = {str(g["group_id"]): g.get("spoken_text", "") for g in scorecard.get("groups") or []}
    phrases = run_dir / "render" / "native_dub" / "audio" / "phrases"
    work = run_dir / "repair_work"
    transcribe = lambda wav: normalize(backend.transcribe(wav, locale="zu-ZA")).get("words") or []
    fixes: list[dict[str, Any]] = []
    for unit in before["units"]:
        ssml = (phrases / f"{unit['unit']}.azure_tts.ssml").read_text(encoding="utf-8")
        for item in unit["dropped"]:
            result = find_respelling(item, texts[unit["unit"]], ssml, transcribe,
                                     lambda text, path: synthesize_ssml(text, key, region, path), work / unit["unit"])
            result["unit"] = unit["unit"]
            named = re.search(r'<voice[^>]*\bname="([^"]+)"', ssml)
            result["voice"] = named.group(1) if named else None
            report["fixes"].append(result)
            progress(f"[repair] {unit['unit']} {result['word']!r}: " + (f"fixed with {result['respelling']!r}" if result["fixed"]
                                                                       else f"not fixed ({result.get('why')})"))
            if result["fixed"]:
                fixes.append(result)
    if fixes:
        add_to_lexicon(Path(settings.work_dir) / "jobs" / job_id / "native_dub" / "pronunciation_names.json", fixes)
        from .dub_render import render_submission

        report["render"] = render_submission(settings, run_id=run_id, clip_id=clip_id, repeat=1, hook=hook)
        if hook:
            from .dub_publish import publish_handoff

            report["handoff"] = publish_handoff(Path(settings.work_dir), job_id=job_id, run_id=run_id, clip_id=clip_id)
        after = check_run(run_dir, backend, normalize)
        report["dropped_after"] = after["dropped_total"]
        report["remaining"] = after["units"]
        report["remembered"] = remember_verified_fixes(Path(settings.work_dir), job_id, report["fixes"], after)
    (run_dir / "repair.json").write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    import argparse

    from .config import load_settings

    parser = argparse.ArgumentParser(prog="mathula_tv.dub_repair")
    parser.add_argument("--job", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--clip-id")
    parser.add_argument("--no-hook", action="store_true")
    args = parser.parse_args(argv)
    report = repair_run(load_settings(), job_id=args.job, run_id=args.run_id, clip_id=args.clip_id, hook=not args.no_hook)
    print(f"dropped before: {report['dropped_before']}, after: {report.get('dropped_after', 'not re-rendered')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
