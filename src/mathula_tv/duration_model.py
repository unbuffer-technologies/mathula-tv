"""A free estimate of how long a voice takes to say a Zulu text, learned from every duration the project has measured.

The agent measures candidate texts with Azure speech (a real synthesis, one measurement each).  Most candidates are far from their window, so a measurement
that was only going to say "20% too long" is spent for nothing.  This module predicts the duration from the text alone, per voice, and every real measurement
(the workbench's ``measure_tts`` / ``measure_tts_batch``, and the accepted dubs of earlier runs) goes into a ledger the model is refitted from, so every job
makes the next job's estimates better.

It never replaces a measurement: its error (about 3% typical, 10% at the 90th percentile when first fitted on 2,300 sentence groups) is larger than the
fit tolerance, so it only tells the agent which candidates are worth a real measurement.  The model is a ridge regression on counts of the text (vowels,
consonants, words, digits, capitals, commas, stops) and the silence of its ``~pause`` markers: numpy only, refitted in milliseconds.

Files, under the work directory: ``duration_measurements.jsonl`` (one row per measured text: voice, rate, text, measured_ms, source) and ``duration_model.json``
(per-voice coefficients with their cross-validated error).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

LEDGER = "duration_measurements.jsonl"
MODEL = "duration_model.json"
FEATURES = ("vowels", "consonants", "words", "digits", "capitals", "commas", "stops", "pause_ms", "bias")
MIN_ROWS_PER_VOICE = 30              # a voice with fewer rows borrows the pooled model
REFIT_AFTER_NEW_ROWS = 40            # the ledger grows by this many rows before the model is fitted again
RIDGE = 1.0
PAUSE = re.compile(r"~pause(\d+)")
MARKUP = re.compile(r"~|\[[^\]]*\]")
VOWELS = re.compile(r"[aeiouAEIOU]")
LETTERS = re.compile(r"[A-Za-z]")


def features(text: str) -> list[float]:
    """The counts the model reads from a text (``~pauseN`` is N ms of silence, counted apart from the speech)."""
    pause_ms = sum(int(n) for n in PAUSE.findall(text or ""))
    spoken = MARKUP.sub("", PAUSE.sub(" ", text or ""))
    vowels = len(VOWELS.findall(spoken))
    letters = len(LETTERS.findall(spoken))
    return [float(vowels), float(letters - vowels), float(len(re.findall(r"\w+", spoken, re.U))), float(sum(c.isdigit() for c in spoken)),
            float(sum(c.isupper() for c in spoken)), float(len(re.findall(r"[,;:]", spoken))), float(len(re.findall(r"[.!?]", spoken))), float(pause_ms), 1.0]


def _normal(rate_pct: float) -> float:
    """How much faster than the voice's natural speed a rate setting makes it (+10% is 1.10); a rate of -100% or less is read as natural."""
    return 1.0 + rate_pct / 100.0 if rate_pct > -90 else 1.0


def rate_of(voice_info: Any) -> float:
    """The speaking-rate setting (percent) of a voice assignment (``base_prosody.rate_percent``); 0 when it has none."""
    try:
        return float(((voice_info or {}).get("base_prosody") or {}).get("rate_percent") or 0.0)
    except (AttributeError, TypeError, ValueError):
        return 0.0


def record(work_dir: Path | str, *, voice: str, text: str, measured_ms: int | float, rate_pct: float = 0.0, source: str = "measure") -> None:
    """One real duration into the ledger.  Best effort: a ledger that cannot be written must never fail a measurement."""
    try:
        if not (voice and text and float(measured_ms) > 0):
            return
        path = Path(work_dir) / LEDGER
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"voice": voice, "rate_pct": rate_pct, "text": text, "measured_ms": int(measured_ms), "source": source}, ensure_ascii=False) + "\n")
    except (OSError, ValueError, TypeError):
        return


def _read_ledger(work_dir: Path) -> list[dict[str, Any]]:
    path = work_dir / LEDGER
    rows: dict[tuple[str, float, str], dict[str, Any]] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                row = json.loads(line)
                rows[(row["voice"], float(row.get("rate_pct") or 0.0), row["text"])] = row          # the same text measured again keeps its latest duration
            except (ValueError, KeyError, TypeError):
                continue
    return list(rows.values())


def _fit(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Ridge fit of the neutral-rate duration on the text counts, with a 5-fold cross-validated error (median and 90th-percentile absolute % error)."""
    if len(rows) < 8:
        return None
    x = np.array([features(r["text"]) for r in rows])
    y = np.array([float(r["measured_ms"]) * _normal(float(r.get("rate_pct") or 0.0)) for r in rows])

    def solve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        penalty = RIDGE * np.eye(a.shape[1])
        penalty[-1, -1] = 0.0                                                   # the bias is not shrunk
        return np.linalg.solve(a.T @ a + penalty, a.T @ b)

    order = np.random.RandomState(0).permutation(len(y))
    folds = np.array_split(order, 5)
    errors: list[float] = []
    for k in range(5):
        test = folds[k]
        train = np.concatenate([folds[j] for j in range(5) if j != k])
        if len(train) < 5 or not len(test):
            continue
        coefficients = solve(x[train], y[train])
        errors += list(np.abs((x[test] @ coefficients - y[test]) / y[test]) * 100.0)
    coefficients = solve(x, y)
    errors_array = np.array(errors) if errors else np.array([10.0])
    return {"coefficients": [round(float(c), 4) for c in coefficients], "rows": len(rows),
            "median_err_pct": round(float(np.median(errors_array)), 2), "p90_err_pct": round(float(np.percentile(errors_array, 90)), 2)}


def refit(work_dir: Path | str) -> dict[str, Any]:
    """Fit every voice with enough rows, plus a pooled model for the rest, from the ledger; write and return the model."""
    work = Path(work_dir)
    rows = _read_ledger(work)
    model: dict[str, Any] = {"schema": "duration-model-v1", "features": list(FEATURES), "ledger_rows": len(rows), "voices": {}}
    by_voice: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_voice.setdefault(row["voice"], []).append(row)
    for voice, group in by_voice.items():
        if len(group) >= MIN_ROWS_PER_VOICE and (fit := _fit(group)):
            model["voices"][voice] = fit
    if (pooled := _fit(rows)):
        model["pooled"] = pooled
    (work / MODEL).write_text(json.dumps(model, indent=1), encoding="utf-8")
    return model


def load(work_dir: Path | str, *, refit_if_stale: bool = True) -> dict[str, Any]:
    """The model, refitted first when the ledger has grown by REFIT_AFTER_NEW_ROWS rows since the last fit (or there is no model yet and a ledger exists)."""
    work = Path(work_dir)
    path = work / MODEL
    model: dict[str, Any] = {}
    if path.is_file():
        try:
            model = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            model = {}
    if refit_if_stale and (work / LEDGER).is_file():
        grown = sum(1 for _ in (work / LEDGER).open(encoding="utf-8", errors="ignore")) - int(model.get("ledger_lines", 0))
        if not model or grown >= REFIT_AFTER_NEW_ROWS:
            model = refit(work)
            model["ledger_lines"] = sum(1 for _ in (work / LEDGER).open(encoding="utf-8", errors="ignore"))
            path.write_text(json.dumps(model, indent=1), encoding="utf-8")
    return model


def predict(model: dict[str, Any], *, voice: str, text: str, rate_pct: float = 0.0) -> dict[str, Any] | None:
    """{predicted_ms, band_pct, basis}: the voice's own fit when it has one, else the pooled fit; None when nothing has been fitted yet.
    band_pct is the model's 90th-percentile error on texts it had not seen (a prediction is inside it nine times in ten)."""
    fit = (model.get("voices") or {}).get(voice)
    basis = voice
    if not fit:
        fit, basis = model.get("pooled"), "pooled"
    if not fit:
        return None
    ms = float(np.array(features(text)) @ np.array(fit["coefficients"])) / _normal(rate_pct)
    return {"predicted_ms": max(0, int(round(ms))), "band_pct": fit["p90_err_pct"], "typical_err_pct": fit["median_err_pct"], "basis": basis, "fitted_on": fit["rows"]}


def _assignments(voices: Any) -> dict[str, Any]:
    """speaker id -> voice assignment from a voice_resolution.json, whichever shape the file has (a dict under 'assignments', or a list of speakers)."""
    voices = voices if isinstance(voices, dict) else {}
    for key in ("assignments", "speakers"):
        found = voices.get(key)
        if isinstance(found, dict):
            return found
        if isinstance(found, list):
            return {str(item.get("speaker_id")): item for item in found if isinstance(item, dict) and item.get("speaker_id")}
    return {}


def bootstrap(work_dir: Path | str) -> int:
    """Seed the ledger from the accepted dubs already on disk: each scorecard group's spoken text and measured duration, with its speaker's voice from the clip's
    voice resolution.  Safe to repeat (the ledger keeps the latest row of a text).  Returns the rows added."""
    work = Path(work_dir)
    added = 0
    for card_path in sorted((work / "experience" / "runs").glob("*/*/r1/scorecard.json")):
        try:
            card = json.loads(card_path.read_text(encoding="utf-8"))
            clip = card_path.parts[-3]
            voices = None
            for candidate in ((work / "experience" / "corpus" / clip / "voice_resolution.json"),):
                if candidate.is_file():
                    voices = json.loads(candidate.read_text(encoding="utf-8"))
            if voices is None:
                job = next(iter((work / "jobs").glob(f"{clip}*")), None)
                if job and (job / "direct_dub" / "voice_resolution.json").is_file():
                    voices = json.loads((job / "direct_dub" / "voice_resolution.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        assignments = _assignments(voices)
        for group in card.get("groups", []):
            speaker = str(group.get("speaker_id") or "")
            info = assignments.get(speaker) if isinstance(assignments, dict) else None
            voice = (info or {}).get("selected_voice") if isinstance(info, dict) else None
            rate = rate_of(info)
            if voice and group.get("measured_ms") and group.get("spoken_text") and not group.get("collides_with"):
                record(work, voice=str(voice), text=str(group["spoken_text"]), measured_ms=int(group["measured_ms"]), rate_pct=rate, source="scorecard")
                added += 1
    return added


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Seed and refit the duration model: python -m mathula_tv.duration_model [--bootstrap] --work ~/mathula-tv/working")
    parser.add_argument("--work", default="working")
    parser.add_argument("--bootstrap", action="store_true", help="add the accepted dubs on disk to the ledger first")
    args = parser.parse_args()
    if args.bootstrap:
        print("rows added from scorecards:", bootstrap(args.work))
    model = refit(args.work)
    model["ledger_lines"] = sum(1 for _ in (Path(args.work) / LEDGER).open(encoding="utf-8", errors="ignore")) if (Path(args.work) / LEDGER).is_file() else 0
    (Path(args.work) / MODEL).write_text(json.dumps(model, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in model.items() if k != "features"}, indent=1)[:1500])


if __name__ == "__main__":
    main()
