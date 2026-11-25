"""Start a run of a job from the best submission any earlier run of that job made, instead of from nothing.

A fresh run of a job the system has already dubbed used to throw that work away: the agent drafted the whole clip again and spent most of its turns finding its way back to timings an earlier run
had already reached.  This module picks the earlier submission to start from and hands it to the repair path (``dub_production`` ``repair_from`` -> ``seed_items``): the units are re-scored under
today's voice and rules, and the agent changes only what misses the bar.  The text the earlier run settled on is kept; only what the new voice or rules make wrong is worked on.

Which submission: runs made with the same voice first (a different voice has a different pace, so its timing says little), then the scorer's own order (``submission_rank.score_key``: hard gates,
cross-speaker collisions, collisions, overruns, dead air).  A run that ended below its own best has that best beside its trace (``best_submission.json``); that one is used.

Every run records which voice made it (``voice_provider`` in its agent_summary.json).  Runs that predate the tag are read as: a Gemini agent run used OmniVoice (the Google stack went in as one,
Azure answering 401), anything else Azure; ``python -m mathula_tv.warm_start backfill`` writes the tag into those runs so the guess is made once and can be read back.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .submission_rank import score_key


def current_voice_provider() -> str:
    """The voice this process dubs with (MATHULA_TTS_PROVIDER; Azure when unset)."""
    return (os.getenv("MATHULA_TTS_PROVIDER") or "azure").strip().lower()


def infer_voice_provider(summary: Mapping[str, Any] | None) -> str:
    """The voice of a run: its recorded tag, else the reading of the Gemini-agent era described above."""
    summary = summary or {}
    tag = str(summary.get("voice_provider") or "").strip().lower()
    if tag:
        return tag
    return "omnivoice" if "gemini" in str(summary.get("model") or "").lower() else "azure"


@dataclass(frozen=True)
class WarmStart:
    run_id: str
    clip_id: str
    directory: Path
    items_path: Path
    texts_path: Path | None          # where the English each unit was written against is read from (a scorecard, or the best submission's own record)
    key: tuple[float, ...]
    voice_provider: str
    same_voice: bool
    from_best: bool                  # the run's best submission, which is not its final one

    def load_items(self) -> list[dict[str, Any]]:
        return list(json.loads(self.items_path.read_text(encoding="utf-8"))["items"])

    def unit_texts(self) -> dict[str, str] | None:
        """The English each unit of the submission was written against (None when it was not kept)."""
        if self.texts_path is None:
            return None
        try:
            data = json.loads(self.texts_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if self.from_best:
            texts = data.get("unit_texts")
            return {str(k): str(v) for k, v in texts.items()} if isinstance(texts, dict) else None
        return {str(g["group_id"]): str(g.get("source_text") or "") for g in data.get("groups") or [] if "group_id" in g}

    def describe(self) -> str:
        key = ", ".join(f"{v:g}" for v in self.key)
        return (f"{self.run_id} ({'its best submission, not its last' if self.from_best else 'its last submission'}; {self.voice_provider} voice"
                f"{'' if self.same_voice else ', not the current voice'}; gates/cross-speaker/collisions/overruns/dead-air% = {key})")


def _read(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def candidates(store: Any, clip_id: str, voice_provider: str, *, exclude: tuple[str, ...] = ()) -> list[WarmStart]:
    """Every earlier run of the clip that left a scored submission, best first."""
    runs = Path(store.root) / "runs"
    found: list[WarmStart] = []
    if not runs.is_dir():
        return found
    for run_dir in sorted(runs.iterdir()):
        if run_dir.name in exclude:
            continue
        directory = run_dir / clip_id / "r1"
        card, final = _read(directory / "scorecard.json"), directory / "submission.json"
        if not isinstance(card, dict) or not final.is_file():
            continue
        summary = _read(directory / "agent_summary.json")
        best_path = directory / "best_submission.json"
        best = _read(best_path) if best_path.is_file() else None
        if isinstance(best, dict) and best.get("items") and best.get("headline"):
            key, items_path, texts_path, from_best = score_key({"headline": best["headline"]}), best_path, best_path, True
        else:
            key, items_path, texts_path, from_best = score_key(card), final, directory / "scorecard.json", False
        provider = infer_voice_provider(summary)
        found.append(WarmStart(run_dir.name, clip_id, directory, items_path, texts_path, key, provider, provider == voice_provider, from_best))
    return sorted(found, key=lambda c: (not c.same_voice, c.key, c.run_id))


def choose(store: Any, clip_id: str, *, voice_provider: str | None = None, exclude: tuple[str, ...] = ()) -> WarmStart | None:
    """The submission a new run of the clip starts from, or None when the clip has never been dubbed."""
    ranked = candidates(store, clip_id, voice_provider or current_voice_provider(), exclude=exclude)
    return ranked[0] if ranked else None


def from_run(store: Any, run_id: str, clip_id: str, *, voice_provider: str | None = None) -> WarmStart:
    """The submission of one named run (``--repair-from``), read the same way."""
    for c in candidates(store, clip_id, voice_provider or current_voice_provider()):
        if c.run_id == run_id:
            return c
    raise ValueError(f"run {run_id!r} has no scored submission for clip {clip_id!r}")


def backfill(store: Any, *, write: bool = True) -> list[tuple[str, str]]:
    """Write the voice tag into every run that lacks one; returns (run id, tag) for each run changed."""
    changed: list[tuple[str, str]] = []
    runs = Path(store.root) / "runs"
    for path in sorted(runs.glob("*/*/r*/agent_summary.json")) if runs.is_dir() else []:
        summary = _read(path)
        if not isinstance(summary, dict) or summary.get("voice_provider"):
            continue
        tag = infer_voice_provider(summary)
        changed.append((f"{path.parts[-4]}/{path.parts[-3]}/{path.parts[-2]}", tag))
        if write:
            summary["voice_provider"] = tag
            summary["voice_provider_inferred"] = True
            path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mathula_tv.warm_start")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("backfill").add_argument("--dry-run", action="store_true")
    show = sub.add_parser("show", help="what a new run of a clip would start from")
    show.add_argument("--clip", required=True)
    show.add_argument("--voice", default=None)
    args = parser.parse_args(argv)
    from .config import load_settings
    from .experience_store import ExperienceStore

    store = ExperienceStore(load_settings().work_dir)
    if args.command == "backfill":
        rows = backfill(store, write=not args.dry_run)
        for name, tag in rows:
            print(f"{name}: {tag}")
        print(f"{len(rows)} run(s) {'would be ' if args.dry_run else ''}tagged")
        return 0
    voice = (args.voice or current_voice_provider()).lower()
    for index, c in enumerate(candidates(store, args.clip, voice)[:12]):
        print(("-> " if index == 0 else "   ") + c.describe())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
