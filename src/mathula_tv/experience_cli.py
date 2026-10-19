"""`mathula-tv experience ...` -- freeze clips, score runs, native-speaker review, compare.

Deliberately thin: runs themselves are made with the existing pipeline commands
or by an agent over the dub-workbench MCP server; this only freezes inputs,
scores outputs, and manages review labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .atomic_io import atomic_write_json, read_json
from .episode_scoring import score_native_dub_dir
from .experience_store import BASELINE_RUN_ID, ExperienceStore
from .review_sheet import (
    build_review_rows, read_review_sheet, scorer_calibration, write_review_sheet,
)

# A clip regresses when it gets clearly worse on a measured dimension or gains a gate failure.
DEAD_AIR_REGRESSION_PTS = 2.0
REVIEW_RATE_REGRESSION = 0.15


def add_parser(commands: Any) -> None:
    experience = commands.add_parser("experience", help="Clip corpus, scoring and native-speaker review")
    sub = experience.add_subparsers(dest="experience_command", required=True)

    freeze = sub.add_parser("freeze", help="Freeze a finished job into the corpus")
    freeze.add_argument("job_id")
    freeze.add_argument("--clip-id")
    freeze.add_argument("--notes", default="")
    freeze.add_argument("--overwrite", action="store_true")

    sub.add_parser("list", help="List frozen clips and scored runs")

    score = sub.add_parser("score", help="Score a run for one or all clips (default: the frozen baselines)")
    score.add_argument("--run", default=BASELINE_RUN_ID)
    score.add_argument("--clip")
    score.add_argument("--repeat", type=int, default=1)

    sheet = sub.add_parser("review-sheet", help="Write the CSV a native-speaker reviewer fills in")
    sheet.add_argument("--run", default=BASELINE_RUN_ID)
    sheet.add_argument("--clip", required=True)
    sheet.add_argument("--repeat", type=int, default=1)
    sheet.add_argument("--spot-check", type=int, default=4)

    ingest = sub.add_parser("ingest-review", help="Load a filled-in review sheet as labels")
    ingest.add_argument("sheet", type=Path)

    calibrate = sub.add_parser("calibrate", help="Scorer precision/recall against native-speaker labels")
    calibrate.add_argument("--run", default=BASELINE_RUN_ID)
    calibrate.add_argument("--clip")
    calibrate.add_argument("--repeat", type=int, default=1)

    compare = sub.add_parser("compare", help="Compare two scored runs clip by clip")
    compare.add_argument("baseline_run")
    compare.add_argument("candidate_run")


def _scorecard_path(store: ExperienceStore, run_id: str, clip_id: str, repeat: int) -> Path:
    return store.repeat_dir(run_id, clip_id, repeat) / "scorecard.json"


def _native_dir(store: ExperienceStore, run_id: str, clip_id: str, repeat: int) -> Path:
    if run_id == BASELINE_RUN_ID:
        return store.baseline_native_dub(clip_id)
    return store.repeat_dir(run_id, clip_id, repeat) / "native_dub"


def score_clip(store: ExperienceStore, run_id: str, clip_id: str, repeat: int = 1) -> dict[str, Any]:
    native = _native_dir(store, run_id, clip_id, repeat)
    card = score_native_dub_dir(native, labels=store.load_labels(clip_id))
    card.update({"run_id": run_id, "clip_id": clip_id, "repeat": repeat})
    atomic_write_json(_scorecard_path(store, run_id, clip_id, repeat), card)
    if not store.read_run(run_id):
        store.write_run(run_id, parameters={"approach": card["approach"]})
    store.append_episode({
        "run_id": run_id, "clip_id": clip_id, "repeat": repeat, "approach": card["approach"],
        "level": card["level"], **card["headline"],
    })
    return card


def load_scorecard(store: ExperienceStore, run_id: str, clip_id: str, repeat: int = 1) -> dict[str, Any]:
    path = _scorecard_path(store, run_id, clip_id, repeat)
    if not path.is_file():
        raise FileNotFoundError(f"no scorecard for run={run_id} clip={clip_id} repeat={repeat}; run `experience score`")
    return read_json(path)


def compare_scorecards(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    b, c = baseline["headline"], candidate["headline"]
    new_gate_failures = {
        name: sorted(set(candidate["gates"][name]) - set(baseline["gates"].get(name, [])))
        for name in candidate["gates"]
    }
    new_gate_failures = {k: v for k, v in new_gate_failures.items() if v}
    resolved = {
        name: sorted(set(baseline["gates"].get(name, [])) - set(candidate["gates"][name]))
        for name in baseline["gates"]
    }
    resolved = {k: v for k, v in resolved.items() if v}
    deltas = {
        key: round(c[key] - b[key], 3)
        for key in ("dead_air_pct", "worst_turn_shortfall_pct", "mean_rush_pct", "groups_rushed",
                    "review_rate", "gate_failures", "tokens_in", "tokens_out")
    }
    regressions = []
    if new_gate_failures:
        regressions.append("new gate failures")
    if deltas["dead_air_pct"] > DEAD_AIR_REGRESSION_PTS:
        regressions.append("more dead air")
    if deltas["groups_rushed"] > 0:
        regressions.append("more rushed units")
    if deltas["review_rate"] > REVIEW_RATE_REGRESSION:
        regressions.append("higher review rate")
    improved = (
        deltas["dead_air_pct"] < -DEAD_AIR_REGRESSION_PTS or deltas["gate_failures"] < 0 or bool(resolved)
    )
    verdict = "regression" if regressions else ("improvement" if improved else "neutral")
    return {
        "verdict": verdict, "regressions": regressions, "deltas": deltas,
        "new_gate_failures": new_gate_failures, "resolved_gate_failures": resolved,
    }


def run(args: argparse.Namespace, settings: Any) -> int:
    store = ExperienceStore(settings.work_dir)
    command = args.experience_command

    if command == "freeze":
        job_root = settings.work_dir / "jobs" / args.job_id
        print(json.dumps(
            store.freeze_clip(job_root, clip_id=args.clip_id, notes=args.notes, overwrite=args.overwrite),
            indent=2,
        ))
    elif command == "list":
        print(json.dumps({"clips": store.load_corpus()["clips"], "episodes": len(store.read_episodes())}, indent=2))
    elif command == "score":
        clips = [args.clip] if args.clip else store.clip_ids()
        for clip_id in clips:
            card = score_clip(store, args.run, clip_id, args.repeat)
            print(json.dumps({"clip_id": clip_id, "level": card["level"], **card["headline"], "gates": card["gates"]}, indent=2))
    elif command == "review-sheet":
        card = load_scorecard(store, args.run, args.clip, args.repeat)
        rows = build_review_rows(args.clip, card, spot_check=args.spot_check)
        path = write_review_sheet(store.review_dir() / f"{args.clip}__{args.run}__r{args.repeat}.csv", rows)
        print(f"{len(rows)} units to review ({sum(1 for r in rows if r['why_flagged'] != 'spot_check (nothing flagged)')} flagged) -> {path}")
    elif command == "ingest-review":
        labels, problems = read_review_sheet(args.sheet)
        for problem in problems:
            print("PROBLEM:", problem)
        by_clip: dict[str, list[dict[str, Any]]] = {}
        for label in labels:
            by_clip.setdefault(label["clip_id"], []).append(label)
        for clip_id, clip_labels in by_clip.items():
            store.append_labels(clip_id, clip_labels)
        print(f"ingested {len(labels)} labels across {len(by_clip)} clip(s); {len(problems)} problem row(s) skipped")
        return 1 if problems and not labels else 0
    elif command == "calibrate":
        clips = [args.clip] if args.clip else store.clip_ids()
        for clip_id in clips:
            card = load_scorecard(store, args.run, clip_id, args.repeat)
            print(clip_id, json.dumps(scorer_calibration(card, store.load_labels(clip_id)), indent=2))
    elif command == "compare":
        overall = "neutral"
        for clip_id in store.clip_ids():
            try:
                baseline = load_scorecard(store, args.baseline_run, clip_id)
                candidate = load_scorecard(store, args.candidate_run, clip_id)
            except FileNotFoundError:
                continue
            result = compare_scorecards(baseline, candidate)
            print(clip_id, json.dumps(result, indent=2))
            if result["verdict"] == "regression":
                overall = "regression"
            elif result["verdict"] == "improvement" and overall != "regression":
                overall = "improvement"
        print("OVERALL:", overall)
        return 1 if overall == "regression" else 0
    return 0
