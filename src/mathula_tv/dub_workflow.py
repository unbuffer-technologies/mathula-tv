"""Orchestrated dubbing: code decides the steps, small agents decide the wording.

``dub_agent.run_agent`` gives one agent the whole clip. Its context grows with every
rewrite (94% of its tokens were re-read history) and on a long clip it settles early.
This is the other shape from the "workflows vs agents" split:

  1. CODE splits the clip into speaker turns (the renderer's own ``_build_speaker_turns``).
  2. A small WORKER AGENT per turn (same tool loop, same tools, same bar) dubs just that
     turn. It still sees the whole clip's English, ledger and glossary as context, so it
     retells with the full story in mind, but its working history is only its own turn.
     Workers run in parallel.
  3. CODE assembles every worker's units into one submission and scores it with the same
     scorer as everything else.
  4. CODE re-dispatches only the turns that miss the bar, with their previous draft and the
     specific problems, for at most ``max_rounds - 1`` extra rounds. The best-scoring
     assembly wins (gates, then overruns, then dead air left).

Nothing here dictates how a worker produces its wording; the method inside a turn is the
agent's. What the code owns is decomposition, checking and the ceilings.

Run:  python -m mathula_tv.dub_workflow --clip 01fa74d8 --run-id wf-v1 --repeat 1
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from .dub_agent import (
    DEFAULT_MAX_PUSHBACKS, DEFAULT_MAX_RESIDUAL_DEAD_AIR_PCT, SYSTEM_PROMPT, Trace, run_agent, unmet_targets,
)
from .dub_workbench import Workbench

WORKER_SCOPE_PROMPT = """

WORKER SCOPE
You are one worker in a team dubbing the same clip. You are responsible for ONE speaker turn only:
the segments marked yours:true when you open the clip. Everything else in the clip is context so your
retelling fits the whole story and does not repeat or contradict the neighbouring turns. Other workers
dub those turns at the same time.
- submit_dub covers only your segments; every one of them must be covered. Use short item_ids (u1, u2...).
- Your units may split or merge your segments however you like, but keep them inside your turn's window.
- measure_tts takes next_start_ms: use the one given in your scope note for your last unit.
- The bar applies to your turn: no overruns, no gate failures, residual dead air within the limit above.
"""


def plan_turns(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Split the clip into the speaker turns the renderer itself uses."""
    from .native_dub import _build_speaker_turns

    groups = [
        {"group_id": str(s["segment_id"]), "speaker_id": str(s.get("speaker_id") or "SPEAKER_00"),
         "start_ms": int(s["start_ms"]), "source_end_ms": int(s["end_ms"])}
        for s in segments
    ]
    turns = _build_speaker_turns(groups)
    speakers = {g["group_id"]: g["speaker_id"] for g in groups}
    plan = []
    for index, turn in enumerate(turns):
        ids = [str(i) for i in turn["member_group_ids"]]
        plan.append({
            "turn_id": f"t{index + 1:02d}", "segment_ids": ids, "speaker_id": speakers[ids[0]],
            "start_ms": int(turn["start_ms"]), "end_ms": int(turn["source_end_ms"]),
            "next_start_ms": int(turns[index + 1]["start_ms"]) if index + 1 < len(turns) else None,
        })
    return plan


def score_key(result: dict[str, Any] | None) -> tuple[float, float, float]:
    """Lower is better: hard gates first, then overruns, then dead air that is left."""
    headline = (result or {}).get("headline") or {}
    return (
        float(headline.get("gate_failures", 1e9)), float(headline.get("cross_speaker_collisions", 0)),
        float(headline.get("collisions", 0)), float(headline.get("groups_rushed", 1e9)),
        float(headline.get("residual_dead_air_pct", 1e9)),
    )


def _task_message(turn: dict[str, Any], draft: dict[str, Any] | None) -> str:
    text = (
        f"Dub speaker turn {turn['turn_id']}: segments {', '.join(turn['segment_ids'])} "
        f"({turn['speaker_id']}, {turn['start_ms']}-{turn['end_ms']} ms; next speech starts at "
        f"{turn['next_start_ms']} ms). Open the clip first; only segments marked yours:true are yours."
    )
    if draft:
        text += (
            "\n\nA previous attempt at this turn was scored and missed the bar. Improve it; keep what works.\n"
            f"Problems: {'; '.join(draft['issues'])}\n"
            f"Previous units: {json.dumps(draft['items'], ensure_ascii=False)}"
        )
    return text


def run_turn_worker(
    client: Any, parent: Workbench, turn: dict[str, Any], *, clip_id: str, run_id: str, repeat: int,
    round_number: int, tts_budget: int, shared_tts: Any, measure_root: Path, draft: dict[str, Any] | None,
    max_turns: int, max_tokens: int, max_dead_air_pct: float, max_pushbacks: int, speed_policy: str,
) -> dict[str, Any]:
    subdir = f"r{round_number}-{turn['turn_id']}"
    worker = Workbench(
        parent.store, lambda: shared_tts, subdir=subdir, measure_root=measure_root,
        scope={"segment_ids": turn["segment_ids"], "next_start_ms": turn["next_start_ms"]},
    )
    directory = parent.store.repeat_dir(run_id, clip_id, repeat) / "turns" / subdir
    summary = run_agent(
        client, worker, clip_id=clip_id, run_id=run_id, repeat=repeat, max_turns=max_turns,
        max_total_tokens=max_tokens, tts_budget=tts_budget, system_prompt=SYSTEM_PROMPT + WORKER_SCOPE_PROMPT,
        speed_policy=speed_policy, max_dead_air_pct=max_dead_air_pct, max_pushbacks=max_pushbacks,
        task_message=_task_message(turn, draft), trace_dir=directory,
    )
    unmet = unmet_targets(worker.last_submit, max_dead_air_pct=max_dead_air_pct) if worker.last_items else ["no submission"]
    return {"turn": turn, "summary": summary, "items": worker.last_items or [], "result": worker.last_submit, "unmet": unmet}


def assemble(workers: dict[str, dict[str, Any]], plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every worker's units in clip order, ids made unique per turn."""
    items: list[dict[str, Any]] = []
    for turn in plan:
        for item in workers[turn["turn_id"]]["items"]:
            items.append({**item, "item_id": f"{turn['turn_id']}-{item['item_id']}"})
    return sorted(items, key=lambda i: int(i["start_ms"]))


def _failing_turns(
    workers: dict[str, dict[str, Any]], result: dict[str, Any] | None, plan: list[dict[str, Any]],
) -> dict[str, list[str]]:
    """Turns that still miss the bar and why, from worker scores and the assembled clip's own flags."""
    failing = {tid: list(w["unmet"]) for tid, w in workers.items() if w["unmet"]}
    for flagged in (result or {}).get("flagged_units", []):
        tid = str(flagged["item_id"]).split("-", 1)[0]
        problems = [*flagged.get("gates", []), *[f for f in flagged.get("flags", []) if f in ("overrun", "dead_air")]]
        if problems and tid in workers:
            failing.setdefault(tid, []).append(f"unit {flagged['item_id']}: {', '.join(problems)}")
    return failing


def run_workflow(
    client: Any, parent: Workbench, *, clip_id: str, run_id: str, repeat: int = 1, tts_budget: int = 600,
    per_worker_tts: int | None = None, max_workers: int = 6, max_rounds: int = 2, worker_max_turns: int = 30,
    worker_max_tokens: int = 400_000, speed_policy: str = "natural",
    max_dead_air_pct: float = DEFAULT_MAX_RESIDUAL_DEAD_AIR_PCT, max_pushbacks: int = DEFAULT_MAX_PUSHBACKS,
) -> dict[str, Any]:
    started = time.monotonic()
    directory = parent.store.repeat_dir(run_id, clip_id, repeat)
    trace = Trace(directory / "workflow_trace.jsonl")
    opened = parent.open_clip(clip_id, run_id, repeat, tts_budget=tts_budget, speed_policy=speed_policy)
    plan = plan_turns(opened["segments"])
    per_worker = per_worker_tts or max(20, tts_budget // max(1, len(plan)))
    measure_root = directory / "measure"
    shared_tts = parent.tts
    trace.log("plan", turns=[{k: t[k] for k in ("turn_id", "segment_ids", "start_ms", "end_ms")} for t in plan],
              per_worker_tts=per_worker)

    workers: dict[str, dict[str, Any]] = {}
    best: dict[str, Any] | None = None
    best_items: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    totals = {"in": 0, "out": 0, "tts": 0, "model_calls": 0}
    pending = list(plan)
    drafts: dict[str, dict[str, Any]] = {}

    for round_number in range(1, max_rounds + 1):
        def work(turn: dict[str, Any]) -> dict[str, Any]:
            return run_turn_worker(
                client, parent, turn, clip_id=clip_id, run_id=run_id, repeat=repeat, round_number=round_number,
                tts_budget=per_worker, shared_tts=shared_tts, measure_root=measure_root,
                draft=drafts.get(turn["turn_id"]), max_turns=worker_max_turns, max_tokens=worker_max_tokens,
                max_dead_air_pct=max_dead_air_pct, max_pushbacks=max_pushbacks, speed_policy=speed_policy,
            )

        with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(pending)))) as pool:
            for outcome in pool.map(work, pending):
                tid = outcome["turn"]["turn_id"]
                previous = workers.get(tid)
                # A re-dispatched turn only replaces its earlier draft if it is no worse.
                if previous is None or score_key(outcome["result"]) <= score_key(previous["result"]) or not previous["items"]:
                    workers[tid] = outcome
                s = outcome["summary"]
                totals["in"] += s["tokens"]["in"]
                totals["out"] += s["tokens"]["out"]
                totals["tts"] += int(s.get("tts_used") or 0)
                totals["model_calls"] += int(s.get("turns") or 0)
                trace.log("worker_done", round=round_number, turn=tid, stop=s["stop_reason"], turns=s["turns"],
                          tts=s.get("tts_used"), pushbacks=s.get("pushbacks"), unmet=outcome["unmet"])

        missing = [t["turn_id"] for t in plan if not workers.get(t["turn_id"], {}).get("items")]
        if missing:
            trace.log("abort", reason="turns without a submission", turns=missing)
            break
        items = assemble(workers, plan)
        result = parent.submit_dub(items, approach_notes=f"workflow round {round_number}: per-turn workers")
        history.append({"round": round_number, "headline": (result or {}).get("headline"), "accepted": (result or {}).get("accepted")})
        trace.log("assembled", round=round_number, accepted=(result or {}).get("accepted"),
                  headline=(result or {}).get("headline"), errors=(result or {}).get("errors"))
        if result and result.get("accepted") and (best is None or score_key(result) < score_key(best)):
            best, best_items = result, items

        failing = _failing_turns(workers, result if result and result.get("accepted") else None, plan)
        if not failing or round_number == max_rounds:
            break
        pending = [t for t in plan if t["turn_id"] in failing]
        drafts = {
            tid: {"items": workers[tid]["items"], "issues": issues}
            for tid, issues in failing.items() if workers[tid]["items"]
        }
        trace.log("redispatch", round=round_number, turns=sorted(failing), issues=failing)

    final = parent.last_submit
    if best is not None and (final is None or score_key(final) > score_key(best)):
        final = parent.submit_dub(best_items, approach_notes="workflow: best round restored")
        trace.log("restored_best", headline=(final or {}).get("headline"))
    summary = {
        "mode": "workflow", "clip_id": clip_id, "run_id": run_id, "repeat": repeat, "turns_planned": len(plan),
        "rounds_run": len(history), "tokens": {"in": totals["in"], "out": totals["out"]},
        "tts_used": totals["tts"], "model_calls": totals["model_calls"],
        "wall_seconds": round(time.monotonic() - started, 1), "speed_policy": speed_policy,
        "final_headline": ((final or {}).get("headline")), "history": history,
        "met_bar": bool(final and final.get("accepted") and not unmet_targets(final, max_dead_air_pct=max_dead_air_pct)),
        "workers": {tid: {"stop": w["summary"]["stop_reason"], "turns": w["summary"]["turns"],
                          "tts": w["summary"].get("tts_used"), "unmet": w["unmet"]} for tid, w in workers.items()},
        "model": getattr(getattr(client, "config", None), "deployment", None),
    }
    trace.log("end", **{k: v for k, v in summary.items() if k != "workers"})
    (directory / "workflow_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mathula_tv.dub_workflow")
    parser.add_argument("--clip", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--tts-budget", type=int, default=600)
    parser.add_argument("--per-worker-tts", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--max-rounds", type=int, default=2)
    parser.add_argument("--speed-policy", choices=("natural", "pipeline"), default="natural")
    args = parser.parse_args(argv)

    from .cli import create_tts_backend
    from .config import load_settings
    from .dub_agent import ResponsesClient
    from .experience_store import ExperienceStore
    from .foundry_grok import native_gpt_config_from_environment

    settings = load_settings()
    parent = Workbench(ExperienceStore(settings.work_dir), lambda: create_tts_backend(settings))
    client = ResponsesClient(native_gpt_config_from_environment(dict(os.environ)))
    summary = run_workflow(
        client, parent, clip_id=args.clip, run_id=args.run_id, repeat=args.repeat, tts_budget=args.tts_budget,
        per_worker_tts=args.per_worker_tts, max_workers=args.max_workers, max_rounds=args.max_rounds,
        speed_policy=args.speed_policy,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "workers"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
