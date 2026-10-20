"""The agent's self-improvement loop: reflect -> curate -> trial -> gate, and an unattended driver.

    batch of runs --(reflect)--> proposed edits --(curate)--> candidate playbook
        --(trial: 3 runs per clip)--> --(gate)--> adopt, or discard

Nothing is adopted on the strength of the reflector's say-so. A candidate playbook has to be run
the standard three times on each clip, and is adopted only if it is no worse than the batch it
was derived from: it must meet the bar at least as often, break no more timing than before, leave no
more dead air (beyond a small tolerance), and not cost materially more tokens. Every step, adopted
or not, is recorded in ``playbooks/history.jsonl`` with its evidence.

Problems in the harness itself (a tool that rejects reasonable input, a rule that freezes a bad draft)
are not strategy and are not "fixed" with lessons: ``dub_smells`` finds them deterministically and they are
reported to a human, and the reflector is told not to write lessons for them.

``autopilot`` repeats the step unattended, bounded by a token budget, a maximum number of rounds and a
patience rule (stop after N rounds without an adoption), so a person reads the history instead of the traces.

Run:
  python -m mathula_tv.dub_improve reflect   --baseline agent-fit --clip 01fa74d8
  python -m mathula_tv.dub_improve step      --baseline agent-fit --clip 01fa74d8 --trial-run pb-trial-1
  python -m mathula_tv.dub_improve autopilot --baseline ab-cand --clip 01fa74d8 --clip 57b20602 --run-prefix auto1
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Mapping, Sequence

from .dub_agent import ChatClient, build_system_prompt, run_agent
from .dub_judge import UnitJudge
from .dub_reflect import digest_run, reflect
from .dub_smells import find_smells, render as render_smells
from .dub_workbench import Workbench
from .playbook import (
    append_history, apply_edits, current_playbook, load_playbook, save_playbook, set_current,
)

TOKEN_GROWTH_TOLERANCE = 1.15      # a candidate may cost up to 15% more tokens than the baseline
DEAD_AIR_TOLERANCE_PTS = 0.3       # ... and leave up to 0.3 points more dead air when everything else ties
REPEATS = 3
MAX_PARALLEL_RUNS = 9


def _clips(clip_id: str | None, clip_ids: Sequence[str] | None) -> list[str]:
    chosen = list(clip_ids) if clip_ids else ([clip_id] if clip_id else [])
    if isinstance(chosen[0] if chosen else None, (list, tuple)):
        chosen = list(chosen[0])
    if not chosen:
        raise ValueError("at least one clip is required")
    return chosen


def collect_batch(store: Any, run_id: str, clip_id: str, *, repeats: int = 5) -> list[dict[str, Any]]:
    """Finished runs of one batch on one clip: outcome, final headline, token cost."""
    batch = []
    for repeat in range(1, repeats + 1):
        directory = store.repeat_dir(run_id, clip_id, repeat)
        summary_path, card_path = directory / "agent_summary.json", directory / "scorecard.json"
        if not (summary_path.is_file() and card_path.is_file()):
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        headline = json.loads(card_path.read_text(encoding="utf-8"))["headline"]
        tokens = summary.get("tokens") or {}
        batch.append({
            "clip": clip_id, "repeat": repeat, "dir": directory, "stop_reason": summary.get("stop_reason"),
            "met_bar": summary.get("stop_reason") == "agent_finished", "headline": headline,
            "processed_tokens": int(tokens.get("in") or 0) + int(tokens.get("out") or 0),
            "cached_tokens": int(tokens.get("cached") or 0), "tts_used": summary.get("tts_used"),
            "playbook": summary.get("playbook"),
        })
    return batch


def collect_batches(store: Any, run_id: str, clip_ids: Sequence[str]) -> list[dict[str, Any]]:
    return [row for clip in clip_ids for row in collect_batch(store, run_id, clip)]


def batch_stats(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        return {"n": 0}
    return {
        "n": len(batch),
        "met_bar": sum(1 for r in batch if r["met_bar"]),
        "tokens_mean": statistics.mean(r["processed_tokens"] for r in batch),
        "violations_mean": statistics.mean(
            float(r["headline"].get("groups_rushed", 0)) + float(r["headline"].get("collisions", 0)) for r in batch),
        "residual_mean": statistics.mean(float(r["headline"].get("residual_dead_air_pct", 0)) for r in batch),
        "gate_failures_mean": statistics.mean(float(r["headline"].get("gate_failures", 0)) for r in batch),
        "meaning_loss_mean": statistics.mean(float(r["headline"].get("meaning_loss_units", 0)) for r in batch),
    }


def gate(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> tuple[bool, list[str]]:
    """Adopt a candidate playbook only if it is no worse than the batch it came from."""
    reasons: list[str] = []
    if candidate.get("n", 0) < baseline.get("n", 0):
        reasons.append(f"candidate has only {candidate.get('n', 0)} finished runs (baseline {baseline.get('n')})")
    if candidate.get("met_bar", 0) < baseline.get("met_bar", 0):
        reasons.append(f"meets the bar less often ({candidate['met_bar']} vs {baseline['met_bar']})")
    if candidate.get("gate_failures_mean", 0) > baseline.get("gate_failures_mean", 0) + 1e-9:
        reasons.append("more hard-gate failures")
    if candidate.get("meaning_loss_mean", 0) > baseline.get("meaning_loss_mean", 0) + 1e-9:
        reasons.append(f"more units lose meaning ({candidate['meaning_loss_mean']:.2f} vs {baseline['meaning_loss_mean']:.2f})")
    if candidate.get("violations_mean", 0) > baseline.get("violations_mean", 0) + 1e-9:
        reasons.append(f"more overruns/collisions ({candidate['violations_mean']:.2f} vs {baseline['violations_mean']:.2f})")
    if candidate.get("met_bar", 0) <= baseline.get("met_bar", 0) and \
            candidate.get("residual_mean", 0) > baseline.get("residual_mean", 0) + DEAD_AIR_TOLERANCE_PTS:
        reasons.append(f"more dead air left ({candidate['residual_mean']:.2f}% vs {baseline['residual_mean']:.2f}%)")
    if candidate.get("tokens_mean", 0) > baseline.get("tokens_mean", 0) * TOKEN_GROWTH_TOLERANCE:
        reasons.append(f"costs materially more tokens ({candidate['tokens_mean']:,.0f} vs {baseline['tokens_mean']:,.0f})")
    return (not reasons), reasons


def run_batch(
    client: Any, store: Any, tts_factory: Callable[[], Any], *, run_id: str, playbook: Mapping[str, Any],
    clip_id: str | None = None, clip_ids: Sequence[str] | None = None, repeats: int = REPEATS,
    tts_budget: int = 600, max_turns: int = 80, max_total_tokens: int = 2_000_000, judge_meaning: bool = True,
) -> list[dict[str, Any]]:
    """Run the agent ``repeats`` times on each clip, in parallel, with one playbook, each in its own workbench."""
    clips = _clips(clip_id, clip_ids)
    shared_tts = tts_factory()
    system_prompt = build_system_prompt(playbook)

    def one(job: tuple[str, int]) -> dict[str, Any]:
        clip, repeat = job
        # each run gets its own judge (its own verdict cache), on the same model deployment as the agent
        judge = UnitJudge(ChatClient(client.config)) if judge_meaning and hasattr(client, "config") else None
        workbench = Workbench(store, lambda: shared_tts, judge=judge)
        return run_agent(
            client, workbench, clip_id=clip, run_id=run_id, repeat=repeat, max_turns=max_turns,
            max_total_tokens=max_total_tokens, tts_budget=tts_budget, system_prompt=system_prompt,
            playbook_version=playbook["version"],
        )

    jobs = [(clip, repeat) for clip in clips for repeat in range(1, repeats + 1)]
    with ThreadPoolExecutor(max_workers=min(len(jobs), MAX_PARALLEL_RUNS)) as pool:
        list(pool.map(one, jobs))
    return collect_batches(store, run_id, clips)


def reflect_on_batch(
    client: Any, store: Any, baseline_run: str, clip_id: Any, playbook: Mapping[str, Any],
) -> dict[str, Any]:
    """Digest the baseline batch, list the harness problems in it, ask the reflector, apply its edits."""
    clips = _clips(clip_id if isinstance(clip_id, str) else None, clip_id if not isinstance(clip_id, str) else None)
    batch = collect_batches(store, baseline_run, clips)
    if not batch:
        raise ValueError(f"no finished runs found for {baseline_run!r} on {clips!r}")
    multi = len(clips) > 1
    digests = {(f"{r['clip']}:r{r['repeat']}" if multi else f"r{r['repeat']}"): digest_run(r["dir"]) for r in batch}
    smells = find_smells(batch)
    note = "all runs use the same playbook; the bar is no overruns/collisions, no gate failures, dead air within limit"
    if multi:
        note += f"; runs cover {len(clips)} different clips, so a lesson must hold across clips"
    if smells:
        note += ("\nKNOWN HARNESS PROBLEMS (reported to a human; do NOT write lessons for these, and do not blame the "
                 "agent's strategy for them):\n" + render_smells(smells))
    reflection = reflect(client, playbook, digests, clip_note=note)
    candidate, applied, rejected = apply_edits(
        playbook, reflection["edits"], note=f"reflection on {baseline_run} ({len(batch)} runs)")
    return {"batch": batch, "digests": digests, "smells": smells, "reflection": reflection, "candidate": candidate,
            "applied": applied, "rejected": rejected}


def improvement_step(
    client: Any, store: Any, tts_factory: Callable[[], Any], *, baseline_run: str, trial_run: str,
    clip_id: str | None = None, clip_ids: Sequence[str] | None = None, repeats: int = REPEATS, run_trial: bool = True,
) -> dict[str, Any]:
    clips = _clips(clip_id, clip_ids)
    playbook = current_playbook(store)
    outcome = reflect_on_batch(client, store, baseline_run, clips, playbook)
    baseline = batch_stats(outcome["batch"])
    record: dict[str, Any] = {
        "baseline_run": baseline_run, "clips": clips, "from_playbook": playbook["version"],
        "diagnosis": outcome["reflection"].get("diagnosis"), "contrast": outcome["reflection"].get("contrast"),
        "harness_problems": outcome["smells"], "applied": outcome["applied"], "rejected": outcome["rejected"],
        "baseline_stats": baseline, "reflection_tokens": outcome["reflection"].get("usage"),
    }
    if not outcome["applied"]:
        record.update(decision="no_change", reasons=["the reflector proposed no valid edits"])
        append_history(store, record)
        return record
    candidate = outcome["candidate"]
    save_playbook(store, candidate)
    record["candidate_playbook"] = candidate["version"]
    if not run_trial:
        record.update(decision="candidate_saved_untested", reasons=[])
        append_history(store, record)
        return record
    trial = run_batch(client, store, tts_factory, clip_ids=clips, run_id=trial_run, playbook=candidate, repeats=repeats)
    stats = batch_stats(trial)
    adopt, reasons = gate(baseline, stats)
    record.update(candidate_stats=stats, trial_run=trial_run, decision="adopted" if adopt else "rejected", reasons=reasons)
    if adopt:
        set_current(store, candidate["version"], reason=f"passed trial {trial_run}")
    append_history(store, record)
    return record


def _cost(record: Mapping[str, Any]) -> int:
    """Processed tokens a step spent: the reflection plus every trial run."""
    usage = record.get("reflection_tokens") or {}
    reflection = int(usage.get("prompt_tokens") or 0) + int(usage.get("completion_tokens") or 0)
    stats = record.get("candidate_stats") or {}
    return reflection + int(stats.get("tokens_mean", 0) * stats.get("n", 0))


def autopilot(
    client: Any, store: Any, tts_factory: Callable[[], Any], *, clip_ids: Sequence[str], baseline_run: str,
    run_prefix: str, max_rounds: int = 3, patience: int = 2, token_budget: int = 6_000_000, repeats: int = REPEATS,
) -> dict[str, Any]:
    """Repeat reflect -> trial -> gate unattended, within a token budget.

    Stops when: the round limit is reached, ``patience`` consecutive rounds adopt nothing, or the next trial would
    not fit the remaining token budget (estimated from the baseline's cost). Returns every round's record.
    """
    rounds: list[dict[str, Any]] = []
    spent = 0
    misses = 0
    stop_reason = "max_rounds"
    for number in range(1, max_rounds + 1):
        baseline_stats = batch_stats(collect_batches(store, baseline_run, clip_ids))
        estimate = int(baseline_stats.get("tokens_mean", 0) * len(clip_ids) * repeats)
        if spent + estimate > token_budget:
            stop_reason = "token_budget"
            break
        record = improvement_step(
            client, store, tts_factory, baseline_run=baseline_run, trial_run=f"{run_prefix}-r{number}",
            clip_ids=clip_ids, repeats=repeats)
        spent += _cost(record)
        rounds.append({k: record.get(k) for k in (
            "decision", "reasons", "from_playbook", "candidate_playbook", "applied", "harness_problems",
            "baseline_stats", "candidate_stats")})
        if record["decision"] == "adopted":
            baseline_run = record["trial_run"]      # the adopted playbook's own trial is the next baseline
            misses = 0
        else:
            misses += 1
            if misses >= patience:
                stop_reason = "patience"
                break
    summary = {
        "clips": list(clip_ids), "rounds": rounds, "tokens_spent": spent, "stop_reason": stop_reason,
        "adopted": sum(1 for r in rounds if r["decision"] == "adopted"), "final_playbook": current_playbook(store)["version"],
    }
    append_history(store, {"autopilot": {k: v for k, v in summary.items() if k != "rounds"}, "decisions": [r["decision"] for r in rounds]})
    return summary


def _print_record(record: Mapping[str, Any]) -> None:
    print(json.dumps({k: v for k, v in record.items() if k not in ("batch", "digests")}, indent=2, ensure_ascii=False, default=str))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mathula_tv.dub_improve")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("reflect", "step", "autopilot", "smells"):
        p = sub.add_parser(name)
        p.add_argument("--baseline", required=True, help="run id of the batch to reflect on")
        p.add_argument("--clip", action="append", required=True, help="repeat for several clips")
        if name == "step":
            p.add_argument("--trial-run", required=True)
        if name in ("step", "autopilot"):
            p.add_argument("--repeats", type=int, default=REPEATS)
        if name == "autopilot":
            p.add_argument("--run-prefix", required=True)
            p.add_argument("--max-rounds", type=int, default=3)
            p.add_argument("--patience", type=int, default=2)
            p.add_argument("--token-budget", type=int, default=6_000_000)
    args = parser.parse_args(argv)

    from .cli import create_tts_backend
    from .config import load_settings
    from .dub_agent import ResponsesClient
    from .experience_store import ExperienceStore
    from .foundry_grok import native_gpt_config_from_environment

    settings = load_settings()
    store = ExperienceStore(settings.work_dir)
    if args.command == "smells":
        print(render_smells(find_smells(collect_batches(store, args.baseline, args.clip))))
        return 0
    client = ResponsesClient(native_gpt_config_from_environment(dict(os.environ)))
    tts_factory = lambda: create_tts_backend(settings)           # noqa: E731
    if args.command == "reflect":
        playbook = current_playbook(store)
        outcome = reflect_on_batch(client, store, args.baseline, args.clip, playbook)
        if outcome["applied"]:
            save_playbook(store, outcome["candidate"])
        print(f"current playbook: {playbook['version']}")
        _print_record({"harness_problems": outcome["smells"], "diagnosis": outcome["reflection"].get("diagnosis"),
                       "contrast": outcome["reflection"].get("contrast"), "applied": outcome["applied"],
                       "rejected": outcome["rejected"],
                       "candidate_playbook": outcome["candidate"]["version"] if outcome["applied"] else None,
                       "reflection_tokens": outcome["reflection"].get("usage")})
    elif args.command == "step":
        _print_record(improvement_step(client, store, tts_factory, baseline_run=args.baseline, trial_run=args.trial_run,
                                       clip_ids=args.clip, repeats=args.repeats))
    else:
        _print_record(autopilot(client, store, tts_factory, clip_ids=args.clip, baseline_run=args.baseline,
                                run_prefix=args.run_prefix, max_rounds=args.max_rounds, patience=args.patience,
                                token_budget=args.token_budget, repeats=args.repeats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
