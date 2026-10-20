"""The agent's self-improvement loop: reflect -> curate -> trial -> gate.

    batch of runs --(reflect)--> proposed edits --(curate)--> candidate playbook
        --(trial: 3 runs)--> --(gate)--> adopt, or discard

Nothing is adopted on the strength of the reflector's say-so. A candidate playbook has to be run
the standard three times on the same clip, and is adopted only if it is no worse than the batch it
was derived from: it must meet the bar at least as often, break no more timing than before, leave no
more dead air (beyond a small tolerance), and not cost materially more tokens. Every step, adopted
or not, is recorded in ``playbooks/history.jsonl`` with its evidence.

Run:
  python -m mathula_tv.dub_improve reflect --baseline agent-fit --clip 01fa74d8
  python -m mathula_tv.dub_improve step    --baseline agent-fit --clip 01fa74d8 --trial-run pb-trial-1
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Mapping

from .dub_agent import build_system_prompt, run_agent
from .dub_reflect import digest_run, reflect
from .dub_workbench import Workbench
from .playbook import (
    append_history, apply_edits, current_playbook, load_playbook, save_playbook, set_current,
)

TOKEN_GROWTH_TOLERANCE = 1.15      # a candidate may cost up to 15% more tokens than the baseline
DEAD_AIR_TOLERANCE_PTS = 0.3       # ... and leave up to 0.3 points more dead air when everything else ties
REPEATS = 3


def collect_batch(store: Any, run_id: str, clip_id: str, *, repeats: int = 5) -> list[dict[str, Any]]:
    """Finished runs of one batch: outcome, final headline, token cost."""
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
            "repeat": repeat, "dir": directory, "stop_reason": summary.get("stop_reason"),
            "met_bar": summary.get("stop_reason") == "agent_finished", "headline": headline,
            "processed_tokens": int(tokens.get("in") or 0) + int(tokens.get("out") or 0),
            "cached_tokens": int(tokens.get("cached") or 0), "tts_used": summary.get("tts_used"),
            "playbook": summary.get("playbook"),
        })
    return batch


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
    if candidate.get("violations_mean", 0) > baseline.get("violations_mean", 0) + 1e-9:
        reasons.append(f"more overruns/collisions ({candidate['violations_mean']:.2f} vs {baseline['violations_mean']:.2f})")
    if candidate.get("met_bar", 0) <= baseline.get("met_bar", 0) and \
            candidate.get("residual_mean", 0) > baseline.get("residual_mean", 0) + DEAD_AIR_TOLERANCE_PTS:
        reasons.append(f"more dead air left ({candidate['residual_mean']:.2f}% vs {baseline['residual_mean']:.2f}%)")
    if candidate.get("tokens_mean", 0) > baseline.get("tokens_mean", 0) * TOKEN_GROWTH_TOLERANCE:
        reasons.append(f"costs materially more tokens ({candidate['tokens_mean']:,.0f} vs {baseline['tokens_mean']:,.0f})")
    return (not reasons), reasons


def run_batch(
    client: Any, store: Any, tts_factory: Callable[[], Any], *, clip_id: str, run_id: str, playbook: Mapping[str, Any],
    repeats: int = REPEATS, tts_budget: int = 600, max_turns: int = 80, max_total_tokens: int = 2_000_000,
) -> list[dict[str, Any]]:
    """Run the agent ``repeats`` times in parallel with one playbook, each in its own workbench."""
    shared_tts = tts_factory()
    system_prompt = build_system_prompt(playbook)

    def one(repeat: int) -> dict[str, Any]:
        workbench = Workbench(store, lambda: shared_tts)
        return run_agent(
            client, workbench, clip_id=clip_id, run_id=run_id, repeat=repeat, max_turns=max_turns,
            max_total_tokens=max_total_tokens, tts_budget=tts_budget, system_prompt=system_prompt,
            playbook_version=playbook["version"],
        )

    with ThreadPoolExecutor(max_workers=repeats) as pool:
        list(pool.map(one, range(1, repeats + 1)))
    return collect_batch(store, run_id, clip_id, repeats=repeats)


def reflect_on_batch(
    client: Any, store: Any, baseline_run: str, clip_id: str, playbook: Mapping[str, Any],
) -> dict[str, Any]:
    """Digest the baseline batch, ask the reflector, apply its edits. Returns everything for the record."""
    batch = collect_batch(store, baseline_run, clip_id)
    if not batch:
        raise ValueError(f"no finished runs found for {baseline_run!r} on {clip_id!r}")
    digests = {f"r{r['repeat']}": digest_run(r["dir"]) for r in batch}
    reflection = reflect(client, playbook, digests, clip_note="all runs: same clip, same playbook; the bar is no overruns/collisions, no gate failures, dead air within limit")
    candidate, applied, rejected = apply_edits(
        playbook, reflection["edits"], note=f"reflection on {baseline_run} ({len(batch)} runs)")
    return {"batch": batch, "digests": digests, "reflection": reflection, "candidate": candidate,
            "applied": applied, "rejected": rejected}


def improvement_step(
    client: Any, store: Any, tts_factory: Callable[[], Any], *, baseline_run: str, trial_run: str, clip_id: str,
    repeats: int = REPEATS, run_trial: bool = True,
) -> dict[str, Any]:
    playbook = current_playbook(store)
    outcome = reflect_on_batch(client, store, baseline_run, clip_id, playbook)
    baseline = batch_stats(outcome["batch"])
    record: dict[str, Any] = {
        "baseline_run": baseline_run, "clip": clip_id, "from_playbook": playbook["version"],
        "diagnosis": outcome["reflection"].get("diagnosis"), "contrast": outcome["reflection"].get("contrast"),
        "applied": outcome["applied"], "rejected": outcome["rejected"], "baseline_stats": baseline,
        "reflection_tokens": outcome["reflection"].get("usage"),
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
    trial = run_batch(client, store, tts_factory, clip_id=clip_id, run_id=trial_run, playbook=candidate, repeats=repeats)
    stats = batch_stats(trial)
    adopt, reasons = gate(baseline, stats)
    record.update(candidate_stats=stats, decision="adopted" if adopt else "rejected", reasons=reasons)
    if adopt:
        set_current(store, candidate["version"], reason=f"passed trial {trial_run}")
    append_history(store, record)
    return record


def _print_record(record: Mapping[str, Any]) -> None:
    print(json.dumps({k: v for k, v in record.items() if k not in ("batch", "digests")}, indent=2, ensure_ascii=False, default=str))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mathula_tv.dub_improve")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("reflect", "step"):
        p = sub.add_parser(name)
        p.add_argument("--baseline", required=True, help="run id of the batch to reflect on")
        p.add_argument("--clip", required=True)
        if name == "step":
            p.add_argument("--trial-run", required=True)
            p.add_argument("--repeats", type=int, default=REPEATS)
    args = parser.parse_args(argv)

    from .cli import create_tts_backend
    from .config import load_settings
    from .dub_agent import ResponsesClient
    from .experience_store import ExperienceStore
    from .foundry_grok import native_gpt_config_from_environment

    settings = load_settings()
    store = ExperienceStore(settings.work_dir)
    client = ResponsesClient(native_gpt_config_from_environment(dict(os.environ)))
    if args.command == "reflect":
        playbook = current_playbook(store)
        outcome = reflect_on_batch(client, store, args.baseline, args.clip, playbook)
        if outcome["applied"]:
            save_playbook(store, outcome["candidate"])
        print(f"current playbook: {playbook['version']}")
        for run, digest in outcome["digests"].items():
            print(f"--- digest {run} ({len(digest)} chars) ---\n{digest[:1500]}\n")
        _print_record({"diagnosis": outcome["reflection"].get("diagnosis"), "contrast": outcome["reflection"].get("contrast"),
                       "applied": outcome["applied"], "rejected": outcome["rejected"],
                       "candidate_playbook": outcome["candidate"]["version"] if outcome["applied"] else None,
                       "reflection_tokens": outcome["reflection"].get("usage")})
    else:
        record = improvement_step(
            client, store, lambda: create_tts_backend(settings), baseline_run=args.baseline,
            trial_run=args.trial_run, clip_id=args.clip, repeats=args.repeats)
        _print_record(record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
