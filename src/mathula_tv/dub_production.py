"""Dub one real clip end to end, and learn from the run without touching the live playbook.

    job (through Pass 1) -> freeze inputs -> agent dubs it once (harness-verified bar, independent meaning check)
        -> plain dubbed video -> one cheap reflection on this single run -> lessons go to a QUEUE

This is the production counterpart of the experiment loop in ``dub_improve``. The experiment loop owns the
decision to change the playbook (repeat runs, a fair trial, a gate). A single production run cannot support that
decision, so here the reflector's output is only *hypotheses*: each is checked by the playbook linter, recorded
with its evidence in ``playbooks/queue.jsonl`` and left there. Nothing in this module edits the live playbook.

Every run leaves ``production_report.json`` in its run directory: what happened, whether it met the bar, what it
cost, where the video is, and which lessons were queued.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .atomic_io import atomic_write_json
from .dub_reflect import digest_run, reflect
from .playbook import apply_edits, current_playbook, load_playbook, playbook_dir

SINGLE_RUN_NOTE = (
    "This is ONE production run of a new clip, so there is no contrast between runs and no way to tell luck from "
    "strategy. Propose a note only if the digest shows a clear, general, avoidable cost (wasted turns, a repeated "
    "mistake, a rejection that better method would have prevented). Otherwise propose no edits. Your edits are "
    "hypotheses that will be tested on later clips, not applied now."
)


def queue_path(store: Any) -> Path:
    return playbook_dir(store) / "queue.jsonl"


def load_queue(store: Any) -> list[dict[str, Any]]:
    path = queue_path(store)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def queue_lessons(store: Any, *, run_id: str, clip_id: str, playbook: Mapping[str, Any],
                  reflection: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Validate the reflector's edits with the playbook linter and append the valid ones to the queue.

    The edits are applied to a throwaway copy only to learn which ones the curator would accept; the live playbook
    and its history are never written here.
    """
    _, applied, rejected = apply_edits(playbook, list(reflection.get("edits") or []), note=f"queued from {run_id}")
    entries = []
    for edit in applied:
        entries.append({
            "at": datetime.now(timezone.utc).isoformat(), "run": run_id, "clip": clip_id,
            "against_playbook": playbook["version"], "edit": edit, "status": "queued",
        })
    if entries:
        path = queue_path(store)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for entry in entries:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entries


def prepare_agent_inputs(settings: Any, job_id: str, *, progress: Any = print) -> dict[str, Any]:
    """Produce the inputs the agent and the renderer need that Pass 1 does not: glossary, pronunciation research,
    speaker prosody modes and the per-speaker voices.

    These come from the existing pipeline's own functions (same prompts, same checksum caches), so a clip prepared
    here is measured and rendered exactly as the pipeline would. None of them translates anything: the old
    candidate-pool translation stage is not run. Each step is cached; calling this twice costs nothing.
    """
    from .atomic_io import read_json
    from .cli import create_native_gpt_backend, create_speech_backend, create_tts_backend
    from .native_dub import (
        _ensure_native_pronunciation_research, _ensure_speaker_seriousness_modes, _ensure_zulu_glossary,
        _load_existing_context_ledger, ensure_native_voice_assignments, native_dub_paths,
    )
    from .orchestrator import Orchestrator

    job_root = Path(settings.work_dir) / "jobs" / job_id
    paths = native_dub_paths(job_root)
    if not paths.pass1.is_file():
        raise FileNotFoundError("run `native-translate --pass 1` first: pass1_restored_transcript.json is missing")
    pass1 = read_json(paths.pass1)
    segments = list(pass1.get("segments") or [])
    provider = create_native_gpt_backend(None, progress=progress)

    ledger, _ = _load_existing_context_ledger(job_root=job_root, pass1=pass1, segments=segments, progress=progress)
    if ledger is None:
        raise FileNotFoundError("the context ledger is missing; Pass 1 should have written it")
    glossary, _ = _ensure_zulu_glossary(job_root=job_root, provider=provider, pass1=pass1, segments=segments,
                                        context_ledger=ledger, progress=progress)
    try:
        stt = create_speech_backend(settings)
    except Exception as exc:             # the round-trip check is optional, as in the pipeline
        progress(f"[prepare] speech credentials unavailable ({exc}); pronunciation round-trip check skipped")
        stt = None
    _ensure_native_pronunciation_research(
        job_root=job_root, context_ledger=ledger, glossary=glossary, progress=progress, tts=create_tts_backend(settings),
        stt_backend=stt, skip_round_trip_verification=stt is None, grok_provider=provider,
        pass1_corrections=pass1.get("corrections") or ())
    _ensure_speaker_seriousness_modes(job_root=job_root, provider=provider, segments=segments, progress=progress)
    job = Orchestrator(settings).jobs.load(job_id)
    voices = ensure_native_voice_assignments(job=job, job_root=job_root, segments=segments, context_ledger=ledger,
                                             provider=provider, progress=progress)
    names = None
    try:
        from .pronunciation_audio import audio_names

        names = audio_names(job_root, voice=primary_voice(Path(voices)), progress=progress)
    except Exception as exc:        # a names-stage problem must never block the dub; the pipeline's own readings still apply
        progress(f"[prepare] names stage skipped: {type(exc).__name__}: {str(exc)[:200]}")
    try:        # words the English transcript got wrong, checked against the speaker's audio (see source_suspects)
        from .source_suspects import run as check_source

        check_source(job_root, provider, speech_backend=stt, progress=progress)
    except Exception as exc:
        progress(f"[prepare] source check skipped: {type(exc).__name__}: {str(exc)[:200]}")
    return {"job_id": job_id, "segments": len(segments), "voice_resolution": str(voices), "names_lexicon": str(names) if names else None,
            "glossary": paths.zulu_glossary.is_file(), "pronunciation_research": paths.pronunciation_research.is_file()}


def primary_voice(voice_resolution: Path, default: str = "zu-ZA-ThembaNeural") -> str:
    """The first speaker's assigned voice from a job's voice_resolution.json ({"assignments": {speaker: {selected_voice}}})."""
    import json

    try:
        data = json.loads(Path(voice_resolution).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default
    assignments = data.get("assignments", data) if isinstance(data, dict) else data
    entries = assignments.values() if isinstance(assignments, dict) else assignments
    for entry in entries:
        if isinstance(entry, dict) and entry.get("selected_voice"):
            return str(entry["selected_voice"])
    return default


def ensure_frozen(store: Any, job_root: Path, clip_id: str) -> dict[str, Any]:
    """Freeze the job's Pass-1 inputs as a clip, once. A new clip has no baseline translation to compare with."""
    from .experience_store import source_corrections_sha

    existing = store.load_corpus()["clips"].get(clip_id)
    if existing and existing.get("source_corrections_sha") == source_corrections_sha(Path(job_root) / "native_dub"):
        return {"clip_id": clip_id, **existing, "already_frozen": True}
    # a new clip, or one frozen before its transcript corrections existed or changed: (re)freeze it
    return store.freeze_clip(job_root, clip_id=clip_id, notes="production", require_baseline=False, overwrite=bool(existing))


def run_production(
    settings: Any, client: Any, *, job_id: str, clip_id: str | None = None, run_id: str | None = None,
    playbook_version: str | None = None, render: bool = True, reflect_after: bool = True, judge_meaning: bool = True,
    max_turns: int = 80, max_total_tokens: int = 2_000_000, tts_budget: int = 600, repair_from: str | None = None,
    hook: bool = False,
) -> dict[str, Any]:
    from .cli import create_tts_backend
    from .dub_agent import ChatClient, build_system_prompt, run_agent
    from .dub_judge import UnitJudge
    from .dub_workbench import Workbench
    from .experience_store import ExperienceStore

    store = ExperienceStore(settings.work_dir)
    job_root = Path(settings.work_dir) / "jobs" / job_id
    clip_id = clip_id or job_id[:8]
    run_id = run_id or f"prod-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    frozen = ensure_frozen(store, job_root, clip_id)
    playbook = load_playbook(store, playbook_version) if playbook_version else current_playbook(store)

    seed_items = None
    if repair_from:
        seed_items = json.loads((store.repeat_dir(repair_from, clip_id, 1) / "submission.json").read_text(encoding="utf-8"))["items"]
    judge = UnitJudge(ChatClient(client.config)) if judge_meaning and hasattr(client, "config") else None
    workbench = Workbench(store, lambda: create_tts_backend(settings), judge=judge)
    summary = run_agent(
        client, workbench, clip_id=clip_id, run_id=run_id, repeat=1, max_turns=max_turns,
        max_total_tokens=max_total_tokens, tts_budget=tts_budget, system_prompt=build_system_prompt(playbook),
        playbook_version=playbook["version"], seed_items=seed_items)
    run_dir = store.repeat_dir(run_id, clip_id, 1)
    met_bar = summary.get("stop_reason") == "agent_finished"
    report: dict[str, Any] = {
        "run_id": run_id, "clip_id": clip_id, "job_id": job_id, "playbook": playbook["version"],
        "stop_reason": summary.get("stop_reason"), "met_bar": met_bar, "unmet_at_end": summary.get("unmet_at_end"),
        "turns": summary.get("turns"), "tokens": summary.get("tokens"), "judge_tokens": summary.get("judge_tokens"),
        "clip": {k: frozen.get(k) for k in ("duration_s", "segment_count")},
        "headline": _headline(run_dir),
    }

    if render and (run_dir / "submission.json").is_file():
        try:
            from .dub_render import render_submission

            report["render"] = render_submission(settings, run_id=run_id, clip_id=clip_id, repeat=1, hook=hook)
            if hook:
                from .dub_publish import publish_handoff

                report["handoff"] = publish_handoff(Path(settings.work_dir), job_id=job_id, run_id=run_id, clip_id=clip_id)
        except Exception as exc:        # a render problem must not lose the dub, the report or the lessons
            report["render_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        if "render" in report:
            try:        # did the voice say every word? A report only: it never changes or blocks the dub
                from .azure_stt import normalize as normalize_stt
                from .cli import create_speech_backend
                from .dub_intelligibility import check_run

                report["intelligibility"] = check_run(run_dir, create_speech_backend(settings), normalize_stt)
                from .pronunciation_memory import record_job_outcomes

                report["intelligibility"]["memory_switched_off"] = record_job_outcomes(job_root, report["intelligibility"])
            except Exception as exc:
                report["intelligibility_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"

    if reflect_after and (run_dir / "trace.jsonl").is_file():
        try:
            reflection = reflect(client, playbook, {"r1": digest_run(run_dir)}, clip_note=SINGLE_RUN_NOTE)
            queued = queue_lessons(store, run_id=run_id, clip_id=clip_id, playbook=playbook, reflection=reflection)
            report["reflection"] = {"tokens": reflection.get("usage"), "diagnosis": reflection.get("diagnosis"),
                                    "queued": [q["edit"] for q in queued]}
        except Exception as exc:
            report["reflection_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"

    atomic_write_json(run_dir / "production_report.json", report)
    return report


def _headline(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "scorecard.json"
    if not path.is_file():
        return {}
    headline = json.loads(path.read_text(encoding="utf-8")).get("headline") or {}
    keep = ("group_count", "groups_rushed", "collisions", "cross_speaker_collisions", "gate_failures",
            "residual_dead_air_pct", "meaning_loss_units", "tts_measurements")
    return {k: headline[k] for k in keep if k in headline}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mathula_tv.dub_production")
    parser.add_argument("--job", required=True, help="job id (a job that has been through native-translate --pass 1)")
    parser.add_argument("--clip", default=None, help="clip id (default: first 8 characters of the job id)")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--playbook", default=None, help="playbook version (default: the adopted one)")
    parser.add_argument("--prepare-only", action="store_true",
                        help="make the glossary, pronunciation research, prosody modes and voices, then stop")
    parser.add_argument("--repair-from", default=None, help="run id whose accepted submission to repair instead of dubbing again")
    parser.add_argument("--hook", action="store_true",
                        help="render the hook edit and footer overlay and write the job-ID-named handoff files for posting")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--no-reflect", action="store_true")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=2_000_000)
    args = parser.parse_args(argv)

    from .config import load_settings
    from .dub_agent import ResponsesClient
    from .foundry_grok import native_gpt_config_from_environment

    settings = load_settings()
    prepared = prepare_agent_inputs(settings, args.job)
    if args.prepare_only:
        print(json.dumps(prepared, indent=2))
        return 0
    client = ResponsesClient(native_gpt_config_from_environment(dict(os.environ)))
    report = run_production(
        settings, client, job_id=args.job, clip_id=args.clip, run_id=args.run_id, playbook_version=args.playbook,
        render=not args.no_render, reflect_after=not args.no_reflect, judge_meaning=not args.no_judge,
        max_total_tokens=args.max_tokens, repair_from=args.repair_from, hook=args.hook)
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return 0 if report["met_bar"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
