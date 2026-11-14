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
from typing import Any, Callable, Mapping

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
    from .cli import create_native_gpt_backend
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
    # Pronunciation is the dictionary's sounds through the rules (job_pronunciations): the web research here only finds the sounds of names. It is not
    # checked by speaking and listening, and no respelling is guessed from the voice (no tts, no speech-to-text, no model for self-supervised guesses).
    _ensure_native_pronunciation_research(
        job_root=job_root, context_ledger=ledger, glossary=glossary, progress=progress, tts=None,
        stt_backend=None, skip_round_trip_verification=True, grok_provider=None,
        pass1_corrections=pass1.get("corrections") or ())
    _ensure_speaker_seriousness_modes(job_root=job_root, provider=provider, segments=segments, progress=progress)
    job = Orchestrator(settings).jobs.load(job_id)
    voices = ensure_native_voice_assignments(job=job, job_root=job_root, segments=segments, context_ledger=ledger,
                                             provider=provider, progress=progress)
    try:        # words the English transcript got wrong, checked against YouTube's captions of the same moment (see source_suspects): no recogniser runs here
        from .source_suspects import run as check_source

        check_source(job_root, provider, progress=progress)
    except Exception as exc:
        progress(f"[prepare] source check skipped: {type(exc).__name__}: {str(exc)[:200]}")
    try:        # how each speaker delivers, measured from the audio (see delivery): the agent writes toward it, the renderer nudges the voice
        from .delivery import ensure_sheet

        ensure_sheet(job_root, progress=progress)
    except Exception as exc:
        progress(f"[prepare] delivery sheet skipped: {type(exc).__name__}: {str(exc)[:200]}")
    try:        # speakers who share a voice differ by pitch: take each speaker's from the sheet where the voice resolution has none (see voice_personas)
        import shutil

        from .delivery import SHEET_NAME, load_sheet
        from .experience_store import ExperienceStore
        from .voice_personas import refresh

        sheet = load_sheet(job_root / "native_dub" / SHEET_NAME)
        changed = refresh(Path(voices), sheet) if sheet else None
        if changed:
            progress("[prepare] voice personas set from the speakers' measured pitch: "
                     + ", ".join(f"{s} {c['pitch_percent_before']:+d}% -> {c['pitch_percent_after']:+d}%" for s, c in changed.items()))
            frozen = ExperienceStore(settings.work_dir).clip_dir(job_id[:8]) / "voice_resolution.json"
            if frozen.is_file():                     # the clip keeps its own copy of the voices; it follows the job's
                shutil.copyfile(Path(voices), frozen)
    except Exception as exc:
        progress(f"[prepare] voice personas not refreshed: {type(exc).__name__}: {str(exc)[:200]}")
    free_memory(progress)
    return {"job_id": job_id, "segments": len(segments), "voice_resolution": str(voices),
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


def _pass1_changed(existing: dict[str, Any], native: Path) -> bool:
    """Whether the job's Pass 1 is no longer the one the clip was frozen from.  A clip frozen with its checksum is compared by it; one frozen before checksums were
    kept is stale when the file was written after the clip was frozen.  (A new Pass 1 can give a byte-identical glossary, so the glossary alone cannot tell.)"""
    from datetime import datetime

    from .experience_store import pass1_sha

    path = Path(native) / "pass1_restored_transcript.json"
    if not path.is_file():
        return False
    if existing.get("pass1_sha"):
        return existing["pass1_sha"] != pass1_sha(native)
    try:
        return path.stat().st_mtime > datetime.fromisoformat(str(existing.get("frozen_at"))).timestamp()
    except (TypeError, ValueError):
        return False


def rss_mb() -> int:
    """This process's resident memory in MB (0 where /proc is not there)."""
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


def free_memory(progress: Callable[[str], None] = print) -> None:
    """Let go of what the preparation loaded (the speaker voice-family model, speechbrain's embeddings, torch's scratch) before the agent's long run: the agent needs none
    of it, and this machine has 6 GB.  The models are loaded again if a later step asks."""
    import gc

    before = rss_mb()
    try:
        from . import voice_family_classifier

        voice_family_classifier._model_cache = None
    except Exception:
        pass
    gc.collect()
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)              # hand the freed pages back to the system
    except Exception:
        pass
    progress(f"[mem] preparation done, models released: {before} MB -> {rss_mb()} MB resident")


def ensure_frozen(store: Any, job_root: Path, clip_id: str) -> dict[str, Any]:
    """Freeze the job's Pass-1 inputs as a clip, once. A new clip has no baseline translation to compare with."""
    from .experience_store import glossary_sha, pass1_sha, source_corrections_sha

    native = Path(job_root) / "native_dub"
    frozen_native = store.clip_dir(clip_id) / "native_dub"
    existing = store.load_corpus()["clips"].get(clip_id)
    frozen_glossary = glossary_sha(frozen_native)          # of the file itself, so clips frozen before checksums were stored still compare
    if (existing and existing.get("source_corrections_sha") == source_corrections_sha(native)
            and frozen_glossary == glossary_sha(native) and not _pass1_changed(existing, native)):
        return {"clip_id": clip_id, **existing, "already_frozen": True}
    # a new clip, or one frozen before its transcript corrections or glossary existed or changed: (re)freeze it. The glossary
    # the clip was dubbed under is kept as previous_glossary.json so a repair run can find the units that use a term that changed.
    old_glossary = frozen_native / "zulu_terminology_glossary.json"
    previous = old_glossary.read_bytes() if existing and old_glossary.is_file() else None
    frozen = store.freeze_clip(job_root, clip_id=clip_id, notes="production", require_baseline=False, overwrite=bool(existing))
    if previous is not None and frozen_glossary != glossary_sha(native):
        (frozen_native / "previous_glossary.json").write_bytes(previous)
    return frozen


def run_production(
    settings: Any, client: Any, *, job_id: str, clip_id: str | None = None, run_id: str | None = None,
    playbook_version: str | None = None, render: bool = True, reflect_after: bool = True, judge_meaning: bool = True,
    max_turns: int | None = None, max_total_tokens: int | None = None, tts_budget: int | None = None, repair_from: str | None = None,
    hook: bool = False, audio_only: bool = False, mode: str | None = None, compact: bool = True, budget: str | None = None, resume: bool = False,
) -> dict[str, Any]:
    from .cli import create_tts_backend
    from .dub_agent import BUDGET_LIMITS, ChatClient, agent_budget, agent_mode, build_system_prompt, run_agent
    from .dub_judge import UnitJudge
    from .dub_workbench import Workbench
    from .experience_store import ExperienceStore

    mode = agent_mode(mode)
    budget = agent_budget(budget)
    limits = BUDGET_LIMITS[budget]
    max_turns = max_turns or limits["max_turns"]
    max_total_tokens = max_total_tokens or limits["max_total_tokens"]
    tts_budget = tts_budget or limits["tts_budget"]
    print(f"[budget] {budget}: up to {max_turns} turns, {max_total_tokens:,} tokens, {tts_budget} measurements" + (" (quality first: the bar is a minimum, the agent decides when it is done)" if budget == "free" else ""))
    print(f"[mode] {mode}" + (" (the agent records its choice points; reasoning summaries are kept)" if mode == "training" else "") + ("; history is not compacted" if not compact else ""))
    store = ExperienceStore(settings.work_dir)
    job_root = Path(settings.work_dir) / "jobs" / job_id
    clip_id = clip_id or job_id[:8]
    run_id = run_id or f"prod-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    frozen = ensure_frozen(store, job_root, clip_id)
    playbook = load_playbook(store, playbook_version) if playbook_version else current_playbook(store)

    seed_items = None
    if resume:
        print(f"[resume] continuing run {run_id} from its checkpoint")
    elif repair_from:
        seed_items = json.loads((store.repeat_dir(repair_from, clip_id, 1) / "submission.json").read_text(encoding="utf-8"))["items"]
        try:        # an earlier run may have typed a chosen spelling into a unit's text: the word goes back, the render respells it
            from .pronunciation_memory import load as load_memory, memory_path, restore_written_words

            seed_items, restored = restore_written_words(seed_items, list(load_memory(memory_path(Path(settings.work_dir)))["entries"].values()))
            for note in restored:
                print(f"[repair] {note}")
        except Exception as exc:
            print(f"[repair] respellings in the seeded text were not checked: {type(exc).__name__}")
    try:        # the stem analyser (ZulMorph) the decisions list is checked with: say up front whether this run can reach it
        from .zulu_stems import status as stem_status

        reached = stem_status()
        print(f"[stems] ZulMorph {'reached ' + str(reached['via']) if reached['available'] else 'NOT reachable: settled terms are compared by plain form only'}")
    except Exception as exc:        # noqa: BLE001 - never stops a run
        print(f"[stems] analyser check failed: {type(exc).__name__}")
    judge = UnitJudge(ChatClient(client.config)) if judge_meaning and hasattr(client, "config") else None
    workbench = Workbench(store, lambda: create_tts_backend(settings), judge=judge)
    if repair_from:        # the English each unit of the earlier submission was written against
        try:
            seed_dir = store.repeat_dir(repair_from, clip_id, 1)
            workbench.seed_unit_texts = {str(g["group_id"]): str(g.get("source_text") or "")
                                         for g in json.loads((seed_dir / "scorecard.json").read_text(encoding="utf-8")).get("groups") or []}
        except (OSError, ValueError, KeyError):
            workbench.seed_unit_texts = None
        try:        # the terms the earlier run settled (submit_dub decisions), so a repair does not word them again
            saved = store.repeat_dir(repair_from, clip_id, 1) / "decisions.json"
            if saved.is_file():
                workbench.seed_decisions = {str(e["english"]).casefold(): e for e in json.loads(saved.read_text(encoding="utf-8")).get("decisions") or [] if isinstance(e, dict) and e.get("english")}
        except (OSError, ValueError):
            pass
    summary = run_agent(
        client, workbench, clip_id=clip_id, run_id=run_id, repeat=1, max_turns=max_turns,
        max_total_tokens=max_total_tokens, tts_budget=tts_budget, system_prompt=build_system_prompt(playbook, mode=mode, budget=budget),
        playbook_version=playbook["version"], seed_items=seed_items, mode=mode, free_finish=(budget == "free"), resume=resume,
        dead_air_slack_pct=limits["dead_air_slack_pct"], compact_after_tokens=limits["compact_after_tokens"] if compact else 0)
    run_dir = store.repeat_dir(run_id, clip_id, 1)
    print(f"[mem] agent finished: {rss_mb()} MB resident")
    met_bar = summary.get("stop_reason") in ("agent_finished", "bar_met")
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
            from .job_pronunciations import build as build_job_pronunciations

            # How the English words are said is the dictionary's sounds through the rules, written for this run's text: nothing the agent or an earlier job chose
            submission = json.loads((run_dir / "submission.json").read_text(encoding="utf-8"))
            report["pronunciations"] = [{"word": s["word"], "spelling": s["spelling"], "reading": s["reading"]}
                                        for s in build_job_pronunciations(job_root, [str(i.get("zulu_text") or "") for i in submission.get("items", [])], progress=lambda _m: None)]

            report["render"] = render_submission(settings, run_id=run_id, clip_id=clip_id, repeat=1, hook=hook, audio_only=audio_only)
            if hook and not audio_only:
                from .dub_publish import publish_handoff

                report["handoff"] = publish_handoff(Path(settings.work_dir), job_id=job_id, run_id=run_id, clip_id=clip_id)
        except Exception as exc:        # a render problem must not lose the dub, the report or the lessons
            report["render_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"

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
    parser.add_argument("--audio-only", action="store_true",
                        help="synthesise, time and mix the dub audio and check it, but encode no video and make no hook (a quick check of a run; not the agent's training mode, see --mode)")
    parser.add_argument("--no-reflect", action="store_true")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=None, help="token ceiling (default from the budget policy)")
    parser.add_argument("--budget", choices=("free", "lean"), default=None,
                        help="free (default, or MATHULA_AGENT_BUDGET): quality first, no pressure to economise, generous limits, the bar is a minimum and the agent decides when it is done. lean: tight limits, the bar ends the run")
    parser.add_argument("--mode", choices=("live", "training"), default=None,
                        help="live (default, or MATHULA_AGENT_MODE): token-lean, what runs in production. training: the agent also records its choice points and unit strategies, the model's reasoning summaries are kept, and thinking_report.md is written beside the trace")
    parser.add_argument("--resume", action="store_true", help="continue the run --run-id from its checkpoint (every turn saves one): nothing is redone, the conversation is restored whole")
    parser.add_argument("--pause", action="store_true", help="ask the running --run-id to stop at the end of its current turn and save its checkpoint; resume it with --resume")
    parser.add_argument("--no-compact", action="store_true", help="keep the whole conversation in context (no compaction): costs many more input tokens; for training runs")
    args = parser.parse_args(argv)

    from .config import load_settings
    from .dub_agent import ResponsesClient, request_pause
    from .foundry_grok import native_gpt_config_from_environment

    settings = load_settings()
    if args.pause or args.resume:
        if not args.run_id:
            parser.error("--pause and --resume need --run-id")
    if args.pause:
        from .experience_store import ExperienceStore

        folder = ExperienceStore(settings.work_dir).repeat_dir(args.run_id, args.clip or args.job[:8], 1)
        print(f"pause requested: {request_pause(folder)}  (the run stops at the end of its current turn; resume with --resume)")
        return 0
    prepared = prepare_agent_inputs(settings, args.job)
    if args.prepare_only:
        print(json.dumps(prepared, indent=2))
        return 0
    client = ResponsesClient(native_gpt_config_from_environment(dict(os.environ)))
    report = run_production(
        settings, client, job_id=args.job, clip_id=args.clip, run_id=args.run_id, playbook_version=args.playbook,
        render=not args.no_render, reflect_after=not args.no_reflect, judge_meaning=not args.no_judge,
        max_total_tokens=args.max_tokens, repair_from=args.repair_from, hook=args.hook, audio_only=args.audio_only,
        mode=args.mode, compact=not args.no_compact, budget=args.budget, resume=args.resume)
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return 0 if report["met_bar"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
