"""Render an agent's dub as a video -- plain dubbed master, no hook edit, no footer overlay.

The agent hands back a *submission*: ordered units, each with the window it fills and its final isiZulu.
The production renderer (``native_dub.render_native_dub``) reads a *committed translation*
(``pass2_natural_translation.json``) in the "temporal-mask groups" shape that ``commit_candidate_pool``
writes. This module is the adapter between the two, and nothing more: it changes no wording and no window.

Fidelity to what was scored:
  * every unit becomes one committed group with the agent's own window and text;
  * the render synthesises each group on its own (no turn-level or island regrouping), the same way
    ``dub_fit`` measured it, so the audio matches the measurement;
  * the hook edit and the footer overlay are both off: the output is the untouched source video with
    the dubbed audio -- no title cards, no opening cut, no editorial model call.

The render happens in its own directory under the run (``.../render``); the original job is only read.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

from .atomic_io import atomic_write_json, read_json

COMMIT_SCHEMA = "mathula-agent-submission-commit-v1"
_FROZEN_INPUTS = ("pass1_restored_transcript.json", "context_ledger.json", "zulu_terminology_glossary.json")
_OPTIONAL_JOB_INPUTS = ("speaker_seriousness_modes.json", "pronunciation_research.json")


class RenderAdapterError(ValueError):
    """The submission cannot be represented as a committed translation (the message says why)."""


def _overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def assign_sentences(
    units: Sequence[Mapping[str, Any]], sentences: Sequence[Mapping[str, Any]],
) -> dict[str, list[str]]:
    """Which sentence records (the renderer's atomic unit) each of the agent's units owns.

    A sentence belongs to the unit that lists its parent segment; when several units list it (a segment the
    agent split), to the one whose window overlaps it most. Every unit must own at least one sentence and the
    units, read in time order, must reproduce the sentences in their original order -- the renderer insists.
    """
    ordered = sorted(units, key=lambda u: (int(u["start_ms"]), int(u["end_ms"])))
    owned: dict[str, list[str]] = {str(u["item_id"]): [] for u in ordered}
    for sentence in sentences:
        parent = str(sentence.get("parent_segment_id") or str(sentence["segment_id"]).split("__s")[0])
        s_start, s_end = int(sentence["start_ms"]), int(sentence["end_ms"])
        listed = [u for u in ordered if parent in {str(x) for x in u.get("source_segment_ids") or []}]
        pool = listed or ordered
        best = max(pool, key=lambda u: (_overlap(s_start, s_end, int(u["start_ms"]), int(u["end_ms"])),
                                        -abs((s_start + s_end) / 2 - (int(u["start_ms"]) + int(u["end_ms"])) / 2)))
        owned[str(best["item_id"])].append(str(sentence["segment_id"]))
    empty = [uid for uid, ids in owned.items() if not ids]
    if empty:
        raise RenderAdapterError(
            f"unit(s) {empty} own no sentence of the source (their segments were merged into another unit's "
            "sentences); merge or re-time them before rendering")
    flattened = [sid for u in ordered for sid in owned[str(u["item_id"])]]
    if flattened != [str(s["segment_id"]) for s in sentences]:
        raise RenderAdapterError(
            "the units, in time order, do not reproduce the source sentences in order (interleaved or overlapping "
            "units); the renderer needs a clean left-to-right cover")
    return owned


def build_committed_translation(
    submission: Mapping[str, Any], sentences: Sequence[Mapping[str, Any]], *, pass1_sha256: str = "",
) -> dict[str, Any]:
    """The ``pass2_natural_translation.json`` artifact for an agent submission."""
    from .native_dub import _build_speaker_turns

    units = sorted(submission["items"], key=lambda u: (int(u["start_ms"]), int(u["end_ms"])))
    owned = assign_sentences(units, sentences)
    by_id = {str(s["segment_id"]): s for s in sentences}
    turn_of: dict[str, str] = {}
    turn_input = [{"group_id": str(u["item_id"]), "speaker_id": str(u["speaker_id"]), "start_ms": int(u["start_ms"]),
                   "source_end_ms": int(u["end_ms"])} for u in units]
    for turn in _build_speaker_turns(turn_input):
        for member in turn["member_group_ids"]:
            turn_of[str(member)] = str(turn["member_group_ids"][0])

    groups: list[dict[str, Any]] = []
    translations: list[dict[str, Any]] = []
    for unit in units:
        uid = str(unit["item_id"])
        segment_ids = owned[uid]
        text = str(unit["zulu_text"]).strip()
        start, end = int(unit["start_ms"]), int(unit["end_ms"])
        groups.append({
            "group_id": uid, "turn_id": turn_of.get(uid), "speaker_id": str(unit["speaker_id"]),
            "segment_ids": segment_ids, "member_group_ids": [uid],
            "start_ms": start, "source_end_ms": end, "source_span_ms": max(1, end - start),
            "source_text": " ".join(str(by_id[sid]["restored_text"]) for sid in segment_ids),
            "discourse_unit_kind": None, "spoken_text": text, "undersized": False, "semantic_guard_warnings": [],
            "selected_candidate_id": f"agent:{uid}", "selected_candidate_variant": "agent",
            "requires_review": False, "candidate_pool_source": False, "agent_submission_source": True,
        })
        for position, segment_id in enumerate(segment_ids):
            translations.append({"segment_id": segment_id, "spoken_text": text if position == 0 else "",
                                 "temporal_mask_group_id": uid})
    return {
        "schema_version": COMMIT_SCHEMA, "candidate_pool_committed": False, "agent_submission_committed": True,
        "provider": "dubbing-agent", "approach_notes": str(submission.get("approach_notes") or ""),
        "source_pass1_sha256": pass1_sha256, "temporal_mask_mode": True, "rolling_syllable_mode": False,
        "timing_optimized": False, "tts_rate_percent": 0, "temporal_mask_groups": groups, "translations": translations,
        "summary": {"mask_count": len(groups), "source_sentence_count": len(sentences)},
    }


def _speed(group: Mapping[str, Any]) -> float:
    """The post-TTS speed-up the renderer applied to one scheduled group (0 when none)."""
    fit = group.get("speed_fit") or {}
    return float(fit.get("speed_percent") or group.get("speed_percent") or 0.0)


def _copy_if_exists(source: Path, target: Path) -> bool:
    if source.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        return True
    return False


def prepare_render_root(
    store: Any, work_dir: Path, *, run_id: str, clip_id: str, repeat: int,
) -> tuple[Path, dict[str, Any], Path]:
    """Build ``<run>/<clip>/r<k>/render`` with the inputs the renderer expects. Returns (root, job info, voice map)."""
    corpus = store.load_corpus()["clips"]
    if clip_id not in corpus:
        raise RenderAdapterError(f"unknown clip {clip_id!r}")
    job_id = corpus[clip_id]["job_id"]
    original = Path(work_dir) / "jobs" / job_id
    frozen = store.clip_dir(clip_id) / "native_dub"
    run_dir = store.repeat_dir(run_id, clip_id, repeat)
    submission_path = run_dir / "submission.json"
    if not submission_path.is_file():
        raise RenderAdapterError(f"no accepted submission at {submission_path}")
    submission = read_json(submission_path)

    root = run_dir / "render"
    native = root / "native_dub"
    native.mkdir(parents=True, exist_ok=True)
    for name in _FROZEN_INPUTS:
        if not _copy_if_exists(frozen / name, native / name):
            raise RenderAdapterError(f"frozen clip is missing {name}")
    for name in _OPTIONAL_JOB_INPUTS:
        _copy_if_exists(original / "native_dub" / name, native / name)

    pass1 = read_json(native / "pass1_restored_transcript.json")
    sentences_path = original / "native_dub" / "pass1_sentences.json"
    if sentences_path.is_file():
        records = list(read_json(sentences_path).get("records") or [])
    else:
        from .native_dub import _build_pass1_sentence_records

        records = _build_pass1_sentence_records(list(pass1["segments"]), job_root=original)
    atomic_write_json(native / "pass1_sentences.json", {
        "schema_version": "mathula-native-pass1-sentences-v1", "sentence_count": len(records), "records": records})

    voice_source = original / "direct_dub" / "voice_resolution.json"
    voice_target = root / "direct_dub" / "voice_resolution.json"
    if not _copy_if_exists(voice_source, voice_target):
        raise RenderAdapterError(f"the original job has no voice resolution at {voice_source}")

    committed = build_committed_translation(submission, records)
    atomic_write_json(native / "pass2_natural_translation.json", committed)
    return root, {"job_id": job_id, "groups": len(committed["temporal_mask_groups"]), "sentences": len(records)}, voice_target


def render_submission(
    settings: Any, *, run_id: str, clip_id: str, repeat: int = 1, force: bool = False, lite: bool = False,
    progress: Any = None,
) -> dict[str, Any]:
    """Prepare the render root and run the production renderer without any hook work."""
    from .cli import create_tts_backend
    from .experience_store import ExperienceStore
    from .native_dub import render_native_dub
    from .orchestrator import Orchestrator

    store = ExperienceStore(settings.work_dir)
    root, info, voice_map = prepare_render_root(store, settings.work_dir, run_id=run_id, clip_id=clip_id, repeat=repeat)
    job = Orchestrator(settings).jobs.load(info["job_id"])               # read-only: used for the source video path
    result = render_native_dub(
        job=job, job_root=root, tts=create_tts_backend(settings), force=force, lite=lite,
        apply_opening_cut=False, render_hook_overlay=False,             # plain dubbed master only
        voice_map_path=voice_map, enable_island_timing=False,
        enable_sdk_group_synthesis=False, enable_turn_group_synthesis=False, progress=progress,
    )
    groups = result.get("groups") or []
    summary = {
        "run": run_id, "clip": clip_id, "repeat": repeat, "video": str(result["output_video"]),
        "groups": result.get("phrase_group_count"), "video_retimed": result.get("video_retimed"),
        "hook_overlay_enabled": result.get("hook_overlay_enabled"), "hook_edit_enabled": result.get("hook_edit_enabled"),
        "timing_quality_review_required": result.get("timing_quality_review_required", False),
        "units_sped_up": sum(1 for g in groups if _speed(g) > 0),
        "max_speed_percent": max((_speed(g) for g in groups), default=0.0),
    }
    atomic_write_json(root / "render_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mathula_tv.dub_render")
    parser.add_argument("--run", required=True)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--lite", action="store_true", help="cap ffmpeg CPU use")
    parser.add_argument("--prepare-only", action="store_true", help="build the committed translation, do not render")
    args = parser.parse_args(argv)

    from .config import load_settings
    from .experience_store import ExperienceStore

    settings = load_settings()
    if args.prepare_only:
        root, info, _ = prepare_render_root(ExperienceStore(settings.work_dir), settings.work_dir,
                                            run_id=args.run, clip_id=args.clip, repeat=args.repeat)
        print(json.dumps({"render_root": str(root), **info}, indent=2))
        return 0
    print(json.dumps(render_submission(settings, run_id=args.run, clip_id=args.clip, repeat=args.repeat,
                                       force=args.force, lite=args.lite), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
