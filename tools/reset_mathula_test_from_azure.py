#!/usr/bin/env python3
"""Archive downstream work and restart one job from immutable Azure STT text.

The PostgreSQL vocabulary remains authoritative. This utility never creates or
modifies learned rules. Before moving any files, it verifies that the current job
can see an active, auto-applicable PostgreSQL rule for the known Azure STT phrase.
It then clears only this job's correction decisions, reruns autocorrection, and
resets downstream workflow state to ``analysis_ready``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


def copy_if_present(source: Path, destination: Path) -> None:
    if not source.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def authoritative_text_values(transcript: Mapping[str, Any]) -> list[str]:
    """Return only effective text fields consumed by downstream language stages.

    Raw Azure word-timing tokens may remain as provenance in nested ``words``
    arrays. They are intentionally excluded here; dubbing-unit construction has
    its own guard that must derive effective text from corrected phrase fields.
    """

    values: list[str] = []
    complete = transcript.get("complete_text")
    if isinstance(complete, str) and complete.strip():
        values.append(complete)

    for collection_name in ("segments", "turns", "phrases", "recognizedPhrases"):
        collection = transcript.get(collection_name)
        if not isinstance(collection, list):
            continue
        for item in collection:
            if not isinstance(item, Mapping):
                continue
            for key in ("source_text", "text", "display_text", "display", "lexical"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    values.append(value)
                    break
    return values


def select_visible_postgres_rule(
    candidates: Iterable[Mapping[str, Any]],
    *,
    heard_fragment: str,
    normalize: Any,
    confirmation_threshold: int,
) -> dict[str, Any] | None:
    """Choose an active learned rule visible to this job without modifying it."""

    fragment = normalize(heard_fragment)
    eligible: list[dict[str, Any]] = []
    for candidate in candidates:
        if str(candidate.get("source")) != "learned_vocabulary":
            continue
        surfaces = candidate.get("surfaces", [])
        if isinstance(surfaces, str):
            surfaces = [surfaces]
        surface_norms = [normalize(str(value)) for value in surfaces]
        if not any(fragment in value or value in fragment for value in surface_norms if value):
            continue
        if int(candidate.get("reversions", 0) or 0) > 0:
            continue
        mode = str(candidate.get("mode") or "")
        confirmations = int(candidate.get("confirmations", 0) or 0)
        if mode != "always" and confirmations < confirmation_threshold:
            continue
        eligible.append(dict(candidate))

    if not eligible:
        return None

    scope_priority = {"job": 3, "user": 2, "project": 1}
    eligible.sort(
        key=lambda item: (
            1 if str(item.get("mode")) == "always" else 0,
            int(item.get("confirmations", 0) or 0),
            scope_priority.get(str(item.get("scope")), 0),
        ),
        reverse=True,
    )
    return eligible[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("job_id")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--heard-fragment",
        default="horse curry",
        help="Raw Azure STT fragment whose learned PostgreSQL rule must be visible.",
    )
    args = parser.parse_args()

    repo = args.repo.resolve()
    sys.path.insert(0, str(repo / "src"))

    from mathula_tv.atomic_io import atomic_write_json, read_json
    from mathula_tv.autocorrect import (
        JobPaths,
        Store,
        normalize,
        reset_source,
        vocabulary_candidates,
    )
    from mathula_tv.autocorrect_stage import (
        AutocorrectStageConfig,
        require_authoritative_transcript,
        run_autocorrection_first,
    )
    from mathula_tv.config import load_settings
    from mathula_tv.media import checksum
    from mathula_tv.models import utcnow

    settings = load_settings(repo)
    config = AutocorrectStageConfig.from_environment(repo)
    job_root = settings.work_dir / "jobs" / args.job_id
    manifest_path = job_root / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"Missing job manifest: {manifest_path}")
    manifest = read_json(manifest_path)

    analysis = job_root / "analysis"
    raw_path = analysis / "transcript_en_raw.json"
    if not raw_path.is_file():
        fallback = analysis / "transcript_en.autocorrect_source.json"
        if fallback.is_file():
            raw_path = fallback
        else:
            raise SystemExit(
                "No immutable pre-autocorrect transcript was found. Expected "
                "analysis/transcript_en_raw.json"
            )
    raw_transcript = read_json(raw_path)

    # PostgreSQL is authoritative. Abort before any destructive move if the
    # required learned rule is missing or not auto-applicable for this job.
    store = Store(config.database_url, schema=config.schema)
    try:
        candidates = vocabulary_candidates(
            store,
            job_id=args.job_id,
            user_id=config.user_id,
            project_id=config.project_id,
        )
        postgres_rule = select_visible_postgres_rule(
            candidates,
            heard_fragment=args.heard_fragment,
            normalize=normalize,
            confirmation_threshold=config.auto_confirmation_threshold,
        )
        if postgres_rule is None:
            raise SystemExit(
                "No active auto-applicable PostgreSQL vocabulary rule visible to "
                f"job {args.job_id!r} matches {args.heard_fragment!r}. Nothing was cleared."
            )
    finally:
        store.close()

    replacement_text = str(postgres_rule.get("replacement_text") or "").strip()
    if not replacement_text:
        raise SystemExit("The visible PostgreSQL rule has an empty replacement; nothing was cleared")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = settings.work_dir / "job_archives" / f"{args.job_id}-before-fresh-test-{stamp}"
    backup.mkdir(parents=True, exist_ok=False)

    # Archive whole generated trees. Original input remains in place.
    for name in ("analysis", "dubbing", "translation", "subtitles", "render", "output", "logs"):
        source = job_root / name
        if source.exists():
            shutil.move(str(source), str(backup / name))

    # Archive generated audio, preserving the source derivatives.
    audio = job_root / "audio"
    if audio.is_dir():
        for child in list(audio.iterdir()):
            if child.name in {"analysis_mono.wav", "mix_source_stereo.wav"}:
                continue
            target = backup / "audio" / child.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(child), str(target))

    # Restore only immutable analysis inputs needed to restart from Azure output.
    archived_analysis = backup / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    atomic_write_json(analysis / "transcript_en_raw.json", raw_transcript)
    for filename in (
        "azure_stt.json",
        "azure_diarization.json",
        "azure_stt_phrase_list.json",
        "pyannote_diarization.json",
    ):
        copy_if_present(archived_analysis / filename, analysis / filename)

    # Clear only this job's correction decisions. Learned PostgreSQL vocabulary
    # is preserved by reset_source and remains the source of the correction.
    paths = JobPaths.for_job(settings.work_dir, args.job_id)
    atomic_write_json(paths.effective, raw_transcript)
    store = Store(config.database_url, schema=config.schema)
    try:
        reset_source(
            store,
            paths=paths,
            job_id=args.job_id,
            actor_id="fresh-test-reset",
        )
    finally:
        store.close()

    class Job:
        job_id = args.job_id

    state = run_autocorrection_first(
        work_dir=settings.work_dir,
        job=Job(),
        reconciled_transcript=raw_transcript,
        config=config,
    )
    if not state.get("ready_for_language_ai"):
        raise SystemExit(
            f"Autocorrection still has {state.get('blocking_review_count', state.get('review_count', 0))} "
            "blocking review item(s)"
        )

    authoritative = require_authoritative_transcript(
        work_dir=settings.work_dir,
        job_id=args.job_id,
    )
    effective = read_json(Path(authoritative["path"]))
    effective_values = authoritative_text_values(effective)
    effective_norm = "\n".join(normalize(value) for value in effective_values)
    heard_norm = normalize(args.heard_fragment)
    replacement_norm = normalize(replacement_text)
    if heard_norm and heard_norm in effective_norm:
        raise SystemExit(
            f"Fresh autocorrection still contains {args.heard_fragment!r}; language AI remains blocked"
        )
    if replacement_norm and replacement_norm not in effective_norm:
        raise SystemExit(
            "Fresh autocorrection did not apply the visible PostgreSQL replacement "
            f"{replacement_text!r}"
        )

    downstream_prefixes = (
        "claude_",
        "azure_tts",
        "speaker_profile",
        "acoustic_analysis",
        "voice_resolution",
        "alignment",
        "background",
        "mix",
        "render",
        "translation",
    )
    completed = [
        stage
        for stage in manifest.get("completed_stages", [])
        if not str(stage).startswith(downstream_prefixes)
    ]
    if "autocorrection" not in completed:
        completed.append("autocorrection")

    counters = {
        key: value
        for key, value in dict(manifest.get("attempt_counters", {})).items()
        if not str(key).startswith(downstream_prefixes)
    }
    history = list(manifest.get("attempt_history", []))
    history.append(
        {
            "stage": "fresh_test_reset_from_azure",
            "status": "completed",
            "from_state": manifest.get("state"),
            "to_state": "analysis_ready",
            "backup_path": str(backup),
            "postgres_rule_id": postgres_rule.get("rule_id"),
            "postgres_rule_scope": postgres_rule.get("scope"),
            "postgres_replacement_text": replacement_text,
            "raw_transcript_sha256": checksum(analysis / "transcript_en_raw.json"),
            "authoritative_transcript_sha256": authoritative["sha256"],
            "autocorrection_state_sha256": authoritative["state_sha256"],
            "completed_at": utcnow(),
        }
    )

    manifest.update(
        {
            "state": "analysis_ready",
            "updated_at": utcnow(),
            "completed_stages": completed,
            "attempt_counters": counters,
            "attempt_history": history,
            "last_error": None,
            "review_readiness": "autocorrection_complete",
        }
    )
    manifest.setdefault("media", {}).update(
        {
            "raw_transcript_sha256": checksum(analysis / "transcript_en_raw.json"),
            "authoritative_transcript_sha256": authoritative["sha256"],
            "autocorrection_state_sha256": authoritative["state_sha256"],
        }
    )
    for key in list(manifest.get("objects", {})):
        if any(token in key for token in ("translation", "tts", "dub", "render", "review", "mix")):
            manifest["objects"].pop(key, None)
    for key in list(manifest.get("providers", {})):
        if any(token in key for token in ("translation", "tts", "synthesis", "voice_resolution")):
            manifest["providers"].pop(key, None)

    atomic_write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "job_id": args.job_id,
                "state": "analysis_ready",
                "backup": str(backup),
                "raw_phrase_removed": True,
                "postgres_rule_id": postgres_rule.get("rule_id"),
                "postgres_rule_scope": postgres_rule.get("scope"),
                "postgres_replacement_text": replacement_text,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
