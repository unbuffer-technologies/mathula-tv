#!/usr/bin/env python3
"""Rerun deterministic autocorrection plus name research for one existing job."""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


def authoritative_text_fingerprint(document: object) -> str:
    """Hash spoken/source content while ignoring regenerated audit metadata."""

    import hashlib

    if not isinstance(document, dict):
        return hashlib.sha256(b"null").hexdigest()
    segments = []
    for index, segment in enumerate(document.get("segments") or [], start=1):
        if not isinstance(segment, dict):
            continue
        segments.append(
            {
                "segment_id": str(
                    segment.get("segment_id")
                    or segment.get("id")
                    or f"segment_{index:04d}"
                ),
                "source_text": str(
                    segment.get("source_text")
                    or segment.get("text")
                    or ""
                ),
                "words": [
                    str(
                        word.get("text")
                        or word.get("word")
                        or word.get("display")
                        or ""
                    )
                    for word in (segment.get("words") or [])
                    if isinstance(word, dict)
                ],
            }
        )
    payload = {
        "complete_text": str(document.get("complete_text") or ""),
        "segments": segments,
        "top_level_words": [
            str(
                word.get("text")
                or word.get("word")
                or word.get("display")
                or ""
            )
            for word in (document.get("words") or [])
            if isinstance(word, dict)
        ],
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    if len(sys.argv) not in {2, 3}:
        raise SystemExit(
            "Usage: rerun_autocorrect_name_research.py JOB_ID [REPO_ROOT]"
        )

    job_id = sys.argv[1]
    repo = (
        Path(sys.argv[2]).expanduser().resolve()
        if len(sys.argv) == 3
        else Path(__file__).resolve().parents[1]
    )
    sys.path.insert(0, str(repo / "src"))

    from mathula_tv.atomic_io import read_json
    from mathula_tv.autocorrect import JobPaths
    from mathula_tv.autocorrect_stage import (
        AutocorrectStageConfig,
        prepare_explicit_name_research_rerun,
        run_autocorrection_first,
    )
    from mathula_tv.config import load_settings
    from mathula_tv.job_store import JobStore
    from mathula_tv.media import checksum

    settings = load_settings(repo)
    jobs = JobStore(settings.work_dir)
    job = jobs.load(job_id)
    job_root = settings.work_dir / "jobs" / job_id
    analysis = job_root / "analysis"
    raw_path = analysis / "transcript_en_raw.json"
    transcript_path = analysis / "transcript_en.json"
    state_path = analysis / "autocorrection_state.json"
    research_path = analysis / "autocorrection_name_research.json"

    if not raw_path.is_file():
        raise SystemExit(f"Missing immutable raw transcript: {raw_path}")

    previous_document = read_json(transcript_path) if transcript_path.is_file() else None
    previous_hash = checksum(transcript_path) if transcript_path.is_file() else None
    previous_text_fingerprint = authoritative_text_fingerprint(previous_document)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_root = analysis / "autocorrection_research_rerun_backup" / stamp
    for path in (transcript_path, state_path, research_path):
        if path.is_file():
            destination = backup_root / path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)

    # Do not overwrite a database-authorised research transcript before
    # ``ensure_job`` checks it. The normal stage will regenerate the deterministic
    # render itself. Only recover legacy states where an older release left the
    # file and registered effective hash out of sync.
    autocorrect_paths = JobPaths.for_job(settings.work_dir, job_id)
    rerun_preflight = prepare_explicit_name_research_rerun(
        config=AutocorrectStageConfig.from_environment(repo),
        paths=autocorrect_paths,
        job_id=job_id,
    )
    if rerun_preflight.get("status") == "recovered":
        print(
            "[autocorrect research] repaired legacy transcript/database "
            "integrity mismatch from deterministic autocorrect render",
            file=sys.stderr,
        )
    elif rerun_preflight.get("status") == "authorized":
        print(
            "[autocorrect research] current effective transcript is "
            "database-authorized; deterministic render will be regenerated",
            file=sys.stderr,
        )

    state = run_autocorrection_first(
        work_dir=settings.work_dir,
        job=job,
        reconciled_transcript=read_json(raw_path),
    )
    current_hash = checksum(transcript_path)
    current_document = read_json(transcript_path)
    current_text_fingerprint = authoritative_text_fingerprint(current_document)
    authoritative_text_changed = previous_text_fingerprint != current_text_fingerprint
    job.media["raw_transcript_sha256"] = state.get("raw_transcript_sha256")
    job.media["authoritative_transcript_sha256"] = current_hash
    job.media["autocorrection_state_sha256"] = checksum(state_path)
    job.media["autocorrection_name_research_sha256"] = (
        checksum(research_path) if research_path.is_file() else None
    )
    if "autocorrection" not in job.completed_stages:
        job.completed_stages.append("autocorrection")
    jobs.save(job)

    translation_paths = [
        job_root / "translation" / "transcript_zu.json",
        job_root / "dubbing" / "translation_zu.json",
    ]
    print(
        json.dumps(
            {
                "job_id": job_id,
                "state": job.state,
                "name_research_status": state.get("name_research_status"),
                "name_research_profile": state.get("name_research_profile"),
                "name_research_candidate_count": state.get(
                    "name_research_candidate_count", 0
                ),
                "name_research_applied_count": state.get(
                    "name_research_applied_count", 0
                ),
                "name_research_unresolved_count": state.get(
                    "name_research_unresolved_count", 0
                ),
                "authoritative_transcript_changed": authoritative_text_changed,
                "authoritative_file_hash_changed": previous_hash != current_hash,
                "authoritative_transcript_sha256": current_hash,
                "research_artifact": str(research_path),
                "backup_directory": str(backup_root),
                "existing_translation_present": any(
                    path.is_file() for path in translation_paths
                ),
                "next_action": (
                    "rerun translation/editorial from the corrected authoritative transcript"
                    if authoritative_text_changed
                    and any(path.is_file() for path in translation_paths)
                    else "continue the normal pipeline"
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
