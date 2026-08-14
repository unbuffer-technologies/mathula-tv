"""Authoritative autocorrection stage executed before language AI.

The stage preserves a raw reconciled transcript, renders the effective transcript
through Mathula TV's existing PostgreSQL autocorrect engine, and blocks all
language-dependent downstream work while corrections still require review.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .atomic_io import atomic_write_json, read_json
from .autocorrect import (
    AutocorrectError,
    JobPaths,
    Store,
    detect_candidates,
    ensure_job,
    load_json,
    render_effective_transcript,
    reset_source,
    sha256_json,
)
from .autocorrect_research import run_name_research_autocorrection
from .media import checksum
from .models import JobManifest, utcnow

AUTOCORRECT_FIRST_STAGE_SCHEMA = "autocorrection-first-stage-v2-name-research"

REVIEW_STATUSES = {"suggested", "needs_review"}
BLOCKING_REVIEW_SOURCE_TYPES = {"ambiguous_hint", "confusion_hint"}


class AutocorrectionReviewRequired(RuntimeError):
    """Raised when language AI must wait for transcript correction review."""


@dataclass(frozen=True)
class AutocorrectStageConfig:
    database_url: str
    schema: str
    hints_path: Path | None
    registry_path: Path | None
    user_id: str | None
    project_id: str | None
    fuzzy_threshold: float
    max_candidates: int
    low_confidence_threshold: float
    auto_confirmation_threshold: int

    @classmethod
    def from_environment(cls, project_root: Path) -> "AutocorrectStageConfig":
        database_url = (
            os.getenv("MATHULA_TV_DATABASE_URL")
            or os.getenv("DATABASE_URL")
            or ""
        ).strip()
        if not database_url:
            raise AutocorrectError(
                "Autocorrection must run before language AI, but "
                "MATHULA_TV_DATABASE_URL or DATABASE_URL is not configured"
            )

        def optional_path(name: str, default: Path) -> Path | None:
            raw = os.getenv(name)
            path = Path(raw).expanduser() if raw else default
            return path if path.is_file() else None

        return cls(
            database_url=database_url,
            schema=os.getenv(
                "MATHULA_TV_AUTOCORRECT_SCHEMA",
                "mathula_autocorrect",
            ).strip(),
            hints_path=optional_path(
                "MATHULA_TV_AUTOCORRECT_HINTS_PATH",
                project_root / "config/autocorrect_hints.json",
            ),
            registry_path=optional_path(
                "MATHULA_TV_AUTOCORRECT_REGISTRY_PATH",
                project_root / "config/entity_registry.json",
            ),
            user_id=os.getenv("MATHULA_TV_AUTOCORRECT_USER_ID") or None,
            project_id=(
                os.getenv("MATHULA_TV_AUTOCORRECT_PROJECT_ID")
                or "mathula-tv"
            ),
            fuzzy_threshold=float(
                os.getenv("MATHULA_TV_AUTOCORRECT_FUZZY_THRESHOLD", "0.68")
            ),
            max_candidates=int(
                os.getenv("MATHULA_TV_AUTOCORRECT_MAX_CANDIDATES", "4")
            ),
            low_confidence_threshold=float(
                os.getenv(
                    "MATHULA_TV_AUTOCORRECT_LOW_CONFIDENCE_THRESHOLD",
                    "0.72",
                )
            ),
            auto_confirmation_threshold=int(
                os.getenv(
                    "MATHULA_TV_AUTOCORRECT_AUTO_CONFIRMATIONS",
                    "2",
                )
            ),
        )


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]



def _configured_autocorrect_hints_path(project_root: Path) -> Path | None:
    raw = os.getenv("MATHULA_TV_AUTOCORRECT_HINTS_PATH")
    path = Path(raw).expanduser() if raw else project_root / "config/autocorrect_hints.json"
    return path if path.is_file() else None


def _rule_flags(names: Any) -> int:
    result = 0
    mapping = {"IGNORECASE": re.IGNORECASE, "MULTILINE": re.MULTILINE}
    for name in names if isinstance(names, list) else []:
        if str(name) not in mapping:
            raise ValueError(f"Unsupported autocorrect regex flag: {name}")
        result |= mapping[str(name)]
    return result


def _authoritative_item_texts(transcript: Mapping[str, Any]) -> list[tuple[str, str]]:
    collection = transcript.get("segments")
    if not isinstance(collection, list):
        collection = transcript.get("turns")
    if not isinstance(collection, list):
        return []
    values: list[tuple[str, str]] = []
    for index, item in enumerate(collection):
        if not isinstance(item, Mapping):
            continue
        item_id = str(
            item.get("segment_id")
            or item.get("turn_id")
            or item.get("phrase_id")
            or item.get("id")
            or f"item_{index + 1:04d}"
        )
        text = next(
            (
                str(item[key])
                for key in ("source_text", "text", "display", "lexical")
                if isinstance(item.get(key), str)
            ),
            "",
        )
        values.append((item_id, text))
    return values


def assert_no_configured_autocorrect_confusions(
    transcript: Mapping[str, Any],
    *,
    project_root: Path | None = None,
) -> None:
    """Block language AI when a curated auto-apply confusion remains unresolved.

    A configured deterministic correction is an application invariant, not an
    advisory suggestion. If its raw Azure spelling is still present in the
    authoritative transcript, autocorrection did not finish correctly and no
    semantic or translation model may see that spelling.
    """

    root = project_root or _project_root()
    path = _configured_autocorrect_hints_path(root)
    if path is None:
        return
    value = json.loads(path.read_text(encoding="utf-8"))
    rules = value.get("rules", []) if isinstance(value, Mapping) else []
    if not isinstance(rules, list):
        raise ValueError(f"autocorrect rules must be a list: {path}")

    unresolved: list[str] = []
    for rule in rules:
        if not isinstance(rule, Mapping):
            continue
        if bool(rule.get("review_required", False)) or not bool(rule.get("auto_apply", True)):
            continue
        pattern_text = str(rule.get("pattern") or "").strip()
        replacement = str(rule.get("replacement") or "").strip()
        if not pattern_text or not replacement:
            continue
        pattern = re.compile(pattern_text, _rule_flags(rule.get("flags", [])))
        for item_id, text in _authoritative_item_texts(transcript):
            match = pattern.search(text)
            if match is None:
                continue
            if " ".join(match.group(0).casefold().split()) == " ".join(replacement.casefold().split()):
                continue
            unresolved.append(f"{match.group(0)!r} in {item_id} -> {replacement!r}")

    if unresolved:
        preview = "; ".join(unresolved[:5])
        raise ValueError(
            "Authoritative autocorrected transcript still contains configured "
            f"auto-apply Azure STT confusion(s): {preview}. Rerun autocorrection "
            "from analysis/transcript_en_raw.json before language AI"
        )

def language_ai_review_counts(state: Mapping[str, Any]) -> dict[str, int]:
    """Split unresolved corrections into blocking and advisory review items.

    Explicit ambiguity/confusion rules block language AI. Generic Azure
    low-confidence detections remain visible for later review but do not stall
    translation when no authoritative replacement is available.
    """

    blocking = 0
    advisory = 0
    corrections = state.get("corrections", [])
    if not isinstance(corrections, list):
        corrections = []

    for correction in corrections:
        if not isinstance(correction, Mapping):
            continue
        if str(correction.get("status") or "") not in REVIEW_STATUSES:
            continue
        source_type = str(correction.get("source_type") or "")
        if source_type in BLOCKING_REVIEW_SOURCE_TYPES:
            blocking += 1
        else:
            advisory += 1

    # Backward-compatible fallback for old/incomplete state payloads.
    total = int(state.get("review_count", blocking + advisory) or 0)
    accounted = blocking + advisory
    if total > accounted:
        advisory += total - accounted

    return {
        "blocking_review_count": blocking,
        "advisory_review_count": advisory,
    }


def prepare_explicit_name_research_rerun(
    *,
    config: AutocorrectStageConfig,
    paths: JobPaths,
    job_id: str,
) -> dict[str, Any]:
    """Authorize a safe effective transcript before an explicit rerun.

    Normal reruns should begin with the currently registered effective transcript.
    ``render_effective_transcript`` will deterministically rebuild it from the
    immutable snapshot after ``ensure_job`` verifies integrity. Older research
    releases could leave the file and PostgreSQL hash out of sync; only that
    legacy mismatch is recovered here, using the deterministic corrected artifact
    after verifying it belongs to the current immutable source generation.
    """

    if not paths.effective.is_file():
        raise FileNotFoundError(paths.effective)

    current_document = read_json(paths.effective)
    current_hash = sha256_json(current_document)
    store = Store(config.database_url, schema=config.schema)
    try:
        with store.transaction() as db:
            row = db.execute(
                "SELECT source_sha256, effective_sha256 FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if row is None:
                return {
                    "status": "unregistered",
                    "current_sha256": current_hash,
                }

            source_hash = str(row["source_sha256"] or "")
            registered_hash = str(row["effective_sha256"] or "")
            if current_hash in {source_hash, registered_hash}:
                return {
                    "status": "authorized",
                    "current_sha256": current_hash,
                    "registered_sha256": registered_hash,
                }

            if not paths.snapshot.is_file():
                raise AutocorrectError(
                    f"Immutable source snapshot is missing: {paths.snapshot}"
                )
            snapshot_hash = sha256_json(read_json(paths.snapshot))
            if snapshot_hash != source_hash:
                raise AutocorrectError(
                    "Cannot recover explicit research rerun: source snapshot "
                    "does not match PostgreSQL state"
                )
            if not paths.corrected.is_file():
                raise AutocorrectError(
                    "Cannot recover explicit research rerun: deterministic "
                    f"autocorrect artifact is missing: {paths.corrected}"
                )

            corrected_document = read_json(paths.corrected)
            corrected_hash = sha256_json(corrected_document)
            autocorrect_meta = corrected_document.get("autocorrect")
            corrected_source_hash = (
                str(autocorrect_meta.get("source_sha256") or "")
                if isinstance(autocorrect_meta, Mapping)
                else ""
            )
            if corrected_source_hash != source_hash:
                raise AutocorrectError(
                    "Cannot recover explicit research rerun: corrected transcript "
                    "was not rendered from the current immutable source"
                )

            # Keep the prior effective document in memory so a database failure
            # cannot leave the filesystem in a newly inconsistent state.
            try:
                atomic_write_json(paths.effective, corrected_document)
                db.execute(
                    """
                    UPDATE jobs
                    SET effective_sha256 = ?, updated_at = ?
                    WHERE job_id = ?
                    """,
                    (corrected_hash, utcnow(), job_id),
                )
                db.execute(
                    """
                    INSERT INTO events(
                        job_id, correction_id, action, actor_id,
                        payload_json, created_at
                    ) VALUES (?, NULL, 'name_research_rerun_integrity_recovered',
                              'system', ?, ?)
                    """,
                    (
                        job_id,
                        json.dumps(
                            {
                                "previous_file_sha256": current_hash,
                                "previous_registered_sha256": registered_hash,
                                "recovered_effective_sha256": corrected_hash,
                            },
                            sort_keys=True,
                        ),
                        utcnow(),
                    ),
                )
            except Exception:
                atomic_write_json(paths.effective, current_document)
                raise

            return {
                "status": "recovered",
                "current_sha256": current_hash,
                "registered_sha256": registered_hash,
                "recovered_sha256": corrected_hash,
            }
    finally:
        store.close()


def _register_researched_effective_transcript(
    *,
    config: AutocorrectStageConfig,
    job_id: str,
    transcript: Mapping[str, Any],
) -> str:
    """Register the research-enriched effective transcript transactionally.

    Name research intentionally runs after deterministic autocorrect rendering.
    Updating only ``transcript_en.json`` would make the next autocorrect run look
    like an unauthorised external edit. Persisting the same canonical JSON hash
    in the autocorrect job row keeps the integrity guard meaningful.
    """

    effective_hash = sha256_json(dict(transcript))
    store = Store(config.database_url, schema=config.schema)
    try:
        with store.transaction() as db:
            db.execute(
                """
                UPDATE jobs
                SET effective_sha256 = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (effective_hash, utcnow(), job_id),
            )
            db.execute(
                """
                INSERT INTO events(
                    job_id, correction_id, action, actor_id,
                    payload_json, created_at
                ) VALUES (?, NULL, 'name_research_rendered', 'system', ?, ?)
                """,
                (
                    job_id,
                    json.dumps(
                        {"effective_sha256": effective_hash},
                        sort_keys=True,
                    ),
                    utcnow(),
                ),
            )
    finally:
        store.close()
    return effective_hash


def run_autocorrection_first(
    *,
    work_dir: Path,
    job: JobManifest,
    reconciled_transcript: Mapping[str, Any],
    config: AutocorrectStageConfig | None = None,
) -> dict[str, Any]:
    """Render the authoritative transcript before classification or translation.

    The reconciled source is persisted independently. If Azure/reconciliation
    output changes, the autocorrect source snapshot is reset before detection.
    Previously confirmed exact rules apply first. A web-grounded research agent
    then verifies suspicious proper-name spellings and repairs strongly evidenced
    STT mistakes in the authoritative transcript before language AI runs.
    """

    project_root = _project_root()
    config = config or AutocorrectStageConfig.from_environment(project_root)
    job_root = Path(work_dir) / "jobs" / job.job_id
    analysis = job_root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    raw_path = analysis / "transcript_en_raw.json"
    atomic_write_json(raw_path, dict(reconciled_transcript))

    paths = JobPaths.for_job(Path(work_dir), job.job_id)
    store = Store(config.database_url, schema=config.schema)
    try:
        raw_hash = sha256_json(dict(reconciled_transcript))
        if paths.snapshot.is_file():
            snapshot_hash = sha256_json(load_json(paths.snapshot))
            if snapshot_hash != raw_hash:
                # A genuinely new STT/reconciliation result starts a new
                # autocorrect source generation. Archive prior correction state.
                atomic_write_json(paths.effective, dict(reconciled_transcript))
                reset_source(
                    store,
                    paths=paths,
                    job_id=job.job_id,
                    actor_id="pipeline",
                )
        else:
            atomic_write_json(paths.effective, dict(reconciled_transcript))

        ensure_job(
            store,
            paths=paths,
            job_id=job.job_id,
            user_id=config.user_id,
            project_id=config.project_id,
        )
        source = load_json(paths.snapshot)
        detect_candidates(
            store,
            job_id=job.job_id,
            user_id=config.user_id,
            project_id=config.project_id,
            source_document=source,
            hints_path=config.hints_path,
            registry_path=config.registry_path,
            fuzzy_threshold=config.fuzzy_threshold,
            max_candidates=config.max_candidates,
            low_confidence_threshold=config.low_confidence_threshold,
            auto_confirmation_threshold=config.auto_confirmation_threshold,
        )
        state = render_effective_transcript(
            store,
            paths=paths,
            job_id=job.job_id,
        )
    finally:
        store.close()

    rendered_transcript = read_json(paths.effective)
    research_result = run_name_research_autocorrection(
        job_root=job_root,
        job=job,
        transcript=rendered_transcript,
        registry_path=config.registry_path,
    )
    researched_transcript = research_result["transcript"]
    research_artifact = research_result["artifact"]
    researched_effective_sha256 = str(state.get("effective_sha256") or "")
    if researched_transcript != rendered_transcript:
        atomic_write_json(paths.effective, researched_transcript)
        try:
            researched_effective_sha256 = _register_researched_effective_transcript(
                config=config,
                job_id=job.job_id,
                transcript=researched_transcript,
            )
        except Exception:
            # Preserve the autocorrect integrity contract if database persistence
            # fails after the atomic file replacement. The deterministic render
            # remains the last database-authorised effective transcript.
            atomic_write_json(paths.effective, rendered_transcript)
            raise

    # Preserve the server's latest deterministic invariant: curated auto-apply
    # confusion rules must be resolved before this stage can declare the
    # authoritative transcript ready for language AI.
    assert_no_configured_autocorrect_confusions(
        read_json(paths.effective),
        project_root=project_root,
    )

    review_counts = language_ai_review_counts(state)
    ready = review_counts["blocking_review_count"] == 0
    enriched = {
        **state,
        "effective_sha256": researched_effective_sha256,
        **review_counts,
        "stage_schema_version": AUTOCORRECT_FIRST_STAGE_SCHEMA,
        "raw_transcript_path": str(raw_path),
        "raw_transcript_sha256": checksum(raw_path),
        "authoritative_transcript_path": str(paths.effective),
        "authoritative_transcript_sha256": checksum(paths.effective),
        "name_research_artifact_path": str(
            job_root / "analysis" / "autocorrection_name_research.json"
        ),
        "name_research_status": research_artifact.get("status"),
        "name_research_profile": (
            research_artifact.get("research_profile", {}).get("name")
            if isinstance(research_artifact.get("research_profile"), Mapping)
            else None
        ),
        "name_research_candidate_count": int(
            research_artifact.get("candidate_count", 0) or 0
        ),
        "name_research_accepted_count": int(
            research_artifact.get("accepted_count", 0) or 0
        ),
        "name_research_applied_count": int(
            research_artifact.get("applied_count", 0) or 0
        ),
        "name_research_unresolved_count": int(
            research_artifact.get("unresolved_count", 0) or 0
        ),
        "ready_for_language_ai": ready,
        "language_ai_blocked": not ready,
        "completed_at": utcnow(),
    }
    atomic_write_json(paths.state, enriched)
    return enriched


def require_authoritative_transcript(
    *,
    work_dir: Path,
    job_id: str,
) -> dict[str, Any]:
    """Return the current authoritative transcript identity or block downstream work."""

    root = Path(work_dir) / "jobs" / job_id
    transcript_path = root / "analysis/transcript_en.json"
    state_path = root / "analysis/autocorrection_state.json"
    if not transcript_path.is_file():
        raise ValueError("Authoritative autocorrected transcript is missing")
    if not state_path.is_file():
        raise ValueError(
            "Autocorrection has not completed. Run analysis reconciliation and "
            "autocorrection before any language AI work"
        )

    transcript = read_json(transcript_path)
    state = read_json(state_path)
    if not isinstance(transcript.get("autocorrect"), Mapping):
        raise ValueError(
            "analysis/transcript_en.json is not an autocorrect-rendered authoritative transcript"
        )
    if not bool(state.get("ready_for_language_ai")):
        blocking_count = int(
            state.get("blocking_review_count", state.get("review_count", 0))
            or 0
        )
        advisory_count = int(state.get("advisory_review_count", 0) or 0)
        suffix = (
            f"; {advisory_count} additional low-confidence suggestion(s) "
            "remain advisory"
            if advisory_count
            else ""
        )
        raise AutocorrectionReviewRequired(
            f"Autocorrection review is required for {blocking_count} blocking "
            "correction(s) before classification, semantic analysis, or "
            f"translation{suffix}"
        )

    actual_hash = checksum(transcript_path)
    recorded_hash = str(state.get("authoritative_transcript_sha256") or "")
    if recorded_hash != actual_hash:
        raise ValueError(
            "Authoritative transcript changed after autocorrection; rerun the "
            "autocorrection stage before language AI"
        )
    assert_no_configured_autocorrect_confusions(transcript)
    return {
        "path": str(transcript_path),
        "sha256": actual_hash,
        "state_path": str(state_path),
        "state_sha256": checksum(state_path),
        "raw_transcript_sha256": state.get("raw_transcript_sha256"),
        "active_correction_count": int(state.get("active_count", 0)),
        "review_count": int(state.get("review_count", 0)),
        "blocking_review_count": int(state.get("blocking_review_count", 0)),
        "advisory_review_count": int(state.get("advisory_review_count", 0)),
        "name_research_status": state.get("name_research_status"),
        "name_research_profile": state.get("name_research_profile"),
        "name_research_applied_count": int(
            state.get("name_research_applied_count", 0) or 0
        ),
        "name_research_unresolved_count": int(
            state.get("name_research_unresolved_count", 0) or 0
        ),
    }
