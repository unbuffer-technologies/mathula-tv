"""Rich terminal review for web-researched transcript autocorrections.

This module deliberately keeps human decisions separate from the provider research
artifact. Approved edits are applied to the authoritative effective transcript,
registered in PostgreSQL, backed up, and recorded in an auditable review artifact.
The immutable Azure/reconciled source snapshot is never modified.
"""

from __future__ import annotations

import argparse
import copy
import getpass
import hashlib
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse

try:
    from rich import box
    from rich.console import Console, Group
    from rich.markup import escape
    from rich.panel import Panel
    from rich.prompt import Confirm, Prompt
    from rich.table import Table
    from rich.text import Text
except ImportError as exc:  # pragma: no cover - installation check
    raise RuntimeError(
        'The autocorrection review interface requires Rich. Run: pip install "rich>=13.9,<15"'
    ) from exc

from .autocorrect import (
    AutocorrectError,
    JobPaths,
    Store,
    atomic_write_json,
    load_json,
    normalize,
    sha256_json,
    utc_now,
)
from .autocorrect_research import (
    _cache_root,
    _candidate_cache_identity,
    _compact_local_context,
    _load_registry_evidence,
    _looks_like_madlanga_source,
    _save_cached_correction,
    _sha256_json,
    _source_metadata,
    _stable_source_metadata,
    apply_researched_corrections,
    build_research_profile,
)
from .autocorrect_stage import AutocorrectStageConfig
from .config import load_settings
from .media import checksum

MANUAL_REVIEW_SCHEMA = "mathula-autocorrect-manual-review-v1"
PROJECT_NOT_NAME_SCHEMA = "mathula-autocorrect-not-name-suppressions-v1"
DOWNSTREAM_STALE_SCHEMA = "mathula-autocorrect-downstream-stale-v1"
FINAL_DECISIONS = {
    "approve",
    "edit",
    "keep_original",
    "not_a_name_job",
    "not_a_name_project",
}


@dataclass(frozen=True)
class ReviewPaths:
    job_root: Path
    analysis: Path
    transcript: Path
    deterministic_transcript: Path
    research: Path
    review: Path
    backups: Path
    downstream_stale: Path
    autocorrection_state: Path
    project_not_names: Path

    @classmethod
    def for_job(cls, work_dir: Path, job_id: str) -> "ReviewPaths":
        job_root = Path(work_dir) / "jobs" / job_id
        analysis = job_root / "analysis"
        cache_root = _cache_root(job_root)
        return cls(
            job_root=job_root,
            analysis=analysis,
            transcript=analysis / "transcript_en.json",
            deterministic_transcript=analysis / "corrected_transcript.json",
            research=analysis / "autocorrection_name_research.json",
            review=analysis / "autocorrection_manual_review.json",
            backups=analysis / "autocorrection_manual_review_backups",
            downstream_stale=analysis / "autocorrection_downstream_stale.json",
            autocorrection_state=analysis / "autocorrection_state.json",
            project_not_names=cache_root / "not_name_suppressions.json",
        )


@dataclass(frozen=True)
class ReviewItem:
    source_status: str
    segment_id: str
    raw_text: str
    suggested_text: str | None
    entity_type: str | None
    confidence: float | None
    reason: str
    validation_reason: str | None
    source_urls: tuple[str, ...]
    story_match_terms: tuple[str, ...]
    context: str

    @property
    def key(self) -> str:
        return decision_key(self.segment_id, self.raw_text)


@dataclass(frozen=True)
class ReviewDecision:
    key: str
    segment_id: str
    raw_text: str
    source_status: str
    action: str
    replacement_text: str | None
    reviewer: str
    reason: str
    learn_for_future_jobs: bool
    decided_at: str
    suggested_text: str | None = None
    entity_type: str | None = None
    confidence: float | None = None
    source_urls: tuple[str, ...] = ()
    story_match_terms: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "segment_id": self.segment_id,
            "raw_text": self.raw_text,
            "source_status": self.source_status,
            "action": self.action,
            "replacement_text": self.replacement_text,
            "reviewer": self.reviewer,
            "reason": self.reason,
            "learn_for_future_jobs": self.learn_for_future_jobs,
            "suggested_text": self.suggested_text,
            "entity_type": self.entity_type,
            "confidence": self.confidence,
            "source_urls": list(self.source_urls),
            "story_match_terms": list(self.story_match_terms),
            "decided_at": self.decided_at,
        }


def decision_key(segment_id: str, raw_text: str) -> str:
    material = f"{segment_id}\0{normalize(raw_text)}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()[:24]


def _read_json_mapping(path: Path, *, required: bool = False) -> dict[str, Any]:
    if not path.is_file():
        if required:
            raise FileNotFoundError(path)
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AutocorrectError(f"Expected a JSON object: {path}")
    return value


def _atomic_write_mapping(path: Path, value: Mapping[str, Any]) -> None:
    atomic_write_json(path, dict(value))


def _segment_id(segment: Mapping[str, Any], index: int) -> str:
    return str(
        segment.get("segment_id")
        or segment.get("turn_id")
        or segment.get("phrase_id")
        or segment.get("id")
        or f"seg-{index:05d}"
    )


def _segment_context(transcript: Mapping[str, Any], segment_id: str, raw_text: str) -> str:
    segments = transcript.get("segments")
    if not isinstance(segments, list):
        return ""
    wanted = normalize(raw_text)
    for index, segment in enumerate(segments, start=1):
        if not isinstance(segment, Mapping):
            continue
        current_id = _segment_id(segment, index)
        text = next(
            (
                str(segment.get(field))
                for field in ("source_text", "text", "display_text", "lexical_text")
                if isinstance(segment.get(field), str)
            ),
            "",
        )
        if current_id == segment_id or (wanted and wanted in normalize(text)):
            return text.strip()
    return ""


def _urls(item: Mapping[str, Any]) -> tuple[str, ...]:
    values = item.get("grounded_source_urls") or item.get("source_urls") or []
    if not isinstance(values, list):
        return ()
    return tuple(str(value).strip() for value in values if str(value).strip())


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def build_review_queue(
    research_artifact: Mapping[str, Any],
    transcript: Mapping[str, Any],
    *,
    existing_decisions: Mapping[str, Mapping[str, Any]] | None = None,
    review_all: bool = False,
) -> list[ReviewItem]:
    existing = existing_decisions or {}
    queue: list[ReviewItem] = []

    groups = (
        ("rejected", research_artifact.get("rejected_corrections") or []),
        ("unresolved", research_artifact.get("unresolved_candidates") or []),
    )
    for source_status, records in groups:
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, Mapping):
                continue
            segment_id = str(record.get("segment_id") or "")
            raw_text = str(record.get("raw_text") or "").strip()
            if not segment_id or not raw_text:
                continue
            key = decision_key(segment_id, raw_text)
            previous_action = str((existing.get(key) or {}).get("action") or "")
            if not review_all and previous_action in FINAL_DECISIONS:
                continue
            suggested = str(record.get("canonical_text") or "").strip() or None
            queue.append(
                ReviewItem(
                    source_status=source_status,
                    segment_id=segment_id,
                    raw_text=raw_text,
                    suggested_text=suggested,
                    entity_type=(str(record.get("entity_type") or "").strip() or None),
                    confidence=_float_or_none(record.get("confidence")),
                    reason=str(record.get("reason") or "").strip(),
                    validation_reason=(
                        str(record.get("validation_reason") or "").strip() or None
                    ),
                    source_urls=_urls(record),
                    story_match_terms=tuple(
                        str(value).strip()
                        for value in (record.get("story_match_terms") or [])
                        if str(value).strip()
                    ),
                    context=_segment_context(transcript, segment_id, raw_text),
                )
            )
    return queue


def _load_review(path: Path, job_id: str) -> dict[str, Any]:
    value = _read_json_mapping(path)
    if not value:
        return {
            "schema_version": MANUAL_REVIEW_SCHEMA,
            "job_id": job_id,
            "status": "draft",
            "decisions": [],
            "created_at": utc_now(),
            "updated_at": utc_now(),
        }
    if value.get("schema_version") != MANUAL_REVIEW_SCHEMA:
        raise AutocorrectError(f"Unsupported manual review schema in {path}")
    if str(value.get("job_id") or "") != job_id:
        raise AutocorrectError(f"Manual review artifact belongs to another job: {path}")
    return value


def _decision_map(review_artifact: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in review_artifact.get("decisions") or []:
        if isinstance(item, Mapping) and str(item.get("key") or ""):
            result[str(item["key"])] = dict(item)
    return result


def _save_draft(
    paths: ReviewPaths,
    artifact: dict[str, Any],
    decisions: Mapping[str, Mapping[str, Any]],
    *,
    reviewer: str,
    research_artifact: Mapping[str, Any],
) -> None:
    artifact.update(
        {
            "schema_version": MANUAL_REVIEW_SCHEMA,
            "job_id": paths.job_root.name,
            "status": "draft",
            "reviewer": reviewer,
            "research_artifact": str(paths.research),
            "research_artifact_sha256": _sha256_json(dict(research_artifact)),
            "decisions": list(decisions.values()),
            "updated_at": utc_now(),
        }
    )
    _atomic_write_mapping(paths.review, artifact)


def _domain(url: str) -> str:
    value = urlparse(url).netloc.casefold().split(":", 1)[0]
    return value[4:] if value.startswith("www.") else value


def _item_renderable(item: ReviewItem, index: int, total: int) -> Group:
    status_style = "yellow" if item.source_status == "rejected" else "cyan"
    heading = Text()
    heading.append(f"Candidate {index}/{total}  ", style="bold white")
    heading.append(item.source_status.upper(), style=f"bold {status_style}")

    summary = Table.grid(padding=(0, 1))
    summary.add_column(style="bold", no_wrap=True)
    summary.add_column()
    summary.add_row("Segment", item.segment_id)
    summary.add_row("Azure heard", Text(item.raw_text, style="bold red"))
    if item.suggested_text:
        summary.add_row("Suggested", Text(item.suggested_text, style="bold green"))
    if item.entity_type:
        summary.add_row("Type", item.entity_type)
    if item.confidence is not None:
        summary.add_row("Confidence", f"{item.confidence:.0%}")
    if item.validation_reason:
        summary.add_row("Validation", item.validation_reason)

    context = Text(item.context or "No segment context available.")
    if item.raw_text and item.context:
        start = item.context.casefold().find(item.raw_text.casefold())
        if start >= 0:
            context.stylize("bold red", start, start + len(item.raw_text))

    evidence = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold")
    evidence.add_column("Evidence", overflow="fold")
    if item.reason:
        evidence.add_row(item.reason)
    for url in item.source_urls:
        evidence.add_row(url)
    if not item.reason and not item.source_urls:
        evidence.add_row("No grounded source was supplied.")

    return Group(
        heading,
        Panel(summary, title="Research decision", border_style=status_style),
        Panel(context, title="Transcript context", border_style="blue"),
        evidence,
    )


def _prompt_reason(console: Console, default: str) -> str:
    return Prompt.ask(
        "[bold]Review note[/bold]",
        default=default[:300] if default else "Reviewed in Mathula TV console",
        console=console,
    ).strip()


def prompt_for_decision(
    console: Console,
    item: ReviewItem,
    *,
    reviewer: str,
) -> ReviewDecision | None:
    options: list[str] = []
    labels: list[tuple[str, str]] = []
    if item.suggested_text:
        options.append("a")
        labels.append(("A", f"Approve: {item.suggested_text}"))
    options.extend(["e", "k", "n", "g", "s", "q"])
    labels.extend(
        [
            ("E", "Enter a corrected spelling"),
            ("K", "Keep the original text"),
            ("N", "Not a name in this job"),
            ("G", "Not a name across Mathula TV"),
            ("S", "Skip for now"),
            ("Q", "Save and quit"),
        ]
    )
    menu = Table.grid(padding=(0, 2))
    for key, label in labels:
        menu.add_row(Text(f"[{key}]", style="bold magenta"), label)
    console.print(menu)
    choice = Prompt.ask(
        "[bold]Decision[/bold]",
        choices=options,
        default="a" if item.suggested_text else "s",
        case_sensitive=False,
        console=console,
    ).casefold()

    if choice == "q":
        return None
    if choice == "s":
        action = "skip"
        replacement = None
        learn = False
        reason = "Deferred for later review"
    elif choice == "a":
        action = "approve"
        replacement = item.suggested_text
        learn = Confirm.ask(
            "Reuse this correction for matching future clips?",
            default=True,
            console=console,
        )
        reason = _prompt_reason(console, item.reason or "Approved grounded suggestion")
    elif choice == "e":
        action = "edit"
        replacement = Prompt.ask(
            "[bold]Correct replacement[/bold]",
            default=item.suggested_text or "",
            console=console,
        ).strip()
        if not replacement:
            console.print("[red]Replacement cannot be empty; item left unresolved.[/red]")
            return ReviewDecision(
                key=item.key,
                segment_id=item.segment_id,
                raw_text=item.raw_text,
                source_status=item.source_status,
                action="skip",
                replacement_text=None,
                reviewer=reviewer,
                reason="Empty replacement was not applied",
                learn_for_future_jobs=False,
                decided_at=utc_now(),
                suggested_text=item.suggested_text,
                entity_type=item.entity_type,
                confidence=item.confidence,
                source_urls=item.source_urls,
                story_match_terms=item.story_match_terms,
            )
        learn = Confirm.ask(
            "Reuse this correction for matching future clips?",
            default=bool(item.source_urls),
            console=console,
        )
        reason = _prompt_reason(console, item.reason or "Reviewer supplied replacement")
    elif choice == "k":
        action = "keep_original"
        replacement = None
        learn = False
        reason = _prompt_reason(console, "Original transcript wording confirmed")
    elif choice == "n":
        action = "not_a_name_job"
        replacement = None
        learn = False
        reason = _prompt_reason(console, "Not a proper-name candidate in this job")
    else:
        action = "not_a_name_project"
        replacement = None
        learn = False
        reason = _prompt_reason(console, "Not a proper-name candidate for Mathula TV")

    return ReviewDecision(
        key=item.key,
        segment_id=item.segment_id,
        raw_text=item.raw_text,
        source_status=item.source_status,
        action=action,
        replacement_text=replacement,
        reviewer=reviewer,
        reason=reason,
        learn_for_future_jobs=learn,
        decided_at=utc_now(),
        suggested_text=item.suggested_text,
        entity_type=item.entity_type,
        confidence=item.confidence,
        source_urls=item.source_urls,
        story_match_terms=item.story_match_terms,
    )


def _load_project_suppressions(path: Path) -> dict[str, Any]:
    value = _read_json_mapping(path)
    if not value:
        return {
            "schema_version": PROJECT_NOT_NAME_SCHEMA,
            "suppressions": [],
            "created_at": utc_now(),
            "updated_at": utc_now(),
        }
    if value.get("schema_version") != PROJECT_NOT_NAME_SCHEMA:
        raise AutocorrectError(f"Unsupported not-name suppression schema: {path}")
    return value


def _save_project_suppressions(
    path: Path,
    decisions: Sequence[Mapping[str, Any]],
) -> int:
    project_decisions = [
        item for item in decisions if item.get("action") == "not_a_name_project"
    ]
    if not project_decisions:
        return 0
    artifact = _load_project_suppressions(path)
    existing = {
        normalize(str(item.get("raw_text") or "")): dict(item)
        for item in artifact.get("suppressions") or []
        if isinstance(item, Mapping) and normalize(str(item.get("raw_text") or ""))
    }
    for item in project_decisions:
        raw = str(item.get("raw_text") or "").strip()
        if not raw:
            continue
        existing[normalize(raw)] = {
            "raw_text": raw,
            "raw_norm": normalize(raw),
            "reason": str(item.get("reason") or ""),
            "reviewer": str(item.get("reviewer") or ""),
            "decided_at": str(item.get("decided_at") or utc_now()),
        }
    artifact["suppressions"] = sorted(existing.values(), key=lambda row: row["raw_norm"])
    artifact["updated_at"] = utc_now()
    _atomic_write_mapping(path, artifact)
    return len(project_decisions)


def _job_manifest(job_root: Path, job_id: str) -> SimpleNamespace:
    manifest = _read_json_mapping(job_root / "manifest.json")
    return SimpleNamespace(
        job_id=job_id,
        source_filename=str(manifest.get("source_filename") or ""),
        local_source_path=str(manifest.get("local_source_path") or ""),
        source_language=str(manifest.get("source_language") or "en-ZA"),
    )


def _candidate_record(
    research: Mapping[str, Any], segment_id: str, raw_text: str
) -> dict[str, Any] | None:
    wanted = (segment_id, normalize(raw_text))
    for item in research.get("candidate_name_spans") or []:
        if not isinstance(item, Mapping):
            continue
        key = (str(item.get("segment_id") or ""), normalize(str(item.get("raw_text") or "")))
        if key == wanted:
            return dict(item)
    return None


def _save_future_corrections(
    *,
    paths: ReviewPaths,
    config: AutocorrectStageConfig,
    research: Mapping[str, Any],
    deterministic_transcript: Mapping[str, Any],
    decisions: Sequence[Mapping[str, Any]],
) -> int:
    selected = [
        item
        for item in decisions
        if item.get("action") in {"approve", "edit"}
        and item.get("learn_for_future_jobs")
        and str(item.get("replacement_text") or "").strip()
    ]
    if not selected:
        return 0

    registry = _load_registry_evidence(config.registry_path)
    job = _job_manifest(paths.job_root, paths.job_root.name)
    source_metadata = _source_metadata(paths.job_root, job)
    candidates = [
        candidate
        for item in selected
        if (candidate := _candidate_record(
            research,
            str(item.get("segment_id") or ""),
            str(item.get("raw_text") or ""),
        ))
        is not None
    ]
    if not candidates:
        return 0
    all_candidates = [
        dict(item)
        for item in research.get("candidate_name_spans") or []
        if isinstance(item, Mapping)
    ]
    local_context = _compact_local_context(
        paths.job_root,
        transcript=deterministic_transcript,
        candidates=all_candidates or candidates,
        include_commission_master_case=_looks_like_madlanga_source(
            deterministic_transcript, source_metadata
        ),
    )
    profile = research.get("research_profile")
    if not isinstance(profile, Mapping):
        profile = build_research_profile(
            transcript=deterministic_transcript,
            source_metadata=source_metadata,
            local_context=local_context,
        )
    cache_root = _cache_root(paths.job_root)
    saved = 0
    for decision in selected:
        candidate = _candidate_record(
            research,
            str(decision.get("segment_id") or ""),
            str(decision.get("raw_text") or ""),
        )
        if candidate is None:
            continue
        identity = _candidate_cache_identity(
            transcript=deterministic_transcript,
            candidate=candidate,
            registry_evidence=registry,
            source_metadata=_stable_source_metadata(source_metadata),
            local_context=local_context,
            research_profile=profile,
        )
        cache_key = _sha256_json(identity)
        source_urls = [
            str(value)
            for value in decision.get("source_urls") or []
            if str(value).strip()
        ]
        correction = {
            "canonical_text": str(decision["replacement_text"]),
            "entity_type": str(decision.get("entity_type") or "other"),
            "confidence": 1.0,
            "reason": str(decision.get("reason") or "Human reviewed correction"),
            "source_urls": source_urls,
            "grounded_source_urls": source_urls,
            "story_match_terms": list(decision.get("story_match_terms") or []),
            "validation_reason": "human_reviewed",
            "reviewed_by": str(decision.get("reviewer") or ""),
            "reviewed_at": str(decision.get("decided_at") or utc_now()),
        }
        _save_cached_correction(
            cache_root=cache_root,
            cache_key=cache_key,
            correction=correction,
        )
        saved += 1
    return saved


def _backup_paths(paths: ReviewPaths) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = paths.backups / stamp
    suffix = 1
    while target.exists():
        target = paths.backups / f"{stamp}-{suffix:02d}"
        suffix += 1
    target.mkdir(parents=True)
    for path in (
        paths.transcript,
        paths.research,
        paths.review,
        paths.downstream_stale,
        paths.autocorrection_state,
    ):
        if path.is_file():
            shutil.copy2(path, target / path.name)
    return target


def _updated_autocorrection_state(
    *,
    paths: ReviewPaths,
    effective_sha256: str,
    reviewer: str,
    decisions: Sequence[Mapping[str, Any]],
    review_path: Path,
) -> dict[str, Any]:
    """Bind the language-AI provenance gate to the reviewed transcript file."""

    state = _read_json_mapping(paths.autocorrection_state, required=True)
    applied = [
        {
            "segment_id": item.get("segment_id"),
            "raw_text": item.get("raw_text"),
            "replacement_text": item.get("replacement_text"),
            "action": item.get("action"),
        }
        for item in decisions
        if item.get("action") in {"approve", "edit"}
    ]
    state.update(
        {
            "effective_sha256": effective_sha256,
            "authoritative_transcript_path": str(paths.transcript),
            "authoritative_transcript_sha256": checksum(paths.transcript),
            "manual_review_status": "completed",
            "manual_review_artifact_path": str(review_path),
            "manual_review_reviewer": reviewer,
            "manual_review_applied_count": len(applied),
            "manual_review_applied_corrections": applied,
            "manual_review_completed_at": utc_now(),
            "completed_at": utc_now(),
        }
    )
    return state


def _stale_artifacts(paths: ReviewPaths) -> list[str]:
    result: list[str] = []
    for relative in ("translation", "dubbing", "direct_dub", "output"):
        root = paths.job_root / relative
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file():
                result.append(str(path.relative_to(paths.job_root)))
                if len(result) >= 500:
                    return result
    return result


def _register_effective_transcript(
    *,
    config: AutocorrectStageConfig,
    paths: ReviewPaths,
    before_document: Mapping[str, Any],
    after_document: Mapping[str, Any],
    reviewer: str,
    decisions: Sequence[Mapping[str, Any]],
    review_path: Path,
) -> str:
    before_hash = sha256_json(dict(before_document))
    after_hash = sha256_json(dict(after_document))
    before_state = _read_json_mapping(paths.autocorrection_state, required=True)
    store = Store(config.database_url, schema=config.schema)
    try:
        with store.transaction() as db:
            row = db.execute(
                "SELECT source_sha256, effective_sha256 FROM jobs WHERE job_id = ? FOR UPDATE",
                (paths.job_root.name,),
            ).fetchone()
            if row is None:
                raise AutocorrectError(
                    f"Autocorrect database has no job row for {paths.job_root.name}"
                )
            allowed = {str(row["source_sha256"] or ""), str(row["effective_sha256"] or "")}
            if before_hash not in allowed:
                raise AutocorrectError(
                    "The authoritative transcript changed outside autocorrect before manual review. "
                    "Run the explicit research rerun integrity recovery first."
                )
            try:
                atomic_write_json(paths.transcript, dict(after_document))
                db.execute(
                    """
                    UPDATE jobs
                    SET effective_sha256 = ?, updated_at = ?
                    WHERE job_id = ?
                    """,
                    (after_hash, utc_now(), paths.job_root.name),
                )
                db.execute(
                    """
                    INSERT INTO events(
                        job_id, correction_id, action, actor_id,
                        payload_json, created_at
                    ) VALUES (?, NULL, 'autocorrect_manual_review_applied', ?, ?, ?)
                    """,
                    (
                        paths.job_root.name,
                        reviewer,
                        json.dumps(
                            {
                                "previous_effective_sha256": before_hash,
                                "effective_sha256": after_hash,
                                "review_artifact": str(review_path),
                                "applied_decisions": [
                                    {
                                        "segment_id": item.get("segment_id"),
                                        "raw_text": item.get("raw_text"),
                                        "replacement_text": item.get("replacement_text"),
                                        "action": item.get("action"),
                                    }
                                    for item in decisions
                                    if item.get("action") in {"approve", "edit"}
                                ],
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        utc_now(),
                    ),
                )
                atomic_write_json(
                    paths.autocorrection_state,
                    _updated_autocorrection_state(
                        paths=paths,
                        effective_sha256=after_hash,
                        reviewer=reviewer,
                        decisions=decisions,
                        review_path=review_path,
                    ),
                )
            except Exception:
                atomic_write_json(paths.transcript, dict(before_document))
                atomic_write_json(paths.autocorrection_state, before_state)
                raise
    finally:
        store.close()
    return after_hash


def synchronize_completed_review_state(
    *,
    paths: ReviewPaths,
    config: AutocorrectStageConfig,
    review_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Repair provenance for a transcript already authorised by manual review.

    v13.6.1 correctly registered the reviewed effective JSON hash in PostgreSQL,
    but did not refresh ``autocorrection_state.json``. This recovery only updates
    the state file after proving that PostgreSQL already authorises the current
    transcript and that the review artifact is completed.
    """

    if str(review_artifact.get("status") or "") != "completed":
        raise AutocorrectError("The manual autocorrection review is not completed")
    transcript = _read_json_mapping(paths.transcript, required=True)
    effective_hash = sha256_json(transcript)
    reviewer = str(review_artifact.get("reviewer") or "human-reviewer")
    decisions = [
        dict(item)
        for item in review_artifact.get("decisions") or []
        if isinstance(item, Mapping)
    ]
    store = Store(config.database_url, schema=config.schema)
    try:
        with store.transaction() as db:
            row = db.execute(
                "SELECT source_sha256, effective_sha256 FROM jobs WHERE job_id = ? FOR UPDATE",
                (paths.job_root.name,),
            ).fetchone()
            if row is None:
                raise AutocorrectError(
                    f"Autocorrect database has no job row for {paths.job_root.name}"
                )
            allowed = {str(row["source_sha256"] or ""), str(row["effective_sha256"] or "")}
            if effective_hash not in allowed:
                raise AutocorrectError(
                    "The reviewed transcript is not authorised by PostgreSQL; "
                    "refusing to repair the language-AI provenance state"
                )
            updated = _updated_autocorrection_state(
                paths=paths,
                effective_sha256=effective_hash,
                reviewer=reviewer,
                decisions=decisions,
                review_path=paths.review,
            )
            atomic_write_json(paths.autocorrection_state, updated)
            db.execute(
                """
                INSERT INTO events(
                    job_id, correction_id, action, actor_id,
                    payload_json, created_at
                ) VALUES (?, NULL, 'autocorrect_manual_review_state_repaired', ?, ?, ?)
                """,
                (
                    paths.job_root.name,
                    reviewer,
                    json.dumps(
                        {
                            "effective_sha256": effective_hash,
                            "authoritative_transcript_sha256": checksum(paths.transcript),
                            "review_artifact": str(paths.review),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    utc_now(),
                ),
            )
    finally:
        store.close()
    return {
        "job_id": paths.job_root.name,
        "status": "state_synchronized",
        "effective_sha256": effective_hash,
        "authoritative_transcript_sha256": checksum(paths.transcript),
        "autocorrection_state": str(paths.autocorrection_state),
    }


def _segment_contains_text(
    transcript: Mapping[str, Any], segment_id: str, text: str
) -> bool:
    segments = transcript.get("segments")
    if not isinstance(segments, list):
        return False
    wanted = normalize(text)
    for index, segment in enumerate(segments, start=1):
        if not isinstance(segment, Mapping):
            continue
        if _segment_id(segment, index) != segment_id:
            continue
        for field in ("source_text", "text", "display_text", "lexical_text"):
            value = segment.get(field)
            if isinstance(value, str) and wanted in normalize(value):
                return True
    return False


def apply_review_decisions(
    *,
    paths: ReviewPaths,
    config: AutocorrectStageConfig,
    review_artifact: dict[str, Any],
    research_artifact: Mapping[str, Any],
    reviewer: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    transcript = _read_json_mapping(paths.transcript, required=True)
    deterministic = (
        _read_json_mapping(paths.deterministic_transcript, required=True)
        if paths.deterministic_transcript.is_file()
        else copy.deepcopy(transcript)
    )
    decisions = [
        dict(item)
        for item in review_artifact.get("decisions") or []
        if isinstance(item, Mapping)
    ]
    corrections = [
        {
            "segment_id": str(item.get("segment_id") or ""),
            "raw_text": str(item.get("raw_text") or ""),
            "canonical_text": str(item.get("replacement_text") or ""),
            "entity_type": str(item.get("entity_type") or "other"),
            "confidence": 1.0,
            "source_urls": list(item.get("source_urls") or []),
            "grounded_source_urls": list(item.get("source_urls") or []),
            "story_match_terms": list(item.get("story_match_terms") or []),
            "reason": str(item.get("reason") or "Human reviewed correction"),
            "reviewed_by": reviewer,
            "reviewed_at": str(item.get("decided_at") or utc_now()),
        }
        for item in decisions
        if item.get("action") in {"approve", "edit"}
        and str(item.get("replacement_text") or "").strip()
    ]
    previous_name_research = copy.deepcopy(
        (transcript.get("autocorrect") or {}).get("name_research")
        if isinstance(transcript.get("autocorrect"), Mapping)
        else None
    )
    corrected, applied = apply_researched_corrections(transcript, corrections)
    intended = {
        (item["segment_id"], normalize(item["raw_text"])) for item in corrections
    }
    applied_keys = {
        (str(item.get("segment_id") or ""), normalize(str(item.get("raw_text") or "")))
        for item in applied
    }
    missing = intended - applied_keys
    unresolved_missing: set[tuple[str, str]] = set()
    for segment_id, raw_norm in missing:
        decision = next(
            (
                item
                for item in corrections
                if item["segment_id"] == segment_id
                and normalize(item["raw_text"]) == raw_norm
            ),
            None,
        )
        if decision is None or not _segment_contains_text(
            transcript, segment_id, str(decision.get("canonical_text") or "")
        ):
            unresolved_missing.add((segment_id, raw_norm))
    if unresolved_missing:
        missing_text = ", ".join(
            f"{segment}:{raw}" for segment, raw in sorted(unresolved_missing)
        )
        raise AutocorrectError(
            "Manual review could not locate every approved raw span in the current "
            f"authoritative transcript: {missing_text}"
        )

    autocorrect_meta = corrected.get("autocorrect")
    if not isinstance(autocorrect_meta, dict):
        autocorrect_meta = {}
        corrected["autocorrect"] = autocorrect_meta
    if previous_name_research is not None:
        autocorrect_meta["name_research"] = previous_name_research
    autocorrect_meta["manual_review"] = {
        "schema_version": MANUAL_REVIEW_SCHEMA,
        "reviewer": reviewer,
        "review_artifact": str(paths.review),
        "applied_count": len(applied),
        "applied_corrections": [
            {
                "segment_id": item.get("segment_id"),
                "raw_text": item.get("raw_text"),
                "canonical_text": item.get("canonical_text"),
                "entity_type": item.get("entity_type"),
                "reviewed_at": item.get("reviewed_at"),
            }
            for item in applied
        ],
        "rendered_at": utc_now(),
    }

    stale_files = _stale_artifacts(paths)
    summary = {
        "job_id": paths.job_root.name,
        "dry_run": dry_run,
        "applied_count": len(applied),
        "applied_corrections": applied,
        "keep_original_count": sum(
            1 for item in decisions if item.get("action") == "keep_original"
        ),
        "job_not_name_count": sum(
            1 for item in decisions if item.get("action") == "not_a_name_job"
        ),
        "project_not_name_count": sum(
            1 for item in decisions if item.get("action") == "not_a_name_project"
        ),
        "stale_artifact_count": len(stale_files),
        "stale_artifacts": stale_files,
        "effective_sha256_before": sha256_json(transcript),
        "effective_sha256_after": sha256_json(corrected),
    }
    if dry_run:
        return summary

    backup = _backup_paths(paths)

    review_artifact.update(
        {
            "schema_version": MANUAL_REVIEW_SCHEMA,
            "job_id": paths.job_root.name,
            "status": "applying",
            "reviewer": reviewer,
            "backup_directory": str(backup),
            "updated_at": utc_now(),
        }
    )
    _atomic_write_mapping(paths.review, review_artifact)

    try:
        effective_hash = _register_effective_transcript(
            config=config,
            paths=paths,
            before_document=transcript,
            after_document=corrected,
            reviewer=reviewer,
            decisions=decisions,
            review_path=paths.review,
        )
    except Exception as exc:
        review_artifact.update(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc)[:2000],
                "updated_at": utc_now(),
            }
        )
        _atomic_write_mapping(paths.review, review_artifact)
        raise

    suppressions_saved = _save_project_suppressions(paths.project_not_names, decisions)
    future_cache_saved = _save_future_corrections(
        paths=paths,
        config=config,
        research=research_artifact,
        deterministic_transcript=deterministic,
        decisions=decisions,
    )

    stale_artifact = {
        "schema_version": DOWNSTREAM_STALE_SCHEMA,
        "job_id": paths.job_root.name,
        "reason": "authoritative_english_transcript_changed_by_manual_autocorrect_review",
        "effective_sha256_before": sha256_json(transcript),
        "effective_sha256_after": effective_hash,
        "stale_artifacts": stale_files,
        "next_action": "rerun translation/editorial from the corrected authoritative transcript",
        "generated_at": utc_now(),
    }
    _atomic_write_mapping(paths.downstream_stale, stale_artifact)

    review_artifact.update(
        {
            "status": "completed",
            "completed_at": utc_now(),
            "effective_sha256_before": sha256_json(transcript),
            "effective_sha256_after": effective_hash,
            "applied_count": len(applied),
            "applied_corrections": applied,
            "project_suppressions_saved": suppressions_saved,
            "future_correction_cache_saved": future_cache_saved,
            "downstream_stale_artifact": str(paths.downstream_stale),
            "next_action": "rerun translation/editorial from the corrected authoritative transcript",
            "updated_at": utc_now(),
        }
    )
    _atomic_write_mapping(paths.review, review_artifact)
    return {
        **summary,
        "dry_run": False,
        "effective_sha256_after": effective_hash,
        "backup_directory": str(backup),
        "project_suppressions_saved": suppressions_saved,
        "future_correction_cache_saved": future_cache_saved,
        "review_artifact": str(paths.review),
        "downstream_stale_artifact": str(paths.downstream_stale),
        "next_action": "rerun translation/editorial from the corrected authoritative transcript",
    }


def _summary_table(decisions: Sequence[Mapping[str, Any]]) -> Table:
    table = Table(title="Review decisions", box=box.ROUNDED)
    table.add_column("Segment", style="cyan", no_wrap=True)
    table.add_column("Azure", style="red")
    table.add_column("Decision", style="bold")
    table.add_column("Replacement", style="green")
    table.add_column("Future reuse", justify="center")
    for item in decisions:
        table.add_row(
            str(item.get("segment_id") or ""),
            str(item.get("raw_text") or ""),
            str(item.get("action") or ""),
            str(item.get("replacement_text") or "—"),
            "yes" if item.get("learn_for_future_jobs") else "no",
        )
    return table


def _reviewer_name(explicit: str | None, console: Console) -> str:
    if explicit and explicit.strip():
        return explicit.strip()
    configured = os.getenv("MATHULA_TV_AUTOCORRECT_REVIEWER", "").strip()
    default = configured or getpass.getuser()
    return Prompt.ask("[bold]Reviewer name[/bold]", default=default, console=console).strip()


def run_console_review(
    *,
    job_id: str,
    reviewed_by: str | None = None,
    live_operation: bool = False,
    dry_run: bool = False,
    review_all: bool = False,
    console: Console | None = None,
) -> dict[str, Any]:
    console = console or Console()
    settings = load_settings()
    paths = ReviewPaths.for_job(settings.work_dir, job_id)
    if not paths.research.is_file():
        raise FileNotFoundError(
            f"No name-research artifact exists for {job_id}: {paths.research}"
        )
    transcript = _read_json_mapping(paths.transcript, required=True)
    research = _read_json_mapping(paths.research, required=True)
    reviewer = _reviewer_name(reviewed_by, console)
    review_artifact = _load_review(paths.review, job_id)
    decisions = _decision_map(review_artifact)
    queue = build_review_queue(
        research,
        transcript,
        existing_decisions=decisions,
        review_all=review_all,
    )

    header = Table.grid(padding=(0, 2))
    header.add_column(style="bold")
    header.add_column()
    header.add_row("Job", job_id)
    header.add_row("Research status", str(research.get("status") or "unknown"))
    header.add_row("Applied by agent", str(research.get("applied_count") or 0))
    header.add_row("Rejected", str(research.get("rejected_count") or 0))
    header.add_row("Unresolved", str(research.get("unresolved_count") or 0))
    header.add_row("Reviewer", reviewer)
    console.print(Panel(header, title="Mathula TV · Autocorrection Review", border_style="magenta"))

    if not queue:
        if str(review_artifact.get("status") or "") == "completed":
            state = _read_json_mapping(paths.autocorrection_state, required=True)
            current_file_hash = checksum(paths.transcript)
            if str(state.get("authoritative_transcript_sha256") or "") != current_file_hash:
                if not live_operation:
                    console.print(
                        "[yellow]The completed review is authorised, but the language-AI "
                        "provenance state is stale. Rerun with --live-operation to repair it.[/yellow]"
                    )
                    return {
                        "job_id": job_id,
                        "status": "state_sync_requires_live_operation",
                        "review_artifact": str(paths.review),
                    }
                config = AutocorrectStageConfig.from_environment(
                    Path(__file__).resolve().parents[2]
                )
                result = synchronize_completed_review_state(
                    paths=paths,
                    config=config,
                    review_artifact=review_artifact,
                )
                console.print(
                    "[green]Completed review provenance synchronized; translation may now continue.[/green]"
                )
                return result
        console.print("[green]No unresolved or rejected candidates remain for review.[/green]")
        return {
            "job_id": job_id,
            "status": "nothing_to_review",
            "review_artifact": str(paths.review),
        }

    quit_requested = False
    for index, item in enumerate(queue, start=1):
        console.rule(style="dim")
        console.print(_item_renderable(item, index, len(queue)))
        decision = prompt_for_decision(console, item, reviewer=reviewer)
        if decision is None:
            quit_requested = True
            break
        decisions[item.key] = decision.to_dict()
        _save_draft(
            paths,
            review_artifact,
            decisions,
            reviewer=reviewer,
            research_artifact=research,
        )

    ordered = list(decisions.values())
    console.print()
    console.print(_summary_table(ordered))
    if quit_requested:
        console.print(f"[yellow]Draft saved to {escape(str(paths.review))}[/yellow]")
        return {
            "job_id": job_id,
            "status": "draft_saved",
            "review_artifact": str(paths.review),
        }

    applicable = [
        item
        for item in ordered
        if item.get("action") in FINAL_DECISIONS
    ]
    if not applicable:
        console.print("[yellow]No final decisions were selected.[/yellow]")
        return {
            "job_id": job_id,
            "status": "draft_saved",
            "review_artifact": str(paths.review),
        }

    if not dry_run and not live_operation:
        console.print(
            "[yellow]Review saved as a draft. Rerun with --live-operation to apply "
            "authoritative transcript changes.[/yellow]"
        )
        return {
            "job_id": job_id,
            "status": "draft_saved_requires_live_operation",
            "review_artifact": str(paths.review),
        }

    if live_operation and not dry_run:
        if not Confirm.ask(
            "Apply these decisions to the authoritative transcript?",
            default=False,
            console=console,
        ):
            console.print("[yellow]No transcript changes were made.[/yellow]")
            return {
                "job_id": job_id,
                "status": "draft_saved",
                "review_artifact": str(paths.review),
            }

    config = AutocorrectStageConfig.from_environment(Path(__file__).resolve().parents[2])
    result = apply_review_decisions(
        paths=paths,
        config=config,
        review_artifact=review_artifact,
        research_artifact=research,
        reviewer=reviewer,
        dry_run=dry_run,
    )
    if dry_run:
        console.print(
            Panel(
                f"Would apply [bold]{result['applied_count']}[/bold] correction(s).\n"
                f"Would mark [bold]{result['project_not_name_count']}[/bold] project-wide not-name suppression(s).\n"
                f"Would invalidate [bold]{result['stale_artifact_count']}[/bold] downstream artifact file(s).",
                title="Dry-run result",
                border_style="yellow",
            )
        )
    else:
        console.print(
            Panel(
                f"Applied [bold green]{result['applied_count']}[/bold green] manual correction(s).\n"
                f"Saved [bold]{result['future_correction_cache_saved']}[/bold] future correction cache entry/entries.\n"
                f"Saved [bold]{result['project_suppressions_saved']}[/bold] project-wide not-name suppression(s).\n"
                f"Backup: {escape(result['backup_directory'])}\n\n"
                "[bold]Next:[/bold] rerun translation/editorial from the corrected authoritative transcript.",
                title="Review completed",
                border_style="green",
            )
        )
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="mathula-tv-review-autocorrect",
        description="Review unresolved and rejected name-research corrections in a Rich terminal UI.",
    )
    result.add_argument("job_id")
    result.add_argument("--reviewed-by")
    result.add_argument("--live-operation", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--review-all", action="store_true")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(list(argv) if argv is not None else None)
    try:
        run_console_review(
            job_id=args.job_id,
            reviewed_by=args.reviewed_by,
            live_operation=args.live_operation,
            dry_run=args.dry_run,
            review_all=args.review_all,
        )
    except (AutocorrectError, FileNotFoundError, ValueError, RuntimeError) as exc:
        Console(stderr=True).print(f"[bold red]error:[/bold red] {escape(str(exc))}")
        return 2
    except KeyboardInterrupt:
        Console(stderr=True).print("\n[yellow]Review interrupted; decisions already entered remain saved as a draft.[/yellow]")
        return 130
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "MANUAL_REVIEW_SCHEMA",
    "PROJECT_NOT_NAME_SCHEMA",
    "ReviewDecision",
    "ReviewItem",
    "ReviewPaths",
    "apply_review_decisions",
    "build_review_queue",
    "decision_key",
    "main",
    "run_console_review",
    "synchronize_completed_review_state",
]
