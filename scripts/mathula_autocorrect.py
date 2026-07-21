#!/usr/bin/env python3
"""Mathula TV production autocorrect CLI backed by PostgreSQL."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from mathula_tv.autocorrect import (
    AutocorrectError,
    JobPaths,
    Store,
    apply_ai_advice,
    atomic_write_json,
    build_ai_request,
    change_correction,
    detect_candidates,
    ensure_job,
    export_adaptive_phrases,
    list_corrections,
    load_json,
    moderate_public_term,
    platform_candidate_report,
    questions,
    render_effective_transcript,
    reset_source,
    undo_all_auto_applied,
)
from mathula_tv.autocorrect_ai import rank_with_anthropic


def emit(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def database_url(explicit: str | None) -> str:
    value = (
        explicit
        or os.environ.get("MATHULA_TV_DATABASE_URL")
        or os.environ.get("DATABASE_URL")
    )
    if not value:
        raise AutocorrectError(
            "Set MATHULA_TV_DATABASE_URL or DATABASE_URL"
        )
    return value


def store_for(args: argparse.Namespace) -> Store:
    return Store(
        database_url(args.database_url),
        schema=args.database_schema,
    )


def common(args: argparse.Namespace) -> tuple[Store, JobPaths]:
    paths = JobPaths.for_job(
        args.working_root,
        args.job_id,
        transcript_relative_path=args.transcript,
    )
    return store_for(args), paths


def command_migrate(args: argparse.Namespace) -> int:
    store = store_for(args)
    emit(
        {
            "database": "connected",
            "schema": store.schema,
            "schema_version": store.CURRENT_SCHEMA_VERSION,
        }
    )
    return 0


def command_prepare(args: argparse.Namespace) -> int:
    store, paths = common(args)
    job = ensure_job(
        store,
        paths=paths,
        job_id=args.job_id,
        user_id=args.user_id,
        project_id=args.project_id,
    )
    source = load_json(paths.snapshot)
    correction_ids = detect_candidates(
        store,
        job_id=args.job_id,
        user_id=args.user_id or job.get("user_id"),
        project_id=args.project_id or job.get("project_id"),
        source_document=source,
        hints_path=args.hints,
        registry_path=args.registry,
        fuzzy_threshold=args.fuzzy_threshold,
        max_candidates=args.max_candidates,
        low_confidence_threshold=args.low_confidence_threshold,
        auto_confirmation_threshold=args.auto_confirmation_threshold,
    )

    ai_used = False
    if args.ai_model:
        request = build_ai_request(
            store,
            job_id=args.job_id,
            source_document=source,
            maximum=args.max_questions,
        )
        atomic_write_json(paths.ai_request, request)
        if request["spans"]:
            advice, metadata = rank_with_anthropic(
                request,
                model=args.ai_model,
            )
            atomic_write_json(paths.ai_response, metadata)
            apply_ai_advice(
                store,
                job_id=args.job_id,
                advice=advice,
            )
            ai_used = True

    state = render_effective_transcript(
        store,
        paths=paths,
        job_id=args.job_id,
    )
    emit(
        {
            "job_id": args.job_id,
            "detected_count": len(correction_ids),
            "ai_used": ai_used,
            "auto_applied": state["status_counts"].get(
                "auto_applied", 0
            ),
            "needs_review": state["review_count"],
            "state_artifact": str(paths.state),
            "effective_transcript": str(paths.effective),
            "source_snapshot": str(paths.snapshot),
        }
    )
    return 0


def command_list(args: argparse.Namespace) -> int:
    store, paths = common(args)
    emit(
        {
            "job_id": args.job_id,
            "state_artifact": str(paths.state),
            "corrections": list_corrections(
                store,
                job_id=args.job_id,
            ),
        }
    )
    return 0


def command_questions(args: argparse.Namespace) -> int:
    store, _paths = common(args)
    emit(
        {
            "job_id": args.job_id,
            "questions": questions(
                store,
                job_id=args.job_id,
                maximum=args.maximum,
            ),
        }
    )
    return 0


def change(args: argparse.Namespace, action: str) -> int:
    store, paths = common(args)
    row = change_correction(
        store,
        job_id=args.job_id,
        correction_id=args.correction_id,
        action=action,
        actor_id=args.actor_id,
        replacement_text=getattr(args, "text", None),
        learn_scopes=getattr(args, "learn_scope", []),
        user_id=getattr(args, "user_id", None),
        project_id=getattr(args, "project_id", None),
        always=getattr(args, "always", False),
        never=getattr(args, "never", False),
        contribute_globally=getattr(
            args,
            "contribute_publicly",
            False,
        ),
        term_category=getattr(args, "term_category", "other"),
        language=getattr(args, "language", "en-ZA"),
        domain=getattr(args, "domain", None),
        consent_version=getattr(
            args,
            "consent_version",
            "public-dictionary-v1",
        ),
    )
    state = render_effective_transcript(
        store,
        paths=paths,
        job_id=args.job_id,
    )
    emit(
        {
            "correction": row,
            "status_counts": state["status_counts"],
        }
    )
    return 0


def command_accept(args: argparse.Namespace) -> int:
    return change(args, "accept")


def command_new_term(args: argparse.Namespace) -> int:
    return change(args, "edit")


def command_revert(args: argparse.Namespace) -> int:
    return change(args, "revert")


def command_redo(args: argparse.Namespace) -> int:
    return change(args, "redo")


def command_dismiss(args: argparse.Namespace) -> int:
    return change(args, "dismiss")


def command_undo_all(args: argparse.Namespace) -> int:
    store, paths = common(args)
    count = undo_all_auto_applied(
        store,
        job_id=args.job_id,
        actor_id=args.actor_id,
    )
    state = render_effective_transcript(
        store,
        paths=paths,
        job_id=args.job_id,
    )
    emit(
        {
            "job_id": args.job_id,
            "reverted_auto_applied": count,
            "status_counts": state["status_counts"],
        }
    )
    return 0


def command_render(args: argparse.Namespace) -> int:
    store, paths = common(args)
    emit(
        render_effective_transcript(
            store,
            paths=paths,
            job_id=args.job_id,
        )
    )
    return 0


def command_reset_source(args: argparse.Namespace) -> int:
    store, paths = common(args)
    emit(
        reset_source(
            store,
            paths=paths,
            job_id=args.job_id,
            actor_id=args.actor_id,
        )
    )
    return 0


def command_export_phrases(args: argparse.Namespace) -> int:
    store, paths = common(args)
    value = export_adaptive_phrases(
        store,
        job_id=args.job_id,
        user_id=args.user_id,
        project_id=args.project_id,
        maximum=args.maximum,
    )
    output = (
        args.output
        or paths.analysis / "adaptive_stt_phrases.json"
    )
    atomic_write_json(output, value)
    emit({"output": str(output), **value})
    return 0


def command_public_report(args: argparse.Namespace) -> int:
    store = store_for(args)
    emit(
        {
            "candidates": platform_candidate_report(
                store,
                minimum_users=args.minimum_users,
                minimum_jobs=args.minimum_jobs,
                status=args.status,
            )
        }
    )
    return 0


def command_public_moderate(
    args: argparse.Namespace,
    action: str,
) -> int:
    store = store_for(args)
    emit(
        moderate_public_term(
            store,
            term_id=args.term_id,
            action=action,
            moderator_id=args.moderator_id,
            minimum_users=args.minimum_users,
            minimum_jobs=args.minimum_jobs,
            force=args.force,
        )
    )
    return 0


def command_public_approve(args: argparse.Namespace) -> int:
    return command_public_moderate(args, "approve")


def command_public_reject(args: argparse.Namespace) -> int:
    return command_public_moderate(args, "reject")


def add_database_options(command: argparse.ArgumentParser) -> None:
    command.add_argument(
        "--database-url",
        help=(
            "PostgreSQL DSN. Defaults to MATHULA_TV_DATABASE_URL "
            "then DATABASE_URL."
        ),
    )
    command.add_argument(
        "--database-schema",
        default=os.environ.get(
            "MATHULA_TV_DATABASE_SCHEMA",
            "mathula_autocorrect",
        ),
    )


def add_job_options(command: argparse.ArgumentParser) -> None:
    command.add_argument("job_id")
    command.add_argument(
        "--working-root",
        type=Path,
        default=Path("working"),
    )
    command.add_argument(
        "--transcript",
        default="analysis/transcript_en.json",
    )
    add_database_options(command)


def add_learning_options(command: argparse.ArgumentParser) -> None:
    command.add_argument("--user-id")
    command.add_argument("--project-id")
    command.add_argument(
        "--learn-scope",
        action="append",
        choices=("job", "user", "project"),
        default=[],
    )
    command.add_argument("--always", action="store_true")
    command.add_argument(
        "--contribute-publicly",
        action="store_true",
        help=(
            "Contribute anonymised evidence to the moderated public "
            "autocorrect dictionary."
        ),
    )
    command.add_argument(
        "--term-category",
        choices=(
            "slang",
            "phrase",
            "brand",
            "place",
            "person",
            "organisation",
            "other",
        ),
        default="other",
    )
    command.add_argument("--language", default="en-ZA")
    command.add_argument("--domain")
    command.add_argument(
        "--consent-version",
        default="public-dictionary-v1",
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description=(
            "Reversible PostgreSQL-backed autocorrect for Mathula TV"
        )
    )
    commands = root.add_subparsers(dest="command", required=True)

    migrate = commands.add_parser("migrate")
    add_database_options(migrate)
    migrate.set_defaults(handler=command_migrate)

    prepare = commands.add_parser("prepare")
    add_job_options(prepare)
    prepare.add_argument("--user-id")
    prepare.add_argument("--project-id")
    prepare.add_argument(
        "--hints",
        type=Path,
        default=Path("config/stt_confusion_overrides.json"),
    )
    prepare.add_argument(
        "--registry",
        type=Path,
        default=Path("config/entity_registry.json"),
    )
    prepare.add_argument("--fuzzy-threshold", type=float, default=0.68)
    prepare.add_argument("--max-candidates", type=int, default=4)
    prepare.add_argument("--max-questions", type=int, default=5)
    prepare.add_argument(
        "--low-confidence-threshold",
        type=float,
        default=0.72,
    )
    prepare.add_argument(
        "--auto-confirmation-threshold",
        type=int,
        default=2,
    )
    prepare.add_argument("--ai-model")
    prepare.set_defaults(handler=command_prepare)

    list_command = commands.add_parser("list")
    add_job_options(list_command)
    list_command.set_defaults(handler=command_list)

    question_command = commands.add_parser("questions")
    add_job_options(question_command)
    question_command.add_argument("--maximum", type=int, default=5)
    question_command.set_defaults(handler=command_questions)

    accept = commands.add_parser("accept")
    add_job_options(accept)
    accept.add_argument("correction_id")
    accept.add_argument("--actor-id", required=True)
    add_learning_options(accept)
    accept.set_defaults(handler=command_accept)

    new_term = commands.add_parser("new-term")
    add_job_options(new_term)
    new_term.add_argument("correction_id")
    new_term.add_argument("--actor-id", required=True)
    new_term.add_argument("--text", required=True)
    add_learning_options(new_term)
    new_term.set_defaults(handler=command_new_term)

    revert = commands.add_parser("revert")
    add_job_options(revert)
    revert.add_argument("correction_id")
    revert.add_argument("--actor-id", required=True)
    add_learning_options(revert)
    revert.add_argument("--never", action="store_true")
    revert.set_defaults(handler=command_revert)

    for name, handler in (
        ("redo", command_redo),
        ("dismiss", command_dismiss),
    ):
        command = commands.add_parser(name)
        add_job_options(command)
        command.add_argument("correction_id")
        command.add_argument("--actor-id", required=True)
        command.set_defaults(handler=handler)

    undo = commands.add_parser("undo-all")
    add_job_options(undo)
    undo.add_argument("--actor-id", required=True)
    undo.set_defaults(handler=command_undo_all)

    render = commands.add_parser("render")
    add_job_options(render)
    render.set_defaults(handler=command_render)

    reset = commands.add_parser("reset-source")
    add_job_options(reset)
    reset.add_argument("--actor-id", required=True)
    reset.set_defaults(handler=command_reset_source)

    phrases = commands.add_parser("export-phrases")
    add_job_options(phrases)
    phrases.add_argument("--user-id")
    phrases.add_argument("--project-id")
    phrases.add_argument("--maximum", type=int, default=100)
    phrases.add_argument("--output", type=Path)
    phrases.set_defaults(handler=command_export_phrases)

    public_report = commands.add_parser("public-report")
    add_database_options(public_report)
    public_report.add_argument("--minimum-users", type=int, default=5)
    public_report.add_argument("--minimum-jobs", type=int, default=5)
    public_report.add_argument(
        "--status",
        choices=("candidate", "approved", "rejected"),
    )
    public_report.set_defaults(handler=command_public_report)

    for name, handler in (
        ("approve-public-term", command_public_approve),
        ("reject-public-term", command_public_reject),
    ):
        command = commands.add_parser(name)
        add_database_options(command)
        command.add_argument("term_id")
        command.add_argument("--moderator-id", required=True)
        command.add_argument("--minimum-users", type=int, default=5)
        command.add_argument("--minimum-jobs", type=int, default=5)
        command.add_argument("--force", action="store_true")
        command.set_defaults(handler=handler)

    return root


def main() -> int:
    load_dotenv(".env", override=False)
    args = parser().parse_args()
    try:
        return int(args.handler(args))
    except (AutocorrectError, FileNotFoundError) as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
