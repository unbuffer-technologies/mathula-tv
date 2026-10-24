"""Re-render a finished clip once, after the pronunciation rules or the shared memory changed.

    python -m mathula_tv.dub_rerender --job <job id> --run-id prod-2

Gives the job the memory's current readings, renders once (with the hook by default), rewrites the handoff files, and runs the
post-render check again. Nothing loops; the dub itself (the agent's submission) is not touched.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable


def rerender(settings: Any, *, job_id: str, run_id: str, clip_id: str | None = None, hook: bool = True,
             progress: Callable[[str], None] = print) -> dict[str, Any]:
    from .azure_stt import normalize
    from .cli import create_speech_backend
    from .dub_intelligibility import check_run
    from .dub_render import render_submission
    from .experience_store import ExperienceStore
    from .pronunciation_memory import apply_for_job, record_job_outcomes

    clip_id = clip_id or job_id[:8]
    work_dir = Path(settings.work_dir)
    job_root = work_dir / "jobs" / job_id
    report: dict[str, Any] = {"memory_applied": apply_for_job(job_root)}
    progress(f"[rerender] shared memory entries in this job: {report['memory_applied'] or 'none'}")
    report["render"] = render_submission(settings, run_id=run_id, clip_id=clip_id, repeat=1, hook=hook)
    if hook:
        from .dub_publish import publish_handoff

        report["handoff"] = publish_handoff(work_dir, job_id=job_id, run_id=run_id, clip_id=clip_id)
    run_dir = ExperienceStore(settings.work_dir).repeat_dir(run_id, clip_id, 1)
    report["intelligibility"] = check_run(run_dir, create_speech_backend(settings), normalize)
    report["memory_switched_off"] = record_job_outcomes(job_root, report["intelligibility"])
    progress(f"[rerender] dropped words after re-render: {report['intelligibility']['dropped_total']}")
    return report


def main(argv: list[str] | None = None) -> int:
    import argparse

    from .config import load_settings

    parser = argparse.ArgumentParser(prog="mathula_tv.dub_rerender")
    parser.add_argument("--job", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--clip-id")
    parser.add_argument("--no-hook", action="store_true")
    args = parser.parse_args(argv)
    rerender(load_settings(), job_id=args.job, run_id=args.run_id, clip_id=args.clip_id, hook=not args.no_hook)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
