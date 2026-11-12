"""Re-render a finished clip once, after the pronunciation rules changed.

    python -m mathula_tv.dub_rerender --job <job id> --run-id prod-2

Writes the job's pronunciations again from the run's text (the dictionary's sounds through the rules, plus the spellings chosen by ear), renders once
(with the hook by default) and rewrites the handoff files. Nothing loops; the dub itself (the agent's submission) is not touched, and no model listens.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable


def rerender(settings: Any, *, job_id: str, run_id: str, clip_id: str | None = None, hook: bool = True,
             progress: Callable[[str], None] = print) -> dict[str, Any]:
    from .dub_render import render_submission
    from .experience_store import ExperienceStore
    from .job_pronunciations import build as build_job_pronunciations

    clip_id = clip_id or job_id[:8]
    work_dir = Path(settings.work_dir)
    job_root = work_dir / "jobs" / job_id
    submission = json.loads((ExperienceStore(settings.work_dir).repeat_dir(run_id, clip_id, 1) / "submission.json").read_text(encoding="utf-8"))
    spelled = build_job_pronunciations(job_root, [str(i.get("zulu_text") or "") for i in submission.get("items", [])], progress=progress)
    report: dict[str, Any] = {"pronunciations": [{"word": s["word"], "spelling": s["spelling"], "reading": s["reading"]} for s in spelled]}
    progress(f"[rerender] pronunciations written by rule for {len(spelled)} word(s)")
    report["render"] = render_submission(settings, run_id=run_id, clip_id=clip_id, repeat=1, hook=hook)
    if hook:
        from .dub_publish import publish_handoff

        report["handoff"] = publish_handoff(work_dir, job_id=job_id, run_id=run_id, clip_id=clip_id)
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
