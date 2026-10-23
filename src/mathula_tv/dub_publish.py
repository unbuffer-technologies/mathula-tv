"""Hand a finished dub over for posting, named the way the production pipeline always named it.

After a render with the hook and footer overlay, this writes into ``<job>/output/``:

    final_dubbed_<job_id>.mp4              the video to post (hook edit + footer overlay)
    dubbed_master_<job_id>.mp4             the clean dubbed master, no overlay
    tiktok_caption_zu_<job_id>.txt         caption, a blank line, then the hashtags: paste-ready on a phone
    tiktok_hashtags_zu_<job_id>.txt, tiktok_cover_zu_<job_id>.txt, tiktok_search_keywords_zu_<job_id>.txt,
    tiktok_seo_zu_<job_id>.json

It reuses ``output_naming`` unchanged (account hashtags from ``config/tiktok_accounts.json``, at most five, community
hashtag first), feeding it the editorial package the native hook step already wrote.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .atomic_io import atomic_write_json


def publish_handoff(work_dir: Path, *, job_id: str, run_id: str, clip_id: str, repeat: int = 1,
                    target_language: str = "zu-ZA") -> dict[str, Any]:
    from .experience_store import ExperienceStore
    from .native_dub import native_dub_paths
    from .output_naming import language_code, load_seo_output_context, publish_dubbed_master_outputs, publish_title_named_outputs

    render_root = ExperienceStore(Path(work_dir)).repeat_dir(run_id, clip_id, repeat) / "render"
    paths = native_dub_paths(render_root)
    for needed in (paths.publication_video, paths.video, paths.seo_package):
        if not Path(needed).is_file():
            raise FileNotFoundError(f"{needed} is missing: render the run with the hook first (dub_render --hook)")
    seo = json.loads(paths.seo_package.read_text(encoding="utf-8"))
    hook = paths.hook_text.read_text(encoding="utf-8").strip() if paths.hook_text.is_file() else ""
    seo = {**seo, "cover_hook": seo.get("cover_hook") or hook or seo.get("title")}

    job_root = Path(work_dir) / "jobs" / job_id
    code = language_code(target_language)
    atomic_write_json(job_root / "translation" / f"tiktok_{code}.json", seo)       # where output_naming expects the SEO
    context = load_seo_output_context(job_root, target_language)
    final = publish_title_named_outputs(job_root, paths.publication_video, context=context)
    master = publish_dubbed_master_outputs(job_root, paths.video, context=context)
    return {"job_id": job_id, "final_video": str(final.final_video), "master_video": str(master.final_video),
            "files": {key: str(path) for key, path in sorted(final.seo_files.items())}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mathula_tv.dub_publish")
    parser.add_argument("--job", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--clip", default=None, help="default: the first 8 characters of the job id")
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args(argv)

    from .config import load_settings

    result = publish_handoff(Path(load_settings().work_dir), job_id=args.job, run_id=args.run,
                             clip_id=args.clip or args.job[:8], repeat=args.repeat)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
