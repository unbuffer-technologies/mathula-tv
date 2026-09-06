#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

from mathula_tv.atomic_io import atomic_write_json, read_json
from mathula_tv.config import load_settings
from mathula_tv.tiktok_editor import (
    TIKTOK_HOOK_PROMPT_VERSION,
    caption_without_hashtags,
)


def _updated_seo(value: Mapping[str, Any], title: str) -> dict[str, Any]:
    updated = dict(value)
    updated["title"] = title
    updated["cover_hook"] = title
    nested = updated.get("tiktok")
    if isinstance(nested, Mapping):
        nested_updated = dict(nested)
        nested_updated["cover_hook"] = title
        updated["tiktok"] = nested_updated
    flags = list(updated.get("human_review_flags") or [])
    flags.append("manual_publication_title_override")
    updated["human_review_flags"] = list(dict.fromkeys(flags))
    return updated


def override_publication_title(*, job_root: Path, title: str) -> Path:
    title = caption_without_hashtags(title)
    if not title:
        raise ValueError("Publication title cannot be empty")
    if len(title) > 90:
        raise ValueError("Publication title must be at most 90 characters")

    job_id = job_root.name
    manifest_path = job_root / "output" / f"tiktok_edit_manifest_{job_id}.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    manifest = read_json(manifest_path)
    if not isinstance(manifest.get("question_anchor_guard"), Mapping):
        raise ValueError("Publication manifest has no protected opening decision")
    if not isinstance(manifest.get("editorial_seo"), Mapping):
        raise ValueError("Publication manifest has no editorial SEO package")

    manifest["selection_prompt_version"] = TIKTOK_HOOK_PROMPT_VERSION
    manifest["hook_text"] = title
    manifest["render_version"] = "manual-title-override-pending-render"
    manifest["editorial_seo"] = _updated_seo(manifest["editorial_seo"], title)
    manifest["title_person_prefix_policy"] = {
        "policy_version": "manual-publication-title-override-v1",
        "input_title": title,
        "selected_title": title,
        "target_language": manifest["editorial_seo"].get("language"),
        "lead_type": "manual",
        "lead_text": title.split(":", 1)[0] if ":" in title else "",
        "prefix_applied": False,
        "manual_override": True,
    }
    title_panel = manifest.get("title_panel")
    if isinstance(title_panel, Mapping):
        panel = dict(title_panel)
        panel["text"] = title
        manifest["title_panel"] = panel
    atomic_write_json(manifest_path, manifest)

    for seo_path in (job_root / "translation").glob("tiktok_*.json"):
        atomic_write_json(seo_path, _updated_seo(read_json(seo_path), title))
    for seo_path in (job_root / "output").glob(f"tiktok_seo_*_{job_id}.json"):
        atomic_write_json(seo_path, _updated_seo(read_json(seo_path), title))
    for cover_path in (job_root / "output").glob(f"tiktok_cover_*_{job_id}.txt"):
        cover_path.write_text(title + "\n", encoding="utf-8")

    hook_path = job_root / "direct_dub" / "tiktok_hook.txt"
    hook_path.parent.mkdir(parents=True, exist_ok=True)
    hook_path.write_text(title + "\n", encoding="utf-8")
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Override a publication title without calling the editorial AI."
    )
    parser.add_argument("job_id")
    parser.add_argument("--title", required=True)
    parser.add_argument("--work-dir", type=Path)
    args = parser.parse_args()

    work_dir = args.work_dir or load_settings().work_dir
    manifest_path = override_publication_title(
        job_root=work_dir / "jobs" / args.job_id,
        title=args.title,
    )
    print(f"Updated publication title: {args.title}")
    print(f"Manifest: {manifest_path}")
    print("Run dub-azure for this job to rerender the card with zero editorial AI calls.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
