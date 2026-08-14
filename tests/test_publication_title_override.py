from __future__ import annotations

import json
from pathlib import Path

from scripts.override_publication_title import override_publication_title


def test_override_updates_manifest_and_seo_without_ai(tmp_path: Path) -> None:
    job_id = "job-title"
    root = tmp_path / "jobs" / job_id
    manifest_path = root / "output" / f"tiktok_edit_manifest_{job_id}.json"
    seo_path = root / "translation" / "tiktok_zu.json"
    manifest_path.parent.mkdir(parents=True)
    seo_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "selection_prompt_version": "old",
                "hook_text": "Kass: old",
                "render_version": "old",
                "question_anchor_guard": {},
                "editorial_seo": {
                    "language": "zu-ZA",
                    "title": "Kass: old",
                    "cover_hook": "Kass: old",
                },
                "title_panel": {"text": "Kass: old"},
            }
        ),
        encoding="utf-8",
    )
    seo_path.write_text(
        json.dumps({"title": "Kass: old", "cover_hook": "Kass: old"}),
        encoding="utf-8",
    )

    title = "UKass: ukwesula akumkhululi uJohnson eKhomishini kaMadlanga nasenkantolo"
    override_publication_title(job_root=root, title=title)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seo = json.loads(seo_path.read_text(encoding="utf-8"))
    assert manifest["hook_text"] == title
    assert manifest["title_panel"]["text"] == title
    assert manifest["render_version"] == "manual-title-override-pending-render"
    assert manifest["title_person_prefix_policy"]["manual_override"] is True
    assert seo["cover_hook"] == title
    assert (root / "direct_dub" / "tiktok_hook.txt").read_text(encoding="utf-8").strip() == title
