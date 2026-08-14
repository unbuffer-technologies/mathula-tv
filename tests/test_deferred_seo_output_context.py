from __future__ import annotations

import json
from pathlib import Path

from mathula_tv.output_naming import load_seo_output_context


def test_missing_seo_is_deferred_without_ai(tmp_path: Path) -> None:
    job_root = tmp_path / "17d8715b86ce4eb7977f91af455c18d5"

    context = load_seo_output_context(job_root, "zu-ZA")

    assert context.title.startswith("Mathula TV")
    assert context.seo_path.is_file()
    seo = json.loads(context.seo_path.read_text(encoding="utf-8"))
    assert seo["status"] == "pending_editorial_call"
    assert seo["ai_generation"] is None
