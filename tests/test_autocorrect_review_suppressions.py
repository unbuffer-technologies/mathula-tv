from __future__ import annotations

import json
from pathlib import Path

from mathula_tv.autocorrect_research import _manual_not_name_suppressions


def test_manual_not_name_suppressions_load_job_and_project(tmp_path: Path) -> None:
    job_root = tmp_path / "working" / "jobs" / "job1"
    analysis = job_root / "analysis"
    analysis.mkdir(parents=True)
    (analysis / "autocorrection_manual_review.json").write_text(
        json.dumps(
            {
                "decisions": [
                    {"action": "not_a_name_job", "raw_text": "Bongi"},
                    {"action": "keep_original", "raw_text": "Other"},
                ]
            }
        ),
        encoding="utf-8",
    )
    cache = tmp_path / "working" / "cache" / "autocorrect_name_research" / "v1"
    cache.mkdir(parents=True)
    (cache / "not_name_suppressions.json").write_text(
        json.dumps(
            {
                "schema_version": "mathula-autocorrect-not-name-suppressions-v1",
                "suppressions": [{"raw_text": "NDPPs"}],
            }
        ),
        encoding="utf-8",
    )
    assert _manual_not_name_suppressions(job_root) == {"bongi", "ndpps"}
