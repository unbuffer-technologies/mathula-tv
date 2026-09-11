"""Tests for master_case_context.py (migrated 2026-08-29 from the sibling
commission-ai project, unchanged logic) -- the compaction helpers that turn
working/case_state/master_case_state.json into a token-budget-bounded payload
for native_dub.py's translation/QA context.
"""
from __future__ import annotations

import json
from pathlib import Path

from mathula_tv import master_case_context as mcc


def test_default_case_state_path_prefers_case_state_subdir(tmp_path: Path):
    (tmp_path / "case_state").mkdir()
    target = tmp_path / "case_state" / "master_case_state.json"
    target.write_text("{}", encoding="utf-8")
    assert mcc.default_case_state_path(tmp_path) == target


def test_default_case_state_path_returns_none_when_missing(tmp_path: Path):
    assert mcc.default_case_state_path(tmp_path) is None


def test_load_master_case_state_returns_empty_dict_for_missing_file(tmp_path: Path):
    assert mcc.load_master_case_state(tmp_path) == {}


def test_load_master_case_state_reads_real_file(tmp_path: Path):
    (tmp_path / "case_state").mkdir()
    data = {"case_key": "madlanga-commission", "standing_threads": []}
    (tmp_path / "case_state" / "master_case_state.json").write_text(
        json.dumps(data), encoding="utf-8",
    )
    assert mcc.load_master_case_state(tmp_path) == data


def test_extract_case_registry_collects_people_with_aliases():
    case_state = {
        "standing_threads": [
            {
                "thread_id": "witness-g-thread",
                "people": ["Witness G", {"name": "Mr Michaels", "aliases": ["Michaels"]}],
            },
        ],
    }
    registry = mcc.extract_case_registry(case_state)
    assert "Witness G" in registry["people"]
    assert "Mr Michaels" in registry["people"]
    assert "Michaels" in registry["people"]["Mr Michaels"]


def test_compact_case_context_includes_standing_threads_and_people():
    case_state = {
        "case_key": "madlanga-commission",
        "standing_threads": [
            {
                "thread_id": "witness-g-security-concerns-and-intelligence-evidence",
                "title": "Testimony of Confidential Intelligence Witness G",
                "status": "developing",
                "why_it_matters": "Management of sensitive intelligence evidence.",
                "people": ["Witness G", "Mr Michaels"],
            },
        ],
        "open_questions": ["Who directed Witness G to meet at Orlando SAPS?"],
    }
    compact = mcc.compact_case_context(case_state)
    assert compact["case_key"] == "madlanga-commission"
    thread_titles = [t["title"] for t in compact["standing_threads"]]
    assert "Testimony of Confidential Intelligence Witness G" in thread_titles
    assert "Who directed Witness G to meet at Orlando SAPS?" in compact["open_questions"]


def test_compact_case_context_shrinks_when_over_max_chars():
    threads = [
        {
            "thread_id": f"thread-{i}",
            "title": f"Thread {i}" * 20,
            "status": "developing",
            "why_it_matters": "x" * 200,
            "people": [f"Person {i}"],
        }
        for i in range(50)
    ]
    case_state = {"case_key": "big-case", "standing_threads": threads}
    compact = mcc.compact_case_context(case_state, max_chars=2000)
    raw = json.dumps(compact, ensure_ascii=False)
    assert len(raw) <= 2000 or len(compact["standing_threads"]) < len(threads)


def test_compact_case_context_handles_empty_state():
    assert mcc.compact_case_context({}) == {
        "case_key": None,
        "trusted_entities": {"people": [], "organizations": [], "companies": [], "locations": []},
        "standing_threads": [],
        "important_relationships": [],
        "timeline_events": [],
        "open_questions": [],
        "contradictions": [],
        "source_refs": [],
    }
