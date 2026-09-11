from __future__ import annotations

import json
from pathlib import Path

import pytest

from mathula_tv.native_dub import (
    CONTEXT_LEDGER_SYSTEM_PROMPT,
    PASS1_SYSTEM_PROMPT,
    _load_case_state_context,
    _normalize_context_ledger,
)


# 2026-08-29: native_phrase_0113/0123's attribution-swap bug traced back to "Witness G" --
# an anonymized, repeatedly-referenced party -- never being captured as a context_ledger
# entity at all (only the transcript's ~6 fully-named principals were). Both prompts that
# can generate the ledger (Pass 1's own combined call, and the standalone fallback for old
# checkpoints) need the same instruction to capture generic/anonymized referents too.

def test_pass1_prompt_instructs_capturing_anonymized_repeatedly_referenced_entities():
    normalized = " ".join(PASS1_SYSTEM_PROMPT.split())
    assert "Witness G" in normalized
    assert "not only fully-named principals" in normalized


def test_context_ledger_fallback_prompt_instructs_capturing_anonymized_entities():
    normalized = " ".join(CONTEXT_LEDGER_SYSTEM_PROMPT.split())
    assert "Witness G" in normalized
    assert "not only fully-named principals" in normalized


# --- _load_case_state_context ------------------------------------------------
# 2026-08-29: master_case_state.json (built by scripts/sync_madlanga_case.py,
# migrated from the sibling commission-ai project) lives at
# working/case_state/, a PROJECT-level sibling of working/jobs/ -- two levels
# up from a job's own job_root. Never a hard requirement: returns None when
# absent so jobs without a case-state file keep working exactly as before.


def test_load_case_state_context_returns_none_when_no_case_state_file(tmp_path: Path):
    job_root = tmp_path / "working" / "jobs" / "some-job"
    job_root.mkdir(parents=True)
    assert _load_case_state_context(job_root=job_root) is None


def test_load_case_state_context_loads_and_compacts_real_file(tmp_path: Path):
    job_root = tmp_path / "working" / "jobs" / "some-job"
    job_root.mkdir(parents=True)
    case_state_dir = tmp_path / "working" / "case_state"
    case_state_dir.mkdir(parents=True)
    (case_state_dir / "master_case_state.json").write_text(
        json.dumps({
            "case_key": "madlanga-commission",
            "standing_threads": [
                {
                    "thread_id": "witness-g-thread",
                    "title": "Testimony of Confidential Intelligence Witness G",
                    "status": "developing",
                    "why_it_matters": "Sensitive intelligence evidence.",
                    "people": ["Witness G"],
                },
            ],
        }),
        encoding="utf-8",
    )
    compact = _load_case_state_context(job_root=job_root)
    assert compact is not None
    assert compact["case_key"] == "madlanga-commission"
    assert any(t["title"] == "Testimony of Confidential Intelligence Witness G" for t in compact["standing_threads"])


def _segments():
    return [
        {"segment_id": segment_id, "speaker_id": "SPEAKER_00", "source_text": "x"}
        for segment_id in (
            "seg-00060", "seg-00064", "seg-00066", "seg-00068", "seg-00072", "seg-00073"
        )
    ]


def test_story_beat_can_span_more_than_four_segments_and_is_canonicalized():
    ledger = _normalize_context_ledger(
        {
            "summary": "x",
            "entities": [],
            "speaker_roles": [],
            "story_beats": [
                {
                    "segment_ids": [
                        "seg-00072", "seg-00064", "seg-00068",
                        "seg-00064", "seg-00060", "seg-00066",
                    ],
                    "note": "beat",
                }
            ],
        },
        _segments(),
    )
    assert ledger["story_beats"][0]["segment_ids"] == [
        "seg-00060", "seg-00064", "seg-00066", "seg-00068", "seg-00072"
    ]


def test_story_beat_still_rejects_unknown_segment_ids():
    with pytest.raises(ValueError, match="unknown story-beat segment IDs"):
        _normalize_context_ledger(
            {
                "summary": "x",
                "entities": [],
                "speaker_roles": [],
                "story_beats": [{"segment_ids": ["seg-99999"], "note": "bad"}],
            },
            _segments(),
        )
