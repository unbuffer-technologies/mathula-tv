from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import mathula_tv.native_dub as native_dub
from mathula_tv.native_dub import _apply_referent_audit
from mathula_tv.referent_audit import REFERENT_AUDIT_SCHEMA_VERSION


class _FakeProvider:
    def __init__(self, responses):
        self._responses = list(responses)
        self.config = SimpleNamespace(max_output_tokens=8192)
        self.calls: list[str] = []

    def complete_json(self, *, operation, system_prompt, payload, schema, max_output_tokens=None):
        self.calls.append(operation)
        expected_operation, data = self._responses.pop(0)
        assert operation == expected_operation
        return SimpleNamespace(data=data, model="grok-test", input_tokens=10, output_tokens=5, attempts=1)


def _pass2(referent_audit_summary=None) -> dict:
    return {
        "input_sha256": "abc123",
        "translations": [{"segment_id": "seg-1", "source_text": "Yes.", "spoken_text": "Yebo."}],
        "referent_audit_summary": referent_audit_summary,
    }


def test_matching_current_schema_checkpoint_is_reused(tmp_path):
    pass2 = _pass2(
        {
            "schema_version": REFERENT_AUDIT_SCHEMA_VERSION,
            "input_sha256": "abc123",
            "groups_checked": 1,
            "groups_flagged": 0,
            "repairs_applied": 0,
            "unresolved_group_ids": [],
            "usage": {"input_tokens": 0, "output_tokens": 0, "attempts": 0},
        }
    )
    provider = _FakeProvider([])  # must not be called at all

    result = _apply_referent_audit(
        job_root=tmp_path,
        provider=provider,
        pass2=pass2,
        force=False,
        progress=None,
        batch_size=12,
        max_repair_rounds=1,
    )

    assert provider.calls == []
    assert result is pass2


def test_older_schema_checkpoint_triggers_a_fresh_audit_without_force(tmp_path):
    pass2 = _pass2(
        {
            # Old shape from before checked_terms/groups_flagged existed.
            "schema_version": "mathula-native-referent-audit-v1",
            "input_sha256": "abc123",
            "groups_checked": 1,
            "issues_found": 0,
            "repairs_applied": 0,
            "unresolved_group_ids": [],
            "usage": {"input_tokens": 0, "output_tokens": 0, "attempts": 0},
        }
    )
    provider = _FakeProvider(
        [("native_referent_audit_batch", {"findings": [{"group_id": "seg-1", "status": "ok", "checked_terms": []}]})]
    )

    result = _apply_referent_audit(
        job_root=tmp_path,
        provider=provider,
        pass2=pass2,
        force=False,
        progress=None,
        batch_size=12,
        max_repair_rounds=1,
    )

    assert provider.calls == ["native_referent_audit_batch"]
    assert result["referent_audit_summary"]["schema_version"] == REFERENT_AUDIT_SCHEMA_VERSION
    assert "groups_flagged" in result["referent_audit_summary"]


def test_no_prior_summary_runs_the_audit(tmp_path):
    pass2 = _pass2(None)
    provider = _FakeProvider(
        [("native_referent_audit_batch", {"findings": [{"group_id": "seg-1", "status": "ok", "checked_terms": []}]})]
    )

    result = _apply_referent_audit(
        job_root=tmp_path,
        provider=provider,
        pass2=pass2,
        force=False,
        progress=None,
        batch_size=12,
        max_repair_rounds=1,
    )

    assert provider.calls == ["native_referent_audit_batch"]
    assert result["referent_audit_summary"]["schema_version"] == REFERENT_AUDIT_SCHEMA_VERSION
