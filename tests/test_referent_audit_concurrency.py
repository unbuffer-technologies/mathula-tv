"""Proves audit_and_repair_pass2's batches actually run concurrently, not just that
correctness survived being routed through a ThreadPoolExecutor with a single batch.

Referent-audit batches are fully independent of each other (unlike Pass 2's temporal-
mask windows, which need the previous window's own real committed output for
continuity) -- real production data showed 152 sentences making 13 sequential audit
calls averaging ~27s each (~6 minutes) with nothing to justify serializing them. This
test forces multiple batches and measures real overlap, not just eventual completion.
"""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from mathula_tv.referent_audit import audit_and_repair_pass2


class _ConcurrencyTrackingProvider:
    """Keyed by which group_ids a batch contains (not call order), so it's safe
    regardless of which order concurrent threads happen to arrive in. Tracks the
    highest number of calls genuinely in flight at once via a short artificial delay.
    """

    def __init__(self, group_ids: list[str], *, delay_seconds: float = 0.05):
        self.config = SimpleNamespace(max_output_tokens=8192)
        self._group_ids = group_ids
        self._delay = delay_seconds
        self._lock = threading.Lock()
        self._in_flight = 0
        self.max_concurrent_seen = 0
        self.calls: list[dict] = []

    def complete_json(self, *, operation, system_prompt, payload, schema, max_output_tokens=None):
        with self._lock:
            self._in_flight += 1
            self.max_concurrent_seen = max(self.max_concurrent_seen, self._in_flight)
        try:
            time.sleep(self._delay)
            self.calls.append({"operation": operation, "payload": payload})
            if operation == "native_referent_audit_batch":
                findings = [{"group_id": item["group_id"], "status": "ok"} for item in payload["groups"]]
                return SimpleNamespace(data={"findings": findings}, model="grok-test", input_tokens=10, output_tokens=5, attempts=1)
            raise AssertionError(f"unexpected operation: {operation}")
        finally:
            with self._lock:
                self._in_flight -= 1


def test_audit_batches_actually_overlap_in_time_not_just_eventually_all_complete():
    group_ids = [f"g{i}" for i in range(20)]
    pass2 = {
        "translations": [
            {"segment_id": gid, "source_text": f"English {gid}.", "spoken_text": f"Zulu {gid}."}
            for gid in group_ids
        ],
    }
    provider = _ConcurrencyTrackingProvider(group_ids, delay_seconds=0.05)

    # batch_size=5 over 20 groups -> 4 batches; max_workers=4 should let all 4 run at once.
    summary = audit_and_repair_pass2(
        provider=provider, pass2=pass2, registry=None, batch_size=5, max_workers=4,
    )

    assert summary["groups_checked"] == 20
    assert len(provider.calls) == 4
    # The real point of this test: more than one batch was genuinely in flight at the
    # same time, not just correctly processed one after another.
    assert provider.max_concurrent_seen > 1
    # Every group_id was covered exactly once across the concurrent batches.
    all_seen_group_ids = sorted(
        item["group_id"] for call in provider.calls for item in call["payload"]["groups"]
    )
    assert all_seen_group_ids == sorted(group_ids)


def test_audit_workers_of_one_falls_back_to_effectively_sequential():
    group_ids = [f"g{i}" for i in range(10)]
    pass2 = {
        "translations": [
            {"segment_id": gid, "source_text": f"English {gid}.", "spoken_text": f"Zulu {gid}."}
            for gid in group_ids
        ],
    }
    provider = _ConcurrencyTrackingProvider(group_ids, delay_seconds=0.02)

    summary = audit_and_repair_pass2(
        provider=provider, pass2=pass2, registry=None, batch_size=5, max_workers=1,
    )

    assert summary["groups_checked"] == 10
    assert provider.max_concurrent_seen == 1
