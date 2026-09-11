"""Proves _run_qa_verdicts's batches actually run concurrently, not just that
correctness survived being routed through a ThreadPoolExecutor with a single
batch.

Extracted this session from what was run_native_qa_back_translation's internal
_audit() closure -- a plain sequential for-loop, the last unparallelized
fan-out in native_dub.py. QA batches are fully independent of each other
(each is its own back-translation/direct-adequacy call over a distinct slice
of sentences), the same property referent_audit.py's own batches already
have, so this uses the identical concurrency-proof pattern as
test_referent_audit_concurrency.py: a real artificial delay per call, keyed
by which refs a batch contains (not call order, so it's safe regardless of
which order concurrent threads happen to arrive in).
"""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from mathula_tv.native_dub import _run_qa_verdicts


class _ConcurrencyTrackingQaProvider:
    def __init__(self, *, delay_seconds: float = 0.05):
        self.config = SimpleNamespace(max_output_tokens=8192)
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
            if operation == "native_qa_back_translation_batch":
                verdicts = [
                    {
                        "ref": pair["ref"], "back_translation": pair["original_english"],
                        "matches_original": True, "discrepancy_summary": "faithful",
                        "category": "none", "severity": "none",
                    }
                    for pair in payload["pairs"]
                ]
                return SimpleNamespace(data={"verdicts": verdicts}, model="grok-test",
                                        input_tokens=10, output_tokens=5, attempts=1)
            raise AssertionError(f"unexpected operation: {operation}")
        finally:
            with self._lock:
                self._in_flight -= 1


def _group(group_id: str, en: str, zu: str) -> dict:
    return {"group_id": group_id, "source_text": en, "spoken_text": zu, "speaker_id": "S"}


def test_qa_batches_actually_overlap_in_time_not_just_eventually_all_complete():
    groups = [_group(f"g{i}", f"English {i}.", f"Zulu {i}.") for i in range(20)]
    provider = _ConcurrencyTrackingQaProvider(delay_seconds=0.05)

    # batch_size=5 over 20 groups -> 4 batches; workers=4 should let all 4 run at once.
    results, usage = _run_qa_verdicts(
        provider=provider, target_groups=groups, batch_size=5,
        direct_adequacy=False, workers=4, progress=None,
    )

    assert results == []  # every pair matched -- nothing flagged
    assert len(provider.calls) == 4
    # The real point of this test: more than one batch was genuinely in
    # flight at once, not just correctly processed one after another.
    assert provider.max_concurrent_seen > 1
    all_seen_refs = sorted(
        item["ref"] for call in provider.calls for item in call["payload"]["pairs"]
    )
    assert all_seen_refs == sorted(f"g{i}" for i in range(20))
    assert usage == {"input_tokens": 40, "output_tokens": 20, "attempts": 4}


def test_qa_workers_of_one_falls_back_to_effectively_sequential():
    groups = [_group(f"g{i}", f"English {i}.", f"Zulu {i}.") for i in range(10)]
    provider = _ConcurrencyTrackingQaProvider(delay_seconds=0.02)

    _run_qa_verdicts(
        provider=provider, target_groups=groups, batch_size=5,
        direct_adequacy=False, workers=1, progress=None,
    )

    assert provider.max_concurrent_seen == 1


def test_qa_verdicts_still_flags_drift_correctly_when_run_in_parallel():
    # Correctness under concurrency, not just speed: one genuinely flagged
    # sentence among several clean ones must still surface correctly no
    # matter which thread happened to process its batch.
    class _MixedProvider:
        def __init__(self):
            self.config = SimpleNamespace(max_output_tokens=8192)

        def complete_json(self, *, operation, payload, **_kwargs):
            verdicts = []
            for pair in payload["pairs"]:
                if pair["ref"] == "g7":
                    verdicts.append({
                        "ref": "g7", "back_translation": "He is here.",
                        "matches_original": False, "discrepancy_summary": "dropped negation",
                        "category": "negation", "severity": "critical",
                    })
                else:
                    verdicts.append({
                        "ref": pair["ref"], "back_translation": pair["original_english"],
                        "matches_original": True, "discrepancy_summary": "faithful",
                        "category": "none", "severity": "none",
                    })
            return SimpleNamespace(data={"verdicts": verdicts}, model="grok-test",
                                    input_tokens=5, output_tokens=5, attempts=1)

    groups = [_group(f"g{i}", f"English {i}.", f"Zulu {i}.") for i in range(15)]
    results, _usage = _run_qa_verdicts(
        provider=_MixedProvider(), target_groups=groups, batch_size=4,
        direct_adequacy=False, workers=4, progress=None,
    )
    assert [item["ref"] for item in results] == ["g7"]
    assert results[0]["severity"] == "critical"
