"""measure_tts_batch accepts a whole clip's worth of units, not just 12."""
from __future__ import annotations

import pytest

from mathula_tv import dub_workbench
from mathula_tv.dub_workbench import MAX_MEASURE_BATCH, BudgetExhausted, Workbench


def _wb(tmp_path, budget=100):
    wb = Workbench(store=None, tts_factory=lambda: object(), measure_root=tmp_path)
    wb.episode = {"dir": str(tmp_path), "tts_budget": budget, "tts_used": 0, "speed_policy": "natural"}
    return wb


def _fake_fit(tts, *, text, start_ms, end_ms, **kwargs):
    return {"measured_ms": len(text), "window_ms": end_ms - start_ms, "direction": None, "fits_naturally": True,
            "required_speed_percent": 0.0, "audio_path": "x"}


def _requests(n):
    return [{"text": "a" * (i + 1), "start_ms": 0, "end_ms": 1000} for i in range(n)]


def test_a_batch_larger_than_twelve_is_measured_in_order_and_charged_once(tmp_path, monkeypatch):
    monkeypatch.setattr(dub_workbench, "measure_text_fit", _fake_fit)
    wb = _wb(tmp_path)
    out = wb.measure_tts_batch(_requests(30))
    assert [r["measured_ms"] for r in out["results"]] == list(range(1, 31))
    assert out["tts_budget_remaining"] == 70 and wb.episode["tts_used"] == 30


def test_the_error_beyond_the_cap_says_the_cap_and_the_count_sent(tmp_path, monkeypatch):
    monkeypatch.setattr(dub_workbench, "measure_text_fit", _fake_fit)
    wb = _wb(tmp_path)
    with pytest.raises(ValueError) as error:
        wb.measure_tts_batch(_requests(MAX_MEASURE_BATCH + 1))
    assert str(MAX_MEASURE_BATCH) in str(error.value) and str(MAX_MEASURE_BATCH + 1) in str(error.value)
    with pytest.raises(ValueError):
        wb.measure_tts_batch([])


def test_the_budget_still_applies_to_big_batches(tmp_path, monkeypatch):
    monkeypatch.setattr(dub_workbench, "measure_text_fit", _fake_fit)
    wb = _wb(tmp_path, budget=20)
    with pytest.raises(BudgetExhausted):
        wb.measure_tts_batch(_requests(30))
    assert wb.episode["tts_used"] == 0
