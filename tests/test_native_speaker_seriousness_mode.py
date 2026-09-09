"""Speaker seriousness mode: real user direction, 2026-09-10, "the system may need
to have 3 modes of seriousness: Mode 1: cross examination, where facts and
chronology is important. Mode 2: political speech, politicians mention a lot of
facts, but they mostly make a lot of empty promises. Mode 3: general news speech,
where can have high flexibility with compacting."

Classified once per job, per speaker_id (never per-segment -- confirmed via
AskUserQuestion this session), via one batched FoundryGrokProvider call, cached
the same checksum-keyed way as _ensure_zulu_glossary/_ensure_native_pronunciation_
research. Any failure degrades EVERY speaker to "cross_examination" (the
strictest tier) -- under-classifying toward stricter is always safe.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from mathula_tv.native_dub import (
    _SPEAKER_SERIOUSNESS_MODE_DEFAULT,
    _ensure_speaker_seriousness_modes,
    _request_speaker_seriousness_mode_batch,
    _speaker_seriousness_mode_samples,
    native_dub_paths,
)


def _job_root(tmp_path: Path) -> Path:
    job_root = tmp_path / "working" / "jobs" / "job"
    native_dub_paths(job_root).root.mkdir(parents=True, exist_ok=True)
    return job_root


# --- _speaker_seriousness_mode_samples -------------------------------------------------


def test_samples_group_by_speaker_in_order():
    segments = [
        {"speaker_id": "S0", "restored_text": "First line."},
        {"speaker_id": "S1", "restored_text": "Other speaker."},
        {"speaker_id": "S0", "restored_text": "Second line."},
    ]
    samples = _speaker_seriousness_mode_samples(segments)
    assert samples == {
        "S0": ["First line.", "Second line."],
        "S1": ["Other speaker."],
    }


def test_samples_cap_per_speaker():
    segments = [{"speaker_id": "S0", "restored_text": f"Line {i}."} for i in range(20)]
    samples = _speaker_seriousness_mode_samples(segments, max_lines_per_speaker=3)
    assert samples == {"S0": ["Line 0.", "Line 1.", "Line 2."]}


def test_samples_skip_blank_speaker_or_text():
    segments = [
        {"speaker_id": "", "restored_text": "No speaker."},
        {"speaker_id": "S0", "restored_text": "   "},
        {"speaker_id": "S0", "restored_text": "Real line."},
    ]
    samples = _speaker_seriousness_mode_samples(segments)
    assert samples == {"S0": ["Real line."]}


def test_samples_fall_back_to_source_text_when_restored_missing():
    segments = [{"speaker_id": "S0", "source_text": "Raw text."}]
    samples = _speaker_seriousness_mode_samples(segments)
    assert samples == {"S0": ["Raw text."]}


# --- _request_speaker_seriousness_mode_batch --------------------------------------------


class _FakeModeProvider:
    def __init__(self, *, responses=None, error=None):
        self.config = SimpleNamespace(max_output_tokens=8192)
        self._responses = responses or {}
        self._error = error
        self.calls: list[dict] = []

    def complete_json(self, *, operation, system_prompt, payload, schema, max_output_tokens=None):
        self.calls.append({"operation": operation, "payload": payload})
        if self._error is not None:
            raise self._error
        speakers = [
            {"speaker_id": speaker_id, **result} for speaker_id, result in self._responses.items()
        ]
        return SimpleNamespace(data={"speakers": speakers}, input_tokens=50, output_tokens=20, attempts=1)


def test_request_batch_returns_mode_and_reasoning_per_speaker():
    provider = _FakeModeProvider(responses={
        "S0": {"mode": "cross_examination", "reasoning": "A witness recounting specific dates."},
        "S1": {"mode": "political_speech", "reasoning": "A candidate listing manifesto promises."},
    })
    by_id, usage = _request_speaker_seriousness_mode_batch(
        provider=provider, speaker_samples={"S0": ["I saw him on May 3rd."], "S1": ["We will scrap all debt."]},
    )
    assert by_id == {
        "S0": {"mode": "cross_examination", "reasoning": "A witness recounting specific dates."},
        "S1": {"mode": "political_speech", "reasoning": "A candidate listing manifesto promises."},
    }
    assert usage == {"input_tokens": 50, "output_tokens": 20, "attempts": 1}
    assert provider.calls[0]["operation"] == "native_speaker_seriousness_mode_batch"


def test_request_batch_raises_on_missing_speaker():
    provider = _FakeModeProvider(responses={"S0": {"mode": "general_news", "reasoning": "A reporter narrating."}})
    with pytest.raises(ValueError, match="missing"):
        _request_speaker_seriousness_mode_batch(provider=provider, speaker_samples={"S0": ["Line."], "S1": ["Line."]})


def test_request_batch_raises_on_unknown_speaker():
    provider = _FakeModeProvider(responses={
        "S0": {"mode": "general_news", "reasoning": "Fine."},
        "GHOST": {"mode": "general_news", "reasoning": "Not requested."},
    })
    with pytest.raises(ValueError, match="unknown"):
        _request_speaker_seriousness_mode_batch(provider=provider, speaker_samples={"S0": ["Line."]})


# --- _ensure_speaker_seriousness_modes ----------------------------------------------------


def test_ensure_modes_writes_artifact_and_returns_modes(tmp_path):
    job_root = _job_root(tmp_path)
    provider = _FakeModeProvider(responses={
        "S0": {"mode": "cross_examination", "reasoning": "Testimony."},
    })
    segments = [{"speaker_id": "S0", "restored_text": "I was there on the 3rd."}]
    modes, usage = _ensure_speaker_seriousness_modes(job_root=job_root, provider=provider, segments=segments)
    assert modes == {"S0": "cross_examination"}
    assert usage["attempts"] == 1
    artifact = native_dub_paths(job_root).speaker_seriousness_modes
    assert artifact.is_file()


def test_ensure_modes_returns_empty_with_zero_cost_when_no_speakers(tmp_path):
    job_root = _job_root(tmp_path)
    provider = _FakeModeProvider()
    modes, usage = _ensure_speaker_seriousness_modes(job_root=job_root, provider=provider, segments=[])
    assert modes == {}
    assert usage == {"input_tokens": 0, "output_tokens": 0, "attempts": 0}
    assert provider.calls == []


def test_ensure_modes_reuses_cache_without_a_second_call(tmp_path):
    job_root = _job_root(tmp_path)
    provider = _FakeModeProvider(responses={"S0": {"mode": "political_speech", "reasoning": "Promises."}})
    segments = [{"speaker_id": "S0", "restored_text": "We will build houses."}]
    _ensure_speaker_seriousness_modes(job_root=job_root, provider=provider, segments=segments)
    modes, _usage = _ensure_speaker_seriousness_modes(job_root=job_root, provider=provider, segments=segments)
    assert modes == {"S0": "political_speech"}
    assert len(provider.calls) == 1  # second call reused the cache


def test_ensure_modes_regenerates_when_segments_change(tmp_path):
    job_root = _job_root(tmp_path)
    provider = _FakeModeProvider(responses={
        "S0": {"mode": "political_speech", "reasoning": "Promises."},
        "S1": {"mode": "general_news", "reasoning": "Narration."},
    })
    _ensure_speaker_seriousness_modes(
        job_root=job_root, provider=provider, segments=[{"speaker_id": "S0", "restored_text": "We will build houses."}],
    )
    modes, _usage = _ensure_speaker_seriousness_modes(
        job_root=job_root, provider=provider,
        segments=[
            {"speaker_id": "S0", "restored_text": "We will build houses."},
            {"speaker_id": "S1", "restored_text": "Reporting live from the scene."},
        ],
    )
    assert modes == {"S0": "political_speech", "S1": "general_news"}
    assert len(provider.calls) == 2  # new speaker content invalidated the cache


def test_ensure_modes_force_bypasses_a_fresh_cache(tmp_path):
    job_root = _job_root(tmp_path)
    provider = _FakeModeProvider(responses={"S0": {"mode": "cross_examination", "reasoning": "Testimony."}})
    segments = [{"speaker_id": "S0", "restored_text": "I was there."}]
    _ensure_speaker_seriousness_modes(job_root=job_root, provider=provider, segments=segments)
    _ensure_speaker_seriousness_modes(job_root=job_root, provider=provider, segments=segments, force=True)
    assert len(provider.calls) == 2


def test_ensure_modes_degrades_every_speaker_to_the_strict_default_on_provider_error(tmp_path):
    job_root = _job_root(tmp_path)
    provider = _FakeModeProvider(error=RuntimeError("boom"))
    segments = [
        {"speaker_id": "S0", "restored_text": "A promise."},
        {"speaker_id": "S1", "restored_text": "Another line."},
    ]
    modes, usage = _ensure_speaker_seriousness_modes(job_root=job_root, provider=provider, segments=segments)
    assert modes == {"S0": _SPEAKER_SERIOUSNESS_MODE_DEFAULT, "S1": _SPEAKER_SERIOUSNESS_MODE_DEFAULT}
    assert usage == {"input_tokens": 0, "output_tokens": 0, "attempts": 0}
