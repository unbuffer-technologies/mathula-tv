"""Tests for _synthesize_group_recovering_stale_cache: the fix for a real,
confirmed-twice-in-production crash where Phase A's initial synthesis call writes
to a FIXED path ({group_id}.wav), so a group whose committed text changes between
runs (a translation fix, or an island-derivation fix changing how a group's
islands are cut) hits a stale-cache idempotency conflict that previously
propagated straight up and killed the entire render.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest

from mathula_tv.azure_tts import AzureTTSIdempotencyError, AzureTTSResult
from mathula_tv.native_dub import _synthesize_group_recovering_stale_cache


def _group():
    return {
        "group_id": "native_phrase_0005",
        "speaker_id": "SPEAKER_00",
        "parts": [{"type": "text", "segment_id": "seg-1", "text": "Sawubona."}],
    }


def _voice_info():
    return {"selected_voice": "zu-ZA-ThandoNeural", "base_prosody": {}}


def _result():
    return SimpleNamespace(duration_ms=1000)


def test_retries_with_force_on_request_mismatch_conflict(tmp_path, monkeypatch):
    calls: list[bool] = []

    def fake_synthesize_group(*, tts, group, voice_info, output_path, preferred_ms, maximum_ms, force, job_root=None):
        calls.append(force)
        if not force:
            raise AzureTTSIdempotencyError("stale content", conflict_kind="request_mismatch")
        return _result()

    monkeypatch.setattr("mathula_tv.native_dub._synthesize_group", fake_synthesize_group)
    warnings: list[str] = []
    result = _synthesize_group_recovering_stale_cache(
        tts=object(), group=_group(), voice_info=_voice_info(),
        output_path=tmp_path / "native_phrase_0005.wav",
        preferred_ms=1000, maximum_ms=1200, force=False, progress=warnings.append,
    )
    assert calls == [False, True]
    assert result.duration_ms == 1000
    assert any("different content than the current committed text" in message for message in warnings)


def test_does_not_auto_retry_an_artifact_integrity_conflict(tmp_path, monkeypatch):
    # A different, rarer conflict kind (incomplete or tampered write) signals a
    # real problem worth surfacing -- must NOT be silently papered over the same
    # way a genuine content change is.
    def fake_synthesize_group(*, tts, group, voice_info, output_path, preferred_ms, maximum_ms, force, job_root=None):
        raise AzureTTSIdempotencyError("corrupted", conflict_kind="artifact_integrity")

    monkeypatch.setattr("mathula_tv.native_dub._synthesize_group", fake_synthesize_group)
    with pytest.raises(AzureTTSIdempotencyError):
        _synthesize_group_recovering_stale_cache(
            tts=object(), group=_group(), voice_info=_voice_info(),
            output_path=tmp_path / "native_phrase_0005.wav",
            preferred_ms=1000, maximum_ms=1200, force=False, progress=None,
        )


def test_does_not_retry_when_already_forced(tmp_path, monkeypatch):
    # If force=True already produced a request_mismatch conflict, retrying with
    # force=True again would just raise the same error a second time for nothing.
    calls: list[bool] = []

    def fake_synthesize_group(*, tts, group, voice_info, output_path, preferred_ms, maximum_ms, force, job_root=None):
        calls.append(force)
        raise AzureTTSIdempotencyError("stale content", conflict_kind="request_mismatch")

    monkeypatch.setattr("mathula_tv.native_dub._synthesize_group", fake_synthesize_group)
    with pytest.raises(AzureTTSIdempotencyError):
        _synthesize_group_recovering_stale_cache(
            tts=object(), group=_group(), voice_info=_voice_info(),
            output_path=tmp_path / "native_phrase_0005.wav",
            preferred_ms=1000, maximum_ms=1200, force=True, progress=None,
        )
    assert calls == [True]


def test_succeeds_immediately_when_no_conflict(tmp_path, monkeypatch):
    calls: list[bool] = []

    def fake_synthesize_group(*, tts, group, voice_info, output_path, preferred_ms, maximum_ms, force, job_root=None):
        calls.append(force)
        return _result()

    monkeypatch.setattr("mathula_tv.native_dub._synthesize_group", fake_synthesize_group)
    result = _synthesize_group_recovering_stale_cache(
        tts=object(), group=_group(), voice_info=_voice_info(),
        output_path=tmp_path / "native_phrase_0005.wav",
        preferred_ms=1000, maximum_ms=1200, force=False, progress=None,
    )
    assert calls == [False]
    assert result.duration_ms == 1000
