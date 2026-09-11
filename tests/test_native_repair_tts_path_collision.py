"""Regression test for a real production crash: "Azure TTS output belongs to a
different synthesis request", raised mid-timing-repair on a re-run of native-dub.

Root cause: repair-round candidate text comes from a fresh, non-deterministic Grok
call every run, but the output path native_dub.py wrote it to was keyed only on
group_id + round number (e.g. "native_phrase_0046.repair1.wav"). A later run whose
repair call returned different text collided with whatever was cached at that path
from an earlier run and hit AzureTTSBackend's real idempotency check, which
correctly refuses to silently overwrite mismatched audio.

The fix hashes the actual repaired text into the path, matching the pattern already
used elsewhere in native_dub.py for temporal-mask candidate measurement
(_measure_temporal_candidate). This test proves the mechanism directly against the
real AzureTTSBackend idempotency layer that broke in production, without needing to
drive the full timing controller.
"""
from __future__ import annotations

import io
import wave
from pathlib import Path

import pytest

from mathula_tv.azure_tts import (
    AzureTTSBackend,
    AzureTTSIdempotencyError,
    AzureTTSRequest,
    AzureTTSResponse,
)
from mathula_tv.native_dub import _hash_payload

VOICES = [{"ShortName": "zu-ZA-ThandoNeural", "Locale": "zu-ZA", "Gender": "Female"}]


class _Boundary:
    def __init__(self, *, voices, audio):
        self.voice_responses = list(voices)
        self.audio_responses = list(audio)

    def get_voices(self, locale, *, timeout_seconds):
        return self.voice_responses.pop(0)

    def synthesize_ssml(self, ssml, *, output_format, timeout_seconds):
        return self.audio_responses.pop(0)


def _json_response(value):
    import json

    return AzureTTSResponse(200, json.dumps(value).encode(), {})


def _pcm_wav(value: int) -> bytes:
    frames = int(value).to_bytes(2, "little", signed=True) * 12_000
    output = io.BytesIO()
    with wave.open(output, "wb") as target:
        target.setparams((1, 2, 24_000, 0, "NONE", "not compressed"))
        target.writeframes(frames)
    return output.getvalue()


def _backend(boundary) -> AzureTTSBackend:
    return AzureTTSBackend(
        boundary,
        configured_voices=("zu-ZA-ThandoNeural",),
        default_voice="zu-ZA-ThandoNeural",
        max_retries=0,
        sleep=lambda _: None,
    )


def _request(text: str) -> AzureTTSRequest:
    return AzureTTSRequest(
        turn_id="native_phrase_0046",
        speaker_id="SPEAKER_00",
        text=text,
        preferred_duration_ms=5700,
        maximum_duration_ms=6040,
    )


def _repair_digest(*, group_id: str, voice: str, repair_map: dict, round_number: int) -> str:
    return _hash_payload(
        {"group_id": group_id, "voice": voice, "repair_map": repair_map, "round": round_number}
    )[:12]


def test_old_group_id_and_round_only_path_collides_across_runs(tmp_path):
    """Demonstrates the actual bug: same path, two different repair-run texts."""

    boundary = _Boundary(
        voices=[_json_response(VOICES)],
        audio=[
            AzureTTSResponse(200, _pcm_wav(1000)),
            AzureTTSResponse(200, _pcm_wav(2000)),
        ],
    )
    service = _backend(boundary)
    output = tmp_path / "native_phrase_0046.repair1.wav"

    service.synthesize(_request("Run one's rhetorical expansion."), output)
    with pytest.raises(AzureTTSIdempotencyError) as error:
        service.synthesize(_request("Run two's DIFFERENT rhetorical expansion."), output)
    assert error.value.conflict_kind == "request_mismatch"


def test_content_addressed_path_avoids_the_collision_across_runs(tmp_path):
    """Proves the fix: hashing the repair_map into the path keeps re-runs distinct."""

    boundary = _Boundary(
        voices=[_json_response(VOICES)],
        audio=[
            AzureTTSResponse(200, _pcm_wav(1000)),
            AzureTTSResponse(200, _pcm_wav(2000)),
        ],
    )
    service = _backend(boundary)
    voice = "zu-ZA-ThandoNeural"

    run_one_map = {"seg-1": "Run one's rhetorical expansion."}
    run_two_map = {"seg-1": "Run two's DIFFERENT rhetorical expansion."}

    digest_one = _repair_digest(group_id="native_phrase_0046", voice=voice, repair_map=run_one_map, round_number=1)
    digest_two = _repair_digest(group_id="native_phrase_0046", voice=voice, repair_map=run_two_map, round_number=1)
    assert digest_one != digest_two

    output_one = tmp_path / f"native_phrase_0046.repair1.{digest_one}.wav"
    output_two = tmp_path / f"native_phrase_0046.repair1.{digest_two}.wav"

    # Neither call raises -- the whole point of the fix.
    service.synthesize(_request(run_one_map["seg-1"]), output_one)
    service.synthesize(_request(run_two_map["seg-1"]), output_two)

    assert output_one.is_file() and output_two.is_file()
    assert output_one.read_bytes() != output_two.read_bytes()


def test_same_repair_text_reuses_the_same_cached_path(tmp_path):
    """A rerun that happens to get the identical repair text back is still a cheap cache hit."""

    boundary = _Boundary(voices=[_json_response(VOICES)], audio=[AzureTTSResponse(200, _pcm_wav(1000))])
    service = _backend(boundary)
    voice = "zu-ZA-ThandoNeural"
    repair_map = {"seg-1": "Identical rhetorical expansion both times."}

    digest = _repair_digest(group_id="native_phrase_0046", voice=voice, repair_map=repair_map, round_number=1)
    output = tmp_path / f"native_phrase_0046.repair1.{digest}.wav"

    first = service.synthesize(_request(repair_map["seg-1"]), output)
    reused = service.synthesize(_request(repair_map["seg-1"]), output)

    assert reused.idempotent_reuse is True
    assert reused.sha256 == first.sha256
    assert len(boundary.audio_responses) == 0  # only one real synthesis call was consumed
