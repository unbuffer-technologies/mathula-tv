"""Tests for native_dub._presynthesize_islands_with_bookmarks: the pre-pass that
synthesizes a whole multi-island group in one bookmarked Speech SDK call instead of
one REST call per island (see azure_tts_sdk.py's module docstring for the rationale).

The giant _run_batch_adaptive_timing_controller itself is not unit-tested anywhere in
this suite (it requires a full TTS/provider stub -- see
test_native_timing_repair_qa_text_saved.py's own note on this), so these tests target
the pre-pass function directly, which is fully exercisable with a fake SDK boundary.
"""
from __future__ import annotations

import io
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from mathula_tv.azure_tts_sdk import SDKBookmarkEvent, SDKGroupSynthesisResponse
from mathula_tv.native_dub import _presynthesize_islands_with_bookmarks
from mathula_tv.tts_ssml import SSMLBounds

VOICE = "zu-ZA-ThandoNeural"


def pcm_wav(*, seconds: float, rate: int = 24_000, value: int = 1200) -> bytes:
    frames = int(value).to_bytes(2, "little", signed=True) * int(rate * seconds)
    output = io.BytesIO()
    with wave.open(output, "wb") as target:
        target.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        target.writeframes(frames)
    return output.getvalue()


class FakeBoundary:
    def __init__(self, *, responses=()):
        self.responses = list(responses)
        self.calls: list[str] = []

    def synthesize_group(self, ssml, *, timeout_seconds):
        self.calls.append(ssml)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _completed_response(bookmarks_ms: dict[str, int], *, seconds: float = 3.0):
    marks = sorted(bookmarks_ms, key=lambda mark: bookmarks_ms[mark])
    return SDKGroupSynthesisResponse(
        audio_data=pcm_wav(seconds=seconds),
        bookmarks=tuple(
            SDKBookmarkEvent(mark=mark, audio_offset_ticks=bookmarks_ms[mark] * 10_000) for mark in marks
        ),
    )


def _fake_tts_backend():
    return SimpleNamespace(
        configured_voices=(VOICE,),
        sample_rate=24_000,
        channels=1,
        sample_width=2,
        ssml_bounds=SSMLBounds(),
    )


def _island(*, parent_id: str, index: int, speaker_id: str = "SPEAKER_00", text: str = "Sawubona.") -> dict:
    island_id = f"{parent_id}__isl{index:02d}"
    return {
        "group_id": island_id,
        "speaker_id": speaker_id,
        "island_index": index,
        "parent_group_id": parent_id,
        "parts": [{"type": "text", "segment_id": island_id, "text": text}],
    }


def _voice_assignments(speaker_id: str = "SPEAKER_00"):
    return {speaker_id: {"selected_voice": VOICE, "base_prosody": {"pitch_percent": 0, "volume_percent": 0}}}


def test_skips_groups_with_only_one_island(tmp_path: Path):
    boundary = FakeBoundary()
    islands = [_island(parent_id="g1", index=0)]
    result = _presynthesize_islands_with_bookmarks(
        timing_input_groups=islands,
        island_counts={"g1": 1},
        voice_assignments=_voice_assignments(),
        tts=_fake_tts_backend(),
        sdk_boundary=boundary,
        audio_dir=tmp_path,
        force=False,
        progress=None,
    )
    assert result == {}
    assert boundary.calls == []


def test_produces_real_azure_tts_results_for_a_multi_island_group(tmp_path: Path):
    boundary = FakeBoundary(
        responses=[_completed_response({"g1__isl00": 0, "g1__isl01": 1000, "g1__isl02": 1800})]
    )
    islands = [
        _island(parent_id="g1", index=0, text="Sawubona."),
        _island(parent_id="g1", index=1, text="Ninjani?"),
        _island(parent_id="g1", index=2, text="Ngiyabonga kakhulu."),
    ]
    result = _presynthesize_islands_with_bookmarks(
        timing_input_groups=islands,
        island_counts={"g1": 3},
        voice_assignments=_voice_assignments(),
        tts=_fake_tts_backend(),
        sdk_boundary=boundary,
        audio_dir=tmp_path,
        force=False,
        progress=None,
    )
    assert set(result) == {"g1__isl00", "g1__isl01", "g1__isl02"}
    assert result["g1__isl00"].backend == "azure_tts_sdk_bookmark_group"
    assert result["g1__isl00"].duration_ms == 1000
    assert result["g1__isl01"].duration_ms == 800
    assert result["g1__isl02"].duration_ms == 1200
    for island_result in result.values():
        assert Path(island_result.output_path).exists()
        assert island_result.voice == VOICE
        assert island_result.speaker_id == "SPEAKER_00"
    assert len(boundary.calls) == 1


def test_falls_back_gracefully_when_boundary_fails(tmp_path: Path):
    boundary = FakeBoundary(responses=[RuntimeError("Azure Speech SDK is not installed")])
    islands = [
        _island(parent_id="g1", index=0, text="Sawubona."),
        _island(parent_id="g1", index=1, text="Ninjani?"),
    ]
    warnings: list[str] = []
    result = _presynthesize_islands_with_bookmarks(
        timing_input_groups=islands,
        island_counts={"g1": 2},
        voice_assignments=_voice_assignments(),
        tts=_fake_tts_backend(),
        sdk_boundary=boundary,
        audio_dir=tmp_path,
        force=False,
        progress=warnings.append,
    )
    assert result == {}
    assert any("FAILED for g1" in message for message in warnings)


def test_only_the_failing_group_falls_back_others_still_presynthesize(tmp_path: Path):
    boundary = FakeBoundary(
        responses=[
            RuntimeError("boom"),
            _completed_response({"g2__isl00": 0, "g2__isl01": 1000}),
        ]
    )
    islands = [
        _island(parent_id="g1", index=0, text="Sawubona."),
        _island(parent_id="g1", index=1, text="Ninjani?"),
        _island(parent_id="g2", index=0, text="Kulungile."),
        _island(parent_id="g2", index=1, text="Siyahamba."),
    ]
    result = _presynthesize_islands_with_bookmarks(
        timing_input_groups=islands,
        island_counts={"g1": 2, "g2": 2},
        voice_assignments=_voice_assignments(),
        tts=_fake_tts_backend(),
        sdk_boundary=boundary,
        audio_dir=tmp_path,
        force=False,
        progress=None,
    )
    assert set(result) == {"g2__isl00", "g2__isl01"}


def test_different_speakers_use_their_own_assigned_voice(tmp_path: Path):
    boundary = FakeBoundary(
        responses=[
            _completed_response({"g1__isl00": 0, "g1__isl01": 1000}),
            _completed_response({"g2__isl00": 0, "g2__isl01": 1000}),
        ]
    )
    islands = [
        _island(parent_id="g1", index=0, speaker_id="SPEAKER_00", text="Sawubona."),
        _island(parent_id="g1", index=1, speaker_id="SPEAKER_00", text="Ninjani?"),
        _island(parent_id="g2", index=0, speaker_id="SPEAKER_01", text="Kulungile."),
        _island(parent_id="g2", index=1, speaker_id="SPEAKER_01", text="Siyahamba."),
    ]
    voice_assignments = {
        "SPEAKER_00": {"selected_voice": "zu-ZA-ThandoNeural", "base_prosody": {}},
        "SPEAKER_01": {"selected_voice": "zu-ZA-ThembaNeural", "base_prosody": {}},
    }
    tts = SimpleNamespace(
        configured_voices=("zu-ZA-ThandoNeural", "zu-ZA-ThembaNeural"),
        sample_rate=24_000, channels=1, sample_width=2, ssml_bounds=SSMLBounds(),
    )
    result = _presynthesize_islands_with_bookmarks(
        timing_input_groups=islands,
        island_counts={"g1": 2, "g2": 2},
        voice_assignments=voice_assignments,
        tts=tts,
        sdk_boundary=boundary,
        audio_dir=tmp_path,
        force=False,
        progress=None,
    )
    assert result["g1__isl00"].voice == "zu-ZA-ThandoNeural"
    assert result["g2__isl00"].voice == "zu-ZA-ThembaNeural"
