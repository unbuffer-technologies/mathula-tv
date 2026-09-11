"""Regression test for a real production case.

native_phrase_0010 measured 34,738ms against a 34,800ms window -- a 62ms
shortfall on a 34.8-second block (99.82% of target). Closing a gap that small
through wording is not achievable: at ~5.3 syllables/second, 62ms is about a
third of one syllable, and any real edit adds or removes at least a full
syllable (~190ms), overshooting either way. Every repair/rescue round left
this group exactly as short as before, which then hit the pipeline's hard
"never leave dead air" failure -- even though there was already 160ms of free
gap before the next block, meaning the shortfall was harmless in practice.

_pad_wav_with_trailing_silence closes a residual gap this small with real
audio (the "expand" side's equivalent of _speed_fit_overflow_wav on the
"compress" side) instead of asking a model to hit an impossible precision.
"""

import math
import struct
import wave
from pathlib import Path

from mathula_tv.native_dub import (
    MAX_UNRESOLVED_EARLY_SILENCE_PAD_MS,
    _pad_wav_with_trailing_silence,
)


def _write_tone(path: Path, duration_ms: int, sample_rate: int = 24000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = round(sample_rate * duration_ms / 1000)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        payload = bytearray()
        for n in range(frames):
            sample = int(7000 * math.sin(2 * math.pi * 220 * n / sample_rate))
            payload += struct.pack("<h", sample)
        handle.writeframes(payload)


def test_pads_the_real_production_shortfall_exactly(tmp_path: Path):
    source = tmp_path / "native_phrase_0010.wav"
    output = tmp_path / "native_phrase_0010.silence-pad.wav"
    _write_tone(source, 34_738)

    result = _pad_wav_with_trailing_silence(
        source_path=source, output_path=output, target_duration_ms=34_800,
    )

    assert result["method"] == "trailing_silence_pad"
    assert result["padding_ms"] == 62
    assert result["final_duration_ms"] == 34_800
    assert result["end_error_ms"] == 0
    assert result["mouth_close_sync_ok"] is True
    assert output.is_file()
    with wave.open(str(output), "rb") as handle:
        params = handle.getparams()
        assert params.nchannels == 1 and params.sampwidth == 2 and params.framerate == 24000


def test_padded_audio_preserves_the_original_content_byte_for_byte(tmp_path: Path):
    source = tmp_path / "source.wav"
    output = tmp_path / "padded.wav"
    _write_tone(source, 1000)

    _pad_wav_with_trailing_silence(source_path=source, output_path=output, target_duration_ms=1100)

    with wave.open(str(source), "rb") as handle:
        original_frames = handle.readframes(handle.getnframes())
    with wave.open(str(output), "rb") as handle:
        padded_frames = handle.readframes(handle.getnframes())
    assert padded_frames[: len(original_frames)] == original_frames
    assert padded_frames[len(original_frames):] == b"\x00" * (len(padded_frames) - len(original_frames))


def test_pad_is_a_no_op_shaped_result_when_source_already_meets_target(tmp_path: Path):
    source = tmp_path / "source.wav"
    output = tmp_path / "padded.wav"
    _write_tone(source, 1000)

    result = _pad_wav_with_trailing_silence(source_path=source, output_path=output, target_duration_ms=1000)

    assert result["padding_ms"] == 0
    assert result["final_duration_ms"] == 1000


def test_silence_pad_cap_is_small_enough_to_be_imperceptible():
    # Sanity guard on the policy constant itself: this must stay a genuinely
    # small, last-mile rounding fix, not a substitute for real repair.
    assert 0 < MAX_UNRESOLVED_EARLY_SILENCE_PAD_MS <= 200
