from __future__ import annotations

from pathlib import Path

import pytest

import mathula_tv.direct_azure_dub as direct
from mathula_tv.direct_azure_dub import DirectDubError, load_approved_segments


def _translation(*, start_ms: int, end_ms: int) -> dict[str, object]:
    return {
        "segments": [
            {
                "segment_id": "seg-tail",
                "speaker_id": "SPEAKER_00",
                "start_ms": start_ms,
                "end_ms": end_ms,
                "source_text": "Hello",
                "translated_text": "Sawubona",
                "tts_text": "Sawubona",
            }
        ]
    }


def test_accepted_translation_tail_is_clamped_to_source() -> None:
    segments = load_approved_segments(
        _translation(start_ms=9_000, end_ms=10_080),
        source_duration_ms=10_000,
    )
    assert len(segments) == 1
    assert segments[0].start_ms == 9_000
    assert segments[0].end_ms == 10_000
    assert segments[0].duration_ms == 1_000


def test_translation_tail_beyond_tolerance_still_fails() -> None:
    with pytest.raises(DirectDubError, match="ends beyond the source video"):
        load_approved_segments(
            _translation(start_ms=9_000, end_ms=10_251),
            source_duration_ms=10_000,
        )


def test_exact_wav_overrun_is_reconciled_without_truncation(monkeypatch) -> None:
    measured = iter((2.512, 2.500))
    monkeypatch.setattr(direct, "probe", lambda _path: {"duration": next(measured)})
    calls: list[dict[str, object]] = []

    def fake_stretch(source, output, **kwargs):
        calls.append({"source": source, "output": output, **kwargs})

    monkeypatch.setattr(direct, "_time_stretch_to_window", fake_stretch)
    audio = Path("block.wav")
    result = direct._enforce_exact_audio_window(
        audio,
        block_id="block_0017_variant_compact",
        target_duration_ms=2_500,
        existing_stretch_ratio=1.0,
        maximum_time_stretch_ratio=1.03,
        sample_rate=48_000,
        runner=lambda *args, **kwargs: None,
    )

    assert result["corrected"] is True
    assert result["duration_ms"] == 2_500
    assert result["correction_ratio"] == pytest.approx(2_512 / 2_500)
    assert len(calls) == 1
    assert calls[0]["source"] == audio
    assert calls[0]["output"] == audio
    assert calls[0]["target_duration_ms"] == 2_500


def test_exact_boundary_rejects_a_real_timing_overflow(monkeypatch) -> None:
    monkeypatch.setattr(direct, "probe", lambda _path: {"duration": 2.550})
    monkeypatch.setattr(
        direct,
        "_time_stretch_to_window",
        lambda *args, **kwargs: pytest.fail("unsafe correction must not run"),
    )

    with pytest.raises(DirectDubError, match="technical correction maximum"):
        direct._enforce_exact_audio_window(
            Path("block.wav"),
            block_id="block_0017_variant_compact",
            target_duration_ms=2_500,
            existing_stretch_ratio=1.0,
            maximum_time_stretch_ratio=1.03,
            sample_rate=48_000,
            runner=lambda *args, **kwargs: None,
        )
