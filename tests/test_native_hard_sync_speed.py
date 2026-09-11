from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

import pytest

from mathula_tv.native_dub import (
    _effective_mouth_close_tolerance,
    _hard_sync_speed_factor,
    _mouth_close_sync_ok,
    _schedule_source_timeline,
    _speed_fit_mode_label,
    _speed_fit_overflow_wav,
    _temporal_required_rush_percent,
    _timing_recast_direction,
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


def test_hard_sync_speed_formula() -> None:
    assert _hard_sync_speed_factor(1100, 1000) == pytest.approx(1.1)
    assert _hard_sync_speed_factor(900, 1000) == 1.0


def test_temporal_required_rush_percent_matches_observed_production_value() -> None:
    # 54.38s raw against a 38.64s mouth window measured +40.7% in a live run.
    assert _temporal_required_rush_percent(54_380, 38_640) == pytest.approx(40.735, abs=0.01)


def test_temporal_required_rush_percent_is_never_negative_when_audio_already_fits() -> None:
    assert _temporal_required_rush_percent(900, 1000) == 0.0
    assert _temporal_required_rush_percent(1000, 1000) == 0.0


# --- speed-fit mode labeling ---------------------------------------------------------
# Two automatic modes: "sentence_end" (no usable gap before the next sentence -- rush
# to the source sentence's own end, unchanged from before this feature) and
# "gap_aware" (a real, already-safe gap exists -- see _effective_protected_late_
# allowance -- so the speed-fit target extends into it). Selection is per-sentence and
# automatic, never a global flag; the label just makes which one applied observable.

def test_speed_fit_mode_is_sentence_end_with_no_usable_gap() -> None:
    assert _speed_fit_mode_label(0) == "sentence_end"


def test_speed_fit_mode_is_gap_aware_with_a_real_gap() -> None:
    assert _speed_fit_mode_label(40) == "gap_aware"
    assert _speed_fit_mode_label(360) == "gap_aware"




def test_mouth_close_tolerance_never_creates_next_block_overlap() -> None:
    assert _effective_mouth_close_tolerance(40, 1000, 1000) == 0
    assert _effective_mouth_close_tolerance(40, 1000, 1015) == 15
    assert _effective_mouth_close_tolerance(40, 1000, 1200) == 40



def test_zero_gap_caps_only_late_tolerance() -> None:
    # No free room means a phrase may not finish late at all, but a small early
    # landing remains safe because it cannot create overlap with the next block.
    late = _effective_mouth_close_tolerance(40, 1000, 1000)
    assert late == 0
    assert _mouth_close_sync_ok(-27, early_tolerance_ms=40, late_tolerance_ms=late) is True
    assert _mouth_close_sync_ok(+1, early_tolerance_ms=40, late_tolerance_ms=late) is False
    assert _timing_recast_direction(973, 1000, max_speed_percent=12, mouth_close_early_tolerance_ms=40, mouth_close_late_tolerance_ms=late) is None

def test_mouth_close_recast_margin_distinguishes_rescuable_lag() -> None:
    # 1160 ms can be brought to <=1040 ms with +12%, so no wording recast.
    assert _timing_recast_direction(1160, 1000, max_speed_percent=12, mouth_close_early_tolerance_ms=40, mouth_close_late_tolerance_ms=40) is None
    # 1180 ms still lands too late at +12%, so wording must compress first.
    assert _timing_recast_direction(1180, 1000, max_speed_percent=12, mouth_close_early_tolerance_ms=40, mouth_close_late_tolerance_ms=40) == "compress"
    # Short speech is never slowed; when timing repair exists it is expanded instead.
    assert _timing_recast_direction(900, 1000, max_speed_percent=12, mouth_close_early_tolerance_ms=40, mouth_close_late_tolerance_ms=40) == "expand"


def test_hard_sync_can_use_max_speed_inside_mouth_close_tolerance(tmp_path: Path) -> None:
    source = tmp_path / "source-tolerance.wav"
    output = tmp_path / "fit-tolerance.wav"
    _write_tone(source, 1160)
    result = _speed_fit_overflow_wav(
        source_path=source,
        output_path=output,
        target_duration_ms=1000,
        max_speed_percent=12,
        allowed_late_ms=40,
    )
    assert result["method"] == "ffmpeg_atempo"
    assert float(result["speed_percent"]) <= 12.0
    assert int(result["end_error_ms"]) <= 40
    assert result["mouth_close_sync_ok"] is True



def test_hard_sync_accepts_small_early_landing_when_next_gap_is_zero(tmp_path: Path) -> None:
    source = tmp_path / "source-zero-gap.wav"
    output = tmp_path / "fit-zero-gap.wav"
    _write_tone(source, 1070)
    result = _speed_fit_overflow_wav(
        source_path=source,
        output_path=output,
        target_duration_ms=1000,
        max_speed_percent=12,
        allowed_late_ms=0,
        allowed_early_ms=40,
    )
    assert -40 <= int(result["end_error_ms"]) <= 0
    assert result["mouth_close_sync_ok"] is True

def test_hard_sync_atempo_lands_inside_source_window(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    output = tmp_path / "fit.wav"
    _write_tone(source, 1100)
    result = _speed_fit_overflow_wav(
        source_path=source,
        output_path=output,
        target_duration_ms=1000,
        max_speed_percent=12,
    )
    assert result["method"] == "ffmpeg_atempo"
    assert 9.5 <= float(result["speed_percent"]) <= 11.0
    assert int(result["final_duration_ms"]) <= 1000
    assert int(result["final_duration_ms"]) >= 970


def test_hard_sync_refuses_unnatural_speed_before_paraphrase(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    _write_tone(source, 1200)
    with pytest.raises(RuntimeError, match="above the natural bound"):
        _speed_fit_overflow_wav(
            source_path=source,
            output_path=tmp_path / "fit.wav",
            target_duration_ms=1000,
            max_speed_percent=12,
        )


def test_hard_sync_schedule_never_shifts_start_or_borrows_gap() -> None:
    scheduled = _schedule_source_timeline([
        {
            "group_id": "g1",
            "speaker_id": "SPEAKER_00",
            "start_ms": 100,
            "source_end_ms": 1100,
            "source_span_ms": 1000,
            "tts_duration_ms": 995,
        },
        {
            "group_id": "g2",
            "speaker_id": "SPEAKER_00",
            "start_ms": 1400,
            "source_end_ms": 2400,
            "source_span_ms": 1000,
            "tts_duration_ms": 900,
        },
    ])
    assert scheduled[0]["scheduled_start_ms"] == 100
    assert scheduled[0]["source_gap_borrowed_ms"] == 0
    assert scheduled[1]["scheduled_start_ms"] == 1400
    assert scheduled[1]["same_speaker_lag_ms"] == 0
    assert scheduled[1]["source_gap_borrowed_ms"] == 0



def test_protected_gap_overflow_preserves_pause_and_rescues_borderline_phrase() -> None:
    from mathula_tv.native_dub import _effective_protected_late_allowance

    late = _effective_protected_late_allowance(
        40,
        1000,
        1480,
        protected_gap_overflow=True,
        max_protected_gap_overflow_ms=400,
        min_protected_pause_ms=120,
    )
    assert late == 360
    # Mirrors native_phrase_0024 geometry: 5.462s raw for a 4.720s source window.
    assert _timing_recast_direction(
        5462,
        4720,
        max_speed_percent=12,
        mouth_close_early_tolerance_ms=40,
        mouth_close_late_tolerance_ms=late,
    ) is None


def test_protected_gap_overflow_does_not_hide_large_mismatch() -> None:
    from mathula_tv.native_dub import _effective_protected_late_allowance

    late = _effective_protected_late_allowance(
        40,
        1000,
        1080,
        protected_gap_overflow=True,
        max_protected_gap_overflow_ms=400,
        min_protected_pause_ms=120,
    )
    assert late == 40
    # Mirrors native_phrase_0021: tiny source gap cannot rescue a large overrun.
    assert _timing_recast_direction(
        6112,
        4320,
        max_speed_percent=12,
        mouth_close_early_tolerance_ms=40,
        mouth_close_late_tolerance_ms=late,
    ) == "compress"


def test_protected_gap_overflow_can_be_disabled() -> None:
    from mathula_tv.native_dub import _effective_protected_late_allowance

    assert _effective_protected_late_allowance(
        40,
        1000,
        1480,
        protected_gap_overflow=False,
        max_protected_gap_overflow_ms=400,
        min_protected_pause_ms=120,
    ) == 40
