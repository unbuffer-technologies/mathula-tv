"""Unit tests for overlap-preserving dialogue rendering and final mixing."""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import requires_ffmpeg, write_wav

from mathula_tv import mixing
from mathula_tv.errors import TimingBoundExceeded


def _clip(unit_id, speaker_id, start_ms, duration_ms, path):
    return {
        "unit_id": unit_id,
        "speaker_id": speaker_id,
        "start_ms": start_ms,
        "duration_ms": duration_ms,
        "path": str(path),
    }


def test_safe_id_sanitizes_and_rejects_empty():
    assert mixing._safe_id("AZURE_1") == "AZURE_1"
    assert mixing._safe_id("speaker/../x") == "speakerx"
    with pytest.raises(ValueError, match="Invalid speaker ID"):
        mixing._safe_id("/@#")


def test_overlap_decisions_skip_same_speaker_and_flag_long_overlaps():
    decisions = mixing.overlap_decisions(
        [
            {"unit_id": "u1", "speaker_id": "A", "start_ms": 0, "duration_ms": 3000},
            {"unit_id": "u2", "speaker_id": "B", "start_ms": 100, "duration_ms": 2500},
            {"unit_id": "u3", "speaker_id": "A", "start_ms": 200, "duration_ms": 200},
        ]
    )
    # Same-speaker A/A overlap is not reported; cross-speaker overlaps are.
    pairs = {(item["left_unit_id"], item["right_unit_id"]) for item in decisions}
    assert ("u1", "u3") not in pairs
    assert ("u1", "u2") in pairs
    long_overlap = next(item for item in decisions if item["left_unit_id"] == "u1" and item["right_unit_id"] == "u2")
    assert long_overlap["duration_ms"] > 2000
    assert long_overlap["human_review_required"] is True


def test_overlap_decisions_ignores_non_overlapping_clips():
    decisions = mixing.overlap_decisions(
        [
            {"unit_id": "u1", "speaker_id": "A", "start_ms": 0, "duration_ms": 500},
            {"unit_id": "u2", "speaker_id": "B", "start_ms": 600, "duration_ms": 500},
        ]
    )
    assert decisions == []


def test_render_dialogue_tracks_requires_clips_and_duration(tmp_path):
    with pytest.raises(ValueError, match="requires clips"):
        mixing.render_dialogue_tracks(
            [], tmp_path / "tracks", tmp_path / "bus.wav", total_duration_ms=0
        )


def test_render_dialogue_tracks_missing_clip_path(tmp_path):
    clips = [_clip("u1", "A", 0, 500, tmp_path / "does_not_exist.wav")]
    with pytest.raises(FileNotFoundError):
        mixing.render_dialogue_tracks(
            clips, tmp_path / "tracks", tmp_path / "bus.wav", total_duration_ms=1000
        )


def test_render_dialogue_tracks_rejects_out_of_bounds_clip(tmp_path):
    audio = write_wav(tmp_path / "a.wav", 500)
    clips = [_clip("u1", "A", 900, 500, audio)]
    with pytest.raises(TimingBoundExceeded):
        mixing.render_dialogue_tracks(
            clips, tmp_path / "tracks", tmp_path / "bus.wav", total_duration_ms=1000
        )


@requires_ffmpeg
def test_render_dialogue_tracks_builds_per_speaker_bus(tmp_path):
    a1 = write_wav(tmp_path / "a1.wav", 500)
    b1 = write_wav(tmp_path / "b1.wav", 500)
    clips = [
        _clip("u1", "A", 0, 500, a1),
        _clip("u2", "B", 400, 500, b1),
    ]
    result = mixing.render_dialogue_tracks(
        clips, tmp_path / "tracks", tmp_path / "dialogue.wav", total_duration_ms=1200
    )
    assert result["schema_version"] == "dialogue-bus-v1"
    assert {track["speaker_id"] for track in result["speaker_tracks"]} == {"A", "B"}
    assert result["overlap_count"] == len(result["overlaps"]) == 1
    assert (tmp_path / "dialogue.wav").is_file()
    assert result["dialogue_bus"]["sample_rate"] == 24000


@requires_ffmpeg
def test_measure_loudness_parses_ebur128(tmp_path):
    audio = write_wav(tmp_path / "loud.wav", 1500, rate=48000, channels=2, amplitude=12000)
    measurement = mixing.measure_loudness(audio)
    assert measurement["measured"] is True
    assert measurement["integrated_lufs"] is not None
    assert measurement["true_peak_dbtp"] is not None


def test_mix_final_buses_orchestrates_and_validates(tmp_path):
    # The ffmpeg mix graph is version-fragile, so drive it with a runner that
    # emulates ffmpeg: it produces a valid master WAV and ebur128 diagnostics.
    background = write_wav(tmp_path / "bg.wav", 1500, rate=48000, channels=2, amplitude=4000)
    dialogue = write_wav(tmp_path / "dlg.wav", 1500, rate=24000, channels=1, amplitude=9000)
    output = tmp_path / "final.wav"

    def fake_runner(command, **kwargs):
        if "ebur128=peak=true" in command:
            class _Result:
                stderr = "I: -14.0 LUFS\nLRA: 8.0 LU\nPeak: -1.6 dBFS\n"

            return _Result()
        write_wav(Path(command[-1]), 1500, rate=48000, channels=2, amplitude=6000)
        return None

    result = mixing.mix_final_buses(background, dialogue, output, runner=fake_runner)
    assert result["schema_version"] == "final-mix-v1"
    assert result["channels"] == 2 and result["sample_rate"] == 48000
    assert result["loudness"] == {
        "integrated_lufs": -14.0,
        "loudness_range_lu": 8.0,
        "true_peak_dbtp": -1.6,
        "measured": True,
    }
    assert result["target_lufs"] == -14 and result["target_true_peak_dbtp"] == -1.5
    assert result["output_sha256"] == mixing.checksum(output)


def test_last_float_returns_last_match():
    text = "I: -20.0 LUFS\nI: -14.5 LUFS"
    assert mixing._last_float(text, r"I:\s*(-?[0-9.]+)\s*LUFS") == -14.5
    assert mixing._last_float("no match here", r"I:\s*(-?[0-9.]+)") is None


def test_validate_pcm_rejects_wrong_format(tmp_path):
    audio = write_wav(tmp_path / "mono.wav", 500, rate=16000, channels=1)
    with pytest.raises(ValueError, match="Unexpected PCM bus format"):
        mixing._validate_pcm(audio, 48000, channels=2)
