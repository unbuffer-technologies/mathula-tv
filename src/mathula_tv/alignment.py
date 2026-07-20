"""Post-OpenVoice timing decisions and pitch-preserving correction."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import wave
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from .errors import InvalidWAV, TimingBoundExceeded
from .media import checksum


def wav_duration_ms(path: Path) -> int:
    try:
        with wave.open(str(path), "rb") as handle:
            if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
                raise InvalidWAV("Alignment input must be mono PCM16")
            if handle.getframerate() <= 0 or handle.getnframes() <= 0:
                raise InvalidWAV("Alignment input has no audio frames")
            return round(handle.getnframes() * 1000 / handle.getframerate())
    except (wave.Error, EOFError) as exc:
        raise InvalidWAV(f"Malformed alignment WAV: {path.name}") from exc


def timing_decision(
    converted_duration_ms: int,
    unit: dict[str, Any],
    *,
    max_speed: float = 1.06,
    tolerance_ms: int = 120,
) -> dict[str, Any]:
    preferred = int(unit["preferred_duration_ms"])
    maximum = int(unit["maximum_duration_ms"])
    if min(converted_duration_ms, preferred, maximum) <= 0 or preferred > maximum:
        raise ValueError("Invalid dubbing-unit timing window")

    if converted_duration_ms <= preferred + tolerance_ms:
        target = max(preferred, converted_duration_ms)
        return {
            "action": "silence_pad" if converted_duration_ms < target else "none",
            "speed": 1.0,
            "target_duration_ms": target,
            "padding_ms": max(0, target - converted_duration_ms),
            "timing_error_ms": converted_duration_ms - preferred,
            "repair_required": False,
        }
    if converted_duration_ms <= maximum + tolerance_ms:
        return {
            "action": "none",
            "speed": 1.0,
            "target_duration_ms": converted_duration_ms,
            "padding_ms": 0,
            "timing_error_ms": converted_duration_ms - preferred,
            "repair_required": False,
        }

    required_speed = converted_duration_ms / maximum
    if required_speed > max_speed:
        return {
            "action": "translation_repair",
            "speed": required_speed,
            "target_duration_ms": maximum,
            "padding_ms": 0,
            "timing_error_ms": converted_duration_ms - preferred,
            "repair_required": True,
            "reason": "required pitch-preserving correction exceeds configured bound",
        }
    return {
        "action": "atempo",
        "speed": required_speed,
        "target_duration_ms": maximum,
        "padding_ms": 0,
        "timing_error_ms": maximum - preferred,
        "repair_required": False,
    }


def atempo_chain(speed: float) -> str:
    """Build a valid FFmpeg atempo chain without changing sample rate."""
    if speed <= 0:
        raise ValueError("Speed must be positive")
    factors: list[float] = []
    remaining = speed
    while remaining > 2:
        factors.append(2.0)
        remaining /= 2
    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5
    factors.append(remaining)
    return ",".join(f"atempo={factor:.8f}" for factor in factors)


def align_converted_unit(
    source: Path,
    output: Path,
    unit: dict[str, Any],
    *,
    min_speed: float = 0.94,
    max_speed: float = 1.06,
    tolerance_ms: int = 120,
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Align one converted unit, never truncating speech or slowing short audio."""
    if not 0 < min_speed <= 1 <= max_speed:
        raise ValueError("Invalid post-conversion speed bounds")
    before = wav_duration_ms(source)
    decision = timing_decision(before, unit, max_speed=max_speed, tolerance_ms=tolerance_ms)
    if decision["repair_required"]:
        raise TimingBoundExceeded(
            f"Unit {unit.get('unit_id')} requires speed {decision['speed']:.4f}, above {max_speed:.4f}",
            details={"unit_id": unit.get("unit_id"), **decision},
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary_dir:
        temporary_root = Path(temporary_dir)
        corrected = temporary_root / "corrected.wav"
        if decision["action"] == "atempo":
            runner(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(source),
                    "-filter:a",
                    atempo_chain(float(decision["speed"])),
                    "-c:a",
                    "pcm_s16le",
                    str(corrected),
                ],
                check=True,
            )
        else:
            shutil.copyfile(source, corrected)

        corrected_ms = wav_duration_ms(corrected)
        target_ms = max(corrected_ms, int(decision["target_duration_ms"]))
        staged = temporary_root / "aligned.wav"
        padding_ms = _copy_with_silence_padding(corrected, staged, target_ms)
        staged.replace(output)

    final_ms = wav_duration_ms(output)
    maximum = int(unit["maximum_duration_ms"])
    if final_ms > maximum + tolerance_ms:
        raise TimingBoundExceeded(
            f"Aligned unit {unit.get('unit_id')} exceeds its hard window",
            details={"final_duration_ms": final_ms, "maximum_duration_ms": maximum},
        )
    return {
        "schema_version": "alignment-result-v1",
        "unit_id": unit.get("unit_id"),
        "speaker_id": unit.get("speaker_id"),
        "output_path": str(output),
        "source_duration_ms": before,
        "corrected_duration_ms": corrected_ms,
        "final_duration_ms": final_ms,
        "speed": round(float(decision["speed"]), 8),
        "method": "ffmpeg_atempo" if decision["action"] == "atempo" else "none",
        "silence_padding_ms": padding_ms,
        "preferred_duration_ms": int(unit["preferred_duration_ms"]),
        "maximum_duration_ms": maximum,
        "final_timing_error_ms": final_ms - int(unit["preferred_duration_ms"]),
        "sha256": checksum(output),
        "warnings": [],
    }


def _copy_with_silence_padding(source: Path, output: Path, target_ms: int) -> int:
    with wave.open(str(source), "rb") as src:
        params = src.getparams()
        frames = src.readframes(src.getnframes())
    current_frames = params.nframes
    target_frames = max(current_frames, round(target_ms * params.framerate / 1000))
    padding_frames = target_frames - current_frames
    with wave.open(str(output), "wb") as dst:
        dst.setparams((params.nchannels, params.sampwidth, params.framerate, 0, "NONE", "not compressed"))
        dst.writeframes(frames)
        dst.writeframes(b"\0" * padding_frames * params.nchannels * params.sampwidth)
    return round(padding_frames * 1000 / params.framerate)


def measured_openvoice_ratios(results: list[dict[str, Any]]) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for item in results:
        source = float(item.get("source_duration_ms") or 0)
        converted = float(item.get("converted_duration_ms") or 0)
        if source > 0 and converted > 0:
            values[str(item["speaker_id"])].append(converted / source)
    return {speaker: sum(items) / len(items) for speaker, items in sorted(values.items())}


def predict_converted_duration_ms(source_duration_ms: int, speaker_id: str, ratios: dict[str, float]) -> int:
    return round(source_duration_ms * ratios.get(speaker_id, 1.0))
