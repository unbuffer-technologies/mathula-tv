"""Overlap-preserving dialogue tracks, buses, loudness, and final mixing."""

from __future__ import annotations

import re
import subprocess
import wave
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from .errors import TimingBoundExceeded
from .media import checksum


def render_dialogue_tracks(
    clips: list[dict[str, Any]],
    output_dir: Path,
    dialogue_output: Path,
    *,
    total_duration_ms: int,
    sample_rate: int = 24000,
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Build smooth, overlap-preserving per-speaker tracks and a dialogue bus.

    The old renderer used 20 ms linear fades on every block.  Back-to-back
    sentence fragments therefore dipped twice at every boundary and could sound
    clipped or choppy.  The quality renderer uses short adaptive equal-power
    fades only for de-clicking, stable sample-clock regeneration, and gentle
    bus compression after timeline placement.
    """
    if total_duration_ms <= 0 or not clips:
        raise ValueError("Dialogue rendering requires clips and a positive duration")
    speakers: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for clip in clips:
        if not Path(clip["path"]).is_file():
            raise FileNotFoundError(clip["path"])
        clip_end_ms = int(clip["start_ms"]) + int(clip["duration_ms"])
        if clip_end_ms > total_duration_ms:
            raise TimingBoundExceeded(
                f"Unit {clip.get('unit_id')} extends beyond the source duration; "
                "dialogue mixing will not truncate speech",
                details={
                    "unit_id": clip.get("unit_id"),
                    "clip_end_ms": clip_end_ms,
                    "source_duration_ms": total_duration_ms,
                },
            )
        speakers[str(clip["speaker_id"])].append(clip)

    output_dir.mkdir(parents=True, exist_ok=True)
    tracks: list[dict[str, Any]] = []
    for speaker, items in sorted(speakers.items()):
        output = output_dir / f"{_safe_id(speaker)}.wav"
        command = _speaker_track_command(items, output, total_duration_ms, sample_rate)
        runner(command, check=True)
        quality = _validate_pcm(output, sample_rate, channels=1)
        tracks.append(
            {
                "speaker_id": speaker,
                "path": str(output),
                "sha256": checksum(output),
                "unit_count": len(items),
                "quality": quality,
            }
        )

    mix_command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    for track in tracks:
        mix_command.extend(["-i", track["path"]])
    inputs = "".join(f"[{index}:a]" for index in range(len(tracks)))
    total_samples = round(total_duration_ms * sample_rate / 1000)
    mix_filter = (
        f"{inputs}amix=inputs={len(tracks)}:normalize=0:dropout_transition=0.08,"
        f"aresample={sample_rate}:async=1:first_pts=0,"
        "acompressor=threshold=0.16:ratio=1.35:attack=6:release=140:makeup=1.0:knee=2.0,"
        "alimiter=limit=0.92:attack=5:release=80,"
        f"apad=whole_len={total_samples},atrim=end_sample={total_samples},asetpts=N/SR/TB[a]"
    )
    mix_command.extend(
        [
            "-filter_complex",
            mix_filter,
            "-map",
            "[a]",
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(dialogue_output),
        ]
    )
    dialogue_output.parent.mkdir(parents=True, exist_ok=True)
    runner(mix_command, check=True)
    quality = _validate_pcm(dialogue_output, sample_rate, channels=1)
    overlaps = overlap_decisions(clips)
    return {
        "schema_version": "dialogue-bus-v2-smooth-boundaries",
        "speaker_tracks": tracks,
        "dialogue_bus": {"path": str(dialogue_output), "sha256": checksum(dialogue_output), **quality},
        "overlaps": overlaps,
        "overlap_count": len(overlaps),
        "gain_management": "adaptive equal-power de-click fades, gentle compression, peak limiter",
        "boundary_smoothing": {
            "fade_curve": "qsin",
            "minimum_fade_ms": 3,
            "maximum_fade_ms": 10,
            "sample_clock_regenerated": True,
        },
    }



def _speaker_track_command(
    clips: list[dict[str, Any]], output: Path, total_duration_ms: int, sample_rate: int
) -> list[str]:
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    ordered = sorted(
        clips,
        key=lambda item: (int(item["start_ms"]), str(item["unit_id"])),
    )
    filters: list[str] = []
    for index, clip in enumerate(ordered):
        command.extend(["-i", str(clip["path"])])
        duration = max(0.001, float(clip.get("duration_ms", 0)) / 1000.0)
        # Enough to eliminate waveform discontinuities, but short enough not to
        # swallow consonants or create a double dip at sentence boundaries.
        fade = min(0.010, max(0.003, duration * 0.025))
        fade_out_start = max(0.0, duration - fade)
        filters.append(
            f"[{index}:a]aresample={sample_rate}:async=1:first_pts=0,"
            "asetpts=N/SR/TB,"
            f"afade=t=in:st=0:d={fade:.4f}:curve=qsin,"
            f"afade=t=out:st={fade_out_start:.4f}:d={fade:.4f}:curve=qsin,"
            f"adelay={int(clip['start_ms'])}:all=1[c{index}]"
        )
    inputs = "".join(f"[c{index}]" for index in range(len(ordered)))
    total_samples = round(total_duration_ms * sample_rate / 1000)
    filters.append(
        f"{inputs}amix=inputs={len(ordered)}:duration=longest:normalize=0:dropout_transition=0.08,"
        "highpass=f=60,"
        "acompressor=threshold=0.14:ratio=1.55:attack=7:release=120:makeup=1.03:knee=2.0,"
        "alimiter=limit=0.92:attack=5:release=70,"
        f"apad=whole_len={total_samples},atrim=end_sample={total_samples},asetpts=N/SR/TB[a]"
    )
    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[a]",
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(output),
        ]
    )
    return command



def overlap_decisions(clips: list[dict[str, Any]]) -> list[dict[str, Any]]:
    decisions = []
    ordered = sorted(clips, key=lambda item: (int(item["start_ms"]), str(item["unit_id"])))
    for index, left in enumerate(ordered):
        left_end = int(left["start_ms"]) + int(left["duration_ms"])
        for right in ordered[index + 1 :]:
            if int(right["start_ms"]) >= left_end:
                break
            if left["speaker_id"] == right["speaker_id"]:
                continue
            end = min(left_end, int(right["start_ms"]) + int(right["duration_ms"]))
            decisions.append(
                {
                    "left_unit_id": left["unit_id"],
                    "right_unit_id": right["unit_id"],
                    "start_ms": int(right["start_ms"]),
                    "end_ms": end,
                    "duration_ms": end - int(right["start_ms"]),
                    "placement": "preserved_on_separate_speaker_tracks",
                    "gain_management": "limiter",
                    "human_review_required": end - int(right["start_ms"]) > 2000,
                }
            )
    return decisions


def mix_final_buses(
    background: Path,
    dialogue: Path,
    output: Path,
    *,
    target_lufs: float = -16.0,
    target_lra: float = 9.0,
    true_peak_dbtp: float = -1.0,
    background_gain_db: float = -5.0,
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Create a stable broadcast-style final mix without abrupt ambience pumping."""
    output.parent.mkdir(parents=True, exist_ok=True)
    filter_graph = (
        "[0:a]aresample=48000:async=1:first_pts=0,"
        f"volume={background_gain_db}dB[bg];"
        "[1:a]aresample=48000:async=1:first_pts=0,"
        "pan=stereo|c0=c0|c1=c0,asplit=2[side][dialogue];"
        "[bg][side]sidechaincompress="
        "threshold=0.025:ratio=7:attack=8:release=450:makeup=1[ducked];"
        "[ducked][dialogue]amix=inputs=2:normalize=0:weights='1 1':dropout_transition=0.25,"
        "acompressor=threshold=0.18:ratio=1.35:attack=12:release=220:makeup=1.0:knee=2.0,"
        f"loudnorm=I={target_lufs}:LRA={target_lra}:TP={true_peak_dbtp},"
        "alimiter=limit=0.89:attack=5:release=100,asetpts=N/SR/TB[mix]"
    )
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(background),
        "-i",
        str(dialogue),
        "-filter_complex",
        filter_graph,
        "-map",
        "[mix]",
        "-ar",
        "48000",
        "-ac",
        "2",
        "-c:a",
        "pcm_s16le",
        str(output),
    ]
    runner(command, check=True)
    quality = _validate_pcm(output, 48000, channels=2)
    loudness = measure_loudness(output, runner=runner)
    return {
        "schema_version": "final-mix-v2-smooth-broadcast",
        "dialogue_sha256": checksum(dialogue),
        "background_sha256": checksum(background),
        "output_path": str(output),
        "output_sha256": checksum(output),
        "target_lufs": target_lufs,
        "target_lra": target_lra,
        "target_true_peak_dbtp": true_peak_dbtp,
        "dialogue_to_background": {
            "background_gain_db": background_gain_db,
            "sidechain_ducking": True,
            "attack_ms": 8,
            "release_ms": 450,
        },
        "sample_clock_regenerated": True,
        "loudness": loudness,
        **quality,
    }



def measure_loudness(path: Path, *, runner: Callable[..., Any] = subprocess.run) -> dict[str, Any]:
    result = runner(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-i",
            str(path),
            "-filter_complex",
            "ebur128=peak=true",
            "-f",
            "null",
            "-",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stderr = getattr(result, "stderr", "") or ""
    integrated = _last_float(stderr, r"I:\s*(-?[0-9.]+)\s*LUFS")
    lra = _last_float(stderr, r"LRA:\s*([0-9.]+)\s*LU")
    peak = _last_float(stderr, r"Peak:\s*(-?[0-9.]+)\s*dBFS")
    return {
        "integrated_lufs": integrated,
        "loudness_range_lu": lra,
        "true_peak_dbtp": peak,
        "measured": integrated is not None,
    }


def _last_float(value: str, pattern: str) -> float | None:
    matches = re.findall(pattern, value)
    return float(matches[-1]) if matches else None


def _validate_pcm(path: Path, sample_rate: int, *, channels: int) -> dict[str, Any]:
    with wave.open(str(path), "rb") as handle:
        if (
            handle.getframerate() != sample_rate
            or handle.getnchannels() != channels
            or handle.getsampwidth() != 2
            or handle.getnframes() <= 0
        ):
            raise ValueError("Unexpected PCM bus format")
        raw = handle.readframes(handle.getnframes())
        duration = handle.getnframes() / sample_rate
    samples = [int.from_bytes(raw[index : index + 2], "little", signed=True) for index in range(0, len(raw), 2)]
    clipping = sum(abs(value) >= 32767 for value in samples)
    peak = max((abs(value) for value in samples), default=0)
    return {
        "duration_seconds": duration,
        "sample_rate": sample_rate,
        "channels": channels,
        "sample_width": 2,
        "peak_pcm16": peak,
        "clipping_count": clipping,
    }


def _safe_id(value: str) -> str:
    safe = "".join(character for character in value if character.isalnum() or character in "-_")
    if not safe:
        raise ValueError("Invalid speaker ID")
    return safe
