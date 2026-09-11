from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .errors import InvalidSourceMedia, UnsupportedSourceDuration


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_media_tools() -> None:
    missing = [name for name in ("ffmpeg", "ffprobe") if not shutil.which(name)]
    if missing:
        raise RuntimeError("Missing required tools: " + ", ".join(missing))


def probe(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Missing or empty media: {path}")
    require_media_tools()
    result = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)], check=True, capture_output=True, text=True)
    data = json.loads(result.stdout)
    duration = float(data.get("format", {}).get("duration") or 0)
    if duration <= 0:
        raise ValueError(f"Media has zero duration: {path}")
    data["duration"] = duration
    return data


def ffmpeg_version() -> str:
    require_media_tools()
    return subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True, text=True).stdout.splitlines()[0]


def extract_clip(
    source: Path, start_seconds: float, end_seconds: float, output: Path, *, pad_seconds: float = 0.15,
) -> Path:
    """Mono 16kHz PCM slice of `source` from start_seconds to end_seconds,
    padded by pad_seconds each side (clamped at 0) for coarticulation context.
    Seek-before-input (fast, not frame-exact) is fine here: callers use this
    for short reference clips fed to phone recognition, not for anything
    requiring sample-accurate timing.
    """
    require_media_tools()
    output.parent.mkdir(parents=True, exist_ok=True)
    padded_start = max(0.0, start_seconds - pad_seconds)
    duration = max(0.05, (end_seconds + pad_seconds) - padded_start)
    temporary = output.with_suffix(".partial.wav")
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{padded_start:.3f}", "-i", str(source), "-t", f"{duration:.3f}",
            "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            str(temporary),
        ],
        check=True, capture_output=True,
    )
    temporary.replace(output)
    return output


def prepare_audio(source: Path, output: Path) -> dict[str, Any]:
    source_info = probe(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".partial.wav")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source), "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(temporary)], check=True)
    temporary.replace(output)
    audio_info = probe(output)
    stream: dict[str, Any] = next(
        (s for s in audio_info["streams"] if s.get("codec_type") == "audio"), {}
    )
    return {"source_duration": source_info["duration"], "normalized_audio_duration": audio_info["duration"], "sample_rate": int(stream.get("sample_rate", 0)), "channels": int(stream.get("channels", 0)), "audio_checksum": checksum(output), "ffmpeg_version": ffmpeg_version()}


def prepare_source_derivatives(
    source: Path,
    analysis_output: Path,
    mix_output: Path,
    *,
    max_duration_seconds: float = 240 * 60,
) -> dict[str, Any]:
    """Preserve distinct STT and production-mix derivatives atomically."""
    source_info = probe(source)
    duration = float(source_info["duration"])
    if duration > max_duration_seconds:
        raise UnsupportedSourceDuration(
            f"Source duration {duration:.3f}s exceeds the reviewed 240-minute limit",
            details={"duration_seconds": duration, "limit_seconds": max_duration_seconds},
        )
    if not any(stream.get("codec_type") == "audio" for stream in source_info.get("streams", [])):
        raise InvalidSourceMedia("Source contains no audio stream")

    outputs = (
        (analysis_output, ["-ac", "1", "-ar", "16000"]),
        (mix_output, ["-ac", "2", "-ar", "48000"]),
    )
    for output, audio_args in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.stem}.partial{output.suffix}")
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source),
                "-map",
                "0:a:0",
                "-vn",
                *audio_args,
                "-c:a",
                "pcm_s16le",
                str(temporary),
            ],
            check=True,
        )
        temporary.replace(output)

    analysis_info = _validate_pcm_derivative(analysis_output, channels=1, sample_rate=16000, expected_duration=duration)
    mix_info = _validate_pcm_derivative(mix_output, channels=2, sample_rate=48000, expected_duration=duration)
    return {
        "schema_version": "source-audio-v1",
        "source_duration": duration,
        "source_probe": source_info,
        "ffmpeg_version": ffmpeg_version(),
        "analysis_mono": analysis_info,
        "mix_source_stereo": mix_info,
        # Backward-compatible names used by the existing Azure worker.
        "normalized_audio_duration": analysis_info["duration"],
        "sample_rate": analysis_info["sample_rate"],
        "channels": analysis_info["channels"],
        "audio_checksum": analysis_info["sha256"],
    }


def _validate_pcm_derivative(path: Path, *, channels: int, sample_rate: int, expected_duration: float) -> dict[str, Any]:
    info = probe(path)
    stream = next((item for item in info.get("streams", []) if item.get("codec_type") == "audio"), None)
    if not stream:
        raise InvalidSourceMedia(f"Prepared derivative has no audio stream: {path.name}")
    actual_channels = int(stream.get("channels") or 0)
    actual_rate = int(stream.get("sample_rate") or 0)
    if actual_channels != channels or actual_rate != sample_rate:
        raise InvalidSourceMedia(
            f"Prepared derivative format mismatch: {path.name}",
            details={"channels": actual_channels, "sample_rate": actual_rate},
        )
    if abs(float(info["duration"]) - expected_duration) > 0.25:
        raise InvalidSourceMedia(f"Prepared derivative duration mismatch: {path.name}")
    return {
        "path": str(path),
        "duration": float(info["duration"]),
        "sample_rate": actual_rate,
        "channels": actual_channels,
        "sample_width": 2,
        "codec": stream.get("codec_name"),
        "sha256": checksum(path),
        "ffprobe": info,
    }


def render_video(source: Path, audio: Path, output: Path) -> None:
    probe(source); probe(audio)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".partial.mp4")
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source), "-i", str(audio), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", str(temporary)]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError:
        command[command.index("copy")] = "libx264"
        # TikTok re-encodes every upload to its own delivery profile, so a slow,
        # high-fidelity local encode here buys nothing durable -- this path only
        # runs when the cheap stream-copy attempt above failed anyway.
        command[command.index("-c:a"):command.index("-c:a")] = ["-preset", "veryfast", "-crf", "20"]
        subprocess.run(command, check=True)
    temporary.replace(output)
    probe(output)
