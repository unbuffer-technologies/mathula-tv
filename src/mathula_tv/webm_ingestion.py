from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


SUPPORTED_VIDEO_INPUT_EXTENSIONS = frozenset({".mp4", ".webm"})


class WebMIngestionError(ValueError):
    """Raised when a WebM source cannot be normalized safely."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_tools() -> None:
    missing = [name for name in ("ffmpeg", "ffprobe") if not shutil.which(name)]
    if missing:
        raise WebMIngestionError("Missing required media tools: " + ", ".join(missing))


def probe_media(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise WebMIngestionError(f"Missing or empty source media: {path}")
    _require_tools()
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise WebMIngestionError(f"ffprobe rejected source media: {result.stderr[-1000:]}")
    data = json.loads(result.stdout)
    duration = float(data.get("format", {}).get("duration") or 0)
    if duration <= 0:
        raise WebMIngestionError(f"Source media has zero duration: {path}")
    streams = data.get("streams", [])
    if not any(item.get("codec_type") == "video" for item in streams):
        raise WebMIngestionError("Source contains no video stream")
    if not any(item.get("codec_type") == "audio" for item in streams):
        raise WebMIngestionError("Source contains no audio stream")
    data["duration"] = duration
    return data


def validate_supported_video(path: Path) -> dict[str, Any]:
    extension = path.suffix.lower()
    if extension not in SUPPORTED_VIDEO_INPUT_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_VIDEO_INPUT_EXTENSIONS))
        raise WebMIngestionError(
            f"Unsupported source video extension: {extension or '<none>'}; supported: {supported}"
        )
    return probe_media(path)


def normalize_webm_to_mp4(
    source: Path,
    output: Path,
    *,
    duration_tolerance_seconds: float = 0.5,
) -> dict[str, Any]:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if source.suffix.lower() != ".webm":
        raise WebMIngestionError("WebM normalization requires a .webm source")

    source_info = validate_supported_video(source)
    source_duration = float(source_info["duration"])
    source_video = next(
        item for item in source_info["streams"] if item.get("codec_type") == "video"
    )
    source_audio = next(
        item for item in source_info["streams"] if item.get("codec_type") == "audio"
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.partial{output.suffix}")
    temporary.unlink(missing_ok=True)
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
            "-sn",
            "-dn",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "48000",
            "-avoid_negative_ts",
            "make_zero",
            "-movflags",
            "+faststart",
            str(temporary),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        temporary.unlink(missing_ok=True)
        raise WebMIngestionError(
            "FFmpeg failed to normalize WebM: " + result.stderr[-1000:]
        )

    try:
        canonical_info = probe_media(temporary)
        canonical_duration = float(canonical_info["duration"])
        duration_delta = abs(canonical_duration - source_duration)
        if duration_delta > duration_tolerance_seconds:
            raise WebMIngestionError(
                "Normalized MP4 duration mismatch: "
                f"source={source_duration:.3f}s canonical={canonical_duration:.3f}s "
                f"delta={duration_delta:.3f}s"
            )
        video = next(
            item for item in canonical_info["streams"] if item.get("codec_type") == "video"
        )
        audio = next(
            item for item in canonical_info["streams"] if item.get("codec_type") == "audio"
        )
        if video.get("codec_name") != "h264":
            raise WebMIngestionError("Canonical MP4 video codec is not H.264")
        if audio.get("codec_name") != "aac":
            raise WebMIngestionError("Canonical MP4 audio codec is not AAC")
        if int(audio.get("sample_rate") or 0) != 48_000:
            raise WebMIngestionError("Canonical MP4 audio is not 48 kHz")
        for field in ("width", "height"):
            if int(video.get(field) or 0) != int(source_video.get(field) or 0):
                raise WebMIngestionError(f"Canonical MP4 changed video {field}")
        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    return {
        "schema_version": "source-normalization-v2",
        "normalized": True,
        "original_path": str(source),
        "original_extension": ".webm",
        "original_sha256": file_sha256(source),
        "original_duration": source_duration,
        "original_video_codec": source_video.get("codec_name"),
        "original_audio_codec": source_audio.get("codec_name"),
        "canonical_path": str(output),
        "canonical_sha256": file_sha256(output),
        "canonical_duration": canonical_duration,
        "canonical_video_codec": video.get("codec_name"),
        "canonical_audio_codec": audio.get("codec_name"),
        "canonical_audio_sample_rate": int(audio.get("sample_rate") or 0),
        "duration_delta": duration_delta,
    }
