"""Validated review rendering without speech truncation or YouTube upload."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Callable

from .atomic_io import atomic_write_json
from .errors import RenderFailure
from .media import checksum, probe


def render_review_mp4(
    source_video: Path,
    final_audio: Path,
    output: Path,
    manifest_path: Path,
    *,
    subtitles: Path | None = None,
    duration_tolerance_seconds: float = 0.12,
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    source_info = probe(source_video)
    audio_info = probe(final_audio)
    source_duration = float(source_info["duration"])
    audio_duration = float(audio_info["duration"])
    if audio_duration > source_duration + duration_tolerance_seconds:
        raise RenderFailure(
            "Final speech is longer than the source video; rendering would truncate speech",
            details={"video_duration": source_duration, "audio_duration": audio_duration},
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.partial{output.suffix}")
    base = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source_video),
        "-i",
        str(final_audio),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
    ]
    audio_options = [
        "-af",
        "apad",
        "-ar",
        "48000",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-t",
        f"{source_duration:.6f}",
        "-movflags",
        "+faststart",
    ]
    stream_copy = subtitles is None
    command = [*base, "-c:v", "copy", *audio_options, str(temporary)]
    if subtitles is not None:
        command = [
            *base,
            "-vf",
            f"subtitles={subtitles}",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            *audio_options,
            str(temporary),
        ]
        stream_copy = False
    try:
        runner(command, check=True)
    except subprocess.CalledProcessError as exc:
        if subtitles is not None:
            raise RenderFailure("FFmpeg failed to render the subtitled review video") from exc
        fallback = [
            *base,
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            *audio_options,
            str(temporary),
        ]
        runner(fallback, check=True)
        stream_copy = False
    rendered = probe(temporary)
    streams = rendered.get("streams", [])
    if not any(item.get("codec_type") == "video" for item in streams):
        raise RenderFailure("Rendered MP4 has no video stream")
    if not any(item.get("codec_type") == "audio" for item in streams):
        raise RenderFailure("Rendered MP4 has no audio stream")
    if abs(float(rendered["duration"]) - source_duration) > max(duration_tolerance_seconds, 0.15):
        raise RenderFailure(
            "Rendered MP4 duration does not match the source video",
            details={"source_duration": source_duration, "rendered_duration": rendered["duration"]},
        )
    source_video_stream = next(item for item in source_info["streams"] if item.get("codec_type") == "video")
    output_video_stream = next(item for item in streams if item.get("codec_type") == "video")
    output_audio_stream = next(item for item in streams if item.get("codec_type") == "audio")
    for field in ("width", "height", "avg_frame_rate"):
        if str(output_video_stream.get(field)) != str(source_video_stream.get(field)):
            raise RenderFailure(f"Rendered MP4 changed source video {field}")
    if output_audio_stream.get("codec_name") != "aac":
        raise RenderFailure("Rendered MP4 audio codec is not AAC")
    if int(output_audio_stream.get("sample_rate") or 0) != 48_000:
        raise RenderFailure("Rendered MP4 audio sample rate is not 48 kHz")
    temporary.replace(output)
    manifest = {
        "schema_version": "render-manifest-v1",
        "source_video": {"path": str(source_video), "sha256": checksum(source_video)},
        "final_audio": {"path": str(final_audio), "sha256": checksum(final_audio)},
        "output": {"path": str(output), "sha256": checksum(output)},
        "video_stream_copy": stream_copy,
        "source_duration_seconds": source_duration,
        "audio_duration_seconds": audio_duration,
        "rendered_duration_seconds": float(rendered["duration"]),
        "source_video_properties": {
            "width": source_video_stream.get("width"),
            "height": source_video_stream.get("height"),
            "frame_rate": source_video_stream.get("avg_frame_rate"),
        },
        "output_video_properties": {
            "width": output_video_stream.get("width"),
            "height": output_video_stream.get("height"),
            "frame_rate": output_video_stream.get("avg_frame_rate"),
        },
        "audio_codec": output_audio_stream.get("codec_name"),
        "audio_sample_rate": int(output_audio_stream.get("sample_rate")),
        "burned_subtitles": str(subtitles) if subtitles else None,
        "youtube_uploaded": False,
    }
    atomic_write_json(manifest_path, manifest)
    return manifest
