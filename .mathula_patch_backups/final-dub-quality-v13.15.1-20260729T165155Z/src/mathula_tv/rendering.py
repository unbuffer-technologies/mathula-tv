"""Validated review rendering without speech truncation or YouTube upload."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any, Callable

from .atomic_io import atomic_write_json
from .errors import RenderFailure
from .media import checksum, probe


def _video_packet_payload_hash(path: Path) -> str:
    """Hash only encoded video packet payloads, ignoring container metadata."""
    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_packets",
                "-show_entries",
                "packet=data_hash",
                "-show_data_hash",
                "sha256",
                "-of",
                "csv=p=0",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RenderFailure(
            "Unable to hash encoded video packets",
            details={"path": str(path), "stderr": exc.stderr},
        ) from exc

    packet_hashes = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not packet_hashes:
        raise RenderFailure(
            "No encoded video packets were available for stream-copy verification",
            details={"path": str(path)},
        )

    digest = hashlib.sha256()
    for packet_hash in packet_hashes:
        digest.update(packet_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


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
    if subtitles is not None:
        raise RenderFailure(
            "Burned subtitles are disabled because the source video stream must remain untouched; use the separate SRT artifact"
        )

    # The product contract is audio replacement only. Never silently fall back to
    # video re-encoding: the source video packets must be stream-copied.
    stream_copy = True
    command = [*base, "-c:v", "copy", *audio_options, str(temporary)]
    try:
        runner(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise RenderFailure(
            "FFmpeg could not stream-copy the source video; video re-encoding is forbidden",
            details={
                "returncode": exc.returncode,
                "stderr": getattr(exc, "stderr", None),
            },
        ) from exc
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
    for field in ("width", "height"):
        if str(output_video_stream.get(field)) != str(source_video_stream.get(field)):
            raise RenderFailure(f"Rendered MP4 changed source video {field}")

    source_video_packet_hash = _video_packet_payload_hash(source_video)
    output_video_packet_hash = _video_packet_payload_hash(temporary)
    if output_video_packet_hash != source_video_packet_hash:
        raise RenderFailure(
            "Rendered MP4 changed the encoded source video packets",
            details={
                "source_video_packet_hash": source_video_packet_hash,
                "output_video_packet_hash": output_video_packet_hash,
            },
        )
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
        "source_video_packet_hash": source_video_packet_hash,
        "output_video_packet_hash": output_video_packet_hash,
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


def render_tiktok_compatible_mp4(
    source_video: Path,
    output: Path,
    manifest_path: Path,
    *,
    frame_rate: int = 25,
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Create a broadly compatible CFR upload derivative from the master MP4."""
    if frame_rate not in {25, 30}:
        raise ValueError("TikTok compatibility frame rate must be 25 or 30 fps")
    source_info = probe(source_video)
    source_streams = source_info.get("streams", [])
    source_video_stream = next(
        (item for item in source_streams if item.get("codec_type") == "video"),
        None,
    )
    source_audio_stream = next(
        (item for item in source_streams if item.get("codec_type") == "audio"),
        None,
    )
    if source_video_stream is None or source_audio_stream is None:
        raise RenderFailure("TikTok compatibility input must contain video and audio")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.partial{output.suffix}")
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source_video),
        "-map",
        "0:v:0",
        "-map",
        "0:a:0",
        "-vf",
        f"fps={frame_rate},format=yuv420p",
        "-fps_mode",
        "cfr",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-profile:v",
        "main",
        "-level:v",
        "4.0",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-ar",
        "48000",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    try:
        runner(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise RenderFailure(
            "FFmpeg could not create the TikTok compatibility derivative",
            details={
                "returncode": exc.returncode,
                "stderr": getattr(exc, "stderr", None),
            },
        ) from exc

    rendered = probe(temporary)
    streams = rendered.get("streams", [])
    video = next(
        (item for item in streams if item.get("codec_type") == "video"),
        None,
    )
    audio = next(
        (item for item in streams if item.get("codec_type") == "audio"),
        None,
    )
    if video is None or audio is None:
        raise RenderFailure("TikTok compatibility derivative is missing a stream")
    if video.get("codec_name") != "h264" or video.get("pix_fmt") != "yuv420p":
        raise RenderFailure("TikTok compatibility video must be H.264 yuv420p")
    if str(video.get("avg_frame_rate")) != f"{frame_rate}/1":
        raise RenderFailure(
            "TikTok compatibility video does not have the requested constant frame rate",
            details={"frame_rate": video.get("avg_frame_rate")},
        )
    if audio.get("codec_name") != "aac" or int(audio.get("sample_rate") or 0) != 48_000:
        raise RenderFailure("TikTok compatibility audio must be 48kHz AAC")
    for field in ("width", "height"):
        if str(video.get(field)) != str(source_video_stream.get(field)):
            raise RenderFailure(f"TikTok compatibility render changed video {field}")
    source_duration = float(source_info["duration"])
    rendered_duration = float(rendered["duration"])
    if abs(rendered_duration - source_duration) > 0.15:
        raise RenderFailure(
            "TikTok compatibility duration does not match the master",
            details={
                "source_duration": source_duration,
                "rendered_duration": rendered_duration,
            },
        )

    temporary.replace(output)
    manifest = {
        "schema_version": "mathula-tiktok-compatible-render-v1",
        "source": {
            "path": str(source_video),
            "sha256": checksum(source_video),
            "frame_rate": source_video_stream.get("avg_frame_rate"),
        },
        "output": {
            "path": str(output),
            "sha256": checksum(output),
            "frame_rate": video.get("avg_frame_rate"),
            "codec": video.get("codec_name"),
            "profile": video.get("profile"),
            "pixel_format": video.get("pix_fmt"),
            "audio_codec": audio.get("codec_name"),
            "audio_sample_rate": int(audio.get("sample_rate") or 0),
        },
        "frame_rate_mode": "constant",
        "video_reencoded": True,
        "master_preserved": True,
        "source_duration_seconds": source_duration,
        "output_duration_seconds": rendered_duration,
    }
    atomic_write_json(manifest_path, manifest)
    return manifest
