"""Validated review rendering without speech truncation or YouTube upload."""

from __future__ import annotations

import hashlib
import math
import os
import subprocess
import time
from pathlib import Path
from fractions import Fraction
from typing import Any, Callable, Mapping, Sequence

from .atomic_io import atomic_write_json
from .errors import RenderFailure
from .media import checksum, probe


def _emit_progress(
    callback: Callable[[dict[str, Any]], None] | None,
    event: Mapping[str, Any],
) -> None:
    if callback is None:
        return
    try:
        callback(dict(event))
    except Exception:
        # Progress output must never alter rendering correctness.
        return


def _ffmpeg_progress_seconds(values: Mapping[str, str]) -> float:
    for key in ("out_time_us", "out_time_ms"):
        raw = str(values.get(key) or "").strip()
        if raw:
            try:
                return max(0.0, float(raw) / 1_000_000.0)
            except ValueError:
                pass
    clock = str(values.get("out_time") or "").strip()
    if clock:
        try:
            hours, minutes, seconds = clock.split(":", 2)
            return max(
                0.0,
                float(hours) * 3600.0
                + float(minutes) * 60.0
                + float(seconds),
            )
        except (TypeError, ValueError):
            pass
    return 0.0


def _positive_float(value: Any) -> float:
    try:
        parsed = float(str(value or "").strip().casefold().removesuffix("x"))
    except (TypeError, ValueError):
        return 0.0
    return parsed if math.isfinite(parsed) and parsed > 0 else 0.0


def _integer(value: Any) -> int:
    try:
        return max(0, int(str(value or "0").strip()))
    except (TypeError, ValueError):
        return 0


def run_ffmpeg_with_progress(
    command: Sequence[str],
    *,
    expected_duration_seconds: float,
    progress_callback: Callable[[dict[str, Any]], None],
    popen_factory: Callable[..., Any] = subprocess.Popen,
    clock: Callable[[], float] = time.monotonic,
    stage: str = "video_render",
) -> None:
    """Run one FFmpeg command and expose machine-readable encode progress."""

    progress_command = [
        *command[:-1],
        "-stats_period",
        "1",
        "-progress",
        "pipe:1",
        "-nostats",
        command[-1],
    ]
    total = max(1, int(round(expected_duration_seconds)))
    started = clock()
    _emit_progress(
        progress_callback,
        {
            "stage": stage,
            "status": "started",
            "message": "Starting FFmpeg clean-master encode",
            "current": 0,
            "total": total,
            "percent": 0.0,
            "stage_elapsed_seconds": 0.0,
        },
    )
    process = popen_factory(
        progress_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    stdout = getattr(process, "stdout", None)
    if stdout is None:
        raise RuntimeError("FFmpeg progress process has no stdout pipe")
    values: dict[str, str] = {}
    last_percent = -1.0
    last_reported_at = started
    final_event: dict[str, Any] | None = None
    for raw_line in stdout:
        line = str(raw_line).strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value
        if key != "progress":
            continue
        now = clock()
        processed = min(
            max(0.0, expected_duration_seconds),
            _ffmpeg_progress_seconds(values),
        )
        percent = (
            min(100.0, processed * 100.0 / expected_duration_seconds)
            if expected_duration_seconds > 0
            else 100.0
        )
        completed = value == "end"
        if completed:
            processed = expected_duration_seconds
            percent = 100.0
        should_report = (
            completed
            or percent >= last_percent + 1.0
            or now - last_reported_at >= 5.0
        )
        if not should_report:
            values.clear()
            continue
        elapsed = max(0.0, now - started)
        speed = _positive_float(values.get("speed"))
        remaining = max(0.0, expected_duration_seconds - processed)
        eta = (
            remaining / speed
            if speed > 0
            else (
                elapsed * remaining / processed
                if processed > 0
                else 0.0
            )
        )
        speed_text = f"{speed:.2f}x" if speed > 0 else "calculating speed"
        frame = _integer(values.get("frame"))
        fps = str(values.get("fps") or "").strip()
        encoded_bytes = _integer(values.get("total_size"))
        detail_parts = [speed_text, f"frame {frame:,}"]
        if fps:
            detail_parts.append(f"{fps} fps")
        if encoded_bytes:
            detail_parts.append(f"{encoded_bytes / (1024 * 1024):.1f} MiB")
        event = {
            "stage": stage,
            "status": "completed" if completed else "running",
            "message": (
                "FFmpeg clean-master encode completed"
                if completed
                else (
                    "FFmpeg clean-master encoding: "
                    + ", ".join(detail_parts)
                )
            ),
            "current": min(total, max(0, int(round(processed)))),
            "total": total,
            "percent": round(percent, 2),
            "processed_seconds": round(processed, 3),
            "expected_duration_seconds": round(expected_duration_seconds, 3),
            "speed": speed,
            "fps": fps,
            "frame": frame,
            "encoded_bytes": encoded_bytes,
            "eta_seconds": round(eta, 3),
            "stage_elapsed_seconds": round(elapsed, 3),
        }
        last_percent = percent
        last_reported_at = now
        if completed:
            final_event = event
        else:
            _emit_progress(progress_callback, event)
        values.clear()

    stderr_stream = getattr(process, "stderr", None)
    stderr = str(stderr_stream.read() or "") if stderr_stream is not None else ""
    returncode = int(process.wait())
    if returncode:
        raise subprocess.CalledProcessError(
            returncode,
            progress_command,
            stderr=stderr,
        )
    if final_event is None:
        final_event = {
            "stage": stage,
            "status": "completed",
            "message": "FFmpeg clean-master encode completed",
            "current": total,
            "total": total,
            "percent": 100.0,
            "processed_seconds": round(expected_duration_seconds, 3),
            "expected_duration_seconds": round(expected_duration_seconds, 3),
            "speed": 0.0,
            "fps": "",
            "frame": 0,
            "encoded_bytes": 0,
            "eta_seconds": 0.0,
            "stage_elapsed_seconds": round(max(0.0, clock() - started), 3),
        }
    _emit_progress(progress_callback, final_event)


def _parse_frame_rate(value: Any) -> Fraction | None:
    if value in (None, "", "N/A", "0/0"):
        return None
    try:
        rate = Fraction(str(value))
    except (ValueError, ZeroDivisionError):
        return None
    numeric = float(rate)
    if not math.isfinite(numeric) or numeric < 12.0 or numeric > 120.0:
        return None
    return rate


def select_playback_frame_rate(
    video_stream: dict[str, Any],
    *,
    requested: int | float | str | Fraction | None = None,
) -> Fraction:
    """Choose a stable CFR rate while preserving the source's real cadence."""
    explicit = _parse_frame_rate(requested)
    if explicit is not None:
        return explicit
    candidate = _parse_frame_rate(video_stream.get("avg_frame_rate"))
    if candidate is None:
        candidate = _parse_frame_rate(video_stream.get("r_frame_rate"))
    if candidate is None:
        return Fraction(25, 1)
    standards = (
        Fraction(24000, 1001),
        Fraction(24, 1),
        Fraction(25, 1),
        Fraction(30000, 1001),
        Fraction(30, 1),
        Fraction(50, 1),
        Fraction(60000, 1001),
        Fraction(60, 1),
    )
    nearest = min(standards, key=lambda item: abs(float(item) - float(candidate)))
    if abs(float(nearest) - float(candidate)) / float(nearest) <= 0.012:
        return nearest
    return candidate.limit_denominator(100_000)


def frame_rate_text(rate: Fraction | str | int | float) -> str:
    parsed = rate if isinstance(rate, Fraction) else _parse_frame_rate(rate)
    if parsed is None:
        raise ValueError(f"Invalid frame rate: {rate!r}")
    return f"{parsed.numerator}/{parsed.denominator}"


def frame_rates_match(left: Any, right: Any, *, tolerance: float = 0.001) -> bool:
    left_rate = _parse_frame_rate(left)
    right_rate = _parse_frame_rate(right)
    if left_rate is None or right_rate is None:
        return False
    return abs(float(left_rate) - float(right_rate)) / float(right_rate) <= tolerance


def _video_level(width: int, height: int, rate: Fraction) -> str:
    pixels_per_second = width * height * float(rate)
    return "4.2" if pixels_per_second > 1920 * 1080 * 30 else "4.1"


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
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    popen_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Render a smooth playback master with regenerated CFR timestamps.

    Set ``MATHULA_TV_FINAL_VIDEO_MODE=stream_copy`` for an emergency archival
    remux.  The production default is a high-quality CFR H.264 master because
    irregular VFR/DTS timestamps are a common cause of visibly choppy playback.
    """
    source_info = probe(source_video)
    audio_info = probe(final_audio)
    source_duration = float(source_info["duration"])
    audio_duration = float(audio_info["duration"])
    if audio_duration > source_duration + duration_tolerance_seconds:
        raise RenderFailure(
            "Final speech is longer than the source video; rendering would truncate speech",
            details={"video_duration": source_duration, "audio_duration": audio_duration},
        )
    if subtitles is not None:
        raise RenderFailure(
            "Burned subtitles are disabled; use the separate SRT artifact"
        )

    source_video_stream = next(
        item for item in source_info["streams"] if item.get("codec_type") == "video"
    )
    target_rate = select_playback_frame_rate(source_video_stream)
    target_rate_text = frame_rate_text(target_rate)
    width = int(source_video_stream.get("width") or 0)
    height = int(source_video_stream.get("height") or 0)
    if width <= 0 or height <= 0:
        raise RenderFailure("Source video has invalid dimensions")

    mode = os.getenv("MATHULA_TV_FINAL_VIDEO_MODE", "smooth_cfr").strip().lower()
    if mode not in {"smooth_cfr", "stream_copy"}:
        raise RenderFailure(
            "MATHULA_TV_FINAL_VIDEO_MODE must be smooth_cfr or stream_copy",
            details={"value": mode},
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.partial{output.suffix}")
    temporary.unlink(missing_ok=True)

    base = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-fflags",
        "+genpts",
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
        "aresample=48000:async=1:first_pts=0,apad,asetpts=N/SR/TB",
        "-ar",
        "48000",
        "-c:a",
        "aac",
        "-b:a",
        os.getenv("MATHULA_TV_FINAL_AUDIO_BITRATE", "256k"),
        "-t",
        f"{source_duration:.6f}",
        "-avoid_negative_ts",
        "make_zero",
        "-movflags",
        "+faststart",
    ]

    if mode == "stream_copy":
        command = [*base, "-c:v", "copy", *audio_options, str(temporary)]
    else:
        preset = os.getenv("MATHULA_TV_FINAL_VIDEO_PRESET", "medium").strip() or "medium"
        crf = os.getenv("MATHULA_TV_FINAL_VIDEO_CRF", "17").strip() or "17"
        keyint = max(12, round(float(target_rate) * 2.0))
        keyint_min = max(6, round(float(target_rate)))
        video_filter = (
            "setpts=PTS-STARTPTS,"
            f"fps=fps={target_rate_text}:round=near,"
            "format=yuv420p"
        )
        command = [
            *base,
            "-vf",
            video_filter,
            "-fps_mode",
            "cfr",
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            crf,
            "-profile:v",
            "high",
            "-level:v",
            _video_level(width, height, target_rate),
            "-pix_fmt",
            "yuv420p",
            "-g",
            str(keyint),
            "-keyint_min",
            str(keyint_min),
            "-sc_threshold",
            "40",
            "-video_track_timescale",
            "90000",
            *audio_options,
            str(temporary),
        ]

    try:
        if progress_callback is not None and popen_factory is not None:
            run_ffmpeg_with_progress(
                command,
                expected_duration_seconds=source_duration,
                progress_callback=progress_callback,
                popen_factory=popen_factory,
                stage="video_render",
            )
        else:
            runner(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise RenderFailure(
            "FFmpeg could not create the final dubbed master",
            details={
                "render_mode": mode,
                "returncode": exc.returncode,
                "stderr": getattr(exc, "stderr", None),
            },
        ) from exc

    rendered = probe(temporary)
    streams = rendered.get("streams", [])
    output_video_stream = next(
        (item for item in streams if item.get("codec_type") == "video"), None
    )
    output_audio_stream = next(
        (item for item in streams if item.get("codec_type") == "audio"), None
    )
    if output_video_stream is None or output_audio_stream is None:
        raise RenderFailure("Rendered MP4 is missing video or audio")
    if abs(float(rendered["duration"]) - source_duration) > max(duration_tolerance_seconds, 0.15):
        raise RenderFailure(
            "Rendered MP4 duration does not match the source video",
            details={"source_duration": source_duration, "rendered_duration": rendered["duration"]},
        )
    for field in ("width", "height"):
        if str(output_video_stream.get(field)) != str(source_video_stream.get(field)):
            raise RenderFailure(f"Rendered MP4 changed source video {field}")
    if output_audio_stream.get("codec_name") != "aac":
        raise RenderFailure("Rendered MP4 audio codec is not AAC")
    if int(output_audio_stream.get("sample_rate") or 0) != 48_000:
        raise RenderFailure("Rendered MP4 audio sample rate is not 48 kHz")

    source_packet_hash = None
    output_packet_hash = None
    if mode == "stream_copy":
        source_packet_hash = _video_packet_payload_hash(source_video)
        output_packet_hash = _video_packet_payload_hash(temporary)
        if output_packet_hash != source_packet_hash:
            raise RenderFailure("Stream-copy mode changed encoded video packets")
    else:
        if output_video_stream.get("codec_name") != "h264":
            raise RenderFailure("Smooth final master must use H.264")
        if output_video_stream.get("pix_fmt") != "yuv420p":
            raise RenderFailure("Smooth final master must use yuv420p")
        if not frame_rates_match(output_video_stream.get("avg_frame_rate"), target_rate):
            raise RenderFailure(
                "Smooth final master does not have the selected constant frame rate",
                details={
                    "expected": target_rate_text,
                    "actual": output_video_stream.get("avg_frame_rate"),
                },
            )

    temporary.replace(output)
    manifest = {
        "schema_version": "render-manifest-v2-smooth-cfr",
        "source_video": {"path": str(source_video), "sha256": checksum(source_video)},
        "final_audio": {"path": str(final_audio), "sha256": checksum(final_audio)},
        "output": {"path": str(output), "sha256": checksum(output)},
        "render_mode": mode,
        "video_stream_copy": mode == "stream_copy",
        "source_video_packet_hash": source_packet_hash,
        "output_video_packet_hash": output_packet_hash,
        "source_duration_seconds": source_duration,
        "audio_duration_seconds": audio_duration,
        "rendered_duration_seconds": float(rendered["duration"]),
        "source_video_properties": {
            "width": source_video_stream.get("width"),
            "height": source_video_stream.get("height"),
            "avg_frame_rate": source_video_stream.get("avg_frame_rate"),
            "r_frame_rate": source_video_stream.get("r_frame_rate"),
        },
        "output_video_properties": {
            "width": output_video_stream.get("width"),
            "height": output_video_stream.get("height"),
            "frame_rate": output_video_stream.get("avg_frame_rate"),
            "codec": output_video_stream.get("codec_name"),
            "pixel_format": output_video_stream.get("pix_fmt"),
        },
        "playback_stability": {
            "target_frame_rate": target_rate_text,
            "constant_frame_rate": mode == "smooth_cfr",
            "timestamps_regenerated": mode == "smooth_cfr",
            "video_track_timescale": 90000 if mode == "smooth_cfr" else None,
        },
        "audio_codec": output_audio_stream.get("codec_name"),
        "audio_sample_rate": int(output_audio_stream.get("sample_rate")),
        "audio_bitrate": os.getenv("MATHULA_TV_FINAL_AUDIO_BITRATE", "256k"),
        "burned_subtitles": None,
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
