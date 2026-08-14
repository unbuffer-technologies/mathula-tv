"""Single-encode upload rendering from timestamped speech clips."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .media import checksum, probe


def _loudnorm_measurement(stderr: str) -> dict[str, float]:
    decoder = json.JSONDecoder()
    values: list[dict[str, Any]] = []
    for index, character in enumerate(stderr):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(stderr[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "input_i" in value:
            values.append(value)
    if not values:
        raise RuntimeError("FFmpeg loudnorm measurement was not returned")
    value = values[-1]
    required = ("input_i", "input_lra", "input_tp", "input_thresh", "target_offset")
    if any(key not in value for key in required):
        raise RuntimeError("FFmpeg loudnorm measurement is incomplete")
    return {key: float(value[key]) for key in required}


def _clip_filters(
    clips: Sequence[Mapping[str, Any]],
    *,
    first_input: int,
    total_duration_ms: int,
    cut_start_ms: int,
    sample_rate: int,
) -> tuple[list[str], str]:
    filters: list[str] = []
    labels: list[str] = []
    for position, clip in enumerate(clips):
        input_index = first_input + position
        label = f"c{position}"
        labels.append(f"[{label}]")
        filters.append(
            f"[{input_index}:a]aresample={sample_rate},asetpts=PTS-STARTPTS,"
            f"afade=t=in:st=0:d=0.02,"
            f"adelay={int(clip['start_ms'])}:all=1[{label}]"
        )
    total_samples = round(total_duration_ms * sample_rate / 1000)
    cut_seconds = cut_start_ms / 1000
    filters.append(
        f"{''.join(labels)}amix=inputs={len(labels)}:normalize=0:"
        "duration=longest:dropout_transition=0,alimiter=limit=0.95,"
        f"apad=whole_len={total_samples},atrim=end_sample={total_samples},"
        f"atrim=start={cut_seconds:.6f},asetpts=PTS-STARTPTS[dialogue]"
    )
    return filters, "dialogue"


def render_consolidated_upload(
    *,
    source_video: Path,
    clips: Sequence[Mapping[str, Any]],
    panel_image: Path,
    output_path: Path,
    total_duration_ms: int,
    cut_start_ms: int,
    target_lufs: float = -16.0,
    target_lra: float = 11.0,
    true_peak_dbtp: float = -1.5,
    sample_rate: int = 48_000,
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Measure audio without video, then encode the upload artifact exactly once."""

    source_video = Path(source_video)
    panel_image = Path(panel_image)
    output_path = Path(output_path)
    if not source_video.is_file():
        raise FileNotFoundError(source_video)
    if not panel_image.is_file():
        raise FileNotFoundError(panel_image)
    if not clips:
        raise ValueError("Consolidated rendering requires speech clips")
    for clip in clips:
        if not Path(str(clip["path"])).is_file():
            raise FileNotFoundError(str(clip["path"]))

    output_duration_ms = total_duration_ms - cut_start_ms
    output_duration_seconds = output_duration_ms / 1000
    output_samples = round(output_duration_ms * sample_rate / 1000)
    raw_fd, raw_name = tempfile.mkstemp(
        prefix="mathula_consolidated_mix_", suffix=".wav", dir="/tmp"
    )
    os.close(raw_fd)
    raw_audio = Path(raw_name)
    raw_audio.unlink()
    mix = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    for clip in clips:
        mix.extend(["-i", str(clip["path"])])
    filters, dialogue_label = _clip_filters(
        clips,
        first_input=0,
        total_duration_ms=total_duration_ms,
        cut_start_ms=cut_start_ms,
        sample_rate=sample_rate,
    )
    mix.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            f"[{dialogue_label}]",
            "-c:a",
            "pcm_s16le",
            "-ar",
            str(sample_rate),
            str(raw_audio),
        ]
    )
    runner(mix, check=True)
    raw_duration = float(probe(raw_audio)["format"]["duration"])
    if abs(raw_duration - output_duration_seconds) > 0.1:
        raw_audio.unlink(missing_ok=True)
        raise RuntimeError(
            "Dialogue mix duration validation failed: "
            f"expected={output_duration_seconds:.3f}s, audio={raw_duration:.3f}s"
        )

    measure = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-y",
        "-i",
        str(raw_audio),
        "-af",
        (
            f"loudnorm=I={target_lufs}:LRA={target_lra}:"
            f"TP={true_peak_dbtp}:print_format=json"
        ),
        "-f",
        "null",
        "-",
    ]
    measured_process = runner(
        measure,
        check=True,
        capture_output=True,
        text=True,
    )
    measured = _loudnorm_measurement(str(measured_process.stderr or ""))

    audio_fd, audio_name = tempfile.mkstemp(
        prefix="mathula_consolidated_audio_", suffix=".wav", dir="/tmp"
    )
    os.close(audio_fd)
    normalized_audio = Path(audio_name)
    normalized_audio.unlink()
    audio_render = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(raw_audio),
        "-af",
        (
            f"loudnorm=I={target_lufs}:LRA={target_lra}:"
            f"TP={true_peak_dbtp}:measured_I={measured['input_i']}:"
            f"measured_LRA={measured['input_lra']}:"
            f"measured_TP={measured['input_tp']}:"
            f"measured_thresh={measured['input_thresh']}:"
            f"offset={measured['target_offset']}:linear=true,"
            f"aresample={sample_rate},"
            f"apad=whole_len={output_samples},"
            f"atrim=end_sample={output_samples},asetpts=PTS-STARTPTS"
        ),
        "-c:a",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        str(normalized_audio),
    ]
    try:
        runner(audio_render, check=True)
    finally:
        raw_audio.unlink(missing_ok=True)
    normalized_duration = float(probe(normalized_audio)["format"]["duration"])
    if abs(normalized_duration - output_duration_seconds) > 0.1:
        normalized_audio.unlink(missing_ok=True)
        raise RuntimeError(
            "Normalized dialogue duration validation failed: "
            f"expected={output_duration_seconds:.3f}s, "
            f"audio={normalized_duration:.3f}s"
        )

    final = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source_video),
        "-i",
        str(normalized_audio),
        "-loop",
        "1",
        "-framerate",
        "25",
        "-i",
        str(panel_image),
    ]
    final_filters = [
        (
            f"[0:v]trim=start={cut_start_ms / 1000:.6f},"
            "setpts=PTS-STARTPTS,fps=25,format=rgba[base]"
        ),
        (
            "[base][2:v]overlay=0:0:shortest=1,"
            "format=yuv420p[video]"
        ),
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.stem}.partial{output_path.suffix}")
    final.extend(
        [
            "-filter_complex",
            ";".join(final_filters),
            "-map",
            "[video]",
            "-map",
            "1:a:0",
            "-t",
            f"{output_duration_seconds:.6f}",
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
            str(sample_rate),
            "-movflags",
            "+faststart",
            str(temporary),
        ]
    )
    try:
        runner(final, check=True)
    finally:
        normalized_audio.unlink(missing_ok=True)
    properties = probe(temporary)
    streams = properties.get("streams", [])
    video_stream = next(
        (stream for stream in streams if stream.get("codec_type") == "video"), None
    )
    audio_stream = next(
        (stream for stream in streams if stream.get("codec_type") == "audio"), None
    )
    if video_stream is None or audio_stream is None:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("Consolidated output must contain video and audio streams")
    video_duration = float(video_stream.get("duration") or properties["format"]["duration"])
    audio_duration = float(audio_stream.get("duration") or 0)
    duration_tolerance_seconds = 0.1
    if (
        abs(video_duration - output_duration_seconds) > duration_tolerance_seconds
        or abs(audio_duration - output_duration_seconds) > duration_tolerance_seconds
    ):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            "Consolidated output duration validation failed: "
            f"expected={output_duration_seconds:.3f}s, "
            f"video={video_duration:.3f}s, audio={audio_duration:.3f}s"
        )
    temporary.replace(output_path)
    return {
        "schema_version": "mathula-consolidated-upload-render-v1",
        "source_video": str(source_video),
        "speech_clip_count": len(clips),
        "cut_start_ms": cut_start_ms,
        "loudness_measurement": measured,
        "video_encode_count": 1,
        "intermediate_media_files": 0,
        "output_path": str(output_path),
        "output_sha256": checksum(output_path),
        "output_probe": properties,
    }


__all__ = ["render_consolidated_upload"]
