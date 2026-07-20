"""Job-local, same-speaker reference-reel selection and construction."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Callable

from .atomic_io import atomic_write_json
from .errors import InsufficientReferenceQuality, MissingSameSpeakerReference
from .media import checksum
from .quality import inspect_wav


PREFERRED_SOURCE_TYPES = ("isolated_dialogue", "clean_source_segment", "denoised_source", "mixed_source_fallback")
PUBLICATION_SOURCE_TYPES = {"isolated_dialogue", "clean_source_segment"}


def select_reference_candidates(
    speaker_id: str,
    candidates: list[dict[str, Any]],
    *,
    minimum_total_seconds: float = 20,
    maximum_total_seconds: float = 45,
    proof_of_concept: bool = False,
) -> dict[str, Any]:
    if speaker_id == "UNKNOWN":
        raise MissingSameSpeakerReference("UNKNOWN cannot have a voice-conversion reference")
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    rank = {value: index for index, value in enumerate(PREFERRED_SOURCE_TYPES)}
    ordered = sorted(
        candidates,
        key=lambda item: (
            rank.get(str(item.get("audio_source_type")), len(rank)),
            -float(item.get("quality_score", 0)),
            -max(0.0, float(item.get("end", 0)) - float(item.get("start", 0))),
        ),
    )
    total = 0.0
    for item in ordered:
        reasons = _rejection_reasons(speaker_id, item)
        duration = max(0.0, float(item.get("end", 0)) - float(item.get("start", 0)))
        if not reasons and total + duration > maximum_total_seconds:
            remaining = maximum_total_seconds - total
            if remaining >= 1.0:
                item = {**item, "end": float(item["start"]) + remaining, "trimmed_for_reel": True}
                duration = remaining
            else:
                reasons.append("maximum reel duration reached")
        if reasons:
            rejected.append({"segment_id": item.get("segment_id"), "reasons": reasons})
            continue
        accepted.append(dict(item))
        total += duration
        if total >= maximum_total_seconds:
            break
    if not accepted:
        raise MissingSameSpeakerReference(f"No clean same-speaker reference exists for {speaker_id}")
    below_preferred = total < minimum_total_seconds
    if below_preferred and not proof_of_concept:
        raise InsufficientReferenceQuality(
            f"Reference reel for {speaker_id} has only {total:.2f}s; {minimum_total_seconds:.2f}s is required",
            details={"speaker_id": speaker_id, "duration_seconds": total},
        )
    source_types = sorted({str(item["audio_source_type"]) for item in accepted})
    return {
        "speaker_id": speaker_id,
        "selected": accepted,
        "rejected": rejected,
        "selected_duration_seconds": round(total, 3),
        "below_preferred_duration": below_preferred,
        "source_types": source_types,
        "publication_reference": not below_preferred and set(source_types) <= PUBLICATION_SOURCE_TYPES,
        "proof_of_concept": proof_of_concept,
    }


def build_reference_reel(
    selection: dict[str, Any],
    output: Path,
    manifest_path: Path,
    *,
    sample_rate: int = 24000,
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    selected = selection["selected"]
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    filters = []
    for index, item in enumerate(selected):
        source = Path(item["path"])
        if not source.is_file():
            raise FileNotFoundError(source)
        command.extend(
            [
                "-ss",
                str(float(item["start"])),
                "-t",
                str(float(item["end"]) - float(item["start"])),
                "-i",
                str(source),
            ]
        )
        filters.append(f"[{index}:a]aresample={sample_rate},aformat=sample_fmts=s16:channel_layouts=mono[r{index}]")
    inputs = "".join(f"[r{index}]" for index in range(len(selected)))
    filters.append(f"{inputs}concat=n={len(selected)}:v=0:a=1[out]")
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = output.with_name(f".{output.name}.partial.wav")
    command.extend(["-filter_complex", ";".join(filters), "-map", "[out]", "-c:a", "pcm_s16le", str(staged)])
    runner(command, check=True)
    quality = inspect_wav(staged)
    if not quality.get("valid") or quality.get("sample_rate") != sample_rate:
        raise InsufficientReferenceQuality("Constructed reference reel failed WAV quality validation")
    staged.replace(output)
    manifest = {
        "schema_version": "speaker-reference-manifest-v1",
        **selection,
        "reference_reel_path": str(output),
        "reference_reel_sha256": checksum(output),
        "final_duration_seconds": quality["duration"],
        "quality_metrics": quality,
        "source_segments": [
            {
                "segment_id": item.get("segment_id"),
                "start": item["start"],
                "end": item["end"],
                "source_transcript": item.get("source_transcript"),
                "audio_source_type": item["audio_source_type"],
                "source_sha256": checksum(Path(item["path"])),
            }
            for item in selected
        ],
        "job_local_identity": True,
        "global_identity_created": False,
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def _rejection_reasons(speaker_id: str, item: dict[str, Any]) -> list[str]:
    reasons = []
    if item.get("speaker_id") != speaker_id:
        reasons.append("cross-speaker candidate")
    if item.get("speaker_id") == "UNKNOWN":
        reasons.append("UNKNOWN speaker")
    if item.get("overlap"):
        reasons.append("overlapping speech")
    duration = float(item.get("end", 0)) - float(item.get("start", 0))
    if duration < 1:
        reasons.append("insufficient voiced duration")
    if float(item.get("silence_ratio", 0)) > 0.3:
        reasons.append("excessive silence")
    if float(item.get("music_ratio", 0)) > 0.2:
        reasons.append("music-heavy")
    if float(item.get("clipping_ratio", 0)) > 0.001:
        reasons.append("clipping")
    if float(item.get("reverberation_score", 0)) > 0.7:
        reasons.append("excessive reverberation")
    if item.get("audio_source_type") not in PREFERRED_SOURCE_TYPES:
        reasons.append("unsupported audio source type")
    if not item.get("path"):
        reasons.append("missing audio source")
    return reasons
