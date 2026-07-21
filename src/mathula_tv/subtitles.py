"""Deterministic isiZulu subtitle artifacts based on approved spoken text."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .atomic_io import atomic_write_json, atomic_write_text
from .media import checksum


def _timestamp(milliseconds: int, separator: str) -> str:
    if milliseconds < 0:
        raise ValueError("Subtitle timestamps cannot be negative")
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}{separator}{millis:03d}"


def subtitle_entries(units: list[dict[str, Any]], alignments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    alignment_by_id = {item["unit_id"]: item for item in alignments}
    entries: list[dict[str, Any]] = []
    for unit in sorted(units, key=lambda item: (int(item["start_ms"]), str(item["unit_id"]))):
        alignment = alignment_by_id.get(unit["unit_id"])
        if alignment is None:
            raise ValueError(f"Missing final alignment for {unit['unit_id']}")
        text = str(unit.get("spoken_text", "")).strip()
        if not text:
            raise ValueError(f"Missing spoken_text for {unit['unit_id']}")
        if text == str(unit.get("tts_text", "")).strip() and unit.get("tts_text_only"):
            raise ValueError("Subtitle text may not be sourced from a TTS-only representation")
        start_ms = int(unit["start_ms"])
        end_ms = start_ms + int(alignment["final_duration_ms"])
        entries.append(
            {
                "subtitle_id": f"sub_{len(entries) + 1:04d}",
                "unit_id": unit["unit_id"],
                "speaker_id": unit["speaker_id"],
                "start_ms": start_ms,
                "end_ms": end_ms,
                "spoken_text": text,
                "has_overlap": bool(unit.get("has_overlap", False)),
                "overlap_group_id": unit.get("overlap_group_id"),
            }
        )
    return entries


def write_subtitles(
    units: list[dict[str, Any]],
    alignments: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    entries = subtitle_entries(units, alignments)
    output_dir.mkdir(parents=True, exist_ok=True)
    spoken_json = output_dir / "spoken_zu.json"
    srt = output_dir / "subtitles_zu.srt"
    vtt = output_dir / "subtitles_zu.vtt"
    atomic_write_json(
        spoken_json,
        {"schema_version": "spoken-subtitles-v1", "language": "zu-ZA", "entries": entries},
    )
    atomic_write_text(
        srt,
        "\n\n".join(
            f"{index}\n{_timestamp(item['start_ms'], ',')} --> {_timestamp(item['end_ms'], ',')}\n{item['spoken_text']}"
            for index, item in enumerate(entries, 1)
        )
        + "\n",
    )
    atomic_write_text(
        vtt,
        "WEBVTT\n\n"
        + "\n\n".join(
            f"{item['unit_id']}\n{_timestamp(item['start_ms'], '.')} --> {_timestamp(item['end_ms'], '.')}\n{item['spoken_text']}"
            for item in entries
        )
        + "\n",
    )
    return {
        "schema_version": "subtitle-manifest-v1",
        "language": "zu-ZA",
        "entry_count": len(entries),
        "artifacts": {
            "spoken_zu": {"path": str(spoken_json), "sha256": checksum(spoken_json)},
            "srt": {"path": str(srt), "sha256": checksum(srt)},
            "vtt": {"path": str(vtt), "sha256": checksum(vtt)},
        },
    }
