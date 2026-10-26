"""YouTube's own captions for a video, as one more reading of what was said.

The system only dubs YouTube videos, and YouTube already ran its own recogniser over the audio.  For a rally that mixes English and
isiZulu its automatic captions are tagged with the language it heard ("zu-orig" for Zulu) and are often right exactly where the English
recogniser is nonsense ("siyayibhadala through tax", "ANC ayinamali, ANC ayibhadali abasebenzi", "nale yabantwana siyayinyusa"), and
wrong where the English one is right (some plain English sentences).  So they are an independent second reading, not a replacement for
the transcript: they join the readings that ``code_switch`` weighs against the sounds of the audio.

Every other automatic track on the video is a machine translation of the original one and is never fetched.  Captions are optional: a
video without them, or a failed fetch, changes nothing.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

DIRECTORY = Path("input") / "youtube_captions"


def find(job_root: Path) -> list[Path]:
    """The caption files stored with a job, the uploader's own subtitles first and YouTube's automatic original-language track after."""
    folder = Path(job_root) / DIRECTORY
    files = sorted(folder.glob("*.json3")) if folder.is_dir() else []
    return sorted(files, key=lambda path: (path.name.split(".")[-2].endswith("-orig"), path.name))


def events(path: Path) -> list[tuple[float, float, str]]:
    """(start, end, text) of each caption line in seconds, in order, without the empty line-break events."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    lines = []
    for event in data.get("events") or []:
        text = "".join(str(seg.get("utf8") or "") for seg in event.get("segs") or []).replace("\n", " ").strip()
        if not text:
            continue
        start = int(event.get("tStartMs") or 0) / 1000.0
        lines.append((start, start + int(event.get("dDurationMs") or 0) / 1000.0, text))
    return lines


def reading(job_root: Path) -> Callable[[float, float], str] | None:
    """A reader returning the caption text that overlaps a window of the audio, or None when the video has no captions."""
    files = find(job_root)
    if not files:
        return None
    lines = events(files[0])
    if not lines:
        return None

    def read(start: float, end: float) -> str:
        seen: list[str] = []
        for line_start, line_end, text in lines:
            if line_end > start and line_start < end and text not in seen:
                seen.append(text)
        return " ".join(seen)

    return read
