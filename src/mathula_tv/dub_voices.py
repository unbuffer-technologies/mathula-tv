"""Which voice will actually speak each speaker's units.

The renderer synthesises every unit with its speaker's assigned voice and prosody
(``direct_dub/voice_resolution.json`` of the original job). Measuring with any other voice -- the generic
default, say -- gives durations that differ by a few percent, enough to turn a "fits" into a quiet speed-up at
render time. So the agent's measurements and the scorer both use these assignments.

A clip's assignments are copied into the frozen corpus the first time they are needed, so the clip stays
self-contained even if the original job is later cleaned up.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence


def load_clip_voices(store: Any, clip_id: str) -> dict[str, dict[str, Any]]:
    """{speaker_id: assignment} for a frozen clip; empty if none can be found (callers fall back to the default)."""
    try:
        frozen = Path(store.clip_dir(clip_id)) / "voice_resolution.json"
        if not frozen.is_file():
            entry = store.load_corpus()["clips"].get(clip_id) or {}
            source = Path(store.root).parent / "jobs" / str(entry.get("job_id")) / "direct_dub" / "voice_resolution.json"
            if not source.is_file():
                return {}
            frozen.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, frozen)
        from .native_dub import load_native_voice_assignments

        return load_native_voice_assignments(frozen.parent, required_speakers=set(), explicit_path=frozen)
    except (AttributeError, OSError, ValueError):
        return {}


def pick_voice_info(
    voices: Mapping[str, Mapping[str, Any]], speaker_id: str | None, start_ms: int, end_ms: int,
    segments: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any] | None:
    """The assignment to measure with, or None (use the default voice) if the clip has no assignments.

    Uses the speaker the caller named when it is a known speaker; otherwise the speaker whose source segment
    overlaps the window most, so an omitted or mistyped speaker_id cannot silently measure with the wrong voice.
    """
    if not voices:
        return None
    if speaker_id in voices:
        return dict(voices[speaker_id])
    best, best_overlap = None, 0
    for segment in segments:
        overlap = min(end_ms, int(segment["end_ms"])) - max(start_ms, int(segment["start_ms"]))
        if overlap > best_overlap and segment.get("speaker_id") in voices:
            best, best_overlap = segment["speaker_id"], overlap
    return dict(voices[best]) if best else dict(next(iter(voices.values())))
