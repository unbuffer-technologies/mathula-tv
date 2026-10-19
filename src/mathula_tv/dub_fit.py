"""Approach-agnostic timing measurement: real Azure TTS duration of a text against a window.

Used by BOTH the dub-workbench MCP server (what an agent measures while it works)
and the episode scorer (what the agent is scored on). Sharing one function is the
point: an agent must never be able to optimise a different number from the one it
is graded by. Wraps the same primitives the candidate-pool pipeline measures with
(``_temporal_mask_geometry`` + ``_measure_temporal_candidate``), so decision-time
numbers are comparable across a pipeline run and an agent run.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping


def _group(
    *, item_id: str, speaker_id: str, start_ms: int, end_ms: int, source_text: str,
) -> dict[str, Any]:
    return {
        "group_id": item_id, "speaker_id": speaker_id, "start_ms": int(start_ms),
        "source_end_ms": int(end_ms), "source_span_ms": max(1, int(end_ms) - int(start_ms)),
        "source_text": source_text, "segment_ids": [f"{item_id}_seg"],
    }


def measure_text_fit(
    tts: Any,
    *,
    text: str,
    start_ms: int,
    end_ms: int,
    next_start_ms: int | None = None,
    speaker_id: str = "SPEAKER_00",
    source_text: str = "",
    item_id: str = "measure",
    output_root: Path,
    voice: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Synthesize ``text`` for real and report how it sits in ``[start_ms, end_ms]``.

    ``next_start_ms`` is where the next speech begins; it bounds how much of the
    gap after this window the text may borrow, exactly as the pipeline does.
    """
    from .native_dub import (
        BRIDGE_MAX_PROTECTED_GAP_OVERFLOW_MS,
        DEFAULT_MAX_NATURAL_SPEED_PERCENT,
        DEFAULT_MAX_PROTECTED_GAP_OVERFLOW_MS,
        DEFAULT_MIN_PROTECTED_PAUSE_MS,
        DEFAULT_PREFERRED_RAW_SPEED_PERCENT,
        MOUTH_CLOSE_TARGET_TOLERANCE_MS,
        _BRIDGE_MAX_SOURCE_SPAN_MS,
        _BRIDGE_MAX_WORD_COUNT,
        _measure_temporal_candidate,
        _temporal_mask_geometry,
    )

    if int(end_ms) <= int(start_ms):
        raise ValueError("end_ms must be greater than start_ms")
    if not str(text or "").strip():
        raise ValueError("text is empty")
    group = _group(
        item_id=item_id, speaker_id=speaker_id, start_ms=start_ms, end_ms=end_ms, source_text=source_text,
    )
    is_bridge = (
        group["source_span_ms"] <= _BRIDGE_MAX_SOURCE_SPAN_MS
        and len(re.findall(r"\w+", source_text)) <= _BRIDGE_MAX_WORD_COUNT
    )
    geometry = _temporal_mask_geometry(
        group,
        next_source_start_ms=int(next_start_ms) if next_start_ms is not None else int(end_ms),
        preferred_raw_speed_percent=DEFAULT_PREFERRED_RAW_SPEED_PERCENT,
        max_natural_speed_percent=DEFAULT_MAX_NATURAL_SPEED_PERCENT,
        mouth_tolerance_ms=MOUTH_CLOSE_TARGET_TOLERANCE_MS,
        max_protected_gap_overflow_ms=(
            BRIDGE_MAX_PROTECTED_GAP_OVERFLOW_MS if is_bridge else DEFAULT_MAX_PROTECTED_GAP_OVERFLOW_MS
        ),
        min_protected_pause_ms=DEFAULT_MIN_PROTECTED_PAUSE_MS,
    )
    voice_info = {"selected_voice": str(voice or tts.default_voice), "base_prosody": {}}
    measured = _measure_temporal_candidate(
        tts=tts, group=group, voice_info=voice_info,
        candidate={"candidate_id": "measure", "spoken_text": text},
        geometry=geometry, output_root=Path(output_root), round_number=0, force=force,
    )
    window_ms = geometry["source_window_ms"]
    measured_ms = int(measured["measured_ms"])
    shortfall_ms = max(0, window_ms - measured_ms) if measured["direction"] == "expand" else 0
    return {
        "measured_ms": measured_ms,
        "window_ms": window_ms,
        "direction": measured["direction"],
        "fits_naturally": bool(measured["fit"]),
        "shortfall_ms": shortfall_ms,
        "shortfall_pct": round(shortfall_ms / window_ms * 100.0, 2) if window_ms else 0.0,
        "required_speed_percent": float(measured["required_speed_percent"]),
        "speed_fit_mode": measured["speed_fit_mode"],
        "rank": list(measured["rank"]),
        "free_gap_after_ms": int(geometry["free_gap_ms"]),
        "late_tolerance_ms": int(geometry["mouth_close_late_tolerance_ms"]),
        "audio_path": measured["path"],
        "voice": voice_info["selected_voice"],
    }
