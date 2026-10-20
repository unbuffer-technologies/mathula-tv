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

from .dub_padding import tail_pad_ms


TRANSIENT_ATTEMPTS = 5
_TRANSIENT_MARKERS = ("HTTP 429", "HTTP 500", "HTTP 502", "HTTP 503", "HTTP 504", "timed out", "Timeout", "Connection")


def _is_transient(error: Exception) -> bool:
    text = str(error)
    return any(marker in text for marker in _TRANSIENT_MARKERS)


def with_transient_retry(call: Any, *, attempts: int = TRANSIENT_ATTEMPTS, sleep: Any = None, jitter: Any = None) -> Any:
    """Run ``call()``, retrying Azure rate limits and server hiccups with exponential backoff and jitter.

    A 429 is the service saying "too many at once", which parallel runs cause routinely; failing the whole
    measurement batch for it wastes a turn and its tokens. Non-transient errors are raised at once, and the
    last transient error is raised after ``attempts`` tries -- the retry is bounded.
    """
    import random
    import time

    sleep = sleep or time.sleep
    jitter = jitter or random.random
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except Exception as error:  # noqa: BLE001 - classified below
            if attempt == attempts or not _is_transient(error):
                raise
            sleep(min(30.0, 2.0 ** attempt) * (0.5 + jitter()))


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
    measured = with_transient_retry(lambda: _measure_temporal_candidate(
        tts=tts, group=group, voice_info=voice_info,
        candidate={"candidate_id": "measure", "spoken_text": text},
        geometry=geometry, output_root=Path(output_root), round_number=0, force=force,
    ))
    window_ms = geometry["source_window_ms"]
    measured_ms = int(measured["measured_ms"])
    shortfall_ms = max(0, window_ms - measured_ms) if measured["direction"] == "expand" else 0
    free_gap_ms = int(geometry["free_gap_ms"])
    pad_ms = tail_pad_ms(shortfall_ms, free_gap_ms)
    return {
        "measured_ms": measured_ms,
        "window_ms": window_ms,
        "direction": measured["direction"],
        "fits_naturally": bool(measured["fit"]),
        "shortfall_ms": shortfall_ms,
        "shortfall_pct": round(shortfall_ms / window_ms * 100.0, 2) if window_ms else 0.0,
        # The renderer closes a small shortfall with real trailing silence (see dub_padding).
        "tail_pad_ms": pad_ms,
        "residual_shortfall_ms": shortfall_ms - pad_ms,
        "residual_shortfall_pct": round((shortfall_ms - pad_ms) / window_ms * 100.0, 2) if window_ms else 0.0,
        "required_speed_percent": float(measured["required_speed_percent"]),
        "speed_fit_mode": measured["speed_fit_mode"],
        "rank": list(measured["rank"]),
        "free_gap_after_ms": free_gap_ms,
        "late_tolerance_ms": int(geometry["mouth_close_late_tolerance_ms"]),
        "audio_path": measured["path"],
        "voice": voice_info["selected_voice"],
    }
