"""How the renderer pads silence, as pure arithmetic.

When a unit's (or a speaker turn's) speech ends before its source window does, the
real renderer does not always leave it as dead air. It has two silence mechanisms,
tried in this order, and a measurement tool that ignores them misreports timing:

  1. TAIL PAD    a residual shortfall of at most ``MAX_UNRESOLVED_EARLY_SILENCE_PAD_MS``
                 that also fits in the free gap before the next speech is closed with
                 real trailing silence (native_dub: ``_pad_wav_with_trailing_silence``).
  2. GAP WIDEN   (Phase 30, multi-sentence turns only) a larger shortfall is spread over
                 the turn's real inter-sentence gaps, in proportion to each gap's real
                 length, each capped at ``_TURN_SILENCE_GAP_CAP_MS``
                 (native_dub: ``_widen_turn_member_gaps``). It can under-shoot, never
                 over-shoot; with no real gap in the turn it closes nothing.

Anything left is dead air: natural speed is kept and the early close is accepted. The
constants are imported from the renderer so they cannot drift; the arithmetic below is
the same as ``_widen_turn_member_gaps`` minus the WAV handling.

Silence is the fallback, not the goal: a native speaker hears padding as a pause. The
numbers here let an agent see how much of a shortfall silence will absorb, so it can put
words where silence would otherwise sit.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence


def _constants() -> tuple[int, int]:
    from .native_dub import MAX_UNRESOLVED_EARLY_SILENCE_PAD_MS, _TURN_SILENCE_GAP_CAP_MS

    return int(MAX_UNRESOLVED_EARLY_SILENCE_PAD_MS), int(_TURN_SILENCE_GAP_CAP_MS)


def tail_pad_ms(shortfall_ms: int, free_gap_ms: int) -> int:
    """Trailing silence the renderer adds to close ``shortfall_ms`` (0 if it would not)."""
    max_pad, _ = _constants()
    shortfall = max(0, int(shortfall_ms))
    if shortfall and shortfall <= max_pad and shortfall <= max(0, int(free_gap_ms)):
        return shortfall
    return 0


def plan_turn_padding(
    members: Sequence[Mapping[str, Any]], *, shortfall_ms: int, last_free_gap_ms: int,
) -> dict[str, Any]:
    """Silence the renderer would add to a turn that finishes ``shortfall_ms`` early.

    ``members`` are the turn's units in order, each with ``start_ms`` and ``end_ms`` (their
    source windows); the real gaps between consecutive members are the widenable room.
    """
    max_pad, gap_cap = _constants()
    shortfall = max(0, int(shortfall_ms))
    result: dict[str, Any] = {
        "shortfall_ms": shortfall, "mechanism": "none", "padded_ms": 0, "widened_gaps_ms": [],
        "real_gaps_ms": [], "residual_ms": shortfall, "tail_pad_limit_ms": max_pad, "gap_cap_ms": gap_cap,
    }
    if not shortfall:
        result["mechanism"] = "not_needed"
        return result
    pad = tail_pad_ms(shortfall, last_free_gap_ms)
    if pad:
        result.update(mechanism="tail_pad", padded_ms=pad, residual_ms=shortfall - pad)
        return result
    real_gaps = [
        max(0, int(members[i + 1]["start_ms"]) - int(members[i]["end_ms"])) for i in range(len(members) - 1)
    ]
    result["real_gaps_ms"] = real_gaps
    total = sum(real_gaps)
    if total <= 0:
        return result
    scale = shortfall / total
    widened = [min(gap_cap, gap * scale) for gap in real_gaps]
    rounded = [int(round(gap)) for gap in widened]
    padded = min(shortfall, sum(rounded))
    result.update(mechanism="gap_widening", padded_ms=padded, widened_gaps_ms=rounded, residual_ms=shortfall - padded)
    return result
