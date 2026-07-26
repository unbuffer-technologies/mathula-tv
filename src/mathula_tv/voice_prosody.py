"""Safe composition of stable speaker prosody and Azure timing adjustments."""

from __future__ import annotations

from typing import Any, Mapping


def bounded_prosody_int(value: Any, *, default: int, lower: int, upper: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(lower, min(upper, parsed))


def voice_and_base_prosody(
    assignment: Mapping[str, Any] | None,
    default_voice: str,
) -> tuple[str, int, int, int]:
    """Return one anchor voice and safe stable base prosody for a speaker."""
    if not assignment:
        return default_voice, 0, 0, 0
    prosody = assignment.get("base_prosody", {})
    if not isinstance(prosody, Mapping):
        prosody = {}
    return (
        str(assignment.get("selected_voice") or default_voice),
        bounded_prosody_int(prosody.get("rate_percent"), default=0, lower=-8, upper=8),
        bounded_prosody_int(prosody.get("pitch_percent"), default=0, lower=-12, upper=12),
        bounded_prosody_int(prosody.get("volume_percent"), default=0, lower=-3, upper=3),
    )


def combined_azure_rate(
    base_rate_percent: int,
    timing_delta_percent: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    """Apply timing search around, rather than instead of, speaker identity rate."""
    return max(minimum, min(maximum, base_rate_percent + timing_delta_percent))


__all__ = [
    "bounded_prosody_int",
    "voice_and_base_prosody",
    "combined_azure_rate",
]
