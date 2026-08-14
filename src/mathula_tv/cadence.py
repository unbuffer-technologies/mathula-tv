"""Phrase-level timing plans and perceptual cadence quality gates.

Approved wording is never changed here. The cadence plan divides that wording
into delivery units and allocates timeline windows for synthesis/rendering
evidence. Azure may still synthesize the parent same-speaker group once.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


CADENCE_PLAN_SCHEMA_VERSION = "mathula-cadence-plan-v1"
CADENCE_QC_SCHEMA_VERSION = "mathula-cadence-qc-v1"
_BOUNDARY = re.compile(r"(?<=[.!?…])\s+|(?<=[,;:])\s+(?=\S)")


@dataclass(frozen=True)
class CadencePhrase:
    phrase_id: str
    text: str
    source_start: int
    source_end: int
    start_ms: int
    end_ms: int
    boundary_after: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CadencePlan:
    block_id: str
    immutable_text: str
    window_start_ms: int
    window_end_ms: int
    phrases: tuple[CadencePhrase, ...]
    schema_version: str = CADENCE_PLAN_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "block_id": self.block_id,
            "immutable_text": self.immutable_text,
            "window_start_ms": self.window_start_ms,
            "window_end_ms": self.window_end_ms,
            "phrase_count": len(self.phrases),
            "phrases": [phrase.to_dict() for phrase in self.phrases],
        }


def _delivery_weight(text: str) -> int:
    words = re.findall(r"\w+(?:[-’']\w+)*", text, re.UNICODE)
    punctuation_pause = sum(text.count(mark) for mark in ",;:")
    terminal_pause = sum(text.count(mark) for mark in ".!?…")
    return max(1, len(words) * 10 + punctuation_pause * 2 + terminal_pause * 4)


def plan_phrase_cadence(
    *,
    block_id: str,
    text: str,
    start_ms: int,
    end_ms: int,
) -> CadencePlan:
    """Split immutable text at natural boundaries and allocate exact windows."""

    value = str(text).strip()
    if not value:
        raise ValueError("Cadence planning requires non-empty immutable text")
    if end_ms <= start_ms:
        raise ValueError("Cadence planning requires a positive timeline window")

    spans: list[tuple[int, int, str]] = []
    cursor = 0
    for match in _BOUNDARY.finditer(value):
        end = match.start()
        phrase = value[cursor:end].strip()
        if phrase:
            actual_start = value.find(phrase, cursor, end + 1)
            spans.append((actual_start, actual_start + len(phrase), phrase))
        cursor = match.end()
    phrase = value[cursor:].strip()
    if phrase:
        actual_start = value.find(phrase, cursor)
        spans.append((actual_start, actual_start + len(phrase), phrase))
    if not spans:
        spans = [(0, len(value), value)]

    weights = [_delivery_weight(phrase_text) for _, _, phrase_text in spans]
    total_weight = sum(weights)
    duration_ms = end_ms - start_ms
    boundaries = [start_ms]
    cumulative = 0
    for weight in weights[:-1]:
        cumulative += weight
        boundaries.append(start_ms + round(duration_ms * cumulative / total_weight))
    boundaries.append(end_ms)

    phrases = []
    for index, ((source_start, source_end, phrase_text), phrase_start, phrase_end) in enumerate(
        zip(spans, boundaries, boundaries[1:]), start=1
    ):
        boundary_after = (
            "terminal"
            if phrase_text.endswith((".", "!", "?", "…"))
            else "clause"
            if phrase_text.endswith((",", ";", ":"))
            else "none"
        )
        phrases.append(
            CadencePhrase(
                phrase_id=f"{block_id}_phrase_{index:03d}",
                text=phrase_text,
                source_start=source_start,
                source_end=source_end,
                start_ms=phrase_start,
                end_ms=phrase_end,
                boundary_after=boundary_after,
            )
        )
    return CadencePlan(
        block_id=block_id,
        immutable_text=value,
        window_start_ms=start_ms,
        window_end_ms=end_ms,
        phrases=tuple(phrases),
    )


def assess_perceptual_cadence(
    *,
    selected_rate_percent: int,
    base_rate_percent: int,
    whole_voice_stretch_ratio: float,
    timeline_fill_action: str,
    pause_fill: Mapping[str, Any] | None = None,
    trailing_room_ms: int = 0,
    maximum_rate_delta_percent: int = 5,
    maximum_whole_voice_change: float = 1.03,
    maximum_added_pause_ms: int = 3000,
    maximum_unexplained_trailing_room_ms: int = 750,
) -> dict[str, Any]:
    """Return a deterministic gate for rushed, dragged, or lopsided delivery."""

    issues: list[dict[str, Any]] = []
    rate_delta = int(selected_rate_percent) - int(base_rate_percent)
    if abs(rate_delta) > maximum_rate_delta_percent:
        issues.append(
            {
                "code": "whole_turn_rate_delta_excessive",
                "value": rate_delta,
                "limit": maximum_rate_delta_percent,
            }
        )
    ratio = float(whole_voice_stretch_ratio)
    if ratio > maximum_whole_voice_change or ratio < 1 / maximum_whole_voice_change:
        issues.append(
            {
                "code": "whole_turn_time_stretch_excessive",
                "value": ratio,
                "limit": maximum_whole_voice_change,
            }
        )
    pause_fill = dict(pause_fill or {})
    maximum_pause = int(pause_fill.get("maximum_added_pause_ms") or 0)
    if maximum_pause > maximum_added_pause_ms:
        issues.append(
            {
                "code": "inserted_pause_excessive",
                "value": maximum_pause,
                "limit": maximum_added_pause_ms,
            }
        )
    trailing_room = max(0, int(trailing_room_ms))
    if trailing_room > maximum_unexplained_trailing_room_ms:
        issues.append(
            {
                "code": "unexplained_trailing_room_excessive",
                "value": trailing_room,
                "limit": maximum_unexplained_trailing_room_ms,
                "required_action": "align target breath groups to source phrase timestamps",
            }
        )
    return {
        "schema_version": CADENCE_QC_SCHEMA_VERSION,
        "passed": not issues,
        "review_required": bool(issues),
        "selected_rate_percent": int(selected_rate_percent),
        "base_rate_percent": int(base_rate_percent),
        "rate_delta_percent": rate_delta,
        "whole_voice_stretch_ratio": ratio,
        "timeline_fill_action": str(timeline_fill_action),
        "pause_fill": pause_fill,
        "trailing_room_ms": trailing_room,
        "limits": {
            "maximum_rate_delta_percent": maximum_rate_delta_percent,
            "maximum_whole_voice_change": maximum_whole_voice_change,
            "maximum_added_pause_ms": maximum_added_pause_ms,
            "maximum_unexplained_trailing_room_ms": maximum_unexplained_trailing_room_ms,
        },
        "issues": issues,
    }


def aggregate_cadence_qc(blocks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failed = [
        str(block.get("block_id"))
        for block in blocks
        if not bool((block.get("cadence_qc") or {}).get("passed", False))
    ]
    return {
        "schema_version": CADENCE_QC_SCHEMA_VERSION,
        "passed": not failed,
        "review_required": bool(failed),
        "failed_block_ids": failed,
        "block_count": len(blocks),
    }


__all__ = [
    "CADENCE_PLAN_SCHEMA_VERSION",
    "CADENCE_QC_SCHEMA_VERSION",
    "CadencePhrase",
    "CadencePlan",
    "aggregate_cadence_qc",
    "assess_perceptual_cadence",
    "plan_phrase_cadence",
]
