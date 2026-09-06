"""Deterministic semantic chunk planning for Mathula TV translation.

The planner keeps every source unit in timeline order, prefers natural speaker/
sentence boundaries, and supplies every chunk with a compact transcript-wide
context spine plus exact adjacent source context.  It has no provider or project
state dependencies, making the plan stable and independently testable.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

CHUNK_PLAN_SCHEMA_VERSION = "mathula.translation.semantic-chunk-plan.v1"
CONTEXT_SPINE_SCHEMA_VERSION = "mathula.translation.context-spine.v1"
SEMANTIC_BATCH_SCHEMA_VERSION = "mathula.translation.semantic-batch.v1"

_SENTENCE_END_RE = re.compile(r"(?:[.!?]|[.!?][\"'”’])\s*$")
_SPACE_RE = re.compile(r"\s+")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def normalise_space(value: Any) -> str:
    return _SPACE_RE.sub(" ", str(value or "")).strip()


def estimate_text_tokens(text: str) -> int:
    """Conservative tokenizer-free estimate suitable for request planning."""
    text = normalise_space(text)
    if not text:
        return 1
    # English and isiZulu JSON payloads usually land between 3 and 4 chars/token.
    # 3.25 deliberately overestimates so chunks stay below the hard ceiling.
    return max(1, math.ceil(len(text) / 3.25))


def estimate_unit_tokens(unit: Mapping[str, Any]) -> int:
    text = normalise_space(unit.get("source_text"))
    protected = " ".join(normalise_space(x) for x in unit.get("protected_spans") or [])
    required = " ".join(normalise_space(x) for x in unit.get("required_facts") or [])
    optional = " ".join(normalise_space(x) for x in unit.get("optional_details") or [])
    # Fixed allowance covers IDs, timing, duration targets and JSON keys.
    return 56 + estimate_text_tokens(" ".join((text, protected, required, optional)))


def adaptive_context_spine_tokens(
    request: Mapping[str, Any],
    *,
    minimum_tokens: int = 1600,
    maximum_tokens: int = 4500,
) -> int:
    """Scale complete-story context to transcript and identity complexity.

    A fixed 4,500-token prefix is useful for multi-hour hearings but wasteful
    for ordinary clips and interviews.  The result remains deterministic so
    checkpoint and prompt-cache identities stay stable.
    """

    if minimum_tokens < 1000:
        raise ValueError("minimum context spine budget must be at least 1000")
    if maximum_tokens < minimum_tokens:
        raise ValueError("maximum context spine budget must not be smaller than minimum")
    units = [
        item for item in request.get("units") or [] if isinstance(item, Mapping)
    ]
    protected = _dedupe_strings(
        [
            *(request.get("global_protected_spans") or []),
            *[
                span
                for unit in units
                for span in (unit.get("protected_spans") or [])
            ],
        ]
    )
    global_context_tokens = estimate_text_tokens(
        canonical_json(request.get("global_context") or {})
    )
    calculated = (
        1325
        + min(2550, len(units) * 16)
        + min(425, len(protected) * 18)
        + min(300, math.ceil(global_context_tokens * 0.12))
    )
    bounded = max(minimum_tokens, min(maximum_tokens, calculated))
    return int(math.ceil(bounded / 100.0) * 100)


def _boundary_after(units: Sequence[Mapping[str, Any]], index: int) -> bool:
    unit = units[index]
    text = normalise_space(unit.get("source_text"))
    if index >= len(units) - 1:
        return True
    next_unit = units[index + 1]
    speaker_change = normalise_space(unit.get("speaker_id")) != normalise_space(
        next_unit.get("speaker_id")
    )
    end_ms = int(unit.get("end_ms") or 0)
    next_start_ms = int(next_unit.get("start_ms") or end_ms)
    gap_ms = max(0, next_start_ms - end_ms)
    return bool(_SENTENCE_END_RE.search(text)) or (speaker_change and gap_ms >= 250) or gap_ms >= 900


@dataclass(frozen=True)
class ChunkRange:
    start: int
    stop: int
    estimated_tokens: int

    def to_dict(self, units: Sequence[Mapping[str, Any]], index: int, count: int) -> dict[str, Any]:
        selected = units[self.start : self.stop]
        return {
            "chunk_index": index,
            "chunk_count": count,
            "start_unit_index": self.start,
            "stop_unit_index_exclusive": self.stop,
            "unit_count": len(selected),
            "estimated_source_tokens": self.estimated_tokens,
            "requested_unit_ids": [normalise_space(item.get("unit_id")) for item in selected],
        }


def plan_semantic_ranges(
    units: Sequence[Mapping[str, Any]],
    *,
    target_tokens: int = 3200,
    hard_tokens: int = 4600,
    max_units: int | None = None,
) -> list[ChunkRange]:
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    if hard_tokens < target_tokens:
        raise ValueError("hard_tokens must be greater than or equal to target_tokens")
    if max_units is not None and max_units <= 0:
        raise ValueError("max_units must be positive when supplied")
    if not units:
        return []

    ranges: list[ChunkRange] = []
    start = 0
    minimum_break_tokens = max(1, int(target_tokens * 0.58))
    while start < len(units):
        total = 0
        index = start
        preferred_stop: int | None = None
        preferred_tokens = 0
        while index < len(units):
            unit_tokens = estimate_unit_tokens(units[index])
            next_total = total + unit_tokens
            unit_limit_hit = max_units is not None and index - start >= max_units
            hard_limit_hit = index > start and next_total > hard_tokens
            if unit_limit_hit or hard_limit_hit:
                break
            total = next_total
            index += 1
            if total >= minimum_break_tokens and _boundary_after(units, index - 1):
                preferred_stop = index
                preferred_tokens = total
            if total >= target_tokens:
                if preferred_stop is not None:
                    index = preferred_stop
                    total = preferred_tokens
                break

        if index == start:
            # A single unusually large unit is never split in the middle.
            index = start + 1
            total = estimate_unit_tokens(units[start])
        ranges.append(ChunkRange(start=start, stop=index, estimated_tokens=total))
        start = index
    return ranges


def _excerpt(text: str, limit: int) -> tuple[str, bool]:
    text = normalise_space(text)
    if len(text) <= limit:
        return text, False
    if limit < 48:
        return text[:limit].rstrip(), True
    tail = max(24, limit // 4)
    head = max(24, limit - tail - 3)
    return f"{text[:head].rstrip()} … {text[-tail:].lstrip()}", True


def _compact_global_context(value: Any, *, character_budget: int) -> tuple[Any, list[str]]:
    """Keep the most useful deterministic global context within a hard budget."""
    if not isinstance(value, Mapping):
        text, truncated = _excerpt(canonical_json(value), max(80, character_budget))
        return text, (["<root>"] if truncated else [])
    result: dict[str, Any] = {}
    omitted: list[str] = []
    # Preserve insertion order because upstream context builders already rank
    # their fields.  Fall back to a stable string representation per value.
    remaining = max(120, character_budget)
    for raw_key, raw_value in value.items():
        key = str(raw_key)
        encoded = canonical_json(raw_value)
        allowance = max(80, min(len(encoded), remaining - 32))
        if allowance <= 80 and result:
            omitted.append(key)
            continue
        if len(encoded) <= allowance:
            candidate: Any = raw_value
        else:
            candidate, _ = _excerpt(encoded, allowance)
            omitted.append(key)
        cost = len(key) + len(canonical_json(candidate)) + 8
        if cost > remaining and result:
            omitted.append(key)
            continue
        result[key] = candidate
        remaining -= min(cost, remaining)
        if remaining < 96:
            omitted.extend(str(item) for item in list(value.keys())[len(result) :])
            break
    return result, list(dict.fromkeys(omitted))


def _dedupe_strings(values: Sequence[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = normalise_space(value)
        key = text.casefold()
        if not text or key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def build_context_spine(
    request: Mapping[str, Any],
    *,
    token_budget: int = 4500,
) -> dict[str, Any]:
    """Build a bounded, complete-coverage story spine for every chunk.

    A long video cannot repeat every raw word in every request without recreating
    the same uncached-input pressure that chunking is meant to solve.  Instead,
    every source unit is covered by one contiguous timeline section.  Each
    section carries its time range, speakers, protected identities and a
    deterministic head/tail excerpt.  The active chunk and adjacent units are
    still supplied verbatim in the dynamic request.
    """
    units = list(request.get("units") or [])
    if token_budget < 1000:
        raise ValueError("context spine token budget must be at least 1000")

    total_chars = max(4200, int(token_budget * 3.05))
    global_budget = max(800, min(3200, int(total_chars * 0.24)))
    compact_context, omitted_context_keys = _compact_global_context(
        request.get("global_context") or {},
        character_budget=global_budget,
    )

    protected_all = _dedupe_strings(request.get("global_protected_spans") or [])
    protected_budget = max(500, min(2200, int(total_chars * 0.16)))
    protected_kept: list[str] = []
    protected_chars = 0
    for value in protected_all:
        cost = len(value) + 4
        if protected_kept and protected_chars + cost > protected_budget:
            break
        protected_kept.append(value)
        protected_chars += cost

    timeline_budget = max(
        2200,
        total_chars
        - len(canonical_json(compact_context))
        - len(canonical_json(protected_kept))
        - 1500,
    )
    # Each section has a fixed JSON overhead.  This cap keeps the prefix stable
    # for both short clips and hearings with thousands of source units.
    max_sections = max(4, min(len(units) or 1, timeline_budget // 360))
    units_per_section = max(1, math.ceil(len(units) / max_sections))
    ranges = [
        (start, min(len(units), start + units_per_section))
        for start in range(0, len(units), units_per_section)
    ]
    excerpt_chars = max(120, min(420, timeline_budget // max(1, len(ranges)) - 180))

    sections: list[dict[str, Any]] = []
    for section_index, (start, stop) in enumerate(ranges, start=1):
        selected = units[start:stop]
        source_joined = " ".join(normalise_space(item.get("source_text")) for item in selected)
        excerpt, truncated = _excerpt(source_joined, excerpt_chars)
        speakers = _dedupe_strings(item.get("speaker_id") for item in selected)
        protected = _dedupe_strings(
            span for item in selected for span in (item.get("protected_spans") or [])
        )
        sections.append(
            {
                "section_index": section_index,
                "first_unit_id": normalise_space(selected[0].get("unit_id")),
                "last_unit_id": normalise_space(selected[-1].get("unit_id")),
                "start_ms": int(selected[0].get("start_ms") or 0),
                "end_ms": int(selected[-1].get("end_ms") or 0),
                "covered_unit_count": len(selected),
                "speakers": speakers[:8],
                "protected_spans": protected[:12],
                "source_excerpt": excerpt,
                "excerpt_truncated": truncated,
            }
        )

    spine = {
        "schema_version": CONTEXT_SPINE_SCHEMA_VERSION,
        "purpose": "complete_story_context_not_translation_targets",
        "complete_timeline_coverage": True,
        "source_unit_count": len(units),
        "covered_unit_count": sum(item["covered_unit_count"] for item in sections),
        "source_request_sha256": normalise_space(request.get("request_sha256")),
        "source_transcript_file_sha256": normalise_space(
            request.get("source_transcript_file_sha256")
        ),
        "source_timeline_file_sha256": normalise_space(
            request.get("source_timeline_file_sha256")
        ),
        "global_context": compact_context,
        "global_context_omitted_or_truncated_keys": omitted_context_keys,
        "global_protected_spans": protected_kept,
        "global_protected_span_count": len(protected_all),
        "global_protected_spans_truncated": len(protected_kept) < len(protected_all),
        "timeline_sections": sections,
        "excerpt_policy": {
            "section_count": len(sections),
            "units_per_section_ceiling": units_per_section,
            "section_excerpt_character_limit": excerpt_chars,
            "exact_active_and_adjacent_text_supplied_in_dynamic_request": True,
        },
    }
    if spine["covered_unit_count"] != len(units):
        raise AssertionError("context spine did not cover every source unit")
    spine["context_spine_sha256"] = sha256_json(spine)
    return spine


def _context_unit(unit: Mapping[str, Any], *, translate: bool) -> dict[str, Any]:
    return {
        "unit_id": normalise_space(unit.get("unit_id")),
        "speaker_id": normalise_space(unit.get("speaker_id")),
        "start_ms": int(unit.get("start_ms") or 0),
        "end_ms": int(unit.get("end_ms") or 0),
        "source_text": normalise_space(unit.get("source_text")),
        "protected_spans": list(unit.get("protected_spans") or []),
        "translate_this_unit": translate,
    }


def build_single_handoff_batch_request(
    request: Mapping[str, Any],
    *,
    context_spine_tokens: int = 4500,
    max_output_tokens: int = 100000,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build one compact translation handoff containing every source unit.

    This is intentionally separate from normal semantic chunk planning. It is
    used by the explicit manual-chatgpt provider so the operator can upload one
    request and install one response for the complete translation pass. The
    active units are already present at the top level, so context_units is kept
    empty to avoid duplicating the full transcript in the handoff.
    """
    source_units = list(request.get("units") or [])
    if not source_units:
        raise ValueError("translation request contains no units")
    if context_spine_tokens < 1000:
        raise ValueError("context_spine_tokens must be at least 1000")
    if max_output_tokens <= 0:
        raise ValueError("max_output_tokens must be positive")

    estimated_source = sum(estimate_unit_tokens(item) for item in source_units)
    unit_count = len(source_units)
    estimated_output = 3500 + int(estimated_source * 5.0) + unit_count * 180
    output_limit = min(max_output_tokens, max(12000, estimated_output))
    spine = build_context_spine(request, token_budget=context_spine_tokens)
    chunk_range = ChunkRange(
        start=0,
        stop=unit_count,
        estimated_tokens=estimated_source,
    )

    dynamic = {
        key: value
        for key, value in request.items()
        if key
        not in {
            "units",
            "request_sha256",
            "translation_batch",
            "global_context",
            "global_protected_spans",
        }
    }
    dynamic["units"] = [dict(item) for item in source_units]
    dynamic["translation_batch"] = {
        "schema_version": SEMANTIC_BATCH_SCHEMA_VERSION,
        **chunk_range.to_dict(source_units, 1, 1),
        "full_request_sha256": normalise_space(request.get("request_sha256")),
        "context_spine_sha256": spine["context_spine_sha256"],
        "context_radius_units": 0,
        "context_units": [],
        "prior_translated_context": [],
        "prior_global_terminology": [],
        "max_output_tokens": output_limit,
        "manual_single_handoff": True,
        "instructions": [
            "Translate every top-level unit and return each unit exactly once.",
            "This manual handoff contains the complete active translation batch.",
            "Use the cached context spine only for global context and terminology.",
            "Do not return context-only units.",
        ],
    }
    dynamic["request_sha256"] = sha256_json(dynamic)

    chunk_dict = chunk_range.to_dict(source_units, 1, 1)
    plan = {
        "schema_version": CHUNK_PLAN_SCHEMA_VERSION,
        "full_request_sha256": normalise_space(request.get("request_sha256")),
        "mode": "manual_single_handoff",
        "target_tokens": estimated_source,
        "hard_tokens": estimated_source,
        "context_units": 0,
        "context_spine_tokens": context_spine_tokens,
        "max_units": unit_count,
        "chunk_count": 1,
        "chunks": [chunk_dict],
        "context_spine": spine,
    }
    plan["plan_sha256"] = sha256_json(plan)
    return [dynamic], plan


def build_semantic_batch_requests(
    request: Mapping[str, Any],
    *,
    target_tokens: int = 3200,
    hard_tokens: int = 4600,
    context_units: int = 3,
    context_spine_tokens: int = 4500,
    max_units: int | None = None,
    max_output_tokens: int = 24000,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if context_units < 0:
        raise ValueError("context_units must not be negative")
    source_units = list(request.get("units") or [])
    if not source_units:
        raise ValueError("translation request contains no units")
    ranges = plan_semantic_ranges(
        source_units,
        target_tokens=target_tokens,
        hard_tokens=hard_tokens,
        max_units=max_units,
    )
    spine = build_context_spine(request, token_budget=context_spine_tokens)
    batches: list[dict[str, Any]] = []
    count = len(ranges)
    for chunk_index, chunk_range in enumerate(ranges, start=1):
        start, stop = chunk_range.start, chunk_range.stop
        dynamic = {
            key: value
            for key, value in request.items()
            if key not in {
                "units",
                "request_sha256",
                "translation_batch",
                "global_context",
                "global_protected_spans",
            }
        }
        dynamic["units"] = [dict(item) for item in source_units[start:stop]]
        before_start = max(0, start - context_units)
        after_stop = min(len(source_units), stop + context_units)
        local_context = [
            _context_unit(item, translate=start <= absolute_index < stop)
            for absolute_index, item in enumerate(
                source_units[before_start:after_stop], start=before_start
            )
        ]
        estimated_source = chunk_range.estimated_tokens
        unit_count = chunk_range.stop - chunk_range.start
        # Multivariant translation returns natural, concise and compact text plus
        # per-unit JSON metadata.  Source-token-only budgeting badly undercounts
        # that expansion on long clips, so reserve both a per-token and per-unit
        # allowance.  The coordinator also caps normal chunks at 18 units.
        estimated_output = 3500 + int(estimated_source * 5.0) + unit_count * 180
        output_limit = min(
            max_output_tokens,
            max(12000, estimated_output),
        )
        dynamic["translation_batch"] = {
            "schema_version": SEMANTIC_BATCH_SCHEMA_VERSION,
            **chunk_range.to_dict(source_units, chunk_index, count),
            "full_request_sha256": normalise_space(request.get("request_sha256")),
            "context_spine_sha256": spine["context_spine_sha256"],
            "context_radius_units": context_units,
            "context_units": local_context,
            "prior_translated_context": [],
            "prior_global_terminology": [],
            "max_output_tokens": output_limit,
            "instructions": [
                "Translate only top-level units and return each requested unit exactly once.",
                "Use the cached context spine and local context only to resolve references and maintain continuity.",
                "Do not return context-only units.",
            ],
        }
        dynamic["request_sha256"] = sha256_json(dynamic)
        batches.append(dynamic)
    plan = {
        "schema_version": CHUNK_PLAN_SCHEMA_VERSION,
        "full_request_sha256": normalise_space(request.get("request_sha256")),
        "target_tokens": target_tokens,
        "hard_tokens": hard_tokens,
        "context_units": context_units,
        "context_spine_tokens": context_spine_tokens,
        "max_units": max_units,
        "chunk_count": count,
        "chunks": [item.to_dict(source_units, i, count) for i, item in enumerate(ranges, 1)],
        "context_spine": spine,
    }
    plan["plan_sha256"] = sha256_json(plan)
    return batches, plan


__all__ = [
    "CHUNK_PLAN_SCHEMA_VERSION",
    "CONTEXT_SPINE_SCHEMA_VERSION",
    "SEMANTIC_BATCH_SCHEMA_VERSION",
    "adaptive_context_spine_tokens",
    "build_context_spine",
    "build_semantic_batch_requests",
    "build_single_handoff_batch_request",
    "estimate_text_tokens",
    "estimate_unit_tokens",
    "plan_semantic_ranges",
    "sha256_json",
]
