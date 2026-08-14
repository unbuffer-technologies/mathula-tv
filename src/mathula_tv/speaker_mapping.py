"""Canonical speaker-ID mapping for voice-family evidence.

Raw diarization providers use IDs such as ``AZURE_1`` while translated speech
uses canonical IDs such as ``SPEAKER_00``.  This module resolves that boundary
from explicit transcript metadata first and timeline overlap second.  It does
not infer voice family and it never converts missing evidence into a masculine
fallback.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping


SPEAKER_ID_MAPPING_SCHEMA_VERSION = "mathula-speaker-id-mapping-v1"


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result


def _bounds(item: Mapping[str, Any]) -> tuple[float, float] | None:
    """Return timeline bounds in seconds for common transcript schemas."""
    start = _number(item.get("start"))
    end = _number(item.get("end"))
    if start is not None and end is not None:
        return start, end

    start_ms = _number(item.get("start_ms"))
    end_ms = _number(
        item.get("end_ms", item.get("preferred_end_ms", item.get("hard_end_ms")))
    )
    if start_ms is not None and end_ms is not None:
        return start_ms / 1000.0, end_ms / 1000.0
    return None


def _canonical_id(item: Mapping[str, Any]) -> str | None:
    value = item.get("speaker_id", item.get("speaker"))
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _raw_id(item: Mapping[str, Any]) -> str | None:
    for key in (
        "raw_speaker",
        "raw_speaker_id",
        "diarization_speaker",
        "source_speaker_id",
    ):
        value = item.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _overlap(left: tuple[float, float], right: tuple[float, float]) -> float:
    return max(0.0, min(left[1], right[1]) - max(left[0], right[0]))


def build_speaker_id_mapping(
    transcript: Mapping[str, Any],
    diarization: Mapping[str, Any],
) -> dict[str, Any]:
    """Map raw diarization speakers to canonical transcript speakers.

    Explicit ``raw_speaker`` metadata in transcript segments is authoritative.
    When older transcripts lack that field, overlap between canonical transcript
    segments and raw diarization turns supplies the mapping.
    """
    transcript_segments = [
        item
        for item in transcript.get("segments", [])
        if isinstance(item, Mapping)
        and _canonical_id(item)
        and _bounds(item) is not None
    ]
    diarization_turns = [
        item
        for item in diarization.get("turns", [])
        if isinstance(item, Mapping)
        and item.get("speaker") is not None
        and _bounds(item) is not None
    ]

    direct_scores: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    overlap_scores: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    raw_duration: dict[str, float] = defaultdict(float)

    for segment in transcript_segments:
        canonical = _canonical_id(segment)
        raw = _raw_id(segment)
        bounds = _bounds(segment)
        if canonical and raw and bounds:
            direct_scores[raw][canonical] += max(0.0, bounds[1] - bounds[0])

    for turn in diarization_turns:
        raw = str(turn["speaker"]).strip()
        turn_bounds = _bounds(turn)
        if not raw or turn_bounds is None:
            continue
        raw_duration[raw] += max(0.0, turn_bounds[1] - turn_bounds[0])
        for segment in transcript_segments:
            canonical = _canonical_id(segment)
            segment_bounds = _bounds(segment)
            if canonical and segment_bounds:
                amount = _overlap(turn_bounds, segment_bounds)
                if amount > 0:
                    overlap_scores[raw][canonical] += amount

    raw_ids = set(raw_duration) | set(direct_scores) | set(overlap_scores)
    raw_to_canonical: dict[str, str] = {}
    mappings: list[dict[str, Any]] = []

    for raw in sorted(raw_ids):
        source = "unresolved"
        scores: Mapping[str, float] = {}
        if direct_scores.get(raw):
            source = "transcript_raw_speaker"
            scores = direct_scores[raw]
        elif overlap_scores.get(raw):
            source = "timeline_overlap"
            scores = overlap_scores[raw]

        if not scores:
            mappings.append(
                {
                    "raw_speaker_id": raw,
                    "canonical_speaker_id": None,
                    "source": source,
                    "confidence": 0.0,
                    "overlap_seconds": 0.0,
                }
            )
            continue

        canonical, score = max(scores.items(), key=lambda item: (item[1], item[0]))
        total = sum(max(0.0, value) for value in scores.values())
        confidence = score / total if total > 0 else 0.0
        raw_to_canonical[raw] = canonical
        mappings.append(
            {
                "raw_speaker_id": raw,
                "canonical_speaker_id": canonical,
                "source": source,
                "confidence": round(confidence, 6),
                "overlap_seconds": round(overlap_scores.get(raw, {}).get(canonical, 0.0), 6),
                "raw_duration_seconds": round(raw_duration.get(raw, 0.0), 6),
                "candidate_scores_seconds": {
                    key: round(value, 6)
                    for key, value in sorted(scores.items())
                },
            }
        )

    canonical_to_raw: dict[str, list[str]] = defaultdict(list)
    for raw, canonical in raw_to_canonical.items():
        canonical_to_raw[canonical].append(raw)
    # A contextual handoff repair may deliberately split one faulty raw Azure
    # label across several canonical speakers.  Preserve those audited direct
    # links even though raw_to_canonical must remain a single dominant mapping
    # for legacy raw acoustic artifacts.
    for raw, candidates in direct_scores.items():
        for canonical, duration in candidates.items():
            if duration > 0 and raw not in canonical_to_raw[canonical]:
                canonical_to_raw[canonical].append(raw)

    return {
        "schema_version": SPEAKER_ID_MAPPING_SCHEMA_VERSION,
        "raw_to_canonical": raw_to_canonical,
        "canonical_to_raw": {
            canonical: sorted(raw_ids)
            for canonical, raw_ids in sorted(canonical_to_raw.items())
        },
        "mappings": mappings,
        "unmapped_raw_speakers": sorted(raw_ids - set(raw_to_canonical)),
        "transcript_speaker_ids": sorted(
            {
                canonical
                for item in transcript_segments
                if (canonical := _canonical_id(item)) is not None
            }
        ),
    }


def _iter_speaker_records(value: Mapping[str, Any]) -> Iterable[tuple[str, Mapping[str, Any]]]:
    raw: Any = value.get("speakers", value.get("results", {}))
    if isinstance(raw, Mapping):
        for key, item in raw.items():
            if isinstance(item, Mapping):
                yield str(key), item
    elif isinstance(raw, list):
        for item in raw:
            if not isinstance(item, Mapping):
                continue
            key = item.get("speaker_id", item.get("speaker"))
            if key is not None:
                yield str(key), item


def canonicalize_voice_family_evidence(
    acoustic_analysis: Mapping[str, Any],
    mapping: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Return strongest acoustic/classifier evidence under canonical IDs."""
    raw_to_canonical = {
        str(key): str(value)
        for key, value in mapping.get("raw_to_canonical", {}).items()
    }
    resolved: dict[str, dict[str, Any]] = {}

    for record_key, item in _iter_speaker_records(acoustic_analysis):
        declared = str(item.get("speaker_id") or record_key).strip()
        raw = str(item.get("raw_speaker_id") or "").strip() or None

        if declared.startswith("SPEAKER_"):
            canonical = declared
        elif raw and raw in raw_to_canonical:
            canonical = raw_to_canonical[raw]
        else:
            canonical = raw_to_canonical.get(declared)

        if not canonical:
            continue

        family = str(item.get("voice_family") or "unknown").strip().lower()
        if family not in {"masculine", "feminine", "unknown"}:
            family = "unknown"
        try:
            confidence = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = min(1.0, max(0.0, confidence))

        evidence = dict(item)
        evidence.update(
            {
                "speaker_id": canonical,
                "source_speaker_id": declared,
                "raw_speaker_id": raw or (declared if not declared.startswith("SPEAKER_") else None),
                "voice_family": family,
                "confidence": confidence,
            }
        )

        previous = resolved.get(canonical)
        previous_rank = _evidence_rank(previous) if previous else (-1, -1.0)
        current_rank = _evidence_rank(evidence)
        if current_rank > previous_rank:
            resolved[canonical] = evidence

    return resolved


def _evidence_rank(item: Mapping[str, Any]) -> tuple[int, float]:
    family = str(item.get("voice_family") or "unknown")
    known = 1 if family in {"masculine", "feminine"} else 0
    try:
        confidence = float(item.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    method = str((item.get("evidence") or {}).get("method") or item.get("method") or "")
    method_bonus = 1 if "ecapa" in method.lower() else 0
    return known + method_bonus, confidence


__all__ = [
    "SPEAKER_ID_MAPPING_SCHEMA_VERSION",
    "build_speaker_id_mapping",
    "canonicalize_voice_family_evidence",
]
