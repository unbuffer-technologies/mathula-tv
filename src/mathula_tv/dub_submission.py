"""The approach-agnostic output contract for a dubbing agent, plus its scorer.

An agent may use any method it likes (candidate pools, single-shot, its own
search over Azure measurements) -- all it must hand back is a ``submission``:
an ordered list of dubbed units, each with the window it intends to occupy, the
Pass-1 segments it covers, and its final isiZulu text. The scorer re-measures
every unit itself through ``dub_fit.measure_text_fit`` (the same function the MCP
server exposes), so an agent cannot claim a fit it did not achieve.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .atomic_io import atomic_write_json, read_json
from .dub_fit import measure_text_fit
from .dub_voices import pick_voice_info
from .episode_scoring import _load_context, score_items

SUBMISSION_SCHEMA = "mathula-dub-submission-v1"
CLIP_END_SLACK_MS = 1500


def validate_submission(
    submission: Mapping[str, Any], segments: Sequence[Mapping[str, Any]],
) -> tuple[list[str], list[str]]:
    """Return (errors, warnings). A submission with errors cannot be scored."""
    errors: list[str] = []
    warnings: list[str] = []
    items = list(submission.get("items") or [])
    if not items:
        return ["submission has no items"], warnings
    segment_ids = {str(s["segment_id"]) for s in segments}
    clip_end = max((int(s.get("end_ms") or 0) for s in segments), default=0)

    seen_ids: set[str] = set()
    previous_end = 0
    covered: dict[str, int] = {}
    for index, item in enumerate(items):
        label = f"item[{index}] ({item.get('item_id')})"
        item_id = str(item.get("item_id") or "")
        if not item_id or item_id in seen_ids:
            errors.append(f"{label}: item_id missing or duplicated")
        seen_ids.add(item_id)
        try:
            start, end = int(item["start_ms"]), int(item["end_ms"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"{label}: start_ms/end_ms must be integers")
            continue
        if end <= start:
            errors.append(f"{label}: end_ms must be greater than start_ms")
        if start < previous_end:
            errors.append(f"{label}: overlaps or precedes the previous item (starts {start} < {previous_end})")
        if start < 0 or end > clip_end + CLIP_END_SLACK_MS:
            errors.append(f"{label}: window {start}-{end} lies outside the clip (0-{clip_end})")
        previous_end = max(previous_end, end)
        if not str(item.get("zulu_text") or "").strip():
            errors.append(f"{label}: zulu_text is empty")
        ids = [str(v) for v in item.get("source_segment_ids") or []]
        if not ids:
            errors.append(f"{label}: source_segment_ids is empty")
        for segment_id in ids:
            if segment_id not in segment_ids:
                errors.append(f"{label}: unknown source segment {segment_id!r}")
            covered[segment_id] = covered.get(segment_id, 0) + 1
        if not str(item.get("speaker_id") or "").strip():
            errors.append(f"{label}: speaker_id is missing")

    for segment_id in sorted(segment_ids - set(covered)):
        errors.append(f"source segment {segment_id!r} is not covered by any item (untranslated content)")
    shared = {sid for sid, count in covered.items() if count > 1}
    by_id = {str(s["segment_id"]): s for s in segments}
    excerpts: dict[str, list[str]] = {sid: [] for sid in shared}
    for index, item in enumerate(items):
        ids = [str(v) for v in item.get("source_segment_ids") or []]
        if not shared.intersection(ids):
            continue          # an unshared unit's English always comes from its segments; any source_text is ignored
        label = f"item[{index}] ({item.get('item_id')})"
        explicit = str(item.get("source_text") or "").strip()
        if not explicit:
            errors.append(f"{label}: shares a source segment with another item, so it must carry an explicit "
                          "source_text (the English it actually covers)")
            continue
        english = _norm_words(" ".join(_segment_english(by_id[sid]) for sid in ids if sid in by_id))
        if f" {_norm_words(explicit)} " not in f" {english} ":
            errors.append(f"{label}: source_text must be copied exactly from the English of its segments; it is "
                          "checked against them, so a paraphrase or shortened version is rejected")
            continue
        for sid in shared.intersection(ids):
            excerpts[sid].append(_norm_words(explicit))
    for sid, pieces in excerpts.items():
        if sid in by_id and pieces:
            carried = set(" ".join(pieces).split())
            missing = [w for w in _norm_words(_segment_english(by_id[sid])).split() if w not in carried]
            if missing:
                errors.append(f"units sharing segment {sid!r} must together carry all of its English; "
                              f"not carried by any unit's source_text: {' '.join(missing[:8])}")
    if shared:
        warnings.append(f"{len(shared)} source segment(s) are split across several items: {sorted(shared)}")
    return errors, warnings


def _norm_words(text: str) -> str:
    return re.sub(r"[\W_]+", " ", str(text or "").casefold()).strip()


def _segment_english(segment: Mapping[str, Any]) -> str:
    return str(segment.get("restored_text") or segment.get("source_text") or "")


def _source_text(item: Mapping[str, Any], segments_by_id: Mapping[str, Mapping[str, Any]],
                 shared: set[str] = frozenset()) -> str:
    """The English a unit is scored against: always read from its source segments by the harness.

    The one exception is a unit that shares a segment with another unit, which may carry the excerpt it covers
    (validate_submission has already checked that excerpt is copied from the segments). Everywhere else an
    agent-supplied source_text is ignored: every gate and the meaning judge measure against this text, so the
    agent must not be able to choose it.
    """
    ids = [str(v) for v in item.get("source_segment_ids") or []]
    explicit = str(item.get("source_text") or "").strip()
    if explicit and shared.intersection(ids):
        return explicit
    return " ".join(_segment_english(segments_by_id[sid]) for sid in ids if sid in segments_by_id).strip()


def score_submission_dir(
    native_dub_dir: Path,
    submission: Mapping[str, Any],
    *,
    tts: Any,
    measure_root: Path,
    labels: Sequence[Mapping[str, Any]] = (),
    usage: Mapping[str, Any] | None = None,
    speed_policy: str = "pipeline",
    segments: Sequence[Mapping[str, Any]] | None = None,
    next_start_after_ms: int | None = None,
    voices: Mapping[str, Mapping[str, Any]] | None = None,
    pronunciation_root: Path | None = None,
) -> dict[str, Any]:
    """Re-measure and score an agent submission against the frozen clip inputs.

    ``segments`` restricts coverage to a subset (one speaker turn handled by a worker) and
    ``next_start_after_ms`` is when the next speech after that subset starts, so the last
    unit's free gap is measured exactly as it would be inside the whole clip."""
    from .native_dub import _build_speaker_turns

    native = Path(native_dub_dir)
    pass1 = read_json(native / "pass1_restored_transcript.json")
    segments = list(segments) if segments is not None else list(pass1.get("segments") or [])
    errors, warnings = validate_submission(submission, segments)
    if errors:
        raise ValueError("invalid submission: " + "; ".join(errors[:8]))
    segments_by_id = {str(s["segment_id"]): s for s in segments}
    glossary, ledger = _load_context(native)

    raw_items = list(submission["items"])
    seen_once: set[str] = set()
    shared_ids: set[str] = set()
    for raw in raw_items:
        for sid in map(str, raw.get("source_segment_ids") or []):
            (shared_ids if sid in seen_once else seen_once).add(sid)
    turn_groups = [
        {"group_id": str(i["item_id"]), "speaker_id": str(i["speaker_id"]),
         "start_ms": int(i["start_ms"]), "source_end_ms": int(i["end_ms"])}
        for i in raw_items
    ]
    turn_by_item: dict[str, str] = {}
    for turn in _build_speaker_turns(turn_groups):
        for item_id in turn["member_group_ids"]:
            turn_by_item[str(item_id)] = str(turn["member_group_ids"][0])

    items = []
    for index, item in enumerate(raw_items):
        next_start = (
            int(raw_items[index + 1]["start_ms"]) if index + 1 < len(raw_items)
            else (int(next_start_after_ms) if next_start_after_ms is not None else None)
        )
        source_text = _source_text(item, segments_by_id, shared_ids)
        fit = measure_text_fit(
            tts, text=str(item["zulu_text"]), start_ms=int(item["start_ms"]), end_ms=int(item["end_ms"]),
            next_start_ms=next_start, speaker_id=str(item["speaker_id"]), source_text=source_text,
            output_root=measure_root,
            voice_info=pick_voice_info(voices or {}, str(item["speaker_id"]), int(item["start_ms"]),
                                       int(item["end_ms"]), segments),
            pronunciation_root=pronunciation_root,
        )
        items.append({
            "group_id": str(item["item_id"]),
            "turn_id": item.get("turn_id") or turn_by_item.get(str(item["item_id"])),
            "speaker_id": str(item["speaker_id"]),
            "start_ms": int(item["start_ms"]),
            "end_ms": int(item["end_ms"]),
            "source_text": source_text,
            "spoken_text": str(item["zulu_text"]),
            "measured_ms": fit["measured_ms"],
            "direction": fit["direction"],
            "required_speed_percent": fit["required_speed_percent"],
            "fits_naturally": fit["fits_naturally"],
            "free_gap_ms": fit["free_gap_after_ms"],
            "tail_pad_ms": fit["tail_pad_ms"],
            "variant_id": "agent",
            "candidate_id": str(item["item_id"]),
            "requires_review": bool(item.get("needs_review")),
        })
    card = score_items(
        items, glossary=glossary, ledger=ledger, labels=labels, usage=usage,
        approach="agent_submission", speed_policy=speed_policy,
    )
    card["submission"] = {
        "approach_notes": str(submission.get("approach_notes") or ""),
        "warnings": warnings,
        "item_count": len(raw_items),
    }
    return card


def write_submission(path: Path, submission: Mapping[str, Any]) -> None:
    atomic_write_json(path, {"schema_version": SUBMISSION_SCHEMA, **dict(submission)})
