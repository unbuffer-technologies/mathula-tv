"""Transcript-first timing evidence for pause-aware, overlapping dialogue.

The planner deliberately consumes Azure word timestamps that Mathula already
produces.  It does not inspect pixels, decode audio, run VAD, import a machine
learning runtime, or require a GPU.  Blocks remain the stable semantic and CLI
contract; speech islands and interaction events are additive timing evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


TRANSCRIPT_LIPSYNC_SCHEMA_VERSION = (
    "mathula-transcript-lipsync-v5-breath-group-islands-v13.19.2"
)
TRANSCRIPT_LIPSYNC_POLICY_VERSION = (
    "azure-word-breath-groups-and-interactions-v5-v13.19.2"
)

MICRO_PAUSE_MIN_MS = 150
PHRASE_PAUSE_MIN_MS = 350
ISLAND_GAP_MIN_MS = 600
INTERACTION_JOIN_GAP_MS = 150
MAX_INTERJECTION_DURATION_MS = 2000
MAX_INTERJECTION_HANDOFF_GAP_MS = 250


@dataclass(frozen=True)
class SpeechIsland:
    island_id: str
    block_id: str
    speaker_id: str
    start_ms: int
    end_ms: int
    word_ids: tuple[str, ...]
    text: str
    timing_evidence: str

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "word_ids": list(self.word_ids),
            "duration_ms": self.duration_ms,
        }


@dataclass(frozen=True)
class InteractionEvent:
    interaction_id: str
    kind: str
    start_ms: int
    end_ms: int
    participant_block_ids: tuple[str, ...]
    participant_speaker_ids: tuple[str, ...]
    interjector_block_id: str | None
    floor_speaker_id: str | None
    overlap_ms: int
    source_authorized: bool
    timing_evidence: str

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "participant_block_ids": list(self.participant_block_ids),
            "participant_speaker_ids": list(self.participant_speaker_ids),
        }


def _block_value(block: Any, field: str, default: Any = None) -> Any:
    if isinstance(block, Mapping):
        return block.get(field, default)
    return getattr(block, field, default)


def _block_id(block: Any) -> str:
    return str(_block_value(block, "block_id") or "").split("_variant_", 1)[0]


def _normalise_words(words: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for index, word in enumerate(words, start=1):
        try:
            start_ms = int(round(float(word.get("start_ms"))))
            end_ms = int(round(float(word.get("end_ms"))))
        except (TypeError, ValueError):
            continue
        speaker_id = str(word.get("speaker_id") or "").strip()
        if not speaker_id or start_ms < 0 or end_ms < start_ms:
            continue
        values.append(
            {
                "word_id": str(word.get("word_id") or f"word-{index:05d}"),
                "speaker_id": speaker_id,
                "text": str(word.get("text") or "").strip(),
                "start_ms": start_ms,
                "end_ms": end_ms,
            }
        )
    return sorted(
        values,
        key=lambda item: (
            item["start_ms"],
            item["end_ms"],
            item["speaker_id"],
            item["word_id"],
        ),
    )


def _words_for_block(
    block: Any,
    words: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    speaker_id = str(_block_value(block, "speaker_id") or "")
    start_ms = int(_block_value(block, "start_ms", 0))
    end_ms = int(_block_value(block, "end_ms", start_ms))
    return [
        dict(word)
        for word in words
        if word["speaker_id"] == speaker_id
        and start_ms <= (int(word["start_ms"]) + int(word["end_ms"])) / 2.0 <= end_ms
    ]


def _build_block_islands(
    block: Any,
    words: Sequence[Mapping[str, Any]],
    *,
    island_gap_ms: int,
) -> tuple[list[SpeechIsland], list[dict[str, Any]], str]:
    block_id = _block_id(block)
    speaker_id = str(_block_value(block, "speaker_id") or "")
    start_ms = int(_block_value(block, "start_ms", 0))
    end_ms = int(_block_value(block, "end_ms", start_ms))
    block_words = _words_for_block(block, words)
    if not block_words:
        return (
            [
                SpeechIsland(
                    island_id=f"{block_id}:island-001",
                    block_id=block_id,
                    speaker_id=speaker_id,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    word_ids=(),
                    text=str(_block_value(block, "source_text") or "").strip(),
                    timing_evidence="block_boundary_fallback",
                )
            ],
            [],
            "block_boundary_fallback",
        )

    pauses: list[dict[str, Any]] = []
    groups: list[list[dict[str, Any]]] = [[block_words[0]]]
    for left, right in zip(block_words, block_words[1:]):
        gap_ms = max(0, int(right["start_ms"]) - int(left["end_ms"]))
        if gap_ms >= MICRO_PAUSE_MIN_MS:
            pauses.append(
                {
                    "start_ms": int(left["end_ms"]),
                    "end_ms": int(right["start_ms"]),
                    "duration_ms": gap_ms,
                    "left_word_id": left["word_id"],
                    "right_word_id": right["word_id"],
                    "kind": (
                        "island_boundary"
                        if gap_ms >= island_gap_ms
                        else "phrase_pause"
                        if gap_ms >= PHRASE_PAUSE_MIN_MS
                        else "micro_pause"
                    ),
                }
            )
        if gap_ms >= island_gap_ms:
            groups.append([])
        groups[-1].append(right)

    islands = [
        SpeechIsland(
            island_id=f"{block_id}:island-{index:03d}",
            block_id=block_id,
            speaker_id=speaker_id,
            start_ms=int(group[0]["start_ms"]),
            end_ms=int(group[-1]["end_ms"]),
            word_ids=tuple(str(word["word_id"]) for word in group),
            text=" ".join(str(word["text"]) for word in group if word["text"]),
            timing_evidence="azure_word_timestamps",
        )
        for index, group in enumerate(groups, start=1)
        if group
    ]
    return islands, pauses, "azure_word_timestamps"


def _overlap_events(
    blocks: Sequence[Any],
    words: Sequence[Mapping[str, Any]],
) -> list[InteractionEvent]:
    candidates: list[dict[str, Any]] = []
    for left_index, left in enumerate(blocks):
        left_id = _block_id(left)
        left_speaker = str(_block_value(left, "speaker_id") or "")
        left_words = _words_for_block(left, words)
        for right in blocks[left_index + 1 :]:
            right_id = _block_id(right)
            right_speaker = str(_block_value(right, "speaker_id") or "")
            if left_speaker == right_speaker:
                continue
            right_words = _words_for_block(right, words)
            for left_word in left_words:
                for right_word in right_words:
                    start_ms = max(
                        int(left_word["start_ms"]),
                        int(right_word["start_ms"]),
                    )
                    end_ms = min(
                        int(left_word["end_ms"]),
                        int(right_word["end_ms"]),
                    )
                    if end_ms <= start_ms:
                        continue
                    candidates.append(
                        {
                            "start_ms": start_ms,
                            "end_ms": end_ms,
                            "block_ids": (left_id, right_id),
                            "speaker_ids": (left_speaker, right_speaker),
                        }
                    )

    candidates.sort(key=lambda item: (item["block_ids"], item["start_ms"]))
    merged: list[dict[str, Any]] = []
    for candidate in candidates:
        if (
            merged
            and merged[-1]["block_ids"] == candidate["block_ids"]
            and candidate["start_ms"] <= merged[-1]["end_ms"] + INTERACTION_JOIN_GAP_MS
        ):
            merged[-1]["end_ms"] = max(merged[-1]["end_ms"], candidate["end_ms"])
        else:
            merged.append(dict(candidate))

    events: list[InteractionEvent] = []
    for index, value in enumerate(merged, start=1):
        start_ms = int(value["start_ms"])
        end_ms = int(value["end_ms"])
        events.append(
            InteractionEvent(
                interaction_id=f"interaction-overlap-{index:03d}",
                kind="source_overlap",
                start_ms=start_ms,
                end_ms=end_ms,
                participant_block_ids=tuple(value["block_ids"]),
                participant_speaker_ids=tuple(value["speaker_ids"]),
                interjector_block_id=None,
                floor_speaker_id=None,
                overlap_ms=end_ms - start_ms,
                source_authorized=True,
                timing_evidence="azure_word_interval_intersection",
            )
        )
    return events


def _sandwiched_interjections(
    blocks: Sequence[Any],
    islands_by_block: Mapping[str, Sequence[SpeechIsland]],
    existing: Sequence[InteractionEvent],
) -> list[InteractionEvent]:
    events: list[InteractionEvent] = []
    for index in range(1, len(blocks) - 1):
        previous = blocks[index - 1]
        middle = blocks[index]
        following = blocks[index + 1]
        previous_speaker = str(_block_value(previous, "speaker_id") or "")
        middle_speaker = str(_block_value(middle, "speaker_id") or "")
        following_speaker = str(_block_value(following, "speaker_id") or "")
        middle_id = _block_id(middle)
        if not (previous_speaker == following_speaker and previous_speaker != middle_speaker):
            continue
        middle_islands = list(islands_by_block.get(middle_id, ()))
        if not middle_islands or any(value.timing_evidence != "azure_word_timestamps" for value in middle_islands):
            continue
        middle_start_ms = min(value.start_ms for value in middle_islands)
        middle_end_ms = max(value.end_ms for value in middle_islands)
        if middle_end_ms - middle_start_ms > MAX_INTERJECTION_DURATION_MS:
            continue
        previous_islands = list(islands_by_block.get(_block_id(previous), ()))
        following_islands = list(islands_by_block.get(_block_id(following), ()))
        if not previous_islands or not following_islands:
            continue
        left_gap_ms = middle_start_ms - max(value.end_ms for value in previous_islands)
        right_gap_ms = min(value.start_ms for value in following_islands) - middle_end_ms
        if left_gap_ms > MAX_INTERJECTION_HANDOFF_GAP_MS or right_gap_ms > MAX_INTERJECTION_HANDOFF_GAP_MS:
            continue
        previous_id = _block_id(previous)
        following_id = _block_id(following)
        overlap_ms = sum(event.overlap_ms for event in existing if middle_id in event.participant_block_ids)
        events.append(
            InteractionEvent(
                interaction_id=f"interaction-interjection-{index:03d}",
                kind="sandwiched_interjection",
                start_ms=middle_start_ms,
                end_ms=middle_end_ms,
                participant_block_ids=(previous_id, middle_id, following_id),
                participant_speaker_ids=(
                    previous_speaker,
                    middle_speaker,
                    following_speaker,
                ),
                interjector_block_id=middle_id,
                floor_speaker_id=previous_speaker,
                overlap_ms=overlap_ms,
                source_authorized=True,
                timing_evidence="azure_word_timestamps_a_b_a",
            )
        )
    return events


def build_transcript_lipsync_plan(
    blocks: Sequence[Any],
    canonical_words: Sequence[Mapping[str, Any]],
    *,
    island_gap_ms: int = ISLAND_GAP_MIN_MS,
) -> dict[str, Any]:
    """Return additive timing evidence while keeping legacy blocks valid."""

    if island_gap_ms < PHRASE_PAUSE_MIN_MS:
        raise ValueError("island_gap_ms must be at least the phrase-pause threshold")
    words = _normalise_words(canonical_words)
    block_records: list[dict[str, Any]] = []
    islands_by_block: dict[str, list[SpeechIsland]] = {}
    for block in blocks:
        islands, pauses, evidence = _build_block_islands(
            block,
            words,
            island_gap_ms=island_gap_ms,
        )
        block_id = _block_id(block)
        islands_by_block[block_id] = islands
        block_records.append(
            {
                "block_id": block_id,
                "speaker_id": str(_block_value(block, "speaker_id") or ""),
                "timing_evidence": evidence,
                "source_speech_envelope": {
                    "start_ms": min(value.start_ms for value in islands),
                    "end_ms": max(value.end_ms for value in islands),
                    "duration_ms": (
                        max(value.end_ms for value in islands)
                        - min(value.start_ms for value in islands)
                    ),
                    "semantic_fit_endpoint_authoritative": (
                        evidence == "azure_word_timestamps"
                    ),
                },
                "speech_islands": [value.to_dict() for value in islands],
                "source_pauses": pauses,
                "interaction_ids": [],
                "interaction_roles": [],
            }
        )

    overlap_events = _overlap_events(blocks, words)
    events = [
        *overlap_events,
        *_sandwiched_interjections(blocks, islands_by_block, overlap_events),
    ]
    record_by_id = {record["block_id"]: record for record in block_records}
    for event in events:
        for block_id in event.participant_block_ids:
            record = record_by_id.get(block_id)
            if record is None:
                continue
            role = (
                "interjector"
                if event.interjector_block_id == block_id
                else "floor"
                if event.floor_speaker_id == record["speaker_id"]
                else "participant"
            )
            record["interaction_ids"].append(event.interaction_id)
            record["interaction_roles"].append(
                {
                    "interaction_id": event.interaction_id,
                    "kind": event.kind,
                    "role": role,
                    "source_authorized": event.source_authorized,
                    "overlap_ms": event.overlap_ms,
                }
            )

    timed_block_count = sum(record["timing_evidence"] == "azure_word_timestamps" for record in block_records)
    return {
        "schema_version": TRANSCRIPT_LIPSYNC_SCHEMA_VERSION,
        "policy_version": TRANSCRIPT_LIPSYNC_POLICY_VERSION,
        "evidence_source": "azure_transcript_word_timestamps",
        "compute_profile": {
            "audio_decoding_required": False,
            "vad_required": False,
            "visual_mouth_analysis_required": False,
            "machine_learning_runtime_required": False,
            "gpu_required": False,
        },
        "thresholds_ms": {
            "micro_pause": MICRO_PAUSE_MIN_MS,
            "phrase_pause": PHRASE_PAUSE_MIN_MS,
            "speech_island_gap": island_gap_ms,
            "interaction_join_gap": INTERACTION_JOIN_GAP_MS,
            "maximum_interjection_duration": MAX_INTERJECTION_DURATION_MS,
            "maximum_interjection_handoff_gap": (MAX_INTERJECTION_HANDOFF_GAP_MS),
        },
        "word_count": len(words),
        "block_count": len(block_records),
        "word_timed_block_count": timed_block_count,
        "fallback_block_count": len(block_records) - timed_block_count,
        "blocks": block_records,
        "interaction_count": len(events),
        "interactions": [value.to_dict() for value in events],
        "compatibility": {
            "existing_block_contract_preserved": True,
            "missing_word_timing_falls_back_to_block_boundaries": True,
            "legacy_cli_flags_unchanged": True,
            "approved_translation_changed": False,
        },
        "render_contract": {
            "word_timed_speech_endpoint_is_authoritative": True,
            "same_speaker_transition_borrow_for_word_timed_blocks": False,
            "island_boundaries_receive_pause_capacity_first": True,
            "phrase_pauses_use_remaining_locked_tempo_capacity": True,
            "phrase_pauses_may_absorb_azure_pause_realization_error": True,
            "island_pauses_never_calibrate_below_requested_floor": True,
            "pause_capacity_can_fill_word_timed_underflow": True,
            "timing_tolerance_applies_to_start_and_end_boundaries": True,
            "legacy_block_timing_fallback_preserved": True,
        },
    }


__all__ = [
    "ISLAND_GAP_MIN_MS",
    "MICRO_PAUSE_MIN_MS",
    "PHRASE_PAUSE_MIN_MS",
    "TRANSCRIPT_LIPSYNC_POLICY_VERSION",
    "TRANSCRIPT_LIPSYNC_SCHEMA_VERSION",
    "build_transcript_lipsync_plan",
]
