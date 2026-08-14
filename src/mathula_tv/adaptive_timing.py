from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence


@dataclass(frozen=True)
class TimingBlock:
    block_id: str
    speaker_id: str
    start_ms: int
    end_ms: int
    text: str
    voice_id: str | None = None

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)


@dataclass
class AdaptiveTimingConfig:
    max_group_duration_ms: int = 45_000
    max_same_speaker_gap_ms: int = 2_500
    max_post_stretch_ratio: float = 1.18
    azure_rate_ratio: float = 1.15
    max_group_blocks: int = 12

    @property
    def total_allowed_compression_ratio(self) -> float:
        return self.azure_rate_ratio * self.max_post_stretch_ratio


@dataclass
class GroupTimingResult:
    blocks: list[TimingBlock]
    window_start_ms: int
    window_end_ms: int
    available_ms: int
    measured_tts_ms: int
    required_compression_ratio: float
    allowed_compression_ratio: float
    fits: bool
    reason: str
    borrowed_silence_before_ms: int = 0
    borrowed_silence_after_ms: int = 0
    diagnostics: dict = field(default_factory=dict)


def _same_voice(a: TimingBlock, b: TimingBlock) -> bool:
    if a.voice_id and b.voice_id:
        return a.voice_id == b.voice_id
    return True


def _can_append(
    group: Sequence[TimingBlock],
    candidate: TimingBlock,
    config: AdaptiveTimingConfig,
) -> bool:
    if not group:
        return True

    last = group[-1]
    if candidate.speaker_id != last.speaker_id:
        return False
    if not _same_voice(last, candidate):
        return False

    gap_ms = candidate.start_ms - last.end_ms
    if gap_ms < 0:
        return False
    if gap_ms > config.max_same_speaker_gap_ms:
        return False
    if len(group) >= config.max_group_blocks:
        return False

    proposed_duration = candidate.end_ms - group[0].start_ms
    if proposed_duration > config.max_group_duration_ms:
        return False

    return True


def adaptive_same_speaker_group(
    *,
    blocks: Sequence[TimingBlock],
    failing_index: int,
    measure_tts_ms: Callable[[Sequence[TimingBlock]], int],
    config: AdaptiveTimingConfig | None = None,
    previous_foreign_end_ms: int | None = None,
    next_foreign_start_ms: int | None = None,
) -> GroupTimingResult:
    """
    Expands a failing block across adjacent same-speaker blocks until the
    measured synthesized speech fits the combined timeline.

    The translated text is never rewritten.
    """
    config = config or AdaptiveTimingConfig()

    if failing_index < 0 or failing_index >= len(blocks):
        raise IndexError(f"Invalid failing_index: {failing_index}")

    seed = blocks[failing_index]
    group: list[TimingBlock] = [seed]

    left = failing_index - 1
    right = failing_index + 1

    # Prefer expanding forward because this preserves the original start point.
    while True:
        window_start = group[0].start_ms
        window_end = group[-1].end_ms

        if previous_foreign_end_ms is not None:
            window_start = max(window_start, previous_foreign_end_ms)
        if next_foreign_start_ms is not None:
            window_end = min(window_end, next_foreign_start_ms)

        available_ms = max(1, window_end - window_start)
        measured_ms = max(1, int(measure_tts_ms(group)))
        required = measured_ms / available_ms
        allowed = config.total_allowed_compression_ratio

        if required <= allowed:
            return GroupTimingResult(
                blocks=list(group),
                window_start_ms=window_start,
                window_end_ms=window_end,
                available_ms=available_ms,
                measured_tts_ms=measured_ms,
                required_compression_ratio=required,
                allowed_compression_ratio=allowed,
                fits=True,
                reason="adaptive_same_speaker_group_fit",
                diagnostics={
                    "block_ids": [b.block_id for b in group],
                    "group_block_count": len(group),
                },
            )

        expanded = False

        if right < len(blocks) and _can_append(group, blocks[right], config):
            group.append(blocks[right])
            right += 1
            expanded = True
        elif left >= 0:
            candidate = blocks[left]
            first = group[0]
            reverse_ok = (
                candidate.speaker_id == first.speaker_id
                and _same_voice(candidate, first)
                and 0 <= first.start_ms - candidate.end_ms <= config.max_same_speaker_gap_ms
                and len(group) < config.max_group_blocks
                and group[-1].end_ms - candidate.start_ms <= config.max_group_duration_ms
            )
            if reverse_ok:
                group.insert(0, candidate)
                left -= 1
                expanded = True

        if not expanded:
            return GroupTimingResult(
                blocks=list(group),
                window_start_ms=window_start,
                window_end_ms=window_end,
                available_ms=available_ms,
                measured_tts_ms=measured_ms,
                required_compression_ratio=required,
                allowed_compression_ratio=allowed,
                fits=False,
                reason="timeline_collision_or_no_compatible_same_speaker_neighbour",
                diagnostics={
                    "block_ids": [b.block_id for b in group],
                    "group_block_count": len(group),
                    "next_foreign_start_ms": next_foreign_start_ms,
                    "previous_foreign_end_ms": previous_foreign_end_ms,
                },
            )


def post_stretch_ratio(result: GroupTimingResult) -> float:
    """
    Ratio to pass to a pitch-preserving post-stretch stage after Azure rate
    adjustment. 1.0 means no additional stretching is needed.
    """
    remaining = result.required_compression_ratio / max(
        result.allowed_compression_ratio / 1.18, 1e-9
    )
    return max(1.0, min(1.18, remaining))
