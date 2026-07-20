from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from .azure_tts import AzureTTSResult


@dataclass(frozen=True)
class TimingRepairSignal:
    reason: str
    preferred_duration_ms: int
    maximum_duration_ms: int
    attempted_rates_percent: tuple[int, ...]
    observed_durations_ms: tuple[int, ...]
    fastest_rate_percent: int
    shortest_duration_ms: int
    required_duration_reduction_ratio: float
    schema_version: str = "azure-tts-timing-repair-v1"

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "reason": self.reason,
            "preferred_duration_ms": self.preferred_duration_ms,
            "maximum_duration_ms": self.maximum_duration_ms,
            "attempted_rates_percent": list(self.attempted_rates_percent),
            "observed_durations_ms": list(self.observed_durations_ms),
            "fastest_rate_percent": self.fastest_rate_percent,
            "shortest_duration_ms": self.shortest_duration_ms,
            "required_duration_reduction_ratio": self.required_duration_reduction_ratio,
        }


@dataclass(frozen=True)
class TimingFitResult:
    accepted: AzureTTSResult | None
    attempts: tuple[AzureTTSResult, ...]
    preferred_fit: bool
    hard_window_fit: bool
    requires_translation_repair: bool
    repair_signal: TimingRepairSignal | None
    warnings: tuple[str, ...] = ()
    schema_version: str = "azure-tts-timing-search-v1"

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "accepted": self.accepted.to_dict() if self.accepted else None,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "preferred_fit": self.preferred_fit,
            "hard_window_fit": self.hard_window_fit,
            "requires_translation_repair": self.requires_translation_repair,
            "repair_signal": self.repair_signal.to_dict() if self.repair_signal else None,
            "warnings": list(self.warnings),
        }


class AzureTTSTimingSearch:
    """Bounded rate search that never accepts speech outside the hard window.

    The callback owns candidate paths and synthesis idempotency. This class only
    selects rates and emits a structured, turn-local repair signal; it does not
    call an AI provider or silently shorten translated text.
    """

    def __init__(
        self,
        *,
        rate_min_percent: int = -12,
        rate_max_percent: int = 15,
        search_attempts: int = 3,
        preferred_end_tolerance_ms: int = 120,
    ):
        if rate_min_percent > 0 or rate_max_percent < 0 or rate_min_percent > rate_max_percent:
            raise ValueError("Azure TTS timing rate bounds must include neutral rate")
        if isinstance(search_attempts, bool) or not 1 <= search_attempts <= 10:
            raise ValueError("Azure TTS timing search attempts must be between 1 and 10")
        if preferred_end_tolerance_ms < 0:
            raise ValueError("Azure TTS preferred-end tolerance cannot be negative")
        self.rate_min_percent = rate_min_percent
        self.rate_max_percent = rate_max_percent
        self.search_attempts = search_attempts
        self.preferred_end_tolerance_ms = preferred_end_tolerance_ms

    def fit(
        self,
        synthesize_rate: Callable[[int], AzureTTSResult],
        *,
        preferred_duration_ms: int,
        maximum_duration_ms: int,
        initial_rate_percent: int = 0,
    ) -> TimingFitResult:
        if preferred_duration_ms <= 0 or maximum_duration_ms <= 0:
            raise ValueError("Azure TTS timing windows must be positive")
        if preferred_duration_ms > maximum_duration_ms:
            raise ValueError("Azure TTS preferred duration cannot exceed its hard maximum")
        if not self.rate_min_percent <= initial_rate_percent <= self.rate_max_percent:
            raise ValueError("Azure TTS initial rate is outside configured bounds")

        attempts: list[AzureTTSResult] = []
        tried: set[int] = set()
        failing: dict[int, AzureTTSResult] = {}
        fitting: dict[int, AzureTTSResult] = {}

        def run(rate: int) -> AzureTTSResult:
            if rate in tried:
                raise AssertionError("Azure TTS timing search selected a duplicate rate")
            result = synthesize_rate(rate)
            if not isinstance(result, AzureTTSResult):
                raise TypeError("Azure TTS timing callback returned an invalid result")
            if result.rate_percent != rate:
                raise ValueError("Azure TTS timing result rate does not match the requested rate")
            if result.duration_ms <= 0:
                raise ValueError("Azure TTS timing result has no measurable duration")
            tried.add(rate)
            attempts.append(result)
            target = fitting if result.duration_ms <= maximum_duration_ms else failing
            target[rate] = result
            return result

        initial = run(initial_rate_percent)
        # Naturally short speech is accepted immediately; available silence is
        # padding space, not a reason to slow otherwise natural delivery.
        if initial.duration_ms <= maximum_duration_ms:
            return self._accepted(initial, attempts, preferred_duration_ms)

        while len(attempts) < self.search_attempts:
            low_fail = max(failing) if failing else initial_rate_percent
            high_fit = min(fitting) if fitting else None
            remaining = self.search_attempts - len(attempts)
            candidate: int | None
            if high_fit is not None:
                candidate = (low_fail + high_fit) // 2
            elif remaining == 1:
                candidate = self.rate_max_percent
            else:
                latest = failing[low_fail]
                current_factor = (100 + low_fail) / 100
                estimated_factor = current_factor * latest.duration_ms / maximum_duration_ms
                candidate = math.ceil((estimated_factor - 1) * 100)
                candidate = max(low_fail + 1, candidate)
            candidate = max(initial_rate_percent, min(candidate, self.rate_max_percent))
            if candidate in tried:
                candidate = self._unused_rate(tried, low_fail=low_fail, high_fit=high_fit)
            if candidate is None:
                break
            run(candidate)

        if fitting:
            # Use the least accelerated passing candidate to preserve delivery.
            selected = fitting[min(fitting)]
            return self._accepted(selected, attempts, preferred_duration_ms)

        shortest = min(attempts, key=lambda item: (item.duration_ms, -item.rate_percent))
        signal = TimingRepairSignal(
            reason="azure_tts_exceeds_hard_timing_window",
            preferred_duration_ms=preferred_duration_ms,
            maximum_duration_ms=maximum_duration_ms,
            attempted_rates_percent=tuple(item.rate_percent for item in attempts),
            observed_durations_ms=tuple(item.duration_ms for item in attempts),
            fastest_rate_percent=max(item.rate_percent for item in attempts),
            shortest_duration_ms=shortest.duration_ms,
            required_duration_reduction_ratio=round(maximum_duration_ms / shortest.duration_ms, 6),
        )
        return TimingFitResult(
            accepted=None,
            attempts=tuple(attempts),
            preferred_fit=False,
            hard_window_fit=False,
            requires_translation_repair=True,
            repair_signal=signal,
            warnings=("No Azure TTS candidate fits the hard timing window",),
        )

    def _unused_rate(self, tried: set[int], *, low_fail: int, high_fit: int | None) -> int | None:
        upper = high_fit if high_fit is not None else self.rate_max_percent + 1
        candidates = [rate for rate in range(low_fail + 1, upper) if rate not in tried]
        if candidates:
            return candidates[len(candidates) // 2]
        if high_fit is None and self.rate_max_percent not in tried:
            return self.rate_max_percent
        return None

    def _accepted(
        self,
        selected: AzureTTSResult,
        attempts: list[AzureTTSResult],
        preferred_duration_ms: int,
    ) -> TimingFitResult:
        preferred_fit = selected.duration_ms <= preferred_duration_ms + self.preferred_end_tolerance_ms
        warnings = () if preferred_fit else ("Azure TTS fits the hard window but exceeds the preferred end tolerance",)
        return TimingFitResult(
            accepted=selected,
            attempts=tuple(attempts),
            preferred_fit=preferred_fit,
            hard_window_fit=True,
            requires_translation_repair=False,
            repair_signal=None,
            warnings=warnings,
        )
