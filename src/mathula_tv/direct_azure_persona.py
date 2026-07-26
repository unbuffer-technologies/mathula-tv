"""Deterministic Azure personas for the direct dubbing renderer.

The native isiZulu voices remain the clean timbre anchors.  This module makes
speakers sharing Thando or Themba audibly distinct with conservative SSML
prosody only; it never performs voice conversion or rewrites approved text.
"""

from __future__ import annotations

import hashlib
import math
from statistics import median
from typing import Any, Mapping

DIRECT_AZURE_PERSONA_VERSION = "mathula-direct-azure-persona-v3-volume-neutral"
MAX_PERSONA_PITCH_PERCENT = 10
MAX_PERSONA_RATE_PERCENT = 8
MAX_PERSONA_VOLUME_PERCENT = 0
SOURCE_DELIVERY_WEIGHT = 0.45

_SOURCE_FEATURE_KEYS = (
    "median_f0_hz",
    "f0_percentile_25",
    "f0_percentile_75",
    "source_speech_rate_wps",
    "voiced_frame_ratio",
    "usable_duration_ms",
    "source_speech_seconds",
    "source_word_count",
)


def collect_source_features(
    profile: Mapping[str, Any] | None,
    acoustic: Mapping[str, Any] | None,
) -> dict[str, float | int]:
    """Collect delivery measurements from the current direct-render evidence."""

    profile = profile or {}
    acoustic = acoustic or {}
    delivery = profile.get("source_delivery")
    if not isinstance(delivery, Mapping):
        delivery = {}

    evidence = acoustic.get("evidence")
    if not isinstance(evidence, Mapping):
        evidence = {}

    output: dict[str, float | int] = {}
    for key in _SOURCE_FEATURE_KEYS:
        value = delivery.get(key)
        if value is None:
            value = acoustic.get(key)
        if value is None:
            value = evidence.get(key)
        number = _finite_number(value)
        if number is None:
            continue
        if key in {"usable_duration_ms", "source_word_count"}:
            output[key] = int(round(number))
        else:
            output[key] = round(number, 6)
    return output


def apply_direct_azure_personas(
    assignments: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Return assignments with source-relative, perceptually separated personas.

    Explicit voice-map overrides remain exact.  Automatic assignments are
    grouped by Azure voice family.  Their source pitch and pace establish the
    natural ordering, then deterministic family-local slots add enough spacing
    to be audible without post-processing the Azure waveform.
    """

    result: dict[str, dict[str, Any]] = {}
    for speaker_id, raw in assignments.items():
        item = dict(raw)
        item["speaker_id"] = str(item.get("speaker_id") or speaker_id)
        item["base_prosody"] = dict(item.get("base_prosody") or {})
        item["source_features"] = dict(item.get("source_features") or {})
        family = _voice_family(item)
        if family in {"masculine", "feminine"}:
            item["voice_family"] = family
        result[str(speaker_id)] = item

    for family in ("masculine", "feminine"):
        family_items = [
            item
            for item in result.values()
            if _voice_family(item) == family
            and item.get("resolution_status") != "explicit_voice_map"
        ]
        if not family_items:
            continue

        centre_f0 = _median_positive(
            item["source_features"].get("median_f0_hz") for item in family_items
        )
        centre_rate = _median_positive(
            item["source_features"].get("source_speech_rate_wps")
            for item in family_items
        )

        for item in family_items:
            base = item["base_prosody"]
            existing_pitch = _bounded_int(
                base.get("pitch_percent"),
                -MAX_PERSONA_PITCH_PERCENT,
                MAX_PERSONA_PITCH_PERCENT,
            )
            existing_rate = _bounded_int(
                base.get("rate_percent"),
                -MAX_PERSONA_RATE_PERCENT,
                MAX_PERSONA_RATE_PERCENT,
            )
            features = item["source_features"]
            source_f0 = _positive_number(features.get("median_f0_hz"))
            source_rate = _positive_number(features.get("source_speech_rate_wps"))

            measured_pitch = existing_pitch
            if source_f0 is not None and centre_f0 is not None:
                semitone_delta = 12.0 * math.log2(source_f0 / centre_f0)
                measured_pitch = _clamp(
                    round(semitone_delta * 2.5),
                    -MAX_PERSONA_PITCH_PERCENT,
                    MAX_PERSONA_PITCH_PERCENT,
                )

            measured_rate = existing_rate
            if source_rate is not None and centre_rate is not None:
                measured_rate = _clamp(
                    round(((source_rate / centre_rate) - 1.0) * 40.0),
                    -MAX_PERSONA_RATE_PERCENT,
                    MAX_PERSONA_RATE_PERCENT,
                )

            item["_measured_persona"] = {
                "pitch_percent": measured_pitch,
                "rate_percent": measured_rate,
                "volume_percent": 0,
            }

        ordered = sorted(family_items, key=_persona_sort_key)
        count = len(ordered)
        for rank, item in enumerate(ordered):
            measured = item.pop("_measured_persona")
            target_pitch, target_rate, target_volume = _persona_target(
                rank=rank,
                count=count,
            )

            if count == 1:
                pitch = int(measured["pitch_percent"])
                rate = int(measured["rate_percent"])
                volume = 0
            else:
                target_weight = 1.0 - SOURCE_DELIVERY_WEIGHT
                pitch = _clamp(
                    round(
                        measured["pitch_percent"] * SOURCE_DELIVERY_WEIGHT
                        + target_pitch * target_weight
                    ),
                    -MAX_PERSONA_PITCH_PERCENT,
                    MAX_PERSONA_PITCH_PERCENT,
                )
                rate = _clamp(
                    round(
                        measured["rate_percent"] * SOURCE_DELIVERY_WEIGHT
                        + target_rate * target_weight
                    ),
                    -MAX_PERSONA_RATE_PERCENT,
                    MAX_PERSONA_RATE_PERCENT,
                )
                # Loudness is normalized once on the completed dialogue bus.
                # Per-speaker volume is never used as an identity dimension.
                volume = 0

            item["base_prosody"] = {
                "strategy": (
                    "source_relative_perceptual_spread"
                    if count > 1
                    else "source_relative_single_speaker"
                ),
                "persona_version": DIRECT_AZURE_PERSONA_VERSION,
                "persona_slot": f"{family}_{rank + 1:02d}",
                "family_persona_count": count,
                "rate_percent": rate,
                "pitch_percent": pitch,
                "volume_percent": 0,
                "volume_policy": "dialogue_bus_loudness_normalization",
                "measured_persona": measured,
                "persona_target": {
                    "pitch_percent": target_pitch,
                    "rate_percent": target_rate,
                    "volume_percent": target_volume,
                },
                "source_delivery_weight": SOURCE_DELIVERY_WEIGHT,
                "separation_adjusted": (
                    pitch != measured["pitch_percent"]
                    or rate != measured["rate_percent"]
                    or volume != measured["volume_percent"]
                ),
            }

    for item in result.values():
        if item.get("resolution_status") != "explicit_voice_map":
            continue
        base = item["base_prosody"]
        requested_volume = _bounded_int(base.get("volume_percent"), -100, 100)
        item["base_prosody"] = {
            **base,
            "strategy": "explicit_voice_map",
            "persona_version": DIRECT_AZURE_PERSONA_VERSION,
            "persona_slot": f"explicit_{item['speaker_id']}",
            "family_persona_count": 1,
            "requested_volume_percent": requested_volume,
            "volume_percent": 0,
            "volume_policy": "dialogue_bus_loudness_normalization",
            "separation_adjusted": requested_volume != 0,
        }

    return result


def _voice_family(item: Mapping[str, Any]) -> str:
    family = str(
        item.get("voice_family") or item.get("anchor_voice_family") or ""
    ).lower()
    if family in {"masculine", "feminine"}:
        return family
    voice = str(item.get("selected_voice") or "").lower()
    if "thando" in voice:
        return "feminine"
    if "themba" in voice:
        return "masculine"
    return "unknown"


def _persona_sort_key(item: Mapping[str, Any]) -> tuple[float, float, float, str]:
    features = item.get("source_features")
    if not isinstance(features, Mapping):
        features = {}
    speaker_id = str(item.get("speaker_id") or "")
    stable = _stable_fraction(speaker_id)
    f0 = _positive_number(features.get("median_f0_hz"))
    rate = _positive_number(features.get("source_speech_rate_wps"))
    if f0 is not None:
        return (0.0, f0, rate if rate is not None else stable, speaker_id)
    if rate is not None:
        return (1.0, rate, stable, speaker_id)
    return (2.0, stable, stable, speaker_id)


def _persona_target(*, rank: int, count: int) -> tuple[int, int, int]:
    if count <= 1:
        return 0, 0, 0
    pitch = round(
        -MAX_PERSONA_PITCH_PERCENT
        + (2 * MAX_PERSONA_PITCH_PERCENT * rank / (count - 1))
    )
    rate_slots = (-6, 6, -3, 3, -8, 8, -1, 1, -5, 5)
    return (
        _clamp(pitch, -MAX_PERSONA_PITCH_PERCENT, MAX_PERSONA_PITCH_PERCENT),
        _clamp(
            rate_slots[rank % len(rate_slots)],
            -MAX_PERSONA_RATE_PERCENT,
            MAX_PERSONA_RATE_PERCENT,
        ),
        0,
    )


def _stable_fraction(value: str) -> float:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)


def _median_positive(values: Any) -> float | None:
    numbers = [number for value in values if (number := _positive_number(value))]
    return float(median(numbers)) if numbers else None


def _positive_number(value: Any) -> float | None:
    number = _finite_number(value)
    if number is None or number <= 0:
        return None
    return number


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _bounded_int(value: Any, lower: int, upper: int) -> int:
    try:
        parsed = round(float(value))
    except (TypeError, ValueError):
        parsed = 0
    return _clamp(parsed, lower, upper)


def _clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, int(value)))


__all__ = [
    "DIRECT_AZURE_PERSONA_VERSION",
    "MAX_PERSONA_PITCH_PERCENT",
    "MAX_PERSONA_RATE_PERCENT",
    "apply_direct_azure_personas",
    "collect_source_features",
]
