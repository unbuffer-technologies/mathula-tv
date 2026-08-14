"""Deterministic Azure voice resolution from speaker context."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .atomic_io import atomic_write_json, read_json
from .azure_voice_catalogue import get_catalogue, resolve_voice_for_family
from .speaker_profiles import SpeakerProfilesArtifact


VOICE_ASSIGNMENTS_SCHEMA_VERSION = "azure-voice-assignments-v1"
PERSONA_V2_VERSION = "azure-anchor-persona-v2"
_MAX_BASE_PITCH_PERCENT = 10
_MAX_BASE_RATE_PERCENT = 8
_PERSONA_SOURCE_WEIGHT = 0.45


@dataclass
class VoiceAssignment:
    """Azure anchor voice and stable relative prosody for one speaker."""

    speaker_id: str
    target_locale: str
    tts_voice_family: str
    selected_voice: str
    resolution_status: str  # "resolved" | "fallback" | "failed"
    resolution_reason: str
    evidence_source: str
    confidence: float
    anchor_voice_family: str = ""
    base_prosody: dict[str, Any] = field(default_factory=dict)
    source_features: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "speaker_id": self.speaker_id,
            "target_locale": self.target_locale,
            "tts_voice_family": self.tts_voice_family,
            "selected_voice": self.selected_voice,
            "resolution_status": self.resolution_status,
            "resolution_reason": self.resolution_reason,
            "evidence_source": self.evidence_source,
            "confidence": self.confidence,
            "anchor_voice_family": self.anchor_voice_family,
            "base_prosody": self.base_prosody,
            "source_features": self.source_features,
        }


@dataclass
class VoiceAssignmentsArtifact:
    """Complete voice assignments for a job."""

    schema_version: str = VOICE_ASSIGNMENTS_SCHEMA_VERSION
    job_id: str = ""
    target_locale: str = ""
    speaker_profiles_checksum: str = ""
    assignments: list[dict[str, Any]] = field(default_factory=list)
    unresolved_speakers: list[str] = field(default_factory=list)
    diagnostic_info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "job_id": self.job_id,
            "target_locale": self.target_locale,
            "speaker_profiles_checksum": self.speaker_profiles_checksum,
            "assignments": self.assignments,
            "unresolved_speakers": self.unresolved_speakers,
            "diagnostic_info": self.diagnostic_info,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VoiceAssignmentsArtifact":
        return cls(
            schema_version=data.get("schema_version", VOICE_ASSIGNMENTS_SCHEMA_VERSION),
            job_id=data.get("job_id", ""),
            target_locale=data.get("target_locale", ""),
            speaker_profiles_checksum=data.get("speaker_profiles_checksum", ""),
            assignments=data.get("assignments", []),
            unresolved_speakers=data.get("unresolved_speakers", []),
            diagnostic_info=data.get("diagnostic_info", {}),
        )


def resolve_azure_voices(
    job_id: str,
    speaker_profiles: SpeakerProfilesArtifact,
    target_locale: str,
    fallback_allowed: bool = False,
    min_confidence: float = 0.5,
) -> VoiceAssignmentsArtifact:
    """Resolve Themba/Thando anchors and relative per-speaker delivery.

    Speakers in the masculine family share the locale's masculine anchor voice;
    speakers in the feminine family share the feminine anchor. Pitch and rate are
    measured relative to the other source speakers in the same family, so multiple
    men or women remain distinct without inventing additional Azure voices.
    """
    try:
        catalogue = get_catalogue(target_locale)
    except ValueError as exc:
        raise ValueError(f"Unsupported target locale '{target_locale}': {exc}") from exc

    pending: list[dict[str, Any]] = []
    unresolved_speakers: list[str] = []
    diagnostic_info: dict[str, Any] = {
        "catalogue_locale": target_locale,
        "available_families": sorted(catalogue.available_families()),
        "available_voices": [voice.name for voice in catalogue.voices],
        "total_speakers": len(speaker_profiles.speakers),
        "anchor_strategy": "source_relative_family_anchor",
    }

    for speaker_profile in speaker_profiles.speakers:
        speaker_id = str(speaker_profile["speaker_id"])
        requested_family = str(speaker_profile.get("tts_voice_family", "unknown"))
        evidence_source = str(speaker_profile.get("tts_voice_family_source", "unknown"))
        confidence = _resolution_confidence(speaker_profile, evidence_source)
        effective_family = requested_family

        try:
            if requested_family == "unknown":
                if not fallback_allowed:
                    raise ValueError(
                        f"Speaker {speaker_id} has unknown voice family and fallback is not allowed"
                    )
                effective_family = "masculine"
                voice_name, resolution_reason = resolve_voice_for_family(
                    target_locale,
                    effective_family,
                    fallback_allowed=True,
                )
                resolution_status = "fallback"
                confidence = 0.1
            else:
                voice_name, resolution_reason = resolve_voice_for_family(
                    target_locale,
                    requested_family,
                    fallback_allowed=fallback_allowed,
                )
                resolution_status = "resolved"

            if resolution_status == "resolved" and confidence < min_confidence:
                if not fallback_allowed:
                    raise ValueError(
                        f"Speaker {speaker_id} confidence {confidence} below threshold {min_confidence}"
                    )
                voice_name, resolution_reason = resolve_voice_for_family(
                    target_locale,
                    effective_family,
                    fallback_allowed=True,
                )
                resolution_status = "fallback"

            source_features = speaker_profile.get("source_delivery", {})
            if not isinstance(source_features, dict):
                source_features = {}
            pending.append(
                {
                    "speaker_id": speaker_id,
                    "requested_family": requested_family,
                    "effective_family": effective_family,
                    "selected_voice": voice_name,
                    "resolution_status": resolution_status,
                    "resolution_reason": resolution_reason,
                    "evidence_source": evidence_source,
                    "confidence": confidence,
                    "source_features": source_features,
                }
            )
        except ValueError as exc:
            unresolved_speakers.append(speaker_id)
            diagnostic_info.setdefault("resolution_failures", {})[speaker_id] = {
                "reason": str(exc),
                "tts_voice_family": requested_family,
                "confidence": confidence,
                "evidence_source": evidence_source,
            }

    if unresolved_speakers and not fallback_allowed:
        raise ValueError(
            f"Unable to resolve an Azure voice for {unresolved_speakers[0]} because the canonical speaker profile has no confident voice family. No Azure synthesis request was made."
        )

    family_centres = _family_feature_centres(pending)
    diagnostic_info["family_relative_centres"] = family_centres
    assignments = []
    for item in pending:
        family = item["effective_family"]
        base_prosody = _relative_base_prosody(
            anchor_voice=item["selected_voice"],
            anchor_family=family,
            source_features=item["source_features"],
            centre=family_centres.get(family, {}),
        )
        assignments.append(
            VoiceAssignment(
                speaker_id=item["speaker_id"],
                target_locale=target_locale,
                tts_voice_family=item["requested_family"],
                selected_voice=item["selected_voice"],
                resolution_status=item["resolution_status"],
                resolution_reason=item["resolution_reason"],
                evidence_source=item["evidence_source"],
                confidence=item["confidence"],
                anchor_voice_family=family,
                base_prosody=base_prosody,
                source_features=item["source_features"],
            ).to_dict()
        )

    assignments = _spread_family_personas(assignments)
    diagnostic_info["persona_version"] = PERSONA_V2_VERSION
    diagnostic_info["persona_separation"] = {
        "strategy": "source_relative_perceptual_spread",
        "source_weight": _PERSONA_SOURCE_WEIGHT,
        "pitch_limit_percent": _MAX_BASE_PITCH_PERCENT,
        "rate_limit_percent": _MAX_BASE_RATE_PERCENT,
    }

    return VoiceAssignmentsArtifact(
        schema_version=VOICE_ASSIGNMENTS_SCHEMA_VERSION,
        job_id=job_id,
        target_locale=target_locale,
        speaker_profiles_checksum=_compute_checksum(speaker_profiles.to_dict()),
        assignments=assignments,
        unresolved_speakers=unresolved_speakers,
        diagnostic_info=diagnostic_info,
    )


def _resolution_confidence(profile: dict[str, Any], evidence_source: str) -> float:
    if evidence_source == "authoritative_identity_gender":
        identity_confidence = profile.get("identity_confidence")
        if identity_confidence is not None:
            return float(identity_confidence)
    return float(profile.get("acoustic_confidence", 0.0) or 0.0)


def _family_feature_centres(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    centres: dict[str, dict[str, Any]] = {}
    for family in ("masculine", "feminine"):
        members = [record for record in records if record["effective_family"] == family]
        f0_values = [
            value
            for member in members
            if (value := _positive_number(member["source_features"].get("median_f0_hz")))
            is not None
        ]
        rate_values = [
            value
            for member in members
            if (value := _positive_number(member["source_features"].get("source_speech_rate_wps")))
            is not None
        ]
        centres[family] = {
            "speaker_count": len(members),
            "median_f0_hz": round(statistics.median(f0_values), 4) if f0_values else None,
            "median_speech_rate_wps": round(statistics.median(rate_values), 4)
            if rate_values
            else None,
            "pitch_measurement_count": len(f0_values),
            "rate_measurement_count": len(rate_values),
        }
    return centres


def _relative_base_prosody(
    *,
    anchor_voice: str,
    anchor_family: str,
    source_features: dict[str, Any],
    centre: dict[str, Any],
) -> dict[str, Any]:
    source_f0 = _positive_number(source_features.get("median_f0_hz"))
    centre_f0 = _positive_number(centre.get("median_f0_hz"))
    pitch_percent = 0
    if source_f0 is not None and centre_f0 is not None:
        semitone_delta = 12.0 * math.log2(source_f0 / centre_f0)
        pitch_percent = _clamp_int(
            round(semitone_delta * 2.5),
            -_MAX_BASE_PITCH_PERCENT,
            _MAX_BASE_PITCH_PERCENT,
        )

    source_rate = _positive_number(source_features.get("source_speech_rate_wps"))
    centre_rate = _positive_number(centre.get("median_speech_rate_wps"))
    rate_percent = 0
    if source_rate is not None and centre_rate is not None:
        rate_percent = _clamp_int(
            round(((source_rate / centre_rate) - 1.0) * 40.0),
            -_MAX_BASE_RATE_PERCENT,
            _MAX_BASE_RATE_PERCENT,
        )

    has_measurement = source_f0 is not None or source_rate is not None
    return {
        "anchor_voice": anchor_voice,
        "anchor_voice_family": anchor_family,
        "strategy": (
            "source_relative_family_anchor" if has_measurement else "neutral_family_anchor"
        ),
        "rate_percent": rate_percent,
        "pitch_percent": pitch_percent,
        "volume_percent": 0,
        "relative_reference": {
            "family_median_f0_hz": centre.get("median_f0_hz"),
            "family_median_speech_rate_wps": centre.get("median_speech_rate_wps"),
            "family_speaker_count": centre.get("speaker_count", 0),
        },
    }



def _spread_family_personas(
    assignments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Blend source delivery with deterministic slots for audible separation."""

    updated: list[dict[str, Any]] = []
    for assignment in assignments:
        item = dict(assignment)
        item["base_prosody"] = dict(item.get("base_prosody") or {})
        item["source_features"] = dict(item.get("source_features") or {})
        updated.append(item)

    for family in ("masculine", "feminine"):
        family_items = [
            item
            for item in updated
            if str(item.get("anchor_voice_family") or "") == family
        ]
        if not family_items:
            continue

        ordered = sorted(family_items, key=_persona_sort_key)
        count = len(ordered)

        for rank, item in enumerate(ordered):
            prosody = item["base_prosody"]
            raw_pitch = _clamp_int(
                _safe_int(prosody.get("pitch_percent")),
                -_MAX_BASE_PITCH_PERCENT,
                _MAX_BASE_PITCH_PERCENT,
            )
            raw_rate = _clamp_int(
                _safe_int(prosody.get("rate_percent")),
                -_MAX_BASE_RATE_PERCENT,
                _MAX_BASE_RATE_PERCENT,
            )
            raw_volume = _clamp_int(
                _safe_int(prosody.get("volume_percent")),
                -3,
                3,
            )

            target_pitch, target_rate, target_volume = _persona_target(
                rank=rank,
                count=count,
            )

            if count == 1:
                final_pitch = raw_pitch
                final_rate = raw_rate
                final_volume = raw_volume
            else:
                target_weight = 1.0 - _PERSONA_SOURCE_WEIGHT
                final_pitch = _clamp_int(
                    round(
                        raw_pitch * _PERSONA_SOURCE_WEIGHT
                        + target_pitch * target_weight
                    ),
                    -_MAX_BASE_PITCH_PERCENT,
                    _MAX_BASE_PITCH_PERCENT,
                )
                final_rate = _clamp_int(
                    round(
                        raw_rate * _PERSONA_SOURCE_WEIGHT
                        + target_rate * target_weight
                    ),
                    -_MAX_BASE_RATE_PERCENT,
                    _MAX_BASE_RATE_PERCENT,
                )
                final_volume = _clamp_int(target_volume, -1, 1)

            prosody.update(
                {
                    "strategy": (
                        "source_relative_family_anchor_perceptual_spread"
                        if count > 1
                        else str(
                            prosody.get("strategy")
                            or "source_relative_family_anchor"
                        )
                    ),
                    "rate_percent": final_rate,
                    "pitch_percent": final_pitch,
                    "volume_percent": final_volume,
                    "persona_version": PERSONA_V2_VERSION,
                    "persona_slot": f"{family}_{rank + 1:02d}",
                    "family_persona_count": count,
                    "persona_v1": {
                        "rate_percent": raw_rate,
                        "pitch_percent": raw_pitch,
                        "volume_percent": raw_volume,
                    },
                    "persona_target": {
                        "rate_percent": target_rate,
                        "pitch_percent": target_pitch,
                        "volume_percent": target_volume,
                    },
                    "separation_adjusted": (
                        final_pitch != raw_pitch
                        or final_rate != raw_rate
                        or final_volume != raw_volume
                    ),
                }
            )

    return updated


def _persona_sort_key(assignment: dict[str, Any]) -> tuple[float, float, str]:
    """Order low/slow to high/fast, with a deterministic ID fallback."""

    features = assignment.get("source_features")
    if not isinstance(features, dict):
        features = {}

    f0 = _positive_number(features.get("median_f0_hz"))
    rate = _positive_number(features.get("source_speech_rate_wps"))
    speaker_id = str(assignment.get("speaker_id") or "")

    digest = hashlib.sha256(speaker_id.encode("utf-8")).digest()
    stable_fraction = int.from_bytes(digest[:8], "big") / float(2**64 - 1)

    return (
        f0 if f0 is not None else 1000.0 + stable_fraction,
        rate if rate is not None else 1000.0 + stable_fraction,
        speaker_id,
    )


def _persona_target(*, rank: int, count: int) -> tuple[int, int, int]:
    """Return an evenly spread pitch plus a complementary rate pattern."""

    if count <= 1:
        return 0, 0, 0

    pitch = round(
        -_MAX_BASE_PITCH_PERCENT
        + (2 * _MAX_BASE_PITCH_PERCENT * rank / (count - 1))
    )
    rate_slots = (-6, 6, -3, 3, -8, 8, -1, 1, -5, 5)
    volume_slots = (-1, 1, 0, 0, -1, 1, 0, 0, -1, 1)

    return (
        _clamp_int(
            pitch,
            -_MAX_BASE_PITCH_PERCENT,
            _MAX_BASE_PITCH_PERCENT,
        ),
        _clamp_int(
            rate_slots[rank % len(rate_slots)],
            -_MAX_BASE_RATE_PERCENT,
            _MAX_BASE_RATE_PERCENT,
        ),
        volume_slots[rank % len(volume_slots)],
    )


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def _positive_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 0:
        return None
    return number


def _clamp_int(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, int(value)))


def write_voice_assignments(
    artifact: VoiceAssignmentsArtifact,
    output_path: Path,
) -> None:
    """Write voice assignments artifact to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = artifact.to_dict()
    data["artifact_sha256"] = _compute_checksum(data)
    atomic_write_json(output_path, data)


def load_voice_assignments(path: Path) -> VoiceAssignmentsArtifact:
    """Load voice assignments artifact from disk."""
    return VoiceAssignmentsArtifact.from_dict(read_json(path))


def validate_assignments_checksum(
    artifact: VoiceAssignmentsArtifact,
    speaker_profiles: SpeakerProfilesArtifact,
) -> bool:
    """Validate that voice assignments match speaker profiles checksum."""
    return artifact.speaker_profiles_checksum == _compute_checksum(speaker_profiles.to_dict())


def get_voice_for_speaker(
    speaker_id: str,
    assignments: VoiceAssignmentsArtifact,
) -> str | None:
    """Get the assigned Azure voice for a specific speaker."""
    for assignment in assignments.assignments:
        if assignment["speaker_id"] == speaker_id:
            return assignment["selected_voice"]
    return None


def _compute_checksum(data: dict[str, Any]) -> str:
    """Compute SHA256 checksum from dictionary."""
    data_copy = {
        key: value
        for key, value in data.items()
        if key not in ("artifact_sha256", "speaker_profiles_checksum")
    }
    canonical = json.dumps(data_copy, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = [
    "VOICE_ASSIGNMENTS_SCHEMA_VERSION",
    "VoiceAssignment",
    "VoiceAssignmentsArtifact",
    "resolve_azure_voices",
    "write_voice_assignments",
    "load_voice_assignments",
    "validate_assignments_checksum",
    "get_voice_for_speaker",
]
