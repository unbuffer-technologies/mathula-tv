from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import wave
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from .atomic_io import atomic_write_json, read_json
from .openvoice import (
    OpenVoiceAttemptLimitExceeded,
    OpenVoiceAssetHashMismatch,
    OpenVoiceBackend,
    OpenVoiceConversionRequest,
    OpenVoiceError,
    OpenVoiceInferenceError,
    OpenVoiceLeaseLost,
    sha256_file,
)
from .quality import inspect_wav


OPENVOICE_PLAN_SCHEMA = "openvoice-conversion-plan-v1"
OPENVOICE_MANIFEST_SCHEMA = "openvoice-worker-manifest-v1"
AZURE_CALIBRATION_SCHEMA = "azure-openvoice-source-calibration-v1"
OPENVOICE_ASSET_PLAN_SCHEMA = "openvoice-asset-plan-v1"
OPENVOICE_ASSET_MANIFEST_SCHEMA = "openvoice-asset-worker-manifest-v1"
OPENVOICE_SOURCE_EMBEDDING_MANIFEST_SCHEMA = (
    "openvoice-source-embedding-manifest-v1"
)
OPENVOICE_TARGET_EMBEDDING_MANIFEST_SCHEMA = (
    "openvoice-target-embedding-manifest-v1"
)
OPENVOICE_VOICE_CANDIDATES_SCHEMA = "openvoice-azure-voice-candidates-v1"
_SAFE_PATH_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")

__all__ = [
    "AzureSourceCalibrationCache",
    "AzureSourceCalibrationSpec",
    "OPENVOICE_ASSET_MANIFEST_SCHEMA",
    "OPENVOICE_ASSET_PLAN_SCHEMA",
    "OPENVOICE_SOURCE_EMBEDDING_MANIFEST_SCHEMA",
    "OPENVOICE_TARGET_EMBEDDING_MANIFEST_SCHEMA",
    "OPENVOICE_VOICE_CANDIDATES_SCHEMA",
    "OpenVoiceAssetPlan",
    "OpenVoiceAssetWorker",
    "OpenVoicePlan",
    "OpenVoiceSourceAsset",
    "OpenVoiceSpeakerAsset",
    "OpenVoiceUnit",
    "OpenVoiceVoiceCandidate",
    "OpenVoiceWorker",
    "write_openvoice_asset_plan",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class OpenVoiceUnit:
    unit_id: str
    speaker_id: str
    azure_voice: str
    target_language: str
    source_audio_path: Path
    source_audio_sha256: str
    source_embedding_path: Path
    source_embedding_sha256: str
    target_embedding_path: Path
    target_embedding_sha256: str
    output_path: Path
    conversion_settings: dict[str, Any] = field(default_factory=dict)
    sample_rate: int = 24000

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> OpenVoiceUnit:
        return cls(
            unit_id=str(value["unit_id"]),
            speaker_id=str(value["speaker_id"]),
            azure_voice=str(value["azure_voice"]),
            target_language=str(value.get("target_language", "zu-ZA")),
            source_audio_path=Path(value["source_audio_path"]),
            source_audio_sha256=str(value["source_audio_sha256"]),
            source_embedding_path=Path(value["source_embedding_path"]),
            source_embedding_sha256=str(value["source_embedding_sha256"]),
            target_embedding_path=Path(value["target_embedding_path"]),
            target_embedding_sha256=str(value["target_embedding_sha256"]),
            output_path=Path(value["output_path"]),
            conversion_settings=dict(value.get("conversion_settings", {})),
            sample_rate=int(value.get("sample_rate", 24000)),
        )


@dataclass(frozen=True)
class OpenVoicePlan:
    job_id: str
    lease_identity: str
    openvoice_revision: str
    units: tuple[OpenVoiceUnit, ...]
    schema_version: str = OPENVOICE_PLAN_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != OPENVOICE_PLAN_SCHEMA:
            raise ValueError("Unsupported OpenVoice plan schema")
        if not self.units:
            raise ValueError("The OpenVoice plan contains no units")
        unit_ids = [unit.unit_id for unit in self.units]
        if len(unit_ids) != len(set(unit_ids)):
            raise ValueError("OpenVoice plan unit IDs must be unique")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> OpenVoicePlan:
        return cls(
            job_id=str(value["job_id"]),
            lease_identity=str(value["lease_identity"]),
            openvoice_revision=str(value["openvoice_revision"]),
            units=tuple(OpenVoiceUnit.from_dict(unit) for unit in value["units"]),
            schema_version=str(value.get("schema_version", "")),
        )


class OpenVoiceWorker:
    """Local conversion runner with per-unit durable resume state.

    Remote lease claim, heartbeat, download, promotion, and completion stay in the
    repository's existing GCS layer.  ``lease_guard`` lets that layer prove
    ownership immediately before and after each expensive unit conversion.
    """

    def __init__(
        self,
        backend: OpenVoiceBackend,
        plan: OpenVoicePlan,
        manifest_path: Path,
        *,
        lease_guard: Callable[[str], bool | None] | None = None,
        max_attempts_per_unit: int = 2,
        allow_lease_rebind: bool = False,
    ) -> None:
        if plan.openvoice_revision != backend.pins.revision:
            raise ValueError("OpenVoice plan and pinned backend revisions differ")
        if max_attempts_per_unit < 1:
            raise ValueError("OpenVoice maximum attempts must be positive")
        self.backend = backend
        self.plan = plan
        self.manifest_path = manifest_path
        self.lease_guard = lease_guard
        self.max_attempts_per_unit = max_attempts_per_unit
        self.allow_lease_rebind = allow_lease_rebind

    def prepare_manifest(self) -> dict[str, Any]:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        if self.manifest_path.is_file():
            manifest = read_json(self.manifest_path)
            self._validate_manifest(manifest)
            if manifest.get("lease_identity") != self.plan.lease_identity:
                if not self.allow_lease_rebind or self.lease_guard is None:
                    raise ValueError(
                        "OpenVoice worker manifest belongs to another lease identity"
                    )
                self._assert_lease("lease-rebind")
                manifest.setdefault("lease_history", []).append(
                    {
                        "lease_identity": manifest.get("lease_identity"),
                        "ended_at": _now(),
                    }
                )
                manifest["lease_identity"] = self.plan.lease_identity
                manifest["updated_at"] = _now()
                atomic_write_json(self.manifest_path, manifest)
            return manifest
        manifest = {
            "schema_version": OPENVOICE_MANIFEST_SCHEMA,
            "job_id": self.plan.job_id,
            "lease_identity": self.plan.lease_identity,
            "backend": "openvoice",
            "backend_revision": self.plan.openvoice_revision,
            "state": "pending",
            "created_at": _now(),
            "updated_at": _now(),
            "checkpoint_hashes": {},
            "runtime": None,
            "units": {
                unit.unit_id: {
                    "unit_id": unit.unit_id,
                    "speaker_id": unit.speaker_id,
                    "azure_voice": unit.azure_voice,
                    "input_identity_sha256": _unit_identity_sha256(unit),
                    "status": "pending",
                    "attempts": 0,
                    "history": [],
                }
                for unit in self.plan.units
            },
        }
        atomic_write_json(self.manifest_path, manifest)
        return manifest

    def run(self, *, force: bool = False) -> dict[str, Any]:
        manifest = self.prepare_manifest()
        for unit in self.plan.units:
            record = manifest["units"][unit.unit_id]
            if not force and self._completed_output_is_valid(record, unit):
                continue

            attempts = int(record.get("attempts", 0))
            if attempts >= self.max_attempts_per_unit:
                error = OpenVoiceAttemptLimitExceeded(
                    f"OpenVoice attempt limit reached for unit {unit.unit_id}"
                )
                record["last_error"] = error.as_record()
                self._set_state_and_write(manifest)
                raise error

            attempt = attempts + 1
            request = OpenVoiceConversionRequest(
                job_id=self.plan.job_id,
                unit_id=unit.unit_id,
                speaker_id=unit.speaker_id,
                source_audio_path=unit.source_audio_path,
                source_audio_sha256=unit.source_audio_sha256,
                source_embedding_path=unit.source_embedding_path,
                source_embedding_sha256=unit.source_embedding_sha256,
                target_embedding_path=unit.target_embedding_path,
                target_embedding_sha256=unit.target_embedding_sha256,
                target_language=unit.target_language,
                azure_voice=unit.azure_voice,
                conversion_settings=unit.conversion_settings,
                output_path=unit.output_path,
                lease_identity=self.plan.lease_identity,
                attempt=attempt,
                sample_rate=unit.sample_rate,
            )
            try:
                self._assert_lease(unit.unit_id)
                result = self.backend.convert(request)
                self._assert_lease(unit.unit_id)
            except OpenVoiceError as exc:
                self._record_failure(manifest, record, attempt, exc)
                raise
            except Exception as exc:
                safe = OpenVoiceInferenceError(
                    f"Unexpected OpenVoice worker failure ({type(exc).__name__})"
                )
                self._record_failure(manifest, record, attempt, safe)
                raise safe from exc

            self._archive_record(record)
            record.update(
                {
                    "status": "completed",
                    "attempts": attempt,
                    "completed_at": _now(),
                    "output_path": str(unit.output_path),
                    "result": result.to_dict(),
                }
            )
            record.pop("last_error", None)
            record.pop("failed_at", None)
            runtime_info = self.backend.runtime_info
            manifest["runtime"] = (
                {
                    "openvoice_revision": runtime_info.openvoice_revision,
                    "python_version": runtime_info.python_version,
                    "torch_version": runtime_info.torch_version,
                    "cuda_available": runtime_info.cuda_available,
                    "cuda_version": runtime_info.cuda_version,
                    "gpu_name": runtime_info.gpu_name,
                    "device": self.backend.device,
                    "precision": self.backend.precision,
                }
                if runtime_info
                else None
            )
            manifest["checkpoint_hashes"] = self.backend.checkpoint_hashes
            self._set_state_and_write(manifest)
        return self._set_state_and_write(manifest)

    def pending_unit_ids(self) -> list[str]:
        manifest = self.prepare_manifest()
        result = []
        for unit in self.plan.units:
            record = manifest["units"][unit.unit_id]
            if not self._completed_output_is_valid(record, unit):
                result.append(unit.unit_id)
        return result

    def _record_failure(
        self,
        manifest: dict[str, Any],
        record: dict[str, Any],
        attempt: int,
        error: OpenVoiceError,
    ) -> None:
        self._archive_record(record)
        record.update(
            {
                "status": "failed",
                "attempts": attempt,
                "failed_at": _now(),
                "last_error": error.as_record(),
            }
        )
        record.pop("completed_at", None)
        record.pop("output_path", None)
        record.pop("result", None)
        self._set_state_and_write(manifest)

    def _assert_lease(self, unit_id: str) -> None:
        if self.lease_guard is None:
            return
        try:
            owned = self.lease_guard(unit_id)
        except OpenVoiceLeaseLost:
            raise
        except Exception as exc:
            raise OpenVoiceLeaseLost(
                f"Lease ownership could not be confirmed for unit {unit_id}"
            ) from exc
        if owned is False:
            raise OpenVoiceLeaseLost(f"Lease ownership was lost before unit {unit_id}")

    def _completed_output_is_valid(
        self, record: Mapping[str, Any], unit: OpenVoiceUnit
    ) -> bool:
        if record.get("status") != "completed" or not unit.output_path.is_file():
            return False
        expected = (record.get("result") or {}).get("output_sha256")
        return bool(expected) and sha256_file(unit.output_path) == expected

    def _set_state_and_write(self, manifest: dict[str, Any]) -> dict[str, Any]:
        statuses = [item.get("status") for item in manifest["units"].values()]
        if all(status == "completed" for status in statuses):
            state = "completed"
        elif any(status == "failed" for status in statuses):
            state = "partial" if any(status == "completed" for status in statuses) else "failed"
        elif any(status == "completed" for status in statuses):
            state = "partial"
        else:
            state = "pending"
        manifest["state"] = state
        manifest["completed_units"] = sum(status == "completed" for status in statuses)
        manifest["pending_units"] = sum(status != "completed" for status in statuses)
        manifest["updated_at"] = _now()
        atomic_write_json(self.manifest_path, manifest)
        return manifest

    def _validate_manifest(self, manifest: Mapping[str, Any]) -> None:
        expected = {
            "schema_version": OPENVOICE_MANIFEST_SCHEMA,
            "job_id": self.plan.job_id,
            "backend_revision": self.plan.openvoice_revision,
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError(f"OpenVoice worker manifest has incompatible {key}")
        if set(manifest.get("units", {})) != {unit.unit_id for unit in self.plan.units}:
            raise ValueError("OpenVoice worker manifest unit set does not match the plan")
        for unit in self.plan.units:
            if (
                manifest["units"][unit.unit_id].get("input_identity_sha256")
                != _unit_identity_sha256(unit)
            ):
                raise ValueError(
                    f"OpenVoice worker manifest inputs changed for unit {unit.unit_id}"
                )

    @staticmethod
    def _archive_record(record: dict[str, Any]) -> None:
        if int(record.get("attempts", 0)) < 1:
            return
        snapshot = {
            key: value
            for key, value in record.items()
            if key not in {"history", "unit_id", "speaker_id", "azure_voice"}
        }
        record.setdefault("history", []).append(snapshot)


@dataclass(frozen=True)
class AzureSourceCalibrationSpec:
    azure_voice: str
    region: str
    output_format: str
    calibration_text_version: str
    calibration_text_sha256: str
    openvoice_revision: str
    sample_rate: int = 24000
    channels: int = 1
    sample_width: int = 2
    minimum_duration_seconds: float = 20.0
    maximum_duration_seconds: float = 40.0
    reviewed: bool = True

    def __post_init__(self) -> None:
        if not _SAFE_PATH_COMPONENT.fullmatch(self.azure_voice):
            raise ValueError("Invalid Azure calibration voice")
        if not self.region.strip() or not self.output_format.strip():
            raise ValueError("Azure calibration region and output format are required")
        if not re.fullmatch(r"[0-9a-f]{64}", self.calibration_text_sha256):
            raise ValueError("Invalid calibration text SHA-256")
        if not self.reviewed:
            raise ValueError("OpenVoice calibration text must be human reviewed")
        if self.channels != 1 or self.sample_width != 2 or self.sample_rate <= 0:
            raise ValueError("Azure calibration must use canonical mono PCM WAV")
        if not 0 < self.minimum_duration_seconds <= self.maximum_duration_seconds:
            raise ValueError("Invalid Azure calibration duration bounds")

    def identity(self) -> dict[str, Any]:
        return {
            "azure_voice": self.azure_voice,
            "region": self.region,
            "output_format": self.output_format,
            "calibration_text_version": self.calibration_text_version,
            "calibration_text_sha256": self.calibration_text_sha256,
            "openvoice_revision": self.openvoice_revision,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "sample_width": self.sample_width,
        }


@dataclass(frozen=True)
class OpenVoiceSourceAsset:
    """One reviewed Azure calibration WAV and its cached source embedding."""

    azure_voice: str
    calibration_audio_path: Path
    calibration_audio_sha256: str
    calibration_manifest_path: Path
    calibration_manifest_sha256: str
    source_embedding_path: Path
    source_embedding_manifest_path: Path
    calibration: AzureSourceCalibrationSpec

    def __post_init__(self) -> None:
        if not _SAFE_PATH_COMPONENT.fullmatch(self.azure_voice):
            raise ValueError("Invalid Azure source voice")
        if self.azure_voice != self.calibration.azure_voice:
            raise ValueError("Azure source asset and calibration voices differ")
        for label, value in (
            ("calibration audio", self.calibration_audio_sha256),
            ("calibration manifest", self.calibration_manifest_sha256),
        ):
            _require_sha256(label, value)
        inputs = {self.calibration_audio_path, self.calibration_manifest_path}
        outputs = {self.source_embedding_path, self.source_embedding_manifest_path}
        if len(outputs) != 2 or inputs & outputs:
            raise ValueError("Azure source embedding outputs must be distinct from inputs")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> OpenVoiceSourceAsset:
        calibration = value.get("calibration")
        if not isinstance(calibration, Mapping):
            raise ValueError("Azure source asset requires calibration identity")
        return cls(
            azure_voice=str(value["azure_voice"]),
            calibration_audio_path=Path(value["calibration_audio_path"]),
            calibration_audio_sha256=str(value["calibration_audio_sha256"]),
            calibration_manifest_path=Path(value["calibration_manifest_path"]),
            calibration_manifest_sha256=str(
                value["calibration_manifest_sha256"]
            ),
            source_embedding_path=Path(value["source_embedding_path"]),
            source_embedding_manifest_path=Path(
                value["source_embedding_manifest_path"]
            ),
            calibration=AzureSourceCalibrationSpec(**dict(calibration)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "azure_voice": self.azure_voice,
            "calibration_audio_path": str(self.calibration_audio_path),
            "calibration_audio_sha256": self.calibration_audio_sha256,
            "calibration_manifest_path": str(self.calibration_manifest_path),
            "calibration_manifest_sha256": self.calibration_manifest_sha256,
            "source_embedding_path": str(self.source_embedding_path),
            "source_embedding_manifest_path": str(
                self.source_embedding_manifest_path
            ),
            "calibration": {
                **self.calibration.identity(),
                "minimum_duration_seconds": self.calibration.minimum_duration_seconds,
                "maximum_duration_seconds": self.calibration.maximum_duration_seconds,
                "reviewed": self.calibration.reviewed,
            },
        }


@dataclass(frozen=True)
class OpenVoiceVoiceCandidate:
    """A voice-specific Azure calibration sentence awaiting job-local conversion."""

    candidate_id: str
    azure_voice: str
    source_audio_path: Path
    source_audio_sha256: str
    output_path: Path
    calibration_text_sha256: str
    target_language: str = "zu-ZA"
    conversion_settings: dict[str, Any] = field(default_factory=dict)
    sample_rate: int = 24000

    def __post_init__(self) -> None:
        if not _SAFE_PATH_COMPONENT.fullmatch(self.candidate_id):
            raise ValueError("Invalid OpenVoice calibration candidate ID")
        if not _SAFE_PATH_COMPONENT.fullmatch(self.azure_voice):
            raise ValueError("Invalid OpenVoice calibration Azure voice")
        _require_sha256("candidate source audio", self.source_audio_sha256)
        _require_sha256("candidate calibration text", self.calibration_text_sha256)
        if not self.target_language.strip() or self.sample_rate <= 0:
            raise ValueError("Calibration candidate language and sample rate are required")
        _canonical_sha256(self.conversion_settings)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> OpenVoiceVoiceCandidate:
        return cls(
            candidate_id=str(value["candidate_id"]),
            azure_voice=str(value["azure_voice"]),
            source_audio_path=Path(value["source_audio_path"]),
            source_audio_sha256=str(value["source_audio_sha256"]),
            output_path=Path(value["output_path"]),
            calibration_text_sha256=str(value["calibration_text_sha256"]),
            target_language=str(value.get("target_language", "zu-ZA")),
            conversion_settings=dict(value.get("conversion_settings", {})),
            sample_rate=int(value.get("sample_rate", 24000)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "azure_voice": self.azure_voice,
            "source_audio_path": str(self.source_audio_path),
            "source_audio_sha256": self.source_audio_sha256,
            "output_path": str(self.output_path),
            "calibration_text_sha256": self.calibration_text_sha256,
            "target_language": self.target_language,
            "conversion_settings": self.conversion_settings,
            "sample_rate": self.sample_rate,
        }


@dataclass(frozen=True)
class OpenVoiceSpeakerAsset:
    """One job-local same-speaker reference and its target embedding."""

    speaker_id: str
    reference_audio_path: Path
    reference_audio_sha256: str
    reference_manifest_path: Path
    reference_manifest_sha256: str
    target_embedding_path: Path
    target_embedding_manifest_path: Path
    candidate_manifest_path: Path | None = None
    voice_candidates: tuple[OpenVoiceVoiceCandidate, ...] = ()
    sample_rate: int = 24000

    def __post_init__(self) -> None:
        if not _SAFE_PATH_COMPONENT.fullmatch(self.speaker_id):
            raise ValueError("Invalid job-local speaker ID")
        if self.speaker_id == "UNKNOWN":
            raise ValueError("UNKNOWN cannot have an OpenVoice target embedding")
        for label, value in (
            ("speaker reference audio", self.reference_audio_sha256),
            ("speaker reference manifest", self.reference_manifest_sha256),
        ):
            _require_sha256(label, value)
        if self.sample_rate <= 0:
            raise ValueError("Speaker reference sample rate must be positive")
        candidate_ids = [item.candidate_id for item in self.voice_candidates]
        voices = [item.azure_voice for item in self.voice_candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("Speaker voice candidate IDs must be unique")
        if len(voices) != len(set(voices)):
            raise ValueError("A speaker may have only one candidate per Azure voice")
        if self.voice_candidates and self.candidate_manifest_path is None:
            raise ValueError("Voice candidates require a deterministic manifest path")
        inputs = {self.reference_audio_path, self.reference_manifest_path}
        outputs = {self.target_embedding_path, self.target_embedding_manifest_path}
        if self.candidate_manifest_path is not None:
            outputs.add(self.candidate_manifest_path)
        outputs.update(item.output_path for item in self.voice_candidates)
        if len(outputs) != 2 + bool(self.candidate_manifest_path) + len(
            self.voice_candidates
        ) or inputs & outputs:
            raise ValueError("Speaker asset outputs must be unique and distinct from inputs")
        if any(item.output_path == item.source_audio_path for item in self.voice_candidates):
            raise ValueError("Voice candidate output must not overwrite its Azure input")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> OpenVoiceSpeakerAsset:
        candidate_manifest = value.get("candidate_manifest_path")
        return cls(
            speaker_id=str(value["speaker_id"]),
            reference_audio_path=Path(value["reference_audio_path"]),
            reference_audio_sha256=str(value["reference_audio_sha256"]),
            reference_manifest_path=Path(value["reference_manifest_path"]),
            reference_manifest_sha256=str(value["reference_manifest_sha256"]),
            target_embedding_path=Path(value["target_embedding_path"]),
            target_embedding_manifest_path=Path(
                value["target_embedding_manifest_path"]
            ),
            candidate_manifest_path=(
                Path(candidate_manifest) if candidate_manifest is not None else None
            ),
            voice_candidates=tuple(
                OpenVoiceVoiceCandidate.from_dict(item)
                for item in value.get("voice_candidates", ())
            ),
            sample_rate=int(value.get("sample_rate", 24000)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "speaker_id": self.speaker_id,
            "reference_audio_path": str(self.reference_audio_path),
            "reference_audio_sha256": self.reference_audio_sha256,
            "reference_manifest_path": str(self.reference_manifest_path),
            "reference_manifest_sha256": self.reference_manifest_sha256,
            "target_embedding_path": str(self.target_embedding_path),
            "target_embedding_manifest_path": str(
                self.target_embedding_manifest_path
            ),
            "candidate_manifest_path": (
                str(self.candidate_manifest_path)
                if self.candidate_manifest_path is not None
                else None
            ),
            "voice_candidates": [item.to_dict() for item in self.voice_candidates],
            "sample_rate": self.sample_rate,
        }


@dataclass(frozen=True)
class OpenVoiceAssetPlan:
    """Deterministic GPU preflight plan prepared and hashed by the server."""

    job_id: str
    lease_identity: str
    openvoice_revision: str
    sources: tuple[OpenVoiceSourceAsset, ...]
    speakers: tuple[OpenVoiceSpeakerAsset, ...]
    schema_version: str = OPENVOICE_ASSET_PLAN_SCHEMA

    def __post_init__(self) -> None:
        for label, value in (
            ("job ID", self.job_id),
            ("lease identity", self.lease_identity),
        ):
            if not _SAFE_PATH_COMPONENT.fullmatch(value):
                raise ValueError(f"Invalid asset-plan {label}")
        if self.schema_version != OPENVOICE_ASSET_PLAN_SCHEMA:
            raise ValueError("Unsupported OpenVoice asset plan schema")
        if not self.openvoice_revision.strip():
            raise ValueError("OpenVoice asset plan requires an immutable revision")
        if not self.sources or not self.speakers:
            raise ValueError("OpenVoice asset plan requires sources and speakers")
        voices = [item.azure_voice for item in self.sources]
        speaker_ids = [item.speaker_id for item in self.speakers]
        if len(voices) != len(set(voices)):
            raise ValueError("Azure source voices must be unique")
        if len(speaker_ids) != len(set(speaker_ids)):
            raise ValueError("Asset-plan speaker IDs must be unique")
        available_voices = set(voices)
        for source in self.sources:
            if source.calibration.openvoice_revision != self.openvoice_revision:
                raise ValueError("Source calibration and asset-plan revisions differ")
        for speaker in self.speakers:
            missing = {
                item.azure_voice for item in speaker.voice_candidates
            } - available_voices
            if missing:
                raise ValueError(
                    f"Speaker voice candidates lack source calibrations: {sorted(missing)}"
                )
        inputs = {
            path
            for source in self.sources
            for path in (
                source.calibration_audio_path,
                source.calibration_manifest_path,
            )
        }
        inputs.update(
            path
            for speaker in self.speakers
            for path in (speaker.reference_audio_path, speaker.reference_manifest_path)
        )
        inputs.update(
            candidate.source_audio_path
            for speaker in self.speakers
            for candidate in speaker.voice_candidates
        )
        outputs = [
            path
            for source in self.sources
            for path in (
                source.source_embedding_path,
                source.source_embedding_manifest_path,
            )
        ]
        for speaker in self.speakers:
            outputs.extend(
                (
                    speaker.target_embedding_path,
                    speaker.target_embedding_manifest_path,
                )
            )
            if speaker.candidate_manifest_path is not None:
                outputs.append(speaker.candidate_manifest_path)
            outputs.extend(item.output_path for item in speaker.voice_candidates)
        if len(outputs) != len(set(outputs)) or inputs & set(outputs):
            raise ValueError("OpenVoice asset-plan outputs collide with inputs or each other")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> OpenVoiceAssetPlan:
        supplied_hash = str(value.get("artifact_sha256", ""))
        _require_sha256("OpenVoice asset plan", supplied_hash)
        payload = {key: item for key, item in value.items() if key != "artifact_sha256"}
        if _canonical_sha256(payload) != supplied_hash:
            raise OpenVoiceAssetHashMismatch(
                "OpenVoice asset plan artifact hash does not match"
            )
        return cls(
            job_id=str(value["job_id"]),
            lease_identity=str(value["lease_identity"]),
            openvoice_revision=str(value["openvoice_revision"]),
            sources=tuple(
                OpenVoiceSourceAsset.from_dict(item) for item in value["sources"]
            ),
            speakers=tuple(
                OpenVoiceSpeakerAsset.from_dict(item) for item in value["speakers"]
            ),
            schema_version=str(value.get("schema_version", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "job_id": self.job_id,
            "lease_identity": self.lease_identity,
            "openvoice_revision": self.openvoice_revision,
            "sources": [item.to_dict() for item in self.sources],
            "speakers": [item.to_dict() for item in self.speakers],
        }

    @property
    def artifact_sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


def write_openvoice_asset_plan(
    plan: OpenVoiceAssetPlan, output_path: Path
) -> dict[str, Any]:
    """Write a canonical asset plan with a self-verifying artifact identity."""

    value = plan.to_dict()
    value["artifact_sha256"] = _canonical_sha256(value)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_path, value)
    return value


class AzureSourceCalibrationCache:
    """Voice/format/revision-specific Azure source embedding cache."""

    def __init__(self, cache_root: Path) -> None:
        self.cache_root = cache_root

    def directory(self, voice: str) -> Path:
        if not _SAFE_PATH_COMPONENT.fullmatch(voice):
            raise ValueError("Invalid Azure calibration voice")
        return self.cache_root / voice

    def get(self, spec: AzureSourceCalibrationSpec) -> dict[str, Any] | None:
        directory = self.directory(spec.azure_voice)
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            return None
        manifest = read_json(manifest_path)
        if manifest.get("schema_version") != AZURE_CALIBRATION_SCHEMA:
            return None
        if any(manifest.get(key) != value for key, value in spec.identity().items()):
            return None
        wav_path = directory / "calibration.wav"
        embedding_path = directory / "source_embedding.bin"
        for label, path, expected in (
            ("calibration WAV", wav_path, manifest.get("wav_sha256")),
            ("source embedding", embedding_path, manifest.get("embedding_sha256")),
        ):
            if not path.is_file() or not expected or sha256_file(path) != expected:
                raise OpenVoiceAssetHashMismatch(
                    f"Cached Azure {label} hash does not match its manifest"
                )
        return manifest

    def ensure(
        self,
        spec: AzureSourceCalibrationSpec,
        calibration_text: str,
        *,
        synthesize: Callable[[str, Path], None],
        backend: OpenVoiceBackend,
        force: bool = False,
    ) -> dict[str, Any]:
        text_hash = hashlib.sha256(calibration_text.encode("utf-8")).hexdigest()
        if text_hash != spec.calibration_text_sha256:
            raise ValueError("Calibration text does not match its reviewed SHA-256")
        if backend.pins.revision != spec.openvoice_revision:
            raise ValueError("Calibration and OpenVoice backend revisions differ")
        if not force:
            cached = self.get(spec)
            expected_checkpoints = {
                checkpoint.name: checkpoint.sha256
                for checkpoint in backend.pins.checkpoints
            }
            if (
                cached is not None
                and cached.get("checkpoint_hashes") == expected_checkpoints
            ):
                return cached

        directory = self.directory(spec.azure_voice)
        directory.mkdir(parents=True, exist_ok=True)
        wav_path = directory / "calibration.wav"
        embedding_path = directory / "source_embedding.bin"
        temporary_wav = _temporary_file(directory, ".calibration.", ".wav")
        try:
            synthesize(calibration_text, temporary_wav)
            quality = _validate_calibration_wav(temporary_wav, spec)
            os.replace(temporary_wav, wav_path)
            wav_hash = sha256_file(wav_path)
            embedding_hash = backend.extract_embedding(
                wav_path,
                embedding_path,
                embedding_kind="source",
                expected_audio_sha256=wav_hash,
            )
        except BaseException:
            try:
                temporary_wav.unlink()
            except FileNotFoundError:
                pass
            raise

        manifest = {
            "schema_version": AZURE_CALIBRATION_SCHEMA,
            **spec.identity(),
            "wav_path": str(wav_path),
            "wav_sha256": wav_hash,
            "wav_duration_ms": round(float(quality["duration"]) * 1000),
            "embedding_path": str(embedding_path),
            "embedding_sha256": embedding_hash,
            "checkpoint_hashes": backend.checkpoint_hashes,
            "created_at": _now(),
        }
        atomic_write_json(directory / "manifest.json", manifest)
        return manifest


class OpenVoiceAssetWorker:
    """GPU preflight for deterministic source/target embedding assets.

    The injected :class:`OpenVoiceBackend` is the only GPU boundary.  This module
    intentionally imports neither Torch nor OpenVoice.  Remote download,
    generation-safe promotion, and lease renewal remain with ``GCSStore`` and the
    thin notebooks.

    ``candidate_evaluator`` is optional.  When supplied it must return exactly
    ``speaker_similarity``, ``word_error_approximation``, and
    ``audio_artifact_count``.  Without it, candidate audio is still converted and
    truthfully marked ``pending_external_qc``.
    """

    def __init__(
        self,
        backend: OpenVoiceBackend,
        plan: OpenVoiceAssetPlan,
        manifest_path: Path,
        *,
        lease_guard: Callable[[str], bool | None] | None = None,
        max_attempts_per_asset: int = 2,
        allow_lease_rebind: bool = False,
        candidate_evaluator: Callable[
            [
                OpenVoiceVoiceCandidate,
                Any,
                OpenVoiceSourceAsset,
                OpenVoiceSpeakerAsset,
            ],
            Mapping[str, Any],
        ]
        | None = None,
    ) -> None:
        if plan.openvoice_revision != backend.pins.revision:
            raise ValueError("OpenVoice asset plan and pinned backend revisions differ")
        if max_attempts_per_asset < 1:
            raise ValueError("OpenVoice asset maximum attempts must be positive")
        self.backend = backend
        self.plan = plan
        self.manifest_path = manifest_path
        self.lease_guard = lease_guard
        self.max_attempts_per_asset = max_attempts_per_asset
        self.allow_lease_rebind = allow_lease_rebind
        self.candidate_evaluator = candidate_evaluator

    def prepare_manifest(self) -> dict[str, Any]:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        if self.manifest_path.is_file():
            manifest = read_json(self.manifest_path)
            self._validate_manifest(manifest)
            if manifest.get("lease_identity") != self.plan.lease_identity:
                if not self.allow_lease_rebind or self.lease_guard is None:
                    raise ValueError(
                        "OpenVoice asset manifest belongs to another lease identity"
                    )
                self._assert_lease("asset-lease-rebind")
                manifest.setdefault("lease_history", []).append(
                    {
                        "lease_identity": manifest.get("lease_identity"),
                        "ended_at": _now(),
                    }
                )
                manifest["lease_identity"] = self.plan.lease_identity
                manifest["updated_at"] = _now()
                atomic_write_json(self.manifest_path, manifest)
            return manifest

        manifest = {
            "schema_version": OPENVOICE_ASSET_MANIFEST_SCHEMA,
            "job_id": self.plan.job_id,
            "lease_identity": self.plan.lease_identity,
            "backend": "openvoice",
            "backend_revision": self.plan.openvoice_revision,
            "plan_artifact_sha256": self.plan.artifact_sha256,
            "state": "pending",
            "created_at": _now(),
            "updated_at": _now(),
            "checkpoint_hashes": {},
            "runtime": None,
            "sources": {
                source.azure_voice: {
                    "azure_voice": source.azure_voice,
                    "input_identity_sha256": _canonical_sha256(source.to_dict()),
                    "status": "pending",
                    "attempts": 0,
                    "history": [],
                }
                for source in self.plan.sources
            },
            "speakers": {
                speaker.speaker_id: {
                    "speaker_id": speaker.speaker_id,
                    "input_identity_sha256": _speaker_embedding_identity(speaker),
                    "status": "pending",
                    "attempts": 0,
                    "history": [],
                    "candidate_manifest_path": (
                        str(speaker.candidate_manifest_path)
                        if speaker.candidate_manifest_path is not None
                        else None
                    ),
                    "candidates": {
                        candidate.candidate_id: {
                            "candidate_id": candidate.candidate_id,
                            "azure_voice": candidate.azure_voice,
                            "input_identity_sha256": _canonical_sha256(
                                candidate.to_dict()
                            ),
                            "status": "pending",
                            "attempts": 0,
                            "history": [],
                        }
                        for candidate in speaker.voice_candidates
                    },
                }
                for speaker in self.plan.speakers
            },
        }
        return self._set_state_and_write(manifest)

    def run(self, *, force: bool = False) -> dict[str, Any]:
        manifest = self.prepare_manifest()
        for source in self.plan.sources:
            self._run_source(manifest, source, force=force)
        sources = {item.azure_voice: item for item in self.plan.sources}
        for speaker in self.plan.speakers:
            self._run_speaker(manifest, speaker, force=force)
            for candidate in speaker.voice_candidates:
                self._run_candidate(
                    manifest,
                    speaker,
                    candidate,
                    sources[candidate.azure_voice],
                    force=force,
                )
            if speaker.voice_candidates:
                self._write_candidate_manifest(manifest, speaker)
        return self._set_state_and_write(manifest)

    def pending_asset_ids(self) -> list[str]:
        manifest = self.prepare_manifest()
        pending = []
        for source in self.plan.sources:
            record = manifest["sources"][source.azure_voice]
            if not self._source_completed(record, source):
                pending.append(f"source:{source.azure_voice}")
        for speaker in self.plan.speakers:
            record = manifest["speakers"][speaker.speaker_id]
            if not self._speaker_completed(record, speaker):
                pending.append(f"speaker:{speaker.speaker_id}")
            for candidate in speaker.voice_candidates:
                candidate_record = record["candidates"][candidate.candidate_id]
                if not self._completed_file(
                    candidate_record, candidate.output_path, "output_sha256"
                ):
                    pending.append(f"candidate:{candidate.candidate_id}")
        return pending

    def _run_source(
        self,
        manifest: dict[str, Any],
        source: OpenVoiceSourceAsset,
        *,
        force: bool,
    ) -> None:
        record = manifest["sources"][source.azure_voice]
        embedding_exists = source.source_embedding_path.is_file()
        embedding_manifest_exists = source.source_embedding_manifest_path.is_file()
        if embedding_exists != embedding_manifest_exists:
            error = OpenVoiceAssetHashMismatch(
                "Azure source cache must contain both embedding and manifest"
            )
            self._record_failure(
                manifest, record, int(record.get("attempts", 0)), error
            )
            raise error
        if not force and embedding_exists:
            try:
                result = self._validate_reusable_source(source)
            except OpenVoiceError as exc:
                self._record_failure(
                    manifest, record, int(record.get("attempts", 0)), exc
                )
                raise
            except Exception as exc:
                error = OpenVoiceAssetHashMismatch(
                    "Azure source embedding cache validation failed"
                )
                self._record_failure(
                    manifest, record, int(record.get("attempts", 0)), error
                )
                raise error from exc
            if record.get("status") == "completed" and self._source_completed(
                record, source
            ):
                return
            self._assert_lease(f"source-cache:{source.azure_voice}")
            _archive_asset_record(record)
            record.update(
                {
                    "status": "completed",
                    "attempts": 0,
                    "completed_at": _now(),
                    "result": result,
                }
            )
            record.pop("last_error", None)
            record.pop("failed_at", None)
            manifest["checkpoint_hashes"] = _pinned_checkpoint_hashes(self.backend)
            self._assert_lease(f"source-cache:{source.azure_voice}")
            self._set_state_and_write(manifest)
            return

        def action(_attempt: int) -> dict[str, Any]:
            self._validate_source_input(source)
            embedding_hash = self.backend.extract_embedding(
                source.calibration_audio_path,
                source.source_embedding_path,
                embedding_kind="source",
                expected_audio_sha256=source.calibration_audio_sha256,
            )
            result = {
                "azure_voice": source.azure_voice,
                "calibration_audio_sha256": source.calibration_audio_sha256,
                "calibration_manifest_sha256": source.calibration_manifest_sha256,
                "calibration_identity": source.calibration.identity(),
                "source_embedding_path": str(source.source_embedding_path),
                "source_embedding_sha256": embedding_hash,
                "openvoice_revision": self.plan.openvoice_revision,
                "cache_reused": False,
            }
            embedding_manifest = {
                "schema_version": OPENVOICE_SOURCE_EMBEDDING_MANIFEST_SCHEMA,
                **result,
                "checkpoint_hashes": self.backend.checkpoint_hashes,
                "status": "ready",
            }
            embedding_manifest["artifact_sha256"] = _canonical_sha256(
                embedding_manifest
            )
            atomic_write_json(
                source.source_embedding_manifest_path, embedding_manifest
            )
            result["source_embedding_manifest_path"] = str(
                source.source_embedding_manifest_path
            )
            result["source_embedding_manifest_sha256"] = sha256_file(
                source.source_embedding_manifest_path
            )
            result["source_embedding_manifest_artifact_sha256"] = (
                embedding_manifest["artifact_sha256"]
            )
            result["checkpoint_hashes"] = self.backend.checkpoint_hashes
            return result

        self._execute(manifest, record, f"source:{source.azure_voice}", action)

    def _run_speaker(
        self,
        manifest: dict[str, Any],
        speaker: OpenVoiceSpeakerAsset,
        *,
        force: bool,
    ) -> None:
        record = manifest["speakers"][speaker.speaker_id]
        if not force and self._speaker_completed(record, speaker):
            self._validate_cached_input(
                manifest, record, lambda: self._validate_speaker_input(speaker)
            )
            return

        def action(_attempt: int) -> dict[str, Any]:
            self._validate_speaker_input(speaker)
            embedding_hash = self.backend.extract_embedding(
                speaker.reference_audio_path,
                speaker.target_embedding_path,
                embedding_kind="target",
                expected_audio_sha256=speaker.reference_audio_sha256,
            )
            result = {
                "speaker_id": speaker.speaker_id,
                "reference_audio_sha256": speaker.reference_audio_sha256,
                "reference_manifest_sha256": speaker.reference_manifest_sha256,
                "target_embedding_path": str(speaker.target_embedding_path),
                "target_embedding_sha256": embedding_hash,
                "job_local_identity": True,
                "openvoice_revision": self.plan.openvoice_revision,
            }
            embedding_manifest = {
                "schema_version": OPENVOICE_TARGET_EMBEDDING_MANIFEST_SCHEMA,
                **result,
                "checkpoint_hashes": self.backend.checkpoint_hashes,
                "status": "ready",
            }
            embedding_manifest["artifact_sha256"] = _canonical_sha256(
                embedding_manifest
            )
            atomic_write_json(
                speaker.target_embedding_manifest_path, embedding_manifest
            )
            result["target_embedding_manifest_path"] = str(
                speaker.target_embedding_manifest_path
            )
            result["target_embedding_manifest_sha256"] = sha256_file(
                speaker.target_embedding_manifest_path
            )
            result["target_embedding_manifest_artifact_sha256"] = (
                embedding_manifest["artifact_sha256"]
            )
            return result

        self._execute(manifest, record, f"speaker:{speaker.speaker_id}", action)

    def _run_candidate(
        self,
        manifest: dict[str, Any],
        speaker: OpenVoiceSpeakerAsset,
        candidate: OpenVoiceVoiceCandidate,
        source: OpenVoiceSourceAsset,
        *,
        force: bool,
    ) -> None:
        speaker_record = manifest["speakers"][speaker.speaker_id]
        record = speaker_record["candidates"][candidate.candidate_id]
        if not force and self._completed_file(
            record, candidate.output_path, "output_sha256"
        ):
            self._validate_cached_input(
                manifest, record, lambda: self._validate_candidate_input(candidate)
            )
            return

        def action(attempt: int) -> dict[str, Any]:
            self._validate_candidate_input(candidate)
            source_result = (
                manifest["sources"][source.azure_voice].get("result") or {}
            )
            speaker_result = speaker_record.get("result") or {}
            source_embedding_hash = source_result.get("source_embedding_sha256")
            target_embedding_hash = speaker_result.get("target_embedding_sha256")
            if not source_embedding_hash or not target_embedding_hash:
                raise OpenVoiceAssetHashMismatch(
                    "Voice calibration candidate lacks verified source or target embedding"
                )
            request = OpenVoiceConversionRequest(
                job_id=self.plan.job_id,
                unit_id=candidate.candidate_id,
                speaker_id=speaker.speaker_id,
                source_audio_path=candidate.source_audio_path,
                source_audio_sha256=candidate.source_audio_sha256,
                source_embedding_path=source.source_embedding_path,
                source_embedding_sha256=str(source_embedding_hash),
                target_embedding_path=speaker.target_embedding_path,
                target_embedding_sha256=str(target_embedding_hash),
                target_language=candidate.target_language,
                azure_voice=candidate.azure_voice,
                conversion_settings=candidate.conversion_settings,
                output_path=candidate.output_path,
                lease_identity=self.plan.lease_identity,
                attempt=attempt,
                sample_rate=candidate.sample_rate,
            )
            conversion = self.backend.convert(request)
            evaluation = None
            if self.candidate_evaluator is not None:
                evaluation = _validate_candidate_evaluation(
                    self.candidate_evaluator(candidate, conversion, source, speaker)
                )
            result = {
                **conversion.to_dict(),
                "candidate_id": candidate.candidate_id,
                "voice": candidate.azure_voice,
                "output_path": str(candidate.output_path),
                "calibration_text_sha256": candidate.calibration_text_sha256,
                "duration_drift_ratio": conversion.duration_ratio,
                "audio_artifact_count": (
                    evaluation["audio_artifact_count"]
                    if evaluation is not None
                    else len(conversion.warnings)
                ),
                "evaluation_status": (
                    "completed" if evaluation is not None else "pending_external_qc"
                ),
                "human_review_required": True,
            }
            if evaluation is not None:
                result.update(evaluation)
            return result

        try:
            self._execute(
                manifest,
                record,
                f"candidate:{candidate.candidate_id}",
                action,
            )
        finally:
            self._write_candidate_manifest(manifest, speaker)

    def _execute(
        self,
        manifest: dict[str, Any],
        record: dict[str, Any],
        asset_id: str,
        action: Callable[[int], dict[str, Any]],
    ) -> dict[str, Any]:
        attempts = int(record.get("attempts", 0))
        if attempts >= self.max_attempts_per_asset:
            error = OpenVoiceAttemptLimitExceeded(
                f"OpenVoice attempt limit reached for asset {asset_id}"
            )
            record["last_error"] = error.as_record()
            self._set_state_and_write(manifest)
            raise error
        attempt = attempts + 1
        try:
            self._assert_lease(asset_id)
            result = action(attempt)
            self._assert_lease(asset_id)
        except OpenVoiceError as exc:
            self._record_failure(manifest, record, attempt, exc)
            raise
        except Exception as exc:
            safe = OpenVoiceInferenceError(
                f"Unexpected OpenVoice asset failure for {asset_id} "
                f"({type(exc).__name__})"
            )
            self._record_failure(manifest, record, attempt, safe)
            raise safe from exc
        _archive_asset_record(record)
        record.update(
            {
                "status": "completed",
                "attempts": attempt,
                "completed_at": _now(),
                "result": result,
            }
        )
        record.pop("last_error", None)
        record.pop("failed_at", None)
        self._record_runtime(manifest)
        self._set_state_and_write(manifest)
        return result

    def _record_failure(
        self,
        manifest: dict[str, Any],
        record: dict[str, Any],
        attempt: int,
        error: OpenVoiceError,
    ) -> None:
        _archive_asset_record(record)
        record.update(
            {
                "status": "failed",
                "attempts": attempt,
                "failed_at": _now(),
                "last_error": error.as_record(),
            }
        )
        record.pop("completed_at", None)
        record.pop("result", None)
        self._set_state_and_write(manifest)

    def _validate_cached_input(
        self,
        manifest: dict[str, Any],
        record: dict[str, Any],
        validation: Callable[[], None],
    ) -> None:
        try:
            validation()
        except OpenVoiceError as exc:
            self._record_failure(
                manifest,
                record,
                max(1, int(record.get("attempts", 0))),
                exc,
            )
            raise
        except Exception as exc:
            safe = OpenVoiceInferenceError(
                f"Unexpected cached OpenVoice asset validation failure "
                f"({type(exc).__name__})"
            )
            self._record_failure(
                manifest,
                record,
                max(1, int(record.get("attempts", 0))),
                safe,
            )
            raise safe from exc

    def _record_runtime(self, manifest: dict[str, Any]) -> None:
        runtime = self.backend.runtime_info
        manifest["runtime"] = (
            {
                "openvoice_revision": runtime.openvoice_revision,
                "python_version": runtime.python_version,
                "torch_version": runtime.torch_version,
                "cuda_available": runtime.cuda_available,
                "cuda_version": runtime.cuda_version,
                "gpu_name": runtime.gpu_name,
                "device": self.backend.device,
                "precision": self.backend.precision,
            }
            if runtime is not None
            else None
        )
        manifest["checkpoint_hashes"] = self.backend.checkpoint_hashes

    def _validate_source_input(self, source: OpenVoiceSourceAsset) -> None:
        _verify_planned_file(
            "Azure calibration manifest",
            source.calibration_manifest_path,
            source.calibration_manifest_sha256,
        )
        _verify_planned_file(
            "Azure calibration audio",
            source.calibration_audio_path,
            source.calibration_audio_sha256,
        )
        calibration_manifest = read_json(source.calibration_manifest_path)
        expected = {
            "schema_version": AZURE_CALIBRATION_SCHEMA,
            **source.calibration.identity(),
            "wav_sha256": source.calibration_audio_sha256,
        }
        if any(calibration_manifest.get(key) != value for key, value in expected.items()):
            raise OpenVoiceAssetHashMismatch(
                "Azure calibration identity does not match the asset plan"
            )
        try:
            _validate_calibration_wav(source.calibration_audio_path, source.calibration)
        except ValueError as exc:
            raise OpenVoiceAssetHashMismatch(
                "Azure calibration audio does not match its reviewed identity"
            ) from exc

    def _validate_reusable_source(
        self, source: OpenVoiceSourceAsset
    ) -> dict[str, Any]:
        self._validate_source_input(source)
        if not source.source_embedding_path.is_file() or not source.source_embedding_manifest_path.is_file():
            raise OpenVoiceAssetHashMismatch(
                "Azure source cache must contain both embedding and manifest"
            )
        embedding_manifest = read_json(source.source_embedding_manifest_path)
        allowed_keys = {
            "schema_version",
            "azure_voice",
            "calibration_audio_sha256",
            "calibration_manifest_sha256",
            "calibration_identity",
            "source_embedding_path",
            "source_embedding_sha256",
            "openvoice_revision",
            "cache_reused",
            "checkpoint_hashes",
            "status",
            "artifact_sha256",
        }
        if set(embedding_manifest) != allowed_keys:
            raise OpenVoiceAssetHashMismatch(
                "Azure source embedding manifest has unexpected fields"
            )
        artifact_sha256 = str(embedding_manifest.get("artifact_sha256", ""))
        _require_sha256("source embedding manifest artifact", artifact_sha256)
        canonical_payload = {
            key: value
            for key, value in embedding_manifest.items()
            if key != "artifact_sha256"
        }
        if _canonical_sha256(canonical_payload) != artifact_sha256:
            raise OpenVoiceAssetHashMismatch(
                "Azure source embedding manifest artifact hash does not match"
            )
        expected_checkpoints = _pinned_checkpoint_hashes(self.backend)
        expected_identity = {
            "schema_version": OPENVOICE_SOURCE_EMBEDDING_MANIFEST_SCHEMA,
            "azure_voice": source.azure_voice,
            "calibration_audio_sha256": source.calibration_audio_sha256,
            "calibration_manifest_sha256": source.calibration_manifest_sha256,
            "calibration_identity": source.calibration.identity(),
            "openvoice_revision": self.plan.openvoice_revision,
            "checkpoint_hashes": expected_checkpoints,
            "status": "ready",
        }
        if any(
            embedding_manifest.get(key) != value
            for key, value in expected_identity.items()
        ):
            raise OpenVoiceAssetHashMismatch(
                "Azure source embedding cache identity does not match immutable pins"
            )
        embedding_sha256 = str(
            embedding_manifest.get("source_embedding_sha256", "")
        )
        _verify_planned_file(
            "cached Azure source embedding",
            source.source_embedding_path,
            embedding_sha256,
        )
        return {
            "azure_voice": source.azure_voice,
            "calibration_audio_sha256": source.calibration_audio_sha256,
            "calibration_manifest_sha256": source.calibration_manifest_sha256,
            "calibration_identity": source.calibration.identity(),
            "source_embedding_path": str(source.source_embedding_path),
            "source_embedding_sha256": embedding_sha256,
            "source_embedding_manifest_path": str(
                source.source_embedding_manifest_path
            ),
            "source_embedding_manifest_sha256": sha256_file(
                source.source_embedding_manifest_path
            ),
            "source_embedding_manifest_artifact_sha256": artifact_sha256,
            "openvoice_revision": self.plan.openvoice_revision,
            "checkpoint_hashes": expected_checkpoints,
            "cache_reused": True,
        }

    def _validate_speaker_input(self, speaker: OpenVoiceSpeakerAsset) -> None:
        _verify_planned_file(
            "speaker reference manifest",
            speaker.reference_manifest_path,
            speaker.reference_manifest_sha256,
        )
        _verify_planned_file(
            "speaker reference audio",
            speaker.reference_audio_path,
            speaker.reference_audio_sha256,
        )
        reference_manifest = read_json(speaker.reference_manifest_path)
        if (
            reference_manifest.get("schema_version")
            != "speaker-reference-manifest-v1"
            or reference_manifest.get("speaker_id") != speaker.speaker_id
            or reference_manifest.get("reference_reel_sha256")
            != speaker.reference_audio_sha256
            or reference_manifest.get("job_local_identity") is not True
            or reference_manifest.get("global_identity_created") is not False
        ):
            raise OpenVoiceAssetHashMismatch(
                "Speaker reference identity does not match the asset plan"
            )
        quality = inspect_wav(speaker.reference_audio_path)
        if (
            not quality.get("valid")
            or quality.get("channels") != 1
            or quality.get("sample_rate") != speaker.sample_rate
        ):
            raise OpenVoiceAssetHashMismatch(
                "Speaker reference is not canonical same-speaker PCM WAV audio"
            )

    @staticmethod
    def _validate_candidate_input(candidate: OpenVoiceVoiceCandidate) -> None:
        _verify_planned_file(
            "Azure voice candidate audio",
            candidate.source_audio_path,
            candidate.source_audio_sha256,
        )
        quality = inspect_wav(candidate.source_audio_path)
        if (
            not quality.get("valid")
            or quality.get("channels") != 1
            or quality.get("sample_rate") != candidate.sample_rate
        ):
            raise OpenVoiceAssetHashMismatch(
                "Azure voice candidate is not canonical mono PCM WAV audio"
            )

    def _write_candidate_manifest(
        self,
        manifest: dict[str, Any],
        speaker: OpenVoiceSpeakerAsset,
    ) -> dict[str, Any]:
        if speaker.candidate_manifest_path is None:
            return {}
        speaker_record = manifest["speakers"][speaker.speaker_id]
        values = []
        for candidate in speaker.voice_candidates:
            record = speaker_record["candidates"][candidate.candidate_id]
            if record.get("status") == "completed":
                values.append(dict(record["result"]))
            else:
                values.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "voice": candidate.azure_voice,
                        "status": record.get("status", "pending"),
                        "attempts": int(record.get("attempts", 0)),
                        "error_code": (record.get("last_error") or {}).get("code"),
                        "human_review_required": True,
                    }
                )
        state = (
            "completed"
            if values and all(item.get("backend") == "openvoice" for item in values)
            else "partial"
            if any(item.get("backend") == "openvoice" for item in values)
            else "pending"
        )
        value = {
            "schema_version": OPENVOICE_VOICE_CANDIDATES_SCHEMA,
            "job_id": self.plan.job_id,
            "speaker_id": speaker.speaker_id,
            "openvoice_revision": self.plan.openvoice_revision,
            "reference_audio_sha256": speaker.reference_audio_sha256,
            "target_embedding_sha256": (
                (speaker_record.get("result") or {}).get("target_embedding_sha256")
            ),
            "state": state,
            "candidates": values,
            "thresholds_used": False,
            "human_review_required": True,
        }
        value["artifact_sha256"] = _canonical_sha256(value)
        speaker.candidate_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(speaker.candidate_manifest_path, value)
        speaker_record["candidate_manifest_sha256"] = value["artifact_sha256"]
        return value

    def _set_state_and_write(self, manifest: dict[str, Any]) -> dict[str, Any]:
        records = list(manifest["sources"].values())
        for speaker in manifest["speakers"].values():
            records.append(speaker)
            records.extend(speaker["candidates"].values())
        statuses = [record.get("status") for record in records]
        if statuses and all(status == "completed" for status in statuses):
            state = "completed"
        elif any(status == "failed" for status in statuses):
            state = (
                "partial" if any(status == "completed" for status in statuses) else "failed"
            )
        elif any(status == "completed" for status in statuses):
            state = "partial"
        else:
            state = "pending"
        manifest["state"] = state
        manifest["completed_assets"] = sum(
            status == "completed" for status in statuses
        )
        manifest["pending_assets"] = sum(
            status != "completed" for status in statuses
        )
        manifest["updated_at"] = _now()
        atomic_write_json(self.manifest_path, manifest)
        return manifest

    def _validate_manifest(self, manifest: Mapping[str, Any]) -> None:
        expected = {
            "schema_version": OPENVOICE_ASSET_MANIFEST_SCHEMA,
            "job_id": self.plan.job_id,
            "backend_revision": self.plan.openvoice_revision,
            "plan_artifact_sha256": self.plan.artifact_sha256,
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError(f"OpenVoice asset manifest has incompatible {key}")
        if set(manifest.get("sources", {})) != {
            item.azure_voice for item in self.plan.sources
        }:
            raise ValueError("OpenVoice asset manifest source set changed")
        if set(manifest.get("speakers", {})) != {
            item.speaker_id for item in self.plan.speakers
        }:
            raise ValueError("OpenVoice asset manifest speaker set changed")
        for source in self.plan.sources:
            if (
                manifest["sources"][source.azure_voice].get("input_identity_sha256")
                != _canonical_sha256(source.to_dict())
            ):
                raise ValueError("OpenVoice source asset inputs changed")
        for speaker in self.plan.speakers:
            record = manifest["speakers"][speaker.speaker_id]
            if record.get("input_identity_sha256") != _speaker_embedding_identity(
                speaker
            ):
                raise ValueError("OpenVoice speaker asset inputs changed")
            if set(record.get("candidates", {})) != {
                item.candidate_id for item in speaker.voice_candidates
            }:
                raise ValueError("OpenVoice speaker candidate set changed")
            for candidate in speaker.voice_candidates:
                if (
                    record["candidates"][candidate.candidate_id].get(
                        "input_identity_sha256"
                    )
                    != _canonical_sha256(candidate.to_dict())
                ):
                    raise ValueError("OpenVoice speaker candidate inputs changed")

    def _assert_lease(self, asset_id: str) -> None:
        if self.lease_guard is None:
            return
        try:
            owned = self.lease_guard(asset_id)
        except OpenVoiceLeaseLost:
            raise
        except Exception as exc:
            raise OpenVoiceLeaseLost(
                f"Lease ownership could not be confirmed for asset {asset_id}"
            ) from exc
        if owned is False:
            raise OpenVoiceLeaseLost(
                f"Lease ownership was lost before asset {asset_id}"
            )

    @staticmethod
    def _completed_file(
        record: Mapping[str, Any], path: Path, hash_field: str
    ) -> bool:
        if record.get("status") != "completed" or not path.is_file():
            return False
        expected = (record.get("result") or {}).get(hash_field)
        return bool(expected) and sha256_file(path) == expected

    @classmethod
    def _source_completed(
        cls, record: Mapping[str, Any], source: OpenVoiceSourceAsset
    ) -> bool:
        return cls._completed_file(
            record, source.source_embedding_path, "source_embedding_sha256"
        ) and cls._completed_file(
            record,
            source.source_embedding_manifest_path,
            "source_embedding_manifest_sha256",
        )

    @classmethod
    def _speaker_completed(
        cls, record: Mapping[str, Any], speaker: OpenVoiceSpeakerAsset
    ) -> bool:
        return cls._completed_file(
            record, speaker.target_embedding_path, "target_embedding_sha256"
        ) and cls._completed_file(
            record,
            speaker.target_embedding_manifest_path,
            "target_embedding_manifest_sha256",
        )


def _temporary_file(directory: Path, prefix: str, suffix: str) -> Path:
    fd, name = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=directory)
    os.close(fd)
    return Path(name)


def _unit_identity_sha256(unit: OpenVoiceUnit) -> str:
    value = {
        "unit_id": unit.unit_id,
        "speaker_id": unit.speaker_id,
        "azure_voice": unit.azure_voice,
        "target_language": unit.target_language,
        "source_audio_sha256": unit.source_audio_sha256,
        "source_embedding_sha256": unit.source_embedding_sha256,
        "target_embedding_sha256": unit.target_embedding_sha256,
        "conversion_settings": unit.conversion_settings,
        "sample_rate": unit.sample_rate,
    }
    try:
        encoded = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("OpenVoice conversion settings must be deterministic JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def _canonical_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("OpenVoice asset data must be deterministic JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(label: str, value: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"Invalid {label} SHA-256")


def _verify_planned_file(label: str, path: Path, expected_sha256: str) -> None:
    _require_sha256(label, expected_sha256)
    if not path.is_file():
        raise OpenVoiceAssetHashMismatch(f"Planned {label} is missing")
    if sha256_file(path) != expected_sha256:
        raise OpenVoiceAssetHashMismatch(f"Planned {label} hash does not match")


def _speaker_embedding_identity(speaker: OpenVoiceSpeakerAsset) -> str:
    return _canonical_sha256(
        {
            "speaker_id": speaker.speaker_id,
            "reference_audio_path": str(speaker.reference_audio_path),
            "reference_audio_sha256": speaker.reference_audio_sha256,
            "reference_manifest_path": str(speaker.reference_manifest_path),
            "reference_manifest_sha256": speaker.reference_manifest_sha256,
            "target_embedding_path": str(speaker.target_embedding_path),
            "target_embedding_manifest_path": str(
                speaker.target_embedding_manifest_path
            ),
            "sample_rate": speaker.sample_rate,
        }
    )


def _pinned_checkpoint_hashes(backend: OpenVoiceBackend) -> dict[str, str]:
    return {
        checkpoint.name: checkpoint.sha256
        for checkpoint in backend.pins.checkpoints
    }


def _archive_asset_record(record: dict[str, Any]) -> None:
    if int(record.get("attempts", 0)) < 1:
        return
    snapshot = {
        key: value
        for key, value in record.items()
        if key
        not in {
            "history",
            "azure_voice",
            "speaker_id",
            "candidate_id",
            "candidates",
            "candidate_manifest_path",
            "candidate_manifest_sha256",
        }
    }
    record.setdefault("history", []).append(snapshot)


def _validate_candidate_evaluation(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("Candidate evaluator must return a mapping")
    required = {
        "speaker_similarity",
        "word_error_approximation",
        "audio_artifact_count",
    }
    if set(value) != required:
        raise ValueError(
            "Candidate evaluator must return only speaker similarity, word error, "
            "and audio artefact count"
        )
    similarity = float(value["speaker_similarity"])
    word_error = float(value["word_error_approximation"])
    artifact_count = int(value["audio_artifact_count"])
    if (
        not math.isfinite(similarity)
        or not math.isfinite(word_error)
        or word_error < 0
        or artifact_count < 0
    ):
        raise ValueError("Candidate evaluator returned invalid finite metrics")
    return {
        "speaker_similarity": similarity,
        "word_error_approximation": word_error,
        "audio_artifact_count": artifact_count,
    }


def _validate_calibration_wav(
    path: Path, spec: AzureSourceCalibrationSpec
) -> dict[str, Any]:
    try:
        with wave.open(str(path), "rb") as audio:
            channels = audio.getnchannels()
            sample_width = audio.getsampwidth()
            sample_rate = audio.getframerate()
        quality = inspect_wav(path)
    except Exception as exc:
        raise ValueError("Azure calibration did not produce a readable PCM WAV") from exc
    if (
        not quality.get("valid")
        or channels != spec.channels
        or sample_width != spec.sample_width
        or sample_rate != spec.sample_rate
    ):
        raise ValueError("Azure calibration WAV does not match the canonical format")
    duration = float(quality["duration"])
    if not spec.minimum_duration_seconds <= duration <= spec.maximum_duration_seconds:
        raise ValueError("Azure calibration WAV is outside the reviewed duration bounds")
    return quality
