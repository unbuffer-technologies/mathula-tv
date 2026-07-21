from __future__ import annotations

import re
import wave
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from .atomic_io import atomic_copy, atomic_write_json, atomic_write_text
from .logging_utils import redact
from .openvoice import sha256_file


COMPATIBILITY_REPORT_SCHEMA = "azure-openvoice-compatibility-report-v1"
REVIEW_PACKAGE_SCHEMA = "azure-openvoice-ab-review-package-v1"
REQUIRED_HUMAN_REVIEWS = (
    "voice_rights",
    "editorial",
    "factual",
    "isizulu_language",
)
HUMAN_REVIEW_STATUSES = {"pending", "approved", "changes_requested", "rejected"}
_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

__all__ = [
    "CompatibilityGateState",
    "CompatibilityThresholds",
    "REQUIRED_HUMAN_REVIEWS",
    "ReviewPackageUnit",
    "UnitCompatibilityObservation",
    "build_ab_review_package",
    "compatibility_report_markdown",
    "evaluate_compatibility",
    "write_compatibility_report",
]


class CompatibilityGateState(str, Enum):
    PENDING = "pending"
    PASSED = "passed"
    FAILED_PRONUNCIATION = "failed_pronunciation"
    FAILED_INTELLIGIBILITY = "failed_intelligibility"
    FAILED_IDENTITY = "failed_identity"
    FAILED_TIMING = "failed_timing"
    FAILED_AUDIO_QUALITY = "failed_audio_quality"
    NEEDS_HUMAN_REVIEW = "needs_human_review"


@dataclass(frozen=True)
class CompatibilityThresholds:
    """Project-calibrated thresholds; no defaults imply universal validity."""

    version: str
    calibrated: bool
    max_openvoice_word_error_rate: float | None
    max_word_error_rate_degradation: float | None
    min_protected_name_accuracy: float | None
    min_number_accuracy: float | None
    min_date_accuracy: float | None
    min_negation_accuracy: float | None
    min_speaker_similarity: float | None
    max_abs_timing_error_ms: int | None
    max_clipping_ratio: float | None
    min_silence_ratio: float | None
    max_silence_ratio: float | None

    def missing_values(self) -> list[str]:
        return [
            name
            for name, value in asdict(self).items()
            if name not in {"version", "calibrated"} and value is None
        ]

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise ValueError("Compatibility threshold version is required")
        bounded_ratios = (
            self.min_protected_name_accuracy,
            self.min_number_accuracy,
            self.min_date_accuracy,
            self.min_negation_accuracy,
            self.max_clipping_ratio,
            self.min_silence_ratio,
            self.max_silence_ratio,
        )
        if any(value is not None and not 0 <= value <= 1 for value in bounded_ratios):
            raise ValueError("Compatibility ratio thresholds must be between zero and one")
        if any(
            value is not None and value < 0
            for value in (
                self.max_openvoice_word_error_rate,
                self.max_word_error_rate_degradation,
            )
        ):
            raise ValueError("Compatibility error thresholds cannot be negative")
        if (
            self.min_speaker_similarity is not None
            and not -1 <= self.min_speaker_similarity <= 1
        ):
            raise ValueError("Speaker-similarity thresholds must be between -1 and 1")
        if self.max_abs_timing_error_ms is not None and self.max_abs_timing_error_ms < 0:
            raise ValueError("Compatibility timing threshold cannot be negative")
        if (
            self.min_silence_ratio is not None
            and self.max_silence_ratio is not None
            and self.min_silence_ratio > self.max_silence_ratio
        ):
            raise ValueError("Compatibility silence thresholds are inverted")


@dataclass(frozen=True)
class UnitCompatibilityObservation:
    unit_id: str
    speaker_id: str
    intended_spoken_text: str
    reference_audio_sha256: str | None
    azure_audio_sha256: str | None
    openvoice_audio_sha256: str | None
    azure_duration_ms: int | None
    openvoice_duration_ms: int | None
    final_timing_error_ms: int | None
    azure_retranscription: str | None
    openvoice_retranscription: str | None
    azure_word_error_rate: float | None
    openvoice_word_error_rate: float | None
    protected_name_accuracy: float | None
    number_accuracy: float | None
    date_accuracy: float | None
    negation_accuracy: float | None
    speaker_similarity: float | None
    azure_speaker_similarity: float | None
    audio_quality_passed: bool | None
    clipping_ratio: float | None
    silence_ratio: float | None
    protected_name_count: int = 0
    number_count: int = 0
    date_count: int = 0
    negation_count: int = 0
    audio_artifact_warnings: tuple[str, ...] = ()
    human_reviews: Mapping[str, str | Mapping[str, Any]] = field(default_factory=dict)
    aligned_audio_sha256: str | None = None

    def __post_init__(self) -> None:
        if (
            not _SAFE_IDENTIFIER.fullmatch(self.unit_id)
            or not _SAFE_IDENTIFIER.fullmatch(self.speaker_id)
            or not self.intended_spoken_text.strip()
        ):
            raise ValueError("Compatibility observations require unit, speaker, and spoken text")
        for value in (
            self.reference_audio_sha256,
            self.azure_audio_sha256,
            self.openvoice_audio_sha256,
            self.aligned_audio_sha256,
        ):
            if value is not None and not _SHA256.fullmatch(value):
                raise ValueError("Compatibility audio hashes must be lowercase SHA-256")
        for count in (
            self.protected_name_count,
            self.number_count,
            self.date_count,
            self.negation_count,
        ):
            if count < 0:
                raise ValueError("Protected item counts cannot be negative")
        bounded_ratios = (
            self.protected_name_accuracy,
            self.number_accuracy,
            self.date_accuracy,
            self.negation_accuracy,
            self.clipping_ratio,
            self.silence_ratio,
        )
        if any(value is not None and not 0 <= value <= 1 for value in bounded_ratios):
            raise ValueError("Compatibility observation ratios must be between zero and one")
        if any(
            value is not None and value < 0
            for value in (self.azure_word_error_rate, self.openvoice_word_error_rate)
        ):
            raise ValueError("Compatibility word-error rates cannot be negative")
        if any(
            value is not None and not -1 <= value <= 1
            for value in (self.speaker_similarity, self.azure_speaker_similarity)
        ):
            raise ValueError("Speaker-similarity observations must be between -1 and 1")
        if any(
            value is not None and value <= 0
            for value in (self.azure_duration_ms, self.openvoice_duration_ms)
        ):
            raise ValueError("Compatibility durations must be positive")
        for role, decision in self.human_reviews.items():
            if role not in REQUIRED_HUMAN_REVIEWS:
                raise ValueError(f"Unknown compatibility human-review role: {role}")
            status = _human_review_status(decision)
            if status not in HUMAN_REVIEW_STATUSES:
                raise ValueError(f"Unknown compatibility human-review status: {status}")
            if status == "approved":
                if not isinstance(decision, Mapping):
                    raise ValueError(
                        f"Approved {role} review requires reviewer and timestamp evidence"
                    )
                reviewer = str(decision.get("reviewer") or "").strip()
                reviewed_at = str(decision.get("reviewed_at") or "").strip()
                if not reviewer or not _is_timezone_aware_timestamp(reviewed_at):
                    raise ValueError(
                        f"Approved {role} review requires reviewer and timestamp evidence"
                    )


def evaluate_compatibility(
    job_id: str,
    observations: Sequence[UnitCompatibilityObservation],
    thresholds: CompatibilityThresholds,
) -> dict[str, Any]:
    if not _SAFE_IDENTIFIER.fullmatch(job_id):
        raise ValueError("Compatibility report job ID is required")
    if not observations:
        raise ValueError("Compatibility report requires at least one unit")
    unit_ids = [item.unit_id for item in observations]
    if len(unit_ids) != len(set(unit_ids)):
        raise ValueError("Compatibility report unit IDs must be unique")

    missing_thresholds = thresholds.missing_values()
    results = [
        _evaluate_unit(item, thresholds, thresholds_missing=bool(missing_thresholds))
        for item in observations
    ]
    state = _aggregate_state([CompatibilityGateState(item["state"]) for item in results])
    warnings = ["Authorised human review is mandatory before publication."]
    if missing_thresholds:
        warnings.append(
            "Compatibility thresholds are incomplete; the gate cannot pass."
        )
    if not thresholds.calibrated:
        warnings.append(
            "Compatibility thresholds are provisional and uncalibrated; "
            "automatic pass is disabled."
        )
    return {
        "schema_version": COMPATIBILITY_REPORT_SCHEMA,
        "job_id": job_id,
        "state": state.value,
        "gate_state": state.value,
        "production_ready": False,
        "thresholds": asdict(thresholds),
        "thresholds_provisional": not thresholds.calibrated,
        "missing_thresholds": missing_thresholds,
        "required_human_reviews": list(REQUIRED_HUMAN_REVIEWS),
        "human_review_required": True,
        "summary": {
            "unit_count": len(results),
            "passed_units": sum(item["state"] == "passed" for item in results),
            "pending_units": sum(item["state"] == "pending" for item in results),
            "needs_human_review_units": sum(
                item["state"] == "needs_human_review" for item in results
            ),
            "failed_units": sum(item["state"].startswith("failed_") for item in results),
        },
        "units": results,
        "warnings": warnings,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _evaluate_unit(
    item: UnitCompatibilityObservation,
    thresholds: CompatibilityThresholds,
    *,
    thresholds_missing: bool,
) -> dict[str, Any]:
    missing = []
    required_values = {
        "reference_audio_sha256": item.reference_audio_sha256,
        "azure_audio_sha256": item.azure_audio_sha256,
        "openvoice_audio_sha256": item.openvoice_audio_sha256,
        "azure_duration_ms": item.azure_duration_ms,
        "openvoice_duration_ms": item.openvoice_duration_ms,
        "final_timing_error_ms": item.final_timing_error_ms,
        "azure_retranscription": item.azure_retranscription,
        "openvoice_retranscription": item.openvoice_retranscription,
        "azure_word_error_rate": item.azure_word_error_rate,
        "openvoice_word_error_rate": item.openvoice_word_error_rate,
        "speaker_similarity": item.speaker_similarity,
        "azure_speaker_similarity": item.azure_speaker_similarity,
        "audio_quality_passed": item.audio_quality_passed,
        "clipping_ratio": item.clipping_ratio,
        "silence_ratio": item.silence_ratio,
    }
    for name, value in required_values.items():
        if value is None:
            missing.append(name)
    protected_metrics = (
        ("protected_name_accuracy", item.protected_name_count, item.protected_name_accuracy),
        ("number_accuracy", item.number_count, item.number_accuracy),
        ("date_accuracy", item.date_count, item.date_accuracy),
        ("negation_accuracy", item.negation_count, item.negation_accuracy),
    )
    for name, count, value in protected_metrics:
        if count and value is None:
            missing.append(name)

    failures: dict[CompatibilityGateState, list[str]] = {
        CompatibilityGateState.FAILED_PRONUNCIATION: [],
        CompatibilityGateState.FAILED_INTELLIGIBILITY: [],
        CompatibilityGateState.FAILED_IDENTITY: [],
        CompatibilityGateState.FAILED_TIMING: [],
        CompatibilityGateState.FAILED_AUDIO_QUALITY: [],
    }
    _compare_minimum(
        failures[CompatibilityGateState.FAILED_PRONUNCIATION],
        "protected-name accuracy",
        item.protected_name_accuracy,
        thresholds.min_protected_name_accuracy,
        applies=item.protected_name_count > 0,
    )
    _compare_minimum(
        failures[CompatibilityGateState.FAILED_PRONUNCIATION],
        "number accuracy",
        item.number_accuracy,
        thresholds.min_number_accuracy,
        applies=item.number_count > 0,
    )
    _compare_minimum(
        failures[CompatibilityGateState.FAILED_PRONUNCIATION],
        "date accuracy",
        item.date_accuracy,
        thresholds.min_date_accuracy,
        applies=item.date_count > 0,
    )
    _compare_minimum(
        failures[CompatibilityGateState.FAILED_PRONUNCIATION],
        "negation accuracy",
        item.negation_accuracy,
        thresholds.min_negation_accuracy,
        applies=item.negation_count > 0,
    )
    if (
        item.openvoice_word_error_rate is not None
        and thresholds.max_openvoice_word_error_rate is not None
        and item.openvoice_word_error_rate > thresholds.max_openvoice_word_error_rate
    ):
        failures[CompatibilityGateState.FAILED_INTELLIGIBILITY].append(
            "OpenVoice retranscription error exceeds the calibrated maximum"
        )
    if (
        item.azure_word_error_rate is not None
        and item.openvoice_word_error_rate is not None
        and thresholds.max_word_error_rate_degradation is not None
        and item.openvoice_word_error_rate - item.azure_word_error_rate
        > thresholds.max_word_error_rate_degradation
    ):
        failures[CompatibilityGateState.FAILED_INTELLIGIBILITY].append(
            "OpenVoice degraded retranscription beyond the calibrated Azure baseline"
        )
    _compare_minimum(
        failures[CompatibilityGateState.FAILED_IDENTITY],
        "speaker similarity",
        item.speaker_similarity,
        thresholds.min_speaker_similarity,
        applies=True,
    )
    if (
        item.final_timing_error_ms is not None
        and thresholds.max_abs_timing_error_ms is not None
        and abs(item.final_timing_error_ms) > thresholds.max_abs_timing_error_ms
    ):
        failures[CompatibilityGateState.FAILED_TIMING].append(
            "Final timing error exceeds the calibrated tolerance"
        )
    if item.audio_quality_passed is False:
        failures[CompatibilityGateState.FAILED_AUDIO_QUALITY].append(
            "Automated audio-quality inspection failed"
        )
    if (
        item.clipping_ratio is not None
        and thresholds.max_clipping_ratio is not None
        and item.clipping_ratio > thresholds.max_clipping_ratio
    ):
        failures[CompatibilityGateState.FAILED_AUDIO_QUALITY].append(
            "Clipping exceeds the calibrated maximum"
        )
    if (
        item.silence_ratio is not None
        and thresholds.min_silence_ratio is not None
        and item.silence_ratio < thresholds.min_silence_ratio
    ):
        failures[CompatibilityGateState.FAILED_AUDIO_QUALITY].append(
            "Silence ratio is below the calibrated range"
        )
    if (
        item.silence_ratio is not None
        and thresholds.max_silence_ratio is not None
        and item.silence_ratio > thresholds.max_silence_ratio
    ):
        failures[CompatibilityGateState.FAILED_AUDIO_QUALITY].append(
            "Silence ratio is above the calibrated range"
        )

    state = CompatibilityGateState.PENDING
    failure_reasons = []
    for candidate in (
        CompatibilityGateState.FAILED_PRONUNCIATION,
        CompatibilityGateState.FAILED_INTELLIGIBILITY,
        CompatibilityGateState.FAILED_IDENTITY,
        CompatibilityGateState.FAILED_TIMING,
        CompatibilityGateState.FAILED_AUDIO_QUALITY,
    ):
        if failures[candidate]:
            state = candidate
            failure_reasons = failures[candidate]
            break
    else:
        human_pending = [
            role
            for role in REQUIRED_HUMAN_REVIEWS
            if _human_review_status(item.human_reviews.get(role)) != "approved"
        ]
        if missing or thresholds_missing:
            state = CompatibilityGateState.PENDING
        elif not thresholds.calibrated or human_pending:
            state = CompatibilityGateState.NEEDS_HUMAN_REVIEW
        else:
            state = CompatibilityGateState.PASSED

    duration_ratio = None
    if item.azure_duration_ms and item.openvoice_duration_ms is not None:
        duration_ratio = round(item.openvoice_duration_ms / item.azure_duration_ms, 6)
    return {
        "unit_id": item.unit_id,
        "speaker_id": item.speaker_id,
        "intended_spoken_text": item.intended_spoken_text,
        "state": state.value,
        "reference_audio_sha256": item.reference_audio_sha256,
        "azure_audio_sha256": item.azure_audio_sha256,
        "openvoice_audio_sha256": item.openvoice_audio_sha256,
        "aligned_audio_sha256": item.aligned_audio_sha256,
        "azure_duration_ms": item.azure_duration_ms,
        "openvoice_duration_ms": item.openvoice_duration_ms,
        "duration_ratio": duration_ratio,
        "final_timing_error_ms": item.final_timing_error_ms,
        "azure_retranscription": item.azure_retranscription,
        "openvoice_retranscription": item.openvoice_retranscription,
        "azure_word_error_rate": item.azure_word_error_rate,
        "openvoice_word_error_rate": item.openvoice_word_error_rate,
        "protected_terms": {
            "name_count": item.protected_name_count,
            "name_accuracy": item.protected_name_accuracy,
            "number_count": item.number_count,
            "number_accuracy": item.number_accuracy,
            "date_count": item.date_count,
            "date_accuracy": item.date_accuracy,
            "negation_count": item.negation_count,
            "negation_accuracy": item.negation_accuracy,
        },
        "similarity": {
            "reference_vs_azure": item.azure_speaker_similarity,
            "reference_vs_openvoice": item.speaker_similarity,
        },
        "audio_quality": {
            "passed": item.audio_quality_passed,
            "clipping_ratio": item.clipping_ratio,
            "silence_ratio": item.silence_ratio,
            "artifact_warnings": redact(list(item.audio_artifact_warnings)),
        },
        "human_reviews": _normalise_human_reviews(item.human_reviews),
        "missing_measurements": missing,
        "failure_reasons": failure_reasons,
    }


def _compare_minimum(
    failures: list[str],
    label: str,
    value: float | None,
    minimum: float | None,
    *,
    applies: bool,
) -> None:
    if applies and value is not None and minimum is not None and value < minimum:
        failures.append(f"{label.capitalize()} is below the calibrated minimum")


def _human_review_status(decision: Any) -> str | None:
    if isinstance(decision, Mapping):
        value = decision.get("status")
        return str(value) if value is not None else None
    return str(decision) if decision is not None else None


def _normalise_human_reviews(
    decisions: Mapping[str, str | Mapping[str, Any]],
) -> dict[str, Any]:
    result = {}
    for role in REQUIRED_HUMAN_REVIEWS:
        decision = decisions.get(role, "pending")
        if isinstance(decision, Mapping):
            result[role] = redact(dict(decision))
        else:
            result[role] = {"status": str(decision)}
    return result


def _is_timezone_aware_timestamp(value: str) -> bool:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _aggregate_state(states: Sequence[CompatibilityGateState]) -> CompatibilityGateState:
    for failure in (
        CompatibilityGateState.FAILED_PRONUNCIATION,
        CompatibilityGateState.FAILED_INTELLIGIBILITY,
        CompatibilityGateState.FAILED_IDENTITY,
        CompatibilityGateState.FAILED_TIMING,
        CompatibilityGateState.FAILED_AUDIO_QUALITY,
    ):
        if failure in states:
            return failure
    if CompatibilityGateState.PENDING in states:
        return CompatibilityGateState.PENDING
    if CompatibilityGateState.NEEDS_HUMAN_REVIEW in states:
        return CompatibilityGateState.NEEDS_HUMAN_REVIEW
    return CompatibilityGateState.PASSED


def write_compatibility_report(
    report: Mapping[str, Any], json_path: Path, markdown_path: Path
) -> None:
    safe_report = redact(dict(report))
    atomic_write_json(json_path, safe_report)
    atomic_write_text(markdown_path, compatibility_report_markdown(safe_report))


def compatibility_report_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Azure TTS to OpenVoice compatibility report",
        "",
        f"- Job: `{report['job_id']}`",
        f"- Gate state: **{report['gate_state']}**",
        f"- Threshold version: `{report['thresholds']['version']}`",
        f"- Thresholds provisional: `{str(report['thresholds_provisional']).lower()}`",
        "- Publication approval: **not granted by this report**",
        "",
        "| Unit | Speaker | State | Azure ms | OpenVoice ms | Ratio | Timing error ms |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for unit in report["units"]:
        lines.append(
            "| {unit_id} | {speaker_id} | {state} | {azure} | {openvoice} | "
            "{ratio} | {timing} |".format(
                unit_id=unit["unit_id"],
                speaker_id=unit["speaker_id"],
                state=unit["state"],
                azure=_display(unit.get("azure_duration_ms")),
                openvoice=_display(unit.get("openvoice_duration_ms")),
                ratio=_display(unit.get("duration_ratio")),
                timing=_display(unit.get("final_timing_error_ms")),
            )
        )
    lines.extend(
        [
            "",
            "Human voice-rights, editorial, factual, and fluent isiZulu reviews "
            "must all be approved before the gate can pass.",
            "",
        ]
    )
    return "\n".join(lines)


@dataclass(frozen=True)
class ReviewPackageUnit:
    unit_id: str
    original_reference: Path
    azure_tts: Path
    openvoice: Path
    aligned: Path

    def __post_init__(self) -> None:
        if not _SAFE_IDENTIFIER.fullmatch(self.unit_id):
            raise ValueError("Invalid A/B review unit ID")


def build_ab_review_package(
    review_root: Path,
    report: Mapping[str, Any],
    assets: Sequence[ReviewPackageUnit],
) -> dict[str, Any]:
    expected_units = {str(item["unit_id"]) for item in report.get("units", [])}
    asset_units = {item.unit_id for item in assets}
    if expected_units != asset_units:
        raise ValueError("A/B review assets do not match compatibility report units")
    for name in (
        "original_reference",
        "azure_tts",
        "openvoice",
        "aligned",
        "comparisons",
    ):
        (review_root / name).mkdir(parents=True, exist_ok=True)

    comparisons = []
    report_units = {str(item["unit_id"]): item for item in report["units"]}
    for item in sorted(assets, key=lambda value: value.unit_id):
        copied = {}
        for label, source in (
            ("original_reference", item.original_reference),
            ("azure_tts", item.azure_tts),
            ("openvoice", item.openvoice),
            ("aligned", item.aligned),
        ):
            if "://" in str(source):
                raise ValueError("A/B review packages require local materialised assets")
            if not source.is_file():
                raise FileNotFoundError(f"Missing {label} audio for unit {item.unit_id}")
            _validate_review_audio(source, label=label, unit_id=item.unit_id)
            destination = review_root / label / f"{item.unit_id}.wav"
            atomic_copy(source, destination)
            actual_hash = sha256_file(destination)
            expected_hash = report_units[item.unit_id].get(
                {
                    "original_reference": "reference_audio_sha256",
                    "azure_tts": "azure_audio_sha256",
                    "openvoice": "openvoice_audio_sha256",
                    "aligned": "aligned_audio_sha256",
                }[label]
            )
            if expected_hash is not None and actual_hash != expected_hash:
                raise ValueError(
                    f"A/B review {label} hash does not match unit {item.unit_id}"
                )
            copied[label] = {
                "path": str(destination.relative_to(review_root)),
                "sha256": actual_hash,
            }
        comparison = {
            "schema_version": REVIEW_PACKAGE_SCHEMA,
            "unit_id": item.unit_id,
            "measurements": report_units[item.unit_id],
            "assets": copied,
        }
        atomic_write_json(review_root / "comparisons" / f"{item.unit_id}.json", comparison)
        comparisons.append(comparison)

    write_compatibility_report(
        report,
        review_root / "compatibility_report.json",
        review_root / "compatibility_report.md",
    )
    atomic_write_text(review_root / "listening_guide.md", _listening_guide(report))
    package_manifest = {
        "schema_version": REVIEW_PACKAGE_SCHEMA,
        "job_id": report["job_id"],
        "gate_state": report["gate_state"],
        "unit_count": len(comparisons),
        "human_review_required": True,
        "required_human_reviews": list(REQUIRED_HUMAN_REVIEWS),
        "units": [item["unit_id"] for item in comparisons],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(review_root / "review_manifest.json", package_manifest)
    return package_manifest


def _listening_guide(report: Mapping[str, Any]) -> str:
    unit_ids = ", ".join(f"`{item['unit_id']}`" for item in report["units"])
    return "\n".join(
        [
            "# Azure/OpenVoice A/B listening guide",
            "",
            f"Job: `{report['job_id']}`",
            "",
            "For each unit, listen in this order: original reference, raw Azure TTS, "
            "OpenVoice conversion, then aligned output. Compare the converted speech "
            "with the intended spoken text in the corresponding comparison JSON.",
            "",
            "Record pronunciation, intelligibility, protected names and numbers, "
            "speaker identity, artefacts, and timing. Do not approve publication from "
            "model inference or automated scores alone.",
            "",
            f"Units: {unit_ids}",
            "",
        ]
    )


def _validate_review_audio(path: Path, *, label: str, unit_id: str) -> None:
    try:
        with wave.open(str(path), "rb") as audio:
            valid = (
                audio.getnframes() > 0
                and audio.getframerate() > 0
                and audio.getnchannels() > 0
                and audio.getsampwidth() > 0
            )
    except (EOFError, wave.Error) as exc:
        raise ValueError(f"Invalid {label} WAV for unit {unit_id}") from exc
    if not valid:
        raise ValueError(f"Empty {label} WAV for unit {unit_id}")


def _display(value: Any) -> str:
    return "-" if value is None else str(value)
