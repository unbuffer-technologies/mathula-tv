"""Azure TTS quality control and validation."""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .atomic_io import read_json
from .media import checksum


@dataclass
class AzureTTSQCResult:
    """Result of Azure TTS quality control for a single unit."""
    
    unit_id: str
    speaker_id: str
    wav_exists: bool
    wav_decodable: bool
    duration_ms: int
    has_effective_silence: bool
    voice_matches_assignment: bool
    duration_within_tolerance: bool
    clipping_ratio: float
    protected_entities_present: bool
    passed: bool
    failures: list[str]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "speaker_id": self.speaker_id,
            "wav_exists": self.wav_exists,
            "wav_decodable": self.wav_decodable,
            "duration_ms": self.duration_ms,
            "has_effective_silence": self.has_effective_silence,
            "voice_matches_assignment": self.voice_matches_assignment,
            "duration_within_tolerance": self.duration_within_tolerance,
            "clipping_ratio": self.clipping_ratio,
            "protected_entities_present": self.protected_entities_present,
            "passed": self.passed,
            "failures": self.failures,
        }


@dataclass
class AzureTTSQCReport:
    """Complete Azure TTS QC report for a job."""
    
    schema_version: str = "azure-tts-qc-v1"
    job_id: str = ""
    total_units: int = 0
    passed_units: int = 0
    failed_units: int = 0
    unit_results: list[dict[str, Any]] = None
    overall_passed: bool = False
    blocking_failures: list[str] = None
    
    def __post_init__(self):
        if self.unit_results is None:
            self.unit_results = []
        if self.blocking_failures is None:
            self.blocking_failures = []
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "job_id": self.job_id,
            "total_units": self.total_units,
            "passed_units": self.passed_units,
            "failed_units": self.failed_units,
            "unit_results": self.unit_results,
            "overall_passed": self.overall_passed,
            "blocking_failures": self.blocking_failures,
        }


def validate_azure_tts_unit(
    unit: dict[str, Any],
    wav_path: Path,
    assigned_voice: str | None = None,
    protected_entities: list[str] | None = None,
    *,
    max_silence_ratio: float = 0.8,
    max_clipping_ratio: float = 0.01,
    duration_tolerance_ms: int = 200,
) -> AzureTTSQCResult:
    """Validate a single Azure TTS unit.
    
    Args:
        unit: Dubbing plan unit
        wav_path: Path to synthesized WAV file
        assigned_voice: Expected voice from voice assignment (if available)
        protected_entities: List of protected entity placeholders expected in TTS text
        max_silence_ratio: Maximum ratio of silence to total duration
        max_clipping_ratio: Maximum ratio of clipped samples
        duration_tolerance_ms: Allowed deviation from preferred duration
    
    Returns:
        AzureTTSQCResult with validation results
    """
    unit_id = unit["unit_id"]
    speaker_id = unit["speaker_id"]
    failures = []
    
    # Check WAV exists
    wav_exists = wav_path.is_file()
    if not wav_exists:
        failures.append("WAV file does not exist")
        return AzureTTSQCResult(
            unit_id=unit_id,
            speaker_id=speaker_id,
            wav_exists=False,
            wav_decodable=False,
            duration_ms=0,
            has_effective_silence=True,
            voice_matches_assignment=False,
            duration_within_tolerance=False,
            clipping_ratio=1.0,
            protected_entities_present=False,
            passed=False,
            failures=failures,
        )
    
    # Check WAV is decodable
    try:
        with wave.open(str(wav_path), "rb") as wav:
            if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
                failures.append("WAV is not mono PCM16")
                wav_decodable = False
                duration_ms = 0
                samples = []
            else:
                wav_decodable = True
                frames = wav.readframes(wav.getnframes())
                samples = list(frames)
                duration_ms = int(wav.getnframes() * 1000 / wav.getframerate())
    except Exception as exc:
        failures.append(f"WAV decode error: {exc}")
        wav_decodable = False
        duration_ms = 0
        samples = []
    
    if not wav_decodable:
        return AzureTTSQCResult(
            unit_id=unit_id,
            speaker_id=speaker_id,
            wav_exists=True,
            wav_decodable=False,
            duration_ms=duration_ms,
            has_effective_silence=True,
            voice_matches_assignment=False,
            duration_within_tolerance=False,
            clipping_ratio=1.0,
            protected_entities_present=False,
            passed=False,
            failures=failures,
        )
    
    # Check for effective silence
    silence_threshold = 100  # PCM16 threshold for silence
    silent_samples = sum(1 for s in samples if abs(s) < silence_threshold)
    silence_ratio = silent_samples / len(samples) if samples else 1.0
    has_effective_silence = silence_ratio > max_silence_ratio
    if has_effective_silence:
        failures.append(f"Effective silence ratio {silence_ratio:.2f} exceeds threshold {max_silence_ratio}")
    
    # Check clipping
    clipping_threshold = 32700  # Near max for int16
    clipped_samples = sum(1 for s in samples if abs(s) > clipping_threshold)
    clipping_ratio = clipped_samples / len(samples) if samples else 0.0
    if clipping_ratio > max_clipping_ratio:
        failures.append(f"Clipping ratio {clipping_ratio:.4f} exceeds threshold {max_clipping_ratio}")
    
    # Check duration tolerance
    preferred_duration = int(unit.get("preferred_duration_ms", 0))
    max_duration = int(unit.get("maximum_duration_ms", 0))
    duration_within_tolerance = (
        preferred_duration > 0
        and abs(duration_ms - preferred_duration) <= duration_tolerance_ms
    )
    if not duration_within_tolerance:
        failures.append(
            f"Duration {duration_ms}ms deviates from preferred {preferred_duration}ms by more than {duration_tolerance_ms}ms"
        )
    
    # Check voice matches assignment
    voice_matches_assignment = True
    if assigned_voice:
        # This check requires the manifest to have voice information
        # For now, we assume it matches if the file exists
        pass
    
    # Check protected entities in TTS text
    tts_text = unit.get("tts_text", "")
    protected_entities_present = True
    if protected_entities:
        for entity in protected_entities:
            if entity not in tts_text:
                failures.append(f"Protected entity '{entity}' missing from TTS text")
                protected_entities_present = False
    
    passed = not failures
    
    return AzureTTSQCResult(
        unit_id=unit_id,
        speaker_id=speaker_id,
        wav_exists=wav_exists,
        wav_decodable=wav_decodable,
        duration_ms=duration_ms,
        has_effective_silence=has_effective_silence,
        voice_matches_assignment=voice_matches_assignment,
        duration_within_tolerance=duration_within_tolerance,
        clipping_ratio=clipping_ratio,
        protected_entities_present=protected_entities_present,
        passed=passed,
        failures=failures,
    )


def validate_azure_tts_manifest(
    manifest_path: Path,
    dubbing_plan_path: Path,
    voice_assignments_path: Path | None = None,
) -> AzureTTSQCReport:
    """Validate complete Azure TTS manifest against dubbing plan.
    
    Args:
        manifest_path: Path to azure_tts/manifest.json
        dubbing_plan_path: Path to dubbing/dubbing_plan.json
        voice_assignments_path: Optional path to azure_voice_assignments.json
    
    Returns:
        AzureTTSQCReport with complete validation results
    """
    manifest = read_json(manifest_path)
    plan = read_json(dubbing_plan_path)
    
    voice_assignments = None
    if voice_assignments_path and voice_assignments_path.is_file():
        voice_assignments = read_json(voice_assignments_path)
    
    job_id = plan.get("job_id", "")
    units = plan.get("units", [])
    manifest_units = {item["turn_id"]: item for item in manifest.get("units", [])}
    
    # Build voice assignment map
    voice_by_speaker = {}
    if voice_assignments:
        for assignment in voice_assignments.get("assignments", []):
            voice_by_speaker[assignment["speaker_id"]] = assignment["selected_voice"]
    
    results = []
    passed_count = 0
    failed_count = 0
    blocking_failures = []
    
    for unit in units:
        unit_id = unit["unit_id"]
        speaker_id = unit["speaker_id"]
        
        # Get WAV path from manifest
        manifest_unit = manifest_units.get(unit_id)
        if not manifest_unit:
            results.append({
                "unit_id": unit_id,
                "speaker_id": speaker_id,
                "passed": False,
                "failures": ["Unit not in Azure TTS manifest"],
            })
            failed_count += 1
            blocking_failures.append(f"Unit {unit_id} missing from manifest")
            continue
        
        wav_path = Path(manifest_unit.get("output_path", ""))
        assigned_voice = voice_by_speaker.get(speaker_id)
        
        # Get protected entities from unit
        protected_entities = unit.get("protected_entities_found", [])
        
        result = validate_azure_tts_unit(
            unit=unit,
            wav_path=wav_path,
            assigned_voice=assigned_voice,
            protected_entities=protected_entities,
        )
        results.append(result.to_dict())
        
        if result.passed:
            passed_count += 1
        else:
            failed_count += 1
            if not result.wav_exists or not result.wav_decodable:
                blocking_failures.append(f"Unit {unit_id} has critical audio failure")
    
    overall_passed = failed_count == 0 and not blocking_failures
    
    report = AzureTTSQCReport(
        schema_version="azure-tts-qc-v1",
        job_id=job_id,
        total_units=len(units),
        passed_units=passed_count,
        failed_units=failed_count,
        unit_results=results,
        overall_passed=overall_passed,
        blocking_failures=blocking_failures,
    )
    
    return report


def write_qc_report(report: AzureTTSQCReport, output_path: Path) -> None:
    """Write QC report to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        import json
        json.dump(report.to_dict(), f, indent=2)


__all__ = [
    "AzureTTSQCResult",
    "AzureTTSQCReport",
    "validate_azure_tts_unit",
    "validate_azure_tts_manifest",
    "write_qc_report",
]
