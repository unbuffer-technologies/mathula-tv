"""CPU-only acoustic voice-family analysis for speaker profiling."""

from __future__ import annotations

import json
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class AcousticAnalysisResult:
    """Result of CPU-only acoustic voice-family analysis."""
    
    speaker_id: str
    voice_family: str  # "masculine" | "feminine" | "unknown"
    confidence: float  # 0.0 to 1.0
    median_f0_hz: float | None
    f0_percentile_25: float | None
    f0_percentile_75: float | None
    voiced_frame_ratio: float
    total_duration_ms: int
    usable_duration_ms: int
    sample_count: int
    evidence: dict[str, Any]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "speaker_id": self.speaker_id,
            "voice_family": self.voice_family,
            "confidence": self.confidence,
            "median_f0_hz": self.median_f0_hz,
            "f0_percentile_25": self.f0_percentile_25,
            "f0_percentile_75": self.f0_percentile_75,
            "voiced_frame_ratio": self.voiced_frame_ratio,
            "total_duration_ms": self.total_duration_ms,
            "usable_duration_ms": self.usable_duration_ms,
            "sample_count": self.sample_count,
            "evidence": self.evidence,
        }


def analyze_speaker_acoustics(
    audio_path: Path,
    speaker_id: str,
    diarization_turns: list[dict[str, Any]],
    *,
    min_duration_ms: int = 2000,
    min_voiced_ratio: float = 0.3,
    masculine_threshold_hz: float = 165.0,
    feminine_threshold_hz: float = 255.0,
) -> AcousticAnalysisResult:
    """Perform CPU-only acoustic voice-family analysis on speaker audio.
    
    Uses pitch (F0) analysis to classify voice family as masculine, feminine, or unknown.
    This is a heuristic analysis - not definitive gender determination.
    
    Args:
        audio_path: Path to mono PCM16 WAV file
        speaker_id: Speaker identifier
        diarization_turns: List of diarization turns for this speaker
        min_duration_ms: Minimum usable audio duration required
        min_voiced_ratio: Minimum ratio of voiced frames to consider analysis valid
        masculine_threshold_hz: F0 threshold below which voice is considered masculine
        feminine_threshold_hz: F0 threshold above which voice is considered feminine
    
    Returns:
        AcousticAnalysisResult with voice family classification
    """
    # Extract speaker-specific segments from diarization
    speaker_turns = [t for t in diarization_turns if t.get("speaker") == speaker_id]
    
    if not speaker_turns:
        return AcousticAnalysisResult(
            speaker_id=speaker_id,
            voice_family="unknown",
            confidence=0.0,
            median_f0_hz=None,
            f0_percentile_25=None,
            f0_percentile_75=None,
            voiced_frame_ratio=0.0,
            total_duration_ms=0,
            usable_duration_ms=0,
            sample_count=0,
            evidence={"reason": "no_diarization_turns_for_speaker"},
        )
    
    # Calculate total speaker duration
    total_duration_ms = sum(
        int(t.get("end", 0) * 1000) - int(t.get("start", 0) * 1000)
        for t in speaker_turns
    )
    
    if total_duration_ms < min_duration_ms:
        return AcousticAnalysisResult(
            speaker_id=speaker_id,
            voice_family="unknown",
            confidence=0.0,
            median_f0_hz=None,
            f0_percentile_25=None,
            f0_percentile_75=None,
            voiced_frame_ratio=0.0,
            total_duration_ms=total_duration_ms,
            usable_duration_ms=0,
            sample_count=0,
            evidence={"reason": "insufficient_duration", "min_required_ms": min_duration_ms},
        )
    
    # Load audio and extract speaker segments
    try:
        audio_samples, sample_rate = _load_wav(audio_path)
    except Exception as exc:
        return AcousticAnalysisResult(
            speaker_id=speaker_id,
            voice_family="unknown",
            confidence=0.0,
            median_f0_hz=None,
            f0_percentile_25=None,
            f0_percentile_75=None,
            voiced_frame_ratio=0.0,
            total_duration_ms=total_duration_ms,
            usable_duration_ms=0,
            sample_count=0,
            evidence={"reason": "audio_load_failed", "error": str(exc)},
        )
    
    # Extract speaker-specific audio segments
    speaker_segments = []
    for turn in speaker_turns:
        start_sample = int(turn.get("start", 0) * sample_rate)
        end_sample = int(turn.get("end", 0) * sample_rate)
        if end_sample > start_sample and end_sample <= len(audio_samples):
            segment = audio_samples[start_sample:end_sample]
            speaker_segments.append(segment)
    
    if not speaker_segments:
        return AcousticAnalysisResult(
            speaker_id=speaker_id,
            voice_family="unknown",
            confidence=0.0,
            median_f0_hz=None,
            f0_percentile_25=None,
            f0_percentile_75=None,
            voiced_frame_ratio=0.0,
            total_duration_ms=total_duration_ms,
            usable_duration_ms=0,
            sample_count=0,
            evidence={"reason": "no_valid_audio_segments"},
        )
    
    # Concatenate all speaker segments
    combined_audio = np.concatenate(speaker_segments)
    usable_duration_ms = int(len(combined_audio) * 1000 / sample_rate)
    
    # Perform pitch analysis using autocorrelation (CPU-only)
    f0_values, voiced_ratio = _estimate_pitch_autocorrelation(
        combined_audio,
        sample_rate,
    )
    
    if voiced_ratio < min_voiced_ratio or len(f0_values) < 10:
        return AcousticAnalysisResult(
            speaker_id=speaker_id,
            voice_family="unknown",
            confidence=0.0,
            median_f0_hz=None,
            f0_percentile_25=None,
            f0_percentile_75=None,
            voiced_frame_ratio=voiced_ratio,
            total_duration_ms=total_duration_ms,
            usable_duration_ms=usable_duration_ms,
            sample_count=len(f0_values),
            evidence={
                "reason": "insufficient_voiced_frames",
                "voiced_ratio": voiced_ratio,
                "min_required_ratio": min_voiced_ratio,
            },
        )
    
    # Calculate statistics
    median_f0 = float(np.median(f0_values))
    f0_p25 = float(np.percentile(f0_values, 25))
    f0_p75 = float(np.percentile(f0_values, 75))
    
    # Classify voice family based on median F0
    # Use a weighted approach considering both median and percentile distribution
    if median_f0 < masculine_threshold_hz:
        voice_family = "masculine"
        # Confidence based on how far below threshold
        confidence = min(1.0, (masculine_threshold_hz - median_f0) / 50.0 + 0.5)
    elif median_f0 > feminine_threshold_hz:
        voice_family = "feminine"
        # Confidence based on how far above threshold
        confidence = min(1.0, (median_f0 - feminine_threshold_hz) / 50.0 + 0.5)
    else:
        # In the ambiguous range between thresholds
        # If both percentiles span both thresholds, evidence is contradictory
        if f0_p25 < masculine_threshold_hz and f0_p75 > feminine_threshold_hz:
            # Contradictory percentile evidence spanning both voice-family thresholds
            voice_family = "unknown"
            confidence = 0.0
        elif f0_p75 > feminine_threshold_hz:
            voice_family = "feminine"
            # Boost confidence when percentile distribution is clear
            percentile_margin = f0_p75 - feminine_threshold_hz
            confidence = min(0.95, percentile_margin / 30.0 + 0.5)
        elif f0_p25 < masculine_threshold_hz:
            voice_family = "masculine"
            # Boost confidence when percentile distribution is clear
            percentile_margin = masculine_threshold_hz - f0_p25
            confidence = min(0.95, percentile_margin / 30.0 + 0.5)
        else:
            # Truly ambiguous
            voice_family = "unknown"
            confidence = 0.0
    
    # Cap confidence at 0.95 for acoustic analysis (not definitive)
    confidence = min(0.95, confidence)
    
    evidence = {
        "turn_count": len(speaker_turns),
        "total_duration_ms": total_duration_ms,
        "usable_duration_ms": usable_duration_ms,
        "sample_rate": sample_rate,
        "f0_sample_count": len(f0_values),
        "voiced_frame_ratio": voiced_ratio,
        "masculine_threshold_hz": masculine_threshold_hz,
        "feminine_threshold_hz": feminine_threshold_hz,
    }
    
    return AcousticAnalysisResult(
        speaker_id=speaker_id,
        voice_family=voice_family,
        confidence=confidence,
        median_f0_hz=median_f0,
        f0_percentile_25=f0_p25,
        f0_percentile_75=f0_p75,
        voiced_frame_ratio=voiced_ratio,
        total_duration_ms=total_duration_ms,
        usable_duration_ms=usable_duration_ms,
        sample_count=len(f0_values),
        evidence=evidence,
    )


def _load_wav(path: Path) -> tuple[np.ndarray, int]:
    """Load mono PCM16 WAV file as numpy array."""
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1:
            raise ValueError("Audio must be mono")
        if wav.getsampwidth() != 2:
            raise ValueError("Audio must be PCM16")
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return samples, sample_rate


def _estimate_pitch_autocorrelation(
    audio: np.ndarray,
    sample_rate: int,
    frame_size_ms: int = 25,
    frame_shift_ms: int = 10,
    min_f0_hz: float = 75.0,
    max_f0_hz: float = 400.0,
) -> tuple[np.ndarray, float]:
    """Estimate pitch using autocorrelation (CPU-only, no external dependencies).
    
    Returns:
        tuple of (f0_values_array, voiced_frame_ratio)
    """
    frame_size = int(frame_size_ms * sample_rate / 1000)
    frame_shift = int(frame_shift_ms * sample_rate / 1000)
    
    min_lag = int(sample_rate / max_f0_hz)
    max_lag = int(sample_rate / min_f0_hz)
    
    f0_values = []
    voiced_count = 0
    total_frames = 0
    
    for i in range(0, len(audio) - frame_size, frame_shift):
        frame = audio[i:i + frame_size]
        
        # Apply windowing
        window = np.hanning(len(frame))
        frame_windowed = frame * window
        
        # Autocorrelation
        autocorr = np.correlate(frame_windowed, frame_windowed, mode='full')
        autocorr = autocorr[len(autocorr) // 2:]
        
        # Find peak in valid lag range
        if len(autocorr) > max_lag:
            relevant = autocorr[min_lag:max_lag]
            if len(relevant) > 0:
                peak_idx = np.argmax(relevant) + min_lag
                
                # Check if peak is significant enough
                peak_value = autocorr[peak_idx]
                if peak_value > 0.3 * autocorr[0]:  # Threshold for voiced frame
                    f0 = sample_rate / peak_idx
                    
                    # Apply octave correction to handle harmonic selection
                    f0 = _correct_octave_error(f0, peak_idx, autocorr, min_lag, max_lag, sample_rate)
                    
                    if min_f0_hz <= f0 <= max_f0_hz:
                        f0_values.append(f0)
                        voiced_count += 1
        
        total_frames += 1
    
    voiced_ratio = voiced_count / total_frames if total_frames > 0 else 0.0
    
    return np.array(f0_values), voiced_ratio


def _correct_octave_error(
    f0: float,
    peak_idx: int,
    autocorr: np.ndarray,
    min_lag: int,
    max_lag: int,
    sample_rate: int,
) -> float:
    """Correct octave doubling errors in pitch detection.
    
    If the detected F0 is implausibly high (likely a harmonic), check if
    dividing by 2 gives a more plausible fundamental frequency by examining
    the autocorrelation at the corresponding lag.
    
    Args:
        f0: Detected fundamental frequency
        peak_idx: Lag index of the autocorrelation peak
        autocorr: Autocorrelation array
        min_lag: Minimum valid lag index
        max_lag: Maximum valid lag index
        sample_rate: Audio sample rate
    
    Returns:
        Corrected F0 (may be unchanged if no correction needed)
    """
    # If F0 is in a range where octave doubling is common (150-300 Hz),
    # check if half the frequency (double the lag) has significant autocorrelation
    if 150.0 <= f0 <= 300.0:
        half_lag = peak_idx * 2
        if half_lag < len(autocorr) and half_lag >= min_lag:
            # Check if autocorrelation at double lag is also significant
            # This suggests the fundamental might be at half the detected frequency
            half_peak_value = autocorr[half_lag]
            zero_lag_value = autocorr[0]
            
            # If the half-lag peak is at least 60% of the detected peak,
            # it's likely the true fundamental
            if half_peak_value > 0.6 * autocorr[peak_idx] and half_peak_value > 0.3 * zero_lag_value:
                corrected_f0 = sample_rate / half_lag
                # Only apply correction if the corrected value is in valid range
                if 75.0 <= corrected_f0 <= 400.0:
                    return corrected_f0
    
    return f0


def _analyze_with_ecapa_classifier(
    job_dir: Path,
    canonical_speaker_id: str,
    audio_path: Path,
    raw_speaker_id: str,
    turns: list[dict[str, Any]],
) -> AcousticAnalysisResult:
    """Analyze speaker using ECAPA voice-family classifier.
    
    Args:
        job_dir: Job directory containing canonical speaker WAV files
        canonical_speaker_id: Canonical speaker ID (e.g., SPEAKER_00)
        audio_path: Path to analysis audio (for F0 diagnostics)
        raw_speaker_id: Raw speaker ID from diarization
        turns: Diarization turns for this speaker
    
    Returns:
        AcousticAnalysisResult with ECAPA classification and F0 diagnostics
    """
    from .voice_family_classifier import classify_voice_family
    
    # Path to canonical speaker WAV file
    speaker_wav_path = job_dir / "audio" / "speakers" / f"{canonical_speaker_id}.wav"
    
    if not speaker_wav_path.exists():
        return AcousticAnalysisResult(
            speaker_id=canonical_speaker_id,
            voice_family="unknown",
            confidence=0.0,
            median_f0_hz=None,
            f0_percentile_25=None,
            f0_percentile_75=None,
            voiced_frame_ratio=0.0,
            total_duration_ms=0,
            usable_duration_ms=0,
            sample_count=0,
            evidence={"reason": "canonical_speaker_wav_missing", "path": str(speaker_wav_path)},
        )
    
    try:
        # Run ECAPA classifier
        classification = classify_voice_family(
            speaker_audio_path=speaker_wav_path,
            speaker_id=canonical_speaker_id,
            confidence_threshold=0.70,
        )
    except Exception as exc:
        return AcousticAnalysisResult(
            speaker_id=canonical_speaker_id,
            voice_family="unknown",
            confidence=0.0,
            median_f0_hz=None,
            f0_percentile_25=None,
            f0_percentile_75=None,
            voiced_frame_ratio=0.0,
            total_duration_ms=0,
            usable_duration_ms=0,
            sample_count=0,
            evidence={"reason": "ecapa_classifier_failed", "error": str(exc)},
        )
    
    # Calculate F0 diagnostics (kept for diagnostic purposes only)
    speaker_turns = [t for t in turns if t.get("speaker") == raw_speaker_id]
    total_duration_ms = sum(
        int(t.get("end", 0) * 1000) - int(t.get("start", 0) * 1000)
        for t in speaker_turns
    ) if speaker_turns else 0
    
    f0_diagnostics = {}
    if total_duration_ms >= 2000 and audio_path.exists():
        try:
            audio_samples, sample_rate = _load_wav(audio_path)
            speaker_segments = []
            for turn in speaker_turns:
                start_sample = int(turn.get("start", 0) * sample_rate)
                end_sample = int(turn.get("end", 0) * sample_rate)
                if end_sample > start_sample and end_sample <= len(audio_samples):
                    segment = audio_samples[start_sample:end_sample]
                    speaker_segments.append(segment)
            
            if speaker_segments:
                combined_audio = np.concatenate(speaker_segments)
                f0_values, voiced_ratio = _estimate_pitch_autocorrelation(combined_audio, sample_rate)
                if len(f0_values) > 0:
                    f0_diagnostics = {
                        "median_f0_hz": float(np.median(f0_values)),
                        "f0_percentile_25": float(np.percentile(f0_values, 25)),
                        "f0_percentile_75": float(np.percentile(f0_values, 75)),
                        "f0_voice_family": "unknown",  # F0 result does not override ECAPA
                    }
        except Exception:
            # F0 diagnostics are optional, don't fail if they fail
            pass
    
    # Build evidence
    evidence = {
        "method": classification.method,
        "model": classification.model,
        "device": classification.device,
        "turn_count": len(speaker_turns),
        "total_duration_ms": total_duration_ms,
    }
    
    # Add F0 diagnostics if available
    if f0_diagnostics:
        evidence["f0_diagnostics"] = f0_diagnostics
    
    # Add ECAPA probabilities
    evidence["ecapa_probabilities"] = classification.probabilities
    
    return AcousticAnalysisResult(
        speaker_id=canonical_speaker_id,
        voice_family=classification.voice_family,
        confidence=classification.confidence,
        median_f0_hz=f0_diagnostics.get("median_f0_hz"),
        f0_percentile_25=f0_diagnostics.get("f0_percentile_25"),
        f0_percentile_75=f0_diagnostics.get("f0_percentile_75"),
        voiced_frame_ratio=f0_diagnostics.get("voiced_frame_ratio", 0.0),
        total_duration_ms=total_duration_ms,
        usable_duration_ms=total_duration_ms,
        sample_count=f0_diagnostics.get("f0_sample_count", 0),
        evidence=evidence,
    )


def analyze_all_speakers(
    audio_path: Path,
    diarization_path: Path,
    transcript_path: Path | None = None,
    output_path: Path | None = None,
    job_dir: Path | None = None,
    use_ecapa_classifier: bool = True,
) -> dict[str, dict[str, Any]]:
    """Analyze acoustics for all speakers in a diarization file.
    
    Args:
        audio_path: Path to mono PCM16 WAV file
        diarization_path: Path to diarization JSON file
        transcript_path: Optional path to transcript for canonical speaker ID mapping
        output_path: Optional path to write results JSON
        job_dir: Optional job directory for canonical speaker WAV files
        use_ecapa_classifier: If True, use ECAPA classifier for voice-family classification
    
    Returns:
        Dictionary mapping canonical speaker_id to analysis results
    """
    diarization = json.loads(diarization_path.read_text(encoding="utf-8"))
    turns = diarization.get("turns", [])
    
    # Build mapping from diarization speaker IDs to canonical speaker IDs
    speaker_mapping = {}
    if transcript_path and transcript_path.is_file():
        transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
        # Extract mapping from transcript segments
        # Each segment has the canonical speaker_id and the raw speaker from diarization
        for segment in transcript.get("segments", []):
            # A contextual turn repair intentionally overrides Azure for one
            # short phrase while preserving Azure's raw label as provenance.
            # It must never redefine the job-wide raw-to-canonical acoustic map.
            if segment.get("speaker_assignment_method") == "contextual_vocative_repair":
                continue
            canonical_id = segment.get("speaker_id", segment.get("speaker"))
            raw_id = segment.get("raw_speaker")  # May not exist in older transcripts
            if canonical_id and raw_id:
                speaker_mapping[raw_id] = canonical_id
    
    # If no mapping from transcript, build it from diarization using first-occurrence order
    if not speaker_mapping:
        raw_speakers = sorted(set(t.get("speaker") for t in turns if t.get("speaker")))
        for idx, raw_speaker in enumerate(raw_speakers):
            speaker_mapping[raw_speaker] = f"SPEAKER_{idx:02d}"
    
    # Get unique speakers from diarization
    raw_speakers = sorted(set(t.get("speaker") for t in turns if t.get("speaker")))
    
    results = {}
    for raw_speaker_id in raw_speakers:
        # Use canonical ID if mapping exists, otherwise use raw ID
        canonical_speaker_id = speaker_mapping.get(raw_speaker_id, raw_speaker_id)
        
        if use_ecapa_classifier and job_dir:
            # Use ECAPA classifier for voice-family classification
            result = _analyze_with_ecapa_classifier(
                job_dir=job_dir,
                canonical_speaker_id=canonical_speaker_id,
                audio_path=audio_path,
                raw_speaker_id=raw_speaker_id,
                turns=turns,
            )
        else:
            # Use F0-based classification (legacy)
            result = analyze_speaker_acoustics(audio_path, raw_speaker_id, turns)
        
        result_dict = result.to_dict()
        # Store under canonical speaker ID
        result_dict["speaker_id"] = canonical_speaker_id
        result_dict["raw_speaker_id"] = raw_speaker_id
        results[canonical_speaker_id] = result_dict
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "schema_version": "acoustic-analysis-v1",
            "audio_path": str(audio_path),
            "audio_sha256": _file_checksum(audio_path),
            "diarization_path": str(diarization_path),
            "transcript_path": str(transcript_path) if transcript_path else None,
            "speaker_mapping": speaker_mapping,
            "speakers": results,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(artifact, f, indent=2)
    
    return results


def _file_checksum(path: Path) -> str:
    """Compute SHA256 checksum of a file."""
    import hashlib
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


__all__ = [
    "AcousticAnalysisResult",
    "analyze_speaker_acoustics",
    "analyze_all_speakers",
]
