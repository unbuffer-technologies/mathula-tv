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
                    if min_f0_hz <= f0 <= max_f0_hz:
                        f0_values.append(f0)
                        voiced_count += 1
        
        total_frames += 1
    
    voiced_ratio = voiced_count / total_frames if total_frames > 0 else 0.0
    
    return np.array(f0_values), voiced_ratio


def analyze_all_speakers(
    audio_path: Path,
    diarization_path: Path,
    output_path: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Analyze acoustics for all speakers in a diarization file.
    
    Args:
        audio_path: Path to mono PCM16 WAV file
        diarization_path: Path to diarization JSON file
        output_path: Optional path to write results JSON
    
    Returns:
        Dictionary mapping speaker_id to analysis results
    """
    diarization = json.loads(diarization_path.read_text(encoding="utf-8"))
    turns = diarization.get("turns", [])
    
    # Get unique speakers
    speakers = sorted(set(t.get("speaker") for t in turns if t.get("speaker")))
    
    results = {}
    for speaker_id in speakers:
        result = analyze_speaker_acoustics(audio_path, speaker_id, turns)
        results[speaker_id] = result.to_dict()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "schema_version": "acoustic-analysis-v1",
            "audio_path": str(audio_path),
            "audio_sha256": _file_checksum(audio_path),
            "diarization_path": str(diarization_path),
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
