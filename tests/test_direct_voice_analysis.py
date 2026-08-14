from __future__ import annotations

import json
import math
import struct
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mathula_tv.direct_voice_analysis import (
    ensure_canonical_speaker_reels,
    ensure_canonical_voice_family_analysis,
)


@dataclass
class VoiceFamilyClassificationResult:
    speaker_id: str
    method: str
    model: str
    device: str
    voice_family: str
    confidence: float
    probabilities: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "speaker_id": self.speaker_id,
            "method": self.method,
            "model": self.model,
            "device": self.device,
            "voice_family": self.voice_family,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
        }


def _write_tone(path: Path, *, seconds: float = 4.0, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        frames = bytearray()
        for index in range(round(seconds * sample_rate)):
            value = round(8000 * math.sin(2 * math.pi * 180 * index / sample_rate))
            frames.extend(struct.pack("<h", value))
        wav.writeframes(bytes(frames))


def test_builds_canonical_speaker_reels_and_runs_classifier(tmp_path: Path) -> None:
    source = tmp_path / "audio" / "analysis_mono.wav"
    _write_tone(source)
    output = tmp_path / "analysis" / "acoustic_analysis.json"
    calls: list[tuple[str, Path]] = []

    def fake_classifier(*, speaker_audio_path, speaker_id, confidence_threshold):
        calls.append((speaker_id, speaker_audio_path))
        family = "feminine" if speaker_id == "SPEAKER_00" else "masculine"
        return VoiceFamilyClassificationResult(
            speaker_id=speaker_id,
            method="ecapa_voice_family_classifier",
            model="test-model",
            device="cpu",
            voice_family=family,
            confidence=0.93,
            probabilities={
                "masculine": 0.07 if family == "feminine" else 0.93,
                "feminine": 0.93 if family == "feminine" else 0.07,
            },
        )

    artifact = ensure_canonical_voice_family_analysis(
        job_id="job1",
        analysis_audio_path=source,
        diarization={
            "turns": [
                {"speaker": "AZURE_1", "start": 0.0, "end": 2.0},
                {"speaker": "AZURE_2", "start": 2.0, "end": 4.0},
            ]
        },
        speaker_mapping={
            "raw_to_canonical": {
                "AZURE_1": "SPEAKER_00",
                "AZURE_2": "SPEAKER_01",
            },
            "canonical_to_raw": {
                "SPEAKER_00": ["AZURE_1"],
                "SPEAKER_01": ["AZURE_2"],
            },
        },
        required_speaker_ids=["SPEAKER_00", "SPEAKER_01"],
        output_path=output,
        speakers_root=tmp_path / "audio" / "speakers",
        classifier=fake_classifier,
    )

    assert [speaker_id for speaker_id, _ in calls] == ["SPEAKER_00", "SPEAKER_01"]
    assert artifact["speakers"]["SPEAKER_00"]["voice_family"] == "feminine"
    assert artifact["speakers"]["SPEAKER_01"]["voice_family"] == "masculine"
    assert (tmp_path / "audio" / "speakers" / "SPEAKER_00.wav").is_file()
    assert (tmp_path / "audio" / "speakers" / "SPEAKER_01.wav").is_file()
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted["speaker_mapping"]["raw_to_canonical"]["AZURE_1"] == "SPEAKER_00"
    assert persisted["failures"] == {}


def test_voice_family_analysis_accepts_clean_subsecond_multi_turn_reel(
    tmp_path: Path,
) -> None:
    source = tmp_path / "audio" / "analysis_mono.wav"
    _write_tone(source, seconds=2.0)
    output = tmp_path / "analysis" / "acoustic_analysis.json"
    observed_duration: list[float] = []

    def fake_classifier(*, speaker_audio_path, speaker_id, confidence_threshold):
        with wave.open(str(speaker_audio_path), "rb") as wav:
            observed_duration.append(wav.getnframes() / wav.getframerate())
        return VoiceFamilyClassificationResult(
            speaker_id=speaker_id,
            method="ecapa_voice_family_classifier",
            model="test-model",
            device="cpu",
            voice_family="feminine",
            confidence=0.96,
            probabilities={"masculine": 0.04, "feminine": 0.96},
        )

    artifact = ensure_canonical_voice_family_analysis(
        job_id="job1",
        analysis_audio_path=source,
        diarization={
            "turns": [
                {"speaker": "AZURE_5", "start": 0.00, "end": 0.60},
                {"speaker": "AZURE_5", "start": 1.00, "end": 1.12},
                {"speaker": "AZURE_5", "start": 1.50, "end": 1.74},
            ]
        },
        speaker_mapping={
            "raw_to_canonical": {"AZURE_5": "SPEAKER_04"},
            "canonical_to_raw": {"SPEAKER_04": ["AZURE_5"]},
        },
        required_speaker_ids=["SPEAKER_04"],
        output_path=output,
        speakers_root=tmp_path / "audio" / "speakers",
        classifier=fake_classifier,
    )

    record = artifact["speakers"]["SPEAKER_04"]
    assert record["voice_family"] == "feminine"
    assert record["extraction"]["turn_count_selected"] == 3
    assert record["extraction"]["duration_seconds"] == 0.96
    assert observed_duration == [0.96]
    assert artifact["failures"] == {}


def test_known_speaker_reel_failure_is_isolated_per_speaker(tmp_path: Path) -> None:
    source = tmp_path / "audio" / "analysis_mono.wav"
    _write_tone(source, seconds=3.0)

    reels = ensure_canonical_speaker_reels(
        analysis_audio_path=source,
        diarization={
            "turns": [
                {"speaker": "AZURE_1", "start": 0.0, "end": 2.0},
                {"speaker": "AZURE_5", "start": 2.0, "end": 2.6},
            ]
        },
        speaker_mapping={
            "raw_to_canonical": {
                "AZURE_1": "SPEAKER_00",
                "AZURE_5": "SPEAKER_04",
            },
            "canonical_to_raw": {
                "SPEAKER_00": ["AZURE_1"],
                "SPEAKER_04": ["AZURE_5"],
            },
        },
        required_speaker_ids=["SPEAKER_00", "SPEAKER_04"],
        speakers_root=tmp_path / "audio" / "speakers",
    )

    assert reels["SPEAKER_00"]["status"] == "ready"
    assert reels["SPEAKER_04"]["status"] == "unavailable"
    assert "too short" in reels["SPEAKER_04"]["error"]
    assert not (tmp_path / "audio" / "speakers" / "SPEAKER_04.wav").exists()


def test_contextual_turns_separate_two_speakers_with_one_raw_azure_id(
    tmp_path: Path,
) -> None:
    source = tmp_path / "audio" / "analysis_mono.wav"
    source.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 16000
    with wave.open(str(source), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        frames = (
            struct.pack("<h", 1000) * sample_rate
            + struct.pack("<h", 0) * sample_rate
            + struct.pack("<h", -2000) * sample_rate
        )
        wav.writeframes(frames)

    observed_first_samples: dict[str, int] = {}

    def fake_classifier(*, speaker_audio_path, speaker_id, confidence_threshold):
        with wave.open(str(speaker_audio_path), "rb") as wav:
            observed_first_samples[speaker_id] = struct.unpack(
                "<h", wav.readframes(1)
            )[0]
        family = "feminine" if speaker_id == "SPEAKER_00" else "masculine"
        return VoiceFamilyClassificationResult(
            speaker_id=speaker_id,
            method="ecapa_voice_family_classifier",
            model="test-model",
            device="cpu",
            voice_family=family,
            confidence=0.95,
            probabilities={family: 0.95},
        )

    mapping = {
        # Legacy consumers retain one dominant raw mapping, while the direct
        # canonical links audit the contextual split.
        "raw_to_canonical": {"AZURE_1": "SPEAKER_00"},
        "canonical_to_raw": {
            "SPEAKER_00": ["AZURE_1"],
            "SPEAKER_01": ["AZURE_1"],
        },
    }
    artifact = ensure_canonical_voice_family_analysis(
        job_id="job-handoff",
        analysis_audio_path=source,
        diarization={
            "turns": [{"speaker": "AZURE_1", "start": 0.0, "end": 3.0}]
        },
        speaker_mapping=mapping,
        required_speaker_ids=["SPEAKER_00", "SPEAKER_01"],
        output_path=tmp_path / "analysis" / "acoustic_analysis.json",
        speakers_root=tmp_path / "audio" / "speakers",
        canonical_speaker_turns={
            "SPEAKER_00": [{"start": 0.0, "end": 1.0}],
            "SPEAKER_01": [{"start": 2.0, "end": 3.0}],
        },
        classifier=fake_classifier,
    )

    assert observed_first_samples == {"SPEAKER_00": 1000, "SPEAKER_01": -2000}
    assert artifact["speakers"]["SPEAKER_00"]["turn_source"] == (
        "contextual_transcript_segments"
    )
    assert artifact["speakers"]["SPEAKER_01"]["turn_source"] == (
        "contextual_transcript_segments"
    )
    assert artifact["speakers"]["SPEAKER_00"]["extraction"][
        "duration_seconds"
    ] == 1.0
    assert artifact["speakers"]["SPEAKER_01"]["extraction"][
        "duration_seconds"
    ] == 1.0
