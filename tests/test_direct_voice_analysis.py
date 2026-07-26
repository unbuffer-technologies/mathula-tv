from __future__ import annotations

import json
import math
import struct
import wave
from pathlib import Path

from mathula_tv.direct_voice_analysis import ensure_canonical_voice_family_analysis
from mathula_tv.voice_family_classifier import VoiceFamilyClassificationResult


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
