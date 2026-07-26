from __future__ import annotations

import json
import math
import wave
from pathlib import Path

import pytest

from mathula_tv.known_speakers import (
    SPEECHBRAIN_MODEL_ID,
    SPEECHBRAIN_MODEL_REVISION,
    enrol_known_speaker,
    identify_known_speaker,
    parse_timeline_range,
)


class FakeEncoder:
    def __init__(self, values: dict[str, list[float]] | None = None):
        self.values = values or {}

    def encode_file(self, path: Path) -> list[float]:
        return self.values.get(path.name, [1.0, 0.0, 0.0])


def _wav(path: Path, seconds: int = 70) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 16000
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        frames = bytearray()
        for index in range(sample_rate * seconds):
            value = round(8000 * math.sin(index * 2 * math.pi * 220 / sample_rate))
            frames.extend(int(value).to_bytes(2, "little", signed=True))
        output.writeframes(bytes(frames))


def test_parse_timeline_range_accepts_hours_and_minutes() -> None:
    assert parse_timeline_range("00:11:38-00:12:08") == (698.0, 728.0)
    assert parse_timeline_range("01:02-01:12") == (62.0, 72.0)


def test_enrolment_requires_two_verified_ranges(tmp_path: Path) -> None:
    audio = tmp_path / "analysis.wav"
    _wav(audio, seconds=10)
    with pytest.raises(ValueError, match="at least 2"):
        enrol_known_speaker(
            work_dir=tmp_path / "working",
            job_id="job",
            analysis_audio_path=audio,
            name="Samkelo Maseko",
            gender="male",
            ranges=["00:00-00:05"],
            encoder=FakeEncoder(),
        )


def test_enrol_and_identify_verified_speaker(tmp_path: Path) -> None:
    audio = tmp_path / "analysis.wav"
    _wav(audio)
    work_dir = tmp_path / "working"

    result = enrol_known_speaker(
        work_dir=work_dir,
        job_id="job-123",
        analysis_audio_path=audio,
        name="Samkelo Maseko",
        gender="male",
        ranges=["00:00:05-00:00:15", "00:00:20-00:00:30"],
        encoder=FakeEncoder(
            {
                "sample_01.wav": [1.0, 0.02, 0.0],
                "sample_02.wav": [0.99, -0.01, 0.0],
            }
        ),
    )

    assert result["person_id"] == "samkelo_maseko"
    assert result["azure_voice"] == "zu-ZA-ThembaNeural"
    profile = json.loads(Path(result["profile_path"]).read_text())
    assert profile["model"] == {
        "id": SPEECHBRAIN_MODEL_ID,
        "revision": SPEECHBRAIN_MODEL_REVISION,
        "embedding_dimensions": 3,
    }

    match = identify_known_speaker(
        audio,
        work_dir=work_dir,
        encoder=FakeEncoder({"analysis.wav": [0.98, 0.01, 0.0]}),
    )
    assert match is not None
    assert match.person_id == "samkelo_maseko"
    assert match.gender == "male"
    assert match.azure_voice == "zu-ZA-ThembaNeural"


def test_identification_rejects_ambiguous_top_matches(tmp_path: Path) -> None:
    root = tmp_path / "working" / "known_speakers"
    profiles = root / "profiles"
    profiles.mkdir(parents=True)
    speakers = {}
    for person_id, embedding in (
        ("first_person", [1.0, 0.0]),
        ("second_person", [0.999, 0.045]),
    ):
        profile = profiles / f"{person_id}.json"
        profile.write_text(json.dumps({"centroid_embedding": embedding}))
        speakers[person_id] = {
            "person_id": person_id,
            "display_name": person_id,
            "gender": "male",
            "azure_voice": "zu-ZA-ThembaNeural",
            "profile_path": str(profile),
            "verified": True,
            "model_id": SPEECHBRAIN_MODEL_ID,
            "model_revision": SPEECHBRAIN_MODEL_REVISION,
        }
    (root / "registry.json").write_text(
        json.dumps(
            {
                "schema_version": "mathula-known-speakers-v1",
                "speakers": speakers,
            }
        )
    )
    probe = tmp_path / "probe.wav"
    _wav(probe, seconds=4)

    assert (
        identify_known_speaker(
            probe,
            work_dir=tmp_path / "working",
            encoder=FakeEncoder({"probe.wav": [1.0, 0.02]}),
        )
        is None
    )


def test_identification_without_registry_does_not_load_model(tmp_path: Path) -> None:
    probe = tmp_path / "probe.wav"
    _wav(probe, seconds=4)
    assert identify_known_speaker(probe, work_dir=tmp_path / "working") is None
