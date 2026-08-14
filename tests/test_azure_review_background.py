from __future__ import annotations

from array import array
import wave

from mathula_tv.background import prepare_background
from mathula_tv.models import JobManifest


def _write_stereo_pcm16(path, *, frames: int = 1000, sample_rate: int = 1000) -> None:
    samples = array("h")
    for _ in range(frames):
        samples.extend((10000, -10000))
    with wave.open(str(path), "wb") as handle:
        handle.setparams((2, 2, sample_rate, 0, "NONE", "not compressed"))
        handle.writeframes(samples.tobytes())


def _read_samples(path) -> array:
    with wave.open(str(path), "rb") as handle:
        values = array("h")
        values.frombytes(handle.readframes(handle.getnframes()))
        return values


def test_azure_only_background_uses_review_ducking(tmp_path):
    source = tmp_path / "source.wav"
    output = tmp_path / "background.wav"
    _write_stereo_pcm16(source)

    manifest = prepare_background(
        source,
        output,
        source_dialogue=[{"start": 0.4, "end": 0.6}],
        azure_only=True,
    )

    original = _read_samples(source)
    ducked = _read_samples(output)

    assert manifest["mode"] == "original_mix_ducked_review_only"
    assert manifest["publication_candidate"] is False
    assert manifest["production_authorized"] is False
    assert manifest["readiness"] == "review_preview_only"
    assert ducked[2 * 100] == original[2 * 100]
    assert abs(ducked[2 * 500]) < abs(original[2 * 500])

def test_verified_mix_state_can_transition_to_rendering():
    job = JobManifest(
        job_id="test-job",
        source_filename="source.mp4",
        local_source_path="/tmp/source.mp4",
        source_checksum="sha256",
        source_duration=1.0,
        state="background_ready",
    )

    job.transition("mix_ready")
    job.transition("rendering")

    assert job.state == "rendering"

