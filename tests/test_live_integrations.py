"""Explicit opt-in integration gates; never selected by the offline suite."""

from __future__ import annotations

import importlib
import os
import uuid
from pathlib import Path

import pytest

from mathula_tv.ai_provider import build_full_clip_payload, create_production_ai_provider
from mathula_tv.azure_stt import AzureFastTranscriptionBackend, derive_endpoint
from mathula_tv.cli import create_tts_backend
from mathula_tv.config import load_settings
from mathula_tv.gcs_store import GCSStore, LeaseConflict


def enabled(name: str) -> None:
    if os.getenv(name) != "1":
        pytest.skip(f"set {name}=1 to run this paid/network/GPU integration gate")


def dotted_callable(variable: str):
    value = os.getenv(variable, "")
    if ":" not in value:
        pytest.skip(f"set {variable}=module:callable for the configured live runtime")
    module, name = value.split(":", 1)
    return getattr(importlib.import_module(module), name)


@pytest.mark.live
def test_live_anthropic_short_translation():
    enabled("MATHULA_TV_RUN_LIVE_ANTHROPIC_TESTS")
    provider = create_production_ai_provider()
    unit = {
        "unit_id": "unit_0001",
        "speaker_id": "AZURE_1",
        "source_text": "The hearing begins on 25 July 2026.",
        "start_ms": 0,
        "preferred_end_ms": 3500,
        "hard_end_ms": 3800,
        "preferred_duration_ms": 3500,
        "maximum_duration_ms": 3800,
    }
    payload = build_full_clip_payload(
        transcript={"schema_version": "transcript-v1", "segments": [unit]},
        dubbing_units=[unit],
        protected_numbers=["25", "2026"],
        protected_dates=["25 July 2026"],
    )
    response = provider.translate_full_clip(payload)
    assert response.data["units"][0]["spoken_text"]
    assert response.metadata.provider == "anthropic"


@pytest.mark.live
def test_live_azure_voice_discovery_and_short_synthesis(tmp_path):
    enabled("MATHULA_TV_RUN_LIVE_AZURE_TTS_TESTS")
    settings = load_settings()
    backend = create_tts_backend(settings)
    voices = backend.validate_configured_voices(force_refresh=True)
    assert voices
    from mathula_tv.azure_tts import AzureTTSRequest

    result = backend.synthesize(
        AzureTTSRequest(
            turn_id="live_test_0001",
            speaker_id="TEST_ONLY",
            text="Sawubona, lokhu ukuhlolwa okufushane.",
            preferred_duration_ms=4000,
            maximum_duration_ms=5000,
        ),
        tmp_path / "azure_live.wav",
        force=True,
    )
    assert result.duration_ms > 0 and result.sha256


@pytest.mark.live
def test_live_azure_stt_retranscription_fixture():
    enabled("MATHULA_TV_RUN_LIVE_AZURE_STT_TESTS")
    audio = Path(os.environ["MATHULA_TV_LIVE_AZURE_STT_AUDIO"])
    settings = load_settings()
    backend = AzureFastTranscriptionBackend(
        settings.azure_speech_endpoint or derive_endpoint(settings.azure_speech_region),
        os.environ["AZURE_SPEECH_KEY"],
        timeout_seconds=settings.azure_speech_timeout_seconds,
    )
    result = backend.transcribe(audio, os.getenv("MATHULA_TV_LIVE_STT_LOCALE", "zu-ZA"))
    assert result["phrases"]


@pytest.mark.live
def test_live_gcs_lease_contention():
    enabled("MATHULA_TV_RUN_LIVE_GCS_TESTS")
    settings = load_settings()
    store = GCSStore(settings.gcs_bucket, settings.gcs_prefix)
    job_id = f"integration-{uuid.uuid4().hex}"
    claim = store.claim(job_id, "lease_test", "worker-one", 1, lease_seconds=60)
    with pytest.raises(LeaseConflict):
        store.claim(job_id, "lease_test", "worker-two", 1, lease_seconds=60)
    completed = store.finish_claim(claim, "completed")
    assert completed["status"] == "completed"


@pytest.mark.live
def test_live_openvoice_conversion_runner():
    enabled("MATHULA_TV_RUN_LIVE_OPENVOICE_TESTS")
    result = dotted_callable("MATHULA_TV_OPENVOICE_LIVE_TEST_RUNNER")()
    assert result["backend"] == "openvoice" and result["output_sha256"]


@pytest.mark.live
def test_live_openvoice_source_and_target_embedding_runner():
    enabled("MATHULA_TV_RUN_LIVE_OPENVOICE_TESTS")
    result = dotted_callable("MATHULA_TV_OPENVOICE_ASSET_LIVE_TEST_RUNNER")()
    assert result["source_embedding_sha256"]
    assert result["target_embedding_sha256"]
    assert result["openvoice_revision"]


@pytest.mark.live
def test_live_speaker_similarity_runner():
    enabled("MATHULA_TV_RUN_LIVE_OPENVOICE_TESTS")
    result = dotted_callable("MATHULA_TV_SPEAKER_SIMILARITY_LIVE_TEST_RUNNER")()
    assert 0 <= float(result["speaker_similarity"]) <= 1
    assert result["reference_audio_sha256"]
    assert result["converted_audio_sha256"]


@pytest.mark.live
def test_live_gcs_partial_promotion_recovery_runner():
    enabled("MATHULA_TV_RUN_LIVE_GCS_TESTS")
    result = dotted_callable("MATHULA_TV_GCS_RECOVERY_LIVE_TEST_RUNNER")()
    assert result["partial_detected"] is True
    assert result["final_hash_verified"] is True
    assert result["stale_worker_blocked"] is True


@pytest.mark.live
def test_live_final_synthetic_render_runner():
    enabled("MATHULA_TV_RUN_LIVE_RENDER_TESTS")
    result = dotted_callable("MATHULA_TV_RENDER_LIVE_TEST_RUNNER")()
    assert result["video_stream"] is True
    assert result["audio_stream"] is True
    assert result["duration_match"] is True
    assert result["output_sha256"]


@pytest.mark.live
def test_live_complete_four_turn_compatibility_job():
    for flag in (
        "MATHULA_TV_RUN_LIVE_AZURE_STT_TESTS",
        "MATHULA_TV_RUN_LIVE_AZURE_TTS_TESTS",
        "MATHULA_TV_RUN_LIVE_ANTHROPIC_TESTS",
        "MATHULA_TV_RUN_LIVE_GCS_TESTS",
        "MATHULA_TV_RUN_LIVE_OPENVOICE_TESTS",
        "MATHULA_TV_RUN_LIVE_FOUR_TURN_TESTS",
    ):
        enabled(flag)
    result = dotted_callable("MATHULA_TV_FOUR_TURN_LIVE_TEST_RUNNER")(
        "19ba6d69f1b84132ba4f20599101834a",
        live_operation=True,
        preserve_translation=True,
    )
    assert result["job_id"] == "19ba6d69f1b84132ba4f20599101834a"
    assert result["compatibility_gate"] in {
        "passed",
        "failed_pronunciation",
        "failed_intelligibility",
        "failed_identity",
        "failed_timing",
        "failed_audio_quality",
        "needs_human_review",
    }
