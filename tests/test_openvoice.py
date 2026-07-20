from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import wave
from dataclasses import replace
from pathlib import Path

import pytest

from mathula_tv.openvoice import (
    CheckpointAsset,
    OpenVoiceAttemptLimitExceeded,
    OpenVoiceAssetHashMismatch,
    OpenVoiceBackend,
    OpenVoiceCheckpointHashMismatch,
    OpenVoiceCheckpointMissing,
    OpenVoiceConversionRequest,
    OpenVoiceCudaOutOfMemory,
    OpenVoiceCudaUnavailable,
    OpenVoiceInferenceError,
    OpenVoiceInvalidOutput,
    OpenVoiceLeaseLost,
    OpenVoicePins,
    OpenVoiceRevisionMismatch,
    OpenVoiceRuntimeInfo,
    OpenVoiceRuntimeMismatch,
    sha256_file,
)
from mathula_tv.openvoice_worker import (
    AzureSourceCalibrationCache,
    AzureSourceCalibrationSpec,
    OpenVoiceAssetPlan,
    OpenVoiceAssetWorker,
    OpenVoicePlan,
    OpenVoiceSourceAsset,
    OpenVoiceSpeakerAsset,
    OpenVoiceUnit,
    OpenVoiceVoiceCandidate,
    OpenVoiceWorker,
    write_openvoice_asset_plan,
)


def write_wav(path: Path, *, frames: int = 2400, rate: int = 24000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as audio:
        audio.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        audio.writeframes((1000).to_bytes(2, "little", signed=True) * frames)


class FakeRuntime:
    def __init__(self) -> None:
        self.load_calls = 0
        self.convert_calls: list[str] = []
        self.embedding_calls = 0
        self.cuda_available = True
        self.revision = "revision-abc123"
        self.python_version = "3.11.9"
        self.torch_version = "2.2.2+cu121"
        self.fail_unit: str | None = None
        self.oom = False
        self.invalid_output = False

    def inspect(self) -> OpenVoiceRuntimeInfo:
        return OpenVoiceRuntimeInfo(
            openvoice_revision=self.revision,
            python_version=self.python_version,
            torch_version=self.torch_version,
            cuda_available=self.cuda_available,
            cuda_version="12.1",
            gpu_name="Synthetic GPU",
        )

    def load_model(self, **kwargs):
        self.load_calls += 1
        return {"loaded": kwargs}

    def extract_embedding(self, model, *, audio_path, output_path, embedding_kind):
        self.embedding_calls += 1
        output_path.write_bytes(f"{embedding_kind}-embedding".encode())

    def convert(
        self,
        model,
        *,
        source_audio_path,
        source_embedding_path,
        target_embedding_path,
        output_path,
        target_language,
        settings,
    ):
        unit = output_path.name.split(".")[1]
        self.convert_calls.append(unit)
        if self.oom:
            raise RuntimeError("CUDA out of memory: private-worker-detail")
        if self.fail_unit == unit:
            self.fail_unit = None
            raise RuntimeError("Bearer secret-worker-token")
        if self.invalid_output:
            output_path.write_bytes(b"not-a-wave")
            return
        write_wav(output_path, frames=2500)


def make_backend(tmp_path: Path, runtime: FakeRuntime | None = None):
    runtime = runtime or FakeRuntime()
    tmp_path.mkdir(parents=True, exist_ok=True)
    checkpoint = tmp_path / "converter.pth"
    checkpoint.write_bytes(b"pinned-checkpoint")
    pins = OpenVoicePins(
        revision="revision-abc123",
        python_version="3.11.9",
        torch_version="2.2.2+cu121",
        checkpoints=(
            CheckpointAsset(
                "converter", checkpoint, hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            ),
        ),
    )
    return OpenVoiceBackend(runtime, pins), runtime


def make_request(tmp_path: Path, unit_id: str = "unit_0001"):
    source = tmp_path / f"{unit_id}-azure.wav"
    source_embedding = tmp_path / "source.bin"
    target_embedding = tmp_path / f"{unit_id}-target.bin"
    write_wav(source)
    source_embedding.write_bytes(b"azure-source-embedding")
    target_embedding.write_bytes(f"target-{unit_id}".encode())
    return OpenVoiceConversionRequest(
        job_id="job-123",
        unit_id=unit_id,
        speaker_id="SPEAKER_00",
        source_audio_path=source,
        source_audio_sha256=sha256_file(source),
        source_embedding_path=source_embedding,
        source_embedding_sha256=sha256_file(source_embedding),
        target_embedding_path=target_embedding,
        target_embedding_sha256=sha256_file(target_embedding),
        target_language="zu-ZA",
        azure_voice="zu-ZA-ThandoNeural",
        conversion_settings={"tau": 0.3},
        output_path=tmp_path / "output" / f"{unit_id}.wav",
        lease_identity="worker-123",
    )


def test_normal_import_does_not_import_torch_or_gpu_openvoice(tmp_path):
    script = """
import builtins
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    level = kwargs.get('level', args[3] if len(args) > 3 else 0)
    if name == 'torch' or (level == 0 and name.startswith('openvoice')):
        raise AssertionError('GPU dependency imported during normal import: ' + name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
import mathula_tv.openvoice
import mathula_tv.openvoice_worker
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
    completed = subprocess.run(
        [sys.executable, "-c", script],
        text=True,
        capture_output=True,
        env=environment,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_pinned_model_is_loaded_once_for_multiple_units(tmp_path):
    backend, runtime = make_backend(tmp_path)
    first = backend.convert(make_request(tmp_path, "unit_0001"))
    second = backend.convert(make_request(tmp_path, "unit_0002"))
    assert runtime.load_calls == 1
    assert first.backend == "openvoice"
    assert first.backend_revision == "revision-abc123"
    assert second.duration_ratio == pytest.approx(1.04, abs=1e-6)
    assert second.output_sha256 == sha256_file(tmp_path / "output" / "unit_0002.wav")


def test_checkpoint_must_exist_and_match_before_runtime_load(tmp_path):
    backend, runtime = make_backend(tmp_path)
    checkpoint = backend.pins.checkpoints[0].path
    checkpoint.unlink()
    with pytest.raises(OpenVoiceCheckpointMissing):
        backend.load()
    assert runtime.load_calls == 0

    checkpoint.write_bytes(b"wrong")
    with pytest.raises(OpenVoiceCheckpointHashMismatch):
        backend.load()
    assert runtime.load_calls == 0


def test_cuda_and_input_hash_failures_are_explicit(tmp_path):
    backend, runtime = make_backend(tmp_path)
    runtime.cuda_available = False
    with pytest.raises(OpenVoiceCudaUnavailable):
        backend.load()

    backend, _ = make_backend(tmp_path / "other")
    request = make_request(tmp_path / "other", "unit_0001")
    request.source_embedding_path.write_bytes(b"tampered")
    with pytest.raises(OpenVoiceAssetHashMismatch, match="source embedding"):
        backend.convert(request)

    request = make_request(tmp_path / "other", "unit_0002")
    request.target_embedding_path.write_bytes(b"tampered")
    with pytest.raises(OpenVoiceAssetHashMismatch, match="target speaker embedding"):
        backend.convert(request)


def test_revision_runtime_and_invalid_output_fail_closed(tmp_path):
    backend, runtime = make_backend(tmp_path / "revision")
    runtime.revision = "unexpected-revision"
    with pytest.raises(OpenVoiceRevisionMismatch):
        backend.load()

    backend, runtime = make_backend(tmp_path / "python")
    runtime.python_version = "3.12.1"
    with pytest.raises(OpenVoiceRuntimeMismatch, match="Python"):
        backend.load()

    backend, runtime = make_backend(tmp_path / "output")
    runtime.invalid_output = True
    request = make_request(tmp_path / "output")
    with pytest.raises(OpenVoiceInvalidOutput, match="readable PCM WAV"):
        backend.convert(request)
    assert not request.output_path.exists()


def test_cuda_oom_and_runtime_errors_are_classified_and_secret_safe(tmp_path):
    backend, runtime = make_backend(tmp_path)
    runtime.oom = True
    with pytest.raises(OpenVoiceCudaOutOfMemory) as error:
        backend.convert(make_request(tmp_path))
    assert "private-worker-detail" not in str(error.value)

    runtime.oom = False
    runtime.fail_unit = "unit_0002"
    with pytest.raises(OpenVoiceInferenceError) as error:
        backend.convert(make_request(tmp_path, "unit_0002"))
    assert "secret-worker-token" not in str(error.value)


def make_plan(tmp_path: Path, count: int = 2):
    units = []
    for index in range(1, count + 1):
        request = make_request(tmp_path, f"unit_{index:04d}")
        units.append(
            OpenVoiceUnit(
                unit_id=request.unit_id,
                speaker_id=request.speaker_id,
                azure_voice=request.azure_voice,
                target_language=request.target_language,
                source_audio_path=request.source_audio_path,
                source_audio_sha256=request.source_audio_sha256,
                source_embedding_path=request.source_embedding_path,
                source_embedding_sha256=request.source_embedding_sha256,
                target_embedding_path=request.target_embedding_path,
                target_embedding_sha256=request.target_embedding_sha256,
                output_path=request.output_path,
                conversion_settings=request.conversion_settings,
            )
        )
    return OpenVoicePlan(
        job_id="job-123",
        lease_identity="worker-123",
        openvoice_revision="revision-abc123",
        units=tuple(units),
    )


def test_worker_persists_partial_manifest_and_resumes_only_pending_units(tmp_path):
    backend, runtime = make_backend(tmp_path)
    plan = make_plan(tmp_path)
    runtime.fail_unit = "unit_0002"
    manifest_path = tmp_path / "manifest.json"
    worker = OpenVoiceWorker(backend, plan, manifest_path)

    with pytest.raises(OpenVoiceInferenceError):
        worker.run()
    partial = json.loads(manifest_path.read_text())
    assert partial["state"] == "partial"
    assert partial["units"]["unit_0001"]["status"] == "completed"
    assert partial["units"]["unit_0002"]["last_error"]["retryable"] is True

    completed = worker.run()
    assert completed["state"] == "completed"
    assert runtime.convert_calls == ["unit_0001", "unit_0002", "unit_0002"]
    assert runtime.load_calls == 1
    assert completed["units"]["unit_0002"]["attempts"] == 2


def test_worker_force_rerun_preserves_attempt_history(tmp_path):
    backend, runtime = make_backend(tmp_path)
    worker = OpenVoiceWorker(
        backend, make_plan(tmp_path, count=1), tmp_path / "manifest.json"
    )
    worker.run()
    rerun = worker.run(force=True)
    record = rerun["units"]["unit_0001"]
    assert record["attempts"] == 2
    assert len(record["history"]) == 1
    assert runtime.convert_calls == ["unit_0001", "unit_0001"]


def test_worker_enforces_attempt_limit_and_lease_ownership(tmp_path):
    backend, runtime = make_backend(tmp_path / "attempts")
    plan = make_plan(tmp_path / "attempts", count=1)
    runtime.fail_unit = "unit_0001"
    worker = OpenVoiceWorker(
        backend,
        plan,
        tmp_path / "attempts" / "manifest.json",
        max_attempts_per_unit=1,
    )
    with pytest.raises(OpenVoiceInferenceError):
        worker.run()
    with pytest.raises(OpenVoiceAttemptLimitExceeded) as error:
        worker.run()
    assert getattr(error.value, "code", None) == "openvoice_attempt_limit_exceeded"
    assert runtime.convert_calls == ["unit_0001"]

    backend, runtime = make_backend(tmp_path / "lease")
    worker = OpenVoiceWorker(
        backend,
        make_plan(tmp_path / "lease", count=1),
        tmp_path / "lease" / "manifest.json",
        lease_guard=lambda unit_id: False,
    )
    with pytest.raises(OpenVoiceLeaseLost):
        worker.run()
    assert runtime.convert_calls == []


def test_worker_rejects_changed_plan_inputs_during_resume(tmp_path):
    backend, _ = make_backend(tmp_path)
    plan = make_plan(tmp_path, count=1)
    manifest_path = tmp_path / "manifest.json"
    OpenVoiceWorker(backend, plan, manifest_path).run()
    changed_unit = replace(plan.units[0], conversion_settings={"tau": 0.99})
    changed_plan = replace(plan, units=(changed_unit,))
    with pytest.raises(ValueError, match="inputs changed"):
        OpenVoiceWorker(backend, changed_plan, manifest_path).prepare_manifest()


def test_reclaimed_worker_requires_explicit_verified_lease_rebind(tmp_path):
    backend, runtime = make_backend(tmp_path)
    plan = make_plan(tmp_path)
    runtime.fail_unit = "unit_0002"
    manifest_path = tmp_path / "manifest.json"
    with pytest.raises(OpenVoiceInferenceError):
        OpenVoiceWorker(backend, plan, manifest_path).run()

    reclaimed_plan = replace(plan, lease_identity="worker-456")
    with pytest.raises(ValueError, match="another lease"):
        OpenVoiceWorker(backend, reclaimed_plan, manifest_path).prepare_manifest()

    reclaimed_backend, reclaimed_runtime = make_backend(tmp_path)
    completed = OpenVoiceWorker(
        reclaimed_backend,
        reclaimed_plan,
        manifest_path,
        lease_guard=lambda unit_id: True,
        allow_lease_rebind=True,
    ).run()
    assert completed["state"] == "completed"
    assert completed["lease_identity"] == "worker-456"
    assert completed["lease_history"][0]["lease_identity"] == "worker-123"
    assert reclaimed_runtime.convert_calls == ["unit_0002"]


def test_calibration_cache_is_stable_and_revision_specific(tmp_path):
    backend, runtime = make_backend(tmp_path)
    text = "Umbhalo wokulinganisa obuyekeziwe."
    text_hash = hashlib.sha256(text.encode()).hexdigest()
    spec = AzureSourceCalibrationSpec(
        azure_voice="zu-ZA-ThandoNeural",
        region="southafricanorth",
        output_format="riff-24khz-16bit-mono-pcm",
        calibration_text_version="reviewed-v1",
        calibration_text_sha256=text_hash,
        openvoice_revision="revision-abc123",
        minimum_duration_seconds=0.09,
        maximum_duration_seconds=0.2,
    )
    syntheses = []

    def synthesize(value, output):
        syntheses.append(value)
        write_wav(output, frames=2400)

    cache = AzureSourceCalibrationCache(tmp_path / "cache")
    first = cache.ensure(spec, text, synthesize=synthesize, backend=backend)
    second = cache.ensure(spec, text, synthesize=synthesize, backend=backend)
    assert first == second
    assert len(syntheses) == 1
    assert runtime.embedding_calls == 1
    assert first["azure_voice"] == "zu-ZA-ThandoNeural"
    assert first["embedding_sha256"] == sha256_file(
        tmp_path / "cache" / "zu-ZA-ThandoNeural" / "source_embedding.bin"
    )

    changed_revision = AzureSourceCalibrationSpec(
        **{**spec.__dict__, "openvoice_revision": "revision-new"}
    )
    assert cache.get(changed_revision) is None


def test_calibration_cache_detects_embedding_tampering(tmp_path):
    backend, _ = make_backend(tmp_path)
    text = "Reviewed calibration"
    spec = AzureSourceCalibrationSpec(
        azure_voice="zu-ZA-ThembaNeural",
        region="southafricanorth",
        output_format="riff-24khz-16bit-mono-pcm",
        calibration_text_version="v1",
        calibration_text_sha256=hashlib.sha256(text.encode()).hexdigest(),
        openvoice_revision="revision-abc123",
        minimum_duration_seconds=0.09,
        maximum_duration_seconds=0.2,
    )
    cache = AzureSourceCalibrationCache(tmp_path / "cache")
    cache.ensure(
        spec,
        text,
        synthesize=lambda value, path: write_wav(path),
        backend=backend,
    )
    (cache.directory(spec.azure_voice) / "source_embedding.bin").write_bytes(b"bad")
    with pytest.raises(OpenVoiceAssetHashMismatch, match="embedding"):
        cache.get(spec)


def make_asset_plan(
    tmp_path: Path,
    *,
    voices: tuple[str, ...] = ("zu-ZA-ThandoNeural", "zu-ZA-ThembaNeural"),
    with_candidates: bool = True,
) -> OpenVoiceAssetPlan:
    sources = []
    for voice in voices:
        root = tmp_path / "azure_sources" / voice
        calibration_audio = root / "calibration.wav"
        write_wav(calibration_audio)
        text_hash = hashlib.sha256(f"reviewed-{voice}".encode()).hexdigest()
        calibration = AzureSourceCalibrationSpec(
            azure_voice=voice,
            region="southafricanorth",
            output_format="riff-24khz-16bit-mono-pcm",
            calibration_text_version="reviewed-v1",
            calibration_text_sha256=text_hash,
            openvoice_revision="revision-abc123",
            minimum_duration_seconds=0.09,
            maximum_duration_seconds=0.2,
        )
        calibration_manifest = root / "manifest.json"
        calibration_manifest.write_text(
            json.dumps(
                {
                    "schema_version": "azure-openvoice-source-calibration-v1",
                    **calibration.identity(),
                    "wav_sha256": sha256_file(calibration_audio),
                    "embedding_sha256": None,
                    "status": "source_embedding_pending_gpu_worker",
                }
            ),
            encoding="utf-8",
        )
        sources.append(
            OpenVoiceSourceAsset(
                azure_voice=voice,
                calibration_audio_path=calibration_audio,
                calibration_audio_sha256=sha256_file(calibration_audio),
                calibration_manifest_path=calibration_manifest,
                calibration_manifest_sha256=sha256_file(calibration_manifest),
                source_embedding_path=root / "source_embedding.bin",
                source_embedding_manifest_path=root / "source_embedding_manifest.json",
                calibration=calibration,
            )
        )

    speaker_root = tmp_path / "speakers" / "SPEAKER_00"
    reference_audio = speaker_root / "reference_reel.wav"
    write_wav(reference_audio)
    reference_manifest = speaker_root / "reference_manifest.json"
    reference_manifest.write_text(
        json.dumps(
            {
                "schema_version": "speaker-reference-manifest-v1",
                "speaker_id": "SPEAKER_00",
                "reference_reel_sha256": sha256_file(reference_audio),
                "job_local_identity": True,
                "global_identity_created": False,
            }
        ),
        encoding="utf-8",
    )
    candidates = []
    if with_candidates:
        sentence_hash = hashlib.sha256(b"reviewed candidate sentence").hexdigest()
        for index, voice in enumerate(voices, 1):
            source_audio = speaker_root / "azure_candidates" / f"{voice}.wav"
            write_wav(source_audio, frames=2400 + index * 100)
            candidates.append(
                OpenVoiceVoiceCandidate(
                    candidate_id=f"candidate_{index:02d}",
                    azure_voice=voice,
                    source_audio_path=source_audio,
                    source_audio_sha256=sha256_file(source_audio),
                    output_path=speaker_root
                    / "converted_candidates"
                    / f"candidate_{index:02d}.wav",
                    calibration_text_sha256=sentence_hash,
                    conversion_settings={"tau": 0.3},
                )
            )
    speaker = OpenVoiceSpeakerAsset(
        speaker_id="SPEAKER_00",
        reference_audio_path=reference_audio,
        reference_audio_sha256=sha256_file(reference_audio),
        reference_manifest_path=reference_manifest,
        reference_manifest_sha256=sha256_file(reference_manifest),
        target_embedding_path=speaker_root / "target_embedding.bin",
        target_embedding_manifest_path=speaker_root
        / "target_embedding_manifest.json",
        candidate_manifest_path=(
            speaker_root / "azure_voice_candidates.json" if candidates else None
        ),
        voice_candidates=tuple(candidates),
    )
    return OpenVoiceAssetPlan(
        job_id="job-123",
        lease_identity="worker-123",
        openvoice_revision="revision-abc123",
        sources=tuple(sources),
        speakers=(speaker,),
    )


def test_asset_plan_is_deterministic_self_verifying_and_revision_bound(tmp_path):
    plan = make_asset_plan(tmp_path)
    path = tmp_path / "asset_plan.json"
    first = write_openvoice_asset_plan(plan, path)
    second = write_openvoice_asset_plan(plan, path)
    assert first == second
    assert first["artifact_sha256"] == plan.artifact_sha256
    loaded = OpenVoiceAssetPlan.from_dict(json.loads(path.read_text()))
    assert loaded == plan

    tampered = json.loads(path.read_text())
    tampered["sources"][0]["calibration_audio_sha256"] = "0" * 64
    with pytest.raises(OpenVoiceAssetHashMismatch, match="artifact hash"):
        OpenVoiceAssetPlan.from_dict(tampered)


def test_asset_worker_extracts_cached_embeddings_and_optional_candidates(tmp_path):
    backend, runtime = make_backend(tmp_path / "runtime")
    plan = make_asset_plan(tmp_path / "assets")
    manifest_path = tmp_path / "assets" / "asset_manifest.json"
    worker = OpenVoiceAssetWorker(backend, plan, manifest_path)

    manifest = worker.run()
    assert manifest["state"] == "completed"
    assert manifest["completed_assets"] == 5
    assert manifest["pending_assets"] == 0
    assert runtime.load_calls == 1
    assert runtime.embedding_calls == 3
    assert runtime.convert_calls == ["candidate_01", "candidate_02"]
    assert worker.pending_asset_ids() == []
    assert manifest["checkpoint_hashes"] == backend.checkpoint_hashes

    source = plan.sources[0]
    speaker = plan.speakers[0]
    assert manifest["sources"][source.azure_voice]["result"][
        "source_embedding_sha256"
    ] == sha256_file(source.source_embedding_path)
    assert manifest["sources"][source.azure_voice]["result"][
        "source_embedding_manifest_sha256"
    ] == sha256_file(source.source_embedding_manifest_path)
    assert manifest["speakers"][speaker.speaker_id]["result"][
        "target_embedding_sha256"
    ] == sha256_file(speaker.target_embedding_path)
    assert manifest["speakers"][speaker.speaker_id]["result"][
        "target_embedding_manifest_sha256"
    ] == sha256_file(speaker.target_embedding_manifest_path)
    candidate_manifest = json.loads(speaker.candidate_manifest_path.read_text())
    assert candidate_manifest["state"] == "completed"
    assert {
        item["evaluation_status"] for item in candidate_manifest["candidates"]
    } == {"pending_external_qc"}
    assert all(item["human_review_required"] for item in candidate_manifest["candidates"])

    rerun = worker.run()
    assert rerun["state"] == "completed"
    assert rerun["completed_assets"] == manifest["completed_assets"]
    assert runtime.embedding_calls == 3
    assert runtime.convert_calls == ["candidate_01", "candidate_02"]


def test_asset_worker_records_injected_candidate_evaluation_without_thresholds(
    tmp_path,
):
    backend, _ = make_backend(tmp_path / "runtime")
    plan = make_asset_plan(
        tmp_path / "assets",
        voices=("zu-ZA-ThandoNeural",),
    )
    calls = []

    def evaluate(candidate, conversion, source, speaker):
        calls.append((candidate.candidate_id, source.azure_voice, speaker.speaker_id))
        return {
            "speaker_similarity": 0.73,
            "word_error_approximation": 0.12,
            "audio_artifact_count": len(conversion.warnings),
        }

    worker = OpenVoiceAssetWorker(
        backend,
        plan,
        tmp_path / "assets" / "asset_manifest.json",
        candidate_evaluator=evaluate,
    )
    worker.run()
    candidate_manifest = json.loads(
        plan.speakers[0].candidate_manifest_path.read_text()
    )
    candidate = candidate_manifest["candidates"][0]
    assert candidate["evaluation_status"] == "completed"
    assert candidate["speaker_similarity"] == 0.73
    assert candidate["word_error_approximation"] == 0.12
    assert candidate_manifest["thresholds_used"] is False
    assert calls == [("candidate_01", "zu-ZA-ThandoNeural", "SPEAKER_00")]


def test_asset_worker_validates_calibration_and_same_speaker_identity(tmp_path):
    backend, runtime = make_backend(tmp_path / "runtime")
    plan = make_asset_plan(tmp_path / "assets", with_candidates=False)
    plan.sources[0].calibration_manifest_path.write_text("{}", encoding="utf-8")
    manifest_path = tmp_path / "assets" / "asset_manifest.json"
    with pytest.raises(OpenVoiceAssetHashMismatch, match="manifest hash"):
        OpenVoiceAssetWorker(backend, plan, manifest_path).run()
    assert runtime.embedding_calls == 0
    failed = json.loads(manifest_path.read_text())
    assert failed["state"] == "failed"
    assert failed["sources"][plan.sources[0].azure_voice]["last_error"]["code"] == (
        "openvoice_asset_hash_mismatch"
    )

    plan = make_asset_plan(tmp_path / "identity", with_candidates=False)
    calibration_manifest = plan.sources[0].calibration_manifest_path
    value = json.loads(calibration_manifest.read_text())
    value["region"] = "unexpected-region"
    calibration_manifest.write_text(json.dumps(value), encoding="utf-8")
    changed_source = replace(
        plan.sources[0],
        calibration_manifest_sha256=sha256_file(calibration_manifest),
    )
    changed_plan = replace(
        plan,
        sources=(changed_source, *plan.sources[1:]),
    )
    with pytest.raises(OpenVoiceAssetHashMismatch, match="calibration identity"):
        OpenVoiceAssetWorker(
            backend,
            changed_plan,
            tmp_path / "identity" / "asset_manifest.json",
        ).run()

    plan = make_asset_plan(tmp_path / "other", with_candidates=False)
    reference_manifest = plan.speakers[0].reference_manifest_path
    value = json.loads(reference_manifest.read_text())
    value["speaker_id"] = "SPEAKER_01"
    reference_manifest.write_text(json.dumps(value), encoding="utf-8")
    changed_speaker = replace(
        plan.speakers[0],
        reference_manifest_sha256=sha256_file(reference_manifest),
    )
    changed_plan = replace(plan, speakers=(changed_speaker,))
    with pytest.raises(OpenVoiceAssetHashMismatch, match="identity"):
        OpenVoiceAssetWorker(
            backend, changed_plan, tmp_path / "other" / "asset_manifest.json"
        ).run()


def test_asset_worker_persists_candidate_partial_state_and_resumes(tmp_path):
    backend, runtime = make_backend(tmp_path / "runtime")
    plan = make_asset_plan(tmp_path / "assets")
    manifest_path = tmp_path / "assets" / "asset_manifest.json"
    runtime.fail_unit = "candidate_02"
    worker = OpenVoiceAssetWorker(backend, plan, manifest_path)

    with pytest.raises(OpenVoiceInferenceError):
        worker.run()
    partial = json.loads(manifest_path.read_text())
    speaker_record = partial["speakers"]["SPEAKER_00"]
    assert partial["state"] == "partial"
    assert speaker_record["candidates"]["candidate_01"]["status"] == "completed"
    assert speaker_record["candidates"]["candidate_02"]["status"] == "failed"
    candidate_manifest = json.loads(
        plan.speakers[0].candidate_manifest_path.read_text()
    )
    assert candidate_manifest["state"] == "partial"

    completed = worker.run()
    assert completed["state"] == "completed"
    assert runtime.convert_calls == ["candidate_01", "candidate_02", "candidate_02"]
    assert runtime.load_calls == 1


def seed_source_cache(tmp_path: Path) -> OpenVoiceAssetPlan:
    backend, _ = make_backend(tmp_path / "runtime")
    plan = make_asset_plan(tmp_path / "assets", with_candidates=False)
    OpenVoiceAssetWorker(
        backend,
        plan,
        tmp_path / "assets" / "asset_manifest.json",
    ).run()
    return plan


def copy_source_cache(source_plan: OpenVoiceAssetPlan, target_plan: OpenVoiceAssetPlan):
    assert len(source_plan.sources) == len(target_plan.sources)
    for source, target in zip(
        source_plan.sources, target_plan.sources, strict=True
    ):
        target.source_embedding_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source.source_embedding_path, target.source_embedding_path)
        shutil.copyfile(
            source.source_embedding_manifest_path,
            target.source_embedding_manifest_path,
        )


def test_asset_worker_reuses_valid_cross_job_source_cache_without_inference(tmp_path):
    seeded = seed_source_cache(tmp_path / "seed")
    plan = replace(
        make_asset_plan(tmp_path / "reuse", with_candidates=False),
        job_id="job-456",
    )
    copy_source_cache(seeded, plan)
    backend, runtime = make_backend(tmp_path / "reuse-runtime")

    manifest = OpenVoiceAssetWorker(
        backend,
        plan,
        tmp_path / "reuse" / "asset_manifest.json",
    ).run()

    assert manifest["state"] == "completed"
    for source in plan.sources:
        record = manifest["sources"][source.azure_voice]
        assert record["status"] == "completed"
        assert record["attempts"] == 0
        assert record["result"]["cache_reused"] is True
        assert record["result"]["checkpoint_hashes"] == {
            item.name: item.sha256 for item in backend.pins.checkpoints
        }
    assert runtime.embedding_calls == 1  # target speaker only
    assert runtime.load_calls == 1


def test_asset_worker_rejects_partial_or_tampered_source_cache(tmp_path):
    seeded = seed_source_cache(tmp_path / "seed")

    partial = make_asset_plan(tmp_path / "partial", with_candidates=False)
    partial_source = partial.sources[0]
    partial_source.source_embedding_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        seeded.sources[0].source_embedding_path,
        partial_source.source_embedding_path,
    )
    backend, runtime = make_backend(tmp_path / "partial-runtime")
    partial_manifest = tmp_path / "partial" / "asset_manifest.json"
    with pytest.raises(OpenVoiceAssetHashMismatch, match="both embedding and manifest"):
        OpenVoiceAssetWorker(backend, partial, partial_manifest).run()
    failed = json.loads(partial_manifest.read_text())
    assert failed["sources"][partial_source.azure_voice]["attempts"] == 0
    assert runtime.embedding_calls == 0

    tampered_embedding = make_asset_plan(
        tmp_path / "tampered-embedding", with_candidates=False
    )
    copy_source_cache(seeded, tampered_embedding)
    tampered_embedding.sources[0].source_embedding_path.write_bytes(b"tampered")
    backend, runtime = make_backend(tmp_path / "tampered-embedding-runtime")
    with pytest.raises(OpenVoiceAssetHashMismatch, match="embedding hash"):
        OpenVoiceAssetWorker(
            backend,
            tampered_embedding,
            tmp_path / "tampered-embedding" / "asset_manifest.json",
        ).run()
    assert runtime.embedding_calls == 0

    artifact_tampered = make_asset_plan(
        tmp_path / "artifact-tampered", with_candidates=False
    )
    copy_source_cache(seeded, artifact_tampered)
    artifact_manifest_path = artifact_tampered.sources[0].source_embedding_manifest_path
    artifact_manifest = json.loads(artifact_manifest_path.read_text())
    artifact_manifest["status"] = "tampered"
    artifact_manifest_path.write_text(json.dumps(artifact_manifest), encoding="utf-8")
    backend, runtime = make_backend(tmp_path / "artifact-tampered-runtime")
    with pytest.raises(OpenVoiceAssetHashMismatch, match="artifact hash"):
        OpenVoiceAssetWorker(
            backend,
            artifact_tampered,
            tmp_path / "artifact-tampered" / "asset_manifest.json",
        ).run()
    assert runtime.embedding_calls == 0

    tampered_manifest = make_asset_plan(
        tmp_path / "tampered-manifest", with_candidates=False
    )
    copy_source_cache(seeded, tampered_manifest)
    source_manifest_path = tampered_manifest.sources[0].source_embedding_manifest_path
    source_manifest = json.loads(source_manifest_path.read_text())
    source_manifest["checkpoint_hashes"] = {"converter": "0" * 64}
    unsigned = {
        key: value
        for key, value in source_manifest.items()
        if key != "artifact_sha256"
    }
    source_manifest["artifact_sha256"] = hashlib.sha256(
        json.dumps(
            unsigned,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    source_manifest_path.write_text(json.dumps(source_manifest), encoding="utf-8")
    backend, runtime = make_backend(tmp_path / "tampered-manifest-runtime")
    with pytest.raises(OpenVoiceAssetHashMismatch, match="immutable pins"):
        OpenVoiceAssetWorker(
            backend,
            tampered_manifest,
            tmp_path / "tampered-manifest" / "asset_manifest.json",
        ).run()
    assert runtime.embedding_calls == 0
