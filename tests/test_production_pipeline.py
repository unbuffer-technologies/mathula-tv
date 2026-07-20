import hashlib
import json
import math
import wave
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from mathula_tv.atomic_io import atomic_write_json, read_json
from mathula_tv.azure_tts import AzureTTSResult
from mathula_tv.compatibility import CompatibilityThresholds, UnitCompatibilityObservation
from mathula_tv.config import Settings
from mathula_tv.errors import CompatibilityGateFailure
from mathula_tv.media import checksum
from mathula_tv.openvoice_worker import OpenVoiceAssetPlan
from mathula_tv.quality import inspect_wav
from mathula_tv.production_pipeline import (
    REVIEWED_AZURE_CALIBRATION_TEXT,
    ProductionDubbingPipeline,
)


def settings(tmp_path):
    return Settings(
        tmp_path,
        "bucket",
        "mathula-tv",
        tmp_path / "case",
        tmp_path / "politics",
        "",
        "eastus",
        "en-ZA",
        "2025-10-15",
        10,
        600,
        "",
        "",
        "preview",
        "model",
    )


def make_wav(path, duration_ms=1000, rate=24000):
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = bytearray()
    for index in range(round(duration_ms * rate / 1000)):
        value = round(1000 * math.sin(index / 8))
        frames.extend(value.to_bytes(2, "little", signed=True))
    with wave.open(str(path), "wb") as handle:
        handle.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        handle.writeframes(frames)


def setup_legacy(tmp_path):
    app = ProductionDubbingPipeline(settings(tmp_path))
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    job = app.jobs.create(
        source_filename="source.mp4",
        local_source_path=str(source),
        source_checksum="a" * 64,
        source_duration=2,
    )
    job.state = "azure_tts_queued"
    app.jobs.save(job)
    paths = app.paths(job.job_id)
    paths.analysis_audio.parent.mkdir(parents=True, exist_ok=True)
    paths.analysis_audio.write_bytes(b"analysis")
    paths.mix_source_audio.write_bytes(b"mix")
    transcript = {
        "schema_version": "transcript-v1",
        "duration": 2,
        "segments": [
            {
                "segment_id": "seg-00001",
                "speaker": "AZURE_1",
                "start": 0,
                "end": 2,
                "source_text": "The number is 25.",
                "source_word_ids": ["word_000001", "word_000002", "word_000003", "word_000004"],
                "source_phrase_ids": ["seg-00001"],
                "overlap": False,
            }
        ],
        "words": [
            {"word_id": f"word_{index:06d}", "source_phrase_id": "seg-00001", "text": text, "start": (index - 1) * .4, "end": index * .4, "speaker": "AZURE_1"}
            for index, text in enumerate(["The", "number", "is", "25"], 1)
        ],
    }
    atomic_write_json(app.jobs.job_dir(job.job_id) / "analysis/transcript_en.json", transcript)
    atomic_write_json(
        app.jobs.job_dir(job.job_id) / "translation/transcript_zu.json",
        {"schema_version": "translation-v1", "turns": [{"segment_id": "seg-00001", "translated_text": "Inombolo ingamashumi amabili nanhlanu."}]},
    )
    return app, job


def test_prepare_plan_preserves_legacy_translation_and_three_text(tmp_path):
    app, job = setup_legacy(tmp_path)
    units = app.build_units(job)
    translation = app.upgrade_validated_translation(job)
    plan = app.prepare_dubbing(job)
    assert units["units"][0]["speaker_id"] == "AZURE_1"
    assert translation["ai_called"] is False
    assert plan["translation_identity"]["ai_called_during_upgrade"] is False
    assert plan["units"][0]["spoken_text"] == plan["units"][0]["tts_text"]
    assert app.jobs.load(job.job_id).state == "azure_tts_queued"


class Backend:
    default_voice = "zu-ZA-ThandoNeural"
    output_format = "riff-24khz-16bit-mono-pcm"
    sample_rate = 24000
    channels = 1
    sample_width = 2

    def validate_configured_voices(self, force_refresh=False):
        return (SimpleNamespace(name=self.default_voice),)

    def synthesize(self, request, output, force=False):
        make_wav(output, 1500)
        digest = hashlib.sha256(output.read_bytes()).hexdigest()
        return AzureTTSResult(
            backend="azure_tts",
            turn_id=request.turn_id,
            speaker_id=request.speaker_id,
            voice=request.voice,
            language=request.language,
            input_text_hash="a" * 64,
            ssml_hash="b" * 64,
            output_path=str(output),
            sample_rate=24000,
            channels=1,
            sample_width=2,
            duration_ms=1500,
            preferred_duration_ms=request.preferred_duration_ms,
            maximum_duration_ms=request.maximum_duration_ms,
            rate_percent=request.rate_percent,
            attempt=1,
            sha256=digest,
            warnings=(),
            request_hash="c" * 64,
            ssml_path=str(output.with_suffix(".ssml")),
            manifest_path=str(output.with_suffix(".json")),
            ssml_summary="safe",
        )


class CalibrationBackend(Backend):
    def synthesize(self, request, output, force=False):
        make_wav(output, 21_000)
        result = super().synthesize(request, output, force=force)
        make_wav(output, 21_000)
        return replace(
            result,
            output_path=str(output),
            duration_ms=21_000,
            sha256=checksum(output),
        )


def test_azure_synthesis_advances_only_after_complete_manifest(tmp_path):
    app, job = setup_legacy(tmp_path)
    app.prepare_dubbing(job)
    manifest = app.synthesize_azure(job, Backend())
    saved = app.jobs.load(job.job_id)
    assert saved.state == "azure_tts_ready"
    assert manifest["backend"] == "azure_tts" and len(manifest["units"]) == 1
    assert Path(manifest["units"][0]["output_path"]).is_file()


def test_compatibility_defaults_pending_without_live_measurements(tmp_path):
    app, job = setup_legacy(tmp_path)
    app.prepare_dubbing(job)
    job.state = "voice_conversion_ready"
    app.jobs.save(job)
    report = app.evaluate_compatibility(job)
    assert report["state"] == "pending"
    assert report["production_ready"] is False
    saved = app.jobs.load(job.job_id)
    assert saved.state == "compatibility_review"
    assert saved.review_readiness == "compatibility_pending"


def test_compatibility_pass_advances_to_timing_review_not_failure(tmp_path):
    app, job = setup_legacy(tmp_path)
    plan = app.prepare_dubbing(job)
    paths = app.paths(job.job_id)
    unit_id = plan["units"][0]["unit_id"]
    reference = app.jobs.job_dir(job.job_id) / "speakers/AZURE_1/reference_reel.wav"
    for path in (
        reference,
        paths.azure_turn(unit_id),
        paths.openvoice_turn(unit_id),
        paths.aligned_turn(unit_id),
    ):
        make_wav(path, 1500)
    atomic_write_json(
        paths.alignment_manifest,
        {
            "schema_version": "alignment-manifest-v1",
            "source_backend": "openvoice",
            "readiness": "compatibility_evidence_only",
            "production_authorized": False,
            "units": [
                {
                    "unit_id": unit_id,
                    "output_path": str(paths.aligned_turn(unit_id)),
                    "sha256": checksum(paths.aligned_turn(unit_id)),
                    "final_timing_error_ms": 10,
                }
            ],
        },
    )
    openvoice_quality = inspect_wav(paths.openvoice_turn(unit_id))
    job.state = "voice_conversion_ready"
    app.jobs.save(job)
    reviews = {
        role: {
            "status": "approved",
            "reviewer": "reviewer",
            "reviewed_at": "2026-07-20T12:00:00+00:00",
        }
        for role in ("voice_rights", "editorial", "factual", "isizulu_language")
    }
    observation = UnitCompatibilityObservation(
        unit_id=plan["units"][0]["unit_id"],
        speaker_id="AZURE_1",
        intended_spoken_text=plan["units"][0]["spoken_text"],
        reference_audio_sha256=checksum(reference),
        azure_audio_sha256=checksum(paths.azure_turn(unit_id)),
        openvoice_audio_sha256=checksum(paths.openvoice_turn(unit_id)),
        azure_duration_ms=1500,
        openvoice_duration_ms=1500,
        final_timing_error_ms=10,
        azure_retranscription=plan["units"][0]["spoken_text"],
        openvoice_retranscription=plan["units"][0]["spoken_text"],
        azure_word_error_rate=0,
        openvoice_word_error_rate=0,
        protected_name_accuracy=1,
        number_accuracy=1,
        date_accuracy=1,
        negation_accuracy=1,
        speaker_similarity=0.9,
        azure_speaker_similarity=0.1,
        audio_quality_passed=True,
        clipping_ratio=float(openvoice_quality["clipped_ratio"]),
        silence_ratio=1.0 - float(openvoice_quality["non_silent_ratio"]),
        number_count=0,
        human_reviews=reviews,
        aligned_audio_sha256=checksum(paths.aligned_turn(unit_id)),
    )
    thresholds = CompatibilityThresholds(
        version="calibrated-v1",
        calibrated=True,
        max_openvoice_word_error_rate=0.2,
        max_word_error_rate_degradation=0.1,
        min_protected_name_accuracy=0.9,
        min_number_accuracy=0.9,
        min_date_accuracy=0.9,
        min_negation_accuracy=0.9,
        min_speaker_similarity=0.8,
        max_abs_timing_error_ms=120,
        max_clipping_ratio=0.001,
        min_silence_ratio=0,
        max_silence_ratio=0.5,
    )
    report = app.evaluate_compatibility(job, [observation], thresholds)
    assert report["state"] == "passed"
    assert app.jobs.load(job.job_id).review_readiness == "needs_timing_review"


def test_evidence_only_alignment_cannot_authorize_background(tmp_path):
    app, job = setup_legacy(tmp_path)
    plan = app.prepare_dubbing(job)
    unit_id = plan["units"][0]["unit_id"]
    make_wav(app.paths(job.job_id).openvoice_turn(unit_id), 1500)
    job.state = "compatibility_review"
    job.compatibility_gate = {"state": "pending"}
    app.jobs.save(job)

    manifest = app.align(job, require_compatibility_pass=False)

    assert manifest["readiness"] == "compatibility_evidence_only"
    assert manifest["production_authorized"] is False
    saved = app.jobs.load(job.job_id)
    assert saved.state == "compatibility_review"
    with pytest.raises(CompatibilityGateFailure, match="Evidence-only"):
        app.prepare_background(saved)


def test_speaker_voice_selection_resets_changed_units_to_azure_queue(tmp_path):
    app, job = setup_legacy(tmp_path)
    app.prepare_dubbing(job)
    paths = app.paths(job.job_id)
    atomic_write_json(
        paths.azure_manifest,
        {"units": [{"turn_id": "unit_0001", "voice": "zu-ZA-ThandoNeural"}]},
    )
    atomic_write_json(
        app.jobs.job_dir(job.job_id) / "speakers/AZURE_1/azure_voice_candidates.json",
        {
            "candidates": [
                {
                    "voice": "zu-ZA-ThandoNeural",
                    "speaker_similarity": 0.5,
                    "word_error_approximation": 0.2,
                    "duration_drift_ratio": 1.1,
                    "audio_artifact_count": 1,
                },
                {
                    "voice": "zu-ZA-ThembaNeural",
                    "speaker_similarity": 0.9,
                    "word_error_approximation": 0.05,
                    "duration_drift_ratio": 1.0,
                    "audio_artifact_count": 0,
                },
            ]
        },
    )
    job.state = "azure_tts_ready"
    app.jobs.save(job)
    result = app.persist_speaker_voice_selections(job)
    assert result["azure_resynthesis_required"] is True
    assert result["affected_units"] == ["unit_0001"]
    assert app.jobs.load(job.job_id).state == "azure_tts_queued"


class FakeGCS:
    def __init__(self):
        self.objects = {}
        self.generations = {}

    def upload(self, job_id, relative, path, *, if_generation_match=None):
        key = (job_id, relative)
        if if_generation_match == 0 and key in self.objects:
            raise ValueError("generation conflict")
        self.objects[key] = Path(path).read_bytes()
        self.generations[key] = self.generations.get(key, 0) + 1
        return f"gs://bucket/{job_id}/{relative}"

    def promote(self, job_id, temporary, final, expected, checksum_fn):
        temporary_key = (job_id, temporary)
        final_key = (job_id, final)
        payload = self.objects[temporary_key]
        actual = hashlib.sha256(payload).hexdigest()
        assert actual == expected
        if final_key in self.objects and self.objects[final_key] != payload:
            raise ValueError("conflicting final")
        self.objects[final_key] = payload
        self.generations[final_key] = self.generations.get(final_key, 0) + 1
        del self.objects[temporary_key]
        return f"gs://bucket/{job_id}/{final}"

    def upload_json(self, job_id, relative, value, *, if_generation_match=None):
        key = (job_id, relative)
        if if_generation_match == 0 and key in self.objects:
            raise ValueError("generation conflict")
        self.objects[key] = json.dumps(value).encode()
        self.generations[key] = self.generations.get(key, 0) + 1
        return f"gs://bucket/{job_id}/{relative}"

    def update_json(self, job_id, relative, value, generation):
        key = (job_id, relative)
        assert self.generations[key] == generation
        return self.upload_json(job_id, relative, value)

    def download(self, job_id, relative, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.objects[(job_id, relative)])

    def download_json(self, job_id, relative):
        key = (job_id, relative)
        if key not in self.objects:
            raise FileNotFoundError(relative)
        return json.loads(self.objects[key]), self.generations[key]

    def put_json(self, job_id, relative, value):
        key = (job_id, relative)
        self.objects[key] = json.dumps(value, sort_keys=True).encode()
        self.generations[key] = self.generations.get(key, 0) + 1

    def put_file(self, job_id, relative, path):
        key = (job_id, relative)
        self.objects[key] = path.read_bytes()
        self.generations[key] = self.generations.get(key, 0) + 1


def setup_openvoice_asset_job(tmp_path):
    app, created = setup_legacy(tmp_path)
    configured = replace(
        settings(tmp_path),
        openvoice_revision="a" * 40,
        openvoice_checkpoint_hashes=(("converter", "b" * 64),),
        azure_tts_voices=("zu-ZA-ThandoNeural",),
    )
    app = ProductionDubbingPipeline(configured)
    job = app.jobs.load(created.job_id)
    app.prepare_dubbing(job)
    paths = app.paths(job.job_id)
    make_wav(paths.azure_turn("unit_0001"), 1500)
    atomic_write_json(
        paths.azure_manifest,
        {
            "units": [
                {
                    "turn_id": "unit_0001",
                    "voice": "zu-ZA-ThandoNeural",
                    "sha256": checksum(paths.azure_turn("unit_0001")),
                }
            ]
        },
    )
    source_root = tmp_path / "cache/openvoice/azure_sources/zu-ZA-ThandoNeural"
    source_audio = source_root / "calibration.wav"
    make_wav(source_audio, 21_000)
    text_hash = hashlib.sha256(
        REVIEWED_AZURE_CALIBRATION_TEXT.encode("utf-8")
    ).hexdigest()
    atomic_write_json(
        source_root / "manifest.json",
        {
            "schema_version": "azure-openvoice-source-calibration-v1",
            "azure_voice": "zu-ZA-ThandoNeural",
            "region": "eastus",
            "output_format": "riff-24khz-16bit-mono-pcm",
            "calibration_text_version": "reviewed-zu-v1",
            "calibration_text_sha256": text_hash,
            "wav_sha256": checksum(source_audio),
            "openvoice_revision": "a" * 40,
            "sample_rate": 24000,
            "channels": 1,
            "sample_width": 2,
            "embedding_sha256": None,
            "status": "source_embedding_pending_gpu_worker",
        },
    )
    speaker_root = app.jobs.job_dir(job.job_id) / "speakers/AZURE_1"
    reference = speaker_root / "reference_reel.wav"
    make_wav(reference, 1000)
    atomic_write_json(
        speaker_root / "reference_manifest.json",
        {
            "schema_version": "speaker-reference-manifest-v1",
            "speaker_id": "AZURE_1",
            "reference_reel_sha256": checksum(reference),
            "job_local_identity": True,
            "global_identity_created": False,
        },
    )
    candidate = speaker_root / "voice_calibration/azure/zu-ZA-ThandoNeural.wav"
    make_wav(candidate, 800)
    sentence_hash = hashlib.sha256(b"reviewed sentence").hexdigest()
    atomic_write_json(
        speaker_root / "voice_calibration/input_manifest.json",
        {
            "speaker_id": "AZURE_1",
            "calibration_sentence_sha256": sentence_hash,
            "azure_candidates": [
                {
                    "voice": "zu-ZA-ThandoNeural",
                    "sha256": checksum(candidate),
                    "sample_rate": 24000,
                }
            ],
        },
    )
    job.state = "azure_tts_ready"
    app.jobs.save(job)
    return app, job


def _artifact(app, value):
    value["artifact_sha256"] = app._canonical_hash(value)
    return value


def test_openvoice_asset_plan_queue_and_reconciliation_are_server_bound(tmp_path):
    app, job = setup_openvoice_asset_job(tmp_path)
    plan_value = app.prepare_openvoice_assets(job, proof_of_concept=True)
    plan = OpenVoiceAssetPlan.from_dict(plan_value)
    assert plan.lease_identity == "unbound"
    assert plan.sources[0].calibration_audio_path == Path(
        "dubbing/openvoice/assets/azure_sources/zu-ZA-ThandoNeural/calibration.wav"
    )

    gcs = FakeGCS()
    app.gcs = gcs
    queued = app.queue_openvoice_assets(job)
    assert queued["worker_mode"] == "assets"
    assert queued["plan_file_sha256"] == checksum(app.paths(job.job_id).openvoice_asset_plan)

    checkpoint_hashes = {"converter": "b" * 64}
    source = plan.sources[0]
    speaker = plan.speakers[0]
    candidate = speaker.voice_candidates[0]
    generated = tmp_path / "generated"
    source_embedding = generated / "source_embedding.bin"
    source_embedding.parent.mkdir(parents=True)
    source_embedding.write_bytes(b"source-embedding")
    source_embedding_hash = checksum(source_embedding)
    source_manifest = generated / "source_embedding_manifest.json"
    source_result = {
        "cache_reused": False,
        "azure_voice": source.azure_voice,
        "calibration_audio_sha256": source.calibration_audio_sha256,
        "calibration_manifest_sha256": source.calibration_manifest_sha256,
        "calibration_identity": source.calibration.identity(),
        "source_embedding_path": "/content/source_embedding.bin",
        "source_embedding_sha256": source_embedding_hash,
        "openvoice_revision": plan.openvoice_revision,
    }
    source_manifest_value = _artifact(
        app,
        {
            "schema_version": "openvoice-source-embedding-manifest-v1",
            **source_result,
            "checkpoint_hashes": checkpoint_hashes,
            "status": "ready",
        },
    )
    atomic_write_json(source_manifest, source_manifest_value)
    source_result.update(
        source_embedding_manifest_path="/content/source_embedding_manifest.json",
        source_embedding_manifest_sha256=checksum(source_manifest),
        source_embedding_manifest_artifact_sha256=source_manifest_value[
            "artifact_sha256"
        ],
    )

    target_embedding = generated / "target_embedding.bin"
    target_embedding.write_bytes(b"target-embedding")
    target_embedding_hash = checksum(target_embedding)
    target_manifest = generated / "target_embedding_manifest.json"
    target_result = {
        "speaker_id": speaker.speaker_id,
        "reference_audio_sha256": speaker.reference_audio_sha256,
        "reference_manifest_sha256": speaker.reference_manifest_sha256,
        "target_embedding_path": "/content/target_embedding.bin",
        "target_embedding_sha256": target_embedding_hash,
        "job_local_identity": True,
        "openvoice_revision": plan.openvoice_revision,
    }
    target_manifest_value = _artifact(
        app,
        {
            "schema_version": "openvoice-target-embedding-manifest-v1",
            **target_result,
            "checkpoint_hashes": checkpoint_hashes,
            "status": "ready",
        },
    )
    atomic_write_json(target_manifest, target_manifest_value)
    target_result.update(
        target_embedding_manifest_path="/content/target_embedding_manifest.json",
        target_embedding_manifest_sha256=checksum(target_manifest),
        target_embedding_manifest_artifact_sha256=target_manifest_value[
            "artifact_sha256"
        ],
    )

    converted = generated / "candidate.wav"
    make_wav(converted, 800)
    candidate_result = {
        "schema_version": "openvoice-conversion-result-v1",
        "backend": "openvoice",
        "backend_revision": plan.openvoice_revision,
        "unit_id": candidate.candidate_id,
        "candidate_id": candidate.candidate_id,
        "speaker_id": speaker.speaker_id,
        "azure_voice": candidate.azure_voice,
        "voice": candidate.azure_voice,
        "source_audio_sha256": candidate.source_audio_sha256,
        "source_embedding_sha256": source_embedding_hash,
        "target_embedding_sha256": target_embedding_hash,
        "output_sha256": checksum(converted),
        "output_path": "/content/candidate.wav",
        "source_duration_ms": 800,
        "converted_duration_ms": 800,
        "duration_ratio": 1.0,
        "duration_drift_ratio": 1.0,
        "sample_rate": 24000,
        "channels": 1,
        "device": "cuda",
        "precision": "fp16",
        "attempt": 1,
        "warnings": [],
        "calibration_text_sha256": candidate.calibration_text_sha256,
        "audio_artifact_count": 0,
        "evaluation_status": "pending_external_qc",
        "human_review_required": True,
    }
    candidate_manifest = generated / "azure_voice_candidates.json"
    candidate_manifest_value = _artifact(
        app,
        {
            "schema_version": "openvoice-azure-voice-candidates-v1",
            "job_id": job.job_id,
            "speaker_id": speaker.speaker_id,
            "openvoice_revision": plan.openvoice_revision,
            "reference_audio_sha256": speaker.reference_audio_sha256,
            "target_embedding_sha256": target_embedding_hash,
            "state": "completed",
            "candidates": [candidate_result],
            "thresholds_used": False,
            "human_review_required": True,
        },
    )
    atomic_write_json(candidate_manifest, candidate_manifest_value)

    gcs.put_file(job.job_id, source.source_embedding_path.as_posix(), source_embedding)
    gcs.put_file(job.job_id, source.source_embedding_manifest_path.as_posix(), source_manifest)
    gcs.put_file(job.job_id, speaker.target_embedding_path.as_posix(), target_embedding)
    gcs.put_file(job.job_id, speaker.target_embedding_manifest_path.as_posix(), target_manifest)
    gcs.put_file(job.job_id, candidate.output_path.as_posix(), converted)
    gcs.put_file(job.job_id, speaker.candidate_manifest_path.as_posix(), candidate_manifest)
    worker_manifest = {
        "schema_version": "openvoice-asset-worker-manifest-v1",
        "job_id": job.job_id,
        "lease_identity": "worker-1",
        "backend": "openvoice",
        "backend_revision": plan.openvoice_revision,
        "plan_artifact_sha256": "c" * 64,
        "server_plan_artifact_sha256": plan.artifact_sha256,
        "server_plan_file_sha256": queued["plan_file_sha256"],
        "state": "completed",
        "completed_assets": 3,
        "pending_assets": 0,
        "checkpoint_hashes": checkpoint_hashes,
        "runtime": {
            "openvoice_revision": plan.openvoice_revision,
            "cuda_available": True,
            "device": "cuda",
        },
        "sources": {
            source.azure_voice: {
                "status": "completed",
                "attempts": 1,
                "result": source_result,
            }
        },
        "speakers": {
            speaker.speaker_id: {
                "status": "completed",
                "attempts": 1,
                "result": target_result,
                "candidate_manifest_sha256": candidate_manifest_value[
                    "artifact_sha256"
                ],
                "candidates": {
                    candidate.candidate_id: {
                        "status": "completed",
                        "attempts": 1,
                        "result": candidate_result,
                    }
                },
            }
        },
    }
    gcs.put_json(job.job_id, "dubbing/openvoice/asset_manifest.json", worker_manifest)
    gcs.put_json(
        job.job_id,
        "worker/openvoice_asset_preparation_claim.json",
        {
            "job_id": job.job_id,
            "task_type": "openvoice_asset_preparation",
            "worker_id": "worker-1",
            "status": "completed",
        },
    )

    receipt = app.reconcile_openvoice_assets(job)
    assert receipt["state"] == "completed"
    assert receipt["candidate_qc_status"] == "pending_external_qc"
    assert (
        tmp_path
        / "cache/openvoice/azure_sources/zu-ZA-ThandoNeural/source_embedding.bin"
    ).read_bytes() == b"source-embedding"
    assert read_json(
        app.jobs.job_dir(job.job_id)
        / "speakers/AZURE_1/azure_voice_candidates.json"
    )["artifact_sha256"] == candidate_manifest_value["artifact_sha256"]

    calibration = app.prepare_azure_source_calibrations(
        job, CalibrationBackend()
    )
    assert calibration["voices"]["zu-ZA-ThandoNeural"]["status"] == "ready"

    reused_plan = OpenVoiceAssetPlan.from_dict(
        app.prepare_openvoice_assets(job, proof_of_concept=True, force=True)
    )
    reused_source = reused_plan.sources[0]
    assert app._resolve_job_path(
        app.paths(job.job_id), reused_source.source_embedding_path
    ).read_bytes() == b"source-embedding"
    second_gcs = FakeGCS()
    app.gcs = second_gcs
    reused_queue = app.queue_openvoice_assets(job, force=True)
    assert reused_queue["source_cache_seed_count"] == 1
