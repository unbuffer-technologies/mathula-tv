from __future__ import annotations

import json
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .atomic_io import atomic_write_json
from .diarization import validate_turns
from .gcs_store import GCSStore
from .media import checksum
from .models import JobManifest

PYANNOTE_MODEL = "pyannote/speaker-diarization-3.1"


def eligible_job(status: dict[str, Any], *, force: bool = False, result_exists: bool = False) -> None:
    if status.get("state") != "analysis_running":
        raise ValueError("Job must be analysis_running")
    if "azure_transcription" not in status.get("completed_stages", []):
        raise ValueError("Job has not completed Azure transcription")
    if not status.get("objects", {}).get("source_audio"):
        raise ValueError("Job has no source audio object")
    if "pyannote_analysis" in status.get("completed_stages", []) and not force:
        raise FileExistsError("Pyannote analysis is already complete; set FORCE=True to replace it")
    if result_exists and not force:
        raise FileExistsError("A Pyannote result already exists; set FORCE=True to replace it")


def discover_eligible_jobs(client: Any, bucket_name: str, prefix: str) -> list[str]:
    result = []
    for blob in client.list_blobs(bucket_name, prefix=f"{prefix.strip('/')}/jobs/"):
        if not blob.name.endswith("/status.json"):
            continue
        try:
            status = json.loads(blob.download_as_text())
            eligible_job(status)
        except (ValueError, FileExistsError, json.JSONDecodeError):
            continue
        result.append(status["job_id"])
    return sorted(set(result))


def require_huggingface_token(userdata: Any) -> str:
    try:
        token = userdata.get("HUGGINGFACE_TOKEN")
    except Exception as exc:
        raise RuntimeError("Unable to read the HUGGINGFACE_TOKEN Colab secret") from exc
    if not token:
        raise RuntimeError("Add HUGGINGFACE_TOKEN in Colab Secrets and grant this notebook access")
    return token


def load_pyannote_pipeline(token: str, model_id: str = PYANNOTE_MODEL, *, pipeline_class: Any = None, torch_module: Any = None):
    try:
        if pipeline_class is None:
            from pyannote.audio import Pipeline
            pipeline_class = Pipeline
        if torch_module is None:
            import torch
            torch_module = torch
        pipeline = pipeline_class.from_pretrained(model_id, token=token)
        if pipeline is None:
            raise RuntimeError("model access was denied")
        if not torch_module.cuda.is_available():
            raise RuntimeError("CUDA is required for this Colab worker")
        pipeline.to(torch_module.device("cuda"))
        return pipeline
    except Exception as exc:
        message = str(exc).lower()
        if any(term in message for term in ("401", "unauthorized", "invalid token")):
            reason = "Hugging Face token is invalid"
        elif any(term in message for term in ("403", "gated", "accept", "restricted")):
            reason = f"Accept the gated model conditions for {model_id}"
        else:
            reason = f"Failed to download/load {model_id}"
        raise RuntimeError(f"{reason}: {type(exc).__name__}") from exc


def normalize_pyannote_output(output: Any, *, job_id: str, model_id: str, audio_duration: float, processing_duration: float) -> dict[str, Any]:
    annotation = getattr(output, "speaker_diarization", output)
    turns: list[dict[str, Any]] = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        start, end = float(turn.start), float(turn.end)
        turns.append({"speaker": str(speaker), "start": start, "end": end, "duration": end - start, "overlap": False})
    validate_turns(turns)
    for index, left in enumerate(turns):
        left["overlap"] = any(
            left["speaker"] != right["speaker"] and min(left["end"], right["end"]) > max(left["start"], right["start"])
            for other, right in enumerate(turns) if other != index
        )
    speakers = sorted({turn["speaker"] for turn in turns})
    return {
        "schema_version": "diarization-v1",
        "job_id": job_id,
        "provider": "pyannote",
        "engine": "pyannote",
        "model_identifier": model_id,
        "device": "cuda",
        "audio_duration": audio_duration,
        "processing_duration": processing_duration,
        "execution_duration": processing_duration,
        "real_time_factor": processing_duration / audio_duration,
        "speaker_count": len(speakers),
        "turns": turns,
        "warnings": ["Speaker identifiers are job-local"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


class LeaseHeartbeat:
    def __init__(self, store: GCSStore, claim: dict, lease_seconds: int, interval: float | None = None):
        self.store, self.claim, self.lease_seconds = store, claim, lease_seconds
        self.interval = interval or max(15.0, lease_seconds / 3)
        self._stop = threading.Event()
        self.error: Exception | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.wait(self.interval):
            try:
                self.claim = self.store.heartbeat(self.claim, self.lease_seconds)
            except Exception as exc:
                self.error = exc
                self._stop.set()

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *args):
        self._stop.set()
        self._thread.join()
        if self.error and not args[0]:
            raise RuntimeError("Lost the Pyannote lease during processing") from self.error


def run_analysis_worker(
    store: GCSStore,
    job_id: str,
    token: str,
    work_dir: Path,
    *,
    worker_id: str | None = None,
    force: bool = False,
    lease_seconds: int = 900,
    pipeline_loader: Callable[[str, str], Any] = load_pyannote_pipeline,
) -> dict[str, Any]:
    status, status_generation = store.download_json(job_id, "status.json")
    final_blob = store.bucket.blob(store.name(job_id, "analysis/pyannote_diarization.json"))
    result_exists = final_blob.exists()
    eligible_job(status, force=force, result_exists=result_exists)
    worker_id = worker_id or f"colab-{uuid.uuid4().hex[:12]}"
    attempt = status.get("attempt_counters", {}).get("pyannote_analysis", 0) + 1
    claim = store.claim(job_id, "pyannote_analysis", worker_id, attempt, lease_seconds)
    upload_started = False
    try:
        audio = work_dir / job_id / "source_audio.wav"
        store.download(job_id, "input/source_audio.wav", audio)
        expected = status.get("media", {}).get("audio_checksum")
        actual = checksum(audio)
        if not expected or actual != expected:
            raise ValueError("Source audio checksum does not match authoritative job state")
        audio_duration = float(status.get("media", {}).get("normalized_audio_duration") or 0)
        if audio_duration <= 0:
            raise ValueError("Authoritative audio duration is missing")
        with LeaseHeartbeat(store, claim, lease_seconds) as heartbeat:
            pipeline = pipeline_loader(token, PYANNOTE_MODEL)
            started = time.perf_counter()
            output = pipeline(str(audio))
            elapsed = time.perf_counter() - started
        claim = heartbeat.claim
        result = normalize_pyannote_output(output, job_id=job_id, model_id=PYANNOTE_MODEL, audio_duration=audio_duration, processing_duration=elapsed)
        result_path = work_dir / job_id / "pyannote_diarization.json"
        atomic_write_json(result_path, result)
        expected_result_hash = checksum(result_path)
        temporary = f"analysis/pyannote_diarization.json.partial.{worker_id}"
        upload_started = True
        store.upload(job_id, temporary, result_path, if_generation_match=0)
        final_generation = 0
        if result_exists and force:
            final_blob.reload(); final_generation = int(final_blob.generation)
        result_uri = store.promote(job_id, temporary, "analysis/pyannote_diarization.json", expected_result_hash, checksum, final_generation=final_generation)
        job = JobManifest.from_dict(status)
        job.attempt_counters["pyannote_analysis"] = attempt
        if "pyannote_analysis" not in job.completed_stages:
            job.completed_stages.append("pyannote_analysis")
        job.objects["pyannote_diarization"] = result_uri
        job.providers["pyannote"] = PYANNOTE_MODEL
        job.media["pyannote_processing_duration"] = round(elapsed, 3)
        job.last_error = None
        store.update_json(job_id, "status.json", job.to_dict(), status_generation)
        claim = store.finish_claim(claim, "completed")
        return result
    except Exception as exc:
        # A transient upload/promotion error retains the renewable claim so the
        # same worker can retry without allowing another worker to complete it.
        if not upload_started:
            try:
                store.finish_claim(claim, "failed", f"{type(exc).__name__}: {exc}")
            except Exception:
                pass
        raise
