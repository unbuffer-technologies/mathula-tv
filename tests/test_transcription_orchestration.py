from types import SimpleNamespace

from mathula_tv.config import Settings
from mathula_tv.job_store import JobStore
from mathula_tv.orchestrator import Orchestrator
from test_azure_fast_stt import RAW


class Backend:
    provider="azure-speech-fast-transcription"; api_version="2025-10-15"; request_duration_seconds=.42
    def __init__(self): self.calls=0
    def transcribe(self, audio, locale): self.calls += 1; return RAW


class GCS:
    def __init__(self): self.uploads=[]
    def upload(self, job_id, relative, path): self.uploads.append((relative,path.read_bytes())); return f"gs://bucket/mathula-tv/jobs/{job_id}/{relative}"
    def upload_json(self, job_id, relative, value): self.uploads.append((relative,value)); return f"gs://bucket/mathula-tv/jobs/{job_id}/{relative}"


def settings(tmp_path):
    return Settings(tmp_path, "bucket", "mathula-tv", tmp_path/"case", tmp_path/"politics", "", "eastus", "en-ZA", "2025-10-15", 10, 600, "", "", "preview", "model")


def test_persistence_gcs_paths_state_idempotency_and_force(tmp_path):
    cfg=settings(tmp_path); gcs=GCS(); app=Orchestrator(cfg,gcs); audio=tmp_path/"audio.wav"; audio.write_bytes(b"audio")
    job=app.jobs.create(source_filename="x.mp4",local_source_path="/x.mp4",source_checksum="a"*64,source_duration=2.5)
    job.objects["local_audio"]=str(audio); job.state="uploaded"; app.jobs.save(job); backend=Backend()
    result=app.transcribe(job,backend)
    assert result.state=="analysis_running" and "azure_transcription" in result.completed_stages and backend.calls==1
    assert (app.jobs.job_dir(job.job_id)/"analysis/azure_stt.json").is_file()
    assert (app.jobs.job_dir(job.job_id)/"analysis/azure_diarization.json").is_file()
    assert [item[0] for item in gcs.uploads] == ["analysis/azure_stt.json","analysis/azure_diarization.json","status.json"]
    assert result.objects["azure_stt"].endswith("analysis/azure_stt.json")
    app.transcribe(app.jobs.load(job.job_id),backend); assert backend.calls==1
    app.transcribe(app.jobs.load(job.job_id),backend,force=True); assert backend.calls==2
