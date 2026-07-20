import json
from pathlib import Path

from mathula_tv.atomic_io import atomic_write_json
from mathula_tv.azure_stt import normalize
from mathula_tv.config import Settings
from mathula_tv.orchestrator import Orchestrator
from test_azure_fast_stt import RAW


class GCS:
    def __init__(self, pyannote): self.pyannote,self.uploads=pyannote,[]
    def download(self,job_id,relative,path): path.parent.mkdir(parents=True,exist_ok=True); atomic_write_json(path,self.pyannote)
    def upload(self,job_id,relative,path): self.uploads.append(relative); return f"gs://bucket/mathula-tv/jobs/{job_id}/{relative}"
    def upload_json(self,job_id,relative,value): self.uploads.append(relative); return f"gs://bucket/{relative}"


def test_server_continues_returned_pyannote_result(tmp_path):
    cfg=Settings(tmp_path,"bucket","mathula-tv",tmp_path/"case",tmp_path/"politics","","eastus","en-ZA","2025-10-15",10,600,"","","preview","model")
    pyannote={"schema_version":"diarization-v1","turns":[{"speaker":"SPEAKER_00","start":0,"end":1,"duration":1,"overlap":False},{"speaker":"SPEAKER_01","start":1,"end":3,"duration":2,"overlap":False}]}
    gcs=GCS(pyannote); app=Orchestrator(cfg,gcs)
    job=app.jobs.create(source_filename="x",local_source_path="/x",source_checksum="a"*64,source_duration=3)
    job.state="analysis_running"; job.completed_stages=["azure_transcription"]; app.jobs.save(job)
    atomic_write_json(app.jobs.job_dir(job.job_id)/"analysis/azure_diarization.json",normalize(RAW))
    message=app.process(job)
    result=app.jobs.load(job.job_id)
    assert result.state=="analysis_ready" and "analysis_reconciliation" in result.completed_stages
    assert (app.jobs.job_dir(job.job_id)/"analysis/transcript_en.json").is_file()
    assert "Analysis ready" in message and "analysis/transcript_en.json" in gcs.uploads
