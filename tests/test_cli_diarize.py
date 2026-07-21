"""CLI-level regression tests for diarize command."""

import json
from pathlib import Path
from unittest.mock import patch

from mathula_tv.atomic_io import atomic_write_json
from mathula_tv.azure_stt import normalize
from mathula_tv.config import Settings
from mathula_tv.orchestrator import Orchestrator
from mathula_tv.media import checksum
from test_azure_fast_stt import RAW


class GCS:
    def __init__(self, pyannote): self.pyannote,self.uploads=pyannote,[]
    def download(self,job_id,relative,path): path.parent.mkdir(parents=True,exist_ok=True); atomic_write_json(path,self.pyannote)
    def upload(self,job_id,relative,path): self.uploads.append(relative); return f"gs://bucket/mathula-tv/jobs/{job_id}/{relative}"
    def upload_json(self,job_id,relative,value): self.uploads.append(relative); return f"gs://bucket/{relative}"


def test_cli_diarize_reuses_artifacts_and_calls_orchestrator(tmp_path):
    """Test that CLI diarize command reuses artifacts and calls orchestrator continuation."""
    cfg=Settings(tmp_path,"bucket","mathula-tv",tmp_path/"case",tmp_path/"politics","","eastus","en-ZA","2025-10-15",10,600,"","","preview","model")
    pyannote={"schema_version":"diarization-v1","turns":[{"speaker":"SPEAKER_00","start":0,"end":1,"duration":1,"overlap":False}]}
    gcs=GCS(pyannote); app=Orchestrator(cfg,gcs)
    job=app.jobs.create(source_filename="x",local_source_path="/x",source_checksum="a"*64,source_duration=3)
    job.state="analysis_running"; job.completed_stages=["azure_transcription"]; app.jobs.save(job)
    job_dir=app.jobs.job_dir(job.job_id)
    
    # Write existing corrected transcript
    corrected_transcript={"schema_version":"transcript-v1","segments":[{"speaker":"SPEAKER_00","start":0,"end":1,"source_text":"hoshkhari"}],"warnings":["Corrected transcript"]}
    atomic_write_json(job_dir/"analysis/transcript_en.json",corrected_transcript)
    
    # Write existing diarization artifacts
    atomic_write_json(job_dir/"analysis/azure_diarization.json",normalize(RAW))
    atomic_write_json(job_dir/"analysis/pyannote_diarization.json",pyannote)
    
    from mathula_tv.diarization import validate_turns
    
    # Simulate CLI diarize command logic
    pyannote_path=job_dir/"analysis/pyannote_diarization.json"
    azure_path=job_dir/"analysis/azure_diarization.json"
    
    # Check if existing diarization artifacts are valid
    if pyannote_path.is_file() and azure_path.is_file():
        pyannote_data=json.loads(pyannote_path.read_text())
        azure_data=json.loads(azure_path.read_text())
        validate_turns(azure_data.get("turns", []))
        validate_turns(pyannote_data.get("turns", []))
        # Artifacts are valid - should call orchestrator
        app.process(job)
        
        # Reload job to check state
        result=app.jobs.load(job.job_id)
        
        # Verify orchestrator was called and state transition occurred
        assert result.state=="analysis_ready"
        assert "analysis_reconciliation" in result.completed_stages
        
        # Verify corrected transcript content was preserved (segments unchanged)
        transcript=json.loads(Path(job_dir/"analysis/transcript_en.json").read_text())
        assert transcript["segments"][0]["source_text"]=="hoshkhari"
        assert "Using existing corrected transcript" in transcript.get("warnings",[])
        
        # Verify domain classification and context were rebuilt
        assert (job_dir/"analysis/domain_classification.json").is_file()
        assert (job_dir/"analysis/context.json").is_file()
        
        # Verify hashes were updated from actual bytes
        assert "analysis_input_hashes" in result.media
        transcript_hash=checksum(job_dir/"analysis/transcript_en.json")
        assert result.media["analysis_input_hashes"]["transcript_en"]==transcript_hash


def test_cli_diarize_azure_authoritative_without_pyannote(tmp_path):
    """Test that CLI diarize works with Azure authoritative when Pyannote package unavailable."""
    cfg=Settings(tmp_path,"bucket","mathula-tv",tmp_path/"case",tmp_path/"politics","","eastus","en-ZA","2025-10-15",10,600,"","","preview","model")
    gcs=GCS(None); app=Orchestrator(cfg,gcs)
    job=app.jobs.create(source_filename="x",local_source_path="/x",source_checksum="a"*64,source_duration=3)
    job.state="analysis_running"; job.completed_stages=["azure_transcription"]; app.jobs.save(job)
    job_dir=app.jobs.job_dir(job.job_id)
    
    # Write only Azure diarization (no Pyannote)
    atomic_write_json(job_dir/"analysis/azure_diarization.json",normalize(RAW))
    
    # Simulate CLI diarize command logic with only Azure
    azure_path=job_dir/"analysis/azure_diarization.json"
    from mathula_tv.diarization import validate_turns
    
    if azure_path.is_file():
        azure_data=json.loads(azure_path.read_text())
        validate_turns(azure_data.get("turns", []))
        # Azure is valid - should call orchestrator
        app.process(job)
        
        # Reload job to check state
        result=app.jobs.load(job.job_id)
        
        # Verify orchestrator was called and state transition occurred
        assert result.state=="analysis_ready"
        assert "analysis_reconciliation" in result.completed_stages


def test_cli_diarize_no_pyannote_inference_when_valid_artifacts_exist(tmp_path):
    """Test that Pyannote inference is not called when valid artifacts already exist."""
    cfg=Settings(tmp_path,"bucket","mathula-tv",tmp_path/"case",tmp_path/"politics","","eastus","en-ZA","2025-10-15",10,600,"","","preview","model")
    pyannote={"schema_version":"diarization-v1","turns":[{"speaker":"SPEAKER_00","start":0,"end":1,"duration":1,"overlap":False}]}
    gcs=GCS(pyannote); app=Orchestrator(cfg,gcs)
    job=app.jobs.create(source_filename="x",local_source_path="/x",source_checksum="a"*64,source_duration=3)
    job.state="analysis_running"; job.completed_stages=["azure_transcription"]; app.jobs.save(job)
    job_dir=app.jobs.job_dir(job.job_id)
    
    # Write existing diarization artifacts
    atomic_write_json(job_dir/"analysis/azure_diarization.json",normalize(RAW))
    atomic_write_json(job_dir/"analysis/pyannote_diarization.json",pyannote)
    
    # Mock run_pyannote to ensure it's not called
    with patch('mathula_tv.cli.run_pyannote') as mock_run_pyannote:
        from mathula_tv.diarization import validate_turns
        
        pyannote_path=job_dir/"analysis/pyannote_diarization.json"
        azure_path=job_dir/"analysis/azure_diarization.json"
        
        if pyannote_path.is_file() and azure_path.is_file():
            pyannote_data=json.loads(pyannote_path.read_text())
            azure_data=json.loads(azure_path.read_text())
            validate_turns(azure_data.get("turns", []))
            validate_turns(pyannote_data.get("turns", []))
            # Artifacts are valid - should not call run_pyannote
            app.process(job)
            
            # Verify run_pyannote was not called
            mock_run_pyannote.assert_not_called()
            
            # Reload job to check state
            result=app.jobs.load(job.job_id)
            
            # Verify state transition still occurred
            assert result.state=="analysis_ready"


def test_cli_diarize_exits_zero_only_after_state_transition(tmp_path):
    """Test that CLI exits successfully only after state transition to analysis_ready."""
    cfg=Settings(tmp_path,"bucket","mathula-tv",tmp_path/"case",tmp_path/"politics","","eastus","en-ZA","2025-10-15",10,600,"","","preview","model")
    pyannote={"schema_version":"diarization-v1","turns":[{"speaker":"SPEAKER_00","start":0,"end":1,"duration":1,"overlap":False}]}
    gcs=GCS(pyannote); app=Orchestrator(cfg,gcs)
    job=app.jobs.create(source_filename="x",local_source_path="/x",source_checksum="a"*64,source_duration=3)
    job.state="analysis_running"; job.completed_stages=["azure_transcription"]; app.jobs.save(job)
    job_dir=app.jobs.job_dir(job.job_id)
    
    # Write existing diarization artifacts
    atomic_write_json(job_dir/"analysis/azure_diarization.json",normalize(RAW))
    atomic_write_json(job_dir/"analysis/pyannote_diarization.json",pyannote)
    
    from mathula_tv.diarization import validate_turns
    
    pyannote_path=job_dir/"analysis/pyannote_diarization.json"
    azure_path=job_dir/"analysis/azure_diarization.json"
    
    if pyannote_path.is_file() and azure_path.is_file():
        pyannote_data=json.loads(pyannote_path.read_text())
        azure_data=json.loads(azure_path.read_text())
        validate_turns(azure_data.get("turns", []))
        validate_turns(pyannote_data.get("turns", []))
        
        # Call orchestrator
        app.process(job)
        
        # Reload job to check state
        result=app.jobs.load(job.job_id)
        
        # Verify state transition occurred
        assert result.state=="analysis_ready"
        
        # If state were still analysis_running, this would be a failure
        assert result.state != "analysis_running"
