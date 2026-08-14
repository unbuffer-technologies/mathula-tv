import json

import pytest
from pathlib import Path

from mathula_tv.atomic_io import atomic_write_json
from mathula_tv.azure_stt import normalize
from mathula_tv.config import Settings
import mathula_tv.orchestrator as orchestrator_module
from mathula_tv.orchestrator import Orchestrator
from mathula_tv.media import checksum
from test_azure_fast_stt import RAW




@pytest.fixture(autouse=True)
def completed_autocorrection_stage(monkeypatch):
    def fake_stage(*, work_dir, job, reconciled_transcript, config=None):
        job_dir = Path(work_dir) / "jobs" / job.job_id
        analysis = job_dir / "analysis"
        raw_path = analysis / "transcript_en_raw.json"
        transcript_path = analysis / "transcript_en.json"
        state_path = analysis / "autocorrection_state.json"
        atomic_write_json(raw_path, dict(reconciled_transcript))
        if transcript_path.is_file():
            effective = json.loads(transcript_path.read_text(encoding="utf-8"))
            effective.setdefault("warnings", []).append("Using existing corrected transcript")
        else:
            effective = dict(reconciled_transcript)
        effective.setdefault(
            "autocorrect",
            {
                "schema_version": "mathula-autocorrect-state-v1",
                "active_correction_count": 0,
                "overlays": [],
            },
        )
        atomic_write_json(transcript_path, effective)
        state = {
            "review_count": 0,
            "active_count": effective["autocorrect"].get("active_correction_count", 0),
            "ready_for_language_ai": True,
            "raw_transcript_sha256": checksum(raw_path),
            "authoritative_transcript_sha256": checksum(transcript_path),
        }
        atomic_write_json(state_path, state)
        return state

    monkeypatch.setattr(
        orchestrator_module,
        "run_autocorrection_first",
        fake_stage,
    )

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


def test_existing_corrected_transcript_is_reused(tmp_path):
    """Test that an existing corrected transcript is reused without re-reconciliation."""
    cfg=Settings(tmp_path,"bucket","mathula-tv",tmp_path/"case",tmp_path/"politics","","eastus","en-ZA","2025-10-15",10,600,"","","preview","model")
    pyannote={"schema_version":"diarization-v1","turns":[{"speaker":"SPEAKER_00","start":0,"end":1,"duration":1,"overlap":False}]}
    gcs=GCS(pyannote); app=Orchestrator(cfg,gcs)
    job=app.jobs.create(source_filename="x",local_source_path="/x",source_checksum="a"*64,source_duration=3)
    job.state="analysis_running"; job.completed_stages=["azure_transcription"]; app.jobs.save(job)
    job_dir=app.jobs.job_dir(job.job_id)
    
    # Write existing corrected transcript
    corrected_transcript={"schema_version":"transcript-v1","segments":[{"speaker":"SPEAKER_00","start":0,"end":1,"source_text":"hoshkhari"}],"warnings":["Corrected transcript"]}
    atomic_write_json(job_dir/"analysis/transcript_en.json",corrected_transcript)
    
    atomic_write_json(job_dir/"analysis/azure_diarization.json",normalize(RAW))
    app.process(job)
    result=app.jobs.load(job.job_id)
    
    # Verify transcript was reused (original content preserved)
    assert result.state=="analysis_ready"
    transcript=json.loads(Path(job_dir/"analysis/transcript_en.json").read_text())
    assert transcript["segments"][0]["source_text"]=="hoshkhari"
    # Check that the transcript warnings indicate reuse
    assert "Using existing corrected transcript" in transcript.get("warnings",[])


def test_analysis_hashes_updated_from_actual_file_bytes(tmp_path):
    """Test that analysis input hashes are computed from actual file bytes."""
    cfg=Settings(tmp_path,"bucket","mathula-tv",tmp_path/"case",tmp_path/"politics","","eastus","en-ZA","2025-10-15",10,600,"","","preview","model")
    pyannote={"schema_version":"diarization-v1","turns":[{"speaker":"SPEAKER_00","start":0,"end":1,"duration":1,"overlap":False}]}
    gcs=GCS(pyannote); app=Orchestrator(cfg,gcs)
    job=app.jobs.create(source_filename="x",local_source_path="/x",source_checksum="a"*64,source_duration=3)
    job.state="analysis_running"; job.completed_stages=["azure_transcription"]; app.jobs.save(job)
    job_dir=app.jobs.job_dir(job.job_id)
    
    atomic_write_json(job_dir/"analysis/azure_diarization.json",normalize(RAW))
    app.process(job)
    result=app.jobs.load(job.job_id)
    
    # Verify hashes are computed from actual bytes
    assert "analysis_input_hashes" in result.media
    transcript_hash=checksum(job_dir/"analysis/transcript_en.json")
    classification_hash=checksum(job_dir/"analysis/domain_classification.json")
    context_hash=checksum(job_dir/"analysis/context.json")
    
    assert result.media["analysis_input_hashes"]["transcript_en"]==transcript_hash
    assert result.media["analysis_input_hashes"]["domain_classification"]==classification_hash
    assert result.media["analysis_input_hashes"]["context"]==context_hash


def test_domain_classification_and_context_rebuilt_from_corrected_transcript(tmp_path):
    """Test that domain classification and context are rebuilt from the corrected transcript."""
    cfg=Settings(tmp_path,"bucket","mathula-tv",tmp_path/"case",tmp_path/"politics","","eastus","en-ZA","2025-10-15",10,600,"","","preview","model")
    pyannote={"schema_version":"diarization-v1","turns":[{"speaker":"SPEAKER_00","start":0,"end":1,"duration":1,"overlap":False}]}
    gcs=GCS(pyannote); app=Orchestrator(cfg,gcs)
    job=app.jobs.create(source_filename="x",local_source_path="/x",source_checksum="a"*64,source_duration=3)
    job.state="analysis_running"; job.completed_stages=["azure_transcription"]; app.jobs.save(job)
    job_dir=app.jobs.job_dir(job.job_id)
    
    # Write corrected transcript with specific content
    corrected_transcript={"schema_version":"transcript-v1","segments":[{"speaker":"SPEAKER_00","start":0,"end":1,"source_text":"Madlanga Commission hearing"}],"warnings":[]}
    atomic_write_json(job_dir/"analysis/transcript_en.json",corrected_transcript)
    
    atomic_write_json(job_dir/"analysis/azure_diarization.json",normalize(RAW))
    app.process(job)
    
    # Verify domain classification and context were rebuilt
    assert (job_dir/"analysis/domain_classification.json").is_file()
    assert (job_dir/"analysis/context.json").is_file()
    classification=json.loads(Path(job_dir/"analysis/domain_classification.json").read_text())
    
    # Verify classification contains content from corrected transcript
    # Domain classifier lowercases text, so check evidence field
    assert "madlanga" in json.dumps(classification).lower() or "commission" in json.dumps(classification).lower()


def test_azure_transcription_not_rerun_when_completed(tmp_path):
    """Test that Azure transcription is not rerun when already completed."""
    cfg=Settings(tmp_path,"bucket","mathula-tv",tmp_path/"case",tmp_path/"politics","","eastus","en-ZA","2025-10-15",10,600,"","","preview","model")
    pyannote={"schema_version":"diarization-v1","turns":[{"speaker":"SPEAKER_00","start":0,"end":1,"duration":1,"overlap":False}]}
    gcs=GCS(pyannote); app=Orchestrator(cfg,gcs)
    job=app.jobs.create(source_filename="x",local_source_path="/x",source_checksum="a"*64,source_duration=3)
    job.state="analysis_running"; job.completed_stages=["azure_transcription"]; app.jobs.save(job)
    job_dir=app.jobs.job_dir(job.job_id)
    
    # Write Azure STT artifact
    azure_stt={"schema_version":"stt-v1","words":[{"word":"test","start":0,"end":1}]}
    atomic_write_json(job_dir/"analysis/azure_stt.json",azure_stt)
    original_stt_bytes=Path(job_dir/"analysis/azure_stt.json").read_bytes()
    
    atomic_write_json(job_dir/"analysis/azure_diarization.json",normalize(RAW))
    app.process(job)
    result=app.jobs.load(job.job_id)
    
    # Verify Azure STT was not rerun (bytes unchanged)
    assert Path(job_dir/"analysis/azure_stt.json").read_bytes()==original_stt_bytes
    assert "azure_transcription" in result.completed_stages


def test_valid_state_transition_analysis_running_to_analysis_ready(tmp_path):
    """Test valid state transition from analysis_running to analysis_ready."""
    cfg=Settings(tmp_path,"bucket","mathula-tv",tmp_path/"case",tmp_path/"politics","","eastus","en-ZA","2025-10-15",10,600,"","","preview","model")
    pyannote={"schema_version":"diarization-v1","turns":[{"speaker":"SPEAKER_00","start":0,"end":1,"duration":1,"overlap":False}]}
    gcs=GCS(pyannote); app=Orchestrator(cfg,gcs)
    job=app.jobs.create(source_filename="x",local_source_path="/x",source_checksum="a"*64,source_duration=3)
    job.state="analysis_running"; job.completed_stages=["azure_transcription"]; app.jobs.save(job)
    
    atomic_write_json(app.jobs.job_dir(job.job_id)/"analysis/azure_diarization.json",normalize(RAW))
    app.process(job)
    result=app.jobs.load(job.job_id)
    
    # Verify valid state transition
    assert result.state=="analysis_ready"
    assert "analysis_reconciliation" in result.completed_stages
