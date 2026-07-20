import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from mathula_tv.colab_worker import (LeaseHeartbeat, eligible_job, load_pyannote_pipeline,
    normalize_pyannote_output, require_huggingface_token, run_analysis_worker)


BASE = {"job_id":"job", "state":"analysis_running", "completed_stages":["azure_transcription"],
        "objects":{"source_audio":"gs://bucket/audio"}, "providers":{}, "attempt_counters":{},
        "media":{"audio_checksum":"", "normalized_audio_duration":2.0}}


def test_explicit_job_eligibility_and_idempotency():
    eligible_job(BASE)
    with pytest.raises(FileExistsError): eligible_job(BASE, result_exists=True)
    with pytest.raises(ValueError, match="Azure"): eligible_job({**BASE, "completed_stages":[]})
    with pytest.raises(ValueError, match="analysis_running"): eligible_job({**BASE, "state":"uploaded"})


def test_missing_huggingface_secret_and_redaction():
    with pytest.raises(RuntimeError, match="HUGGINGFACE_TOKEN") as error:
        require_huggingface_token(SimpleNamespace(get=lambda key: None))
    assert "secret-value" not in str(error.value)


@pytest.mark.parametrize("message,expected", [("401 invalid token","token is invalid"),("403 gated repository","Accept the gated")])
def test_model_access_errors_are_clear_and_secret_safe(message, expected):
    class Pipeline:
        @staticmethod
        def from_pretrained(*args, **kwargs): raise RuntimeError(message)
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:True), device=lambda value:value)
    with pytest.raises(RuntimeError, match=expected) as error:
        load_pyannote_pipeline("secret-value", pipeline_class=Pipeline, torch_module=torch)
    assert "secret-value" not in str(error.value)


class Segment:
    def __init__(self,start,end): self.start,self.end=start,end
class Annotation:
    def itertracks(self,yield_label=False):
        yield Segment(0,1),None,"SPEAKER_00"
        yield Segment(.8,2),None,"SPEAKER_01"


def test_successful_normalization_preserves_overlap():
    result=normalize_pyannote_output(Annotation(),job_id="job",model_id="model",audio_duration=2,processing_duration=1)
    assert result["speaker_count"]==2 and all(t["overlap"] for t in result["turns"])
    assert result["real_time_factor"]==.5 and result["device"]=="cuda"


def test_lease_heartbeat_updates_generation():
    class Store:
        def heartbeat(self,claim,seconds): return {**claim,"generation":claim["generation"]+1}
    with LeaseHeartbeat(Store(),{"generation":1},1,interval=.01) as heartbeat: time.sleep(.03)
    assert heartbeat.claim["generation"] > 1


def test_source_checksum_mismatch_marks_claim_failed(tmp_path):
    class Blob:
        def exists(self): return False
    class Bucket:
        def blob(self,name): return Blob()
    class Store:
        bucket=Bucket()
        def name(self,j,r): return r
        def download_json(self,j,r): return ({**BASE,"media":{**BASE["media"],"audio_checksum":"bad"}},1)
        def claim(self,*args): return {"job_id":"job","task_type":"pyannote_analysis","generation":1}
        def download(self,j,r,path): path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(b"audio")
        def finish_claim(self,claim,status,error=None): self.finished=status
    store=Store()
    with pytest.raises(ValueError,match="checksum"):
        run_analysis_worker(store,"job","token",tmp_path)
    assert store.finished=="failed"
