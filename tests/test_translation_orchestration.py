import json
from pathlib import Path

import pytest

from mathula_tv.atomic_io import atomic_write_json
from mathula_tv.config import Settings
from mathula_tv.orchestrator import Orchestrator
from mathula_tv.translation import AzureAIError


SEG={"segment_id":"seg-00001","speaker":"SPEAKER_00","start":0.0,"end":2.0,"source_text":"He alleged it happened!","recognition_confidence":.9,"speaker_assignment_confidence":.9,"diarization_source":"pyannote","overlap":False,"warnings":[]}
TURN={"segment_id":"seg-00001","speaker":"SPEAKER_00","start":0.0,"end":2.0,"source_text":"He alleged it happened!","translated_text":"Uthe kusolwa ukuthi kwenzeka!","target_duration":2.0,"delivery_style":"sensational","emotional_intensity":"high","statement_type":"allegation","attribution_required":True,"uncertainty_flags":["allegation"],"pronunciation_hints":[]}
SEO={"title":"Uthi Kwenzekeni? Isimangalo Esishaqisayo","description":"Incazelo enobufakazi.","tags_csv":"izindaba, isimangalo","tags":["izindaba","isimangalo"],"thumbnail_hook":"UTHI KWENZEKENI?","category":"News & Politics","primary_domain":"commission","secondary_domains":[],"entities":["Witness"],"main_entities":["Witness"],"claim_attributions":["Isimangalo sibhekiswe kofakazi"],"context_providers_used":["commission-master-case"],"factual_grounding_notes":["Based on transcript"],"uncertainty_warnings":["Allegation"],"human_review_flags":["Review attribution"]}


class Client:
    deployment="gpt-5.4"; api_version="preview"; repair_attempts=0
    def __init__(self,fail=None): self.calls=[]; self.request_durations=[]; self.fail=fail
    def complete_json(self,prompt,payload):
        self.calls.append((prompt,payload)); self.request_durations.append(.1 if len(self.calls)==1 else .2)
        if self.fail: raise self.fail
        return {"turns":[dict(TURN)]} if len(self.calls)==1 else dict(SEO)


class GCS:
    def __init__(self): self.data={}; self.promotions=[]
    def upload(self,job,relative,path,if_generation_match=None): self.data[relative]=path.read_bytes(); return f"gs://bucket/{relative}"
    def promote(self,job,temp,final,digest,checksum_fn): self.data[final]=self.data[temp]; del self.data[temp]; self.promotions.append(final); return f"gs://bucket/{final}"
    def upload_json(self,job,relative,value): self.data[relative]=json.dumps(value).encode(); return f"gs://bucket/{relative}"


def settings(tmp): return Settings(tmp,"bucket","mathula-tv",tmp/"case",tmp/"politics","","eastus","en-ZA","2025-10-15",10,600,"","","preview","model")


def setup(tmp_path):
    gcs=GCS(); app=Orchestrator(settings(tmp_path),gcs); job=app.jobs.create(source_filename="x",local_source_path="/x",source_checksum="a"*64,source_duration=2)
    job.state="analysis_ready"; job.completed_stages=["analysis_reconciliation"]; app.jobs.save(job); root=app.jobs.job_dir(job.job_id)
    atomic_write_json(root/"analysis/transcript_en.json",{"schema_version":"transcript-v1","language":"en-ZA","segments":[SEG]})
    atomic_write_json(root/"analysis/domain_classification.json",{"primary_domain":"commission","secondary_domains":[],"entities":{}})
    atomic_write_json(root/"analysis/context.json",{"providers":[{"provider_name":"commission-master-case","enrichment_applied":True,"relevant_facts":[{"fact":"Background only"}]},{"provider_name":"none","enrichment_applied":False}]})
    return app,job,gcs


def test_analysis_ready_process_dispatch_and_complete_state_path(tmp_path):
    app,job,gcs=setup(tmp_path); client=Client(); result=app.process(job,translation_client=client)
    saved=app.jobs.load(job.job_id)
    assert saved.state=="synthesis_queued" and result["translated_segments"]==1 and len(client.calls)==2
    assert client.calls[0][1]["grounded_context"][0]["provider_name"]=="commission-master-case" and len(client.calls[0][1]["grounded_context"])==1
    assert gcs.promotions==["translation/transcript_zu.json","translation/youtube_zu.json"]
    assert saved.attempt_counters=={"translation":1,"seo_packaging":1}


def test_translation_and_seo_idempotency(tmp_path):
    app,job,gcs=setup(tmp_path); first=Client(); app.translate_and_package(job,first)
    second=Client(); result=app.translate_and_package(app.jobs.load(job.job_id),second)
    assert result["state"]=="synthesis_queued" and second.calls==[]


@pytest.mark.parametrize("retryable,state",[(True,"failed_retryable"),(False,"failed_terminal")])
def test_azure_failure_state(tmp_path,retryable,state):
    app,job,_=setup(tmp_path); client=Client(AzureAIError("safe failure",retryable=retryable))
    with pytest.raises(AzureAIError): app.translate_and_package(job,client)
    assert app.jobs.load(job.job_id).state==state
