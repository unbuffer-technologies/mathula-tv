"""DEPRECATED migration reference: not registered, not selectable, not production."""

from __future__ import annotations

import json
import importlib.metadata
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .atomic_io import atomic_write_json, read_json
from .colab_worker import LeaseHeartbeat
from .gcs_store import GCSStore
from .media import checksum
from .models import JobManifest
from .omnivoice_adapter import RealOmniVoiceAdapter
from .quality import inspect_wav
from .synthesis import assemble_timeline, extract_reference_wav, plan_synthesis_units, reference_text, select_speaker_references


def discover_synthesis_jobs(client: Any, bucket: str, prefix: str) -> list[str]:
    jobs=[]
    for blob in client.list_blobs(bucket,prefix=f"{prefix.strip('/')}/jobs/"):
        if blob.name.endswith("/status.json"):
            try:
                value=json.loads(blob.download_as_text())
                if value.get("state")=="synthesis_queued" and "omnivoice_synthesis" not in value.get("completed_stages",[]): jobs.append(value["job_id"])
            except Exception: continue
    return sorted(set(jobs))


def build_calibration_manifest(*,model_id,package_versions,language_id,reference,sentence,results):
    return {"schema_version":"omnivoice-calibration-v1","model_id":model_id,"package_versions":package_versions,"language_id":language_id,"test_sentence":sentence,"reference_duration":reference["duration"],"reference_word_count":len(reference["transcript"].split()),"profiles":{name:{"inference_steps":item.generation_parameters["num_step"],"speed":item.generation_parameters["speed"],"duration_setting":item.generation_parameters["duration"],"generated_duration":item.duration,"real_time_factor":item.real_time_factor,"sha256":checksum(Path(item.output_path))} for name,item in sorted(results.items())}}


def calibrate_local_job(job_dir:Path,adapter=None,sentence:str|None=None):
    adapter=adapter or RealOmniVoiceAdapter(); transcript=read_json(job_dir/"analysis/transcript_en.json"); pyannote=read_json(job_dir/"analysis/pyannote_diarization.json"); translation=read_json(job_dir/"translation/transcript_zu.json")
    speaker=translation["turns"][0]["speaker"]; refs=select_speaker_references(pyannote["turns"])
    if speaker not in refs: raise ValueError(f"No safe reference for calibration speaker {speaker}")
    ref=refs[speaker][0]; ref["transcript"]=reference_text(transcript["words"],ref["start"],ref["end"],speaker); path=job_dir/f"output/calibration/{speaker}_reference.wav"; extract_reference_wav(job_dir/"input/source_audio.wav",path,ref["start"],ref["end"]); ref["path"]=str(path)
    sentence=sentence or plan_synthesis_units(translation["turns"][0])[0]["unit_text"]; adapter.load(); directory=job_dir/"output/calibration"; results={profile:adapter.synthesize(sentence,path,ref["transcript"],directory/f"{profile}_settings.wav",generation_profile=profile) for profile in ("current","space_matched")}
    try: version=importlib.metadata.version("omnivoice")
    except importlib.metadata.PackageNotFoundError: version="unknown"
    manifest=build_calibration_manifest(model_id=adapter.model_id,package_versions={"mathula_tv":__import__('mathula_tv').__version__,"omnivoice":version},language_id=adapter.language_id,reference=ref,sentence=sentence,results=results); atomic_write_json(directory/"comparison_manifest.json",manifest); return manifest


class OmniVoiceSynthesisWorker:
    def __init__(self,store:GCSStore,job_id:str,work_dir:Path,*,worker_id:str|None=None,lease_seconds:int=900,force:bool=False,adapter=None,fallback_references:dict[str,dict]|None=None):
        self.store,self.job_id,self.root=store,job_id,work_dir/job_id
        self.worker_id=worker_id or f"omnivoice-{uuid.uuid4().hex[:12]}"; self.lease_seconds,self.force=lease_seconds,force
        self.adapter=adapter or RealOmniVoiceAdapter(); self.fallback_references=fallback_references or {}; self.fallbacks=[]; self.claim=None; self.status=None; self.references={}; self.segment_results=[]; self.clips=[]; self.report=None

    def _load_status(self): self.status,self.status_generation=self.store.download_json(self.job_id,"status.json"); return self.status
    def _save_status(self): self.store.update_json(self.job_id,"status.json",self.status,self.status_generation); self._load_status()

    def prepare(self):
        status=self._load_status()
        final=self.store.bucket.blob(self.store.name(self.job_id,"output/dubbed_zu.wav"))
        if status.get("state")=="synthesis_ready" and final.exists() and not self.force: raise FileExistsError("Valid synthesis is already complete")
        if status.get("state")=="failed_retryable" and (status.get("last_error") or {}).get("stage")=="omnivoice_synthesis":
            recovered=JobManifest.from_dict(status); recovered.transition("synthesis_queued"); recovered.last_error=None; self.status=recovered.to_dict(); self._save_status(); status=self.status
        if status.get("state")!="synthesis_queued": raise ValueError("Job must be synthesis_queued")
        attempt=status.get("attempt_counters",{}).get("omnivoice_synthesis",0)+1
        self.claim=self.store.claim(self.job_id,"omnivoice_synthesis",self.worker_id,attempt,self.lease_seconds)
        job=JobManifest.from_dict(status); job.transition("synthesis_claimed"); job.attempt_counters["omnivoice_synthesis"]=attempt
        self.status=job.to_dict(); self._save_status()
        paths={"audio":("input/source_audio.wav",self.root/"input/source_audio.wav"),"transcript":("analysis/transcript_en.json",self.root/"analysis/transcript_en.json"),"pyannote":("analysis/pyannote_diarization.json",self.root/"analysis/pyannote_diarization.json"),"translation":("translation/transcript_zu.json",self.root/"translation/transcript_zu.json")}
        for _,(remote,local) in paths.items(): self.store.download(self.job_id,remote,local)
        expected={"audio":status["media"]["audio_checksum"],"transcript":status["media"]["analysis_input_hashes"]["transcript_en"],"translation":status["media"]["transcript_zu_sha256"]}
        for key in expected:
            if checksum(paths[key][1])!=expected[key]: raise ValueError(f"Downloaded {key} checksum mismatch")
        self.audio,self.transcript,self.pyannote,self.translation=paths["audio"][1],read_json(paths["transcript"][1]),read_json(paths["pyannote"][1]),read_json(paths["translation"][1])
        turns=select_speaker_references(self.pyannote["turns"])
        speakers={t["speaker"] for t in self.translation["turns"]}
        unresolved=sorted(speakers-set(turns))
        for speaker in list(unresolved):
            fallback=self.fallback_references.get(speaker)
            if fallback and Path(fallback.get("path","")).is_file() and str(fallback.get("transcript","")).strip():
                self.references[speaker]={"speaker":speaker,"path":str(fallback["path"]),"transcript":fallback["transcript"],"start":None,"end":None,"duration":None,"sha256":checksum(Path(fallback["path"])),"configured_fallback":True}
                self.fallbacks.append(speaker); unresolved.remove(speaker)
        if unresolved: raise ValueError("No safe same-speaker reference or explicit fallback for: "+", ".join(unresolved))
        for speaker,items in turns.items():
            item=items[0]; item["transcript"]=reference_text(self.transcript["words"],item["start"],item["end"],speaker)
            path=self.root/f"references/{speaker}.wav"; extract_reference_wav(self.audio,path,item["start"],item["end"]); item["path"]=str(path); item["sha256"]=checksum(path)
            reference_quality=inspect_wav(path)
            if not reference_quality["valid"] or reference_quality["non_silent_ratio"]<.7: raise ValueError(f"Unsafe high-silence reference audio for {speaker}")
            self.references[speaker]=item
        return self.references

    def load_model(self):
        with LeaseHeartbeat(self.store,self.claim,self.lease_seconds) as heartbeat: info=self.adapter.load()
        self.claim=heartbeat.claim; self.model_info=info
        job=JobManifest.from_dict(self.status); job.transition("synthesizing"); self.status=job.to_dict(); self._save_status(); return info

    def calibrate(self,sentence:str|None=None):
        speaker=sorted(self.references)[0]; ref=self.references[speaker]
        if sentence is None: sentence=plan_synthesis_units(self.translation["turns"][0])[0]["unit_text"]
        directory=self.root/"output/calibration"; results={}
        for profile in ("current","space_matched"):
            results[profile]=self.adapter.synthesize(sentence,Path(ref["path"]),ref["transcript"],directory/f"{profile}_settings.wav",generation_profile=profile)
        try: omnivoice_version=importlib.metadata.version("omnivoice")
        except importlib.metadata.PackageNotFoundError: omnivoice_version="unknown"
        manifest=build_calibration_manifest(model_id=self.adapter.model_id,package_versions={"mathula_tv":__import__('mathula_tv').__version__,"omnivoice":omnivoice_version},language_id=self.adapter.language_id,reference=ref,sentence=sentence,results=results)
        self.calibration_manifest_path=directory/"comparison_manifest.json"; atomic_write_json(self.calibration_manifest_path,manifest)
        for name,path in (("current_settings.wav",directory/"current_settings.wav"),("space_matched_settings.wav",directory/"space_matched_settings.wav"),("comparison_manifest.json",self.calibration_manifest_path)):
            relative=f"output/calibration/{name}"; temp=f"{relative}.partial.{self.worker_id}"; self.store.upload(self.job_id,temp,path,if_generation_match=0); self.store.promote(self.job_id,temp,relative,checksum(path),checksum)
        return manifest

    def synthesize_turns(self):
        units=[unit for turn in self.translation["turns"] for unit in plan_synthesis_units(turn)]
        placement_cursor=0.0
        with LeaseHeartbeat(self.store,self.claim,self.lease_seconds) as heartbeat:
            for turn in units:
                ref=self.references[turn["speaker"]]; path=self.root/f"segments/{turn['unit_id']}.wav"
                result=self.adapter.synthesize(turn["unit_text"],Path(ref["path"]),ref["transcript"],path,style={"delivery_style":turn.get("delivery_style"),"emotional_intensity":turn.get("emotional_intensity")})
                available=float(turn["end"])-float(turn["start"]); warnings=[]
                if result.duration>available*1.25: warnings.append("Generated speech exceeds available source duration")
                placed=max(float(turn["start"]),placement_cursor); placement_cursor=placed+result.duration
                record={"unit_id":turn["unit_id"],"segment_id":turn["segment_id"],"speaker":turn["speaker"],"target_text":turn["unit_text"],"approved_segment_text":turn["translated_text"],"planned_start":turn["start"],"placed_start":placed,"reference_interval":[ref["start"],ref["end"]],"available_duration":available,"generated_duration":result.duration,"generation_time":result.generation_time,"real_time_factor":result.real_time_factor,"language_id":result.language_id,"generation_parameters":result.generation_parameters,"warnings":warnings,"status":"synthesized"}
                self.segment_results.append(record); self.clips.append({"segment_id":turn["unit_id"],"start":placed,"duration":result.duration,"path":path})
                if hasattr(self.adapter,"torch") and self.adapter.torch and hasattr(self.adapter.torch,"cuda"): self.adapter.torch.cuda.empty_cache()
        self.claim=heartbeat.claim; return self.segment_results

    def assemble_and_verify(self):
        output=self.root/"output/dubbed_zu.wav"; source_duration=float(self.status["media"]["normalized_audio_duration"])
        assembly=assemble_timeline(self.clips,output,source_duration,24000); quality=inspect_wav(output)
        if not quality["valid"] or quality.get("channels")!=1 or quality.get("sample_rate")!=24000: raise ValueError("Final dubbed WAV failed quality validation")
        synthesized={r["segment_id"] for r in self.segment_results}; accepted={t["segment_id"] for t in self.translation["turns"]}
        if synthesized!=accepted: raise ValueError("Not every accepted translation segment was synthesized")
        try: omnivoice_version=importlib.metadata.version("omnivoice")
        except importlib.metadata.PackageNotFoundError: omnivoice_version="unknown"
        self.output=output; self.report={"schema_version":"processing-report-v1","job_id":self.job_id,"models":{"omnivoice":self.adapter.model_id},"model_id":self.adapter.model_id,"package_versions":{"mathula_tv":__import__('mathula_tv').__version__,"omnivoice":omnivoice_version},"device":self.adapter.device_map,"source_duration":source_duration,"final_duration":quality["duration"],"speaker_count":len(self.references),"selected_references":list(self.references.values()),"segment_results":self.segment_results,"unresolved_speakers":[],"missing_turns":[],"duration_mismatches":[r["segment_id"] for r in self.segment_results if r["warnings"]],"overlaps":assembly["overlaps"],"time_adjustments":assembly["time_adjustments"],"synthesis_failures":[],"fallbacks":self.fallbacks,"loudness_measurements":quality,"total_model_load_time":self.adapter.load_duration,"total_generation_time":sum(r["generation_time"] for r in self.segment_results),"overall_real_time_factor":sum(r["generation_time"] for r in self.segment_results)/source_duration,"output_sha256":checksum(output),"human_review_warnings":["Human review required before publication"]+(["Configured fallback references require extra review"] if self.fallbacks else []),"review_warnings":["Human review required before publication"],"generated_at":datetime.now(timezone.utc).isoformat()}
        self.report_path=self.root/"output/processing_report.json"; atomic_write_json(self.report_path,self.report); return self.report

    def upload_results(self):
        for relative,path,key in (("output/dubbed_zu.wav",self.output,"dubbed_audio"),("output/processing_report.json",self.report_path,"processing_report")):
            digest=checksum(path); temp=f"{relative}.partial.{self.worker_id}"; self.store.upload(self.job_id,temp,path,if_generation_match=0)
            self.status["objects"][key]=self.store.promote(self.job_id,temp,relative,digest,checksum)
        return self.status["objects"]

    def complete(self):
        job=JobManifest.from_dict(self.status); job.completed_stages.append("omnivoice_synthesis"); job.providers["synthesis"]="k2-fsa/OmniVoice"; job.media["synthesis_duration"]=self.report["final_duration"]; job.media["synthesis_generation_time"]=round(self.report["total_generation_time"],3); job.transition("synthesis_ready")
        self.status=job.to_dict(); self._save_status(); self.claim=self.store.finish_claim(self.claim,"completed"); return self.status

    def fail(self,exc:Exception):
        if self.status and self.status.get("state") not in {"synthesis_ready","failed_retryable","failed_terminal"}:
            try:
                job=JobManifest.from_dict(self.status); job.last_error={"stage":"omnivoice_synthesis","error_type":type(exc).__name__,"message":str(exc)[:500],"retryable":True,"at":datetime.now(timezone.utc).isoformat()}; job.transition("failed_retryable"); self.status=job.to_dict(); self._save_status()
            except Exception: pass
        if self.claim:
            try: self.store.finish_claim(self.claim,"failed",f"{type(exc).__name__}: {exc}")
            except Exception: pass
