from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any

from .atomic_io import atomic_write_json, read_json
from .config import load_settings
from .gcs_store import GCSStore
from .job_store import JobStore
from .media import checksum, probe
from .mixing import render_dialogue_tracks
from .rendering import render_review_mp4

PLAN_SCHEMA = "mathula-direct-openvoice-colab-plan-v1"
MANIFEST_SCHEMA = "mathula-direct-openvoice-colab-manifest-v1"


def _sha(path: Path) -> str:
    return checksum(path)


def _store() -> GCSStore:
    settings = load_settings()
    return GCSStore(settings.gcs_bucket, prefix=settings.gcs_prefix)


def prepare_direct_openvoice_job(job_id: str, *, force: bool = False) -> dict[str, Any]:
    settings = load_settings()
    jobs = JobStore(settings.work_dir)
    job = jobs.load(job_id)
    job_root = jobs.job_dir(job_id)
    direct_root = job_root / "direct_dub"
    report_path = direct_root / "report.json"
    resolution_path = direct_root / "voice_resolution.json"
    if not report_path.is_file() or not resolution_path.is_file():
        raise FileNotFoundError("Run the direct Azure dub and voice resolution first")
    report = read_json(report_path)
    resolution = read_json(resolution_path)
    resolved = {str(x["speaker_id"]): x for x in resolution.get("speakers", []) if x.get("status") == "resolved"}
    store = _store()
    items=[]
    uploaded_refs=set()
    for block in report.get("blocks", []):
        block_id=str(block["block_id"])
        speaker_id=str(block["speaker_id"])
        source=Path(block["output_path"])
        reference=job_root / "audio" / "speakers" / f"{speaker_id}.wav"
        if not source.is_file(): raise FileNotFoundError(source)
        if not reference.is_file(): raise FileNotFoundError(reference)
        if speaker_id not in resolved: raise RuntimeError(f"Unresolved speaker {speaker_id}")
        source_rel=f"dubbing/openvoice/direct/input/blocks/{block_id}.wav"
        ref_rel=f"dubbing/openvoice/direct/input/references/{speaker_id}.wav"
        if force or speaker_id not in uploaded_refs:
            store.upload(job_id, ref_rel, reference)
            uploaded_refs.add(speaker_id)
        store.upload(job_id, source_rel, source)
        items.append({
            "block_id": block_id,
            "speaker_id": speaker_id,
            "start_ms": int(block["start_ms"]),
            "target_duration_ms": int(block["duration_ms"]),
            "azure_voice": str(block.get("voice") or resolved[speaker_id].get("selected_voice") or ""),
            "source_audio": {"relative": source_rel, "sha256": _sha(source)},
            "target_reference": {"relative": ref_rel, "sha256": _sha(reference)},
            "output_relative": f"dubbing/openvoice/direct/output/blocks/{block_id}.wav",
        })
    if not items: raise RuntimeError("Direct dub report contains no blocks")
    plan={
        "schema_version": PLAN_SCHEMA,
        "job_id": job_id,
        "source": "direct_azure_dub",
        "reference_policy": "original_english_canonical_speaker_audio",
        "source_voice_policy": "azure_isiZulu_block_audio",
        "items": items,
    }
    canonical=json.dumps(plan,ensure_ascii=False,sort_keys=True,separators=(",",":" )).encode()
    plan["plan_sha256"]=hashlib.sha256(canonical).hexdigest()
    local=direct_root/"openvoice_colab"/"plan.json"
    atomic_write_json(local,plan)
    store.upload_json(job_id,"dubbing/openvoice/direct/plan.json",plan)
    return {"plan":str(local),"items":len(items),"gcs":f"gs://{settings.gcs_bucket}/{store.name(job_id,'dubbing/openvoice/direct/plan.json')}"}


def reconcile_direct_openvoice_job(job_id: str, *, force: bool = False) -> dict[str, Any]:
    settings=load_settings(); jobs=JobStore(settings.work_dir); job=jobs.load(job_id); root=jobs.job_dir(job_id)
    direct=root/"direct_dub"; out=direct/"openvoice_colab"; out.mkdir(parents=True,exist_ok=True)
    store=_store(); manifest,_=store.download_json(job_id,"dubbing/openvoice/direct/manifest.json")
    if manifest.get("schema_version") != MANIFEST_SCHEMA: raise RuntimeError("Unsupported OpenVoice direct manifest")
    clips=[]; downloaded=[]
    for item in manifest.get("items",[]):
        block_id=str(item["block_id"]); path=out/"blocks"/f"{block_id}.wav"
        if force or not path.is_file(): store.download(job_id,str(item["output_relative"]),path)
        if _sha(path) != item["output_sha256"]: raise RuntimeError(f"Checksum mismatch for {block_id}")
        duration_ms=round(probe(path)["duration"]*1000)
        clips.append({"unit_id":block_id,"speaker_id":item["speaker_id"],"path":str(path),"start_ms":int(item["start_ms"]),"duration_ms":duration_ms})
        downloaded.append({**item,"local_path":str(path),"duration_ms":duration_ms})
    source=Path(job.local_source_path); total=round(probe(source)["duration"]*1000)
    dialogue=out/"dialogue.wav"
    dialogue_manifest=render_dialogue_tracks(clips,out/"dialogue_tracks",dialogue,total_duration_ms=total,sample_rate=24000)
    video=out/"final_cloned.mp4"; render_manifest=render_review_mp4(source,dialogue,video,out/"render_manifest.json")
    report={"schema_version":"mathula-direct-openvoice-colab-render-v1","job_id":job_id,"reference_policy":"original_english_canonical_speaker_audio","items":downloaded,"dialogue":dialogue_manifest,"render":render_manifest,"output_video":str(video)}
    atomic_write_json(out/"report.json",report)
    return report
