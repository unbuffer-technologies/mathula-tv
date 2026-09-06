#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from mathula_tv.atomic_io import atomic_write_json, read_json
from mathula_tv.config import load_settings
from mathula_tv.job_store import JobStore
from mathula_tv.media import checksum
from mathula_tv.output_naming import load_seo_output_context, publish_job_id_transcript_alias, publish_title_named_outputs

def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("job_id"); args=p.parse_args()
    settings=load_settings(); jobs=JobStore(settings.work_dir); job=jobs.load(args.job_id); root=jobs.job_dir(job.job_id)
    result={"job_id":job.job_id}
    transcript=root/"analysis"/"transcript_en.json"
    if transcript.is_file():
        alias=publish_job_id_transcript_alias(root,job.job_id); result["transcript"]=str(alias)
        job.objects["transcript_manual"]=str(alias)
    video=root/"direct_dub"/"final_dubbed.mp4"
    if video.is_file():
        context=load_seo_output_context(root,job.target_language)
        outputs=publish_title_named_outputs(root,video,context=context)
        result["final_video"]=str(outputs.final_video)
        result["seo_files"]={k:str(v) for k,v in sorted(outputs.seo_files.items())}
        job.media["direct_dub_video"]=str(outputs.final_video)
        job.media["direct_dub_video_sha256"]=checksum(outputs.final_video)
        job.media["direct_dub_seo_files"]=result["seo_files"]
    jobs.save(job)
    report=root/"output"/f"manual_artifacts_{job.job_id}.json"; atomic_write_json(report,result)
    print(json.dumps(result,indent=2,ensure_ascii=False)); return 0
if __name__=="__main__": raise SystemExit(main())
