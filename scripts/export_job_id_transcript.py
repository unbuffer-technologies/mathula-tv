#!/usr/bin/env python3
from __future__ import annotations
import argparse
from mathula_tv.config import load_settings
from mathula_tv.job_store import JobStore
from mathula_tv.output_naming import publish_job_id_transcript_alias

def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("job_id"); a=p.parse_args()
    jobs=JobStore(load_settings().work_dir); job=jobs.load(a.job_id)
    path=publish_job_id_transcript_alias(jobs.job_dir(job.job_id),job.job_id)
    job.objects["transcript_manual"]=str(path); jobs.save(job); print(path); return 0
if __name__=="__main__": raise SystemExit(main())
