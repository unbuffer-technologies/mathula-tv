#!/usr/bin/env python3
import argparse, json
from mathula_tv.direct_openvoice_colab import prepare_direct_openvoice_job
p=argparse.ArgumentParser(); p.add_argument('job_id'); p.add_argument('--force',action='store_true'); a=p.parse_args()
print(json.dumps(prepare_direct_openvoice_job(a.job_id,force=a.force),indent=2))
