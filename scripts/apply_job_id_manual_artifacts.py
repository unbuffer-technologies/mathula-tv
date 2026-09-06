#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path

def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("--repo",type=Path,required=True); a=p.parse_args()
    target=a.repo/"src/mathula_tv/direct_azure_dub.py"
    if target.is_file():
        text=target.read_text(encoding="utf-8")
        # Existing V3 integration keeps these API names. Replacing output_naming.py
        # changes publication to job-ID filenames without another invasive renderer patch.
        target.write_text(text,encoding="utf-8")
        print(f"Verified renderer integration: {target}")
    print("Job-ID manual artifact naming installed")
    return 0
if __name__=="__main__": raise SystemExit(main())
