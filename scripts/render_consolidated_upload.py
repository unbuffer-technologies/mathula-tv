import argparse
from pathlib import Path

from mathula_tv.atomic_io import atomic_write_json, read_json
from mathula_tv.consolidated_render import render_consolidated_upload


parser = argparse.ArgumentParser()
parser.add_argument("plan")
args = parser.parse_args()
plan_path = Path(args.plan).resolve()
plan = read_json(plan_path)
result = render_consolidated_upload(
    source_video=Path(plan["source_video"]),
    clips=plan["clips"],
    panel_image=Path(plan["panel_image"]),
    output_path=Path(plan["output_path"]),
    total_duration_ms=int(plan["total_duration_ms"]),
    cut_start_ms=int(plan["cut_start_ms"]),
)
report_path = Path(plan["report_path"])
atomic_write_json(report_path, result)
print(report_path)
