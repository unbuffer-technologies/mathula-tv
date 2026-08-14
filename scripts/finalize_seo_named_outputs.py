#!/usr/bin/env python3
"""Create title-named MP4 and individual SEO files for an existing job."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mathula_tv.atomic_io import atomic_write_json, read_json
from mathula_tv.config import load_settings
from mathula_tv.job_store import JobStore
from mathula_tv.media import checksum
from mathula_tv.output_naming import (
    load_seo_output_context,
    publish_title_named_outputs,
)


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser()
    command.add_argument("job_id")
    return command


def _file_checksums(paths: dict[str, Path]) -> dict[str, str]:
    return {key: checksum(path) for key, path in sorted(paths.items())}


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    settings = load_settings()
    jobs = JobStore(settings.work_dir)
    job = jobs.load(args.job_id)
    job_root = jobs.job_dir(job.job_id)
    canonical_video = job_root / "direct_dub" / "final_dubbed.mp4"

    context = load_seo_output_context(job_root, job.target_language)
    outputs = publish_title_named_outputs(
        job_root,
        canonical_video,
        context=context,
    )
    seo_files = {key: str(path) for key, path in sorted(outputs.seo_files.items())}
    seo_checksums = _file_checksums(dict(outputs.seo_files))

    report_path = job_root / "direct_dub" / "report.json"
    if report_path.is_file():
        report = read_json(report_path)
        report_outputs = report.setdefault("outputs", {})
        report_outputs.pop("seo_package", None)
        report_outputs.update(
            {
                "canonical_final_video": str(canonical_video),
                "final_video": str(outputs.final_video),
                "seo_files": seo_files,
                "seo_title": outputs.title,
                "filename_stem": outputs.filename_stem,
            }
        )
        atomic_write_json(report_path, report)

    job.media.pop("direct_dub_seo_package", None)
    job.media.pop("direct_dub_seo_package_sha256", None)
    job.media["direct_dub_canonical_video"] = str(canonical_video)
    job.media["direct_dub_video"] = str(outputs.final_video)
    job.media["direct_dub_video_sha256"] = checksum(outputs.final_video)
    job.media["direct_dub_seo_files"] = seo_files
    job.media["direct_dub_seo_file_sha256"] = seo_checksums
    job.media["direct_dub_seo_title"] = outputs.title
    jobs.save(job)

    payload: dict[str, Any] = {
        "job_id": job.job_id,
        **outputs.to_dict(),
        "final_video_sha256": checksum(outputs.final_video),
        "seo_file_sha256": seo_checksums,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
