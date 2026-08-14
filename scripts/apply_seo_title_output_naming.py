#!/usr/bin/env python3
"""Surgically integrate SEO-title deliverable naming into direct_azure_dub.py."""

from __future__ import annotations

import argparse
from pathlib import Path


IMPORT_ANCHOR = "from .models import JobManifest, utcnow\n"
IMPORT_REPLACEMENT = (
    IMPORT_ANCHOR
    + "from .output_naming import (\n"
    + "    load_seo_output_context,\n"
    + "    publish_title_named_outputs,\n"
    + ")\n"
)

SOURCE_ANCHOR = """        source_video = Path(job.local_source_path)\n        if not source_video.is_file():\n            raise FileNotFoundError(source_video)\n\n        translation_path = translation_path or (\n"""
SOURCE_REPLACEMENT = """        source_video = Path(job.local_source_path)\n        if not source_video.is_file():\n            raise FileNotFoundError(source_video)\n\n        seo_context = load_seo_output_context(\n            self.jobs.job_dir(job.job_id),\n            job.target_language,\n        )\n\n        translation_path = translation_path or (\n"""

FINGERPRINT_CALL_ANCHOR = """            options,\n            clean_background_stem,\n        )\n"""
FINGERPRINT_CALL_REPLACEMENT = """            options,\n            clean_background_stem,\n            seo_context.seo_path,\n        )\n"""

RENDER_ANCHOR = """        render_manifest = render_review_mp4(\n            source_video,\n            artifacts.final_mix,\n            artifacts.final_video,\n            artifacts.render_manifest,\n            runner=self.runner,\n        )\n\n        report = {\n"""
RENDER_REPLACEMENT = """        render_manifest = render_review_mp4(\n            source_video,\n            artifacts.final_mix,\n            artifacts.final_video,\n            artifacts.render_manifest,\n            runner=self.runner,\n        )\n        named_outputs = publish_title_named_outputs(\n            self.jobs.job_dir(job.job_id),\n            artifacts.final_video,\n            context=seo_context,\n        )\n\n        report = {\n"""

OUTPUTS_ANCHOR = """                \"final_mix\": str(artifacts.final_mix),\n                \"final_video\": str(artifacts.final_video),\n                \"report\": str(artifacts.report),\n"""
OUTPUTS_REPLACEMENT = """                \"final_mix\": str(artifacts.final_mix),\n                \"canonical_final_video\": str(artifacts.final_video),\n                \"final_video\": str(named_outputs.final_video),\n                \"seo_package\": str(named_outputs.seo_package),\n                \"seo_title\": named_outputs.title,\n                \"filename_stem\": named_outputs.filename_stem,\n                \"report\": str(artifacts.report),\n"""

JOB_MEDIA_ANCHOR = """        job.providers[\"direct_dubbing\"] = \"azure-tts\"\n        job.media[\"direct_dub_video\"] = str(artifacts.final_video)\n        job.media[\"direct_dub_video_sha256\"] = checksum(artifacts.final_video)\n        job.media[\"direct_dub_report\"] = str(artifacts.report)\n"""
JOB_MEDIA_REPLACEMENT = """        job.providers[\"direct_dubbing\"] = \"azure-tts\"\n        job.media[\"direct_dub_canonical_video\"] = str(artifacts.final_video)\n        job.media[\"direct_dub_video\"] = str(named_outputs.final_video)\n        job.media[\"direct_dub_video_sha256\"] = checksum(named_outputs.final_video)\n        job.media[\"direct_dub_seo_package\"] = str(named_outputs.seo_package)\n        job.media[\"direct_dub_seo_package_sha256\"] = checksum(named_outputs.seo_package)\n        job.media[\"direct_dub_seo_title\"] = named_outputs.title\n        job.media[\"direct_dub_report\"] = str(artifacts.report)\n"""

SIGNATURE_ANCHOR = """        options: DirectDubOptions,\n        clean_background_stem: Path | None,\n    ) -> str:\n"""
SIGNATURE_REPLACEMENT = """        options: DirectDubOptions,\n        clean_background_stem: Path | None,\n        seo_path: Path,\n    ) -> str:\n"""

PAYLOAD_ANCHOR = """            \"translation_sha256\": checksum(translation_path),\n            \"blocks\": [block.to_dict() for block in blocks],\n"""
PAYLOAD_REPLACEMENT = """            \"translation_sha256\": checksum(translation_path),\n            \"seo_sha256\": checksum(seo_path),\n            \"blocks\": [block.to_dict() for block in blocks],\n"""

REUSE_ANCHOR = """        if not artifacts.report.is_file() or not artifacts.final_video.is_file():\n            return None\n        report = read_json(artifacts.report)\n        if report.get(\"request_fingerprint\") != request_fingerprint:\n            return None\n        output = dict(report)\n"""
REUSE_REPLACEMENT = """        if not artifacts.report.is_file() or not artifacts.final_video.is_file():\n            return None\n        report = read_json(artifacts.report)\n        if report.get(\"request_fingerprint\") != request_fingerprint:\n            return None\n        outputs = report.get(\"outputs\")\n        if not isinstance(outputs, Mapping):\n            return None\n        for key in (\"final_video\", \"seo_package\"):\n            path = Path(str(outputs.get(key) or \"\"))\n            if not path.is_file() or path.stat().st_size <= 0:\n                return None\n        output = dict(report)\n"""


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"Expected one {label} anchor, found {count}")
    return text.replace(old, new, 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()

    target = args.repo / "src" / "mathula_tv" / "direct_azure_dub.py"
    text = target.read_text(encoding="utf-8")
    for old, new, label in (
        (IMPORT_ANCHOR, IMPORT_REPLACEMENT, "import"),
        (SOURCE_ANCHOR, SOURCE_REPLACEMENT, "SEO context"),
        (FINGERPRINT_CALL_ANCHOR, FINGERPRINT_CALL_REPLACEMENT, "fingerprint call"),
        (RENDER_ANCHOR, RENDER_REPLACEMENT, "named output publication"),
        (OUTPUTS_ANCHOR, OUTPUTS_REPLACEMENT, "report outputs"),
        (JOB_MEDIA_ANCHOR, JOB_MEDIA_REPLACEMENT, "job media"),
        (SIGNATURE_ANCHOR, SIGNATURE_REPLACEMENT, "fingerprint signature"),
        (PAYLOAD_ANCHOR, PAYLOAD_REPLACEMENT, "fingerprint payload"),
        (REUSE_ANCHOR, REUSE_REPLACEMENT, "reuse validation"),
    ):
        text = replace_once(text, old, new, label)
    target.write_text(text, encoding="utf-8")
    print(f"Patched {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
