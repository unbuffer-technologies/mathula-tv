#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"Missing file: {path}")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {path}: {exc}") from exc


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def items_from_transcript(value: Any, path: Path) -> tuple[str, list[dict[str, Any]]]:
    if not isinstance(value, dict):
        raise SystemExit(f"Transcript root must be an object: {path}")

    for key in ("segments", "turns", "units"):
        items = value.get(key)
        if isinstance(items, list):
            if not all(isinstance(item, dict) for item in items):
                raise SystemExit(f"{key} must contain JSON objects: {path}")
            return key, items

    raise SystemExit(
        f"Transcript must contain one of segments, turns, or units: {path}"
    )


def first_text(item: dict[str, Any], candidates: tuple[str, ...]) -> str:
    for key in candidates:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def validate_alignment(
    source: dict[str, Any],
    translated: dict[str, Any],
    source_path: Path,
    translated_path: Path,
) -> dict[str, Any]:
    source_key, source_items = items_from_transcript(source, source_path)
    translated_key, translated_items = items_from_transcript(
        translated, translated_path
    )

    if len(source_items) != len(translated_items):
        raise SystemExit(
            "Segment count mismatch: "
            f"source={len(source_items)}, translated={len(translated_items)}"
        )

    missing_translation: list[int] = []
    timing_mismatches: list[int] = []

    for index, (src, dst) in enumerate(zip(source_items, translated_items)):
        translated_text = first_text(
            dst,
            (
                "translated_text",
                "target_text",
                "text_zu",
                "tts_text",
                "display_translation",
                "text",
                "source_text",
            ),
        )
        if not translated_text:
            missing_translation.append(index)

        for start_key, end_key in (
            ("start", "end"),
            ("start_seconds", "end_seconds"),
            ("offset", "end"),
        ):
            if start_key in src and start_key in dst:
                try:
                    if abs(float(src[start_key]) - float(dst[start_key])) > 0.02:
                        timing_mismatches.append(index)
                except (TypeError, ValueError):
                    timing_mismatches.append(index)
                break

    if missing_translation:
        preview = ", ".join(map(str, missing_translation[:12]))
        raise SystemExit(
            f"Translated transcript has empty text at segment indexes: {preview}"
        )

    if timing_mismatches:
        preview = ", ".join(map(str, sorted(set(timing_mismatches))[:12]))
        raise SystemExit(
            "Translated transcript changed source timings at segment indexes: "
            f"{preview}"
        )

    return {
        "source_structure": source_key,
        "translated_structure": translated_key,
        "segment_count": len(source_items),
    }


def clean_title(value: str) -> str:
    value = " ".join(value.replace("：", ":").split())
    return value[:100].rstrip(" -:|") or "Mathula TV isiZulu Dub"


def load_source_title(job_dir: Path, manifest: dict[str, Any]) -> str:
    youtube_source = job_dir / "input" / "youtube_source.json"
    if youtube_source.is_file():
        data = read_json(youtube_source)
        title = data.get("title")
        if isinstance(title, str) and title.strip():
            return clean_title(title)

    source_name = manifest.get("source_filename")
    if isinstance(source_name, str) and source_name.strip():
        return clean_title(Path(source_name).stem)

    return "Mathula TV isiZulu Dub"


def build_fallback_seo(
    *,
    job_id: str,
    title: str,
    translated: dict[str, Any],
) -> dict[str, Any]:
    description = (
        f"{title}\n\n"
        "Le vidiyo ihunyushelwe futhi yadabhelwa ngesiZulu yi-Mathula TV."
    )
    tags = [
        "Mathula TV",
        "izindaba ngesiZulu",
        "isiZulu dub",
        "South Africa",
        "Zulu news",
    ]
    hashtags = [
        "#IsiZulu",
        "#IzindabaNgesiZulu",
    ]

    return {
        "schema_version": "mathula-youtube-zu-v1",
        "job_id": job_id,
        "language": "zu-ZA",
        "title": title,
        "youtube_title": title,
        "description": description,
        "youtube_description": description,
        "tags": tags,
        "youtube_tags": tags,
        "thumbnail": title[:70],
        "youtube_thumbnail": title[:70],
        "chapters": [],
        "youtube_chapters": [],
        "tiktok_caption": title,
        "tiktok_hashtags": hashtags,
        "hashtags": hashtags,
        "source": "manual-translation-fallback",
        "translated_segment_count": len(
            items_from_transcript(translated, Path("translated transcript"))[1]
        ),
    }


def backup_existing(path: Path, backup_dir: Path) -> None:
    if path.exists():
        backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup_dir / path.name)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Install a manually translated Mathula TV transcript and create "
            "all compatibility artifacts required by dub-azure."
        )
    )
    parser.add_argument("job_id")
    parser.add_argument("translated_json", type=Path)
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path("/home/mokgethwa/mathula-tv"),
    )
    parser.add_argument(
        "--tiktok-seo-json",
        "--seo-json",
        dest="seo_json",
        type=Path,
        help=(
            "Optional reviewed TikTok SEO JSON. When omitted, the configured "
            "single editorial AI call is deferred until rendered dub blocks exist."
        ),
    )
    parser.add_argument(
        "--skip-migrate",
        action="store_true",
        help="Do not call migrate-dubbing-state after installation.",
    )
    args = parser.parse_args()

    repo = args.repo.resolve()
    job_dir = repo / "working" / "jobs" / args.job_id
    manifest_path = job_dir / "manifest.json"
    if not manifest_path.is_file():
        alternate = job_dir / "job.json"
        if alternate.is_file():
            manifest_path = alternate
        else:
            raise SystemExit(f"Job manifest not found under: {job_dir}")

    translated_path = args.translated_json.resolve()
    source_path = job_dir / "analysis" / "transcript_en.json"

    manifest = read_json(manifest_path)
    source = read_json(source_path)
    translated = read_json(translated_path)
    validation = validate_alignment(
        source, translated, source_path, translated_path
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = job_dir / "translation" / "backups" / stamp

    translation_target = job_dir / "translation" / "transcript_zu.json"
    dubbing_target = job_dir / "dubbing" / "translation_zu.json"
    language_code = str(manifest.get("target_language") or "zu-ZA").split("-", 1)[0]
    seo_target = job_dir / "translation" / f"tiktok_{language_code}.json"

    if args.seo_json:
        seo = read_json(args.seo_json.resolve())
        if not isinstance(seo, dict):
            raise SystemExit("SEO JSON root must be an object")
        seo.setdefault("job_id", args.job_id)
        seo.setdefault("language", "zu-ZA")
        seo.setdefault("schema_version", "mathula-tiktok-seo-v1")
        seo.setdefault("platform", "tiktok")
    else:
        from mathula_tv.localized_seo import build_deferred_localized_seo

        seo = build_deferred_localized_seo(
            job_id=args.job_id,
            target_language=str(manifest.get("target_language") or "zu-ZA"),
        )

    # Resolve and validate every input, including external AI output, before
    # replacing any installed artifact.
    for path in (translation_target, dubbing_target, seo_target):
        backup_existing(path, backup_dir)

    write_json(translation_target, translated)
    write_json(dubbing_target, translated)
    write_json(seo_target, seo)

    migration_output = ""
    job_state = str(manifest.get("state", ""))

    if not args.skip_migrate and job_state == "synthesis_queued":
        command = [
            str(repo / ".venv" / "bin" / "python"),
            "-m",
            "mathula_tv.cli",
            "migrate-dubbing-state",
            args.job_id,
        ]
        completed = subprocess.run(
            command,
            cwd=repo,
            text=True,
            capture_output=True,
        )
        migration_output = (completed.stdout + completed.stderr).strip()
        if completed.returncode != 0:
            raise SystemExit(
                "Artifacts were installed, but migrate-dubbing-state failed:\n"
                + migration_output
            )
    elif not args.skip_migrate:
        migration_output = (
            f"Migration skipped: job state {job_state!r} does not require "
            "legacy synthesis_queued migration"
        )

    result = {
        "job_id": args.job_id,
        "status": "manual_translation_installed",
        **validation,
        "translation_file": str(translation_target),
        "dubbing_compatibility_file": str(dubbing_target),
        "seo_file": str(seo_target),
        "backup_directory": str(backup_dir) if backup_dir.exists() else None,
        "migration_output": migration_output,
        "next_command": (
            f"python -m mathula_tv.cli dub-azure {args.job_id} --live-operation"
        ),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
