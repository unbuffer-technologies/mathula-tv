from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


IMPORT_LINE = "from .webm_ingestion import file_sha256, normalize_webm_to_mp4, validate_supported_video\n"
MARKER = "# Mathula WebM ingestion wrapper v2"

WRAPPER = '''    # Mathula WebM ingestion wrapper v2
    def submit(self, source: Path, target_language: str = "zu-ZA", force: bool = False):
        source = Path(source).expanduser().resolve()
        source_info = validate_supported_video(source)
        original_sha256 = file_sha256(source)

        if source.suffix.lower() == ".mp4":
            job = self._submit_canonical_video(source, target_language, force)
            canonical = Path(job.local_source_path)
            job.media.setdefault("source_ingestion", {
                "schema_version": "source-ingestion-v2",
                "normalized": False,
                "original_filename": source.name,
                "original_path": str(canonical),
                "original_extension": ".mp4",
                "original_sha256": original_sha256,
                "original_duration": float(source_info["duration"]),
                "canonical_path": str(canonical),
                "canonical_sha256": file_sha256(canonical),
                "canonical_duration": float(source_info["duration"]),
            })
            job.objects.setdefault("source_media", str(canonical))
            job.objects.setdefault("canonical_media", str(canonical))
            self.jobs.save(job)
            return job

        import tempfile
        with tempfile.TemporaryDirectory(prefix="mathula-webm-") as temporary_dir:
            canonical_input = Path(temporary_dir) / "source.mp4"
            normalization = normalize_webm_to_mp4(source, canonical_input)
            job = self._submit_canonical_video(canonical_input, target_language, force)

        job_dir = self.jobs.job_dir(job.job_id)
        preserved_original = job_dir / "input" / source.name
        preserved_original.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, preserved_original)
        canonical = Path(job.local_source_path)

        job.source_filename = source.name
        job.objects["original_media"] = str(preserved_original)
        job.objects["source_media"] = str(canonical)
        job.objects["canonical_media"] = str(canonical)
        job.media["source_ingestion"] = {
            "schema_version": "source-ingestion-v2",
            **normalization,
            "original_filename": source.name,
            "original_path": str(preserved_original),
            "original_sha256": original_sha256,
        }
        self.jobs.save(job)
        return job

'''


def latest_backup(pattern: str) -> Path | None:
    matches = sorted(Path("/tmp").glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def restore_v1_backups(repo: Path) -> None:
    targets = {
        "mathula-tv-orchestrator.before-webm.*.py": repo / "src/mathula_tv/orchestrator.py",
        "mathula-tv-media.before-webm.*.py": repo / "src/mathula_tv/media.py",
    }
    for pattern, target in targets.items():
        backup = latest_backup(pattern)
        if backup is None:
            raise SystemExit(f"Cannot locate required v1 backup in /tmp: {pattern}")
        shutil.copy2(backup, target)
        print(f"Restored {target} from {backup}")


def patch_orchestrator(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        print(f"Already patched: {path}")
        return
    if "class Orchestrator" not in text:
        raise SystemExit("Orchestrator class was not found")
    if IMPORT_LINE not in text:
        import_anchor = re.search(r"(?:^from \.\w[^\n]*\n)+", text, flags=re.MULTILINE)
        if not import_anchor:
            raise SystemExit("Could not locate local import block in orchestrator.py")
        insert_at = import_anchor.end()
        text = text[:insert_at] + IMPORT_LINE + text[insert_at:]

    method_pattern = re.compile(r"^    def submit\(", flags=re.MULTILINE)
    match = method_pattern.search(text)
    if not match:
        raise SystemExit("Could not locate Orchestrator.submit")
    text = text[:match.start()] + WRAPPER + text[match.start():]
    text = method_pattern.sub("    def _submit_canonical_video(", text, count=1)
    path.write_text(text, encoding="utf-8")
    print(f"Patched {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--restore-v1-backups", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    if args.restore_v1_backups:
        restore_v1_backups(repo)
    patch_orchestrator(repo / "src/mathula_tv/orchestrator.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
