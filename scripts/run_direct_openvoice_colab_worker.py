#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PLAN_SCHEMA = "mathula-direct-openvoice-colab-plan-v1"
MANIFEST_SCHEMA = "mathula-direct-openvoice-colab-manifest-v1"
QUEUE_SCHEMA = "mathula-direct-openvoice-colab-queue-v1"
QUEUE_PENDING_ROOT = "workers/openvoice/direct/queue/pending"
QUEUE_COMPLETED_ROOT = "workers/openvoice/direct/queue/completed"
QUEUE_FAILED_ROOT = "workers/openvoice/direct/queue/failed"
SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_plan_sha256(plan: dict[str, Any]) -> str:
    payload = dict(plan)
    claimed = str(payload.pop("plan_sha256", ""))
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    actual = hashlib.sha256(encoded).hexdigest()
    if claimed and claimed != actual:
        raise RuntimeError(
            f"plan SHA-256 mismatch: claimed={claimed} actual={actual}"
        )
    return actual


def global_name(prefix: str, relative: str) -> str:
    clean_prefix = prefix.strip("/")
    clean_relative = relative.strip("/")
    return f"{clean_prefix}/{clean_relative}" if clean_prefix else clean_relative


def job_name(prefix: str, job_id: str, relative: str) -> str:
    return global_name(prefix, f"jobs/{job_id}/{relative.strip('/')}")


def queue_name(prefix: str, state: str, job_id: str) -> str:
    roots = {
        "pending": QUEUE_PENDING_ROOT,
        "completed": QUEUE_COMPLETED_ROOT,
        "failed": QUEUE_FAILED_ROOT,
    }
    return global_name(prefix, f"{roots[state]}/{job_id}.json")


def load_json_blob(blob: Any) -> dict[str, Any]:
    return json.loads(blob.download_as_text())


def upload_json_blob(blob: Any, payload: dict[str, Any]) -> None:
    blob.upload_from_string(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        content_type="application/json",
    )


class DirectOpenVoiceWorker:
    def __init__(
        self,
        *,
        bucket_name: str,
        prefix: str,
        device: str,
        tau: float,
        converter_root: Path,
        force: bool = False,
    ) -> None:
        from google.cloud import storage

        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        self.bucket_name = bucket_name
        self.prefix = prefix.strip("/")
        self.device = device
        self.tau = tau
        self.converter_root = converter_root
        self.force = force
        self._converter: Any | None = None
        self._se_extractor: Any | None = None
        self._target_embedding_cache: dict[str, Any] = {}
        self.work_root = Path(
            os.environ.get(
                "MATHULA_TV_OPENVOICE_WORK_ROOT",
                "/content/mathula-openvoice-direct",
            )
        ).resolve()
        self.work_root.mkdir(parents=True, exist_ok=True)

    def _load_runtime(self) -> None:
        if self._converter is not None:
            return
        try:
            from openvoice import se_extractor
            from openvoice.api import ToneColorConverter
        except ImportError as exc:
            raise RuntimeError(
                "OpenVoice is not installed in this Colab runtime. "
                "Run the notebook's existing OpenVoice setup cell once."
            ) from exc

        config = self.converter_root / "config.json"
        checkpoint = self.converter_root / "checkpoint.pth"
        if not config.is_file() or not checkpoint.is_file():
            raise FileNotFoundError(
                "OpenVoice converter files were not found. Expected: "
                f"{config} and {checkpoint}"
            )
        converter = ToneColorConverter(str(config), device=self.device)
        converter.load_ckpt(str(checkpoint))
        self._converter = converter
        self._se_extractor = se_extractor
        print(
            json.dumps(
                {
                    "event": "openvoice_runtime_loaded",
                    "device": self.device,
                    "converter_root": str(self.converter_root),
                },
                indent=2,
            )
        )

    def pending_jobs(self) -> list[dict[str, Any]]:
        root = global_name(self.prefix, f"{QUEUE_PENDING_ROOT}/")
        jobs: list[dict[str, Any]] = []
        for blob in self.client.list_blobs(self.bucket, prefix=root):
            if not blob.name.endswith(".json"):
                continue
            marker = load_json_blob(blob)
            if marker.get("schema_version") != QUEUE_SCHEMA:
                print(f"Skipping unsupported queue marker: gs://{self.bucket_name}/{blob.name}")
                continue
            job_id = str(marker.get("job_id") or "")
            if not SAFE_ID_RE.fullmatch(job_id):
                print(f"Skipping invalid queued job ID: {job_id!r}")
                continue
            marker["_blob_name"] = blob.name
            jobs.append(marker)
        jobs.sort(key=lambda item: str(item.get("queued_at") or ""))
        return jobs

    def _manifest_is_current(self, job_id: str, plan_sha256: str) -> bool:
        blob = self.bucket.blob(
            job_name(
                self.prefix,
                job_id,
                "dubbing/openvoice/direct/manifest.json",
            )
        )
        if not blob.exists(self.client):
            return False
        try:
            manifest = load_json_blob(blob)
        except Exception:
            return False
        return (
            manifest.get("schema_version") == MANIFEST_SCHEMA
            and manifest.get("plan_sha256") == plan_sha256
        )

    def _finish_queue(
        self,
        *,
        job_id: str,
        pending_blob_name: str | None,
        state: str,
        payload: dict[str, Any],
    ) -> None:
        status_blob = self.bucket.blob(queue_name(self.prefix, state, job_id))
        upload_json_blob(status_blob, payload)
        if pending_blob_name:
            self.bucket.blob(pending_blob_name).delete()

    def process_job(
        self,
        job_id: str,
        *,
        queued_plan_sha256: str | None = None,
        pending_blob_name: str | None = None,
    ) -> dict[str, Any]:
        if not SAFE_ID_RE.fullmatch(job_id):
            raise ValueError(f"Unsafe job ID: {job_id!r}")

        plan_blob = self.bucket.blob(
            job_name(
                self.prefix,
                job_id,
                "dubbing/openvoice/direct/plan.json",
            )
        )
        if not plan_blob.exists(self.client):
            raise FileNotFoundError(
                f"OpenVoice plan not found for job {job_id}"
            )
        plan = load_json_blob(plan_blob)
        if plan.get("schema_version") != PLAN_SCHEMA:
            raise RuntimeError(f"Unsupported plan schema for job {job_id}")
        plan_sha256 = canonical_plan_sha256(plan)
        if queued_plan_sha256 and queued_plan_sha256 != plan_sha256:
            raise RuntimeError(
                f"Queued plan SHA does not match current plan for {job_id}"
            )

        if not self.force and self._manifest_is_current(job_id, plan_sha256):
            result = {
                "schema_version": "mathula-direct-openvoice-queue-result-v1",
                "job_id": job_id,
                "status": "already_complete",
                "plan_sha256": plan_sha256,
                "completed_at": utc_now(),
            }
            self._finish_queue(
                job_id=job_id,
                pending_blob_name=pending_blob_name,
                state="completed",
                payload=result,
            )
            print(json.dumps(result, indent=2))
            return result

        self._load_runtime()
        assert self._converter is not None
        assert self._se_extractor is not None

        job_root = self.work_root / "jobs" / job_id
        if job_root.exists():
            shutil.rmtree(job_root)
        source_root = job_root / "source"
        reference_root = job_root / "references"
        output_root = job_root / "output"
        source_root.mkdir(parents=True, exist_ok=True)
        reference_root.mkdir(parents=True, exist_ok=True)
        output_root.mkdir(parents=True, exist_ok=True)

        results: list[dict[str, Any]] = []
        for index, item in enumerate(plan.get("items", []), start=1):
            block_id = str(item["block_id"])
            speaker_id = str(item["speaker_id"])
            if not SAFE_ID_RE.fullmatch(block_id) or not SAFE_ID_RE.fullmatch(speaker_id):
                raise ValueError(
                    f"Unsafe block or speaker ID: {block_id!r}, {speaker_id!r}"
                )

            source_path = source_root / f"{block_id}.wav"
            reference_path = reference_root / f"{speaker_id}.wav"
            output_path = output_root / f"{block_id}.wav"

            source_blob = self.bucket.blob(
                job_name(
                    self.prefix,
                    job_id,
                    str(item["source_audio"]["relative"]),
                )
            )
            source_blob.download_to_filename(source_path)
            if sha256_file(source_path) != item["source_audio"]["sha256"]:
                raise RuntimeError(f"Source checksum mismatch for {block_id}")

            reference_sha = str(item["target_reference"]["sha256"])
            if reference_sha not in self._target_embedding_cache:
                reference_blob = self.bucket.blob(
                    job_name(
                        self.prefix,
                        job_id,
                        str(item["target_reference"]["relative"]),
                    )
                )
                reference_blob.download_to_filename(reference_path)
                if sha256_file(reference_path) != reference_sha:
                    raise RuntimeError(
                        f"Reference checksum mismatch for {speaker_id}"
                    )
                target_se, _ = self._se_extractor.get_se(
                    str(reference_path),
                    self._converter,
                    vad=True,
                )
                self._target_embedding_cache[reference_sha] = target_se

            source_se, _ = self._se_extractor.get_se(
                str(source_path),
                self._converter,
                vad=False,
            )
            self._converter.convert(
                audio_src_path=str(source_path),
                src_se=source_se,
                tgt_se=self._target_embedding_cache[reference_sha],
                output_path=str(output_path),
                message="Mathula TV direct OpenVoice conversion",
                tau=self.tau,
            )
            output_sha256 = sha256_file(output_path)
            output_relative = str(item["output_relative"])
            self.bucket.blob(
                job_name(self.prefix, job_id, output_relative)
            ).upload_from_filename(output_path)
            results.append(
                {
                    "block_id": block_id,
                    "speaker_id": speaker_id,
                    "start_ms": int(item["start_ms"]),
                    "target_duration_ms": int(item["target_duration_ms"]),
                    "output_relative": output_relative,
                    "output_sha256": output_sha256,
                }
            )
            print(
                f"[{index}/{len(plan['items'])}] {job_id} {block_id} "
                f"speaker={speaker_id}"
            )

        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "job_id": job_id,
            "plan_sha256": plan_sha256,
            "reference_policy": "original_english_canonical_speaker_audio",
            "worker_mode": "queue",
            "device": self.device,
            "tau": self.tau,
            "completed_at": utc_now(),
            "items": results,
        }
        manifest_blob_name = job_name(
            self.prefix,
            job_id,
            "dubbing/openvoice/direct/manifest.json",
        )
        upload_json_blob(self.bucket.blob(manifest_blob_name), manifest)
        result = {
            "schema_version": "mathula-direct-openvoice-queue-result-v1",
            "job_id": job_id,
            "status": "completed",
            "plan_sha256": plan_sha256,
            "completed_items": len(results),
            "manifest": f"gs://{self.bucket_name}/{manifest_blob_name}",
            "completed_at": utc_now(),
        }
        self._finish_queue(
            job_id=job_id,
            pending_blob_name=pending_blob_name,
            state="completed",
            payload=result,
        )
        print(json.dumps(result, indent=2))
        return result

    def process_pending_once(self, *, max_jobs: int = 0) -> int:
        pending = self.pending_jobs()
        if max_jobs > 0:
            pending = pending[:max_jobs]
        if not pending:
            print("No pending Mathula OpenVoice jobs.")
            return 0

        completed = 0
        for marker in pending:
            job_id = str(marker["job_id"])
            try:
                self.process_job(
                    job_id,
                    queued_plan_sha256=str(marker.get("plan_sha256") or ""),
                    pending_blob_name=str(marker.get("_blob_name") or "") or None,
                )
                completed += 1
            except Exception as exc:
                failure = {
                    "schema_version": "mathula-direct-openvoice-queue-result-v1",
                    "job_id": job_id,
                    "status": "failed",
                    "plan_sha256": marker.get("plan_sha256"),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "failed_at": utc_now(),
                }
                self._finish_queue(
                    job_id=job_id,
                    pending_blob_name=str(marker.get("_blob_name") or "") or None,
                    state="failed",
                    payload=failure,
                )
                print(json.dumps(failure, indent=2))
        return completed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Process one job or continuously watch the Mathula OpenVoice "
            "GCS queue. With no job ID, all pending jobs are processed once."
        )
    )
    parser.add_argument("job_id", nargs="?")
    parser.add_argument(
        "--bucket",
        default=os.getenv("MATHULA_TV_GCS_BUCKET", ""),
    )
    parser.add_argument(
        "--prefix",
        default=os.getenv("MATHULA_TV_GCS_PREFIX", "mathula-tv"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tau", type=float, default=0.3)
    parser.add_argument(
        "--converter-root",
        default=os.getenv(
            "OPENVOICE_CONVERTER_ROOT",
            "/content/private/openvoice-v2/converter",
        ),
    )
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--max-jobs", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.bucket:
        raise SystemExit("MATHULA_TV_GCS_BUCKET or --bucket is required")
    if args.poll_seconds < 5:
        raise SystemExit("--poll-seconds must be at least 5")

    worker = DirectOpenVoiceWorker(
        bucket_name=args.bucket,
        prefix=args.prefix,
        device=args.device,
        tau=args.tau,
        converter_root=Path(args.converter_root).resolve(),
        force=args.force,
    )

    if args.job_id:
        worker.process_job(args.job_id)
        return 0

    if not args.watch:
        worker.process_pending_once(max_jobs=args.max_jobs)
        return 0

    print(
        json.dumps(
            {
                "event": "watching_openvoice_queue",
                "bucket": args.bucket,
                "prefix": args.prefix,
                "poll_seconds": args.poll_seconds,
                "device": args.device,
            },
            indent=2,
        )
    )
    try:
        while True:
            worker.process_pending_once(max_jobs=args.max_jobs)
            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        print("OpenVoice queue worker stopped.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
