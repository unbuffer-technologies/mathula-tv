from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .logging_utils import redact


class LeaseConflict(RuntimeError):
    pass


class LeaseLost(RuntimeError):
    pass


def object_name(prefix: str, job_id: str, relative: str) -> str:
    return f"{prefix.strip('/')}/jobs/{job_id}/{relative.strip('/')}"


def gs_uri(bucket: str, name: str) -> str:
    return f"gs://{bucket}/{name}"


@dataclass
class GCSStore:
    bucket_name: str
    prefix: str = "mathula-tv"
    client: Any = None

    def __post_init__(self) -> None:
        if not self.bucket_name:
            raise ValueError("MATHULA_TV_GCS_BUCKET is required")
        if self.client is None:
            try:
                from google.cloud import storage
            except ImportError as exc:
                raise RuntimeError("Install google-cloud-storage for GCS support") from exc
            self.client = storage.Client()
        self.bucket = self.client.bucket(self.bucket_name)

    def name(self, job_id: str, relative: str) -> str:
        return object_name(self.prefix, job_id, relative)

    def upload(self, job_id: str, relative: str, path: Path, *, if_generation_match: int | None = None) -> str:
        blob = self.bucket.blob(self.name(job_id, relative))
        blob.upload_from_filename(str(path), if_generation_match=if_generation_match)
        return gs_uri(self.bucket_name, blob.name)

    def upload_json(self, job_id: str, relative: str, value: dict, *, if_generation_match: int | None = None) -> str:
        blob = self.bucket.blob(self.name(job_id, relative))
        blob.upload_from_string(json.dumps(redact(value), ensure_ascii=False), content_type="application/json", if_generation_match=if_generation_match)
        return gs_uri(self.bucket_name, blob.name)

    def download(self, job_id: str, relative: str, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.bucket.blob(self.name(job_id, relative)).download_to_filename(str(path))

    def download_json(self, job_id: str, relative: str) -> tuple[dict, int]:
        blob = self.bucket.blob(self.name(job_id, relative))
        if not blob.exists():
            raise FileNotFoundError(gs_uri(self.bucket_name, blob.name))
        blob.reload()
        return json.loads(blob.download_as_text()), int(blob.generation)

    def claim(self, job_id: str, task_type: str, worker_id: str, attempt: int, lease_seconds: int = 900) -> dict:
        now = datetime.now(timezone.utc)
        relative = f"worker/{task_type}_claim.json"
        blob = self.bucket.blob(self.name(job_id, relative))
        generation = 0
        if blob.exists():
            blob.reload()
            current = json.loads(blob.download_as_text())
            if datetime.fromisoformat(current["lease_expiry"]) > now:
                raise LeaseConflict(f"Unexpired {task_type} lease held by {current['worker_id']}")
            generation = blob.generation
        claim = {"job_id": job_id, "task_type": task_type, "worker_id": worker_id, "claim_timestamp": now.isoformat(), "heartbeat_timestamp": now.isoformat(), "lease_expiry": (now + timedelta(seconds=lease_seconds)).isoformat(), "attempt_number": attempt}
        try:
            blob.upload_from_string(json.dumps(claim), content_type="application/json", if_generation_match=generation)
        except Exception as exc:
            if "condition" in str(exc).lower() or "412" in str(exc):
                raise LeaseConflict("Lease changed concurrently") from exc
            raise
        claim["generation"] = blob.generation
        return claim

    def heartbeat(self, claim: dict, lease_seconds: int = 900) -> dict:
        now = datetime.now(timezone.utc)
        updated = dict(claim, heartbeat_timestamp=now.isoformat(), lease_expiry=(now + timedelta(seconds=lease_seconds)).isoformat())
        generation = updated.pop("generation")
        blob = self.bucket.blob(self.name(claim["job_id"], f"worker/{claim['task_type']}_claim.json"))
        blob.upload_from_string(json.dumps(updated), content_type="application/json", if_generation_match=generation)
        updated["generation"] = blob.generation
        return updated

    def assert_claim(self, claim: dict) -> dict:
        """Verify generation, owner, task, and expiry without extending a lease."""
        blob = self.bucket.blob(self.name(claim["job_id"], f"worker/{claim['task_type']}_claim.json"))
        if not blob.exists():
            raise LeaseLost("Worker lease object no longer exists")
        blob.reload()
        current = json.loads(blob.download_as_text())
        if int(blob.generation) != int(claim["generation"]):
            raise LeaseLost("Worker lease generation changed")
        for field in ("job_id", "task_type", "worker_id", "attempt_number"):
            if current.get(field) != claim.get(field):
                raise LeaseLost(f"Worker lease {field} changed")
        expiry = datetime.fromisoformat(current["lease_expiry"])
        if expiry <= datetime.now(timezone.utc):
            raise LeaseLost("Worker lease expired")
        if current.get("status") in {"completed", "failed"}:
            raise LeaseLost("Worker lease is already final")
        return {**current, "generation": int(blob.generation)}

    def finish_claim(self, claim: dict, status: str, error: str | None = None) -> dict:
        if status not in {"completed", "failed"}:
            raise ValueError("Lease status must be completed or failed")
        now = datetime.now(timezone.utc)
        updated = {k: v for k, v in claim.items() if k != "generation"}
        updated.update(status=status, completed_timestamp=now.isoformat(), lease_expiry=now.isoformat())
        if error:
            updated["error"] = str(error)[:500]
        blob = self.bucket.blob(self.name(claim["job_id"], f"worker/{claim['task_type']}_claim.json"))
        blob.upload_from_string(json.dumps(updated), content_type="application/json", if_generation_match=claim["generation"])
        updated["generation"] = blob.generation
        return updated

    def update_json(self, job_id: str, relative: str, value: dict, generation: int) -> str:
        return self.upload_json(job_id, relative, value, if_generation_match=generation)

    def promote(self, job_id: str, temporary_relative: str, final_relative: str, expected_sha256: str, checksum_fn, *, final_generation: int = 0) -> str:
        temporary_name = self.name(job_id, temporary_relative)
        final_name = self.name(job_id, final_relative)
        temporary = self.bucket.blob(temporary_name)
        if not temporary.exists():
            raise ValueError("Temporary output is missing")
        temporary.reload()
        if not temporary.size or int(temporary.size) <= 0:
            raise ValueError("Temporary output is empty")
        temporary_hash = self._remote_sha256(temporary, checksum_fn)
        if temporary_hash != expected_sha256:
            raise ValueError("Uploaded temporary output checksum mismatch")

        final = self.bucket.blob(final_name)
        if final.exists():
            final.reload()
            self._verify_final(final, expected_sha256, checksum_fn)
            temporary.delete(if_generation_match=temporary.generation)
            return gs_uri(self.bucket_name, final.name)

        try:
            # A final object must not be silently replaced. The legacy argument
            # remains API-compatible, but promotion always creates generation 0.
            final = self.bucket.copy_blob(temporary, self.bucket, final_name, if_generation_match=0)
        except Exception as exc:
            # Another worker may have won the create race. Treat an identical,
            # fully verified final object as success; otherwise reject it.
            raced = self.bucket.blob(final_name)
            if not raced.exists():
                raise RuntimeError(f"GCS promotion failed: {redact(str(exc))}") from exc
            raced.reload()
            try:
                self._verify_final(raced, expected_sha256, checksum_fn)
            except ValueError as conflict:
                raise ValueError("A conflicting final output already exists") from conflict
            final = raced

        # copy_blob's return value is not guaranteed to have loaded metadata.
        final.reload()
        self._verify_final(final, expected_sha256, checksum_fn)
        temporary.delete(if_generation_match=temporary.generation)
        return gs_uri(self.bucket_name, final.name)

    @staticmethod
    def _remote_sha256(blob: Any, checksum_fn) -> str:
        import tempfile
        with tempfile.NamedTemporaryFile() as handle:
            blob.download_to_filename(handle.name)
            return checksum_fn(Path(handle.name))

    def _verify_final(self, blob: Any, expected_sha256: str, checksum_fn) -> None:
        if not blob.size or int(blob.size) <= 0:
            raise ValueError("Promoted final output is empty")
        if self._remote_sha256(blob, checksum_fn) != expected_sha256:
            raise ValueError("A conflicting final output already exists")


class GCSLeaseHeartbeat:
    """Generic background heartbeat shared by GPU and diagnostic workers."""

    def __init__(
        self,
        store: GCSStore,
        claim: dict,
        lease_seconds: int = 900,
        interval_seconds: float | None = None,
    ) -> None:
        self.store = store
        self.claim = claim
        self.lease_seconds = lease_seconds
        self.interval_seconds = interval_seconds or max(15.0, lease_seconds / 3)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.error: Exception | None = None
        self._lock = threading.Lock()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                with self._lock:
                    self.claim = self.store.heartbeat(self.claim, self.lease_seconds)
            except Exception as exc:
                self.error = exc
                self._stop.set()

    def assert_owned(self, _unit_id: str | None = None) -> bool:
        if self.error:
            raise LeaseLost("Background lease heartbeat failed") from self.error
        with self._lock:
            self.claim = self.store.assert_claim(self.claim)
        return True

    def __enter__(self) -> "GCSLeaseHeartbeat":
        self.store.assert_claim(self.claim)
        self._thread.start()
        return self

    def __exit__(self, exc_type, _exc, _traceback) -> None:
        self._stop.set()
        self._thread.join()
        if self.error is not None and exc_type is None:
            raise LeaseLost("Background lease heartbeat failed") from self.error
