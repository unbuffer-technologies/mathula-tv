from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from .atomic_io import atomic_write_json, read_json
from .gcs_store import LeaseConflict, LeaseLost
from .logging_utils import redact


def _safe_relative(relative: str) -> Path:
    value = PurePosixPath(str(relative).replace('\\', '/'))
    if value.is_absolute() or not value.parts or any(part in {'', '.', '..'} for part in value.parts):
        raise ValueError(f'Unsafe local artifact path: {relative!r}')
    return Path(*value.parts)


@dataclass
class LocalStore:
    """Filesystem-backed artifact store used by normal Mathula runs.

    It intentionally mirrors the subset of ``GCSStore`` used by the server
    pipeline.  This keeps the existing durable-artifact/promotion contracts
    intact while making ``working/jobs/<job_id>`` the authoritative store.
    GCS remains available explicitly for Colab/remote-worker workflows.
    """

    work_dir: Path

    def __post_init__(self) -> None:
        self.root = Path(self.work_dir).expanduser().resolve() / 'jobs'
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def backend(self) -> str:
        return 'local'

    def name(self, job_id: str, relative: str) -> str:
        return (Path(job_id) / _safe_relative(relative)).as_posix()

    def path(self, job_id: str, relative: str) -> Path:
        if not job_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in job_id):
            raise ValueError('Invalid job ID')
        destination = self.root / job_id / _safe_relative(relative)
        try:
            destination.resolve().relative_to((self.root / job_id).resolve())
        except ValueError as exc:
            raise ValueError('Local artifact path escapes the job directory') from exc
        return destination

    def upload(
        self,
        job_id: str,
        relative: str,
        path: Path,
        *,
        if_generation_match: int | None = None,
    ) -> str:
        source = Path(path).expanduser().resolve()
        if not source.is_file() or source.stat().st_size <= 0:
            raise ValueError(f'Local artifact source is missing or empty: {source}')
        destination = self.path(job_id, relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if if_generation_match == 0 and destination.exists() and destination.resolve() != source:
            raise FileExistsError(str(destination))
        if destination.resolve() != source:
            temporary = destination.with_name(f'.{destination.name}.{os.getpid()}.upload')
            shutil.copy2(source, temporary)
            os.replace(temporary, destination)
        return str(destination.resolve())

    def upload_json(
        self,
        job_id: str,
        relative: str,
        value: dict,
        *,
        if_generation_match: int | None = None,
    ) -> str:
        destination = self.path(job_id, relative)
        if if_generation_match == 0 and destination.exists():
            raise FileExistsError(str(destination))
        if if_generation_match not in {None, 0} and destination.exists():
            current = int(destination.stat().st_mtime_ns)
            if current != int(if_generation_match):
                raise RuntimeError('Local artifact generation changed concurrently')
        atomic_write_json(destination, redact(value))
        return str(destination.resolve())

    def download(self, job_id: str, relative: str, path: Path) -> None:
        source = self.path(job_id, relative)
        if not source.is_file():
            raise FileNotFoundError(str(source))
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() == destination.resolve():
            return
        shutil.copy2(source, destination)

    def download_json(self, job_id: str, relative: str) -> tuple[dict, int]:
        source = self.path(job_id, relative)
        if not source.is_file():
            raise FileNotFoundError(str(source))
        return read_json(source), int(source.stat().st_mtime_ns)

    def update_json(self, job_id: str, relative: str, value: dict, generation: int) -> str:
        return self.upload_json(
            job_id,
            relative,
            value,
            if_generation_match=int(generation),
        )

    def promote(
        self,
        job_id: str,
        temporary_relative: str,
        final_relative: str,
        expected_sha256: str,
        checksum_fn,
        *,
        final_generation: int = 0,
    ) -> str:
        temporary = self.path(job_id, temporary_relative)
        final = self.path(job_id, final_relative)
        if not temporary.is_file() or temporary.stat().st_size <= 0:
            raise ValueError('Temporary output is missing or empty')
        if checksum_fn(temporary) != expected_sha256:
            raise ValueError('Uploaded temporary output checksum mismatch')
        final.parent.mkdir(parents=True, exist_ok=True)
        if final.exists():
            if final.stat().st_size <= 0 or checksum_fn(final) != expected_sha256:
                raise ValueError('A conflicting final output already exists')
            temporary.unlink(missing_ok=True)
            return str(final.resolve())
        os.replace(temporary, final)
        if final.stat().st_size <= 0 or checksum_fn(final) != expected_sha256:
            raise ValueError('Promoted final output checksum mismatch')
        return str(final.resolve())

    # The methods below preserve the GCSStore worker/lease interface for local
    # single-machine operation.  Colab commands still instantiate GCSStore.
    def claim(
        self,
        job_id: str,
        task_type: str,
        worker_id: str,
        attempt: int,
        lease_seconds: int = 900,
    ) -> dict:
        now = datetime.now(timezone.utc)
        relative = f'worker/{task_type}_claim.json'
        path = self.path(job_id, relative)
        if path.exists():
            current = read_json(path)
            expiry = datetime.fromisoformat(current['lease_expiry'])
            if expiry > now and current.get('status') not in {'completed', 'failed'}:
                raise LeaseConflict(f"Unexpired {task_type} lease held by {current.get('worker_id')}")
        claim = {
            'job_id': job_id,
            'task_type': task_type,
            'worker_id': worker_id,
            'claim_timestamp': now.isoformat(),
            'heartbeat_timestamp': now.isoformat(),
            'lease_expiry': (now + timedelta(seconds=lease_seconds)).isoformat(),
            'attempt_number': attempt,
        }
        atomic_write_json(path, claim)
        claim['generation'] = int(path.stat().st_mtime_ns)
        return claim

    def heartbeat(self, claim: dict, lease_seconds: int = 900) -> dict:
        current = self.assert_claim(claim)
        now = datetime.now(timezone.utc)
        updated = {k: v for k, v in current.items() if k != 'generation'}
        updated['heartbeat_timestamp'] = now.isoformat()
        updated['lease_expiry'] = (now + timedelta(seconds=lease_seconds)).isoformat()
        path = self.path(claim['job_id'], f"worker/{claim['task_type']}_claim.json")
        atomic_write_json(path, updated)
        updated['generation'] = int(path.stat().st_mtime_ns)
        return updated

    def assert_claim(self, claim: dict) -> dict:
        path = self.path(claim['job_id'], f"worker/{claim['task_type']}_claim.json")
        if not path.is_file():
            raise LeaseLost('Worker lease object no longer exists')
        current = read_json(path)
        for field in ('job_id', 'task_type', 'worker_id', 'attempt_number'):
            if current.get(field) != claim.get(field):
                raise LeaseLost(f'Worker lease {field} changed')
        if datetime.fromisoformat(current['lease_expiry']) <= datetime.now(timezone.utc):
            raise LeaseLost('Worker lease expired')
        if current.get('status') in {'completed', 'failed'}:
            raise LeaseLost('Worker lease is already final')
        return {**current, 'generation': int(path.stat().st_mtime_ns)}

    def finish_claim(self, claim: dict, status: str, error: str | None = None) -> dict:
        if status not in {'completed', 'failed'}:
            raise ValueError('Lease status must be completed or failed')
        current = self.assert_claim(claim)
        now = datetime.now(timezone.utc)
        updated = {k: v for k, v in current.items() if k != 'generation'}
        updated.update(
            status=status,
            completed_timestamp=now.isoformat(),
            lease_expiry=now.isoformat(),
        )
        if error:
            updated['error'] = str(error)[:500]
        path = self.path(claim['job_id'], f"worker/{claim['task_type']}_claim.json")
        atomic_write_json(path, updated)
        updated['generation'] = int(path.stat().st_mtime_ns)
        return updated


__all__ = ['LocalStore']
