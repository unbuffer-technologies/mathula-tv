from __future__ import annotations

import uuid
from pathlib import Path

from .atomic_io import atomic_write_json, read_json
from .media import checksum
from .models import JobManifest
from .models import utcnow, validate_job_id


class JobStore:
    def __init__(self, work_dir: Path):
        self.root = work_dir / "jobs"
        self.root.mkdir(parents=True, exist_ok=True)

    def job_dir(self, job_id: str) -> Path:
        return self.root / validate_job_id(job_id)

    def path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "manifest.json"

    def create(self, **kwargs) -> JobManifest:
        job = JobManifest(job_id=kwargs.pop("job_id", uuid.uuid4().hex), **kwargs)
        if self.path(job.job_id).exists():
            raise FileExistsError(job.job_id)
        self.save(job)
        return job

    def save(self, job: JobManifest) -> None:
        atomic_write_json(self.path(job.job_id), job.to_dict())

    def load(self, job_id: str) -> JobManifest:
        return JobManifest.from_dict(read_json(self.path(job_id)))

    def list(self) -> list[JobManifest]:
        return [JobManifest.from_dict(read_json(path)) for path in sorted(self.root.glob("*/manifest.json"))]

    def complete_stage(self, job: JobManifest, stage: str, target_state: str, *, force: bool = False) -> bool:
        if stage in job.completed_stages and not force:
            return False
        if stage not in job.completed_stages:
            job.completed_stages.append(stage)
        job.transition(target_state)
        self.save(job)
        return True

    def begin_attempt(self, job: JobManifest, stage: str, *, forced: bool = False) -> int:
        """Record an immutable attempt entry before external or costly work."""
        attempt = job.attempt_counters.get(stage, 0) + 1
        job.attempt_counters[stage] = attempt
        job.attempt_history.append(
            {
                "stage": stage,
                "attempt": attempt,
                "forced": bool(forced),
                "started_at": utcnow(),
                "status": "running",
            }
        )
        self.save(job)
        return attempt

    def finish_attempt(self, job: JobManifest, stage: str, attempt: int, status: str) -> None:
        if status not in {"completed", "failed", "partial", "lease_lost"}:
            raise ValueError("Invalid attempt status")
        for item in reversed(job.attempt_history):
            if item.get("stage") == stage and item.get("attempt") == attempt:
                if item.get("status") != "running":
                    raise ValueError("Attempt is already final")
                item.update(status=status, completed_at=utcnow())
                self.save(job)
                return
        raise ValueError("Attempt history entry is missing")

    def migrate_synthesis_queued(
        self,
        job: JobManifest,
        *,
        dry_run: bool = False,
        allow_legacy_review_reset: bool = False,
    ) -> dict:
        """Migrate the one supported legacy state without changing translations."""
        if job.state == "azure_tts_queued":
            return {"changed": False, "state": job.state, "translation_hashes": self._translation_hashes(job)}
        legacy_review_reset = job.state == "review_ready"
        if legacy_review_reset:
            if not allow_legacy_review_reset:
                raise ValueError(
                    "Legacy review_ready reset requires --allow-legacy-review-reset"
                )
            if (
                job.providers.get("synthesis") != "k2-fsa/OmniVoice"
                or "omnivoice_synthesis" not in job.completed_stages
            ):
                raise ValueError(
                    "Only a verified legacy OmniVoice review_ready job can be reset"
                )
        elif job.state != "synthesis_queued":
            raise ValueError(f"Only legacy synthesis_queued can migrate, not {job.state}")
        hashes = self._translation_hashes(job)
        result = {
            "changed": True,
            "from_state": job.state,
            "state": "azure_tts_queued",
            "translation_hashes": hashes,
            "translations_preserved": True,
            "dry_run": dry_run,
            "legacy_review_reset": legacy_review_reset,
        }
        if dry_run:
            return result
        if legacy_review_reset:
            job.state = "azure_tts_queued"
            job.updated_at = utcnow()
        else:
            job.transition("azure_tts_queued")
        job.schema_version = "job-manifest-v2"
        job.compatibility_gate = {"state": "pending", "calibrated_thresholds": False}
        job.review_readiness = "compatibility_pending"
        job.attempt_history.append(
            {
                "stage": "dubbing_state_migration",
                "attempt": 1,
                "status": "completed",
                "from_state": result["from_state"],
                "to_state": "azure_tts_queued",
                "translation_hashes": hashes,
                "completed_at": utcnow(),
            }
        )
        self.save(job)
        if hashes != self._translation_hashes(job):
            raise RuntimeError("Translation artifacts changed during state migration")
        return result

    def _translation_hashes(self, job: JobManifest) -> dict[str, str]:
        root = self.job_dir(job.job_id)
        paths = {
            "translation": root / "translation/transcript_zu.json",
            "seo": root / "translation/youtube_zu.json",
        }
        missing = [name for name, path in paths.items() if not path.is_file() or not path.stat().st_size]
        if missing:
            raise ValueError("Legacy migration requires validated translation artifacts: " + ", ".join(missing))
        return {name: checksum(path) for name, path in paths.items()}

    def recover_failed_stage(self, job: JobManifest, stage: str, target_state: str) -> None:
        allowed = {"translation": "analysis_ready", "seo_packaging": "translation_ready"}
        if job.state not in {"failed_retryable", "failed_terminal"} or (job.last_error or {}).get("stage") != stage:
            raise ValueError("Job failure does not match the requested recovery stage")
        if allowed.get(stage) != target_state:
            raise ValueError(f"Invalid {stage} recovery target")
        job.state, job.updated_at, job.last_error = target_state, utcnow(), None
        self.save(job)
