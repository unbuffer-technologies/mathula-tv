"""Experience store: frozen clip corpus, run directories, episode index, human labels.

Layout under ``<work_dir>/experience``:

    corpus.json                      clip manifest
    corpus/<clip>/native_dub/        frozen replay inputs (Pass-1, ledger, glossary, ...)
    corpus/<clip>/analysis/          raw word index (optional)
    corpus/<clip>/baseline/native_dub/  inputs + outputs as they stood at freeze time
    runs/<run>/run.json              version stamp + parameters
    runs/<run>/<clip>/r<k>/          one replay (native_dub/ + scorecard.json)
    labels/<clip>.jsonl              human review labels
    episodes.jsonl                   one headline line per scored (run, clip, repeat)
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .atomic_io import atomic_write_json, read_json

def source_corrections_sha(native_dir: Path) -> str | None:
    """Checksum of a job's source corrections, so a clip frozen before they existed (or with others) can be refrozen."""
    path = Path(native_dir) / "source_corrections.json"
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


CORPUS_SCHEMA = "mathula-experience-corpus-v1"
RUN_SCHEMA = "mathula-experience-run-v1"

REQUIRED_INPUTS = (
    "pass1_restored_transcript.json",
    "context_ledger.json",
    "zulu_terminology_glossary.json",
)
OPTIONAL_INPUTS = (
    "speaker_seriousness_modes.json",
    "pronunciation_research.json",
    "pronunciation_names.json",
    "source_corrections.json",
    "manual_web_overrides.json",
)
BASELINE_OUTPUTS = (
    "pass2_candidate_pool.json",
    "pass2_natural_translation.json",
    "timing_plan.json",
)
BASELINE_RUN_ID = "baseline"


def text_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def version_stamp(repo_root: Path | None = None) -> dict[str, Any]:
    """Git SHA, dirty flag, and every prompt/schema version constant in native_dub."""
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]

    def _git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args], cwd=root, capture_output=True, text=True, timeout=15, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    from . import native_dub

    versions = {
        name: value
        for name, value in vars(native_dub).items()
        if re.search(r"(_PROMPT_VERSION|_SCHEMA_VERSION)$", name) and isinstance(value, str)
    }
    return {
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "versions": dict(sorted(versions.items())),
    }


class ExperienceStore:
    def __init__(self, work_dir: Path):
        self.root = Path(work_dir) / "experience"

    @property
    def corpus_path(self) -> Path:
        return self.root / "corpus.json"

    @property
    def episodes_path(self) -> Path:
        return self.root / "episodes.jsonl"

    def clip_dir(self, clip_id: str) -> Path:
        return self.root / "corpus" / clip_id

    def baseline_native_dub(self, clip_id: str) -> Path:
        return self.clip_dir(clip_id) / "baseline" / "native_dub"

    def run_dir(self, run_id: str) -> Path:
        return self.root / "runs" / run_id

    def repeat_dir(self, run_id: str, clip_id: str, repeat: int) -> Path:
        return self.run_dir(run_id) / clip_id / f"r{repeat}"

    def labels_path(self, clip_id: str) -> Path:
        return self.root / "labels" / f"{clip_id}.jsonl"

    def review_dir(self) -> Path:
        return self.root / "review"

    def load_corpus(self) -> dict[str, Any]:
        if self.corpus_path.is_file():
            return read_json(self.corpus_path)
        return {"schema_version": CORPUS_SCHEMA, "clips": {}}

    def clip_ids(self) -> list[str]:
        return sorted(self.load_corpus()["clips"])

    def freeze_clip(
        self, job_root: Path, *, clip_id: str | None = None, notes: str = "", overwrite: bool = False,
        require_baseline: bool = True,
    ) -> dict[str, Any]:
        """Copy a finished job's replay inputs and current outputs into the corpus.

        Inputs are what a replay needs to re-run the candidate-pool stage with different
        code; the baseline snapshot is the job's committed state, scored as run
        ``baseline`` so every later run has something real to be compared against.

        ``require_baseline=False`` freezes a *new* clip that has only been through Pass 1 (production use): there is no
        committed translation to compare against, only the inputs the agent needs.
        """
        job_root = Path(job_root)
        native = job_root / "native_dub"
        missing = [name for name in REQUIRED_INPUTS if not (native / name).is_file()]
        if missing:
            raise FileNotFoundError(f"{job_root.name} is missing required replay inputs: {missing}")
        if require_baseline and not (native / "pass2_candidate_pool.json").is_file():
            raise FileNotFoundError(f"{job_root.name} has no pass2_candidate_pool.json to baseline against")

        clip_id = clip_id or job_root.name[:8]
        corpus = self.load_corpus()
        if clip_id in corpus["clips"] and not overwrite:
            raise FileExistsError(f"clip {clip_id!r} already frozen; pass overwrite=True to refresh it")

        clip_dir = self.clip_dir(clip_id)
        if clip_dir.exists():
            shutil.rmtree(clip_dir)
        frozen = clip_dir / "native_dub"
        baseline = self.baseline_native_dub(clip_id)
        frozen.mkdir(parents=True)
        baseline.mkdir(parents=True)
        for name in (*REQUIRED_INPUTS, *OPTIONAL_INPUTS):
            if (native / name).is_file():
                shutil.copy2(native / name, frozen / name)
                shutil.copy2(native / name, baseline / name)
        for name in BASELINE_OUTPUTS:
            if (native / name).is_file():
                shutil.copy2(native / name, baseline / name)
        raw = job_root / "analysis" / "transcript_en_raw.json"
        if raw.is_file():
            (clip_dir / "analysis").mkdir(parents=True, exist_ok=True)
            shutil.copy2(raw, clip_dir / "analysis" / raw.name)

        # Words the English transcript got wrong (see source_suspects) are corrected in the FROZEN copy only: the agent, the
        # gates, the meaning check and the renderer all read this copy, and the job's own Pass 1 keeps its checksum.
        from .source_suspects import apply_to_frozen

        apply_to_frozen(frozen / "pass1_restored_transcript.json", frozen / "source_corrections.json")
        segments = read_json(frozen / "pass1_restored_transcript.json").get("segments") or []
        entry = {
            "source_corrections_sha": source_corrections_sha(native),
            "job_id": job_root.name,
            "frozen_at": _now(),
            "duration_s": round(max((int(s.get("end_ms") or 0) for s in segments), default=0) / 1000.0, 1),
            "segment_count": len(segments),
            "has_render_timing": (baseline / "timing_plan.json").is_file(),
            "has_raw_word_index": raw.is_file(),
            "notes": notes,
        }
        corpus["clips"][clip_id] = entry
        corpus["schema_version"] = CORPUS_SCHEMA
        atomic_write_json(self.corpus_path, corpus)
        return {"clip_id": clip_id, **entry}

    def seed_replay_dir(self, clip_id: str, target_job_root: Path) -> Path:
        """Create ``target_job_root`` holding exactly the frozen inputs for ``clip_id``."""
        clip_dir = self.clip_dir(clip_id)
        if not (clip_dir / "native_dub").is_dir():
            raise FileNotFoundError(f"clip {clip_id!r} is not frozen")
        target = Path(target_job_root)
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(clip_dir / "native_dub", target / "native_dub")
        if (clip_dir / "analysis").is_dir():
            shutil.copytree(clip_dir / "analysis", target / "analysis")
        return target

    def write_run(self, run_id: str, *, parameters: dict[str, Any]) -> dict[str, Any]:
        record = {
            "schema_version": RUN_SCHEMA,
            "run_id": run_id,
            "created_at": _now(),
            "stamp": version_stamp(),
            "parameters": parameters,
        }
        atomic_write_json(self.run_dir(run_id) / "run.json", record)
        return record

    def read_run(self, run_id: str) -> dict[str, Any]:
        path = self.run_dir(run_id) / "run.json"
        return read_json(path) if path.is_file() else {}

    def append_episode(self, entry: dict[str, Any]) -> None:
        self.episodes_path.parent.mkdir(parents=True, exist_ok=True)
        with self.episodes_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")

    def read_episodes(self) -> list[dict[str, Any]]:
        if not self.episodes_path.is_file():
            return []
        return [
            json.loads(line)
            for line in self.episodes_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def append_labels(self, clip_id: str, labels: Iterable[dict[str, Any]]) -> int:
        path = self.labels_path(clip_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with path.open("a", encoding="utf-8") as handle:
            for label in labels:
                handle.write(json.dumps(label, ensure_ascii=False, sort_keys=True) + "\n")
                count += 1
        return count

    def load_labels(self, clip_id: str) -> list[dict[str, Any]]:
        path = self.labels_path(clip_id)
        if not path.is_file():
            return []
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
