"""Keep the server disk from filling up with scratch audio.

Every agent run measures hundreds of candidate sentences with TTS and keeps the audio in ``<run>/measure/``. Eleven gigabytes
of it once filled the disk so completely that a render died writing its video ("No space left on device") after the agent had
finished the work. That audio is scratch: a measurement is recomputed from the text whenever it is needed, and nothing reads an
old run's copy. Everything that is a record (submission, scorecard, trace, judge, the render folder) is never touched here.

* ``clean_measure`` removes one run's ``measure`` folder, called when the run ends.
* ``ensure_free_space`` runs before a render: when the disk is low it removes the scratch audio of every OTHER run, and when that is
  not enough it stops with a message that says what is using the space, instead of a crashed ffmpeg.

Set ``MATHULA_DUB_KEEP_MEASURE=1`` to keep the scratch audio (for debugging a measurement).
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Callable

SCRATCH_NAME = "measure"
DEFAULT_MIN_FREE_GB = 4.0


def _is_run_scratch(path: Path, runs_root: Path) -> bool:
    """Only ``<runs_root>/<run>/<clip>/r<n>/measure`` qualifies: exact name, exact depth, inside the runs folder."""
    try:
        relative = path.resolve().relative_to(runs_root.resolve())
    except ValueError:
        return False
    parts = relative.parts
    return len(parts) == 4 and parts[3] == SCRATCH_NAME and parts[2].startswith("r") and path.is_dir() and not path.is_symlink()


def _size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def keep_measure() -> bool:
    return os.getenv("MATHULA_DUB_KEEP_MEASURE", "").strip().lower() in {"1", "true", "yes"}


def clean_measure(experience_root: Path, run_dir: Path | None = None, *, except_run_dir: Path | None = None) -> int:
    """Remove scratch audio. With ``run_dir``: that run's. Without: every run's, except ``except_run_dir``. Returns bytes freed."""
    runs_root = Path(experience_root) / "runs"
    if run_dir is not None:
        targets = [Path(run_dir) / SCRATCH_NAME]
    else:
        targets = list(runs_root.glob(f"*/*/r*/{SCRATCH_NAME}"))
    freed = 0
    keep = Path(except_run_dir).resolve() if except_run_dir is not None else None
    for target in targets:
        if not _is_run_scratch(target, runs_root):
            continue
        if keep is not None and target.resolve().parent == keep:
            continue
        freed += _size(target)
        shutil.rmtree(target, ignore_errors=True)
    return freed


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1024 ** 3


def ensure_free_space(experience_root: Path, *, need_gb: float | None = None, keep_run_dir: Path | None = None,
                      progress: Callable[[str], None] = print) -> float:
    """Free GB available, after removing other runs' scratch audio if the disk is below ``need_gb``. Raises if it is still too low."""
    need = float(need_gb if need_gb is not None else os.getenv("MATHULA_MIN_FREE_GB", DEFAULT_MIN_FREE_GB))
    root = Path(experience_root)
    root.mkdir(parents=True, exist_ok=True)
    free = free_gb(root)
    if free >= need:
        return free
    freed = clean_measure(root, except_run_dir=keep_run_dir) / 1024 ** 3
    free = free_gb(root)
    progress(f"[housekeeping] the disk had less than {need:.0f} GB free; removed {freed:.1f} GB of old scratch audio, {free:.1f} GB free now")
    if free < need:
        biggest = sorted(((_size(p), p) for p in root.parent.iterdir() if p.is_dir()), reverse=True)[:4]
        raise RuntimeError(f"only {free:.1f} GB free on the disk (need {need:.0f} GB) after removing scratch audio. Largest folders in "
                           f"{root.parent}: " + ", ".join(f"{p.name} {size / 1024 ** 3:.1f} GB" for size, p in biggest))
    return free
