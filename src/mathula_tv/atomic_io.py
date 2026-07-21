from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable


def _atomic_replace(path: Path, mode: str, writer: Callable[[Any], None], *, fsync_dir: bool = False) -> None:
    """Write ``path`` atomically by staging to a temp file and ``os.replace``.

    ``writer`` receives the open file handle and is responsible for writing the
    payload. The handle is flushed and fsynced before the atomic rename. When
    ``fsync_dir`` is set the parent directory is fsynced afterwards so the rename
    itself is durable.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, mode, encoding=None if "b" in mode else "utf-8", newline=None if "b" in mode else "\n") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if fsync_dir:
            _fsync_directory(path.parent)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _fsync_directory(directory: Path) -> None:
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_json(path: Path, data: Any) -> None:
    def write(handle: Any) -> None:
        json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    _atomic_replace(path, "w", write)


def atomic_write_text(path: Path, value: str) -> None:
    _atomic_replace(path, "w", lambda handle: handle.write(value))


def atomic_write_bytes(path: Path, value: bytes, *, fsync_dir: bool = False) -> None:
    _atomic_replace(path, "wb", lambda handle: handle.write(value), fsync_dir=fsync_dir)


def atomic_copy(source: Path, destination: Path) -> None:
    def write(handle: Any) -> None:
        with Path(source).open("rb") as input_file:
            shutil.copyfileobj(input_file, handle)

    _atomic_replace(destination, "wb", write)


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)
