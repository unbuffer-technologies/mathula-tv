"""Loanwords the agent created, shared by every job.

When no everyday Zulu word exists for an English noun, the agent creates one (``loan_adapt``) and writes it in the Zulu text.  This file remembers the
form chosen, so the next clip that says the same English word gets the same Zulu word instead of a new invention.

The file is ``<work_dir>/experience/created_loanwords.json``: {"entries": {english word, lowercase: {"english", "zulu", "jobs", "first_seen", "note"}}}.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = "mathula-created-loanwords-v1"


def path_for(work_dir: Path) -> Path:
    return Path(work_dir) / "experience" / "created_loanwords.json"


def _key(english: str) -> str:
    return re.sub(r"[^a-z]", "", english.casefold())


def load(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        return {"schema": SCHEMA, "entries": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("entries", {})
    return data


def remember(path: Path, english: str, zulu: str, *, job_id: str, note: str = "") -> dict[str, Any]:
    """Record the Zulu word created for an English word; the same word again counts one more job, a different one replaces it."""
    data = load(path)
    key = _key(english)
    entry = data["entries"].get(key)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if entry and entry["zulu"] == zulu:
        entry["jobs"] = sorted({*entry["jobs"], job_id})
    else:
        entry = {"english": english, "zulu": zulu, "jobs": [job_id], "first_seen": now, "note": note}
    data["entries"][key] = entry
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(data, indent=1, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    temporary.replace(path)
    return entry


def zulu_for(path: Path, english: str) -> str | None:
    entry = load(path)["entries"].get(_key(english))
    return str(entry["zulu"]) if entry else None
