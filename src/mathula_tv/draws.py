"""Which draw of a sentence the listener approved, kept as a number and not as audio.

The voice is exactly repeatable: the same text, reference clip, settings and seed give the same audio again.  A sentence it reads right in most draws and wrong in a few ("Phineas" 3 of 4,
"yoMnyango" in all but the one the render happened to use) is a lottery with one fixed ticket per sentence.  The listener hears draws of it in the console and commits the one that sounds
right; what is kept is a small whole number, the retake, which goes into the seed of that sentence (``retake`` 0 is the plain seed, so a sentence nobody touched is unchanged).

A retake belongs to what it was heard with: the exact text of the span, and the reference clip the voice cloned.  Change the wording or the clip and the draw is another one, so the mark
is ignored (it stays in the file, marked stale in the console) and the sentence is read again from retake 0.  Nothing here listens, scores or chooses.

The file is ``retakes.json`` in the job's corpus folder: {"marks": {span key: {retake, ref: [start_s, end_s], text, unit, note, at}}}.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

FILE = "retakes.json"
MAX_RETAKE = 9


def span_key(text: str) -> str:
    return hashlib.sha1(" ".join(str(text or "").split()).encode("utf-8")).hexdigest()[:12]


def _path(job_root: Path | str | None) -> Path | None:
    return Path(str(job_root)) / FILE if job_root else None


def load(job_root: Path | str | None) -> dict[str, Any]:
    path = _path(job_root)
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path is not None else {}
    except (OSError, ValueError):
        data = {}
    marks = data.get("marks") if isinstance(data, dict) else None
    return {"marks": dict(marks) if isinstance(marks, dict) else {}}


def _same_reference(mark: Mapping[str, Any], ref_start: float, ref_end: float) -> bool:
    ref = mark.get("ref") or [None, None]
    try:
        return abs(float(ref[0]) - float(ref_start)) < 0.011 and abs(float(ref[1]) - float(ref_end)) < 0.011
    except (TypeError, ValueError):
        return False


def retake_for(store: Mapping[str, Any], text: str, ref_start: float, ref_end: float) -> int:
    """The approved retake of this exact span read with this reference clip, else 0."""
    mark = (store.get("marks") or {}).get(span_key(text))
    if not isinstance(mark, Mapping) or not _same_reference(mark, ref_start, ref_end):
        return 0
    try:
        return max(0, min(MAX_RETAKE, int(mark.get("retake") or 0)))
    except (TypeError, ValueError):
        return 0


def is_stale(store: Mapping[str, Any], text: str, ref_start: float, ref_end: float) -> bool:
    """A mark exists for this text but was made with another reference clip: it no longer applies."""
    mark = (store.get("marks") or {}).get(span_key(text))
    return isinstance(mark, Mapping) and not _same_reference(mark, ref_start, ref_end)


def take_for(store: Mapping[str, Any], text: str, ref_start: float, ref_end: float) -> int | None:
    """The take (0, 1, 2: the three draws the voice makes of every sentence) the listener committed for this exact span and reference clip, or None when none was."""
    mark = (store.get("marks") or {}).get(span_key(text))
    if not isinstance(mark, Mapping) or mark.get("take") is None or not _same_reference(mark, ref_start, ref_end):
        return None
    try:
        return max(0, int(mark["take"]))
    except (TypeError, ValueError):
        return None


def commit(job_root: Path | str, text: str, ref_start: float, ref_end: float, retake: int, *, unit: str = "", note: str = "", take: int | None = None) -> dict[str, Any]:
    if not isinstance(retake, int) or not 0 <= retake <= MAX_RETAKE:
        raise ValueError(f"a retake is a whole number from 0 to {MAX_RETAKE}")
    path = _path(job_root)
    assert path is not None
    store = load(job_root)
    if take is not None and (not isinstance(take, int) or isinstance(take, bool) or not 0 <= take <= MAX_RETAKE):
        raise ValueError(f"a take is a whole number from 0 to {MAX_RETAKE}")
    mark = {"retake": retake, **({"take": take} if take is not None else {}), "ref": [round(float(ref_start), 2), round(float(ref_end), 2)], "text": " ".join(str(text).split())[:200], "unit": unit, "note": note[:300],
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    store["marks"][span_key(text)] = mark
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(store, indent=1, ensure_ascii=False), encoding="utf-8")
    return mark


def revoke(job_root: Path | str, text: str) -> bool:
    path = _path(job_root)
    store = load(job_root)
    if store["marks"].pop(span_key(text), None) is None:
        return False
    assert path is not None
    path.write_text(json.dumps(store, indent=1, ensure_ascii=False), encoding="utf-8")
    return True
