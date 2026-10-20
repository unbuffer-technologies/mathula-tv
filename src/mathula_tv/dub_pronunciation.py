"""The clip's pronunciation dictionary, so measurement hears what the renderer will say.

The renderer registers each job's web-researched pronunciation corrections (names, acronyms, organisations) and
applies them to the text it hands Azure. Those substitutions can lengthen or shorten the audio (a respelled name,
a spelled-out acronym), so a measurement made without them can disagree with the render by several percent --
enough to turn a unit that "fits" into one that is quietly sped up. This module registers the same dictionary for
measurement.

The research file is copied into the frozen corpus the first time it is needed (like the clip's voices), and the
dictionary is registered under the clip's corpus directory, a key that is stable and unique per clip, so clips
measured side by side in one process never share a dictionary.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .atomic_io import read_json

_REGISTERED: set[str] = set()


def register_clip_pronunciation(store: Any, clip_id: str) -> Path | None:
    """Register the clip's pronunciation dictionary; returns the key (``job_root``) to measure with, or None.

    None means the clip has no research file, and measurement then uses the default dictionary -- exactly what the
    renderer does for a job without one.
    """
    try:
        key = Path(store.clip_dir(clip_id))
        frozen = key / "pronunciation_research.json"
        if not frozen.is_file():
            entry = store.load_corpus()["clips"].get(clip_id) or {}
            source = (Path(store.root).parent / "jobs" / str(entry.get("job_id")) / "native_dub"
                      / "pronunciation_research.json")
            if not source.is_file():
                return None
            frozen.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, frozen)
        if str(key) not in _REGISTERED:
            from .native_dub import _apply_native_pronunciation_research_to_job_cache

            _apply_native_pronunciation_research_to_job_cache(key, read_json(frozen))
            _REGISTERED.add(str(key))
        return key
    except (AttributeError, OSError, ValueError):
        return None
