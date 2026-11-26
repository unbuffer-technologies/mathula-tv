"""The MMS forced aligner, loaded once for this process, for finding where a name the speaker says falls in the SOURCE audio.

The recogniser's word times are rough and its spelling of a name is a guess ("Bronkhorstspruit" was heard as "Bronco Spirit" at 72.92-73.38 s; the name runs to 73.60 s, and the primer cut from the
recogniser's edges lost its last 0.2 s).  The name's own spelling is known, so the aligner can say where it sounds: the same model the voice service uses on the voice's own audio (word_align).
Where torch, torchaudio or the weights are not here, there is no aligner and the recogniser's edges are used as before.
"""
from __future__ import annotations

from typing import Any, Callable

_state: list[Any] = [None, False]          # [aligner or None, tried]


def shared_aligner() -> Callable[..., Any] | None:
    """(text, mono float array, rate) -> [{"start", "end", "score", "letters"} or None per word], or None when it cannot be loaded.  Loaded on first use, on the GPU when there is one."""
    if not _state[1]:
        _state[1] = True
        try:
            import torch

            from . import word_align

            _state[0] = word_align.load("cuda" if torch.cuda.is_available() else "cpu")
        except Exception:        # noqa: BLE001 - an optimisation of the primer's edges: never a reason a run cannot go on
            _state[0] = None
    return _state[0]
