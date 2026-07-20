from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any


def validate_turns(turns: list[dict]) -> None:
    if not turns:
        raise ValueError("Diarization returned no turns")
    for turn in turns:
        if float(turn["end"]) <= float(turn["start"]) or not turn.get("speaker"):
            raise ValueError("Invalid diarization turn")


def select_authoritative(pyannote: dict | None, azure: dict | None, pyannote_error: str | None = None) -> tuple[list[dict], str, list[str]]:
    """Return Azure labels and report Pyannote only as a diagnostic signal."""
    warnings = []
    try:
        turns = (azure or {}).get("turns", [])
        validate_turns(turns)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Authoritative Azure diarization is unavailable: {exc}") from exc

    if pyannote_error:
        warnings.append(f"Pyannote diagnostic unavailable: {pyannote_error}")
    elif pyannote is not None:
        try:
            diagnostic_turns = pyannote.get("turns", [])
            validate_turns(diagnostic_turns)
            azure_speakers = len({item["speaker"] for item in turns})
            pyannote_speakers = len({item["speaker"] for item in diagnostic_turns})
            if azure_speakers != pyannote_speakers:
                warnings.append(
                    f"Pyannote diagnostic disagreement: Azure found {azure_speakers} speakers; "
                    f"Pyannote found {pyannote_speakers}"
                )
        except (ValueError, TypeError) as exc:
            warnings.append(f"Pyannote diagnostic invalid: {exc}")
    return turns, "azure", warnings


def run_pyannote(audio: Path, model_id: str, token: str | None = None) -> dict[str, Any]:
    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise RuntimeError("Install the diarization extra to run Pyannote") from exc
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logging.warning("Pyannote is running on CPU; diarization may be slow")
    pipeline = Pipeline.from_pretrained(model_id, token=token)
    pipeline.to(torch.device(device))
    started = time.perf_counter()
    output = pipeline(str(audio))
    elapsed = time.perf_counter() - started
    turns = [{"speaker": speaker, "start": float(turn.start), "end": float(turn.end), "duration": float(turn.end - turn.start), "overlap": False, "confidence": None, "model_identifier": model_id, "device": device} for turn, _, speaker in output.itertracks(yield_label=True)]
    validate_turns(turns)
    audio_duration = max(t["end"] for t in turns)
    return {"schema_version": "diarization-v1", "engine": "pyannote", "model_identifier": model_id, "model_revision": None, "device": device, "execution_duration": elapsed, "audio_duration": audio_duration, "real_time_factor": elapsed / audio_duration, "turns": turns}
