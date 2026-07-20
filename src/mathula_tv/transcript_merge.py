from __future__ import annotations

from typing import Any


def _overlap(start: float, end: float, turn: dict) -> float:
    return max(0.0, min(end, float(turn["end"])) - max(start, float(turn["start"])))


def assign_words(words: list[dict], diarization: list[dict], tolerance: float = 0.35) -> list[dict]:
    assigned = []
    for word in words:
        start, end = float(word["start"]), float(word["end"])
        scored = [(turn, _overlap(start, end, turn)) for turn in diarization]
        best, amount = max(scored, key=lambda item: item[1], default=(None, 0.0))
        warning = None
        if amount <= 0:
            midpoint = (start + end) / 2
            near = [(turn, min(abs(midpoint - float(turn["start"])), abs(midpoint - float(turn["end"])))) for turn in diarization]
            best, distance = min(near, key=lambda item: item[1], default=(None, float("inf")))
            if distance > tolerance:
                best = None
                warning = "no speaker within tolerance"
        duration = max(end - start, 0.001)
        assigned.append({**word, "speaker": best["speaker"] if best else "UNKNOWN", "speaker_assignment_confidence": min(1.0, amount / duration) if amount else (0.25 if best else 0.0), "overlap": sum(1 for _, score in scored if score > 0) > 1, "assignment_warning": warning})
    return assigned


def reconcile(words: list[dict], diarization: list[dict], *, source: str = "azure", tolerance: float = 0.35, max_pause: float = 1.2) -> dict[str, Any]:
    enriched = assign_words(words, diarization, tolerance)
    segments: list[dict] = []
    warnings: list[str] = []
    current: list[dict] = []
    for word in enriched:
        if current and (word["speaker"] != current[-1]["speaker"] or float(word["start"]) - float(current[-1]["end"]) > max_pause or word["overlap"] != current[-1]["overlap"]):
            segments.append(_segment(current, len(segments), source))
            current = []
        current.append(word)
        if word.get("assignment_warning"):
            warnings.append(f"{word.get('text', '')}: {word['assignment_warning']}")
    if current:
        segments.append(_segment(current, len(segments), source))
    return {"schema_version": "transcript-v1", "language": "en-ZA", "diarization_source": source, "segments": segments, "words": enriched, "warnings": warnings}


def _segment(words: list[dict], index: int, source: str) -> dict:
    confidences = [float(w.get("confidence", 0)) for w in words if w.get("confidence") is not None]
    assignments = [float(w["speaker_assignment_confidence"]) for w in words]
    return {
        "segment_id": f"seg-{index + 1:05d}",
        "speaker": words[0]["speaker"],
        "start": float(words[0]["start"]),
        "end": float(words[-1]["end"]),
        "source_text": " ".join(str(w.get("text", "")).strip() for w in words).strip(),
        "source_word_ids": [w["word_id"] for w in words if w.get("word_id")],
        "source_phrase_ids": list(dict.fromkeys(w["source_phrase_id"] for w in words if w.get("source_phrase_id"))),
        "recognition_confidence": sum(confidences) / len(confidences) if confidences else None,
        "speaker_assignment_confidence": sum(assignments) / len(assignments),
        "diarization_source": source,
        "overlap": any(w["overlap"] for w in words),
        "warnings": [w["assignment_warning"] for w in words if w.get("assignment_warning")],
    }
