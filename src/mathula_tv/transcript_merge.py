from __future__ import annotations

import re
from typing import Any


_CHAIR_VOCATIVE_AT_START = re.compile(
    r"^\s*(?:well[,.]?\s+|yes[,.]?\s+|no[,.]?\s+|thank\s+you[,.]?\s+)?"
    r"chair(?:person|man|woman)?\b",
    re.IGNORECASE,
)
_CHAIR_VOCATIVE_AT_END = re.compile(
    r"\bchair(?:person|man|woman)?\s*[.!?]*\s*$",
    re.IGNORECASE,
)
_WITHIN_PHRASE_PAUSE_SECONDS = 2.5
_MAX_CONTEXTUAL_RESPONDENT_DISTANCE_SECONDS = 180.0
_BROADCAST_HANDOFF_MINIMUM_GAP_SECONDS = 1.5
_BROADCAST_HANDOFF_MAXIMUM_GAP_SECONDS = 15.0
_BROADCAST_HANDOFF_CONTEXT_SECONDS = 120.0
_BROADCAST_HANDOFF_CUE = re.compile(
    r"\b(?:let(?:'s| us)\s+(?:take you|cross|go)\s+(?:then\s+)?to|"
    r"take you\s+(?:then\s+)?to|joining us live|over to)\b.*"
    r"\b(?:anchor|reporter|correspondent|journalist)\b|"
    r"\b(?:anchor|reporter|correspondent|journalist)\b.*"
    r"\b(?:joining us live|live (?:there|from|in)|over to)\b",
    re.IGNORECASE | re.DOTALL,
)
_BROADCAST_REPLY_CUE = re.compile(
    r"^\s*(?:well|yes|thank you|thanks)[,.]?\b.{0,80}"
    r"\b(?:good (?:morning|afternoon|evening)|as you (?:say|said)|"
    r"to (?:you|the)\b)",
    re.IGNORECASE | re.DOTALL,
)
CONTEXTUAL_SPEAKER_TURN_REPAIR_VERSION = (
    "v13.18.31-broadcast-handoff-and-chair-vocative-turn-repair"
)


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
        raw_speaker = best["speaker"] if best else "UNKNOWN"
        assigned.append({**word, "speaker": raw_speaker, "raw_speaker": raw_speaker, "speaker_assignment_confidence": min(1.0, amount / duration) if amount else (0.25 if best else 0.0), "overlap": sum(1 for _, score in scored if score > 0) > 1, "assignment_warning": warning})
    return assigned


def _phrase_runs(words: list[dict]) -> list[list[dict]]:
    """Group consecutive Azure words by recognition phrase provenance."""

    runs: list[list[dict]] = []
    for index, word in enumerate(words):
        phrase_id = word.get("source_phrase_id")
        key = ("phrase", phrase_id) if phrase_id is not None else ("word", index)
        if runs:
            previous = runs[-1][0]
            previous_id = previous.get("source_phrase_id")
            previous_key = (
                ("phrase", previous_id)
                if previous_id is not None
                else ("word", index - 1)
            )
            if key == previous_key:
                runs[-1].append(word)
                continue
        runs.append([word])
    return runs


def _run_text(run: list[dict]) -> str:
    return " ".join(str(word.get("text") or "").strip() for word in run).strip()


def _run_speaker(run: list[dict]) -> str:
    counts: dict[str, int] = {}
    for word in run:
        speaker = str(word.get("speaker") or "UNKNOWN")
        counts[speaker] = counts.get(speaker, 0) + 1
    return max(counts, key=lambda speaker: (counts[speaker], speaker))


def _run_start(run: list[dict]) -> float:
    return float(run[0]["start"])


def _run_end(run: list[dict]) -> float:
    return float(run[-1]["end"])


def _is_short_chair_vocative(run: list[dict]) -> bool:
    text = _run_text(run)
    if len(run) > 14 or _run_end(run) - _run_start(run) > 6.0:
        return False
    return bool(
        _CHAIR_VOCATIVE_AT_START.search(text)
        or _CHAIR_VOCATIVE_AT_END.search(text)
    )


def _repair_chair_vocative_turns(words: list[dict]) -> list[dict[str, Any]]:
    """Repair high-confidence vocative replies that Azure assigns to the chair.

    Fast diarization occasionally absorbs a very short interjection into the
    dominant speaker on either side.  In commission and court footage this can
    make the chair appear to answer themself (for example, ``19 August,
    Chair``).  Repeated correctly diarized uses of the vocative establish both
    the chair and likely respondents.  Only short, whole Azure recognition
    phrases are eligible; the acoustic/raw speaker remains attached as audit
    provenance.
    """

    runs = _phrase_runs(words)
    if len(runs) < 3:
        return []

    speakers = [_run_speaker(run) for run in runs]
    chair_links: dict[str, int] = {}
    vocative_indexes = [
        index for index, run in enumerate(runs) if _is_short_chair_vocative(run)
    ]
    for index in vocative_indexes:
        current = speakers[index]
        for neighbour_index in (index - 1, index + 1):
            if not 0 <= neighbour_index < len(runs):
                continue
            neighbour = speakers[neighbour_index]
            if neighbour == current or neighbour == "UNKNOWN":
                continue
            if neighbour_index < index:
                gap = _run_start(runs[index]) - _run_end(runs[neighbour_index])
            else:
                gap = _run_start(runs[neighbour_index]) - _run_end(runs[index])
            if gap <= 5.0:
                chair_links[neighbour] = chair_links.get(neighbour, 0) + 1

    if not chair_links:
        return []
    chair_speaker, chair_evidence = max(
        chair_links.items(), key=lambda item: (item[1], item[0])
    )
    # One isolated "Chair" is not enough authority to rewrite diarization.
    if chair_evidence < 2:
        return []

    respondent_vocatives: dict[str, int] = {}
    for index in vocative_indexes:
        speaker = speakers[index]
        if speaker not in {chair_speaker, "UNKNOWN"}:
            respondent_vocatives[speaker] = respondent_vocatives.get(speaker, 0) + 1
    if not respondent_vocatives:
        return []

    speaker_centres: dict[str, list[float]] = {}
    for run, speaker in zip(runs, speakers):
        if speaker != chair_speaker:
            speaker_centres.setdefault(speaker, []).append(
                (_run_start(run) + _run_end(run)) / 2.0
            )

    repairs: list[dict[str, Any]] = []
    for index in vocative_indexes:
        run = runs[index]
        if speakers[index] != chair_speaker:
            continue
        centre = (_run_start(run) + _run_end(run)) / 2.0
        ranked: list[tuple[float, str, float]] = []
        for candidate, evidence in respondent_vocatives.items():
            distances = [
                abs(centre - candidate_centre)
                for candidate_centre in speaker_centres.get(candidate, [])
            ]
            if not distances:
                continue
            distance = min(distances)
            if distance > _MAX_CONTEXTUAL_RESPONDENT_DISTANCE_SECONDS:
                continue
            ranked.append((evidence * 30.0 - distance, candidate, distance))
        if not ranked:
            continue
        _score, replacement, distance = max(ranked)
        phrase_id = run[0].get("source_phrase_id")
        repair = {
            "repair_version": CONTEXTUAL_SPEAKER_TURN_REPAIR_VERSION,
            "source_phrase_id": phrase_id,
            "text": _run_text(run),
            "original_speaker": chair_speaker,
            "repaired_speaker": replacement,
            "reason": "chair_cannot_be_own_vocative_respondent",
            "chair_evidence_count": chair_evidence,
            "respondent_evidence_count": respondent_vocatives[replacement],
            "nearest_respondent_distance_seconds": round(distance, 3),
        }
        for word in run:
            word["speaker"] = replacement
            word["speaker_assignment_method"] = "contextual_vocative_repair"
            word["speaker_repair"] = repair
        speakers[index] = replacement
        repairs.append(repair)
    return repairs


def _repair_broadcast_handoff_turns(words: list[dict]) -> list[dict[str, Any]]:
    """Split an explicit live handoff that Azure merged into one speaker.

    Broadcast mixes can make a studio presenter and remote correspondent sound
    similar enough for Azure to assign both to one raw speaker.  The boundary
    remains explicit in the words: the presenter introduces a named news role,
    asks a question, a satellite/production pause follows, and the remote voice
    replies with a greeting.  Only that complete pattern is authoritative
    enough to create a new effective speaker label.
    """

    runs = _phrase_runs(words)
    if len(runs) < 2:
        return []
    speakers = [_run_speaker(run) for run in runs]
    repairs: list[dict[str, Any]] = []
    split_raw_speakers: set[str] = set()

    for boundary_index in range(1, len(runs)):
        previous = runs[boundary_index - 1]
        following = runs[boundary_index]
        original_speaker = speakers[boundary_index]
        if (
            not original_speaker
            or original_speaker == "UNKNOWN"
            or original_speaker in split_raw_speakers
            or speakers[boundary_index - 1] != original_speaker
        ):
            continue
        gap = _run_start(following) - _run_end(previous)
        if not (
            _BROADCAST_HANDOFF_MINIMUM_GAP_SECONDS
            <= gap
            <= _BROADCAST_HANDOFF_MAXIMUM_GAP_SECONDS
        ):
            continue

        context_runs: list[list[dict]] = []
        for candidate_index in range(boundary_index - 1, -1, -1):
            candidate = runs[candidate_index]
            if speakers[candidate_index] != original_speaker:
                break
            if (
                _run_end(previous) - _run_start(candidate)
                > _BROADCAST_HANDOFF_CONTEXT_SECONDS
            ):
                break
            context_runs.append(candidate)
        context_runs.reverse()
        introduction_text = " ".join(_run_text(run) for run in context_runs)
        reply_text = _run_text(following)
        if not _BROADCAST_HANDOFF_CUE.search(introduction_text):
            continue
        if not _BROADCAST_REPLY_CUE.search(reply_text):
            continue
        # A live handoff normally ends in a direct question.  Requiring that
        # punctuation prevents ordinary story narration from splitting a
        # speaker merely because it mentions a reporter.
        if "?" not in introduction_text[-500:]:
            continue

        repaired_speaker = f"{original_speaker}__CONTEXT_HANDOFF_01"
        affected_runs = [
            run
            for index, run in enumerate(runs)
            if index >= boundary_index and speakers[index] == original_speaker
        ]
        repair = {
            "repair_version": CONTEXTUAL_SPEAKER_TURN_REPAIR_VERSION,
            "source_phrase_id": following[0].get("source_phrase_id"),
            "text": reply_text,
            "original_speaker": original_speaker,
            "repaired_speaker": repaired_speaker,
            "reason": "explicit_broadcast_handoff_merged_by_diarization",
            "boundary_gap_seconds": round(gap, 3),
            "introduction_text": introduction_text,
            "affected_phrase_count": len(affected_runs),
        }
        for run in affected_runs:
            for word in run:
                word["speaker"] = repaired_speaker
                word["speaker_assignment_method"] = (
                    "contextual_broadcast_handoff_repair"
                )
                word["speaker_repair"] = repair
        for index in range(boundary_index, len(runs)):
            if speakers[index] == original_speaker:
                speakers[index] = repaired_speaker
        split_raw_speakers.add(original_speaker)
        repairs.append(repair)

    return repairs


def repair_contextual_speaker_turns(words: list[dict]) -> list[dict[str, Any]]:
    """Apply conservative transcript-authoritative speaker-turn repairs."""

    repairs = _repair_chair_vocative_turns(words)
    repairs.extend(_repair_broadcast_handoff_turns(words))
    return repairs


def reconcile(words: list[dict], diarization: list[dict], *, source: str = "azure", tolerance: float = 0.35, max_pause: float = 1.2) -> dict[str, Any]:
    # Canonical IDs are scoped to one transcript.  Retaining the function cache
    # across jobs can otherwise shift SPEAKER_XX IDs after a rebuild.
    if hasattr(_map_to_canonical_speaker_id, "_mapping"):
        del _map_to_canonical_speaker_id._mapping
    enriched = assign_words(words, diarization, tolerance)
    speaker_turn_repairs = repair_contextual_speaker_turns(enriched)
    segments: list[dict] = []
    warnings: list[str] = []
    current: list[dict] = []
    for word in enriched:
        same_source_phrase = bool(
            current
            and word.get("source_phrase_id") is not None
            and word.get("source_phrase_id") == current[-1].get("source_phrase_id")
        )
        allowed_pause = max_pause
        if same_source_phrase:
            allowed_pause = max(max_pause, _WITHIN_PHRASE_PAUSE_SECONDS)
        if current and (word["speaker"] != current[-1]["speaker"] or float(word["start"]) - float(current[-1]["end"]) > allowed_pause or word["overlap"] != current[-1]["overlap"]):
            segments.append(_segment(current, len(segments), source))
            current = []
        current.append(word)
        if word.get("assignment_warning"):
            warnings.append(f"{word.get('text', '')}: {word['assignment_warning']}")
    if current:
        segments.append(_segment(current, len(segments), source))
    return {"schema_version": "transcript-v1", "language": "en-ZA", "diarization_source": source, "segments": segments, "words": enriched, "speaker_turn_repairs": speaker_turn_repairs, "warnings": warnings}


def _segment(words: list[dict], index: int, source: str) -> dict:
    confidences = [float(w.get("confidence", 0)) for w in words if w.get("confidence") is not None]
    assignments = [float(w["speaker_assignment_confidence"]) for w in words]
    effective_speaker = str(words[0]["speaker"])
    raw_speakers = list(
        dict.fromkeys(str(word.get("raw_speaker") or word["speaker"]) for word in words)
    )
    raw_speaker = raw_speakers[0]
    
    # Map to canonical speaker ID (SPEAKER_00, SPEAKER_01, etc.)
    canonical_speaker = _map_to_canonical_speaker_id(effective_speaker, index)
    
    return {
        "segment_id": f"seg-{index + 1:05d}",
        "speaker": canonical_speaker,
        "speaker_id": canonical_speaker,
        "raw_speaker": raw_speaker,
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
        "speaker_assignment_method": (
            next(
                (
                    str(w["speaker_assignment_method"])
                    for w in words
                    if str(w.get("speaker_assignment_method") or "").startswith(
                        "contextual_"
                    )
                ),
                "diarization_overlap",
            )
        ),
        "speaker_repairs": list(
            {
                str(w["speaker_repair"]): w["speaker_repair"]
                for w in words
                if isinstance(w.get("speaker_repair"), dict)
            }.values()
        ),
    }


def _map_to_canonical_speaker_id(raw_speaker: str, segment_index: int) -> str:
    """Map raw diarization speaker ID to canonical speaker ID.
    
    This function creates a stable mapping from raw diarization IDs (like AZURE_1)
    to canonical speaker IDs (like SPEAKER_00) based on first occurrence order.
    
    The mapping is deterministic: the first unique raw speaker encountered
    becomes SPEAKER_00, the second becomes SPEAKER_01, etc.
    
    Args:
        raw_speaker: Raw speaker ID from diarization (e.g., "AZURE_1")
        segment_index: Index of the current segment (for ordering)
    
    Returns:
        Canonical speaker ID (e.g., "SPEAKER_00")
    """
    # Use a module-level cache to maintain mapping across calls
    if not hasattr(_map_to_canonical_speaker_id, "_mapping"):
        _map_to_canonical_speaker_id._mapping = {}
    
    mapping = _map_to_canonical_speaker_id._mapping
    
    if raw_speaker not in mapping:
        # Assign next available canonical ID
        canonical_id = f"SPEAKER_{len(mapping):02d}"
        mapping[raw_speaker] = canonical_id
    
    return mapping[raw_speaker]
