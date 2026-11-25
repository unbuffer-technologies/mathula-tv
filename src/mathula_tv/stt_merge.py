"""Words with times + speaker turns -> the transcription JSON the rest of the pipeline already reads (Azure Fast Transcription's shape).

The recogniser (Whisper) gives words with start/end seconds and no speaker; the diarizer (pyannote) gives who spoke when and no words.  This joins them and writes the response in the shape
``azure_stt.normalize`` reads (``combinedPhrases``, ``durationMilliseconds`` and ``phrases`` of one speaker each, every phrase with its words and punctuation), so replacing the provider
changes nothing downstream.  The service (gcp/stt_service) carries a copy of this file; a test keeps the two identical.

* a word takes the speaker whose turn it overlaps most; a word in a gap takes the nearest turn within ``NEAR_S`` seconds, else the speaker before it;
* speakers are numbered 1, 2, ... by the time they first speak (Azure's convention; the labels are job-local);
* a phrase is a sentence of one speaker: it ends at a speaker change, at a word that closes a sentence, at a long silence, or at a hard length.
"""
from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

NEAR_S = 1.0               # a word in a silence takes the turn that is at most this far away
GAP_BREAK_S = 1.2          # a silence this long ends a phrase even without punctuation
MAX_PHRASE_WORDS = 45
MAX_PHRASE_S = 25.0
_SENTENCE_END = re.compile(r"[.?!][\"')\]]*$")
_ABBREVIATIONS = frozenset({"mr", "mrs", "ms", "dr", "prof", "adv", "hon", "rev", "sen", "gen", "col", "capt", "st", "jr", "sr", "vs", "etc", "inc", "ltd", "no", "eg", "ie"})          # a full stop after these does not end a sentence
_STRONG_END = re.compile(r"[?!][\"')\]]*$")
_OPEN = re.compile(r"^[\"'(\[]*[A-Z0-9]")


def speaker_numbers(turns: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Diarizer label -> 1, 2, ... in the order the speakers first speak."""
    order: dict[str, int] = {}
    for turn in sorted(turns, key=lambda t: float(t["start"])):
        order.setdefault(str(turn["speaker"]), len(order) + 1)
    return order


def assign_speakers(words: Sequence[Mapping[str, Any]], turns: Sequence[Mapping[str, Any]]) -> list[int | None]:
    """The speaker number of each word (None only when there are no turns at all)."""
    numbers = speaker_numbers(turns)
    ranked = sorted(turns, key=lambda t: float(t["start"]))
    result: list[int | None] = []
    previous: int | None = None
    for word in words:
        start, end = float(word["start"]), float(word["end"])
        best, best_overlap = None, 0.0
        for turn in ranked:
            if float(turn["start"]) >= end:
                break
            overlap = min(end, float(turn["end"])) - max(start, float(turn["start"]))
            if overlap > best_overlap:
                best, best_overlap = turn, overlap
        if best is None and ranked:        # a word in a silence: the nearest turn, when it is close
            middle = (start + end) / 2
            nearest = min(ranked, key=lambda t: 0.0 if float(t["start"]) <= middle <= float(t["end"]) else min(abs(middle - float(t["start"])), abs(middle - float(t["end"]))))
            distance = 0.0 if float(nearest["start"]) <= middle <= float(nearest["end"]) else min(abs(middle - float(nearest["start"])), abs(middle - float(nearest["end"])))
            if distance <= NEAR_S or previous is None:
                best = nearest
        speaker = numbers[str(best["speaker"])] if best is not None else previous
        result.append(speaker)
        previous = speaker if speaker is not None else previous
    return result


def build_phrases(words: Sequence[Mapping[str, Any]], speakers: Sequence[int | None]) -> list[dict[str, Any]]:
    """Group the words into phrases (sentences of one speaker), each as Azure's ``phrases`` entry."""
    phrases: list[dict[str, Any]] = []
    current: list[tuple[Mapping[str, Any], int | None]] = []

    def flush() -> None:
        if not current:
            return
        first, last = current[0][0], current[-1][0]
        start_ms, end_ms = int(round(float(first["start"]) * 1000)), int(round(float(last["end"]) * 1000))
        probabilities = [float(w["probability"]) for w, _ in current if w.get("probability") is not None]
        entry: dict[str, Any] = {
            "offsetMilliseconds": start_ms, "durationMilliseconds": max(1, end_ms - start_ms), "text": " ".join(str(w["text"]).strip() for w, _ in current),
            "words": [{"text": str(w["text"]).strip(), "offsetMilliseconds": int(round(float(w["start"]) * 1000)), "durationMilliseconds": max(1, int(round((float(w["end"]) - float(w["start"])) * 1000)))}
                      for w, _ in current],
            "locale": "", "confidence": round(sum(probabilities) / len(probabilities), 4) if probabilities else None}
        if current[0][1] is not None:
            entry["speaker"] = current[0][1]
        phrases.append(entry)
        current.clear()

    for index, (word, speaker) in enumerate(zip(words, speakers)):
        text = str(word["text"]).strip()
        if not text:
            continue
        if current:
            gap = float(word["start"]) - float(current[-1][0]["end"])
            before = str(current[-1][0]["text"]).strip()
            abbreviation = before.rstrip(".").lower().lstrip("\"'([") in _ABBREVIATIONS and before.endswith(".")
            ended = bool(_SENTENCE_END.search(before)) and not abbreviation and (bool(_OPEN.match(text)) or bool(_STRONG_END.search(before)))          # "Mr. Smith" does not end a sentence; "?" and "!" do
            if speaker != current[0][1] or ended or gap >= GAP_BREAK_S or len(current) >= MAX_PHRASE_WORDS or float(word["end"]) - float(current[0][0]["start"]) > MAX_PHRASE_S:
                flush()
        current.append((word, speaker))
    flush()
    return phrases


def to_azure_raw(words: Sequence[Mapping[str, Any]], turns: Sequence[Mapping[str, Any]], *, duration_s: float, locale: str) -> dict[str, Any]:
    """The whole response: ``words`` (text, start, end, probability), the diarizer's ``turns`` (speaker, start, end; may be empty), the audio length and the locale asked for."""
    speakers = assign_speakers(words, turns) if turns else [None] * len(words)
    phrases = build_phrases(words, speakers)
    for phrase in phrases:
        phrase["locale"] = locale
    return {"durationMilliseconds": int(round(duration_s * 1000)), "combinedPhrases": [{"text": " ".join(p["text"] for p in phrases).strip()}], "phrases": phrases}
