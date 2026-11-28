"""The words a speaker likes to pad their sentences with, read from the English source, and what a unit needs to reach the edge of its window.

A dub unit that comes out short of its window leaves dead air.  The Zulu text can be lengthened with anything, but the speaker's own habits are the right padding: a speaker who says
"you know" and "honestly" all through a talk sounds like themselves when the Zulu carries the same kind of word in the same places, and does not when the agent invents filler (the
anchor's "ntambama").  This module finds those words in the speaker's English; it picks nothing for the agent, which finds the Zulu form itself (human_translations, attested_zulu).
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# Discourse and stance markers an English speaker pads with: not content words, and not the function words ("so", "now", "also", "just") every speaker uses.
DISCOURSE_MARKERS: tuple[str, ...] = (
    "you know", "i mean", "you see", "honestly", "to be honest", "frankly", "obviously", "clearly", "of course", "in fact", "actually", "really", "indeed", "certainly", "definitely",
    "absolutely", "truly", "literally", "basically", "essentially", "fortunately enough", "fortunately", "unfortunately", "at the end of the day", "as i said", "i think", "i believe",
    "i would say", "let me say", "i can tell you", "remember", "kind of", "sort of", "very", "quite", "particularly", "especially", "exactly", "well", "look", "right",
)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def fillers_by_speaker(segments: Iterable[Mapping[str, Any]], *, min_count: int = 2, top: int = 6) -> dict[str, list[dict[str, Any]]]:
    """For each speaker, the discourse markers they use most in the English source, with how many times and one sentence they said it in.  A marker is listed when the speaker says it at least
    ``min_count`` times (once when they speak only a few sentences).  The English is the restored transcript of each segment."""
    texts: dict[str, list[str]] = {}
    for segment in segments:
        text = str(segment.get("restored_text") or segment.get("source_text") or "").strip()
        if text:
            texts.setdefault(str(segment.get("speaker_id") or "SPEAKER_00"), []).append(text)
    found: dict[str, list[dict[str, Any]]] = {}
    for speaker, pieces in texts.items():
        joined = " ".join(pieces)
        sentences = _sentences(joined)
        floor = min_count if len(sentences) >= 6 else 1
        counts: Counter[str] = Counter()
        example: dict[str, str] = {}
        for marker in DISCOURSE_MARKERS:
            pattern = re.compile(rf"(?<![A-Za-z']){re.escape(marker)}(?![A-Za-z'])", re.IGNORECASE)
            hits = pattern.findall(joined)
            if len(hits) >= floor:
                counts[marker] = len(hits)
                example[marker] = next((s for s in sentences if pattern.search(s)), "")
        # "fortunately enough" is also a "fortunately": count the longer phrase, not both
        for longer, shorter in (("fortunately enough", "fortunately"), ("to be honest", "honestly")):
            if longer in counts and shorter in counts:
                counts[shorter] -= counts[longer]
                if counts[shorter] < floor:
                    del counts[shorter]
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top]
        found[speaker] = [{"phrase": phrase, "count": count, "example": example[phrase][:110]} for phrase, count in ranked]
    return found


CONVERSATION_FILE = "code_switching/soap_opera/balanced_engzul.jsonl"          # unscripted-sounding dialogue, each segment tagged zul or eng
STRUCTURED_FILE = "spoken/anv_zul.jsonl"                                       # people talking about health, farming, transport: structured speech
MIN_USES, MIN_STRUCTURED, MIN_RATIO, MAX_LETTERS, FRAGMENT_SHARE = 12, 40, 20.0, 9, 0.25


def conversational_fillers(resources: Path, top: int = 8) -> list[dict[str, Any]]:
    """The Zulu words conversation uses far more than structured speech: the fallback padding for a speaker whose English has none.  Read from two human-transcribed corpora, nothing typed in:
    the soap-opera dialogue (words tagged Zulu, so English code-switching is out) against the 2-million-word structured speech (a word must exist there too, so a name or a slip is out), the
    ratio of their rates at least MIN_RATIO.  Bound morphemes the transcripts split with an underscore (``e_ mood _ini``) are left out.  Evidence for the agent, never an answer picked for it."""
    zulu: Counter[str] = Counter()
    english: Counter[str] = Counter()
    fragment: Counter[str] = Counter()
    example: dict[str, str] = {}
    total = 0
    try:
        for line in (resources / CONVERSATION_FILE).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            whole = " ".join(str(seg.get("text") or "") for seg in row.get("segments") or [])
            for seg in row.get("segments") or []:
                for raw in str(seg.get("text") or "").lower().split():
                    word = re.sub(r"[^a-z'\-]", "", raw)
                    if not word:
                        continue
                    if seg.get("lang") == "zul":
                        zulu[word] += 1
                        total += 1
                        example.setdefault(word, whole[:110])
                        if "_" in raw:
                            fragment[word] += 1
                    else:
                        english[word] += 1
        structured: Counter[str] = Counter()
        structured_total = 0
        for line in (resources / STRUCTURED_FILE).read_text(encoding="utf-8").splitlines():
            for word in re.findall(r"[a-z][a-z'\-]*", str(json.loads(line).get("transcript") or "").lower()):
                structured[word] += 1
                structured_total += 1
    except (OSError, ValueError):
        return []
    if not total or not structured_total:
        return []
    found: list[dict[str, Any]] = []
    for word, uses in zulu.items():
        if uses < MIN_USES or len(word) > MAX_LETTERS or structured[word] < MIN_STRUCTURED or english[word] * 3 > uses or fragment[word] > FRAGMENT_SHARE * uses:
            continue
        ratio = (uses / total) / ((structured[word] + 1) / structured_total)
        if ratio >= MIN_RATIO:
            found.append({"word": word, "count": uses, "ratio": round(ratio, 1), "example": example[word]})
    return sorted(found, key=lambda x: (-x["count"], x["word"]))[:top]


def _letters(text: str) -> int:
    return sum(ch.isalpha() for ch in re.sub(r"~\w+", "", text))


def budget_rows(items: Sequence[Mapping[str, Any]], unit_ids: Iterable[str], estimate_ms: Any) -> list[dict[str, Any]]:
    """For each named unit: its window, the length its text is estimated to take (``estimate_ms(text, speaker_id, start_ms, end_ms)``, the free estimate with the unit's own pinned pace),
    how far short of the window or over it that is, and how many letters would close the gap at that pace (positive: add, negative: cut)."""
    wanted = {str(u) for u in unit_ids}
    rows: list[dict[str, Any]] = []
    for item in items:
        if str(item.get("item_id")) not in wanted:
            continue
        text = str(item.get("zulu_text") or item.get("text") or "")
        start, end = int(item["start_ms"]), int(item["end_ms"])
        speaker = str(item.get("speaker_id") or "SPEAKER_00")
        estimated = estimate_ms(text, speaker, start, end)
        if not estimated or not _letters(text):
            continue
        window = end - start
        letters_per_s = _letters(text) / (estimated / 1000.0)
        gap_s = (window - estimated) / 1000.0
        rows.append({"item_id": str(item["item_id"]), "speaker_id": speaker, "window_s": round(window / 1000.0, 1), "estimated_s": round(estimated / 1000.0, 1), "gap_s": round(gap_s, 1),
                     "letters_now": _letters(text), "letters_to_change": int(round(gap_s * letters_per_s)), "letters_within": max(2, int(round(0.5 * letters_per_s)))})
    return rows


def budget_note(rows: Sequence[Mapping[str, Any]], fillers: Mapping[str, Sequence[Mapping[str, Any]]], conversational: Sequence[Mapping[str, Any]] = (), few: int = 3) -> str:
    """The part of a repair task message that says, for each unit in scope, what to write: how many letters to add or cut to reach its window, and the speaker's own padding words."""
    if not rows:
        return ""
    lines = ["Each unit in scope and what it needs (OmniVoice's length formula at the pace this unit will be spoken at; good to about 2%): "]
    for row in rows:
        if row["gap_s"] >= 0.3:
            need = f"{row['gap_s']:.1f} s short: ADD about {row['letters_to_change']} letters"
        elif row["gap_s"] <= -0.1:
            need = f"{-row['gap_s']:.1f} s over: CUT about {-row['letters_to_change']} letters"
        else:
            need = "already at its edge: leave it"
        lines.append(f"{row['item_id']} {row['speaker_id']} ({row['window_s']} s window, {row['estimated_s']} s now, {row['letters_now']} letters): {need}, to land within about {row['letters_within']} letters of that.")
    speakers = {row["speaker_id"] for row in rows if row["gap_s"] >= 0.3}
    for speaker in sorted(speakers):
        own = list(fillers.get(speaker) or [])
        if own:
            said = "; ".join(f"'{x['phrase']}' x{x['count']}" for x in own)
            lines.append(f"{speaker} pads their English with: {said}. To lengthen a short unit use the Zulu form of one of THESE, where this speaker would say it (find the form with human_translations or attested_zulu), before adding anything else; do not invent filler.")
        if len(own) < few and conversational:
            heard = "; ".join(f"'{x['word']}' x{x['count']} (\"{x['example'][:60]}\")" for x in conversational[:6])
            lines.append(f"{speaker} has {'few' if own else 'no'} filling words of their own in the English. What Zulu speakers in conversation say far more than in structured speech (soap-opera dialogue against 2 million words of talks; use one only where {speaker} would sound natural saying it, "
                         f"and only if it fits the sentence): {heard}.")
    lines.append("Write ONE new version of each unit to its budget, then ONE check_texts over all of them and ONE measure_tts_batch of the units you changed; touch a unit again only if it is still more than that many letters off or over its window. ")
    return "\n".join(lines) + "\n"
