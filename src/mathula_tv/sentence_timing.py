"""Where each Zulu sentence of a unit will land against the source speaker's sentences, shown to the agent before the renderer does it.

The agent used to see only a unit's total length against its window (units_overview, turns_ending_early).  A long unit can fit its window and still run behind the speaker's mouth in the middle
(prod-106 u8: the first sentence ran 2.5 s longer than the English one, so the next sentences started late until the last one caught up), and nothing it saw said so.  The renderer moves silence
between sentences (``sentence_retiming``) but cannot move speech, so the only cure for a late sentence is in the words: take time out of the sentences before it, or put words later.

This runs that same retiming plan on the audio the harness already measured for each unit and reports, per unit, the sentences that would still start well after (or before) the speaker's matching
sentence once the renderer has done its best, and how many sentences could be matched at all (a unit with 3 of 30 matched is mostly unchecked).  Evidence, never a gate.
"""
from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

LATE_S = 0.8                  # a matched sentence that still starts this much after the speaker's is behind the mouth
EARLY_S = 1.2                 # ... and this much before it is far ahead (less often a problem: the gap before it is silence)
ANCHOR_WORDS = 3              # a sentence is placed by a word the aligner matched among its first words, so a re-ordered sentence does not read as a late one
MIN_SENTENCES_FOR_COVERAGE = 6
LOW_COVERAGE = 0.5            # fewer than this share of the Zulu sentences matched: the unit is mostly unchecked
MAX_UNITS = 8
MAX_LATE_PER_UNIT = 4
_BOUNDARY = re.compile(r"[.?!][\"'”’)]*\s*(?:~pause(\d+)\s*)?")


def sentence_openings(text: str) -> list[str]:
    """The first words of each sentence of the unit's text after the first, for the boundaries the voice leaves a silence of 0.8 s or more at (a full stop, or a longer ~pause)."""
    pieces = re.split(r"(?<=[.?!])[\"'”’)]*\s+", re.sub(r"\s*~pause(\d+)", lambda m: f" ~pause{m.group(1)}", text.strip()))
    openings = []
    for piece in pieces[1:]:
        clean = re.sub(r"~pause\d+", "", piece).strip()
        openings.append(" ".join(clean.split()[:5]))
    return openings


def sentence_tokens(text: str) -> list[list[str]]:
    """The words of each sentence of the unit's text (split at . ? !), without the ~pause markers."""
    pieces = re.split(r"(?<=[.?!])[\"'”’)]*\s+", re.sub(r"~pause\d+", " ", text.strip()))
    return [piece.split() for piece in pieces if piece.split()]


def _aligned_lags(item: Mapping[str, Any], onsets_after: Sequence[float], words: Sequence[Mapping[str, Any]], aligner: Any, chars_per_s: float) -> list[float | None]:
    """For each sentence after the first, how late it starts against the English the aligner says it says: the Zulu word nearest the sentence's start that matches an English word is placed
    in time by its characters at the speaker's rate (only among the sentence's first ANCHOR_WORDS words), and compared with when the speaker said that English word.  None where no word of the sentence could be matched."""
    start_s = int(item["start_ms"]) / 1000.0
    inside = sorted((w for w in words if start_s - 0.05 <= float(w["start"]) and float(w["end"]) <= int(item["end_ms"]) / 1000.0 + 0.05), key=lambda w: float(w["start"]))
    english = [str(w.get("text") or "") for w in inside]
    sentences = sentence_tokens(str(item.get("zulu_text") or ""))
    flat = [token for sentence in sentences for token in sentence]
    pairs = aligner(english, flat) if english and flat else []
    by_zulu: dict[int, int] = {}
    for i, j in pairs:
        by_zulu.setdefault(j, i)
    lags: list[float | None] = []
    first = 0
    for k, sentence in enumerate(sentences):
        if k:
            seen = 0
            found: float | None = None
            for offset, token in enumerate(sentence[:ANCHOR_WORDS]):          # an anchor deep in the sentence says little: Zulu reorders (the reporter's name before the broadcaster, where the English has it after)
                if first + offset in by_zulu:
                    spoken_before = seen / chars_per_s if chars_per_s else 0.0
                    found = (onsets_after[k - 1] + spoken_before) - (float(inside[by_zulu[first + offset]]["start"]) - start_s)
                    break
                seen += len(token) + 1
            lags.append(found)
        first += len(sentence)
    return lags


def check_unit(item: Mapping[str, Any], audio_path: str | Path | None, words: Sequence[Mapping[str, Any]], *, shortfall_ms: int = 0, chars_per_s: float = 0.0, aligner: Any = None) -> dict[str, Any] | None:
    """The sentence timing of one unit, or None when there is nothing to say (one sentence, no audio, or every matched sentence lands within tolerance and most are matched)."""
    from . import sentence_retiming

    if not audio_path or not Path(str(audio_path)).is_file():
        return None
    with tempfile.TemporaryDirectory() as tmp:
        report = sentence_retiming.retime(Path(str(audio_path)), Path(tmp) / "out.wav", list(words), int(item["start_ms"]), int(item["end_ms"]), max(0, int(shortfall_ms)), diagnose=True)
    ends, starts = int(report.get("zulu_sentence_ends") or 0), int(report.get("source_sentence_starts") or 0)
    sentences = report.get("sentences")
    if not ends:
        return None
    total = ends + 1
    matched = int(report.get("matched") or 0)
    openings = sentence_openings(str(item.get("zulu_text") or ""))
    late: list[dict[str, Any]] = []
    early: list[dict[str, Any]] = []
    by_content: list[float | None] = []
    if aligner is not None and sentences and len(sentence_tokens(str(item.get("zulu_text") or ""))) == ends + 1:
        try:
            by_content = _aligned_lags(item, [float(r["zulu_starts_after_s"]) for r in sentences], words, aligner, chars_per_s)
        except Exception:                                             # noqa: BLE001 - an advisory view must never fail a submission
            by_content = []
    if by_content:
        matched = sum(1 for lag in by_content if lag is not None)
    for k, row in enumerate(sentences or []):
        if by_content:
            lag = by_content[k] if k < len(by_content) else None
            if lag is None:
                continue
        else:
            target = row.get("source_starts_s")
            if target is None:
                continue
            lag = float(row["zulu_starts_after_s"]) - float(target)
        entry = {"sentence": k + 2, "starts": f"{lag:+.1f}s against the speaker's", **({"opens": openings[k]} if len(openings) == ends and k < len(openings) else {})}
        if lag >= LATE_S:
            if chars_per_s:
                entry["take_out_about_chars"] = int(round(lag * chars_per_s))
            late.append(entry)
        elif lag <= -EARLY_S:
            early.append(entry)
    coverage = matched / ends if ends else 1.0
    thin = total >= MIN_SENTENCES_FOR_COVERAGE and coverage < LOW_COVERAGE
    if not late and not early and not thin:
        return None
    out: dict[str, Any] = {"unit": str(item.get("item_id")), "sentences": total, "speaker_sentences": starts + 1, "matched": f"{matched} of {ends} boundaries" + (" (by what the words say)" if by_content else "")}
    if late:
        out["behind_the_mouth"] = late[:MAX_LATE_PER_UNIT]
    if early:
        out["ahead_of_the_mouth"] = early[:MAX_LATE_PER_UNIT]
    if thin and not late:
        out["mostly_unchecked"] = True
    return out


def check(items: Sequence[Mapping[str, Any]], card: Mapping[str, Any], words: Sequence[Mapping[str, Any]], rates: Mapping[str, float] | None = None, aligner: Any = None) -> list[dict[str, Any]]:
    """The units of a scored submission with something to say about where their sentences land; ``card`` is the scorecard (its ``audio_paths`` and groups)."""
    audio = card.get("audio_paths") or {}
    groups = {str(g.get("group_id")): g for g in card.get("groups") or []}
    found: list[dict[str, Any]] = []
    for item in items:
        unit = str(item.get("item_id"))
        group = groups.get(unit) or {}
        try:
            entry = check_unit(item, audio.get(unit), words, shortfall_ms=int(group.get("shortfall_ms") or 0), chars_per_s=float((rates or {}).get(str(item.get("speaker_id")), 0.0)), aligner=aligner)
        except Exception:                                             # noqa: BLE001 - an advisory view must never fail a submission
            entry = None
        if entry:
            found.append(entry)
    found.sort(key=lambda e: (0 if e.get("behind_the_mouth") else 1, -(len(e.get("behind_the_mouth") or []))))
    return found[:MAX_UNITS]
