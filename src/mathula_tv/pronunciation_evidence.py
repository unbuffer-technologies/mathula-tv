"""What the speaker in the source audio actually said, word by word.

A name's whole phrase is rarely found verbatim in the speech-to-text transcript ("eThekwini Metro Municipality" was heard
as "Etegini Metro Municipality"), but its English words usually are: the speaker said "Municipality" and "Moses" and
"Party" many times. Each English-shaped word of a name is therefore located on its own, clipped from the source audio, and
the sounds the speaker used are chosen from the dictionary form and its reduced-vowel variants. A word the speaker never
said has no evidence, and the caller must leave that name alone: a synthetic voice reading the name is not the speaker.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .pronunciation_assess import phone_tokens, strip_length


def _letters(text: Any) -> str:
    return re.sub(r"[^a-z0-9']", "", str(text or "").casefold())


def word_spans(words: Sequence[Mapping[str, Any]], word: str, limit: int = 4) -> list[tuple[float, float]]:
    """(start, end) seconds of up to ``limit`` occurrences of ``word`` in the transcript, spread across the recording."""
    target = _letters(word)
    found = [(float(item.get("start") or 0.0), float(item.get("end") or 0.0))
             for item in words if target and _letters(item.get("text")) == target]
    if len(found) <= limit:
        return found
    step = (len(found) - 1) / (limit - 1) if limit > 1 else 0
    return [found[round(i * step)] for i in range(limit)]


ONSET_SLACK_SECONDS = 0.20      # the transcript's start for a word is often late by this much; its first sounds ("oʊ" of OK, "w ɛ" of Western) fall before it
OFFSET_SLACK_SECONDS = 0.12


def word_windows(words: Sequence[Mapping[str, Any]], word: str, limit: int = 4) -> list[tuple[float, float, float, float]]:
    """(start, end, earliest, latest) seconds of up to ``limit`` occurrences of ``word``: the transcript's own span, and how far the word's sounds may
    reach either side of it, which is its slack but never into the neighbouring words (the previous word's end, the next word's start).

    The recogniser's word times are coarse: decoding only inside the span cut the first sounds off OK ("k eː"), SG ("s tʃ iː", the "e" of "es" lost) and
    Western ("ɪ s ə n"), and the lab then matched spellings to those clipped sounds."""
    target = _letters(word)
    found: list[tuple[float, float, float, float]] = []
    for index, item in enumerate(words):
        if not target or _letters(item.get("text")) != target:
            continue
        start, end = float(item.get("start") or 0.0), float(item.get("end") or 0.0)
        before = float(words[index - 1].get("end") or 0.0) if index else 0.0
        after = float(words[index + 1].get("start") or end + OFFSET_SLACK_SECONDS) if index + 1 < len(words) else end + OFFSET_SLACK_SECONDS
        earliest = min(start, max(before, start - ONSET_SLACK_SECONDS))
        latest = max(end, min(after, end + OFFSET_SLACK_SECONDS))
        found.append((start, end, earliest, latest))
    if len(found) <= limit:
        return found
    step = (len(found) - 1) / (limit - 1) if limit > 1 else 0
    return [found[round(i * step)] for i in range(limit)]


def edit_distance(a: Sequence[str], b: Sequence[str]) -> int:
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        row = [i]
        for j, y in enumerate(b, 1):
            row.append(min(previous[j] + 1, row[-1] + 1, previous[j - 1] + (x != y)))
        previous = row
    return previous[-1]


def consensus(heard: Sequence[Sequence[str]]) -> list[str]:
    """The decoding closest to all the others (the medoid): one noisy occurrence cannot decide the word.

    Only decodings at least as long as the median one compete.  A truncated decoding ("d iː" for Heidi, the others "aɪ d iː" and "h eː l d i") is
    close to every longer one by deletion alone, so the plain medoid picked it, and "Dyi" then scored 1.0 against sounds the speaker did not make."""
    heard = [list(h) for h in heard]
    lengths = sorted(len(h) for h in heard)
    median = lengths[len(lengths) // 2]
    longer = [h for h in heard if len(h) >= median]
    return list(min(longer, key=lambda mine: sum(edit_distance(mine, other) for other in heard)))


def heard_word(source: Path, transcript: Sequence[Mapping[str, Any]], word: str, gop: Any, work: Path, limit: int = 6,
               context_seconds: float = 0.25, slack_seconds: float = 0.03) -> dict[str, Any] | None:
    """The phones the speaker used for ``word``, decoded straight from the audio of each occurrence (no dictionary).

    The clip carries ``context_seconds`` either side so the recogniser hears the word whole; only the phones whose middle
    falls inside the transcript's own time span for the word are kept, so the neighbouring words do not leak in.
    """
    from .media import extract_clip

    windows = word_windows(transcript, word, limit)
    if not windows:
        return None
    spans = [(start, end) for start, end, _, _ in windows]
    decoded, timed, clips = [], [], []
    for k, (start, end, earliest, latest) in enumerate(windows):
        lead = min(context_seconds, start)
        clip = extract_clip(source, start, end, work / f"{_letters(word)}_{k}.wav", pad_seconds=context_seconds)
        low, high = lead - (start - earliest), lead + (latest - start)                # the span widened by the slack, never into a neighbouring word
        inside_timed = [(token, begin, finish) for token, begin, finish in gop.decode_timed(clip)
                        if min(lead - slack_seconds, low) <= (begin + finish) / 2 <= max(lead + (end - start) + slack_seconds, high)]
        decoded.append([token for token, _, _ in inside_timed])
        timed.append(inside_timed)
        clips.append(clip)
    return {"phones": consensus([d for d in decoded if d] or decoded), "decoded": decoded, "spans": spans, "timed": timed, "clips": clips}


def speaker_sounds(job_root: Path, name: str, base: str, gop: Any, work: Path, *, looks_isizulu: Callable[[str], bool],
                   limit: int = 4) -> dict[str, Any]:
    """Sounds each word of ``name`` had in the speaker's own audio.

    ``base`` is the research's pronunciation note, one sound group per word. Returns ``words`` (one IPA string per word,
    or None when any English-shaped word has no occurrence in the source audio), ``missing`` (those words) and
    ``evidence`` (how each word was decided).
    """
    import json

    from .media import extract_clip

    diarization = Path(job_root) / "analysis" / "azure_diarization.json"
    source = Path(job_root) / "audio" / "analysis_mono.wav"
    transcript = json.loads(diarization.read_text(encoding="utf-8")).get("words") or [] if diarization.is_file() and source.is_file() else []
    sounds: list[str] = []
    evidence: list[dict[str, Any]] = []
    missing: list[str] = []
    for index, (word, group) in enumerate(zip(name.split(), base.split())):
        fixed = looks_isizulu(word) or (len(word) > 1 and word.isupper())
        if fixed:
            sounds.append(strip_length(group) if looks_isizulu(word) else group)
            evidence.append({"word": word, "source": "not changed by rule: dictionary form"})
            continue
        spans = word_spans(transcript, word, limit)
        if not spans:
            missing.append(word)
            sounds.append(group)
            evidence.append({"word": word, "source": "no occurrence in the source audio"})
            continue
        clips = [extract_clip(source, start, end, work / f"w{index}_{_letters(word)}_{k}.wav") for k, (start, end) in enumerate(spans)]
        picked = gop.choose_expected(clips, group)
        sounds.append(picked["chosen"])
        evidence.append({"word": word, "source": f"speaker, {len(spans)} occurrence(s) at "
                         + ", ".join(f"{start:.1f}s" for start, _ in spans), "dictionary_form": picked["base"],
                         "chosen": picked["chosen"], "chosen_excess": round(picked["chosen_excess"], 3),
                         "dictionary_excess": round(picked["base_excess"], 3)})
    return {"words": None if missing else sounds, "missing": missing, "evidence": evidence}
