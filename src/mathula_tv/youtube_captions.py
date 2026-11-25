"""YouTube's own captions for a video, as one more reading of what was said.

The system only dubs YouTube videos, and YouTube already ran its own recogniser over the audio.  For a rally that mixes English and
isiZulu its automatic captions are tagged with the language it heard ("zu-orig" for Zulu) and are often right exactly where the English
recogniser is nonsense ("siyayibhadala through tax", "ANC ayinamali, ANC ayibhadali abasebenzi", "nale yabantwana siyayinyusa"), and
wrong where the English one is right (some plain English sentences).  So they are an independent second reading, not a replacement for
the transcript: pass 1 of the native pipeline (the STT restoration, the pipeline's autocorrect) is shown the caption text of the segments where the
English recogniser and YouTube disagree, and restores what the speaker really said there.  There is no separate language-switch phase.

Every other automatic track on the video is a machine translation of the original one and is never fetched.  Captions are optional: a
video without them, or a failed fetch, changes nothing.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

DIRECTORY = Path("input") / "youtube_captions"
SOURCE_RECORDS = ("youtube_source.json", "x_source.json")          # where a downloaded source says what it is: title, channel, the uploader's own text
DESCRIPTION_CHARS = 1500          # how much of the uploader's description pass 1 and the agent are shown


def video_context(job_root: Path) -> dict[str, str]:
    """What the uploader wrote about the video: its title, channel, description and chapter titles (``input/youtube_source.json``, saved when the video was downloaded).  A human-written source that spells the names of the
    story (job 833ea12f: the recogniser wrote "Spico Moussimane" and YouTube's automatic captions "manito" for a stretch whose title says "Mosimane"); empty when the job has none."""
    data: Any = None
    for name in SOURCE_RECORDS:          # a job comes from YouTube or from X: the same fields, the one that exists
        try:
            data = json.loads((Path(job_root) / "input" / name).read_text(encoding="utf-8"))
            break
        except (OSError, ValueError):
            continue
    if not isinstance(data, dict):
        return {}
    out = {k: " ".join(str(data.get(k) or "").split())[:200] for k in ("title", "channel") if data.get(k)}
    description = "\n".join(line.strip() for line in str(data.get("description") or "").strip().splitlines() if line.strip())
    if description:
        out["description"] = description[:DESCRIPTION_CHARS]
    chapters = [" ".join(str(c).split())[:80] for c in data.get("chapters") or [] if str(c).strip()]
    if chapters:
        out["chapters"] = "; ".join(chapters)[:400]
    return out


def find(job_root: Path) -> list[Path]:
    """The caption files stored with a job, the uploader's own subtitles first and YouTube's automatic original-language track after."""
    folder = Path(job_root) / DIRECTORY
    files = sorted(folder.glob("*.json3")) if folder.is_dir() else []
    return sorted(files, key=lambda path: (path.name.split(".")[-2].endswith("-orig"), path.name))


def events(path: Path) -> list[tuple[float, float, str]]:
    """(start, end, text) of each caption line in seconds, in order, without the empty line-break events."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    lines = []
    for event in data.get("events") or []:
        text = "".join(str(seg.get("utf8") or "") for seg in event.get("segs") or []).replace("\n", " ").strip()
        if not text:
            continue
        start = int(event.get("tStartMs") or 0) / 1000.0
        lines.append((start, start + int(event.get("dDurationMs") or 0) / 1000.0, text))
    return lines


def reading(job_root: Path) -> Callable[[float, float], str] | None:
    """A reader returning the caption text that overlaps a window of the audio, or None when the video has no captions."""
    files = find(job_root)
    if not files:
        return None
    lines = events(files[0])
    if not lines:
        return None

    def read(start: float, end: float) -> str:
        seen: list[str] = []
        for line_start, line_end, text in lines:
            if line_end > start and line_start < end and text not in seen:
                seen.append(text)
        return " ".join(seen)

    return read


HINT_PAD_SECONDS = 0.6          # caption lines are a few seconds long and start a little off the audio: look this far either side of a segment
HINT_MIN_WORDS = 3
HINT_MIN_MISSING = 0.4          # share of a segment's words that the captions for the same moment do not have
HINT_MAX_SEGMENTS = 40
HINT_MAX_CHARS = 500


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z]+", str(text).lower())


def disagreements(job_root: Path, segments: Sequence[Mapping[str, Any]], *, text_key: str = "source_text", minimum: float = HINT_MIN_MISSING,
                  min_words: int = HINT_MIN_WORDS) -> list[dict[str, Any]]:
    """The segments whose text (``text_key``) the captions for the same moment do not support, in time order: ``{"segment_id", "order", "share", "caption", "track"}``, where
    ``share`` is the part of the segment's words (three or more of them) that the overlapping captions lack.  A text comparison only: it says where two readings differ,
    never which one is right.  Empty when the video has no captions."""
    files = find(job_root)
    lines = events(files[0]) if files else []
    if not lines:
        return []
    track = files[0].name.split(".")[-2]
    found: list[dict[str, Any]] = []
    for order, segment in enumerate(segments):
        words = _words(segment.get(text_key, ""))
        if len(words) < min_words:
            continue
        start, end = float(segment["start_ms"]) / 1000.0 - HINT_PAD_SECONDS, float(segment["end_ms"]) / 1000.0 + HINT_PAD_SECONDS
        seen: list[str] = []
        for line_start, line_end, text in lines:
            if line_end > start and line_start < end and text not in seen:
                seen.append(text)
        caption = " ".join(seen)
        have = set(_words(caption))
        missing = sum(1 for w in words if w not in have) / len(words)
        if caption and missing >= minimum:
            found.append({"segment_id": str(segment["segment_id"]), "order": order, "share": missing, "caption": caption[:HINT_MAX_CHARS], "track": track})
    return found


def hints(job_root: Path, segments: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The caption text, for the segments whose English the captions do not support: ``{"track": "zu-orig", "segments": {segment_id: caption text}}``, or ``{}`` when the
    video has no captions or no segment differs.  A segment is one when at least ``HINT_MIN_MISSING`` of its words (three or more) are not in the captions that
    overlap it: an English recogniser's nonsense for isiZulu or Sesotho speech, or a different reading of a name.  A text comparison only: it picks which segments the
    restoration pass is shown a second reading of, and says nothing about which reading is right.  The strongest differences first, in time order, at most
    ``HINT_MAX_SEGMENTS``."""
    found = disagreements(job_root, segments)
    if not found:
        return {}
    chosen = sorted(sorted(found, key=lambda row: -row["share"])[:HINT_MAX_SEGMENTS], key=lambda row: row["order"])
    return {"track": chosen[0]["track"], "segments": {row["segment_id"]: row["caption"] for row in chosen}}


def all_words(job_root: Path) -> set[str]:
    """Every word of the job's caption track (lower case): what YouTube heard, whatever the segments say."""
    files = find(job_root)
    return {w for _, _, text in (events(files[0]) if files else []) for w in _words(text)}


GAP_MIN_SECONDS = 1.5           # silence in the transcript, between two segments, that is worth looking at
GAP_MIN_WORDS = 3
GAP_MAX_GAPS = 30
GAP_MAX_CHARS = 300
_NOISE_TAG = re.compile(r"\[[^\]]*\]|\([^)]*\)")           # [applause], [ukuhalalisa], (music): not speech


def gaps(job_root: Path, segments: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The caption text for each stretch of the video that the transcript has no segment for: ``{"after": segment_id, "start_ms", "end_ms", "yt": text}``, in time order.

    A gap is at least ``GAP_MIN_SECONDS`` between one segment's end and the next one's start; its text is the caption lines that overlap it with noise tags removed,
    and a gap needs ``GAP_MIN_WORDS`` words of it.  The lines are a few seconds long, so the text can repeat words the neighbouring segments already hold: the restoration
    pass decides what is new.  It is where an English recogniser skipped speech altogether (a short burst of isiZulu over a crowd, a sentence under applause), which a
    correction of the words it did write cannot recover."""
    files = find(job_root)
    lines = events(files[0]) if files else []
    if not lines:
        return []
    found: list[dict[str, Any]] = []
    for before, after in zip(segments, segments[1:]):
        start, end = int(before["end_ms"]), int(after["start_ms"])
        if (end - start) / 1000.0 < GAP_MIN_SECONDS:
            continue
        seen: list[str] = []
        for line_start, line_end, text in lines:
            if line_end > start / 1000.0 and line_start < end / 1000.0 and text not in seen:
                seen.append(text)
        text = " ".join(_NOISE_TAG.sub(" ", " ".join(seen)).split())
        if len(_words(text)) >= GAP_MIN_WORDS:
            found.append({"after": str(before["segment_id"]), "start_ms": start, "end_ms": end, "yt": text[:GAP_MAX_CHARS]})
    return found[:GAP_MAX_GAPS]
