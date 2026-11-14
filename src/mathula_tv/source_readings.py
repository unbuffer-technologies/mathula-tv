"""What the pipeline actually heard at one moment of the source, for the dub agent: the independent readings and how sure the recogniser was.

The agent translates the English transcript, which is a recogniser's text corrected by pass 1.  When a sentence looks wrong or ambiguous (a double negative, a name, a number: prod-97 asked for
"audio-level confirmation of the speaker's intended polarity" in a sentence about an eight-year term) it had nothing to check it against.  The audio cannot be played to a model, but the pipeline holds
two independent readings of every moment and the recogniser's per-word confidence:

* the recogniser's own text, and what pass 1 changed in it (with its reason);
* YouTube's automatic caption for the same moment, a second recogniser run on the same audio;
* the recogniser's confidence for each word, and the words it was unsure of;
* the words the two readings disagree on, and the sentences either side.

Two readings that agree are strong evidence; a low-confidence word is the one to doubt; a disagreement is where to look.  It is evidence to weigh, never an answer picked for the agent, and it cannot
settle what a speaker meant.  Free: no model, no speech.
"""
from __future__ import annotations

import difflib
import re
from typing import Any, Mapping, Sequence

LOW_CONFIDENCE = 0.8          # a word the recogniser was less sure of than this is named
MAX_MATCHES = 3
CONTEXT_CHARS = 240
EXCERPT_WORDS = 25            # words of the English kept either side of the phrase asked about
WINDOW_SECONDS = 2.5          # seconds of audio kept either side of it (for the confidences and the caption)
FALLBACK_CHARS = 700          # a long segment asked about without a findable phrase is cut to this
NOTE = ("Two independent readings of the same audio (the recogniser and YouTube's caption): where they agree the words are very probably right; a low-confidence word is the one to doubt; where they "
        "differ is where to look. This cannot play the audio or tell what the speaker meant: it shows what was heard.")
_WORD = re.compile(r"[a-z0-9']+")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(str(text or "").lower().replace("’", "'"))


def find_segments(segments: Sequence[Mapping[str, Any]], segment_id: str = "", phrase: str = "") -> list[Mapping[str, Any]]:
    """The segment with this id, else up to three whose English holds the phrase (case and punctuation ignored)."""
    wanted = str(segment_id or "").strip()
    if wanted:
        return [s for s in segments if str(s.get("segment_id")) == wanted][:1]
    needle = " ".join(_tokens(phrase))
    if not needle:
        return []
    found = [s for s in segments if needle in " ".join(_tokens(str(s.get("restored_text") or s.get("source_text") or "")))
             or needle in " ".join(_tokens(str(s.get("source_text") or "")))]
    return found[:MAX_MATCHES]


def _disagreement(recogniser: str, caption: str) -> dict[str, list[str]]:
    """The words the two readings differ on, between the first and the last place they agree: the English excerpt can run past the caption's window and the caption can begin before it,
    and those ends are not disagreements."""
    a, b = _tokens(recogniser), _tokens(caption)
    matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
    blocks = [m for m in matcher.get_matching_blocks() if m.size]
    a_lo, a_hi = (blocks[0].a, blocks[-1].a + blocks[-1].size) if blocks else (0, len(a))
    b_lo, b_hi = (blocks[0].b, blocks[-1].b + blocks[-1].size) if blocks else (0, len(b))
    only_a, only_b = [], []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal" or (i2 <= a_lo and j2 <= b_lo) or (i1 >= a_hi and j1 >= b_hi):
            continue
        if tag in ("replace", "delete"):
            only_a.append(" ".join(a[i1:i2]))
        if tag in ("replace", "insert"):
            only_b.append(" ".join(b[j1:j2]))
    return {"only_in_the_recogniser": [x for x in only_a if x][:10], "only_in_the_caption": [x for x in only_b if x][:10]}


def _span(haystack: Sequence[str], needle: Sequence[str]) -> tuple[int, int] | None:
    """Where the token sequence ``needle`` sits in ``haystack`` (first and last index), allowing the recogniser's words to differ from it in a place or two."""
    n = len(needle)
    if not n or not haystack:
        return None
    for i in range(0, max(1, len(haystack) - n + 1)):
        if list(haystack[i:i + n]) == list(needle):
            return i, i + n - 1
    matcher = difflib.SequenceMatcher(None, list(haystack), list(needle), autojunk=False)
    blocks = [b for b in matcher.get_matching_blocks() if b.size >= max(2, n // 2)]
    if blocks:
        return min(b.a for b in blocks), max(b.a + b.size - 1 for b in blocks)
    return None


def excerpt(text: str, phrase: str, radius: int = EXCERPT_WORDS) -> str:
    """The words of ``text`` round the phrase, with an ellipsis where it is cut; the start of the text when the phrase is not in it."""
    words = str(text or "").split()
    norm = [" ".join(_tokens(w)) for w in words]
    found = _span([n for n in norm], _tokens(phrase)) if phrase else None
    if found is None:
        return " ".join(words) if len(" ".join(words)) <= FALLBACK_CHARS else " ".join(words)[:FALLBACK_CHARS].rstrip() + " …"
    first, last = max(0, found[0] - radius), min(len(words), found[1] + radius + 1)
    return ("… " if first else "") + " ".join(words[first:last]) + (" …" if last < len(words) else "")


def readings_for(segment: Mapping[str, Any], *, words: Sequence[Mapping[str, Any]], captions: Sequence[tuple[float, float, str]], track: str,
                 before: str = "", after: str = "", phrase: str = "") -> dict[str, Any]:
    """Everything the pipeline holds about what was said in one segment (see the module docstring); with a ``phrase``, only a window round it (a segment can be a whole minute of speech)."""
    start, end = float(segment.get("start_ms") or 0) / 1000.0, float(segment.get("end_ms") or 0) / 1000.0
    inside = [w for w in words if float(w.get("start", -1)) >= start - 0.05 and float(w.get("end", 1e9)) <= end + 0.05]
    window = None
    if phrase and inside:
        found = _span([" ".join(_tokens(str(w.get("text")))) for w in inside], _tokens(phrase))
        if found is not None:
            window = (max(start, float(inside[found[0]]["start"]) - WINDOW_SECONDS), min(end, float(inside[found[1]]["end"]) + WINDOW_SECONDS))
            inside = [w for w in inside if float(w["start"]) >= window[0] - 0.05 and float(w["end"]) <= window[1] + 0.05]
    lo, hi = window or (start, end)
    confident = [(w, float(w["confidence"])) for w in inside if isinstance(w.get("confidence"), (int, float))]
    low = [{"word": str(w.get("text")), "confidence": round(c, 2), "at_s": round(float(w.get("start", 0)), 1)} for w, c in confident if c < LOW_CONFIDENCE][:12]
    lines: list[str] = []
    for line_start, line_end, text in captions:
        if line_end > lo and line_start < hi and text not in lines:
            lines.append(text)
    caption = " ".join(lines)
    restored_full = str(segment.get("restored_text") or segment.get("source_text") or "")
    recogniser_full = str(segment.get("source_text") or restored_full)
    restored, recogniser = excerpt(restored_full, phrase), excerpt(recogniser_full, phrase)
    result: dict[str, Any] = {
        "segment_id": str(segment.get("segment_id")), "speaker_id": segment.get("speaker_id"), "segment_from_s": round(start, 1), "segment_to_s": round(end, 1),
        **({"window_from_s": round(window[0], 1), "window_to_s": round(window[1], 1)} if window else {}),
        "english_you_have": restored,
        "restoration_changed_it": bool(segment.get("changed")),
    }
    if segment.get("changed"):
        result["recogniser_wrote"] = recogniser
        result["restoration_reason"] = segment.get("reason_code")
    result["mean_word_confidence"] = round(sum(c for _, c in confident) / len(confident), 2) if confident else None
    result["words_the_recogniser_was_unsure_of"] = low
    if caption:
        result[f"youtube_caption ({track})"] = caption
        result["where_the_two_readings_differ"] = _disagreement(restored, caption)
    else:
        result["youtube_caption"] = "none for this moment"
    if before and not window:
        result["sentence_before"] = before[-CONTEXT_CHARS:]
    if after and not window:
        result["sentence_after"] = after[:CONTEXT_CHARS]
    return result


def build(segments: Sequence[Mapping[str, Any]], *, words: Sequence[Mapping[str, Any]], captions: Sequence[tuple[float, float, str]], track: str,
          segment_id: str = "", phrase: str = "") -> dict[str, Any]:
    matches = find_segments(segments, segment_id, phrase)
    if not matches:
        return {"error": "no such segment", "hint": "give a segment_id from open_clip, or a phrase of the English (a few words)", "note": NOTE}
    order = {str(s.get("segment_id")): i for i, s in enumerate(segments)}
    out = []
    for segment in matches:
        i = order.get(str(segment.get("segment_id")), 0)
        before = str(segments[i - 1].get("restored_text") or "") if i > 0 else ""
        after = str(segments[i + 1].get("restored_text") or "") if i + 1 < len(segments) else ""
        out.append(readings_for(segment, words=words, captions=captions, track=track, before=before, after=after, phrase=phrase))
    return {"segments": out, "note": NOTE}
