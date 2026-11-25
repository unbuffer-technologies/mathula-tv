"""Catch words the English transcript got wrong, using YouTube's own captions of the same moment.

Every check the dubbing agent has compares its Zulu with the English transcript. When the transcript itself is wrong, all of
them pass: in job 13eacc6f the speaker said "masipala" (municipality) fast, the transcript wrote the name "Masia", and the dub
told its audience that a person called uMasia must know about the tenders.

Pass 1 already repairs transcript errors from the text (and YouTube's captions where they disagree with a whole segment). This is a
second look at single words: a capitalised word in the middle of a sentence, that is not one of the clip's known names, and that is
rare in the clip, is shown with the caption text YouTube has for the same moment, and one batched model call decides, from the
sentence and that second reading, whether it is a real name or a misheard word. Only a clear "misheard" verdict with a replacement
is acted on.  No recogniser is run here: the audio is not decoded and nothing is transcribed again.

The corrections are written to ``native_dub/source_corrections.json`` and applied to the FROZEN copy of the transcript that
the agent, the gates, the meaning check and the renderer read. The job's own Pass 1 file is never edited, so its caches stay valid.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

SCHEMA = "mathula-source-corrections-v1"
FILE = "source_corrections.json"
MAX_SUSPECTS = 15
MAX_OCCURRENCES = 2
_WORD = re.compile(r"[A-Za-z][A-Za-z'’]*")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_NOT_NAMES = {
    "i", "i'm", "i've", "i'll", "i'd", "mr", "mrs", "ms", "dr", "prof", "monday", "tuesday", "wednesday", "thursday", "friday",
    "saturday", "sunday", "january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
    "november", "december", "south", "african", "africa", "ward", "president", "minister", "mayor", "english", "zulu",
}

SYSTEM_PROMPT = (
    "You check an English speech-recognition transcript of a South African political speech, using the sentence around each "
    "word and 'youtube', the caption text YouTube has for the same moment (an independent reading: often right where this "
    "transcript is wrong, and wrong where it is right; it may be empty). Each token is a capitalised word "
    "in the middle of a sentence that is not a known name of this clip. Decide for each: 'proper_name' (a real person, place or "
    "organisation, spelt plausibly), 'misrecognised' (the recogniser wrote a name for what is really a common word or a "
    "South African loanword, such as 'masipala' for municipality), or 'unsure'. Be conservative: answer 'misrecognised' only "
    "when the sentence clearly needs a different word AND, where there is a caption, it does not contradict it. For 'misrecognised' give the "
    "English words that should stand in the transcript as 'replacement' (for example 'the municipality'); otherwise leave "
    "replacement empty."
)


def schema(ids: Sequence[int]) -> dict[str, Any]:
    return {
        "type": "object", "additionalProperties": False, "required": ["verdicts"],
        "properties": {"verdicts": {"type": "array", "minItems": len(ids), "maxItems": len(ids), "items": {
            "type": "object", "additionalProperties": False, "required": ["id", "verdict", "replacement", "reason"],
            "properties": {"id": {"type": "integer", "enum": list(ids)},
                           "verdict": {"type": "string", "enum": ["proper_name", "misrecognised", "unsure"]},
                           "replacement": {"type": "string"}, "reason": {"type": "string"}}}}},
    }


def find_suspects(segments: Sequence[Mapping[str, Any]], known_names: Iterable[str], *, limit: int = MAX_SUSPECTS) -> list[dict[str, Any]]:
    """Capitalised mid-sentence words that no known name of the clip explains and that occur at most twice."""
    known = {w.casefold() for name in known_names for w in _WORD.findall(str(name))}
    texts = {str(s.get("segment_id")): str(s.get("restored_text") or s.get("source_text") or "") for s in segments}
    everything = " ".join(texts.values()).casefold()
    suspects: list[dict[str, Any]] = []
    seen: set[str] = set()
    for segment_id, text in texts.items():
        sentences = _SENTENCE_END.split(text)
        for number, sentence in enumerate(sentences):
            for position, match in enumerate(_WORD.finditer(sentence)):
                token = match.group(0)
                norm = token.casefold()
                if (position == 0 or not token[0].isupper() or token.isupper() or len(token) < 3 or norm in known
                        or norm in _NOT_NAMES or norm in seen):
                    continue
                if len(re.findall(rf"\b{re.escape(norm)}\b", everything)) > MAX_OCCURRENCES:
                    continue
                seen.add(norm)
                suspects.append({"id": len(suspects), "segment_id": segment_id, "token": token, "sentence": sentence.strip(),
                                 "previous_sentence": sentences[number - 1].strip() if number else ""})
                if len(suspects) >= limit:
                    return suspects
    return suspects


def judge(provider: Any, suspects: list[dict[str, Any]], known_names: Sequence[str]) -> list[dict[str, Any]]:
    """One batched model call for all suspects. Returns the verdict for each (id, verdict, replacement, reason)."""
    if not suspects:
        return []
    payload = {"known_names_of_this_clip": list(known_names)[:60],
               "tokens": [{k: s[k] for k in ("id", "token", "sentence", "previous_sentence", "youtube") if s.get(k)} for s in suspects]}
    response = provider.complete_json(operation="native_source_suspect_check", system_prompt=SYSTEM_PROMPT, payload=payload,
                                      schema=schema([s["id"] for s in suspects]), max_output_tokens=min(4000, 300 + 120 * len(suspects)))
    return list(response.data.get("verdicts") or [])


def corrections_from(suspects: Sequence[Mapping[str, Any]], verdicts: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_id = {int(v["id"]): v for v in verdicts}
    found = []
    for suspect in suspects:
        verdict = by_id.get(int(suspect["id"]))
        if verdict and verdict.get("verdict") == "misrecognised" and str(verdict.get("replacement") or "").strip():
            found.append({"segment_id": suspect["segment_id"], "token": suspect["token"], "replacement": verdict["replacement"].strip(),
                          "reason": str(verdict.get("reason") or "").strip(), "sentence": suspect["sentence"],
                          "audio_phones": suspect.get("audio_phones")})
    return found


def apply_corrections(pass1: dict[str, Any], corrections: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Replace each corrected word in its segment's English (in place). Returns what actually changed."""
    changed = []
    segments = pass1.get("segments") or []
    for correction in corrections:
        # not \b: a token such as "is—" ends in punctuation, where \b would never match
        pattern = re.compile(r"(?<![A-Za-z'])" + re.escape(correction["token"]) + r"(?![A-Za-z'])")
        for segment in segments:
            if str(segment.get("segment_id")) != str(correction["segment_id"]):
                continue
            for field in ("restored_text", "source_text"):
                text = str(segment.get(field) or "")
                if pattern.search(text):
                    segment[field] = pattern.sub(correction["replacement"], text, count=1)
                    if field == "restored_text":
                        changed.append({"segment_id": segment["segment_id"], "was": correction["token"], "now": correction["replacement"],
                                        "reason": correction.get("reason", "")})
                        _extend_window(segment, segments, correction.get("extends_to_ms"))
    return changed


def _extend_window(segment: dict[str, Any], segments: Sequence[Mapping[str, Any]], extends_to_ms: Any) -> None:
    """Speech recovered after a segment's end: its window grows to cover it (never into the next segment)."""
    try:
        target = int(extends_to_ms)
    except (TypeError, ValueError):
        return
    later = [int(s["start_ms"]) for s in segments if int(s["start_ms"]) > int(segment["start_ms"])]
    if later:
        target = min(target, min(later) - 50)
    if target > int(segment["end_ms"]):
        segment["end_ms"] = target
        if "end" in segment:
            segment["end"] = target / 1000.0


START_LOOKBACK_MS = 4000          # how far before a segment's first recognised word its restored opening words may have been spoken
START_QUIET_MS = 300              # a silence this long ends the stretch of speech that leads into the segment


def prepends_words(source_text: str, restored_text: str) -> bool:
    """True when the restored text opens with words the recogniser never heard (a name it skipped: "Samkele Maseko there, giving us ...")."""
    words = lambda t: re.findall(r"[a-z0-9']+", str(t).lower())
    source, restored = words(source_text), words(restored_text)
    if not source or len(restored) <= len(source) or restored[0] == source[0]:
        return False
    return source[0] in restored[1:4]


def speech_onset_before(samples: Any, rate: int, start_ms: int, floor_ms: int) -> int | None:
    """Where the speech that runs into ``start_ms`` began, looking back no further than ``floor_ms`` (the end of the segment before) and ``START_LOOKBACK_MS``: walk back
    from the start while the audio is speech, stop at a silence of START_QUIET_MS.  None when there is no earlier speech joined to the start."""
    import numpy as np

    x = np.asarray(samples, dtype=float)
    window = max(1, int(0.02 * rate))
    count = len(x) // window
    if count < 2:
        return None
    energy = np.sqrt((x[: count * window].reshape(count, window) ** 2).mean(axis=1))
    first = max(0, int(max(floor_ms, start_ms - START_LOOKBACK_MS) / 1000 * rate) // window)
    last = min(count - 1, int(start_ms / 1000 * rate) // window)
    if last <= first:
        return None
    span = energy[first:last + 1]
    voiced = span[span > 150.0]
    loud = span > max(150.0, 0.08 * float(np.median(voiced))) if len(voiced) else np.zeros(len(span), dtype=bool)
    quiet_windows = max(1, int(START_QUIET_MS / 1000 * rate) // window)
    onset: int | None = None
    silent = 0
    for k in range(len(loud) - 1, -1, -1):
        if loud[k]:
            onset, silent = k, 0
        else:
            silent += 1
            if silent >= quiet_windows:
                break
    if onset is None or (onset + first) * window / rate * 1000 >= start_ms - 150:
        return None
    return int((onset + first) * window / rate * 1000)


def extend_start_to_speech(segment: dict[str, Any], previous_end_ms: int, audio_path: Path) -> int | None:
    """A segment whose restored text opens with words the recogniser skipped starts where those words were spoken, not at its first recognised word: its window grows back to
    the speech onset (never into the segment before).  Returns the new start, or None when nothing changed."""
    if not prepends_words(str(segment.get("source_text") or ""), str(segment.get("restored_text") or "")) or not Path(audio_path).is_file():
        return None
    import wave

    import numpy as np

    with wave.open(str(audio_path), "rb") as src:
        rate, channels, width = src.getframerate(), src.getnchannels(), src.getsampwidth()
        if width != 2:
            return None
        begin = max(0, int((int(segment["start_ms"]) - START_LOOKBACK_MS - 200) / 1000 * rate))
        src.setpos(min(begin, src.getnframes()))
        raw = src.readframes(int((int(segment["start_ms"]) / 1000 * rate)) - begin + int(0.1 * rate))
    samples = np.frombuffer(raw, dtype=np.int16).reshape(-1, channels)[:, 0]
    offset_ms = begin / rate * 1000
    onset = speech_onset_before(samples, rate, int(int(segment["start_ms"]) - offset_ms), max(0, int(previous_end_ms + 50 - offset_ms)))
    if onset is None:
        return None
    new_start = int(onset + offset_ms)
    if new_start >= int(segment["start_ms"]):
        return None
    segment["start_ms"] = new_start
    if "start" in segment:
        segment["start"] = new_start / 1000.0
    segment["window_extended_back_to_speech"] = True
    return new_start


def apply_to_frozen(pass1_path: Path, corrections_path: Path) -> list[dict[str, Any]]:
    """Apply a job's source corrections to the frozen copy of Pass 1 (called when a clip is frozen)."""
    corrections_path = Path(corrections_path)
    if not corrections_path.is_file():
        return []
    corrections = json.loads(corrections_path.read_text(encoding="utf-8")).get("corrections") or []
    if not corrections:
        return []
    pass1 = json.loads(Path(pass1_path).read_text(encoding="utf-8"))
    changed = apply_corrections(pass1, corrections)
    pass1.setdefault("corrections", []).extend(
        {"original_text": c["was"], "corrected_text": c["now"], "reason": c["reason"], "source": "audio source check"} for c in changed)
    Path(pass1_path).write_text(json.dumps(pass1, ensure_ascii=False, indent=1), encoding="utf-8")
    return changed


def run(job_root: Path, provider: Any, *, progress: Callable[[str], None] = print) -> Path | None:
    """Examine a job's transcript and write ``native_dub/source_corrections.json``. Never raises for one bad word."""
    from . import youtube_captions

    job_root = Path(job_root)
    native = job_root / "native_dub"
    pass1 = json.loads((native / "pass1_restored_transcript.json").read_text(encoding="utf-8"))
    ledger = pass1.get("context_ledger") or {}
    names = [str(e.get("canonical_text") or e.get("name") or "") for e in (ledger.get("entities") or [])]
    research = native / "pronunciation_research.json"
    if research.is_file():
        names += [str(c.get("canonical_text") or "") for c in json.loads(research.read_text(encoding="utf-8")).get("accepted_corrections") or []]
    names = [n for n in names if n]
    segments = pass1.get("segments") or []
    suspects = find_suspects(segments, names)
    caption = youtube_captions.reading(job_root)            # the captions' text for a window of the video: a second reading, not a recogniser run here
    if suspects and caption is not None:
        windows = {str(s.get("segment_id")): (float(s.get("start_ms") or 0) / 1000.0, float(s.get("end_ms") or 0) / 1000.0) for s in segments}
        for suspect in suspects:
            start, end = windows.get(str(suspect["segment_id"]), (0.0, 0.0))
            if end > start:
                suspect["youtube"] = caption(start - 0.5, end + 0.5)[:400]
    try:
        verdicts = judge(provider, suspects, names)
    except Exception as exc:
        progress(f"[source] the source check could not run ({type(exc).__name__}: {str(exc)[:120]}); transcript left as it is")
        verdicts = []
    corrections = corrections_from(suspects, verdicts)
    for item in corrections:
        progress(f"[source] {item['token']!r} is probably a misheard word -> {item['replacement']!r}  ({item['reason'][:100]})")
    path = native / FILE
    path.write_text(json.dumps({"schema": SCHEMA, "corrections": corrections, "examined": [
        {"token": s["token"], "segment_id": s["segment_id"], "youtube": s.get("youtube"),
         "verdict": next((v.get("verdict") for v in verdicts if int(v["id"]) == s["id"]), None)} for s in suspects]},
        indent=1, ensure_ascii=False), encoding="utf-8")
    progress(f"[source] checked {len(suspects)} unfamiliar capitalised word(s); {len(corrections)} correction(s)")
    return path
