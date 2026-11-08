"""What was said, and in which language: the questions the pipeline cannot settle itself, the listener's answers, and what the pipeline learns from them.

The speaker may move between English, isiZulu, Sepedi, Setswana, Sesotho and Afrikaans in one breath, and the English recogniser writes all of them as English.
YouTube's captions are a second reading, and pass 1 of the native pipeline restores what was said from the two.  Where the readings stay confused, nothing here
decides: the console asks the listener (two kinds of question), and the answer is kept.

* a word question: a token of the Zulu dub that the English dictionary knows but that nobody said (not in the transcript, the restored text or the captions) -- ``Wehle``,
  a Zulu verb that is also a surname in CMUdict.  The answer is the word's language.  English words are respelled by the rules; every other language is left as written.
* a stretch question: a segment whose restored text the captions still do not support.  The answer is which reading was right, or the listener's own text.

The answers are lessons, shared by every job (``language_lessons.json`` beside the pronunciation book):

* a word answered once is settled in every later job;
* a word that looks like words already answered the same way (its *signature*: Zulu-shaped or not, capitalised inside a sentence or not) is settled the same way
  without asking, once ``LEARN_MIN_ANSWERS`` answers agree to ``LEARN_AGREEMENT``; the console shows what was settled by itself so it can be taken back;
* the listener's corrections of the model's text go into the next pass-1 prompt as examples (``examples``), so the model learns when the caption or the transcript is
  the one to trust.

A stretch answer is also applied to the job's restored transcript (``apply_to_segments``) whenever pass 1 runs or is reused.  No recogniser or score is used here.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, MutableSequence, Sequence

from . import youtube_captions

LANGUAGES = ("english", "zulu", "sepedi", "setswana", "sesotho", "afrikaans", "other")
CHOICES = ("model", "transcript", "typed")
LESSONS_NAME = "language_lessons.json"
ANSWERS_NAME = "language_answers.json"
LEARN_MIN_ANSWERS = 3          # answers with one signature before the pipeline settles a new word of that signature by itself
LEARN_AGREEMENT = 0.8          # share of those answers that must agree
STRETCH_MIN_SHARE = 0.5        # share of a restored segment's words the captions lack before the listener is asked
STRETCH_MIN_WORDS = 5          # a shorter segment overlaps the neighbours' caption lines too much for a difference to mean anything
STRETCH_MAX = 8                # unanswered stretch questions per job: the strongest differences
EXAMPLES_MAX = 6               # listener corrections shown to pass 1
EXAMPLE_CHARS = 260
TEXT_MAX = 1500


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _words(text: Any) -> list[str]:
    return re.findall(r"[a-z]+", str(text or "").lower())


def key_of(word: str) -> str:
    return re.sub(r"[^a-z]", "", str(word).casefold())


def _read(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write(path: Path, data: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


# -- the stores ---------------------------------------------------------------------------------------------------------------------------------
def work_dir_of(job_root: Path) -> Path:
    """The work directory a job lives in (``<work>/jobs/<job id>``), where the shared lessons are kept; the job's own folder when it is not laid out that way."""
    job_root = Path(job_root).resolve()
    return job_root.parents[1] if job_root.parent.name == "jobs" else job_root


def lessons_path(work_dir: Path) -> Path:
    return Path(work_dir) / LESSONS_NAME


def load_lessons(work_dir: Path) -> dict[str, Any]:
    data = _read(lessons_path(work_dir))
    return {"words": data.get("words") if isinstance(data.get("words"), dict) else {}, "stretches": data.get("stretches") if isinstance(data.get("stretches"), list) else []}


def answers_path(job_root: Path) -> Path:
    return Path(job_root) / "native_dub" / ANSWERS_NAME


def load_answers(job_root: Path) -> dict[str, Any]:
    data = _read(answers_path(job_root))
    return {"stretches": data.get("stretches") if isinstance(data.get("stretches"), dict) else {}}


# -- words --------------------------------------------------------------------------------------------------------------------------------------
def said_words(job_root: Path) -> set[str] | None:
    """Every word anybody said in this job: the transcript, the restored text and the captions.  None when the job has no restored transcript yet (nothing to compare with)."""
    segments = _read(Path(job_root) / "native_dub" / "pass1_restored_transcript.json").get("segments")
    if not isinstance(segments, list) or not segments:
        return None
    said: set[str] = set()
    for segment in segments:
        if isinstance(segment, Mapping):
            for field in ("source_text", "restored_text", "model_text"):
                said.update(_words(segment.get(field)))
    return said | youtube_captions.all_words(Path(job_root))


def heard(said: set[str], word: str) -> bool:
    """Whether the word was said, in any of its forms: itself, its plural or singular, or a word with the same stem (the speaker said "Pakistanis" and "Somalis"; the dub
    keeps "Pakistan" and "Somalia" under a Zulu prefix).  The stem is all but the last two letters, at least four."""
    key = key_of(word)
    if not key:
        return False
    if key in said or key.rstrip("s") in said or key + "s" in said:
        return True
    need = max(4, len(key) - 2)
    return len(key) >= 4 and any(len(w) >= need and w[:need] == key[:need] for w in said)


def signature(word: str, *, inside: bool) -> str:
    """What a word looks like to the pipeline, for settling a new word by the way earlier ones were answered: whether it is spelled with isiZulu's letters in open
    syllables, and whether a capital marks it inside a sentence (a name) or not."""
    from .interpreter import zulu_shaped

    return f"{'zulu-shaped' if zulu_shaped(word) else 'not-zulu-shaped'}|{'capital-inside' if inside else 'no-capital-inside'}"


def verdict(lessons: Mapping[str, Any], sig: str) -> str | None:
    """The language that earlier answers settle for words with this signature, or None when there are too few answers or they disagree."""
    answers = [str(e.get("language")) for e in (lessons.get("words") or {}).values() if isinstance(e, Mapping) and e.get("signature") == sig]
    if len(answers) < LEARN_MIN_ANSWERS:
        return None
    language, count = Counter(answers).most_common(1)[0]
    return language if count / len(answers) >= LEARN_AGREEMENT else None


def answer_word(work_dir: Path, word: str, language: str, sig: str = "") -> dict[str, Any]:
    if language not in LANGUAGES:
        raise ValueError(f"language must be one of {', '.join(LANGUAGES)}")
    key = key_of(word)
    if not key:
        raise ValueError("a word is required")
    lessons = load_lessons(work_dir)
    lessons["words"][key] = {"word": str(word)[:80], "language": language, "signature": sig, "at": _now()}
    _write(lessons_path(work_dir), lessons)
    return lessons["words"][key]


def forget_word(work_dir: Path, word: str) -> bool:
    lessons = load_lessons(work_dir)
    found = lessons["words"].pop(key_of(word), None) is not None
    if found:
        _write(lessons_path(work_dir), lessons)
    return found


# -- stretches ----------------------------------------------------------------------------------------------------------------------------------
def _clip(text: Any, limit: int = EXAMPLE_CHARS) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def stretch_questions(job_root: Path) -> list[dict[str, Any]]:
    """The stretches the listener is asked about, in time order: the ``STRETCH_MAX`` strongest differences between the restored text and the captions that have no answer
    yet, and every stretch that has one (so an answer can be changed)."""
    job_root = Path(job_root)
    segments = _read(job_root / "native_dub" / "pass1_restored_transcript.json").get("segments")
    if not isinstance(segments, list) or not segments:
        return []
    answers = load_answers(job_root)["stretches"]
    by_id = {str(s.get("segment_id")): s for s in segments if isinstance(s, Mapping)}
    probes = [{"segment_id": sid, "start_ms": s.get("start_ms", 0), "end_ms": s.get("end_ms", 0), "text": s.get("model_text") or s.get("restored_text") or s.get("source_text") or ""}
              for sid, s in by_id.items()]
    found = youtube_captions.disagreements(job_root, probes, text_key="text", minimum=STRETCH_MIN_SHARE, min_words=STRETCH_MIN_WORDS)
    pick = [row for row in sorted((r for r in found if r["segment_id"] not in answers), key=lambda r: -r["share"])[:STRETCH_MAX]]
    read = youtube_captions.reading(job_root)
    chosen: dict[str, dict[str, Any]] = {row["segment_id"]: row for row in pick}
    for sid in answers:
        if sid in by_id and sid not in chosen:
            segment = by_id[sid]
            chosen[sid] = {"segment_id": sid, "share": 0.0, "caption": read(float(segment.get("start_ms", 0)) / 1000, float(segment.get("end_ms", 0)) / 1000) if read else "",
                           "track": (youtube_captions.find(job_root) or [Path("none.none.json3")])[0].name.split(".")[-2]}
    rows = []
    for sid, row in chosen.items():
        segment = by_id[sid]
        rows.append({"kind": "stretch", "id": sid, "start_ms": int(segment.get("start_ms", 0)), "end_ms": int(segment.get("end_ms", 0)), "speaker": str(segment.get("speaker_id") or ""),
                     "transcript": str(segment.get("source_text") or ""), "model": str(segment.get("model_text") or segment.get("restored_text") or ""),
                     "caption": row["caption"], "track": row["track"], "share": round(float(row["share"]), 2), "reason": str(segment.get("reason_code") or ""),
                     "answer": answers.get(sid)})
    return sorted(rows, key=lambda r: r["start_ms"])


def answer_stretch(job_root: Path, work_dir: Path, segment_id: str, choice: str, text: str = "") -> dict[str, Any]:
    """Keep the listener's answer for one stretch of this job and add it to the shared lessons."""
    if choice not in CHOICES:
        raise ValueError(f"choice must be one of {', '.join(CHOICES)}")
    text = " ".join(str(text or "").split())[:TEXT_MAX]
    if choice == "typed" and not text:
        raise ValueError("type what was said")
    job_root = Path(job_root)
    segments = _read(job_root / "native_dub" / "pass1_restored_transcript.json").get("segments") or []
    segment = next((s for s in segments if isinstance(s, Mapping) and str(s.get("segment_id")) == segment_id), None)
    if segment is None:
        raise ValueError("no such stretch")
    answers = load_answers(job_root)
    answers["stretches"][segment_id] = {"choice": choice, "text": text if choice == "typed" else "", "at": _now()}
    _write(answers_path(job_root), answers)
    model = str(segment.get("model_text") or segment.get("restored_text") or "")
    transcript = str(segment.get("source_text") or "")
    final = {"model": model, "transcript": transcript, "typed": text}[choice]
    read = youtube_captions.reading(job_root)
    caption = read(float(segment.get("start_ms", 0)) / 1000, float(segment.get("end_ms", 0)) / 1000) if read else ""
    files = youtube_captions.find(job_root)
    lessons = load_lessons(work_dir)
    lessons["stretches"] = [e for e in lessons["stretches"] if not (e.get("job") == job_root.name and e.get("segment_id") == segment_id)]
    lessons["stretches"].append({"job": job_root.name, "segment_id": segment_id, "track": files[0].name.split(".")[-2] if files else "", "transcript": _clip(transcript, 600),
                                 "model": _clip(model, 600), "caption": _clip(caption, 600), "choice": choice, "final": _clip(final, 600), "at": _now()})
    _write(lessons_path(work_dir), lessons)
    return answers["stretches"][segment_id]


def apply_to_segments(job_root: Path, segments: MutableSequence[Any]) -> int:
    """Put the listener's answers into pass 1's restored segments (the list is changed in place): the segment's text becomes the model's, the transcript's or the listener's
    own.  The model's own text is kept as ``model_text``.  Returns how many segments' text changed; running it again changes nothing."""
    answers = load_answers(job_root)["stretches"]
    changed = 0
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        answer = answers.get(str(segment.get("segment_id")))
        if not answer:
            continue
        segment.setdefault("model_text", segment.get("restored_text", ""))
        target = {"model": segment["model_text"], "transcript": segment.get("source_text", ""), "typed": answer.get("text", "")}.get(answer.get("choice"), "")
        if not str(target).strip():
            continue
        if segment.get("restored_text") != target:
            segment["restored_text"] = target
            segment["changed"] = target != segment.get("source_text")
            segment["reason_code"] = "listener_confirmed"
            changed += 1
        segment["listener_confirmed"] = answer.get("choice")
    return changed


def examples(work_dir: Path) -> list[dict[str, str]]:
    """The listener's latest corrections of the model's restored text, for pass 1's prompt: what the transcript and the caption said, what the listener decided, and which reading
    won.  The model learns from them when the caption or the transcript is the one to trust."""
    rows = load_lessons(work_dir)["stretches"][-EXAMPLES_MAX:]
    return [{"transcript": _clip(r.get("transcript")), "caption": _clip(r.get("caption")), "caption_track": str(r.get("track") or ""), "model_wrote": _clip(r.get("model")),
             "listener_decided": _clip(r.get("final")), "winner": {"model": "the model's text", "transcript": "the transcript", "typed": "the listener's own words"}.get(str(r.get("choice")), "")}
            for r in rows]
