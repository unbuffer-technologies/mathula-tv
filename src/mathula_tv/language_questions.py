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
WORDS_MAX = 80                 # words the listener named a language for, shown to pass 1
EXAMPLE_CHARS = 260
TEXT_MAX = 1500
COMMENT_MAX = 600              # the listener's own words to the model about one question


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
    return {"stretches": data.get("stretches") if isinstance(data.get("stretches"), dict) else {},
            "evidence": data.get("evidence") if isinstance(data.get("evidence"), dict) else {}}


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
    answers = [str(e["language"]) for e in (lessons.get("words") or {}).values() if isinstance(e, Mapping) and e.get("language") and e.get("signature") == sig]
    if len(answers) < LEARN_MIN_ANSWERS:
        return None
    language, count = Counter(answers).most_common(1)[0]
    return language if count / len(answers) >= LEARN_AGREEMENT else None


def _comment(value: Any) -> str:
    return " ".join(str(value or "").split())[:COMMENT_MAX]


def answer_word(work_dir: Path, word: str, language: str = "", sig: str = "", comment: str | None = None) -> dict[str, Any]:
    """Keep the listener's answer for a word: its language, and/or a comment to the model (``comment`` None keeps the one it has).  A comment alone does not settle the word."""
    if language and language not in LANGUAGES:
        raise ValueError(f"language must be one of {', '.join(LANGUAGES)}")
    key = key_of(word)
    if not key:
        raise ValueError("a word is required")
    note = None if comment is None else _comment(comment)
    lessons = load_lessons(work_dir)
    old = lessons["words"].get(key) or {}
    if not language and not old.get("language") and not (note if note is not None else old.get("comment")):
        raise ValueError("choose a language or write a comment")
    entry = {"word": str(word)[:80], "language": language or old.get("language", ""), "signature": sig if language else old.get("signature", ""),
             "comment": old.get("comment", "") if note is None else note, "at": _now()}
    lessons["words"][key] = entry
    _write(lessons_path(work_dir), lessons)
    return entry


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
    pick = [row for row in sorted((r for r in found if not (answers.get(r["segment_id"]) or {}).get("choice")), key=lambda r: -r["share"])[:STRETCH_MAX]]
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
                     "answer": answers.get(sid) if (answers.get(sid) or {}).get("choice") else None, "comment": str((answers.get(sid) or {}).get("comment") or "")})
    return sorted(rows, key=lambda r: r["start_ms"])


def answer_stretch(job_root: Path, work_dir: Path, segment_id: str, choice: str = "", text: str = "", comment: str | None = None) -> dict[str, Any]:
    """Keep the listener's answer for one stretch of this job (which reading was right, and/or a comment to the model) and, once there is a decision, add it to the shared
    lessons.  A comment alone steers this video's next pass 1 and decides nothing."""
    if choice and choice not in CHOICES:
        raise ValueError(f"choice must be one of {', '.join(CHOICES)}")
    text = " ".join(str(text or "").split())[:TEXT_MAX]
    if choice == "typed" and not text:
        raise ValueError("type what was said")
    note = None if comment is None else _comment(comment)
    job_root = Path(job_root)
    segments = _read(job_root / "native_dub" / "pass1_restored_transcript.json").get("segments") or []
    segment = next((s for s in segments if isinstance(s, Mapping) and str(s.get("segment_id")) == segment_id), None)
    if segment is None:
        raise ValueError("no such stretch")
    answers = load_answers(job_root)
    old = answers["stretches"].get(segment_id) or {}
    if not choice and not old.get("choice") and not (note if note is not None else old.get("comment")):
        raise ValueError("choose which reading was right or write a comment")
    kept = {"choice": choice or old.get("choice", ""), "text": (text if choice == "typed" else "") if choice else old.get("text", ""),
            "comment": old.get("comment", "") if note is None else note, "at": _now()}
    answers["stretches"][segment_id] = kept
    _write(answers_path(job_root), answers)
    if not kept["choice"]:
        return kept                                              # a comment only: nothing was decided, so it is no lesson for other videos yet
    choice, text = kept["choice"], kept["text"]
    model = str(segment.get("model_text") or segment.get("restored_text") or "")
    transcript = str(segment.get("source_text") or "")
    final = {"model": model, "transcript": transcript, "typed": text}[choice]
    read = youtube_captions.reading(job_root)
    caption = read(float(segment.get("start_ms", 0)) / 1000, float(segment.get("end_ms", 0)) / 1000) if read else ""
    files = youtube_captions.find(job_root)
    lessons = load_lessons(work_dir)
    lessons["stretches"] = [e for e in lessons["stretches"] if not (e.get("job") == job_root.name and e.get("segment_id") == segment_id)]
    lessons["stretches"].append({"job": job_root.name, "segment_id": segment_id, "track": files[0].name.split(".")[-2] if files else "", "transcript": _clip(transcript, 600),
                                 "model": _clip(model, 600), "caption": _clip(caption, 600), "choice": choice, "final": _clip(final, 600), "comment": kept["comment"], "at": _now()})
    _write(lessons_path(work_dir), lessons)
    return kept


def apply_to_segments(job_root: Path, segments: MutableSequence[Any]) -> int:
    """Put the listener's answers into pass 1's restored segments (the list is changed in place): the segment's text becomes the model's, the transcript's or the listener's
    own.  The model's own text is kept as ``model_text``.  Returns how many segments' text changed; running it again changes nothing."""
    answers = load_answers(job_root)["stretches"]
    changed = 0
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        answer = answers.get(str(segment.get("segment_id")))
        if not answer or not answer.get("choice"):
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


def known_words(work_dir: Path) -> list[dict[str, str]]:
    """The words the listener named a language for (the latest ``WORDS_MAX``), for pass 1's prompt: a word the listener said is Sepedi is kept as Sepedi, one they said is
    English is kept as English, wherever the recogniser wrote it as something else."""
    rows = sorted((e for e in load_lessons(work_dir)["words"].values() if isinstance(e, Mapping) and (e.get("language") or e.get("comment"))), key=lambda e: str(e.get("at") or ""))[-WORDS_MAX:]
    return [{"word": str(e.get("word") or ""), **({"language": str(e["language"])} if e.get("language") else {}), **({"comment": _clip(e["comment"], COMMENT_MAX)} if e.get("comment") else {})}
            for e in rows if e.get("word")]


def examples(work_dir: Path) -> list[dict[str, str]]:
    """The listener's latest corrections of the model's restored text, for pass 1's prompt: what the transcript and the caption said, what the listener decided, and which reading
    won.  The model learns from them when the caption or the transcript is the one to trust."""
    rows = load_lessons(work_dir)["stretches"][-EXAMPLES_MAX:]
    return [{"transcript": _clip(r.get("transcript")), "caption": _clip(r.get("caption")), "caption_track": str(r.get("track") or ""), "model_wrote": _clip(r.get("model")),
             "listener_decided": _clip(r.get("final")), "winner": {"model": "the model's text", "transcript": "the transcript", "typed": "the listener's own words"}.get(str(r.get("choice")), ""),
             **({"listener_comment": _clip(r["comment"], COMMENT_MAX)} if r.get("comment") else {})}
            for r in rows]


# -- what the agent said it needed and was not given --------------------------------------------------------------------------------------------------
EVIDENCE_BEFORE_S = 1.0        # seconds of the source played before the words the agent asked about
EVIDENCE_WINDOW_S = 9.0        # and after them
EVIDENCE_FALLBACK_S = 20.0     # the first seconds of the unit, when what the agent asked about cannot be found in the source words
_QUOTED = re.compile(r"""['‘’"“”]([^'‘’"“”]{3,80})['‘’"“”]""")


def _letters_of(text: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text or "").lower())


def locate_in_source(wanted: str, words: Sequence[Mapping[str, Any]], start_s: float, end_s: float) -> tuple[float, float, str] | None:
    """Where in the source audio the words an evidence note names are said: the first quoted phrase of ``wanted`` (``the garbled words immediately after 'third party'``) found in the
    source words inside the unit, as (from_s, to_s, the words heard there).  The window starts just before the phrase and runs past it, because the agent asks about what comes next to it."""
    inside = [w for w in words if float(w["start"]) >= start_s - 0.05 and float(w["end"]) <= end_s + 0.05]
    heard = [_letters_of(w.get("text")) for w in inside]
    for quoted in _QUOTED.findall(str(wanted or "")):
        phrase = [t for t in (_letters_of(p) for p in quoted.split()) if t]
        if not phrase:
            continue
        for i in range(len(heard) - len(phrase) + 1):
            if heard[i:i + len(phrase)] == phrase:
                begin = max(start_s, float(inside[i]["start"]) - EVIDENCE_BEFORE_S)
                finish = min(end_s, float(inside[i]["start"]) + EVIDENCE_WINDOW_S)
                said = " ".join(str(w.get("text") or "") for w in inside if begin <= float(w["start"]) <= finish)
                return begin, finish, said
    return None


def evidence_questions(job_root: Path, run_dir: Path, items: Sequence[Mapping[str, Any]], words: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The listener's questions from the agent's own ``missing_evidence`` notes in this run: what it wanted and why, the stretch of the source to hear (found from the words it quoted, else the
    start of the unit), what the recogniser heard there, and the answer if there is one.  One question per note, kept across runs of the job by ``experience_ledger.evidence_key``."""
    from . import experience_ledger, thinking_report

    job_root = Path(job_root)
    answers = load_answers(job_root)["evidence"]
    by_unit = {str(i.get("item_id")): (n, i) for n, i in enumerate(items)}
    rows = []
    for m in thinking_report.load(Path(run_dir)).get("missing") or []:
        unit, wanted = str(m.get("unit") or ""), str(m.get("wanted") or "")
        key = experience_ledger.evidence_key(job_root.name, unit, wanted)
        index, item = by_unit.get(unit, (None, None))
        row: dict[str, Any] = {"kind": "evidence", "id": key, "unit": unit, "wanted": wanted, "why": str(m.get("why") or ""), "item": index, "transcript": "",
                               "answer": None, "comment": str((answers.get(key) or {}).get("comment") or "")}
        if item is not None and isinstance(item.get("start_ms"), (int, float)) and isinstance(item.get("end_ms"), (int, float)):
            start_s, end_s = float(item["start_ms"]) / 1000, float(item["end_ms"]) / 1000
            found = locate_in_source(wanted, words, start_s, end_s)
            begin, finish, said = found if found else (start_s, min(end_s, start_s + EVIDENCE_FALLBACK_S), "")
            row.update({"start_ms": int(begin * 1000), "end_ms": int(finish * 1000), "transcript": said, "located": found is not None})
        answer = answers.get(key)
        if isinstance(answer, Mapping) and answer.get("text"):
            row["answer"] = {"text": str(answer["text"]), "at": str(answer.get("at") or "")}
        rows.append(row)
    return rows


def answer_evidence(job_root: Path, work_dir: Path, key: str, *, wanted: str = "", unit: str = "", text: str = "", comment: str | None = None) -> dict[str, Any]:
    """Keep the listener's answer to one of the agent's missing-evidence questions: what was really said (``text``) and/or a comment.  The next run on this video is shown it
    (``notes_for_agent``), and the experience ledger marks the note ``answered``."""
    from . import experience_ledger

    if not re.fullmatch(r"[0-9a-f]{16}", str(key or "")):
        raise ValueError("no such question")
    text = " ".join(str(text or "").split())[:TEXT_MAX]
    note = None if comment is None else _comment(comment)
    answers = load_answers(job_root)
    old = answers["evidence"].get(key) or {}
    if not text and not old.get("text") and not (note if note is not None else old.get("comment")):
        raise ValueError("write what was said or a comment")
    kept = {"wanted": str(wanted or old.get("wanted") or "")[:300], "unit": str(unit or old.get("unit") or "")[:20], "text": text or old.get("text", ""),
            "comment": old.get("comment", "") if note is None else note, "at": _now()}
    answers["evidence"][key] = kept
    _write(answers_path(job_root), answers)
    if kept["text"]:
        experience_ledger.mark_answered(work_dir, job_id=Path(job_root).name, key=key, text=kept["text"])
    return kept


def notes_for_agent(job_root: Path) -> list[dict[str, str]]:
    """What the listener said to the agent's missing-evidence questions on this video, for ``open_clip``: ``you_asked_for`` (what it wanted), ``listener_says`` and any comment.  The units may be cut
    differently next time, so the answer is tied to what was wanted, not to a unit id."""
    out = []
    for e in load_answers(job_root)["evidence"].values():
        if isinstance(e, Mapping) and (e.get("text") or e.get("comment")):
            out.append({"you_asked_for": _clip(e.get("wanted"), 300), **({"listener_says": _clip(e["text"], TEXT_MAX)} if e.get("text") else {}),
                        **({"listener_comment": _clip(e["comment"], COMMENT_MAX)} if e.get("comment") else {})})
    return out


def notes(job_root: Path) -> list[dict[str, str]]:
    """The listener's comments on this video's own stretches, for pass 1's prompt: ``{"id": segment_id, "note": comment}``, in segment order."""
    return [{"id": sid, "note": _clip(a["comment"], COMMENT_MAX)} for sid, a in sorted(load_answers(job_root)["stretches"].items()) if isinstance(a, Mapping) and a.get("comment")]
