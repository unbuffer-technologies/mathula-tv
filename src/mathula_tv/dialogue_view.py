"""The clip as a conversation: turns in order, who is answering what, and the short sounds of other people inside a turn.

The agent used to be given the segments one by one and a list of speaker ids in order.  Everything else about the conversation it had to infer from the English: that a turn answers a
question the other speaker just asked, that an interviewer's "Mm-hmm." sits inside the analyst's answer (job 37bae6ec: the agent copied it into the Zulu and the voice read it letter by
letter), that a turn opens with the speaker's own "Mhm.".  This module states those facts from evidence the pipeline already holds:

* turns: consecutive segments of one speaker, from the diarization the segments carry;
* the question a turn answers: the last sentence of the turn before it, when that ends with a question mark (the transcript's own punctuation, no model);
* the question a turn asks: its own last sentence, when it ends with a question mark;
* murmurs: YouTube's captions mark a change of speaker with ">>"; a short ">>" line made only of a murmur (Mm-hmm, Mhm, Uh-huh) that falls inside a turn's window is another person
  listening, and a turn that opens with a murmur has the speaker's own.  Neither is speech to translate.

Evidence the agent reads when the clip opens (it stays in the opened clip, which compaction keeps); never a gate.  Free: no model, no speech.
"""
from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from .voice_limits import MURMUR

QUESTION_CHARS = 170
MAX_TURNS_SHOWN = 60
_MURMURS_ONLY = re.compile(r"^(?:\W|" + MURMUR.pattern + r")+$", re.IGNORECASE | re.UNICODE)
_SENTENCES = re.compile(r"(?<=[.!?])\s+")


def _text(segment: Mapping[str, Any]) -> str:
    return str(segment.get("restored_text") or segment.get("source_text") or segment.get("english") or "").strip()


def _clip_question(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= QUESTION_CHARS else "…" + text[-QUESTION_CHARS:].lstrip()


def _last_sentence(text: str) -> str:
    parts = [p for p in _SENTENCES.split(" ".join(str(text or "").split())) if p.strip()]
    return parts[-1] if parts else ""


def murmur_lines(lines: Sequence[tuple[float, float, str]]) -> list[dict[str, Any]]:
    """The caption lines (start, end, text) that open with ">>" and are made only of a murmur: ``[{"at_s": 723.0, "text": "Mm-hmm."}]``, and the lines that open with ">>" and a murmur
    before words (the speaker's own "Mhm. There's quite a lot ...")."""
    found = []
    for start, _end, text in lines:
        if not text.lstrip().startswith(">>"):
            continue
        body = text.lstrip()[2:].strip()
        if not body:
            continue
        if _MURMURS_ONLY.match(body):
            found.append({"at_s": round(float(start), 2), "text": body, "alone": True})
        elif MURMUR.match(body):
            found.append({"at_s": round(float(start), 2), "text": body.split(".")[0].strip() + ".", "alone": False})
    return found


_NAME_STOP = {"the", "and", "van", "der", "den", "von", "mr", "mrs", "ms", "dr", "prof", "adv", "general", "sergeant", "advocate", "minister"}


def people_from_ledger(ledger: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    """Who each speaker is, from the context ledger: ``{speaker id: {"name": "Kenny Maphanga", "role": "Live reporter"}}``.  The ledger lists the speakers by role (``speaker_roles``) and the people
    of the story with a role each (``entities``); a person whose role holds a speaker's role ("Studio anchor addressed by the reporter" holds "Studio anchor") is that speaker."""
    people: dict[str, dict[str, str]] = {}
    entities = [e for e in ledger.get("entities") or [] if isinstance(e, Mapping) and e.get("name") and e.get("role")]
    for role in ledger.get("speaker_roles") or []:
        if not isinstance(role, Mapping) or not role.get("speaker_id") or not role.get("role"):
            continue
        wanted = str(role["role"]).casefold()
        match = next((e for e in entities if wanted in str(e["role"]).casefold()), None)
        if match:
            people[str(role["speaker_id"])] = {"name": str(match["name"]), "role": str(role["role"])}
    return people


def _said_names(text: str, person: Mapping[str, str]) -> list[str]:
    """The parts of a person's name (first name, surname) that a turn says, capitalised as a name, in the order they first appear."""
    found: list[tuple[int, str]] = []
    for part in re.findall(r"[A-Za-z][A-Za-z’'-]+", str(person.get("name") or "")):
        if len(part) < 3 or part.casefold() in _NAME_STOP:
            continue
        hit = re.search(rf"(?<![A-Za-z]){re.escape(part)}(?![A-Za-z])", text)
        if hit:
            found.append((hit.start(), part))
    return [part for _at, part in sorted(found)]


def build(segments: Sequence[Mapping[str, Any]], lines: Sequence[tuple[float, float, str]] | None = None, roles: Mapping[str, str] | None = None,
          people: Mapping[str, Mapping[str, str]] | None = None) -> list[dict[str, Any]]:
    """The turns of the clip, one dict each (see the module docstring).  ``lines`` are the YouTube caption lines (youtube_captions.events); ``roles`` maps a speaker id to a name; ``people``
    (``people_from_ledger``) says who each speaker is, so a turn can show which of the other speakers' names it says."""
    groups: list[list[Mapping[str, Any]]] = []
    for segment in segments:
        if groups and str(groups[-1][0].get("speaker_id") or "") == str(segment.get("speaker_id") or ""):
            groups[-1].append(segment)
        else:
            groups.append([segment])
    murmurs = murmur_lines(lines or [])
    shown = []
    for number, group in enumerate(groups, 1):
        speaker = str(group[0].get("speaker_id") or "")
        start, end = float(group[0].get("start_ms") or 0) / 1000.0, float(group[-1].get("end_ms") or 0) / 1000.0
        text = " ".join(_text(s) for s in group)
        item: dict[str, Any] = {"turn": number, "speaker": (roles or {}).get(speaker) or speaker, "segments": [str(s.get("segment_id")) for s in group],
                                "from_s": round(start, 1), "to_s": round(end, 1)}
        for other, person in (people or {}).items():
            if other != speaker:
                said = _said_names(text, person)
                if said:          # colleagues: the name of the person spoken to is an address, not another person in the story
                    item.setdefault("says_the_name_of", []).append({"name": " / ".join(said), "is": f"{person.get('name')}, the {str(person.get('role') or '').lower()}", "speaks_in_turn": [n for n, g in enumerate(groups, 1) if str(g[0].get("speaker_id") or "") == other][:3]})
        if number > 1:
            before = _last_sentence(" ".join(_text(s) for s in groups[number - 2]))
            if before.endswith("?"):
                item["answers"] = _clip_question(before)
        last = _last_sentence(text)
        if last.endswith("?"):
            item["asks"] = _clip_question(last)
        inside = [m for m in murmurs if m["alone"] and start + 0.5 <= m["at_s"] <= end + 0.5]
        if inside:
            item["murmur_of_another_speaker_inside"] = [f"{m['text']} at {m['at_s']:.1f} s" for m in inside]
        opening = [m for m in murmurs if not m["alone"] and start - 1.0 <= m["at_s"] <= start + 2.0]
        if opening:
            item["opens_with_the_speakers_own_murmur"] = opening[0]["text"]
        shown.append(item)
    return shown[:MAX_TURNS_SHOWN]


NOTE = ("The clip as a conversation. Each turn is one speaker's run of segments; 'answers' is the question the speaker before just asked, 'asks' the question this turn ends on. "
        "A murmur of another speaker inside a turn (Mm-hmm, Mhm, Uh-huh) is that person listening, a sound and not speech: leave it out of the Zulu, or write Yebo / Cha when it is plainly an answer. "
        "A turn that opens with the speaker's own murmur does not need it either. Write each turn as the reply it is, not as a monologue. "
        "'says_the_name_of' lists a name in this turn that belongs to the other person in the conversation: the speaker is talking to that colleague, so it is a name of address, not a "
        "second person in the story. Speech-to-text cannot show the intonation that tells a listener 'Kenny, Vusimuzi Matlala…' is two people; the Zulu has to, so put a comma after the name of address "
        "whether or not the speaker paused.")
