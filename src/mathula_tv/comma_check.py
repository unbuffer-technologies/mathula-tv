"""Where the Zulu text puts commas against where the SOURCE speakers actually paused.

A comma costs the listener about 0.15 s of silence (measured on both voices).  The agent writes commas from habit, among them one on each side of a name of address ("Impela, Ayanda,
sethemba").  A person talking to a colleague does not stop round a name: in this clip's source "Indeed, Ayanda, hopefully" was said with no pause on either side of Ayanda (0.0 s).
A dub that stops where nobody stopped sounds like a script being read.  (The dub's comma count overall was in line with the speakers' pauses: 6 of 7 units had fewer commas than
pauses; the defect is WHERE the commas sit, chiefly round names of address.)

Two measurements per unit, both from the source audio's word timings (``analysis/azure_diarization.json``), both only evidence for the agent (never a gate):

* a comma beside a name that the speaker said with no pause on that side (a name of address is said in the same breath as the words round it);
* how many commas the unit has against how many times the speaker really paused (0.25 s or more) in the same stretch.
"""
from __future__ import annotations

import difflib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

NO_PAUSE_S = 0.12            # a gap below this is the speaker running the words together: a comma here is the dub's, not the speaker's
REAL_PAUSE_S = 0.25          # a gap of this much is a pause the speaker took, and a comma may follow it
OVER_PUNCTUATED = 2.0        # commas per real pause above which a stretch is flagged (with at least MIN_COMMAS commas)
MIN_COMMAS = 6
_PREFIX = re.compile(r"^(?:ngo|nga|nge|kwa|kuka|ku|ka|na|no|ne|wa|we|ya|u|o|e|a)-?(?=[A-Z])")          # grammar glued before a name: u-Ayanda, noSenzo, eNorth
_TOKEN = re.compile(r"[^\s,]+|,")


def load_words(job_root: Path) -> list[dict[str, Any]]:
    """The source's word timings, in order (``text``, ``start``, ``end`` in seconds); empty when the job has none."""
    try:
        words = json.loads((Path(job_root) / "analysis" / "azure_diarization.json").read_text(encoding="utf-8")).get("words") or []
    except (OSError, ValueError):
        return []
    return sorted((w for w in words if isinstance(w, Mapping) and "start" in w and "end" in w and w.get("text")), key=lambda w: float(w["start"]))


def _letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", str(text).lower())


def _similar(a: str, b: str) -> bool:
    return len(a) >= 3 and len(b) >= 3 and (a == b or difflib.SequenceMatcher(None, a, b).ratio() >= 0.75)          # the recogniser spells names its own way (Mpanga for Maphanga)


def _window(words: list[dict[str, Any]], start_ms: int, end_ms: int) -> list[dict[str, Any]]:
    return [w for w in words if float(w["start"]) >= start_ms / 1000 - 0.05 and float(w["end"]) <= end_ms / 1000 + 0.05]


def _gap_before(inside: list[dict[str, Any]], index: int) -> float | None:
    return max(0.0, float(inside[index]["start"]) - float(inside[index - 1]["end"])) if index else None


def _gap_after(inside: list[dict[str, Any]], index: int) -> float | None:
    return max(0.0, float(inside[index + 1]["start"]) - float(inside[index]["end"])) if index + 1 < len(inside) else None


def phrased(english: str, words: list[dict[str, Any]], start_ms: int, end_ms: int) -> str | None:
    """The segment's English with the pauses its speaker really took marked in the text, ``|0.7s|`` for a pause of a quarter second or more (its length in seconds).  The words
    of the written English are lined up with the timed words the recogniser heard (they differ where pass 1 corrected a name), so every mark sits after the right word.  The agent
    reads this beside the punctuation of the English: a comma with no mark after it was not a pause, and a mark with no comma is one the speaker took anyway."""
    inside = _window(words, start_ms, end_ms)
    tokens = str(english or "").split()
    if len(inside) < 2 or not tokens:
        return None
    heard = [_letters(w["text"]) for w in inside]
    written = [_letters(t) for t in tokens]
    mapped: dict[int, int] = {}
    for block in difflib.SequenceMatcher(None, heard, written, autojunk=False).get_matching_blocks():
        for offset in range(block.size):
            mapped[block.a + offset] = block.b + offset
    marks: dict[int, str] = {}
    last = -1
    for k in range(len(inside) - 1):
        if k in mapped:
            last = mapped[k]
        gap = float(inside[k + 1]["start"]) - float(inside[k]["end"])
        if gap >= REAL_PAUSE_S and last >= 0 and last not in marks:
            marks[last] = f"|{gap:.1f}s|"          # the pause and how long it was: a ~pause marker in the Zulu is sized from it
    if not marks:
        return " ".join(tokens)
    return " ".join(t + (" " + marks[i] if i in marks else "") for i, t in enumerate(tokens))


def conversation(segments: Iterable[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Who talks in the clip and in what order: a clip of several speakers is a conversation, and the units are turns in it, each one an answer to the speaker before."""
    turns = []
    for segment in segments:
        speaker = str(segment.get("speaker_id") or "")
        if speaker and (not turns or turns[-1] != speaker):
            turns.append(speaker)
    speakers = list(dict.fromkeys(turns))
    if len(speakers) < 2:
        return None
    return {"speakers": len(speakers), "turn_order": turns[:40],
            "note": (f"This clip is a conversation between {len(speakers)} people, not a monologue. Each turn answers the one before it and often hands over to the next speaker by name "
                     "(\"Yeah, Mfundo, ...\", \"Kenny, ...\", \"Indeed, Ayanda, ...\"). People who talk to each other say a name of address in the same breath as the words round it, and "
                     "take a pause where the thought changes. 'phrasing' shows where each speaker really paused.")}


def locate(source_text: str, words: list[dict[str, Any]]) -> tuple[int, int] | None:
    """Where in the source audio this English was said (start_ms, end_ms), found by lining its words up with the recogniser's timed words; None when too little of it matches.
    check_text is given the English of a unit, not its window, and the commas of a draft should be checked while it is a draft."""
    written = [_letters(t) for t in str(source_text or "").split() if _letters(t)]
    heard = [_letters(w["text"]) for w in words]
    if len(written) < 3 or not heard:
        return None
    blocks = [b for b in difflib.SequenceMatcher(None, heard, written, autojunk=False).get_matching_blocks() if b.size >= 2]
    if sum(b.size for b in blocks) < 0.5 * len(written):
        return None
    first, last = min(b.a for b in blocks), max(b.a + b.size - 1 for b in blocks)
    return int(float(words[first]["start"]) * 1000), int(float(words[last]["end"]) * 1000)


def draft_check(zulu_text: str, source_text: str, words: list[dict[str, Any]]) -> list[str]:
    """The comma_check evidence of a draft, as short sentences (empty when nothing is worth saying)."""
    window = locate(source_text, words)
    found = comma_check(zulu_text, words, *window) if window else None
    if not found:
        return []
    return [*found.get("commas_where_the_speaker_did_not_pause", []), *([found["over_punctuated"]] if found.get("over_punctuated") else []),
            *found.get("pauses_the_speaker_held_that_the_dub_does_not", [])]


def _capitalised(token: str) -> bool:
    return token != "," and _PREFIX.sub("", token.strip(".!?:;\"'“”"))[:1].isupper()


def _stands_alone(tokens: list[str], at: int, side: str) -> bool:
    """Is the name touching this comma a name of address, standing between marks (", Ayanda,", ". Kenny Maphanga,", "lapho, Ayanda.")?  A name inside a clause ("uJenene Mkhwanazi,
    okwakumele...", "nge-IDAC, nakuba...") is not spoken to anyone: its comma is a clause boundary, which the Zulu may need whatever the English did."""
    step = 1 if side == "before" else -1          # a comma before the name: the name runs on to the right; a comma after it: to the left
    end = at
    while 0 <= end + step < len(tokens) and _capitalised(tokens[end + step]) and not tokens[end].endswith((".", "!", "?")) and (end - at) * step < 2:
        end += step
    far = end + step                                                   # the token just outside the name, on the side away from the comma
    if side == "before":
        return far >= len(tokens) or tokens[far] == "," or tokens[end].endswith((".", "!", "?", ":", ";"))
    return far < 0 or tokens[far] == "," or tokens[far].endswith((".", "!", "?", ":", ";"))


LONG_PAUSE_S = 0.8           # a pause of this much is the speaker holding the thought: the dub should hold it too, and a comma (0.15 s) does not
NEAR_CHARS = 35
NEAR_FRACTION = 0.2          # how far from its proportional place a silence may sit and still be the speaker's pause: this share of the sentence (a retold sentence moves words), at least NEAR_CHARS              # "the same place" in the Zulu: this many letters either side of where the pause falls in the sentence (about 2.5 s of speech)
HELD_AT_STOP_MS = 920        # the silence a full stop leaves by itself
RETIMING_REACH_MS = 900      # sentence_retiming moves a sentence end's silence by up to this much toward the speaker's
_END_MARK = re.compile(r"[.?!…][\"'”’)\]]*$")
_CLAUSE_MARK = re.compile(r"[,;:—–][\"'”’)\]]*$")


def _zulu_sentences(text: str) -> list[list[dict[str, Any]]]:
    """The Zulu as sentences of words, each with where it ends (letters from the sentence start) and the silence the voice leaves after it: 0.92 s after a full stop, 0.15 s after
    a comma, or the ``~pauseN`` written after it (the TOTAL silence there)."""
    sentences: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    cursor = 0
    for token in str(text or "").split():
        marked = re.fullmatch(r"~pause(\d{3,4})", token, re.IGNORECASE)
        if marked:
            last = current[-1] if current else sentences[-1][-1] if sentences else None          # a marker after a full stop belongs to the sentence that stop ended
            if last is not None:
                last["silence"] = max(last["silence"], int(marked.group(1)))
            continue
        cursor += len(token) + 1
        silence = HELD_AT_STOP_MS if _END_MARK.search(token) else 150 if _CLAUSE_MARK.search(token) else 0
        current.append({"word": token, "end": cursor, "silence": silence, "stop": silence == HELD_AT_STOP_MS})
        if silence == HELD_AT_STOP_MS:
            sentences.append(current)
            current, cursor = [], 0
    if current:
        sentences.append(current)
    return sentences


def _english_sentences(inside: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    sentences: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for word in inside:
        current.append(word)
        if str(word["text"]).rstrip("\"'”’)").endswith((".", "?", "!")):
            sentences.append(current)
            current = []
    if current:
        sentences.append(current)
    return sentences


def _place_in_zulu(k: int, pairs: list[tuple[int, int]], e_sentence: list[dict[str, Any]], z_sentence: list[dict[str, Any]],
                   cumulative_e: list[int], total_z: int) -> tuple[float, int]:
    """Where in the Zulu (letters from the sentence start) the silence after English word ``k`` belongs, and how far from it a silence may sit and still answer: found between the two
    aligned words that enclose ``k`` (word_alignment) by how far ``k`` is from each, so the error is the stretch between two anchors and not the length of the sentence.  A word that is
    itself aligned puts the place at the end of its Zulu word."""
    for e_index, z_index in pairs:
        if e_index == k:
            return float(z_sentence[z_index]["end"]), NEAR_CHARS
    before = [(e, z) for e, z in pairs if e < k]
    after = [(e, z) for e, z in pairs if e > k]
    e0, z0 = (before[-1][0], z_sentence[before[-1][1]]["end"]) if before else (-1, 0)
    if after:
        e1, z1 = after[0][0], z_sentence[after[0][1]]["end"] - len(str(z_sentence[after[0][1]]["word"]))
        e1_start = cumulative_e[e1 - 1] if e1 else 0                    # the English letters before the next aligned word starts
    else:
        e1, z1, e1_start = len(e_sentence), total_z, cumulative_e[-1]
    e0_end = cumulative_e[e0] if e0 >= 0 else 0
    z1 = max(z1, z0)
    span_e = max(1, e1_start - e0_end)
    fraction = min(1.0, max(0.0, (cumulative_e[k] - e0_end) / span_e))
    return z0 + fraction * (z1 - z0), max(NEAR_CHARS, int(NEAR_FRACTION * (z1 - z0)))


def _held_pauses_aligned(zulu: list[list[dict[str, Any]]], inside: list[dict[str, Any]], aligner: Any) -> list[str] | None:
    """The held pauses found by the aligned words (word_alignment) over the whole unit, or None when nothing could be aligned (the position-only check then runs).  The speaker's
    sentences and the Zulu's do not pair by number (``Yes, yes, yes.`` is three Zulu words in two sentences, and every pause after it is one sentence out), so no sentence pairing is
    used: the silence after English word k is looked for at the place the words round it are said, within the stretch between the two aligned words that enclose it."""
    flat: list[dict[str, Any]] = []
    offset = 0
    for sentence in zulu:
        flat.extend({**w, "end": w["end"] + offset} for w in sentence)
        offset += sentence[-1]["end"]
    pairs = aligner([str(w["text"]) for w in inside], [str(w["word"]) for w in flat])
    if not pairs:
        return None
    lengths = [len(str(w["text"])) + 1 for w in inside]
    cumulative_e = [sum(lengths[:i + 1]) for i in range(len(lengths))]
    total_z = flat[-1]["end"]
    used: set[int] = set()
    found: list[str] = []
    for k, word in enumerate(inside):
        gap = _gap_after(inside, k)
        if gap is None or gap < LONG_PAUSE_S:
            continue
        sentence_end = str(word["text"]).rstrip("\"'”’)").endswith((".", "?", "!"))
        need_ms = int(round(gap * 1000))
        target, window = _place_in_zulu(k, pairs, inside, flat, cumulative_e, total_z)
        candidates = [w for w in flat if id(w) not in used and abs(w["end"] - target) <= window
                      and (w["silence"] >= 0.75 * need_ms or (sentence_end and w["stop"] and w["silence"] + RETIMING_REACH_MS >= need_ms))]          # a stop's silence is moved toward the speaker's at a sentence end
        if candidates:
            used.add(id(min(candidates, key=lambda w: abs(w["end"] - target))))
            continue
        near = [w for w in flat if abs(w["end"] - target) <= NEAR_CHARS] or [min(flat, key=lambda w: abs(w["end"] - target))]
        best = max(near, key=lambda w: w["silence"])
        anchor = min(near, key=lambda w: abs(w["end"] - target))
        index = flat.index(anchor)
        around = " ".join(w["word"] for w in flat[max(0, index - 1):index + 2])
        said = " ".join(str(w["text"]) for w in inside[max(0, k - 2):k + 1])
        found.append(f"after “{said}” the speaker held {gap:.1f} s; near “{around}” your Zulu leaves {best['silence'] / 1000:.2f} s "
                     f"(~pause{int(round(gap * 100)) * 10} after the word that ends the thought would match)")
    return list(dict.fromkeys(found))[:4]


def held_pauses(text: str, inside: list[dict[str, Any]], aligner: Any = None) -> list[str]:
    """The pauses of 0.8 s or more that the source speaker held inside this unit and the Zulu does not hold: at that place in the sentence the Zulu has no stop and no ``~pauseN``
    that long (a comma leaves 0.15 s, a full stop 0.92 s).  With an ``aligner`` (word_alignment: names, numbers and words the parallel corpus translates tie the Zulu's words to the
    speaker's) the place is found between the two aligned words that enclose the pause, over the whole unit; on prod-97 and prod-99 this removed the pauses the agent had rightly declined
    (its ~pause stood right after the translated phrase while the sentence's proportional place fell a sentence away, because ``Yes, yes, yes.`` is two Zulu sentences) and reported misses the
    position check never saw.  Without one, or when nothing aligns, the place is found by how far into its sentence the pause falls (sentence by sentence when the English and the Zulu
    have as many sentences, over the whole unit otherwise), so it is a position, not an exact word: a silence within NEAR_FRACTION (0.2) of the sentence, and at least NEAR_CHARS letters, of
    that place answers the pause, and one silence answers one pause.  prod-97 and prod-98 showed why the window has to scale with the sentence: the agent declined three points because its
    pause was already in the unit a few words from where the proportional place fell (a retold sentence moves its words); on the agents' final texts the points shown fell by about half
    (11 -> 4, 14 -> 6, 17 -> 7, 6 -> 3), and with every ~pause marker deleted the check still reported 10-15 of the 33-37 held pauses."""
    zulu = _zulu_sentences(text)
    english = _english_sentences(inside)
    if not zulu or not english:
        return []
    if aligner is not None:
        found_aligned = _held_pauses_aligned(zulu, inside, aligner)
        if found_aligned is not None:
            return found_aligned
    if len(zulu) != len(english):                       # the sentences do not pair up: the whole unit is one stretch on each side
        offset = 0
        flat: list[dict[str, Any]] = []
        for sentence in zulu:
            flat.extend({**w, "end": w["end"] + offset} for w in sentence)
            offset += sentence[-1]["end"]
        zulu = [flat]
        english = [[w for s in english for w in s]]
    position = {id(w): i for i, w in enumerate(inside)}
    found: list[str] = []
    for z_sentence, e_sentence in zip(zulu, english):
        lengths = [len(str(w["text"])) + 1 for w in e_sentence]
        total_e, total_z = sum(lengths), z_sentence[-1]["end"]
        tolerance = max(NEAR_CHARS, int(NEAR_FRACTION * total_z))
        inner = z_sentence[:-1] or z_sentence                          # the stop that ends the sentence belongs to the sentence's own end, which is after the speaker's next words
        used: set[int] = set()                                          # one silence is one pause: it cannot answer two of the speaker's
        cumulative = 0
        for k, word in enumerate(e_sentence):
            cumulative += lengths[k]
            gap = _gap_after(inside, position[id(word)])
            if gap is None or gap < LONG_PAUSE_S:
                continue
            at_end = k == len(e_sentence) - 1
            target = total_z if at_end else cumulative / total_e * total_z
            need_ms = int(round(gap * 1000))
            if at_end:
                last = z_sentence[-1]
                if last["silence"] >= 0.75 * need_ms or (last["stop"] and last["silence"] + RETIMING_REACH_MS >= need_ms):
                    continue            # at a sentence end the renderer moves the stop's silence toward the speaker's (up to RETIMING_REACH_MS), so a stop covers a pause that much longer
                near, best = [last], last
            else:
                candidates = [w for w in inner if id(w) not in used and abs(w["end"] - target) <= tolerance and w["silence"] >= 0.75 * need_ms]
                if candidates:
                    used.add(id(min(candidates, key=lambda w: abs(w["end"] - target))))
                    continue
                near = [w for w in inner if abs(w["end"] - target) <= NEAR_CHARS] or [min(inner, key=lambda w: abs(w["end"] - target))]
                best = max(near, key=lambda w: w["silence"])
            anchor = min(near, key=lambda w: abs(w["end"] - target))
            index = z_sentence.index(anchor)
            around = " ".join(w["word"] for w in z_sentence[max(0, index - 1):index + 2])
            said = " ".join(str(w["text"]) for w in e_sentence[max(0, k - 2):k + 1])
            found.append(f"after “{said}” the speaker held {gap:.1f} s; near “{around}” your Zulu leaves {best['silence'] / 1000:.2f} s "
                         f"(~pause{int(round(gap * 100)) * 10} after the word that ends the thought would match)")
    return list(dict.fromkeys(found))[:4]


def comma_check(text: str, words: list[dict[str, Any]], start_ms: int, end_ms: int, aligner: Any = None) -> dict[str, Any] | None:
    """What is worth telling the agent about this unit's commas and pauses, or None when there is nothing."""
    inside = _window(words, start_ms, end_ms)
    if not inside:
        return None
    held = held_pauses(text, inside, aligner)
    text = re.sub(r"~pause\d+", " ", text)
    tokens = _TOKEN.findall(text)
    commas = [i for i, token in enumerate(tokens) if token == ","]
    if not commas:
        return {"pauses_the_speaker_held_that_the_dub_does_not": held} if held else None
    gaps = [float(inside[i + 1]["start"]) - float(inside[i]["end"]) for i in range(len(inside) - 1)]
    pauses = sum(1 for g in gaps if g >= REAL_PAUSE_S)
    spoken = [_letters(w["text"]) for w in inside]
    found: list[str] = []
    for i in commas:
        for side, at in (("before", i + 1), ("after", i - 1)):          # a comma before a name is the gap before it; a comma after a name is the gap after it
            if not 0 <= at < len(tokens) or tokens[at] == ",":
                continue
            raw = tokens[at].strip(".!?:;\"'“”")
            name = _PREFIX.sub("", raw)
            if not name[:1].isupper() or not _stands_alone(tokens, at, side):
                continue
            key = _letters(name)
            hits = [k for k, heard in enumerate(spoken) if _similar(key, heard)]
            measured = [(_gap_before(inside, k) if side == "before" else _gap_after(inside, k)) for k in hits]
            measured = [m for m in measured if m is not None]
            if measured and all(m < NO_PAUSE_S for m in measured):
                left = " ".join(t for t in tokens[max(0, i - 2):i] if t != ",")
                right = " ".join(t for t in tokens[i + 1:i + 3] if t != ",")
                found.append(f"“{left}, {right}” (the speaker said {name} with no pause {side} it: {min(measured):.2f} s)")
    over = len(commas) >= MIN_COMMAS and len(commas) > OVER_PUNCTUATED * max(pauses, 1)
    if not found and not over and not held:
        return None
    result: dict[str, Any] = {"commas": len(commas), "speaker_pauses_of_a_quarter_second_or_more": pauses}
    if held:
        result["pauses_the_speaker_held_that_the_dub_does_not"] = held
    if found:
        result["commas_where_the_speaker_did_not_pause"] = list(dict.fromkeys(found))[:6]
    if over:
        result["over_punctuated"] = f"{len(commas)} commas (about {len(commas) * 0.15:.1f} s of silence) against {pauses} pauses the speaker really took"
    return result


def comma_checks(items: Iterable[Mapping[str, Any]], words: list[dict[str, Any]], aligner: Any = None) -> list[dict[str, Any]]:
    found = []
    for item in items:
        try:
            checked = comma_check(str(item.get("zulu_text") or ""), words, int(item.get("start_ms")), int(item.get("end_ms")), aligner)
        except (TypeError, ValueError):
            continue
        if checked:
            found.append({"item_id": str(item.get("item_id")), **checked})
    return found
