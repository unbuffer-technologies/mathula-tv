"""A pronunciation lab for the dubbing agent: hear what the voice does with a spelling, and compare it with the speaker.

The agent writes text and never hears audio, and the Zulu TTS voice takes only text, so the only way to make it say an English word
the way the speaker did is to try spellings. This is that loop, made available to the agent as tools:

* ``source(word)``      the sounds the speaker produced for the word, decoded from their own audio (all occurrences, consensus);
* ``try_spelling``      synthesise a spelling in the Zulu voice, decode the sounds it produced with the same recogniser, and say how
                         they differ from the speaker's, in plain terms ("voice drops the d", "vowel o instead of ɔɪ");
* ``candidate_words``   the English words the Zulu text keeps, found in the speaker's audio.

It measures; it never decides. The agent chooses a spelling from the differences, the number of tries is capped, and a person's ear stays
the check on a first use. The comparison below is guidance for the agent, not a pass/fail score.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from collections import Counter
from typing import Any, Callable, Iterable, Mapping, Sequence
from xml.sax.saxutils import escape

MAX_TRIES_PER_WORD = 5
MAX_TRIES_PER_CLIP = 15
MAX_SPELLINGS_PER_CALL = 3
SUGGEST_ROUNDS = 1            # how many times the best spellings found are varied again
LOAN_POOL = 10               # Zulu forms made for a word, each spoken once
LOAN_SHOWN = 5
STANDALONE_AFTER = " kakhulu."   # what follows a spelling when it is spoken alone: a word that ends the utterance loses or devoices its last vowel
SUGGEST_FIRST = 28            # spellings worked out backwards from the speaker's sounds that are spoken first
SUGGEST_PER_ROUND = 20        # varied spellings spoken in each round
SUGGEST_SHOWN = 5             # the distinct best of them that are measured in full and shown
SUGGEST_BY_OPENING = 2        # of those, how many are chosen for the speaker's vowel opening instead of for sounds

_VOWEL_LETTERS = set("aeiouɑɒɔəɛɜɪʊʌæɐøɨ")
_VOICING = {("p", "b"), ("t", "d"), ("k", "ɡ"), ("k", "g"), ("s", "z"), ("f", "v"), ("ʃ", "ʒ"), ("tʃ", "dʒ"), ("θ", "ð")}
_EDGE_COST = 0.3          # sounds at the edges of a word window are noisy in the recogniser: skipping them costs little
_MARKS = "ːˑʰʲʷ̩̯̃"


class PronunciationBudgetExhausted(Exception):
    pass


def _is_vowel(phone: str) -> bool:
    return any(c in _VOWEL_LETTERS for c in phone)


def _base(phone: str) -> str:
    return "".join(c for c in phone if c not in _MARKS)


def substitution_cost(a: str, b: str) -> float:
    if a == b:
        return 0.0
    base_a, base_b = _base(a), _base(b)
    if base_a == base_b:
        return 0.1                                    # length or aspiration only
    if _is_vowel(a) and _is_vowel(b):
        return 0.3 if base_a[:1] == base_b[:1] else 0.6
    if _is_vowel(a) != _is_vowel(b):
        return 1.0
    if (base_a, base_b) in _VOICING or (base_b, base_a) in _VOICING:
        return 0.3
    return 0.7


def align(source: Sequence[str], heard: Sequence[str]) -> dict[str, Any]:
    """Weighted alignment of the speaker's sounds with the voice's. Returns ``distance`` (0 = identical), ``closeness`` (1 - distance)
    and ``differences`` in plain words.

    The speaker's window is cut from running speech by word timings, so its first and last sounds are noisy: skipping them costs
    little. The voice's window is cut cleanly with SSML bookmarks, so a sound the voice adds anywhere (the "ma" of "mafector") is real."""
    n, m = len(source), len(heard)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    back: list[list[tuple[int, int, str] | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0], back[i][0] = i * _EDGE_COST, (i - 1, 0, "skip_source")
    for j in range(1, m + 1):
        dp[0][j], back[0][j] = j * 1.0, (0, j - 1, "skip_heard")
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            options = [(dp[i - 1][j - 1] + substitution_cost(source[i - 1], heard[j - 1]), (i - 1, j - 1, "pair")),
                       (dp[i - 1][j] + 1.0, (i - 1, j, "skip_source")),
                       (dp[i][j - 1] + 1.0, (i, j - 1, "skip_heard"))]
            dp[i][j], back[i][j] = min(options, key=lambda option: option[0])
    # the speaker's last sounds may be skipped cheaply (noisy edge); the voice's extra last sounds may not
    finish = min([(dp[n][j] + (m - j) * 1.0, n, j) for j in range(m + 1)] + [(dp[i][m] + (n - i) * _EDGE_COST, i, m) for i in range(n + 1)])
    total, i, j = finish
    differences: list[str] = []
    if m - j:
        differences.append(f"the voice adds {' '.join(heard[j:])} at the end")
    steps = []
    while (i, j) != (0, 0):
        previous_i, previous_j, step = back[i][j]  # type: ignore[misc]
        steps.append((step, i - 1, j - 1))
        i, j = previous_i, previous_j
    steps.reverse()
    for step, si, hj in steps:
        speaker_edge = si <= 0 or si >= n - 1
        if step == "pair" and source[si] != heard[hj] and substitution_cost(source[si], heard[hj]) > 0.1:
            kind = "vowel" if _is_vowel(source[si]) else "consonant"
            differences.append(f"speaker {source[si]}, voice {heard[hj]} ({kind})")
        elif step == "skip_source" and not speaker_edge:
            differences.append(f"the voice leaves out {source[si]}")
        elif step == "skip_heard":
            differences.append(f"the voice adds {heard[hj]}" + (" at the start" if hj == 0 else ""))
    distance = total / max(n, m, 1)
    return {"distance": round(distance, 3), "closeness": round(max(0.0, 1.0 - distance), 3), "differences": differences[:6]}


def final_consonant_lost(occurrences: Sequence[str], heard: Sequence[str]) -> str | None:
    """The speaker's final consonant when the voice's word has none like it.  ``align`` lets the speaker's last sound go almost free because it is
    cut from running speech, so a plural s that the voice drops ("MaFighter" says f aɪ t a for the speaker's f aɪ t a s) scored 0.94 with no
    difference named.  It is believed only when every decode of the speaker's word (at least two) ends with the same consonant."""
    finals = [seq.split()[-1] for seq in occurrences if seq.strip()]
    if len(finals) < 2 or any(_is_vowel(f) for f in finals) or len({_same_sound(f) for f in finals}) != 1:
        return None
    wanted = _same_sound(finals[0])
    if any(_same_sound(p) == wanted for p in list(heard)[-2:] if not _is_vowel(p)):
        return None
    return f"the voice's word ends without the final {finals[0]} that all {len(finals)} of the speaker's occurrences end with"


# A Zulu prefix written straight onto an English word, no hyphen ("weScience", "iScience", "NeTechnology", "MaFighter"): one to four letters,
# then a capital that starts the English word.  Missed before, so "Science" was never offered for checking and was spoken with Zulu letter rules.
_GLUED_PREFIX = re.compile(r"^[A-Za-z][a-z]{0,3}(?=[A-Z][a-z]{3,})")


def _word_pattern(word: str) -> "re.Pattern[str]":
    """The English word wherever the Zulu text writes it, with a Zulu prefix glued on ("weScience", "i-Technology") or not.  Group 1 is the
    prefix, group 2 the English word itself."""
    english = _GLUED_PREFIX.sub("", word.split("-")[-1])
    return re.compile(r"(?<![A-Za-z'])((?:[A-Za-z][a-z]{0,3}-|[A-Za-z][a-z]{0,3}(?=[A-Z]))?)((?i:" + re.escape(english) + r"))(?![A-Za-z'])")


IN_SENTENCE_MARGINS_MS = ((40, 120), (300, 300), (600, 600), (1000, 1000), (2000, 2000))   # real audio kept before / after the word when it is decoded


def _readings(in_sentence: Sequence[Any]) -> list[list[str]]:
    """The decodes of one word's audio: a single list of sounds counts as one decode."""
    if in_sentence and isinstance(in_sentence[0], str):
        return [list(in_sentence)]
    return [list(reading) for reading in in_sentence] or [[]]


_VOICED_PARTNER = {voiced: voiceless for voiceless, voiced in _VOICING}


def _same_sound(phone: str) -> str:
    """A voiced and a voiceless partner (d/t, dʒ/tʃ, ɡ/k) count as one sound when the recogniser's labels are voted on: it flips between the two
    on half-voiced stops, and how voiced a stop is gets measured on the audio itself (``phone_acoustics``), not read off a label."""
    base = _base(phone)
    return _VOICED_PARTNER.get("k" if base == "ɡ" else base, "k" if base == "ɡ" else base)


def _consonants(phones: Sequence[str]) -> Counter:
    return Counter(_same_sound(p) for p in phones if not _is_vowel(p))


def written_prefix(word: str) -> str:
    """The isiZulu prefix glued straight onto the English word as the Zulu writes it ("Ma" of "MaFighter", "we" of "weScience"); "" when none."""
    found = _GLUED_PREFIX.match(word.split("-")[-1])
    return found.group(0) if found else ""


def _token(text: Any) -> str:
    return re.sub(r"[^a-z0-9']", "", str(text).casefold())


def restore_words(words: Sequence[Mapping[str, Any]], corrections: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The recogniser's timed words with the transcript restorer's word-for-word corrections applied ("patient fund" -> "pension fund", "SASA" -> "SASSA",
    "Leopard creed" -> "Leopard Creek"), so a word the recogniser misheard is still found in the audio at the time it was said.  A correction that
    changes the number of words has no word-to-word match and is left as heard; the original text stays in ``heard_as``."""
    from difflib import SequenceMatcher

    out = [dict(w) for w in words]
    tokens = [_token(w.get("text")) for w in out]
    cursor = 0
    for correction in corrections:
        original, corrected = str(correction.get("original_text") or "").split(), str(correction.get("corrected_text") or "").split()
        wanted = [_token(t) for t in original]
        if not wanted:
            continue
        start = next((i for i in list(range(cursor, len(tokens) - len(wanted) + 1)) + list(range(0, min(cursor, len(tokens) - len(wanted) + 1)))
                      if tokens[i:i + len(wanted)] == wanted), -1)
        if start < 0:
            continue
        cursor = start + len(wanted)
        for tag, a1, a2, b1, b2 in SequenceMatcher(None, wanted, [_token(t) for t in corrected], autojunk=False).get_opcodes():
            if tag == "replace" and a2 - a1 == b2 - b1:
                for n in range(a2 - a1):
                    out[start + a1 + n]["heard_as"] = out[start + a1 + n].get("text")
                    out[start + a1 + n]["text"] = corrected[b1 + n].strip(".,!?;:\"")
    return out


def without_prefix(spelling: str, prefix: str) -> str:
    """``spelling`` without the Zulu prefix it starts with: the prefix is grammar, not the English word, so it is not what an English listener
    or the speaker's sounds are compared with.  A spelling that does not start with the prefix (the agent dropped it) is judged as it is."""
    if prefix and spelling.casefold().startswith(prefix.casefold()) and len(spelling) > len(prefix) + 1:
        return spelling[len(prefix):].lstrip("-")
    return spelling


def lost_in_the_sentence(speaker: Sequence[str], alone: Sequence[str], in_sentence: Sequence[Any]) -> list[str]:
    """Consonants the speaker makes and the voice makes when the word is alone, that the voice does not make in the sentence the word is
    spoken in.  The voice reads the same letters differently in a sentence ("Technology" ended with j alone and with a hard g in its
    sentence), and a sound it can make alone is the one a spelling can be blamed for losing.  Neighbouring words' sounds in the window, and
    sounds the voice never makes for this spelling, are not counted.

    ``in_sentence`` is one decode of the word's audio or several (the same audio cut with different margins): the recogniser is unsure of a
    borderline sound (a half-voiced affricate came out as j in six decodes and ch in one), so a sound is lost only when most decodes lack it."""
    readings = _readings(in_sentence)
    wanted, can = _consonants(speaker), _consonants(alone)
    return [sound for sound in dict.fromkeys(_base(p) for p in alone if not _is_vowel(p))
            if _same_sound(sound) in wanted and 2 * sum(1 for r in readings if _consonants(r)[_same_sound(sound)] >= min(wanted[_same_sound(sound)], can[_same_sound(sound)])) < len(readings)]


def kept_in_the_sentence(speaker: Sequence[str], in_sentence: Sequence[Any]) -> list[str]:
    """The speaker's consonants that most decodes of the word's audio in its sentence have."""
    readings = _readings(in_sentence)
    return [sound for sound in dict.fromkeys(_base(p) for p in speaker if not _is_vowel(p))
            if 2 * sum(1 for r in readings if _consonants(r)[_same_sound(sound)] >= 1) >= len(readings)]


def _stem(word: str) -> str:
    word = re.sub(r"[^a-z]", "", word.casefold())
    return word[:-1] if word.endswith("s") and len(word) > 4 else word


def candidate_words(texts: Iterable[str], source_words: Sequence[Mapping[str, Any]], *, looks_isizulu: Callable[[str], bool],
                    exclude: set[str] = frozenset(), limit: int = 30, named: Sequence[Mapping[str, str]] = ()) -> list[dict[str, Any]]:
    """English words that the Zulu text keeps and that the speaker says in the source audio.

    A token counts when its stem (after any isiZulu prefix glued on, with a hyphen or straight onto a capital: "weScience") is a word of the
    English transcript.  A glued prefix proves the writer meant an English word, so it is kept even when the word happens to end in a vowel and
    so looks isiZulu ("Science").  Names already given a reading (``exclude``) are skipped.

    ``named`` is what a model read out of the Zulu text itself ({"written": "iScience", "english": "science"}): it finds the English words
    by meaning, whatever their spelling or prefix.  Only the speaker's side (the transcript) is still matched here, because the lab needs
    the speaker's own sounds for the word."""
    texts = list(texts)
    zulu_by_stem = {_stem(str(item.get("english") or "")): str(item.get("zulu") or "").strip() for item in named
                    if str(item.get("kind") or "").lower() != "name" and str(item.get("zulu") or "").strip()}
    name_stems = {_stem(str(item.get("english") or "")) for item in named if str(item.get("kind") or "").lower() == "name"} - {""}
    name_written = {str(item.get("written") or "").casefold() for item in named if str(item.get("kind") or "").lower() == "name"}
    spoken: dict[str, dict[str, Any]] = {}
    spelt_alike = {re.sub(r"[^a-z']", "", str(w.get("text") or "").casefold()) for w in source_words} - {""}     # words as the transcript spells them
    # the Zulu-shaped words that are names or another language's words, not Zulu: a word the transcript restorer corrected to a name, or one the model
    # read as a name in the Zulu text.  "Imali" is Zulu (the speaker code-switches) and the Zulu voice says it right: it is not offered.
    name_words = {re.sub(r"[^a-z']", "", w.casefold()) for item in named if str(item.get("kind") or "").lower() == "name"
                  for field in ("written", "english") for w in re.split(r"[\s\-]+", str(item.get(field) or ""))}
    name_words |= {re.sub(r"[^a-z']", "", str(w.get("text") or "").casefold()) for w in source_words if w.get("heard_as")}
    name_words -= {""}
    for word in source_words:
        text = str(word.get("text") or "")
        stem = _stem(text)
        if len(stem) >= 4:
            spoken.setdefault(stem, {"as_spoken": re.sub(r"[^A-Za-z']", "", text), "count": 0})["count"] += 1
    found: dict[str, dict[str, Any]] = {}
    for text in texts:
        for token in re.findall(r"[A-Za-z][A-Za-z'\-]*", text):
            written = token.split("-")[-1]
            unglued = _GLUED_PREFIX.sub("", written)
            for variant, glued in ((written, None), (unglued, "prefix") if unglued != written else (written, None)):
                stem = _stem(variant)
                if stem in name_stems or written.casefold() in name_written:
                    continue                                 # the model says it is a name: not an English word to listen for
                carried = looks_isizulu(variant) and not glued and variant[:1].isupper() and re.sub(r"[^a-z']", "", variant.casefold()) in (spelt_alike & name_words)     # capitalised: a name
                if len(stem) < 4 or stem not in spoken or (looks_isizulu(variant) and not glued and not carried) or written.casefold() in exclude or variant.casefold() in exclude:
                    continue
                entry = found.setdefault(written, {
                    "word": written, "occurrences_in_dub": 0, "speaker_says": spoken[stem]["as_spoken"],
                    "times_the_speaker_says_it": spoken[stem]["count"],
                    "note": ("a Zulu prefix is glued on: the speaker says the English word as it is" if glued else
                             "written the same in the speaker's transcript although it is shaped like isiZulu: a name or a word of another language "
                             "(Sesotho, Afrikaans); the voice reads it with Zulu letter rules, so compare the speaker's sounds" if carried else "")})
                entry["occurrences_in_dub"] += 1
                if zulu_by_stem.get(stem):
                    entry["zulu_word"] = zulu_by_stem[stem]
                break
    for item in named:
        written, english = str(item.get("written") or "").strip(), str(item.get("english") or "").strip()
        if str(item.get("kind") or "").lower() == "name":
            continue                                         # names go through the names pipeline; an English listener cannot judge them
        stem = _stem(english)
        if len(stem) < 4 or stem not in spoken or written in found or written.casefold() in exclude or english.casefold() in exclude:
            continue
        count = sum(text.count(written) for text in texts)
        if count:
            found[written] = {"word": written, "occurrences_in_dub": count, "speaker_says": spoken[stem]["as_spoken"],
                              "times_the_speaker_says_it": spoken[stem]["count"], **({"zulu_word": zulu_by_stem[stem]} if zulu_by_stem.get(stem) else {}),
                              "note": "a Zulu prefix is glued on: the speaker says the English word as it is" if written.casefold() != english.casefold() else ""}
    return _one_entry_per_word(found.values())[:limit]


def _one_entry_per_word(entries: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """The same English word written several ways ("benefit", "ama-benefit", "inama-benefit") is one thing to check, not three: the
    lab compares the speaker's word with the voice's, whichever prefix the Zulu put on it.  The most common form leads."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        groups.setdefault(_stem(entry["speaker_says"]), []).append(entry)
    merged = []
    for group in groups.values():
        group.sort(key=lambda e: (-e["occurrences_in_dub"], e["word"]))
        lead = dict(group[0])
        lead["occurrences_in_dub"] = sum(e["occurrences_in_dub"] for e in group)
        if len(group) > 1:
            lead["also_written"] = [e["word"] for e in group[1:]]
        lead["note"] = next((e["note"] for e in group if e["note"]), "")
        if not lead.get("zulu_word"):
            lead.pop("zulu_word", None)
            for member in group:
                if member.get("zulu_word"):
                    lead["zulu_word"] = member["zulu_word"]
                    break
        merged.append(lead)
    return sorted(merged, key=lambda e: (-e["occurrences_in_dub"], e["word"]))


GOOD_ENOUGH = 0.85


def who_keeps_each_sound(tried: Mapping[str, Mapping[str, Any]], speaker: Sequence[str]) -> str:
    """For each consonant of the speaker that some spelling keeps in the sentence and another loses: which keep it and which lose it.  The
    letters that keep a sound in one spelling can be put together with the letters that keep another; this only shows where each sound came
    from, it does not write the spelling."""
    rows = [r for r in tried.values() if "kept_in_its_sentence" in r]
    if len(rows) < 2:
        return ""
    makes = set().union(*({_base(p) for p in r["voice_sounds"].split() if not _is_vowel(p)} | set(r["kept_in_its_sentence"]) for r in rows))
    notes = []
    for sound in dict.fromkeys(_base(p) for p in speaker if not _is_vowel(p)):
        if sound not in makes:
            continue
        keep = [repr(r["spelling"]) for r in rows if sound in r["kept_in_its_sentence"]]
        lose = [repr(r["spelling"]) for r in rows if sound not in r["kept_in_its_sentence"]]
        if keep and lose:
            notes.append(f"{sound}: kept by {', '.join(keep)}; lost by {', '.join(lose)}")
    if not notes:
        return ""
    return ("In the sentence, the speaker's sounds that only some spellings keep: " + " | ".join(notes)
            + ". Put the letters that keep each sound together in one spelling.")


def _advice(tried: Mapping[str, Mapping[str, Any]], best: Mapping[str, Any], left_word: int, left_clip: int, speaker: Sequence[str] = ()) -> str:
    """What to do next, from the tries so far.  It reads what an English listener wrote and the differences; it adds no score and does not
    choose for the agent."""
    heard = {record["spelling"]: record["english_hears"] for record in tried.values() if record.get("english_hears") is not None}
    understood = [record["spelling"] for record in tried.values() if record.get("english_listener_understands") and not record.get("lost_in_its_sentence")]
    parts = []
    lost = {record["spelling"]: record["lost_in_its_sentence"] for record in tried.values() if record.get("lost_in_its_sentence")}
    if lost:
        parts.append("In the sentence the word is spoken in, the voice loses a sound it makes when the word is alone: "
                     + "; ".join(f"{spelling!r} loses {' '.join(sounds)} (most of {tried[spelling].get('readings_of_the_audio', 1)} readings of its audio lack it; the tightest says {tried[spelling].get('said_in_its_sentence')!r})" for spelling, sounds in lost.items())
                     + ". A spelling is judged where it is heard, so one that loses a sound the speaker makes is not an answer; try letters that keep it.")
        keeping = who_keeps_each_sound(tried, speaker)
        if keeping:
            parts.append(keeping)
    if best.get("how_the_sounds_are_made_in_its_sentence"):
        parts.append(f"Measured in its sentence ({best['spelling']!r}; the median over several cuts of the same audio): "
                     + " | ".join(best["how_the_sounds_are_made_in_its_sentence"]) + ".")
    if heard:
        shown = "; ".join(f"{spelling!r} -> {text!r}" for spelling, text in list(heard.items())[-4:])
        if understood:
            top = max(understood, key=lambda spelling: tried[spelling]["closeness"])
            if best["spelling"] not in understood and best["closeness"] > tried[top]["closeness"]:
                parts.append(
                    f"The listener and the sounds point at different spellings. The listener wrote the word for {', '.join(repr(x) for x in understood)} "
                    f"(the closest of those in sounds: {top!r}, {tried[top]['closeness']}). The closest in sounds is {best['spelling']!r} "
                    f"({best['closeness']}), for which the listener wrote {heard.get(best['spelling'])!r}. What the listener wrote: {shown}. "
                    "Each check is blind somewhere: closeness skips the speaker's last sounds, so a missing ending can still score high, and the listener "
                    "hears one short clip with no sentence around it and guesses a word (a plural, a name or a technical term often comes out as another "
                    "real word). Compare the sounds and the measured differences of both before you choose.")
            else:
                parts.append(f"An English listener writes the word for {', '.join(repr(x) for x in understood)}. What the listener wrote: {shown}. "
                             "Use one of those; between them, the one whose sounds are closest.")
        elif not any(record.get("english_listener_understands") for record in tried.values()):
            if best["closeness"] >= GOOD_ENOUGH and not best.get("lost_in_its_sentence"):
                parts.append(f"The sounds of {best['spelling']!r} match the speaker's (closeness {best['closeness']}). What an English listener wrote: {shown}. "
                             "The listener hears one short clip with no sentence around it and guesses a word from it (a plural, a name or a technical term often "
                             "comes out as a different real word), so weigh the two: the sounds say it matches, the listener says it did not write the word.")
            else:
                parts.append(f"No spelling is understood yet. What an English listener wrote: {shown}. Change the sounds that make them hear the wrong word.")
    elif best["closeness"] >= GOOD_ENOUGH:
        return " ".join(parts + [f"{best['spelling']!r} is close to the speaker's sounds; use it."])
    distinct = {record["voice_sounds"] for record in tried.values()}
    if len(distinct) < len(tried):
        parts.append(f"{len(tried)} spellings gave only {len(distinct)} different sound(s): spellings that sound the same count as one idea, "
                     "so change the letters behind the differences, not the letters around them.")
    if best.get("how_the_sounds_are_made_differently"):
        parts.append(f"Measured in the closest ({best['spelling']!r}): " + " | ".join(best["how_the_sounds_are_made_differently"]) + ".")
    if not understood and best.get("differences"):
        parts.append(f"Still different in the closest ({best['spelling']!r}, {best['closeness']}): " + "; ".join(best["differences"]) + ".")
    if not understood and min(left_word, left_clip) > 0:
        parts.append(f"{left_word} tries left for this word, {left_clip} in the clip: try spellings that change those sounds before settling.")
    return " ".join(parts)


def _letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", str(text).casefold())


def understood_as(heard: str, word: str) -> bool:
    """Whether what an English listener wrote is the word (spacing, case, punctuation and a plural -s do not matter)."""
    heard_letters, word_letters = _letters(heard), _letters(word)
    return bool(word_letters) and (heard_letters == word_letters or heard_letters.rstrip("s") == word_letters.rstrip("s"))


def failing_readings(lab: Any, entries: Iterable[Mapping[str, Any]]) -> set[str]:
    """Display texts (casefolded) of saved English-word readings that an English listener does not understand, or that lose a sound in the
    sentence they are spoken in.

    Every saved reading of an English word is re-checked, whoever chose it (a person's ear included: the agent has audio tools that hear more
    than an ear does, and a pick made once should not outlast a better measurement); a name or a Zulu word is not an English word to listen
    for.  A reading that cannot be checked counts as fine.  The ones returned go back on the agent's list, so a run corrects a bad earlier pick
    instead of trusting it for ever."""
    failing: set[str] = set()
    for entry in entries:
        if entry.get("kind") == "english_code_switch" and not entry.get("disabled"):
            try:
                word, spelling = str(entry["display_text"]), str(entry["tts_text"])
                broken = lab.reading_fails(word, spelling) if hasattr(lab, "reading_fails") else lab.reading_understood(word, spelling) is False
                if broken:
                    failing.add(word.casefold())
            except Exception:
                continue
    return failing


class PronunciationLab:
    """The sounds of a word as the speaker said it, and as the voice says a spelling. One instance per clip; models load on first use."""

    def __init__(self, job_root: Path, *, voice: str, key: str, region: str, work_dir: Path, gop: Any = None, stt: Any = None) -> None:
        self.job_root, self.voice, self.key, self.region, self.work_dir = Path(job_root), voice, key, region, Path(work_dir)
        self._gop = gop
        self._stt = stt                                  # an Azure speech-to-text backend; made on first use when None
        self._voice_audio: dict[str, Path] = {}
        self._ending: dict[str, dict[str, Any] | None] = {}                # spelling -> does its word end with a vowel (the signal), when the voice says it alone
        self._in_sentence_ending: dict[tuple[str, str, str], dict[str, Any] | None] = {}
        self._created: dict[str, set[str]] = {}                          # word -> the Zulu forms create_loanword made for it in this run
        self._word_end: dict[str, float] = {}             # audio file -> second (in that file) where the spelling's text ends; later sounds are the next word's
        self._reading_checks: dict[tuple[str, str], bool | None] = {}
        self.sentences: list[str] = []                    # the Zulu the agent has written: a word is also heard in the sentence it is spoken in
        self.readings: dict[str, str] = {}                # readings already chosen for other words, applied in those sentences as the render does
        self._in_sentence: dict[tuple[str, str, str], list[list[str]]] = {}
        self._in_sentence_acoustics: dict[tuple[str, str, str], list[list[dict[str, Any]]]] = {}
        self._windows: Any = None
        self._transcript: list[Mapping[str, Any]] | None = None
        self.tries_per_word: dict[str, int] = {}
        self.tried: dict[str, dict[str, dict[str, Any]]] = {}
        self._source_cache: dict[str, dict[str, Any]] = {}

    @property
    def gop(self) -> Any:
        if self._gop is None:
            from .pronunciation_assess import LocalGop

            self._gop = LocalGop()
        self._gop._load()
        return self._gop

    @property
    def transcript(self) -> list[Mapping[str, Any]]:
        if self._transcript is None:
            path = self.job_root / "analysis" / "azure_diarization.json"
            words = json.loads(path.read_text(encoding="utf-8")).get("words") or [] if path.is_file() else []
            restored = self.job_root / "native_dub" / "pass1_restored_transcript.json"
            try:
                corrections = json.loads(restored.read_text(encoding="utf-8")).get("corrections") or [] if restored.is_file() else []
                words = restore_words(words, corrections)
            except (OSError, ValueError):                       # the restorer's file is an extra; without it the words are as heard
                pass
            self._transcript = words
        return self._transcript

    @property
    def total_tries(self) -> int:
        return sum(self.tries_per_word.values())

    def source(self, word: str) -> dict[str, Any]:
        """The sounds the speaker used for ``word`` (consensus over its occurrences in the source audio)."""
        spoken_stems = {_stem(str(w.get("text"))): str(w["text"]) for w in self.transcript}
        stem = next((c for c in (_stem(word), _stem(without_prefix(word.split("-")[-1], written_prefix(word)))) if c in spoken_stems), _stem(word))
        if stem not in self._source_cache:
            from .pronunciation_evidence import heard_word

            spoken_form = spoken_stems.get(stem, word)
            heard = heard_word(self.job_root / "audio" / "analysis_mono.wav", self.transcript, re.sub(r"[^A-Za-z']", "", spoken_form), self.gop,
                               self.work_dir / "source", limit=4)
            self._source_cache[stem] = ({"found": True, "phones": heard["phones"], "occurrences": len(heard["spans"]), "spoken_form": spoken_form,
                                         "each_occurrence": [" ".join(d) for d in heard["decoded"]],
                                         "_timed": heard.get("timed"), "_clips": heard.get("clips")} if heard else {"found": False, "phones": []})
        return self._source_cache[stem]

    def _audio_for(self, spelling: str) -> Path:
        """The voice saying ``spelling`` alone, as rendered (no pitch change)."""
        from . import pronunciation_search as ps

        if self._windows is None:
            self._windows = ps.NameWindows(ps.bookmark_synthesizer(self.voice, self.key, self.region), lambda wav: (), self.work_dir / "voice")
        # the word alone, but followed by a word: said as the last thing before silence it loses or devoices its final vowel ("Saensa" came out as
        # "saens"), which made every spelling ending in a vowel look closer to the speaker than it sounds.  The next word's sounds are cut off by time.
        context, candidate = ps.Context(before="", after=STANDALONE_AFTER, pitch="+0%"), ps.Candidate(spelling)
        audio = Path(self._windows.audio(context, candidate))
        try:
            marks = json.loads((self._windows.work_dir / f"{self._windows._key(context, candidate)}.json").read_text(encoding="utf-8"))
            self._word_end[str(audio)] = (float(marks["n1"]) - float(marks["n0"]) + ps.LEAD_MS) / 1000
            from . import phone_acoustics as pa

            whole = self._windows.work_dir / f"{self._windows._key(context, candidate)}.wav"
            self._ending[spelling] = pa.final_vowel(pa.load(whole), float(marks["n0"]) / 1000, float(marks["n1"]) / 1000 + 0.02)
        except (OSError, ValueError, KeyError):
            pass
        self._voice_audio[spelling] = audio
        return audio

    def _word_timed(self, audio: Path) -> list[tuple[str, float, float]]:
        """The sounds of the voice's audio for a spelling, without the sounds of the word that follows it."""
        timed = self.gop.decode_timed(Path(audio))
        end = self._word_end.get(str(audio))
        return timed if end is None else [heard for heard in timed if heard[1] <= end]

    def _apply_readings(self, text: str, skip: str) -> str:
        text = re.sub(r"~[a-z]+\d*\s*", "", text)          # ~pause800, ~en: markers for the renderer, not words to speak
        for display, spoken in self.readings.items():
            if _stem(display) != _stem(skip):
                text = _word_pattern(display).sub(lambda m, spoken=spoken: m.group(1) + spoken, text)
        return text

    def _located(self, word: str) -> tuple[Any, str]:
        """The sentence context of the word (see ``_context``) and the Zulu prefix the sentence has on it ("" when none)."""
        from . import pronunciation_search as ps

        pattern = _word_pattern(word)
        for text in self.sentences:
            pieces = re.split(r"(?<=[.?!])\s+", text)
            for index, piece in enumerate(pieces):
                found = pattern.search(piece)
                if found:
                    after = piece[found.end(2):] + (" " + pieces[index + 1] if index + 1 < len(pieces) else "")
                    after = after[:200].rsplit(" ", 1)[0] if len(after) > 200 else after
                    return (ps.Context(before=escape(self._apply_readings(piece[:found.start(2)], word)),
                                       after=escape(self._apply_readings(after, word)), pitch="+0%"), found.group(1))
        return None, ""

    def _context(self, word: str) -> Any:
        """The first sentence of the Zulu text that has the word (and the one after it), as the context to hear a spelling in; None when no
        sentence has it.  The Zulu prefix the sentence has on the word stays in the context: a spelling replaces the English word only."""
        return self._located(word)[0]

    def sentence_shown(self, word: str) -> str | None:
        """The sentence a word is heard in, with the spelling's place marked, for the agent to read; None when no sentence has the word."""
        context = self._context(word)
        return f"{context.before}[the spelling]{context.after}".strip() if context else None

    def readings_in_sentence(self, word: str, spelling: str) -> list[list[str]] | None:
        """The sounds the voice makes for ``spelling`` of ``word`` in the sentence the word is spoken in, decoded from the same audio cut with
        several margins (see ``IN_SENTENCE_MARGINS_MS``); None when no sentence has the word."""
        from . import pronunciation_search as ps

        context = self._context(word)
        if context is None:
            return None
        key = (word.casefold(), spelling, f"{context.before}|{context.after}")
        if key not in self._in_sentence:
            if self._windows is None:
                self._windows = ps.NameWindows(ps.bookmark_synthesizer(self.voice, self.key, self.region), lambda wav: (), self.work_dir / "voice")
            cuts = self._windows.audio_with_margins(context, ps.Candidate(without_prefix(spelling, self._located(word)[1])), IN_SENTENCE_MARGINS_MS)
            timed = [self.gop.decode_timed(Path(cut)) for cut in cuts]
            self._in_sentence[key] = [[phone for phone, _, _ in heard] for heard in timed]
            try:
                from . import phone_acoustics as pa

                whole, marks = self._windows.whole(context, ps.Candidate(without_prefix(spelling, self._located(word)[1])))
                self._in_sentence_ending[key] = pa.final_vowel(pa.load(whole), float(marks["n0"]) / 1000, float(marks["n1"]) / 1000 + 0.02)
            except Exception:
                self._in_sentence_ending[key] = None
            try:
                from . import phone_acoustics as pa

                self._in_sentence_acoustics[key] = [pa.timed_features(pa.load(Path(cut)), heard) for cut, heard in zip(cuts, timed)]
            except Exception:                                   # a measuring failure leaves out the acoustics, never the sounds
                self._in_sentence_acoustics[key] = []
        return self._in_sentence[key]

    def ending_note(self, word: str, spelling: str, source: Mapping[str, Any]) -> str | None:
        """Where the voice's ending differs from the speaker's, in plain words: a final vowel the voice speaks that he does not (a Zulu voice says
        every written vowel in full), or the opposite.  From the signal after the last s / ts, in the word's sentence; None when it matches or cannot
        be judged (a word that does not end in a hiss)."""
        context = self._context(word)
        ending = None
        if context is not None:
            ending = self._in_sentence_ending.get((word.casefold(), spelling, f"{context.before}|{context.after}"))
        if ending is None:
            ending = self._ending.get(without_prefix(spelling, written_prefix(word)))
        phones = list(source.get("phones") or [])
        if not ending or not phones:
            return None
        speaker_vowel = _is_vowel(phones[-1])
        if ending["present"] and not speaker_vowel:
            return (f"the voice ends the word with a vowel of about {ending['ms']} ms after the last s (a Zulu voice says every written vowel in full), "
                    f"and the speaker's word ends in the consonant {phones[-1]}")
        if not ending["present"] and speaker_vowel:
            return f"the voice's word ends at the last s with no vowel after it, and the speaker's word ends in a short vowel ({phones[-1]})"
        return None

    def _speaker_features(self, source: Mapping[str, Any]) -> list[list[dict[str, Any]]]:
        from . import phone_acoustics as pa

        timed, clips = source.get("_timed"), source.get("_clips")
        if not timed or not clips:
            return []
        if "_speaker_features" not in source:
            source["_speaker_features"] = [pa.timed_features(pa.load(Path(clip)), occurrence) for clip, occurrence in zip(clips, timed) if occurrence]  # type: ignore[index]
        return source["_speaker_features"]                                                      # type: ignore[index]

    def acoustics_in_sentence(self, word: str, spelling: str, source: Mapping[str, Any]) -> list[str]:
        """How the stops and affricates of the voice's word in its sentence are made differently from the speaker's, measured on the audio in
        every cut and compared by the median (see ``phone_acoustics.differences_over_cuts``): the answer to "j or ch?" that the recogniser's
        label, which flips, cannot give."""
        from . import phone_acoustics as pa

        if self.readings_in_sentence(word, spelling) is None:
            return []
        context = self._context(word)
        cuts = self._in_sentence_acoustics.get((word.casefold(), spelling, f"{context.before}|{context.after}")) or []
        try:
            speaker = self._speaker_features(source)
            return pa.differences_over_cuts(speaker, cuts, voice_open_f1=self.voice_open_f1()) if speaker and cuts else []
        except Exception:
            return []

    def sounds_in_sentence(self, word: str, spelling: str) -> list[str] | None:
        """The tightest of those decodes, to show the agent what the voice says (the wider ones carry the neighbouring words' sounds)."""
        readings = self.readings_in_sentence(word, spelling)
        return None if readings is None else readings[0]

    def sounds_lost_in_sentence(self, word: str, spelling: str, speaker: Sequence[str], alone: Sequence[str]) -> list[str]:
        readings = self.readings_in_sentence(word, spelling)
        return [] if readings is None else lost_in_the_sentence(speaker, alone, readings)

    def reading_fails(self, word: str, spelling: str) -> bool:
        """Whether a saved reading should be looked at again: an English listener does not understand it, or the voice loses a sound of it in
        the sentence it is spoken in.  A reading that cannot be checked counts as fine."""
        if self.reading_understood(word, spelling) is False:
            return True
        source = self.source(word)
        if not source.get("found") or self._context(word) is None:
            return False
        return bool(self.sounds_lost_in_sentence(word, spelling, source["phones"], self.voice_reading(without_prefix(spelling, written_prefix(word)))))

    def voice_reading(self, spelling: str) -> list[str]:
        """The sounds the Zulu voice produces for ``spelling``, decoded from its own audio with the same recogniser."""
        phones = [phone for phone, _, _ in self._word_timed(self._audio_for(spelling))]
        ending = self._ending.get(spelling)
        if ending and ending["present"] and phones and not _is_vowel(phones[-1]):
            phones.append("ə")                      # the recogniser did not list a final vowel that the signal shows: it is in the sounds the voice makes
        return phones

    def reading_understood(self, word: str, spelling: str) -> bool | None:
        """Whether an English listener writes ``word`` when the voice says ``spelling``; None when that cannot be asked.  No sounds are decoded
        and no try is spent: this re-checks a reading that was saved earlier."""
        key = (word.casefold(), spelling)
        if key not in self._reading_checks:
            spelling = without_prefix(spelling, written_prefix(word))        # an English listener hears the English word, not the Zulu "ma"
            self._audio_for(spelling)
            heard = self.english_hears(spelling)
            self._reading_checks[key] = None if heard is None else understood_as(heard, word)
        return self._reading_checks[key]

    def sound_differences(self, source: Mapping[str, Any], spelling: str) -> list[str]:
        """How the stops of the voice's word are made differently from the speaker's, measured from both audios (release, closure, vocal-fold
        vibration, voicing delay, hiss).  The phone recogniser gives two different sounds the same label; these measures do not."""
        from . import phone_acoustics as pa

        timed, clips = source.get("_timed"), source.get("_clips")
        audio = self._voice_audio.get(spelling)
        if not timed or not clips or audio is None:
            return []
        try:
            if "_speaker_features" not in source:
                source["_speaker_features"] = [pa.timed_features(pa.load(Path(clip)), occurrence) for clip, occurrence in zip(clips, timed) if occurrence]
            voice = pa.timed_features(pa.load(Path(audio)), self._word_timed(audio))
            return pa.differences(source["_speaker_features"], voice)
        except Exception:                                       # a measuring failure leaves out the extra information, never the try
            return []

    def zulu_hears(self, spelling: str) -> str | None:
        """What a Zulu listener (Azure speech-to-text, zu-ZA) writes for the voice saying ``spelling`` alone; None when it cannot be asked.  For a word
        made in Zulu: a listener that writes it back has understood it."""
        return self._hears(spelling, "zu-ZA")

    def english_hears(self, spelling: str) -> str | None:
        return self._hears(spelling, "en-ZA")

    def _hears(self, spelling: str, locale: str) -> str | None:
        """What an English listener (Azure speech-to-text, en-ZA) writes for the voice saying ``spelling`` alone; None when it cannot be asked.

        The recogniser is trained on real speech and knows which sound sequences are words, so it hears what a person would, where the phone
        recogniser only hears sounds.  It filters; it does not approve (a person's ear decides)."""
        audio = self._voice_audio.get(spelling)
        if audio is None or not Path(audio).is_file():
            return None
        try:
            if self._stt is None:
                from .cli import create_speech_backend
                from .config import load_settings

                self._stt = create_speech_backend(load_settings())
            from .azure_stt import normalize

            clip = self.work_dir / f"{locale}_ear" / f"{abs(hash(spelling))}.wav"
            clip.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(audio), "-ac", "1", "-ar", "16000", str(clip)], check=True)
            words = normalize(self._stt.transcribe(clip, locale=locale)).get("words") or []
            return " ".join(str(w.get("text")) for w in words).strip()
        except Exception as exc:                          # an empty transcription is an answer ("nothing"); a real failure is "could not ask"
            return "" if "empty transcription" in str(exc).lower() else None

    def _measure(self, word: str, spelling: str, source: Mapping[str, Any]) -> dict[str, Any]:
        """Everything the lab measures about one spelling of ``word`` against the speaker: sounds, how the stops are made, what an English
        listener writes, and what the voice does with it in its sentence.  Spends no try."""
        judged = without_prefix(spelling, written_prefix(word))             # the Zulu prefix is spoken, but it is not what is compared with the speaker
        heard = self.voice_reading(judged)
        record: dict[str, Any] = {"spelling": spelling, "voice_sounds": " ".join(heard), **align(source["phones"], heard)}
        lost_end = final_consonant_lost(source.get("each_occurrence") or [], heard)
        if lost_end:
            record["differences"] = [lost_end, *record["differences"]][:6]
        notes = self.sound_differences(source, judged)
        if notes:
            record["how_the_sounds_are_made_differently"] = notes
        listener = self.english_hears(judged)
        if listener is not None:
            record["english_hears"] = listener or "(nothing)"
            record["english_listener_understands"] = understood_as(listener, source.get("spoken_form") or word)
        readings = self.readings_in_sentence(word, spelling)
        if readings is not None:
            record["said_in_its_sentence"] = " ".join(_readings(readings)[0])
            record["kept_in_its_sentence"] = kept_in_the_sentence(source["phones"], readings)
            lost = lost_in_the_sentence(source["phones"], heard, readings)
            if lost:
                record["lost_in_its_sentence"] = lost
                record["readings_of_the_audio"] = len(_readings(readings))
            made = self.acoustics_in_sentence(word, spelling, source)
            if made:
                record["how_the_sounds_are_made_in_its_sentence"] = made
        note = self.ending_note(word, spelling, source)
        if note:
            record["ending"] = note
        record["_heard"] = heard
        return record

    def units_with(self, word: str, item_ids: Sequence[str] = ()) -> list[str]:
        """Which of the Zulu units say the word (by their ids, in the order of the text given to the lab)."""
        pattern = _word_pattern(word)
        ids = list(item_ids) or [str(index + 1) for index in range(len(self.sentences))]
        return [ids[index] for index, text in enumerate(self.sentences) if index < len(ids) and pattern.search(text)]

    def as_written(self, word: str, spelling: str | None = None) -> dict[str, Any] | None:
        """The same measurements for the word exactly as the Zulu text writes it (or, given ``spelling``, for a reading saved for it), so that
        "does it need a look?" is answered from evidence.  Spends no try; None when the speaker never says the word."""
        source = self.source(word)
        if not source.get("found"):
            return None
        record = self._measure(word, spelling or word, source)
        record.pop("_heard", None)
        return {"word": word, "spelling": spelling, "speaker_sounds": " ".join(source["phones"]), "voice_sounds": record["voice_sounds"], "closeness": record["closeness"],
                "differences": record["differences"][:3], "ending": record.get("ending"), "english_hears": record.get("english_hears"),
                "english_listener_understands": record.get("english_listener_understands"),
                "lost_in_its_sentence": record.get("lost_in_its_sentence"), "made_differently": (record.get("how_the_sounds_are_made_in_its_sentence") or record.get("how_the_sounds_are_made_differently") or [])[:2]}

    def _letter_table(self) -> dict[str, Any]:
        """Which sounds this voice makes for which letters (see ``letter_sounds``): built from its own audio once and kept with the shared memory."""
        from . import letter_sounds

        if getattr(self, "_letters", None) is None:
            try:
                path = Path(self.job_root).parents[1] / "experience" / f"letter_sounds_{self.voice}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
            except (IndexError, OSError):
                path = self.work_dir / f"letter_sounds_{self.voice}.json"
            self._letters = letter_sounds.load_or_build(path, self.voice_reading, voice=self.voice)
        return self._letters

    def voice_open_f1(self) -> float | None:
        """How open the mouth is when this voice says a plain "a" (F1 in Hz): the scale the opening of any vowel is read against."""
        if getattr(self, "_open_reference", "unset") == "unset":
            readings = [self.first_vowel_opening(word) for word in ("baba", "mama")]
            readings = [value for value in readings if value]
            self._open_reference = float(sorted(readings)[len(readings) // 2]) if readings else None
        return self._open_reference

    def first_vowel_opening(self, spelling: str) -> float | None:
        """How open the mouth is at the start of the first vowel when the voice says ``spelling`` alone (F1 in Hz); None when not measurable."""
        from . import phone_acoustics as pa

        try:
            audio = Path(self._audio_for(spelling))
            features = pa.timed_features(pa.load(audio), self._word_timed(audio))
        except Exception:
            return None
        return next((f["onset_f1"] for f in features if "onset_f1" in f), None)

    def _speaker_opening(self, source: Mapping[str, Any]) -> float | None:
        """The speaker's mouth opening at the start of his first vowel of the word (median over his occurrences)."""
        try:
            readings = [next((f["onset_f1"] for f in occurrence if "onset_f1" in f), None) for occurrence in self._speaker_features(source)]
        except Exception:
            return None
        readings = [value for value in readings if value is not None]
        return float(sorted(readings)[len(readings) // 2]) if readings else None

    def create_loanword(self, word: str) -> dict[str, Any]:
        """A Zulu word made for an English word with none, the way Zulu adapts loans (see ``loan_adapt``), from the speaker's own sounds: the forms
        are spoken by the Zulu voice and measured.  Counts as one try.  Nothing here is scored against the speaker's English sounds: a loanword
        differs from them by design (isayensi is not ``s aɪ ə n s``), so those figures are shown for information only."""
        from difflib import SequenceMatcher

        from . import loan_adapt

        source = self.source(word)
        if not source["found"]:
            return {"word": word, "error": "the speaker never says this word in the source audio, so there are no sounds to build a Zulu word from"}
        if self.total_tries >= MAX_TRIES_PER_CLIP or self.tries_per_word.get(word, 0) >= MAX_TRIES_PER_WORD:
            raise PronunciationBudgetExhausted(
                f"pronunciation tries used: {self.tries_per_word.get(word, 0)}/{MAX_TRIES_PER_WORD} for {word!r}, {self.total_tries}/{MAX_TRIES_PER_CLIP} in all")
        self.tries_per_word[word] = self.tries_per_word.get(word, 0) + 1
        candidates = loan_adapt.adapt(source["phones"], limit=LOAN_POOL)
        rows, seen = [], set()
        for candidate in candidates:
            heard = self.voice_reading(candidate["stem"])
            if tuple(heard) in seen:
                continue                                                         # another way of writing sounds already shown
            seen.add(tuple(heard))
            listener = self.zulu_hears(candidate["stem"])
            rows.append({"zulu_word": candidate["stem"], "with_i": candidate["with_i"], "syllables": candidate["syllables"],
                         "changes_from_the_most_typical": candidate["choices"], "voice_sounds": " ".join(heard),
                         "a_zulu_listener_writes": listener, "zulu_listener_similarity": None if listener is None else round(SequenceMatcher(None, _token(listener), _token(candidate["stem"])).ratio(), 2),
                         "closeness_to_the_speakers_english": align(source["phones"], heard)["closeness"]})
            self._created.setdefault(word, set()).add(candidate["stem"])
            if len(rows) >= LOAN_SHOWN:
                break
        return {"word": word, "speaker_sounds": " ".join(source["phones"]), "zulu_words": rows,
                "how_these_were_made": "the speaker's sounds rebuilt in Zulu syllables: English p/t/k/b written ph/th/kh/bh, vowels reduced to a e i o u, a vowel "
                                       "inserted where Zulu has no cluster (u after a labial or velar, i after the others) and at the end of the word; the first "
                                       "is the most typical, the others change one choice each (see changes_from_the_most_typical)",
                "next": "choose one, write it in the Zulu text where the English word stands, with the noun prefix the sentence needs (i-, isi-, ama-, u-), "
                        "then call use_loanword so later clips use the same word",
                "tries_left_for_word": MAX_TRIES_PER_WORD - self.tries_per_word[word], "tries_left_in_clip": MAX_TRIES_PER_CLIP - self.total_tries}

    def suggest_spellings(self, word: str) -> dict[str, Any]:
        """Spellings of ``word`` worked out backwards from the speaker's sounds and the letters this voice turns into those sounds, spoken and
        measured like any tried spelling (they can be used with ``use_pronunciation``).  Counts as one try: it replaces guessing from English
        spelling with a search the agent could not do by hand, and the agent still decides."""
        from . import letter_sounds

        source = self.source(word)
        if not source["found"]:
            return {"word": word, "error": "the speaker never says this word in the source audio, so there is nothing to match"}
        if self.total_tries >= MAX_TRIES_PER_CLIP or self.tries_per_word.get(word, 0) >= MAX_TRIES_PER_WORD:
            raise PronunciationBudgetExhausted(
                f"pronunciation tries used: {self.tries_per_word.get(word, 0)}/{MAX_TRIES_PER_WORD} for {word!r}, {self.total_tries}/{MAX_TRIES_PER_CLIP} in all")
        self.tries_per_word[word] = self.tries_per_word.get(word, 0) + 1
        prefix = written_prefix(word)
        stem = without_prefix(word.split("-")[-1], prefix)
        capital = stem[:1].isupper()
        table = self._letter_table()
        from itertools import zip_longest

        def styled(spelling: str) -> str:
            return spelling.capitalize() if capital else spelling

        spoken: dict[str, tuple[float, tuple[str, ...]]] = {}          # spelling -> (closeness, sounds): everything spoken once

        def speak(spellings: Iterable[str]) -> None:
            for spelling in spellings:
                spelling = styled(spelling.lower())
                if spelling and spelling not in spoken:
                    heard = tuple(self.voice_reading(spelling))
                    spoken[spelling] = (align(source["phones"], list(heard))["closeness"], heard)

        # seeds: spellings worked out backwards from the speaker's sounds, the word as it is, and the best of what has been tried for it
        tried_best = sorted(self.tried.get(word, {}).values(), key=lambda r: -r["closeness"])[:3]
        speak([spelling for spelling, _ in letter_sounds.candidates(source["phones"], table, limit=SUGGEST_FIRST)] + [stem]
              + [without_prefix(r["spelling"], prefix) for r in tried_best])
        for _ in range(SUGGEST_ROUNDS):                                # then the best so far are varied by small orthographic edits and spoken
            leaders = sorted(spoken, key=lambda spelling: -spoken[spelling][0])[:4]
            lists = [letter_sounds.neighbours(leader, table) for leader in leaders]
            speak([edit for group in zip_longest(*lists) for edit in group if edit][:SUGGEST_PER_ROUND])         # one edit of each leader in turn
        pool = list(spoken)
        by_sounds: dict[tuple[str, ...], tuple[float, str]] = {}
        for spelling, (closeness, heard) in spoken.items():
            if heard not in by_sounds or (closeness, -len(spelling)) > (by_sounds[heard][0], -len(by_sounds[heard][1])):
                by_sounds[heard] = (closeness, spelling)
        ranked = sorted(by_sounds.values(), key=lambda item: (-item[0], len(item[1]), item[1]))
        best = ranked[:SUGGEST_SHOWN - SUGGEST_BY_OPENING]
        # a portfolio, not one score: the closest in sounds (closeness is nearly blind to the quality of a vowel), and the closest in how open the
        # mouth is at the start of the first vowel, which a person hears at once ("Seinsa" starts nearly shut where the speaker's mouth is wide open)
        wanted = self._speaker_opening(source)
        if wanted is not None:
            # every spelling spoken is a candidate here, even one that sounds like a better-scoring one: the same decoded sounds can open the mouth differently
            rest = sorted(((closeness, spelling) for spelling, (closeness, _) in spoken.items() if (closeness, spelling) not in best),
                          key=lambda item: (-item[0], len(item[1]), item[1]))
            opening = {item[1]: self.first_vowel_opening(item[1]) for item in rest[:16]}
            best += sorted((item for item in rest if opening.get(item[1]) is not None), key=lambda item: abs(opening[item[1]] - wanted))[:SUGGEST_BY_OPENING]
        best = (best + [item for item in ranked if item not in best])[:SUGGEST_SHOWN]
        results = []
        for _, spelling in best:
            full = prefix + spelling if prefix else spelling
            record = self._measure(word, full, source)
            record.pop("_heard", None)
            self.tried.setdefault(word, {})[full] = record
            results.append(record)
        left_word, left_clip = MAX_TRIES_PER_WORD - self.tries_per_word[word], MAX_TRIES_PER_CLIP - self.total_tries
        out: dict[str, Any] = {
            "word": word, "speaker_sounds": " ".join(source["phones"]), "speaker_occurrences": source["occurrences"],
            "how_these_were_found": f"spellings were worked out backwards from the speaker's sounds through the letters this voice turns into them, "
                                    f"then the best were varied by small edits (a letter group swapped, left out or added); {len(pool)} were spoken, "
                                    f"{len(by_sounds)} different sound sequences came out; shown are the closest in sounds and the closest in how open the mouth is "
                                    "at the start of the first vowel",
            "results": results, "tries_left_for_word": left_word, "tries_left_in_clip": left_clip}
        if results:
            nearest = max(results, key=lambda r: (not r.get("lost_in_its_sentence"), r["closeness"]))
            out["closest_so_far"] = {"spelling": nearest["spelling"], "closeness": nearest["closeness"]}
            out["understood_by_the_listener"] = [r["spelling"] for r in results if r.get("english_listener_understands") and not r.get("lost_in_its_sentence")]
            out["advice"] = _advice(self.tried[word], nearest, left_word, left_clip, speaker=source["phones"])
        shown = self.sentence_shown(word)
        if shown:
            out["also_heard_in_this_sentence"] = shown
        return out

    def try_spellings(self, word: str, spellings: Sequence[str]) -> dict[str, Any]:
        """Try up to three spellings of ``word``; each is compared with the speaker's sounds. Counts against the budget."""
        spellings = [s.strip() for s in spellings if str(s).strip()][:MAX_SPELLINGS_PER_CALL]
        if not spellings:
            raise ValueError("give at least one spelling to try")
        source = self.source(word)
        if not source["found"]:
            return {"word": word, "error": "the speaker never says this word in the source audio, so there is nothing to match"}
        results = []
        for spelling in spellings:
            if self.total_tries >= MAX_TRIES_PER_CLIP or self.tries_per_word.get(word, 0) >= MAX_TRIES_PER_WORD:
                raise PronunciationBudgetExhausted(
                    f"pronunciation tries used: {self.tries_per_word.get(word, 0)}/{MAX_TRIES_PER_WORD} for {word!r}, {self.total_tries}/{MAX_TRIES_PER_CLIP} in all")
            self.tries_per_word[word] = self.tries_per_word.get(word, 0) + 1
            record = self._measure(word, spelling, source)
            heard = record.pop("_heard")
            twin = next((other for other, tried in self.tried.get(word, {}).items() if tried["voice_sounds"] == record["voice_sounds"]), None)
            if twin:
                record["same_sounds_as"] = twin                     # a different spelling, the same sounds: not a new idea
            self.tried.setdefault(word, {})[spelling] = record
            results.append(record)
        # "closest" is by sounds only; the spellings the English listener wrote the word for are listed beside it, not ranked above it:
        # closeness skips the speaker's last sounds and the listener guesses from one short clip, so neither check may decide alone
        best = max(self.tried[word].values(), key=lambda r: (not r.get("lost_in_its_sentence"), r["closeness"]))
        left_word, left_clip = MAX_TRIES_PER_WORD - self.tries_per_word[word], MAX_TRIES_PER_CLIP - self.total_tries
        out = {"word": word, "speaker_sounds": " ".join(source["phones"]), "speaker_occurrences": source["occurrences"], "results": results,
               "closest_so_far": {"spelling": best["spelling"], "closeness": best["closeness"]},
               "understood_by_the_listener": [r["spelling"] for r in self.tried[word].values()
                                              if r.get("english_listener_understands") and not r.get("lost_in_its_sentence")],
               "advice": _advice(self.tried[word], best, left_word, left_clip, speaker=source["phones"]),
               "tries_left_for_word": left_word, "tries_left_in_clip": left_clip}
        shown = self.sentence_shown(word)
        if shown:
            out["also_heard_in_this_sentence"] = shown
        return out
