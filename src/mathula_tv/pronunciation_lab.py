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
    name_stems = {_stem(str(item.get("english") or "")) for item in named if str(item.get("kind") or "").lower() == "name"} - {""}
    name_written = {str(item.get("written") or "").casefold() for item in named if str(item.get("kind") or "").lower() == "name"}
    spoken: dict[str, dict[str, Any]] = {}
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
                if len(stem) < 4 or stem not in spoken or (looks_isizulu(variant) and not glued) or written.casefold() in exclude or variant.casefold() in exclude:
                    continue
                entry = found.setdefault(written, {
                    "word": written, "occurrences_in_dub": 0, "speaker_says": spoken[stem]["as_spoken"],
                    "times_the_speaker_says_it": spoken[stem]["count"],
                    "note": "a Zulu prefix is glued on: the speaker says the English word as it is" if glued else ""})
                entry["occurrences_in_dub"] += 1
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
                              "times_the_speaker_says_it": spoken[stem]["count"],
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
    if heard:
        shown = "; ".join(f"{spelling!r} -> {text!r}" for spelling, text in list(heard.items())[-4:])
        if understood:
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
        self._reading_checks: dict[tuple[str, str], bool | None] = {}
        self.sentences: list[str] = []                    # the Zulu the agent has written: a word is also heard in the sentence it is spoken in
        self.readings: dict[str, str] = {}                # readings already chosen for other words, applied in those sentences as the render does
        self._in_sentence: dict[tuple[str, str, str], list[list[str]]] = {}
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
            self._transcript = json.loads(path.read_text(encoding="utf-8")).get("words") or [] if path.is_file() else []
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
        # the word alone, with nothing before or after it, so the neighbouring words' sounds do not leak into its window
        audio = Path(self._windows.audio(ps.Context(before="", after=".", pitch="+0%"), ps.Candidate(spelling)))
        self._voice_audio[spelling] = audio
        return audio

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
            self._in_sentence[key] = [list(self.gop.decode(Path(cut))) for cut in cuts]
        return self._in_sentence[key]

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
        return list(self.gop.decode(self._audio_for(spelling)))

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
            voice = pa.timed_features(pa.load(Path(audio)), self.gop.decode_timed(audio))
            return pa.differences(source["_speaker_features"], voice)
        except Exception:                                       # a measuring failure leaves out the extra information, never the try
            return []

    def english_hears(self, spelling: str) -> str | None:
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

            clip = self.work_dir / "english_ear" / f"{abs(hash(spelling))}.wav"
            clip.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(audio), "-ac", "1", "-ar", "16000", str(clip)], check=True)
            words = normalize(self._stt.transcribe(clip, locale="en-ZA")).get("words") or []
            return " ".join(str(w.get("text")) for w in words).strip()
        except Exception as exc:                          # an empty transcription is an answer ("nothing"); a real failure is "could not ask"
            return "" if "empty transcription" in str(exc).lower() else None

    def _measure(self, word: str, spelling: str, source: Mapping[str, Any]) -> dict[str, Any]:
        """Everything the lab measures about one spelling of ``word`` against the speaker: sounds, how the stops are made, what an English
        listener writes, and what the voice does with it in its sentence.  Spends no try."""
        judged = without_prefix(spelling, written_prefix(word))             # the Zulu prefix is spoken, but it is not what is compared with the speaker
        heard = self.voice_reading(judged)
        record: dict[str, Any] = {"spelling": spelling, "voice_sounds": " ".join(heard), **align(source["phones"], heard)}
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
        record["_heard"] = heard
        return record

    def as_written(self, word: str) -> dict[str, Any] | None:
        """The same measurements for the word exactly as the Zulu text writes it, so that "does it need a look?" is answered from evidence.
        Spends no try; None when the speaker never says the word."""
        source = self.source(word)
        if not source.get("found"):
            return None
        record = self._measure(word, word, source)
        record.pop("_heard", None)
        return {"word": word, "speaker_sounds": " ".join(source["phones"]), "voice_sounds": record["voice_sounds"], "closeness": record["closeness"],
                "differences": record["differences"][:3], "english_hears": record.get("english_hears"),
                "english_listener_understands": record.get("english_listener_understands"),
                "lost_in_its_sentence": record.get("lost_in_its_sentence"), "made_differently": (record.get("how_the_sounds_are_made_differently") or [])[:2]}

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
        best = max(self.tried[word].values(), key=lambda r: (bool(r.get("english_listener_understands")) and not r.get("lost_in_its_sentence"),
                                                            not r.get("lost_in_its_sentence"), r["closeness"]))
        left_word, left_clip = MAX_TRIES_PER_WORD - self.tries_per_word[word], MAX_TRIES_PER_CLIP - self.total_tries
        out = {"word": word, "speaker_sounds": " ".join(source["phones"]), "speaker_occurrences": source["occurrences"], "results": results,
               "closest_so_far": {"spelling": best["spelling"], "closeness": best["closeness"]},
               "advice": _advice(self.tried[word], best, left_word, left_clip, speaker=source["phones"]),
               "tries_left_for_word": left_word, "tries_left_in_clip": left_clip}
        shown = self.sentence_shown(word)
        if shown:
            out["also_heard_in_this_sentence"] = shown
        return out
