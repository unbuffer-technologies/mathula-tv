"""A second reference for a word the speaker's own audio cannot give: a standard South African English voice saying the corrected text.

The lab matches the Zulu voice's reading of a word to "the speaker's sounds", decoded by a phone recogniser from the stretch of audio the speech service timed as
the word.  That fails when the stretch is short and fast: in the Nonvalo interview "High" lasted 0.16 s inside "the Western Cape High Court" said in 1.2 s, the
service's times were right (a separate forced alignment agrees within 0.04 s), and the recogniser still read "p aː" from it.  A person understands the word from
the context; the decoder cannot.

For those words only, a clean rendering of the corrected phrase is decoded by the same recogniser, which puts the reference in the recogniser's own phone space
(its habits cancel when two readings are compared) and in a South African accent, and the target word's sounds are cut out with a forced alignment.  Decoded this
way: High "h aɪ", Western "w ɛ s t ə n", OK "ɔ k eɪ", Heidi "h aɪ d i", SG "e s dʒ i", where the speaker's audio gave "p aː", "ɪ s ə n", "k eː ð ɛ", "d iː" and
"s tʃ iː".  The speaker's own sounds remain the reference whenever they are usable; every output that shows a reference says which one it is.
"""
from __future__ import annotations

import re
from pathlib import Path
from statistics import median
from typing import Any, Callable, Mapping, Sequence

EN_VOICE = "en-ZA-LeahNeural"
SHORT_SPAN_SECONDS = 0.26       # a word the service times shorter than this (median over its occurrences) is too fast to decode on its own
CONTEXT_WORDS = 2               # words either side of the target that the phrase carries, so the voice says it as it is said in a sentence
SLACK_SECONDS = 0.06


def weakness(spans: Sequence[tuple[float, float]], warning: str | None, word: str = "", phones: Sequence[str] = ()) -> str | None:
    """Why the speaker's audio is not enough for this word, or None when it is."""
    if warning:
        return "the speaker's sounds do not fit the word's letters"
    letters = re.sub(r"[^A-Za-z]", "", word)
    if letters and phones is not None and (2 <= len(letters) <= 5 and letters.isupper()) and len(phones) < len(letters):
        return f"an initialism said letter by letter, but only {len(phones)} sound(s) were decoded for its {len(letters)} letters"       # AI came out as one sound
    if len(letters) >= 3 and phones is not None and 0 < len(phones) < 2:
        return "only one sound was decoded for it"
    durations = [end - start for start, end in spans if end > start]
    if durations and median(durations) < SHORT_SPAN_SECONDS:
        return f"the speaker says it in {median(durations):.2f} s, too fast to decode on its own"
    return None


def is_initialism(token: str) -> bool:
    clean = re.sub(r"[^\w']", "", token)
    return 2 <= len(clean) <= 5 and clean.isalpha() and clean.isupper()


def _spoken(token: str) -> str:
    """An initialism is said by the names of its letters (SSML say-as); with the letters spaced ("A I") or run together the voice said the AI of
    "concocted pictures AI pictures" as a single short vowel and the recogniser heard one sound."""
    clean = re.sub(r"[^\w']", "", token)
    return f'<say-as interpret-as="characters">{clean}</say-as>' if is_initialism(token) else clean


def phrase_around(transcript: Sequence[Mapping[str, Any]], word: str) -> tuple[list[str], int] | None:
    """(the words of the phrase around the word's longest occurrence, the target's position in it), or None when the transcript never has the word."""
    target = re.sub(r"[^a-z0-9']", "", str(word).casefold())
    best: tuple[float, int] | None = None
    for index, item in enumerate(transcript):
        if re.sub(r"[^a-z0-9']", "", str(item.get("text") or "").casefold()) == target:
            length = float(item.get("end") or 0) - float(item.get("start") or 0)
            if best is None or length > best[0]:
                best = (length, index)
    if best is None:
        return None
    first = max(0, best[1] - CONTEXT_WORDS)
    window = transcript[first: best[1] + CONTEXT_WORDS + 1]
    words = [str(w.get("text") or "") for w in window]
    return words, best[1] - first


def tts_reference(word: str, transcript: Sequence[Mapping[str, Any]], *, work: Path, gop: Any, listener: Any,
                  synthesize: Callable[[str, Path], Path], voice: str = EN_VOICE) -> dict[str, Any] | None:
    """The target word's sounds as ``voice`` says the corrected phrase around it, or None when it cannot be made (the caller keeps the speaker's)."""
    found = phrase_around(transcript, word)
    if found is None:
        return None
    words, position = found
    tokens: list[tuple[int, str]] = [(i, re.sub(r"[^\w']", "", w)) for i, w in enumerate(words) if re.sub(r"[^\w']", "", w)]
    text = " ".join(_spoken(words[i]) for i, _ in tokens)
    ssml = ('<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-ZA">'
            f'<voice name="{voice}">{text}</voice></speak>')
    work.mkdir(parents=True, exist_ok=True)
    wav = synthesize(ssml, work / (re.sub(r"[^A-Za-z0-9]", "", word) + "_en.wav"))
    rows = listener.align(wav, [t for _, t in tokens], 0.0, None)
    if len(rows) != len(tokens):
        return None
    own = [row for (owner, _), row in zip(tokens, rows) if owner == position and row.get("start") is not None]
    if not own:
        return None
    start, end = min(r["start"] for r in own), max(r["end"] for r in own)
    before = [row["end"] for (owner, _), row in zip(tokens, rows) if owner < position and row.get("end") is not None]
    after = [row["start"] for (owner, _), row in zip(tokens, rows) if owner > position and row.get("start") is not None]
    # the clean rendering has a gap between words, so the neighbours' own times bound the target: the letters the aligner follows end before the sounds do
    # (the "ay" of OK comes after the K), and a fixed slack cut it off
    # but the aligner's start for a word is where its first letter's sound begins, and the end of the word before can run past its last letter ("Cape" ends
    # in a p released after the aligner's end), so on the left the target keeps only a little slack and never goes before the neighbour's end
    # letter names are not the letters the aligner follows, so an initialism starts after the word before it; that word's last sound ("pictures" ends in a z)
    # is still sounding a few hundredths of a second after the aligner's end for it, hence the margin
    low = before[-1] + 0.08 if (before and is_initialism(words[position])) else max([start - SLACK_SECONDS, *before[-1:]])
    high = after[0] if after else end + 4 * SLACK_SECONDS
    timed = [(token, begin, finish) for token, begin, finish in gop.decode_timed(wav) if low <= (begin + finish) / 2 <= high]
    phones = [token for token, _, _ in timed]
    if not phones:
        return None
    return {"phones": phones, "timed": timed, "clip": wav, "voice": voice, "phrase": text, "span": (start, end)}
