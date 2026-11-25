"""The model's reading of why a word the listener pressed sounds rushed or clipped: an opinion the listener asks for, kept nowhere.

The listener's ear is the measurement.  A word is rushed or clipped because the voice swallows a syllable of it or squeezes a stretch together, and nothing counted on the audio (pace, syllables, loudness
peaks, a recogniser's phones) told that apart from ordinary speech on the real dub.  So the Studio's Repair button sends the pressed word with its sentence and the next one to a model and shows what it
reads as the cause in the construction of the Zulu.  It stores nothing and changes no text; what the ear decides is recorded with the Studio's flag (console.studio_flag).
"""
from __future__ import annotations

from typing import Any

INVESTIGATE_PROMPT = """\
You advise the writer of isiZulu that a synthetic voice reads aloud in a news broadcast. The listener pressed one word in the dub and says the voice rushes it or clips it. You get the "sentence" that holds it, the
"next_sentence" that follows it in the same breath of the dub (the voice often fuses a sentence's end with the start of the next), the "word" and the "phrase" around it. A word sounds rushed when the voice
swallows a syllable of it or squeezes the syllables of a stretch together, and its pace in syllables a second can look perfectly normal, so do not reason from speed. Say what in the CONSTRUCTION is the cause,
as a feature of the Zulu and not of this word, looking at the junction with the next sentence as well: a run of names fused to a conjunction, a long chain of modifiers (verb-noun, adverb, adjective) taken for one
compound, heavy consonant clusters, nouns stacked with nothing saying how they relate, a concord that does not agree, a sentence that ends on a long word and a next one that starts on one. The audience is general
listeners and the reader is a synthetic voice, not an expert Zulu speaker: plain, speakable isiZulu. Give "rule": one sentence a writer can apply to any text, starting with "When", that does not mention these words;
"rewrite": the "sentence" rewritten so that it can be said, with the same meaning, every name, number and date unchanged (only that sentence, never the next); and "next_note": one line on what to change in the
next sentence's opening if it contributes, else "". Be honest: if the cause is not in the text, use "cause": "unclear" and an empty rule and rewrite.
Reply with JSON only: {"items":[{"id":"1","cause":"...","rule":"...","rewrite":"...","next_note":"..."}]}
"""


def investigate(client: Any, *, sentence: str, next_sentence: str, word: str, phrase: str) -> dict[str, str]:
    """The model's reading of why ONE word the listener pressed is rushed or clipped, with the sentence it is in and the next: {cause, rule, rewrite, next_note}.  An opinion: the listener decides."""
    from .dub_judge import _call

    payload = [{"id": "1", "sentence": sentence, "next_sentence": next_sentence, "word": word, "phrase": phrase}]
    answer = _call(client, INVESTIGATE_PROMPT, payload, {}).get("1") or {}
    return {key: str(answer.get(key) or "").strip() for key in ("cause", "rule", "rewrite", "next_note")}
