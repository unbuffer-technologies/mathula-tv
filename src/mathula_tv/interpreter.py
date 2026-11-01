"""An English word to Zulu spellings, derived sound by sound and letter by letter, with the derivation kept.

The agent used to find a spelling by guessing and measuring.  This is the derivation it can start from, in the four steps a person would take:

1. ``readings``: the word's English sounds from the dictionary (``english_sounds``); a run of capitals is the names of its letters.
2. ``aim``: for each English sound, what Zulu speakers do with it (``phonology``, learned from people who say the same names in both languages)
   limited to the sounds this voice can make, so the aim is a sound the voice can say.
3. ``spellings``: the letters of isiZulu spelling that make the voice say the aim, in the order that keeps the sounds closest (``letter_sounds``,
   built from the voice's own audio); the ruler is ``phone_features``.
4. The caller speaks and measures them (the pronunciation lab); nothing here is judged without the voice saying it.

``table`` lists, for every English sound, the row a reader can check: what the sound is, what Zulu speakers say, what the voice can make and the letters
that make it.  A word with no dictionary reading and no capitals (a Zulu name such as Nkandla) has no reading here and is left as it is written.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from . import letter_sounds
from .english_sounds import EnglishSounds
from .phone_features import distance
from .phonology import corpus_symbol

AIM_MINIMUM_SHARE = 0.08          # a Zulu-speaker choice rarer than this is not an aim
SHARE_BONUS = 0.3                 # a sound all Zulu speakers choose is this much nearer than its distance on the chart says


def makeable(table: Mapping) -> set[str]:
    """Every sound some letter group of isiZulu spelling makes in this voice (the voice's inventory as the letter table saw it)."""
    sounds: set[str] = set()
    for value in (table.get("graphemes") or {}).values():
        for option in letter_sounds._options(value):
            sounds.update(option["sounds"])
    return sounds


class Interpreter:
    def __init__(self, sounds: EnglishSounds, table: Mapping, speakers: Any = None) -> None:
        self.sounds, self.table, self.speakers = sounds, table, speakers
        self._can = makeable(table) or set(letter_sounds.PRIOR)        # what the voice's own audio produced; the hand-written list only when there is none

    def readings(self, written: str) -> list[dict[str, Any]]:
        """The dictionary's readings; for a word it lacks that is not Zulu-shaped (Litchfield, not Nkandla), the guesser's, labelled ``guessed``."""
        from .pronunciation_search import looks_isizulu

        return self.sounds.say(written, guess=self.sounds.guesser is not None and not looks_isizulu(written))

    def nearest_makeable(self, phone: str) -> tuple[str, float]:
        """The sound the voice can make that is nearest to ``phone`` on the chart, and how far."""
        if phone in self._can:
            return phone, 0.0
        scored = [(away, other) for other in self._can if (away := distance(phone, other)) is not None]
        if not scored:
            return phone, 1.0
        away, nearest = min(scored)
        return nearest, away

    def aim_for(self, phone: str) -> dict[str, Any]:
        """What to aim at for one English sound.  The sound itself when the voice can make it: the goal is to sound like the speaker, and Zulu speakers
        reading written names partly follow the letters (they say the PRICE vowel of "High" as i a third of the time).  When the voice cannot make
        it, the nearest sound it has, with the Zulu speakers' choices weighed in: each costs its distance from the English sound less a bonus for
        how many of them chose it.  ``choices`` lists them all, best first."""
        saying = [o for o in (self.speakers.options(corpus_symbol(phone)) if self.speakers else []) if o["zulu"] and o["share"] >= AIM_MINIMUM_SHARE]
        made, away = self.nearest_makeable(phone)
        if away == 0.0:
            choices = [{"zulu_speakers_say": next((o["zulu"] for o in saying if o["zulu"] == made), None), "share": None, "voice_makes": made, "distance": 0.0}]
            return {"english": phone, "aim": made, "choices": choices, "why": "the voice makes this sound"}
        options: dict[str, dict[str, Any]] = {made: {"zulu_speakers_say": None, "share": 0.0, "voice_makes": made}}
        for option in saying:
            reached, _ = self.nearest_makeable(option["zulu"])
            if reached not in options or option["share"] > (options[reached]["share"] or 0.0):
                options[reached] = {"zulu_speakers_say": option["zulu"], "share": option["share"], "voice_makes": reached}
        for option in options.values():
            option["distance"] = round(distance(phone, option["voice_makes"]) or 1.0, 2)
            option["cost"] = option["distance"] - SHARE_BONUS * (option["share"] or 0.0)
        choices = sorted(options.values(), key=lambda o: (o["cost"], o["voice_makes"]))
        return {"english": phone, "aim": choices[0]["voice_makes"], "choices": [{k: v for k, v in c.items() if k != "cost"} for c in choices],
                "why": "the voice cannot make this sound: nearest it has, weighed by what Zulu speakers choose"}

    def aims(self, phones: Sequence[str]) -> list[dict[str, Any]]:
        return [self.aim_for(phone) for phone in phones]          # a diphthong stays one sound: the corpus and the voice both treat it as one

    def spellings(self, written: str, *, limit: int = 8) -> dict[str, Any]:
        """Spellings for a written word, best first, each with the reading and aim it came from; ``readings`` is empty when the dictionary and
        the capitals rule say nothing (the word is left to the caller as it is written)."""
        readings = self.readings(written)
        found: dict[str, dict[str, Any]] = {}
        for reading in readings:
            aims = self.aims(reading["phones"])
            for spelling, cost in letter_sounds.candidates([a["aim"] for a in aims], self.table, limit=limit * 2, noisy_edges=False):
                best = found.get(spelling)
                if best is None or cost < best["cost"]:
                    found[spelling] = {"spelling": spelling, "cost": round(cost, 3), "reading": reading["how"], "accent": reading["accent"],
                                       "english": reading["phones"], "aim": [a["aim"] for a in aims]}
        ranked = sorted(found.values(), key=lambda r: (r["cost"], len(r["spelling"]), r["spelling"]))[:limit]
        return {"word": written, "readings": readings, "spellings": ranked}

    @staticmethod
    def held_variants(spelling: str) -> list[str]:
        """A letter-by-letter spelling with its last letter held.  English stresses the last letter of an initialism, and an English listener hears it
        only when the voice holds it (about 170 ms): "Ey Aayi" is heard as AI, "Ey Ayi" and "Ey Ai" as A; "Ey En Siyi" as ANC, "Ey En Si" as A.  The
        voice holds a final "yi" (and "aayi" for the letter I) longer than the bare vowel."""
        low = spelling.lower()
        out = [low[:-2] + "aayi"] if low.endswith("ai") else []
        if low and low[-1] in "aeiou":
            out.append(low + "yi")
        return list(dict.fromkeys(out))

    def dictionary_note(self, written: str, speaker_phones: Sequence[str]) -> dict[str, Any] | None:
        """What the dictionary says the word sounds like, set beside the speaker's decode, for the agent to read: the reading nearest the decode (a
        speaker who does not say r after a vowel gets that reading), the stress, and the sounds only the decode has (the next word's, usually, when the decode
        was cut from running speech).  None when the dictionary and the capitals rule have no reading (a Zulu name)."""
        from .pronunciation_lab import align

        readings = self.readings(written)
        if not readings:
            return None
        best = max(readings, key=lambda r: align(r["phones"], list(speaker_phones))["closeness"])
        compared = align(best["phones"], list(speaker_phones))
        extra = [d.replace("the voice adds ", "").replace(" at the end", " (at the end)") for d in compared["differences"] if d.startswith("the voice adds")]
        note: dict[str, Any] = {
            "reads_as": best["how"], "sounds": " ".join(best["phones"]),
            "stress": " ".join(f"{p}({s})" if s else p for p, s in zip(best["phones"], best["stress"])),
            "closeness_to_the_speakers_decode": compared["closeness"]}
        if extra:
            note["only_in_the_speakers_decode"] = extra
            note["so"] = "sounds the dictionary does not have are often the next word's, cut in with this one; do not spell them into the word"
        if best["how"] == "guessed":
            note["reliability"] = "the dictionary does not have this word: a model guessed these sounds from its spelling. Treat them as a hint; the speaker's own decode comes first"
        if best["how"] == "letters":
            note["how_english_says_it"] = ("a word said letter by letter: English stresses the LAST letter and the letters before it are short. An English listener "
                                           "hears the last letter only when the voice holds it (about 170 ms): a spelling ending -yi or -aayi holds it, a bare vowel does not")
        return note

    def table_rows(self, phones: Sequence[str]) -> list[dict[str, Any]]:
        """For each English sound: what Zulu speakers say, what the voice makes, and the letters that make the aim."""
        rows = []
        for aim in self.aims(phones):
            letters = [(letters, option["seen"] / max(1, option["of"])) for letters, value in (self.table.get("graphemes") or {}).items()
                       for option in letter_sounds._options(value) if option["sounds"] == [aim["aim"]]]
            letters += [(g, 0.0) for g in letter_sounds.PRIOR.get(aim["aim"], []) if g not in {x for x, _ in letters}]
            rows.append({**aim, "letters_the_voice_reads_so": [g for g, _ in sorted(letters, key=lambda item: -item[1])][:6]})
        return rows
