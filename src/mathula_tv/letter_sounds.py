"""Which sounds the Zulu voice makes for which letters, learned from the voice itself, and spellings worked out backwards from sounds.

The agent used to find a spelling for an English word by guessing from English spelling ("science" -> sayensi, sayans, sayents: one family, five
tries) and only then measuring it.  This is the linguistic method behind a search instead:

1. ``build_table``: the voice says each letter or letter group of isiZulu spelling between several different neighbours (vowels around a
   consonant, consonants around a vowel) and the phone recogniser writes down what it heard each time.  A letter group is read differently next
   to different letters, so the table keeps every outcome seen and how often.  It is data from this voice's own audio, not hand-written rules,
   and it is built once per voice and kept.
2. ``candidates``: the speaker's sounds for the word are cut into letter groups that make those sounds (or sounds near them), and every way of
   doing so is ranked by how far each letter group's sounds are from the speaker's.  The letter groups come from two places: ``PRIOR``, the
   ordinary ways Latin letters write each sound (aɪ as ai, ay, ei, ey; a schwa as a, e, u, ah, uh), and the table above, whose probes the
   recogniser decodes unevenly for vowels, so it adds what it learned and does not overrule the prior.

3. ``neighbours``: a spelling is varied by small orthographic edits (a letter group swapped for one the voice treats the same way, a group left
   out, a vowel or glide added), the way a person tries "-ay-" for "-ai-".  Which groups are substitutes comes from the table too (two groups that
   share their most frequent outcome).

The result is a pool of spellings that reach the speaker's sounds by different letters.  They are only candidates: the voice reads a letter group
differently next to other letters ("saintsa" came out as ``s aɪ n t``), and the table comes from short nonsense syllables the recogniser decodes
unevenly, so every candidate is spoken and measured, the best are varied again, and only measured results are shown to the agent.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

SCHEMA = "mathula-letter-sounds-v2"

# Letters and letter groups of isiZulu spelling (clicks c, q, x are left out: a respelling must never add one).
CONSONANTS = ["b", "bh", "d", "dl", "f", "g", "h", "hl", "j", "k", "kh", "l", "m", "n", "ng", "nk", "nj", "nd", "mb", "p", "ph", "r", "s", "sh",
              "t", "th", "ts", "tsh", "nts", "nt", "ns", "nz", "v", "w", "y", "z"]
VOWELS = ["a", "e", "i", "o", "u", "aa", "ee", "ii", "oo", "uu", "ai", "ay", "ei", "ey", "au", "aw", "oi", "oy", "ae", "ia", "ie", "ua", "ya", "ye", "yi"]

EDGE_SKIP_COST = 0.3          # the speaker's first and last sounds are noisy when cut from running speech (see pronunciation_lab.align)
GRAPHEME_COST = 0.05          # a small push toward fewer letter groups
NEAR_VOWEL_COST = 0.6         # vowels are reached through near vowels (a for the schwa); consonants only through their voicing partner


# How Latin letters write each sound the recogniser reports, the most ordinary spelling first.  It is what any phonology of isiZulu spelling says
# and it holds for every word; the voice then decides, by speaking each spelling, which of them it really reads that way.
PRIOR: dict[str, list[str]] = {
    "a": ["a", "ah"], "ɑ": ["a", "ah"], "ɐ": ["a", "ah"], "aː": ["aa", "a"], "ɛ": ["e", "eh", "a"], "e": ["e", "eh"], "eː": ["ee", "e"],
    "ə": ["a", "e", "u", "ah", "uh"], "ʌ": ["a", "u", "ah", "uh"], "ɜ": ["e", "a", "u"], "æ": ["a", "e"], "ɒ": ["o", "a"],
    "ɪ": ["i", "e", "ih"], "i": ["i", "ee"], "iː": ["ii", "ee", "i"], "ɔ": ["o", "oh"], "o": ["o", "oh"], "oː": ["oo", "o"],
    "ʊ": ["u", "uh", "oo"], "u": ["u", "oo"], "uː": ["uu", "oo", "u"],
    "aɪ": ["ai", "ay", "ei", "ey", "ae", "ayi", "ahi", "aai", "aayi"], "eɪ": ["ei", "ey", "ay", "ai", "aye"], "əɪ": ["ai", "ay", "ei"], "aʊ": ["au", "aw", "ow"],
    "oʊ": ["ou", "ow", "o"], "əʊ": ["ou", "ow", "o"], "ɔɪ": ["oi", "oy"],
    "p": ["p", "ph", "b"], "b": ["b", "bh", "mb"], "t": ["t", "th", "d"], "d": ["d", "nd", "t"], "k": ["k", "kh", "g"], "ɡ": ["g", "k", "ng"],
    "g": ["g", "k"], "tʃ": ["tsh", "j", "tj"], "dʒ": ["j", "dj"], "f": ["f", "ph"], "v": ["v", "f"], "s": ["s", "ss", "z"], "z": ["z", "s"],
    "ʃ": ["sh", "s"], "ʒ": ["j", "zh"], "h": ["h"], "m": ["m"], "n": ["n", "nn"], "ŋ": ["ng", "nk"], "l": ["l"], "ɹ": ["r"], "r": ["r"],
    "ɾ": ["r"], "j": ["y"], "w": ["w"], "ts": ["ts", "nts", "ns", "tz"], "dz": ["dz", "z"], "θ": ["th", "t", "s"], "ð": ["d", "z", "th"],
    "ɬ": ["hl", "l"], "ɕ": ["dl", "hl"], "ç": ["hl"],
}
PRIOR_RANK_COST = 0.04        # each place down the list is a slightly less ordinary spelling of the sound


def _is_vowel(phone: str) -> bool:
    from .pronunciation_lab import _is_vowel as vowel

    return vowel(phone)


def _middle(phones: Sequence[str], *, carrier_is_vowel: bool) -> list[str]:
    """The sounds between the fixed neighbours of a probe."""
    heard = list(phones)
    if heard and _is_vowel(heard[0]) == carrier_is_vowel:
        heard = heard[1:]
    if heard and _is_vowel(heard[-1]) == carrier_is_vowel:
        heard = heard[:-1]
    return heard


CONSONANT_FRAMES = ("a{}a", "i{}a", "a{}i")           # a consonant between vowels
VOWEL_FRAMES = ("s{}n", "t{}t", "m{}l", "s{}s")       # a vowel between consonants
OUTCOMES_KEPT = 3                                      # the most frequent outcomes of a letter group that are kept


def build_table(reading: Callable[[str], Sequence[str]], *, voice: str = "", progress: Callable[[str], None] = lambda message: None) -> dict:
    """The voice's sounds for each letter group, in every neighbourhood probed.  ``reading(spelling)`` is the voice saying a spelling, decoded
    to phones.  Each letter group gets its most frequent outcomes: ``{"sounds": [...], "seen": k, "of": n}``."""
    from collections import Counter

    graphemes: dict[str, list[dict]] = {}
    for kind, letters_groups, frames in (("consonant", CONSONANTS, CONSONANT_FRAMES), ("vowel", VOWELS, VOWEL_FRAMES)):
        for group in letters_groups:
            outcomes: Counter = Counter()
            asked = 0
            for frame in frames:
                try:
                    middle = _middle(reading(frame.format(group)), carrier_is_vowel=(kind == "consonant"))
                except Exception as exc:                   # one probe failing leaves that neighbourhood out
                    progress(f"[letter sounds] {frame.format(group)!r}: {type(exc).__name__}")
                    continue
                asked += 1
                if middle:
                    outcomes[tuple(middle)] += 1
            if outcomes:
                graphemes[group] = [{"sounds": list(sounds), "seen": seen, "of": asked} for sounds, seen in outcomes.most_common(OUTCOMES_KEPT)]
    return {"schema": SCHEMA, "voice": voice, "graphemes": graphemes}


def _options(value: object) -> list[dict]:
    """A letter group's outcomes, whether the table kept several (v2) or just one list of sounds."""
    if value and isinstance(value, list) and isinstance(value[0], str):
        return [{"sounds": list(value), "seen": 1, "of": 1}]
    return [dict(option) for option in value or [] if option.get("sounds")]


def load_or_build(path: Path, reading: Callable[[str], Sequence[str]], *, voice: str = "", progress: Callable[[str], None] = lambda message: None) -> dict:
    path = Path(path)
    try:
        table = json.loads(path.read_text(encoding="utf-8"))
        if table.get("schema") == SCHEMA and table.get("graphemes"):
            return table
    except (OSError, ValueError):
        pass
    table = build_table(reading, voice=voice, progress=progress)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(table, ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError:
        pass                                                # an unwritable place means the table is built again next time, nothing worse
    return table


def _step_cost(target: str, made: str) -> float | None:
    """What it costs to reach the speaker's sound ``target`` with a letter group that makes ``made``; None when it cannot."""
    from .pronunciation_lab import substitution_cost

    cost = substitution_cost(target, made)
    if cost == 0.0:
        return 0.0
    if _is_vowel(target) and _is_vowel(made):
        return cost if cost <= NEAR_VOWEL_COST else None
    return cost if cost <= 0.3 else None


def candidates(target: Sequence[str], table: Mapping, *, limit: int = 24, beam: int = 400) -> list[tuple[str, float]]:
    """Spellings (lowercase) whose letter groups make the speaker's sounds ``target`` in the voice's table, best first, as (spelling, cost)."""
    graphemes = [(letters, list(option["sounds"]), 0.1 * (1.0 - option["seen"] / max(1, option["of"])))
                 for letters, value in (table.get("graphemes") or {}).items() for option in _options(value)]
    graphemes += [(letters, [phone], PRIOR_RANK_COST * rank) for phone, spellings in PRIOR.items() for rank, letters in enumerate(spellings)]
    count = len(target)
    if not count or not graphemes:
        return []
    states: list[dict[str, float]] = [dict() for _ in range(count + 1)]
    states[0][""] = 0.0
    for position in range(count + 1):
        for spelling, cost in sorted(states[position].items(), key=lambda item: item[1])[:beam]:
            if count > 1 and position in (0, count - 1):          # a noisy edge sound of the speaker may be left to the voice
                skipped = states[position + 1]
                skipped[spelling] = min(skipped.get(spelling, 99.0), cost + EDGE_SKIP_COST)
            for letters, phones, rarity in graphemes:
                end = position + len(phones)
                if end > count:
                    continue
                total = 0.0
                for wanted, made in zip(target[position:end], phones):
                    step = _step_cost(wanted, made)
                    if step is None:
                        break
                    total += step
                else:
                    reached = states[end]
                    key = spelling + letters
                    reached[key] = min(reached.get(key, 99.0), cost + total + GRAPHEME_COST + rarity)
    return sorted(states[count].items(), key=lambda item: (item[1], len(item[0]), item[0]))[:limit]


INSERTS = ["a", "e", "i", "u", "o", "h", "y", "n"]      # what may be added to a spelling: a vowel, a breath or glide, a nasal


def substitutes(table: Mapping) -> dict[str, list[str]]:
    """For each letter group, the other groups that write the same sound: those that share a sound in ``PRIOR`` (most ordinary first), then those
    the voice read the same way in the probes (its most frequent outcome is among theirs)."""
    found: dict[str, list[str]] = {}
    for spellings in PRIOR.values():
        for letters in spellings:
            for other in spellings:
                if other != letters and other not in found.setdefault(letters, []):
                    found[letters].append(other)
    outcomes: dict[str, set[tuple[str, ...]]] = {}
    top: dict[str, tuple[str, ...]] = {}
    for letters, value in (table.get("graphemes") or {}).items():
        options = _options(value)
        if options:
            top[letters] = tuple(max(options, key=lambda option: option["seen"])["sounds"])
            outcomes[letters] = {tuple(option["sounds"]) for option in options}
    for letters in top:
        for other in sorted(o for o in top if o != letters and top[letters] in outcomes[o]):
            if other not in found.setdefault(letters, []):
                found[letters].append(other)
    return found


def split(spelling: str, table: Mapping) -> list[str]:
    """A spelling as the letter groups the table knows (longest first); a letter it does not know stands alone."""
    known = sorted({*(table.get("graphemes") or {}), *(letters for spellings in PRIOR.values() for letters in spellings)}, key=len, reverse=True)
    parts, text = [], spelling.lower()
    while text:
        group = next((letters for letters in known if text.startswith(letters)), text[0])
        parts.append(group)
        text = text[len(group):]
    return parts


def neighbours(spelling: str, table: Mapping, *, limit: int = 60) -> list[str]:
    """Spellings one small edit away (lowercase): a letter group swapped for a substitute, then a group left out, then a letter added."""
    parts = split(spelling, table)
    swaps = substitutes(table)
    found: dict[str, None] = {}
    # the vowels first: the nucleus of a syllable is what a listener hears first ("sei" for "sai" is a different word), and its letters are the
    # ones with the most ways of being written; consonants after
    order = sorted(range(len(parts)), key=lambda index: (0 if any(letter in "aeiou" for letter in parts[index]) else 1, index))
    for index in order:
        for other in swaps.get(parts[index], []):
            found.setdefault("".join(parts[:index] + [other] + parts[index + 1:]), None)
    for index in range(len(parts)):
        if len(parts) > 2:
            found.setdefault("".join(parts[:index] + parts[index + 1:]), None)
    for index in range(len(parts) + 1):
        for added in INSERTS:
            found.setdefault("".join(parts[:index] + [added] + parts[index:]), None)
    found.pop(spelling.lower(), None)
    return list(found)[:limit]


def main(argv: list[str] | None = None) -> int:
    """Build (or show) the letter-sound table for a voice: ``python -m mathula_tv.letter_sounds --voice zu-ZA-ThembaNeural``."""
    import argparse
    import os

    from .pronunciation_lab import PronunciationLab

    parser = argparse.ArgumentParser(prog="mathula_tv.letter_sounds")
    parser.add_argument("--voice", default="zu-ZA-ThembaNeural")
    parser.add_argument("--work", default="working")
    parser.add_argument("--force", action="store_true", help="rebuild even if a table is kept")
    args = parser.parse_args(argv)
    path = Path(args.work) / "experience" / f"letter_sounds_{args.voice}.json"
    if args.force and path.exists():
        path.unlink()
    lab = PronunciationLab(Path(args.work) / "jobs" / "none", voice=args.voice, key=os.environ["AZURE_SPEECH_KEY"],
                           region=os.environ["AZURE_SPEECH_REGION"].strip().lower(), work_dir=Path(args.work) / "letter_sounds_probe")
    table = load_or_build(path, lab.voice_reading, voice=args.voice, progress=print)
    for letters, options in table["graphemes"].items():
        print(f"{letters:<5} -> " + "  |  ".join(f"{' '.join(o['sounds'])} ({o['seen']}/{o['of']})" for o in options))
    print(f"{len(table['graphemes'])} letter groups kept in {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
