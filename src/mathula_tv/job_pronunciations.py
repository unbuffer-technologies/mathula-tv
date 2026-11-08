"""A job's pronunciations by rule alone: its English words, names and initialisms spelled the way a Zulu speaker says them.

    python -m mathula_tv.job_pronunciations --job <job id> --run <run id>

Nothing is spoken, heard or scored.  The words come from the dub text of one run; each is spelled by ``zulu_rules`` from its
dictionary sounds (the job's web research first, then CMUdict, then the guesser) and written to the job's
``native_dub/pronunciation_names.json`` for both voices.  A person listens and the rules are changed, not the words.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

SOURCE = "rules"
VOICES = ("zu-ZA-ThandoNeural", "zu-ZA-ThembaNeural")
_TOKEN = re.compile(r"(?<![A-Za-z0-9])[A-Za-z][A-Za-z'’]*(?:-[A-Za-z][A-Za-z'’]*)*")          # not the "th" of 4th or the "nd" of 22nd: those are the ordinal's, read as a number
ZULU_COMMON = 60               # how often the Zulu side of the corpus writes a form before it counts as a Zulu word (High 42, October 35, Party 2; Pho 79, wale 106, mina 340)
_SHORT_PREFIX = 4            # "i-", "ne-", "ngo-", "ama-": a short piece before a hyphen is Zulu grammar, not the word


def core(token: str) -> str:
    """The English part of a written token: ``i-MK`` -> ``MK``, ``ngo-WhatsApp`` -> ``WhatsApp``; a glued prefix (``weScience``) stays for the renderer."""
    parts = token.split("-")
    if len(parts) > 1 and (len(parts[0]) <= _SHORT_PREFIX or (parts[0].islower() and parts[1][:1].isupper())):          # a capital only opens a sentence: "I-Western Cape High Court"; "njenge-SG"
        return "-".join(parts[1:])
    return token


def candidates(texts: Iterable[str], *, researched: Mapping[str, Any], dictionary: Callable[[str], bool], zulu_count: Callable[[str], int],
               said: Callable[[str], bool] | None = None, lessons: Mapping[str, Any] | None = None, unsure: list[dict[str, str]] | None = None,
               resolved: list[dict[str, str]] | None = None) -> list[str]:
    """The tokens of the dub text that are English as spoken: initialisms in capitals, words the job's research has sounds for, and words the dictionary
    knows that real Zulu text hardly ever writes (``ZULU_COMMON`` times or more is a Zulu word that happens to be in the English dictionary: uma, mina, into)
    or that are capitalised inside a sentence (a name).  In order of first appearance, one spelling of each.

    A dictionary word must also have been *said*: ``said`` tells whether anybody said it (transcript, restored text or captions).  One nobody said was written by the
    translator (``Wehle``, a Zulu verb that is a surname in the dictionary), and the speaker may have been code-switching into a language this cannot tell, so it is not
    respelled as English.  The listener decides (``lessons``, ``language_questions``): a word answered "english" is respelled, any other answer leaves it as written, and a
    word whose signature earlier answers settle is settled the same way (appended to ``resolved``); the rest are appended to ``unsure`` and left as written until asked."""
    from . import language_questions
    from .english_sounds import plural_initialism
    from .interpreter import zulu_shaped
    from .pronunciation_lab import without_prefix, written_prefix

    lessons = lessons or {}
    answered = lessons.get("words") or {}
    seen: dict[str, str] = {}
    for text in texts:
        text = re.sub(r"~pause\d+", " ", text)
        for match in _TOKEN.finditer(text):
            word = core(match.group(0)).strip("-")
            word = without_prefix(word, written_prefix(word))          # "uJacob": the u is grammar
            key = word.casefold()
            if len(word) < 2 or key in seen or "-" in word:
                continue
            before = text[:match.start()].rstrip()
            inside = word[:1].isupper() and bool(before) and before[-1] not in ".?!:—"          # a capital after a word, not one that opens a sentence
            zulu_name = zulu_shaped(word) and not dictionary(key)          # Zuma, Ramaphosa, Nomvalo: already isiZulu spelling, left alone even where web research gave them English sounds
            answer = answered.get(language_questions.key_of(word))
            if answer:                                          # the listener said what language this is: English is respelled, anything else is left as written
                if answer.get("language") == "english":
                    seen[key] = word
                continue
            if (word.isupper() and word.isalpha()) or plural_initialism(word) or (key in researched and not zulu_name):
                seen[key] = word
            elif dictionary(key) and (zulu_count(key) < ZULU_COMMON or inside):
                if said is None or said(word):
                    seen[key] = word
                    continue
                sig = language_questions.signature(word, inside=inside)
                settled = language_questions.verdict(lessons, sig)
                if settled == "english":
                    seen[key] = word
                if settled and resolved is not None:
                    resolved.append({"word": word, "language": settled, "signature": sig})
                elif not settled and unsure is not None:
                    unsure.append({"word": word, "signature": sig})
    return list(seen.values())


_ORDINAL = re.compile(r"(?<![A-Za-z0-9])(\d{1,2})(?:st|nd|rd|th)(?![A-Za-z0-9])")
_ONES = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
         "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def ordinal_words(number: int) -> list[str]:
    """The English words a written ordinal is said with (22nd -> twenty two): the day of a date is a cardinal after the Zulu prefix (ngo-, ka-), as in "ngomhla ka-22",
    and the cardinal is shorter than "twenty-second"."""
    if number < 20:
        return [_ONES[number]]
    tens, ones = divmod(number, 10)
    return [_TENS[tens], *([_ONES[ones]] if ones else [])]


def ordinals(texts: Iterable[str]) -> list[str]:
    """The written ordinals of the text (``22nd``), once each."""
    return list(dict.fromkeys(m.group(0) for text in texts for m in _ORDINAL.finditer(text) if 1 <= int(m.group(1)) < 100))


def write_lexicon(path: Path, spelled: Sequence[Mapping[str, Any]], voices: Sequence[str] = VOICES, by_ear: Mapping[str, str] | None = None) -> Path:
    """The job's names lexicon with only these rule spellings in it (the file it replaced is kept beside it, once)."""
    path = Path(path)
    keep = path.with_name(path.stem + ".before_job_rules.json")
    if path.is_file() and not keep.is_file():
        shutil.copyfile(path, keep)
    entries = [{"display_text": s["word"], "tts_text": s["word"], "tts_text_by_voice": {voice: (s.get("by_voice") or {}).get(voice, s["spelling"]) for voice in voices},
                "kind": "english_code_switch", "confidence": 0.9, "source": s.get("source", SOURCE), "reads_as": s["reading"], "english": " ".join(s["english"]),
                "derivation": s["derivation"], **({"decision": s["decision"]} if s.get("decision") else {})} for s in spelled]
    entries += [{"display_text": text, "tts_text": text, "tts_text_by_voice": {voice: spelling for voice in voices}, "kind": "english_code_switch", "confidence": 0.9,
                 "source": "ear"} for text, spelling in (by_ear or {}).items()]
    path.write_text(json.dumps({"schema": "mathula-names-lexicon-v1", "voice": voices[0], "entries": entries}, indent=1, ensure_ascii=False), encoding="utf-8")
    return path


def apply_book(spelled: list[dict[str, Any]], entries: Mapping[str, Any], voices: Sequence[str] = VOICES) -> list[dict[str, Any]]:
    """Put the listener's committed spellings over the rules': a word the listener committed (approved as the rules wrote it, or a custom spelling) is spoken exactly as committed, in each voice
    that has a committed spelling; the rules' spelling stays beside it as ``rule_spelling``."""
    from .pronunciation_book import key_of, spelling_for

    out = []
    for item in spelled:
        by_voice = {voice: spelling_for(entries, item["word"], voice) for voice in voices}
        if any(by_voice.values()):
            item = {**item, "by_voice": {voice: spelling for voice, spelling in by_voice.items() if spelling}, "source": "ear",
                    "decision": entries[key_of(item["word"])].get("kind"), "rule_spelling": item["spelling"]}
        out.append(item)
    return out


class Prepared:
    """Everything needed to spell a job's English words: the dictionary, the sounds the job's web research found for names, and how often Zulu text writes a form."""

    def __init__(self, job_root: Path) -> None:
        from . import attestation
        from .english_sounds import EnglishSounds, g2p_guesser
        from .interpreter import Interpreter
        from .zulu_rules import research_readings

        self.job_root = Path(job_root)
        self.work_dir = self.job_root.parents[1]
        self.sounds = EnglishSounds.load(self.work_dir / "resources" / "cmudict" / "cmudict.dict")
        self.sounds.guesser = g2p_guesser()
        self.researched: dict[str, Any] = {}
        for research in (self.job_root / "analysis" / "autocorrection_name_research.json", self.job_root / "native_dub" / "pronunciation_research.json"):          # the second pass wins where both found a word
            if research.is_file():
                self.researched.update(research_readings(json.loads(research.read_text(encoding="utf-8"))))
        self.interpreter = Interpreter(self.sounds, {}, None, self.researched)
        self.corpus = attestation.load(self.work_dir)
        self.refresh_lessons()
        self.unsure: list[dict[str, str]] = []          # dictionary words in the dub that nobody said, with no answer yet: asked of the listener, left as written
        self.resolved: list[dict[str, str]] = []        # the same, settled by the way earlier answers went

    def refresh_lessons(self) -> None:
        """Read what was said in the job and the listener's shared lessons again (after an answer, or after pass 1 changed)."""
        from . import language_questions

        said = language_questions.said_words(self.job_root)
        self.said = None if said is None else (lambda word, _said=said: language_questions.heard(_said, word))
        self.lessons = language_questions.load_lessons(self.work_dir)

    def zulu_count(self, word: str) -> int:
        return self.corpus.frequency.get(word, 0) if self.corpus is not None else 0


def spell(prepared: Prepared, texts: Iterable[str], *, progress: Callable[[str], None] = print) -> list[dict[str, Any]]:
    """The English words, names, initialisms and written ordinals of ``texts``, each with the spelling the rules give it (nothing yet from the book)."""
    texts = list(texts)
    interpreter = prepared.interpreter
    prepared.unsure.clear()
    prepared.resolved.clear()
    words = candidates(texts, researched=prepared.researched, dictionary=lambda w: bool(prepared.sounds.word(w)), zulu_count=prepared.zulu_count, said=prepared.said,
                       lessons=prepared.lessons, unsure=prepared.unsure, resolved=prepared.resolved)
    spelled = []
    for written in ordinals(texts):              # 22nd: the voice reads the digits in its own way ("2020"); the day is said in English, as the number
        parts = [interpreter.rules_spelling(word) for word in ordinal_words(int(re.match(r"\d+", written).group(0)))]
        if all(parts):
            spelled.append({"word": written, "spelling": "-".join(p["spelling"] for p in parts), "reading": "number", "english": [w for p in parts for w in p["english"]],
                            "derivation": [f"{written} is said in English as the number {' '.join(ordinal_words(int(re.match(r'[0-9]+', written).group(0))))}", *parts[0]["derivation"]]})
            progress(f"  {written} -> {spelled[-1]['spelling']}  [number]")
    for word in words:
        found = interpreter.rules_spelling(word)
        if found is None:
            progress(f"  {word}: no reading, left as written")
            continue
        spelled.append({"word": word, **found})
        progress(f"  {word} -> {found['spelling']}  [{found['reading']}]")
    return spelled


def with_committed_words(spelled: list[dict[str, Any]], entries: Mapping[str, Any], texts: Iterable[str], voices: Sequence[str] = VOICES) -> list[dict[str, Any]]:
    """The rules' spellings with the listener's committed ones over them (``apply_book``), plus the committed words the rules did not find in the text at all
    (a Zulu word the voice mangles, such as sotswebhu): they are spoken as committed wherever the text writes them."""
    from .ear_clips import word_pattern
    from .pronunciation_book import key_of, spelling_for

    done = apply_book(spelled, entries, voices)
    seen = {key_of(item["word"]) for item in done}
    joined = " ".join(texts)
    for key, entry in entries.items():
        if key in seen or not word_pattern(str(entry.get("word"))).search(joined):          # bare, after a hyphenated prefix, or with a Zulu prefix glued on
            continue
        by_voice = {voice: spelling_for(entries, entry["word"], voice) for voice in voices}
        spelling = next((v for v in by_voice.values() if v), None)
        if spelling:
            done.append({"word": entry["word"], "spelling": spelling, "reading": "committed", "english": [], "derivation": ["committed by the listener; no rule applies"],
                         "by_voice": {voice: sp for voice, sp in by_voice.items() if sp}, "source": "ear", "decision": entry.get("kind"), "rule_spelling": ""})
    return done


def build(job_root: Path, texts: Iterable[str], *, progress: Callable[[str], None] = print, book_path: Path | None = None, prepared: Prepared | None = None) -> list[dict[str, Any]]:
    """Spell the English words of ``texts`` for the job and write its lexicon.  Returns what was spelled.  Spellings the listener has committed (the pronunciation book)
    replace the rules' for their words."""
    from .pronunciation_book import load as load_book

    texts = list(texts)
    prepared = prepared or Prepared(job_root)
    spelled = spell(prepared, texts, progress=progress)
    book = load_book(book_path or prepared.work_dir / "pronunciation_book.json")["entries"]
    spelled = with_committed_words(spelled, book, texts)
    for item in spelled:
        if item.get("decision"):
            progress(f"  {item['word']}: the listener's {item['decision']} spelling {item['by_voice']} stands over the rules' {item.get('rule_spelling') or '(none)'}")
    write_lexicon(Path(job_root) / "native_dub" / "pronunciation_names.json", spelled, by_ear=by_ear(Path(job_root)))
    return spelled


EAR_FILE = "pronunciation_by_ear.json"


def by_ear(job_root: Path) -> dict[str, str]:
    """Spellings a person chose by listening to this job, written as {text: spelling} in ``native_dub/pronunciation_by_ear.json``: a word the voice mangles that no
    rule covers (the Zulu "sotswebhu", which the voice says with a p).  They stand beside the rule spellings and are spoken by both voices."""
    path = Path(job_root) / "native_dub" / EAR_FILE
    return {str(k): str(v) for k, v in json.loads(path.read_text(encoding="utf-8")).items()} if path.is_file() else {}


def main(argv: list[str] | None = None) -> int:
    import argparse

    from .config import load_settings
    from .experience_store import ExperienceStore

    parser = argparse.ArgumentParser(prog="mathula_tv.job_pronunciations")
    parser.add_argument("--job", required=True)
    parser.add_argument("--run", required=True, help="the run whose dub text holds the words")
    parser.add_argument("--clip-id")
    args = parser.parse_args(argv)
    settings = load_settings()
    clip_id = args.clip_id or args.job[:8]
    submission = json.loads((ExperienceStore(settings.work_dir).repeat_dir(args.run, clip_id, 1) / "submission.json").read_text(encoding="utf-8"))
    spelled = build(Path(settings.work_dir) / "jobs" / args.job, [str(i.get("zulu_text") or "") for i in submission["items"]])
    print(len(spelled), "words spelled by rule for", args.job)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
