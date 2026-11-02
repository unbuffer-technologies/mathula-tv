"""A pronunciation lexicon built in advance, for each voice, from the English words and initialisms that real Zulu text keeps in Latin letters.

The agent used to find a spelling for each word in each clip by trying and measuring, which took 16 to 38 minutes a clip, skipped words, and measured on one voice
while the dub speaks two.  This does that work once per voice, offline:

1. ``words``: the English words and initialisms the Zulu corpus keeps (counted from its Zulu side), minus Zulu words that happen to be capitalised, names, and
   initialisms that are said as words (COVID, SASSA), plus a seed of the words South African news is made of.
2. ``build``: for each word and a voice, the interpreter's candidate spellings are spoken by that voice, decoded, and put to two listeners (the English speech-to-text and,
   for a word said letter by letter, the phone recogniser's letters).  A word that is already right as written is left alone; otherwise the closest spelling a listener
   takes for the word is kept.  Nothing is judged without the voice saying it.
3. ``merge``: the kept spellings go into the shared pronunciation memory for that voice (source ``lexicon``: an agent's choice from a clip's own audio replaces it, never
   the other way round), and the renderer speaks each voice's spelling for that voice only.
"""
from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

SEED_WORDS = (
    "minister parliament commission president allegations corruption secretary general leadership decision court party western news science education deputy "
    "october november december january february august september committee government department municipality constitution election budget treasury police "
    "investigation judge tribunal inquiry report statement spokesperson premier mayor council chairperson director manager official union strike protest").split()
SEED_INITIALISMS = (
    "ANC EFF DA MK MKP IFP SABC SAPS SARS NPA IEC ID UK US EU UN AU SA VAT GDP CEO CFO TV AI API CPU GPU HIV TB KZN NGO NHI SMS ICT GBV SMME ICC SADC NDP MEC ARV IDP "
    "EPWP CIDB GCIS DPME MTEF FBI CIA BBC CNN UIF SANDF OK PTA SCA NDPP IPID PRASA ESKOM").split()
SAID_AS_WORDS = frozenset({"COVID", "SASSA", "PAIA", "GEMS", "BRICS", "NATO", "NOMA", "UMA", "ESKOM", "PRASA", "IPID", "SARS"})          # spoken as a word, not by letters
NAMES = frozenset({"durban", "natal", "singh", "jean", "ngor", "dass", "jacob", "jessie", "cyril", "johnson", "scott", "eric", "paris", "aaron", "russia", "phoenix",
                   "johannesburg", "pretoria", "gauteng", "limpopo", "zuma", "ramaphosa", "malema"})      # the corpus writes these with a capital mostly, but not always
MAX_LETTERS_UNSEEDED = 3        # a counted initialism longer than this is taken only if it is seeded: PAIA and GEMS are words
MIN_LOWER_SHARE = 0.5           # a word mostly written with a capital in the corpus is a name (Durban, Singh)
CANDIDATES = 10                 # spellings spoken per word and voice
LISTENED = 4                    # the closest of them the speech-to-text is asked about, besides the word as written
CLOSE_ENOUGH = 0.7              # a spelling must be at least this close to a reading to be kept
CONFIDENT = 0.85                # a word's spelling this close to the dictionary reading needs no listener to say so


# What the English speech-to-text writes for a letter said alone (it writes "Eh" or "Hey" for A), and spellings worth trying beside the interpreter's.
LETTER_ACCEPT = {"A": "a eh ay hey", "B": "b be bee", "C": "c see sea", "D": "d dee", "E": "e", "F": "f ef eff", "G": "g gee", "H": "h age aitch ach", "I": "i eye hi", "J": "j jay",
                 "K": "k kay okay", "L": "l el elle", "M": "m em", "N": "n en", "O": "o oh", "P": "p pee", "Q": "q queue cue", "R": "r are", "S": "s es ess", "T": "t tea tee",
                 "U": "u you", "V": "v vee", "W": "w", "X": "x ex", "Y": "y why", "Z": "z zed zee"}
LETTER_ENDINGS = ("i", "a", "yi", "ayi", "ii", "aa", "e", "u", "ee", "ay", "yii", "aayi")
HELD_MS = 140                   # an English listener hears a last letter only when the voice holds its vowel about this long
SHORT_MS = 90
BUILD = "rules-first-1"         # how the candidates were made and chosen; a word said by letters built under another is built again with ``again_letters``


def _tail_ms(lab: Any, spelling: str) -> int | None:
    """How long the voice holds the last sound of ``spelling`` said alone: from that sound's start to where the word ends."""
    audio = lab._audio_for(spelling)
    timed = lab._word_timed(audio)
    end = lab._word_end.get(str(audio))
    return round((end - timed[-1][1]) * 1000) if timed and end is not None else None


def build_letters(lab: Any, out_path: Path, *, shard: int = 0, shards: int = 1, progress: Callable[[str], None] = print) -> int:
    """For each letter of the alphabet, spellings of its English name spoken by the lab's voice and judged (the speech-to-text writes the letter, or the phone recogniser
    decodes its name), with how long each holds its last sound; appends one JSON line per letter (a stopped build resumes)."""
    from . import letter_sounds, zulu_rules
    from .english_sounds import LETTER_NAMES, to_ipa
    from .pronunciation_lab import _letters_closeness

    out_path = Path(out_path)
    done = {json.loads(line)["letter"] for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip()} if out_path.is_file() else set()
    interpreter, count = lab._interpreter(), 0
    with out_path.open("a", encoding="utf-8") as handle:
        for index, letter in enumerate(sorted(LETTER_NAMES)):
            if index % shards != shard or letter in done:
                continue
            target = to_ipa(LETTER_NAMES[letter])
            measured = [s for s, _ in letter_sounds.candidates([a["aim"] for a in interpreter.aims(target)], lab._letter_table(), limit=8, noisy_edges=False)] if interpreter else []
            # the English-to-Zulu rules' spellings of the letter's name (brief, held, longer held) first, then the letters that make the voice say it
            pool = [r["spelling"] for r in zulu_rules.respell_initialism(letter, limit=10)] + measured
            pool += [s + e for s in measured[:3] for e in LETTER_ENDINGS if not s.endswith(e)]
            tried = []
            for spelling in dict.fromkeys(pool):
                heard = lab.voice_reading(spelling)
                listener = lab.english_hears(spelling) or ""
                said = any(w in LETTER_ACCEPT[letter].split() for w in re.sub(r"[^a-z ]", "", listener.lower()).split())
                tried.append({"spelling": spelling, "accepted": said or _letters_closeness(target, heard) >= 0.85, "listener": listener, "sounds": " ".join(heard),
                              "tail_ms": _tail_ms(lab, spelling)})
            handle.write(json.dumps({"letter": letter, "voice": lab.voice, "tried": tried}, ensure_ascii=False) + "\n")
            handle.flush()
            count += 1
            progress(f"{lab.voice} letter {letter}: {sum(t['accepted'] for t in tried)} of {len(tried)} accepted")
    return count


def letter_forms(records: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, str]]:
    """From the letter records (``build_letters``): for each letter, the spelling the voice says briefly and the one it holds, both of which a listener takes for the letter."""
    forms: dict[str, dict[str, str]] = {}
    for record in records:
        good = [t for t in record["tried"] if t["accepted"] and t["tail_ms"] is not None]
        if not good:
            continue
        short = min(good, key=lambda t: (t["tail_ms"] > SHORT_MS, len(t["spelling"]), t["tail_ms"]))
        held = [t for t in good if t["tail_ms"] >= HELD_MS]
        forms[record["letter"]] = {"short": short["spelling"], "held": min(held, key=lambda t: (t["tail_ms"], len(t["spelling"])))["spelling"] if held else short["spelling"]}
    return forms


def compose(word: str, forms: Mapping[str, Mapping[str, str]]) -> list[str]:
    """Spellings of a word said letter by letter, made letter by letter: the earlier letters brief and the last one held (English stresses the last letter), with the
    letters spaced or hyphenated (run together, Zulu letter rules turn n and s into nts), and the last one brief as well."""
    letters = [c for c in word.upper() if c.isalpha()]
    if not letters or any(letter not in forms for letter in letters):
        return []
    out = []
    for last in ("held", "short"):
        pieces = [forms[letter]["short"] for letter in letters[:-1]] + [forms[letters[-1]][last]]
        for separator in (" ", "-"):
            spelling = separator.join(piece.capitalize() if separator == " " else piece for piece in pieces)
            if spelling not in out:
                out.append(spelling)
    return out


def collect(work_dir: Path, *, words_limit: int = 150, initialisms_limit: int = 40, lines: Iterable[str] | None = None) -> dict[str, Any]:
    """The words and initialisms for the lexicon, most frequent first (seeds first)."""
    from .english_sounds import EnglishSounds
    from .pronunciation_search import looks_isizulu

    resources = Path(work_dir) / "resources" / "attestation"
    sounds = EnglishSounds.load(Path(work_dir) / "resources" / "cmudict" / "cmudict.dict")
    zulu_forms: set[str] = set()
    for path in (resources / "lexicon").glob("*.txt"):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            zulu_forms.update(t.lower() for t in re.findall(r"[A-Za-z']+", line))
    if lines is None:
        lines = (resources / "autshumato.zu.txt").read_text(encoding="utf-8", errors="replace").splitlines()
    lower, capital, caps = collections.Counter(), collections.Counter(), collections.Counter()
    for line in lines:
        for token in re.findall(r"[A-Za-z][A-Za-z']*", line):
            if re.fullmatch(r"[A-Z]{2,5}", token):
                caps[token] += 1
            elif len(token) >= 4 and token.lower() not in zulu_forms and not looks_isizulu(token.lower()) and sounds is not None and sounds.word(token.lower()):
                (capital if token[:1].isupper() else lower)[token.lower()] += 1
    words = [w for w, _ in sorted(((w, lower[w] + capital[w]) for w in set(lower) | set(capital)), key=lambda kv: -kv[1])
             if lower[w] >= MIN_LOWER_SHARE * (lower[w] + capital[w]) and lower[w] > 0 and w not in NAMES]
    seeded_initialisms = [i for i in SEED_INITIALISMS if i not in SAID_AS_WORDS]
    # a counted run of capitals that is also an English word (INK, ART, OR, ME) is a word in capitals, not an initialism
    counted = [i for i, _ in caps.most_common() if i.lower() not in zulu_forms and not looks_isizulu(i.lower()) and i not in SAID_AS_WORDS
               and (len(i) <= MAX_LETTERS_UNSEEDED or i in SEED_INITIALISMS) and i not in seeded_initialisms and not (sounds is not None and sounds.word(i.lower()))]
    return {"initialisms": seeded_initialisms + counted[:initialisms_limit],
            "words": list(dict.fromkeys([*SEED_WORDS, *words[:words_limit]]))}


def decide(written: str, tried: Sequence[Mapping[str, Any]], *, letters: bool = False) -> dict[str, Any]:
    """From the spellings tried (each ``{"spelling", "closeness", "accepted"}``; ``written`` is the word as the text writes it): leave a word that is already right, keep
    the closest spelling a listener takes for the word, or report that none was.  A word said letter by letter (``letters``) keeps one whose last sound the voice holds
    (``tail_ms`` of at least HELD_MS) when a listener takes any such: English stresses the last letter (the I of AI is held, the I of HIV is not), and the closeness of
    the sounds cannot tell a held vowel from a brief one."""
    by_spelling = {t["spelling"]: t for t in tried}
    here = by_spelling.get(written)
    if letters:
        if here and here["accepted"]:
            return {"status": "fine as written"}
        good = [t for t in tried if t["accepted"] and t["closeness"] >= CLOSE_ENOUGH and t["spelling"] != written]
    else:
        # a word has a dictionary reading, and the sounds the voice makes are measured against it: sounds that close need no listener, and a listener that merely knows
        # the word ("News" written for [n eɪ]) does not make a reading right.  A listener's approval counts only for sounds that are fairly close.
        if here and (here["closeness"] >= CONFIDENT or (here["accepted"] and here["closeness"] >= CLOSE_ENOUGH)):
            return {"status": "fine as written"}
        good = [t for t in tried if t["closeness"] >= CLOSE_ENOUGH and (t["accepted"] or t["closeness"] >= CONFIDENT) and t["spelling"] != written]
    if not good:
        return {"status": "unresolved"}
    if letters:
        held = [t for t in good if (t.get("tail_ms") or 0) >= HELD_MS]
        good = held or good
    # equally close spellings: the one tried first, which is the English-to-Zulu rules' main choice, then its variants, then the letters composed from the voice's
    # forms; the rules are the prior, so the voice's measurement overrides them only where it finds a spelling that is closer
    order = {t["spelling"]: index for index, t in enumerate(tried)}
    best = max(good, key=lambda t: (t["closeness"], -order[t["spelling"]]))
    return {"status": "respell", "spelling": best["spelling"], "closeness": best["closeness"], "held": bool(letters and (best.get("tail_ms") or 0) >= HELD_MS)}


def evaluate(lab: Any, word: str, forms: Mapping[str, Mapping[str, str]] | None = None) -> dict[str, Any]:
    """Speak the candidates for ``word`` in the lab's voice and decide (see ``decide``).  ``forms`` are the voice's letter forms (``letter_forms``): a word said letter by
    letter is also composed from them."""
    from .english_sounds import is_initialism
    from .pronunciation_lab import align, understood_as

    interpreter = lab._interpreter()
    readings = interpreter.readings(word) if interpreter else []
    if not readings:
        return {"word": word, "status": "no reading"}
    written = word if is_initialism(word) else word.lower()
    composed = compose(word, forms) if forms and readings[0]["how"] == "letters" else []
    ruled = [r["spelling"] for r in interpreter.rule_spellings(word, readings)]                # the English-to-Zulu rules' spellings come first: they are the prior
    spellings = list(dict.fromkeys([written, *ruled, *composed, *lab._interpreter_spellings(word)]))[:CANDIDATES + 1 + len(ruled) + len(composed)]
    tried: list[dict[str, Any]] = []
    for spelling in spellings:
        heard = lab.voice_reading(spelling)
        closeness = max((align(heard, reading["phones"], epenthetic_source=True)["closeness"] for reading in readings), default=0.0) if heard else 0.0
        tried.append({"spelling": spelling, "closeness": round(closeness, 3), "sounds": " ".join(heard), "heard": heard})
    asked = {written, *composed, *ruled} |{t["spelling"] for t in sorted(tried, key=lambda t: -t["closeness"])[:LISTENED]}
    for t in tried:
        heard = t.pop("heard")
        t["accepted"] = False
        if t["spelling"] in asked:
            t["listener"] = lab.english_hears(t["spelling"])                       # asked once: the text is kept and the verdict is made from it
            t["accepted"] = bool(t["listener"] is not None and understood_as(t["listener"], word)) or bool(lab._letters_heard(word, heard))
    letters = readings[0]["how"] == "letters"
    if letters:
        for t in tried:
            if t["accepted"]:
                t["tail_ms"] = _tail_ms(lab, t["spelling"])
    return {"word": word, "reads_as": readings[0]["how"], "build": BUILD, **decide(written, tried, letters=letters), "tried": tried}


def redecide(lab: Any, records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The records of an earlier build (``build``), decided again under the length rule: for a word said letter by letter, how long the voice holds the last sound of each
    spelling a listener took is measured from the audio the build kept, and ``decide`` chooses again.  The listeners are not asked again.  Returns the records whose
    choice changed."""
    from .pronunciation_lab import align

    interpreter = lab._interpreter()
    changed = []
    for record in records:
        if record.get("status") == "error" or not record.get("tried"):
            continue
        letters = record.get("reads_as") == "letters"
        tried = [dict(t) for t in record.get("tried", [])]
        readings = interpreter.readings(record["word"]) if interpreter and not letters else []
        for t in tried:
            if readings and t.get("sounds"):               # the sounds the voice made are kept: the closeness is measured again, with the comparison as it is now
                t["closeness"] = round(max(align(t["sounds"].split(), r["phones"], epenthetic_source=True)["closeness"] for r in readings), 3)
        if letters:
            for t in tried:
                if t["accepted"]:
                    t["tail_ms"] = _tail_ms(lab, t["spelling"])
        written = tried[0]["spelling"] if tried else record["word"]
        decision = decide(written, tried, letters=letters)
        if (decision["status"], decision.get("spelling")) != (record.get("status"), record.get("spelling")):
            changed.append({**{k: v for k, v in record.items() if k not in ("status", "spelling", "closeness", "held", "tried")}, **decision, "tried": tried})
    return changed


def build(lab: Any, words: Sequence[str], out_path: Path, *, shard: int = 0, shards: int = 1, progress: Callable[[str], None] = print,
          forms: Mapping[str, Mapping[str, str]] | None = None, again_letters: bool = False) -> int:
    """Evaluate this shard's words and append one JSON line per word to ``out_path``; words already in the file are skipped (a stopped build resumes).  ``again_letters``
    evaluates the words said letter by letter once more (the way they are composed changed); the later line replaces the earlier."""
    out_path = Path(out_path)
    # a word that ended in an error (the service was down, the key failed) is tried again; only a decision is final
    done = ({r["word"] for r in (json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip())
             if r["status"] != "error" and not (again_letters and r.get("reads_as") == "letters" and r.get("build") != BUILD)}
            if out_path.is_file() else set())
    count = 0
    with out_path.open("a", encoding="utf-8") as handle:
        for index, word in enumerate(words):
            if index % shards != shard or word in done:
                continue
            try:
                record = evaluate(lab, word, forms)
            except Exception as error:  # noqa: BLE001 - one word's failure must not stop the build
                record = {"word": word, "status": "error", "error": str(error)[:200]}
            record["voice"] = lab.voice
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            count += 1
            progress(f"{lab.voice} {word}: {record['status']}" + (f" -> {record['spelling']} ({record['closeness']})" if record.get("spelling") else ""))
    return count


RULES_BUILD = "rules-only-1"


def build_rules(interpreter: Any, words: Sequence[str], voices: Sequence[str], directory: Path) -> dict[str, int]:
    """The lexicon by rule alone: each word's spelling from its dictionary (or researched) sounds through ``zulu_rules``, the same for every voice, with the derivation kept.
    Nothing is spoken, heard or scored; a person listens and the rules are changed.  One file per voice (``built_<voice>_rules.jsonl``, replaced each time) in the
    shape ``merge`` reads, so the kept spellings reach the shared memory the same way.  Returns how many words got a spelling and how many had no reading."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tally = {"spelled": 0, "no reading": 0}
    lines = []
    for word in words:
        found = interpreter.rules_spelling(word)
        if found is None:
            lines.append({"word": word, "status": "no reading", "build": RULES_BUILD})
            tally["no reading"] += 1
        else:
            lines.append({"word": word, "status": "respell", "spelling": found["spelling"], "closeness": None, "reads_as": found["reading"], "build": RULES_BUILD,
                          "derivation": found["derivation"], "english": " ".join(found["english"])})
            tally["spelled"] += 1
    for voice in voices:
        (directory / f"built_{voice}_rules.jsonl").write_text("".join(json.dumps({**line, "voice": voice}, ensure_ascii=False) + "\n" for line in lines), encoding="utf-8")
    return tally


def merge(results: Iterable[Path], memory_file: Path, *, job_id: str = "lexicon") -> dict[str, int]:
    """Put the kept spellings into the pronunciation memory, for the voice each was built on; returns how many words ended in each status."""
    from .pronunciation_memory import forget_lexicon, remember

    latest: dict[tuple[str, str], dict[str, Any]] = {}            # the last line for a word and voice stands: a retry replaces an error
    for path in results:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                latest[(record["word"], record["voice"])] = record
    tally: collections.Counter = collections.Counter(record["status"] for record in latest.values())
    forget_lexicon(memory_file, [(r["word"], r["voice"]) for r in latest.values() if r["status"] == "respell"])      # the latest build's choice replaces the lexicon's earlier one
    for record in latest.values():
        if record["status"] == "respell":
            remember(memory_file, record["word"], record["spelling"], source="lexicon", voice=record["voice"], job_id=job_id, kind="english_code_switch",
                     closeness=record["closeness"], note=f"built in advance for {record['voice']}")
    return dict(tally)


def main(argv: list[str] | None = None) -> int:
    import argparse
    import os

    parser = argparse.ArgumentParser(prog="mathula_tv.lexicon_builder")
    sub = parser.add_subparsers(dest="command", required=True)
    words_cmd = sub.add_parser("words", help="write the word list from the Zulu corpus")
    words_cmd.add_argument("--work-dir", required=True)
    letters_cmd = sub.add_parser("letters", help="find how one voice says each letter of the alphabet (one shard)")
    letters_cmd.add_argument("--job-root", required=True)
    letters_cmd.add_argument("--voice", required=True)
    letters_cmd.add_argument("--shard", type=int, default=0)
    letters_cmd.add_argument("--shards", type=int, default=1)
    build_cmd = sub.add_parser("build", help="speak and judge the candidates for one voice (one shard)")
    build_cmd.add_argument("--job-root", required=True)
    build_cmd.add_argument("--voice", required=True)
    build_cmd.add_argument("--shard", type=int, default=0)
    build_cmd.add_argument("--shards", type=int, default=1)
    build_cmd.add_argument("--limit", type=int, default=0)
    build_cmd.add_argument("--again-letters", action="store_true", help="evaluate the words said letter by letter once more")
    build_cmd.add_argument("--words", default="", help="build these words (comma separated, any case) instead of the corpus list: a clip's names, for example")
    redecide_cmd = sub.add_parser("redecide", help="choose again among the spellings already judged, with the length rule for letters (no listener is asked)")
    redecide_cmd.add_argument("--job-root", required=True)
    redecide_cmd.add_argument("--voice", required=True)
    redecide_cmd.add_argument("--shard", type=int, default=0)
    redecide_cmd.add_argument("--dry", action="store_true", help="print the changed choices and write nothing")
    redecide_cmd.add_argument("--names", action="store_true", help="the file of a chosen list of words (build --words), not the corpus list")
    rules_cmd = sub.add_parser("rules", help="spell words by rule alone from their dictionary sounds: no voice, no listener, no score (writes the files merge reads)")
    rules_cmd.add_argument("--work-dir", required=True)
    rules_cmd.add_argument("--job-root", default="", help="a job whose web research supplies sounds for its names")
    rules_cmd.add_argument("--words", default="", help="comma separated words, any case, instead of the corpus list")
    rules_cmd.add_argument("--voices", default="zu-ZA-ThandoNeural,zu-ZA-ThembaNeural")
    merge_cmd = sub.add_parser("merge", help="put the kept spellings into the shared memory")
    merge_cmd.add_argument("--work-dir", required=True)
    args = parser.parse_args(argv)
    if args.command == "words":
        directory = Path(args.work_dir) / "resources" / "lexicon"
        directory.mkdir(parents=True, exist_ok=True)
        found = collect(Path(args.work_dir))
        (directory / "words.json").write_text(json.dumps(found, ensure_ascii=False, indent=1), encoding="utf-8")
        print(len(found["initialisms"]), "initialisms,", len(found["words"]), "words ->", directory / "words.json")
        return 0
    if args.command == "rules":
        from .english_sounds import EnglishSounds, g2p_guesser
        from .interpreter import Interpreter
        from .zulu_rules import research_readings

        work_dir = Path(args.work_dir)
        directory = work_dir / "resources" / "lexicon"
        sounds = EnglishSounds.load(work_dir / "resources" / "cmudict" / "cmudict.dict")
        sounds.guesser = g2p_guesser()
        research_file = Path(args.job_root) / "native_dub" / "pronunciation_research.json" if args.job_root else None
        researched = research_readings(json.loads(research_file.read_text(encoding="utf-8"))) if research_file and research_file.is_file() else {}
        found = json.loads((directory / "words.json").read_text(encoding="utf-8"))
        every = [w.strip() for w in args.words.split(",") if w.strip()] or [*found["initialisms"], *found["words"]]
        print(build_rules(Interpreter(sounds, {}, None, researched), every, [v for v in args.voices.split(",") if v], directory), "->", directory)
        return 0
    if args.command in ("letters", "build", "redecide"):
        from .pronunciation_lab import PronunciationLab

        job_root = Path(args.job_root)
        work_dir = job_root.parents[1]
        directory = work_dir / "resources" / "lexicon"
        # redecide reads the audio the build kept, so it uses the build's folder
        lab = PronunciationLab(job_root, voice=args.voice, key=os.environ["AZURE_SPEECH_KEY"], region=os.environ["AZURE_SPEECH_REGION"].strip().lower(),
                               work_dir=Path("/tmp") / f"lexicon_{'build' if args.command == 'redecide' else args.command}_{args.voice}_{args.shard}")
        if args.command == "redecide":
            path = directory / f"built_{args.voice}_{'names' if args.names else ''}{args.shard}.jsonl"
            latest = {}
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    record = json.loads(line)
                    latest[record["word"]] = record
            changed = redecide(lab, latest.values())
            with path.open("a", encoding="utf-8") as handle:
                for record in changed:
                    if not args.dry:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    before = latest[record["word"]]
                    print(f"{args.voice} {record['word']}: {before['status']} {before.get('spelling') or ''} -> {record['status']} {record.get('spelling') or ''}", flush=True)
            print(len(changed), "choices changed" + (" (dry run: nothing written)" if args.dry else ""))
            return 0
        if args.command == "letters":
            out = directory / f"letters_{args.voice}_{args.shard}.jsonl"
            print(build_letters(lab, out, shard=args.shard, shards=args.shards, progress=lambda m: print(m, flush=True)), "letters built")
            return 0
        found = json.loads((directory / "words.json").read_text(encoding="utf-8"))
        every = [w.strip() for w in args.words.split(",") if w.strip()] or [*found["initialisms"], *found["words"]]
        if args.limit:
            every = every[:args.limit]
        forms = letter_forms(json.loads(line) for path in sorted(directory.glob(f"letters_{args.voice}_*.jsonl")) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        print(len(forms), "letters known for", args.voice)
        # a list of chosen words goes to its own file, which a merge reads after the corpus lists and so lets stand
        out = directory / f"built_{args.voice}_{'names' if args.words else ''}{args.shard}.jsonl"
        print(build(lab, every, out, shard=args.shard, shards=args.shards, progress=lambda m: print(m, flush=True), forms=forms, again_letters=args.again_letters), "words built")
        return 0
    from .pronunciation_memory import memory_path

    work_dir = Path(args.work_dir)
    print(merge(sorted((work_dir / "resources" / "lexicon").glob("built_*.jsonl")), memory_path(work_dir)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
