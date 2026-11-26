"""OmniVoice (the Zulu fine-tune on Google Cloud Run) as the voice of the dubbing pipeline, in place of Azure Speech.

``OmniVoiceTTSBackend`` takes over ``native_dub._synthesize_group`` (that function hands a group to a backend that has ``synthesize_group``), so the agent's
``measure_tts`` and the episode scorer speak every candidate through it, and what the agent measures is what it is scored on, as before.

What differs from Azure, and what the backend does about it:
* there is no SSML: the pronunciation dictionary's typed parts are flattened to text (a respelling becomes its spoken form, an acronym stays as written),
  and a held ``~pause`` or a break becomes real silence between two generations;
* the voice is CLONED from a reference clip, and the clip also sets the pace (OmniVoice sizes its output from the reference's speaking rate).  The reference is cut
  from the job's own English audio, inside the window being fitted, because a diarization label can cover two people (the anchor and the reporter were both
  SPEAKER_00 on 2026-11-18) and one clip per label would give the anchor the reporter's voice;
* output is stochastic, so every measurement is written to its content-addressed path and reused: the audio that was measured is the audio that ships.
"""
from __future__ import annotations

import base64
from difflib import SequenceMatcher
import hashlib
import io
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import soundfile as sf

from . import omnivoice_duration as od
from . import abbreviations

CLIP_MIN_S = 3.0
CLIP_MAX_S = 9.0
NAME_SIMILARITY = 0.8        # how alike (spelt as the voice would say them) a name of the dub and the words the speaker said must be: "Canny Maphanga" and the recogniser's "Kanima Panga" are 0.87, "Ayanda Nyathi" and "Ayanda Nyati" 0.96, unrelated words under 0.3
NAME_SIMILARITY_ONE_WORD = 0.85
NAME_SIMILARITY_LONG = 0.65          # a long name (NAME_LONG_LETTERS letters or more) said alone: an unrelated word is almost never this alike to it, and the recogniser's guesses at a long place name are far from it ("Likuhuli" and "Iguruleni" for Ekurhuleni: 0.67, 0.74)
NAME_LONG_LETTERS = 8
NAME_SKELETON_MIN = 0.6              # a name that passes only by the lowered bars must also share this much of its consonants with the word heard
NAME_CONNECTORS = frozenset({"and", "&", "of", "the", "in", "at"})          # a word the speaker says between two names that the dub joins with a Zulu prefix ("Tswane and Johannesburg": "e-Tshwane nase-Johannesburg"): it is in the clip and in its transcript
PRIMER_REPS = 3              # a name the source says is put in front of the reference this many times (heard on "Phineas": once 0 of 3 right, three times 2 of 3, and 3 of 3 with the names first)
PRIMER_MAX_S = 9.0           # the names, all repetitions together, never take more of the reference than this (the model asks for 3-10 s in all; the clip it is put in front of is about 8.5 s)
PRIMER_MAX_SNIPPET_S = 4.0   # a run of names longer than this was cut wrongly
PRIMER_SLOW_FACTOR = 2.0      # a name the speaker draws out to more than this many times what it takes at their usual pace is not the one to put in front of the reference: it drags the pace of the whole primed sentence
PRIMER_SLOW_MIN_S = 0.6       # (and is at least this long: "ANC" for 1.26 s in prod-s2 unit 2, 0.36-0.40 s everywhere else: the primed reference ran at 10.7 characters a second instead of 13.5 and the voice took 25% longer over two sentences)
PRIMER_ALIGN_CONTEXT = 3      # words of the source either side of a name that are aligned with it, so the aligner has the sentence around it
PRIMER_ALIGN_LEAD_S = 0.05   # what is kept in front of the aligned name, and
PRIMER_ALIGN_TAIL_S = 0.10   # behind it (the aligner puts a word's edges a little inside the sound; the last letter's release carries on)
PRIMER_GAP_S = 0.05          # the least silence between the copies (more reads to the model as a slower speaker: 0.25 s made the sentence 13% longer, 0.05 s 6%); ReferenceBank.primed sets the real gap from the clip's pace
PRIMER_GAP_MAX_S = 0.6       # and the most
CLIP_TARGET_S = 8.0          # a longer clip says more about the voice; 3-10 s is what the model asks for
REFERENCE_RATE = 24000
DEFAULT_NUM_STEP = 64         # refinement steps of every synthesis: the agent's measurements, its submissions, the render and the Studio (32 is the model's default; 64 gave the tightest draws and a slower reading of a rushed word, and one setting everywhere means the first measurement is the final audio)
SECONDS_PER_COPY = 20.0       # a copy of the voice service (one GPU) for each cluster's worth of speech (CLUSTER_MAX_S) of source audio, held for the length of a job (omnivoice_scale): a unit's spans are spoken side by side, one per copy
MAX_COPIES = 4                # the service's own cap (--max-instances), well under the project's L4 quota
DEFAULT_SPEED = 1.0           # the voice speaks at the pace of its reference clip, which is the speaker's own: no slowing, no stretching
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-ZÀ-ɏ\"'(])")
MAX_SPAN_CHARS = 280          # about 15-20 s of speech: what one generation was judged to hold; a longer sentence is still one span
MIN_SENTENCE_CHARS = 20       # a sentence shorter than this is spoken together with the next one
SPAN_GAP_MS = 150


@dataclass(frozen=True)
class Reference:
    wav: bytes
    text: str
    seconds: float
    start_s: float
    end_s: float
    speaker: str


def syllables(text: str) -> int:
    """A rough syllable count of a text, for pace: an acronym says its letters (W says three), any other word its groups of vowels."""
    total = 0
    for word in re.findall(r"[A-Za-z']+", str(text or "")):
        if word.isupper() and len(word) <= 5:
            total += len(word) + word.count("W") * 2
            continue
        low = word.lower()
        groups = len(re.findall(r"[aeiouy]+", low))
        if low.endswith("e") and groups > 1 and not low.endswith(("le", "ne", "se", "me")):
            groups -= 1
        total += max(groups, 1)
    return total


def stretch_audio(piece: np.ndarray, rate: int, factor: float) -> np.ndarray:
    """``piece`` made ``factor`` times longer without changing its pitch (ffmpeg's atempo, which takes 0.5 to 2 a step); the piece itself when ffmpeg is not there or fails."""
    if abs(factor - 1.0) < 0.02:
        return piece
    import shutil
    import tempfile

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return piece
    tempo, steps = 1.0 / factor, []
    while tempo < 0.5:
        steps.append("atempo=0.5")
        tempo /= 0.5
    while tempo > 2.0:
        steps.append("atempo=2.0")
        tempo /= 2.0
    steps.append(f"atempo={tempo:.4f}")
    try:
        with tempfile.TemporaryDirectory() as folder:
            source, target = os.path.join(folder, "a.wav"), os.path.join(folder, "b.wav")
            sf.write(source, piece, rate, subtype="PCM_16")
            subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", source, "-filter:a", ",".join(steps), target], check=True, timeout=30)
            out, out_rate = sf.read(target, dtype="float32")
        return (out.mean(axis=1) if out.ndim > 1 else out) if out_rate == rate else piece
    except Exception:                    # noqa: BLE001 - a primer is an optimisation of the reference: unstretched is what it was
        return piece


class ReferenceBank:
    """English reference clips for one job, cut from its own audio: sentence-bounded runs of one speaker's words, 3-9 s long, with their transcript."""

    def __init__(self, words: Sequence[Mapping[str, Any]], speaker_of_raw: Mapping[str, str], audio_path: Path, cache_dir: Path | None = None) -> None:
        self.audio_path = Path(audio_path)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.words = list(words)                       # the source's timed words: where its speakers paused (span_cuts)
        self.speaker_of_raw = {str(k): str(v) for k, v in speaker_of_raw.items()}
        self.aligner: Any = None                          # (text, mono audio, rate) -> timed words with letters (source_align.shared_aligner): where a name the speaker says falls in their audio, from its spelling
        self.known_names: list[list[str]] = []          # the job's own names (its pronunciation research): "Canny" alone is the start of "Canny Maphanga"
        self._candidates = self._build(words, speaker_of_raw)
        self._pace = self._speaker_pace(self._candidates, words, speaker_of_raw)
        self._tone: tuple[np.ndarray, float] | None = None

    @staticmethod
    def _speaker_pace(candidates: Sequence[Mapping[str, Any]], words: Sequence[Mapping[str, Any]] = (), speaker_of_raw: Mapping[str, str] | None = None) -> dict[str, float]:
        """Each speaker's usual words a second, and "" for everyone: what a fluent clip of that voice should sound like.  It is the speaker's pace over ALL their speech (their words divided by the time
        they spend speaking, runs of words with no stop over 0.5 s between them), not the median of the clips found: the clips are short fluent sentences with no hesitation, so on a clip with few of
        them the median sits well above how the speaker really speaks (Mbalula clip: median of five clips 2.84 words/s, the speaker 2.45), and the voice takes its pace from the reference it is shown."""
        paces: dict[str, list[float]] = {"": []}
        for c in candidates:
            paces.setdefault(str(c["speaker"]), []).append(float(c["wps"]))
            paces[""].append(float(c["wps"]))
        usual = {speaker: float(np.median(values)) for speaker, values in paces.items() if values}
        spoken: dict[str, list[float]] = {}                                           # speaker -> [words, seconds of speech]
        run: list[Mapping[str, Any]] = []

        def close() -> None:
            if run:
                for key in (str((speaker_of_raw or {}).get(str(run[0].get("raw_speaker")), run[0].get("raw_speaker"))), ""):
                    tally = spoken.setdefault(key, [0.0, 0.0])
                    tally[0] += len(run)
                    tally[1] += float(run[-1]["end"]) - float(run[0]["start"])
                run.clear()

        for word in words:
            if word.get("overlap") or (run and (word["raw_speaker"] != run[-1]["raw_speaker"] or float(word["start"]) - float(run[-1]["end"]) > 0.5)):
                close()
                if word.get("overlap"):
                    continue
            run.append(word)
        close()
        for key, (count, seconds) in spoken.items():
            if count >= 20 and seconds >= 5.0:                                       # enough speech to know how they speak; else the clips' median stands
                usual[key] = count / seconds
        return usual

    @classmethod
    def for_corpus(cls, corpus_root: Path | str) -> "ReferenceBank":
        """From an experience corpus folder (``working/experience/corpus/<clip>``): its transcript, and the audio of the job it was made from."""
        root = Path(corpus_root)
        transcript = json.loads((root / "analysis" / "transcript_en_raw.json").read_text(encoding="utf-8"))
        speaker_of_raw = {str(s.get("raw_speaker")): str(s.get("speaker_id") or s.get("speaker")) for s in transcript.get("segments") or [] if s.get("raw_speaker")}
        work = root.parents[2] if len(root.parents) > 2 else root
        jobs = work / "jobs"
        audio = None
        for job in sorted(jobs.glob(f"{root.name}*")) if jobs.is_dir() else []:
            for name in ("mix_source_stereo.wav", "analysis_mono.wav"):
                if (job / "audio" / name).is_file():
                    audio = job / "audio" / name
                    break
            if audio:
                break
        if audio is None:
            raise FileNotFoundError(f"no source audio found for corpus {root.name} under {jobs}")
        bank = cls(transcript.get("words") or [], speaker_of_raw, audio, cache_dir=root / "omnivoice_refs")
        try:
            research = json.loads((root / "pronunciation_research.json").read_text(encoding="utf-8"))
            for entry in research.get("accepted_corrections") or []:
                tokens = [t for t in re.split(r"\s+", str(entry.get("canonical_text") or "")) if len(t) >= 3 and t[0].isupper()]
                if len(tokens) > 1 and str(entry.get("entity_type") or "") == "person":
                    bank.known_names.append(tokens)
        except (OSError, ValueError):
            pass
        try:
            restored = json.loads((root / "native_dub" / "pass1_restored_transcript.json").read_text(encoding="utf-8"))
            bank.restore(restored.get("segments") or [])
        except (OSError, ValueError):
            pass
        return bank

    def restore(self, segments: Sequence[Mapping[str, Any]]) -> int:
        """Put the job's own corrected spelling on the recogniser's words, so the name search compares the dub's name with what the job already knows the speaker said ("kenny mapanga" was restored to
        "Canny Maphanga" in pass 1) and not with a raw guess.  Each restored segment is lined up word for word with the raw words inside its window; only a word the restoration replaced by exactly
        one word (or a run by a run of the same length) gets ``restored``, and a segment whose words do not line up one for one is left as the recogniser heard it.  Returns the words restored."""
        def tokens(text: str) -> list[str]:
            return [t for t in (re.sub(r"[^\w']", "", raw).lower() for raw in str(text or "").split()) if t]

        count = 0
        for segment in segments:
            if not segment.get("changed", True):
                continue
            start, end = float(segment.get("start_ms") or 0) / 1000.0 - 0.05, float(segment.get("end_ms") or 0) / 1000.0 + 0.05
            inside = sorted((w for w in self.words if start <= float(w.get("start") or 0.0) < end), key=lambda w: float(w["start"]))
            heard, written = tokens(segment.get("source_text")), str(segment.get("restored_text") or "").split()
            if not inside or len(heard) != len(inside) or [t for t in tokens(" ".join(str(w["text"]) for w in inside))] != heard:
                continue                                             # the words do not line up one for one: no guessing
            clean = [re.sub(r"^[^\w]+|[^\w]+$", "", raw) for raw in written]
            kept = [(raw, c) for raw, c in zip(written, clean) if c]
            for tag, i1, i2, j1, j2 in SequenceMatcher(None, heard, [c.lower() for _, c in kept], autojunk=False).get_opcodes():
                if tag == "replace" and i2 - i1 == j2 - j1:
                    for offset in range(i2 - i1):
                        inside[i1 + offset]["restored"] = kept[j1 + offset][1]
                        count += 1
        # a name the recogniser misheard that the job corrected in one place is the same name wherever it spelt it that way again (prod-s1: "Iguruleni" was restored to Ekurhuleni in the first unit and left as
        # heard in the second speaker's, 0.74 alike to Ekurhuleni, under the bar for one word, so the second speaker's Ekurhuleni was never primed)
        respelt: dict[str, set[str]] = {}
        for word in self.words:
            fixed = str(word.get("restored") or "")
            heard = re.sub(r"[^\w']", "", str(word["text"])).lower()
            if fixed[:1].isupper() and len(heard) >= 5 and heard != fixed.lower() and SequenceMatcher(None, self._sound_key(heard), self._sound_key(fixed)).ratio() >= 0.7:
                respelt.setdefault(heard, set()).add(fixed)
        for word in self.words:
            heard = re.sub(r"[^\w']", "", str(word["text"])).lower()
            if not word.get("restored") and len(respelt.get(heard, ())) == 1:          # only where the job corrected it one way
                word["restored"] = next(iter(respelt[heard]))
                count += 1
        return count

    @staticmethod
    def _said(word: Mapping[str, Any]) -> str:
        """What the speaker said in this word, as the job restored it, else as the recogniser heard it."""
        return str(word.get("restored") or word["text"])

    @staticmethod
    def _build(words: Sequence[Mapping[str, Any]], speaker_of_raw: Mapping[str, str]) -> list[dict[str, Any]]:
        runs: list[list[Mapping[str, Any]]] = []
        current: list[Mapping[str, Any]] = []
        for word in words:
            if word.get("overlap"):
                current = []
                continue
            if current and (word["raw_speaker"] != current[-1]["raw_speaker"] or float(word["start"]) - float(current[-1]["end"]) > 0.5):
                current = []
            if not current:
                runs.append(current)
            current.append(word)
        found: list[dict[str, Any]] = []
        order = {id(w): i for i, w in enumerate(words)}
        for run in runs:
            first = 0
            for index, word in enumerate(run):
                if str(word["text"]).rstrip().endswith((".", "?", "!")) or index == len(run) - 1:
                    part = list(run[first:index + 1])
                    first = index + 1
                    while part and float(part[-1]["end"]) - float(part[0]["start"]) > CLIP_MAX_S and len(part) > 3:
                        part.pop()
                    if not part:
                        continue
                    start, end = float(part[0]["start"]), float(part[-1]["end"])
                    if end - start < CLIP_MIN_S or end - start > CLIP_MAX_S:          # trimming stops at three words: a run still longer than the most a clip may be (a few words the recogniser stretched over half a minute) is not a clip of anyone's voice
                        continue
                    raw = str(part[0]["raw_speaker"])
                    at = order.get(id(part[0]), 0)
                    last_at = order.get(id(part[-1]), len(words) - 1)
                    prev_end = float(words[at - 1]["end"]) if at > 0 else start - 0.3
                    next_start = float(words[last_at + 1]["start"]) if last_at + 1 < len(words) else end + 0.3
                    gaps = sum(1 for a, b in zip(part, part[1:]) if float(b["start"]) - float(a["end"]) > 0.25)
                    found.append({"start": start, "end": end, "text": " ".join(str(w["text"]) for w in part), "wps": len(part) / max(end - start, 0.1), "gaps": gaps,
                                  "prev_end": min(prev_end, start), "next_start": max(next_start, end),
                                  "confidence": float(np.mean([float(w.get("confidence") or 0.0) for w in part])),
                                  "speaker": speaker_of_raw.get(raw, raw)})
        return found

    def _score(self, candidate: Mapping[str, Any]) -> float:
        """How good a clip is to clone: clearly transcribed, fluent (no hesitation gaps), at the speaker's USUAL pace (a slow or rushed clip teaches the model that pace, and its pauses), and long
        enough to say something about the voice."""
        usual = self._pace.get(str(candidate["speaker"])) or self._pace.get("") or float(candidate["wps"])
        return (float(candidate["confidence"]) - 1.0 * abs(float(candidate["wps"]) / usual - 1.0) - 0.3 * int(candidate["gaps"])
                - 0.02 * abs((candidate["end"] - candidate["start"]) - CLIP_TARGET_S))

    @staticmethod
    def _sound_key(text: str) -> str:
        """A name spelt roughly as it sounds, so that the dub's spelling and a recogniser's guess at the same name can be compared (c, q -> k, ph -> f, y -> i, w -> u; letters only)."""
        t = str(text).lower()
        for old, new in (("ph", "f"), ("ck", "k"), ("c", "k"), ("q", "k"), ("x", "ks"), ("y", "i"), ("w", "u")):
            t = t.replace(old, new)
        return "".join(ch for ch in t if ch.isalpha())

    @staticmethod
    def _skeleton(key: str) -> str:
        """The consonants of a sound key (a recogniser's guess at a name keeps its consonants better than its vowels)."""
        return re.sub(r"[aeiou]", "", key)

    TITLES = abbreviations.title_words()          # said before a name (Mnu, Dr, Prof ... and written out); the speaker's own words for them are "Mr", "Doctor": never part of the name

    @staticmethod
    def dub_name_runs(text: str) -> list[list[str]]:
        """The runs of name-like words in the dub text (a capital, three letters or more; or an English word in lower case after a hyphenated Zulu prefix), with a Zulu prefix taken off ("uCanny" and "u-Phineas" -> Canny, Phineas; "we-JMPD" -> JMPD).  A comma or a full stop ends a run.
        Zulu words that merely begin a sentence are in here too: they are only names when the speaker said something like them (name_matches)."""
        runs: list[list[str]] = []
        current: list[str] = []
        previous = ""
        for raw in str(text or "").split():
            core = re.sub(r"^[^\w]+|[^\w]+$", "", raw)
            stem = core.split("-")[-1]
            spaced = abbreviations.english_stem(previous.lower(), core)          # "ama details": the same word after the space the voice text is written with
            previous = core
            if spaced or ("-" in core and stem.isalpha() and stem.islower() and len(stem) >= 4):          # an English word kept in English after a Zulu class prefix ("ama-details", "ezingu-forty"): when the speaker says it, its audio is the pronunciation, as a name's is (prod-x1: the voice dropped the t of "details" in every draw)
                if current:
                    runs.append(current)
                runs.append([stem])
                current = []
                continue
            core = stem
            core = re.sub(r"^(?:[a-z]+|[A-Z])(?=[A-Z][a-z])", "", core)          # uCanny, kwamaHawks (a prefix of any length), uClive (a capital U at the start of a sentence), but not JMPD
            ends = raw.rstrip().endswith((",", ".", "?", "!", ";", ":"))
            if core.lower() in ReferenceBank.TITLES:          # "Mnu Mashaba": the name is Mashaba, and the title is spelled out by the dictionary (a title taken into the name was protected from it and read as "mnu")
                if current:
                    runs.append(current)
                current = []
                continue
            if len(core) >= 3 and core[0].isupper():
                current.append(core)
                if ends:
                    runs.append(current)
                    current = []
            else:
                if current:
                    runs.append(current)
                current = []
        if current:
            runs.append(current)
        return runs

    def name_matches(self, speaker_id: str, start_s: float, end_s: float, text: str) -> list[dict[str, Any]]:
        """The names of the dub text that the speaker also says in the window, found by how they sound and not by how they are spelt: [{tokens, text, first, last, source_text, similarity}], longest first.
        (The recogniser heard "Canny Maphanga" as "Kanima Panga"; the dub spells it Canny Maphanga: 0.87 alike.)"""
        words = [(i, w) for i, w in enumerate(self.words)
                 if not w.get("overlap") and float(w.get("start") or 0.0) >= start_s - 0.05 and float(w.get("end") or 0.0) <= end_s + 0.05
                 and self.speaker_of_raw.get(str(w.get("raw_speaker")), str(w.get("raw_speaker"))) == speaker_id]
        if not words:
            return []
        found: list[dict[str, Any]] = []
        for run in self.dub_name_runs(text):
            if len(run) == 1:                              # "Canny," on its own: the job knows her whole name, and the speaker said it whole
                whole = next((full for full in self.known_names if run[0].lower() in [t.lower() for t in full]), None)
                run = list(whole) if whole else run
            used = [False] * len(run)
            for length in range(len(run), 0, -1):
                for at in range(0, len(run) - length + 1):
                    if any(used[at: at + length]):
                        continue
                    tokens = run[at: at + length]
                    key = self._sound_key(" ".join(tokens))
                    if len(key) < 3:
                        continue
                    best, best_at, best_inside, best_counted = 0.0, None, None, ""
                    for n in ((length, length + 1) if length > 1 else (1, 2)):          # as many words as the name has: a word of the text that the audio does not say, or the other way round, would teach the model the wrong thing; the exceptions are a connector between two names ("Tswane and Johannesburg") and a one-word name the recogniser wrote as two ("Action SA")
                        for j in range(0, len(words) - n + 1):
                            window = words[j: j + n]
                            if any(window[k + 1][0] != window[k][0] + 1 for k in range(n - 1)):
                                continue
                            inside = None
                            counted = [self._said(w) for _, w in window]
                            if n > length and length > 1:
                                inside = next((q for q in range(1, n - 1) if re.sub(r"\W", "", counted[q]).lower() in NAME_CONNECTORS), None)
                                if inside is None:
                                    continue
                                counted = [x for q, x in enumerate(counted) if q != inside]
                            joiner = "" if (n > length and length == 1) else " "
                            similarity = SequenceMatcher(None, key, self._sound_key(joiner.join(counted))).ratio()
                            if similarity > best + (0.04 if (n > length and length == 1) else 0.0):          # the two-word reading must be clearly better than the one word
                                best, best_at, best_inside, best_counted = similarity, window, inside, joiner.join(counted)
                    needed = NAME_SIMILARITY if length > 1 else (NAME_SIMILARITY_LONG if len(key) >= NAME_LONG_LETTERS else NAME_SIMILARITY_ONE_WORD)
                    if best_at is None or best < needed or (length == 1 and len(key) < 4 and not tokens[0].isupper()):
                        continue
                    if length == 1 and best < NAME_SIMILARITY_ONE_WORD and SequenceMatcher(None, self._skeleton(key), self._skeleton(self._sound_key(best_counted))).ratio() < NAME_SKELETON_MIN:
                        continue                                                       # "national" is 0.75 like ActionSA by its letters but shares half of its consonants; the recogniser's real guesses ("Likuhuli", "Iguruleni" for Ekurhuleni) share two thirds
                    for k in range(at, at + length):
                        used[k] = True
                    spoken = list(tokens)
                    if best_inside is not None:                                       # the connector is in the clip, so it is in the transcript the model reads, between the names it joins
                        spoken.insert(best_inside, re.sub(r"^\W+|\W+$", "", self._said(best_at[best_inside][1])))
                    found.append({"tokens": tokens, "text": " ".join(spoken), "first": best_at[0][0], "last": best_at[-1][0],
                                  "source_text": " ".join(str(w["text"]) for _, w in best_at), "restored_text": " ".join(self._said(w) for _, w in best_at), "similarity": round(best, 2)})
        return sorted(found, key=lambda m: -len(m["tokens"]))

    def primer_for(self, speaker_id: str, start_s: float, end_s: float, span_text: str, usual_pace: bool = True, aligned: bool = True) -> dict[str, Any] | None:
        """The names this span says that the speaker also says in the source window, cut from the source in the speaker's own voice: {names, snippets: [(text, wav, seconds)], reps, seconds}, or None.
        The snippet's text is the dub's spelling of the name (what the audio really says), not the recogniser's.  The model copies how a name is said from its reference when the reference carries that
        word, and one copy is not enough (0 of 3 right), three are (2 of 3; 3 of 3 with them first)."""
        matches = self.name_matches(speaker_id, start_s, end_s, span_text)
        snippets, seen = [], set()
        for m in matches:
            key = " ".join(self._sound_key(t) for t in m["tokens"])
            if key in seen or any(key in other for other in seen):                  # the same name again: the first time is enough
                continue
            seen.add(key)
            a, z = m["first"], m["last"]
            prev_end = float(self.words[a - 1]["end"]) if a > 0 else float(self.words[a]["start"]) - 0.3
            next_start = float(self.words[z + 1]["start"]) if z + 1 < len(self.words) else float(self.words[z]["end"]) + 0.3
            first, last = self._edges_of(a, z)
            refined = self._aligned_edges(a, z, m["text"]) if aligned else None
            if refined is not None:
                first, last = refined
            expected = (len(m["text"]) + 1) / self.typical_chars_per_s(speaker_id)
            if usual_pace and last - first > max(PRIMER_SLOW_MIN_S, PRIMER_SLOW_FACTOR * expected):          # this time the speaker drew it out: take the speaker's usual saying of it from elsewhere in the job (not when only the name is used, as in a splice: the host keeps its pace and the drawn-out name is how the speaker said it)
                other = self._usual_saying(speaker_id, m, expected)
                if other is not None:
                    first, last = other
            if not 0.2 <= last - first <= PRIMER_MAX_SNIPPET_S:
                continue
            snippets.append((m["text"], self._clip(first, last), last - first))
        if not snippets:
            return None
        snippets.sort(key=lambda x: -len(x[0].split()))                                # the longest names first
        reps = PRIMER_REPS
        total = lambda: sum(x[2] for x in snippets) + PRIMER_GAP_S * len(snippets)
        while len(snippets) > 1 and total() * reps > PRIMER_MAX_S:                       # three copies are what works, so names are dropped before copies are
            snippets.pop()
        while reps > 1 and total() * reps > PRIMER_MAX_S:
            reps -= 1
        said = {re.sub(r"^\W+|\W+$", "", w).lower() for text, _, _ in snippets for w in text.split()} - NAME_CONNECTORS
        return {"names": sorted(said), "snippets": snippets, "reps": reps, "seconds": total() * reps}

    def _edges_of(self, a: int, z: int) -> tuple[float, float]:
        """The audio edges of the words ``a``..``z`` of the source (a primer is the whole word: the snap trims the quiet, and when it cut into the word the edges move outward only)."""
        prev_end = float(self.words[a - 1]["end"]) if a > 0 else float(self.words[a]["start"]) - 0.3
        next_start = float(self.words[z + 1]["start"]) if z + 1 < len(self.words) else float(self.words[z]["end"]) + 0.3
        edges = (float(self.words[a]["start"]), float(self.words[z]["end"]), min(prev_end, float(self.words[a]["start"])), max(next_start, float(self.words[z]["end"])))
        first, last = self.snap(*edges)
        if last - first < 0.2:
            first, last = self.snap(*edges, inward=0.0)
        return first, last

    def _aligned_edges(self, a: int, z: int, name_text: str) -> tuple[float, float] | None:
        """The audio edges of the name in the source words ``a``..``z``, from where its spelling sounds (forced alignment of the sentence around it, the name written as the dub writes it), or None when there is
        no aligner or it does not find the name plausibly (not 0.2-4 s, or farther than the recogniser's words by half a second).  The edges are the word's with a little kept either side, then moved to the
        quiet between it and its neighbours as any clip's are."""
        if self.aligner is None:
            return None
        try:
            lo, hi = max(0, a - PRIMER_ALIGN_CONTEXT), min(len(self.words) - 1, z + PRIMER_ALIGN_CONTEXT)
            t0, t1 = max(0.0, float(self.words[lo]["start"]) - 0.1), float(self.words[hi]["end"]) + 0.1
            info = sf.info(str(self.audio_path))
            audio, rate = sf.read(str(self.audio_path), start=int(t0 * info.samplerate), stop=int(t1 * info.samplerate), dtype="float32", always_2d=True)
            before = [str(self.words[i]["text"]) for i in range(lo, a)]
            after = [str(self.words[i]["text"]) for i in range(z + 1, hi + 1)]
            name = name_text.split()
            found = self.aligner(" ".join(before + name + after), audio.mean(axis=1), rate)
        except Exception:        # noqa: BLE001 - the recogniser's edges are the fallback
            return None
        if not isinstance(found, list) or len(found) != len(before) + len(name) + len(after):
            return None
        mine = found[len(before): len(before) + len(name)]
        if not name or any(w is None for w in mine):
            return None
        start, end = t0 + float(mine[0]["start"]), t0 + float(mine[-1]["end"])
        if not (0.2 <= end - start <= PRIMER_MAX_SNIPPET_S) or abs(start - float(self.words[a]["start"])) > 0.5 or abs(end - float(self.words[z]["end"])) > 0.6:
            return None
        previous = found[len(before) - 1] if before else None
        following = found[len(before) + len(name)] if after else None
        prev_end = min(t0 + float(previous["end"]), start) if previous else start - 0.3
        next_start = max(t0 + float(following["start"]), end) if following else end + 0.3
        first, last = self.snap(start - PRIMER_ALIGN_LEAD_S, end + PRIMER_ALIGN_TAIL_S, prev_end, next_start, inward=0.0)
        return (first, last) if last - first >= 0.2 else None

    def typical_chars_per_s(self, speaker_id: str) -> float:
        """How many characters a second this speaker's clean clips run at (their median), 13.5 when there are none."""
        rates = [len(c["text"]) / max(c["end"] - c["start"], 0.1) for c in self._candidates if c["speaker"] == speaker_id]
        return float(np.median(rates)) if rates else 13.5

    def _usual_saying(self, speaker_id: str, match: Mapping[str, Any], expected_s: float) -> tuple[float, float] | None:
        """Where else in the job this speaker says the same name (the same number of words, as alike as ``name_matches`` asks), as (first, last) audio edges: the one whose length is nearest what it takes at
        their usual pace and is not itself drawn out; None when the job has no other."""
        key = self._sound_key(" ".join(match["tokens"]))
        words = [(i, w) for i, w in enumerate(self.words) if not w.get("overlap") and self.speaker_of_raw.get(str(w.get("raw_speaker")), str(w.get("raw_speaker"))) == speaker_id]
        n = int(match["last"]) - int(match["first"]) + 1
        needed = NAME_SIMILARITY if len(match["tokens"]) > 1 else (NAME_SIMILARITY_LONG if len(key) >= NAME_LONG_LETTERS else NAME_SIMILARITY_ONE_WORD)
        best, best_score = None, None
        for j in range(0, len(words) - n + 1):
            window = words[j: j + n]
            if window[0][0] == match["first"] or any(window[k + 1][0] != window[k][0] + 1 for k in range(n - 1)):
                continue
            said = [self._said(w) for _, w in window if re.sub(r"\W", "", self._said(w)).lower() not in NAME_CONNECTORS]
            if SequenceMatcher(None, key, self._sound_key(" ".join(said))).ratio() < needed:
                continue
            first, last = self._edges_of(window[0][0], window[-1][0])
            if not 0.2 <= last - first <= max(PRIMER_SLOW_MIN_S, PRIMER_SLOW_FACTOR * expected_s):
                continue
            score = abs((last - first) / max(expected_s, 0.1) - 1.0)
            if best_score is None or score < best_score:
                best, best_score = (first, last), score
        return best

    def primed(self, base: Reference, primer: Mapping[str, Any] | None, match_syllables: float = 0.0) -> Reference:
        """The reference with the speaker's own names, each run repeated ``reps`` times, put in front of it (names first, then the fluent clip), and the transcript to match.

        The voice takes its pace from the whole reference: characters said per second.  Names cut from the source are said fast and tight, so a run of them made the voice speak 19% faster than the
        fluent clip does (prod-126, unit-01 sentence 3: 7.0 s against 8.6 s, 21 characters per second) and drop the weak syllables ("ngemisebenzi" -> "nomsebenzi").  The silence between the copies is
        therefore the time the names lack to be said at the clip's own pace, never less than PRIMER_GAP_S (names slower than the clip cannot be sped up) and never more than PRIMER_GAP_MAX_S."""
        if not primer:
            return base
        head, rate = sf.read(io.BytesIO(base.wav), dtype="float32")
        if head.ndim > 1:
            head = head.mean(axis=1)
        pieces: list[np.ndarray] = []
        texts: list[str] = []
        clip_syllables = syllables(base.text) / max(base.seconds, 0.1)          # how many syllables a second the reference clip is spoken at
        for _ in range(int(primer["reps"])):
            for text, wav, _seconds in primer["snippets"]:
                piece, piece_rate = sf.read(io.BytesIO(wav), dtype="float32")
                if piece_rate != rate:
                    return base
                piece = piece.mean(axis=1) if piece.ndim > 1 else piece
                if match_syllables > 1.0:          # a name said faster than the clip is spoken slows toward the clip's syllable speed (at most ``match_syllables`` times longer), pitch kept
                    faster = (syllables(text) / max(len(piece) / rate, 0.05)) / max(clip_syllables, 0.1)
                    if faster > 1.15:
                        piece = stretch_audio(piece, rate, min(faster, match_syllables))
                pieces.append(piece)
                texts.append(text)
        clip_pace = len(base.text) / max(base.seconds, 0.1)
        lacking = sum(len(t) + 1 for t in texts) / clip_pace - sum(len(x) for x in pieces) / rate
        gap = np.zeros(int(min(PRIMER_GAP_MAX_S, max(PRIMER_GAP_S, lacking / max(len(pieces), 1))) * rate), dtype=np.float32)
        audio = np.concatenate([part for piece in pieces for part in (piece, gap)] + [head])
        return Reference(wav=wav_bytes(audio, rate), text=" ".join(texts + [base.text]), seconds=len(audio) / rate, start_s=base.start_s, end_s=base.end_s, speaker=base.speaker)

    def ranked(self, speaker_id: str, start_ms: int, end_ms: int) -> list[dict[str, Any]]:
        """The clips to clone for a unit, best first.  The person talking in the unit's window is who is cloned (a merged label cannot give the wrong voice: the speaker of the best clip inside the window, else the
        unit's own label), and the clip is that person's BEST clip anywhere in the job, so they have one voice and one pace in every turn.  The unit's own window used to decide the clip, and every turn had its
        own reference and its own pace: the second speaker's 3.7 s turn held only a 3.6 s clip at 3.57 words/s (their usual is 2.75) and came out at 21 characters a second against 14 in their other turns (prod-s1).
        After that the other speakers' clips, the closest in time first."""
        start, end = start_ms / 1000.0, end_ms / 1000.0
        if not self._candidates:
            raise LookupError("the job has no clean 3-9 s clip of any speaker to clone")
        inside = sorted((c for c in self._candidates if c["start"] >= start - 0.05 and c["end"] <= end + 0.05), key=self._score, reverse=True)
        talking = inside[0]["speaker"] if inside else speaker_id
        mine = sorted((c for c in self._candidates if c["speaker"] == talking), key=lambda c: (-self._score(c), c["start"]))
        middle = (start + end) / 2.0
        others = sorted((c for c in self._candidates if c["speaker"] != talking), key=lambda c: (abs((c["start"] + c["end"]) / 2.0 - middle) + (0.0 if c["speaker"] == speaker_id else 25.0), -self._score(c)))
        return mine + others

    PAUSE_CLIPS = 40                      # the longest clips of the speaker, looked at for their pauses
    PAUSE_MIN_RUNS = 8                    # fewer pauses than this say nothing about the speaker

    def natural_pause_s(self, speaker_id: str, percentile: float = 95.0) -> float | None:
        """How long the speaker's own pauses run, in the quiet stretches (14 dB under their speech, at least 0.12 s) of their clean clips: the ``percentile`` of them, or None when there are too few to say.
        Read from the audio, because the transcript's word times run on without gaps (96% of them have none).  Two uses: a take of the voice is only marked down for quiet stretches longer than the 95th (the
        pauses the speaker makes are phrasing, not hesitation), and a full stop is given at least the 75th as its breath."""
        cache = self.__dict__.setdefault("_pause_cache", {})
        if speaker_id not in cache:
            from .internal_silence import FRAME_S, silent_runs

            mine = sorted((c for c in self._candidates if str(c["speaker"]) == speaker_id), key=lambda c: -(float(c["end"]) - float(c["start"])))[:self.PAUSE_CLIPS]
            lengths: list[float] = []
            for candidate in mine:
                try:
                    audio, rate = sf.read(io.BytesIO(self._clip(float(candidate["start"]), float(candidate["end"]))), dtype="float32")
                except Exception:                                          # noqa: BLE001 - a clip that cannot be read says nothing
                    continue
                audio = audio.mean(axis=1) if audio.ndim > 1 else audio
                lengths += [(last - first) * FRAME_S for first, last in silent_runs(audio, rate, min_run_s=0.12, margin_db=14.0)]
            cache[speaker_id] = lengths if len(lengths) >= self.PAUSE_MIN_RUNS else []
        runs = cache[speaker_id]
        return round(float(np.percentile(runs, percentile)), 3) if runs else None

    def room_tone(self) -> tuple[np.ndarray, float]:
        """The source recording's own floor of room sound and how far under its speech it sits (room_tone): what fills the joins between generations, cached beside the clips."""
        if self._tone is None:
            cached = self.cache_dir / "room_tone.npz" if self.cache_dir else None
            if cached is not None and cached.is_file():
                data = np.load(cached)
                from .room_tone import VERSION as TONE_VERSION

                if (int(data["version"]) if "version" in data.files else 1) >= TONE_VERSION:
                    self._tone = (data["tone"].astype(np.float32), float(data["delta_db"]))
            if self._tone is None:
                from .room_tone import find_room_tone

                self._tone = find_room_tone(self.audio_path)
                if cached is not None:
                    cached.parent.mkdir(parents=True, exist_ok=True)
                    from .room_tone import VERSION as TONE_VERSION

                    np.savez(cached, tone=self._tone[0], delta_db=self._tone[1], version=TONE_VERSION)
        return self._tone

    def choose(self, speaker_id: str, start_ms: int, end_ms: int) -> dict[str, Any]:
        return self.ranked(speaker_id, start_ms, end_ms)[0]

    def _levels(self) -> np.ndarray:
        """The source audio's level in dB per 10 ms (read once)."""
        if getattr(self, "_db", None) is None:
            audio, rate = sf.read(str(self.audio_path), dtype="float32", always_2d=True)
            mono = audio.mean(axis=1)
            hop = max(1, int(0.01 * rate))
            count = len(mono) // hop
            self._db = 20.0 * np.log10(np.sqrt((mono[: count * hop].reshape(count, hop) ** 2).mean(axis=1)) + 1e-9)
        return self._db

    def snap(self, start: float, end: float, prev_end: float, next_start: float, inward: float = 0.08) -> tuple[float, float]:
        """The clip's edges moved to the quietest point between it and its neighbouring words.  The recogniser's word times are rough, and a clip cut at them starts and ends in the middle of
        speech (20 of 25 clips of prod-119 ended within a few dB of the speech level); the model clones from the clip and an abrupt edge leaves a stray sound at the start of what it says."""
        db = self._levels()
        smooth = np.convolve(np.pad(db, 1, mode="edge"), np.ones(3) / 3, mode="valid") if len(db) else db

        def quietest(lo: float, hi: float, current: float) -> float:
            i0, i1 = max(0, int(lo * 100)), min(len(smooth) - 1, int(hi * 100))
            if i1 <= i0:
                return current
            best = i0 + int(np.argmin(smooth[i0:i1 + 1]))
            at = min(max(int(current * 100), 0), len(smooth) - 1)
            return best / 100.0 if smooth[best] <= smooth[at] - 3.0 else current              # only when it is clearly quieter than where the recogniser put the edge

        new_start = quietest(max(prev_end, start - 0.15), start + inward, start)          # ``inward`` is how far into the word an edge may move (the recogniser's times are rough); 0 for a short word, whose closed first sound and weak last sound
        new_end = quietest(end - inward, min(next_start, end + 0.15), end)                 # are the quietest points of it ("details", 0.34 s: cut to 0.18 s without its d and its s)
        return (new_start, new_end) if new_end - new_start <= CLIP_MAX_S + 0.3 else (start, end)

    def as_reference(self, chosen: Mapping[str, Any]) -> Reference:
        start, end = self.snap(chosen["start"], chosen["end"], float(chosen.get("prev_end", chosen["start"] - 0.3)), float(chosen.get("next_start", chosen["end"] + 0.3)))
        return Reference(wav=self._clip(start, end), text=chosen["text"], seconds=end - start, start_s=start, end_s=end, speaker=chosen["speaker"])

    def pick(self, speaker_id: str, start_ms: int, end_ms: int) -> Reference:
        chosen = self.choose(speaker_id, start_ms, end_ms)
        return Reference(wav=self._clip(chosen["start"], chosen["end"]), text=chosen["text"], seconds=chosen["end"] - chosen["start"],
                         start_s=chosen["start"], end_s=chosen["end"], speaker=chosen["speaker"])

    def _clip(self, start: float, end: float) -> bytes:
        cached = self.cache_dir / f"{int(start * 1000)}_{int(end * 1000)}.wav" if self.cache_dir else None
        if cached is not None and cached.is_file():
            return cached.read_bytes()
        info = sf.info(str(self.audio_path))
        audio, rate = sf.read(str(self.audio_path), start=int(start * info.samplerate), stop=int(end * info.samplerate), dtype="float32", always_2d=True)
        mono = audio.mean(axis=1)
        if rate != REFERENCE_RATE:
            if rate % REFERENCE_RATE == 0:
                factor = rate // REFERENCE_RATE
                mono = mono[: len(mono) // factor * factor].reshape(-1, factor).mean(axis=1)
            else:
                mono = np.interp(np.linspace(0, len(mono) - 1, int(len(mono) * REFERENCE_RATE / rate)), np.arange(len(mono)), mono).astype("float32")
        buffer = io.BytesIO()
        sf.write(buffer, mono, REFERENCE_RATE, format="WAV", subtype="PCM_16")
        data = buffer.getvalue()
        if cached is not None:
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_bytes(data)
        return data


def _group_text(group: Mapping[str, Any]) -> str:
    """The text a group is to say, as one string (its pauses left out): what names the dub carries."""
    return " ".join(str(part.get("text") or "") for part in group.get("parts") or [] if part.get("type") != "break")


PREFIXED_NAME = re.compile(r"\b([a-z]+)([A-Z][A-Za-z'\u2019]+)\b")


def separate_name_prefixes(text: str, names: "set[str] | frozenset[str]") -> str:
    """The text with a Zulu prefix written apart from a name the source says ("uPhineas" -> "u-Phineas", "kaManyama" -> "ka-Manyama"), for the names in ``names`` (lower-case).  The model copies how a name
    is said from its reference clip when it sees the same word there; glued to a prefix the word is a different one and the Zulu letter rules read it ("Penius").  Heard on prod-120: the glued form was
    wrong in 4 of 4 draws, the separated one right in 8 of 11."""
    if not names:
        return text
    return PREFIXED_NAME.sub(lambda m: f"{m.group(1)}-{m.group(2)}" if m.group(2).lower() in names else m.group(0), text)


def sentence_marks(span: str, word_times: Sequence[Any] | None, n_lead: int, offset_s: float) -> list[dict[str, Any]]:
    """The sentences of a span as the audio kept for it has them: {text, chars, start, end} in seconds from the start of that audio.  ``word_times`` is the service's list for the text it was sent (the lead
    sentence first, then the span, then one word of the next span), ``offset_s`` what was cut from the front of the generation."""
    tokens = str(span or "").split()
    times = list(word_times or [])[n_lead:n_lead + len(tokens)]
    marks, first = [], 0
    for i, token in enumerate(tokens):
        if token.rstrip().endswith((".", "?", "!")) or i == len(tokens) - 1:
            timed = [(tokens[k], times[k]) for k in range(first, i + 1) if k < len(times) and times[k]]
            if timed:
                marks.append({"text": " ".join(t for t, _ in timed)[:60], "chars": len(" ".join(tokens[first:i + 1])),
                              "start": round(max(0.0, float(timed[0][1]["start"]) - offset_s), 2), "end": round(max(0.0, float(timed[-1][1]["end"]) - offset_s), 2)})
            first = i + 1
    return marks


def word_marks(span: str, word_times: Sequence[Any] | None, n_lead: int, offset_s: float) -> list[dict[str, Any]]:
    """Every word of a span with its time in the audio kept for it: {text, start, end} in seconds from the start of that audio (what the studio shows as buttons).  A word the service could not time is left out."""
    tokens = str(span or "").split()
    times = list(word_times or [])[n_lead:n_lead + len(tokens)]
    return [{"text": token, "start": round(max(0.0, float(times[k]["start"]) - offset_s), 2), "end": round(max(0.0, float(times[k]["end"]) - offset_s), 2)}
            for k, token in enumerate(tokens) if k < len(times) and times[k]]


def source_sentences(words: Sequence[Mapping[str, Any]], speaker_of_raw: Mapping[str, str], speaker_id: str, start_s: float, end_s: float) -> list[dict[str, Any]]:
    """The speaker's sentences in the source window, from the recogniser's timed words: {start, end} in seconds from the start of the window (a sentence ends at a word ending . ? or !)."""
    inside = [w for w in words if not w.get("overlap") and float(w.get("start") or 0.0) >= start_s - 0.05 and float(w.get("end") or 0.0) <= end_s + 0.05
              and speaker_of_raw.get(str(w.get("raw_speaker")), str(w.get("raw_speaker"))) == speaker_id]
    out, current = [], []
    for w in inside:
        current.append(w)
        if str(w.get("text") or "").rstrip().endswith((".", "?", "!")):
            out.append({"start": round(float(current[0]["start"]) - start_s, 2), "end": round(float(current[-1]["end"]) - start_s, 2)})
            current = []
    if current:
        out.append({"start": round(float(current[0]["start"]) - start_s, 2), "end": round(float(current[-1]["end"]) - start_s, 2)})
    return out


def match_source(dub: Sequence[Mapping[str, Any]], source: Sequence[Mapping[str, Any]]) -> list[int | None]:
    """For each dub sentence the source sentence that covers the same share of the speech (by the middle of its share of the total speaking time), or None."""
    if not dub or not source:
        return [None] * len(dub)
    d_total = sum(float(x["end"]) - float(x["start"]) for x in dub) or 1.0
    s_total = sum(float(x["end"]) - float(x["start"]) for x in source) or 1.0
    s_spans, at = [], 0.0
    for x in source:
        length = (float(x["end"]) - float(x["start"])) / s_total
        s_spans.append((at, at + length))
        at += length
    out, cum = [], 0.0
    for x in dub:
        length = (float(x["end"]) - float(x["start"])) / d_total
        mid = cum + length / 2.0
        cum += length
        out.append(next((i for i, (a, z) in enumerate(s_spans) if a <= mid <= z), len(source) - 1))
    return out


LONG_PREFIX_NAME = re.compile(r"\b([a-z]{5,})-([A-Z][A-Za-z'\u2019]+)")


def decouple_names(text: str, names: "set[str] | frozenset[str]") -> str:
    """A name the speaker says, written as its own word after a long prefix, with the prefix's last syllable set apart: "wayekwi-Hawks" -> "waye-kwi Hawks".  Heard on prod-123: with the hyphen the
    voice ran "kwi" into "Hawks" ("khwi-awks"); with the last syllable of the verb apart and the name a word of its own, it said Hawks.  Short prefixes (u-, ka-, we-, ye-) are left as they are."""
    if not names:
        return text

    def split(match: "re.Match[str]") -> str:
        prefix, name = match.group(1), match.group(2)
        if name.lower() not in names:
            return match.group(0)
        syllable = re.search(r"[^aeiou]*[aeiou]+$", prefix)
        stem = prefix[: syllable.start()] if syllable else ""
        head = f"{stem}-{syllable.group(0)}" if syllable and len(stem) >= 2 else prefix
        return f"{head} {name}"

    return LONG_PREFIX_NAME.sub(split, text)


def protect_names(text: str, tokens: Sequence[str]) -> tuple[str, dict[str, str]]:
    """The text with each of ``tokens`` (names the speaker says in the source) swapped for a placeholder word the pronunciation dictionary will not match, and what to put back.  A Zulu prefix
    stays in front of it ("we-JMPD" -> "we-Zqprotab").  The dictionary's respellings (JMPD -> Jey Em Pii Dii, Canny -> Kheni) were made for a voice that has no source audio to learn a name from;
    for a name the speaker says, the source audio is the pronunciation and the dub text keeps the name as written."""
    mapping: dict[str, str] = {}
    out = text
    for number, token in enumerate(tokens):
        placeholder = "Zqprot" + "".join(chr(97 + int(digit)) for digit in f"{number:02d}")          # letters only: no number or figure rule touches it
        pattern = re.compile(rf"(?:\b|(?<=[a-z]))({re.escape(token)})(?![A-Za-z'\u2019])")          # after a word boundary, or glued to a Zulu prefix of any length (kwamaHawks, ngowayengu-Superintendent)

        def put(match: "re.Match[str]", placeholder: str = placeholder, token: str = token) -> str:
            mapping[placeholder] = token
            return placeholder

        out = pattern.sub(put, out)
    return out, mapping


def restore_names(text: str, mapping: Mapping[str, str]) -> str:
    for placeholder, token in mapping.items():
        text = text.replace(placeholder, token)
    return text


def stable_seed(*parts: Any) -> int:
    """A seed for the model's random generator that depends only on what is being generated (the text sent, the take, the pass, the reference): the same unit gives the same audio every time, so the length
    the agent measures is the length the render has.  Without it every generation is a fresh draw and a unit varies by 2-3 s from one to the next."""
    return int.from_bytes(hashlib.sha256("\x1f".join(str(part) for part in parts).encode("utf-8")).digest()[:4], "big") % (2 ** 31 - 1)


def _first_word(span: str) -> str | None:
    """The first word of a span when it has letters in it (a number or a symbol cannot be placed by the aligner), else None."""
    words = str(span or "").split()
    return words[0] if words and any(ch.isalpha() for ch in words[0]) else None


def _lead_words_ok(word_times: Any, lead: str) -> bool:
    """Whether the service's word times for a span that starts with ``lead`` can place the end of the lead: a list with the lead's last word (or one before it) and a word after it."""
    if not isinstance(word_times, list):
        return False
    n_lead = len(str(lead).split())
    return any(word_times[:n_lead]) and any(word_times[n_lead:])


class OmniVoiceClient:
    """The Cloud Run service's ``/v1/synthesize``.  Authenticates as a Google service account (``GOOGLE_APPLICATION_CREDENTIALS``) with an ID token for the service
    URL; ``OMNIVOICE_ID_TOKEN`` or a signed-in gcloud also work, for a developer's PC."""

    def __init__(self, url: str, *, retries: int = 3, timeout: float = 900.0, sleep: Callable[[float], None] = time.sleep, cache: Any = None, cache_tag: str = "") -> None:
        self.url = url.rstrip("/")
        self.retries, self.timeout, self.sleep = retries, timeout, sleep
        self.cache = cache                       # omnivoice_cache.SpanCache: what the voice has said before, a span at a time (None: every request goes to the service)
        self.cache_tag = cache_tag
        self.cache_stats = {"hits": 0, "misses": 0}
        self.parallel = 1                        # how many requests may be in the air at once: the copies of the voice held for the job (set_parallel); a unit's spans are spoken side by side across them
        self._slots: Any = None
        self.default_generation: dict[str, float] = {"num_step": DEFAULT_NUM_STEP}          # decoding settings of every request unless the caller sets its own
        self._token: tuple[str, float] | None = None

    def _id_token(self) -> str:
        if self._token and time.time() < self._token[1]:
            return self._token[0]
        token = None
        try:
            import google.auth.transport.requests
            import google.oauth2.id_token

            token = google.oauth2.id_token.fetch_id_token(google.auth.transport.requests.Request(), self.url)
        except Exception:
            token = (os.environ.get("OMNIVOICE_ID_TOKEN") or "").strip() or None
        if not token:
            for command in (["gcloud", "auth", "print-identity-token"], ["powershell", "-NoProfile", "-Command", "gcloud auth print-identity-token"]):
                try:
                    lines = subprocess.run(command, capture_output=True, text=True, timeout=60).stdout.strip().splitlines()
                    if lines and not lines[-1].startswith("ERROR"):
                        token = lines[-1].strip()
                        break
                except (OSError, subprocess.SubprocessError):
                    continue
        if not token:
            raise RuntimeError("no credentials for the OmniVoice service: set GOOGLE_APPLICATION_CREDENTIALS to the service account key, or OMNIVOICE_ID_TOKEN")
        self._token = (token, time.time() + 40 * 60)
        return token

    def warm(self, copies: int = 1, *, hold_s: float = 0.0, patience_s: float = 240.0, progress: Callable[[str], None] | None = None) -> list[str]:
        """Wake the voice before the work starts: ``copies`` requests to ``/v1/warm`` (no GPU), answered by a loaded copy, and the ids of the copies that answered.  A copy that is not running
        takes a minute or more to start and the call waits for it (up to ``patience_s``), so the first measurement of a job does not pay for the cold start.  This does NOT make Cloud Run run
        several copies: requests wait for a busy copy rather than start a new one (measured 2026-11-24: four holds of 30 s were answered one after another by one copy for four minutes), and a
        new copy only starts once real work keeps the first one busy for a minute or two."""
        import threading

        copies = max(1, min(int(copies), MAX_COPIES))
        started = time.time()
        seen: set[str] = set()
        lock = threading.Lock()
        headers = {"Authorization": f"Bearer {self._id_token()}", "Content-Type": "application/json"}

        def ask() -> None:
            try:
                from .dub_agent import post_with_retry

                answer = post_with_retry(f"{self.url}/v1/warm", headers, {"hold_s": hold_s}, timeout=hold_s + patience_s, retries=1, sleep=self.sleep)
                with lock:
                    seen.add(str(answer.get("instance") or ""))
            except Exception:          # noqa: BLE001 - warming is an optimisation: a failed ask is just a copy that did not answer
                pass

        threads = [threading.Thread(target=ask) for _ in range(copies)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        seen.discard("")
        if progress:
            progress(f"[voice] {len(seen)} copy of the voice answered after {time.time() - started:.0f} s" if len(seen) == 1 else f"[voice] {len(seen)} copies of the voice answered after {time.time() - started:.0f} s")
        return sorted(seen)

    def synthesize(self, *, spans: Sequence[str], gaps_ms: Sequence[int], reference: Reference | None, durations_s: Sequence[float | None] | None = None,
                   speed: float = 1.0, lora: bool = True, align: bool = False, languages: Sequence[str | None] | None = None,
                   generation: Mapping[str, float] | None = None, seeds: Sequence[int | None] | None = None) -> dict[str, Any]:
        """The service's reply for these spans, where a span that was said before (same text, reference, settings and seed) is taken from the cache and only the others are asked for (omnivoice_cache)."""
        ask = dict(spans=spans, gaps_ms=gaps_ms, reference=reference, durations_s=durations_s, speed=speed, lora=lora, align=align, languages=languages, generation=generation, seeds=seeds)
        cache = self.cache
        spans = list(spans)
        if cache is None or not spans or not seeds or any(not str(x).strip() for x in spans):
            return self._dispatch(**ask)
        try:
            return self._cached(cache, spans, ask)
        except Exception:                    # noqa: BLE001 - the cache is an optimisation: whatever goes wrong in it, the service is asked
            return self._dispatch(**ask)

    def _cached(self, cache: Any, spans: list[str], ask: dict[str, Any]) -> dict[str, Any]:
        from . import omnivoice_cache as oc

        count = len(spans)
        align = bool(ask["align"])
        merged = {**self.default_generation, **(ask["generation"] or {})}
        reference = ask["reference"]
        reference_id = oc.reference_digest(reference.wav if reference is not None else None, reference.text if reference is not None else None)
        seeds = list(ask["seeds"]) + [None] * (count - len(ask["seeds"]))
        durations = list(ask["durations_s"] or []) + [None] * (count - len(ask["durations_s"] or []))
        languages = [(x or "zu") for x in (ask["languages"] or [])] + ["zu"] * (count - len(ask["languages"] or []))
        keys = [None if seeds[i] is None else oc.span_key(text=spans[i], language=languages[i], reference=reference_id, duration=durations[i], speed=float(ask["speed"]), lora=bool(ask["lora"]),
                                                          generation=merged, seed=int(seeds[i]), tag=self.cache_tag) for i in range(count)]
        found = [cache.get(k, need_words=align) if k else None for k in keys]
        missing = [i for i in range(count) if found[i] is None]
        self.cache_stats["hits"] += count - len(missing)
        self.cache_stats["misses"] += len(missing)
        if len(missing) == count:                                              # nothing said before: the request as it was, and its spans remembered
            reply = self._dispatch(**ask)
            try:
                parts, rate = oc.split_reply(reply)
                words = reply.get("words") if align and isinstance(reply.get("words"), list) else None
                for i, part in enumerate(parts[:count]):
                    if keys[i] is not None:
                        cache.put(keys[i], part, rate, words[i] if words is not None and i < len(words) else None)
            except Exception:                # noqa: BLE001
                pass
            return reply
        gaps = [max(0, int(g)) for g in ask["gaps_ms"]][: count - 1]
        gaps += [0] * (count - 1 - len(gaps))
        new_parts: dict[int, Any] = {}
        new_words: dict[int, Any] = {}
        rate = next(f["rate"] for f in found if f is not None)
        gpu_s, words_back = 0.0, True
        if missing:
            sub = self._dispatch(spans=[spans[i] for i in missing], gaps_ms=[0] * (len(missing) - 1), reference=reference, durations_s=[durations[i] for i in missing], speed=ask["speed"], lora=ask["lora"],
                               align=align, languages=[languages[i] for i in missing], generation=ask["generation"], seeds=[seeds[i] for i in missing])
            parts, sub_rate = oc.split_reply(sub)
            if sub_rate != rate or len(parts) != len(missing):
                return self._dispatch(**ask)
            gpu_s = float(sub.get("gpu_s") or 0.0)
            sub_words = sub.get("words") if isinstance(sub.get("words"), list) else None
            words_back = sub_words is not None
            for k, i in enumerate(missing):
                new_parts[i] = parts[k]
                new_words[i] = sub_words[k] if sub_words is not None and k < len(sub_words) else None
                if keys[i] is not None:
                    cache.put(keys[i], parts[k], rate, new_words[i] if align else None)
        if any(f is not None and f["rate"] != rate for f in found):
            return self._dispatch(**ask)
        pieces = [found[i]["audio"] if found[i] is not None else new_parts[i] for i in range(count)]
        words = [found[i]["words"] if found[i] is not None else new_words[i] for i in range(count)] if align and words_back else None
        reply = oc.assemble(pieces, rate, gaps, words, gpu_s)
        reply["cache"] = {"hits": count - len(missing), "spans": count}
        return reply

    def set_parallel(self, copies: int) -> None:
        """Allow this many requests at once (the copies of the voice that are held), and spread a unit's spans over them."""
        import threading

        self.parallel = max(1, min(MAX_COPIES, int(copies)))
        self._slots = threading.BoundedSemaphore(self.parallel) if self.parallel > 1 else None

    def _dispatch(self, **ask: Any) -> dict[str, Any]:
        if self.parallel > 1 and len(list(ask["spans"])) > 1:
            return self._remote_parallel(**ask)
        return self._remote(**ask)

    def _remote_parallel(self, **ask: Any) -> dict[str, Any]:
        """The spans of a request spoken side by side: each copy of the voice takes a share (the longest spans first into the lightest share), and the replies are put together as one service reply, in the
        order of the spans, with the gaps between them.  A span is generated on its own with its own seed, so where it is spoken changes nothing about it."""
        from concurrent.futures import ThreadPoolExecutor

        from . import omnivoice_cache as oc

        spans = list(ask["spans"])
        count = len(spans)
        shares: list[list[int]] = [[] for _ in range(min(self.parallel, count))]
        loads = [0] * len(shares)
        for i in sorted(range(count), key=lambda i: -len(spans[i])):
            lightest = loads.index(min(loads))
            shares[lightest].append(i)
            loads[lightest] += len(spans[i]) + 40
        shares = [sorted(share) for share in shares if share]

        def fill(sequence: Any) -> list[Any]:
            return list(sequence or []) + [None] * (count - len(sequence or []))

        durations, languages, seeds = fill(ask["durations_s"]), fill(ask["languages"]), fill(ask["seeds"])

        def one(share: list[int]) -> dict[str, Any]:
            return self._remote(spans=[spans[i] for i in share], gaps_ms=[0] * (len(share) - 1), reference=ask["reference"], durations_s=[durations[i] for i in share] if ask["durations_s"] else None,
                                speed=ask["speed"], lora=ask["lora"], align=ask["align"], languages=[languages[i] for i in share] if ask["languages"] else None, generation=ask["generation"],
                                seeds=[seeds[i] for i in share] if ask["seeds"] else None)

        with ThreadPoolExecutor(max_workers=len(shares)) as pool:
            replies = list(pool.map(one, shares))
        parts: dict[int, Any] = {}
        words: dict[int, Any] = {}
        rate = None
        gpu_s = 0.0
        words_ok = bool(ask["align"])
        for share, reply in zip(shares, replies):
            got, got_rate = oc.split_reply(reply)
            if (rate is not None and got_rate != rate) or len(got) != len(share):
                return self._remote(**ask)
            rate = got_rate
            gpu_s += float(reply.get("gpu_s") or 0.0)
            back = reply.get("words") if isinstance(reply.get("words"), list) else None
            words_ok = words_ok and back is not None and len(back) == len(share)
            for k, i in enumerate(share):
                parts[i] = got[k]
                words[i] = back[k] if back is not None and k < len(back) else None
        gaps = [max(0, int(g)) for g in ask["gaps_ms"]][: count - 1]
        gaps += [0] * (count - 1 - len(gaps))
        reply = oc.assemble([parts[i] for i in range(count)], int(rate or 24000), gaps, [words[i] for i in range(count)] if words_ok else None, gpu_s)
        reply["copies"] = len(shares)
        return reply

    def _remote(self, *, spans: Sequence[str], gaps_ms: Sequence[int], reference: Reference | None, durations_s: Sequence[float | None] | None = None,
                speed: float = 1.0, lora: bool = True, align: bool = False, languages: Sequence[str | None] | None = None,
                generation: Mapping[str, float] | None = None, seeds: Sequence[int | None] | None = None) -> dict[str, Any]:
        from .dub_agent import post_with_retry

        body: dict[str, Any] = {"spans": list(spans), "gaps_ms": [max(0, int(g)) for g in gaps_ms], "speed": speed, "lora": lora}          # a join closer than the model's padding is made when the audio is assembled
        if align:
            body["align"] = True                    # the service also returns where every word of every span falls in its audio (a service that predates this ignores it)
        if seeds:
            body["seeds"] = [None if x is None else int(x) for x in seeds]          # per span; a service that predates this generates unseeded
        generation = {**self.default_generation, **(generation or {})}
        if generation:
            body["generation"] = {str(k): v for k, v in generation.items()}          # the model's decoding settings (position_temperature, guidance_scale, ...); a service that predates this ignores them
        if languages:
            body["languages"] = [l or "zu" for l in languages]          # per span; a service that predates this reads every span as Zulu
        if durations_s:
            body["durations_s"] = [d for d in durations_s]
        if reference is not None:
            body["ref_wav_b64"] = base64.b64encode(reference.wav).decode("ascii")
            body["ref_text"] = reference.text
        headers = {"Authorization": f"Bearer {self._id_token()}", "Content-Type": "application/json"}
        slots = self._slots
        if slots is None:
            return post_with_retry(f"{self.url}/v1/synthesize", headers, body, timeout=self.timeout, retries=self.retries, sleep=self.sleep)
        with slots:                                  # no more requests in the air than there are copies to take them
            return post_with_retry(f"{self.url}/v1/synthesize", headers, body, timeout=self.timeout, retries=self.retries, sleep=self.sleep)


def copies_for(source_seconds: float) -> int:
    """How many copies of the voice a clip of this much source audio should have running: one for each three minutes, between one and four.  Several units are measured side by side and each takes
    seconds to minutes, so a long clip is worth more GPUs and a short one is not."""
    import math

    return max(1, min(MAX_COPIES, math.ceil(max(0.0, float(source_seconds)) / SECONDS_PER_COPY)))


def release_voice() -> None:
    """Give back the copies this process holds (omnivoice_scale.release): at the end of a job, and whenever the process exits."""
    try:
        from . import omnivoice_scale as sc
        from .config import load_settings

        sc.release(load_settings().work_dir)
    except Exception as exc:        # noqa: BLE001 - a copy left held is put back by the next job (release_if_stale)
        print(f"[voice] could not release the copies: {type(exc).__name__}: {str(exc)[:120]}")


def warm_voice(source_seconds: float, *, progress: Callable[[str], None] = print, environ: Mapping[str, str] | None = None, patience_s: float = 300.0) -> list[str] | None:
    """Wake the voice for a job, and hold as many copies as the clip needs (copies_for: one per cluster's worth of speech, up to four) until the job ends, waiting for them to load before anything else is done.
    Every request after it spreads a unit's spans over the copies that answered.  None when the voice is not OmniVoice (no OMNIVOICE_URL)."""
    env = dict(os.environ if environ is None else environ)
    url = (env.get("OMNIVOICE_URL") or "").strip()
    if not url:
        return None
    wanted = copies_for(source_seconds)
    if wanted > 1 and (env.get("OMNIVOICE_SCALE", "1")).strip().lower() not in {"0", "false", "no"}:
        try:
            import atexit

            from . import omnivoice_scale as sc
            from .config import load_settings

            work_dir = load_settings().work_dir
            sc.release_if_stale(work_dir, progress=progress)
            sc.acquire(wanted, work_dir, progress=progress)
            atexit.register(release_voice)
        except Exception as exc:        # noqa: BLE001 - one copy still does the job
            progress(f"[voice] could not hold {wanted} copies ({type(exc).__name__}: {str(exc)[:120]}): one copy it is")
            wanted = 1
    else:
        wanted = 1
    progress(f"[voice] waking the voice ({source_seconds:.0f} s of source audio: {wanted} cop{'y' if wanted == 1 else 'ies'})")
    client = OmniVoiceClient(url)
    seen: set[str] = set()
    started = time.time()
    while True:
        answered = client.warm(wanted, hold_s=4.0 if wanted > 1 else 0.0, patience_s=patience_s)          # a few seconds held, so the requests go to different copies
        seen.update(answered)
        if len(seen) >= wanted or time.time() - started > patience_s:
            break
        if not answered:
            time.sleep(3.0)
    progress(f"[voice] {len(seen)} cop{'y' if len(seen) == 1 else 'ies'} of the voice answered after {time.time() - started:.0f} s")
    if environ is None:
        os.environ["OMNIVOICE_PARALLEL"] = str(max(1, len(seen)))          # the backends this job makes spread a unit's spans over these copies
    return sorted(seen)


def split_marked_pauses(text: str) -> list[str | int]:
    """The text as speech with its ``~pauseN`` markers as silence (ms) between the pieces, for OmniVoice.

    ``~pauseN`` is the TOTAL silence at that spot.  ``delivery.split_pauses`` subtracts what Azure's punctuation already leaves (0.92 s after a full stop), which is
    Azure's behaviour and not OmniVoice's: OmniVoice trims each generation and keeps 100 ms at each end, so the gap between two spans is N minus those 200 ms.  A
    marker below the shortest held pause is left to the model's own phrasing; the same per-marker and total caps apply."""
    from .delivery import MAX_MARKED_PAUSE_MS, MAX_MARKED_PAUSE_TOTAL_MS, MIN_MARKED_PAUSE_MS, PAUSE_MARKER

    pieces: list[str | int] = []
    position, total = 0, 0
    for match in PAUSE_MARKER.finditer(text):
        before = text[position:match.start()]
        position = match.end()
        if before.strip():
            pieces.append(before)
        wanted = min(MAX_MARKED_PAUSE_MS, int(match.group(1)))
        gap = min(wanted, MAX_MARKED_PAUSE_TOTAL_MS - total) - int(od.EDGE_PAD_S * 1000)
        if gap >= MIN_MARKED_PAUSE_MS and pieces and not isinstance(pieces[-1], int):
            pieces.append(gap)
            total += gap
    if text[position:].strip():
        pieces.append(text[position:])
    return pieces


def flatten_parts(parts: Sequence[Any]) -> list[str | int]:
    """The dictionary's typed SSML parts as plain text and silence: a string is speech, an int is a pause in ms.  Emphasis and character parts keep their text,
    a substitution speaks its respelling (the form the voice should say), a break is silence."""
    out: list[str | int] = []
    for part in parts:
        name = type(part).__name__
        if name == "BreakPart":
            out.append(int(part.duration_ms))
        elif name == "SubstitutionPart":
            out.append(str(part.tts_text))
        else:
            out.append(str(getattr(part, "text", "")))
    return out


def merge_spans(spans: Sequence[str], gaps: Sequence[int], per_span: int, max_chars: int = 320, fits: Callable[[str], bool] | None = None) -> tuple[list[str], list[int]]:
    """Up to ``per_span`` consecutive sentences spoken in ONE generation (no more than ``max_chars`` characters), the gap after the last of them kept between generations.  The voice then plans the
    intonation across those sentences, as it does for one long sentence; the price measured on 2026-11-22 was 15-19% faster, shorter speech with weak syllables dropped, so this is for listening
    experiments (a run's voice.json says ``span_sentences``), not the default.
    ``per_span`` is a cap on the sentences (FILL_SENTENCES: none); the size is ``fits``, a test of the text the cluster would then be (the estimated speech time, see OmniVoiceTTSBackend.plan), or ``max_chars`` without one."""
    merged: list[str] = []
    after: list[int] = []
    count = 0
    for index, text in enumerate(spans):
        gap = int(gaps[index]) if index < len(gaps) else 0
        if merged and count < per_span and (fits(f"{merged[-1]} {text}") if fits is not None else len(merged[-1]) + 1 + len(text) <= max_chars):
            merged[-1] = f"{merged[-1]} {text}"
            after[-1] = gap
            count += 1
        else:
            merged.append(text)
            after.append(gap)
            count = 1
    return merged, after[:-1]


FILL_SENTENCES = 99          # span_sentences = fill: a cluster takes sentences for as long as they fit the time budget
CLUSTER_MAX_S = 20.0         # the longest speech a cluster is made to hold (estimated from the reference's pace): the model chunks a generation above 30 s, and a long one drifts, so a cluster stays well under
SWAP_FADE_S = 0.012          # the two sounds at a splice overlap this long, one fading out as the other fades in
SWAP_KEEP_S = 0.05           # a sentence swapped into a cluster keeps this much of the quiet on each side of it, so the pause around it stays as the cluster made it


def _plain_word(word: str) -> str:
    """A word as a name is matched: no punctuation around it, lower case, and only what follows an 'i-' / 'u-' / 'ngo-' prefix."""
    return re.sub(r"^\W+|\W+$", "", str(word)).lower().split("-")[-1]


def name_runs(words: Sequence[str], names: Sequence[str] | set[str]) -> list[tuple[int, int]]:
    """The runs of consecutive words (first, last index) that are the speaker's names: every word of a run is one of ``names`` (the lower-case words ReferenceBank.primer_for says it carries)."""
    wanted = {str(n).lower() for n in names}
    runs: list[tuple[int, int]] = []
    start = None
    for i, word in enumerate(words):
        if _plain_word(word) in wanted:
            start = i if start is None else start
        elif start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(words) - 1))
    return runs


def _letters(text: str) -> int:
    return sum(1 for ch in str(text) if ch.isalpha())


def name_lengths(words: Sequence[str], times: Sequence[Mapping[str, Any] | None], names: Sequence[str] | set[str]) -> list[dict[str, Any]]:
    """How long the voice said each name of ``words`` (``times`` are the aligned times of the same words): [{text, s, pace}].  ``pace`` is how fast the name was said, in letters a second, against the median
    of the other words of the same sentence (1.0 = as fast as they are, 0.4 = two and a half times slower: a word said twice, or drawn out)."""
    import statistics

    runs = name_runs(words, names)
    inside = {i for first, last in runs for i in range(first, last + 1)}
    rest = []
    for i, word in enumerate(words):
        t = times[i] if i < len(times) else None
        if i not in inside and t and float(t["end"]) - float(t["start"]) > 0.05 and _letters(word):
            rest.append(_letters(word) / (float(t["end"]) - float(t["start"])))
    typical = statistics.median(rest) if len(rest) >= 3 else None
    out = []
    for first, last in runs:
        a, z = (times[first] if first < len(times) else None), (times[last] if last < len(times) else None)
        if a and z:
            seconds = float(z["end"]) - float(a["start"])
            entry = {"text": " ".join(words[first:last + 1]), "s": round(seconds, 3),
                     "words": [{"text": words[i], "s": round(float(times[i]["end"]) - float(times[i]["start"]), 3)} for i in range(first, last + 1) if i < len(times) and times[i]]}
            if typical and seconds > 0.05:
                entry["pace"] = round(_letters(" ".join(words[first:last + 1])) / seconds / typical, 2)
            sure = [float(item[2]) for i in range(first, min(last + 1, len(times))) if times[i] for item in (times[i].get("letters") or []) if len(item) > 2]
            if sure:                                                                         # how sure the aligner is that each letter of the name was said (0-1; a swallowed letter is near 0): the mean
                entry["clarity"] = round(sum(sure) / len(sure), 3)
            out.append(entry)
    return out


STRETCH = 1.5          # a name said this many times longer than the median of its draws is not a better draw of it but a worse one (a word said twice: prod-x1 "wase-Ekurhuleni", 1.72 s of 0.98/1.72/0.94)


def stretched_draws(scores: Sequence[Mapping[str, Any]]) -> list[bool]:
    """Which draws said a name far longer than the other draws did: the timing of the aligner, not its confidence (the echo drew the highest confidence of the three).  It takes three draws to say what is
    usual, so fewer say nothing."""
    import statistics

    flags = [False] * len(scores)
    if len(scores) < 3:
        return flags
    by_name: dict[tuple[str, int], list[tuple[int, float]]] = {}
    for t, sc in enumerate(scores):
        for n in sc.get("names") or []:          # word by word: a run of two names dilutes the one that was said twice (1.86, 2.37 and 1.94 s for the run, 0.98, 1.72 and 0.94 s for the word)
            for k, word in enumerate(n.get("words") or [{"text": n["text"], "s": n["s"]}]):
                by_name.setdefault((str(n["text"]), k), []).append((t, float(word["s"])))
    for seen in by_name.values():
        if len(seen) < 3:
            continue
        typical = statistics.median(s for _, s in seen)
        for t, s in seen:
            if typical > 0 and s > STRETCH * typical:
                flags[t] = True
    return flags


def pick_take(scores: Sequence[Mapping[str, Any]]) -> int:
    """Which draw of a span is used: where the span carries names the speaker says, the draw in which the voice said them longest (the audio score, voice_takes.score_span, breaking a tie); otherwise the draw
    with the least wrong in the audio.  The way a name goes wrong is that it is swallowed, said shorter than the speaker says it, and of the three draws of "i-NEC" and of "yi-ANC" the longest name was the
    one the ear chose both times (0.401 s of 0.38/0.30/0.40, and 0.641 s of 0.42/0.64/0.54).
    Where the aligner also says how sure it is of each letter of the names (``clarity``, the mean over them: a swallowed letter is near 0), the draw whose names were said most clearly is used: the user picks
    draws by enunciation, and the clearest draw was the one picked in 3 of 3 clear verdicts of 2026-11-26 where the longest name was not (be-ActionSA: 1.02 s of 1.02/0.70/0.74 was the bad one).  A draw the
    listener committed is never overruled (``pinned`` in speak)."""
    if not any(float(sc.get("name_s") or 0.0) > 0.0 for sc in scores):
        return min(range(len(scores)), key=lambda t: scores[t]["score"])
    stretched = stretched_draws(scores)
    kept = [t for t in range(len(scores)) if not stretched[t]] or list(range(len(scores)))          # a draw that said a name far longer than the others is set aside (stretched_draws)
    if all(scores[t].get("clarity") is not None for t in kept):
        return max(kept, key=lambda t: (float(scores[t]["clarity"]), float(scores[t].get("name_s") or 0.0), -float(scores[t]["score"])))
    return max(kept, key=lambda t: (float(scores[t].get("name_s") or 0.0), -float(scores[t]["score"])))


def joined_with_the_word_before(words: Sequence[str], first: int) -> int:
    """Where a name's swap begins: the name itself, or the word before it when the name carries a Zulu prefix ("i-NEC", "yi-ANC") and that word ends in a vowel ("futhi i-NEC"): the two vowels run
    into each other, the aligner puts the boundary somewhere inside them, and a cut there loses the prefix, so the pair is taken from one generation."""
    if first > 0 and "-" in str(words[first]).strip(".,;:!?") and re.search(r"[aeiou]\W*$", str(words[first - 1]).lower()):
        return first - 1
    return first


def prefix_share(words: Sequence[str], first: int) -> float | None:
    """For a name that carries a Zulu prefix after a word ending in a vowel ("baphothule ama-details"): the share of the name's letters that the prefix is (0.3 for "ama" of "ama-details"), else None.
    The prefix is said the same way in every generation of the sentence, so a splice can be joined inside it instead of taking the word before as well."""
    word = str(words[first]).strip(".,;:!?")
    if first > 0 and "-" in word and re.search(r"[aeiou]\W*$", str(words[first - 1]).lower()):
        letters = len(re.sub(r"\W", "", word))
        head = len(word.split("-", 1)[0])
        if letters and head:
            return head / letters
    return None


TRAILING_FOLLOW = "Futhi"      # spoken after the last span of a unit (cut away afterwards): with nothing after it the model's generation ends ON the last word, which is then chopped at full volume (prod-x1 "ngomphumela", draw 3 of the primed span ended at -49 dB)
FINAL_TAIL_KEEP_S = 0.30       # the end of a unit keeps up to this much after its last word (the word's own decay), and fades out over FINAL_FADE_S
FINAL_FADE_S = 0.08


MERGE_GAP_WORDS = 1          # names this close (at most this many words between them) are taken as one stretch from the primed sentence


def swap_regions(words: Sequence[str], names: Sequence[str] | set[str], with_word_before: bool = True) -> list[tuple[int, int]]:
    """The stretches of a sentence that are taken from the primed generation: each name run (with the vowel-ended word before it that it runs into), and a stretch that would leave only a word or so of
    the cluster between two of them is made one stretch, taken whole ("yebo, ama-details okugcina, silinde i-ANC": the 0.6 s of "okugcina" left between the two names came from another generation,
    at another pace, with a splice on each side of it, and every draw of it sounded messy)."""
    regions: list[tuple[int, int]] = []
    for first, last in name_runs(words, names):
        first = joined_with_the_word_before(words, first) if with_word_before else first          # with_word_before False: the name alone, joined inside its prefix (swap_word_run inside=)
        if regions and first - regions[-1][1] - 1 <= MERGE_GAP_WORDS:
            regions[-1] = (regions[-1][0], last)
        else:
            regions.append((first, last))
    return regions


def _crossfade(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    n = min(n, len(a), len(b))
    if n < 2:
        return np.concatenate([a, b])
    ramp = np.linspace(0.0, 1.0, n, dtype=a.dtype)
    return np.concatenate([a[:-n], a[-n:] * (1.0 - ramp) + b[:n] * ramp, b[n:]])


def _crossfade_equal(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """Like _crossfade but with equal-power curves (cos/sin), so the level does not dip in the middle of the blend."""
    n = min(n, len(a), len(b))
    if n < 2:
        return np.concatenate([a, b])
    t = np.linspace(0.0, np.pi / 2.0, n, dtype=a.dtype)
    return np.concatenate([a[:-n], a[-n:] * np.cos(t) + b[:n] * np.sin(t), b[n:]])


def _alike_cut(host: np.ndarray, at: int, snippet: np.ndarray, around: int, window: int, reach: int, before: bool) -> int:
    """Where in ``snippet``, within ``reach`` samples of ``around``, to cut so that the snippet looks most like the host at the join: for a join into the snippet (``before``) the ``window`` samples of the
    snippet that come just before the cut are compared with the ``window`` samples of the host before ``at`` (normalised correlation); for the join back out they are the ones just after the cut against
    the host's just after ``at``.  Signals that are both near silence are left where they were."""
    ref = host[max(0, at - window): at] if before else host[at: at + window]
    if len(ref) < window or float(np.sqrt(np.mean(ref ** 2))) < 1e-4:
        return around
    best, best_corr = around, -2.0
    norm_ref = float(np.linalg.norm(ref)) + 1e-9
    for j in range(around - reach, around + reach + 1, max(1, reach // 60)):
        seg = snippet[j - window: j] if before else snippet[j: j + window]
        if j - window < 0 or len(seg) < window or j + window > len(snippet):
            continue
        corr = float(np.dot(ref, seg)) / (norm_ref * (float(np.linalg.norm(seg)) + 1e-9))
        if corr > best_corr:
            best, best_corr = j, corr
    return best


def _speech_level(audio: np.ndarray, rate: int) -> float:
    """The loudness of the speech in a stretch: the rms of its louder half of 10 ms frames."""
    n = max(1, int(0.01 * rate))
    count = len(audio) // n
    if count < 4:
        return float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0
    rms = np.sqrt((audio[: count * n].reshape(count, n) ** 2).mean(axis=1))
    return float(np.sqrt(np.mean(np.sort(rms)[count // 2:] ** 2)))


def _prefix_cut(part: np.ndarray, host_from: int, host_to: int, snippet: np.ndarray, snip_from: int, snip_to: int, n: int, step: int) -> tuple[float, int, int] | None:
    """Where to join two generations of the same prefix, inside it: (correlation, cut in ``part``, cut in ``snippet``).  The ``n`` samples that are cross-faded are the ``n`` of ``part`` just before the
    host cut and the ``n`` of ``snippet`` just after the snippet cut; both must lie inside the prefix (``host_from``..``host_to`` and ``snip_from``..``snip_to``), and the pair that look most alike
    (normalised correlation of the two overlapping stretches, so also in phase) is the join."""
    host_from = max(host_from, 0)
    host_to = min(host_to, len(part))
    snip_from = max(snip_from, 0)
    snip_to = min(snip_to, len(snippet))
    if host_to - host_from < n or snip_to - snip_from < n:
        return None
    seg = snippet[snip_from:snip_to].astype(np.float64)
    csum = np.concatenate([[0.0], np.cumsum(seg ** 2)])
    seg_norm = np.sqrt(np.maximum(csum[n:] - csum[:-n], 0.0)) + 1e-9
    best: tuple[float, int, int] | None = None
    host_room, snip_room = host_to - host_from - n, snip_to - snip_from - n
    for hi in range(host_from + n, host_to + 1, max(1, step)):
        ref = part[hi - n: hi].astype(np.float64)
        ref_norm = float(np.linalg.norm(ref))
        if ref_norm < 1e-6:
            continue
        corr = np.correlate(seg, ref, "valid") / (seg_norm * ref_norm)
        centre = (hi - n - host_from) / max(1, host_room) * snip_room          # the same way through the prefix in the snippet: a-m-a has two vowels alike, and a join from the end of one into the start of the other says the prefix twice
        lo_m, hi_m = max(0, int(centre - PREFIX_JOIN_SLACK * (snip_to - snip_from))), min(len(corr) - 1, int(centre + PREFIX_JOIN_SLACK * (snip_to - snip_from)))
        if hi_m < lo_m:
            continue
        m = lo_m + int(np.argmax(corr[lo_m: hi_m + 1]))
        if best is None or float(corr[m]) > best[0]:
            best = (float(corr[m]), hi, snip_from + m)
    return best


PREFIX_JOIN_SLACK = 0.2             # the two cuts are as far through their prefixes, give or take this share of the prefix
PREFIX_JOIN_MIN_CORR = 0.5          # a join inside a prefix is used when the two overlapping stretches correlate at least this well, else the name is cut at its word boundaries as any other


def pause_bounded(first: int, last: int, count: int, host_times: Sequence[Mapping[str, Any] | None], host_offset: int, snippet_times: Sequence[Mapping[str, Any] | None], snippet_offset: int,
                  min_gap_s: float = 0.10, max_extra: int = 3) -> tuple[int, int]:
    """The words ``first``..``last`` of a sentence (``count`` words), widened until both ends sit against a real pause (a silence of ``min_gap_s`` or more, in the host AND in the snippet's generation), by at
    most ``max_extra`` words each way: a cut in the middle of continuous speech is heard, a cut in a pause is not."""
    def gap(times: Sequence[Mapping[str, Any] | None], i: int, after: bool) -> float:
        a, b = (i, i + 1) if after else (i - 1, i)
        if a < 0 or b >= len(times) or not times[a] or not times[b]:
            return 9.0
        return float(times[b]["start"]) - float(times[a]["end"])

    lo, hi, extra = first, last, 0
    while extra < max_extra and lo > 0 and min(gap(host_times, host_offset + lo, False), gap(snippet_times, snippet_offset + lo, False)) < min_gap_s:
        lo, extra = lo - 1, extra + 1
    extra = 0
    while extra < max_extra and hi < count - 1 and min(gap(host_times, host_offset + hi, True), gap(snippet_times, snippet_offset + hi, True)) < min_gap_s:
        hi, extra = hi + 1, extra + 1
    return lo, hi


def merge_regions(regions: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Stretches of words (first, last) with those that overlap or touch made one, in order."""
    merged: list[tuple[int, int]] = []
    for lo, hi in sorted(regions):
        if merged and lo <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def swap_word_run(part: np.ndarray, rate: int, times: Sequence[Mapping[str, Any] | None], first: int, last: int,
                  snippet: np.ndarray, snippet_times: Sequence[Mapping[str, Any] | None], s_first: int, s_last: int,
                  cut_at: Callable[[np.ndarray, float, float], float], keep_s: float | None = None, refine: bool = False, inside: float | None = None) -> tuple[np.ndarray, list[Mapping[str, Any] | None]] | None:
    """``part`` with its words ``first``..``last`` replaced by the words ``s_first``..``s_last`` of ``snippet`` (another generation of the same words), and every word's time after the swap:
    (new audio, new word times), or None when a word has no time.  Both are cut in the quiet between the word before and the run (and between the run and the word after), joined with a short
    crossfade; the words after the swap move by the difference in length, and the run takes its times from the snippet.  ``times`` are seconds from the start of ``part``; ``cut_at(audio, from_s,
    to_s)`` says where to cut between two times.  With ``keep_s`` (a whole sentence, bounded by real pauses) the cut is no farther than that from the words, so the pause stays whole."""
    a_word, z_word, s_a, s_z = times[first], times[last], snippet_times[s_first], snippet_times[s_last]
    if not (a_word and z_word and s_a and s_z):
        return None

    def edges(audio: np.ndarray, all_times: Sequence[Mapping[str, Any] | None], f: int, l: int) -> tuple[float, float]:
        begin, end = float(all_times[f]["start"]), float(all_times[l]["end"])
        before = all_times[f - 1] if f > 0 else None
        after = all_times[l + 1] if l + 1 < len(all_times) else None
        edge = keep_s if keep_s is not None else 0.02
        cut_a = cut_at(audio, float(before["end"]), begin) if before else max(0.0, begin - edge)
        cut_z = cut_at(audio, end, float(after["start"])) if after else min(len(audio) / rate, end + edge)
        if keep_s is not None:
            cut_a, cut_z = max(cut_a, begin - keep_s), min(cut_z, end + keep_s)
        return max(0.0, min(cut_a, begin)), max(cut_z, end)

    a, z = edges(part, times, first, last)
    sa, sz = edges(snippet, snippet_times, s_first, s_last)
    i_a, i_z, j_a, j_z = (int(round(t * rate)) for t in (a, z, sa, sz))
    n = int(SWAP_FADE_S * rate)
    if refine:                                                                  # cuts where the two generations look alike, a gain that puts the name at the host's speech level, an equal-power blend (longer across quiet)
        window, reach = int(0.015 * rate), int(0.03 * rate)
        j_a = _alike_cut(part, i_a, snippet, j_a, window, reach, True)
        j_z = _alike_cut(part, i_z, snippet, j_z, window, reach, False)
        n = int(0.02 * rate)
        quiet = max(_speech_level(part[max(0, i_a - window): i_a + window], rate), _speech_level(part[max(0, i_z - window): i_z + window], rate)) < 0.25 * max(_speech_level(part[i_a:i_z], rate), 1e-6)
        n = int(0.04 * rate) if quiet else n
    if inside is not None and refine:                                           # the name alone, joined inside its prefix: both tracks say it the same way, so the cross-fade lies wholly within it, where the two look most alike
        window = int(0.02 * rate)
        h0, h1, s0, s1 = float(a_word["start"]), float(a_word["end"]), float(s_a["start"]), float(s_a["end"])
        found = _prefix_cut(part, int(h0 * rate), int((h0 + inside * (h1 - h0)) * rate), snippet, int(s0 * rate), int((s0 + inside * (s1 - s0)) * rate), window, max(1, rate // 1000))
        if found is not None and found[0] >= PREFIX_JOIN_MIN_CORR:
            i_a, j_a, n = found[1], found[2], window
    middle = snippet[j_a:j_z]
    if refine and len(middle) > 0:
        host_level, own_level = _speech_level(part[i_a:i_z], rate), _speech_level(middle, rate)
        if host_level > 1e-6 and own_level > 1e-6:
            middle = middle * float(np.clip(host_level / own_level, 0.7, 1.4))
    if len(middle) < 2 * n or i_a < n or len(part) - i_z < n:
        return None
    blend = _crossfade_equal if refine else _crossfade
    joined = blend(blend(part[:i_a], middle, n), part[i_z:], n)
    delta = ((i_a - n + len(middle) - n) - i_z) / rate                      # how far everything after the run moves
    moved: list[Mapping[str, Any] | None] = []
    for k, t in enumerate(times):
        if t is None or k < first:
            moved.append(t)
        elif k <= last:
            sw = snippet_times[s_first + (k - first)]
            if not sw:
                return None
            moved.append({**t, "start": (i_a - n + (float(sw["start"]) * rate - j_a)) / rate, "end": (i_a - n + (float(sw["end"]) * rate - j_a)) / rate})
        else:
            moved.append({**t, "start": float(t["start"]) + delta, "end": float(t["end"]) + delta})
    return joined, moved


MIX_FADE_S = 0.08              # a name is mixed in and out over this long, at the quietest point around the words either side of it
MIX_BOUND_REACH_S = 0.12       # a join is looked for this far back into the word before the name (and on into the word after it), never inside the name
MIX_NAME_MARGIN_S = 0.02       # a cross-fade clears the name's own edge by this much, since the aligner's edge is not exact
MIX_VALLEY_DB = -25.0           # a quiet point worth joining at is at least this far below the track's speech level (a pause, a stop's closure)
MIX_JOIN_REACH_S = 0.40        # a join is looked for this far back from the name, or on from it, in each track
MIX_LETTER_REJECT = 2.5         # where the aligner gave letter times, a pair of quiet points more letters apart than this is not the same moment of the words and is never joined (a pause with the closure inside the next word)
MIX_SHALLOW_DB = -12.0          # when one track has no valley as deep as that, the search is made again with dips this far under speech (the voice ran the name into the next word)
MIX_SHAPE_DB = 40.0            # a pair of quiet points is worth this much (dB of depth) for every 1.0 of similarity between the loudness shapes around them
MIX_DIST_DB_PER_S = 40.0       # and costs this much per second of difference in how far from the name each is
MIX_LETTER_DB = 15.0           # and this much per letter of difference in where in the words each sits, when the aligner gave the time of each letter (the same text is read in both tracks, so the same event is at the same letter)
MIX_SHAPE_SPAN_S = 0.45        # the loudness shape compared runs this far from a quiet point, on the side that is played from the other track
MIX_MIN_FADE_S = 0.03          # the shortest cross-fade, in a silence too short for the full one
MIX_EXTEND_WORDS = 2           # a stretch with no quiet pair beside it takes the next word too, at most this many times a side
MIX_ONSET_S = 0.04             # where the next word has no room beside the name, a join is made this far into it, past its first sound
MIX_PAUSE_KEEP_S = 0.04        # a pause is shortened to this at the least


def _quietest(audio: np.ndarray, rate: int, t0: float, t1: float) -> int:
    """The sample at the centre of the quietest 20 ms of ``audio`` between the times ``t0`` and ``t1``: a mix there has no speech to echo."""
    i0, i1 = max(0, int(t0 * rate)), min(len(audio), int(t1 * rate))
    window = int(0.02 * rate)
    if i1 - i0 <= window:
        return (i0 + i1) // 2
    csum = np.concatenate([[0.0], np.cumsum(audio[i0:i1].astype(np.float64) ** 2)])
    return i0 + int(np.argmin(csum[window:] - csum[:-window])) + window // 2


def _squeeze_silence(audio: np.ndarray, remove: int, rate: int) -> tuple[np.ndarray, int]:
    """``audio`` shorter by up to ``remove`` samples, all taken out of its silent stretches (each keeps MIX_PAUSE_KEEP_S): the primed sentence pauses around a name where the host does not."""
    hop = int(0.005 * rate)
    count = len(audio) // hop
    if remove <= 0 or count < 4:
        return audio, 0
    rms = np.sqrt((audio[: count * hop].reshape(count, hop) ** 2).mean(axis=1))
    quiet = rms < 0.04 * float(np.percentile(rms, 90))
    runs: list[tuple[int, int, int]] = []
    i = 0
    while i < count:
        if quiet[i]:
            j = i
            while j < count and quiet[j]:
                j += 1
            spare = (j - i) * hop - int(MIX_PAUSE_KEEP_S * rate)
            if spare > 0:
                runs.append((i * hop, j * hop, spare))
            i = j
        else:
            i += 1
    total = sum(spare for _, _, spare in runs)
    if total <= 0:
        return audio, 0
    take = min(remove, total)
    pieces, at, removed = [], 0, 0
    for a, b, spare in runs:
        cut = int(round(spare * take / total))
        middle = (a + b) // 2
        pieces.append(audio[at: middle - cut // 2])
        at = middle + (cut - cut // 2)
        removed += cut
    pieces.append(audio[at:])
    n = int(0.004 * rate)
    out = pieces[0]
    for piece in pieces[1:]:
        out = _crossfade_equal(out, piece, n) if len(out) > n and len(piece) > n else np.concatenate([out, piece])
    return out, removed


def _valleys(audio: np.ndarray, rate: int, t0: float, t1: float, speech: float, depth_db: float | None = None) -> list[tuple[int, float]]:
    """The quiet points of ``audio`` between the times ``t0`` and ``t1``: (sample at the centre, dB below the track's speech level) of each stretch whose 5 ms level is the lowest within 30 ms either side and
    deeper than MIX_VALLEY_DB.  A held silence is one point, at its centre."""
    hop = int(0.005 * rate)
    i0, i1 = max(0, int(t0 * rate)), min(len(audio), int(t1 * rate))
    count = (i1 - i0) // hop
    if count < 5:
        return []
    level = np.sqrt((audio[i0: i0 + count * hop].astype(np.float64).reshape(count, hop) ** 2).mean(axis=1))
    db = np.convolve(20.0 * np.log10(np.maximum(level / max(speech, 1e-9), 1e-5)), np.ones(3) / 3.0, mode="same")
    limit = MIX_VALLEY_DB if depth_db is None else depth_db
    low = [i for i in range(1, count - 1) if db[i] <= limit and db[i] <= db[max(0, i - 6): i + 7].min() + 1e-9]
    points: list[tuple[int, float]] = []
    run: list[int] = []
    for i in low + [None]:
        if i is not None and (not run or i - run[-1] <= 6):
            run.append(i)
            continue
        if run:
            middle = run[len(run) // 2]
            points.append((i0 + middle * hop + hop // 2, float(db[run].min())))
        run = [i] if i is not None else []
    return points


def _letter_times(entries: Sequence[Mapping[str, Any] | None]) -> list[float] | None:
    """The time of every letter of the words ``entries`` (each a word's timing with its "letters": [[letter, seconds]]), in order, or None when a word is missing or has no letter times."""
    out: list[float] = []
    for entry in entries:
        letters = entry.get("letters") if entry else None
        if not letters:
            return None
        out.extend(float(item[1]) for item in letters)
    return out if len(out) >= 2 else None


def _letter_position(times: Sequence[float], t: float) -> float:
    """Where ``t`` falls in the run of letters: 3.0 at the third letter's time, 3.5 half way to the next; before the first or after the last, the distance in letters at 0.1 s a letter."""
    if t <= times[0]:
        return (t - times[0]) / 0.1
    if t >= times[-1]:
        return len(times) - 1 + (t - times[-1]) / 0.1
    k = max(i for i in range(len(times)) if times[i] <= t)
    return k + (t - times[k]) / max(times[k + 1] - times[k], 1e-6)


def _silence_extent(audio: np.ndarray, rate: int, centre: int, speech: float) -> int:
    """How many samples around ``centre`` stay more than MIX_VALLEY_DB below the track's speech level (5 ms steps, at most 0.5 s each way): the width of the silence or closure that point is the middle of."""
    hop = int(0.005 * rate)
    reach = int(0.5 * rate)

    def frames(direction: int) -> int:
        n, at = 0, centre
        while n * hop < reach:
            lo, hi = (at, at + hop) if direction > 0 else (at - hop, at)
            if lo < 0 or hi > len(audio):
                break
            level = 20.0 * np.log10(max(float(np.sqrt(np.mean(audio[lo:hi].astype(np.float64) ** 2))) / max(speech, 1e-9), 1e-5))
            if level > MIX_VALLEY_DB:
                break
            n, at = n + 1, at + direction * hop
        return n * hop

    return frames(1) + frames(-1)


def _loudness_db(audio: np.ndarray, rate: int, start: int, span: float, speech: float) -> np.ndarray | None:
    """The loudness of ``audio`` in 5 ms steps over ``span`` seconds from sample ``start`` (a negative ``span`` runs back from it), in dB below the track's speech level (floored at -40 dB), or None past an end."""
    hop = int(0.005 * rate)
    count = int(abs(span) / 0.005)
    lo = start if span > 0 else start - count * hop
    if lo < 0 or lo + count * hop > len(audio) or count < 4:
        return None
    level = np.sqrt((audio[lo: lo + count * hop].astype(np.float64).reshape(count, hop) ** 2).mean(axis=1))
    return np.clip(20.0 * np.log10(np.maximum(level / max(speech, 1e-9), 1e-5)), -40.0, 6.0)


def _shape_match(host: np.ndarray, c_host: int, primed: np.ndarray, c_primed: int, rate: int, host_speech: float, primed_speech: float, side: str) -> float:
    """How alike the two tracks are around a pair of quiet points where it matters (correlation, -1..1): for a join after the name, the MIX_SHAPE_SPAN_S that follows it (the host plays on from there, so
    it must be the same syllables as the primed sentence has after its point); for a join before the name, the same span that precedes it.  A pause is not like the closure inside a word, though both are quiet
    and both follow the same loud speech."""
    span = MIX_SHAPE_SPAN_S if side == "after" else -MIX_SHAPE_SPAN_S
    a, b = _loudness_db(host, rate, c_host, span, host_speech), _loudness_db(primed, rate, c_primed, span, primed_speech)
    if a is None or b is None or float(np.std(a)) < 1e-6 or float(np.std(b)) < 1e-6:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


MIX_SYNC_REACH_S = 0.04        # the primed track is moved at most this far to line its syllables up with the host's at a join
MIX_SYNC_CONTEXT_S = 0.15      # about a syllable: the stretch of each track whose loudness shape is compared
MIX_SYNC_MIN_CORR = 0.4        # the shapes must be at least this alike (correlation) to trust the lag
MIX_LOUD = 0.15                # a join has speech in it when either track is this share of its own speech level there


def _loudness_shape(audio: np.ndarray, hop: int) -> np.ndarray:
    count = len(audio) // hop
    return np.sqrt((audio[: count * hop].astype(np.float64).reshape(count, hop) ** 2).mean(axis=1))


def sync_join(host: np.ndarray, c_host: int, snippet: np.ndarray, c_snippet: int, rate: int, width: int) -> dict[str, float | bool]:
    """Similarity-guided syllable sync at a join.  The cross-fade runs over ``width`` samples centred on ``c_host`` in the host and ``c_snippet`` in the primed sentence.  Where either track has speech there
    and the two are not in step, the cross-fade is heard as an echo: two renditions of the same syllable, a few hundredths of a second apart.  The lag that lines the primed track up is found from the
    loudness shape of a syllable (about MIX_SYNC_CONTEXT_S) of each, within MIX_SYNC_REACH_S, then fine-tuned to the sample from the waveforms over the cross-fade itself.  It is used only when
    there is speech at the join and the shapes agree (MIX_SYNC_MIN_CORR).  Returns the lag (samples to add to ``c_snippet``), whether it applies, and what it was measured from."""
    hop = max(1, rate // 400)
    context, reach, half = int(MIX_SYNC_CONTEXT_S * rate), int(MIX_SYNC_REACH_S * rate), width // 2
    out: dict[str, float | bool] = {"lag": 0, "applied": False, "env_corr": 0.0, "wave_corr": 0.0, "level_host": 0.0, "level_snippet": 0.0}
    if c_host - half < 0 or c_host + half > len(host) or c_snippet - half < 0 or c_snippet + half > len(snippet):
        return out
    host_speech, snippet_speech = _speech_level(host, rate), _speech_level(snippet, rate)
    out["level_host"] = round(float(np.sqrt(np.mean(host[c_host - half: c_host + half] ** 2))) / max(host_speech, 1e-9), 3)
    out["level_snippet"] = round(float(np.sqrt(np.mean(snippet[c_snippet - half: c_snippet + half] ** 2))) / max(snippet_speech, 1e-9), 3)
    h0 = c_host - context // 2
    if h0 < 0 or h0 + context > len(host):
        return out
    shape = _loudness_shape(host[h0: h0 + context], hop)
    if float(np.std(shape)) < 1e-9:
        return out
    best_lag, best = 0, -2.0
    for lag in range(-reach, reach + 1, hop):
        s0 = c_snippet - context // 2 + lag
        if s0 < 0 or s0 + context > len(snippet):
            continue
        other = _loudness_shape(snippet[s0: s0 + context], hop)
        if len(other) != len(shape) or float(np.std(other)) < 1e-9:
            continue
        corr = float(np.corrcoef(shape, other)[0, 1])
        if corr > best:
            best_lag, best = lag, corr
    out["env_corr"] = round(best, 3)
    ref = host[c_host - half: c_host + half].astype(np.float64)
    ref_norm = float(np.linalg.norm(ref)) + 1e-9
    fine_lag, fine = best_lag, -2.0
    for lag in range(best_lag - hop, best_lag + hop + 1):
        a = c_snippet + lag - half
        if a < 0 or a + width > len(snippet):
            continue
        seg = snippet[a: a + width].astype(np.float64)
        corr = float(np.dot(ref, seg)) / (ref_norm * (float(np.linalg.norm(seg)) + 1e-9))
        if corr > fine:
            fine_lag, fine = lag, corr
    out["wave_corr"] = round(fine, 3)
    if max(float(out["level_host"]), float(out["level_snippet"])) >= MIX_LOUD and best >= MIX_SYNC_MIN_CORR:
        out["lag"], out["applied"] = fine_lag, True
    return out


def mix_word_run(part: np.ndarray, rate: int, times: Sequence[Mapping[str, Any] | None], first: int, last: int,
                 snippet: np.ndarray, snippet_times: Sequence[Mapping[str, Any] | None], s_first: int, s_last: int, opens_sentence: bool = False,
                 sync: bool = True, notes: list[dict[str, Any]] | None = None, extend: int = MIX_EXTEND_WORDS) -> tuple[np.ndarray, list[Mapping[str, Any] | None]] | None:
    """``part`` with its words ``first``..``last`` (a name) taken from ``snippet`` (another generation of the sentence): (new audio, new word times), or None when a word the mix needs has no time.

    It is a mix, not a splice: the host plays into the primed sentence and back, cross-faded for MIX_FADE_S in a real silence beside the name in both tracks (a shorter fade where the silence is shorter), or else at the quiet point of the bounding word (found in
    each track on its own), so the energy and the rhythm carry over.  The host's own stretch from join to join (the placeholder) is dropped whole and nothing is cut out of the primed speech: its pauses
    are shortened where it pauses more than the host did, and the rest of the sentence moves by the difference in length.  Where the next word starts too close to the name in either track for the
    cross-fade to fit between them, the join is made inside that word past its first sound (MIX_ONSET_S), at the same point in both tracks: the fade never straddles a word's start, which is heard as a hiccup.
    A name that opens its sentence (``opens_sentence``) puts the primed sentence's own beginning in front of it, since what is in front of it is the lead, cut away afterwards.  With ``sync`` each join
    where there is speech is first lined up with sync_join (the primed track moved by the lag it finds); ``notes`` gets one record per mix: the joins' levels, lags and similarities."""
    wb, wa = (times[first - 1] if first > 0 else None), (times[last + 1] if last + 1 < len(times) else None)
    sb, sa = (snippet_times[s_first - 1] if s_first > 0 else None), (snippet_times[s_last + 1] if s_last + 1 < len(snippet_times) else None)
    if not (wb and wa and sb and sa and times[first] and times[last] and snippet_times[s_first] and snippet_times[s_last]):
        return None
    width = int(MIX_FADE_S * rate)
    clear = width / 2 / rate + MIX_NAME_MARGIN_S                  # from the centre of a join to the name's edge: the whole cross-fade lies in the bounding word, clear of the name in BOTH tracks

    def before_name(audio: np.ndarray, word: Mapping[str, Any], name_start: float) -> int:
        latest = name_start - clear
        return _quietest(audio, rate, min(max(float(word["start"]) + width / 2 / rate, latest - MIX_BOUND_REACH_S), latest - 0.01), latest + 0.01)

    def after_name(audio: np.ndarray, word: Mapping[str, Any], name_end: float) -> int:
        earliest = name_end + clear
        return _quietest(audio, rate, earliest - 0.01, max(min(float(word["end"]) - width / 2 / rate, earliest + MIX_BOUND_REACH_S), earliest + 0.01))

    h_n0, h_n1 = float(times[first]["start"]), float(times[last]["end"])
    s_n0, s_n1 = float(snippet_times[s_first]["start"]), float(snippet_times[s_last]["end"])
    c_hl, c_sl = before_name(part, wb, h_n0), before_name(snippet, sb, s_n0)
    c_hr, c_sr = after_name(part, wa, h_n1), after_name(snippet, sa, s_n1)
    h_speech, s_speech = _speech_level(part, rate), _speech_level(snippet, rate)
    half = width / 2 / rate
    wl = wr = width

    def join_between(side: str) -> tuple[int, int, int] | None:
        """The pair of quiet points (one in each track), and the fade to use, that a cross-fade beside the name is best made at: both deep (a pause, or a stop's closure inside a word), the loudness around
        them alike (so the pause is paired with the pause and the closure with the closure), and about as far from the name in each.  The fade is as long as the silence they sit in where that is the shorter
        (never less than MIX_MIN_FADE_S), else MIX_FADE_S; it clears the name by MIX_NAME_MARGIN_S.  The word times of a spelled acronym are not exact, so only what is heard decides."""
        if side == "before":
            h_ref, s_ref = h_n0, s_n0
            h_lo, s_lo = max(float(wb["start"]) + half, min(h_ref, float(wb["end"])) - MIX_JOIN_REACH_S), max(float(sb["start"]) + half, min(s_ref, float(sb["end"])) - MIX_JOIN_REACH_S)          # the reach is counted from the word before, so a long pause in one track does not hide the word's own closures
            h_hi, s_hi = h_ref - 0.02, s_ref - 0.02
        else:
            h_ref, s_ref = h_n1, s_n1
            h_lo, s_lo = h_ref, s_ref
            h_hi, s_hi = max(float(wa["end"]) - half, h_lo + 0.02), max(float(sa["end"]) - half, s_lo + 0.02)
            h_hi, s_hi = min(h_hi, max(h_ref, float(wa["start"])) + MIX_JOIN_REACH_S), min(s_hi, max(s_ref, float(sa["start"])) + MIX_JOIN_REACH_S)          # counted from the next word, for the same reason
        words_host = (times[first - 1], times[first]) if side == "before" else (times[last], times[last + 1])
        words_primed = (snippet_times[s_first - 1], snippet_times[s_first]) if side == "before" else (snippet_times[s_last], snippet_times[s_last + 1])
        letters_host, letters_primed = _letter_times(words_host), _letter_times(words_primed)
        if letters_host is not None and (letters_primed is None or len(letters_host) != len(letters_primed)):
            letters_host = letters_primed = None
        for depth in (None, MIX_SHALLOW_DB):                 # the deep quiet points first; where they give no pair that is the same moment in both tracks, the shallower dips
            host_valleys, primed_valleys = _valleys(part, rate, h_lo, h_hi, h_speech, depth), _valleys(snippet, rate, s_lo, s_hi, s_speech, depth)
            best: tuple[float, int, int, int] | None = None
            for c_h, db_h in host_valleys:
                ext_h = _silence_extent(part, rate, c_h, h_speech)
                for c_s, db_s in primed_valleys:
                    ext_s = _silence_extent(snippet, rate, c_s, s_speech)
                    fade = max(int(MIX_MIN_FADE_S * rate), min(width, ext_h, ext_s) if min(ext_h, ext_s) >= int(0.05 * rate) else width)
                    margin = MIX_NAME_MARGIN_S * rate
                    clear_ok = (c_h + fade / 2 <= h_ref * rate - margin and c_s + fade / 2 <= s_ref * rate - margin) if side == "before" else (c_h - fade / 2 >= h_ref * rate + margin and c_s - fade / 2 >= s_ref * rate + margin)
                    if fade > min(ext_h, ext_s) and not clear_ok:                                    # a fade that fits inside both silences has nothing spoken under it; a closure's neighbours are speech, which must not be the name's
                        continue
                    apart = abs(abs(c_h / rate - h_ref) - abs(c_s / rate - s_ref))
                    cost = (db_h + db_s) / 2.0 - MIX_SHAPE_DB * _shape_match(part, c_h, snippet, c_s, rate, h_speech, s_speech, side) + MIX_DIST_DB_PER_S * apart
                    if letters_host is not None and letters_primed is not None:
                        letters_apart = abs(_letter_position(letters_host, c_h / rate) - _letter_position(letters_primed, c_s / rate))
                        if letters_apart > MIX_LETTER_REJECT:
                            continue
                        cost += MIX_LETTER_DB * letters_apart
                    if best is None or cost < best[0]:
                        best = (cost, c_h, c_s, fade)
            if best is not None:
                return (best[1], best[2], best[3])
        return None

    found_before, found_after = join_between("before"), join_between("after")
    if found_after is None and extend > 0 and last + 2 < len(times) and last + 2 < len(snippet_times) and s_last + 2 < len(snippet_times) and times[last + 2] and snippet_times[s_last + 2]:
        return mix_word_run(part, rate, times, first, last + 1, snippet, snippet_times, s_first, s_last + 1, opens_sentence, sync, notes, extend - 1)          # no quiet pair after the name (the voice ran it into the next word): the next word is taken too, and the join is made after it
    if found_before is None and not opens_sentence and extend > 0 and first >= 2 and s_first >= 2 and times[first - 2] and snippet_times[s_first - 2]:
        return mix_word_run(part, rate, times, first - 1, last, snippet, snippet_times, s_first - 1, s_last, opens_sentence, sync, notes, extend - 1)
    if found_before:
        c_hl, c_sl, wl = found_before
    if found_after:
        c_hr, c_sr, wr = found_after
    joins: list[dict[str, Any]] = []
    if not opens_sentence:
        joins.append({"side": "before", **sync_join(part, c_hl, snippet, c_sl, rate, wl)})
    joins.append({"side": "after", **sync_join(part, c_hr, snippet, c_sr, rate, wr)})
    for j in joins:
        if sync and j["applied"]:
            if j["side"] == "before":
                lag = min(int(j["lag"]), int((s_n0 - (wl / 2 / rate + MIX_NAME_MARGIN_S)) * rate) - c_sl)          # a lag never takes the fade over the primed name's edge (the name would be faded in, clipped)
                c_sl += lag
            else:
                lag = max(int(j["lag"]), int((s_n1 + wr / 2 / rate + MIX_NAME_MARGIN_S) * rate) - c_sr)
                c_sr += lag
            j["lag"] = lag
    hl0, s0, hr0, t0 = c_hl - wl // 2, c_sl - wl // 2, c_hr - wr // 2, c_sr - wr // 2
    mid_s = snippet[c_sl:t0] if opens_sentence else snippet[s0 + wl:t0]
    mid_h = hr0 - c_hl if opens_sentence else hr0 - (hl0 + wl)
    if mid_h < int(0.05 * rate) or len(mid_s) < int(0.05 * rate) or hl0 < 0 or s0 < 0 or hr0 + wr > len(part) or t0 + wr > len(snippet):
        return None
    need = len(mid_s) - mid_h                                   # > 0: the name is longer than its stretch
    asked, squeezed = need, 0
    if need > 0:
        mid_s, squeezed = _squeeze_silence(mid_s, need, rate)
        need -= squeezed
    mid = mid_s
    ramp_after = np.linspace(0.0, np.pi / 2.0, wr, dtype=np.float32)
    out_of = snippet[t0:t0 + wr] * np.cos(ramp_after) + part[hr0:hr0 + wr] * np.sin(ramp_after)
    if opens_sentence:
        joined = np.concatenate([snippet[:c_sl], mid, out_of, part[hr0 + wr:]])
        anchor, name_from, delta = c_sl, c_sl, (c_sl + len(mid) - hr0) / rate
    else:
        ramp_before = np.linspace(0.0, np.pi / 2.0, wl, dtype=np.float32)
        into = part[hl0:hl0 + wl] * np.cos(ramp_before) + snippet[s0:s0 + wl] * np.sin(ramp_before)
        joined = np.concatenate([part[:hl0], into, mid, out_of, part[hr0 + wr:]])
        anchor, name_from, delta = hl0 + wl, s0 + wl, (len(mid) - mid_h) / rate
    scale = len(mid) / max(1, len(mid_s))
    moved: list[Mapping[str, Any] | None] = []
    for k, t in enumerate(times):
        if t is None:
            moved.append(t)
        elif k < first:
            j = s_first - (first - k)
            moved.append({**t, "start": float(snippet_times[j]["start"]), "end": float(snippet_times[j]["end"])} if opens_sentence and j >= 0 and snippet_times[j] else t)
        elif k <= last:
            sw = snippet_times[s_first + (k - first)]
            if not sw:
                return None
            moved.append({**t, "start": (anchor + (float(sw["start"]) * rate - name_from) * scale) / rate, "end": (anchor + (float(sw["end"]) * rate - name_from) * scale) / rate})
        else:
            moved.append({**t, "start": float(t["start"]) + delta, "end": float(t["end"]) + delta})
    if notes is not None:
        notes.append({"words": [first, last], "synced": sync, "longer_ms": round(1000.0 * asked / rate), "pauses_out_ms": round(1000.0 * squeezed / rate), "joins": [{"side": j["side"], "level_host": j["level_host"], "level_snippet": j["level_snippet"], "env_corr": j["env_corr"], "wave_corr": j["wave_corr"],
                                                                         "lag_ms": round(1000.0 * int(j["lag"]) / rate, 1), "applied": bool(sync and j["applied"]),
                                                                         "host_s": round((c_hl if j["side"] == "before" else c_hr) / rate, 3), "primed_s": round((c_sl if j["side"] == "before" else c_sr) / rate, 3),
                                                                         "fade_ms": round(1000.0 * (wl if j["side"] == "before" else wr) / rate)} for j in joins]})
    return joined.astype(part.dtype), moved


def to_spans(pieces: Sequence[str | int], sentence_gaps: Sequence[float] | None = None) -> tuple[list[str], list[int]]:
    """Speech and silence -> (spans, gaps_ms).  One sentence is one span, so the voice speaks each at its own natural pace; a real pause (a held silence) cuts one as well.

    Two sentences in one generation came out 15-19% shorter than the same two spoken apart (prod-126, unit-01 sentences 3-4: the voice speeds up and drops its weakest syllables, "ngemisebenzi" ->
    "nomsebenzi"), and the agent fits the wording to the windows afterwards, so the voice is not asked to hurry.  ``sentence_gaps`` says how long the SOURCE speaker paused after each sentence of the unit
    (span_cuts): the join after a sentence is as long as that pause, or as short as it can be.  Without it (no timed words, or its count does not match the text) the gap is SPAN_GAP_MS.
    A sentence shorter than MIN_SENTENCE_CHARS is kept with the next one: a generation of a word or two is not a voice."""
    from . import span_cuts

    spans: list[str] = []
    gaps: list[int] = []
    current = ""

    def close(gap: int) -> None:
        nonlocal current
        if current.strip():
            spans.append(current.strip())
            gaps.append(gap)
        current = ""

    split = lambda text: [s.strip() for s in SENTENCE_END.split(text.strip()) if s.strip()]      # noqa: E731
    total = sum(len(split(piece)) for piece in pieces if isinstance(piece, str))
    chosen = sentence_gaps if sentence_gaps is not None and len(sentence_gaps) == total - 1 else None
    index = 0
    for piece in pieces:
        if isinstance(piece, int):
            if current.strip():
                close(piece)
            elif gaps:
                gaps[-1] += piece                 # two silences in a row add up; a silence before any speech is dropped
            continue
        sentences = split(piece)
        for position, sentence in enumerate(sentences):
            current = f"{current} {sentence}".strip()
            if position < len(sentences) - 1 and len(current) >= MIN_SENTENCE_CHARS:
                close(span_cuts.join_gap_ms(chosen[index + position]) if chosen is not None else SPAN_GAP_MS)          # as long as the speaker's own pause there, or as short as it can be
        index += len(sentences)
    close(0)
    return spans, gaps[:-1] if gaps else []




def _decode_answer(answer: Mapping[str, Any]) -> tuple[np.ndarray, int]:
    audio, rate = sf.read(io.BytesIO(base64.b64decode(answer["audio_b64"])), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, int(rate)


def wav_bytes(audio: np.ndarray, rate: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16), rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


class OmniVoiceTTSBackend:
    """A voice backend with the one method ``_synthesize_group`` needs from it, ``synthesize_group``, plus the attributes callers read."""

    provider = "omnivoice"

    def __init__(self, client: OmniVoiceClient, *, default_voice: str = "omnivoice-zu", speed: float = DEFAULT_SPEED, lora: bool = True,
                 banks: Mapping[str, ReferenceBank] | None = None, bank_factory: Callable[[Path], ReferenceBank] = ReferenceBank.for_corpus) -> None:
        self.client = client
        self.default_voice = default_voice
        self.speed = speed
        self.follow_word = True          # a span is spoken with the first word of the next one after it (cut off afterwards) so that its last word is not the last thing the model makes; an experiment can turn it off
        self.name_draws = False          # the final render: a span with names is drawn as many times as the Studio draws it (the names differ draw to draw), and the longest name is used, so the video is the audio that was listened to; a run's measurements make one
        self.primer_usual_pace = True       # a name the speaker drew out in the window is replaced in the primer by their usual saying of it (the drawn-out one drags a whole primed sentence); False for a splice, which takes only the name
        self.host_spaced = False            # the host of a splice is sent the same text a reference-only sentence is (an English word after a Zulu prefix spaced), so it is that sentence exactly
        self.host_per_draw = False          # each draw speaks the cluster's first pass again with its own seed (one speaker's sentence can range from 4% to 15% of it in breaks between seeds, and a host shared by the draws cannot be re-rolled)
        self.splice_refine = False          # a splice cuts where the two generations look alike, matches the name's level to its host and blends with equal power
        self.primer_align = True            # the names the speaker says are cut from their audio at the edges the aligner finds from the name's spelling, not at the recogniser's word times
        self.splice_mix = True              # a name is mixed in from its primed sentence (cross-faded at the quietest points around it, mix_word_run) instead of cut in; it implies host_spaced and the speaker's own saying of a name in the primer
        self.splice_sync = True             # each join of a mix where there is speech is lined up first (sync_join): the primed track moved by the lag at which the two syllables' loudness shapes agree
        self.splice_inside_prefix = False   # a name with a Zulu prefix is taken alone and joined inside its prefix, at the moment the two generations look most alike (not with the word before it)
        self.splice_phrases = False         # and widens each swapped stretch to the nearest real pauses (at most 3 words each way), so its joins fall in silence
        self.priming = True          # False: the reference clip only; no name of the speaker is put in front of it and none is kept out of the pronunciation dictionary (phonetic respelling decides how a name is read)
        self.pin_to_reference = False          # a primed sentence is spoken for the length the plain reference clip gives it (its pace), not the length its primed reference gives it
        self.match_primer_syllables = 0.0          # 0: the names are put in front of the reference as the speaker said them; 1.5 (say): a name said faster than the reference clip is slowed toward its syllable speed, at most this many times longer
        self.splice_names = True         # a cluster's names are cut from their primed sentences and put in; False: the whole span is spoken primed (every name of the span in front of the reference) and used as it is, with no splice
        self.nameless_draws = 1          # how many draws the Studio makes of a unit that carries no names (the draws of a name are the ones listened to; a draw of the speech around it is a call for nothing)
        self.span_sentences = 1          # how many sentences one generation speaks (1 = each its own; an experiment can ask for 2 or 3, see merge_spans)
        self.generation: dict[str, float] | None = None          # the model's decoding settings (position_temperature, guidance_scale, num_step ...); None = the model's own defaults
        self.lora = lora
        self._banks: dict[str, ReferenceBank] = dict(banks or {})
        self._bank_factory = bank_factory

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> "OmniVoiceTTSBackend":
        env = dict(os.environ if environ is None else environ)
        url = (env.get("OMNIVOICE_URL") or "").strip()
        if not url:
            raise RuntimeError("OMNIVOICE_URL is not set (the Cloud Run service URL)")
        client = OmniVoiceClient(url, cache_tag=(env.get("OMNIVOICE_CACHE_TAG") or "").strip())
        if (env.get("OMNIVOICE_CACHE", "1")).strip().lower() not in {"0", "false", "no"}:          # what the voice said before, kept between runs, the studio and the render (omnivoice_cache)
            folder = (env.get("OMNIVOICE_CACHE_DIR") or "").strip()
            if not folder and environ is None:
                try:
                    from .config import load_settings

                    folder = str(Path(load_settings().work_dir) / "voice_cache")
                except Exception:            # noqa: BLE001 - no settings, no cache
                    folder = ""
            if folder:
                from .omnivoice_cache import DEFAULT_MAX_BYTES, SpanCache

                client.cache = SpanCache(folder, max_bytes=int(float(env.get("OMNIVOICE_CACHE_MB") or 0) * 1024 * 1024) or DEFAULT_MAX_BYTES)
        if (env.get("OMNIVOICE_NUM_STEP") or "").strip():
            client.default_generation["num_step"] = int(env["OMNIVOICE_NUM_STEP"])
        backend = cls(client, speed=float(env.get("OMNIVOICE_SPEED") or DEFAULT_SPEED), lora=(env.get("OMNIVOICE_LORA", "1").strip().lower() not in {"0", "false", "no"}))
        backend.name_draws = (env.get("OMNIVOICE_NAME_DRAWS") or "").strip().lower() in {"1", "true", "yes"}
        backend.splice_names = (env.get("OMNIVOICE_SPLICE") or "1").strip().lower() not in {"0", "false", "no", "off"}
        backend.primer_usual_pace = (env.get("OMNIVOICE_PRIMER_USUAL_PACE") or "1").strip().lower() not in {"0", "false", "no", "off"}
        backend.host_spaced = (env.get("OMNIVOICE_HOST_SPACED") or "").strip().lower() in {"1", "true", "yes", "on"}
        backend.host_per_draw = (env.get("OMNIVOICE_HOST_PER_DRAW") or "").strip().lower() in {"1", "true", "yes", "on"}
        backend.splice_refine = (env.get("OMNIVOICE_SPLICE_REFINE") or "").strip().lower() in {"1", "true", "yes", "on"}
        backend.primer_align = (env.get("OMNIVOICE_PRIMER_ALIGN") or "1").strip().lower() not in {"0", "false", "no", "off"}
        backend.splice_mix = (env.get("OMNIVOICE_SPLICE_MIX") or "1").strip().lower() not in {"0", "false", "no", "off"}
        backend.splice_sync = (env.get("OMNIVOICE_SPLICE_SYNC") or "1").strip().lower() not in {"0", "false", "no", "off"}
        backend.splice_inside_prefix = (env.get("OMNIVOICE_SPLICE_INSIDE_PREFIX") or "").strip().lower() in {"1", "true", "yes", "on"}
        backend.splice_phrases = (env.get("OMNIVOICE_SPLICE_PHRASES") or "").strip().lower() in {"1", "true", "yes", "on"}
        backend.priming = (env.get("OMNIVOICE_PRIMING") or "1").strip().lower() not in {"0", "false", "no", "off"}
        backend.pin_to_reference = (env.get("OMNIVOICE_PIN_TO_REFERENCE") or "").strip().lower() in {"1", "true", "yes", "on"}
        try:
            backend.match_primer_syllables = float(env.get("OMNIVOICE_PRIMER_SYLLABLES") or 0.0)
        except ValueError:
            backend.match_primer_syllables = 0.0
        if (env.get("OMNIVOICE_PARALLEL") or "").strip().isdigit():          # set by warm_voice: the copies of the voice this job holds
            client.set_parallel(int(env["OMNIVOICE_PARALLEL"]))
        per_span = (env.get("OMNIVOICE_SPAN_SENTENCES") or "").strip()
        if per_span.lower() == "fill":
            backend.span_sentences = FILL_SENTENCES
        elif per_span.isdigit() and 1 <= int(per_span) <= 4:          # an experiment for a whole run: how many sentences one generation speaks (see merge_spans); the studio's per-run voice.json says the same for one run
            backend.span_sentences = int(per_span)
        return backend

    def bank_for(self, job_root: Path | str | None) -> ReferenceBank:
        if job_root is None:
            raise RuntimeError("the OmniVoice voice needs the job's corpus folder (job_root) to cut reference clips from its audio")
        key = str(job_root)
        if key not in self._banks:
            self._banks[key] = self._bank_factory(Path(job_root))
            if self.primer_align:
                from . import source_align

                self._banks[key].aligner = source_align.shared_aligner()
        return self._banks[key]

    def span_primers(self, group: Mapping[str, Any], spans: Sequence[str], job_root: Path | str | None) -> list[dict[str, Any] | None]:
        """For each span, the speaker's own names it carries (ReferenceBank.primer_for), or None."""
        if not self.priming:
            return [None] * len(spans)
        try:
            bank = self.bank_for(job_root)
            start = int(group.get("start_ms") or 0)
            end = int(group.get("source_end_ms") or (start + int(group.get("source_span_ms") or 0)))
            return [bank.primer_for(str(group.get("speaker_id") or ""), start / 1000.0, end / 1000.0, span, self.primer_usual_pace and not (self.splice_mix and self.splice_names), self.primer_align) for span in spans]
        except Exception:                      # noqa: BLE001 - a primer is an improvement, never a reason a unit cannot be spoken
            return [None] * len(spans)

    def reference_for(self, group: Mapping[str, Any], job_root: Path | str | None) -> Reference:
        start = int(group.get("start_ms") or 0)
        end = int(group.get("source_end_ms") or (start + int(group.get("source_span_ms") or 0)))
        return self.bank_for(job_root).pick(str(group.get("speaker_id") or ""), start, end)

    def plan(self, group: Mapping[str, Any], voice_info: Mapping[str, Any], job_root: Path | str | None) -> tuple[list[str], list[int]]:
        """The group's text as spans and silences, after the same pronunciation dictionary the Azure path applies."""
        from .native_dub import _native_dub_tts_ready_parts

        try:
            start = int(group.get("start_ms") or 0)
            end = int(group.get("source_end_ms") or (start + int(group.get("source_span_ms") or 0)))
            matches = self.bank_for(job_root).name_matches(str(group.get("speaker_id") or ""), start / 1000.0, end / 1000.0, _group_text(group)) if self.priming else []          # no priming: no name is left to the speaker's audio, every one goes through the dictionary
        except Exception:                      # noqa: BLE001 - no bank or no timed words: the text stays as written
            matches = []
        covered = sorted({t for m in matches for t in m["tokens"]}, key=lambda t: (-len(t), t))          # names the speaker says in this window: the source audio is their pronunciation
        names = {t.lower() for t in covered}
        pieces: list[str | int] = []
        for part in group["parts"]:
            if part["type"] == "break":
                pieces.append(int(part["duration_ms"]))
                continue
            for piece in split_marked_pauses(str(part["text"])):
                if isinstance(piece, int):
                    pieces.append(piece)
                    continue
                piece = abbreviations.expand(piece)[0]          # "Mnu Mashaba" -> "Mnumzana Mashaba", "Dr. Smith" -> "Doctor Smith", "20 km" -> "20 kilometres": every abbreviation with a spoken form, before the names are put out of the dictionary's reach
                protected, mapping = protect_names(piece, covered)                          # the dictionary reads the rest, not these
                ready = _native_dub_tts_ready_parts(protected, job_root=job_root, voice=str(voice_info.get("selected_voice") or ""))
                for out in ([ready] if isinstance(ready, str) else flatten_parts(ready)):
                    pieces.append(restore_names(out, mapping) if isinstance(out, str) else out)
        pieces = [decouple_names(separate_name_prefixes(piece, names), names) if isinstance(piece, str) else piece for piece in pieces]
        spans, gaps = to_spans(pieces, self.sentence_gaps(group, job_root))
        if self.span_sentences > 1:
            try:                                                           # as many sentences as fit: the cluster's estimated speech time (the free estimate, from the reference's pace) stays within CLUSTER_MAX_S
                reference = self.reference_for(group, job_root)
                fits = lambda text: od.estimate_seconds([text], [], reference.text, reference.seconds, self.speed) <= CLUSTER_MAX_S      # noqa: E731
            except Exception:                                              # noqa: BLE001 - no reference to read the pace from: the character cap stands
                fits = None
            return merge_spans(spans, gaps, self.span_sentences, fits=fits)
        return spans, gaps

    def sentence_gaps(self, group: Mapping[str, Any], job_root: Path | str | None) -> list[float] | None:
        """How long the source speaker paused after each sentence of this unit's text (span_cuts), or None when it cannot be told: the spans are then cut for length alone."""
        try:
            from . import span_cuts
            from .comma_check import _window

            text = " ".join(str(part.get("text") or "") for part in group["parts"] if part.get("type") != "break")
            start = int(group.get("start_ms") or 0)
            end = int(group.get("source_end_ms") or (start + int(group.get("source_span_ms") or 0)))
            inside = _window(self.bank_for(job_root).words, start, end)
            root = Path(str(job_root))
            aligner = span_cuts.get_aligner(str(root.parents[2])) if len(root.parents) > 2 else None
            return span_cuts.boundary_gaps(text, inside, aligner)
        except Exception:                      # noqa: BLE001 - where to cut is an improvement, never a reason a unit cannot be spoken
            return None

    def speak(self, group: Mapping[str, Any], spans: Sequence[str], gaps: Sequence[int], job_root: Path | str | None, retakes: Mapping[int, int] | None = None, takes_out: list[Any] | None = None,
              all_takes: bool = False, raw_out: list[dict[str, Any]] | None = None) -> tuple[np.ndarray, int, dict[str, Any], dict[str, Any], Reference]:
        """The unit spoken as the voice would naturally say it, cloned from the best clip of that speaker: (audio, rate, an answer-shaped description, a report, the reference used).

        Nothing is pinned, stretched or cut.  The voice varies from run to run (a generation can start with a pitch spike or leave quiet half-voiced stretches), so the unit is synthesised up to
        MAX_TAKES times and the best take of each span is kept (voice_takes); the spans are joined over the source's room tone, not digital silence.  When even the best takes leave a lot of the
        voice's own silence inside the speech, one more synthesis is made with the next-best reference clip and used if it scores better overall."""
        from . import voice_takes as vt

        bank = self.bank_for(job_root)
        ranked = bank.ranked(str(group.get("speaker_id") or ""), int(group.get("start_ms") or 0),
                             int(group.get("source_end_ms") or (int(group.get("start_ms") or 0) + int(group.get("source_span_ms") or 0))))
        clip_edges = (float(ranked[0]["start"]), float(ranked[0]["end"]))          # a draw the listener committed belongs to the clip as the Studio names it (the clip's own edges), not to where this render snaps them to the quiet: the lookup by the snapped edges missed by 0.06 s and no commit ever applied
        tone, delta = bank.room_tone()
        pause_ok = bank.natural_pause_s(str(group.get("speaker_id") or ""))          # the longest pause the speaker themself makes inside a sentence: shorter ones are not held against a take
        breath = bank.natural_pause_s(str(group.get("speaker_id") or ""), 75.0) or 0.0          # what the speaker leaves at least after a sentence: a full stop is a breath (prod-129 0:39: 'ethile.' and 'Umane' 0.06 s apart)
        primers = self.span_primers(group, spans, job_root)
        from . import draws as dr

        marked = dr.load(job_root)                                      # the draws the listener approved (draws.py); ``retakes`` tries others without keeping them

        leads = [None] + [vt.lead_sentence(spans[i - 1]) for i in range(1, len(spans))]
        follows = [(_first_word(spans[i + 1]) if i + 1 < len(spans) else TRAILING_FOLLOW) if self.follow_word else None for i in range(len(spans))]          # the first word of the span after: spoken after this one so its last word is not the last thing the model makes (the last span of a unit has none: a plain word, cut away like the others)
        notes: list[list[dict[str, Any]]] = []
        take_count = [0]                                                 # which take this is: every take is a draw of its own, and the same one every time
        join_notes: list[dict[str, Any]] = []
        pin_notes: list[dict[str, Any]] = []
        first_pass: dict[Any, Any] = {}                                    # the unprimed pass of the cluster, made by the first draw and shared by the others
        raw_notes: list[dict[str, Any]] = []                              # the primed sentences the names were cut from, exactly as the voice made them (the studio plays them)

        def synth(*, keep: Sequence[Iterable[str]] | None = None, **ask: Any) -> dict[str, Any]:
            """One request to the voice.  An English word kept after a glued Zulu prefix ("ama-details") is sent as two words, because the voice reads the glued form as one Zulu word and drops the
            word's consonants; the word times come back folded to the words of the text, so nothing else sees the difference.  A word in ``keep[i]`` (the names of span i the speaker says: they are cut out of
            a primed sentence and put in) stays glued: the voice then leaves the word after it whole, and the word itself is the speaker's (abbreviations.separate_stems)."""
            texts = list(ask["spans"])
            spaced = [abbreviations.separate_stems(text, keep[i] if keep and i < len(keep) and keep[i] else ()) for i, text in enumerate(texts)]
            if all(all(g == 1 for g in groups) for _, groups in spaced):
                return self.client.synthesize(**ask)
            reply = self.client.synthesize(**{**ask, "spans": [text for text, _ in spaced]})
            words = reply.get("words")
            if isinstance(words, list) and len(words) == len(spaced):
                reply = {**reply, "words": [abbreviations.fold_words(w, groups) for w, (_, groups) in zip(words, spaced)]}
            return reply

        def take(reference: Reference) -> tuple[list[tuple[np.ndarray, int, int]], list[dict[str, Any]], int, float]:
            """One synthesis of the unit.  A span that follows another is spoken with the last sentence of the one before in front of it, so the voice does not start cold, and after the generation the
            repeated sentence is cut off between its last word and the first new one.  Where that is: the service aligns the known text with its own audio (one request carries every span of the unit and the word
            times of each); a service that cannot is answered by speaking each lead alone as well and warping it onto the start of the generation (voice_takes.align_end)."""
            use_words = getattr(self, "service_aligns", None) is not False          # a unit of one sentence has no lead to cut, but its words are the Studio's word strip and the cut of its stray ends

            refs = [bank.primed(reference, primer, self.match_primer_syllables) for primer in primers]
            take_no = take_count[0]
            take_count[0] += 1
            salt = f"{reference.start_s:.2f}-{reference.end_s:.2f}"

            def seed_for(index: int, text: str, *parts: Any) -> int:
                """The seed of one generation: the text, the take and what the generation is, plus the span's approved retake (nothing added at 0, so an untouched sentence keeps its draw)."""
                if index >= len(spans):
                    n = 0
                elif retakes is not None and index in retakes:
                    n = int(retakes[index])
                else:
                    n = dr.retake_for(marked, spans[index], *clip_edges)
                return stable_seed(text, take_no, *parts, *((("retake", n),) if n else ()))

            def primed_pass(sent: list[str], pool: Any) -> dict[str, Any]:
                """Pronunciation from the names, pace from the speaker.  Every span is spoken once with the fluent clip (the audio of the spans that carry no names, and the length the speaker's own pace gives the
                rest); each span that carries names is then spoken again with the names-then-clip reference, and replaces that span.  The two lengths are the model's own readings of the same sentence; the
                slower one wins.  A primed span that comes out SHORTER (the names, said fast, made the voice read the reference as a faster speaker: 7.0 s against 8.6 s, "ngemisebenzi" -> "nomsebenzi") is spoken
                once more pinned to the fluent length, which only stretches it.  One that comes out LONGER is left as it is: pinning it down squeezed the short syllable in front of a name ("ngu-Eric" heard as
                "Eric": 0 of 5 clean pinned, 5 of 5 clean unpinned, prod-125 1:01)."""
                host_key = (salt, tuple(sent), take_no if self.host_per_draw else None)          # the cluster is spoken once and its draws differ in how the names are primed - or, with host_per_draw, each draw has a first pass of its own
                reused = host_key in first_pass
                first_future = None if reused else pool.submit(synth, spans=sent, gaps_ms=[0] * (len(sent) - 1), reference=reference, speed=self.speed, lora=self.lora, generation=self.generation, align=True,
                                                               seeds=[seed_for(i, text, "first", salt) for i, text in enumerate(sent)], keep=None if (self.host_spaced or (self.splice_mix and self.splice_names)) else [p["names"] if p else () for p in primers])
                futures: dict[Any, Any] = {}

                def primed_kwargs(index: int, text: str, tag: str) -> dict[str, Any]:
                    kwargs = dict(spans=[text], gaps_ms=[], reference=refs[index], speed=self.speed, lora=self.lora, generation=self.generation, align=True, seeds=[seed_for(index, text, tag, salt, primers[index]["names"])])
                    if self.pin_to_reference:          # the length is what the PLAIN reference gives this text (the model's own estimate from the clip's characters a second): the names in front of it say how the names sound, and nothing else
                        kwargs["durations_s"] = [round(od.estimate_seconds([text], [], reference.text, reference.seconds, self.speed), 3)]
                    return kwargs

                def fetch(key: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
                    """The reply of a primed generation: the one already on its way (submitted with the first pass, so they are spoken side by side), or made now."""
                    future = futures.get(key)
                    return future.result() if future is not None else synth(**kwargs)

                def sentence_plan(index: int, j: int, sentences_in: list[str]) -> tuple[str | None, str | None, str, int]:
                    """For the j-th sentence of a cluster: what is spoken before it, the word after it, the text of the generation and how many words of it come first (the way the pipeline speaks any sentence)."""
                    before = vt.lead_sentence(sentences_in[j - 1] if j > 0 else (leads[index] or ""))
                    after = _first_word(sentences_in[j + 1]) if j + 1 < len(sentences_in) else follows[index]
                    return before, after, " ".join(x for x in (before, sentences_in[j], after) if x), len(before.split()) if before else 0

                for index, primer in enumerate(primers):                          # nothing a primed generation says depends on the first pass: they are asked for now, and used when the first pass has come back
                    if primer is None:
                        continue
                    sentences_in = [x for x in vt.SENTENCE_END.split(spans[index].strip()) if x.strip()]
                    if self.splice_names and self.span_sentences > 1 and len(sentences_in) >= 1 and sum(len(x.split()) for x in sentences_in) == len(spans[index].split()):
                        for j in reversed(range(len(sentences_in))):
                            if name_runs(sentences_in[j].split(), primer["names"]):
                                text_in = sentence_plan(index, j, sentences_in)[2]
                                futures[("s", index, text_in)] = pool.submit(synth, **primed_kwargs(index, text_in, "primed-sentence"))
                    else:
                        futures[("b", index)] = pool.submit(synth, **primed_kwargs(index, sent[index], "primed"))
                first = first_pass[host_key] if reused else first_future.result()
                if not isinstance(first.get("words"), list):
                    return first                                                       # the service does not align: the caller asks the old way, with one reference
                first_pass[host_key] = first
                audio_all, rate_seen = _decode_answer(first)
                parts = vt.split_spans(first, audio_all, rate_seen)
                words_by_span = [first["words"][i] if i < len(first["words"]) else None for i in range(len(sent))]
                gpu, aligned = (0.0 if reused else float(first.get("gpu_s") or 0.0)), bool(first.get("aligned", True))
                for index, primer in enumerate(primers):
                    if primer is None:
                        continue
                    sentences_in = [x for x in vt.SENTENCE_END.split(spans[index].strip()) if x.strip()]
                    n_lead = len(leads[index].split()) if leads[index] else 0
                    now_times = words_by_span[index]
                    if self.splice_names and self.span_sentences > 1 and len(sentences_in) >= 1 and isinstance(now_times, list) and len(now_times) == len(sent[index].split()) and sum(len(x.split()) for x in sentences_in) == len(spans[index].split()):
                        ranges, at = [], n_lead
                        for x in sentences_in:
                            ranges.append((at, at + len(x.split()) - 1, x))
                            at += len(x.split())
                        audio_now, times_now, redone = parts[index], list(now_times), 0
                        for j in reversed(range(len(ranges))):                     # a cluster is spoken whole; each sentence that carries a name is spoken again, primed, and only its name is put in place of the cluster's own
                            first_w, last_w, text_s = ranges[j]
                            if not name_runs(text_s.split(), primer["names"]) or not times_now[first_w] or not times_now[last_w]:
                                continue
                            before, after, text_in, n_before = sentence_plan(index, j, sentences_in)          # spoken the way the pipeline speaks any sentence: the one before it in front, the first word of the next after it (both cut away)
                            reply = fetch(("s", index, text_in), primed_kwargs(index, text_in, "primed-sentence"))
                            gpu += float(reply.get("gpu_s") or 0.0)
                            if not isinstance(reply.get("words"), list) or not reply["words"] or len(reply["words"][0]) != len(text_in.split()):
                                continue
                            sentence_all, sentence_rate = _decode_answer(reply)
                            if sentence_rate != rate_seen:
                                continue
                            cut_at = lambda audio, t0, t1: vt.find_cut(audio, rate_seen, (t0 + t1) / 2.0, before_s=abs(t1 - t0) / 2.0 + 0.03, after_s=abs(t1 - t0) / 2.0 + 0.03)[0]      # noqa: E731
                            primed_audio = vt.split_spans(reply, sentence_all, sentence_rate)[0]
                            raw_notes.append({"span": index, "take": take_no, "text": text_in, "audio": primed_audio, "rate": rate_seen, "names": name_lengths(text_in.split(), reply["words"][0], primer["names"]), "source": [{"text": t, "s": round(x, 3)} for t, _, x in primer["snippets"]]})
                            swapped = 0
                            words_s = text_s.split()
                            regions = swap_regions(words_s, primer["names"], with_word_before=not (self.splice_inside_prefix or self.splice_mix))          # the names are taken from the primed sentence (with a word or so between two of them); the cluster keeps everything else
                            if self.splice_phrases:                                    # each stretch widened to real pauses, so the joins fall where nothing is said; stretches that now touch are one
                                regions = merge_regions([pause_bounded(a, b, len(words_s), times_now, first_w, reply["words"][0], n_before) for a, b in regions])
                            mix_notes: list[dict[str, Any]] = []
                            for run_first, run_last in reversed(regions):
                                out = mix_word_run(audio_now, rate_seen, times_now, first_w + run_first, first_w + run_last, primed_audio, reply["words"][0], n_before + run_first, n_before + run_last, run_first == 0, sync=self.splice_sync, notes=mix_notes) if self.splice_mix else None
                                if out is None:                                    # no mix (switched off, or a word beside the name has no time): the name is cut in at its words' edges, or inside its prefix
                                    out = swap_word_run(audio_now, rate_seen, times_now, first_w + run_first, first_w + run_last, primed_audio, reply["words"][0], n_before + run_first, n_before + run_last, cut_at, refine=self.splice_refine or self.splice_mix,
                                                        inside=prefix_share(words_s, run_first) if (self.splice_inside_prefix or self.splice_mix) else None)
                                if out is not None:
                                    (audio_now, times_now), swapped = out, swapped + 1
                            redone += swapped
                            pin_notes.append({"span": index, "sentence": text_s[:40], "lead": bool(before), "follow": bool(after), "names_swapped": swapped, "mix": mix_notes})
                        if redone:
                            parts[index], words_by_span[index] = audio_now, times_now
                            continue
                    natural = len(vt.trim_zero_pads(parts[index])[0]) / rate_seen
                    reply = fetch(("b", index), primed_kwargs(index, sent[index], "primed"))
                    if not isinstance(reply.get("words"), list) or not reply["words"]:
                        return reply
                    audio_part, rate_part = _decode_answer(reply)
                    primed_s = len(vt.trim_zero_pads(vt.split_spans(reply, audio_part, rate_part)[0])[0]) / rate_part
                    pinned = primed_s < natural
                    if pinned:                                                         # the names made it faster than the speaker is: stretch it to the speaker's length
                        gpu += float(reply.get("gpu_s") or 0.0)
                        again = pool.submit(synth, spans=[sent[index]], gaps_ms=[], reference=refs[index], speed=self.speed, lora=self.lora, generation=self.generation, align=True, durations_s=[round(natural, 3)],
                                            seeds=[seed_for(index, sent[index], "stretched", salt, primer["names"])]).result()          # through the same pool: never more requests in the air than copies
                        if isinstance(again.get("words"), list) and again["words"]:
                            again_audio, again_rate = _decode_answer(again)
                            if len(vt.trim_zero_pads(vt.split_spans(again, again_audio, again_rate)[0])[0]) / again_rate > primed_s:          # the pin is not exact: the slower reading wins
                                reply, audio_part, rate_part = again, again_audio, again_rate
                    parts[index] = vt.split_spans(reply, audio_part, rate_part)[0]
                    raw_notes.append({"span": index, "take": take_no, "text": sent[index], "audio": parts[index], "rate": rate_part, "names": name_lengths(sent[index].split(), reply["words"][0], primer["names"]), "source": [{"text": t, "s": round(x, 3)} for t, _, x in primer["snippets"]]})          # the sentence as the voice made it with the names in front, before any cut
                    pin_notes.append({"span": index, "fluent_s": round(natural, 2), "primed_s": round(primed_s, 2), "stretched": pinned, "got_s": round(len(vt.trim_zero_pads(parts[index])[0]) / rate_part, 2)})
                    words_by_span[index] = reply["words"][0]
                    gpu += float(reply.get("gpu_s") or 0.0)
                    aligned = aligned and bool(reply.get("aligned", True))
                starts, position = [], 0
                for part in parts[:len(sent)]:
                    starts.append(position / rate_seen)
                    position += len(part)
                return {"audio_b64": base64.b64encode(wav_bytes(np.concatenate(parts[:len(sent)]), rate_seen)).decode("ascii"), "sample_rate": rate_seen, "span_starts_s": starts,
                        "span_ms": [int(len(part) / rate_seen * 1000) for part in parts[:len(sent)]], "gpu_s": round(gpu, 2), "words": words_by_span, "aligned": aligned}

            def ask_per_reference(sent: list[str]) -> dict[str, Any]:
                """primed_pass with as many requests in the air as there are copies of the voice (one, when the job holds one: the requests then go one after another as they always did)."""
                from concurrent.futures import ThreadPoolExecutor

                pool = ThreadPoolExecutor(max_workers=max(1, int(getattr(self.client, "parallel", 1) or 1)))
                try:
                    return primed_pass(sent, pool)
                finally:
                    pool.shutdown(wait=True)

            def ask(with_words: bool) -> tuple[dict[str, Any], int]:
                heard = [] if with_words else [lead for lead in leads if lead]
                sent = [" ".join(part for part in (lead, span, follow if with_words else None) if part) for lead, span, follow in zip(leads, spans, follows)] + heard
                if with_words and any(ref is not reference for ref in refs):
                    return ask_per_reference(sent), 0
                reply = synth(spans=sent, gaps_ms=[max(0, g) for g in gaps] + [0] * len(heard), reference=reference, speed=self.speed, lora=self.lora, generation=self.generation, **({"align": True} if with_words else {}),
                                               seeds=[seed_for(i, text, "first", salt) for i, text in enumerate(sent)])
                return reply, len(heard)

            answer, heard_count = ask(use_words)
            if use_words:
                words_by_span = answer.get("words")
                if not isinstance(words_by_span, list):                       # the service does not align: remember it, and ask the old way
                    self.service_aligns = False
                    answer, heard_count = ask(False)
                    use_words = False
                else:
                    self.service_aligns = True
                    if any(leads[i] and not _lead_words_ok(words_by_span[i] if i < len(words_by_span) else None, leads[i]) for i in range(len(spans))):
                        answer, heard_count = ask(False)                      # this answer is missing words for a span: speak the leads alone, this once
                        use_words = False
            sent_words = [" ".join(part for part in (lead, span, follow if use_words else None) if part).split() for lead, span, follow in zip(leads, spans, follows)]
            audio, rate = _decode_answer(answer)
            segments = vt.split_spans(answer, audio, rate)
            alone = iter(segments[len(spans):])
            cores: list[tuple[np.ndarray, int, int]] = []
            scores: list[dict[str, Any]] = []
            info: list[dict[str, Any]] = []
            natural: list[float | None] = []                                       # after each span: how long the voice itself left between its last word and the first word of the next span
            for index in range(len(spans)):
                gap_s = None
                if use_words and follows[index]:
                    times = answer["words"][index]
                    n_before = len(leads[index].split()) if leads[index] else 0
                    last_word = next((w for w in reversed(times[n_before:-1]) if w), None) if len(times) == len(sent_words[index]) else None
                    if last_word and times[-1]:
                        gap_s = max(0.0, float(times[-1]["start"]) - float(last_word["end"]))
                natural.append(gap_s)
            target = [max(vt.join_target_s(gaps[i], natural[i]), breath if spans[i].rstrip().endswith((".", "?", "!")) else 0.0) for i in range(len(spans) - 1)]
            for index, segment in enumerate(segments[:len(spans)]):
                core, lead_pad, tail_pad = vt.trim_zero_pads(segment)
                extra = 0.0
                model_pad_s, head_cut_s = lead_pad / rate, 0.0                      # the model's own padding in front of the core, and what was cut from the core's front
                entry: dict[str, Any] = {"span": index}
                if use_words and follows[index]:
                    word_times = answer["words"][index]
                    shift = lead_pad / rate
                    n_lead = len(leads[index].split()) if leads[index] else 0
                    after = word_times[-1] if len(word_times) == len(sent_words[index]) else None
                    before = next((w for w in reversed(word_times[n_lead:-1]) if w), None)
                    if after and before:                                           # the span's last word is cut off at the quiet after it; the word the model said after it goes
                        end_word, start_next = float(before["end"]) - shift, float(after["start"]) - shift
                        low, high = min(end_word, start_next) - 0.03, max(end_word, start_next) + 0.03
                        tail_s, tail_found = vt.find_cut(core, rate, (low + high) / 2, before_s=(high - low) / 2, after_s=(high - low) / 2)
                        last_span = index == len(spans) - 1
                        keep_tail = FINAL_TAIL_KEEP_S if last_span else vt.edge_keep_s(target[index], share=0.6, most_s=vt.TAIL_KEEP_S, least_s=vt.TAIL_MIN_KEEP_S)          # a short join keeps little of the quiet; the end of the unit keeps the word's whole decay
                        tail_s = min(max(tail_s, end_word + vt.TAIL_MIN_KEEP_S), end_word + keep_tail)
                        core = vt.fade_out(core[:int(tail_s * rate)], rate, FINAL_FADE_S if last_span else 0.012)
                        tail_pad, kept_tail = 0, round(tail_s - end_word, 3)         # the join is measured from the word, not from the generation's end: see build()
                        entry.update(tail_by="words", tail_word_end_s=round(end_word, 2), tail_next_start_s=round(start_next, 2), tail_cut_s=round(tail_s, 2), tail_found=tail_found, kept_tail_s=kept_tail)
                if use_words and not leads[index]:                                         # a span that begins cold: sound before its first word is not in the text
                    own = [w for w in answer["words"][index][:len(spans[index].split())] if w]
                    if own:
                        lead_in = vt.stray_before(core, rate, float(own[0]["start"]) - model_pad_s)
                        if lead_in:
                            core, head_cut_s = vt.fade_in(core[int(lead_in * rate):], rate), lead_in
                            entry.update(stray_head_s=lead_in, kept_head_s=round(float(own[0]["start"]) - model_pad_s - lead_in, 3))
                if use_words and not follows[index]:                                       # and one that nothing follows: sound after its last word is not either
                    own = [w for w in answer["words"][index][len(leads[index].split()) if leads[index] else 0:][:len(spans[index].split())] if w]
                    if own:
                        after = vt.stray_after(core, rate, float(own[-1]["end"]) - model_pad_s - head_cut_s)
                        if after:
                            core = vt.fade_out(core[:int(after * rate)], rate)
                            entry.update(stray_tail_s=after)
                if leads[index]:
                    n_lead = len(leads[index].split())
                    if use_words:
                        word_times = answer["words"][index]
                        shift = lead_pad / rate                                # word times run from the start of the span's audio, the core starts after the model's own padding
                        last = next((w for w in reversed(word_times[:n_lead]) if w), None)
                        first = next((w for w in word_times[n_lead:] if w), None)
                        end_lead, start_new = float(last["end"]) - shift, float(first["start"]) - shift
                        low, high = min(end_lead, start_new) - 0.03, max(end_lead, start_new) + 0.03
                        cut_s, found = vt.find_cut(core, rate, (low + high) / 2, before_s=(high - low) / 2, after_s=(high - low) / 2)
                        cut_s = vt.head_cut(cut_s, end_lead, start_new, vt.edge_keep_s(target[index - 1] if index > 0 else 0.0, share=0.4, most_s=vt.HEAD_KEEP_S, least_s=vt.HEAD_MIN_KEEP_S))         # a fixed small lead-in before the first new word, so the silence at the join can be set exactly
                        entry.update(lead_chars=len(leads[index]), by="words", lead_end_s=round(end_lead, 2), new_start_s=round(start_new, 2), cut_s=round(cut_s, 2), found=found, kept_head_s=round(start_new - cut_s, 3))
                        found = True                                           # the words say where the boundary is; a cut with no quiet stretch in it is still at the boundary
                    else:
                        standalone = next(alone, None)
                        alone_core = vt.trim_zero_pads(standalone)[0] if standalone is not None else np.zeros(0, dtype=np.float32)
                        start_s, end_s = vt.active_extent(alone_core, rate) if len(alone_core) else (0.0, 0.0)
                        on_s, _ = vt.active_extent(core, rate)
                        aligned = vt.align_end(core, alone_core, rate) if len(alone_core) else None          # where the repeated sentence really ends in this generation
                        if aligned is not None:
                            cut_s, found = vt.find_cut(core, rate, aligned, before_s=vt.CUT_WINDOW_ALIGNED_S, after_s=vt.CUT_WINDOW_ALIGNED_S)
                        else:
                            cut_s, found = vt.find_cut(core, rate, on_s + (end_s - start_s))
                        entry.update(lead_chars=len(leads[index]), by="warp", lead_s=round(end_s - start_s, 2), aligned_s=aligned, cut_s=round(cut_s, 2), found=found)
                    core, lead_pad = vt.fade_in(core[int(cut_s * rate):], rate), 0
                    head_cut_s = cut_s
                    extra = 0.0 if found else 1.5                                   # the end of the repeated sentence was not clear: this take is worth less
                if len(entry) > 1:
                    info.append(entry)
                core, head_cut, tail_cut = vt.strip_stray_ends(core, rate)          # the model's end-of-generation click (and the zeros before it) is not speech: drop it, keep the timeline
                if head_cut:
                    core = vt.fade_in(core, rate)
                lead_pad, tail_pad = lead_pad + head_cut, tail_pad + tail_cut
                marks = sentence_marks(spans[index], answer["words"][index] if use_words else None, len(leads[index].split()) if leads[index] else 0, model_pad_s + head_cut_s + head_cut / rate)
                word_list = word_marks(spans[index], answer["words"][index] if use_words else None, len(leads[index].split()) if leads[index] else 0, model_pad_s + head_cut_s + head_cut / rate)
                cores.append((core, lead_pad, tail_pad))
                scores.append({**vt.score_span(core, rate, pause_ok), "uncertain_cut": bool(extra), "stray_removed_s": round((head_cut + tail_cut) / rate, 3), "sentence_marks": marks, "word_marks": word_list,
                               "kept_head_s": entry.get("kept_head_s") if not head_cut else None, "kept_tail_s": entry.get("kept_tail_s") if not tail_cut else None,
                               "natural_gap_s": None if natural[index] is None else round(natural[index], 3), "target_after_s": round(target[index], 3) if index < len(target) else None})
                scores[-1]["score"] = round(scores[-1]["score"] + extra, 3)
            for index, scored in enumerate(scores):                          # how long the voice said the names of each span in this take (the primed sentences the names come from)
                scored["names"] = [n for note in raw_notes if note["span"] == index and note["take"] == take_no for n in note.get("names") or []]
                scored["name_s"] = round(sum(float(n["s"]) for n in scored["names"]), 3)
                scored["clarity"] = round(sum(float(n["clarity"]) for n in scored["names"]) / len(scored["names"]), 3) if scored["names"] and all(n.get("clarity") is not None for n in scored["names"]) else None
            notes.append(info)
            return cores, scores, rate, float(answer.get("gpu_s") or 0.0)

        def build(takes: list[tuple[list[tuple[np.ndarray, int, int]], list[dict[str, Any]], int, float]]) -> tuple[np.ndarray, int, list[float], list[int], list[dict[str, Any]], list[tuple[np.ndarray, int, int]]]:
            rate = takes[0][2]
            picks = [pinned[i] if pinned[i] is not None and pinned[i] < len(takes) else pick_take([takes[t][1][i] for t in range(len(takes))]) for i in range(len(spans))]          # a take the listener committed is the one kept
            cores = [takes[picks[i]][0][i] for i in range(len(spans))]
            chosen = [takes[picks[i]][1][i] for i in range(len(spans))]
            effective, joins_made = list(gaps), []
            for i in range(len(spans) - 1):                                      # a join between two spans cut at their words: exactly the planned silence between the words, the cuts' own quiet counted
                tail_kept, head_kept = chosen[i].get("kept_tail_s"), chosen[i + 1].get("kept_head_s")
                if tail_kept is not None and head_kept is not None:
                    wanted = chosen[i].get("target_after_s")
                    wanted = vt.join_target_s(gaps[i], chosen[i].get("natural_gap_s")) if wanted is None else wanted
                    fill = vt.join_fill_s(wanted, tail_kept, head_kept)
                    effective[i] = int(round((fill - (cores[i][2] + cores[i + 1][1]) / rate) * 1000))
                    joins_made.append({"after_span": i, "planned_silence_s": round(max(0.0, vt.PAD_S + gaps[i] / 1000.0), 3), "natural_gap_s": chosen[i].get("natural_gap_s"), "target_silence_s": round(wanted, 3),
                                       "kept_tail_s": tail_kept, "kept_head_s": head_kept, "fill_s": round(fill, 3), "silence_s": round(tail_kept + head_kept + fill, 3)})
                elif tail_kept is not None or head_kept is not None:                  # only one side is cut at its word: the padding the other side keeps stands for its half of the join
                    effective[i] = int(round(gaps[i] + 100 - (tail_kept if tail_kept is not None else head_kept) * 1000))
            join_notes[:] = joins_made
            audio, starts = vt.assemble(cores, effective, rate, tone, delta)
            return audio, rate, starts, picks, chosen, cores

        reference = bank.as_reference(ranked[0])
        pinned = [dr.take_for(marked, text, *clip_edges) for text in spans]
        takes = []
        wanted_takes = 1 + max([p for p in pinned if p is not None] or [0])          # a run makes one (and the later take the listener committed, if any)
        if all_takes:                                                    # the Studio: three draws where a span carries names (they differ in how the names are primed), one where it does not
            wanted_takes = max(wanted_takes, vt.MAX_TAKES if any(primers) else self.nameless_draws)
        if self.name_draws and any(primers):
            wanted_takes = max(wanted_takes, vt.MAX_TAKES)          # draws are made by hand: the Studio asks for all of them, a run makes one (and the later take the listener committed, if any)
        for _ in range(wanted_takes):
            takes.append(take(reference))
        audio, rate, starts, picks, kept, cores = build(takes)
        if raw_out is not None:
            for i in range(len(spans)):
                flags = stretched_draws([takes[t][1][i] for t in range(len(takes))])
                for note in raw_notes:
                    if note["span"] == i and note["take"] < len(flags):
                        note["stretched"] = flags[note["take"]]
            raw_out[:] = raw_notes
        if takes_out is not None:
            takes_out[:] = [[takes[t][0][i][0] for t in (range(len(takes)) if primers[i] is not None or pinned[i] is not None or self.nameless_draws > 1 else range(1))] for i in range(len(spans))]          # every take of every span, as kept (the studio plays them as the draws)
        report: dict[str, Any] = {"takes": len(takes), "picked": picks, "spans": kept, "reference_s": [round(reference.start_s, 1), round(reference.end_s, 1)],
                                  "room_tone": {"below_speech_db": round(delta, 1)},
                                  "overlap": [next((n for n in notes[picks[i]] if n["span"] == i), None) for i in range(1, len(spans))] if notes else []}
        report["tails"] = [next((n for n in notes[picks[i]] if n["span"] == i and "tail_by" in n), None) for i in range(len(spans) - 1)] if notes else []          # where each span's last word was cut off from the next span's first word
        report["silences"] = list(join_notes)
        report["primed"] = list(pin_notes)                                              # per span with names and take: its length at the fluent clip's pace, with the names in the reference, whether it was stretched to the first, and the length it ended with
        report["primers"] = [None if p is None else {"names": p["names"], "reps": p["reps"], "texts": [t for t, _, _ in p["snippets"]], "seconds": round(p["seconds"], 2)} for p in primers]
        report["joins"] = [{"span": i, "start_s": starts[i], "dropouts_per_take": [t[1][i].get("dropouts", 0) for t in takes], "picked_dropouts": kept[i].get("dropouts", 0),
                            "peak_first_40ms": round(float(np.abs(audio[int(starts[i] * rate): int(starts[i] * rate) + int(0.04 * rate)]).max(initial=0.0)), 3)} for i in range(len(spans))]
        window_start = int(group.get("start_ms") or 0) / 1000.0
        window_end = int(group.get("source_end_ms") or (int(group.get("start_ms") or 0) + int(group.get("source_span_ms") or 0))) / 1000.0
        dub_sentences = []
        for i, chosen_span in enumerate(kept):
            base = starts[i] + cores[i][1] / rate                                          # where the kept audio of this span begins in the unit
            for mark in chosen_span.get("sentence_marks") or []:
                dub_sentences.append({"text": mark["text"], "chars": mark["chars"], "start": round(base + mark["start"], 2), "end": round(base + mark["end"], 2)})
        try:
            theirs = source_sentences(bank.words, bank.speaker_of_raw, str(group.get("speaker_id") or ""), window_start, window_end)
        except Exception:                      # noqa: BLE001 - the source's sentences are context for the agent, never a reason a unit cannot be spoken
            theirs = []
        matched = match_source(dub_sentences, theirs)
        report["sentences"] = [{"n": n + 1, "text": x["text"], "chars": x["chars"], "start_s": x["start"], "end_s": x["end"], "dub_s": round(x["end"] - x["start"], 2),
                                "source_start_s": None if matched[n] is None else theirs[matched[n]]["start"], "source_s": None if matched[n] is None else round(theirs[matched[n]]["end"] - theirs[matched[n]]["start"], 2)}
                               for n, x in enumerate(dub_sentences)]
        report["words"] = [{"span": i, "text": mark["text"], "start_s": round(starts[i] + cores[i][1] / rate + mark["start"], 2), "end_s": round(starts[i] + cores[i][1] / rate + mark["end"], 2)}
                           for i, chosen_span in enumerate(kept) for mark in chosen_span.get("word_marks") or []]          # every word of the unit on its timeline, for the studio
        for chosen_span in kept:
            chosen_span.pop("sentence_marks", None)
            chosen_span.pop("word_marks", None)
        lengths = [int((len(core) + lead + tail) / rate * 1000) for core, lead, tail in cores]
        answer = {"span_starts_s": starts, "span_ms": lengths, "gpu_s": round(sum(t[3] for t in takes), 2)}
        return audio, rate, answer, report, reference

    def estimate_seconds(self, group: Mapping[str, Any], voice_info: Mapping[str, Any], job_root: Path | str | None) -> float:
        """The free estimate of this group's natural length (see omnivoice_duration), from its reference clip's pace: no synthesis, no GPU."""
        spans, gaps = self.plan(group, voice_info, job_root)
        reference = self.reference_for(group, job_root)
        return od.estimate_seconds(spans, gaps, reference.text, reference.seconds, self.speed)          # a span with names runs about 6% longer than this (the names slow the voice); the measurement has the real length

    def synthesize_group(self, *, group: Mapping[str, Any], voice_info: Mapping[str, Any], output_path: Path, preferred_ms: int, maximum_ms: int,
                         force: bool, job_root: Path | str | None = None) -> Any:
        from .azure_tts import AzureTTSResult

        output_path = Path(output_path)
        spans, gaps = self.plan(group, voice_info, job_root)
        if not spans:
            raise ValueError("nothing to say: the group has no text")
        reference = self.reference_for(group, job_root)
        from . import draws as dr

        marked = dr.load(job_root)
        request_hash = hashlib.sha256(json.dumps({"spans": spans, "gaps": gaps, "ref": [reference.start_s, reference.end_s], "speed": self.speed, "lora": self.lora,
                                                  "natural": 10, "generation": {**getattr(self.client, "default_generation", {}), **(self.generation or {})}, "retakes": [dr.retake_for(marked, sp, reference.start_s, reference.end_s) for sp in spans], "primers": [None if p is None else [p["names"], p["reps"]] for p in self.span_primers(group, spans, job_root)]},
                                                 ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        manifest = output_path.with_suffix(".omnivoice.json")
        reuse = False
        if output_path.is_file() and manifest.is_file() and not force:
            try:
                if json.loads(manifest.read_text(encoding="utf-8")).get("request_hash") == request_hash:
                    reuse = True
            except ValueError:
                reuse = False
        if not reuse:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            audio, rate, answer, natural, reference = self.speak(group, spans, gaps, job_root)
            output_path.write_bytes(wav_bytes(audio, rate))
            starts = answer.get("span_starts_s")
            manifest.write_text(json.dumps({"request_hash": request_hash, "spans": spans, "gaps_ms": gaps, "reference": {"start_s": reference.start_s, "end_s": reference.end_s, "text": reference.text},
                                            "span_starts_s": starts, "span_ms": answer.get("span_ms"), "gpu_s": answer.get("gpu_s"),
                                            "natural": natural}, ensure_ascii=False, indent=1), encoding="utf-8")
        info = sf.info(str(output_path))
        data = output_path.read_bytes()
        return AzureTTSResult(
            backend=self.provider, turn_id=str(group.get("group_id") or ""), speaker_id=str(group.get("speaker_id") or ""), voice=str(voice_info.get("selected_voice") or self.default_voice),
            language="zu", input_text_hash=hashlib.sha256(" ".join(spans).encode()).hexdigest(), ssml_hash="", output_path=str(output_path), sample_rate=int(info.samplerate),
            channels=int(info.channels), sample_width=2, duration_ms=int(round(info.frames / info.samplerate * 1000)), preferred_duration_ms=int(preferred_ms),
            maximum_duration_ms=int(maximum_ms), rate_percent=0, attempt=1, sha256=hashlib.sha256(data).hexdigest(), warnings=(), request_hash=request_hash, ssml_path="",
            manifest_path=str(manifest), ssml_summary=f"omnivoice {len(spans)} span(s), reference {reference.start_s:.1f}-{reference.end_s:.1f}s", idempotent_reuse=reuse)
