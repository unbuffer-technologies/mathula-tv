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
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import soundfile as sf

from . import omnivoice_duration as od

CLIP_MIN_S = 3.0
CLIP_MAX_S = 9.0
NAME_SIMILARITY = 0.8        # how alike (spelt as the voice would say them) a name of the dub and the words the speaker said must be: "Canny Maphanga" and the recogniser's "Kanima Panga" are 0.87, "Ayanda Nyathi" and "Ayanda Nyati" 0.96, unrelated words under 0.3
NAME_SIMILARITY_ONE_WORD = 0.85
PRIMER_REPS = 3              # a name the source says is put in front of the reference this many times (heard on "Phineas": once 0 of 3 right, three times 2 of 3, and 3 of 3 with the names first)
PRIMER_MAX_S = 9.0           # the names, all repetitions together, never take more of the reference than this (the model asks for 3-10 s in all; the clip it is put in front of is about 8.5 s)
PRIMER_MAX_SNIPPET_S = 4.0   # a run of names longer than this was cut wrongly
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


class ReferenceBank:
    """English reference clips for one job, cut from its own audio: sentence-bounded runs of one speaker's words, 3-9 s long, with their transcript."""

    def __init__(self, words: Sequence[Mapping[str, Any]], speaker_of_raw: Mapping[str, str], audio_path: Path, cache_dir: Path | None = None) -> None:
        self.audio_path = Path(audio_path)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.words = list(words)                       # the source's timed words: where its speakers paused (span_cuts)
        self.speaker_of_raw = {str(k): str(v) for k, v in speaker_of_raw.items()}
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
    def dub_name_runs(text: str) -> list[list[str]]:
        """The runs of name-like words in the dub text (a capital, three letters or more), with a Zulu prefix taken off ("uCanny" and "u-Phineas" -> Canny, Phineas; "we-JMPD" -> JMPD).  A comma or a full stop ends a run.
        Zulu words that merely begin a sentence are in here too: they are only names when the speaker said something like them (name_matches)."""
        runs: list[list[str]] = []
        current: list[str] = []
        for raw in str(text or "").split():
            core = re.sub(r"^[^\w]+|[^\w]+$", "", raw)
            core = core.split("-")[-1]
            core = re.sub(r"^(?:[a-z]+|[A-Z])(?=[A-Z][a-z])", "", core)          # uCanny, kwamaHawks (a prefix of any length), uClive (a capital U at the start of a sentence), but not JMPD
            ends = raw.rstrip().endswith((",", ".", "?", "!", ";", ":"))
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
                    best, best_at = 0.0, None
                    for n in (length,):                                              # as many words as the name has: a word of the text that the audio does not say, or the other way round, would teach the model the wrong thing
                        for j in range(0, len(words) - n + 1):
                            window = words[j: j + n]
                            if any(window[k + 1][0] != window[k][0] + 1 for k in range(n - 1)):
                                continue
                            similarity = SequenceMatcher(None, key, self._sound_key(" ".join(self._said(w) for _, w in window))).ratio()
                            if similarity > best:
                                best, best_at = similarity, window
                    needed = NAME_SIMILARITY if length > 1 else NAME_SIMILARITY_ONE_WORD
                    if best_at is None or best < needed or (length == 1 and len(key) < 4 and not tokens[0].isupper()):
                        continue
                    for k in range(at, at + length):
                        used[k] = True
                    found.append({"tokens": tokens, "text": " ".join(tokens), "first": best_at[0][0], "last": best_at[-1][0],
                                  "source_text": " ".join(str(w["text"]) for _, w in best_at), "restored_text": " ".join(self._said(w) for _, w in best_at), "similarity": round(best, 2)})
        return sorted(found, key=lambda m: -len(m["tokens"]))

    def primer_for(self, speaker_id: str, start_s: float, end_s: float, span_text: str) -> dict[str, Any] | None:
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
            first, last = self.snap(float(self.words[a]["start"]), float(self.words[z]["end"]), min(prev_end, float(self.words[a]["start"])), max(next_start, float(self.words[z]["end"])))
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
        said = {re.sub(r"^\W+|\W+$", "", w).lower() for text, _, _ in snippets for w in text.split()}
        return {"names": sorted(said), "snippets": snippets, "reps": reps, "seconds": total() * reps}

    def primed(self, base: Reference, primer: Mapping[str, Any] | None) -> Reference:
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
        for _ in range(int(primer["reps"])):
            for text, wav, _seconds in primer["snippets"]:
                piece, piece_rate = sf.read(io.BytesIO(wav), dtype="float32")
                if piece_rate != rate:
                    return base
                pieces.append(piece.mean(axis=1) if piece.ndim > 1 else piece)
                texts.append(text)
        clip_pace = len(base.text) / max(base.seconds, 0.1)
        lacking = sum(len(t) + 1 for t in texts) / clip_pace - sum(len(x) for x in pieces) / rate
        gap = np.zeros(int(min(PRIMER_GAP_MAX_S, max(PRIMER_GAP_S, lacking / max(len(pieces), 1))) * rate), dtype=np.float32)
        audio = np.concatenate([part for piece in pieces for part in (piece, gap)] + [head])
        return Reference(wav=wav_bytes(audio, rate), text=" ".join(texts + [base.text]), seconds=len(audio) / rate, start_s=base.start_s, end_s=base.end_s, speaker=base.speaker)

    def ranked(self, speaker_id: str, start_ms: int, end_ms: int) -> list[dict[str, Any]]:
        """The clips to clone for a unit, best first: those inside its window, then the closest in time (preferring the same speaker) for a window too short to hold one."""
        start, end = start_ms / 1000.0, end_ms / 1000.0
        if not self._candidates:
            raise LookupError("the job has no clean 3-9 s clip of any speaker to clone")
        inside = sorted((c for c in self._candidates if c["start"] >= start - 0.05 and c["end"] <= end + 0.05), key=self._score, reverse=True)
        middle = (start + end) / 2.0
        others = sorted((c for c in self._candidates if c not in inside),
                        key=lambda c: (abs((c["start"] + c["end"]) / 2.0 - middle) + (0.0 if c["speaker"] == speaker_id else 25.0), -self._score(c)))
        return inside + others

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

    def snap(self, start: float, end: float, prev_end: float, next_start: float) -> tuple[float, float]:
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

        new_start = quietest(max(prev_end, start - 0.15), start + 0.08, start)
        new_end = quietest(end - 0.08, min(next_start, end + 0.15), end)
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


def name_lengths(words: Sequence[str], times: Sequence[Mapping[str, Any] | None], names: Sequence[str] | set[str]) -> list[dict[str, Any]]:
    """How long the voice said each name of ``words`` (``times`` are the aligned times of the same words): [{text, s}]."""
    out = []
    for first, last in name_runs(words, names):
        a, z = (times[first] if first < len(times) else None), (times[last] if last < len(times) else None)
        if a and z:
            out.append({"text": " ".join(words[first:last + 1]), "s": round(float(z["end"]) - float(a["start"]), 3)})
    return out


def pick_take(scores: Sequence[Mapping[str, Any]]) -> int:
    """Which draw of a span is used: where the span carries names the speaker says, the draw in which the voice said them longest (the audio score, voice_takes.score_span, breaking a tie); otherwise the draw
    with the least wrong in the audio.  The way a name goes wrong is that it is swallowed, said shorter than the speaker says it, and of the three draws of "i-NEC" and of "yi-ANC" the longest name was the
    one the ear chose both times (0.401 s of 0.38/0.30/0.40, and 0.641 s of 0.42/0.64/0.54)."""
    if not any(float(sc.get("name_s") or 0.0) > 0.0 for sc in scores):
        return min(range(len(scores)), key=lambda t: scores[t]["score"])
    return max(range(len(scores)), key=lambda t: (float(scores[t].get("name_s") or 0.0), -float(scores[t]["score"])))


def joined_with_the_word_before(words: Sequence[str], first: int) -> int:
    """Where a name's swap begins: the name itself, or the word before it when the name carries a Zulu prefix ("i-NEC", "yi-ANC") and that word ends in a vowel ("futhi i-NEC"): the two vowels run
    into each other, the aligner puts the boundary somewhere inside them, and a cut there loses the prefix, so the pair is taken from one generation."""
    if first > 0 and "-" in str(words[first]).strip(".,;:!?") and re.search(r"[aeiou]\W*$", str(words[first - 1]).lower()):
        return first - 1
    return first


def _crossfade(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    n = min(n, len(a), len(b))
    if n < 2:
        return np.concatenate([a, b])
    ramp = np.linspace(0.0, 1.0, n, dtype=a.dtype)
    return np.concatenate([a[:-n], a[-n:] * (1.0 - ramp) + b[:n] * ramp, b[n:]])


def swap_word_run(part: np.ndarray, rate: int, times: Sequence[Mapping[str, Any] | None], first: int, last: int,
                  snippet: np.ndarray, snippet_times: Sequence[Mapping[str, Any] | None], s_first: int, s_last: int,
                  cut_at: Callable[[np.ndarray, float, float], float], keep_s: float | None = None) -> tuple[np.ndarray, list[Mapping[str, Any] | None]] | None:
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
    middle = snippet[j_a:j_z]
    if len(middle) < 2 * n or i_a < n or len(part) - i_z < n:
        return None
    joined = _crossfade(_crossfade(part[:i_a], middle, n), part[i_z:], n)
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
        return self._banks[key]

    def span_primers(self, group: Mapping[str, Any], spans: Sequence[str], job_root: Path | str | None) -> list[dict[str, Any] | None]:
        """For each span, the speaker's own names it carries (ReferenceBank.primer_for), or None."""
        try:
            bank = self.bank_for(job_root)
            start = int(group.get("start_ms") or 0)
            end = int(group.get("source_end_ms") or (start + int(group.get("source_span_ms") or 0)))
            return [bank.primer_for(str(group.get("speaker_id") or ""), start / 1000.0, end / 1000.0, span) for span in spans]
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
            matches = self.bank_for(job_root).name_matches(str(group.get("speaker_id") or ""), start / 1000.0, end / 1000.0, _group_text(group))
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
        tone, delta = bank.room_tone()
        pause_ok = bank.natural_pause_s(str(group.get("speaker_id") or ""))          # the longest pause the speaker themself makes inside a sentence: shorter ones are not held against a take
        breath = bank.natural_pause_s(str(group.get("speaker_id") or ""), 75.0) or 0.0          # what the speaker leaves at least after a sentence: a full stop is a breath (prod-129 0:39: 'ethile.' and 'Umane' 0.06 s apart)
        primers = self.span_primers(group, spans, job_root)
        from . import draws as dr

        marked = dr.load(job_root)                                      # the draws the listener approved (draws.py); ``retakes`` tries others without keeping them

        leads = [None] + [vt.lead_sentence(spans[i - 1]) for i in range(1, len(spans))]
        follows = [_first_word(spans[i + 1]) if i + 1 < len(spans) and self.follow_word else None for i in range(len(spans))]          # the first word of the span after: spoken after this one so its last word is not the last thing the model makes
        notes: list[list[dict[str, Any]]] = []
        take_count = [0]                                                 # which take this is: every take is a draw of its own, and the same one every time
        join_notes: list[dict[str, Any]] = []
        pin_notes: list[dict[str, Any]] = []
        first_pass: dict[Any, Any] = {}                                    # the unprimed pass of the cluster, made by the first draw and shared by the others
        raw_notes: list[dict[str, Any]] = []                              # the primed sentences the names were cut from, exactly as the voice made them (the studio plays them)

        def take(reference: Reference) -> tuple[list[tuple[np.ndarray, int, int]], list[dict[str, Any]], int, float]:
            """One synthesis of the unit.  A span that follows another is spoken with the last sentence of the one before in front of it, so the voice does not start cold, and after the generation the
            repeated sentence is cut off between its last word and the first new one.  Where that is: the service aligns the known text with its own audio (one request carries every span of the unit and the word
            times of each); a service that cannot is answered by speaking each lead alone as well and warping it onto the start of the generation (voice_takes.align_end)."""
            use_words = getattr(self, "service_aligns", None) is not False and any(leads)

            refs = [bank.primed(reference, primer) for primer in primers]
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
                    n = dr.retake_for(marked, spans[index], reference.start_s, reference.end_s)
                return stable_seed(text, take_no, *parts, *((("retake", n),) if n else ()))

            def primed_pass(sent: list[str], pool: Any) -> dict[str, Any]:
                """Pronunciation from the names, pace from the speaker.  Every span is spoken once with the fluent clip (the audio of the spans that carry no names, and the length the speaker's own pace gives the
                rest); each span that carries names is then spoken again with the names-then-clip reference, and replaces that span.  The two lengths are the model's own readings of the same sentence; the
                slower one wins.  A primed span that comes out SHORTER (the names, said fast, made the voice read the reference as a faster speaker: 7.0 s against 8.6 s, "ngemisebenzi" -> "nomsebenzi") is spoken
                once more pinned to the fluent length, which only stretches it.  One that comes out LONGER is left as it is: pinning it down squeezed the short syllable in front of a name ("ngu-Eric" heard as
                "Eric": 0 of 5 clean pinned, 5 of 5 clean unpinned, prod-125 1:01)."""
                reused = (salt, tuple(sent)) in first_pass          # the cluster is spoken once: the draws of a span with names differ in how its names are primed, not in the speech around them
                first_future = None if reused else pool.submit(self.client.synthesize, spans=sent, gaps_ms=[0] * (len(sent) - 1), reference=reference, speed=self.speed, lora=self.lora, generation=self.generation, align=True,
                                                               seeds=[seed_for(i, text, "first", salt) for i, text in enumerate(sent)])
                futures: dict[Any, Any] = {}

                def primed_kwargs(index: int, text: str, tag: str) -> dict[str, Any]:
                    return dict(spans=[text], gaps_ms=[], reference=refs[index], speed=self.speed, lora=self.lora, generation=self.generation, align=True, seeds=[seed_for(index, text, tag, salt, primers[index]["names"])])

                def fetch(key: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
                    """The reply of a primed generation: the one already on its way (submitted with the first pass, so they are spoken side by side), or made now."""
                    future = futures.get(key)
                    return future.result() if future is not None else self.client.synthesize(**kwargs)

                def sentence_plan(index: int, j: int, sentences_in: list[str]) -> tuple[str | None, str | None, str, int]:
                    """For the j-th sentence of a cluster: what is spoken before it, the word after it, the text of the generation and how many words of it come first (the way the pipeline speaks any sentence)."""
                    before = vt.lead_sentence(sentences_in[j - 1] if j > 0 else (leads[index] or ""))
                    after = _first_word(sentences_in[j + 1]) if j + 1 < len(sentences_in) else follows[index]
                    return before, after, " ".join(x for x in (before, sentences_in[j], after) if x), len(before.split()) if before else 0

                for index, primer in enumerate(primers):                          # nothing a primed generation says depends on the first pass: they are asked for now, and used when the first pass has come back
                    if primer is None:
                        continue
                    sentences_in = [x for x in vt.SENTENCE_END.split(spans[index].strip()) if x.strip()]
                    if self.span_sentences > 1 and len(sentences_in) > 1 and sum(len(x.split()) for x in sentences_in) == len(spans[index].split()):
                        for j in reversed(range(len(sentences_in))):
                            if name_runs(sentences_in[j].split(), primer["names"]):
                                text_in = sentence_plan(index, j, sentences_in)[2]
                                futures[("s", index, text_in)] = pool.submit(self.client.synthesize, **primed_kwargs(index, text_in, "primed-sentence"))
                    else:
                        futures[("b", index)] = pool.submit(self.client.synthesize, **primed_kwargs(index, sent[index], "primed"))
                first = first_pass[(salt, tuple(sent))] if reused else first_future.result()
                if not isinstance(first.get("words"), list):
                    return first                                                       # the service does not align: the caller asks the old way, with one reference
                first_pass[(salt, tuple(sent))] = first
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
                    if self.span_sentences > 1 and len(sentences_in) > 1 and isinstance(now_times, list) and len(now_times) == len(sent[index].split()) and sum(len(x.split()) for x in sentences_in) == len(spans[index].split()):
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
                            for run_first, run_last in reversed(name_runs(words_s, primer["names"])):          # only the name is taken from the primed sentence; the cluster keeps everything else
                                run_first = joined_with_the_word_before(words_s, run_first)
                                out = swap_word_run(audio_now, rate_seen, times_now, first_w + run_first, first_w + run_last, primed_audio, reply["words"][0], n_before + run_first, n_before + run_last, cut_at)
                                if out is not None:
                                    (audio_now, times_now), swapped = out, swapped + 1
                            redone += swapped
                            pin_notes.append({"span": index, "sentence": text_s[:40], "lead": bool(before), "follow": bool(after), "names_swapped": swapped})
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
                        again = pool.submit(self.client.synthesize, spans=[sent[index]], gaps_ms=[], reference=refs[index], speed=self.speed, lora=self.lora, generation=self.generation, align=True, durations_s=[round(natural, 3)],
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
                reply = self.client.synthesize(spans=sent, gaps_ms=[max(0, g) for g in gaps] + [0] * len(heard), reference=reference, speed=self.speed, lora=self.lora, generation=self.generation, **({"align": True} if with_words else {}),
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
                        keep_tail = vt.edge_keep_s(target[index], share=0.6, most_s=vt.TAIL_KEEP_S, least_s=vt.TAIL_MIN_KEEP_S)          # a short join keeps little of the quiet
                        tail_s = min(max(tail_s, end_word + vt.TAIL_MIN_KEEP_S), end_word + keep_tail)
                        core = vt.fade_out(core[:int(tail_s * rate)], rate)
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
                scored["name_s"] = round(sum(float(n["s"]) for note in raw_notes if note["span"] == index and note["take"] == take_no for n in note.get("names") or []), 3)
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
        pinned = [dr.take_for(marked, text, reference.start_s, reference.end_s) for text in spans]
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
