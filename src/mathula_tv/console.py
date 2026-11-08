"""The pronunciation console: the listener's place to decide how English words are said in the Zulu dubs.

    python -m mathula_tv.console set-password      # once, on the server: asks for the password and stores only its hash
    python -m mathula_tv.console serve             # 127.0.0.1:8765, behind nginx at https://ai.bakgethwa.com

The workflow is the listener's: the dictionary's sounds go through the rules and make a spelling; the console plays it in its real sentence in each voice; the listener says
correct (the spelling is committed to the pronunciation book and every job uses it) or not correct (alternatives from the same rules are played, or the listener types
a spelling of their own, and that is committed instead).  There are no exceptions: a spelling that differs from the rules' is kept as a lesson for the rules (what the
dictionary heard, how the rules built theirs, what was written instead), the Rules page shows the patterns across words, and the model can be asked why.
Nothing here scores anything.

The source speakers are here too: ``source_words`` finds where a word is said in a job's English transcript (word timings) and ``source_audio`` cuts that word, or the
phrase around it, out of the job's own source audio, so the listener hears how the speaker said it before deciding how the Zulu voice should.

A recogniser appears on this page and nowhere else: ``source_hear`` shows the phones a phoneme recogniser hears in a source clip and the word a letter recogniser reads
there, only for the listener's ear to compare.  Nothing in the pipeline or the agent uses it, and nothing here chooses or scores with it.

``Console.handle`` takes a request and returns a response, so the whole thing is tested without a socket; ``serve`` only carries bytes to it.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import queue
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import parse_qs, urlsplit

from . import ear_clips, language_questions as language, pronunciation_book as book, zulu_rules as rules
from .job_pronunciations import VOICES, Prepared, spell, with_committed_words

SESSION_SECONDS = 12 * 3600
LOGIN_FAILURES = 5
LOGIN_WINDOW_SECONDS = 600
MAX_BODY = 64 * 1024
MAX_AUDIO_PER_DAY = 800                 # new syntheses a day (a clip made ahead counts too): the console can spend Azure credit, so it has a ceiling
JOB_ID = re.compile(r"^[0-9a-f]{32}$")
RUN_ID = re.compile(r"^[A-Za-z0-9._-]{1,40}$")
AUDIO_NAME = re.compile(r"^[0-9a-f]{32}\.wav$")
SEGMENT_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,60}$")
PBKDF = {"n": 2 ** 14, "r": 8, "p": 1}


class ConsoleError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status, self.message = status, message


def _clean_word(value: Any) -> str:
    word = str(value or "").strip()
    if not word or len(word) > 80 or re.search(r"[\x00-\x1f<>\"`]", word):
        raise ConsoleError(400, "a word is required")
    return word


class Console:
    def __init__(self, work_dir: Path, *, synthesize: Callable[..., Path] | None = None, clock: Callable[[], float] = time.time, max_audio_per_day: int = MAX_AUDIO_PER_DAY,
                 analyse: Callable[[dict[str, Any]], dict[str, Any]] | None = None, cut: Callable[[Path, float, float, Path], None] | None = None,
                 recognise: Callable[[Path], dict[str, str]] | None = None) -> None:
        self.work_dir = Path(work_dir)
        self.dir = self.work_dir / "console"
        self.audio_dir = self.dir / "audio"
        self.book_path = self.work_dir / "pronunciation_book.json"
        self.clock = clock
        self.max_audio_per_day = max_audio_per_day
        self._synthesize = synthesize
        self._lock = threading.Lock()
        self._failures: list[float] = []
        self._prepared: dict[str, Prepared] = {}
        self._tts: Any = None
        self._analyse = analyse
        self._cut = cut or self._ffmpeg_cut
        self._recognise = recognise or self._recognise_local
        self._recognisers: tuple[Any, Any] | None = None
        self._unload_timer: threading.Timer | None = None
        self._recognise_lock = threading.Lock()
        self._sources: dict[str, tuple[float, list[dict[str, Any]], Path, str]] = {}
        self._ahead: queue.Queue[tuple[str, str, str, str, str]] = queue.Queue()          # clips to make before they are asked for
        self._queued: set[str] = set()
        self._worker: threading.Thread | None = None
        self.dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self._secret = self._load_secret()

    # -- login -------------------------------------------------------------------------------------------------------------------------------
    def _load_secret(self) -> bytes:
        path = self.dir / "secret.key"
        if not path.is_file():
            path.write_bytes(secrets.token_bytes(32))
            try:
                path.chmod(0o600)
            except OSError:
                pass
        return path.read_bytes()

    def set_password(self, password: str) -> None:
        if len(password) < 10:
            raise ValueError("the password needs at least 10 characters")
        salt = secrets.token_bytes(16)
        digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, **PBKDF, dklen=32)
        path = self.dir / "auth.json"
        path.write_text(json.dumps({"salt": salt.hex(), "hash": digest.hex(), **PBKDF}), encoding="utf-8")
        try:
            path.chmod(0o600)
        except OSError:
            pass

    def _password_ok(self, password: str) -> bool:
        try:
            stored = json.loads((self.dir / "auth.json").read_text(encoding="utf-8"))
            digest = hashlib.scrypt(password.encode("utf-8"), salt=bytes.fromhex(stored["salt"]), n=int(stored["n"]), r=int(stored["r"]), p=int(stored["p"]), dklen=32)
            return hmac.compare_digest(digest.hex(), stored["hash"])
        except (OSError, ValueError, KeyError):
            return False

    def login(self, password: str) -> str:
        now = self.clock()
        self._failures = [t for t in self._failures if now - t < LOGIN_WINDOW_SECONDS]
        if len(self._failures) >= LOGIN_FAILURES:
            raise ConsoleError(429, "too many wrong passwords; wait ten minutes")
        if not self._password_ok(password):
            self._failures.append(now)
            time.sleep(0.5)
            raise ConsoleError(401, "wrong password")
        self._failures.clear()
        expires = int(now) + SESSION_SECONDS
        body = f"{expires}.{secrets.token_hex(8)}"
        return f"{body}.{hmac.new(self._secret, body.encode(), hashlib.sha256).hexdigest()}"

    def session_ok(self, cookie: str | None) -> bool:
        try:
            expires, nonce, signature = (cookie or "").split(".")
            expected = hmac.new(self._secret, f"{expires}.{nonce}".encode(), hashlib.sha256).hexdigest()
            return hmac.compare_digest(signature, expected) and int(expires) > self.clock()
        except ValueError:
            return False

    # -- what there is to review -----------------------------------------------------------------------------------------------------------------
    def _store(self) -> Any:
        from .experience_store import ExperienceStore

        return ExperienceStore(self.work_dir)

    def jobs(self) -> list[dict[str, Any]]:
        store = self._store()
        found: dict[str, dict[str, Any]] = {}
        for clip, info in (store.load_corpus().get("clips") or {}).items():
            job = str(info.get("job_id") or "")
            runs = []
            for run_dir in (self.work_dir / "experience" / "runs").glob(f"*/{clip}/r1/submission.json"):
                runs.append({"run": run_dir.parents[2].name, "at": datetime.fromtimestamp(run_dir.stat().st_mtime, timezone.utc).isoformat(timespec="minutes")})
            if not runs or not JOB_ID.match(job):
                continue
            try:
                manifest = json.loads((self.work_dir / "jobs" / job / "manifest.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                manifest = {}
            found[job] = {"job": job, "clip": clip, "name": manifest.get("source_filename") or clip, "created": str(manifest.get("created_at") or "")[:16],
                          "runs": sorted(runs, key=lambda r: r["at"], reverse=True)}
        return sorted(found.values(), key=lambda j: j["runs"][0]["at"], reverse=True)

    def _items(self, job: str, run: str) -> tuple[Path, list[dict[str, Any]]]:
        if not JOB_ID.match(job) or not RUN_ID.match(run):
            raise ConsoleError(400, "unknown job or run")
        job_root = self.work_dir / "jobs" / job
        clip = job[:8]
        path = self.work_dir / "experience" / "runs" / run / clip / "r1" / "submission.json"
        try:
            items = json.loads(path.read_text(encoding="utf-8"))["items"]
        except (OSError, ValueError, KeyError):
            raise ConsoleError(404, "no dub text for that run")
        return job_root, items

    def _prepared_for(self, job_root: Path) -> Prepared:
        key = str(job_root)
        if key not in self._prepared:
            self._prepared[key] = Prepared(job_root)
        return self._prepared[key]

    def _spelled(self, job: str, run: str) -> tuple[Path, list[dict[str, Any]], list[dict[str, Any]], Prepared]:
        job_root, items = self._items(job, run)
        prepared = self._prepared_for(job_root)
        texts = [str(i.get("zulu_text") or "") for i in items]
        spelled = spell(prepared, texts, progress=lambda _m: None)
        return job_root, items, with_committed_words(spelled, book.load(self.book_path)["entries"], texts), prepared

    def words(self, job: str, run: str) -> list[dict[str, Any]]:
        _, items, spelled, _ = self._spelled(job, run)
        joined = " ".join(str(i.get("zulu_text") or "") for i in items)
        rows = []
        for item in spelled:
            decision = item.get("decision")
            rows.append({"word": item["word"], "rules": item.get("rule_spelling") or item["spelling"] if not decision else item.get("rule_spelling", ""),
                         "spelling": (item.get("by_voice") or {}).get(VOICES[0]) or item["spelling"], "status": decision or "unreviewed", "reading": item["reading"],
                         "times": len(ear_clips.word_pattern(item["word"]).findall(joined))})
        order = {"unreviewed": 0, "custom": 1, "approved": 2}
        return sorted(rows, key=lambda r: (order.get(r["status"], 0), -r["times"], r["word"].casefold()))

    def word(self, job: str, run: str, word: str) -> dict[str, Any]:
        job_root, items, spelled, prepared = self._spelled(job, run)
        voices = self._voices(job_root)
        mine = next((s for s in spelled if s["word"].casefold() == word.casefold()), None)
        if mine is None:
            raise ConsoleError(404, "that word is not in this dub")
        listed = self._listed(prepared, mine["word"]) if mine["reading"] != "number" else []
        samples = ear_clips.pick(items, [mine["word"]], voices, per_word=2)
        entry = book.load(self.book_path)["entries"].get(book.key_of(mine["word"]))
        return {"word": mine["word"], "reading": mine["reading"], "english": " ".join(mine.get("english") or []), "rules": (mine.get("rule_spelling") or mine["spelling"]) if mine.get("decision") else mine["spelling"],
                "derivation": mine["derivation"], "committed": entry, "alternatives": listed,
                "samples": [{"voice": s["voice"], "unit": s["unit"], "text": s["text"]} for s in samples], "voices": list(VOICES)}

    @staticmethod
    def _listed(prepared: Prepared, word: str) -> list[dict[str, Any]]:
        """The rules' main spelling and their next ones, each with the open choices that separate it from the main one."""
        options = prepared.interpreter.alternatives(word, limit=8)
        main = options[0]["choice"] if options else None
        return [{"spelling": o["spelling"], "derivation": o["derivation"],
                 "differences": rules.choice_differences(main, o["choice"]) if main is not None and o["choice"] is not None else []} for o in options]

    def _voices(self, job_root: Path) -> dict[str, str]:
        try:
            resolution = json.loads((job_root / "direct_dub" / "voice_resolution.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        assignments = resolution.get("assignments", resolution)
        return {speaker: str(a.get("selected_voice") or "") for speaker, a in assignments.items() if isinstance(a, Mapping)}

    # -- hearing it -------------------------------------------------------------------------------------------------------------------------
    def _budget(self) -> tuple[Path, dict[str, Any]]:
        path = self.dir / "budget.json"
        today = datetime.fromtimestamp(self.clock(), timezone.utc).strftime("%Y-%m-%d")
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            state = {}
        if state.get("date") != today:
            state = {"date": today, "count": 0}
        return path, state

    def _clip(self, job: str, run: str, word: str, spelling: str, voice: str, sample: int | None = None) -> tuple[Path, str, str, str]:
        """What would be spoken for this spelling of the word in this voice: (job_root, speaker, text, file name).  The sentence is the first one that holds the word
        for this voice, else the first one that holds it; with no sentence at all the spelling is said alone."""
        if voice not in VOICES:
            raise ConsoleError(400, "unknown voice")
        try:
            spelling = book._valid_spelling(spelling)
        except ValueError as exc:
            raise ConsoleError(400, str(exc))
        job_root, items = self._items(job, run)
        voices = self._voices(job_root)
        rows = ear_clips.pick(items, [word], voices, per_word=2)
        if sample is None:
            sample = next((i for i, row in enumerate(rows) if row["voice"] == voice), 0)
        sentence = rows[min(max(int(sample), 0), len(rows) - 1)]["text"] if rows else word
        speaker = next((row["speaker_id"] for row in rows if row["voice"] == voice), None) or next((s for s, v in voices.items() if v == voice), None) or "SPEAKER_00"
        text = ear_clips.substitute(sentence, word, spelling) if rows else spelling
        return job_root, speaker, text, hashlib.sha256(f"{job}|{voice}|{text}".encode("utf-8")).hexdigest()[:32] + ".wav"

    def audio(self, job: str, run: str, word: str, spelling: str, voice: str, sample: int | None = None) -> dict[str, Any]:
        job_root, speaker, text, name = self._clip(job, run, word, spelling, voice, sample)
        self._ensure(name, job_root, speaker, voice, text)
        return {"url": f"/audio/{name}", "text": text, "voice": voice}

    def _ensure(self, name: str, job_root: Path, speaker: str, voice: str, text: str) -> None:
        target = self.audio_dir / name
        if not target.is_file():                                          # a clip made ahead is answered at once; only a synthesis holds the lock
            with self._lock:
                if not target.is_file():
                    path, state = self._budget()
                    if state["count"] >= self.max_audio_per_day:
                        raise ConsoleError(429, f"today's limit of {self.max_audio_per_day} new clips is used; hearing a clip again is free")
                    self._speak(job_root, speaker, voice, text, target)
                    state["count"] += 1
                    path.write_text(json.dumps(state), encoding="utf-8")

    # -- the source speakers: how the word was said in the English audio ---------------------------------------------------------------------------
    SOURCE_CONTEXT = 5                 # words either side of the one asked for, for the phrase clip
    SOURCE_MAX_PHRASE = 8.0            # seconds
    SOURCE_MIN_WORD = 0.7              # seconds: the shortest word clip
    SOURCE_PAUSE = 0.8                 # seconds of silence that end a phrase

    def _source(self, job: str) -> tuple[list[dict[str, Any]], Path, str] | None:
        """A job's English words with their times, its source audio, and its name; None when the job has no transcript or no audio."""
        if not JOB_ID.match(job):
            raise ConsoleError(400, "unknown job")
        root = self.work_dir / "jobs" / job
        transcript = next((root / "analysis" / n for n in ("corrected_transcript.json", "transcript_en.json") if (root / "analysis" / n).is_file()), None)
        audio = next((root / "audio" / n for n in ("analysis_mono.wav", "mix_source_stereo.wav") if (root / "audio" / n).is_file()), None)
        if transcript is None or audio is None:
            return None
        stamp = transcript.stat().st_mtime
        known = self._sources.get(job)
        if known is None or known[0] != stamp:
            try:
                words = [w for w in json.loads(transcript.read_text(encoding="utf-8")).get("words", []) if isinstance(w, Mapping) and w.get("start") is not None and w.get("end") is not None]
            except (OSError, ValueError):
                return None
            try:
                name = str(json.loads((root / "manifest.json").read_text(encoding="utf-8")).get("source_filename") or job[:8])
            except (OSError, ValueError):
                name = job[:8]
            known = self._sources[job] = (stamp, words, audio, name)
        return known[1], known[2], known[3]

    @staticmethod
    def _plain(text: Any) -> str:
        return re.sub(r"(?:['’]s)?[^\w]*$", "", re.sub(r"^[^\w]+", "", str(text or ""))).casefold()

    def source_words(self, word: str, job: str | None = None, limit: int = 5) -> list[dict[str, Any]]:
        """Where ``word`` is said in the source audio: in the order it is said, at least eight seconds apart, at most three to a job when looking across jobs.  Nothing
        is scored or ranked: the listener's ear decides.  Each has the words around it (``pre``, ``word``, ``post``), the time, and the ``index`` that ``source_audio`` takes."""
        word = _clean_word(word)
        key = self._plain(word)
        if not key:
            return []
        if job:
            jobs = [job]
        else:
            folder = self.work_dir / "jobs"
            jobs = sorted((d.name for d in folder.iterdir() if JOB_ID.match(d.name)), key=lambda n: (folder / n).stat().st_mtime, reverse=True) if folder.is_dir() else []
        found: list[dict[str, Any]] = []
        for one in jobs:
            source = self._source(one)
            if source is None:
                continue
            words, _, name = source
            hits = [i for i, w in enumerate(words) if self._plain(w.get("text")) == key]
            chosen: list[int] = []
            for i in hits:
                if all(abs(float(words[i]["start"]) - float(words[j]["start"])) >= 8 for j in chosen) and len(chosen) < (limit if job else 3):
                    chosen.append(i)
            for i in chosen:
                w = words[i]
                seconds = int(float(w["start"]))
                found.append({"job": one, "job_name": name, "index": i, "at": f"{seconds // 60}:{seconds % 60:02d}", "speaker": str(w.get("speaker") or ""),
                              "pre": " ".join(str(x.get("text")) for x in words[max(0, i - 6):i]), "word": str(w.get("text")),
                              "post": " ".join(str(x.get("text")) for x in words[i + 1:i + 7])})
        return found[:limit]

    def source_audio(self, job: str, index: int, mode: str) -> dict[str, Any]:
        """The source audio of one word (``mode`` "word") or of the phrase around it ("phrase"), cut from the job's own audio.  Local and free: no voice is asked."""
        if mode not in ("word", "phrase"):
            raise ConsoleError(400, "mode is word or phrase")
        source = self._source(job)
        if source is None:
            raise ConsoleError(404, "this job has no source audio here")
        words, audio, _ = source
        if not isinstance(index, int) or not 0 <= index < len(words):
            raise ConsoleError(400, "no such word")
        if mode == "word":
            start, end = float(words[index]["start"]) - 0.12, float(words[index]["end"]) + 0.12
            if end - start < self.SOURCE_MIN_WORD:                         # a word the recogniser timed at a fifth of a second is too short to hear: the same centre, wider
                middle = (start + end) / 2
                start, end = middle - self.SOURCE_MIN_WORD / 2, middle + self.SOURCE_MIN_WORD / 2
        else:
            lo = hi = index
            for _ in range(self.SOURCE_CONTEXT):                      # a few words either side, stopping at a pause (the phrase ends there) or at the longest clip
                if lo > 0 and float(words[lo]["start"]) - float(words[lo - 1]["end"]) <= self.SOURCE_PAUSE and float(words[hi]["end"]) - float(words[lo - 1]["start"]) <= self.SOURCE_MAX_PHRASE:
                    lo -= 1
                if hi < len(words) - 1 and float(words[hi + 1]["start"]) - float(words[hi]["end"]) <= self.SOURCE_PAUSE and float(words[hi + 1]["end"]) - float(words[lo]["start"]) <= self.SOURCE_MAX_PHRASE:
                    hi += 1
            start, end = float(words[lo]["start"]) - 0.1, float(words[hi]["end"]) + 0.1
        start = max(0.0, start)
        name = hashlib.sha256(f"source|{job}|{start:.3f}|{end:.3f}".encode("utf-8")).hexdigest()[:32] + ".wav"
        target = self.audio_dir / name
        if not target.is_file():
            self._cut(audio, start, max(0.2, end - start), target)
        return {"url": f"/audio/{name}", "start": round(start, 2), "end": round(end, 2), "mode": mode}

    def source_hear(self, job: str, index: int, mode: str) -> dict[str, Any]:
        """What a recogniser hears in this source clip, to compare with the ear: ``phones`` (a phoneme recogniser's IPA for the stretch) and ``reads_as`` (what an English
        letter recogniser writes for it, an alternative reading of the word).  Computed on request, once per clip, on this machine; shown and never used."""
        clip = self.source_audio(job, index, mode)
        name = clip["url"].rsplit("/", 1)[-1]
        cache = self.dir / "heard" / (name[:-4] + ".json")
        try:
            heard = json.loads(cache.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            with self._recognise_lock:                                  # one clip at a time: the models are big and this machine is small
                heard = {k: str(v) for k, v in self._recognise(self.audio_dir / name).items()}
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(heard, ensure_ascii=False), encoding="utf-8")
        return {"mode": mode, "url": clip["url"], **heard}

    RECOGNISER_IDLE_SECONDS = 300      # the two models take about 1.7 GB of this machine's 6: they are let go this long after the last clip

    def _recognise_local(self, wav: Path) -> dict[str, str]:
        if self._recognisers is None:
            from .english_listener import EnglishListener
            from .pronunciation_assess import LocalGop

            self._recognisers = (LocalGop(), EnglishListener())
        phones, letters = self._recognisers
        try:
            return {"phones": " ".join(phones.decode(wav)), "reads_as": " ".join(letters.transcribe(wav).split())}
        finally:
            if self._unload_timer is not None:
                self._unload_timer.cancel()
            self._unload_timer = threading.Timer(self.RECOGNISER_IDLE_SECONDS, self.unload_recognisers)
            self._unload_timer.daemon = True
            self._unload_timer.start()

    def unload_recognisers(self) -> bool:
        """Let go of the recogniser models (they load again on the next request).  True when there was something to let go of."""
        with self._recognise_lock:
            had = self._recognisers is not None
            self._recognisers = None
        if had:
            import gc

            gc.collect()
            try:
                import ctypes

                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception:
                pass
        return had

    @staticmethod
    def _ffmpeg_cut(audio: Path, start: float, duration: float, target: Path) -> None:
        if shutil.which("ffmpeg") is None:
            raise ConsoleError(503, "ffmpeg is not installed here, so the source audio cannot be cut")
        tmp = target.with_suffix(".part.wav")
        fade = min(0.015, duration / 4)
        result = subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(audio), "-ac", "1", "-ar", "24000",
                                 "-af", f"afade=t=in:d={fade:.3f},afade=t=out:st={max(0.0, duration - fade):.3f}:d={fade:.3f}", "-c:a", "pcm_s16le", str(tmp)],
                                capture_output=True, timeout=60)
        if result.returncode != 0 or not tmp.is_file():
            tmp.unlink(missing_ok=True)
            raise ConsoleError(500, "the source audio could not be cut")
        tmp.replace(target)

    # -- questions: what was said and in which language, asked where the pipeline cannot settle it -----------------------------------------------
    SEGMENT_AUDIO_MAX = 30.0           # seconds of a stretch played: the first part of a long one

    @staticmethod
    def _around(text: str, pattern: "re.Pattern[str]", width: int = 110) -> str:
        found = pattern.search(text)
        if not found:
            return text[:2 * width]
        start, end = max(0, found.start() - width), min(len(text), found.end() + width)
        return ("…" if start else "") + text[start:end].strip() + ("…" if end < len(text) else "")

    def questions(self, job: str, run: str) -> dict[str, Any]:
        """What the pipeline could not settle about this dub, for the listener: words of the Zulu dub that nobody said (``words``, with their sentence and the English of
        that moment), words that earlier answers settled by themselves (``resolved``, so they can be taken back), the answers already given (``answered``), and stretches
        where the restored text and the captions still differ (``stretches``).  ``open`` counts what has no answer."""
        job_root, items = self._items(job, run)
        prepared = self._prepared_for(job_root)
        prepared.refresh_lessons()
        spell(prepared, [str(i.get("zulu_text") or "") for i in items], progress=lambda _m: None)          # fills prepared.unsure and prepared.resolved

        def place(word: str) -> dict[str, str]:
            pattern = ear_clips.word_pattern(word)
            item = next((i for i in items if pattern.search(str(i.get("zulu_text") or ""))), None)
            return {"dub": self._around(str(item.get("zulu_text") or ""), pattern), "source": str(item.get("source_text") or "")[:400]} if item else {"dub": "", "source": ""}

        answered = prepared.lessons["words"]
        rows = [{"kind": "word", "id": language.key_of(u["word"]), "word": u["word"], **place(u["word"]), "answer": None} for u in prepared.unsure]
        done = [{"kind": "word", "id": key, "word": e["word"], **place(e["word"]), "answer": {"language": e["language"]}} for key, e in answered.items()
                if any(ear_clips.word_pattern(e["word"]).search(str(i.get("zulu_text") or "")) for i in items)]
        settled = [{"kind": "word", "id": language.key_of(r["word"]), "word": r["word"], **place(r["word"]), "answer": {"language": r["language"], "by_itself": True}} for r in prepared.resolved]
        stretches = language.stretch_questions(job_root)
        return {"words": rows, "answered": done, "resolved": settled, "stretches": stretches,
                "open": len(rows) + sum(1 for s in stretches if not s["answer"]), "languages": list(language.LANGUAGES)}

    def _refresh_lessons(self) -> None:
        for prepared in self._prepared.values():
            prepared.refresh_lessons()

    def answer(self, job: str, run: str, data: Mapping[str, Any]) -> dict[str, Any]:
        """Keep the listener's answer: a word's language (``kind`` "word", ``word``, ``language``) or which reading of a stretch was right (``kind`` "stretch", ``id``,
        ``choice`` model / transcript / typed, ``text``).  Both are lessons for every later job."""
        job_root, _ = self._items(job, run)
        kind = str(data.get("kind") or "")
        try:
            if kind == "word":
                word = _clean_word(data.get("word"))
                prepared = self._prepared_for(job_root)
                known = next((u for u in [*prepared.unsure, *prepared.resolved] if language.key_of(u["word"]) == language.key_of(word)), None)
                sig = (known or {}).get("signature") or (prepared.lessons["words"].get(language.key_of(word)) or {}).get("signature") or ""
                stored = language.answer_word(self.work_dir, word, str(data.get("language") or ""), sig)
            elif kind == "stretch":
                segment_id = str(data.get("id") or "")
                if not SEGMENT_ID.match(segment_id):
                    raise ConsoleError(400, "no such stretch")
                stored = language.answer_stretch(job_root, self.work_dir, segment_id, str(data.get("choice") or ""), str(data.get("text") or ""))
            else:
                raise ConsoleError(400, "kind is word or stretch")
        except ValueError as exc:
            raise ConsoleError(400, str(exc))
        self._refresh_lessons()
        return {"stored": stored, **self.questions(job, run)}

    def answer_revoke(self, job: str, run: str, word: str) -> dict[str, Any]:
        self._items(job, run)
        language.forget_word(self.work_dir, _clean_word(word))
        self._refresh_lessons()
        return self.questions(job, run)

    def segment_audio(self, job: str, segment_id: str) -> dict[str, Any]:
        """The source audio of one restored segment (the first ``SEGMENT_AUDIO_MAX`` seconds of it), cut from the job's own audio."""
        if not SEGMENT_ID.match(segment_id):
            raise ConsoleError(400, "no such stretch")
        source = self._source(job)
        if source is None:
            raise ConsoleError(404, "this job has no source audio here")
        segments = json.loads((self.work_dir / "jobs" / job / "native_dub" / "pass1_restored_transcript.json").read_text(encoding="utf-8")).get("segments") or []
        segment = next((s for s in segments if str(s.get("segment_id")) == segment_id), None)
        if segment is None:
            raise ConsoleError(404, "no such stretch")
        start = max(0.0, float(segment["start_ms"]) / 1000 - 0.15)
        end = min(float(segment["end_ms"]) / 1000 + 0.15, start + self.SEGMENT_AUDIO_MAX)
        name = hashlib.sha256(f"segment|{job}|{start:.3f}|{end:.3f}".encode("utf-8")).hexdigest()[:32] + ".wav"
        if not (self.audio_dir / name).is_file():
            self._cut(source[1], start, max(0.2, end - start), self.audio_dir / name)
        return {"url": f"/audio/{name}", "start": round(start, 2), "end": round(end, 2)}

    # -- wordsmith: any word typed in, heard in both voices -----------------------------------------------------------------------------------
    SMITH_JOB = "_wordsmith"                  # a job root with nothing in it: the rules and the book decide the spelling, no job's research or lexicon

    def smith(self, text: str, as_typed: bool = False) -> dict[str, Any]:
        """The spelling each voice would say for a typed word or short phrase: a word in the pronunciation book as committed, else the rules' spelling from the
        dictionary's sounds, else the word as typed (a Zulu word, or ``as_typed``).  Alternatives from the rules are listed when the text is one word."""
        from .job_pronunciations import core

        text = _clean_word(text)
        prepared = self._prepared_for(self.work_dir / "jobs" / self.SMITH_JOB)
        entries = book.load(self.book_path)["entries"]
        pieces: list[dict[str, Any]] = []
        for token in text.split():
            match = re.fullmatch(r"[^A-Za-z0-9]*(.*?)([.,]*)[^A-Za-z0-9.,]*", token)
            body, tail = (match.group(1), match.group(2)) if match else (token, "")
            english = core(body) if body else body
            prefix = body[:len(body) - len(english)]
            word = english.strip("-")
            piece: dict[str, Any] = {"token": token, "word": word, "source": "as typed", "reading": "", "derivation": [], "english": ""}
            said = {voice: word for voice in VOICES}
            if word and not as_typed and re.fullmatch(r"[A-Za-z][A-Za-z'’]*", word):
                key = book.key_of(word)
                if key in entries:
                    said = {voice: book.spelling_for(entries, word, voice) or word for voice in VOICES}
                    piece.update(source="book", reading="committed", derivation=["committed by the listener; no rule applies"])
                else:
                    found = prepared.interpreter.rules_spelling(word)
                    if found:
                        said = {voice: found["spelling"] for voice in VOICES}
                        piece.update(source="rules", reading=found["reading"], derivation=found["derivation"], english=" ".join(found["english"]))
            piece["by_voice"] = {voice: f"{prefix}{said[voice]}{tail}" if word else token for voice in VOICES}
            piece["bare"] = bool(word) and not prefix and not tail and token == word
            pieces.append(piece)
        if not pieces:
            raise ConsoleError(400, "a word is required")
        alternatives = self._listed(prepared, pieces[0]["word"]) if len(pieces) == 1 and pieces[0]["word"] and not as_typed else []
        sources = {p["source"] for p in pieces}
        committable = len(pieces) == 1 and pieces[0]["bare"] and bool(re.fullmatch(r"[A-Za-z][A-Za-z'’]*", pieces[0]["word"]))          # a single bare word: no Zulu prefix or punctuation to leave out
        return {"text": text, "committable": committable, "source": sources.pop() if len(sources) == 1 else "mixed", "spelling": {voice: " ".join(p["by_voice"][voice] for p in pieces) for voice in VOICES},
                "pieces": pieces, "alternatives": alternatives, "voices": list(VOICES)}

    def smith_commit(self, word: str, spellings: Mapping[str, Any], note: str = "") -> dict[str, Any]:
        """The listener heard these spellings (one per voice) in the Wordsmith and they are right: commit them (see ``_commit``)."""
        word = _clean_word(word)
        if not re.fullmatch(r"[A-Za-z][A-Za-z\'’]*", word):
            raise ConsoleError(400, "only a single word can be committed")
        said = {voice: book._valid_spelling(str(spellings.get(voice) or "")) for voice in VOICES}
        prepared = self._prepared_for(self.work_dir / "jobs" / self.SMITH_JOB)
        found = prepared.interpreter.rules_spelling(word)
        return self._commit(word, said, ruled=found["spelling"] if found else "", alternatives=self._listed(prepared, word), english=" ".join(found["english"]) if found else "",
                            reading=found["reading"] if found else "", derivation=found["derivation"] if found else [], note=note or "committed in the Wordsmith")

    def smith_audio(self, spelling: str, voice: str) -> dict[str, Any]:
        """The spelling spoken alone by one voice."""
        if voice not in VOICES:
            raise ConsoleError(400, "unknown voice")
        try:
            spelling = book._valid_spelling(spelling)
        except ValueError as exc:
            raise ConsoleError(400, str(exc))
        name = hashlib.sha256(f"wordsmith|{voice}|{spelling}".encode("utf-8")).hexdigest()[:32] + ".wav"
        self._ensure(name, self.work_dir / "jobs" / self.SMITH_JOB, "SPEAKER_00", voice, spelling)
        return {"url": f"/audio/{name}", "text": spelling, "voice": voice}

    def ready(self, job: str, run: str, word: str, items: list[Mapping[str, Any]]) -> list[bool]:
        """Which of these clips (each ``{spelling, voice}``) are already made."""
        return [(self.audio_dir / self._clip(job, run, word, str(i.get("spelling") or ""), str(i.get("voice")))[3]).is_file() for i in items[:40]]

    def prefetch(self, job: str, run: str, words: list[str], *, alternatives: bool = False) -> int:
        """Make the clips of these words ahead of the listener (the spelling that is current for each voice, and with ``alternatives`` every alternative of the rules too),
        one at a time in the background.  Returns how many were queued."""
        _, _, spelled, prepared = self._spelled(job, run)
        by_word = {s["word"].casefold(): s for s in spelled}
        wanted: list[tuple[str, str, str]] = []
        for word in words[:8]:
            found = by_word.get(str(word).casefold())
            if not found:
                continue
            for voice in VOICES:
                wanted.append((found["word"], (found.get("by_voice") or {}).get(voice) or found["spelling"], voice))
            if alternatives and found["reading"] not in ("number", "committed"):
                for alt in prepared.interpreter.alternatives(found["word"], limit=8)[1:]:
                    wanted.extend((found["word"], alt["spelling"], voice) for voice in VOICES)
        queued = 0
        for word, spelling, voice in wanted:
            name = self._clip(job, run, word, spelling, voice)[3]
            if name in self._queued or (self.audio_dir / name).is_file():
                continue
            self._queued.add(name)
            self._ahead.put((job, run, word, spelling, voice))
            queued += 1
        if queued and (self._worker is None or not self._worker.is_alive()):
            self._worker = threading.Thread(target=self._make_ahead, name="clips-ahead", daemon=True)
            self._worker.start()
        return queued

    def _make_ahead(self) -> None:
        while True:
            try:
                job, run, word, spelling, voice = self._ahead.get(timeout=30)
            except queue.Empty:
                return
            try:
                self.audio(job, run, word, spelling, voice)
            except ConsoleError as exc:
                if exc.status == 429:                                      # today's ceiling: stop making clips ahead
                    while not self._ahead.empty():
                        self._ahead.get_nowait()
                        self._ahead.task_done()
            except Exception as exc:                                       # one clip that cannot be made must not stop the others
                sys.stderr.write(f"clip ahead failed: {type(exc).__name__}: {str(exc)[:160]}\n")
            finally:
                self._queued.discard(self._clip_name_or_none(job, run, word, spelling, voice))
                self._ahead.task_done()

    def _clip_name_or_none(self, job: str, run: str, word: str, spelling: str, voice: str) -> str | None:
        try:
            return self._clip(job, run, word, spelling, voice)[3]
        except ConsoleError:
            return None

    def wait_idle(self, timeout: float = 5.0) -> bool:
        """For tests: wait until the clips queued ahead are made."""
        end = time.time() + timeout
        while time.time() < end:
            if self._ahead.unfinished_tasks == 0:
                return True
            time.sleep(0.01)
        return False

    def _speak(self, job_root: Path, speaker: str, voice: str, text: str, target: Path) -> None:
        if self._synthesize is not None:
            self._synthesize(job_root=job_root, speaker=speaker, voice=voice, text=text, target=target)
            return
        from .cli import create_tts_backend
        from .config import load_settings
        from .native_dub import _apply_native_pronunciation_research_to_job_cache, _synthesize_group

        if self._tts is None:
            self._tts = create_tts_backend(load_settings())
        _apply_native_pronunciation_research_to_job_cache(job_root, {"accepted_corrections": []})          # the dictionary the renderer reads: this job's lexicon
        voices = self._voices(job_root)
        info = {}
        try:
            info = json.loads((job_root / "direct_dub" / "voice_resolution.json").read_text(encoding="utf-8")).get("assignments", {}).get(speaker, {})
        except (OSError, ValueError):
            pass
        info = {**info, "selected_voice": voice, "base_prosody": info.get("base_prosody") or {}}
        group = {"group_id": "console", "speaker_id": speaker, "parts": [{"type": "text", "text": text}]}
        _synthesize_group(tts=self._tts, group=group, voice_info=info, output_path=target, preferred_ms=max(2000, 120 * len(text)), maximum_ms=max(4000, 240 * len(text)),
                          force=False, job_root=job_root)

    # -- deciding ---------------------------------------------------------------------------------------------------------------------------
    def correct(self, job: str, run: str, word: str, spelling: str, voice: str | None = None, note: str = "") -> dict[str, Any]:
        """The listener heard this spelling and it is right: commit it, for both voices or for ``voice`` only."""
        detail = self.word(job, run, word)
        said = {v: spelling for v in ([voice] if voice else VOICES)}
        if any(v not in VOICES for v in said):
            raise ConsoleError(400, "unknown voice")
        return self._commit(word, said, ruled=detail["rules"], alternatives=detail["alternatives"], english=detail["english"], reading=detail["reading"], derivation=detail["derivation"],
                            job=job, note=note)

    def _commit(self, word: str, spellings: Mapping[str, str], *, ruled: str, alternatives: list[dict[str, Any]], english: str = "", reading: str = "", derivation: Any = (),
                job: str = "", note: str = "") -> dict[str, Any]:
        """Commit a spelling for each voice given.  What the rules wrote is "approved"; anything else is "custom", and is also kept as feedback for the rules (the
        sounds the dictionary heard, how the rules built their spelling, what was chosen, how they differ), for every custom spelling, so that patterns can show."""
        said = {voice: book._valid_spelling(spelling) for voice, spelling in spellings.items()}
        whole = set(said) == set(VOICES)
        if whole and len(set(said.values())) == 1:
            groups = [(book.ALL_VOICES, next(iter(said.values())))]
        else:
            groups = list(said.items())
        entry: dict[str, Any] = {}
        for i, (voice, spelling) in enumerate(groups):
            entry = book.decide(self.book_path, word, spelling, kind="approved" if spelling == ruled else "custom", voice=voice, rule_spelling=ruled, job=job, note=note,
                                replace=whole and i == 0)
        for spelling in dict.fromkeys(said.values()):
            if ruled and spelling != ruled:
                differences = next((a["differences"] for a in alternatives if a["spelling"] == spelling), [])
                book.record_feedback(self.book_path, word=word, rule_spelling=ruled, chosen=spelling, differences=differences, job=job, note=note, english=english, reading=reading,
                                     derivation=derivation)
        return entry

    def book_view(self) -> dict[str, Any]:
        data = book.load(self.book_path)
        entries = sorted(data["entries"].values(), key=lambda e: str(e.get("decided_at")), reverse=True)
        return {"entries": entries, "feedback": data["feedback"][-100:][::-1], "log": data["log"][-100:][::-1], "tally": book.rule_tally(data)}

    def rules_view(self) -> dict[str, Any]:
        tally = book.rule_tally(book.load(self.book_path))["rules"]
        defaults = rules.rule_defaults()
        try:
            history = json.loads(rules.rules_config_path().read_text(encoding="utf-8")).get("history", [])
        except (OSError, ValueError):
            history = []
        listed = []
        for name, (label, options) in rules.RULE_LABELS.items():
            asked = tally.get(name, {})
            listed.append({"rule": name, "label": label, "options": [{"value": v, "means": m} for v, m in options.items()], "as_written": next(iter(options)),
                           "default": defaults.get(name), "wanted": asked.get("wanted", {}), "words": asked.get("words", {})})
        data = book.load(self.book_path)
        try:
            analysis = json.loads((self.dir / "analysis.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            analysis = None
        return {"rules": listed, "history": history[-30:][::-1], "patterns": book.patterns(data), "lessons": len(data["feedback"]), "analysis": analysis}

    # -- learning: ask the model why the listener's spellings differ from the rules' ---------------------------------------------------------------
    MAX_LESSONS = 80
    MAX_ANALYSES_PER_DAY = 10

    def learn(self) -> dict[str, Any]:
        """One call to the model with the listener's custom spellings (each with the sounds the dictionary heard and how the rules built theirs), the rules and their open
        choices: it says what the spellings have in common, why the rules were wrong and what each change to a rule would be.  It changes nothing; the answer is shown
        on the Rules page.  One call, no retries, ten a day."""
        data = book.load(self.book_path)
        lessons = data["feedback"][-self.MAX_LESSONS:]
        if not lessons:
            raise ConsoleError(400, "no custom spellings yet: commit a spelling of your own first, and the rules have something to learn from")
        with self._lock:
            path, state = self._budget()
            if state.get("analyses", 0) >= self.MAX_ANALYSES_PER_DAY:
                raise ConsoleError(429, f"today's {self.MAX_ANALYSES_PER_DAY} analyses are used")
            state["analyses"] = state.get("analyses", 0) + 1
            path.write_text(json.dumps(state), encoding="utf-8")
        payload = {"the_rules": (rules.__doc__ or "").strip(), "open_choices": {name: {"label": label, "options": {str(k): v for k, v in options.items()}}
                                                                                    for name, (label, options) in rules.RULE_LABELS.items()},
                   "defaults_now": {k: v for k, v in rules.rule_defaults().items()},
                   "lessons": [{"word": r["word"], "english_sounds": r.get("english"), "reading": r.get("reading"), "how_the_rules_built_it": r.get("derivation"),
                                "rules_spelling": r["rule_spelling"], "listener_wrote": r["chosen"], "syllable_changes": r.get("diff"), "listener_note": r.get("note"),
                                "open_choices_that_differ": [d.get("label") for d in r.get("differences") or []]} for r in lessons],
                   "patterns_seen": book.patterns(data)}
        result = (self._analyse or self._ask_model)(payload)
        record = {"at": datetime.fromtimestamp(self.clock(), timezone.utc).isoformat(timespec="seconds"), "lessons": len(lessons), "result": result}
        (self.dir / "analysis.json").write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
        return record

    ANALYSIS_SYSTEM = (
        "You help improve a set of rules that spell English words, names and initialisms the way South African isiZulu speakers say them, so that an Azure isiZulu "
        "text-to-speech voice (Thando, Themba) pronounces them right.  A listener hears the rules' spelling and, when it is not right, writes their own.  Every such "
        "spelling is a lesson.  For each lesson you are given the English sounds, how the rules built their spelling, what the listener wrote, and how the syllables "
        "differ.  Find what the lessons have in common, say WHY the rules' spelling was wrong in terms of the sounds and of how the voice reads Zulu spelling (the "
        "listener usually does not know why; the voice reads some letters and doubled letters differently from what a person expects), and say what change to the rules "
        "would make them write the listener's spelling for this kind of word, without breaking words that were right.  Be concrete: name the English sound or "
        "letter, the position in the word, the old and the new spelling.  When a change is one of the rules' open choices, name it and the option.  When the evidence is "
        "one word, say so and say what a second example would settle; never invent a rule from a single example without saying it is a guess.  ``confidence`` is exactly one of "
        "the strings high, medium or low; ``open_choice`` is the name of one of the open choices, or an empty string when the change is not one of them.")
    ANALYSIS_SCHEMA = {
        "type": "object", "additionalProperties": False, "required": ["summary", "findings"],
        "properties": {
            "summary": {"type": "string"},
            "findings": {"type": "array", "maxItems": 12, "items": {
                "type": "object", "additionalProperties": False,
                "required": ["pattern", "why", "words", "proposed_change", "open_choice", "confidence"],
                "properties": {"pattern": {"type": "string"}, "why": {"type": "string"}, "words": {"type": "array", "items": {"type": "string"}, "maxItems": 20},
                               "proposed_change": {"type": "string"}, "open_choice": {"type": "string"},          # the name of one of the rules' open choices, or an empty string
                               "confidence": {"type": "string"}}}}}}          # "high", "medium" or "low" (said in the prompt: the shared client's compact contract shows an enum badly)

    def _ask_model(self, payload: dict[str, Any]) -> dict[str, Any]:
        import dataclasses

        from .foundry_grok import FoundryGrokProvider, native_gpt_config_from_environment

        config = dataclasses.replace(native_gpt_config_from_environment(dict(os.environ)), max_schema_repairs=0, max_retries=2, reasoning_effort="medium", max_output_tokens=6000)
        return dict(FoundryGrokProvider(config).complete_json(operation="pronunciation_rule_lessons", system_prompt=self.ANALYSIS_SYSTEM, payload=payload,
                                                              schema=self.ANALYSIS_SCHEMA).data)

    # -- http --------------------------------------------------------------------------------------------------------------------------------
    def handle(self, method: str, target: str, headers: Mapping[str, str], body: bytes = b"") -> tuple[int, dict[str, str], bytes]:
        from .console_ui import PAGE

        parts = urlsplit(target)
        path, query = parts.path, {k: v[0] for k, v in parse_qs(parts.query).items()}
        base = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer"}

        def reply(status: int, payload: Any = None, extra: Mapping[str, str] | None = None) -> tuple[int, dict[str, str], bytes]:
            return status, {**base, "Content-Type": "application/json", **(extra or {})}, json.dumps(payload if payload is not None else {}, ensure_ascii=False).encode("utf-8")

        try:
            if method == "GET" and path == "/":
                nonce = secrets.token_urlsafe(12)
                csp = (f"default-src 'none'; script-src 'nonce-{nonce}'; style-src 'nonce-{nonce}'; connect-src 'self'; media-src 'self'; img-src 'self' data:; "
                       "base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
                return 200, {**base, "Content-Type": "text/html; charset=utf-8", "Content-Security-Policy": csp}, PAGE.replace("__NONCE__", nonce).encode("utf-8")
            cookies = dict(c.strip().split("=", 1) for c in (headers.get("Cookie") or headers.get("cookie") or "").split(";") if "=" in c)
            if method in ("POST", "DELETE") and (headers.get("X-Requested-With") or headers.get("x-requested-with")) != "console":
                raise ConsoleError(403, "not from the console page")
            if len(body) > MAX_BODY:
                raise ConsoleError(413, "too large")
            data = json.loads(body.decode("utf-8")) if body else {}
            if method == "POST" and path == "/api/login":
                cookie = self.login(str(data.get("password") or ""))
                return reply(200, {"ok": True}, {"Set-Cookie": f"session={cookie}; Path=/; HttpOnly; Secure; SameSite=Strict; Max-Age={SESSION_SECONDS}"})
            if not self.session_ok(cookies.get("session")):
                raise ConsoleError(401, "sign in")
            if method == "GET" and path == "/api/session":
                return reply(200, {"ok": True, "voices": list(VOICES)})
            if method == "POST" and path == "/api/logout":
                return reply(200, {"ok": True}, {"Set-Cookie": "session=; Path=/; HttpOnly; Secure; SameSite=Strict; Max-Age=0"})
            if method == "GET" and path == "/api/jobs":
                return reply(200, self.jobs())
            if method == "GET" and path == "/api/words":
                return reply(200, self.words(query.get("job", ""), query.get("run", "")))
            if method == "GET" and path == "/api/word":
                return reply(200, self.word(query.get("job", ""), query.get("run", ""), _clean_word(query.get("word"))))
            if method == "POST" and path == "/api/audio":
                sample = data.get("sample")
                return reply(200, self.audio(str(data.get("job")), str(data.get("run")), _clean_word(data.get("word")), str(data.get("spelling") or ""), str(data.get("voice")), None if sample is None else int(sample)))
            if method == "GET" and path == "/api/source":
                return reply(200, {"occurrences": self.source_words(query.get("word", ""), query.get("job") or None)})
            if method == "POST" and path == "/api/source-hear":
                return reply(200, self.source_hear(str(data.get("job")), data.get("index"), str(data.get("mode"))))
            if method == "POST" and path == "/api/source-audio":
                return reply(200, self.source_audio(str(data.get("job")), data.get("index"), str(data.get("mode"))))
            if method == "POST" and path == "/api/smith":
                return reply(200, self.smith(data.get("text"), bool(data.get("as_typed"))))
            if method == "POST" and path == "/api/smith-commit":
                spellings = data.get("spelling") if isinstance(data.get("spelling"), Mapping) else {}
                return reply(200, self.smith_commit(str(data.get("word") or ""), spellings, str(data.get("note") or "")[:400]))
            if method == "POST" and path == "/api/smith-audio":
                return reply(200, self.smith_audio(str(data.get("spelling") or ""), str(data.get("voice"))))
            if method == "POST" and path == "/api/prefetch":
                words = [_clean_word(w) for w in list(data.get("words") or [])[:8]]
                return reply(200, {"queued": self.prefetch(str(data.get("job")), str(data.get("run")), words, alternatives=bool(data.get("alternatives")))})
            if method == "POST" and path == "/api/ready":
                items = [i for i in list(data.get("items") or [])[:40] if isinstance(i, Mapping)]
                return reply(200, {"ready": self.ready(str(data.get("job")), str(data.get("run")), _clean_word(data.get("word")), items)})
            if method == "POST" and path == "/api/correct":
                return reply(200, self.correct(str(data.get("job")), str(data.get("run")), _clean_word(data.get("word")), str(data.get("spelling") or ""), data.get("voice") or None,
                                               str(data.get("note") or "")[:400]))
            if method == "GET" and path == "/api/questions":
                return reply(200, self.questions(query.get("job", ""), query.get("run", "")))
            if method == "POST" and path == "/api/answer":
                return reply(200, self.answer(str(data.get("job")), str(data.get("run")), data))
            if method == "POST" and path == "/api/answer-revoke":
                return reply(200, self.answer_revoke(str(data.get("job")), str(data.get("run")), data.get("word")))
            if method == "POST" and path == "/api/segment-audio":
                return reply(200, self.segment_audio(str(data.get("job")), str(data.get("id"))))
            if method == "POST" and path == "/api/learn":
                return reply(200, self.learn())
            if method == "GET" and path == "/api/book":
                return reply(200, self.book_view())
            if method == "POST" and path == "/api/book/revoke":
                return reply(200, {"revoked": book.revoke(self.book_path, _clean_word(data.get("word")))})
            if method == "GET" and path == "/api/rules":
                return reply(200, self.rules_view())
            if method == "POST" and path == "/api/rules":
                rules.write_rule_default(str(data.get("rule")), data.get("pick"), note=str(data.get("note") or "")[:400])
                return reply(200, self.rules_view())
            if method == "GET" and path.startswith("/audio/"):
                name = path[len("/audio/"):]
                if not AUDIO_NAME.match(name) or not (self.audio_dir / name).is_file():
                    raise ConsoleError(404, "no such clip")
                return 200, {**base, "Content-Type": "audio/wav", "Cache-Control": "private, max-age=3600"}, (self.audio_dir / name).read_bytes()
            raise ConsoleError(404, "not found")
        except ConsoleError as exc:
            return reply(exc.status, {"error": exc.message})
        except (ValueError, KeyError) as exc:
            return reply(400, {"error": str(exc)[:200]})
        except Exception as exc:                                         # never show a stack trace to the browser
            sys.stderr.write(f"console error: {type(exc).__name__}: {exc}\n")
            return reply(500, {"error": "something went wrong on the server"})


def serve(console: Console, host: str = "127.0.0.1", port: int = 8765) -> None:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _run(self, method: str) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(min(length, MAX_BODY + 1)) if length else b""
            status, headers, payload = console.handle(method, self.path, dict(self.headers), body)
            self.send_response(status)
            for name, value in headers.items():
                self.send_header(name, value)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:
            self._run("GET")

        def do_POST(self) -> None:
            self._run("POST")

        def do_DELETE(self) -> None:
            self._run("DELETE")

        def log_message(self, format: str, *args: Any) -> None:          # no request lines: they would carry words and job ids into the journal for nothing
            pass

    ThreadingHTTPServer((host, port), Handler).serve_forever()


def main(argv: list[str] | None = None) -> int:
    import argparse
    import getpass

    from .config import load_settings

    parser = argparse.ArgumentParser(prog="mathula_tv.console")
    parser.add_argument("command", choices=["serve", "set-password"])
    parser.add_argument("--port", type=int, default=int(os.environ.get("MATHULA_TV_CONSOLE_PORT", "8765")))
    args = parser.parse_args(argv)
    console = Console(Path(load_settings().work_dir))
    if args.command == "set-password":
        first = getpass.getpass("New console password (at least 10 characters): ")
        if first != getpass.getpass("Again: "):
            print("The two did not match; nothing was changed.")
            return 1
        console.set_password(first)
        print("Password stored (only its hash).")
        return 0
    serve(console, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
