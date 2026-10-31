"""POWSM as the listener of last resort for a word the other recognisers cannot read from the speaker's audio.

POWSM (Phonetic Open Whisper-style Speech Model) reads a stretch of audio into phones, blind; given a phrase of a few words, in the Nonvalo interview it
heard the "h aɪ" of a 0.16 s "High" that the XLSR phone recogniser and ZIPA both read as "pa".  It is large (about 20 s of CPU per call at a fixed 20 s window) and
installs only under Python 3.11, so it runs in a separate environment as a long-lived worker (``powsm_serve.py``) that the lab starts on first use and that keeps
the model loaded; answers are kept on disk beside the job so a word is read once.

Set MATHULA_POWSM_PYTHON to the interpreter of that environment (default ``~/powsm_lab/env311/bin/python``).  Without it ``available()`` is False and the lab uses
the English-voice reference instead.
"""
from __future__ import annotations

import atexit
import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Sequence

DEFAULT_PYTHON = "~/powsm_lab/env311/bin/python"
START_TIMEOUT_SECONDS = 240
CALL_TIMEOUT_SECONDS = 240
PAD_SECONDS = 0.4                # the word's stretch is read with this much audio either side, so the model hears it in its surroundings


def python_path() -> Path:
    return Path(os.environ.get("MATHULA_POWSM_PYTHON") or DEFAULT_PYTHON).expanduser()


class PowsmListener:
    def __init__(self, cache_path: Path | None = None, command: Sequence[str] | None = None) -> None:
        self.command = list(command) if command else [str(python_path()), "-u", str(Path(__file__).with_name("powsm_serve.py"))]
        self.cache_path = Path(cache_path) if cache_path else None
        self._cache: dict[str, list[str]] | None = None
        self._process: subprocess.Popen | None = None
        self._lines: "queue.Queue[str | None]" = queue.Queue()

    @staticmethod
    def available() -> bool:
        return python_path().is_file()

    def _read_cache(self) -> dict[str, list[str]]:
        if self._cache is None:
            try:
                self._cache = json.loads(self.cache_path.read_text(encoding="utf-8")) if self.cache_path and self.cache_path.is_file() else {}
            except (OSError, ValueError):
                self._cache = {}
        return self._cache

    def _line(self, timeout: float) -> dict[str, Any]:
        """The next ``@@`` line from the worker (its other output is skipped)."""
        while True:
            try:
                raw = self._lines.get(timeout=timeout)
            except queue.Empty:
                raise TimeoutError("the POWSM worker did not answer in time") from None
            if raw is None:
                raise RuntimeError("the POWSM worker ended")
            if raw.startswith("@@"):
                return json.loads(raw[2:])

    def _start(self) -> None:
        if self._process and self._process.poll() is None:
            return
        self._process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding="utf-8", bufsize=1)
        self._lines = queue.Queue()
        threading.Thread(target=self._pump, args=(self._process, self._lines), daemon=True).start()
        atexit.register(self.close)
        self._line(START_TIMEOUT_SECONDS)                 # {"ready": true} once the model is loaded

    @staticmethod
    def _pump(process: subprocess.Popen, lines: "queue.Queue[str | None]") -> None:
        """Move the worker's output lines to a queue, so that waiting for one can time out on any platform."""
        assert process.stdout
        for raw in process.stdout:
            lines.put(raw)
        lines.put(None)

    def close(self) -> None:
        if self._process and self._process.poll() is None:
            self._process.terminate()
        self._process = None

    def phones(self, wav: Path, start: float, end: float) -> list[str] | None:
        """The phones POWSM hears in the stretch ``start``..``end`` of the audio (up to 20 s), or None when it cannot be asked or fails."""
        key = json.dumps([str(wav), round(start, 2), round(end, 2), Path(wav).stat().st_size if Path(wav).is_file() else 0])
        cache = self._read_cache()
        if key in cache:
            return cache[key] or None
        try:
            self._start()
            assert self._process and self._process.stdin
            self._process.stdin.write(json.dumps({"wav": str(wav), "start": max(0.0, start), "end": end}) + "\n")
            self._process.stdin.flush()
            reply = self._line(CALL_TIMEOUT_SECONDS)
        except Exception:
            self.close()
            return None
        phones = [str(p) for p in reply.get("phones") or []]
        if reply.get("error") or not phones:
            return None
        cache[key] = phones
        if self.cache_path:
            try:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                self.cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
            except OSError:
                pass
        return phones
