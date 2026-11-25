"""A cache of what the voice has already said, one span at a time.

The voice service generates every span of a request on its own, with the random generator set to that span's seed first (gcp/omnivoice_service/app.py: ``synthesize_rest``), so the audio of a span
(and the times of its words) depends only on its text, language, reference clip and its text, pinned length, speed, adapter, decoding settings and seed, never on the other spans of the request.  The
same span asked for again is therefore the same audio, and the agent that re-measures a unit after changing one sentence, the studio that builds the unit it just measured, and a render of it, each ask
for mostly spans that were already spoken.  A span that is asked for with no seed is never cached: it is not repeatable.

``OmniVoiceClient.synthesize`` looks every seeded span up here, sends the service only the spans that are missing, and puts the reply together as the service would have (the silence of each gap between
spans, where every span starts, how long it is, the times of its words).  A request with nothing cached goes to the service as it was and its reply is returned untouched.

The key carries a tag (``OMNIVOICE_CACHE_TAG``, ``VERSION`` here): change it when the voice itself changes (a new adapter, a new model, another aligner), and nothing said before is used again.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import soundfile as sf

VERSION = "1"
DEFAULT_MAX_BYTES = 800 * 1024 * 1024
KEEP_AFTER_EVICTION = 0.8          # when the cache is over its size, the least recently used spans go until it is this full


def reference_digest(wav: bytes | None, text: str | None) -> str:
    return hashlib.sha256((wav or b"") + b"\0" + str(text or "").encode("utf-8")).hexdigest()


def span_key(*, text: str, language: str, reference: str, duration: float | None, speed: float, lora: bool, generation: Mapping[str, Any], seed: int, tag: str = "") -> str:
    """What decides the audio of one span, as one hash."""
    knobs = sorted((str(k), float(v)) for k, v in (generation or {}).items())
    payload = json.dumps([VERSION, tag, text, language or "zu", reference, None if not duration else round(float(duration), 3), None if duration else round(float(speed), 4), bool(lora), knobs, int(seed)], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class SpanCache:
    def __init__(self, folder: Path | str, *, max_bytes: int = DEFAULT_MAX_BYTES) -> None:
        self.folder = Path(folder)
        self.max_bytes = int(max_bytes)
        self.folder.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.folder / f"{key}.npz"

    def get(self, key: str, *, need_words: bool) -> dict[str, Any] | None:
        """{audio (int16), rate, words (list | None)} of a span said before, or None (also when its words are wanted and were not kept)."""
        path = self._path(key)
        try:
            with np.load(path, allow_pickle=False) as data:
                meta = json.loads(str(data["meta"]))
                audio = np.asarray(data["audio"], dtype=np.int16)
        except (OSError, ValueError, KeyError):
            return None
        words = meta.get("words")
        if need_words and (words is None or words == "none"):
            return None
        try:
            os.utime(path, None)                                     # used just now: the last to go
        except OSError:
            pass
        return {"audio": audio, "rate": int(meta["rate"]), "words": None if words == "none" else words}

    def put(self, key: str, audio: np.ndarray, rate: int, words: Any) -> None:
        path = self._path(key)
        temporary = path.with_name(f"{path.stem}.{os.getpid()}.{time.time_ns()}.tmp.npz")
        try:
            np.savez(temporary, audio=np.asarray(audio, dtype=np.int16), meta=np.array(json.dumps({"rate": int(rate), "words": "none" if words is None else words}, ensure_ascii=False)))
            os.replace(temporary, path)
        except OSError:
            try:
                temporary.unlink()
            except OSError:
                pass
            return
        self.evict()

    def evict(self) -> None:
        try:
            files = [(p, p.stat()) for p in self.folder.glob("*.npz")]
        except OSError:
            return
        total = sum(s.st_size for _, s in files)
        if total <= self.max_bytes:
            return
        for path, stat in sorted(files, key=lambda x: x[1].st_mtime):
            try:
                path.unlink()
            except OSError:
                continue
            total -= stat.st_size
            if total <= self.max_bytes * KEEP_AFTER_EVICTION:
                break


def decode_wav(audio_b64: str) -> tuple[np.ndarray, int]:
    data, rate = sf.read(io.BytesIO(base64.b64decode(audio_b64)), dtype="int16")
    return (data.mean(axis=1).astype(np.int16) if data.ndim > 1 else data), int(rate)


def encode_wav(audio: np.ndarray, rate: int) -> str:
    buffer = io.BytesIO()
    sf.write(buffer, np.asarray(audio, dtype=np.int16), rate, format="WAV", subtype="PCM_16")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def split_reply(reply: Mapping[str, Any]) -> tuple[list[np.ndarray], int]:
    """The audio of each span of a service reply, and the sample rate (cut where the reply says each span starts and how long it is, as every consumer of a reply does)."""
    audio, rate = decode_wav(str(reply["audio_b64"]))
    parts = [audio[int(float(s) * rate): int((float(s) + ms / 1000.0) * rate)] for s, ms in zip(reply.get("span_starts_s") or [], reply.get("span_ms") or [])]
    return parts, rate


def assemble(parts: Sequence[np.ndarray], rate: int, gaps_ms: Sequence[int], words: Sequence[Any] | None, gpu_s: float) -> dict[str, Any]:
    """A reply as the service makes it: the spans one after another with the silence of each gap between them."""
    chunks: list[np.ndarray] = []
    starts: list[float] = []
    lengths: list[int] = []
    elapsed = 0
    millisecond = rate // 1000 if rate % 1000 == 0 else 1
    for index, part in enumerate(parts):
        if len(part) % millisecond:                                  # a span is as long as a whole number of milliseconds (what the reply can say), by a few zeros at its end, never by cutting it
            part = np.concatenate([part, np.zeros(millisecond - len(part) % millisecond, dtype=np.int16)])
        starts.append(round(elapsed / rate, 3))
        lengths.append(len(part) // millisecond)
        chunks.append(part)
        elapsed += len(part)
        if index < len(parts) - 1:
            silence = np.zeros(int((int(gaps_ms[index]) if index < len(gaps_ms) else 0) / 1000.0 * rate), dtype=np.int16)
            chunks.append(silence)
            elapsed += len(silence)
    track = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)
    answer: dict[str, Any] = {"audio_b64": encode_wav(track, rate), "sample_rate": rate, "duration_ms": int(len(track) / rate * 1000), "span_starts_s": starts, "span_ms": lengths, "gpu_s": round(float(gpu_s), 2)}
    if words is not None:
        answer["aligned"] = bool(words and all(w is not None for w in words))
        answer["words"] = list(words)
    return answer
