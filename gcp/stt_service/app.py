"""Speech-to-text and speaker diarization as a REST service (Cloud Run, one L4 GPU), in place of Azure Fast Transcription.

POST /v1/transcribe   multipart: ``audio`` (any file ffmpeg reads) and ``params`` (JSON):
    {"locale": "en-ZA", "diarize": true, "max_speakers": 10, "min_speakers": null, "phrases": ["Madlanga", ...]}
  -> the JSON azure_stt.normalize reads: combinedPhrases, durationMilliseconds, phrases (one speaker each, with words), plus ``engine`` (what ran and how long each part took).

* English only: faster-whisper large-v3 with word timestamps and VAD; ``phrases`` become the initial prompt (a spelling hint, Whisper's way of biasing towards names).
* diarization: pyannote (DIARIZER, default the Community-1 pipeline); every word takes the speaker whose turn it overlaps most (stt_merge).
One generation at a time (GPU_LOCK): the GPU is one L4.
"""
import json
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import torch  # noqa: F401 - imported first so its bundled cuDNN/cuBLAS are the ones CTranslate2 loads
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

import stt_merge

HF_TOKEN = (os.environ.get("HF_TOKEN") or "").strip()          # a secret piped in from a terminal keeps its trailing newline, which is an illegal HTTP header value
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3")
WHISPER_ROOT = os.environ.get("WHISPER_ROOT", "/models/whisper")
DIARIZER = os.environ.get("DIARIZER", "pyannote/speaker-diarization-community-1")
PROMPT_MAX_CHARS = 600           # about 150 tokens: Whisper reads at most the last 224 of the prompt
GPU_LOCK = threading.Lock()
_models: dict[str, object] = {}

app = FastAPI(title="mathula-stt")


def _whisper():
    if "whisper" not in _models:
        from faster_whisper import WhisperModel

        _models["whisper"] = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16", download_root=WHISPER_ROOT)
    return _models["whisper"]


def _diarizer():
    if "diarizer" not in _models:
        from pyannote.audio import Pipeline

        pipeline = Pipeline.from_pretrained(DIARIZER, token=HF_TOKEN or None)
        pipeline.to(torch.device("cuda"))
        _models["diarizer"] = pipeline
    return _models["diarizer"]


def _prepare(source: Path, target: Path) -> float:
    """16 kHz mono wav: what both models read.  Returns the length in seconds."""
    done = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(source), "-ac", "1", "-ar", "16000", "-f", "wav", str(target)], capture_output=True, text=True)
    if done.returncode:
        raise HTTPException(status_code=422, detail=f"ffmpeg could not read the audio: {done.stderr[-300:]}")
    import soundfile as sf

    info = sf.info(str(target))
    return info.frames / float(info.samplerate)


def _language(locale: str) -> str:
    return (locale or "en").split("-")[0].lower()


def _whisper_words(wav: Path, locale: str, phrases: list[str]) -> list[dict]:
    import soundfile as sf

    prompt = (", ".join(p for p in phrases if p.strip()))[:PROMPT_MAX_CHARS] or None
    samples, _ = sf.read(str(wav), dtype="float32")          # the 16 kHz mono samples go in directly: faster-whisper's own decoder (PyAV) is not used
    segments, _ = _whisper().transcribe(samples, language=_language(locale), beam_size=5, word_timestamps=True, vad_filter=True, condition_on_previous_text=False,
                                        initial_prompt=prompt, temperature=[0.0, 0.2, 0.4])
    return [{"text": w.word, "start": float(w.start), "end": float(w.end), "probability": float(w.probability)} for segment in segments for w in (segment.words or [])]


def _turns(wav: Path, min_speakers, max_speakers) -> list[dict]:
    import soundfile as sf

    options = {k: v for k, v in (("min_speakers", min_speakers), ("max_speakers", max_speakers)) if v}
    samples, rate = sf.read(str(wav), dtype="float32")          # the waveform goes in directly: reading a file would need torchcodec
    output = _diarizer()({"waveform": torch.from_numpy(samples).unsqueeze(0), "sample_rate": int(rate)}, **options)
    annotation = getattr(output, "exclusive_speaker_diarization", None) or getattr(output, "speaker_diarization", output)          # pyannote 4 returns a result object; 3 an Annotation
    return [{"speaker": str(label), "start": float(turn.start), "end": float(turn.end)} for turn, _, label in annotation.itertracks(yield_label=True)]


@app.get("/health")
def health():
    return {"ok": True, "cuda": torch.cuda.is_available(), "loaded": sorted(_models)}


@app.post("/v1/transcribe")
def transcribe(audio: UploadFile = File(...), params: str = Form("{}")):
    try:
        options = json.loads(params or "{}")
    except ValueError:
        raise HTTPException(status_code=400, detail="params is not JSON")
    locale = str(options.get("locale") or "en-ZA")
    if _language(locale) != "en":
        raise HTTPException(status_code=400, detail=f"only English is transcribed here (asked for {locale})")
    diarize = bool(options.get("diarize", True))
    timings: dict[str, float] = {}
    with tempfile.TemporaryDirectory() as folder, GPU_LOCK:
        source = Path(folder) / (Path(audio.filename or "audio").name or "audio")
        source.write_bytes(audio.file.read())
        wav = Path(folder) / "audio16k.wav"
        started = time.perf_counter()
        duration = _prepare(source, wav)
        timings["prepare"] = round(time.perf_counter() - started, 2)
        started = time.perf_counter()
        words = _whisper_words(wav, locale, list(options.get("phrases") or []))
        timings["words"] = round(time.perf_counter() - started, 2)
        turns: list[dict] = []
        if diarize:
            started = time.perf_counter()
            turns = _turns(wav, options.get("min_speakers"), options.get("max_speakers"))
            timings["diarize"] = round(time.perf_counter() - started, 2)
    raw = stt_merge.to_azure_raw(words, turns, duration_s=duration, locale=locale)
    raw["engine"] = {"model": WHISPER_MODEL, "diarizer": DIARIZER if diarize else None, "words": len(words), "turns": len(turns),
                     "speakers": len({t["speaker"] for t in turns}), "audio_seconds": round(duration, 2), "seconds": timings}
    raw["diarization_turns"] = turns
    return raw


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
