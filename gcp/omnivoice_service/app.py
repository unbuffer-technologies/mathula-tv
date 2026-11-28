"""Speak the Zulu units of a finished Mathula dub (job ada0e557, prod-110) with the fine-tuned OmniVoice LoRA.

Tab 1 "Try units": a few units, side options (base vs tuned, cloning the source speaker, exact pauses, forced window length).
Tab 2 "Whole dub": every unit placed on the job's timeline -> ONE Zulu audio file as long as the final dub, to compare with it.

The adapter (mzansilives/omnivoice-zulu-checkpoints, checkpoint-8000) is bundled in ./adapter, so no token is needed.
Sentences are spoken one at a time (a unit can be two minutes long), and the pauses between them come from our ~pauseN markers or a short default.
"""
import io
import json
import os
import re
import threading
import time
import zipfile

import gradio as gr
import numpy as np
import soundfile as sf
import torch

try:                                                      # Cloud Run keeps a real GPU attached for the life of the instance: there is no ZeroGPU quota to ask for
    import spaces
except ImportError:
    class spaces:                                         # noqa: N801 - the same decorator the Hugging Face Space uses, doing nothing here
        @staticmethod
        def GPU(*args, **kwargs):
            if args and callable(args[0]):
                return args[0]
            return lambda function: function

import omnivoice_duration as od                          # the free length estimate (copied from src/mathula_tv/omnivoice_duration.py)
import word_align                                        # where each word of a span falls in its audio (MMS forced alignment)

HERE = os.path.dirname(os.path.abspath(__file__))
FIT_TOL_S = 0.25                                          # an exact-fit unit that misses its window by more than this is generated once more
DATA = json.load(open(os.path.join(HERE, "units.json"), encoding="utf-8"))
UNITS = DATA["units"]
TOTAL_S = DATA.get("total_s", 993.4)                      # the final dub's length, so the new track lines up with it
REFS = json.load(open(os.path.join(HERE, "refs", "refs.json"), encoding="utf-8"))          # one English clip per source speaker, cut from the job's own audio
UNIT_REFS = json.load(open(os.path.join(HERE, "refs", "unit", "refs_unit.json"), encoding="utf-8"))    # one clip per unit; a unit too short for its own borrows the closest-pitched one
DEFAULT = ["u02", "u03", "u11", "u14", "u15"]            # the five short units: quick to hear
PAUSE = re.compile(r"\s*~pause(\d+)\s*")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-ZÀ-ɏ\"'(])")
MAX_SPAN_CHARS = 280                                      # about 15-20 s of speech: the most one generation is asked to say before a short break
SPAN_GAP_MS = 150                                         # silence between two spans that were cut only because of their length
JOIN_GAP_MS = 300                                         # least silence kept between two units when one runs into the next
GPU_LOCK = threading.Lock()                               # one GPU, one generation at a time: the REST API and the test UI both go through it


def _load(use_adapter: bool):
    """Base OmniVoice (k2-fsa/OmniVoice), with the Zulu LoRA merged in memory when use_adapter is true (so the two are separate instances)."""
    from omnivoice import OmniVoice
    from peft import PeftModel

    # ZeroGPU only attaches a real GPU while a @spaces.GPU function runs; at start-up cuda is emulated, so peft must read the adapter on the CPU.
    model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cpu", dtype=torch.float32)
    if use_adapter:
        adapter = os.path.join(HERE, "adapter")                                                              # checkpoint-8000 ships inside this Space: no token needed
        model = PeftModel.from_pretrained(model, adapter, torch_device="cpu").merge_and_unload()           # the same merge omnivoice.utils.lora.load_lora_adapter does
    model = model.to(torch.float16)
    model.audio_tokenizer.to(torch.float32)                  # the audio decoder stays in float32 (omnivoice-infer never casts it); in fp16 it produced NaN audio
    return model.to("cuda")                                  # ZeroGPU moves it to the real GPU for each call


MODELS = {"tuned": _load(True), "base": _load(False)}
try:                                                      # the aligner is an extra: a service that cannot load it still speaks (callers then fall back to measuring the audio)
    ALIGNER = word_align.load("cuda")
    ALIGNER_ERROR = None
except Exception as error:                                # noqa: BLE001
    ALIGNER, ALIGNER_ERROR = None, f"{type(error).__name__}: {str(error)[:200]}"


def _by_id():
    return {u["id"]: u for u in UNITS}


def _gpu_seconds_units(ids, *args, **kwargs):
    """GPU time for the units tab: the models are loaded, so only speech is generated; capped low for ZeroGPU quotas."""
    return int(min(90, 20 + 0.5 * sum(_by_id()[i]["window_s"] for i in ids)))


def _gpu_seconds_one(uid, *args, **kwargs):
    return int(min(120, 20 + 0.5 * _by_id()[uid]["window_s"]))


def _pieces(unit, keep_pauses):
    """(sentences, gaps_ms): the unit's text cut into sentences, and the silence after each one (our ~pause markers when kept, else a short default)."""
    if keep_pauses and "~pause" in unit["text_pauses"]:
        parts = PAUSE.split(unit["text_pauses"])                           # text, ms, text, ms, ...
        texts, marks = parts[0::2], [int(g) for g in parts[1::2]]
    else:
        texts, marks = [unit["text"]], []
    # Sentences are only cut apart where the text has a real ~pause, or when a span would grow past MAX_SPAN_CHARS: sentence by sentence generation made every
    # sentence its own island (identical 0.5 s gaps, a new start each time).  Inside a span the model speaks the sentences as one run.
    spans, gaps = [], []
    for index, text in enumerate(texts):
        current = ""
        for sentence in [s.strip() for s in SENTENCE_END.split(text.strip()) if s.strip()]:
            if current and len(current) + 1 + len(sentence) > MAX_SPAN_CHARS:
                spans.append(current)
                gaps.append(SPAN_GAP_MS)
                current = sentence
            else:
                current = f"{current} {sentence}".strip()
        if current:
            spans.append(current)
            gaps.append(marks[index] if index < len(marks) else 0)
    return spans, gaps[:-1] if gaps else []


def _syllables(text: str) -> int:
    return max(1, len(re.findall(r"[aeiouAEIOU]", text)))                 # Zulu: one syllable per vowel


def _say(model, unit, *, ref_audio, ref_text, match_window, keep_pauses, speed=1.0, window_s=None):
    """One unit's audio, span by span.  An exact fit (match_window) asks each span for a length: what is left of the window after the silences and the 200 ms of padding
    the model adds to every generation, shared between the spans by their syllable count; otherwise the model's own pace is used, scaled by `speed`."""
    rate = model.sampling_rate
    spans, gaps = _pieces(unit, keep_pauses)
    speech_s = max(0.5, (window_s or unit["window_s"]) - sum(gaps) / 1000.0 - od.EDGE_PAD_S * len(spans))
    total_syllables = sum(_syllables(s) for s in spans) or 1
    chunks, starts = [], []
    elapsed = 0
    for index, span in enumerate(spans):
        wanted = speech_s * _syllables(span) / total_syllables if match_window else None
        with GPU_LOCK:
            audio = np.asarray(model.generate(text=span, language="zu", ref_audio=ref_audio, ref_text=ref_text if ref_audio else None,
                                              duration=wanted, speed=1.0 if match_window else speed)[0]).squeeze()
        starts.append(elapsed / rate)
        chunks.append(audio)
        elapsed += len(audio)
        if index < len(gaps):
            silence = np.zeros(int(gaps[index] / 1000.0 * rate), dtype=audio.dtype)
            chunks.append(silence)
            elapsed += len(silence)
    return np.concatenate(chunks), spans, starts


def _reference(unit, clone, ref_audio, ref_text):
    if ref_audio:                                                          # an uploaded clip wins over the source speaker's clip
        return ref_audio, ref_text or None
    if clone and unit["id"] in UNIT_REFS:                                  # the unit's own English audio: a diarization label can mix two people, the unit cannot
        return os.path.join(HERE, "refs", "unit", f"{unit['id']}.wav"), UNIT_REFS[unit["id"]]["text"]
    if clone and unit["speaker"] in REFS:
        return os.path.join(HERE, "refs", f"{unit['speaker']}.wav"), REFS[unit["speaker"]]
    return None, None


@spaces.GPU(duration=_gpu_seconds_units)
def speak(ids, use_adapter, match_window, clone, keep_pauses, ref_audio, ref_text, progress=gr.Progress()):
    model = MODELS["tuned" if use_adapter else "base"]
    by_id = _by_id()
    results, log = [], []
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        for n, uid in enumerate(ids):
            unit = by_id[uid]
            progress((n, len(ids)), desc=uid)
            started = time.time()
            use_ref, use_text = _reference(unit, clone, ref_audio, ref_text)
            try:
                audio, sentences, _ = _say(model, unit, ref_audio=use_ref, ref_text=use_text, match_window=match_window, keep_pauses=keep_pauses)
            except Exception as error:                      # one failing unit must not lose the others
                log.append(f"{uid}: FAILED {type(error).__name__}: {error}")
                continue
            if not np.isfinite(audio).all():
                log.append(f"{uid}: the model returned NaN audio")
                continue
            seconds = len(audio) / model.sampling_rate
            tag = f"{uid}_{'tuned' if use_adapter else 'base'}{'_clone' if use_ref else ''}{'_pauses' if keep_pauses else ''}"
            path = f"/tmp/{tag}.wav"
            sf.write(path, audio, model.sampling_rate)
            bundle.write(path, f"{tag}.wav")
            results.append((path, f"{uid} ({unit['speaker']}): {seconds:.1f}s of a {unit['window_s']}s window"))
            log.append(f"{uid}: {seconds:.2f}s spoken, window {unit['window_s']}s, {len(sentences)} sentences, generated in {time.time() - started:.0f}s\n   {unit['text'][:140]}")
    zip_path = "/tmp/units.zip"
    open(zip_path, "wb").write(buffer.getvalue())
    return results, "\n".join(log), zip_path


@spaces.GPU(duration=_gpu_seconds_one)
def render_unit(uid, use_adapter, match_window, clone, keep_pauses, speed=1.0):
    """One unit for the whole-dub build (its own GPU call, so a two-minute turn stays inside the per-call limit)."""
    model = MODELS["tuned" if use_adapter else "base"]
    unit = _by_id()[uid]
    use_ref, use_text = _reference(unit, clone, None, None)
    started = time.time()
    audio, sentences, starts = _say(model, unit, ref_audio=use_ref, ref_text=use_text, match_window=match_window, keep_pauses=keep_pauses, speed=speed)
    attempts = 1
    if match_window:                                  # one bounded correction: a unit that misses its window by more than FIT_TOL_S is asked again, shifted by the miss
        miss = len(audio) / model.sampling_rate - unit["window_s"]
        if abs(miss) > FIT_TOL_S:
            again, again_sentences, again_starts = _say(model, unit, ref_audio=use_ref, ref_text=use_text, match_window=True, keep_pauses=keep_pauses, window_s=unit["window_s"] - miss)
            attempts = 2
            if abs(len(again) / model.sampling_rate - unit["window_s"]) < abs(miss):
                audio, sentences, starts = again, again_sentences, again_starts
    return audio.astype(np.float32), model.sampling_rate, sentences, starts, time.time() - started, attempts


def build_whole(use_adapter, match_window, clone, keep_pauses, limit, speed=1.0, progress=gr.Progress()):
    """Every unit on the job's timeline, one Zulu track as long as the final dub.  A unit starts at its source start, or just after the previous unit when that one ran long.
    Stops at the first unit that fails or returns NaN audio, so a broken setup cannot spend the day's ZeroGPU quota."""
    units = UNITS[: int(limit)]
    rate, track, rows, log = 24000, None, [], []
    track = np.zeros(int(TOTAL_S * rate), dtype=np.float32)
    free_from = 0.0
    spent = 0.0
    for n, unit in enumerate(units):
        progress((n, len(units)), desc=unit["id"])
        try:
            audio, rate_back, sentences, starts, took, attempts = render_unit(unit["id"], use_adapter, match_window, clone, keep_pauses, speed)
        except Exception as error:
            log.append(f"STOPPED at {unit['id']}: {type(error).__name__}: {error}")
            break
        if not np.isfinite(audio).all() or not len(audio):
            log.append(f"STOPPED at {unit['id']}: the model returned empty or NaN audio")
            break
        spent += took
        start = max(unit["start_ms"] / 1000.0, free_from)
        end = start + len(audio) / rate_back
        if end > len(track) / rate_back:
            track = np.concatenate([track, np.zeros(int((end - len(track) / rate_back) * rate_back) + rate_back, dtype=np.float32)])
        track[int(start * rate_back): int(start * rate_back) + len(audio)] += audio
        free_from = end + JOIN_GAP_MS / 1000.0
        late = start - unit["start_ms"] / 1000.0
        over = end - unit["end_ms"] / 1000.0
        # collision at the SOURCE timeline (no pushing): how far this unit's audio, started where the source speaker started, runs into the next unit's start
        following = UNITS[UNITS.index(unit) + 1] if unit is not UNITS[-1] else None
        collides = (unit["start_ms"] / 1000.0 + len(audio) / rate_back - following["start_ms"] / 1000.0) if following else None
        cross = bool(following and following["speaker"] != unit["speaker"])
        use_ref, use_text = _reference(unit, clone, None, None)
        predicted = None
        if use_ref and use_ref.startswith(HERE) and unit["id"] in UNIT_REFS and not match_window:            # the free estimate, to check it against what was spoken
            ref = UNIT_REFS[unit["id"]]
            spans_, gaps_ = _pieces(unit, keep_pauses)
            predicted = round(od.estimate_seconds(spans_, gaps_, ref["text"], ref["end_s"] - ref["start_s"], speed), 2)
        rows.append({"unit": unit["id"], "speaker": unit["speaker"], "source_start_s": round(unit["start_ms"] / 1000.0, 2), "placed_start_s": round(start, 2),
                     "late_by_s": round(late, 2), "spoken_s": round(len(audio) / rate_back, 2), "window_s": unit["window_s"],
                     "past_window_end_s": round(over, 2), "collides_into_next_s": None if collides is None else round(collides, 2), "next_is_other_speaker": cross,
                     "predicted_s": predicted, "attempts": attempts, "sentences": len(sentences), "gpu_s": round(took, 1),
                     "sentence_starts_s": [round(start + s, 2) for s in starts]})
        flag = "" if collides is None or collides <= 0.02 else (f"  ** COLLIDES {collides:.2f}s into the next unit" + (" (a different speaker)" if cross else ""))
        log.append(f"{unit['id']} ({unit['speaker']}): {len(audio) / rate_back:.1f}s spoken in a {unit['window_s']}s window"
                   f"{'' if predicted is None else ' (free estimate said %.1fs)' % predicted}, {'late by %.1fs, ' % late if late > 0.05 else ''}"
                   f"{'ends %.1fs past its window' % over if over > 0.05 else 'fits'}{flag}; {took:.0f}s of GPU, {attempts} attempt(s), {len(sentences)} spans")
    peak = float(np.abs(track).max()) or 1.0
    track = (track / peak * 0.9).astype(np.float32)                       # one gain for the whole track, so every unit keeps its level against the others
    path = "/tmp/zulu_omnivoice_dub.flac"               # lossless and about half the size: Cloud Run refuses a single HTTP response over 32 MiB, and a 993 s wav is 47 MB
    sf.write(path, track, rate, subtype="PCM_16")
    report = "/tmp/timing_report.json"
    json.dump({"job": DATA.get("job"), "setup": {"lora": use_adapter, "clone": clone, "keep_pauses": keep_pauses, "match_window": match_window}, "units": rows}, open(report, "w"), indent=1)
    log.append(f"\nGPU seconds used by the units: {spent:.0f}. Track: {len(track) / rate:.0f}s long (the final dub is {TOTAL_S:.0f}s).")
    return path, path, report, "\n".join(log)


from fastapi import FastAPI, HTTPException                # noqa: E402 - the REST API for the dubbing pipeline, beside the test UI

import base64                                              # noqa: E402
import hashlib                                             # noqa: E402
import tempfile                                            # noqa: E402

api = FastAPI(title="Mathula OmniVoice")
_REF_FILES: dict[str, str] = {}
MAX_SPANS = 60


def _reference_file(wav_b64: str) -> str:
    """The reference clip the caller sent, written once to a temp file (keyed by its hash) because OmniVoice reads references from a path."""
    raw = base64.b64decode(wav_b64)
    key = hashlib.sha1(raw).hexdigest()
    path = _REF_FILES.get(key)
    if not path or not os.path.exists(path):
        path = os.path.join(tempfile.gettempdir(), f"ref_{key}.wav")
        with open(path, "wb") as handle:
            handle.write(raw)
        _REF_FILES[key] = path
    return path


@api.post("/v1/estimate")
def estimate_rest(body: dict) -> dict:
    """OmniVoice's own numbers for a reference clip, asked the way ``generate`` asks them: {ref_wav_b64, ref_text, lora?, texts?, speed?}.

    The clip goes through ``create_voice_clone_prompt`` (silences removed, end punctuation added, the audio tokenised), so the token count is the one the length formula really divides by.
    Returns {ref_tokens, ref_seconds, ref_text_used, ref_weight, tokens_per_weight, frame_rate}: how many tokens of audio one unit of text weight takes at this clip's pace.  With ``texts``
    it also returns {targets_tokens, targets_s}: the length the model itself would fix for each text with this clip as its reference (``_estimate_target_tokens``, ``speed`` applied)."""
    if not body.get("ref_wav_b64"):
        raise HTTPException(status_code=400, detail="send ref_wav_b64 and ref_text")
    model = MODELS["tuned" if body.get("lora", True) else "base"]
    reference = _reference_file(body["ref_wav_b64"])
    with GPU_LOCK:
        prompt = model.create_voice_clone_prompt(ref_audio=reference, ref_text=(body.get("ref_text") or None))
    tokens = int(prompt.ref_audio_tokens.shape[-1])
    frame_rate = float(model.audio_tokenizer.config.frame_rate)
    weight = float(model.duration_estimator.calculate_total_weight(prompt.ref_text))
    if weight <= 0 or tokens <= 0:
        raise HTTPException(status_code=422, detail="the clip has no text weight or no audio after silence removal")
    answer: dict = {"ref_tokens": tokens, "ref_seconds": round(tokens / frame_rate, 4), "ref_text_used": prompt.ref_text, "ref_weight": weight, "tokens_per_weight": tokens / weight, "frame_rate": frame_rate, "instance": INSTANCE_ID}
    texts = [str(t) for t in (body.get("texts") or [])]
    if texts:
        speed = float(body.get("speed") or 1.0)
        targets = [int(model._estimate_target_tokens(t, prompt.ref_text, tokens, speed=speed)) for t in texts]
        answer["targets_tokens"] = targets
        answer["targets_s"] = [round(t / frame_rate, 3) for t in targets]
    return answer


INSTANCE_ID = os.urandom(4).hex()                         # which copy of the service answered (Cloud Run may run several)


@api.post("/v1/warm")
def warm(body: dict) -> dict:
    """Hold this request for ``hold_s`` seconds (at most 240) without touching the GPU.  With one request per copy (concurrency 1), a caller who sends N of them at once makes Cloud Run
    run N copies, each loaded and kept warm afterwards; the answer says which copy this was, so the caller can count them."""
    hold = max(0.0, min(float(body.get("hold_s") or 0.0), 240.0))
    time.sleep(hold)
    return {"ok": True, "instance": INSTANCE_ID, "held_s": hold, "models": sorted(MODELS)}


@api.get("/health")                                       # not /healthz: Cloud Run's front end reserves that path and answers 404 itself
def health() -> dict:
    return {"ok": True, "instance": INSTANCE_ID, "models": sorted(MODELS), "units": len(UNITS), "aligner": ALIGNER is not None, "aligner_error": ALIGNER_ERROR}


GENERATION_KNOBS = {"num_step", "guidance_scale", "t_shift", "layer_penalty_factor", "position_temperature", "class_temperature"}


@api.post("/v1/synthesize")
def synthesize_rest(body: dict) -> dict:
    """Speak these spans with the Zulu model: {spans, gaps_ms, ref_wav_b64, ref_text, durations_s, speed, lora}.

    ``languages`` (optional, one per span) sets the model's language tag for that span: "zu" by default, "en" to read the text as English, "none" for the language-agnostic mode.
    ``seeds`` (optional, one per span, null = unseeded) sets the random generator before that span, so the same text, reference, settings and seed give the same audio again.
    ``generation`` (optional) sets decoding knobs for every span: num_step (32), guidance_scale (2.0), t_shift (0.1), layer_penalty_factor (5.0), position_temperature (5.0; 0 = greedy), class_temperature (0.0).
    ``durations_s`` pins a span's length (or null to let the model decide); ``gaps_ms`` are the silences between spans; the reference clip sets the voice AND the pace.
    ``align`` (true) also returns ``words``: for each span the start, end and score of every whitespace-separated word of its text in seconds from the start of THAT span's audio (None for a word
    with no letter in it), found by forced alignment of the known text; ``aligned`` says whether it worked.  One request therefore carries every span of a unit, audio and word times together.
    Returns {audio_b64 (16-bit wav), sample_rate, duration_ms, span_starts_s, span_ms, gpu_s, words?, aligned?, align_error?}.  Only one generation runs at a time."""
    spans = [str(s) for s in (body.get("spans") or []) if str(s).strip()]
    if not spans or len(spans) > MAX_SPANS:
        raise HTTPException(status_code=400, detail=f"send between 1 and {MAX_SPANS} non-empty spans")
    gaps = [max(0, int(g)) for g in (body.get("gaps_ms") or [])][: len(spans) - 1]
    gaps += [0] * (len(spans) - 1 - len(gaps))
    durations = list(body.get("durations_s") or [])
    durations += [None] * (len(spans) - len(durations))
    speed = float(body.get("speed") or 1.0)
    languages = [(str(l) if l else None) for l in (body.get("languages") or [])][: len(spans)]          # per span: "zu" (the default), "en", ... or "none" for the model's language-agnostic mode
    languages += ["zu"] * (len(spans) - len(languages))
    model = MODELS["tuned" if body.get("lora", True) else "base"]
    seeds = [(int(x) if x is not None else None) for x in (body.get("seeds") or [])][: len(spans)]          # per span: the random generator is set to this before the span is generated, so the same seed gives the same audio
    seeds += [None] * (len(spans) - len(seeds))
    knobs = {k: (int(v) if k == "num_step" else float(v)) for k, v in (body.get("generation") or {}).items() if k in GENERATION_KNOBS}          # num_step counts steps: a float breaks the schedule (HTTP 500)          # decoding settings, e.g. position_temperature=0 for greedy
    reference = _reference_file(body["ref_wav_b64"]) if body.get("ref_wav_b64") else None
    ref_text = (body.get("ref_text") or None) if reference else None
    started = time.time()
    rate = model.sampling_rate
    want_words = bool(body.get("align")) and ALIGNER is not None
    chunks, starts, lengths, elapsed, words, align_error = [], [], [], 0, [], None
    for index, span in enumerate(spans):
        pinned = float(durations[index]) if durations[index] else None
        with GPU_LOCK:
            if seeds[index] is not None:
                torch.manual_seed(seeds[index])
                torch.cuda.manual_seed_all(seeds[index])
            audio = np.asarray(model.generate(text=span, language=languages[index] or "zu", ref_audio=reference, ref_text=ref_text, duration=pinned, speed=1.0 if pinned else speed, **knobs)[0]).squeeze()
            if want_words and len(audio) and np.isfinite(audio).all():
                try:
                    words.append(ALIGNER(span, audio, rate))
                except Exception as error:                # noqa: BLE001 - the audio is still good; the caller measures it instead
                    words.append(None)
                    align_error = f"span {index}: {type(error).__name__}: {str(error)[:160]}"
        if not len(audio) or not np.isfinite(audio).all():
            raise HTTPException(status_code=500, detail=f"the model returned empty or NaN audio for span {index}")
        starts.append(elapsed / rate)
        lengths.append(int(len(audio) / rate * 1000))
        chunks.append(audio)
        elapsed += len(audio)
        if index < len(gaps):
            chunks.append(np.zeros(int(gaps[index] / 1000.0 * rate), dtype=audio.dtype))
            elapsed += len(chunks[-1])
    track = np.concatenate(chunks)
    buffer = io.BytesIO()
    sf.write(buffer, (np.clip(track, -1.0, 1.0) * 32767).astype(np.int16), rate, format="WAV", subtype="PCM_16")
    answer = {"audio_b64": base64.b64encode(buffer.getvalue()).decode("ascii"), "sample_rate": rate, "duration_ms": int(len(track) / rate * 1000),
              "span_starts_s": [round(s, 3) for s in starts], "span_ms": lengths, "gpu_s": round(time.time() - started, 2)}
    if body.get("align"):
        answer["aligned"] = bool(want_words and words and all(w is not None for w in words))
        if want_words:
            answer["words"] = words
        if align_error or (body.get("align") and ALIGNER is None):
            answer["align_error"] = align_error or ALIGNER_ERROR or "no aligner"
    return answer


with gr.Blocks(title="OmniVoice Zulu test") as demo:
    gr.Markdown("Zulu units of the last complete Mathula job (ada0e557, prod-110), spoken by the fine-tuned OmniVoice LoRA.")
    with gr.Tab("Whole dub"):
        gr.Markdown("One Zulu audio as long as the final dub, every unit at its place on the timeline, to compare with the final dub. "
                    "Stops at the first bad unit. Try 5 units first: the quota is small.")
        with gr.Row():
            w_tuned = gr.Checkbox(True, label="Use the Zulu LoRA")
            w_clone = gr.Checkbox(True, label="Clone each source speaker")
            w_pauses = gr.Checkbox(True, label="Keep our ~pause silences")
            w_window = gr.Checkbox(False, label="Fit each unit exactly to its window (precise; one retry if it misses by over 0.25 s)")
        w_speed = gr.Slider(0.6, 1.2, value=0.85, step=0.05, label="Speed when not pacing to the window (1.0 = the model's own pace, lower is slower)")
        w_limit = gr.Slider(1, len(UNITS), value=len(UNITS), step=1, label="Build the first N units")
        w_go = gr.Button("Build the whole dub", variant="primary")
        w_audio = gr.Audio(label="Zulu track", type="filepath")
        w_files = gr.File(label="Download: track (wav) and timing report", file_count="multiple")
        w_log = gr.Textbox(label="Log", lines=14)

        def run_whole(a, b, c, d, e, f, progress=gr.Progress()):
            path, same, report, log = build_whole(a, b, c, d, e, f, progress)
            return path, [same, report], log

        w_go.click(run_whole, [w_tuned, w_window, w_clone, w_pauses, w_limit, w_speed], [w_audio, w_files, w_log])

    with gr.Tab("Try units"):
        ids = gr.CheckboxGroup([(f"{u['id']} · {u['speaker']} · {u['window_s']}s · {u['text'][:50]}", u["id"]) for u in UNITS], value=DEFAULT, label="Units")
        with gr.Row():
            tuned = gr.Checkbox(True, label="Use the Zulu LoRA (untick for base OmniVoice, to compare)")
            window = gr.Checkbox(False, label="Force each unit to its source window length (duration=)")
        with gr.Row():
            clone = gr.Checkbox(False, label="Clone the source speaker (their own English clip from the job)")
            pauses = gr.Checkbox(False, label="Keep our ~pause markers as exact silences (only units that have them)")
        with gr.Accordion("Optional: your own reference voice (overrides the source speaker's)", open=False):
            ref = gr.Audio(type="filepath", label="Reference clip (a few seconds)")
            ref_t = gr.Textbox(label="What the reference clip says (leave empty to let the model transcribe it)")
        go = gr.Button("Speak", variant="primary")
        out_files = gr.File(label="All audio (zip)")
        log = gr.Textbox(label="Log", lines=10)
        players = gr.State()

        @gr.render(inputs=players)
        def show(items):
            for path, label in items or []:
                gr.Audio(path, label=label)

        go.click(speak, [ids, tuned, window, clone, pauses, ref, ref_t], [players, log, out_files])

demo.queue(default_concurrency_limit=1)
app = gr.mount_gradio_app(api, demo, path="/")            # the REST API under /v1 and /healthz, the test UI at /

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")), timeout_keep_alive=620)
