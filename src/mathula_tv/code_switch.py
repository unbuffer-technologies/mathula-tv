"""Work out what a speaker really said where they switch language and the English transcript turns it into nonsense.

At a rally the speaker moves between English, isiZulu and sometimes Sepedi, siSwati, Xitsonga or Afrikaans, and which ones depend on the
place.  The English recogniser writes the other languages as nonsense English ("Siyipatala through tax. Mauteng alai shop"), and the
meaning is lost before the agent sees it.  No recogniser here reads Sepedi, so this does not try to transcribe: it gathers independent
evidence about the SAME audio and lets a model reconstruct the words from it.

For every stretch where the English recogniser and the isiZulu recogniser read the audio very differently it collects
  * the English reading, the isiZulu reading and the Afrikaans reading, and
  * the phones a language-independent phone model hears (sounds, not words),
and the model reconstructs the words, tags each with its language, and gives the English meaning and how sure it is.  The result goes to
the agent as evidence with its confidence; it never rewrites the transcript, and where the sounds do not support a reading the model is
told to say so rather than invent.
"""
from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

SCHEMA_VERSION = "mathula-code-switch-evidence-v1"
FILE = "code_switch_evidence.json"
MIN_DISAGREEMENT = 0.5            # how differently the English and isiZulu recognisers read a stretch (0 = the same words, 1 = nothing shared)
MAX_STRETCHES = 15
MAX_STRETCH_SECONDS = 14.0
PHONE_WORD_GAP_SECONDS = 0.12     # phones separated by more than this are grouped as separate words
LANGUAGES = ("english", "isizulu", "sepedi", "sesotho", "setswana", "siswati", "xitsonga", "afrikaans", "unclear")

SYSTEM_PROMPT = (
    "You reconstruct what a South African politician really said in stretches where machine transcripts disagree. At a rally he moves "
    "between English, isiZulu and, depending on the place, Sepedi (Northern Sotho), Sesotho, Setswana, siSwati, Xitsonga or Afrikaans; the "
    "place of the rally tells you which are likely (a prior only: the sounds decide). For each stretch you get three recogniser readings "
    "(English, isiZulu, Afrikaans) of the same audio and the raw phone sequence a language-independent phone model heard (IPA, grouped "
    "where he pauses; noisy, with errors). The English recogniser turns other languages into nonsense English; the isiZulu recogniser "
    "reads isiZulu well and other languages badly. Treat the phones as the ground truth for SOUNDS and the readings as hypotheses about "
    "words. Reconstruct the most likely words he said, tag each word or phrase with its language, give the English meaning, and say which "
    "phones or readings support it. Names of people, places and organisations are names, not another language. If the sounds do not "
    "support a confident reconstruction, say so with confidence 'low': never invent."
)


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z]+", str(text).lower())


def disagreement(first: str, second: str) -> float:
    """0 when two readings share all their words in order, 1 when they share none; 0 if either is empty."""
    a, b = _words(first), _words(second)
    return 1.0 - SequenceMatcher(None, a, b).ratio() if a and b else 0.0


def select_stretches(rows: Sequence[Mapping[str, Any]], *, floor: float = MIN_DISAGREEMENT, cap: int = MAX_STRETCHES) -> list[dict[str, Any]]:
    """The stretches the two recognisers read most differently (at least ``floor``), the worst first, at most ``cap``, in time order."""
    scored = [(disagreement(r["english"], r["isizulu"]), r) for r in rows]
    worst = sorted((s for s in scored if s[0] >= floor), key=lambda s: -s[0])[:cap]
    return sorted((dict(r, disagreement=round(score, 2)) for score, r in worst), key=lambda r: r["start_s"])


def group_phones(timed: Sequence[tuple[str, float, float]], gap: float = PHONE_WORD_GAP_SECONDS) -> str:
    """Phones joined into word-like groups wherever the speaker pauses, groups separated by ' | '."""
    groups: list[str] = []
    current: list[str] = []
    last_end: float | None = None
    for phone, start, end in timed:
        if last_end is not None and start - last_end > gap and current:
            groups.append("".join(current))
            current = []
        current.append(phone)
        last_end = end
    if current:
        groups.append("".join(current))
    return " | ".join(groups)


def segment_for(segments: Sequence[Mapping[str, Any]], start_s: float, end_s: float) -> str | None:
    """The transcript segment a stretch mostly falls in."""
    best, best_overlap = None, 0.0
    for segment in segments:
        overlap = min(end_s, int(segment["end_ms"]) / 1000) - max(start_s, int(segment["start_ms"]) / 1000)
        if overlap > best_overlap:
            best, best_overlap = str(segment["segment_id"]), overlap
    return best


def _schema(ids: Sequence[int]) -> dict[str, Any]:
    return {"type": "object", "additionalProperties": False, "required": ["stretches"], "properties": {"stretches": {
        "type": "array", "minItems": len(ids), "maxItems": len(ids), "items": {
            "type": "object", "additionalProperties": False,
            "required": ["id", "reconstruction", "english_meaning", "confidence", "evidence"],
            "properties": {
                "id": {"type": "integer", "enum": list(ids)},
                "reconstruction": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["text", "language"],
                                                              "properties": {"text": {"type": "string"}, "language": {"type": "string", "enum": list(LANGUAGES)}}}},
                "english_meaning": {"type": "string"},
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                "evidence": {"type": "string"}}}}}}


def reconstruct(provider: Any, stretches: Sequence[Mapping[str, Any]], context: str) -> list[dict[str, Any]]:
    """One batched model call over all selected stretches."""
    if not stretches:
        return []
    payload = {"rally": context, "stretches": [
        {"id": s["id"], "start_s": s["start_s"], "english_recogniser": s["english"], "isizulu_recogniser": s["isizulu"],
         "afrikaans_recogniser": s["afrikaans"], "phones_grouped_by_pauses": s["phones"]} for s in stretches]}
    response = provider.complete_json(operation="native_code_switch_reconstruction", system_prompt=SYSTEM_PROMPT, payload=payload,
                                      schema=_schema([s["id"] for s in stretches]), max_output_tokens=min(8000, 600 + 450 * len(stretches)))
    return list(response.data.get("stretches") or [])


def build(
    job_root: Path, provider: Any, *, zulu_reading: Callable[[float, float], str], afrikaans_reading: Callable[[float, float], str],
    phones: Callable[[float, float], list[tuple[str, float, float]]], progress: Callable[[str], None] = print,
) -> Path | None:
    """Write ``native_dub/code_switch_evidence.json`` for a job.  The readers take (start_s, end_s) of the job's analysis audio."""
    import hashlib

    job_root = Path(job_root)
    native = job_root / "native_dub"
    pass1_path, analysis = native / "pass1_restored_transcript.json", job_root / "analysis" / "azure_diarization.json"
    if not (pass1_path.is_file() and analysis.is_file()):
        return None
    sha = hashlib.sha256(pass1_path.read_bytes()).hexdigest()
    target = native / FILE
    if target.is_file():
        try:
            if json.loads(target.read_text(encoding="utf-8")).get("pass1_sha256") == sha:
                return target
        except ValueError:
            pass
    pass1 = json.loads(pass1_path.read_text(encoding="utf-8"))
    segments = pass1.get("segments") or []
    phrases = json.loads(analysis.read_text(encoding="utf-8")).get("phrases") or []
    from concurrent.futures import ThreadPoolExecutor

    spans = [(float(p["start"]), min(float(p["end"]), float(p["start"]) + MAX_STRETCH_SECONDS)) for p in phrases]
    with ThreadPoolExecutor(max_workers=4) as pool:
        zulu = list(pool.map(lambda span: zulu_reading(*span), spans))
    rows = [{"id": index, "start_s": round(start, 1), "end_s": round(end, 1), "english": str(phrases[index].get("text") or ""), "isizulu": zulu[index]}
            for index, (start, end) in enumerate(spans)]
    chosen = select_stretches(rows)
    progress(f"[code-switch] {len(chosen)} of {len(rows)} stretches are read very differently by the English and isiZulu recognisers")
    for stretch in chosen:
        stretch["afrikaans"] = afrikaans_reading(stretch["start_s"], stretch["end_s"])
        stretch["phones"] = group_phones(phones(stretch["start_s"], stretch["end_s"]))
    ledger = {}
    ledger_path = native / "context_ledger.json"
    if ledger_path.is_file():
        data = json.loads(ledger_path.read_text(encoding="utf-8"))
        ledger = data.get("context_ledger", data)
    context = str(ledger.get("summary") or "")
    results = {int(r["id"]): r for r in reconstruct(provider, chosen, context)}
    found = []
    for stretch in chosen:
        result = results.get(stretch["id"])
        if not result:
            continue
        found.append({
            "segment_id": segment_for(segments, stretch["start_s"], stretch["end_s"]), "start_s": stretch["start_s"], "end_s": stretch["end_s"],
            "english_reading": stretch["english"], "isizulu_reading": stretch["isizulu"], "disagreement": stretch["disagreement"],
            "reconstruction": result["reconstruction"], "english_meaning": result["english_meaning"], "confidence": result["confidence"],
            "evidence": result["evidence"]})
        progress(f"[code-switch] {stretch['start_s']:.0f}s ({result['confidence']}): {result['english_meaning'][:90]}")
    target.write_text(json.dumps({"schema_version": SCHEMA_VERSION, "pass1_sha256": sha, "stretches_examined": len(rows), "stretches": found},
                                 ensure_ascii=False, indent=1), encoding="utf-8")
    return target


def run(job_root: Path, provider: Any, *, speech_backend: Any, gop: Any = None, progress: Callable[[str], None] = print) -> Path | None:
    """``build`` with the real recognisers: Azure speech for the isiZulu and Afrikaans readings, the wav2vec2 phone model for the sounds."""
    import subprocess
    import tempfile
    import threading

    from .azure_stt import normalize

    job_root = Path(job_root)
    audio = job_root / "audio" / "analysis_mono.wav"
    if speech_backend is None or not audio.is_file():
        progress("[code-switch] no speech recogniser or analysis audio; skipped")
        return None
    scratch = Path(tempfile.mkdtemp(prefix="code_switch_"))
    counter = iter(range(10_000_000))
    lock = threading.Lock()

    def clip(start: float, end: float) -> Path:
        with lock:
            number = next(counter)
        path = scratch / f"c{number}.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", str(audio), "-ac", "1", "-ar", "16000", str(path)],
                       check=True, capture_output=True)
        return path

    def reading(locale: str) -> Callable[[float, float], str]:
        def read(start: float, end: float) -> str:
            try:
                return " ".join(str(w.get("text")) for w in normalize(speech_backend.transcribe(clip(start, end), locale=locale)).get("words") or [])
            except Exception:                  # Azure answers an error for a window with nothing in it: nothing was heard
                return ""
        return read

    if gop is None:
        from .pronunciation_assess import LocalGop

        gop = LocalGop()
    loaded = []

    def sounds(start: float, end: float) -> list[tuple[str, float, float]]:
        if not loaded:
            gop._load()
            loaded.append(True)
        return gop.decode_timed(clip(start, end))

    return build(job_root, provider, zulu_reading=reading("zu-ZA"), afrikaans_reading=reading("af-ZA"), phones=sounds, progress=progress)


def load(path: Path) -> dict[str, Any] | None:
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return doc if doc.get("schema_version") == SCHEMA_VERSION else None


AGENT_NOTE = (
    "Where the speaker switches language, the English transcript of that stretch is often nonsense. These are reconstructions of what he "
    "really said, made from the sounds in the audio and several recognisers, with how sure they are. Use them to understand the MEANING "
    "of a garbled stretch; never translate the garbled English itself. Words tagged isizulu are his own isiZulu: keep them as isiZulu, in "
    "plain wording. Words tagged sepedi, sesotho, setswana, siswati or xitsonga are not understood by most isiZulu viewers: say the meaning in "
    "plain isiZulu (a short greeting or slogan the audience knows may stay). With confidence 'low', use only what the clear parts of the "
    "English support and do not fill the rest in."
)


def for_agent(doc: Mapping[str, Any], segment_ids: Sequence[str] | None = None) -> dict[str, Any] | None:
    wanted = set(segment_ids) if segment_ids is not None else None
    rows = []
    for stretch in doc.get("stretches") or []:
        if wanted is not None and stretch.get("segment_id") not in wanted:
            continue
        rows.append({"segment_id": stretch.get("segment_id"), "at_s": stretch["start_s"], "english_transcript_has": stretch["english_reading"][:160],
                     "he_said": " ".join(f"{p['text']} [{p['language']}]" for p in stretch["reconstruction"]),
                     "meaning": stretch["english_meaning"], "confidence": stretch["confidence"]})
    return {"note": AGENT_NOTE, "stretches": rows} if rows else None
