"""Work out what a speaker really said where they switch language and the English transcript turns it into nonsense.

At a rally the speaker moves between English, isiZulu and sometimes Sepedi, siSwati, Xitsonga or Afrikaans, depending on the place.  The
English recogniser writes the other languages as nonsense English ("Siyipatala through tax. Mauteng alai shop"), and the meaning is lost
before the agent sees it.  No recogniser here reads Sepedi, so this does not try to transcribe: it gathers independent evidence about the
SAME audio and reconstructs the words, one stretch at a time, in time order (a rolling window):

  1. evidence: the English, isiZulu and Afrikaans readings, and the phones a language-independent model hears (sounds, not words);
  2. context that rolls forward: the stretches already resolved just before, earlier sentences of the speech about the same things, the
     paragraph it sits in;
  3. the model proposes a few genuinely different candidate readings;
  4. each candidate is SPOKEN by the isiZulu voice and the phones it produces are compared with the phones of the original audio -- a
     candidate that fits the argument but sounds nothing like the audio loses;
  5. the model chooses, and the stretch counts as resolved only if its choice also sounds best, clearly better than every candidate that
     means something different; otherwise it is uncertain and goes on a short list for a person to listen to.

The result goes to the agent as evidence with its status.  It never rewrites the transcript, and only a resolved reading of missed speech is
added to the dub.
"""
from __future__ import annotations

import json
import re
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

SCHEMA_VERSION = "mathula-code-switch-evidence-v1"
EVIDENCE_VERSION = 6              # bump when what the model is given or asked changes: cached evidence is then rebuilt
FILE = "code_switch_evidence.json"
MIN_DISAGREEMENT = 0.5            # how differently the English and isiZulu recognisers read a stretch (0 = the same words, 1 = nothing shared)
MAX_STRETCHES = 15
MAX_STRETCH_SECONDS = 14.0
PHONE_WORD_GAP_SECONDS = 0.12     # phones separated by more than this are grouped as separate words
MAX_CANDIDATES = 4
MIN_CLOSENESS_MARGIN = 0.04       # a resolved reading must sound this much closer to the audio than any rival that means something else
CARRIED_FORWARD = 6               # how many resolved stretches roll forward as context
MEMORY_SENTENCES = 12
LANGUAGES = ("english", "isizulu", "sepedi", "sesotho", "setswana", "siswati", "xitsonga", "afrikaans", "unclear")

PROPOSE_PROMPT = (
    "You reconstruct what a South African politician really said in a stretch where machine transcripts disagree. At a rally he moves "
    "between English, isiZulu and, depending on the place, Sepedi (Northern Sotho), Sesotho, Setswana, siSwati, Xitsonga or Afrikaans. You get "
    "recogniser readings of the same audio (English, isiZulu, Afrikaans, and YouTube's own automatic captions, which are often tagged isiZulu: "
    "the English one turns other languages into nonsense, the isiZulu ones read isiZulu well and other languages badly, YouTube's can be right "
    "on isiZulu where the others are wrong and wrong on English) and the raw phones a language-independent phone model heard (IPA, grouped "
    "where he pauses; noisy). You also get what he said just before and after, the paragraph it sits in, the stretches already resolved just "
    "before it, and earlier sentences of the speech about the same things. The argument is strong evidence: "
    "a speaker builds patterns (the grant for the old, then the grant for the children; a figure, then the next figure), and every group of "
    "people or thing he has already named is a possible subject here. Propose 2 to 4 GENUINELY DIFFERENT candidate readings (different words "
    "or a different meaning, not spelling variants of one). For each give the words tagged by language, the spelling that would make a "
    "isiZulu speaker say it the way he did (it will be spoken aloud and compared with the phones), the English meaning, and why it fits. "
    "Names of people, places and organisations are names, not another language. The English meaning must say only what the words of THIS stretch say: do not add what the next stretch says, "
    "and do not fill in a subject or figure he did not say here. Never invent: include a candidate only if the sounds could plausibly carry it."
)

CHOOSE_PROMPT = (
    "You choose between candidate readings of a stretch of a South African politician's speech. Each candidate was spoken aloud by a isiZulu "
    "voice and its phones compared with the phones of the original audio: 'closeness' is how near its sounds are to the audio (higher is "
    "closer; differences of a few hundredths are small). Weigh the sounds first, then how well each continues the argument. Pick the candidate "
    "you believe he said, or -1 if none fits both the sounds and the argument. Say how sure you are: 'high' only when the choice is clear on "
    "both counts."
)


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z]+", str(text).lower())


def disagreement(first: str, second: str) -> float:
    """0 when two readings share all their words in order, 1 when they share none; 0 if either is empty."""
    a, b = _words(first), _words(second)
    return 1.0 - SequenceMatcher(None, a, b).ratio() if a and b else 0.0


def select_stretches(rows: Sequence[Mapping[str, Any]], *, floor: float = MIN_DISAGREEMENT, cap: int = MAX_STRETCHES) -> list[dict[str, Any]]:
    """The stretches the two recognisers read most differently (at least ``floor``), the worst first, at most ``cap``, in time order."""
    scored = [(disagreement(r["english"], r["isizulu"]), r) for r in rows]      # YouTube's captions are evidence, not a trigger: its isiZulu model garbles plain English
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


def neighbours(phrases: Sequence[Mapping[str, Any]], start_s: float, end_s: float, count: int = 2) -> tuple[str, str]:
    """What the first-pass transcript says in the ``count`` phrases just before and just after a stretch."""
    before = [str(p.get("text") or "") for p in phrases if float(p["end"]) <= start_s + 0.05][-count:]
    after = [str(p.get("text") or "") for p in phrases if float(p["start"]) >= end_s - 0.05][:count]
    return " ".join(before), " ".join(after)


def segment_for(segments: Sequence[Mapping[str, Any]], start_s: float, end_s: float) -> str | None:
    """The transcript segment a stretch mostly falls in."""
    best, best_overlap = None, 0.0
    for segment in segments:
        overlap = min(end_s, int(segment["end_ms"]) / 1000) - max(start_s, int(segment["start_ms"]) / 1000)
        if overlap > best_overlap:
            best, best_overlap = str(segment["segment_id"]), overlap
    return best


_STOP = {"that", "this", "with", "from", "have", "will", "your", "they", "them", "what", "were", "been", "there", "their", "which", "would",
         "about", "because", "going", "where", "when", "these", "those", "into", "than", "then", "also", "just", "more", "some", "other"}


def _content(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{4,}", str(text).lower()) if w not in _STOP}


def topic_memory(segments: Sequence[Mapping[str, Any]], before_s: float, probe: str, limit: int = MEMORY_SENTENCES, skip_segment: str | None = None) -> list[str]:
    """Earlier sentences of the speech that share content words with the stretch's surroundings: what he has already said about the same things."""
    wanted = _content(probe)
    scored = []
    for segment in segments:
        start = int(segment["start_ms"]) / 1000
        if start >= before_s or str(segment.get("segment_id")) == skip_segment:        # a stretch's own paragraph is given to the model whole
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", str(segment.get("restored_text") or segment.get("source_text") or "")):
            overlap = len(wanted & _content(sentence))
            if overlap:
                scored.append((overlap, start, f"[{start:.0f}s] {sentence.strip()}"))
    best = sorted(scored, key=lambda s: (-s[0], -s[1]))[:limit]
    return [text for _, _, text in sorted(best, key=lambda s: s[1])]


def speakable(spelling: str) -> str:
    """The spelling as plain words for the voice: sentence punctuation next to the end-of-text bookmark makes the voice drop the bookmark."""
    return re.sub(r"\s+", " ", re.sub(r"[?!.,;:…—–]+", " ", str(spelling))).strip()


def _propose_schema() -> dict[str, Any]:
    word = {"type": "object", "additionalProperties": False, "required": ["text", "language"],
            "properties": {"text": {"type": "string"}, "language": {"type": "string", "enum": list(LANGUAGES)}}}
    candidate = {"type": "object", "additionalProperties": False, "required": ["words", "spelling_for_sounds", "english_meaning", "fits_because"],
                 "properties": {"words": {"type": "array", "items": word}, "spelling_for_sounds": {"type": "string"},
                                "english_meaning": {"type": "string"}, "fits_because": {"type": "string"}}}
    return {"type": "object", "additionalProperties": False, "required": ["candidates"],
            "properties": {"candidates": {"type": "array", "minItems": 1, "maxItems": MAX_CANDIDATES, "items": candidate}}}


def _choose_schema(count: int) -> dict[str, Any]:
    return {"type": "object", "additionalProperties": False, "required": ["chosen", "confidence", "reason"],
            "properties": {"chosen": {"type": "integer", "enum": list(range(-1, count))},
                           "confidence": {"type": "string", "enum": ["high", "medium", "low"]}, "reason": {"type": "string"}}}


def _same_meaning(a: str, b: str) -> bool:
    return set(_words(a)) == set(_words(b))


def is_resolved(candidates: Sequence[Mapping[str, Any]], chosen: int, confidence: str, checked: bool = False) -> bool:
    """Resolved only if the model is sure AND the choice sounds best, clearly better than every rival that means something different."""
    if chosen < 0 or confidence != "high":
        return False
    pick = candidates[chosen]
    closeness = [c.get("closeness") for c in candidates]
    if any(value is None for value in closeness):
        return not checked                       # a check was possible but did not run for every candidate: not resolved; no check at all: the model's word is all there is
    rivals = [c["closeness"] for c in candidates if c is not pick and not _same_meaning(c["english_meaning"], pick["english_meaning"])]
    return pick["closeness"] >= max(closeness) - 1e-9 and all(pick["closeness"] - r >= MIN_CLOSENESS_MARGIN for r in rivals)


def solve_stretch(
    provider: Any, stretch: Mapping[str, Any], *, rally: str, resolved: Sequence[Mapping[str, Any]], memory: Sequence[str],
    source_phones: Sequence[str], speak: Callable[[str], list[str]] | None,
) -> dict[str, Any]:
    """Propose candidates, test each against the audio's phones, choose.  Two model calls and a few spoken candidates."""
    from .pronunciation_lab import align

    evidence = {
        "start_s": stretch["start_s"], "english_recogniser": stretch["english"], "isizulu_recogniser": stretch["isizulu"],
        "afrikaans_recogniser": stretch["afrikaans"], "youtube_caption_reading": stretch.get("youtube", ""), "phones_grouped_by_pauses": stretch["phones"],
        "said_just_before": stretch.get("before", ""), "said_just_after": stretch.get("after", ""), "paragraph_it_sits_in": stretch.get("paragraph", "")}
    carried = [f"[{r['start_s']:.0f}s] {' '.join(p['text'] for p in r['reconstruction'])} = {r['english_meaning']} ({r['status']})" for r in resolved[-CARRIED_FORWARD:]]
    context = {"rally_story": rally, "already_resolved_just_before": carried, "earlier_sentences_about_the_same_things": list(memory)}
    proposed = provider.complete_json(
        operation="native_code_switch_candidates", system_prompt=PROPOSE_PROMPT, payload={"stretch": evidence, **context},
        schema=_propose_schema(), max_output_tokens=2500).data.get("candidates") or []
    candidates = []
    for candidate in proposed[:MAX_CANDIDATES]:
        closeness, untested = None, ""
        if speak is None:
            untested = "no voice check available"
        elif not source_phones:
            untested = "the audio of this stretch gave no phones"
        else:
            for attempt in (1, 2):                 # one retry: a throttled or dropped call is the usual reason
                try:
                    closeness, untested = align(list(source_phones), speak(speakable(candidate["spelling_for_sounds"])))["closeness"], ""
                    break
                except Exception as exc:
                    untested = f"{type(exc).__name__}: {str(exc)[:80]}"
                    time.sleep(1.5)
        candidates.append({**candidate, "closeness": closeness, **({"untested": untested} if untested else {})})
    if not candidates:
        return {"status": "uncertain", "candidates": [], "chosen": -1, "confidence": "low", "reason": "no candidate"}
    shown = [{"index": i, "english_meaning": c["english_meaning"], "words": " ".join(w["text"] for w in c["words"]), "fits_because": c["fits_because"],
              "closeness": c["closeness"]} for i, c in enumerate(candidates)]
    choice = provider.complete_json(
        operation="native_code_switch_choice", system_prompt=CHOOSE_PROMPT, payload={"stretch": evidence, "candidates": shown, **context},
        schema=_choose_schema(len(candidates)), max_output_tokens=800).data
    chosen, confidence = int(choice["chosen"]), str(choice["confidence"])
    return {"status": "resolved" if is_resolved(candidates, chosen, confidence, checked=speak is not None) else "uncertain", "candidates": candidates, "chosen": chosen,
            "confidence": confidence, "reason": str(choice.get("reason") or "")}


def build(
    job_root: Path, provider: Any, *, zulu_reading: Callable[[float, float], str], afrikaans_reading: Callable[[float, float], str],
    phones: Callable[[float, float], list[tuple[str, float, float]]], speak: Callable[[str], list[str]] | None = None,
    youtube_reading: Callable[[float, float], str] | None = None, progress: Callable[[str], None] = print,
) -> Path | None:
    """Write ``native_dub/code_switch_evidence.json`` for a job.  The readers take (start_s, end_s) of the job's analysis audio;
    ``speak(text)`` returns the phones the isiZulu voice produces for a text."""
    import hashlib
    from concurrent.futures import ThreadPoolExecutor

    job_root = Path(job_root)
    native = job_root / "native_dub"
    pass1_path, analysis = native / "pass1_restored_transcript.json", job_root / "analysis" / "azure_diarization.json"
    if not (pass1_path.is_file() and analysis.is_file()):
        return None
    corrections_path = native / "source_corrections.json"
    unresolved = (json.loads(corrections_path.read_text(encoding="utf-8")).get("unresolved") or []) if corrections_path.is_file() else []
    # the cache key: the transcript, the missed-speech stretches to reconstruct and the version of this method
    sha = hashlib.sha256(pass1_path.read_bytes() + json.dumps([EVIDENCE_VERSION, youtube_reading is not None, [[i.get("gap_start_s"), i.get("gap_end_s")] for i in unresolved]]).encode()).hexdigest()
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
    spans = [(float(p["start"]), min(float(p["end"]), float(p["start"]) + MAX_STRETCH_SECONDS)) for p in phrases]
    with ThreadPoolExecutor(max_workers=4) as pool:
        zulu = list(pool.map(lambda span: zulu_reading(*span), spans))
    rows = [{"id": index, "start_s": round(start, 1), "end_s": round(end, 1), "english": str(phrases[index].get("text") or ""), "isizulu": zulu[index],
             "youtube": youtube_reading(start, end) if youtube_reading else ""} for index, (start, end) in enumerate(spans)]
    chosen = select_stretches(rows)
    progress(f"[code-switch] {len(chosen)} of {len(rows)} stretches are read very differently by the English and isiZulu recognisers")
    # speech the transcript missed that the English check rejected (another language): reconstructed too, whatever the readings say
    for item in unresolved:
        start, end = float(item["gap_start_s"]), min(float(item["gap_end_s"]), float(item["gap_start_s"]) + MAX_STRETCH_SECONDS)
        chosen.append({"id": len(rows) + len(chosen), "start_s": round(start, 1), "end_s": round(end, 1), "english": str(item.get("heard") or ""),
                       "isizulu": zulu_reading(start, end), "youtube": youtube_reading(start, end) if youtube_reading else "", "disagreement": 1.0, "unresolved": item})
    if unresolved:
        progress(f"[code-switch] {len(unresolved)} stretch(es) of missed speech that were not English are reconstructed as well")
    chosen.sort(key=lambda s: s["start_s"])                       # the rolling window moves forward in time
    texts = {str(s["segment_id"]): str(s.get("restored_text") or s.get("source_text") or "") for s in segments}
    ledger = {}
    ledger_path = native / "context_ledger.json"
    if ledger_path.is_file():
        data = json.loads(ledger_path.read_text(encoding="utf-8"))
        ledger = data.get("context_ledger", data)
    beats = [str(b.get("text") or b.get("summary") or b) if isinstance(b, Mapping) else str(b) for b in ledger.get("story_beats") or []]
    rally = " ".join([str(ledger.get("summary") or ""), *beats])[:1800]
    found: list[dict[str, Any]] = []
    for stretch in chosen:
        stretch["afrikaans"] = afrikaans_reading(stretch["start_s"], stretch["end_s"])
        timed = phones(stretch["start_s"], stretch["end_s"])
        stretch["phones"] = group_phones(timed)
        stretch["before"], stretch["after"] = neighbours(phrases, stretch["start_s"], stretch["end_s"])
        gap = stretch.get("unresolved")
        owner = str(gap["segment_id"]) if gap else segment_for(segments, stretch["start_s"], stretch["end_s"])
        stretch["paragraph"] = texts.get(owner or "", "")[:900]
        probe = " ".join([stretch["english"], stretch["isizulu"], stretch["before"], stretch["after"], stretch["paragraph"]])
        solved = solve_stretch(provider, stretch, rally=rally, resolved=[f for f in found if f["status"] != "none"],
                               memory=topic_memory(segments, stretch["start_s"], probe, skip_segment=owner), source_phones=[p for p, _, _ in timed], speak=speak)
        pick = solved["candidates"][solved["chosen"]] if solved["chosen"] >= 0 and solved["candidates"] else None
        rivals = [c for c in solved["candidates"] if c is not pick]
        found.append({
            "segment_id": owner, "start_s": stretch["start_s"], "end_s": stretch["end_s"],
            "english_reading": stretch["english"], "isizulu_reading": stretch["isizulu"], "youtube_reading": stretch.get("youtube", ""), "disagreement": stretch["disagreement"],
            "reconstruction": pick["words"] if pick else [], "english_meaning": pick["english_meaning"] if pick else "",
            "confidence": solved["confidence"], "status": solved["status"] if pick else "uncertain", "reason": solved["reason"],
            "candidates": [{"english_meaning": c["english_meaning"], "words": " ".join(w["text"] for w in c["words"]), "closeness": c["closeness"],
                            **({"untested": c["untested"]} if c.get("untested") else {})}
                           for c in solved["candidates"]],
            "alternative": rivals[0]["english_meaning"] if rivals and not _same_meaning(rivals[0]["english_meaning"], pick["english_meaning"] if pick else "") else "",
            **({"missed_speech": {"after_word": gap.get("after_word"), "extends_to_ms": gap.get("extends_to_ms")}} if gap else {})})
        progress(f"[code-switch] {stretch['start_s']:.0f}s {found[-1]['status']} ({found[-1]['confidence']}): {found[-1]['english_meaning'][:90]}")
    target.write_text(json.dumps({"schema_version": SCHEMA_VERSION, "pass1_sha256": sha, "stretches_examined": len(rows), "stretches": found},
                                 ensure_ascii=False, indent=1), encoding="utf-8")
    return target


def add_missed_speech_to_corrections(job_root: Path, progress: Callable[[str], None] = print) -> list[dict[str, Any]]:
    """Missed speech in another language that was RESOLVED joins the source corrections, so the dub says it.

    Its English meaning is written into the segment the speech continues and the segment's window grows to cover it (as for English
    speech the first pass skipped).  Uncertain readings stay evidence for the agent only: a wrong sentence put in a politician's mouth is
    worse than a pause.  Safe to call again: a correction is added once."""
    from .source_gaps import insertion

    native = Path(job_root) / "native_dub"
    doc, corrections_path = load(native / FILE), native / "source_corrections.json"
    if not doc or not corrections_path.is_file():
        return []
    file = json.loads(corrections_path.read_text(encoding="utf-8"))
    pass1 = json.loads((native / "pass1_restored_transcript.json").read_text(encoding="utf-8"))
    texts = {str(s["segment_id"]): str(s.get("restored_text") or s.get("source_text") or "") for s in pass1.get("segments") or []}
    existing = {(str(c.get("segment_id")), str(c.get("replacement"))) for c in file.get("corrections") or []}
    added = []
    for stretch in doc.get("stretches") or []:
        gap = stretch.get("missed_speech")
        if not gap or stretch.get("status") != "resolved" or not stretch.get("segment_id"):
            continue
        change = insertion(texts.get(str(stretch["segment_id"]), ""), str(gap.get("after_word") or ""), stretch["english_meaning"])
        if change is None or (str(stretch["segment_id"]), change["replacement"]) in existing:
            continue
        correction = {"segment_id": stretch["segment_id"], **change, "kind": "missing speech (other language)", "sentence": "", "audio_phones": None,
                      "reason": f"{stretch['end_s'] - stretch['start_s']:.1f} s the speaker said in another language, reconstructed from the sounds: "
                                f"{' '.join(p['text'] for p in stretch['reconstruction'])!r}",
                      "heard": stretch["english_meaning"], "extends_to_ms": gap.get("extends_to_ms")}
        file.setdefault("corrections", []).append(correction)
        added.append(correction)
        progress(f"[code-switch] speech in another language at {stretch['start_s']:.0f} s added to the dub: {stretch['english_meaning']!r}")
    if added:
        corrections_path.write_text(json.dumps(file, indent=1, ensure_ascii=False), encoding="utf-8")
    return added


def review_items(doc: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """The stretches a person should listen to: what the recognisers made of them and the readings still in contention."""
    items = []
    for stretch in (doc or {}).get("stretches") or []:
        if stretch.get("status") == "uncertain" and stretch.get("candidates"):
            items.append({"at_s": stretch["start_s"], "to_s": stretch["end_s"], "english_transcript": stretch["english_reading"][:100],
                          "isizulu_recogniser": stretch["isizulu_reading"][:100],
                          "readings": [f"{c['words']}  = {c['english_meaning']}" + (f"  (sounds {c['closeness']:.2f})" if c.get("closeness") is not None else "")
                                       for c in stretch["candidates"]]})
    return items


def run(job_root: Path, provider: Any, *, speech_backend: Any, gop: Any = None, progress: Callable[[str], None] = print) -> Path | None:
    """``build`` with the real recognisers: Azure speech for the isiZulu and Afrikaans readings, the wav2vec2 phone model for the sounds,
    and the isiZulu voice (spoken through the pronunciation lab) to test each candidate against the audio."""
    import os
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

    speak = None
    key, region = os.environ.get("AZURE_SPEECH_KEY"), os.environ.get("AZURE_SPEECH_REGION", "").strip().lower()
    if key and region:
        from .pronunciation_lab import PronunciationLab

        voice = "zu-ZA-ThembaNeural"
        voices_path = job_root / "direct_dub" / "voice_resolution.json"
        if voices_path.is_file():
            from .dub_production import primary_voice

            voice = primary_voice(voices_path)
        lab = PronunciationLab(job_root, voice=voice, key=key, region=region, work_dir=scratch / "lab", gop=gop)
        speak = lab.voice_reading
    from . import youtube_captions

    path = build(job_root, provider, zulu_reading=reading("zu-ZA"), afrikaans_reading=reading("af-ZA"), phones=sounds, speak=speak,
                 youtube_reading=youtube_captions.reading(job_root), progress=progress)
    if path:
        add_missed_speech_to_corrections(job_root, progress)           # every run: the source check rewrites its corrections file each time
    return path


def load(path: Path) -> dict[str, Any] | None:
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return doc if doc.get("schema_version") == SCHEMA_VERSION else None


AGENT_NOTE = (
    "Where the speaker switches language, the English transcript of that stretch is often nonsense. These are reconstructions of what he "
    "really said, made from the sounds in the audio, several recognisers and the argument of the speech. status 'resolved' means the sounds "
    "and the argument agree and it is clearly better than the other readings; 'uncertain' means another reading is still possible (it is "
    "listed): do not rely on it. Use them to understand the MEANING of a garbled stretch; never translate the garbled English itself. Words "
    "tagged isizulu are his own isiZulu: keep them as isiZulu, in plain wording. Words tagged sepedi, sesotho, setswana, siswati or xitsonga are "
    "not understood by most isiZulu viewers: say the meaning in plain isiZulu (a short greeting or slogan the audience knows may stay). When "
    "uncertain, use only what the clear parts of the English support and do not fill the rest in."
)


def for_agent(doc: Mapping[str, Any], segment_ids: Sequence[str] | None = None) -> dict[str, Any] | None:
    wanted = set(segment_ids) if segment_ids is not None else None
    rows = []
    for stretch in doc.get("stretches") or []:
        if wanted is not None and stretch.get("segment_id") not in wanted:
            continue
        if not stretch.get("reconstruction"):
            continue
        row = {"segment_id": stretch.get("segment_id"), "at_s": stretch["start_s"], "english_transcript_has": stretch["english_reading"][:160],
               "he_said": " ".join(f"{p['text']} [{p['language']}]" for p in stretch["reconstruction"]),
               "meaning": stretch["english_meaning"], "status": stretch.get("status", "uncertain")}
        if stretch.get("status") != "resolved" and stretch.get("alternative"):
            row["or_perhaps"] = stretch["alternative"]
        rows.append(row)
    return {"note": AGENT_NOTE, "stretches": rows} if rows else None
