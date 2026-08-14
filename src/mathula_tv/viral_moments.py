#!/usr/bin/env python3
"""Mathula TV viral-moments module for long Madlanga Commission hearings.

Pipeline
--------
1. Extract mono 16 kHz FLAC from a 1–2 hour hearing video with ffmpeg.
2. Transcribe English with Azure Speech fast transcription (en-ZA, diarization,
   word timestamps, Madlanga-specific phrase boosting).
3. Build plausible 20–90 second moments from timestamped utterances.
4. Pre-rank candidates with deterministic editorial heuristics.
5. Optionally use an Azure OpenAI deployment to score and trim the strongest
   candidates using structured output.
6. De-duplicate overlapping moments and render clean H.264/AAC source clips.
7. Write JSON/JSONL manifests that can be fed into a downstream TikTok pipeline.

This is a first-class Mathula TV package module. It can consume an existing
Mathula TV job, reuse its authoritative English transcript, or invoke the existing
production Azure Speech backend when transcription is missing. Output clips retain
the source framing so the existing TikTok editor remains the next stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import requests


LOG = logging.getLogger("mathula_tv.viral_moments")
SCHEMA_VERSION = "mathula.viral-moments.v2"
AZURE_SPEECH_API_VERSION = "2025-10-15"

# Phrase boosting is deliberately limited to established Commission vocabulary.
# A text file passed through --phrases-file can extend this list per hearing.
DEFAULT_PHRASES = [
    "Madlanga Commission",
    "Justice Mbuyiseli Madlanga",
    "Justice Madlanga",
    "Commissioner Baloyi",
    "Commissioner Khumalo",
    "Advocate Sandile Khumalo SC",
    "Ms Ncube",
    "Advocate Ncube",
    "Advocate Ofentse Motlhasedi",
    "Major General Feroz Khan",
    "Julius Malema",
    "Andrea Johnson",
    "South African Police Service",
    "SAPS",
    "Independent Police Investigative Directorate",
    "IPID",
    "National Prosecuting Authority",
    "NPA",
    "Crime Intelligence",
    "Political Killings Task Team",
    "KwaZulu-Natal",
    "Mkhwanazi",
    "Khumalo",
    "Madlanga",
]

PROCEDURAL_PATTERNS = [
    r"\bthank you (?:chair|chairperson|commissioner|advocate)\b",
    r"\bmay i proceed\b",
    r"\bplease turn to page\b",
    r"\bpage \d+\b",
    r"\bparagraph \d+\b",
    r"\bbundle\b",
    r"\bexhibit\b",
    r"\bwe (?:will|shall) adjourn\b",
    r"\btea break\b",
    r"\blunch break\b",
    r"\bhousekeeping matter\b",
    r"\bis my microphone on\b",
    r"\bcan you hear me\b",
    r"\bplease repeat the question\b",
]

HOOK_PATTERNS = [
    r"\bi (?:never|did not|didn't|cannot|can't|refuse|deny|admit|accept)\b",
    r"\bthat is (?:false|not true|a lie|incorrect|untrue)\b",
    r"\byou are (?:lying|misleading|wrong)\b",
    r"\bwho (?:authorised|ordered|instructed|paid|called)\b",
    r"\bwhy did you\b",
    r"\bhow do you explain\b",
    r"\bwhat happened\b",
    r"\bthe evidence (?:shows|proves|suggests)\b",
    r"\bwe found\b",
    r"\bfor the first time\b",
    r"\bunder oath\b",
    r"\bcontradict(?:s|ed|ion)\b",
    r"\bcover[- ]?up\b",
    r"\binterference\b",
    r"\bcorrupt(?:ion|ed)?\b",
    r"\bthreat(?:ened|s)?\b",
    r"\bkill(?:ed|ing|ings)?\b",
    r"\bbribe(?:d|ry)?\b",
    r"\bsecret\b",
    r"\bdeleted\b",
    r"\bphone records?\b",
    r"\bintelligence report\b",
]

LEGAL_CAVEAT_PATTERNS = [
    r"\balleg(?:e|ed|ation|ations)\b",
    r"\baccording to\b",
    r"\bthe witness said\b",
    r"\bthe evidence leader put it to\b",
    r"\bnot yet tested\b",
]


@dataclass(frozen=True)
class Utterance:
    index: int
    start_ms: int
    end_ms: int
    speaker: str
    text: str
    confidence: float | None = None
    words: list[dict[str, Any]] = field(default_factory=list)

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)


@dataclass
class Candidate:
    candidate_id: str
    utterance_start: int
    utterance_end: int
    start_ms: int
    end_ms: int
    text: str
    speakers: list[str]
    heuristic_score: float
    heuristic_features: dict[str, float]
    ai_score: float | None = None
    title: str = ""
    hook: str = ""
    reason: str = ""
    category: str = "other"
    selected_start_utterance: int | None = None
    selected_end_utterance: int | None = None
    final_score: float = 0.0

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)


@dataclass
class RenderedClip:
    rank: int
    clip_id: str
    file: str
    sidecar_file: str
    score: float
    title: str
    hook: str
    reason: str
    category: str
    source_start_ms: int
    source_end_ms: int
    duration_ms: int
    source_start_timecode: str
    source_end_timecode: str
    speakers: list[str]
    transcript: str
    heuristic_score: float
    ai_score: float | None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def run_command(command: Sequence[str], *, description: str) -> None:
    LOG.info("%s", description)
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"{description} failed (exit {completed.returncode}):\n{stderr}")


def require_binary(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"Required executable '{name}' was not found on PATH")
    return path


def ffprobe_duration_ms(video_path: Path) -> int:
    ffprobe = require_binary("ffprobe")
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr.strip()}")
    try:
        return int(round(float(result.stdout.strip()) * 1000))
    except ValueError as exc:
        raise RuntimeError(f"Could not parse video duration: {result.stdout!r}") from exc


def extract_audio(video_path: Path, audio_path: Path) -> None:
    ffmpeg = require_binary("ffmpeg")
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "flac",
            "-compression_level",
            "8",
            str(audio_path),
        ],
        description="Extracting 16 kHz mono FLAC for Azure Speech",
    )


def normalize_endpoint(endpoint: str) -> str:
    endpoint = endpoint.strip().rstrip("/")
    if not endpoint.startswith(("https://", "http://")):
        endpoint = "https://" + endpoint
    return endpoint


def load_phrases(path: Path | None) -> list[str]:
    phrases = list(DEFAULT_PHRASES)
    if path:
        for line in path.read_text(encoding="utf-8").splitlines():
            value = line.strip()
            if value and not value.startswith("#"):
                phrases.append(value)
    # Stable de-duplication, case-insensitive.
    seen: set[str] = set()
    output: list[str] = []
    for phrase in phrases:
        key = phrase.casefold()
        if key not in seen:
            seen.add(key)
            output.append(phrase)
    return output


def azure_fast_transcribe(
    *,
    audio_path: Path,
    endpoint: str,
    subscription_key: str,
    locale: str,
    max_speakers: int,
    phrases: list[str],
    timeout_seconds: int,
    max_attempts: int = 4,
) -> dict[str, Any]:
    endpoint = normalize_endpoint(endpoint)
    url = (
        f"{endpoint}/speechtotext/transcriptions:transcribe"
        f"?api-version={AZURE_SPEECH_API_VERSION}"
    )
    definition: dict[str, Any] = {
        "locales": [locale],
        "profanityFilterMode": "None",
        "diarization": {"enabled": True, "maxSpeakers": max_speakers},
        "phraseList": {"phrases": phrases, "biasingWeight": 1.5},
    }
    headers = {"Ocp-Apim-Subscription-Key": subscription_key}

    for attempt in range(1, max_attempts + 1):
        LOG.info(
            "Submitting Azure Speech fast transcription (attempt %d/%d, %.1f MiB)",
            attempt,
            max_attempts,
            audio_path.stat().st_size / (1024 * 1024),
        )
        try:
            with audio_path.open("rb") as audio_file:
                response = requests.post(
                    url,
                    headers=headers,
                    files={"audio": (audio_path.name, audio_file, "audio/flac")},
                    data={"definition": json.dumps(definition)},
                    timeout=(30, timeout_seconds),
                )
        except requests.RequestException as exc:
            if attempt == max_attempts:
                raise RuntimeError(f"Azure Speech request failed: {exc}") from exc
            delay = min(60, 3 * (2 ** (attempt - 1)))
            LOG.warning("Azure Speech request error: %s; retrying in %ss", exc, delay)
            time.sleep(delay)
            continue

        if response.status_code < 400:
            try:
                return response.json()
            except ValueError as exc:
                raise RuntimeError("Azure Speech returned a non-JSON success response") from exc

        retryable = response.status_code in {408, 409, 429, 500, 502, 503, 504}
        body = response.text[:4000]
        if not retryable or attempt == max_attempts:
            raise RuntimeError(
                f"Azure Speech failed with HTTP {response.status_code}: {body}"
            )
        retry_after = response.headers.get("Retry-After")
        delay = int(retry_after) if retry_after and retry_after.isdigit() else min(
            90, 5 * (2 ** (attempt - 1))
        )
        LOG.warning("Azure Speech HTTP %s; retrying in %ss", response.status_code, delay)
        time.sleep(delay)

    raise AssertionError("unreachable")


def normalize_transcript(raw: dict[str, Any]) -> list[Utterance]:
    phrases = raw.get("phrases")
    if not isinstance(phrases, list) or not phrases:
        raise RuntimeError("Azure transcript did not contain a non-empty 'phrases' array")

    utterances: list[Utterance] = []
    for index, item in enumerate(phrases):
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        start_ms = int(item.get("offsetMilliseconds") or item.get("offset_ms") or 0)
        duration_ms = int(item.get("durationMilliseconds") or item.get("duration_ms") or 0)
        if duration_ms <= 0:
            words = item.get("words") or []
            if words:
                last = words[-1]
                last_start = int(last.get("offsetMilliseconds") or start_ms)
                last_duration = int(last.get("durationMilliseconds") or 0)
                duration_ms = max(1, last_start + last_duration - start_ms)
        speaker_value = item.get("speaker")
        speaker = f"SPEAKER_{int(speaker_value):02d}" if isinstance(speaker_value, int) else "SPEAKER_UNKNOWN"
        confidence_value = item.get("confidence")
        confidence = float(confidence_value) if isinstance(confidence_value, (int, float)) else None
        utterances.append(
            Utterance(
                index=len(utterances),
                start_ms=start_ms,
                end_ms=start_ms + max(1, duration_ms),
                speaker=speaker,
                text=text,
                confidence=confidence,
                words=list(item.get("words") or []),
            )
        )

    utterances.sort(key=lambda value: (value.start_ms, value.end_ms))
    # Reindex after sorting.
    return [
        Utterance(
            index=i,
            start_ms=u.start_ms,
            end_ms=u.end_ms,
            speaker=u.speaker,
            text=u.text,
            confidence=u.confidence,
            words=u.words,
        )
        for i, u in enumerate(utterances)
    ]


def serialize_utterances(utterances: Sequence[Utterance]) -> list[dict[str, Any]]:
    return [asdict(item) for item in utterances]


def load_normalized_transcript(path: Path) -> list[Utterance]:
    value = json.loads(path.read_text(encoding="utf-8"))
    items = value.get("utterances") if isinstance(value, dict) else value
    if not isinstance(items, list):
        raise RuntimeError("Reusable transcript must be a list or contain an 'utterances' list")
    output: list[Utterance] = []
    for i, item in enumerate(items):
        start_ms = int(item["start_ms"])
        end_ms = int(item["end_ms"])
        output.append(
            Utterance(
                index=i,
                start_ms=start_ms,
                end_ms=end_ms,
                speaker=str(item.get("speaker") or "SPEAKER_UNKNOWN"),
                text=str(item["text"]).strip(),
                confidence=(
                    float(item["confidence"])
                    if item.get("confidence") is not None
                    else None
                ),
                words=list(item.get("words") or []),
            )
        )
    return output


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text, flags=re.UNICODE))


def sentence_like_end(text: str) -> bool:
    return bool(re.search(r"[.!?][\"')\]]?\s*$", text.strip()))


def text_ngram_set(text: str, n: int = 3) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z0-9']+", text.casefold())
    if len(words) < n:
        return {tuple(words)} if words else set()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def jaccard_similarity(a: str, b: str) -> float:
    left = text_ngram_set(a)
    right = text_ngram_set(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def interval_iou(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    overlap = max(0, min(a_end, b_end) - max(a_start, b_start))
    if overlap <= 0:
        return 0.0
    union = max(a_end, b_end) - min(a_start, b_start)
    return overlap / union if union else 0.0


def count_pattern_hits(text: str, patterns: Sequence[str]) -> int:
    lowered = text.casefold()
    return sum(1 for pattern in patterns if re.search(pattern, lowered, flags=re.IGNORECASE))


def score_candidate(
    *,
    utterances: Sequence[Utterance],
    start: int,
    end: int,
    target_ms: int,
) -> tuple[float, dict[str, float]]:
    section = utterances[start : end + 1]
    text = " ".join(item.text for item in section)
    lowered = text.casefold()
    duration_ms = section[-1].end_ms - section[0].start_ms
    words = word_count(text)
    speakers = {item.speaker for item in section}
    speaker_changes = sum(
        1 for left, right in zip(section, section[1:]) if left.speaker != right.speaker
    )
    questions = text.count("?")
    exclamations = text.count("!")
    hook_hits = count_pattern_hits(text, HOOK_PATTERNS)
    procedural_hits = count_pattern_hits(text, PROCEDURAL_PATTERNS)
    legal_caveats = count_pattern_hits(text, LEGAL_CAVEAT_PATTERNS)
    numbers = len(re.findall(r"\b(?:R\s*)?\d[\d,.:%/-]*\b", text))
    quoted_language = len(re.findall(r"[\"“”']", text))
    names = len(re.findall(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})+\b", text))
    confidence_values = [u.confidence for u in section if u.confidence is not None]
    average_confidence = (
        sum(confidence_values) / len(confidence_values) if confidence_values else 0.80
    )

    duration_fit = max(0.0, 1.0 - abs(duration_ms - target_ms) / max(target_ms, 1))
    density = min(1.0, words / max(1.0, duration_ms / 1000.0 * 2.3))
    interaction = min(1.0, (speaker_changes + questions * 1.5) / 8.0)
    specificity = min(1.0, (numbers + names + quoted_language * 0.2) / 8.0)
    hook = min(1.0, hook_hits / 3.0)
    completeness = 1.0 if sentence_like_end(section[-1].text) else 0.55
    standalone = min(1.0, 0.35 + 0.14 * len(speakers) + 0.08 * legal_caveats + 0.2 * completeness)
    procedural_penalty = min(1.0, procedural_hits / 3.0)
    low_confidence_penalty = max(0.0, 0.82 - average_confidence) / 0.25
    repetition_penalty = min(1.0, max(0, lowered.count("thank you") - 1) / 4.0)

    score = 100.0 * (
        0.24 * hook
        + 0.16 * interaction
        + 0.13 * specificity
        + 0.13 * standalone
        + 0.12 * duration_fit
        + 0.10 * density
        + 0.07 * completeness
        + 0.05 * min(1.0, exclamations / 2.0)
        - 0.18 * procedural_penalty
        - 0.08 * min(1.0, low_confidence_penalty)
        - 0.06 * repetition_penalty
    )
    # Keep the deterministic score useful even for calm but substantive testimony.
    score = max(0.0, min(100.0, score + min(10.0, words / 20.0)))
    features = {
        "hook": round(hook, 4),
        "interaction": round(interaction, 4),
        "specificity": round(specificity, 4),
        "standalone": round(standalone, 4),
        "duration_fit": round(duration_fit, 4),
        "speech_density": round(density, 4),
        "completeness": round(completeness, 4),
        "procedural_penalty": round(procedural_penalty, 4),
        "average_stt_confidence": round(average_confidence, 4),
        "word_count": float(words),
        "speaker_changes": float(speaker_changes),
        "question_count": float(questions),
        "hook_pattern_hits": float(hook_hits),
    }
    return round(score, 3), features


def build_candidates(
    utterances: Sequence[Utterance],
    *,
    min_ms: int,
    target_ms: int,
    max_ms: int,
    candidate_limit: int,
) -> list[Candidate]:
    if not utterances:
        return []

    candidates: list[Candidate] = []
    total = len(utterances)
    # Starting every two utterances limits combinatorial growth while retaining
    # enough boundaries for question/answer exchanges.
    for start in range(0, total, 2):
        start_ms = utterances[start].start_ms
        best_ends: list[int] = []
        for end in range(start, total):
            if end > start:
                internal_gap = utterances[end].start_ms - utterances[end - 1].end_ms
                if internal_gap > 8_000:
                    break
            duration = utterances[end].end_ms - start_ms
            if duration > max_ms:
                break
            if duration < min_ms:
                continue
            gap_after = (
                utterances[end + 1].start_ms - utterances[end].end_ms
                if end + 1 < total
                else 999_999
            )
            boundary_bonus = sentence_like_end(utterances[end].text) or gap_after >= 1200
            near_target = abs(duration - target_ms) <= 15_000
            if boundary_bonus or near_target:
                best_ends.append(end)
        # Evaluate at most three endings per start: nearest target, shortest,
        # and longest. This creates useful alternatives without flooding AI.
        best_ends = sorted(
            set(best_ends),
            key=lambda end: abs((utterances[end].end_ms - start_ms) - target_ms),
        )[:3]
        for end in best_ends:
            section = utterances[start : end + 1]
            text = " ".join(item.text for item in section).strip()
            if word_count(text) < 35:
                continue
            heuristic_score, features = score_candidate(
                utterances=utterances,
                start=start,
                end=end,
                target_ms=target_ms,
            )
            digest = hashlib.sha1(
                f"{section[0].start_ms}:{section[-1].end_ms}:{text[:160]}".encode("utf-8")
            ).hexdigest()[:10]
            candidates.append(
                Candidate(
                    candidate_id=f"cand_{digest}",
                    utterance_start=start,
                    utterance_end=end,
                    start_ms=section[0].start_ms,
                    end_ms=section[-1].end_ms,
                    text=text,
                    speakers=sorted({item.speaker for item in section}),
                    heuristic_score=heuristic_score,
                    heuristic_features=features,
                )
            )

    candidates.sort(key=lambda value: value.heuristic_score, reverse=True)
    # Deterministic early de-duplication before sending candidates to AI.
    kept: list[Candidate] = []
    for candidate in candidates:
        duplicate = any(
            (
                interval_iou(candidate.start_ms, candidate.end_ms, item.start_ms, item.end_ms) >= 0.72
                and jaccard_similarity(candidate.text, item.text) >= 0.72
            )
            for item in kept
        )
        if not duplicate:
            kept.append(candidate)
        if len(kept) >= candidate_limit:
            break
    return kept


def format_timecode(milliseconds: int) -> str:
    milliseconds = max(0, int(milliseconds))
    hours, rem = divmod(milliseconds, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def normalize_openai_base_url(endpoint: str) -> str:
    endpoint = normalize_endpoint(endpoint)
    if endpoint.endswith("/openai/v1"):
        return endpoint + "/"
    if endpoint.endswith("/openai/v1/"):
        return endpoint
    return endpoint.rstrip("/") + "/openai/v1/"


def split_batches(items: Sequence[Candidate], batch_size: int) -> Iterable[list[Candidate]]:
    for index in range(0, len(items), batch_size):
        yield list(items[index : index + batch_size])


def ai_rank_candidates(
    candidates: list[Candidate],
    utterances: Sequence[Utterance],
    *,
    endpoint: str,
    api_key: str,
    deployment: str,
    batch_size: int,
    timeout_seconds: int,
) -> None:
    try:
        from openai import OpenAI
        from pydantic import BaseModel, ConfigDict, Field
    except ImportError as exc:
        raise RuntimeError(
            "AI ranking requires 'openai' and 'pydantic'. Install requirements.txt."
        ) from exc

    class CandidateReview(BaseModel):
        model_config = ConfigDict(extra="forbid")
        candidate_id: str
        keep: bool
        score: float = Field(ge=0, le=100)
        title: str
        hook: str
        reason: str
        category: str
        start_utterance: int
        end_utterance: int

    class ReviewBatch(BaseModel):
        model_config = ConfigDict(extra="forbid")
        reviews: list[CandidateReview]

    client = OpenAI(
        api_key=api_key,
        base_url=normalize_openai_base_url(endpoint),
        timeout=timeout_seconds,
        max_retries=2,
    )
    candidate_map = {candidate.candidate_id: candidate for candidate in candidates}

    system_prompt = """You are Mathula TV's senior short-form news editor.
Evaluate candidate excerpts from a South African public commission hearing for clean TikTok source clips.

A strong moment:
- hooks immediately or has a precise utterance where the hook begins;
- contains a revelation, contradiction, admission, denial, forceful challenge, consequential evidence, or unusually clear explanation;
- is understandable without inventing context;
- preserves legal accuracy: allegations remain allegations and testimony is not converted into established fact;
- ends on a complete answer or decisive beat;
- is worth watching as video, not merely useful as a written quote.

Penalize greetings, page references, repetitive procedure, incomplete accusations, low-context fragments, and moments that rely on a title to become interesting.
The start/end utterances must be IDs shown inside that candidate and must define one continuous clip. Prefer 25–75 seconds; use up to 90 seconds only when needed for context. Do not rewrite testimony. Return one review for every candidate ID."""

    for batch_number, batch in enumerate(split_batches(candidates, batch_size), start=1):
        blocks: list[str] = []
        for candidate in batch:
            lines = [
                f"CANDIDATE {candidate.candidate_id}",
                f"Heuristic score: {candidate.heuristic_score:.1f}",
                f"Window: {format_timecode(candidate.start_ms)} --> {format_timecode(candidate.end_ms)}",
            ]
            for item in utterances[candidate.utterance_start : candidate.utterance_end + 1]:
                lines.append(
                    f"u{item.index} | {format_timecode(item.start_ms)} | {item.speaker} | {item.text}"
                )
            blocks.append("\n".join(lines))
        user_prompt = (
            "Review all candidates below. Category should be a compact editorial class such as "
            "revelation, contradiction, confrontation, admission, denial, evidence, explanation, or other.\n\n"
            + "\n\n".join(blocks)
        )
        LOG.info("Azure OpenAI viral ranking batch %d (%d candidates)", batch_number, len(batch))
        try:
            completion = client.beta.chat.completions.parse(
                model=deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=ReviewBatch,
            )
            parsed = completion.choices[0].message.parsed
            if parsed is None:
                refusal = completion.choices[0].message.refusal
                raise RuntimeError(f"Azure OpenAI returned no parsed ranking: {refusal}")
        except Exception as exc:  # SDK exception types vary across versions.
            raise RuntimeError(f"Azure OpenAI ranking batch {batch_number} failed: {exc}") from exc

        received: set[str] = set()
        for review in parsed.reviews:
            candidate = candidate_map.get(review.candidate_id)
            if candidate is None:
                continue
            received.add(review.candidate_id)
            candidate.ai_score = float(review.score) if review.keep else max(0.0, float(review.score) - 20.0)
            candidate.title = review.title.strip()
            candidate.hook = review.hook.strip()
            candidate.reason = review.reason.strip()
            candidate.category = re.sub(r"[^a-z0-9_-]+", "_", review.category.casefold()).strip("_") or "other"
            low = max(candidate.utterance_start, int(review.start_utterance))
            high = min(candidate.utterance_end, int(review.end_utterance))
            if low <= high:
                duration = utterances[high].end_ms - utterances[low].start_ms
                if 15_000 <= duration <= 105_000:
                    candidate.selected_start_utterance = low
                    candidate.selected_end_utterance = high
        missing = [item.candidate_id for item in batch if item.candidate_id not in received]
        if missing:
            LOG.warning("AI omitted %d candidate(s): %s", len(missing), ", ".join(missing[:5]))


def finalize_candidate_boundaries(
    candidate: Candidate,
    utterances: Sequence[Utterance],
    *,
    video_duration_ms: int,
    min_ms: int,
    max_ms: int,
    start_padding_ms: int,
    end_padding_ms: int,
) -> None:
    start_index = (
        candidate.selected_start_utterance
        if candidate.selected_start_utterance is not None
        else candidate.utterance_start
    )
    end_index = (
        candidate.selected_end_utterance
        if candidate.selected_end_utterance is not None
        else candidate.utterance_end
    )
    start_index = max(candidate.utterance_start, min(candidate.utterance_end, start_index))
    end_index = max(start_index, min(candidate.utterance_end, end_index))

    start_ms = max(0, utterances[start_index].start_ms - start_padding_ms)
    end_ms = min(video_duration_ms, utterances[end_index].end_ms + end_padding_ms)

    # Expand an over-aggressive AI trim until it reaches the configured minimum.
    left = start_index
    right = end_index
    while end_ms - start_ms < min_ms and (left > candidate.utterance_start or right < candidate.utterance_end):
        left_gap = (
            utterances[left].start_ms - utterances[left - 1].start_ms
            if left > candidate.utterance_start
            else math.inf
        )
        right_gap = (
            utterances[right + 1].end_ms - utterances[right].end_ms
            if right < candidate.utterance_end
            else math.inf
        )
        if right_gap <= left_gap and right < candidate.utterance_end:
            right += 1
        elif left > candidate.utterance_start:
            left -= 1
        else:
            break
        start_ms = max(0, utterances[left].start_ms - start_padding_ms)
        end_ms = min(video_duration_ms, utterances[right].end_ms + end_padding_ms)

    # If still too long, trim from the less information-dense edge, preserving
    # complete utterance boundaries.
    while end_ms - start_ms > max_ms and left < right:
        left_words = word_count(utterances[left].text)
        right_words = word_count(utterances[right].text)
        if left_words <= right_words:
            left += 1
        else:
            right -= 1
        start_ms = max(0, utterances[left].start_ms - start_padding_ms)
        end_ms = min(video_duration_ms, utterances[right].end_ms + end_padding_ms)

    candidate.utterance_start = left
    candidate.utterance_end = right
    candidate.start_ms = start_ms
    candidate.end_ms = end_ms
    candidate.text = " ".join(item.text for item in utterances[left : right + 1]).strip()
    candidate.speakers = sorted({item.speaker for item in utterances[left : right + 1]})


def select_final_candidates(
    candidates: list[Candidate],
    utterances: Sequence[Utterance],
    *,
    video_duration_ms: int,
    clip_count: int,
    min_ms: int,
    max_ms: int,
    start_padding_ms: int,
    end_padding_ms: int,
) -> list[Candidate]:
    for candidate in candidates:
        ai_component = candidate.ai_score if candidate.ai_score is not None else candidate.heuristic_score
        # The heuristic remains a guardrail against model enthusiasm for weak,
        # low-context snippets.
        candidate.final_score = round(0.72 * ai_component + 0.28 * candidate.heuristic_score, 3)
        if not candidate.title:
            candidate.title = make_fallback_title(candidate.text)
        if not candidate.hook:
            candidate.hook = best_hook_sentence(candidate.text)
        if not candidate.reason:
            candidate.reason = "Selected by deterministic hearing-moment ranking."
        finalize_candidate_boundaries(
            candidate,
            utterances,
            video_duration_ms=video_duration_ms,
            min_ms=min_ms,
            max_ms=max_ms,
            start_padding_ms=start_padding_ms,
            end_padding_ms=end_padding_ms,
        )

    candidates.sort(key=lambda value: value.final_score, reverse=True)
    selected: list[Candidate] = []
    category_counts: Counter[str] = Counter()
    for candidate in candidates:
        if candidate.duration_ms < min_ms * 0.75 or candidate.duration_ms > max_ms * 1.08:
            continue
        too_similar = any(
            (
                interval_iou(candidate.start_ms, candidate.end_ms, item.start_ms, item.end_ms) >= 0.45
            )
            or (
                interval_iou(candidate.start_ms, candidate.end_ms, item.start_ms, item.end_ms) >= 0.20
                and jaccard_similarity(candidate.text, item.text) >= 0.30
            )
            or (
                jaccard_similarity(candidate.text, item.text) >= 0.65
                and abs(candidate.start_ms - item.start_ms) < 180_000
            )
            or (
                slugify(candidate.title) == slugify(item.title)
                and abs(candidate.start_ms - item.start_ms) < 300_000
            )
            for item in selected
        )
        if too_similar:
            continue
        # Encourage a varied output set without forcing weak clips.
        diversity_penalty = max(0, category_counts[candidate.category] - 2) * 5.0
        if candidate.final_score - diversity_penalty < 30.0 and len(selected) >= max(3, clip_count // 2):
            continue
        candidate.final_score = round(candidate.final_score - diversity_penalty, 3)
        selected.append(candidate)
        category_counts[candidate.category] += 1
        if len(selected) >= clip_count:
            break

    selected.sort(key=lambda value: value.final_score, reverse=True)
    return selected


def first_complete_sentence(text: str, max_chars: int = 180) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    match = re.search(r"^(.{20,}?[.!?])(?:\s|$)", text)
    value = match.group(1) if match else text[:max_chars]
    return value[:max_chars].strip()


def best_hook_sentence(text: str, max_chars: int = 180) -> str:
    sentences = [
        value.strip()
        for value in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text).strip())
        if value.strip()
    ]
    if not sentences:
        return first_complete_sentence(text, max_chars=max_chars)

    def sentence_score(sentence: str) -> float:
        hook_hits = count_pattern_hits(sentence, HOOK_PATTERNS)
        procedure = count_pattern_hits(sentence, PROCEDURAL_PATTERNS)
        questions = sentence.count("?")
        specificity = len(re.findall(r"\b(?:R\s*)?\d[\d,.:%/-]*\b", sentence))
        specificity += len(re.findall(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})+\b", sentence))
        return 5.0 * hook_hits + 2.0 * questions + min(3.0, specificity) - 5.0 * procedure

    best = max(sentences, key=sentence_score)
    return best[:max_chars].strip()


def make_fallback_title(text: str) -> str:
    sentence = best_hook_sentence(text, max_chars=95)
    sentence = re.sub(r"^[^A-Za-z0-9]+", "", sentence)
    return sentence.rstrip(".?!") or "Commission hearing moment"


def slugify(value: str, max_length: int = 56) -> str:
    value = value.casefold()
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    return (value[:max_length].rstrip("-") or "commission-moment")


def render_clip(
    *,
    video_path: Path,
    output_path: Path,
    start_ms: int,
    end_ms: int,
    preset: str,
    crf: int,
) -> None:
    ffmpeg = require_binary("ffmpeg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    start_seconds = start_ms / 1000.0
    duration_seconds = max(0.1, (end_ms - start_ms) / 1000.0)
    run_command(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start_seconds:.3f}",
            "-i",
            str(video_path),
            "-t",
            f"{duration_seconds:.3f}",
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-movflags",
            "+faststart",
            "-avoid_negative_ts",
            "make_zero",
            str(output_path),
        ],
        description=f"Rendering {output_path.name}",
    )


def candidate_debug_dict(candidate: Candidate) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "utterance_start": candidate.utterance_start,
        "utterance_end": candidate.utterance_end,
        "start_ms": candidate.start_ms,
        "end_ms": candidate.end_ms,
        "duration_ms": candidate.duration_ms,
        "text": candidate.text,
        "speakers": candidate.speakers,
        "heuristic_score": candidate.heuristic_score,
        "heuristic_features": candidate.heuristic_features,
        "ai_score": candidate.ai_score,
        "final_score": candidate.final_score,
        "title": candidate.title,
        "hook": candidate.hook,
        "reason": candidate.reason,
        "category": candidate.category,
        "selected_start_utterance": candidate.selected_start_utterance,
        "selected_end_utterance": candidate.selected_end_utterance,
    }



@dataclass(frozen=True)
class ViralMomentsOptions:
    """Runtime settings for one viral-moment extraction run."""

    locale: str = "en-ZA"
    max_speakers: int = 12
    clip_count: int = 15
    min_clip_seconds: float = 22.0
    target_clip_seconds: float = 52.0
    max_clip_seconds: float = 90.0
    candidate_limit: int = 60
    ai_batch_size: int = 10
    phrases_file: Path | None = None
    no_ai: bool = False
    dry_run: bool = False
    ffmpeg_preset: str = "veryfast"
    crf: int = 20
    start_padding_ms: int = 350
    end_padding_ms: int = 650
    speech_timeout_seconds: int = 7200
    openai_timeout_seconds: int = 900
    force: bool = False
    downstream_target_language: str = "zu-ZA"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_safe_options(options: ViralMomentsOptions) -> dict[str, Any]:
    value = asdict(options)
    value["phrases_file"] = str(options.phrases_file) if options.phrases_file else None
    return value


def _file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _input_fingerprint(
    video_path: Path,
    transcript_path: Path | None,
    options: ViralMomentsOptions,
) -> tuple[str, dict[str, Any]]:
    payload = {
        "source": _file_identity(video_path),
        "transcript": _file_identity(transcript_path) if transcript_path else None,
        "options": _json_safe_options(options),
        "azure_openai_deployment": (
            os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip()
            or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "").strip()
            or os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "").strip()
            or None
        ),
        "schema_version": SCHEMA_VERSION,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), payload


def _manifest_outputs_exist(manifest: dict[str, Any], output_dir: Path) -> bool:
    clips = manifest.get("clips")
    if not isinstance(clips, list) or not clips:
        return False
    dry_run = bool((manifest.get("settings") or {}).get("dry_run"))
    if dry_run:
        return True
    return all(
        isinstance(item, dict)
        and item.get("file")
        and (output_dir / str(item["file"])).is_file()
        for item in clips
    )


def _seconds_or_milliseconds(value: Any, *, milliseconds_key_present: bool) -> int:
    if value is None:
        return 0
    number = float(value)
    return int(round(number if milliseconds_key_present else number * 1000.0))


def utterances_from_mathula_transcript(value: Any) -> list[Utterance]:
    """Adapt Mathula transcript-v1, this module's transcript, or raw Azure JSON."""

    if isinstance(value, list):
        items = value
        source_kind = "utterances"
    elif isinstance(value, dict) and isinstance(value.get("utterances"), list):
        items = value["utterances"]
        source_kind = "utterances"
    elif isinstance(value, dict) and isinstance(value.get("segments"), list):
        items = value["segments"]
        source_kind = "segments"
    elif isinstance(value, dict) and isinstance(value.get("phrases"), list):
        return normalize_transcript(value)
    else:
        raise RuntimeError(
            "Transcript must contain 'segments', 'utterances', or Azure 'phrases'"
        )

    output: list[Utterance] = []
    transcript_words = value.get("words", []) if isinstance(value, dict) else []
    for item in items:
        if not isinstance(item, dict):
            continue
        text = str(item.get("source_text") or item.get("text") or "").strip()
        if not text:
            continue
        if source_kind == "segments":
            has_start_ms = item.get("start_ms") is not None
            has_end_ms = item.get("end_ms") is not None
            start_ms = _seconds_or_milliseconds(
                item.get("start_ms") if has_start_ms else item.get("start"),
                milliseconds_key_present=has_start_ms,
            )
            end_ms = _seconds_or_milliseconds(
                item.get("end_ms") if has_end_ms else item.get("end"),
                milliseconds_key_present=has_end_ms,
            )
        else:
            start_ms = int(item.get("start_ms") or 0)
            end_ms = int(item.get("end_ms") or 0)
        if end_ms <= start_ms:
            continue
        speaker = str(
            item.get("speaker")
            or item.get("speaker_id")
            or item.get("azure_speaker")
            or "SPEAKER_UNKNOWN"
        )
        confidence_value = item.get("confidence")
        confidence = (
            float(confidence_value)
            if isinstance(confidence_value, (int, float))
            else None
        )
        words = list(item.get("words") or [])
        if not words and transcript_words and source_kind == "segments":
            words = [
                word
                for word in transcript_words
                if isinstance(word, dict)
                and float(word.get("start", 0.0)) * 1000 >= start_ms - 1
                and float(word.get("end", 0.0)) * 1000 <= end_ms + 1
                and str(word.get("speaker") or speaker) == speaker
            ]
        output.append(
            Utterance(
                index=len(output),
                start_ms=start_ms,
                end_ms=end_ms,
                speaker=speaker,
                text=text,
                confidence=confidence,
                words=words,
            )
        )

    output.sort(key=lambda item: (item.start_ms, item.end_ms))
    return [
        Utterance(
            index=index,
            start_ms=item.start_ms,
            end_ms=item.end_ms,
            speaker=item.speaker,
            text=item.text,
            confidence=item.confidence,
            words=item.words,
        )
        for index, item in enumerate(output)
    ]


def _extract_mathula_analysis_audio(video_path: Path, audio_path: Path) -> None:
    """Create the 16 kHz mono PCM WAV expected by Mathula's Azure backend."""

    ffmpeg = require_binary("ffmpeg")
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(audio_path),
        ],
        description="Extracting Mathula Azure-STT analysis WAV",
    )


def _production_speech_backend() -> Any:
    """Construct Mathula TV's configured production SpeechBackend lazily."""

    try:
        from .config import load_settings
        from .cli import create_speech_backend
    except ImportError as exc:
        raise RuntimeError(
            "Could not import Mathula TV's production Azure Speech backend"
        ) from exc
    return create_speech_backend(load_settings())


def _configure_backend(
    backend: Any,
    phrases: Sequence[str],
    options: ViralMomentsOptions,
) -> None:
    """Apply job-local Fast-STT options without replacing Mathula's backend."""

    targets = [backend]
    fast = getattr(backend, "fast", None)
    if fast is not None and fast is not backend:
        targets.append(fast)
    for target in targets:
        for name, value in (
            ("phrase_list", tuple(phrases)),
            ("phrase_biasing_weight", 1.5),
            ("timeout_seconds", options.speech_timeout_seconds),
            ("max_speakers", options.max_speakers),
        ):
            if hasattr(target, name):
                try:
                    setattr(target, name, value)
                except (AttributeError, TypeError):
                    LOG.debug("Azure backend option %s is immutable", name)


def _normalise_mathula_azure_raw(raw: dict[str, Any], raw_reference: str) -> list[Utterance]:
    # Test doubles and future backend adapters may already return Mathula's
    # normalized transcript contract rather than the raw Azure phrase envelope.
    if isinstance(raw.get("segments"), list) or isinstance(raw.get("utterances"), list):
        return utterances_from_mathula_transcript(raw)
    try:
        from .azure_stt import normalize as normalize_mathula_azure
    except ImportError:
        return normalize_transcript(raw)
    normalized = normalize_mathula_azure(raw, raw_reference=raw_reference)
    return utterances_from_mathula_transcript(normalized)


def _validate_options(options: ViralMomentsOptions) -> None:
    if not 2 <= options.max_speakers <= 35:
        raise RuntimeError("max_speakers must be between 2 and 35")
    if options.clip_count < 1:
        raise RuntimeError("clip_count must be at least 1")
    if not (
        5
        <= options.min_clip_seconds
        < options.target_clip_seconds
        <= options.max_clip_seconds
        <= 180
    ):
        raise RuntimeError(
            "Clip durations must satisfy 5 <= min < target <= max <= 180 seconds"
        )
    if options.candidate_limit < options.clip_count:
        raise RuntimeError("candidate_limit must be at least clip_count")
    if not 0 <= options.crf <= 51:
        raise RuntimeError("crf must be between 0 and 51")


def _resolve_job_source(job_root: Path) -> Path:
    status_path = job_root / "status.json"
    if status_path.is_file():
        status = _read_json(status_path)
        for value in (
            status.get("local_source_path"),
            status.get("objects", {}).get("local_source"),
            status.get("objects", {}).get("source_video"),
        ):
            if value and Path(str(value)).is_file():
                return Path(str(value)).resolve()
    input_dir = job_root / "input"
    candidates = sorted(
        path
        for path in input_dir.glob("*")
        if path.is_file() and path.suffix.casefold() in {".mp4", ".mov", ".mkv", ".webm", ".m4v"}
    )
    if len(candidates) == 1:
        return candidates[0].resolve()
    if not candidates:
        raise RuntimeError(f"No source video found for Mathula job: {job_root.name}")
    raise RuntimeError(
        f"Multiple source videos found for job {job_root.name}; status.json must identify the source"
    )


def _default_job_transcript(job_root: Path) -> Path | None:
    for path in (
        job_root / "analysis/transcript_en.json",
        job_root / "analysis/transcript_en_raw.json",
        job_root / "analysis/azure_diarization.json",
        job_root / "analysis/azure_stt.json",
    ):
        if path.is_file():
            return path
    return None


def _absolute_clip_record(
    clip: RenderedClip,
    output_dir: Path,
    target_language: str,
) -> dict[str, Any]:
    value = asdict(clip)
    value["absolute_file"] = str((output_dir / clip.file).resolve())
    value["tiktok_input"] = value["absolute_file"]
    value["pipeline_stage"] = "clean_source_clip"
    value["downstream_target_language"] = target_language
    value["submit_argv"] = [
        "python",
        "-m",
        "mathula_tv.cli",
        "submit",
        value["absolute_file"],
        "--target-language",
        target_language,
    ]
    return value


def _write_downstream_submit_script(
    path: Path,
    clip_records: Sequence[dict[str, Any]],
) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'cd "${MATHULA_TV_REPO_ROOT:-/home/mokgethwa/mathula-tv}"',
        'source .venv/bin/activate',
        "",
    ]
    for record in clip_records:
        argv = record.get("submit_argv") or []
        lines.append(" ".join(shlex.quote(str(value)) for value in argv))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o755)


def run_viral_moments(
    *,
    video_path: Path,
    output_dir: Path,
    options: ViralMomentsOptions | None = None,
    transcript_path: Path | None = None,
    speech_backend: Any | None = None,
    source_job_id: str | None = None,
) -> dict[str, Any]:
    """Transcribe/reuse, rank, render, and return the complete module manifest."""

    options = options or ViralMomentsOptions()
    _validate_options(options)
    video_path = Path(video_path).resolve()
    output_dir = Path(output_dir).resolve()
    if not video_path.is_file():
        raise RuntimeError(f"Input video does not exist: {video_path}")
    if transcript_path is not None and not Path(transcript_path).is_file():
        raise RuntimeError(f"Reusable transcript does not exist: {transcript_path}")

    manifest_path = output_dir / "clips_manifest.json"
    fingerprint, fingerprint_payload = _input_fingerprint(
        video_path,
        Path(transcript_path).resolve() if transcript_path is not None else None,
        options,
    )
    if manifest_path.is_file() and not options.force:
        existing = _read_json(manifest_path)
        if (
            existing.get("input_fingerprint") == fingerprint
            and _manifest_outputs_exist(existing, output_dir)
        ):
            LOG.info("Reusing completed viral-moments manifest: %s", manifest_path)
            return {**existing, "idempotent_reuse": True}
        LOG.info("Viral-moments inputs changed or outputs are missing; rebuilding")

    require_binary("ffmpeg")
    require_binary("ffprobe")
    output_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = output_dir / "tiktok_inputs"
    clips_dir.mkdir(parents=True, exist_ok=True)

    video_duration_ms = ffprobe_duration_ms(video_path)
    LOG.info("Source duration: %s", format_timecode(video_duration_ms))
    normalized_transcript_path = output_dir / "transcript_normalized.json"
    raw_transcript_path = output_dir / "azure_stt_raw.json"
    transcript_source: str
    stt_provider: str | None = None

    if transcript_path is not None:
        transcript_path = Path(transcript_path).resolve()
        LOG.info("Reusing Mathula English transcript: %s", transcript_path)
        utterances = utterances_from_mathula_transcript(_read_json(transcript_path))
        transcript_source = str(transcript_path)
    else:
        phrases = load_phrases(options.phrases_file)
        backend = speech_backend or _production_speech_backend()
        _configure_backend(backend, phrases, options)
        with tempfile.TemporaryDirectory(prefix="mathula-viral-stt-") as temp_dir:
            audio_path = Path(temp_dir) / "hearing.wav"
            _extract_mathula_analysis_audio(video_path, audio_path)
            LOG.info("Transcribing hearing with Mathula TV's Azure Speech backend")
            raw = backend.transcribe(audio_path, options.locale)
        atomic_write_json(raw_transcript_path, raw)
        utterances = _normalise_mathula_azure_raw(
            raw,
            raw_reference=str(raw_transcript_path.relative_to(output_dir)),
        )
        transcript_source = str(raw_transcript_path)
        stt_provider = str(getattr(backend, "provider", "azure-speech"))

    if not utterances:
        raise RuntimeError("English transcription contained no usable timestamped utterances")
    atomic_write_json(
        normalized_transcript_path,
        {
            "schema_version": "mathula.viral-moments.transcript.v1",
            "created_at": utc_now_iso(),
            "source_video": str(video_path),
            "source_job_id": source_job_id,
            "source_transcript": transcript_source,
            "locale": options.locale,
            "utterance_count": len(utterances),
            "utterances": serialize_utterances(utterances),
        },
    )
    LOG.info("Loaded %d timestamped English utterances", len(utterances))

    min_ms = int(options.min_clip_seconds * 1000)
    target_ms = int(options.target_clip_seconds * 1000)
    max_ms = int(options.max_clip_seconds * 1000)
    candidates = build_candidates(
        utterances,
        min_ms=min_ms,
        target_ms=target_ms,
        max_ms=max_ms,
        candidate_limit=options.candidate_limit,
    )
    if not candidates:
        raise RuntimeError("No viable viral-moment candidates were found")
    LOG.info("Retained %d candidate moments before editorial AI", len(candidates))

    ai_enabled = not options.no_ai
    ai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    ai_key = (
        os.getenv("AZURE_OPENAI_API_KEY", "").strip()
        or os.getenv("AZURE_OPENAI_KEY", "").strip()
    )
    ai_deployment = (
        os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip()
        or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "").strip()
        or os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "").strip()
    )
    if ai_enabled:
        if ai_endpoint and ai_key and ai_deployment:
            ai_rank_candidates(
                candidates,
                utterances,
                endpoint=ai_endpoint,
                api_key=ai_key,
                deployment=ai_deployment,
                batch_size=options.ai_batch_size,
                timeout_seconds=options.openai_timeout_seconds,
            )
        else:
            LOG.warning(
                "Azure OpenAI editorial settings are incomplete; using deterministic ranking"
            )
            ai_enabled = False

    selected = select_final_candidates(
        candidates,
        utterances,
        video_duration_ms=video_duration_ms,
        clip_count=options.clip_count,
        min_ms=min_ms,
        max_ms=max_ms,
        start_padding_ms=options.start_padding_ms,
        end_padding_ms=options.end_padding_ms,
    )
    if not selected:
        raise RuntimeError("No non-overlapping clips passed final selection")

    rendered: list[RenderedClip] = []
    for rank, candidate in enumerate(selected, start=1):
        clip_id = f"viral_{rank:02d}_{candidate.candidate_id.removeprefix('cand_')}"
        filename = (
            f"{rank:02d}_{round(candidate.final_score):03d}_"
            f"{slugify(candidate.title)}.mp4"
        )
        output_path = clips_dir / filename
        sidecar_path = output_path.with_suffix(".json")
        if not options.dry_run:
            render_clip(
                video_path=video_path,
                output_path=output_path,
                start_ms=candidate.start_ms,
                end_ms=candidate.end_ms,
                preset=options.ffmpeg_preset,
                crf=options.crf,
            )
        clip = RenderedClip(
            rank=rank,
            clip_id=clip_id,
            file=str(output_path.relative_to(output_dir)),
            sidecar_file=str(sidecar_path.relative_to(output_dir)),
            score=candidate.final_score,
            title=candidate.title,
            hook=candidate.hook,
            reason=candidate.reason,
            category=candidate.category,
            source_start_ms=candidate.start_ms,
            source_end_ms=candidate.end_ms,
            duration_ms=candidate.duration_ms,
            source_start_timecode=format_timecode(candidate.start_ms),
            source_end_timecode=format_timecode(candidate.end_ms),
            speakers=candidate.speakers,
            transcript=candidate.text,
            heuristic_score=candidate.heuristic_score,
            ai_score=candidate.ai_score,
        )
        atomic_write_json(
            sidecar_path,
            _absolute_clip_record(clip, output_dir, options.downstream_target_language),
        )
        rendered.append(clip)

    clip_records = [
        _absolute_clip_record(item, output_dir, options.downstream_target_language)
        for item in rendered
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now_iso(),
        "module": "mathula_tv.viral_moments",
        "profile": "madlanga_commission",
        "source_job_id": source_job_id,
        "source_video": str(video_path),
        "input_fingerprint": fingerprint,
        "input_fingerprint_payload": fingerprint_payload,
        "idempotent_reuse": False,
        "source_duration_ms": video_duration_ms,
        "source_duration_timecode": format_timecode(video_duration_ms),
        "transcript_file": str(normalized_transcript_path.relative_to(output_dir)),
        "transcript_source": transcript_source,
        "azure_stt_raw_file": (
            str(raw_transcript_path.relative_to(output_dir))
            if raw_transcript_path.exists()
            else None
        ),
        "azure_stt_provider": stt_provider,
        "output_contract": {
            "video": "clean source-aspect H.264/AAC MP4",
            "subtitles_burned_in": False,
            "portrait_reframing_applied": False,
            "intended_next_stage": "Mathula TV TikTok pipeline",
            "queue_file": "tiktok_queue.jsonl",
            "submit_script": "submit_clips_to_mathula_tv.sh",
            "downstream_target_language": options.downstream_target_language,
        },
        "settings": _json_safe_options(options),
        "clip_count": len(rendered),
        "clips": clip_records,
    }
    atomic_write_json(manifest_path, manifest)

    for filename in ("clips_manifest.jsonl", "tiktok_queue.jsonl"):
        with (output_dir / filename).open("w", encoding="utf-8") as handle:
            for record in clip_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    _write_downstream_submit_script(
        output_dir / "submit_clips_to_mathula_tv.sh",
        clip_records,
    )

    atomic_write_json(
        output_dir / "ranking_debug.json",
        {
            "schema_version": "mathula.viral-moments.ranking-debug.v1",
            "created_at": utc_now_iso(),
            "candidate_count": len(candidates),
            "selected_candidate_ids": [item.candidate_id for item in selected],
            "candidates": [candidate_debug_dict(item) for item in candidates],
        },
    )
    LOG.info("Complete: %d ranked clip(s)", len(rendered))
    LOG.info("TikTok queue: %s", output_dir / "tiktok_queue.jsonl")
    return manifest


def run_viral_moments_for_job(
    *,
    job_id: str,
    work_dir: Path,
    options: ViralMomentsOptions | None = None,
    output_dir: Path | None = None,
    force_transcribe: bool = False,
    speech_backend: Any | None = None,
) -> dict[str, Any]:
    """Run against a normal Mathula TV job without changing its state machine."""

    job_root = Path(work_dir).resolve() / "jobs" / job_id
    if not job_root.is_dir():
        raise RuntimeError(f"Mathula TV job does not exist: {job_id}")
    video_path = _resolve_job_source(job_root)
    transcript = None if force_transcribe else _default_job_transcript(job_root)
    destination = Path(output_dir).resolve() if output_dir else job_root / "viral_moments"
    return run_viral_moments(
        video_path=video_path,
        output_dir=destination,
        options=options,
        transcript_path=transcript,
        speech_backend=speech_backend,
        source_job_id=job_id,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create ranked TikTok source clips from a Mathula TV job ID or a long "
            "Madlanga Commission video."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("source", help="Mathula TV job ID or source video path")
    parser.add_argument("--work-dir", type=Path, default=Path(os.getenv("MATHULA_TV_WORK_DIR", "working")))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--reuse-transcript", type=Path)
    parser.add_argument("--force-transcribe", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--locale", default="en-ZA")
    parser.add_argument("--max-speakers", type=int, default=12)
    parser.add_argument("--clip-count", type=int, default=15)
    parser.add_argument("--min-clip-seconds", type=float, default=22.0)
    parser.add_argument("--target-clip-seconds", type=float, default=52.0)
    parser.add_argument("--max-clip-seconds", type=float, default=90.0)
    parser.add_argument("--candidate-limit", type=int, default=60)
    parser.add_argument("--ai-batch-size", type=int, default=10)
    parser.add_argument("--phrases-file", type=Path)
    parser.add_argument("--no-ai", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ffmpeg-preset", default="veryfast")
    parser.add_argument("--crf", type=int, default=20)
    parser.add_argument("--start-padding-ms", type=int, default=350)
    parser.add_argument("--end-padding-ms", type=int, default=650)
    parser.add_argument("--speech-timeout-seconds", type=int, default=7200)
    parser.add_argument("--openai-timeout-seconds", type=int, default=900)
    parser.add_argument("--downstream-target-language", default="zu-ZA")
    parser.add_argument("--verbose", action="store_true")
    return parser


def _options_from_args(args: argparse.Namespace) -> ViralMomentsOptions:
    return ViralMomentsOptions(
        locale=args.locale,
        max_speakers=args.max_speakers,
        clip_count=args.clip_count,
        min_clip_seconds=args.min_clip_seconds,
        target_clip_seconds=args.target_clip_seconds,
        max_clip_seconds=args.max_clip_seconds,
        candidate_limit=args.candidate_limit,
        ai_batch_size=args.ai_batch_size,
        phrases_file=args.phrases_file,
        no_ai=args.no_ai,
        dry_run=args.dry_run,
        ffmpeg_preset=args.ffmpeg_preset,
        crf=args.crf,
        start_padding_ms=args.start_padding_ms,
        end_padding_ms=args.end_padding_ms,
        speech_timeout_seconds=args.speech_timeout_seconds,
        openai_timeout_seconds=args.openai_timeout_seconds,
        force=args.force,
        downstream_target_language=args.downstream_target_language,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[viral-clips %(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        options = _options_from_args(args)
        source_path = Path(args.source)
        if source_path.is_file():
            output_dir = args.output_dir or Path("working/viral_moments") / (
                datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            )
            manifest = run_viral_moments(
                video_path=source_path,
                output_dir=output_dir,
                options=options,
                transcript_path=args.reuse_transcript,
            )
        else:
            manifest = run_viral_moments_for_job(
                job_id=args.source,
                work_dir=args.work_dir,
                options=options,
                output_dir=args.output_dir,
                force_transcribe=args.force_transcribe,
            )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    except KeyboardInterrupt:
        LOG.error("Interrupted")
        return 130
    except Exception as exc:
        LOG.error("%s", exc)
        if args.verbose:
            LOG.exception("Detailed failure")
        return 1


__all__ = [
    "Candidate",
    "RenderedClip",
    "Utterance",
    "ViralMomentsOptions",
    "build_candidates",
    "run_viral_moments",
    "run_viral_moments_for_job",
    "select_final_candidates",
    "utterances_from_mathula_transcript",
]


if __name__ == "__main__":
    raise SystemExit(main())
