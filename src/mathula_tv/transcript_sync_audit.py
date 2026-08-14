"""Deterministic transcript, translation, subtitle, and audio-timeline audit.

The audit treats the authoritative autocorrected transcript as the source of
truth.  It runs before paid language AI and again after rendering so a job
cannot be reported complete when corrected wording was lost, protected
wordplay was translated away, or dubbed dialogue disappeared from later
windows on the timeline.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import wave
from array import array
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .atomic_io import atomic_write_json, read_json


AUDIT_SCHEMA_VERSION = "transcript-sync-audit-v1"
CONTRACT_SCHEMA_VERSION = "transcript-contract-v1"
_TOKEN = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)


class TranscriptSyncAuditFailure(ValueError):
    """Raised when a strict transcript/synchronisation audit fails."""

    def __init__(self, message: str, *, report: Mapping[str, Any]):
        super().__init__(message)
        self.report = dict(report)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tokens(value: Any) -> list[str]:
    return [token.casefold() for token in _TOKEN.findall(str(value or ""))]


def _count_token_sequence(value: Any, phrase: Any) -> int:
    haystack = _tokens(value)
    needle = _tokens(phrase)
    if not needle:
        return 0
    return sum(
        1
        for index in range(len(haystack) - len(needle) + 1)
        if haystack[index : index + len(needle)] == needle
    )


def _contains_tokens(value: Any, phrase: Any) -> bool:
    return _count_token_sequence(value, phrase) > 0


def _segment_text(segment: Mapping[str, Any]) -> str:
    return str(segment.get("source_text") or segment.get("text") or "").strip()


def _unit_list(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    units = value.get("units")
    if not isinstance(units, list):
        raise ValueError("Artifact does not contain a units list")
    return [dict(unit) for unit in units if isinstance(unit, Mapping)]


def _issue(
    issues: list[dict[str, Any]],
    severity: str,
    code: str,
    message: str,
    **details: Any,
) -> None:
    issues.append(
        {
            "severity": severity,
            "code": code,
            "message": message,
            "details": details,
        }
    )


def _load_contract(repo_root: Path, job_root: Path, raw_sha256: str) -> tuple[Path | None, dict[str, Any] | None]:
    candidates = [
        job_root / "analysis/transcript_contract.json",
        repo_root / "config/transcript_contracts" / f"{raw_sha256}.json",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        contract = read_json(path)
        if contract.get("schema_version") != CONTRACT_SCHEMA_VERSION:
            raise ValueError(f"Unsupported transcript contract schema: {path}")
        expected_hash = str(contract.get("raw_transcript_sha256") or "")
        if expected_hash and expected_hash != raw_sha256:
            raise ValueError(f"Transcript contract does not belong to the raw transcript: {path}")
        return path, contract
    return None, None


def _validate_raw_transcript(raw: Mapping[str, Any], issues: list[dict[str, Any]]) -> None:
    segments = raw.get("segments")
    words = raw.get("words")
    if not isinstance(segments, list) or not segments:
        _issue(issues, "error", "raw_segments_missing", "Raw transcript has no segments")
        return
    if not isinstance(words, list) or not words:
        _issue(issues, "error", "raw_words_missing", "Raw transcript has no timed words")
        return

    previous_segment_end = -math.inf
    segment_ids: set[str] = set()
    for index, segment in enumerate(segments):
        if not isinstance(segment, Mapping):
            _issue(issues, "error", "raw_segment_invalid", "Raw segment is not an object", index=index)
            continue
        segment_id = str(segment.get("segment_id") or "")
        start = float(segment.get("start") or 0)
        end = float(segment.get("end") or 0)
        if not segment_id or segment_id in segment_ids:
            _issue(issues, "error", "raw_segment_id_invalid", "Raw segment IDs must be unique", segment_id=segment_id)
        segment_ids.add(segment_id)
        if end <= start:
            _issue(issues, "error", "raw_segment_duration_invalid", "Raw segment has a non-positive duration", segment_id=segment_id)
        if start < previous_segment_end - 0.001:
            _issue(issues, "error", "raw_segment_order_invalid", "Raw segments overlap or are out of order", segment_id=segment_id)
        previous_segment_end = max(previous_segment_end, end)

        timed_words = [
            word
            for word in words
            if isinstance(word, Mapping)
            and str(word.get("speaker")) == str(segment.get("speaker"))
            and float(word.get("start") or 0) >= start - 0.001
            and float(word.get("end") or 0) <= end + 0.001
        ]
        if _tokens(_segment_text(segment)) != _tokens(" ".join(str(word.get("text") or "") for word in timed_words)):
            _issue(
                issues,
                "error",
                "raw_segment_word_mismatch",
                "Raw segment text does not match its timed Azure words",
                segment_id=segment_id,
            )

    previous_word_start = -math.inf
    for index, word in enumerate(words):
        if not isinstance(word, Mapping):
            _issue(issues, "error", "raw_word_invalid", "Raw word is not an object", index=index)
            continue
        start = float(word.get("start") or 0)
        end = float(word.get("end") or 0)
        if end <= start:
            _issue(issues, "error", "raw_word_duration_invalid", "Raw word has a non-positive duration", index=index)
        if start < previous_word_start - 0.001:
            _issue(issues, "error", "raw_word_order_invalid", "Raw words are out of order", index=index)
        previous_word_start = start


def _validate_authoritative_transcript(
    raw: Mapping[str, Any],
    authoritative: Mapping[str, Any],
    contract: Mapping[str, Any] | None,
    issues: list[dict[str, Any]],
) -> None:
    raw_segments = raw.get("segments") if isinstance(raw.get("segments"), list) else []
    corrected_segments = authoritative.get("segments") if isinstance(authoritative.get("segments"), list) else []
    if len(raw_segments) != len(corrected_segments):
        _issue(
            issues,
            "error",
            "authoritative_segment_count_changed",
            "Autocorrection changed the segment count",
            raw_count=len(raw_segments),
            authoritative_count=len(corrected_segments),
        )
        return

    raw_by_id: dict[str, Mapping[str, Any]] = {}
    corrected_by_id: dict[str, Mapping[str, Any]] = {}
    for raw_segment, corrected_segment in zip(raw_segments, corrected_segments):
        for field in ("segment_id", "speaker"):
            if str(raw_segment.get(field)) != str(corrected_segment.get(field)):
                _issue(
                    issues,
                    "error",
                    "authoritative_segment_identity_changed",
                    "Autocorrection changed segment identity",
                    field=field,
                    raw=raw_segment.get(field),
                    authoritative=corrected_segment.get(field),
                )
        for field in ("start", "end"):
            if abs(float(raw_segment.get(field) or 0) - float(corrected_segment.get(field) or 0)) > 0.001:
                _issue(
                    issues,
                    "error",
                    "authoritative_timing_changed",
                    "Autocorrection changed source timing",
                    segment_id=raw_segment.get("segment_id"),
                    field=field,
                )
        if not _segment_text(corrected_segment):
            _issue(issues, "error", "authoritative_text_empty", "Corrected segment text is empty", segment_id=raw_segment.get("segment_id"))
        segment_id = str(raw_segment.get("segment_id") or "")
        if segment_id:
            raw_by_id[segment_id] = raw_segment
            corrected_by_id[segment_id] = corrected_segment

    autocorrect = authoritative.get("autocorrect")
    if not isinstance(autocorrect, Mapping):
        _issue(issues, "error", "autocorrect_metadata_missing", "Authoritative transcript has no autocorrection metadata")
        return

    overlays = autocorrect.get("overlays")
    if not isinstance(overlays, list):
        _issue(issues, "error", "autocorrect_overlays_missing", "Authoritative transcript has no autocorrection overlay list")
        return
    active_count = int(autocorrect.get("active_correction_count", len(overlays)) or 0)
    if active_count != len(overlays):
        _issue(
            issues,
            "error",
            "autocorrect_overlay_count_mismatch",
            "Autocorrection metadata count does not match rendered overlays",
            active_correction_count=active_count,
            overlay_count=len(overlays),
        )

    grouped: dict[tuple[str, tuple[str, ...], tuple[str, ...]], list[Mapping[str, Any]]] = {}
    for index, overlay in enumerate(overlays):
        if not isinstance(overlay, Mapping):
            _issue(issues, "error", "autocorrect_overlay_invalid", "Autocorrection overlay is not an object", index=index)
            continue
        item_id = str(overlay.get("item_id") or "")
        source = str(overlay.get("azure_text") or overlay.get("heard_text") or "").strip()
        replacement = str(overlay.get("corrected_text") or overlay.get("replacement_text") or "").strip()
        status = str(overlay.get("status") or "")
        raw_segment = raw_by_id.get(item_id)
        corrected_segment = corrected_by_id.get(item_id)
        if raw_segment is None or corrected_segment is None:
            _issue(issues, "error", "autocorrect_overlay_item_missing", "Autocorrection overlay references an unknown segment", item_id=item_id, source=source, replacement=replacement)
            continue
        if not source or not replacement:
            _issue(issues, "error", "autocorrect_overlay_text_missing", "Autocorrection overlay is missing source or replacement text", item_id=item_id, source=source, replacement=replacement)
            continue
        if status and status not in {"auto_applied", "confirmed", "edited"}:
            _issue(issues, "error", "autocorrect_overlay_status_invalid", "Rendered autocorrection overlay is not active", item_id=item_id, status=status, source=source, replacement=replacement)
        raw_text = _segment_text(raw_segment)
        corrected_text = _segment_text(corrected_segment)
        if not _contains_tokens(raw_text, source):
            _issue(issues, "error", "autocorrect_overlay_source_missing", "Applied correction source is absent from its raw segment", item_id=item_id, source=source, replacement=replacement)
        if not _contains_tokens(corrected_text, replacement):
            _issue(issues, "error", "autocorrect_overlay_replacement_missing", "Applied correction replacement is absent from the authoritative segment", item_id=item_id, source=source, replacement=replacement)
        segment_start_ms = round(float(raw_segment.get("start") or 0) * 1000)
        segment_end_ms = round(float(raw_segment.get("end") or 0) * 1000)
        overlay_start = overlay.get("start_ms")
        overlay_end = overlay.get("end_ms")
        if overlay_start is not None and int(overlay_start) < segment_start_ms - 1:
            _issue(issues, "error", "autocorrect_overlay_timing_invalid", "Applied correction begins before its source segment", item_id=item_id, source=source, start_ms=overlay_start, segment_start_ms=segment_start_ms)
        if overlay_end is not None and int(overlay_end) > segment_end_ms + 1:
            _issue(issues, "error", "autocorrect_overlay_timing_invalid", "Applied correction ends after its source segment", item_id=item_id, source=source, end_ms=overlay_end, segment_end_ms=segment_end_ms)
        grouped.setdefault((item_id, tuple(_tokens(source)), tuple(_tokens(replacement))), []).append(overlay)

    # Validate only corrections that the autocorrection stage says it actually
    # rendered. Static transcript contracts must not duplicate PostgreSQL
    # vocabulary rules because the two sources drift and create false failures.
    for (item_id, source_tokens, replacement_tokens), matching in grouped.items():
        if not source_tokens or not replacement_tokens or source_tokens == replacement_tokens:
            continue
        raw_text = _segment_text(raw_by_id[item_id])
        corrected_text = _segment_text(corrected_by_id[item_id])
        source_phrase = str(matching[0].get("azure_text") or matching[0].get("heard_text") or "")
        replacement_phrase = str(matching[0].get("corrected_text") or matching[0].get("replacement_text") or "")
        raw_source_count = _count_token_sequence(raw_text, source_phrase)
        corrected_source_count = _count_token_sequence(corrected_text, source_phrase)
        expected_source_max = max(0, raw_source_count - len(matching))
        if corrected_source_count > expected_source_max:
            _issue(
                issues,
                "error",
                "applied_correction_source_survived",
                "An applied autocorrection still contains its raw STT wording",
                item_id=item_id,
                source=source_phrase,
                replacement=replacement_phrase,
                applied_count=len(matching),
                raw_source_count=raw_source_count,
                authoritative_source_count=corrected_source_count,
            )

    # Optional static assertions remain available for deliberately curated job
    # contracts, but are opt-in. PostgreSQL is otherwise the sole authority for
    # which corrections must be applied.
    if contract and contract.get("enforce_corrections") is True:
        corrected_complete = str(authoritative.get("complete_text") or " ".join(_segment_text(segment) for segment in corrected_segments))
        raw_complete = str(raw.get("complete_text") or " ".join(_segment_text(segment) for segment in raw_segments))
        for correction in contract.get("corrections", []):
            if not isinstance(correction, Mapping):
                continue
            source = str(correction.get("source") or "").strip()
            replacement = str(correction.get("replacement") or "").strip()
            required = bool(correction.get("required", True))
            if source and not _contains_tokens(raw_complete, source):
                _issue(issues, "error" if required else "warning", "contract_source_missing", "Contract source phrase is absent from raw transcript", source=source)
            if replacement and not _contains_tokens(corrected_complete, replacement):
                _issue(issues, "error" if required else "warning", "required_correction_missing", "Required autocorrection is absent from the authoritative transcript", source=source, replacement=replacement)
            if source and correction.get("forbid_source_after_correction", True) and _contains_tokens(corrected_complete, source):
                _issue(issues, "error" if required else "warning", "raw_misrecognition_survived", "Raw STT wording survived autocorrection", source=source, replacement=replacement)

    if not contract:
        return
    corrected_complete = str(authoritative.get("complete_text") or " ".join(_segment_text(segment) for segment in corrected_segments))
    for advisory in contract.get("advisories", []):
        if not isinstance(advisory, Mapping):
            continue
        phrase = str(advisory.get("source") or "").strip()
        if phrase and _contains_tokens(corrected_complete, phrase):
            _issue(
                issues,
                "warning",
                "transcript_advisory",
                str(advisory.get("message") or "Transcript phrase requires human confirmation"),
                source=phrase,
                suggestion=advisory.get("suggestion"),
            )


def _validate_units(
    authoritative: Mapping[str, Any],
    units_artifact: Mapping[str, Any],
    issues: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    try:
        units = _unit_list(units_artifact)
    except ValueError as exc:
        _issue(issues, "error", "dubbing_units_missing", str(exc))
        return []

    expected_ids = [f"unit_{index + 1:04d}" for index in range(len(units))]
    returned_ids = [str(unit.get("unit_id")) for unit in units]
    if returned_ids != expected_ids:
        _issue(issues, "error", "dubbing_unit_ids_invalid", "Dubbing unit IDs are not sequential", returned_ids=returned_ids)

    authoritative_tokens = _tokens(" ".join(_segment_text(segment) for segment in authoritative.get("segments", [])))
    unit_tokens = _tokens(" ".join(str(unit.get("source_text") or "") for unit in units))
    if unit_tokens != authoritative_tokens:
        _issue(
            issues,
            "error",
            "dubbing_unit_text_coverage_mismatch",
            "Dubbing units do not reproduce the authoritative corrected transcript exactly",
            authoritative_token_count=len(authoritative_tokens),
            unit_token_count=len(unit_tokens),
        )

    previous_start = -1
    for unit in units:
        unit_id = str(unit.get("unit_id"))
        start = int(unit.get("start_ms") or 0)
        preferred_end = int(unit.get("preferred_end_ms") or 0)
        hard_end = int(unit.get("hard_end_ms") or 0)
        if start < previous_start:
            _issue(issues, "error", "dubbing_unit_order_invalid", "Dubbing unit starts are out of order", unit_id=unit_id)
        previous_start = start
        if not start <= preferred_end <= hard_end:
            _issue(issues, "error", "dubbing_unit_window_invalid", "Dubbing unit timing window is invalid", unit_id=unit_id)
        if int(unit.get("preferred_duration_ms") or -1) != preferred_end - start:
            _issue(issues, "error", "preferred_duration_mismatch", "Preferred duration does not match the unit window", unit_id=unit_id)
        if int(unit.get("maximum_duration_ms") or -1) != hard_end - start:
            _issue(issues, "error", "maximum_duration_mismatch", "Maximum duration does not match the hard window", unit_id=unit_id)
    return units


def _units_overlapping(units: Sequence[Mapping[str, Any]], start_ms: int, end_ms: int) -> list[Mapping[str, Any]]:
    return [
        unit
        for unit in units
        if int(unit.get("hard_end_ms") or unit.get("preferred_end_ms") or 0) > start_ms
        and int(unit.get("start_ms") or 0) < end_ms
    ]


def _validate_plan_and_contract(
    source_units: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
    contract: Mapping[str, Any] | None,
    issues: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    try:
        plan_units = _unit_list(plan)
    except ValueError as exc:
        _issue(issues, "error", "dubbing_plan_missing", str(exc))
        return []
    if [str(unit.get("unit_id")) for unit in plan_units] != [str(unit.get("unit_id")) for unit in source_units]:
        _issue(issues, "error", "plan_unit_identity_mismatch", "Dubbing plan units do not match source units")
    for source, translated in zip(source_units, plan_units):
        unit_id = str(source.get("unit_id"))
        for field in ("start_ms", "preferred_end_ms", "hard_end_ms", "preferred_duration_ms", "maximum_duration_ms"):
            if int(source.get(field) or 0) != int(translated.get(field) or 0):
                _issue(issues, "error", "plan_timing_changed", "Translation changed a source timing field", unit_id=unit_id, field=field)
        for field in ("spoken_text", "tts_text"):
            if not str(translated.get(field) or "").strip():
                _issue(issues, "error", "translated_text_missing", f"{field} is empty", unit_id=unit_id)

    if not contract:
        return plan_units

    for span in contract.get("protected_spans", []):
        if not isinstance(span, Mapping):
            continue
        start_ms = int(span.get("start_ms") or 0)
        end_ms = int(span.get("end_ms") or start_ms)
        overlapping = _units_overlapping(plan_units, start_ms, end_ms)
        if not overlapping:
            _issue(issues, "error", "protected_span_has_no_unit", "Protected transcript span is not covered by a dubbing unit", span_id=span.get("span_id"))
            continue
        source_joined = " ".join(str(unit.get("source_text") or "") for unit in overlapping)
        spoken_joined = " ".join(str(unit.get("spoken_text") or "") for unit in overlapping)
        tts_joined = " ".join(str(unit.get("tts_text") or "") for unit in overlapping)
        source_expected = str(span.get("source_expected") or span.get("display_expected") or "")
        display_expected = str(span.get("display_expected") or source_expected)
        tts_expected = str(span.get("tts_expected") or display_expected)
        if source_expected and not _contains_tokens(source_joined, source_expected):
            _issue(issues, "error", "protected_source_span_missing", "Corrected source phrase is missing from its timing span", span_id=span.get("span_id"), expected=source_expected)
        if display_expected and not _contains_tokens(spoken_joined, display_expected):
            _issue(issues, "error", "protected_spoken_span_missing", "Subtitle/spoken translation lost a protected phrase", span_id=span.get("span_id"), expected=display_expected, units=[unit.get("unit_id") for unit in overlapping])
        if tts_expected and not _contains_tokens(tts_joined, tts_expected):
            _issue(issues, "error", "protected_tts_span_missing", "TTS text lost the required pronunciation form", span_id=span.get("span_id"), expected=tts_expected, units=[unit.get("unit_id") for unit in overlapping])
    return plan_units


def _read_pcm16(path: Path) -> tuple[int, int, array]:
    with wave.open(str(path), "rb") as handle:
        if handle.getsampwidth() != 2 or handle.getcomptype() != "NONE":
            raise ValueError(f"Expected uncompressed PCM16 WAV: {path}")
        rate = handle.getframerate()
        channels = handle.getnchannels()
        frames = handle.readframes(handle.getnframes())
    samples = array("h")
    samples.frombytes(frames)
    if samples.itemsize != 2:
        raise ValueError("Unexpected PCM16 sample width")
    return rate, channels, samples


def _mono_samples(samples: array, channels: int) -> list[float]:
    if channels == 1:
        return [float(value) for value in samples]
    return [
        sum(samples[index : index + channels]) / channels
        for index in range(0, len(samples), channels)
    ]


def _rms(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return math.sqrt(sum(value * value for value in values) / len(values))


def _strongest_window(values: Sequence[float], rate: int, usable_ms: int, window_ms: int = 120) -> tuple[int, int, float]:
    usable_frames = min(len(values), max(1, round(usable_ms * rate / 1000)))
    window_frames = min(usable_frames, max(1, round(window_ms * rate / 1000)))
    stride = max(1, window_frames // 2)
    best = (0, window_frames, 0.0)
    for start in range(0, max(1, usable_frames - window_frames + 1), stride):
        end = min(usable_frames, start + window_frames)
        value = _rms(values[start:end])
        if value > best[2]:
            best = (start, end, value)
    return best


def _slice_ms(values: Sequence[float], rate: int, start_ms: int, end_ms: int) -> Sequence[float]:
    begin = max(0, round(start_ms * rate / 1000))
    finish = min(len(values), round(end_ms * rate / 1000))
    return values[begin:finish]


def _validate_alignment_and_audio(
    job_root: Path,
    plan_units: Sequence[Mapping[str, Any]],
    alignment: Mapping[str, Any],
    issues: list[dict[str, Any]],
    *,
    hard_tolerance_ms: int = 120,
) -> list[dict[str, Any]]:
    try:
        aligned_units = _unit_list(alignment)
    except ValueError as exc:
        _issue(issues, "error", "alignment_manifest_missing", str(exc))
        return []
    aligned_by_id = {str(unit.get("unit_id")): unit for unit in aligned_units}

    dialogue_path = job_root / "audio/dub_dialogue.wav"
    dialogue_rate = 0
    dialogue_values: list[float] = []
    if dialogue_path.is_file():
        try:
            dialogue_rate, dialogue_channels, dialogue_samples = _read_pcm16(dialogue_path)
            dialogue_values = _mono_samples(dialogue_samples, dialogue_channels)
        except Exception as exc:
            _issue(issues, "error", "dialogue_bus_invalid", "Dialogue bus is not valid PCM16", error=str(exc))
    else:
        _issue(issues, "error", "dialogue_bus_missing", "Dialogue bus is missing")

    for plan_unit in plan_units:
        unit_id = str(plan_unit.get("unit_id"))
        aligned = aligned_by_id.get(unit_id)
        if aligned is None:
            _issue(issues, "error", "alignment_unit_missing", "Alignment result is missing", unit_id=unit_id)
            continue
        start_ms = int(plan_unit.get("start_ms") or 0)
        preferred_ms = int(plan_unit.get("preferred_duration_ms") or 0)
        maximum_ms = int(plan_unit.get("maximum_duration_ms") or 0)
        final_ms = int(aligned.get("final_duration_ms") or 0)
        if final_ms < preferred_ms - 20:
            _issue(issues, "error", "aligned_unit_short", "Aligned audio is shorter than the preferred timeline window", unit_id=unit_id, final_duration_ms=final_ms, preferred_duration_ms=preferred_ms)
        if final_ms > maximum_ms + hard_tolerance_ms:
            _issue(issues, "error", "aligned_unit_exceeds_hard_window", "Aligned audio exceeds the hard timing window", unit_id=unit_id, final_duration_ms=final_ms, maximum_duration_ms=maximum_ms)
        if start_ms + final_ms > int(plan_unit.get("hard_end_ms") or 0) + hard_tolerance_ms:
            _issue(issues, "error", "aligned_unit_absolute_end_invalid", "Aligned audio ends outside its source timing window", unit_id=unit_id)

        aligned_path = Path(str(aligned.get("output_path") or job_root / "dubbing/aligned/turns" / f"{unit_id}.wav"))
        if not aligned_path.is_absolute():
            aligned_path = job_root / aligned_path
        if not aligned_path.is_file():
            fallback = job_root / "dubbing/aligned/turns" / f"{unit_id}.wav"
            aligned_path = fallback if fallback.is_file() else aligned_path
        if not aligned_path.is_file():
            _issue(issues, "error", "aligned_wav_missing", "Aligned unit WAV is missing", unit_id=unit_id, path=str(aligned_path))
            continue
        try:
            rate, channels, samples = _read_pcm16(aligned_path)
            values = _mono_samples(samples, channels)
        except Exception as exc:
            _issue(issues, "error", "aligned_wav_invalid", "Aligned unit WAV is invalid", unit_id=unit_id, error=str(exc))
            continue
        measured_ms = round(len(values) * 1000 / rate)
        if abs(measured_ms - final_ms) > 20:
            _issue(issues, "error", "alignment_manifest_duration_mismatch", "Alignment manifest does not match the WAV duration", unit_id=unit_id, manifest_ms=final_ms, measured_ms=measured_ms)

        usable_ms = int(aligned.get("corrected_duration_ms") or aligned.get("source_duration_ms") or final_ms)
        peak_start, peak_end, peak_rms = _strongest_window(values, rate, usable_ms)
        if peak_rms < 30:
            _issue(issues, "error", "aligned_unit_effectively_silent", "Aligned unit is effectively silent", unit_id=unit_id, peak_rms=peak_rms)
            continue
        if dialogue_values and dialogue_rate:
            absolute_start_ms = start_ms + round(peak_start * 1000 / rate)
            absolute_end_ms = start_ms + round(peak_end * 1000 / rate)
            bus_rms = _rms(_slice_ms(dialogue_values, dialogue_rate, absolute_start_ms, absolute_end_ms))
            minimum = max(30.0, peak_rms * 0.08)
            if bus_rms < minimum:
                _issue(
                    issues,
                    "error",
                    "dialogue_bus_missing_unit_audio",
                    "Dub dialogue bus has no audible energy at this unit's source timestamp",
                    unit_id=unit_id,
                    start_ms=absolute_start_ms,
                    end_ms=absolute_end_ms,
                    aligned_peak_rms=round(peak_rms, 3),
                    dialogue_bus_rms=round(bus_rms, 3),
                )
    return aligned_units


def _validate_subtitles(
    job_root: Path,
    plan_units: Sequence[Mapping[str, Any]],
    aligned_units: Sequence[Mapping[str, Any]],
    issues: list[dict[str, Any]],
) -> None:
    path = job_root / "subtitles/spoken_zu.json"
    if not path.is_file():
        _issue(issues, "error", "subtitle_manifest_missing", "spoken_zu.json is missing")
        return
    data = read_json(path)
    entries = data.get("entries") if isinstance(data.get("entries"), list) else []
    if len(entries) != len(plan_units):
        _issue(issues, "error", "subtitle_count_mismatch", "Subtitle count does not match dubbing units", subtitle_count=len(entries), unit_count=len(plan_units))
        return
    aligned_by_id = {str(item.get("unit_id")): item for item in aligned_units}
    plan_by_id = {str(item.get("unit_id")): item for item in plan_units}
    for entry in entries:
        unit_id = str(entry.get("unit_id"))
        plan = plan_by_id.get(unit_id)
        aligned = aligned_by_id.get(unit_id)
        if plan is None or aligned is None:
            _issue(issues, "error", "subtitle_unit_unknown", "Subtitle references an unknown unit", unit_id=unit_id)
            continue
        expected_start = int(plan.get("start_ms") or 0)
        expected_end = expected_start + int(aligned.get("final_duration_ms") or 0)
        if abs(int(entry.get("start_ms") or 0) - expected_start) > 1 or abs(int(entry.get("end_ms") or 0) - expected_end) > 1:
            _issue(issues, "error", "subtitle_timing_mismatch", "Subtitle timing does not match the aligned speech", unit_id=unit_id, expected_start_ms=expected_start, expected_end_ms=expected_end, actual_start_ms=entry.get("start_ms"), actual_end_ms=entry.get("end_ms"))
        if _tokens(entry.get("spoken_text")) != _tokens(plan.get("spoken_text")):
            _issue(issues, "error", "subtitle_text_mismatch", "Subtitle text does not match spoken_text", unit_id=unit_id)


def _validate_final_media(job_root: Path, issues: list[dict[str, Any]]) -> None:
    render_manifest_path = job_root / "render/render_manifest.json"
    if not render_manifest_path.is_file():
        _issue(issues, "error", "render_manifest_missing", "Render manifest is missing")
        return
    manifest = read_json(render_manifest_path)
    if manifest.get("video_stream_copy") is not True:
        _issue(issues, "error", "video_not_stream_copied", "Rendered video did not preserve the source video bitstream")
    if str(manifest.get("source_video_packet_hash") or "") != str(manifest.get("output_video_packet_hash") or ""):
        _issue(issues, "error", "video_packet_hash_mismatch", "Rendered MP4 changed encoded source video packets")
    source_duration = float(manifest.get("source_duration_seconds") or 0)
    rendered_duration = float(manifest.get("rendered_duration_seconds") or 0)
    if source_duration <= 0 or abs(rendered_duration - source_duration) > 0.15:
        _issue(issues, "error", "render_duration_mismatch", "Rendered duration does not match the source video", source_duration=source_duration, rendered_duration=rendered_duration)
    final_mix = job_root / "audio/final_mix.wav"
    if not final_mix.is_file():
        _issue(issues, "error", "final_mix_missing", "Final mixed WAV is missing")
    review_video = job_root / "output/review_zu.mp4"
    if not review_video.is_file():
        _issue(issues, "error", "review_video_missing", "Review video alias is missing")


def validate_pre_translation(
    *,
    repo_root: Path,
    job_root: Path,
    strict: bool = True,
) -> dict[str, Any]:
    """Validate raw→autocorrected→unit text before any paid language AI call."""

    raw_path = job_root / "analysis/transcript_en_raw.json"
    authoritative_path = job_root / "analysis/transcript_en.json"
    units_path = job_root / "dubbing/dubbing_units.json"
    issues: list[dict[str, Any]] = []
    for path, code in (
        (raw_path, "raw_transcript_missing"),
        (authoritative_path, "authoritative_transcript_missing"),
        (units_path, "dubbing_units_missing"),
    ):
        if not path.is_file():
            _issue(issues, "error", code, f"Required pre-translation artifact is missing: {path}")
    if any(issue["severity"] == "error" for issue in issues):
        report = _report("pre_translation", issues, None, {})
        if strict:
            raise TranscriptSyncAuditFailure("Pre-translation transcript audit failed", report=report)
        return report

    raw_sha256 = _sha256(raw_path)
    contract_path, contract = _load_contract(repo_root, job_root, raw_sha256)
    raw = read_json(raw_path)
    authoritative = read_json(authoritative_path)
    units_artifact = read_json(units_path)
    _validate_raw_transcript(raw, issues)
    _validate_authoritative_transcript(raw, authoritative, contract, issues)
    units = _validate_units(authoritative, units_artifact, issues)
    report = _report(
        "pre_translation",
        issues,
        contract_path,
        {
            "raw_transcript_sha256": raw_sha256,
            "segment_count": len(raw.get("segments", [])),
            "word_count": len(raw.get("words", [])),
            "unit_count": len(units),
        },
    )
    if strict and not report["overall_passed"]:
        raise TranscriptSyncAuditFailure(_failure_message(report), report=report)
    return report


def validate_translation_plan(
    *,
    repo_root: Path,
    job_root: Path,
    strict: bool = True,
) -> dict[str, Any]:
    """Validate translated/spoken/TTS text against source-timed contract before TTS."""

    paths = {
        "raw": job_root / "analysis/transcript_en_raw.json",
        "authoritative": job_root / "analysis/transcript_en.json",
        "units": job_root / "dubbing/dubbing_units.json",
        "plan": job_root / "dubbing/dubbing_plan.json",
    }
    issues: list[dict[str, Any]] = []
    for name, path in paths.items():
        if not path.is_file():
            _issue(issues, "error", f"{name}_artifact_missing", f"Required translation audit artifact is missing: {path}")
    if any(issue["severity"] == "error" for issue in issues):
        report = _report("pre_tts", issues, None, {})
        if strict:
            raise TranscriptSyncAuditFailure(_failure_message(report), report=report)
        return report

    raw_sha256 = _sha256(paths["raw"])
    contract_path, contract = _load_contract(repo_root, job_root, raw_sha256)
    raw = read_json(paths["raw"])
    authoritative = read_json(paths["authoritative"])
    units_artifact = read_json(paths["units"])
    plan = read_json(paths["plan"])
    _validate_raw_transcript(raw, issues)
    _validate_authoritative_transcript(raw, authoritative, contract, issues)
    source_units = _validate_units(authoritative, units_artifact, issues)
    plan_units = _validate_plan_and_contract(source_units, plan, contract, issues)
    report = _report(
        "pre_tts",
        issues,
        contract_path,
        {
            "raw_transcript_sha256": raw_sha256,
            "segment_count": len(raw.get("segments", [])),
            "word_count": len(raw.get("words", [])),
            "unit_count": len(source_units),
            "translated_unit_count": len(plan_units),
        },
    )
    report_path = job_root / "review/translation_contract_audit.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(report_path, report)
    report["report_path"] = str(report_path)
    if strict and not report["overall_passed"]:
        raise TranscriptSyncAuditFailure(_failure_message(report), report=report)
    return report


def run_transcript_sync_audit(
    *,
    repo_root: Path,
    job_root: Path,
    strict: bool = True,
    require_final_media: bool = True,
) -> dict[str, Any]:
    """Run the full text and timing audit and persist its JSON report."""

    issues: list[dict[str, Any]] = []
    required = {
        "raw": job_root / "analysis/transcript_en_raw.json",
        "authoritative": job_root / "analysis/transcript_en.json",
        "units": job_root / "dubbing/dubbing_units.json",
        "plan": job_root / "dubbing/dubbing_plan.json",
        "alignment": job_root / "dubbing/aligned/manifest.json",
    }
    for name, path in required.items():
        if not path.is_file():
            _issue(issues, "error", f"{name}_artifact_missing", f"Required audit artifact is missing: {path}")
    if any(issue["severity"] == "error" for issue in issues):
        report = _report("final", issues, None, {})
        if strict:
            raise TranscriptSyncAuditFailure(_failure_message(report), report=report)
        return report

    raw_path = required["raw"]
    raw_sha256 = _sha256(raw_path)
    contract_path, contract = _load_contract(repo_root, job_root, raw_sha256)
    raw = read_json(raw_path)
    authoritative = read_json(required["authoritative"])
    units_artifact = read_json(required["units"])
    plan = read_json(required["plan"])
    alignment = read_json(required["alignment"])

    _validate_raw_transcript(raw, issues)
    _validate_authoritative_transcript(raw, authoritative, contract, issues)
    source_units = _validate_units(authoritative, units_artifact, issues)
    plan_units = _validate_plan_and_contract(source_units, plan, contract, issues)
    aligned_units = _validate_alignment_and_audio(job_root, plan_units, alignment, issues)
    _validate_subtitles(job_root, plan_units, aligned_units, issues)
    if require_final_media:
        _validate_final_media(job_root, issues)

    report = _report(
        "final",
        issues,
        contract_path,
        {
            "raw_transcript_sha256": raw_sha256,
            "segment_count": len(raw.get("segments", [])),
            "word_count": len(raw.get("words", [])),
            "unit_count": len(source_units),
            "aligned_unit_count": len(aligned_units),
        },
    )
    report_path = job_root / "review/transcript_sync_audit.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(report_path, report)
    report["report_path"] = str(report_path)
    if strict and not report["overall_passed"]:
        raise TranscriptSyncAuditFailure(_failure_message(report), report=report)
    return report


def _report(stage: str, issues: list[dict[str, Any]], contract_path: Path | None, metrics: Mapping[str, Any]) -> dict[str, Any]:
    errors = [issue for issue in issues if issue["severity"] == "error"]
    warnings = [issue for issue in issues if issue["severity"] == "warning"]
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "stage": stage,
        "overall_passed": not errors,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "contract_path": str(contract_path) if contract_path else None,
        "metrics": dict(metrics),
        "issues": issues,
    }


def _failure_message(report: Mapping[str, Any]) -> str:
    errors = [issue for issue in report.get("issues", []) if issue.get("severity") == "error"]
    parts: list[str] = []
    for issue in errors[:8]:
        details = issue.get("details") if isinstance(issue.get("details"), Mapping) else {}
        context = (
            details.get("source")
            or details.get("span_id")
            or details.get("unit_id")
            or details.get("item_id")
        )
        suffix = f" [{context}]" if context else ""
        parts.append(f"{issue.get('code')}: {issue.get('message')}{suffix}")
    summary = "; ".join(parts)
    if len(errors) > 8:
        summary += f"; and {len(errors) - 8} more"
    return f"Transcript/timing audit failed: {summary}"


__all__ = [
    "AUDIT_SCHEMA_VERSION",
    "CONTRACT_SCHEMA_VERSION",
    "TranscriptSyncAuditFailure",
    "run_transcript_sync_audit",
    "validate_pre_translation",
    "validate_translation_plan",
]
