"""Retranscription, speaker-similarity, and production review reports."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from .atomic_io import atomic_write_json
from .logging_utils import redact
from .media import checksum


class SpeakerSimilarityBackend(Protocol):
    backend: str
    version: str

    def compare(self, reference: Path, candidate: Path) -> float: ...


def retranscription_comparison(
    spoken_text: str,
    recognized_text: str,
    *,
    protected_terms: list[str] | None = None,
) -> dict[str, Any]:
    expected = _tokens(spoken_text)
    actual = _tokens(recognized_text)
    distance = _levenshtein(expected, actual)
    protected = protected_terms or []
    protected_results = [
        {
            "term": term,
            "expected": _contains(spoken_text, term),
            "recognized": _contains(recognized_text, term),
        }
        for term in protected
    ]
    expected_numbers = _numbers(spoken_text)
    actual_numbers = _numbers(recognized_text)
    expected_dates = _dates(spoken_text)
    actual_dates = _dates(recognized_text)
    expected_negation = _negations(expected)
    actual_negation = _negations(actual)
    expected_acronyms = _acronyms(spoken_text)
    actual_acronyms = _acronyms(recognized_text)
    return {
        "schema_version": "retranscription-qc-v1",
        "expected_spoken_text": spoken_text,
        "recognized_text": recognized_text,
        "word_error_approximation": round(distance / max(len(expected), 1), 4),
        "protected_terms": protected_results,
        "protected_terms_passed": all(not item["expected"] or item["recognized"] for item in protected_results),
        "numbers_expected": expected_numbers,
        "numbers_recognized": actual_numbers,
        "numbers_preserved": expected_numbers == actual_numbers,
        "dates_expected": expected_dates,
        "dates_recognized": actual_dates,
        "dates_preserved": expected_dates == actual_dates,
        "negation_expected": expected_negation,
        "negation_recognized": actual_negation,
        "negation_preserved": expected_negation == actual_negation,
        "acronyms_expected": expected_acronyms,
        "acronyms_recognized": actual_acronyms,
        "acronyms_preserved": expected_acronyms == actual_acronyms,
        "warning": "Azure isiZulu retranscription is a QC signal; fluent human review is authoritative",
    }


def speaker_similarity_comparison(
    reference: Path,
    raw_azure: Path,
    converted: Path,
    backend: SpeakerSimilarityBackend,
    *,
    threshold: float | None = None,
    threshold_version: str | None = None,
    unrelated: list[Path] | None = None,
) -> dict[str, Any]:
    converted_score = float(backend.compare(reference, converted))
    azure_score = float(backend.compare(reference, raw_azure))
    unrelated_scores = [float(backend.compare(path, converted)) for path in (unrelated or [])]
    calibrated = threshold is not None and bool(threshold_version)
    if threshold is not None and threshold_version:
        result = "passed" if converted_score >= threshold else "failed_identity"
    else:
        result = "needs_human_review"
    return {
        "schema_version": "speaker-similarity-qc-v1",
        "embedding_backend": backend.backend,
        "backend_version": backend.version,
        "input_hashes": {
            "reference": checksum(reference),
            "raw_azure_tts": checksum(raw_azure),
            "openvoice": checksum(converted),
        },
        "reference_to_openvoice": converted_score,
        "reference_to_raw_azure": azure_score,
        "openvoice_to_unrelated_speakers": unrelated_scores,
        "calibration_status": "calibrated" if calibrated else "uncalibrated",
        "threshold": threshold,
        "threshold_version": threshold_version,
        "result": result,
        "warnings": [] if calibrated else ["Similarity threshold is provisional and cannot pass the gate"],
    }


def review_readiness(report: dict[str, Any]) -> str:
    compatibility = str(report.get("compatibility", {}).get("state", "pending"))
    if compatibility in {"pending", "needs_human_review"}:
        return "compatibility_pending"
    if compatibility.startswith("failed"):
        return "compatibility_failed"
    if report.get("openvoice_skipped"):
        return "azure_tts_only_review"
    ordered = (
        ("translation_review_required", "needs_translation_review"),
        ("pronunciation_review_required", "needs_pronunciation_review"),
        ("voice_review_required", "needs_voice_review"),
        ("timing_review_required", "needs_timing_review"),
        ("background_review_required", "needs_background_review"),
        ("audio_review_required", "needs_audio_review"),
    )
    for field, value in ordered:
        if report.get(field):
            return value
    if report.get("failed_units") or report.get("qc_failed"):
        return "failed"
    return "review_ready"


def write_qc_report(
    job_id: str,
    values: dict[str, Any],
    json_path: Path,
    markdown_path: Path,
) -> dict[str, Any]:
    report = {
        "schema_version": "production-qc-report-v1",
        "job_id": job_id,
        **redact(values),
        "human_review_mandatory": True,
        "publication_approved": False,
        "youtube_uploaded": False,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    report["readiness"] = review_readiness(report)
    atomic_write_json(json_path, report)
    markdown = _markdown_report(report)
    _atomic_write_text(markdown_path, markdown)
    return report


def _markdown_report(report: dict[str, Any]) -> str:
    compatibility = report.get("compatibility", {})
    lines = [
        "# Mathula TV dubbing QC report",
        "",
        f"- Job: `{report['job_id']}`",
        f"- Readiness: `{report['readiness']}`",
        f"- Compatibility gate: `{compatibility.get('state', 'pending')}`",
        f"- Speakers: {report.get('number_of_speakers', 0)}",
        f"- Dubbing units: {report.get('number_of_dubbing_units', 0)}",
        f"- Overlaps: {report.get('number_of_overlaps', 0)}",
        f"- Background strategy: `{report.get('background_strategy', 'not_run')}`",
        "",
        "Human voice-use, editorial, factual, timing, audio, and fluent isiZulu review remain mandatory.",
        "This report never constitutes publication approval.",
        "",
    ]
    flags = report.get("human_review_requirements", [])
    if flags:
        lines.extend(["## Human review requirements", ""] + [f"- {item}" for item in flags] + [""])
    return "\n".join(lines)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[\w'-]+", text.casefold(), flags=re.UNICODE)


def _levenshtein(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for index, left_item in enumerate(left, 1):
        current = [index]
        for other, right_item in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[other] + 1, previous[other - 1] + (left_item != right_item)))
        previous = current
    return previous[-1]


def _contains(text: str, term: str) -> bool:
    return term.casefold() in text.casefold()


def _numbers(text: str) -> list[str]:
    return re.findall(r"\b\d+(?:[.,]\d+)?%?\b", text)


def _dates(text: str) -> list[str]:
    months = "january|february|march|april|may|june|july|august|september|october|november|december|ujanuwari|ufebhuwari|umashi|uephreli|umeyi|ujuni|ujulayi|uagasti|usepthemba|u-okthoba|unovemba|udisemba"
    return re.findall(rf"\b(?:\d{{1,2}}[/-]\d{{1,2}}[/-]\d{{2,4}}|\d{{1,2}}\s+(?:{months})\s+\d{{4}})\b", text, flags=re.I)


def _negations(tokens: list[str]) -> list[str]:
    markers = {"not", "no", "never", "akukho", "hhayi", "ngeke", "angeke", "aku"}
    return [token for token in tokens if token in markers or token.startswith("anga")]


def _acronyms(text: str) -> list[str]:
    return re.findall(r"\b[A-Z]{2,}\b", text)
