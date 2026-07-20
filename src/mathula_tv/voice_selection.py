"""Once-per-speaker Azure base-voice calibration after OpenVoice conversion."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .atomic_io import atomic_write_json


def rank_voice_candidates(
    candidates: list[dict[str, Any]],
    *,
    allowed_voices: list[str] | tuple[str, ...],
    human_override: str | None = None,
) -> dict[str, Any]:
    allowed = list(dict.fromkeys(allowed_voices))
    if not allowed:
        raise ValueError("At least one Azure isiZulu voice must be allowed")
    by_voice = {str(item["voice"]): dict(item) for item in candidates}
    if set(by_voice) != set(allowed):
        missing = sorted(set(allowed) - set(by_voice))
        unexpected = sorted(set(by_voice) - set(allowed))
        raise ValueError(f"Voice calibration candidates mismatch; missing={missing}, unexpected={unexpected}")
    for item in by_voice.values():
        required = {"speaker_similarity", "word_error_approximation", "duration_drift_ratio", "audio_artifact_count"}
        missing_fields = required - set(item)
        if missing_fields:
            raise ValueError(f"Voice candidate {item['voice']} is missing {sorted(missing_fields)}")
        item["ranking_score"] = round(
            float(item["speaker_similarity"])
            - float(item["word_error_approximation"])
            - abs(float(item["duration_drift_ratio"]) - 1.0) * 0.5
            - int(item["audio_artifact_count"]) * 0.05,
            6,
        )
    ranked = sorted(by_voice.values(), key=lambda item: (-item["ranking_score"], item["voice"]))
    if human_override is not None:
        if human_override not in allowed:
            raise ValueError("Human voice override is not in the configured allowlist")
        selected = by_voice[human_override]
        selection_method = "human_override"
    else:
        selected = ranked[0]
        selection_method = "measured_ranking"
    return {
        "schema_version": "speaker-azure-voice-selection-v1",
        "selected_voice": selected["voice"],
        "selection_method": selection_method,
        "human_override": human_override,
        "candidates": ranked,
        "ranking_method": "similarity minus WER, duration drift, and artefact penalties",
        "thresholds_used": False,
        "human_review_required": True,
    }


def calibrate_speaker_voice(
    speaker_id: str,
    allowed_voices: list[str] | tuple[str, ...],
    evaluate: Callable[[str], dict[str, Any]],
    output_path: Path,
    *,
    calibration_sentence: str,
    human_override: str | None = None,
) -> dict[str, Any]:
    if not calibration_sentence.strip():
        raise ValueError("Reviewed calibration sentence is required")
    candidates = []
    for voice in allowed_voices:
        result = evaluate(voice)
        candidates.append({"voice": voice, **result})
    manifest = rank_voice_candidates(candidates, allowed_voices=allowed_voices, human_override=human_override)
    manifest.update(
        speaker_id=speaker_id,
        calibration_sentence=calibration_sentence,
        calibration_scope="job-local-speaker",
    )
    atomic_write_json(output_path, manifest)
    return manifest
