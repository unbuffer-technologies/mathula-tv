"""Versioned deterministic production dubbing plan construction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .atomic_io import atomic_write_json
from .logging_utils import redact
from .media import checksum
from .pronunciation import PronunciationDictionary
from .target_text import TargetText, build_target_text, record_full_text_change


DUBBING_PLAN_SCHEMA_VERSION = "dubbing-plan-v1"


def upgrade_legacy_translation_artifact(
    legacy: Mapping[str, Any],
    units: list[dict[str, Any]],
    *,
    pronunciation_dictionary: PronunciationDictionary | None = None,
) -> dict[str, Any]:
    """Distribute approved legacy text without calling an AI or dropping words."""
    turns = legacy.get("turns")
    if not isinstance(turns, list) or not turns:
        raise ValueError("Legacy translation must contain approved turns")
    by_id = {str(item.get("segment_id")): item for item in turns}
    if None in by_id or "None" in by_id:
        raise ValueError("Legacy translation turns require segment IDs")
    unit_ids_by_phrase: dict[str, list[str]] = {}
    for unit in units:
        for phrase_id in unit["source_phrase_ids"]:
            unit_ids_by_phrase.setdefault(str(phrase_id), []).append(unit["unit_id"])
    missing = sorted(set(by_id) - set(unit_ids_by_phrase))
    if missing:
        raise ValueError(f"Dubbing units do not preserve legacy turn IDs: {missing}")

    per_phrase_chunks: dict[str, dict[str, str]] = {}
    for phrase_id, turn in by_id.items():
        approved = str(turn.get("translated_text", "")).strip()
        if not approved:
            raise ValueError(f"Legacy approved text is empty for {phrase_id}")
        target_units = unit_ids_by_phrase[phrase_id]
        weights = [
            max(1, len(str(next(unit for unit in units if unit["unit_id"] == unit_id)["source_text"])))
            for unit_id in target_units
        ]
        chunks = _split_approved_text(approved, weights)
        per_phrase_chunks[phrase_id] = dict(zip(target_units, chunks, strict=True))

    translated_units = []
    reconstructed: dict[str, list[str]] = {phrase_id: [] for phrase_id in by_id}
    for unit in units:
        pieces = []
        phrase_ids = [str(value) for value in unit["source_phrase_ids"]]
        for phrase_id in phrase_ids:
            if phrase_id not in per_phrase_chunks:
                raise ValueError(f"No approved legacy translation for source phrase {phrase_id}")
            piece = per_phrase_chunks[phrase_id].get(unit["unit_id"], "")
            if piece:
                pieces.append(piece)
                reconstructed[phrase_id].append(piece)
        approved_for_unit = " ".join(pieces).strip()
        if not approved_for_unit:
            raise ValueError(f"Legacy upgrade produced empty text for {unit['unit_id']}")
        target = (
            build_target_text(approved_for_unit, dictionary=pronunciation_dictionary)
            if pronunciation_dictionary is not None
            else TargetText.from_legacy_approved_text(approved_for_unit)
        )
        translated_units.append(
            {
                "unit_id": unit["unit_id"],
                **target.to_dict(),
                "legacy_source_phrase_ids": phrase_ids,
                "migration": {
                    "method": "deterministic_word_boundary_distribution",
                    "ai_called": False,
                    "approved_wording_preserved": True,
                    "human_review_required": len(unit_ids_by_phrase[phrase_ids[0]]) > 1,
                },
            }
        )
    for phrase_id, turn in by_id.items():
        before = _words(str(turn["translated_text"]))
        after = _words(" ".join(reconstructed[phrase_id]))
        if before != after:
            raise RuntimeError(f"Legacy translation content changed while splitting {phrase_id}")
    return {
        "schema_version": "three-text-translation-v1",
        "language": "zu-ZA",
        "source_schema_version": legacy.get("schema_version"),
        "source_artifact_sha256": _canonical_sha256(legacy),
        "ai_called": False,
        "approved_content_preserved": True,
        "units": translated_units,
    }


def normalize_translation_units(
    translation: Mapping[str, Any],
    units: list[dict[str, Any]],
    *,
    pronunciation_dictionary: PronunciationDictionary | None = None,
) -> dict[str, Any]:
    if "turns" in translation:
        return upgrade_legacy_translation_artifact(
            translation,
            units,
            pronunciation_dictionary=pronunciation_dictionary,
        )
    values = translation.get("units")
    if not isinstance(values, list):
        raise ValueError("Translation artifact contains neither turns nor units")
    by_id = {str(item.get("unit_id")): item for item in values}
    expected = [unit["unit_id"] for unit in units]
    if list(by_id) != expected:
        raise ValueError("Translated unit IDs do not match deterministic dubbing units")
    normalized = []
    for unit_id in expected:
        item = by_id[unit_id]
        faithful = str(item.get("faithful_translation", "")).strip()
        spoken = str(item.get("spoken_text", "")).strip()
        supplied_tts = str(item.get("tts_text", "")).strip()
        if not faithful or not spoken or not supplied_tts:
            raise ValueError(f"Translated unit {unit_id} is missing one of the three text forms")
        
        # Validate protected entity integrity
        expected_entities = item.get("protected_entities_expected", 0)
        restored_entities = item.get("protected_entities_restored", 0)
        missing_entities = item.get("protected_entities_missing", [])
        unexpected_entities = item.get("protected_entities_unexpected", [])
        
        if expected_entities != restored_entities:
            raise ValueError(
                f"Unit {unit_id} entity integrity failed: expected {expected_entities}, "
                f"restored {restored_entities}, missing {missing_entities}, "
                f"unexpected {unexpected_entities}"
            )
        if missing_entities:
            raise ValueError(f"Unit {unit_id} has missing protected entities: {missing_entities}")
        if unexpected_entities:
            raise ValueError(f"Unit {unit_id} has unexpected protected entities: {unexpected_entities}")
        
        protected_terms = tuple(str(value) for value in item.get("protected_entities_found", ()))
        changes = list(item.get("text_changes", ()))
        if faithful != spoken and not changes:
            timing_strategy = str(item.get("timing_strategy") or "timing-aware spoken delivery")
            detail = item.get("omitted_or_compressed_detail") or []
            change_type = "contraction" if len(spoken) < len(faithful) else "paraphrase"
            changes.append(
                record_full_text_change(
                    faithful,
                    spoken,
                    layer="faithful_to_spoken",
                    change_type=change_type,
                    reason=f"Claude timing strategy: {timing_strategy}; recorded detail: {detail}",
                    protected_terms=protected_terms,
                    human_review_required=True,
                )
            )
        target = build_target_text(
            faithful,
            spoken_text=spoken,
            dictionary=pronunciation_dictionary,
            changes=changes,
            protected_terms=protected_terms,
        )
        if pronunciation_dictionary is None and supplied_tts != target.tts_text:
            raise ValueError(
                f"Translated unit {unit_id} proposes spoken_text/tts_text differences "
                "without a reviewed pronunciation dictionary"
            )
        proposal_review = None
        if pronunciation_dictionary is not None and supplied_tts != target.tts_text:
            proposal_review = {
                "claude_tts_text_proposal": supplied_tts,
                "application_tts_text": target.tts_text,
                "applied": False,
                "reason": "Only reviewed application pronunciation entries may alter TTS text",
                "human_review_required": True,
            }
        normalized.append(
            {
                **item,
                **target.to_dict(),
                "unit_id": unit_id,
                "tts_proposal_review": proposal_review,
            }
        )
    return {
        "schema_version": "three-text-translation-v1",
        "language": "zu-ZA",
        "source_schema_version": translation.get("schema_version"),
        "source_artifact_sha256": _canonical_sha256(translation),
        "ai_called": bool(translation.get("ai_called", True)),
        "units": normalized,
        "seo": translation.get("seo", {}),
        "metadata": translation.get("metadata", {}),
    }


def build_dubbing_plan(
    *,
    job: Mapping[str, Any],
    source_media: Path,
    analysis_audio: Path,
    mix_source_audio: Path,
    unit_artifact: Mapping[str, Any],
    translation_artifact: Mapping[str, Any],
    output_path: Path,
    pronunciation_dictionary: PronunciationDictionary | None = None,
    speaker_references: Mapping[str, Any] | None = None,
    source_calibrations: Mapping[str, Any] | None = None,
    voice_selections: Mapping[str, Any] | None = None,
    azure_only: bool = False,
) -> dict[str, Any]:
    units = [dict(value) for value in unit_artifact.get("units", [])]
    if not units:
        raise ValueError("Dubbing plan requires deterministic units")
    translation = normalize_translation_units(
        translation_artifact,
        units,
        pronunciation_dictionary=pronunciation_dictionary,
    )
    translated_by_id = {item["unit_id"]: item for item in translation["units"]}
    planned_units = []
    for unit in units:
        target = translated_by_id[unit["unit_id"]]
        planned_units.append(
            {
                **unit,
                "faithful_translation": target["faithful_translation"],
                "spoken_text": target["spoken_text"],
                "tts_text": target["tts_text"],
                "text_changes": target.get("text_changes", []),
                "pronunciation_substitutions": target.get("pronunciation_substitutions", []),
                "protected_entities_expected": target.get("protected_entities_expected", 0),
                "protected_entities_restored": target.get("protected_entities_restored", 0),
                "pronunciation_calibration_required": target.get("pronunciation_calibration_required", False),
                "azure_tts": {"status": "pending", "attempts": 0},
                "openvoice": {"status": "skipped" if azure_only else "pending", "attempts": 0},
                "alignment": {"status": "pending"},
                "review_flags": list(target.get("human_review_flags", [])),
            }
        )
    plan = {
        "schema_version": DUBBING_PLAN_SCHEMA_VERSION,
        "job_id": job["job_id"],
        "configuration": {
            "dubbing_mode": "azure_only" if azure_only else "standard",
            "gpu_required": False if azure_only else True,
            "providers": {
                "speech_generation": "azure_tts",
                "voice_conversion": "none" if azure_only else "openvoice",
                "alignment_source": "azure_tts" if azure_only else "openvoice",
            },
        },
        "source_media": {"path": str(source_media), "sha256": checksum(source_media)},
        "analysis_audio": {"path": str(analysis_audio), "sha256": checksum(analysis_audio)},
        "mix_source_audio": {"path": str(mix_source_audio), "sha256": checksum(mix_source_audio)},
        "source_transcript_identity": {
            "schema_version": unit_artifact.get("source_schema_version"),
            "sha256": unit_artifact.get("source_sha256"),
        },
        "dubbing_units_identity": {
            "schema_version": unit_artifact.get("schema_version"),
            "sha256": unit_artifact.get("artifact_sha256"),
        },
        "translation_identity": {
            "schema_version": translation["schema_version"],
            "source_schema_version": translation.get("source_schema_version"),
            "source_sha256": translation["source_artifact_sha256"],
            "ai_called_during_upgrade": translation["ai_called"],
        },
        "pronunciation_dictionary": (
            {
                "schema_version": pronunciation_dictionary.schema_version,
                "version": pronunciation_dictionary.dictionary_version,
                "sha256": pronunciation_dictionary.sha256,
            }
            if pronunciation_dictionary
            else None
        ),
        "entity_registry": job.get("media", {}).get("entity_registry_sha256"),
        "entity_bindings": job.get("media", {}).get("entity_bindings_sha256"),
        "speakers": {
            speaker_id: {
                "reference": dict((speaker_references or {}).get(speaker_id, {})),
                "selected_azure_voice": dict((voice_selections or {}).get(speaker_id, {})),
                "job_local_identity": True,
            }
            for speaker_id in sorted({unit["speaker_id"] for unit in units})
        },
        "azure_source_calibrations": dict(source_calibrations or {}),
        "units": planned_units,
        "background": {"status": "pending", "mode": None},
        "compatibility": {"state": "pending", "human_review_required": True},
        "review_flags": ["Human voice-use, editorial, factual, audio, and fluent isiZulu review required"],
    }
    safe_plan = redact(plan)
    if safe_plan != plan:
        raise ValueError("Dubbing plan contained a credential-like value")
    plan["artifact_sha256"] = _canonical_sha256(plan)
    atomic_write_json(output_path, plan)
    return plan


def _split_approved_text(text: str, weights: list[int]) -> list[str]:
    if len(weights) == 1:
        return [text.strip()]
    words = text.split()
    if len(words) < len(weights):
        raise ValueError("Approved translation has fewer words than derived dubbing units")
    total_weight = sum(weights)
    boundaries = [0]
    cumulative = 0
    for index, weight in enumerate(weights[:-1], 1):
        cumulative += weight
        target = round(len(words) * cumulative / total_weight)
        minimum = boundaries[-1] + 1
        maximum = len(words) - (len(weights) - index)
        boundaries.append(max(minimum, min(maximum, target)))
    boundaries.append(len(words))
    return [" ".join(words[boundaries[index] : boundaries[index + 1]]) for index in range(len(weights))]


def _words(value: str) -> list[str]:
    return value.split()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "DUBBING_PLAN_SCHEMA_VERSION",
    "build_dubbing_plan",
    "normalize_translation_units",
    "upgrade_legacy_translation_artifact",
]
