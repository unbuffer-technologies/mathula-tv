"""Two-phase production dubbing orchestration built on existing job state."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from .ai_provider import (
    AIProvider,
    TURN_REPAIR_SCHEMA,
    build_full_clip_payload,
    build_semantic_language_payload,
)
from .alignment import align_converted_unit
from .artifacts import DubbingArtifacts
from .atomic_io import atomic_write_json, read_json
from .autocorrect_stage import require_authoritative_transcript
from .azure_stt import SpeechBackend
from .azure_tts import (
    AzureTTSBackend,
    AzureTTSIdempotencyError,
    AzureTTSRequest,
    AzureTTSResult,
)
from .background import (
    ResidualSpeechDetector,
    SeparationProvider,
    prepare_background,
    residual_dialogue_qc,
    write_background_manifest,
)
from .compatibility import (
    CompatibilityThresholds,
    UnitCompatibilityObservation,
    evaluate_compatibility,
    write_compatibility_report,
)
from .config import Settings
from .consent import build_voice_rights_review, require_generation_rights
from .dubbing_plan import build_dubbing_plan, normalize_translation_units
from .dubbing_units import DubbingUnitConfig, build_dubbing_units
from .entity_registry import (
    ENTITY_BINDINGS_SCHEMA,
    ENTITY_MATCHER_VERSION,
    ENTITY_REGISTRY_SCHEMA,
    EntityBindingsArtifact,
    EntityRegistry,
    protect_text_with_placeholders,
    restore_display_text,
    restore_tts_text,
    validate_placeholder_integrity,
)
from .errors import (
    AIInvalidStructuredOutput,
    CompatibilityGateFailure,
    TranslationTimingFailure,
)
from .gcs_store import GCSStore
from .local_store import LocalStore
from .intelligibility import IntelligibilityAuditor
from .job_store import JobStore
from .logging_utils import redact
from .media import checksum, prepare_source_derivatives
from .mixing import mix_final_buses, render_dialogue_tracks
from .models import JobManifest, utcnow
from .multivariant_translation import (
    PROMPT_VERSION as MULTIVARIANT_PROMPT_VERSION,
    build_job_translation_request,
    build_translation_batch_requests,
    merge_multivariant_batch_responses,
    normalize_multivariant_response,
    prepare_installed_translation,
    validate_multivariant_response,
    write_translation_artifacts,
)
from .one_call_translation import translate_full_transcript_once
from .openvoice_worker import (
    OPENVOICE_ASSET_MANIFEST_SCHEMA,
    OPENVOICE_SOURCE_EMBEDDING_MANIFEST_SCHEMA,
    OPENVOICE_TARGET_EMBEDDING_MANIFEST_SCHEMA,
    OPENVOICE_VOICE_CANDIDATES_SCHEMA,
    AzureSourceCalibrationSpec,
    OpenVoiceAssetPlan,
    OpenVoicePlan,
    OpenVoiceSourceAsset,
    OpenVoiceSpeakerAsset,
    OpenVoiceUnit,
    OpenVoiceVoiceCandidate,
    write_openvoice_asset_plan,
)
from .pronunciation import (
    PronunciationDictionary,
    PronunciationEntry,
    build_initialism_ssml_parts,
    with_default_organisation_initialisms,
    with_web_researched_organisation_pronunciations,
)
from .quality import inspect_wav
from .qc import write_qc_report
from .references import build_reference_reel, select_reference_candidates
from .rendering import render_review_mp4
from .subtitles import write_subtitles
from .tts_timing import AzureTTSTimingSearch
from .voice_selection import rank_voice_candidates
from .voice_prosody import voice_and_base_prosody


REVIEWED_AZURE_CALIBRATION_TEXT = (
    "Namuhla sixoxa ngezindaba ezibalulekile ezithinta imiphakathi yaseNingizimu Afrika. "
    "Sicela nilalele ngokucophelela amagama abantu, amagama ezindawo, izinombolo, izinsuku, "
    "amaphesenti kanye nemali eshiwo kulo mbiko. Izintatheli zethu zihlola amaqiniso, zigcine "
    "ukuthi ubani oshilo, futhi zehlukanise izinsolo ebufakazini obuqinisekisiwe. Uma udaba "
    "lusadingidwa, siyakusho lokho ngokucacile ngaphandle kokwandisa isimangalo."
)
REVIEWED_SPEAKER_VOICE_CALIBRATION_SENTENCE = (
    "Sawubona, namuhla ngikhuluma ngokucacile ukuze kuhlolwe izwi, isigqi, "
    "ukuphimisa kanye nokuqondakala kwalo mbiko."
)




def _timing_repair_generation_sha256(translation: Mapping[str, Any], translation_path: Path) -> str:
    """Return the stable repair-budget generation for one full translation.

    A newly generated full translation receives a new budget. All bounded timing
    repairs derived from that translation keep the same generation so repeated
    repairs cannot evade the per-unit limit merely by changing the wording.
    """
    existing = str(translation.get("timing_repair_generation_sha256") or "").strip()
    if re.fullmatch(r"[0-9a-f]{64}", existing):
        return existing
    return checksum(translation_path)


def _timing_repair_stage(unit_id: str, generation_sha256: str) -> str:
    return f"gpt_timing_repair:{unit_id}:{generation_sha256[:16]}"


def _legacy_timing_repair_stage(unit_id: str, generation_sha256: str) -> str:
    return f"claude_timing_repair:{unit_id}:{generation_sha256[:16]}"


def _ai_call_count(job: JobManifest) -> int:
    return sum(
        count
        for stage, count in job.attempt_counters.items()
        if stage.startswith(("gpt_", "ai_", "claude_"))
    )


def _clear_azure_timing_repair_markers(*marker_paths: Path) -> None:
    """Remove one-shot repair markers after Azure TTS fully succeeds."""
    for marker_path in marker_paths:
        marker_path.unlink(missing_ok=True)

def _synthesize_azure_candidate(
    backend: AzureTTSBackend,
    request: AzureTTSRequest,
    output_path: Path,
    *,
    force: bool,
) -> AzureTTSResult:
    """Reuse matching Azure audio and refresh only a stale request candidate.

    Cache corruption, incomplete artifacts, and manifest-integrity failures remain
    blocking. Only a valid artifact created for a different deterministic request
    is replaced, and only for the candidate currently being evaluated.
    """
    try:
        return backend.synthesize(request, output_path, force=force)
    except AzureTTSIdempotencyError as exc:
        if force or exc.conflict_kind != "request_mismatch":
            raise
        return backend.synthesize(request, output_path, force=True)


class ProductionDubbingPipeline:
    """Server-owned stages; GPU inference remains in the leased worker."""

    def __init__(self, settings: Settings, gcs: GCSStore | None = None, stt_backend: SpeechBackend | None = None):
        self.settings = settings
        self.jobs = JobStore(settings.work_dir)
        # Local filesystem is the normal artifact backend.  Colab/remote-worker
        # commands opt into GCS by passing a GCSStore explicitly.
        self.gcs = gcs if gcs is not None else LocalStore(settings.work_dir)
        self._stt_backend = stt_backend

    def paths(self, job_id: str) -> DubbingArtifacts:
        try:
            target_locale = self.jobs.load(job_id).target_language
        except Exception:
            # Offline unit tests and pre-manifest migration helpers may create
            # artifact paths before a job manifest exists. Preserve the historic
            # isiZulu default in that narrow case.
            target_locale = "zu-ZA"
        return DubbingArtifacts(
            self.jobs.job_dir(job_id),
            target_locale=target_locale,
        )

    def inspect(self, job: JobManifest) -> dict[str, Any]:
        paths = self.paths(job.job_id)
        known = {
            "manifest": self.jobs.path(job.job_id),
            "source_media": Path(job.local_source_path),
            "analysis_audio": paths.analysis_audio,
            "mix_source_audio": paths.mix_source_audio,
            "raw_transcript": self.jobs.job_dir(job.job_id) / "analysis/transcript_en_raw.json",
            "autocorrection_state": self.jobs.job_dir(job.job_id) / "analysis/autocorrection_state.json",
            "transcript": self.jobs.job_dir(job.job_id) / "analysis/transcript_en.json",
            "multivariant_translation": paths.raw_multivariant_translation,
            "legacy_translation": self.jobs.job_dir(job.job_id) / "translation/transcript_zu.json",
            "dubbing_units": paths.dubbing_units,
            "dubbing_plan": paths.plan,
            "azure_tts_manifest": paths.azure_manifest,
            "openvoice_asset_plan": paths.openvoice_asset_plan,
            "openvoice_asset_manifest": paths.openvoice_asset_manifest,
            "openvoice_manifest": paths.openvoice_manifest,
            "compatibility_report": paths.compatibility_json,
            "final_video": paths.final_video,
        }
        return {
            "job_id": job.job_id,
            "state": job.state,
            "schema_version": job.schema_version,
            "compatibility_gate": job.compatibility_gate,
            "review_readiness": job.review_readiness,
            "providers": job.providers,
            "completed_stages": job.completed_stages,
            "attempt_counters": job.attempt_counters,
            "artifacts": {
                label: {
                    "path": str(path),
                    "exists": path.is_file(),
                    "size": path.stat().st_size if path.is_file() else 0,
                    "sha256": checksum(path) if path.is_file() else None,
                }
                for label, path in known.items()
            },
            "live_operation_performed": False,
        }

    def validate_job(self, job: JobManifest) -> dict[str, Any]:
        """Verify materialised production inputs and recorded hashes without mutation."""

        paths = self.paths(job.job_id)
        checks: list[dict[str, Any]] = []

        def verify(label: str, path: Path, expected: str | None, *, required: bool) -> None:
            if not path.is_file() or path.stat().st_size == 0:
                checks.append(
                    {
                        "label": label,
                        "path": str(path),
                        "status": "missing" if required else "not_present",
                        "required": required,
                    }
                )
                return
            actual = checksum(path)
            checks.append(
                {
                    "label": label,
                    "path": str(path),
                    "status": "verified" if expected is None or actual == expected else "hash_mismatch",
                    "required": required,
                    "sha256": actual,
                    "expected_sha256": expected,
                }
            )

        verify("job_manifest", self.jobs.path(job.job_id), None, required=True)
        verify("source_media", Path(job.local_source_path), job.source_checksum, required=True)
        analysis_expected = (
            job.media.get("analysis_mono", {}).get("sha256")
            if isinstance(job.media.get("analysis_mono"), Mapping)
            else job.media.get("audio_checksum")
        )
        mix_expected = (
            job.media.get("mix_source_stereo", {}).get("sha256")
            if isinstance(job.media.get("mix_source_stereo"), Mapping)
            else None
        )
        verify("analysis_mono", paths.analysis_audio, analysis_expected, required=True)
        verify("mix_source_stereo", paths.mix_source_audio, mix_expected, required=True)
        verify(
            "validated_translation",
            (
                paths.job_root / "translation/transcript_zu.json"
                if job.state == "synthesis_queued"
                else paths.raw_multivariant_translation
            ),
            (
                job.media.get("transcript_zu_sha256")
                if job.state == "synthesis_queued"
                else job.media.get("translation_multivariant_sha256")
            ),
            required=job.state in {
                "synthesis_queued",
                "translation_ready",
                "azure_tts_queued",
                "azure_tts_running",
                "azure_tts_ready",
            },
        )
        verify(
            "validated_seo",
            paths.job_root / "translation/youtube_zu.json",
            job.media.get("youtube_zu_sha256"),
            required=job.state == "synthesis_queued",
        )

        for label, manifest_path, unit_key, path_field, hash_field in (
            ("azure_tts", paths.azure_manifest, "units", "output_path", "sha256"),
            ("alignment", paths.alignment_manifest, "units", "output_path", "sha256"),
        ):
            if not manifest_path.is_file():
                continue
            value = read_json(manifest_path)
            for item in value.get(unit_key, []):
                output_value = item.get(path_field)
                if output_value:
                    verify(
                        f"{label}:{item.get('unit_id') or item.get('turn_id')}",
                        Path(str(output_value)),
                        item.get(hash_field),
                        required=True,
                    )
        if paths.openvoice_manifest.is_file():
            openvoice = read_json(paths.openvoice_manifest)
            for unit_id, record in openvoice.get("units", {}).items():
                result = record.get("result", {})
                verify(
                    f"openvoice:{unit_id}",
                    paths.openvoice_turn(str(unit_id)),
                    result.get("output_sha256"),
                    required=record.get("status") == "completed",
                )
        if paths.openvoice_asset_plan.is_file():
            asset_plan_value = read_json(paths.openvoice_asset_plan)
            asset_plan = OpenVoiceAssetPlan.from_dict(asset_plan_value)
            verify(
                "openvoice_asset_plan",
                paths.openvoice_asset_plan,
                job.media.get("openvoice_asset_plan_sha256"),
                required=False,
            )
            asset_manifest = (
                read_json(paths.openvoice_asset_manifest)
                if paths.openvoice_asset_manifest.is_file()
                else {}
            )
            source_records = asset_manifest.get("sources", {})
            speaker_records = asset_manifest.get("speakers", {})
            for source in asset_plan.sources:
                result = source_records.get(source.azure_voice, {}).get("result", {})
                verify(
                    f"openvoice_source_embedding:{source.azure_voice}",
                    self._resolve_job_path(paths, source.source_embedding_path),
                    result.get("source_embedding_sha256"),
                    required=asset_manifest.get("state") == "completed",
                )
            for speaker in asset_plan.speakers:
                record = speaker_records.get(speaker.speaker_id, {})
                result = record.get("result", {})
                verify(
                    f"openvoice_target_embedding:{speaker.speaker_id}",
                    self._resolve_job_path(paths, speaker.target_embedding_path),
                    result.get("target_embedding_sha256"),
                    required=asset_manifest.get("state") == "completed",
                )
                for candidate in speaker.voice_candidates:
                    candidate_result = (
                        record.get("candidates", {})
                        .get(candidate.candidate_id, {})
                        .get("result", {})
                    )
                    verify(
                        f"openvoice_voice_candidate:{candidate.candidate_id}",
                        self._resolve_job_path(paths, candidate.output_path),
                        candidate_result.get("output_sha256"),
                        required=asset_manifest.get("state") == "completed",
                    )
        for label, path in (
            ("dubbing_plan", paths.plan),
            ("final_mix", paths.final_mix),
            ("final_video", paths.final_video),
        ):
            verify(label, path, None, required=False)

        failures = [
            item
            for item in checks
            if item["status"] == "hash_mismatch"
            or (item["required"] and item["status"] == "missing")
        ]
        if failures:
            raise ValueError(
                "Job validation failed: "
                + ", ".join(f"{item['label']}={item['status']}" for item in failures)
            )
        return {
            "schema_version": "production-job-validation-v1",
            "job_id": job.job_id,
            "state": job.state,
            "valid": True,
            "checks": checks,
            "mutated": False,
        }

    def migrate(
        self,
        job: JobManifest,
        *,
        dry_run: bool = False,
        allow_legacy_review_reset: bool = False,
    ) -> dict[str, Any]:
        return self.jobs.migrate_synthesis_queued(
            job,
            dry_run=dry_run,
            allow_legacy_review_reset=allow_legacy_review_reset,
        )

    def ensure_source_derivatives(self, job: JobManifest, *, force: bool = False) -> dict[str, Any]:
        paths = self.paths(job.job_id)
        if paths.analysis_audio.is_file() and paths.mix_source_audio.is_file() and not force:
            return job.media
        media = prepare_source_derivatives(
            Path(job.local_source_path),
            paths.analysis_audio,
            paths.mix_source_audio,
            max_duration_seconds=self.settings.max_source_duration_minutes * 60,
        )
        job.media.update(media)
        job.objects.update(
            local_audio=str(paths.analysis_audio),
            analysis_mono=str(paths.analysis_audio),
            mix_source_stereo=str(paths.mix_source_audio),
        )
        self.jobs.save(job)
        return media

    def build_units(self, job: JobManifest, *, force: bool = False) -> dict[str, Any]:
        paths = self.paths(job.job_id)
        authoritative = require_authoritative_transcript(
            work_dir=self.settings.work_dir,
            job_id=job.job_id,
        )
        transcript_path = Path(authoritative["path"])

        # Dubbing units are derived from the effective transcript, which may be
        # rewritten in place by the autocorrection workflow.  Never reuse a
        # cached unit artifact unless its deterministic source identity still
        # matches the current effective transcript.  This prevents corrected
        # phrases such as "Madlanga ward" from falling back to stale Azure STT
        # text such as "malanga ward" during translation.
        transcript = read_json(transcript_path)
        candidate = build_dubbing_units(transcript, DubbingUnitConfig())
        candidate["source_transcript_sha256"] = authoritative["sha256"]
        candidate["source_autocorrection_state_sha256"] = authoritative[
            "state_sha256"
        ]
        if paths.dubbing_units.is_file() and not force:
            existing = read_json(paths.dubbing_units)
            if (
                existing.get("source_sha256") == candidate.get("source_sha256")
                and existing.get("config") == candidate.get("config")
                and existing.get("source_transcript_sha256")
                == authoritative["sha256"]
                and existing.get("source_autocorrection_state_sha256")
                == authoritative["state_sha256"]
            ):
                return existing

        artifact = candidate
        atomic_write_json(paths.dubbing_units, artifact)
        job.media["dubbing_units_sha256"] = checksum(paths.dubbing_units)
        job.media["dubbing_unit_count"] = len(artifact["units"])
        job.providers["authoritative_diarization"] = "azure"
        self.jobs.save(job)
        return artifact

    def translation_is_current(self, job: JobManifest) -> bool:
        """Return whether translation was derived from the current corrected transcript."""

        paths = self.paths(job.job_id)
        if not paths.three_text_translation.is_file():
            return False
        authoritative = require_authoritative_transcript(
            work_dir=self.settings.work_dir,
            job_id=job.job_id,
        )
        units = self.build_units(job, force=False)
        translation = read_json(paths.three_text_translation)
        return (
            translation.get("source_transcript_sha256")
            == authoritative["sha256"]
            and translation.get("source_dubbing_units_sha256")
            == checksum(paths.dubbing_units)
            and bool(units.get("units"))
        )

    def bind_entities(self, job: JobManifest, *, force: bool = False) -> dict[str, Any]:
        """Bind entities to dubbing units and create job-local binding artifact."""
        paths = self.paths(job.job_id)
        registry = EntityRegistry()
        transcript_path = self.jobs.job_dir(job.job_id) / "analysis/transcript_en.json"

        # Build or refresh units before validating entity bindings.  The
        # effective transcript can change through autocorrection while keeping
        # the same path, so transcript-path identity alone is insufficient.
        units = self.build_units(job, force=force)
        units_sha256 = checksum(paths.dubbing_units)

        # Check if bindings already exist and are valid for both the effective
        # transcript and the exact dubbing-unit artifact used for translation.
        if paths.entity_bindings.is_file() and not force:
            existing = read_json(paths.entity_bindings)
            existing_registry_sha = existing.get("registry_sha256")
            existing_transcript_sha = existing.get("source_transcript_sha256")
            existing_units_sha = existing.get("source_dubbing_units_sha256")
            if (
                existing_registry_sha == registry.sha256
                and existing_transcript_sha == checksum(transcript_path)
                and existing_units_sha == units_sha256
            ):
                return existing
        
        # Match entities for each unit
        per_unit_bindings = {}
        protected_source_text_per_unit = {}
        placeholder_mapping = {}
        entities_found_global = set()
        unresolved_ambiguous = set()
        
        for unit in units["units"]:
            unit_id = unit["unit_id"]
            source_text = unit.get("source_text", "")
            
            # Match entities
            matches = registry.match_entities(source_text, unit_id)
            
            # Protect text with placeholders
            protected_text, bindings = protect_text_with_placeholders(source_text, matches)
            
            per_unit_bindings[unit_id] = {
                "unit_id": unit_id,
                "matches": [asdict(m) for m in matches],
                "protected_text": protected_text,
                "placeholder_count": len(bindings),
            }
            
            protected_source_text_per_unit[unit_id] = protected_text
            placeholder_mapping.update(bindings)
            entities_found_global.update(m.entity_id for m in matches)
        
        # Create binding artifact
        artifact = EntityBindingsArtifact(
            schema_version=ENTITY_BINDINGS_SCHEMA,
            job_id=job.job_id,
            registry_path=str(registry._registry_path),
            registry_sha256=registry.sha256,
            registry_schema_version=ENTITY_REGISTRY_SCHEMA,
            source_transcript_sha256=checksum(transcript_path),
            generated_at=utcnow(),
            entities_found_global=tuple(sorted(entities_found_global)),
            per_unit_bindings=per_unit_bindings,
            unresolved_ambiguous_matches=tuple(sorted(unresolved_ambiguous)),
            protected_source_text_per_unit=protected_source_text_per_unit,
            placeholder_mapping=placeholder_mapping,
            summary_counts={
                "total_entities": len(entities_found_global),
                "total_units": len(units["units"]),
                "total_placeholders": len(placeholder_mapping),
                "alias_collisions": len(registry.alias_collisions),
            },
        )
        
        # Write artifact.  Record the exact unit artifact identity so a later
        # autocorrection cannot leave valid-looking but stale entity bindings.
        paths.entity_bindings.parent.mkdir(parents=True, exist_ok=True)
        artifact_dict = artifact.to_dict()
        artifact_dict["matcher_version"] = ENTITY_MATCHER_VERSION
        artifact_dict["source_dubbing_units_sha256"] = units_sha256
        atomic_write_json(paths.entity_bindings, artifact_dict)
        
        # Update job manifest
        job.media["entity_bindings_sha256"] = checksum(paths.entity_bindings)
        job.media["entity_matcher_version"] = ENTITY_MATCHER_VERSION
        job.media["entity_registry_sha256"] = registry.sha256
        self.jobs.save(job)
        
        return artifact.to_dict()

    def _semantic_analysis_path(self, job: JobManifest) -> Path:
        return self.paths(job.job_id).job_root / "analysis/semantic_language_features.json"

    def _semantic_pronunciation_dictionary(
        self,
        job: JobManifest,
        analysis: Mapping[str, Any],
    ) -> PronunciationDictionary:
        base = self._pronunciation_dictionary(job)
        retained = [
            entry
            for entry in base.job_overrides
            if entry.source not in {"gpt_semantic_analysis", "claude_semantic_analysis"}
        ]
        generated: list[PronunciationEntry] = []
        seen: set[str] = {entry.display_text.casefold() for entry in retained}
        for span in analysis.get("spans", []):
            if not isinstance(span, Mapping):
                continue
            policy = str(span.get("policy"))
            if policy not in {"preserve_verbatim", "spell_out", "human_review"}:
                continue
            display = str(span.get("display_text", "")).strip()
            if not display or display.casefold() in seen:
                continue
            confidence = float(span.get("confidence", 0))
            proposed_tts = str(span.get("tts_text", display)).strip() or display
            safe_alias = (
                policy == "spell_out"
                and str(span.get("category")) in {
                    "letter_name", "acronym", "wordplay", "deliberate_code_switch"
                }
                and confidence >= 0.85
            )
            tts_text = proposed_tts if safe_alias else display
            generated.append(
                PronunciationEntry(
                    display_text=display,
                    spoken_text=display,
                    tts_text=tts_text,
                    language=job.target_language,
                    source="gpt_semantic_analysis",
                    confidence=confidence,
                    kind="english_code_switch",
                    notes=(
                        f"semantic_span_id={span.get('span_id')}",
                        f"policy={policy}",
                        str(span.get("rationale", "")),
                    ),
                )
            )
            seen.add(display.casefold())
        dictionary = base.with_job_overrides(job.job_id, [*retained, *generated])
        atomic_write_json(self.paths(job.job_id).pronunciation_dictionary, dictionary.to_dict())
        return dictionary

    def _translation_pronunciation_dictionary(
        self,
        job: JobManifest,
        translated_units: list[Mapping[str, Any]],
    ) -> PronunciationDictionary:
        """Create safe job-local TTS aliases from the one-call translation response."""

        base = self._pronunciation_dictionary(job)
        obsolete_sources = {
            "gpt_semantic_analysis",
            "gpt_full_context_translation",
            "claude_semantic_analysis",
            "claude_full_context_translation",
        }
        retained = [entry for entry in base.job_overrides if entry.source not in obsolete_sources]
        generated: list[PronunciationEntry] = []
        seen = {entry.display_text.casefold() for entry in (*base.entries, *retained)}
        for unit in translated_units:
            spoken_text = str(unit.get("spoken_text", ""))
            for feature in unit.get("language_features", []):
                if not isinstance(feature, Mapping):
                    continue
                policy = str(feature.get("policy", ""))
                if policy not in {"preserve_verbatim", "spell_out", "preserve_and_flag"}:
                    continue
                display = str(feature.get("display_text") or feature.get("source_text") or "").strip()
                if not display or display.casefold() in seen:
                    continue
                if display.casefold() not in spoken_text.casefold():
                    continue
                tts_text = (
                    str(feature.get("tts_text") or display).strip()
                    if policy == "spell_out"
                    else display
                )
                generated.append(
                    PronunciationEntry(
                        display_text=display,
                        spoken_text=display,
                        tts_text=tts_text,
                        language=job.target_language,
                        source="gpt_full_context_translation",
                        confidence=0.95 if policy == "spell_out" else 0.90,
                        kind="english_code_switch",
                        notes=(
                            f"unit_id={unit.get('unit_id')}",
                            f"policy={policy}",
                            str(feature.get("reason", "")),
                        ),
                    )
                )
                seen.add(display.casefold())
        dictionary = base.with_job_overrides(job.job_id, [*retained, *generated])
        atomic_write_json(self.paths(job.job_id).pronunciation_dictionary, dictionary.to_dict())
        return dictionary

    def _write_translation_language_features(
        self,
        job: JobManifest,
        translated_units: list[Mapping[str, Any]],
        metadata: Mapping[str, Any],
    ) -> dict[str, Any]:
        spans: list[dict[str, Any]] = []
        for unit in translated_units:
            for feature in unit.get("language_features", []):
                if not isinstance(feature, Mapping):
                    continue
                spans.append(
                    {
                        "span_id": f"span_{len(spans) + 1:04d}",
                        "unit_id": str(unit.get("unit_id")),
                        "source_text": str(feature.get("source_text", "")),
                        "policy": str(feature.get("policy", "")),
                        "display_text": str(feature.get("display_text", "")),
                        "tts_text": str(feature.get("tts_text", "")),
                        "reason": str(feature.get("reason", "")),
                    }
                )
        authoritative = require_authoritative_transcript(
            work_dir=self.settings.work_dir,
            job_id=job.job_id,
        )
        artifact = {
            "schema_version": "semantic-language-features-v2",
            "method": "single_full_context_translation_call",
            "source_transcript_sha256": authoritative["sha256"],
            "source_autocorrection_state_sha256": authoritative["state_sha256"],
            "provider_metadata": dict(metadata),
            "spans": spans,
        }
        atomic_write_json(self._semantic_analysis_path(job), artifact)
        return artifact

    def analyze_semantic_language(
        self,
        job: JobManifest,
        provider: AIProvider,
        *,
        transcript: Mapping[str, Any],
        units: Mapping[str, Any],
        context: list[Mapping[str, Any]],
        glossary: Mapping[str, Any],
        force: bool = False,
    ) -> dict[str, Any]:
        path = self._semantic_analysis_path(job)
        authoritative = require_authoritative_transcript(
            work_dir=self.settings.work_dir,
            job_id=job.job_id,
        )
        if path.is_file() and not force:
            existing = read_json(path)
            if (
                existing.get("source_transcript_sha256")
                == authoritative["sha256"]
            ):
                return existing
        if _ai_call_count(job) >= self.settings.max_ai_calls_per_job:
            raise RuntimeError("MATHULA_TV_MAX_AI_CALLS_PER_JOB reached before semantic analysis")
        payload = build_semantic_language_payload(
            transcript=transcript,
            dubbing_units=units,
            speaker_roles=transcript.get("speaker_roles", {}),
            job_context=context,
            terminology_glossary=glossary,
        )
        attempt = self.jobs.begin_attempt(job, "gpt_semantic_analysis", forced=force)
        try:
            response = provider.analyze_semantic_language(payload)
            artifact = {
                **response.data,
                "metadata": response.metadata.to_dict(),
            }
            atomic_write_json(path, artifact)
            self.jobs.finish_attempt(job, "gpt_semantic_analysis", attempt, "completed")
            return artifact
        except Exception as exc:
            job.last_error = self._safe_error("gpt_semantic_analysis", exc)
            self.jobs.finish_attempt(job, "gpt_semantic_analysis", attempt, "failed")
            raise

    def translate(
        self,
        job: JobManifest,
        provider: AIProvider,
        *,
        force: bool = False,
        batch_size: int | None = None,
        context_units: int | None = None,
        request_timeout_seconds: float | None = None,
        batch_max_retries: int | None = None,
        max_provider_calls: int | None = None,
        parallelism: int | None = None,
        restart_batches: bool = False,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Translate the complete source-bound timeline through Azure OpenAI GPT.

        Cloud translation uses the exact request, rendered system prompt,
        rendered user prompt and response schema exported for manual overrides.
        No batching, provider retry or automatic structured-output repair is
        permitted by this stage. Non-GPT translation providers are rejected.
        """
        try:
            return translate_full_transcript_once(
                self,
                job,
                provider,
                force=force,
                batch_size=batch_size,
                context_units=context_units,
                request_timeout_seconds=request_timeout_seconds,
                batch_max_retries=batch_max_retries,
                max_provider_calls=max_provider_calls,
                parallelism=parallelism,
                restart_batches=restart_batches,
                progress=progress,
            )
        except AIInvalidStructuredOutput as original_error:
            # ``translate_full_transcript_once`` owns the immutable run
            # directory and writes failure.json before re-raising. The provider
            # now includes the complete parsed candidate in that error. Try one
            # local, source-bound metadata recovery before allowing another paid
            # provider call. No spoken text is generated or edited here.
            refreshed = self.jobs.load(job.job_id)
            try:
                recovered = self._recover_failed_translation_response(
                    refreshed,
                    missing_ok=True,
                    dry_run=False,
                )
            except Exception as recovery_error:
                if progress is not None:
                    progress(
                        {
                            "stage": "local_recovery",
                            "status": "warning",
                            "message": (
                                "GPT response was preserved, but local validation "
                                f"recovery did not pass: {recovery_error}"
                            ),
                        }
                    )
                raise original_error from recovery_error
            if recovered is None:
                raise
            if progress is not None:
                progress(
                    {
                        "stage": "local_recovery",
                        "status": "completed",
                        "message": (
                            "Installed the preserved GPT response after safe local "
                            "metadata normalization; no second provider call was made"
                        ),
                        "current": 1,
                        "total": 1,
                    }
                )
            return recovered["translation"]

    def _translation_failure_run_dir(
        self,
        job: JobManifest,
        *,
        run_dir: Path | None = None,
    ) -> Path | None:
        cloud_root = self.jobs.job_dir(job.job_id) / "translation" / "cloud_runs"
        cloud_root_resolved = cloud_root.resolve()
        if run_dir is not None:
            candidate = run_dir.resolve()
            if candidate.parent != cloud_root_resolved:
                raise ValueError(
                    "Translation recovery --run-dir must be one direct child of "
                    f"{cloud_root}"
                )
            return candidate

        preferred = str(job.media.get("translation_run_directory") or "").strip()
        ordered: list[Path] = []
        if preferred:
            preferred_path = Path(preferred).resolve()
            if preferred_path.parent == cloud_root_resolved:
                ordered.append(preferred_path)
        if cloud_root.is_dir():
            ordered.extend(
                sorted(
                    (path.resolve() for path in cloud_root.iterdir() if path.is_dir()),
                    reverse=True,
                )
            )
        seen: set[Path] = set()
        for candidate in ordered:
            if candidate in seen:
                continue
            seen.add(candidate)
            if (candidate / "failure.json").is_file():
                return candidate
        return None

    def _recover_failed_translation_response(
        self,
        job: JobManifest,
        *,
        run_dir: Path | None = None,
        missing_ok: bool = False,
        dry_run: bool = False,
    ) -> dict[str, Any] | None:
        selected_run = self._translation_failure_run_dir(job, run_dir=run_dir)
        if selected_run is None:
            if missing_ok:
                return None
            raise ValueError("No failed full-transcript GPT run is available")

        failure_path = selected_run / "failure.json"
        request_path = selected_run / "translation_request.json"
        if not request_path.is_file():
            raise ValueError(
                f"Failed translation run is missing its source-bound request: {request_path}"
            )
        failure = read_json(failure_path)
        error = failure.get("error") if isinstance(failure, Mapping) else None
        details = error.get("details") if isinstance(error, Mapping) else None
        details = details if isinstance(details, Mapping) else {}
        candidate = details.get("candidate_output")
        if not isinstance(candidate, Mapping):
            if missing_ok:
                return None
            raise ValueError(
                "The failed run predates response preservation and has no recoverable "
                "candidate; one new GPT call is unavoidable"
            )

        request = read_json(request_path)
        current_request, _source_paths = build_job_translation_request(
            job_root=self.jobs.job_dir(job.job_id),
            job_id=job.job_id,
            target_locale=job.target_language,
        )
        for field, message in (
            ("job_id", "Recovered response belongs to a different job"),
            ("target_locale", "Recovered response targets a different language"),
            (
                "source_transcript_sha256",
                "Recovered response is stale because the authoritative transcript changed",
            ),
            (
                "source_timeline_sha256",
                "Recovered response is stale because the dubbing units changed",
            ),
        ):
            if request.get(field) != current_request.get(field):
                raise ValueError(message)

        # Re-run the deterministic server normalizer. This may repair only
        # source-grounded metadata contradictions, such as a compact omission
        # label that the model omitted from optional_details. Spoken text is
        # never changed by this step.
        normalized_candidate = normalize_multivariant_response(
            dict(candidate),
            request=request,
        )
        provider_metadata = {
            key: details.get(key)
            for key in (
                "provider",
                "model_requested",
                "model_returned",
                "prompt_version",
                "prompt_hash",
                "request_timestamp",
                "completion_timestamp",
                "token_usage",
                "provider_attempts",
                "stop_reason",
            )
            if details.get(key) is not None
        }
        provider_metadata.update(
            {
                "recovered_from_validation_failure": True,
                "recovery_provider_http_requests_started": 0,
                "source_run_directory": str(selected_run),
            }
        )
        installed, validation = prepare_installed_translation(
            response=normalized_candidate,
            request=request,
            mode="cloud_validation_recovery",
            provider_metadata=provider_metadata,
        )
        if not dry_run and job.state not in {
            "analysis_ready",
            "translation_running",
            "translation_ready",
            "azure_tts_queued",
        }:
            raise ValueError(
                "Recovered translation can only be installed from analysis_ready, "
                "translation_running, translation_ready or azure_tts_queued; "
                f"got {job.state}"
            )

        summary = {
            "schema_version": "mathula.translation.cloud-recovery-result.v1",
            "job_id": job.job_id,
            "target_locale": job.target_language,
            "status": "validated" if dry_run else "installed",
            "dry_run": dry_run,
            "run_directory": str(selected_run),
            "request_path": str(request_path),
            "failure_path": str(failure_path),
            "candidate_output_sha256": details.get("candidate_output_sha256"),
            "unit_count": validation["unit_count"],
            "provider_http_requests_started": 0,
            "validation": validation,
        }
        if dry_run:
            return {"summary": summary, "translation": {}}

        recovery_backups = selected_run / "recovery_backups"
        atomic_write_json(
            selected_run / "translation_response.recovered.json",
            normalized_candidate,
        )
        atomic_write_json(selected_run / "validation.recovered.json", validation)
        atomic_write_json(
            selected_run / "installed_translation.recovered.json",
            installed,
        )
        written = write_translation_artifacts(
            job_root=self.jobs.job_dir(job.job_id),
            target_locale=job.target_language,
            installed=installed,
            pronunciation_dictionary=self._pronunciation_dictionary(job),
            backup_dir=recovery_backups,
        )
        summary.update(
            {
                "installed_translation_sha256": written["translation_sha256"],
                "installed_dubbing_sha256": written["dubbing_sha256"],
                "backups": written["backups"],
                "completed_at": utcnow(),
            }
        )
        atomic_write_json(selected_run / "recovery_manifest.json", summary)

        status_path = selected_run / "status.json"
        status = read_json(status_path) if status_path.is_file() else {}
        if not isinstance(status, dict):
            status = {}
        status.update(
            {
                "status": "completed_recovered",
                "completed": True,
                "recovered_without_provider_call": True,
                "recovery_manifest": str(selected_run / "recovery_manifest.json"),
                "error": None,
            }
        )
        atomic_write_json(status_path, status)

        previous_state = job.state
        if job.state == "analysis_ready":
            job.transition("translation_running")
        if job.state == "translation_running":
            job.transition("translation_ready")
        if job.state == "translation_ready":
            job.transition("azure_tts_queued")
        if job.state != "azure_tts_queued":
            raise ValueError(
                "Recovered translation can only complete from analysis_ready, "
                f"translation_running, translation_ready or azure_tts_queued; got {job.state}"
            )
        if "translation" not in job.completed_stages:
            job.completed_stages.append("translation")
        job.providers["translation"] = str(details.get("provider") or "azure-openai-gpt")
        job.providers["translation_model"] = str(
            details.get("model_returned")
            or details.get("model_requested")
            or "unknown"
        )
        job.providers["translation_prompt_version"] = str(
            details.get("prompt_version") or MULTIVARIANT_PROMPT_VERSION
        )
        job.media["translation_run_directory"] = str(selected_run)
        job.media["translation_multivariant_sha256"] = written["translation_sha256"]
        job.media["translation_sha256"] = written["dubbing_sha256"]
        job.compatibility_gate = {"state": "pending", "calibrated_thresholds": False}
        job.review_readiness = "compatibility_pending"
        job.last_error = None
        job.attempt_history.append(
            {
                "stage": "cloud_translation_validation_recovery",
                "status": "completed",
                "from_state": previous_state,
                "to_state": job.state,
                "provider_http_requests_started": 0,
                "source_run_directory": str(selected_run),
                "candidate_output_sha256": details.get("candidate_output_sha256"),
                "completed_at": utcnow(),
            }
        )
        self.jobs.save(job)
        translation = read_json(self.paths(job.job_id).three_text_translation)
        return {"summary": summary, "translation": translation}

    def recover_translation_response(
        self,
        job: JobManifest,
        *,
        run_dir: Path | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        recovered = self._recover_failed_translation_response(
            job,
            run_dir=run_dir,
            missing_ok=False,
            dry_run=dry_run,
        )
        if recovered is None:  # pragma: no cover - guarded by missing_ok=False
            raise ValueError("No recoverable translation response is available")
        return recovered["summary"]

    def upgrade_validated_translation(self, job: JobManifest, *, force: bool = False) -> dict[str, Any]:
        paths = self.paths(job.job_id)
        if paths.three_text_translation.is_file() and not force:
            return read_json(paths.three_text_translation)
        legacy_path = self.jobs.job_dir(job.job_id) / "translation/transcript_zu.json"
        if not legacy_path.is_file():
            raise ValueError("Validated legacy isiZulu translation is missing")
        authoritative = require_authoritative_transcript(
            work_dir=self.settings.work_dir,
            job_id=job.job_id,
        )
        units = self.build_units(job, force=False)
        normalized = normalize_translation_units(read_json(legacy_path), units["units"])
        normalized["source_transcript_sha256"] = authoritative["sha256"]
        normalized["source_autocorrection_state_sha256"] = authoritative[
            "state_sha256"
        ]
        normalized["source_dubbing_units_sha256"] = checksum(paths.dubbing_units)
        normalized["migration"] = {
            "source_path": "translation/transcript_zu.json",
            "source_sha256": checksum(legacy_path),
            "ai_called": False,
            "approved_content_preserved": True,
            "migrated_at": utcnow(),
        }
        atomic_write_json(paths.three_text_translation, normalized)
        job.media["translation_three_text_sha256"] = checksum(paths.three_text_translation)
        self.jobs.save(job)
        return normalized

    def repair_translation_units(
        self,
        job: JobManifest,
        provider: AIProvider,
        *,
        force: bool = False,
    ) -> dict[str, Any]:
        """Repair only Azure timing failures and preserve the faithful meaning."""

        paths = self.paths(job.job_id)
        queue_path = paths.azure_tts_root / "translation_repair_required.json"
        if not queue_path.is_file():
            raise ValueError("No Azure timing-repair queue exists")
        queue = read_json(queue_path)
        signals = queue.get("signals")
        if not isinstance(signals, list) or not signals:
            raise ValueError("Azure timing-repair queue has no affected units")
        translation = read_json(paths.three_text_translation)
        current_translation_sha256 = checksum(paths.three_text_translation)
        queued_translation_sha256 = str(queue.get("source_translation_sha256") or "")
        if queued_translation_sha256 and queued_translation_sha256 != current_translation_sha256:
            raise ValueError(
                "Azure timing-repair queue is stale because the translation changed; "
                "rerun Azure synthesis to create a current repair queue"
            )
        repair_generation_sha256 = str(
            queue.get("timing_repair_generation_sha256")
            or _timing_repair_generation_sha256(translation, paths.three_text_translation)
        )
        units_artifact = read_json(paths.dubbing_units)
        by_id = {str(item["unit_id"]): dict(item) for item in translation["units"]}
        source_by_id = {str(item["unit_id"]): item for item in units_artifact["units"]}
        ordered_ids = [str(item["unit_id"]) for item in translation["units"]]
        completed: list[dict[str, Any]] = []

        for signal in signals:
            unit_id = str(signal.get("unit_id") or "")
            if unit_id not in by_id or unit_id not in source_by_id:
                raise ValueError(f"Timing-repair unit is not in the dubbing plan: {unit_id}")
            stage = _timing_repair_stage(unit_id, repair_generation_sha256)
            previous_repairs = job.attempt_counters.get(stage, 0) + job.attempt_counters.get(
                _legacy_timing_repair_stage(unit_id, repair_generation_sha256), 0
            )
            if previous_repairs >= self.settings.max_translation_repairs_per_unit and not force:
                raise RuntimeError(
                    f"MATHULA_TV_MAX_TRANSLATION_REPAIRS_PER_UNIT reached for {unit_id}"
                )
            if _ai_call_count(job) >= self.settings.max_ai_calls_per_job:
                raise RuntimeError("MATHULA_TV_MAX_AI_CALLS_PER_JOB reached during timing repair")

            current = by_id[unit_id]
            index = ordered_ids.index(unit_id)
            neighbours = [
                by_id[ordered_ids[other]]
                for other in range(max(0, index - 1), min(len(ordered_ids), index + 2))
                if other != index
            ]
            pronunciation_dictionary = self._pronunciation_dictionary(job)
            source_text = str(source_by_id[unit_id].get("source_text", ""))
            protected_verbatim_phrases = []
            for entry in (*pronunciation_dictionary.job_overrides, *pronunciation_dictionary.entries):
                if entry.kind != "english_code_switch":
                    continue
                phrase = entry.spoken_text
                if phrase.casefold() in source_text.casefold():
                    protected_verbatim_phrases.append(phrase)
            payload = {
                "schema_version": "turn-timing-repair-request-v1",
                "unit_id": unit_id,
                "source_unit": source_by_id[unit_id],
                "current_translation": current,
                "neighbouring_translations": neighbours,
                "timing_failure": signal,
                "constraints": {
                    "faithful_translation_must_not_change": True,
                    "protected_entities_must_be_preserved": current.get(
                        "protected_entities_found", []
                    ),
                    "protected_verbatim_phrases_must_be_preserved": protected_verbatim_phrases,
                    "maximum_duration_ms": source_by_id[unit_id]["maximum_duration_ms"],
                    "human_review_required": True,
                },
                "content_trust": "untrusted_data",
            }
            attempt = self.jobs.begin_attempt(job, stage, forced=force)
            try:
                response = provider.repair_turn(
                    payload,
                    output_schema=TURN_REPAIR_SCHEMA,
                )
                repaired = dict(response.data["unit"])
                if str(repaired.get("unit_id")) != unit_id:
                    raise ValueError("GPT timing repair returned a different unit ID")
                faithful_changed_by_model = repaired.get("faithful_translation") != current.get("faithful_translation")
                # The archival faithful translation is server-owned and never changes
                # during a delivery-only timing repair. Models can paraphrase
                # echoed fields despite the instruction.
                repaired["faithful_translation"] = current.get("faithful_translation")
                if not repaired.get("numbers_preserved") or not repaired.get("dates_preserved"):
                    raise ValueError("GPT timing repair did not preserve numbers or dates")
                if not repaired.get("negation_preserved"):
                    raise ValueError("GPT timing repair did not preserve negation")
                found = {str(value).casefold() for value in repaired.get("protected_entities_found", [])}
                preserved = {
                    str(value).casefold()
                    for value in repaired.get("protected_entities_preserved", [])
                }
                if not found <= preserved:
                    raise ValueError("GPT timing repair lost a protected entity")
                history = list(current.get("timing_repair_history", []))
                history.append(
                    {
                        "attempt": attempt,
                        "repair_generation_sha256": repair_generation_sha256,
                        "signal": signal,
                        "faithful_translation_preserved_by_server": faithful_changed_by_model,
                        "metadata": response.metadata.to_dict(),
                    }
                )
                repaired["timing_repair_history"] = history
                by_id[unit_id] = repaired
                repair_root = (
                    paths.job_root
                    / "dubbing/repairs"
                    / unit_id
                    / repair_generation_sha256[:16]
                )
                atomic_write_json(repair_root / f"attempt_{attempt:02d}.json", response.to_artifact())
                self.jobs.finish_attempt(job, stage, attempt, "completed")
                completed.append(
                    {
                        "unit_id": unit_id,
                        "attempt": attempt,
                        "spoken_text_changed": repaired.get("spoken_text") != current.get("spoken_text"),
                    }
                )
            except Exception:
                self.jobs.finish_attempt(job, stage, attempt, "failed")
                raise

        candidate_translation = {
            **translation,
            "timing_repair_generation_sha256": repair_generation_sha256,
            "units": [by_id[unit_id] for unit_id in ordered_ids],
        }
        normalized = normalize_translation_units(
            candidate_translation,
            units_artifact["units"],
            pronunciation_dictionary=self._pronunciation_dictionary(job),
        )
        # normalize_translation_units rebuilds the unit records, but timing
        # repair must retain provenance and other full-translation metadata.
        for key, value in candidate_translation.items():
            if key != "units":
                normalized.setdefault(key, value)

        normalized["timing_repair_summary"] = {
            "schema_version": "translation-timing-repair-summary-v1",
            "provider": "azure-openai-gpt",
            "affected_units": completed,
            "full_clip_resent": False,
            "repair_generation_sha256": repair_generation_sha256,
        }
        atomic_write_json(paths.three_text_translation, normalized)
        atomic_write_json(
            paths.azure_tts_root / "translation_repair_applied.json",
            normalized["timing_repair_summary"],
        )
        self.prepare_dubbing(job, force=True)
        job.attempt_history.append(
            {
                "stage": "azure_tts_timing_repair_reset",
                "status": "completed",
                "from_state": job.state,
                "to_state": "azure_tts_queued",
                "affected_units": [item["unit_id"] for item in completed],
                "at": utcnow(),
            }
        )
        job.state = "azure_tts_queued"
        job.last_error = None
        self.jobs.save(job)
        return normalized["timing_repair_summary"]

    def prepare_dubbing(self, job: JobManifest, *, force: bool = False, azure_only: bool = False) -> dict[str, Any]:
        paths = self.paths(job.job_id)
        authoritative = require_authoritative_transcript(
            work_dir=self.settings.work_dir,
            job_id=job.job_id,
        )
        self.ensure_source_derivatives(job, force=False)
        units = self.build_units(job, force=False)
        units_sha256 = checksum(paths.dubbing_units)
        translation = (
            read_json(paths.three_text_translation)
            if paths.three_text_translation.is_file()
            else self.upgrade_validated_translation(job)
        )
        translation_sha256 = checksum(paths.three_text_translation)
        if (
            translation.get("source_transcript_sha256")
            != authoritative["sha256"]
            or translation.get("source_dubbing_units_sha256")
            != units_sha256
        ):
            raise ValueError(
                "Translation is stale because the authoritative autocorrected "
                "transcript changed; rerun translation before preparing dubbing"
            )
        if paths.plan.is_file() and not force:
            existing = read_json(paths.plan)
            if (
                existing.get("source_transcript_sha256")
                == authoritative["sha256"]
                and existing.get("source_dubbing_units_sha256")
                == units_sha256
                and existing.get("source_translation_sha256")
                == translation_sha256
            ):
                return existing
        pronunciation = self._pronunciation_dictionary(job)
        plan = build_dubbing_plan(
            job=job.to_dict(),
            source_media=Path(job.local_source_path),
            analysis_audio=paths.analysis_audio,
            mix_source_audio=paths.mix_source_audio,
            unit_artifact=units,
            translation_artifact=translation,
            output_path=paths.plan,
            pronunciation_dictionary=pronunciation,
            azure_only=azure_only,
        )
        plan["source_transcript_sha256"] = authoritative["sha256"]
        plan["source_autocorrection_state_sha256"] = authoritative[
            "state_sha256"
        ]
        plan["source_dubbing_units_sha256"] = units_sha256
        plan["source_translation_sha256"] = translation_sha256
        atomic_write_json(paths.plan, plan)
        job.media["dubbing_plan_sha256"] = checksum(paths.plan)
        self.jobs.save(job)
        return plan

    def synthesize_azure(
        self,
        job: JobManifest,
        backend: AzureTTSBackend,
        *,
        force: bool = False,
        voice_assignments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if job.state == "azure_tts_ready" and force:
            self._forced_stage_reset(job, "azure_tts", "azure_tts_queued")
        if job.state not in {"azure_tts_queued", "azure_tts_running"}:
            raise ValueError(f"Azure TTS requires azure_tts_queued, not {job.state}")
        plan = self.prepare_dubbing(job)
        paths = self.paths(job.job_id)
        available = backend.validate_configured_voices(force_refresh=force)
        previous_manifest = read_json(paths.azure_manifest) if paths.azure_manifest.is_file() else {}
        previous_by_unit = {
            str(item.get("turn_id")): item for item in previous_manifest.get("units", [])
        }
        repair_required_path = paths.azure_tts_root / "translation_repair_required.json"
        repair_applied_path = paths.azure_tts_root / "translation_repair_applied.json"
        repair_applied = read_json(repair_applied_path) if repair_applied_path.is_file() else {}
        repaired_ids = {
            str(item.get("unit_id")) for item in repair_applied.get("affected_units", [])
        }
        if job.state == "azure_tts_queued":
            job.transition("azure_tts_running")
        attempt = self.jobs.begin_attempt(job, "azure_tts", forced=force)
        search = AzureTTSTimingSearch(
            rate_min_percent=self.settings.azure_rate_min_percent,
            rate_max_percent=self.settings.azure_rate_max_percent,
            search_attempts=min(
                self.settings.azure_rate_search_attempts,
                self.settings.max_azure_tts_attempts_per_unit,
            ),
            preferred_end_tolerance_ms=self.settings.preferred_end_tolerance_ms,
        )
        results = []
        repair_signals = []
        
        # Load complete assignments so every speaker keeps a stable family-anchor
        # voice plus its own source-relative pitch and base speaking rate.
        assignment_by_speaker: dict[str, Mapping[str, Any]] = {}
        if voice_assignments:
            for assignment in voice_assignments.get("assignments", []):
                if isinstance(assignment, Mapping) and assignment.get("speaker_id"):
                    assignment_by_speaker[str(assignment["speaker_id"])] = assignment

        try:
            for unit in plan["units"]:
                assignment = assignment_by_speaker.get(str(unit["speaker_id"]))
                if assignment_by_speaker:
                    voice, base_rate, base_pitch, base_volume = voice_and_base_prosody(
                        assignment,
                        backend.default_voice,
                    )
                else:
                    voice = self._selected_voice(plan, unit["speaker_id"], backend.default_voice)
                    base_rate = base_pitch = base_volume = 0
                request = AzureTTSRequest(
                    turn_id=unit["unit_id"],
                    speaker_id=unit["speaker_id"],
                    text=unit["tts_text"],
                    preferred_duration_ms=int(unit["preferred_duration_ms"]),
                    maximum_duration_ms=int(unit["maximum_duration_ms"]),
                    voice=voice,
                    rate_percent=base_rate,
                    pitch_percent=base_pitch,
                    volume_percent=base_volume,
                    parts=build_initialism_ssml_parts(
                        unit["tts_text"],
                        language=job.target_language,
                    ),
                )
                previous = previous_by_unit.get(str(unit["unit_id"]))
                unit_force = force or unit["unit_id"] in repaired_ids or (
                    previous is not None and previous.get("voice") != voice
                )

                def synthesize_rate(rate: int) -> AzureTTSResult:
                    # AzureTTSTimingSearch uses absolute Azure rate percentages.
                    # Start at the speaker's identity rate, then search upward
                    # within the configured global bounds when timing requires it.
                    return _synthesize_azure_candidate(
                        backend,
                        request.with_rate(rate),
                        paths.azure_candidate(unit["unit_id"], rate),
                        force=unit_force,
                    )

                fit = search.fit(
                    synthesize_rate,
                    preferred_duration_ms=request.preferred_duration_ms,
                    maximum_duration_ms=request.maximum_duration_ms,
                    initial_rate_percent=base_rate,
                )
                if fit.accepted is None:
                    if fit.repair_signal is None:
                        raise RuntimeError("Timing search rejected a unit without a repair signal")
                    repair_signals.append(
                        {"unit_id": unit["unit_id"], **fit.repair_signal.to_dict()}
                    )
                    continue
                canonical = paths.azure_turn(unit["unit_id"])
                self._atomic_copy(Path(fit.accepted.output_path), canonical)
                result = fit.accepted.to_dict()
                result.update(
                    output_path=str(canonical),
                    sha256=checksum(canonical),
                    timing_search=fit.to_dict(),
                    speaker_base_prosody={
                        "rate_percent": base_rate,
                        "pitch_percent": base_pitch,
                        "volume_percent": base_volume,
                    },
                    timing_rate_delta_percent=fit.accepted.rate_percent - base_rate,
                )
                results.append(result)
            if repair_signals:
                atomic_write_json(
                    repair_required_path,
                    {
                        "schema_version": "translation-timing-repair-queue-v2",
                        "provider": "azure-openai-gpt",
                        "repair_scope": "affected_units_only",
                        "source_translation_sha256": checksum(paths.three_text_translation),
                        "timing_repair_generation_sha256": _timing_repair_generation_sha256(
                            read_json(paths.three_text_translation),
                            paths.three_text_translation,
                        ),
                        "signals": repair_signals,
                    },
                )
                raise TranslationTimingFailure(
                    f"{len(repair_signals)} unit(s) require bounded GPT timing repair",
                    details={"repair_artifact": str(repair_required_path)},
                )
            manifest = {
                "schema_version": "azure-tts-manifest-v1",
                "backend": "azure_tts",
                "available_voices": [voice.name for voice in available],
                "units": results,
                "attempt": attempt,
            }
            atomic_write_json(paths.azure_manifest, manifest)
            self._update_plan_stage(
                job,
                "azure_tts",
                {str(item["turn_id"]): item for item in results},
            )
            
            # Run Azure TTS intelligibility audit BEFORE state transition
            # This ensures audit failures block the success state
            if hasattr(self, '_stt_backend') and self._stt_backend:
                self._audit_intelligibility(job, "azure-tts", plan["units"], paths.azure_tts_root)
            
            job.providers["speech_generation"] = "azure_tts"
            job.media["azure_tts_manifest_sha256"] = checksum(paths.azure_manifest)
            
            if "azure_tts" not in job.completed_stages:
                job.completed_stages.append("azure_tts")
            job.transition("azure_tts_ready")
            self.jobs.finish_attempt(job, "azure_tts", attempt, "completed")
            _clear_azure_timing_repair_markers(
                repair_required_path,
                repair_applied_path,
            )
            return manifest
        except Exception as exc:
            job.last_error = self._safe_error("azure_tts", exc)
            self.jobs.finish_attempt(job, "azure_tts", attempt, "partial" if results else "failed")
            raise

    def prepare_azure_source_calibrations(
        self,
        job: JobManifest,
        backend: AzureTTSBackend,
        *,
        calibration_text: str = REVIEWED_AZURE_CALIBRATION_TEXT,
        force: bool = False,
    ) -> dict[str, Any]:
        if not re.fullmatch(r"[0-9a-f]{40}", self.settings.openvoice_revision):
            raise ValueError("A reviewed immutable OpenVoice commit is required before source calibration")
        voices = backend.validate_configured_voices(force_refresh=force)
        text_hash = hashlib.sha256(calibration_text.encode("utf-8")).hexdigest()
        manifests = {}
        for voice in voices:
            root = self.settings.work_dir / "cache/openvoice/azure_sources" / voice.name
            manifest_path = root / "manifest.json"
            previous = read_json(manifest_path) if manifest_path.is_file() else {}
            candidate = root / "calibration_candidate.wav"
            request = AzureTTSRequest(
                turn_id=f"calibration_{voice.name}",
                speaker_id="AZURE_SOURCE",
                text=calibration_text,
                preferred_duration_ms=30_000,
                maximum_duration_ms=40_000,
                voice=voice.name,
            )
            result = backend.synthesize(request, candidate, force=force)
            if not 20_000 <= result.duration_ms <= 40_000:
                raise ValueError(
                    f"Azure source calibration for {voice.name} must be 20-40 seconds, got {result.duration_ms}ms"
                )
            canonical = root / "calibration.wav"
            self._atomic_copy(candidate, canonical)
            manifest = {
                "schema_version": "azure-openvoice-source-calibration-v1",
                "azure_voice": voice.name,
                "region": self.settings.azure_speech_region,
                "output_format": backend.output_format,
                "calibration_text_version": "reviewed-zu-v1",
                "calibration_text_sha256": text_hash,
                "wav_sha256": checksum(canonical),
                "openvoice_revision": self.settings.openvoice_revision,
                "embedding_sha256": None,
                "status": "source_embedding_pending_gpu_worker",
                "sample_rate": backend.sample_rate,
                "channels": backend.channels,
                "sample_width": backend.sample_width,
            }
            embedding_path = root / "source_embedding.bin"
            embedding_manifest_path = root / "source_embedding_manifest.json"
            reusable_identity = {
                "schema_version": manifest["schema_version"],
                "azure_voice": manifest["azure_voice"],
                "region": manifest["region"],
                "output_format": manifest["output_format"],
                "calibration_text_version": manifest[
                    "calibration_text_version"
                ],
                "calibration_text_sha256": manifest[
                    "calibration_text_sha256"
                ],
                "wav_sha256": manifest["wav_sha256"],
                "openvoice_revision": manifest["openvoice_revision"],
                "sample_rate": manifest["sample_rate"],
                "channels": manifest["channels"],
                "sample_width": manifest["sample_width"],
            }
            if (
                not force
                and previous.get("status") == "ready"
                and all(
                    previous.get(key) == value
                    for key, value in reusable_identity.items()
                )
                and embedding_path.is_file()
                and previous.get("embedding_sha256") == checksum(embedding_path)
                and embedding_manifest_path.is_file()
                and isinstance(previous.get("checkpoint_hashes"), Mapping)
                and previous.get("checkpoint_hashes")
            ):
                manifest.update(
                    {
                        "embedding_path": str(embedding_path),
                        "embedding_sha256": previous["embedding_sha256"],
                        "embedding_manifest_path": str(
                            embedding_manifest_path
                        ),
                        "checkpoint_hashes": dict(
                            previous["checkpoint_hashes"]
                        ),
                        "status": "ready",
                    }
                )
            atomic_write_json(manifest_path, manifest)
            if manifest["status"] == "ready":
                embedding_manifest = read_json(embedding_manifest_path)
                try:
                    self._verify_embedding_manifest(
                        embedding_manifest_path,
                        schema=OPENVOICE_SOURCE_EMBEDDING_MANIFEST_SCHEMA,
                        expected_artifact_hash=str(
                            embedding_manifest.get("artifact_sha256") or ""
                        ),
                        expected_values={
                            "azure_voice": voice.name,
                            "calibration_audio_sha256": checksum(canonical),
                            "calibration_manifest_sha256": checksum(
                                manifest_path
                            ),
                            "source_embedding_sha256": checksum(
                                embedding_path
                            ),
                            "openvoice_revision": self.settings.openvoice_revision,
                            "checkpoint_hashes": manifest[
                                "checkpoint_hashes"
                            ],
                            "status": "ready",
                        },
                    )
                except ValueError:
                    manifest = {
                        **{key: manifest[key] for key in reusable_identity},
                        "embedding_sha256": None,
                        "status": "source_embedding_pending_gpu_worker",
                    }
                    atomic_write_json(manifest_path, manifest)
            manifests[voice.name] = manifest
        return {"schema_version": "azure-source-calibrations-v1", "voices": manifests}

    def prepare_reference_reels(
        self,
        job: JobManifest,
        candidates_by_speaker: Mapping[str, list[dict[str, Any]]],
        *,
        proof_of_concept: bool = False,
        force: bool = False,
    ) -> dict[str, Any]:
        plan = self.prepare_dubbing(job)
        results = {}
        for speaker_id in sorted(plan["speakers"]):
            root = self.jobs.job_dir(job.job_id) / "speakers" / speaker_id
            manifest_path = root / "reference_manifest.json"
            if manifest_path.is_file() and not force:
                results[speaker_id] = read_json(manifest_path)
                continue
            selection = select_reference_candidates(
                speaker_id,
                candidates_by_speaker.get(speaker_id, []),
                proof_of_concept=proof_of_concept,
            )
            results[speaker_id] = build_reference_reel(
                selection,
                root / "reference_reel.wav",
                manifest_path,
            )
        return results

    def default_reference_candidates(self, job: JobManifest) -> dict[str, list[dict[str, Any]]]:
        """Create truthful mixed-source proof candidates; never publication references."""
        self.ensure_source_derivatives(job)
        transcript = read_json(self.jobs.job_dir(job.job_id) / "analysis/transcript_en.json")
        result: dict[str, list[dict[str, Any]]] = {}
        for segment in transcript["segments"]:
            speaker = str(segment["speaker"])
            result.setdefault(speaker, []).append(
                {
                    "segment_id": segment["segment_id"],
                    "speaker_id": speaker,
                    "start": float(segment["start"]),
                    "end": float(segment["end"]),
                    "path": str(self.paths(job.job_id).mix_source_audio),
                    "source_transcript": segment["source_text"],
                    "audio_source_type": "mixed_source_fallback",
                    "overlap": bool(segment.get("overlap", False)),
                    "quality_score": float(segment.get("speaker_assignment_confidence") or 0.5),
                }
            )
        return result

    def persist_speaker_voice_selections(
        self,
        job: JobManifest,
        evaluations: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if job.state != "azure_tts_ready":
            raise ValueError("Speaker voice selection requires completed Azure timing-fit audio")
        plan = self.prepare_dubbing(job)
        if evaluations is not None:
            evaluation_speakers = evaluations.get("speakers")
            if (
                not isinstance(evaluation_speakers, Mapping)
                or set(evaluation_speakers) != set(plan["speakers"])
            ):
                raise ValueError(
                    "Candidate evaluations must exactly match the job-local speaker set"
                )
        azure_manifest = read_json(self.paths(job.job_id).azure_manifest)
        azure_by_unit = {str(item["turn_id"]): item for item in azure_manifest["units"]}
        selected = {}
        for speaker in sorted(plan["speakers"]):
            root = self.jobs.job_dir(job.job_id) / "speakers" / speaker
            candidates_path = root / "azure_voice_candidates.json"
            if not candidates_path.is_file():
                raise ValueError(
                    f"GPU voice-calibration results are missing for {speaker}: {candidates_path}"
                )
            value = read_json(candidates_path)
            if evaluations is not None:
                self._apply_candidate_evaluations(
                    job,
                    speaker,
                    value,
                    evaluations,
                )
                atomic_write_json(candidates_path, value)
            pending = [
                str(item.get("voice") or item.get("azure_voice") or "unknown")
                for item in value.get("candidates", [])
                if item.get("evaluation_status") == "pending_external_qc"
            ]
            if pending:
                raise ValueError(
                    f"External speaker-similarity/intelligibility QC is pending for {speaker}: "
                    + ", ".join(pending)
                )
            selection = rank_voice_candidates(
                value["candidates"],
                allowed_voices=self.settings.azure_tts_voices,
                human_override=value.get("human_override"),
            )
            selection["speaker_id"] = speaker
            atomic_write_json(root / "azure_voice_selection.json", selection)
            selected[speaker] = selection
            plan["speakers"][speaker]["selected_azure_voice"] = selection
        self._write_plan(self.paths(job.job_id).plan, plan)
        changed_units = [
            unit["unit_id"]
            for unit in plan["units"]
            if azure_by_unit[unit["unit_id"]]["voice"]
            != selected[unit["speaker_id"]]["selected_voice"]
        ]
        if changed_units:
            atomic_write_json(
                self.paths(job.job_id).azure_tts_root / "voice_reselection_required.json",
                {
                    "schema_version": "azure-tts-voice-reselection-v1",
                    "affected_units": changed_units,
                    "reason": "Measured per-speaker Azure base voice changed",
                },
            )
            job.attempt_history.append(
                {
                    "stage": "azure_voice_selection",
                    "status": "completed",
                    "from_state": "azure_tts_ready",
                    "to_state": "azure_tts_queued",
                    "affected_units": changed_units,
                    "at": utcnow(),
                }
            )
            job.transition("azure_tts_queued")
        job.media["dubbing_plan_sha256"] = checksum(self.paths(job.job_id).plan)
        self.jobs.save(job)
        return {
            "schema_version": "job-speaker-azure-voice-selections-v1",
            "speakers": selected,
            "azure_resynthesis_required": bool(changed_units),
            "affected_units": changed_units,
        }

    def prepare_speaker_voice_calibration_audio(
        self,
        job: JobManifest,
        backend: AzureTTSBackend,
        *,
        calibration_sentence: str = REVIEWED_SPEAKER_VOICE_CALIBRATION_SENTENCE,
        force: bool = False,
    ) -> dict[str, Any]:
        """Generate the same reviewed Azure sentence for every speaker/voice pair."""

        if job.state != "azure_tts_ready":
            raise ValueError("Speaker voice calibration requires azure_tts_ready")
        plan = self.prepare_dubbing(job)
        voices = backend.validate_configured_voices(force_refresh=force)
        sentence_hash = hashlib.sha256(calibration_sentence.encode("utf-8")).hexdigest()
        speakers: dict[str, Any] = {}
        for speaker in sorted(plan["speakers"]):
            reference = self.paths(job.job_id).job_root / "speakers" / speaker / "reference_reel.wav"
            if not reference.is_file():
                raise ValueError(f"Speaker reference reel is missing for {speaker}")
            results = []
            for index, voice in enumerate(voices, 1):
                output = (
                    self.paths(job.job_id).job_root
                    / "speakers"
                    / speaker
                    / "voice_calibration"
                    / "azure"
                    / f"{voice.name}.wav"
                )
                result = backend.synthesize(
                    AzureTTSRequest(
                        turn_id=f"voice_cal_{speaker}_{index}",
                        speaker_id=speaker,
                        text=calibration_sentence,
                        preferred_duration_ms=8_000,
                        maximum_duration_ms=15_000,
                        voice=voice.name,
                    ),
                    output,
                    force=force,
                )
                results.append(result.to_dict())
            manifest = {
                "schema_version": "speaker-azure-voice-calibration-inputs-v1",
                "speaker_id": speaker,
                "calibration_sentence": calibration_sentence,
                "calibration_sentence_sha256": sentence_hash,
                "reference_reel_path": str(reference),
                "reference_reel_sha256": checksum(reference),
                "azure_candidates": results,
                "openvoice_conversion_status": "pending_gpu_asset_worker",
                "quality_measurement_status": "pending",
                "human_review_required": True,
            }
            manifest_path = (
                self.paths(job.job_id).job_root
                / "speakers"
                / speaker
                / "voice_calibration"
                / "input_manifest.json"
            )
            atomic_write_json(manifest_path, manifest)
            speakers[speaker] = {
                "manifest_path": str(manifest_path),
                "manifest_sha256": checksum(manifest_path),
                "voice_count": len(results),
            }
        return {
            "schema_version": "job-speaker-azure-voice-calibration-inputs-v1",
            "speakers": speakers,
            "next": "Run the pinned GPU asset-preflight worker",
        }

    def prepare_openvoice_assets(
        self,
        job: JobManifest,
        *,
        lease_identity: str = "unbound",
        proof_of_concept: bool = False,
        force: bool = False,
    ) -> dict[str, Any]:
        """Build the server-bound GPU preflight plan without importing OpenVoice."""

        if job.state != "azure_tts_ready":
            raise ValueError(
                f"OpenVoice asset preparation requires azure_tts_ready, not {job.state}"
            )
        revision = self.settings.openvoice_revision
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError(
                "MATHULA_TV_OPENVOICE_REVISION must be a reviewed immutable "
                "40-character commit SHA"
            )
        plan = self.prepare_dubbing(job)
        paths = self.paths(job.job_id)
        self._require_voice_rights(job, plan, proof_of_concept=proof_of_concept)
        if paths.openvoice_asset_plan.is_file() and not force:
            existing = read_json(paths.openvoice_asset_plan)
            parsed = OpenVoiceAssetPlan.from_dict(existing)
            if parsed.job_id != job.job_id or parsed.openvoice_revision != revision:
                raise ValueError("Existing OpenVoice asset plan belongs to another job or revision")
            return existing

        allowed_voices = tuple(dict.fromkeys(self.settings.azure_tts_voices))
        if not allowed_voices:
            raise ValueError("At least one configured Azure isiZulu voice is required")
        sources: list[OpenVoiceSourceAsset] = []
        for voice in allowed_voices:
            cache_root = self.settings.work_dir / "cache/openvoice/azure_sources" / voice
            calibration_audio = cache_root / "calibration.wav"
            calibration_manifest = cache_root / "manifest.json"
            for required in (calibration_audio, calibration_manifest):
                if not required.is_file() or required.stat().st_size == 0:
                    raise ValueError(f"Azure source calibration asset is missing: {required}")
            calibration_value = read_json(calibration_manifest)
            calibration_audio_hash = checksum(calibration_audio)
            expected_identity = {
                "schema_version": "azure-openvoice-source-calibration-v1",
                "azure_voice": voice,
                "openvoice_revision": revision,
                "wav_sha256": calibration_audio_hash,
            }
            if any(
                calibration_value.get(key) != value
                for key, value in expected_identity.items()
            ):
                raise ValueError(
                    f"Azure source calibration identity is stale or invalid for {voice}"
                )
            staged_root = paths.openvoice_root / "assets/azure_sources" / voice
            staged_audio = staged_root / "calibration.wav"
            staged_manifest = staged_root / "calibration_manifest.json"
            self._atomic_copy(calibration_audio, staged_audio)
            self._atomic_copy(calibration_manifest, staged_manifest)
            source_asset = OpenVoiceSourceAsset(
                    azure_voice=voice,
                    calibration_audio_path=self._relative_job_path(paths, staged_audio),
                    calibration_audio_sha256=calibration_audio_hash,
                    calibration_manifest_path=self._relative_job_path(
                        paths, staged_manifest
                    ),
                    calibration_manifest_sha256=checksum(staged_manifest),
                    source_embedding_path=self._relative_job_path(
                        paths, staged_root / "source_embedding.bin"
                    ),
                    source_embedding_manifest_path=self._relative_job_path(
                        paths, staged_root / "source_embedding_manifest.json"
                    ),
                    calibration=AzureSourceCalibrationSpec(
                        azure_voice=voice,
                        region=str(calibration_value.get("region") or ""),
                        output_format=str(
                            calibration_value.get("output_format") or ""
                        ),
                        calibration_text_version=str(
                            calibration_value.get("calibration_text_version") or ""
                        ),
                        calibration_text_sha256=str(
                            calibration_value.get("calibration_text_sha256") or ""
                        ),
                        openvoice_revision=revision,
                        sample_rate=int(calibration_value.get("sample_rate") or 0),
                        channels=int(calibration_value.get("channels") or 0),
                        sample_width=int(calibration_value.get("sample_width") or 0),
                        reviewed=True,
                    ),
            )
            cached_embedding = cache_root / "source_embedding.bin"
            cached_embedding_manifest = (
                cache_root / "source_embedding_manifest.json"
            )
            cache_declared_ready = calibration_value.get("status") == "ready"
            if cache_declared_ready and (
                cached_embedding.is_file() != cached_embedding_manifest.is_file()
            ):
                raise ValueError(
                    f"Azure source embedding cache is partial for {voice}"
                )
            if cache_declared_ready:
                if not cached_embedding.is_file():
                    raise ValueError(
                        f"Azure source embedding cache is missing for {voice}"
                    )
                cached_value = read_json(cached_embedding_manifest)
                expected_cached = {
                    "azure_voice": voice,
                    "calibration_audio_sha256": calibration_audio_hash,
                    "calibration_manifest_sha256": checksum(calibration_manifest),
                    "source_embedding_sha256": checksum(cached_embedding),
                    "openvoice_revision": revision,
                    "status": "ready",
                    "checkpoint_hashes": calibration_value.get(
                        "checkpoint_hashes"
                    ),
                }
                self._verify_embedding_manifest(
                    cached_embedding_manifest,
                    schema=OPENVOICE_SOURCE_EMBEDDING_MANIFEST_SCHEMA,
                    expected_artifact_hash=str(
                        cached_value.get("artifact_sha256") or ""
                    ),
                    expected_values=expected_cached,
                )
                self._atomic_copy(
                    cached_embedding,
                    self._resolve_job_path(
                        paths, source_asset.source_embedding_path
                    ),
                )
                self._atomic_copy(
                    cached_embedding_manifest,
                    self._resolve_job_path(
                        paths, source_asset.source_embedding_manifest_path
                    ),
                )
            sources.append(source_asset)

        speakers: list[OpenVoiceSpeakerAsset] = []
        for speaker_id in sorted(plan["speakers"]):
            speaker_root = paths.job_root / "speakers" / speaker_id
            reference_audio = speaker_root / "reference_reel.wav"
            reference_manifest = speaker_root / "reference_manifest.json"
            for required in (reference_audio, reference_manifest):
                if not required.is_file() or required.stat().st_size == 0:
                    raise ValueError(f"Speaker reference asset is missing: {required}")
            reference_hash = checksum(reference_audio)
            reference_value = read_json(reference_manifest)
            if (
                reference_value.get("schema_version")
                != "speaker-reference-manifest-v1"
                or reference_value.get("speaker_id") != speaker_id
                or reference_value.get("reference_reel_sha256") != reference_hash
                or reference_value.get("job_local_identity") is not True
                or reference_value.get("global_identity_created") is not False
            ):
                raise ValueError(
                    f"Speaker reference identity is invalid for {speaker_id}"
                )
            calibration_manifest = (
                speaker_root / "voice_calibration" / "input_manifest.json"
            )
            if not calibration_manifest.is_file():
                raise ValueError(
                    f"Azure speaker-voice calibration inputs are missing for {speaker_id}"
                )
            calibration_value = read_json(calibration_manifest)
            if calibration_value.get("speaker_id") != speaker_id:
                raise ValueError("Speaker-voice calibration manifest identity changed")
            sentence_hash = str(
                calibration_value.get("calibration_sentence_sha256") or ""
            )
            candidates_by_voice = {
                str(item.get("voice")): item
                for item in calibration_value.get("azure_candidates", [])
            }
            if set(candidates_by_voice) != set(allowed_voices):
                raise ValueError(
                    f"Speaker-voice candidates do not match the Azure allowlist for {speaker_id}"
                )
            candidates: list[OpenVoiceVoiceCandidate] = []
            for voice in allowed_voices:
                item = candidates_by_voice[voice]
                source_audio = (
                    speaker_root / "voice_calibration" / "azure" / f"{voice}.wav"
                )
                if not source_audio.is_file() or checksum(source_audio) != item.get("sha256"):
                    raise ValueError(
                        f"Azure voice-calibration WAV is missing or changed for {speaker_id}/{voice}"
                    )
                candidates.append(
                    OpenVoiceVoiceCandidate(
                        candidate_id=f"{speaker_id}--{voice}",
                        azure_voice=voice,
                        source_audio_path=self._relative_job_path(paths, source_audio),
                        source_audio_sha256=str(item["sha256"]),
                        output_path=Path(
                            f"speakers/{speaker_id}/voice_calibration/openvoice/{voice}.wav"
                        ),
                        calibration_text_sha256=sentence_hash,
                        conversion_settings={"tau": 0.3},
                        sample_rate=int(item.get("sample_rate") or self.settings.azure_tts_sample_rate),
                    )
                )
            speakers.append(
                OpenVoiceSpeakerAsset(
                    speaker_id=speaker_id,
                    reference_audio_path=self._relative_job_path(paths, reference_audio),
                    reference_audio_sha256=reference_hash,
                    reference_manifest_path=self._relative_job_path(
                        paths, reference_manifest
                    ),
                    reference_manifest_sha256=checksum(reference_manifest),
                    target_embedding_path=Path(
                        f"speakers/{speaker_id}/target_embedding.bin"
                    ),
                    target_embedding_manifest_path=Path(
                        f"speakers/{speaker_id}/target_embedding_manifest.json"
                    ),
                    candidate_manifest_path=Path(
                        f"speakers/{speaker_id}/azure_voice_candidates.json"
                    ),
                    voice_candidates=tuple(candidates),
                    sample_rate=self.settings.azure_tts_sample_rate,
                )
            )

        worker_plan = OpenVoiceAssetPlan(
            job_id=job.job_id,
            lease_identity=lease_identity,
            openvoice_revision=revision,
            sources=tuple(sources),
            speakers=tuple(speakers),
        )
        value = write_openvoice_asset_plan(worker_plan, paths.openvoice_asset_plan)
        job.media["openvoice_asset_plan_sha256"] = checksum(
            paths.openvoice_asset_plan
        )
        job.media["openvoice_asset_plan_artifact_sha256"] = value[
            "artifact_sha256"
        ]
        job.attempt_history.append(
            {
                "stage": "openvoice_asset_plan",
                "status": "completed",
                "forced": force,
                "artifact_sha256": value["artifact_sha256"],
                "at": utcnow(),
            }
        )
        self.jobs.save(job)
        return value

    def prepare_openvoice(
        self,
        job: JobManifest,
        *,
        lease_identity: str,
        prepare_only: bool = False,
        proof_of_concept: bool = False,
        force: bool = False,
    ) -> dict[str, Any]:
        revision = self.settings.openvoice_revision
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("MATHULA_TV_OPENVOICE_REVISION must be a reviewed immutable 40-character commit SHA")
        plan = self.prepare_dubbing(job)
        paths = self.paths(job.job_id)
        self._require_voice_rights(job, plan, proof_of_concept=proof_of_concept)
        if prepare_only:
            return self.prepare_openvoice_assets(
                job,
                lease_identity=lease_identity,
                proof_of_concept=proof_of_concept,
                force=force,
            )

        if job.state != "azure_tts_ready":
            raise ValueError(
                f"OpenVoice conversion preparation requires azure_tts_ready, not {job.state}"
            )
        reconciliation_path = paths.openvoice_root / "asset_reconciliation.json"
        if (
            not reconciliation_path.is_file()
            or read_json(reconciliation_path).get("state") != "completed"
        ):
            raise ValueError(
                "Reconcile the completed OpenVoice GPU asset-preflight worker first"
            )
        for speaker_id in sorted(plan["speakers"]):
            selection = plan["speakers"][speaker_id].get("selected_azure_voice")
            if not isinstance(selection, Mapping) or not selection.get("selected_voice"):
                raise ValueError(
                    f"Measured Azure base-voice selection is missing for {speaker_id}"
                )
        azure_manifest = read_json(paths.azure_manifest)
        azure_by_unit = {item["turn_id"]: item for item in azure_manifest["units"]}
        units = []
        missing = []
        staged_speaker_references = {}
        for speaker in sorted(plan["speakers"]):
            reference = self.jobs.job_dir(job.job_id) / "speakers" / speaker / "reference_reel.wav"
            if not reference.is_file():
                missing.append(str(reference))
            else:
                staged_speaker_references[speaker] = {
                    "path": f"speakers/{speaker}/reference_reel.wav",
                    "sha256": checksum(reference),
                }
        for unit in plan["units"]:
            azure = azure_by_unit[unit["unit_id"]]
            voice = azure["voice"]
            selected_voice = self._selected_voice(plan, unit["speaker_id"], voice)
            if selected_voice != voice:
                raise ValueError(
                    f"Azure TTS for {unit['unit_id']} uses {voice}, but speaker calibration selected "
                    f"{selected_voice}; rerun only the affected Azure units first"
                )
            source_root = self.settings.work_dir / "cache/openvoice/azure_sources" / voice
            source_embedding = source_root / "source_embedding.bin"
            target_embedding = self.jobs.job_dir(job.job_id) / "speakers" / unit["speaker_id"] / "target_embedding.bin"
            source_manifest_path = source_root / "manifest.json"
            target_manifest_path = (
                self.jobs.job_dir(job.job_id)
                / "speakers"
                / unit["speaker_id"]
                / "target_embedding_manifest.json"
            )
            for path in (source_embedding, target_embedding):
                if not path.is_file():
                    missing.append(str(path))
            if not source_embedding.is_file() or not target_embedding.is_file():
                continue
            source_manifest = read_json(source_manifest_path)
            target_manifest = read_json(target_manifest_path)
            if (
                source_manifest.get("status") != "ready"
                or source_manifest.get("azure_voice") != voice
                or source_manifest.get("openvoice_revision") != revision
                or source_manifest.get("embedding_sha256")
                != checksum(source_embedding)
            ):
                raise ValueError(f"Cached Azure source embedding identity is invalid for {voice}")
            if (
                target_manifest.get("schema_version")
                != OPENVOICE_TARGET_EMBEDDING_MANIFEST_SCHEMA
                or target_manifest.get("status") != "ready"
                or target_manifest.get("speaker_id") != unit["speaker_id"]
                or target_manifest.get("job_local_identity") is not True
                or target_manifest.get("openvoice_revision") != revision
                or target_manifest.get("target_embedding_sha256")
                != checksum(target_embedding)
            ):
                raise ValueError(
                    f"Job-local target embedding identity is invalid for {unit['speaker_id']}"
                )
            staged_source = paths.openvoice_root / "assets/azure_sources" / voice / "source_embedding.bin"
            staged_target = paths.openvoice_root / "assets/target_speakers" / unit["speaker_id"] / "target_embedding.bin"
            self._atomic_copy(source_embedding, staged_source)
            self._atomic_copy(target_embedding, staged_target)
            units.append(
                OpenVoiceUnit(
                    unit_id=unit["unit_id"],
                    speaker_id=unit["speaker_id"],
                    azure_voice=voice,
                    target_language="zu-ZA",
                    source_audio_path=Path(f"dubbing/azure_tts/turns/{unit['unit_id']}.wav"),
                    source_audio_sha256=azure["sha256"],
                    source_embedding_path=Path(f"dubbing/openvoice/assets/azure_sources/{voice}/source_embedding.bin"),
                    source_embedding_sha256=checksum(staged_source),
                    target_embedding_path=Path(f"dubbing/openvoice/assets/target_speakers/{unit['speaker_id']}/target_embedding.bin"),
                    target_embedding_sha256=checksum(staged_target),
                    output_path=Path(f"dubbing/openvoice/turns/{unit['unit_id']}.wav"),
                    conversion_settings={"tau": 0.3},
                )
            )
        if missing:
            raise ValueError("OpenVoice embeddings are not ready: " + ", ".join(sorted(set(missing))))
        worker_plan = OpenVoicePlan(job.job_id, lease_identity, revision, tuple(units))
        value = {
            "schema_version": worker_plan.schema_version,
            "job_id": worker_plan.job_id,
            "lease_identity": worker_plan.lease_identity,
            "openvoice_revision": worker_plan.openvoice_revision,
            "speaker_references": staged_speaker_references,
            "units": [self._openvoice_unit_dict(unit) for unit in worker_plan.units],
        }
        atomic_write_json(paths.openvoice_plan, value)
        openvoice_by_unit = {unit.unit_id: unit for unit in worker_plan.units}
        for planned_unit in plan["units"]:
            worker_unit = openvoice_by_unit[planned_unit["unit_id"]]
            planned_unit["openvoice"] = {
                "status": "prepared",
                "attempts": 0,
                "azure_voice": worker_unit.azure_voice,
                "source_audio_sha256": worker_unit.source_audio_sha256,
                "source_embedding_sha256": worker_unit.source_embedding_sha256,
                "target_embedding_sha256": worker_unit.target_embedding_sha256,
                "openvoice_revision": revision,
            }
        self._write_plan(paths.plan, plan)
        job.media["dubbing_plan_sha256"] = checksum(paths.plan)
        job.media["openvoice_plan_sha256"] = checksum(paths.openvoice_plan)
        self.jobs.save(job)
        return value

    def queue_openvoice_assets(
        self, job: JobManifest, *, force: bool = False
    ) -> dict[str, Any]:
        """Promote every immutable asset-plan input with generation preconditions."""

        if job.state != "azure_tts_ready":
            raise ValueError(
                f"OpenVoice asset queue requires azure_tts_ready, not {job.state}"
            )
        paths = self.paths(job.job_id)
        if not paths.openvoice_asset_plan.is_file():
            raise ValueError("Prepare the OpenVoice asset plan before queueing it")
        value = read_json(paths.openvoice_asset_plan)
        plan = OpenVoiceAssetPlan.from_dict(value)
        if plan.job_id != job.job_id or plan.lease_identity != "unbound":
            raise ValueError("OpenVoice asset plan has an invalid server handoff identity")
        rights_path = paths.review_root / "voice_rights.json"
        if not rights_path.is_file():
            raise PermissionError(
                "Voice-rights review artifact is required before GPU asset queueing"
            )
        require_generation_rights(read_json(rights_path))
        if not self.gcs:
            self.gcs = GCSStore(self.settings.gcs_bucket, self.settings.gcs_prefix)
        expected_checkpoints = dict(self.settings.openvoice_checkpoint_hashes)
        if not expected_checkpoints:
            raise ValueError(
                "Pinned OpenVoice checkpoint hashes are required before GPU queueing"
            )
        attempt = self.jobs.begin_attempt(
            job, "openvoice_asset_queue", forced=force
        )
        try:
            inputs: dict[Path, str] = {}
            cache_seed_count = 0
            for source in plan.sources:
                inputs[source.calibration_audio_path] = source.calibration_audio_sha256
                inputs[source.calibration_manifest_path] = (
                    source.calibration_manifest_sha256
                )
                source_embedding = self._resolve_job_path(
                    paths, source.source_embedding_path
                )
                source_embedding_manifest = self._resolve_job_path(
                    paths, source.source_embedding_manifest_path
                )
                if source_embedding.is_file() != source_embedding_manifest.is_file():
                    raise ValueError(
                        f"OpenVoice source cache seed is partial for {source.azure_voice}"
                    )
                if source_embedding.is_file():
                    seed_value = read_json(source_embedding_manifest)
                    checkpoints = seed_value.get("checkpoint_hashes")
                    if not isinstance(checkpoints, Mapping) or not checkpoints:
                        raise ValueError(
                            f"OpenVoice source cache checkpoint identity is missing for {source.azure_voice}"
                        )
                    self._verify_embedding_manifest(
                        source_embedding_manifest,
                        schema=OPENVOICE_SOURCE_EMBEDDING_MANIFEST_SCHEMA,
                        expected_artifact_hash=str(
                            seed_value.get("artifact_sha256") or ""
                        ),
                        expected_values={
                            "azure_voice": source.azure_voice,
                            "calibration_audio_sha256": source.calibration_audio_sha256,
                            "calibration_manifest_sha256": source.calibration_manifest_sha256,
                            "source_embedding_sha256": checksum(source_embedding),
                            "openvoice_revision": plan.openvoice_revision,
                            "checkpoint_hashes": dict(checkpoints),
                            "status": "ready",
                        },
                    )
                    inputs[source.source_embedding_path] = checksum(
                        source_embedding
                    )
                    inputs[source.source_embedding_manifest_path] = checksum(
                        source_embedding_manifest
                    )
                    cache_seed_count += 1
            for speaker in plan.speakers:
                inputs[speaker.reference_audio_path] = speaker.reference_audio_sha256
                inputs[speaker.reference_manifest_path] = (
                    speaker.reference_manifest_sha256
                )
                for candidate in speaker.voice_candidates:
                    inputs[candidate.source_audio_path] = candidate.source_audio_sha256
            promoted: dict[str, str] = {}
            for relative_path, expected in sorted(
                inputs.items(), key=lambda item: item[0].as_posix()
            ):
                local = self._resolve_job_path(paths, relative_path)
                if not local.is_file() or checksum(local) != expected:
                    raise ValueError(
                        f"OpenVoice asset-plan input hash mismatch: {relative_path}"
                    )
                relative = relative_path.as_posix()
                promoted[relative] = self._promote_file(
                    job.job_id, relative, local, expected
                )
            plan_file_hash = checksum(paths.openvoice_asset_plan)
            plan_uri = self._promote_file(
                job.job_id,
                "dubbing/openvoice/asset_plan.json",
                paths.openvoice_asset_plan,
                plan_file_hash,
            )
            job.objects["openvoice_asset_plan"] = plan_uri
            job.media["openvoice_asset_plan_sha256"] = plan_file_hash
            job.media["openvoice_asset_plan_artifact_sha256"] = value[
                "artifact_sha256"
            ]
            job.media["openvoice_asset_queue"] = {
                "status": "queued",
                "attempt": attempt,
                "input_count": len(promoted),
                "queued_at": utcnow(),
            }
            self.jobs.finish_attempt(
                job, "openvoice_asset_queue", attempt, "completed"
            )
            self._upload_job_status(job)
            return {
                "schema_version": "openvoice-asset-queue-receipt-v1",
                "job_id": job.job_id,
                "state": "queued",
                "plan_uri": plan_uri,
                "plan_file_sha256": plan_file_hash,
                "plan_artifact_sha256": value["artifact_sha256"],
                "input_count": len(promoted),
                "source_cache_seed_count": cache_seed_count,
                "worker_mode": "assets",
                "lease_task": "openvoice_asset_preparation",
            }
        except Exception:
            self.jobs.finish_attempt(job, "openvoice_asset_queue", attempt, "failed")
            raise

    def reconcile_openvoice_assets(self, job: JobManifest) -> dict[str, Any]:
        """Materialise verified GPU preflight outputs and update the source cache."""

        if job.state != "azure_tts_ready":
            raise ValueError(
                f"OpenVoice asset reconciliation requires azure_tts_ready, not {job.state}"
            )
        if not self.gcs:
            self.gcs = GCSStore(self.settings.gcs_bucket, self.settings.gcs_prefix)
        paths = self.paths(job.job_id)
        plan_value = read_json(paths.openvoice_asset_plan)
        plan = OpenVoiceAssetPlan.from_dict(plan_value)
        plan_file_hash = checksum(paths.openvoice_asset_plan)
        claim, _ = self.gcs.download_json(
            job.job_id, "worker/openvoice_asset_preparation_claim.json"
        )
        if claim.get("status") != "completed":
            return {
                "state": "pending",
                "next": "Wait for the openvoice_asset_preparation lease to complete",
            }
        if claim.get("job_id") != job.job_id or claim.get("task_type") != (
            "openvoice_asset_preparation"
        ):
            raise ValueError("Completed OpenVoice asset lease identity is invalid")

        attempt = self.jobs.begin_attempt(
            job, "openvoice_asset_reconciliation", forced=False
        )
        staged: list[tuple[Path, Path]] = []
        manifest_download: Path | None = None
        try:
            manifest_download = self._stage_gcs_download(
                job.job_id,
                "dubbing/openvoice/asset_manifest.json",
                paths.openvoice_asset_manifest,
            )
            manifest_file_hash = checksum(manifest_download)
            remote = read_json(manifest_download)
            expected_asset_count = len(plan.sources) + len(plan.speakers) + sum(
                len(speaker.voice_candidates) for speaker in plan.speakers
            )
            expected = {
                "schema_version": OPENVOICE_ASSET_MANIFEST_SCHEMA,
                "job_id": job.job_id,
                "backend": "openvoice",
                "backend_revision": plan.openvoice_revision,
                "server_plan_artifact_sha256": plan.artifact_sha256,
                "server_plan_file_sha256": plan_file_hash,
                "state": "completed",
                "pending_assets": 0,
                "completed_assets": expected_asset_count,
                "lease_identity": claim.get("worker_id"),
            }
            mismatches = [
                key for key, expected_value in expected.items()
                if remote.get(key) != expected_value
            ]
            if mismatches:
                raise ValueError(
                    "OpenVoice asset worker manifest identity mismatch: "
                    + ", ".join(mismatches)
                )
            if not re.fullmatch(
                r"[0-9a-f]{64}", str(remote.get("plan_artifact_sha256") or "")
            ):
                raise ValueError("Worker-local OpenVoice asset-plan identity is invalid")
            runtime = remote.get("runtime") or {}
            if (
                runtime.get("openvoice_revision") != plan.openvoice_revision
                or runtime.get("cuda_available") is not True
                or not str(runtime.get("device") or "").startswith("cuda")
            ):
                raise ValueError("OpenVoice GPU runtime identity is invalid")
            checkpoint_hashes = remote.get("checkpoint_hashes")
            expected_checkpoints = dict(self.settings.openvoice_checkpoint_hashes)
            if not expected_checkpoints or checkpoint_hashes != expected_checkpoints:
                raise ValueError("OpenVoice checkpoint hashes do not match server pins")

            remote_sources = remote.get("sources") or {}
            if set(remote_sources) != {source.azure_voice for source in plan.sources}:
                raise ValueError("OpenVoice asset manifest source set changed")
            remote_speakers = remote.get("speakers") or {}
            if set(remote_speakers) != {
                speaker.speaker_id for speaker in plan.speakers
            }:
                raise ValueError("OpenVoice asset manifest speaker set changed")

            for source in plan.sources:
                record = remote_sources[source.azure_voice]
                result = self._completed_asset_result(
                    record,
                    f"source:{source.azure_voice}",
                    self.settings.openvoice_max_attempts,
                )
                if (
                    result.get("azure_voice") != source.azure_voice
                    or result.get("calibration_audio_sha256")
                    != source.calibration_audio_sha256
                    or result.get("calibration_manifest_sha256")
                    != source.calibration_manifest_sha256
                    or result.get("calibration_identity")
                    != source.calibration.identity()
                    or result.get("openvoice_revision") != plan.openvoice_revision
                    or result.get("cache_reused") not in {True, False}
                    or (
                        result.get("cache_reused") is True
                        and result.get("checkpoint_hashes")
                        != dict(checkpoint_hashes)
                    )
                ):
                    raise ValueError(
                        f"OpenVoice source result identity changed for {source.azure_voice}"
                    )
                embedding = self._stage_planned_output(
                    job,
                    paths,
                    source.source_embedding_path,
                    str(result.get("source_embedding_sha256") or ""),
                    staged,
                )
                embedding_manifest = self._stage_planned_output(
                    job,
                    paths,
                    source.source_embedding_manifest_path,
                    str(result.get("source_embedding_manifest_sha256") or ""),
                    staged,
                )
                self._verify_embedding_manifest(
                    embedding_manifest,
                    schema=OPENVOICE_SOURCE_EMBEDDING_MANIFEST_SCHEMA,
                    expected_artifact_hash=str(
                        result.get("source_embedding_manifest_artifact_sha256")
                        or ""
                    ),
                    expected_values={
                        "azure_voice": source.azure_voice,
                        "calibration_audio_sha256": source.calibration_audio_sha256,
                        "calibration_manifest_sha256": source.calibration_manifest_sha256,
                        "source_embedding_sha256": checksum(embedding),
                        "openvoice_revision": plan.openvoice_revision,
                        "checkpoint_hashes": dict(checkpoint_hashes),
                        "status": "ready",
                    },
                )

            candidate_qc_pending = False
            for speaker in plan.speakers:
                record = remote_speakers[speaker.speaker_id]
                result = self._completed_asset_result(
                    record,
                    f"speaker:{speaker.speaker_id}",
                    self.settings.openvoice_max_attempts,
                )
                if (
                    result.get("speaker_id") != speaker.speaker_id
                    or result.get("reference_audio_sha256")
                    != speaker.reference_audio_sha256
                    or result.get("reference_manifest_sha256")
                    != speaker.reference_manifest_sha256
                    or result.get("job_local_identity") is not True
                    or result.get("openvoice_revision") != plan.openvoice_revision
                ):
                    raise ValueError(
                        f"OpenVoice target result identity changed for {speaker.speaker_id}"
                    )
                target = self._stage_planned_output(
                    job,
                    paths,
                    speaker.target_embedding_path,
                    str(result.get("target_embedding_sha256") or ""),
                    staged,
                )
                target_manifest = self._stage_planned_output(
                    job,
                    paths,
                    speaker.target_embedding_manifest_path,
                    str(result.get("target_embedding_manifest_sha256") or ""),
                    staged,
                )
                self._verify_embedding_manifest(
                    target_manifest,
                    schema=OPENVOICE_TARGET_EMBEDDING_MANIFEST_SCHEMA,
                    expected_artifact_hash=str(
                        result.get("target_embedding_manifest_artifact_sha256")
                        or ""
                    ),
                    expected_values={
                        "speaker_id": speaker.speaker_id,
                        "reference_audio_sha256": speaker.reference_audio_sha256,
                        "reference_manifest_sha256": speaker.reference_manifest_sha256,
                        "target_embedding_sha256": checksum(target),
                        "job_local_identity": True,
                        "openvoice_revision": plan.openvoice_revision,
                        "checkpoint_hashes": dict(checkpoint_hashes),
                        "status": "ready",
                    },
                )
                candidate_records = record.get("candidates") or {}
                if set(candidate_records) != {
                    candidate.candidate_id for candidate in speaker.voice_candidates
                }:
                    raise ValueError(
                        f"OpenVoice candidate set changed for {speaker.speaker_id}"
                    )
                results_by_id: dict[str, dict[str, Any]] = {}
                source_hashes = {
                    source.azure_voice: str(
                        remote_sources[source.azure_voice]["result"][
                            "source_embedding_sha256"
                        ]
                    )
                    for source in plan.sources
                }
                for candidate in speaker.voice_candidates:
                    candidate_result = self._completed_asset_result(
                        candidate_records[candidate.candidate_id],
                        f"candidate:{candidate.candidate_id}",
                        self.settings.openvoice_max_attempts,
                    )
                    expected_candidate = {
                        "backend": "openvoice",
                        "backend_revision": plan.openvoice_revision,
                        "candidate_id": candidate.candidate_id,
                        "speaker_id": speaker.speaker_id,
                        "azure_voice": candidate.azure_voice,
                        "voice": candidate.azure_voice,
                        "source_audio_sha256": candidate.source_audio_sha256,
                        "source_embedding_sha256": source_hashes[
                            candidate.azure_voice
                        ],
                        "target_embedding_sha256": str(
                            result["target_embedding_sha256"]
                        ),
                        "calibration_text_sha256": candidate.calibration_text_sha256,
                        "sample_rate": candidate.sample_rate,
                        "channels": 1,
                    }
                    if any(
                        candidate_result.get(key) != expected_value
                        for key, expected_value in expected_candidate.items()
                    ):
                        raise ValueError(
                            f"OpenVoice voice candidate identity changed for {candidate.candidate_id}"
                        )
                    candidate_audio = self._stage_planned_output(
                        job,
                        paths,
                        candidate.output_path,
                        str(candidate_result.get("output_sha256") or ""),
                        staged,
                    )
                    quality = inspect_wav(candidate_audio)
                    if (
                        not quality.get("valid")
                        or quality.get("channels") != 1
                        or quality.get("sample_rate") != candidate.sample_rate
                    ):
                        raise ValueError(
                            f"OpenVoice voice candidate WAV is invalid for {candidate.candidate_id}"
                        )
                    self._verify_conversion_durations(candidate_result)
                    evaluation_status = candidate_result.get("evaluation_status")
                    if evaluation_status == "completed":
                        self._validate_candidate_metrics(candidate_result)
                    elif evaluation_status == "pending_external_qc":
                        candidate_qc_pending = True
                    else:
                        raise ValueError(
                            f"OpenVoice candidate evaluation status is invalid for {candidate.candidate_id}"
                        )
                    results_by_id[candidate.candidate_id] = candidate_result
                if speaker.candidate_manifest_path is not None:
                    candidate_manifest_path = self._stage_gcs_download(
                        job.job_id,
                        speaker.candidate_manifest_path.as_posix(),
                        self._resolve_job_path(paths, speaker.candidate_manifest_path),
                    )
                    staged.append(
                        (
                            candidate_manifest_path,
                            self._resolve_job_path(
                                paths, speaker.candidate_manifest_path
                            ),
                        )
                    )
                    candidate_manifest = read_json(candidate_manifest_path)
                    self._verify_canonical_artifact(
                        candidate_manifest,
                        str(record.get("candidate_manifest_sha256") or ""),
                    )
                    if (
                        candidate_manifest.get("schema_version")
                        != OPENVOICE_VOICE_CANDIDATES_SCHEMA
                        or candidate_manifest.get("job_id") != job.job_id
                        or candidate_manifest.get("speaker_id")
                        != speaker.speaker_id
                        or candidate_manifest.get("openvoice_revision")
                        != plan.openvoice_revision
                        or candidate_manifest.get("reference_audio_sha256")
                        != speaker.reference_audio_sha256
                        or candidate_manifest.get("target_embedding_sha256")
                        != result.get("target_embedding_sha256")
                    ):
                        raise ValueError(
                            f"OpenVoice candidate manifest identity changed for {speaker.speaker_id}"
                        )
                    manifest_results = {
                        str(item.get("candidate_id")): item
                        for item in candidate_manifest.get("candidates", [])
                    }
                    if manifest_results != results_by_id:
                        raise ValueError(
                            f"OpenVoice candidate manifest results changed for {speaker.speaker_id}"
                        )

            for temporary, destination in staged:
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary.replace(destination)
            staged.clear()
            for source in plan.sources:
                source_path = self._resolve_job_path(
                    paths, source.source_embedding_path
                )
                source_manifest_path = self._resolve_job_path(
                    paths, source.source_embedding_manifest_path
                )
                cache_root = (
                    self.settings.work_dir
                    / "cache/openvoice/azure_sources"
                    / source.azure_voice
                )
                self._atomic_copy(source_path, cache_root / "source_embedding.bin")
                cache_manifest_path = cache_root / "manifest.json"
                cache_manifest = read_json(cache_manifest_path)
                source_result = remote_sources[source.azure_voice]["result"]
                cache_manifest.update(
                    {
                        "embedding_path": str(cache_root / "source_embedding.bin"),
                        "embedding_sha256": source_result[
                            "source_embedding_sha256"
                        ],
                        "embedding_manifest_path": str(
                            cache_root / "source_embedding_manifest.json"
                        ),
                        "checkpoint_hashes": dict(checkpoint_hashes),
                        "status": "ready",
                    }
                )
                atomic_write_json(cache_manifest_path, cache_manifest)
                reusable_manifest = read_json(source_manifest_path)
                reusable_manifest.update(
                    {
                        "calibration_manifest_sha256": checksum(
                            cache_manifest_path
                        ),
                        "source_embedding_path": str(
                            cache_root / "source_embedding.bin"
                        ),
                        "source_embedding_sha256": source_result[
                            "source_embedding_sha256"
                        ],
                        "checkpoint_hashes": dict(checkpoint_hashes),
                        "status": "ready",
                    }
                )
                reusable_manifest["artifact_sha256"] = self._canonical_hash(
                    {
                        key: value
                        for key, value in reusable_manifest.items()
                        if key != "artifact_sha256"
                    }
                )
                atomic_write_json(
                    cache_root / "source_embedding_manifest.json",
                    reusable_manifest,
                )

            dubbing_plan = read_json(paths.plan)
            dubbing_plan["azure_source_calibrations"] = {
                source.azure_voice: {
                    "calibration_audio_sha256": source.calibration_audio_sha256,
                    "calibration_manifest_sha256": source.calibration_manifest_sha256,
                    "source_embedding_sha256": remote_sources[source.azure_voice][
                        "result"
                    ]["source_embedding_sha256"],
                    "source_embedding_manifest_sha256": remote_sources[
                        source.azure_voice
                    ]["result"]["source_embedding_manifest_sha256"],
                    "openvoice_revision": plan.openvoice_revision,
                    "status": "ready",
                }
                for source in plan.sources
            }
            for speaker in plan.speakers:
                speaker_result = remote_speakers[speaker.speaker_id]["result"]
                speaker_plan = dubbing_plan["speakers"][speaker.speaker_id]
                speaker_plan["reference"] = {
                    "path": str(speaker.reference_audio_path),
                    "sha256": speaker.reference_audio_sha256,
                    "manifest_sha256": speaker.reference_manifest_sha256,
                    "job_local_identity": True,
                }
                speaker_plan["target_embedding"] = {
                    "path": str(speaker.target_embedding_path),
                    "sha256": speaker_result["target_embedding_sha256"],
                    "manifest_sha256": speaker_result[
                        "target_embedding_manifest_sha256"
                    ],
                    "openvoice_revision": plan.openvoice_revision,
                    "status": "ready",
                }
            for unit in dubbing_plan["units"]:
                speaker_result = remote_speakers[unit["speaker_id"]]["result"]
                unit["openvoice"].update(
                    {
                        "asset_status": "ready",
                        "target_embedding_sha256": speaker_result[
                            "target_embedding_sha256"
                        ],
                        "openvoice_revision": plan.openvoice_revision,
                    }
                )
            self._write_plan(paths.plan, dubbing_plan)

            atomic_write_json(paths.openvoice_asset_manifest, remote)
            receipt = {
                "schema_version": "openvoice-asset-reconciliation-v1",
                "job_id": job.job_id,
                "state": "completed",
                "worker_manifest_sha256": manifest_file_hash,
                "server_plan_file_sha256": plan_file_hash,
                "server_plan_artifact_sha256": plan.artifact_sha256,
                "source_count": len(plan.sources),
                "speaker_count": len(plan.speakers),
                "candidate_count": sum(
                    len(speaker.voice_candidates) for speaker in plan.speakers
                ),
                "candidate_qc_status": (
                    "pending_external_qc"
                    if candidate_qc_pending
                    else "completed"
                ),
                "completed_at": utcnow(),
            }
            receipt_path = paths.openvoice_root / "asset_reconciliation.json"
            atomic_write_json(receipt_path, receipt)
            job.media["openvoice_asset_manifest_sha256"] = checksum(
                paths.openvoice_asset_manifest
            )
            job.media["openvoice_asset_reconciliation"] = receipt
            job.objects["openvoice_asset_manifest"] = (
                "dubbing/openvoice/asset_manifest.json"
            )
            if "openvoice_assets" not in job.completed_stages:
                job.completed_stages.append("openvoice_assets")
            self.jobs.finish_attempt(
                job,
                "openvoice_asset_reconciliation",
                attempt,
                "completed",
            )
            if manifest_download is not None:
                manifest_download.unlink(missing_ok=True)
            return receipt
        except Exception:
            for temporary, _destination in staged:
                temporary.unlink(missing_ok=True)
            if manifest_download is not None:
                manifest_download.unlink(missing_ok=True)
            self.jobs.finish_attempt(
                job, "openvoice_asset_reconciliation", attempt, "failed"
            )
            raise

    def queue_openvoice(self, job: JobManifest, *, force: bool = False) -> dict[str, Any]:
        if job.state == "voice_conversion_queued" and not force:
            plan_path = self.paths(job.job_id).openvoice_plan
            return {
                "schema_version": "openvoice-conversion-queue-receipt-v1",
                "job_id": job.job_id,
                "state": "voice_conversion_queued",
                "plan_file_sha256": checksum(plan_path),
                "worker_mode": "conversion",
            }
        if job.state != "azure_tts_ready":
            raise ValueError(f"Voice conversion queue requires azure_tts_ready, not {job.state}")
        paths = self.paths(job.job_id)
        rights_path = paths.review_root / "voice_rights.json"
        if not rights_path.is_file():
            raise PermissionError("Voice-rights review artifact is required before OpenVoice queueing")
        require_generation_rights(read_json(rights_path))
        value = read_json(paths.openvoice_plan)
        plan = self._validate_openvoice_conversion_plan(job, value)
        if not self.gcs:
            self.gcs = GCSStore(self.settings.gcs_bucket, self.settings.gcs_prefix)
        if not self.settings.openvoice_checkpoint_hashes:
            raise ValueError(
                "Pinned OpenVoice checkpoint hashes are required before conversion queueing"
            )
        for unit in plan.units:
            for relative_path, expected in (
                (unit.source_audio_path, unit.source_audio_sha256),
                (unit.source_embedding_path, unit.source_embedding_sha256),
                (unit.target_embedding_path, unit.target_embedding_sha256),
            ):
                relative = relative_path.as_posix()
                local = self._resolve_job_path(paths, relative_path)
                if checksum(local) != expected:
                    raise ValueError(f"OpenVoice plan input hash mismatch: {relative}")
                self._promote_file(job.job_id, relative, local, expected)
        for reference in value.get("speaker_references", {}).values():
            local = self.jobs.job_dir(job.job_id) / reference["path"]
            if checksum(local) != reference["sha256"]:
                raise ValueError("Speaker reference hash mismatch before queueing")
            self._promote_file(job.job_id, reference["path"], local, reference["sha256"])
        plan_file_hash = checksum(paths.openvoice_plan)
        job.objects["openvoice_plan"] = self._promote_file(
            job.job_id,
            "dubbing/openvoice/plan.json",
            paths.openvoice_plan,
            plan_file_hash,
        )
        job.transition("voice_conversion_queued")
        self.jobs.save(job)
        self._upload_job_status(job)
        return {
            "schema_version": "openvoice-conversion-queue-receipt-v1",
            "job_id": job.job_id,
            "state": job.state,
            "plan_uri": job.objects["openvoice_plan"],
            "plan_file_sha256": plan_file_hash,
            "unit_count": len(value["units"]),
            "worker_mode": "conversion",
            "lease_task": "openvoice_conversion",
        }

    def reconcile_openvoice(self, job: JobManifest) -> dict[str, Any]:
        if job.state not in {"voice_conversion_queued", "voice_conversion_running"}:
            raise ValueError("OpenVoice reconciliation requires a queued or running conversion")
        if not self.gcs:
            self.gcs = GCSStore(self.settings.gcs_bucket, self.settings.gcs_prefix)
        paths = self.paths(job.job_id)
        plan_value = read_json(paths.openvoice_plan)
        plan = self._validate_openvoice_conversion_plan(job, plan_value)
        plan_file_hash = checksum(paths.openvoice_plan)
        try:
            claim, _ = self.gcs.download_json(
                job.job_id, "worker/openvoice_conversion_claim.json"
            )
        except FileNotFoundError:
            return {
                "state": "pending",
                "next": "Wait for the openvoice_conversion lease claim",
            }
        if claim.get("status") != "completed":
            return {
                "state": str(claim.get("status") or "pending"),
                "next": "Wait for the openvoice_conversion lease to complete",
            }
        if (
            claim.get("job_id") != job.job_id
            or claim.get("task_type") != "openvoice_conversion"
            or not claim.get("worker_id")
        ):
            raise ValueError("Completed OpenVoice conversion lease identity is invalid")

        attempt = self.jobs.begin_attempt(
            job, "openvoice_reconciliation", forced=False
        )
        staged: list[tuple[Path, Path]] = []
        manifest_download: Path | None = None
        try:
            manifest_download = self._stage_gcs_download(
                job.job_id,
                "dubbing/openvoice/manifest.json",
                paths.openvoice_manifest,
            )
            remote = read_json(manifest_download)
            expected_top = {
                "schema_version": "openvoice-worker-manifest-v1",
                "job_id": job.job_id,
                "lease_identity": claim["worker_id"],
                "backend": "openvoice",
                "backend_revision": plan.openvoice_revision,
                "server_plan_file_sha256": plan_file_hash,
                "state": "completed",
                "completed_units": len(plan.units),
                "pending_units": 0,
            }
            mismatches = [
                key
                for key, expected in expected_top.items()
                if remote.get(key) != expected
            ]
            if mismatches:
                raise ValueError(
                    "OpenVoice conversion manifest identity mismatch: "
                    + ", ".join(mismatches)
                )
            runtime = remote.get("runtime") or {}
            if (
                runtime.get("openvoice_revision") != plan.openvoice_revision
                or runtime.get("cuda_available") is not True
                or not str(runtime.get("device") or "").startswith("cuda")
                or runtime.get("precision") not in {"fp16", "fp32", "bf16"}
            ):
                raise ValueError("OpenVoice conversion runtime identity is invalid")
            expected_checkpoints = dict(self.settings.openvoice_checkpoint_hashes)
            if not expected_checkpoints or remote.get("checkpoint_hashes") != (
                expected_checkpoints
            ):
                raise ValueError("OpenVoice conversion checkpoint hashes changed")
            records = remote.get("units")
            if not isinstance(records, Mapping) or set(records) != {
                unit.unit_id for unit in plan.units
            }:
                raise ValueError("OpenVoice conversion manifest unit set changed")

            results: dict[str, dict[str, Any]] = {}
            for unit in plan.units:
                record = records[unit.unit_id]
                result = self._completed_asset_result(
                    record,
                    f"unit:{unit.unit_id}",
                    self.settings.max_openvoice_attempts_per_unit,
                )
                attempts = int(record["attempts"])
                expected_record = {
                    "unit_id": unit.unit_id,
                    "speaker_id": unit.speaker_id,
                    "azure_voice": unit.azure_voice,
                    "input_identity_sha256": self._openvoice_unit_identity_hash(
                        unit
                    ),
                }
                if any(
                    record.get(key) != expected
                    for key, expected in expected_record.items()
                ):
                    raise ValueError(
                        f"OpenVoice conversion record identity changed for {unit.unit_id}"
                    )
                expected_result = {
                    "schema_version": "openvoice-conversion-result-v1",
                    "backend": "openvoice",
                    "backend_revision": plan.openvoice_revision,
                    "unit_id": unit.unit_id,
                    "speaker_id": unit.speaker_id,
                    "azure_voice": unit.azure_voice,
                    "source_audio_sha256": unit.source_audio_sha256,
                    "source_embedding_sha256": unit.source_embedding_sha256,
                    "target_embedding_sha256": unit.target_embedding_sha256,
                    "sample_rate": unit.sample_rate,
                    "channels": 1,
                    "attempt": attempts,
                }
                if any(
                    result.get(key) != expected
                    for key, expected in expected_result.items()
                ) or not str(result.get("device") or "").startswith("cuda"):
                    raise ValueError(
                        f"OpenVoice conversion result identity changed for {unit.unit_id}"
                    )
                self._verify_conversion_durations(result)
                converted = self._stage_planned_output(
                    job,
                    paths,
                    unit.output_path,
                    str(result.get("output_sha256") or ""),
                    staged,
                )
                quality = inspect_wav(converted)
                if (
                    not quality.get("valid")
                    or quality.get("channels") != 1
                    or quality.get("sample_rate") != unit.sample_rate
                ):
                    raise ValueError(
                        f"OpenVoice conversion WAV is invalid for {unit.unit_id}"
                    )
                results[unit.unit_id] = result

            for temporary, destination in staged:
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary.replace(destination)
            staged.clear()
            atomic_write_json(paths.openvoice_manifest, remote)
            self._update_plan_stage(job, "openvoice", results)
            
            # Run OpenVoice intelligibility audit BEFORE state transition
            # This ensures audit failures block the success state
            if hasattr(self, '_stt_backend') and self._stt_backend:
                self._audit_intelligibility(job, "openvoice", plan["units"], paths.openvoice_root / "turns")
            
            if job.state == "voice_conversion_queued":
                job.transition("voice_conversion_running")
            job.transition("voice_conversion_ready")
            job.providers["voice_conversion"] = "openvoice"
            job.media["openvoice_manifest_sha256"] = checksum(
                paths.openvoice_manifest
            )
            job.media["openvoice_reconciliation"] = {
                "schema_version": "openvoice-conversion-reconciliation-v1",
                "worker_manifest_sha256": checksum(manifest_download),
                "server_plan_file_sha256": plan_file_hash,
                "unit_count": len(plan.units),
                "completed_at": utcnow(),
            }
            
            if "openvoice_conversion" not in job.completed_stages:
                job.completed_stages.append("openvoice_conversion")
            self.jobs.finish_attempt(
                job, "openvoice_reconciliation", attempt, "completed"
            )
            manifest_download.unlink(missing_ok=True)
            return remote
        except Exception:
            for temporary, _destination in staged:
                temporary.unlink(missing_ok=True)
            if manifest_download is not None:
                manifest_download.unlink(missing_ok=True)
            self.jobs.finish_attempt(
                job, "openvoice_reconciliation", attempt, "failed"
            )
            raise

    def evaluate_compatibility(
        self,
        job: JobManifest,
        observations: list[UnitCompatibilityObservation] | None = None,
        thresholds: CompatibilityThresholds | None = None,
    ) -> dict[str, Any]:
        plan = self.prepare_dubbing(job)
        if observations is None:
            observations = [self._pending_observation(job, unit) for unit in plan["units"]]
        thresholds = thresholds or CompatibilityThresholds(
            version="provisional-unconfigured",
            calibrated=False,
            max_openvoice_word_error_rate=None,
            max_word_error_rate_degradation=None,
            min_protected_name_accuracy=None,
            min_number_accuracy=None,
            min_date_accuracy=None,
            min_negation_accuracy=None,
            min_speaker_similarity=None,
            max_abs_timing_error_ms=None,
            max_clipping_ratio=None,
            min_silence_ratio=None,
            max_silence_ratio=None,
        )
        evidence_binding = self._validate_compatibility_observations(
            job,
            plan,
            observations,
            require_alignment=thresholds.calibrated,
        )
        report = evaluate_compatibility(job.job_id, observations, thresholds)
        report["evidence_binding"] = evidence_binding
        paths = self.paths(job.job_id)
        write_compatibility_report(report, paths.compatibility_json, paths.compatibility_markdown)
        dubbing_plan = read_json(paths.plan)
        dubbing_plan["compatibility"] = {
            "state": report["state"],
            "human_review_required": True,
            "report_path": str(paths.compatibility_json),
            "report_sha256": checksum(paths.compatibility_json),
            "threshold_version": thresholds.version,
            "thresholds_calibrated": thresholds.calibrated,
        }
        self._write_plan(paths.plan, dubbing_plan)
        if job.state == "voice_conversion_ready":
            job.transition("compatibility_review")
        job.compatibility_gate = {
            "state": report["state"],
            "report_path": str(paths.compatibility_json),
            "report_sha256": checksum(paths.compatibility_json),
            "threshold_version": thresholds.version,
            "calibrated": thresholds.calibrated,
        }
        if report["state"] in {"pending", "needs_human_review"}:
            job.review_readiness = "compatibility_pending"
        elif report["state"] == "passed":
            job.review_readiness = "needs_timing_review"
        else:
            job.review_readiness = "compatibility_failed"
        self.jobs.save(job)
        return report

    def align(
        self,
        job: JobManifest,
        *,
        skip_openvoice: bool = False,
        require_compatibility_pass: bool = True,
        force: bool = False,
        azure_only: bool = False,
    ) -> dict[str, Any]:
        gate = job.compatibility_gate.get("state", "pending")
        
        # For azure_only mode, skip compatibility gate requirements
        if azure_only:
            skip_openvoice = True
            require_compatibility_pass = False
            production_authorized = True
            allowed = {"azure_tts_ready"}
        else:
            if require_compatibility_pass and not skip_openvoice and gate != "passed":
                raise CompatibilityGateFailure(f"OpenVoice compatibility gate is {gate}")
            production_authorized = not skip_openvoice and gate == "passed"
            allowed = {"azure_tts_ready"} if skip_openvoice else {"compatibility_review"}
        
        if job.state not in allowed and not (
            production_authorized and force and job.state == "alignment_ready"
        ):
            raise ValueError(f"Alignment cannot start from {job.state}")
        if production_authorized and force and job.state == "alignment_ready":
            self._forced_stage_reset(job, "alignment", "compatibility_review")
        if production_authorized:
            job.transition("alignment_running")
            self.jobs.save(job)
        plan = self.prepare_dubbing(job)
        paths = self.paths(job.job_id)
        results = []
        for unit in plan["units"]:
            source = paths.azure_turn(unit["unit_id"]) if skip_openvoice else paths.openvoice_turn(unit["unit_id"])
            result = align_converted_unit(
                source,
                paths.aligned_turn(unit["unit_id"]),
                unit,
                min_speed=self.settings.post_conversion_min_speed,
                max_speed=self.settings.post_conversion_max_speed,
                tolerance_ms=self.settings.final_timing_tolerance_ms,
            )
            results.append(result)
        manifest = {
            "schema_version": "alignment-manifest-v1",
            "source_backend": "azure_tts" if skip_openvoice else "openvoice",
            "readiness": (
                "azure_tts_production_ready"
                if azure_only
                else "compatibility_passed"
                if production_authorized
                else "azure_tts_only_review"
                if skip_openvoice
                else "compatibility_evidence_only"
            ),
            "production_authorized": production_authorized,
            "compatibility_gate_state": gate,
            "azure_only_mode": azure_only,
            "units": results,
        }
        atomic_write_json(paths.alignment_manifest, manifest)
        if production_authorized:
            self._update_plan_stage(
                job,
                "alignment",
                {str(item["unit_id"]): item for item in results},
            )
            
            # Run aligned intelligibility audit BEFORE state transition
            # This ensures audit failures block the success state
            if hasattr(self, '_stt_backend') and self._stt_backend:
                plan = self.prepare_dubbing(job)
                self._audit_intelligibility(job, "aligned", plan["units"], paths.aligned_root / "turns")
            
            job.review_readiness = "needs_background_review"
            job.transition("alignment_ready")
        else:
            evidence_plan = read_json(paths.plan)
            by_unit = {str(item["unit_id"]): item for item in results}
            for unit in evidence_plan["units"]:
                unit["alignment"] = {
                    "status": "evidence_only",
                    "production_authorized": False,
                    "result": by_unit[unit["unit_id"]],
                }
            self._write_plan(paths.plan, evidence_plan)
            job.review_readiness = (
                "azure_tts_only_review"
                if skip_openvoice
                else "compatibility_pending"
            )
        self.jobs.save(job)
        return manifest

    def prepare_background(
        self,
        job: JobManifest,
        *,
        clean_stem: Path | None = None,
        separation_provider: SeparationProvider | None = None,
        residual_detector: ResidualSpeechDetector | None = None,
        proof_of_concept: bool = False,
        azure_only: bool = False,
    ) -> dict[str, Any]:
        paths = self.paths(job.job_id)
        if not paths.alignment_manifest.is_file():
            raise CompatibilityGateFailure(
                "Production background requires a verified alignment manifest"
            )
        alignment = read_json(paths.alignment_manifest)
        
        # For azure_only mode, accept azure_tts_production_ready alignment
        if azure_only:
            if (
                alignment.get("readiness") != "azure_tts_production_ready"
                or alignment.get("azure_only_mode") is not True
                or alignment.get("source_backend") != "azure_tts"
            ):
                raise CompatibilityGateFailure(
                    "Azure-only background requires azure_tts_production_ready alignment"
                )
        else:
            if (
                job.compatibility_gate.get("state") != "passed"
                or alignment.get("readiness") != "compatibility_passed"
                or alignment.get("production_authorized") is not True
                or alignment.get("source_backend") != "openvoice"
            ):
                raise CompatibilityGateFailure(
                    "Evidence-only alignment cannot authorize background production"
                )
        
        if job.state != "alignment_ready":
            raise ValueError("Background preparation requires alignment_ready")
        job.transition("background_processing")
        self.jobs.save(job)
        transcript = read_json(self.jobs.job_dir(job.job_id) / "analysis/transcript_en.json")
        
        # Use separate preview paths for proof-of-concept
        output_path = paths.preview_background if proof_of_concept else paths.background
        manifest_path = paths.preview_root / "background_manifest.json" if proof_of_concept else paths.background_manifest
        
        manifest = prepare_background(
            paths.mix_source_audio,
            output_path,
            source_dialogue=transcript["segments"],
            clean_stem=clean_stem,
            cached_separation=self.jobs.job_dir(job.job_id) / "audio/separated_background.wav",
            separation_provider=separation_provider,
            reconstructed_ambience=self.jobs.job_dir(job.job_id) / "audio/reconstructed_ambience.wav",
            allow_review_ducking=proof_of_concept,
            azure_only=azure_only,
        )
        # For review-only ducked background, skip residual QC
        if manifest.get("mode") == "original_mix_ducked_review_only":
            residual = {
                "status": "not_run",
                "passed": False,
                "reason": "Review-only ducked source; residual source speech may remain",
            }
        elif residual_detector is not None:
            residual = residual_dialogue_qc(
                output_path,
                " ".join(str(item.get("source_text", "")) for item in transcript["segments"]),
                residual_detector,
            )
        else:
            residual = {
                "status": "not_run",
                "passed": False,
                "reason": "Configure an injectable speech detector/retranscriber",
            }
        write_background_manifest(manifest_path, manifest, residual)
        dubbing_plan = read_json(paths.plan)
        
        # Check residual dialogue QC for production safety
        # Allow only explicitly approved production states
        # Skip this check for review-only ducked background
        approved_residual_states = {"passed", "clean", "no_residual"}
        residual_status = residual.get("status", "not_run")
        review_only_mode = manifest.get("mode") == "original_mix_ducked_review_only"
        
        if not proof_of_concept and not review_only_mode and residual_status not in approved_residual_states:
            raise ValueError(
                f"Residual dialogue QC status '{residual_status}' is not approved for production. "
                f"Approved states: {', '.join(approved_residual_states)}. "
                f"Reason: {residual.get('reason', 'Unknown reason')}. "
                "Use proof_of_concept=True for preview-only ducking, or provide a clean stem."
            )
        
        dubbing_plan["background"] = {
            "status": "ready",
            "mode": manifest["mode"],
            "manifest_path": str(manifest_path),
            "manifest_sha256": checksum(manifest_path),
            "residual_dialogue_status": residual["status"],
            "proof_of_concept_ducking": proof_of_concept,
            "readiness": manifest.get("readiness", "production_ready"),
            "production_authorized": manifest.get("production_authorized", True),
        }
        self._write_plan(paths.plan, dubbing_plan)
        
        if proof_of_concept:
            job.review_readiness = "preview_review_only"
            # Do not transition to background_ready for previews
        elif review_only_mode:
            job.review_readiness = "review_preview_only"
            job.transition("background_ready")
        else:
            job.review_readiness = "needs_background_review"
            job.transition("background_ready")
        self.jobs.save(job)
        return {**manifest, "residual_dialogue_qc": residual}

    def mix(self, job: JobManifest, *, proof_of_concept: bool = False, azure_only: bool = False) -> dict[str, Any]:
        if job.state != "background_ready":
            raise ValueError("Mixing requires background_ready")
        paths = self.paths(job.job_id)
        alignment = read_json(paths.alignment_manifest)
        # For azure_only mode, accept azure_tts_production_ready alignment
        if azure_only:
            if (
                alignment.get("readiness") != "azure_tts_production_ready"
                or alignment.get("azure_only_mode") is not True
                or alignment.get("source_backend") != "azure_tts"
            ):
                raise CompatibilityGateFailure(
                    "Azure-only mix requires azure_tts_production_ready alignment"
                )
        else:
            if (
                job.compatibility_gate.get("state") != "passed"
                or alignment.get("production_authorized") is not True
                or alignment.get("readiness") != "compatibility_passed"
            ):
                raise CompatibilityGateFailure(
                    "Compatibility-passed OpenVoice alignment is required before mixing"
                )
        
        # For proof-of-concept, use preview background
        background_path = paths.preview_background if proof_of_concept else paths.background
        final_mix_path = paths.preview_final_mix if proof_of_concept else paths.final_mix
        manifest_path = paths.preview_root / "mix_manifest.json" if proof_of_concept else (self.jobs.job_dir(job.job_id) / "audio/mix_manifest.json")
        
        # Read background manifest to inherit authorization
        background_manifest_path = paths.preview_root / "background_manifest.json" if proof_of_concept else paths.background_manifest
        background_manifest = read_json(background_manifest_path)
        
        plan = self.prepare_dubbing(job)
        aligned_by_id = {item["unit_id"]: item for item in alignment["units"]}
        clips = [
            {
                "unit_id": unit["unit_id"],
                "speaker_id": unit["speaker_id"],
                "start_ms": unit["start_ms"],
                "duration_ms": aligned_by_id[unit["unit_id"]]["final_duration_ms"],
                "path": str(paths.aligned_turn(unit["unit_id"])),
            }
            for unit in plan["units"]
        ]
        total_ms = round(float(job.source_duration) * 1000)
        dialogue = render_dialogue_tracks(
            clips,
            self.jobs.job_dir(job.job_id) / "audio/speakers",
            paths.dialogue,
            total_duration_ms=total_ms,
        )
        final_mix = mix_final_buses(
            background_path,
            paths.dialogue,
            final_mix_path,
            target_lufs=self.settings.target_lufs,
            target_lra=self.settings.target_lra,
            true_peak_dbtp=self.settings.true_peak_dbtp,
        )
        # Inherit authorization from background manifest
        background_readiness = background_manifest.get("readiness", "production_ready")
        background_authorized = background_manifest.get("production_authorized", True)
        
        manifest = {
            "schema_version": "mix-manifest-v1",
            "dialogue": dialogue,
            "final_mix": final_mix,
            "readiness": background_readiness,
            "production_authorized": background_authorized,
        }
        atomic_write_json(manifest_path, manifest)
        
        # Run final-mix intelligibility audit BEFORE state transition
        if hasattr(self, '_stt_backend') and self._stt_backend:
            # Get baseline WER from aligned stage for degradation comparison
            baseline_wer = None
            aligned_report_path = paths.qc_intelligibility_root / "aligned" / "report.json"
            if aligned_report_path.exists():
                aligned_report = read_json(aligned_report_path)
                baseline_wer = aligned_report.get("aggregate_blind_wer")
            
            # Audit final mix
            self._audit_intelligibility(
                job, "final-mix", plan["units"], final_mix_path.parent, baseline_wer=baseline_wer
            )
        
        if proof_of_concept:
            job.review_readiness = "preview_review_only"
            # Do not transition to mix_ready for previews
        else:
            job.transition("mix_ready")
        self.jobs.save(job)
        return manifest

    def subtitles(self, job: JobManifest) -> dict[str, Any]:
        paths = self.paths(job.job_id)
        plan = self.prepare_dubbing(job)
        alignment = read_json(paths.alignment_manifest)
        return write_subtitles(plan["units"], alignment["units"], paths.subtitle_root)

    def render(self, job: JobManifest, *, burn_subtitles: bool = False, azure_only: bool = False) -> dict[str, Any]:
        paths = self.paths(job.job_id)
        if job.state == "background_ready":
            self.mix(job, azure_only=azure_only)
            job = self.jobs.load(job.job_id)
        if job.state != "mix_ready":
            raise ValueError("Rendering requires mix_ready after a verified mix")
        if not paths.final_mix.is_file():
            raise ValueError("Rendering requires the canonical final mix")

        alignment = read_json(paths.alignment_manifest)
        # For azure_only mode, accept azure_tts_production_ready alignment.
        if azure_only:
            if (
                alignment.get("readiness") != "azure_tts_production_ready"
                or alignment.get("azure_only_mode") is not True
                or alignment.get("source_backend") != "azure_tts"
            ):
                raise CompatibilityGateFailure(
                    "Azure-only render requires azure_tts_production_ready alignment"
                )
        else:
            if (
                job.compatibility_gate.get("state") != "passed"
                or alignment.get("production_authorized") is not True
                or alignment.get("readiness") != "compatibility_passed"
            ):
                raise CompatibilityGateFailure(
                    "Compatibility-passed OpenVoice alignment is required before rendering"
                )

        subtitle_manifest = self.subtitles(job)
        mix_manifest_path = self.jobs.job_dir(job.job_id) / "audio/mix_manifest.json"
        mix_manifest = read_json(mix_manifest_path)
        job.transition("rendering")
        self.jobs.save(job)

        manifest = render_review_mp4(
            Path(job.local_source_path),
            paths.final_mix,
            paths.final_video,
            paths.render_manifest,
            subtitles=Path(subtitle_manifest["artifacts"]["srt"]["path"]) if burn_subtitles else None,
        )

        paths.review_video.parent.mkdir(parents=True, exist_ok=True)
        staged_review = paths.review_video.with_name(f".{paths.review_video.name}.partial")
        shutil.copyfile(paths.final_video, staged_review)
        staged_review.replace(paths.review_video)

        manifest = {
            **manifest,
            "readiness": mix_manifest.get("readiness", "production_ready"),
            "production_authorized": mix_manifest.get("production_authorized", True),
            "review_alias_path": str(paths.review_video),
            "review_alias_sha256": checksum(paths.review_video),
        }
        atomic_write_json(paths.render_manifest, manifest)

        job.objects["final_video"] = str(paths.final_video)
        job.objects["review_video"] = str(paths.review_video)
        job.review_readiness = str(manifest["readiness"])
        job.transition("review_ready")
        self.jobs.save(job)
        return manifest

    def review(self, job: JobManifest) -> dict[str, Any]:
        paths = self.paths(job.job_id)
        plan = self.prepare_dubbing(job)
        compatibility: dict[str, Any] = (
            read_json(paths.compatibility_json)
            if paths.compatibility_json.is_file()
            else {"state": "pending", "units": []}
        )
        background = (
            read_json(paths.background_manifest)
            if paths.background_manifest.is_file()
            else {"mode": "not_run"}
        )
        mix_manifest = self.jobs.job_dir(job.job_id) / "audio/mix_manifest.json"
        mix = read_json(mix_manifest) if mix_manifest.is_file() else {}
        residual_qc = background.get("residual_dialogue_qc")
        residual_passed = (
            bool(residual_qc.get("passed", False))
            if isinstance(residual_qc, Mapping)
            else False
        )
        azure_manifest = self._read_optional(paths.azure_manifest, {"units": []})
        openvoice_manifest = self._read_optional(paths.openvoice_manifest, {"units": {}})
        alignment_manifest = self._read_optional(paths.alignment_manifest, {"units": []})
        translation = self._read_optional(paths.three_text_translation, {"units": []})
        worker_plan = self._read_optional(paths.openvoice_plan, {"units": []})
        azure_units = {str(item.get("turn_id")): item for item in azure_manifest.get("units", [])}
        openvoice_units = {
            str(unit_id): record.get("result", {})
            for unit_id, record in openvoice_manifest.get("units", {}).items()
        }
        alignment_units = {
            str(item.get("unit_id")): item for item in alignment_manifest.get("units", [])
        }
        compatibility_units = {
            str(item.get("unit_id")): item for item in compatibility.get("units", [])
        }
        references: dict[str, Any] = {}
        selections: dict[str, Any] = {}
        for speaker in sorted(plan["speakers"]):
            speaker_root = paths.job_root / "speakers" / speaker
            references[speaker] = self._read_optional(
                speaker_root / "reference_manifest.json", {"status": "missing"}
            )
            selections[speaker] = self._read_optional(
                speaker_root / "azure_voice_selection.json", {"status": "pending"}
            )
        source_embedding_hashes = {
            str(item.get("azure_voice")): item.get("source_embedding_sha256")
            for item in worker_plan.get("units", [])
            if item.get("source_embedding_sha256")
        }
        target_embedding_hashes = {
            str(item.get("speaker_id")): item.get("target_embedding_sha256")
            for item in worker_plan.get("units", [])
            if item.get("target_embedding_sha256")
        }
        artifact_paths = {
            "source_media": Path(job.local_source_path),
            "analysis_audio": paths.analysis_audio,
            "mix_source_audio": paths.mix_source_audio,
            "dubbing_plan": paths.plan,
            "azure_tts_manifest": paths.azure_manifest,
            "openvoice_manifest": paths.openvoice_manifest,
            "alignment_manifest": paths.alignment_manifest,
            "background_manifest": paths.background_manifest,
            "final_mix": paths.final_mix,
            "render_manifest": paths.render_manifest,
            "final_video": paths.final_video,
        }
        translation_flags = [
            flag
            for unit in translation.get("units", [])
            for flag in unit.get("human_review_flags", [])
        ]
        final_mix = mix.get("final_mix", {})
        loudness = final_mix.get("loudness") or {}
        failed_units = [
            unit_id
            for unit_id, item in compatibility_units.items()
            if str(item.get("state", "")).startswith("failed_")
        ]
        timing_errors = {
            unit_id: item.get("final_timing_error_ms")
            for unit_id, item in alignment_units.items()
        }
        timing_review_required = any(
            value is None or abs(int(value)) > self.settings.final_timing_tolerance_ms
            for value in timing_errors.values()
        ) or len(timing_errors) != len(plan["units"])
        audio_review_required = not bool(loudness.get("measured")) or bool(
            final_mix.get("clipping_count", 0)
        )
        values = {
            "source_media_hash": job.source_checksum,
            "source_duration_seconds": job.source_duration,
            "number_of_speakers": len(plan["speakers"]),
            "number_of_dubbing_units": len(plan["units"]),
            "number_of_overlaps": sum(bool(item["has_overlap"]) for item in plan["units"]),
            "azure_stt_backend": job.providers.get("azure_stt"),
            "azure_stt_api_version": job.providers.get("azure_stt_api_version"),
            "authoritative_diarization": "azure",
            "ai_provider": job.providers.get("translation", "azure-openai-gpt"),
            "gpt_model": job.providers.get("translation_model", self.settings.gpt_model),
            "gpt_prompt_version": job.providers.get("translation_prompt_version"),
            "azure_tts_voices": sorted(
                {str(item.get("voice")) for item in azure_units.values() if item.get("voice")}
            ),
            "openvoice_revision": self.settings.openvoice_revision or None,
            "speaker_reference_quality": references,
            "selected_base_voice_per_speaker": selections,
            "azure_source_embedding_hashes": source_embedding_hashes,
            "target_embedding_hashes": target_embedding_hashes,
            "azure_duration_ms_per_unit": {
                unit_id: item.get("duration_ms") for unit_id, item in azure_units.items()
            },
            "openvoice_duration_ms_per_unit": {
                unit_id: item.get("converted_duration_ms")
                for unit_id, item in openvoice_units.items()
            },
            "openvoice_duration_ratios": {
                unit_id: item.get("duration_ratio") for unit_id, item in openvoice_units.items()
            },
            "final_timing_errors_ms": timing_errors,
            "translation_repair_count": sum(
                len(item.get("timing_repair_history", []))
                for item in translation.get("units", [])
            ),
            "tts_attempts_per_unit": {
                unit_id: len(item.get("timing_search", {}).get("attempts", []))
                for unit_id, item in azure_units.items()
            },
            "openvoice_attempts_per_unit": {
                unit_id: item.get("attempt") for unit_id, item in openvoice_units.items()
            },
            "retranscription_results": {
                unit_id: {
                    "azure": item.get("azure_retranscription"),
                    "openvoice": item.get("openvoice_retranscription"),
                    "azure_word_error_rate": item.get("azure_word_error_rate"),
                    "openvoice_word_error_rate": item.get("openvoice_word_error_rate"),
                }
                for unit_id, item in compatibility_units.items()
            },
            "protected_term_results": {
                unit_id: item.get("protected_terms")
                for unit_id, item in compatibility_units.items()
            },
            "speaker_similarity_results": {
                unit_id: item.get("similarity") for unit_id, item in compatibility_units.items()
            },
            "residual_english_results": residual_qc,
            "compatibility": compatibility,
            "background_strategy": background.get("mode", "not_run"),
            "background_review_required": not residual_passed,
            "loudness": loudness,
            "true_peak_dbtp": loudness.get("true_peak_dbtp"),
            "clipping_count": final_mix.get("clipping_count"),
            "silence_ratios": {
                unit_id: (item.get("audio_quality") or {}).get("silence_ratio")
                for unit_id, item in compatibility_units.items()
            },
            "failed_units": failed_units,
            "fallback_units": (
                [item["unit_id"] for item in plan["units"]]
                if alignment_manifest.get("source_backend") == "azure_tts"
                else []
            ),
            "translation_review_required": bool(translation_flags),
            "pronunciation_review_required": compatibility.get("state")
            == "failed_pronunciation",
            "voice_review_required": compatibility.get("state")
            in {"pending", "needs_human_review", "failed_identity"},
            "timing_review_required": timing_review_required,
            "audio_review_required": audio_review_required,
            "artifact_hashes": {
                name: checksum(path) for name, path in artifact_paths.items() if path.is_file()
            },
            "attempt_history": job.attempt_history,
            "voice_rights_review": job.consent_review,
            "human_review_requirements": [
                "Voice rights and consent",
                "Editorial and factual accuracy",
                "Fluent isiZulu language and pronunciation",
                "Speaker identity, timing, background removal, and audio quality",
            ],
        }
        report = write_qc_report(job.job_id, values, paths.qc_json, paths.qc_markdown)
        job.review_readiness = report["readiness"]
        self.jobs.save(job)
        return report

    @staticmethod
    def _read_optional(path: Path, default: dict[str, Any]) -> dict[str, Any]:
        return read_json(path) if path.is_file() else default

    def _pronunciation_dictionary(self, job: JobManifest) -> PronunciationDictionary:
        path = self.paths(job.job_id).pronunciation_dictionary
        if path.is_file():
            dictionary = PronunciationDictionary.from_dict(read_json(path))
        else:
            dictionary = PronunciationDictionary(
                dictionary_version="v1",
                language=job.target_language,
                job_id=job.job_id,
            )
        dictionary = with_default_organisation_initialisms(dictionary)
        research_path = (
            self.paths(job.job_id).job_root
            / "analysis"
            / "autocorrection_name_research.json"
        )
        dictionary = with_web_researched_organisation_pronunciations(
            dictionary,
            read_json(research_path) if research_path.is_file() else None,
        )
        atomic_write_json(path, dictionary.to_dict())
        return dictionary

    def _require_voice_rights(
        self,
        job: JobManifest,
        plan: Mapping[str, Any],
        *,
        proof_of_concept: bool,
    ) -> dict[str, Any]:
        path = self.paths(job.job_id).review_root / "voice_rights.json"
        if path.is_file():
            review = read_json(path)
            if proof_of_concept and not review.get("generation_allowed"):
                statuses = {
                    str(item.get("consent_or_rights_status"))
                    for item in review.get("speakers", [])
                }
                if statuses <= {"unknown", "pending_review"}:
                    review = build_voice_rights_review(
                        list(review["speakers"]),
                        intended_use="internal compatibility proof",
                        proof_of_concept=True,
                    )
                    review["supersedes_sha256"] = checksum(path)
                    atomic_write_json(path, review)
        else:
            review = build_voice_rights_review(
                [
                    {
                        "speaker_id": speaker,
                        "reference_source": str(
                            self.jobs.job_dir(job.job_id) / "speakers" / speaker / "reference_reel.wav"
                        ),
                    }
                    for speaker in sorted(plan["speakers"])
                ],
                intended_use="internal compatibility proof" if proof_of_concept else "publication review",
                proof_of_concept=proof_of_concept,
            )
            atomic_write_json(path, review)
        require_generation_rights(review)
        job.consent_review = {
            "path": str(path),
            "proof_of_concept": bool(review.get("proof_of_concept")),
            "publication_approved": bool(review.get("publication_approved")),
        }
        self.jobs.save(job)
        return review

    def _pending_observation(self, job: JobManifest, unit: Mapping[str, Any]) -> UnitCompatibilityObservation:
        paths = self.paths(job.job_id)
        azure = paths.azure_turn(unit["unit_id"])
        converted = paths.openvoice_turn(unit["unit_id"])
        aligned = paths.aligned_turn(unit["unit_id"])
        reference = self.jobs.job_dir(job.job_id) / "speakers" / unit["speaker_id"] / "reference_reel.wav"
        azure_quality = inspect_wav(azure) if azure.is_file() else {}
        converted_quality = inspect_wav(converted) if converted.is_file() else {}
        alignment_by_unit: dict[str, Any] = {}
        if paths.alignment_manifest.is_file():
            alignment_by_unit = {
                str(item["unit_id"]): item
                for item in read_json(paths.alignment_manifest).get("units", [])
            }
        alignment = alignment_by_unit.get(str(unit["unit_id"]), {})
        spoken = str(unit["spoken_text"])
        protected_names = [str(value) for value in unit.get("protected_terms", [])]
        numbers = re.findall(r"\b\d+(?:[.,]\d+)?%?\b", spoken)
        dates = re.findall(r"\b\d{1,2}(?:[/-]\d{1,2}[/-]\d{2,4}|\s+\w+\s+\d{4})\b", spoken)
        negations = re.findall(r"\b(?:akukho|hhayi|ngeke|angeke|not|no|never)\b", spoken, re.I)
        return UnitCompatibilityObservation(
            unit_id=unit["unit_id"],
            speaker_id=unit["speaker_id"],
            intended_spoken_text=unit["spoken_text"],
            reference_audio_sha256=checksum(reference) if reference.is_file() else None,
            azure_audio_sha256=checksum(azure) if azure.is_file() else None,
            openvoice_audio_sha256=checksum(converted) if converted.is_file() else None,
            azure_duration_ms=(round(float(azure_quality["duration"]) * 1000) if azure_quality else None),
            openvoice_duration_ms=(
                round(float(converted_quality["duration"]) * 1000)
                if converted_quality
                else None
            ),
            final_timing_error_ms=(
                int(alignment["final_timing_error_ms"])
                if alignment.get("final_timing_error_ms") is not None
                else None
            ),
            azure_retranscription=None,
            openvoice_retranscription=None,
            azure_word_error_rate=None,
            openvoice_word_error_rate=None,
            protected_name_accuracy=None,
            number_accuracy=None,
            date_accuracy=None,
            negation_accuracy=None,
            speaker_similarity=None,
            azure_speaker_similarity=None,
            audio_quality_passed=(bool(converted_quality.get("valid")) if converted_quality else None),
            clipping_ratio=(
                float(converted_quality["clipped_ratio"])
                if converted_quality.get("clipped_ratio") is not None
                else None
            ),
            silence_ratio=(
                1.0 - float(converted_quality["non_silent_ratio"])
                if converted_quality.get("non_silent_ratio") is not None
                else None
            ),
            protected_name_count=len(protected_names),
            number_count=len(numbers),
            date_count=len(dates),
            negation_count=len(negations),
            audio_artifact_warnings=(
                () if converted_quality.get("valid") else ("OpenVoice WAV quality inspection did not pass",)
            ),
            aligned_audio_sha256=checksum(aligned) if aligned.is_file() else None,
        )

    def _validate_compatibility_observations(
        self,
        job: JobManifest,
        plan: Mapping[str, Any],
        observations: list[UnitCompatibilityObservation],
        *,
        require_alignment: bool,
    ) -> dict[str, Any]:
        """Bind gate evidence to the complete plan and materialised audio set."""

        planned_units = list(plan.get("units", []))
        planned_ids = [str(unit["unit_id"]) for unit in planned_units]
        observed_ids = [item.unit_id for item in observations]
        if observed_ids != planned_ids:
            raise ValueError(
                "Compatibility observations must exactly match deterministic plan order"
            )
        paths = self.paths(job.job_id)
        alignment_by_id: dict[str, Any] = {}
        if paths.alignment_manifest.is_file():
            alignment_value = read_json(paths.alignment_manifest)
            alignment_by_id = {
                str(item["unit_id"]): item
                for item in alignment_value.get("units", [])
            }
        if require_alignment and set(alignment_by_id) != set(planned_ids):
            raise ValueError(
                "Calibrated compatibility evidence requires exact diagnostic alignment outputs"
            )
        bound_units: list[dict[str, Any]] = []
        for unit, observation in zip(planned_units, observations, strict=True):
            unit_id = str(unit["unit_id"])
            if (
                observation.speaker_id != unit["speaker_id"]
                or observation.intended_spoken_text != unit["spoken_text"]
            ):
                raise ValueError(
                    f"Compatibility speaker or intended text changed for {unit_id}"
                )
            reference = (
                paths.job_root
                / "speakers"
                / str(unit["speaker_id"])
                / "reference_reel.wav"
            )
            azure = paths.azure_turn(unit_id)
            converted = paths.openvoice_turn(unit_id)
            aligned = paths.aligned_turn(unit_id)
            required_files = {
                "reference_audio_sha256": reference,
                "azure_audio_sha256": azure,
                "openvoice_audio_sha256": converted,
            }
            if require_alignment:
                required_files["aligned_audio_sha256"] = aligned
            for field, path in required_files.items():
                supplied = getattr(observation, field)
                if not path.is_file():
                    if require_alignment or supplied is not None:
                        raise ValueError(
                            f"Compatibility artifact binding failed for {unit_id}/{field}"
                        )
                    continue
                if supplied != checksum(path):
                    raise ValueError(
                        f"Compatibility artifact binding failed for {unit_id}/{field}"
                    )
            if observation.aligned_audio_sha256 is not None:
                if not aligned.is_file() or observation.aligned_audio_sha256 != checksum(
                    aligned
                ):
                    raise ValueError(
                        f"Compatibility aligned-audio hash changed for {unit_id}"
                    )
            azure_quality = inspect_wav(azure) if azure.is_file() else {}
            openvoice_quality = inspect_wav(converted) if converted.is_file() else {}
            if azure_quality:
                measured_azure_ms = round(
                    float(azure_quality.get("duration") or 0) * 1000
                )
                if observation.azure_duration_ms != measured_azure_ms:
                    raise ValueError(
                        f"Compatibility Azure duration evidence changed for {unit_id}"
                    )
            elif observation.azure_duration_ms is not None:
                raise ValueError(
                    f"Compatibility Azure duration lacks an artifact for {unit_id}"
                )
            if openvoice_quality:
                measured_openvoice_ms = round(
                    float(openvoice_quality.get("duration") or 0) * 1000
                )
                if observation.openvoice_duration_ms != measured_openvoice_ms:
                    raise ValueError(
                        f"Compatibility OpenVoice duration evidence changed for {unit_id}"
                    )
            elif observation.openvoice_duration_ms is not None:
                raise ValueError(
                    f"Compatibility OpenVoice duration lacks an artifact for {unit_id}"
                )
            if observation.audio_quality_passed is True and not openvoice_quality.get(
                "valid"
            ):
                raise ValueError(
                    f"Compatibility audio-quality evidence is invalid for {unit_id}"
                )
            if openvoice_quality and observation.clipping_ratio is not None and not math.isclose(
                observation.clipping_ratio,
                float(openvoice_quality.get("clipped_ratio") or 0),
                abs_tol=1e-6,
            ):
                raise ValueError(
                    f"Compatibility clipping evidence changed for {unit_id}"
                )
            measured_silence = 1.0 - float(openvoice_quality.get("non_silent_ratio") or 0)
            if openvoice_quality and observation.silence_ratio is not None and not math.isclose(
                observation.silence_ratio,
                measured_silence,
                abs_tol=1e-6,
            ):
                raise ValueError(
                    f"Compatibility silence evidence changed for {unit_id}"
                )
            if not openvoice_quality and (
                observation.clipping_ratio is not None
                or observation.silence_ratio is not None
                or observation.audio_quality_passed is not None
            ):
                raise ValueError(
                    f"Compatibility audio-quality metrics lack an artifact for {unit_id}"
                )
            alignment = alignment_by_id.get(unit_id)
            if require_alignment:
                if (
                    not isinstance(alignment, Mapping)
                    or observation.final_timing_error_ms
                    != alignment.get("final_timing_error_ms")
                    or observation.aligned_audio_sha256 != alignment.get("sha256")
                    or Path(str(alignment.get("output_path") or "")) != aligned
                ):
                    raise ValueError(
                        f"Compatibility timing evidence changed for {unit_id}"
                    )
            elif alignment is not None and observation.final_timing_error_ms is not None:
                if observation.final_timing_error_ms != alignment.get(
                    "final_timing_error_ms"
                ):
                    raise ValueError(
                        f"Compatibility timing evidence changed for {unit_id}"
                    )
            expected_counts = {
                "number_count": len(
                    re.findall(
                        r"\b\d+(?:[.,]\d+)?%?\b", observation.intended_spoken_text
                    )
                ),
                "date_count": len(
                    re.findall(
                        r"\b\d{1,2}(?:[/-]\d{1,2}[/-]\d{2,4}|\s+\w+\s+\d{4})\b",
                        observation.intended_spoken_text,
                    )
                ),
                "negation_count": len(
                    re.findall(
                        r"\b(?:akukho|hhayi|ngeke|angeke|not|no|never)\b",
                        observation.intended_spoken_text,
                        re.I,
                    )
                ),
            }
            if any(
                getattr(observation, field) != count
                for field, count in expected_counts.items()
            ):
                raise ValueError(
                    f"Compatibility protected-value counts changed for {unit_id}"
                )
            bound_units.append(
                {
                    "unit_id": unit_id,
                    "speaker_id": observation.speaker_id,
                    "reference_audio_sha256": observation.reference_audio_sha256,
                    "azure_audio_sha256": observation.azure_audio_sha256,
                    "openvoice_audio_sha256": observation.openvoice_audio_sha256,
                    "aligned_audio_sha256": observation.aligned_audio_sha256,
                }
            )
        return {
            "schema_version": "compatibility-evidence-binding-v1",
            "dubbing_plan_sha256": checksum(paths.plan),
            "complete_unit_set": True,
            "alignment_required": require_alignment,
            "units": bound_units,
        }

    @staticmethod
    def _selected_voice(plan: Mapping[str, Any], speaker_id: str, default: str) -> str:
        selected = plan.get("speakers", {}).get(speaker_id, {}).get("selected_azure_voice", {})
        return str(selected.get("selected_voice") or default)

    @staticmethod
    def _openvoice_unit_dict(unit: OpenVoiceUnit) -> dict[str, Any]:
        return {
            "unit_id": unit.unit_id,
            "speaker_id": unit.speaker_id,
            "azure_voice": unit.azure_voice,
            "target_language": unit.target_language,
            "source_audio_path": str(unit.source_audio_path),
            "source_audio_sha256": unit.source_audio_sha256,
            "source_embedding_path": str(unit.source_embedding_path),
            "source_embedding_sha256": unit.source_embedding_sha256,
            "target_embedding_path": str(unit.target_embedding_path),
            "target_embedding_sha256": unit.target_embedding_sha256,
            "output_path": str(unit.output_path),
            "conversion_settings": unit.conversion_settings,
            "sample_rate": unit.sample_rate,
        }

    def _validate_openvoice_conversion_plan(
        self, job: JobManifest, value: Mapping[str, Any]
    ) -> OpenVoicePlan:
        plan = OpenVoicePlan.from_dict(value)
        revision = self.settings.openvoice_revision
        if (
            plan.job_id != job.job_id
            or plan.lease_identity != "unbound"
            or plan.openvoice_revision != revision
            or not re.fullmatch(r"[0-9a-f]{40}", revision)
        ):
            raise ValueError("OpenVoice conversion plan identity is invalid")
        paths = self.paths(job.job_id)
        dubbing_plan = read_json(paths.plan)
        planned_units = list(dubbing_plan.get("units", []))
        if [unit.unit_id for unit in plan.units] != [
            str(unit["unit_id"]) for unit in planned_units
        ]:
            raise ValueError("OpenVoice conversion plan unit order changed")
        azure_manifest = read_json(paths.azure_manifest)
        azure_by_id = {
            str(item["turn_id"]): item for item in azure_manifest.get("units", [])
        }
        if set(azure_by_id) != {unit.unit_id for unit in plan.units}:
            raise ValueError("OpenVoice conversion plan Azure unit set changed")
        planned_by_id = {
            str(unit["unit_id"]): unit for unit in planned_units
        }
        for unit in plan.units:
            planned = planned_by_id[unit.unit_id]
            azure = azure_by_id[unit.unit_id]
            selected_voice = self._selected_voice(
                dubbing_plan,
                str(planned["speaker_id"]),
                str(azure.get("voice") or ""),
            )
            expected_paths = {
                "source_audio_path": Path(
                    f"dubbing/azure_tts/turns/{unit.unit_id}.wav"
                ),
                "source_embedding_path": Path(
                    "dubbing/openvoice/assets/azure_sources"
                )
                / unit.azure_voice
                / "source_embedding.bin",
                "target_embedding_path": Path(
                    "dubbing/openvoice/assets/target_speakers"
                )
                / unit.speaker_id
                / "target_embedding.bin",
                "output_path": Path(
                    f"dubbing/openvoice/turns/{unit.unit_id}.wav"
                ),
            }
            if (
                unit.speaker_id != planned["speaker_id"]
                or unit.azure_voice != selected_voice
                or azure.get("voice") != unit.azure_voice
                or azure.get("sha256") != unit.source_audio_sha256
                or unit.target_language != "zu-ZA"
                or unit.sample_rate != self.settings.azure_tts_sample_rate
                or any(
                    getattr(unit, field) != self._validated_relative_path(expected)
                    for field, expected in expected_paths.items()
                )
            ):
                raise ValueError(
                    f"OpenVoice conversion plan inputs changed for {unit.unit_id}"
                )
            if set(unit.conversion_settings) - {"tau"} or not 0 <= float(
                unit.conversion_settings.get("tau", 0.3)
            ) <= 1:
                raise ValueError(
                    f"OpenVoice conversion settings are invalid for {unit.unit_id}"
                )
            self._canonical_hash(unit.conversion_settings)
            for path, expected_hash in (
                (unit.source_audio_path, unit.source_audio_sha256),
                (unit.source_embedding_path, unit.source_embedding_sha256),
                (unit.target_embedding_path, unit.target_embedding_sha256),
            ):
                local = self._resolve_job_path(paths, path)
                if not local.is_file() or checksum(local) != expected_hash:
                    raise ValueError(
                        f"OpenVoice conversion plan local input changed: {path}"
                    )
        references = value.get("speaker_references")
        speakers = {unit.speaker_id for unit in plan.units}
        if not isinstance(references, Mapping) or set(references) != speakers:
            raise ValueError("OpenVoice plan speaker-reference set changed")
        for speaker_id in speakers:
            reference = references[speaker_id]
            expected_path = Path(f"speakers/{speaker_id}/reference_reel.wav")
            if (
                not isinstance(reference, Mapping)
                or Path(str(reference.get("path") or "")) != expected_path
                or not re.fullmatch(
                    r"[0-9a-f]{64}", str(reference.get("sha256") or "")
                )
            ):
                raise ValueError(
                    f"OpenVoice speaker-reference identity changed for {speaker_id}"
                )
            local = self._resolve_job_path(paths, expected_path)
            if not local.is_file() or checksum(local) != reference["sha256"]:
                raise ValueError(
                    f"OpenVoice speaker-reference artifact changed for {speaker_id}"
                )
        return plan

    @classmethod
    def _openvoice_unit_identity_hash(cls, unit: OpenVoiceUnit) -> str:
        return cls._canonical_hash(
            {
                "unit_id": unit.unit_id,
                "speaker_id": unit.speaker_id,
                "azure_voice": unit.azure_voice,
                "target_language": unit.target_language,
                "source_audio_sha256": unit.source_audio_sha256,
                "source_embedding_sha256": unit.source_embedding_sha256,
                "target_embedding_sha256": unit.target_embedding_sha256,
                "conversion_settings": unit.conversion_settings,
                "sample_rate": unit.sample_rate,
            }
        )

    @staticmethod
    def _atomic_copy(source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.partial")
        shutil.copyfile(source, temporary)
        temporary.replace(destination)

    @staticmethod
    def _relative_job_path(paths: DubbingArtifacts, value: Path) -> Path:
        try:
            relative = value.resolve().relative_to(paths.job_root.resolve())
        except ValueError as exc:
            raise ValueError("OpenVoice asset path is outside the job directory") from exc
        return ProductionDubbingPipeline._validated_relative_path(relative)

    @staticmethod
    def _validated_relative_path(value: Path) -> Path:
        text = value.as_posix()
        if (
            not text
            or value.is_absolute()
            or "\\" in text
            or any(
                part in {"", ".", ".."}
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", part)
                for part in value.parts
            )
        ):
            raise ValueError("OpenVoice paths must be safe job-relative POSIX paths")
        return value

    @classmethod
    def _resolve_job_path(
        cls, paths: DubbingArtifacts, relative: Path | str
    ) -> Path:
        safe = cls._validated_relative_path(Path(str(relative)))
        destination = paths.job_root / safe
        try:
            destination.resolve().relative_to(paths.job_root.resolve())
        except ValueError as exc:
            raise ValueError("OpenVoice path escapes the job directory") from exc
        return destination

    def _stage_gcs_download(
        self, job_id: str, relative: str, destination: Path
    ) -> Path:
        if not self.gcs:
            raise RuntimeError("GCS is not configured")
        temporary = destination.with_name(
            f".{destination.name}.{uuid.uuid4().hex}.reconcile"
        )
        try:
            self.gcs.download(job_id, relative, temporary)
            if (
                not temporary.is_file()
                or temporary.stat().st_size == 0
                or temporary.stat().st_size
                > self.settings.max_artifact_download_bytes
            ):
                raise ValueError("Downloaded OpenVoice artifact violates the size limit")
            return temporary
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    def _stage_planned_output(
        self,
        job: JobManifest,
        paths: DubbingArtifacts,
        relative: Path,
        expected_sha256: str,
        staged: list[tuple[Path, Path]],
    ) -> Path:
        if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            raise ValueError(f"OpenVoice output hash is invalid for {relative}")
        destination = self._resolve_job_path(paths, relative)
        temporary = self._stage_gcs_download(
            job.job_id, relative.as_posix(), destination
        )
        if checksum(temporary) != expected_sha256:
            temporary.unlink(missing_ok=True)
            raise ValueError(f"OpenVoice output hash mismatch for {relative}")
        staged.append((temporary, destination))
        return temporary

    @staticmethod
    def _completed_asset_result(
        record: Mapping[str, Any], label: str, maximum_attempts: int
    ) -> dict[str, Any]:
        attempts = int(record.get("attempts") or 0)
        result = record.get("result")
        cache_reused = (
            isinstance(result, Mapping) and result.get("cache_reused") is True
        )
        if (
            record.get("status") != "completed"
            or not isinstance(result, Mapping)
            or (attempts < 1 and not (attempts == 0 and cache_reused))
            or attempts > maximum_attempts
        ):
            raise ValueError(f"OpenVoice asset is not validly completed: {label}")
        return dict(result)

    @classmethod
    def _verify_embedding_manifest(
        cls,
        path: Path,
        *,
        schema: str,
        expected_artifact_hash: str,
        expected_values: Mapping[str, Any],
    ) -> None:
        value = read_json(path)
        cls._verify_canonical_artifact(value, expected_artifact_hash)
        if value.get("schema_version") != schema or any(
            value.get(key) != expected for key, expected in expected_values.items()
        ):
            raise ValueError("OpenVoice embedding manifest identity changed")

    def _audit_intelligibility(
        self,
        job: JobManifest,
        stage: str,
        units: list[dict[str, Any]],
        audio_root: Path,
    ) -> dict[str, Any]:
        """Run intelligibility audit for a stage."""
        if not self._stt_backend:
            return {"state": "skipped", "reason": "STT backend not configured"}
        
        paths = self.paths(job.job_id)
        auditor = IntelligibilityAuditor(self.settings, self._stt_backend)
        
        # Load entity bindings for protected entity checking
        protected_entities = {}
        if paths.entity_bindings.is_file():
            entity_bindings = read_json(paths.entity_bindings)
            registry = EntityRegistry()
            for unit_id, unit_data in entity_bindings.get("per_unit_bindings", {}).items():
                unit_entities = {}
                for match in unit_data.get("matches", []):
                    entity_id = match.get("entity_id")
                    if entity_id:
                        entity = registry.get_entity(entity_id)
                        unit_entities[entity_id] = {
                            "canonical_text": entity.canonical_text,
                            "aliases": list(entity.aliases),
                            "stt_phrases": list(entity.stt_phrases),
                        }
                protected_entities[unit_id] = unit_entities
        
        # Run audit
        report = auditor.audit_stage(
            job_id=job.job_id,
            stage=stage,
            units=units,
            audio_root=audio_root,
            locale=job.target_language,
            with_phrase_hints=False,
            protected_entities=protected_entities,
        )
        
        # Write reports
        output_dir = paths.qc_intelligibility_root / stage
        json_path, md_path = auditor.write_report(report, output_dir)
        
        # Store in job media
        job.media[f"intelligibility_{stage}_sha256"] = checksum(json_path)
        job.media[f"intelligibility_{stage}_state"] = report.state
        self.jobs.save(job)
        
        # Check if gate passed
        if report.state == "failed":
            raise ValueError(
                f"Intelligibility audit failed for stage {stage}: {report.pass_fail_reasons}"
            )
        
        return {"state": report.state, "report_path": str(json_path)}

    def rebuild_with_entities(
        self,
        job: JobManifest,
        *,
        force: bool = False,
    ) -> dict[str, Any]:
        """Safely rebuild downstream artifacts after entity registry changes.
        
        This command invalidates translation and downstream artifacts while preserving
        source artifacts (transcript, dubbing units, source derivatives). It forces
        entity re-binding and re-translation with the updated registry.
        """
        paths = self.paths(job.job_id)
        
        # Validate that source artifacts exist
        if not paths.dubbing_units.is_file():
            raise ValueError("Source dubbing units are missing - cannot rebuild")
        if not (self.jobs.job_dir(job.job_id) / "analysis/transcript_en.json").is_file():
            raise ValueError("Source transcript is missing - cannot rebuild")
        
        # Check current state
        valid_states = {"analysis_ready", "translation_ready", "azure_tts_queued", "azure_tts_ready"}
        if job.state not in valid_states:
            raise ValueError(
                f"Rebuild requires job in {valid_states}, not {job.state}. "
                "Use migrate() for legacy jobs."
            )
        
        # Record what will be invalidated
        invalidated = {
            "entity_bindings": paths.entity_bindings.is_file(),
            "translation": paths.three_text_translation.is_file(),
            "dubbing_plan": paths.plan.is_file(),
            "azure_tts": paths.azure_manifest.is_file(),
            "openvoice": paths.openvoice_manifest.is_file(),
            "alignment": paths.alignment_manifest.is_file(),
            "background": paths.background_manifest.is_file(),
            "final_mix": paths.final_mix.is_file(),
        }
        
        # Safely remove downstream artifacts
        if paths.entity_bindings.is_file():
            paths.entity_bindings.unlink()
        
        if paths.three_text_translation.is_file():
            paths.three_text_translation.unlink()
        
        if paths.plan.is_file():
            paths.plan.unlink()
        
        # Remove Azure TTS outputs
        if paths.azure_tts_root.exists():
            shutil.rmtree(paths.azure_tts_root, ignore_errors=True)
        
        # Remove OpenVoice outputs
        if paths.openvoice_root.exists():
            shutil.rmtree(paths.openvoice_root, ignore_errors=True)
        
        # Remove alignment outputs
        if paths.aligned_root.exists():
            shutil.rmtree(paths.aligned_root, ignore_errors=True)
        
        # Remove background
        if paths.background.is_file():
            paths.background.unlink()
        if paths.background_manifest.is_file():
            paths.background_manifest.unlink()
        
        # Remove final mix
        if paths.dialogue.is_file():
            paths.dialogue.unlink()
        if paths.final_mix.is_file():
            paths.final_mix.unlink()
        
        # Remove intelligibility reports
        if paths.qc_intelligibility_root.exists():
            shutil.rmtree(paths.qc_intelligibility_root, ignore_errors=True)
        
        # Update job manifest
        job.media.pop("entity_bindings_sha256", None)
        job.media.pop("entity_registry_sha256", None)
        job.media.pop("translation_sha256", None)
        job.media.pop("dubbing_plan_sha256", None)
        job.media.pop("azure_tts_manifest_sha256", None)
        job.media.pop("openvoice_manifest_sha256", None)
        job.media.pop("alignment_manifest_sha256", None)
        
        # Reset state to analysis_ready
        job.transition("analysis_ready")
        
        # Clear completed stages that depend on entities
        job.completed_stages = [
            stage for stage in job.completed_stages
            if stage in {"source_derivatives", "dubbing_units"}
        ]
        
        # Record rebuild in attempt history
        job.attempt_history.append({
            "stage": "entity_registry_rebuild",
            "status": "completed",
            "invalidated_artifacts": invalidated,
            "preserved_artifacts": {
                "dubbing_units": True,
                "source_transcript": True,
                "source_derivatives": True,
            },
            "at": utcnow(),
        })
        
        self.jobs.save(job)
        
        return {
            "schema_version": "entity-rebuild-v1",
            "job_id": job.job_id,
            "state": job.state,
            "invalidated": invalidated,
            "next_step": "Run translate() to re-bind entities and re-translate",
        }

    @classmethod
    def _verify_canonical_artifact(
        cls, value: Mapping[str, Any], expected_artifact_hash: str
    ) -> None:
        embedded = str(value.get("artifact_sha256") or "")
        if (
            not re.fullmatch(r"[0-9a-f]{64}", expected_artifact_hash)
            or embedded != expected_artifact_hash
            or cls._canonical_hash(
                {key: item for key, item in value.items() if key != "artifact_sha256"}
            )
            != embedded
        ):
            raise ValueError("OpenVoice canonical artifact hash mismatch")

    @staticmethod
    def _verify_conversion_durations(result: Mapping[str, Any]) -> None:
        source = float(result.get("source_duration_ms") or 0)
        converted = float(result.get("converted_duration_ms") or 0)
        ratio = float(result.get("duration_ratio") or 0)
        if (
            not all(math.isfinite(value) for value in (source, converted, ratio))
            or source <= 0
            or converted <= 0
            or ratio <= 0
            or not math.isclose(ratio, converted / source, rel_tol=1e-4, abs_tol=1e-4)
        ):
            raise ValueError("OpenVoice conversion duration metadata is invalid")

    @staticmethod
    def _validate_candidate_metrics(result: Mapping[str, Any]) -> None:
        for field in ("speaker_similarity", "word_error_approximation"):
            value = float(result.get(field, math.nan))
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"OpenVoice candidate metric is invalid: {field}")
        artifacts = result.get("audio_artifact_count")
        if not isinstance(artifacts, int) or isinstance(artifacts, bool) or artifacts < 0:
            raise ValueError("OpenVoice candidate audio_artifact_count is invalid")

    def _apply_candidate_evaluations(
        self,
        job: JobManifest,
        speaker_id: str,
        candidate_manifest: dict[str, Any],
        evaluations: Mapping[str, Any],
    ) -> None:
        if evaluations.get("schema_version") != "openvoice-candidate-evaluations-v1":
            raise ValueError("Unsupported OpenVoice candidate-evaluation schema")
        speakers = evaluations.get("speakers")
        if not isinstance(speakers, Mapping) or speaker_id not in speakers:
            raise ValueError(f"Candidate evaluations are missing for {speaker_id}")
        review = speakers[speaker_id]
        if not isinstance(review, Mapping) or review.get("status") != "approved":
            raise ValueError(f"Candidate evaluation is not approved for {speaker_id}")
        reviewer = str(review.get("reviewer") or "").strip()
        reviewed_at = str(review.get("reviewed_at") or "")
        try:
            timestamp = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(
                f"Candidate evaluation timestamp is invalid for {speaker_id}"
            ) from exc
        if not reviewer or timestamp.tzinfo is None:
            raise ValueError(
                f"Candidate evaluation requires a reviewer and timezone for {speaker_id}"
            )
        raw_candidates = review.get("candidates")
        if not isinstance(raw_candidates, Mapping):
            raise ValueError(f"Candidate evaluation values are missing for {speaker_id}")
        manifest_candidates = {
            str(item.get("voice") or item.get("azure_voice")): item
            for item in candidate_manifest.get("candidates", [])
        }
        if set(raw_candidates) != set(manifest_candidates):
            raise ValueError(
                f"Candidate evaluation voices do not match the manifest for {speaker_id}"
            )
        for voice, item in manifest_candidates.items():
            metrics = raw_candidates[voice]
            if not isinstance(metrics, Mapping):
                raise ValueError(f"Candidate metrics are invalid for {speaker_id}/{voice}")
            merged = {
                **item,
                "speaker_similarity": metrics.get("speaker_similarity"),
                "word_error_approximation": metrics.get(
                    "word_error_approximation"
                ),
                "audio_artifact_count": metrics.get("audio_artifact_count"),
            }
            self._validate_candidate_metrics(merged)
            item.update(
                speaker_similarity=merged["speaker_similarity"],
                word_error_approximation=merged["word_error_approximation"],
                audio_artifact_count=merged["audio_artifact_count"],
                evaluation_status="completed",
                external_qc={
                    "reviewer": reviewer,
                    "reviewed_at": timestamp.isoformat(),
                    "source_sha256": self._canonical_hash(dict(metrics)),
                },
            )
        override = review.get("human_override")
        if override is not None:
            candidate_manifest["human_override"] = str(override)
        candidate_manifest["external_qc_status"] = "approved"
        candidate_manifest["external_qc_reviewer"] = reviewer
        candidate_manifest["external_qc_reviewed_at"] = timestamp.isoformat()
        candidate_manifest["job_id"] = job.job_id
        candidate_manifest["artifact_sha256"] = self._canonical_hash(
            {
                key: value
                for key, value in candidate_manifest.items()
                if key != "artifact_sha256"
            }
        )

    def _upload_job_status(self, job: JobManifest) -> str:
        if not self.gcs:
            raise RuntimeError("GCS is not configured")
        try:
            _current, generation = self.gcs.download_json(job.job_id, "status.json")
        except FileNotFoundError:
            return self.gcs.upload_json(
                job.job_id,
                "status.json",
                job.to_dict(),
                if_generation_match=0,
            )
        return self.gcs.update_json(
            job.job_id, "status.json", job.to_dict(), generation
        )

    def _forced_stage_reset(self, job: JobManifest, stage: str, target: str) -> None:
        job.attempt_history.append(
            {"stage": stage, "status": "forced_rerun", "from_state": job.state, "to_state": target, "at": utcnow()}
        )
        job.state = target
        self.jobs.save(job)

    def _promote_file(self, job_id: str, relative: str, local: Path, expected: str) -> str:
        if not self.gcs:
            raise RuntimeError("GCS is not configured")
        temporary = f"{relative}.partial.server-{uuid.uuid4().hex[:12]}"
        self.gcs.upload(job_id, temporary, local, if_generation_match=0)
        return self.gcs.promote(job_id, temporary, relative, expected, checksum)

    def _update_plan_stage(
        self,
        job: JobManifest,
        stage: str,
        results: Mapping[str, Mapping[str, Any]],
    ) -> None:
        if stage not in {"azure_tts", "openvoice", "alignment"}:
            raise ValueError(f"Unsupported dubbing-plan stage: {stage}")
        path = self.paths(job.job_id).plan
        plan = read_json(path)
        for unit in plan["units"]:
            result = results.get(str(unit["unit_id"]))
            if result is None:
                raise ValueError(f"{stage} result is missing for {unit['unit_id']}")
            attempts: int | None = None
            if stage == "azure_tts":
                attempts = len(result.get("timing_search", {}).get("attempts", []))
            elif stage == "openvoice":
                attempts = int(result.get("attempt") or 1)
            unit[stage] = {
                "status": "ready",
                "attempts": attempts,
                "result": dict(result),
            }
        self._write_plan(path, plan)

    def _write_plan(self, path: Path, plan: dict[str, Any]) -> None:
        plan["artifact_sha256"] = self._canonical_hash(
            {key: value for key, value in plan.items() if key != "artifact_sha256"}
        )
        atomic_write_json(path, plan)

    @staticmethod
    def _canonical_hash(value: Any) -> str:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _safe_error(stage: str, exc: Exception) -> dict[str, Any]:
        if hasattr(exc, "to_dict"):
            value = exc.to_dict()
            return {"stage": stage, **value, "at": utcnow()}
        return {
            "stage": stage,
            "error_type": type(exc).__name__,
            "message": str(redact(str(exc)))[:500],
            "retryable": False,
            "at": utcnow(),
        }


__all__ = ["ProductionDubbingPipeline"]
