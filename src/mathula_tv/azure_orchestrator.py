"""Azure-only dubbing orchestration for CPU-only professional TTS dubbing."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .acoustic_analysis import analyze_all_speakers
from .atomic_io import read_json
from .azure_qc import validate_azure_tts_manifest, write_qc_report
from .azure_voice_catalogue import get_catalogue
from .speaker_profiles import build_speaker_profiles, write_speaker_profiles
from .voice_resolver import resolve_azure_voices, write_voice_assignments, validate_assignments_checksum
from .production_pipeline import ProductionDubbingPipeline
from .models import JobManifest


logger = logging.getLogger(__name__)


class AzureDubbingOrchestrator:
    """Orchestrates the complete Azure-only dubbing pipeline."""
    
    def __init__(
        self,
        pipeline: ProductionDubbingPipeline,
        *,
        fallback_allowed: bool = False,
        min_confidence: float = 0.5,
    ):
        self.pipeline = pipeline
        self.fallback_allowed = fallback_allowed
        self.min_confidence = min_confidence
    
    def run_full_pipeline(
        self,
        job: JobManifest,
        azure_tts_backend: Any,
        *,
        force: bool = False,
        skip_to: str | None = None,
    ) -> dict[str, Any]:
        """Run the complete Azure-only dubbing pipeline.
        
        Args:
            job: Job manifest
            azure_tts_backend: Azure TTS backend
            force: Force re-run of stages
            skip_to: Skip to a specific stage ("speaker_profiles", "voice_resolution", 
                      "azure_tts", "alignment", "background", "mix", "render")
        
        Returns:
            Summary of pipeline execution
        """
        paths = self.pipeline.paths(job.job_id)
        summary = {"job_id": job.job_id, "stages_completed": []}
        
        # Stage 1: Build speaker profiles
        if skip_to is None or skip_to == "speaker_profiles":
            logger.info(f"Building speaker profiles for job {job.job_id}")
            speaker_profiles = self._build_speaker_profiles(job)
            summary["stages_completed"].append("speaker_profiles")
            summary["speaker_profiles_path"] = str(paths.speaker_profiles)
        
        # Stage 2: Run acoustic analysis
        if skip_to is None or skip_to == "acoustic_analysis":
            logger.info(f"Running acoustic analysis for job {job.job_id}")
            acoustic_results = self._run_acoustic_analysis(job)
            summary["stages_completed"].append("acoustic_analysis")
            summary["acoustic_analysis_path"] = str(paths.acoustic_analysis)
        
        # Stage 3: Resolve Azure voices
        if skip_to is None or skip_to == "voice_resolution":
            logger.info(f"Resolving Azure voices for job {job.job_id}")
            voice_assignments = self._resolve_voices(job)
            summary["stages_completed"].append("voice_resolution")
            summary["voice_assignments_path"] = str(paths.azure_voice_assignments)
        
        # Stage 4: Prepare dubbing plan (azure_only mode)
        if skip_to is None or skip_to == "dubbing_plan":
            logger.info(f"Preparing Azure-only dubbing plan for job {job.job_id}")
            plan = self.pipeline.prepare_dubbing(job, azure_only=True, force=force)
            summary["stages_completed"].append("dubbing_plan")
        
        # Stage 5: Synthesize Azure TTS
        if skip_to is None or skip_to == "azure_tts":
            logger.info(f"Synthesizing Azure TTS for job {job.job_id}")
            voice_assignments = read_json(paths.azure_voice_assignments)
            azure_manifest = self.pipeline.synthesize_azure(
                job,
                azure_tts_backend,
                force=force,
                voice_assignments=voice_assignments,
            )
            summary["stages_completed"].append("azure_tts")
        
        # Stage 6: Run Azure TTS QC
        if skip_to is None or skip_to == "azure_qc":
            logger.info(f"Running Azure TTS QC for job {job.job_id}")
            qc_report = self._run_azure_qc(job)
            summary["stages_completed"].append("azure_qc")
            summary["azure_qc_passed"] = qc_report.overall_passed
            if not qc_report.overall_passed:
                raise ValueError(f"Azure TTS QC failed: {qc_report.blocking_failures}")
        
        # Stage 7: Align directly from Azure TTS
        if skip_to is None or skip_to == "alignment":
            logger.info(f"Aligning Azure TTS for job {job.job_id}")
            alignment_manifest = self.pipeline.align(job, azure_only=True, force=force)
            summary["stages_completed"].append("alignment")
        
        # Stage 8: Prepare background
        if skip_to is None or skip_to == "background":
            logger.info(f"Preparing background for job {job.job_id}")
            background_manifest = self.pipeline.prepare_background(job, azure_only=True)
            summary["stages_completed"].append("background")
        
        # Stage 9: Mix final audio
        if skip_to is None or skip_to == "mix":
            logger.info(f"Mixing final audio for job {job.job_id}")
            mix_manifest = self.pipeline.mix(job, azure_only=True)
            summary["stages_completed"].append("mix")
        
        # Stage 10: Generate subtitles and review video
        if skip_to is None or skip_to == "render":
            logger.info(f"Generating review video for job {job.job_id}")
            render_manifest = self.pipeline.render(job, azure_only=True)
            summary["stages_completed"].append("render")
        
        summary["status"] = "completed"
        return summary
    
    def _build_speaker_profiles(self, job: JobManifest) -> dict[str, Any]:
        """Build speaker profiles from transcript and diarization."""
        paths = self.pipeline.paths(job.job_id)
        
        transcript_path = paths.job_root / "analysis/transcript_en.json"
        diarization_path = paths.job_root / "analysis/azure_diarization.json"
        
        speaker_profiles = build_speaker_profiles(
            job_id=job.job_id,
            transcript_path=transcript_path,
            diarization_path=diarization_path,
            speaker_registry_path=None,  # TODO: Add speaker registry support
            entity_registry_path=None,  # TODO: Add entity registry support
            acoustic_analysis_results=None,  # Acoustic analysis runs separately
        )
        
        write_speaker_profiles(speaker_profiles, paths.speaker_profiles)
        return speaker_profiles.to_dict()
    
    def _run_acoustic_analysis(self, job: JobManifest) -> dict[str, Any]:
        """Run CPU-only acoustic voice-family analysis."""
        paths = self.pipeline.paths(job.job_id)
        
        audio_path = paths.analysis_audio
        diarization_path = paths.job_root / "analysis/azure_diarization.json"
        
        acoustic_results = analyze_all_speakers(
            audio_path=audio_path,
            diarization_path=diarization_path,
            output_path=paths.acoustic_analysis,
        )
        
        # Rebuild speaker profiles with acoustic analysis results
        speaker_profiles = read_json(paths.speaker_profiles)
        updated_profiles = build_speaker_profiles(
            job_id=job.job_id,
            transcript_path=paths.job_root / "analysis/transcript_en.json",
            diarization_path=diarization_path,
            acoustic_analysis_results=acoustic_results,
        )
        write_speaker_profiles(updated_profiles, paths.speaker_profiles)
        
        return acoustic_results
    
    def _resolve_voices(self, job: JobManifest) -> dict[str, Any]:
        """Resolve Azure voices from speaker profiles."""
        paths = self.pipeline.paths(job.job_id)
        
        speaker_profiles_data = read_json(paths.speaker_profiles)
        from .speaker_profiles import SpeakerProfilesArtifact
        speaker_profiles = SpeakerProfilesArtifact.from_dict(speaker_profiles_data)
        
        voice_assignments = resolve_azure_voices(
            job_id=job.job_id,
            speaker_profiles=speaker_profiles,
            target_locale=job.target_language,
            fallback_allowed=self.fallback_allowed,
            min_confidence=self.min_confidence,
        )
        
        write_voice_assignments(voice_assignments, paths.azure_voice_assignments)
        return voice_assignments.to_dict()
    
    def _run_azure_qc(self, job: JobManifest) -> dict[str, Any]:
        """Run Azure TTS quality control."""
        paths = self.pipeline.paths(job.job_id)
        
        qc_report = validate_azure_tts_manifest(
            manifest_path=paths.azure_manifest,
            dubbing_plan_path=paths.plan,
            voice_assignments_path=paths.azure_voice_assignments,
        )
        
        write_qc_report(qc_report, paths.azure_qc_report)
        return qc_report


def create_azure_orchestrator(
    pipeline: ProductionDubbingPipeline,
    *,
    fallback_allowed: bool = False,
    min_confidence: float = 0.5,
) -> AzureDubbingOrchestrator:
    """Create an Azure dubbing orchestrator.
    
    Args:
        pipeline: Production dubbing pipeline
        fallback_allowed: Allow fallback voice resolution
        min_confidence: Minimum confidence threshold
    
    Returns:
        AzureDubbingOrchestrator instance
    """
    return AzureDubbingOrchestrator(
        pipeline,
        fallback_allowed=fallback_allowed,
        min_confidence=min_confidence,
    )


__all__ = [
    "AzureDubbingOrchestrator",
    "create_azure_orchestrator",
]
