"""Direct Azure dubbing from an approved segment-level translation.

This module deliberately avoids the production orchestration/state-machine path.
The approved translation segments remain authoritative. Adjacent fragments from
one speaker are coalesced only to restore natural target-language speech timing;
the translated words are never redistributed or rewritten.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import wave
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .atomic_io import atomic_write_json, read_json
from .azure_tts import (
    AzureTTSBackend,
    AzureTTSHTTPBoundary,
    AzureTTSRequest,
    AzureTTSResult,
)
from .background import prepare_background
from .config import Settings
from .job_store import JobStore
from .media import checksum, prepare_source_derivatives, probe
from .mixing import mix_final_buses, render_dialogue_tracks
from .models import JobManifest, utcnow
from .output_naming import (
    load_seo_output_context,
    publish_title_named_outputs,
)
from .rendering import render_review_mp4, render_tiktok_compatible_mp4
from .tts_ssml import SSMLBounds
from .voice_prosody import voice_and_base_prosody
from .speaker_mapping import build_speaker_id_mapping, canonicalize_voice_family_evidence
from .direct_voice_analysis import (
    ensure_canonical_speaker_reels,
    ensure_canonical_voice_family_analysis,
)
from .known_speakers import identify_known_speaker
from .direct_azure_persona import (
    DIRECT_AZURE_PERSONA_VERSION,
    apply_direct_azure_personas,
    collect_source_features,
)


DIRECT_DUB_SCHEMA_VERSION = "mathula-direct-azure-dub-v6-exact-sample-loudness"


class DirectDubError(RuntimeError):
    """Raised when the direct renderer cannot safely produce a complete dub."""


class TimingOverflowError(DirectDubError):
    """Raised when synthesized speech cannot fit its current timeline window."""

    def __init__(
        self,
        message: str,
        *,
        measured_duration_ms: int,
        target_duration_ms: int,
        stretch_ratio: float,
    ) -> None:
        super().__init__(message)
        self.measured_duration_ms = measured_duration_ms
        self.target_duration_ms = target_duration_ms
        self.stretch_ratio = stretch_ratio


@dataclass(frozen=True)
class ApprovedSegment:
    segment_id: str
    speaker_id: str
    start_ms: int
    end_ms: int
    source_text: str
    translated_text: str
    tts_text: str

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


@dataclass(frozen=True)
class SpeechBlock:
    block_id: str
    speaker_id: str
    start_ms: int
    end_ms: int
    segment_ids: tuple[str, ...]
    source_text: str
    translated_text: str
    tts_text: str

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "segment_ids": list(self.segment_ids),
            "duration_ms": self.duration_ms,
        }


@dataclass(frozen=True)
class DirectDubArtifacts:
    job_root: Path

    @property
    def root(self) -> Path:
        return self.job_root / "direct_dub"

    @property
    def blocks_root(self) -> Path:
        return self.root / "blocks"

    def candidate(self, request: AzureTTSRequest) -> Path:
        payload = {
            "turn_id": request.turn_id,
            "speaker_id": request.speaker_id,
            "voice": request.voice,
            "text": request.text,
            "rate_percent": request.rate_percent,
            "pitch_percent": request.pitch_percent,
            "volume_percent": request.volume_percent,
        }
        digest = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()[:12]
        return (
            self.blocks_root
            / request.turn_id
            / f"azure_rate_{request.rate_percent:+d}_{digest}.wav"
        )

    def final_block(self, block_id: str) -> Path:
        return self.blocks_root / block_id / "final.wav"

    @property
    def dialogue_tracks(self) -> Path:
        return self.root / "dialogue_tracks"

    @property
    def dialogue(self) -> Path:
        return self.root / "dialogue.wav"

    @property
    def background(self) -> Path:
        return self.root / "background.wav"

    @property
    def final_mix(self) -> Path:
        return self.root / "final_mix.wav"

    @property
    def final_video(self) -> Path:
        return self.root / "final_dubbed.mp4"

    @property
    def tiktok_render_manifest(self) -> Path:
        return self.root / "tiktok_render_manifest.json"

    @property
    def render_manifest(self) -> Path:
        return self.root / "render_manifest.json"

    @property
    def report(self) -> Path:
        return self.root / "report.json"

    @property
    def voice_resolution(self) -> Path:
        return self.root / "voice_resolution.json"


@dataclass(frozen=True)
class DirectDubOptions:
    same_speaker_gap_ms: int = 1200
    max_azure_rate_percent: int = 25
    max_time_stretch_ratio: float = 1.35
    collision_max_time_stretch_ratio: float = 1.60
    max_cross_speaker_boundary_shift_ms: int = 500
    male_voice: str = "zu-ZA-ThembaNeural"
    female_voice: str = "zu-ZA-ThandoNeural"
    min_voice_family_confidence: float = 0.70
    include_background: bool = False
    dialogue_target_lufs: float = -16.0
    dialogue_true_peak_dbfs: float = -1.5
    dialogue_loudness_range_lu: float = 11.0

    def validate(self, backend: AzureTTSBackend) -> None:
        if self.same_speaker_gap_ms < 0:
            raise ValueError("same-speaker gap must be non-negative")
        lower = backend.ssml_bounds.rate_min_percent
        upper = backend.ssml_bounds.rate_max_percent
        if not lower <= self.max_azure_rate_percent <= upper:
            raise ValueError(
                "max Azure rate must be within the active SSML bounds "
                f"[{lower}, {upper}]"
            )
        if not 1.0 <= self.max_time_stretch_ratio <= 2.0:
            raise ValueError("max time-stretch ratio must be between 1.0 and 2.0")
        if not self.max_time_stretch_ratio <= self.collision_max_time_stretch_ratio <= 2.0:
            raise ValueError(
                "collision time-stretch ratio must be between the normal limit and 2.0"
            )
        if not 0 <= self.max_cross_speaker_boundary_shift_ms <= 1000:
            raise ValueError(
                "cross-speaker boundary shift must be between 0 and 1000ms"
            )
        if not 0.0 <= self.min_voice_family_confidence <= 1.0:
            raise ValueError("minimum voice-family confidence must be between 0 and 1")
        if not -24.0 <= self.dialogue_target_lufs <= -12.0:
            raise ValueError("dialogue target must be between -24 and -12 LUFS")
        if not -3.0 <= self.dialogue_true_peak_dbfs <= -0.1:
            raise ValueError("dialogue true-peak target must be between -3.0 and -0.1 dBFS")
        if not 1.0 <= self.dialogue_loudness_range_lu <= 20.0:
            raise ValueError("dialogue loudness range must be between 1 and 20 LU")


def create_direct_azure_backend(
    settings: Settings,
    *,
    max_rate_percent: int = 25,
) -> AzureTTSBackend:
    """Create the only external backend required by the direct renderer."""
    key = os.getenv("AZURE_SPEECH_KEY", "")
    if not key:
        raise ValueError("AZURE_SPEECH_KEY is missing")
    if not settings.azure_speech_region:
        raise ValueError("AZURE_SPEECH_REGION is missing")
    bounds = SSMLBounds(
        rate_min_percent=settings.azure_rate_min_percent,
        rate_max_percent=max_rate_percent,
    )
    return AzureTTSBackend(
        AzureTTSHTTPBoundary(settings.azure_speech_region, key),
        configured_voices=settings.azure_tts_voices,
        default_voice=settings.azure_tts_default_voice,
        sample_rate=settings.azure_tts_sample_rate,
        channels=settings.azure_tts_channels,
        sample_width=settings.azure_tts_sample_width,
        timeout_seconds=settings.azure_tts_timeout_seconds,
        max_retries=settings.azure_tts_max_retries,
        ssml_bounds=bounds,
    )


class DirectAzureDubRenderer:
    """Render an Azure dub directly from the approved translation artifact."""

    def __init__(
        self,
        settings: Settings,
        backend: AzureTTSBackend,
        *,
        runner: Callable[..., Any] = subprocess.run,
    ) -> None:
        self.settings = settings
        self.backend = backend
        self.jobs = JobStore(settings.work_dir)
        self.runner = runner

    def render(
        self,
        job: JobManifest,
        *,
        translation_path: Path | None = None,
        voice_map_path: Path | None = None,
        clean_background_stem: Path | None = None,
        options: DirectDubOptions | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        options = options or DirectDubOptions(
            max_azure_rate_percent=min(25, self.settings.azure_rate_max_percent)
        )
        options.validate(self.backend)

        artifacts = DirectDubArtifacts(self.jobs.job_dir(job.job_id))
        artifacts.root.mkdir(parents=True, exist_ok=True)
        source_video = Path(job.local_source_path)
        if not source_video.is_file():
            raise FileNotFoundError(source_video)

        seo_context = load_seo_output_context(
            self.jobs.job_dir(job.job_id),
            job.target_language,
        )

        translation_path = translation_path or (
            self.jobs.job_dir(job.job_id) / "translation" / "transcript_zu.json"
        )
        translation = read_json(translation_path)
        segments = load_approved_segments(translation, source_duration_ms=round(probe(source_video)["duration"] * 1000))
        blocks = coalesce_same_speaker_segments(
            segments,
            max_gap_ms=options.same_speaker_gap_ms,
        )
        if not blocks:
            raise DirectDubError("Approved translation contains no speakable segments")

        assignments = self._voice_assignments(
            job,
            voice_map_path,
            options,
            required_speaker_ids=sorted({block.speaker_id for block in blocks}),
            output_path=artifacts.voice_resolution,
            refresh_voice_analysis=force,
        )
        request_fingerprint = self._request_fingerprint(
            job,
            translation_path,
            blocks,
            assignments,
            options,
            clean_background_stem,
            seo_context.seo_path,
        )
        if not force:
            reused = self._reuse_completed(artifacts, request_fingerprint)
            if reused is not None:
                return reused

        render_blocks = list(blocks)
        block_results: list[dict[str, Any]] = []
        clips: list[dict[str, Any]] = []
        for block_index in range(len(render_blocks)):
            block = render_blocks[block_index]
            assignment = assignments.get(block.speaker_id)
            if assignment is None:
                raise DirectDubError(
                    f"No resolved voice assignment for {block.speaker_id}"
                )
            voice, base_rate, pitch, volume = voice_and_base_prosody(
                assignment,
                "",
            )
            if not voice:
                raise DirectDubError(
                    f"Resolved assignment for {block.speaker_id} has no Azure voice"
                )
            try:
                fitted = self._synthesize_block(
                    block,
                    artifacts,
                    voice=voice,
                    base_rate_percent=base_rate,
                    pitch_percent=pitch,
                    volume_percent=volume,
                    max_azure_rate_percent=options.max_azure_rate_percent,
                    max_time_stretch_ratio=options.max_time_stretch_ratio,
                    force=force,
                )
            except TimingOverflowError as exc:
                expanded, timing_adjustment = borrow_safe_silence_window(
                    render_blocks,
                    block_index,
                )
                fitted: dict[str, Any] | None = None
                latest_overflow = exc
                if expanded.duration_ms > block.duration_ms:
                    try:
                        fitted = self._synthesize_block(
                            expanded,
                            artifacts,
                            voice=voice,
                            base_rate_percent=base_rate,
                            pitch_percent=pitch,
                            volume_percent=volume,
                            max_azure_rate_percent=options.max_azure_rate_percent,
                            max_time_stretch_ratio=options.collision_max_time_stretch_ratio,
                            # Reuse the already synthesized Azure candidates. Only the
                            # pitch-preserving fit is redone for the expanded window.
                            force=False,
                        )
                    except TimingOverflowError as expanded_exc:
                        latest_overflow = expanded_exc
                    else:
                        block = expanded
                        render_blocks[block_index] = expanded
                        fitted["timing_adjustment"] = timing_adjustment

                if fitted is None:
                    required_window_ms = math.ceil(
                        latest_overflow.measured_duration_ms
                        / options.collision_max_time_stretch_ratio
                    )
                    expanded, shifted_next, timing_adjustment = (
                        rebalance_cross_speaker_handoff(
                            render_blocks,
                            block_index,
                            required_window_ms=required_window_ms,
                            max_shift_ms=options.max_cross_speaker_boundary_shift_ms,
                        )
                    )
                    if shifted_next is None:
                        raise DirectDubError(
                            genuine_speaker_collision_message(
                                render_blocks,
                                block_index,
                                latest_overflow,
                                required_window_ms=required_window_ms,
                            )
                        ) from latest_overflow
                    render_blocks[block_index] = expanded
                    render_blocks[block_index + 1] = shifted_next
                    block = expanded
                    fitted = self._synthesize_block(
                        expanded,
                        artifacts,
                        voice=voice,
                        base_rate_percent=base_rate,
                        pitch_percent=pitch,
                        volume_percent=volume,
                        max_azure_rate_percent=options.max_azure_rate_percent,
                        max_time_stretch_ratio=options.collision_max_time_stretch_ratio,
                        force=False,
                    )
                    fitted["timing_adjustment"] = timing_adjustment
            except DirectDubError:
                # Non-timing failures must remain visible and must not trigger
                # timeline mutation.
                raise
            block_results.append(fitted)
            clips.append(
                {
                    "unit_id": block.block_id,
                    "speaker_id": block.speaker_id,
                    "path": fitted["output_path"],
                    "start_ms": fitted["start_ms"],
                    "duration_ms": fitted["duration_ms"],
                }
            )

        source_duration_ms = round(probe(source_video)["duration"] * 1000)
        dialogue_manifest = render_dialogue_tracks(
            clips,
            artifacts.dialogue_tracks,
            artifacts.dialogue,
            total_duration_ms=source_duration_ms,
            sample_rate=self.backend.sample_rate,
            runner=self.runner,
        )
        loudness_manifest = normalize_dialogue_loudness(
            artifacts.dialogue,
            target_lufs=options.dialogue_target_lufs,
            true_peak_dbfs=options.dialogue_true_peak_dbfs,
            loudness_range_lu=options.dialogue_loudness_range_lu,
            sample_rate=self.backend.sample_rate,
            runner=self.runner,
        )
        dialogue_manifest = {
            **dialogue_manifest,
            "output_path": str(artifacts.dialogue),
            "output_sha256": checksum(artifacts.dialogue),
            "loudness_normalization": loudness_manifest,
        }
        if options.include_background:
            mix_source = self._ensure_mix_source(job, source_video)
            background_manifest = prepare_background(
                mix_source,
                artifacts.background,
                source_dialogue=[
                    {
                        "start": block.start_ms / 1000.0,
                        "end": block.end_ms / 1000.0,
                    }
                    for block in blocks
                ],
                clean_stem=clean_background_stem,
                azure_only=True,
            )
            mix_manifest = mix_final_buses(
                artifacts.background,
                artifacts.dialogue,
                artifacts.final_mix,
                runner=self.runner,
            )
            audio_mode = "dialogue_with_background"
        else:
            # The original soundtrack usually still contains the English speakers.
            # Dialogue-only is therefore the safe production default. Background
            # can be enabled explicitly only when the caller wants it.
            artifacts.background.unlink(missing_ok=True)
            shutil.copy2(artifacts.dialogue, artifacts.final_mix)
            background_manifest = {
                "schema_version": "mathula-background-disabled-v1",
                "mode": "disabled",
                "reason": "dialogue_only_default",
                "output_path": None,
            }
            mix_manifest = {
                "schema_version": "mathula-dialogue-only-mix-v1",
                "mode": "dialogue_only",
                "dialogue_path": str(artifacts.dialogue),
                "output_path": str(artifacts.final_mix),
                "output_sha256": checksum(artifacts.final_mix),
            }
            audio_mode = "dialogue_only"

        render_manifest = render_review_mp4(
            source_video,
            artifacts.final_mix,
            artifacts.final_video,
            artifacts.render_manifest,
            runner=self.runner,
        )
        named_outputs = publish_title_named_outputs(
            self.jobs.job_dir(job.job_id),
            artifacts.final_video,
            context=seo_context,
        )
        named_seo_files = {
            key: str(path) for key, path in sorted(named_outputs.seo_files.items())
        }
        tiktok_upload_video = (
            self.jobs.job_dir(job.job_id)
            / "output"
            / f"tiktok_upload_{job.job_id}.mp4"
        )
        tiktok_render_manifest = render_tiktok_compatible_mp4(
            named_outputs.final_video,
            tiktok_upload_video,
            artifacts.tiktok_render_manifest,
            runner=self.runner,
        )

        report = {
            "schema_version": DIRECT_DUB_SCHEMA_VERSION,
            "job_id": job.job_id,
            "created_at": utcnow(),
            "request_fingerprint": request_fingerprint,
            "translation": {
                "path": str(translation_path),
                "sha256": checksum(translation_path),
                "schema_version": translation.get("schema_version"),
                "approved_segment_count": len(segments),
                "translated_words_preserved": True,
                "translation_repair_used": False,
            },
            "voice_resolution": {
                "path": str(artifacts.voice_resolution),
                "sha256": checksum(artifacts.voice_resolution),
                "unknown_defaults_to_masculine": False,
                "persona_version": DIRECT_AZURE_PERSONA_VERSION,
                "personas": {
                    speaker_id: dict(assignment.get("base_prosody") or {})
                    for speaker_id, assignment in assignments.items()
                },
            },
            "strategy": {
                "pipeline": "direct_approved_segments_to_azure",
                "orchestrator_used": False,
                "generated_micro_units": False,
                "same_speaker_coalescing_only": True,
                "speech_block_count": len(blocks),
                "same_speaker_gap_ms": options.same_speaker_gap_ms,
                "max_azure_rate_percent": options.max_azure_rate_percent,
                "max_time_stretch_ratio": options.max_time_stretch_ratio,
                "audio_mode": audio_mode,
                "background_included": options.include_background,
                "speaker_volume_percent": 0,
                "dialogue_target_lufs": options.dialogue_target_lufs,
                "dialogue_true_peak_dbfs": options.dialogue_true_peak_dbfs,
                "dialogue_loudness_range_lu": options.dialogue_loudness_range_lu,
            },
            "blocks": block_results,
            "dialogue": dialogue_manifest,
            "background": background_manifest,
            "mix": mix_manifest,
            "render": render_manifest,
            "tiktok_render": tiktok_render_manifest,
            "outputs": {
                "dialogue": str(artifacts.dialogue),
                "background": (
                    str(artifacts.background) if options.include_background else None
                ),
                "final_mix": str(artifacts.final_mix),
                "canonical_final_video": str(artifacts.final_video),
                "final_video": str(named_outputs.final_video),
                "tiktok_upload_video": str(tiktok_upload_video),
                "seo_files": named_seo_files,
                "seo_title": named_outputs.title,
                "filename_stem": named_outputs.filename_stem,
                "report": str(artifacts.report),
            },
            "idempotent_reuse": False,
        }
        atomic_write_json(artifacts.report, report)

        job.providers["direct_dubbing"] = "azure-tts"
        job.media.pop("direct_dub_seo_package", None)
        job.media.pop("direct_dub_seo_package_sha256", None)
        job.media["direct_dub_canonical_video"] = str(artifacts.final_video)
        job.media["direct_dub_video"] = str(named_outputs.final_video)
        job.media["direct_dub_video_sha256"] = checksum(named_outputs.final_video)
        job.media["tiktok_upload_video"] = str(tiktok_upload_video)
        job.media["tiktok_upload_video_sha256"] = checksum(tiktok_upload_video)
        job.media["direct_dub_seo_files"] = named_seo_files
        job.media["direct_dub_seo_file_sha256"] = {
            key: checksum(path)
            for key, path in sorted(named_outputs.seo_files.items())
        }
        job.media["direct_dub_seo_title"] = named_outputs.title
        job.media["direct_dub_report"] = str(artifacts.report)
        self.jobs.save(job)
        return report

    def _ensure_analysis_audio(self, job: JobManifest) -> Path:
        job_root = self.jobs.job_dir(job.job_id)
        analysis_audio = job_root / "audio" / "analysis_mono.wav"
        if analysis_audio.is_file() and analysis_audio.stat().st_size > 0:
            return analysis_audio
        source_video = Path(job.local_source_path)
        if not source_video.is_file():
            raise FileNotFoundError(source_video)
        mix_source = job_root / "audio" / "mix_source_stereo.wav"
        prepare_source_derivatives(
            source_video,
            analysis_audio,
            mix_source,
            max_duration_seconds=self.settings.max_source_duration_minutes * 60,
        )
        return analysis_audio

    def _ensure_mix_source(self, job: JobManifest, source_video: Path) -> Path:
        job_root = self.jobs.job_dir(job.job_id)
        analysis_audio = job_root / "audio" / "analysis_mono.wav"
        mix_source = job_root / "audio" / "mix_source_stereo.wav"
        if mix_source.is_file() and mix_source.stat().st_size > 0:
            return mix_source
        prepare_source_derivatives(
            source_video,
            analysis_audio,
            mix_source,
            max_duration_seconds=self.settings.max_source_duration_minutes * 60,
        )
        return mix_source

    def _voice_assignments(
        self,
        job: JobManifest,
        voice_map_path: Path | None,
        options: DirectDubOptions,
        *,
        required_speaker_ids: Sequence[str],
        output_path: Path,
        refresh_voice_analysis: bool = False,
    ) -> dict[str, Mapping[str, Any]]:
        """Resolve required canonical speakers from mapped classifier evidence.

        Orchestrator-era ``azure_voice_assignments.json`` is intentionally ignored:
        it may contain the historical unknown-to-Themba fallback that this direct
        renderer is replacing. Explicit ``--voice-map`` remains the only manual
        override surface.
        """
        job_root = self.jobs.job_dir(job.job_id)
        analysis_root = job_root / "analysis"

        transcript_path = analysis_root / "transcript_en.json"
        diarization_path = analysis_root / "azure_diarization.json"
        acoustic_path = analysis_root / "acoustic_analysis.json"
        profiles_path = analysis_root / "speaker_profiles.json"
        legacy_assignment_path = analysis_root / "azure_voice_assignments.json"

        transcript = read_json(transcript_path) if transcript_path.is_file() else {"segments": []}
        diarization = read_json(diarization_path) if diarization_path.is_file() else {"turns": []}
        mapping = build_speaker_id_mapping(transcript, diarization)

        acoustic_raw = read_json(acoustic_path) if acoustic_path.is_file() else {}
        acoustic = canonicalize_voice_family_evidence(acoustic_raw, mapping)
        voice_analysis_refresh_error: str | None = None
        known_matches: dict[str, Any] = {}
        known_match_errors: dict[str, str] = {}

        known_registry_path = (
            self.settings.work_dir / "known_speakers" / "registry.json"
        )
        if known_registry_path.is_file():
            try:
                analysis_audio_path = self._ensure_analysis_audio(job)
                reels = ensure_canonical_speaker_reels(
                    analysis_audio_path=analysis_audio_path,
                    diarization=diarization,
                    speaker_mapping=mapping,
                    required_speaker_ids=required_speaker_ids,
                    speakers_root=job_root / "audio" / "speakers",
                    force=refresh_voice_analysis,
                )
                for speaker_id, reel in reels.items():
                    try:
                        match = identify_known_speaker(
                            Path(str(reel["speaker_audio_path"])),
                            work_dir=self.settings.work_dir,
                        )
                        if match is not None:
                            known_matches[speaker_id] = match
                    except Exception as exc:
                        known_match_errors[speaker_id] = str(exc)
            except Exception as exc:
                for speaker_id in required_speaker_ids:
                    known_match_errors[str(speaker_id)] = str(exc)

        def evidence_is_resolved(speaker_id: str) -> bool:
            item = acoustic.get(speaker_id, {})
            family = str(item.get("voice_family") or "unknown").lower()
            try:
                confidence = float(item.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            return (
                family in {"masculine", "feminine"}
                and confidence >= options.min_voice_family_confidence
            )

        speakers_needing_analysis = [
            speaker_id
            for speaker_id in required_speaker_ids
            if speaker_id not in known_matches
            and (refresh_voice_analysis or not evidence_is_resolved(speaker_id))
        ]
        if speakers_needing_analysis:
            try:
                analysis_audio_path = self._ensure_analysis_audio(job)
                acoustic_raw = ensure_canonical_voice_family_analysis(
                    job_id=job.job_id,
                    analysis_audio_path=analysis_audio_path,
                    diarization=diarization,
                    speaker_mapping=mapping,
                    required_speaker_ids=speakers_needing_analysis,
                    output_path=acoustic_path,
                    speakers_root=job_root / "audio" / "speakers",
                    confidence_threshold=options.min_voice_family_confidence,
                    force=refresh_voice_analysis,
                )
                acoustic = canonicalize_voice_family_evidence(acoustic_raw, mapping)
            except Exception as exc:
                voice_analysis_refresh_error = str(exc)

        profiles_by_id: dict[str, Mapping[str, Any]] = {}
        if profiles_path.is_file():
            profiles = read_json(profiles_path)
            for item in profiles.get("speakers", []):
                if isinstance(item, Mapping) and item.get("speaker_id"):
                    profiles_by_id[str(item["speaker_id"])] = item

        overrides: dict[str, dict[str, Any]] = {}
        if voice_map_path is not None:
            override_artifact = read_json(voice_map_path)
            overrides = {
                speaker_id: value
                for speaker_id, value in _iter_voice_overrides(override_artifact)
            }

        result: dict[str, Mapping[str, Any]] = {}
        resolution_records: list[dict[str, Any]] = []
        unresolved: list[str] = []

        for speaker_id in required_speaker_ids:
            profile = profiles_by_id.get(speaker_id, {})
            acoustic_item = acoustic.get(speaker_id, {})
            raw_ids = list(mapping.get("canonical_to_raw", {}).get(speaker_id, []))

            family = "unknown"
            confidence = 0.0
            source = "unresolved"
            selected_known_voice: str | None = None
            known_match_record: dict[str, Any] | None = None
            known_match_error = known_match_errors.get(speaker_id)

            known_match = known_matches.get(speaker_id)
            if known_match is not None:
                known_match_record = known_match.to_dict()
                family = (
                    "masculine" if known_match.gender == "male" else "feminine"
                )
                confidence = known_match.score
                source = "speechbrain_ecapa_known_speaker"
                selected_known_voice = known_match.azure_voice

            identity_gender = str(profile.get("identity_gender") or "").lower()
            if family == "unknown" and identity_gender in {"male", "female"}:
                family = "masculine" if identity_gender == "male" else "feminine"
                confidence = float(profile.get("identity_confidence") or 1.0)
                source = str(profile.get("identity_gender_source") or "authoritative_identity_gender")
            elif family == "unknown":
                acoustic_family = str(acoustic_item.get("voice_family") or "unknown").lower()
                try:
                    acoustic_confidence = float(acoustic_item.get("confidence", 0.0))
                except (TypeError, ValueError):
                    acoustic_confidence = 0.0
                if (
                    acoustic_family in {"masculine", "feminine"}
                    and acoustic_confidence >= options.min_voice_family_confidence
                ):
                    family = acoustic_family
                    confidence = acoustic_confidence
                    evidence = acoustic_item.get("evidence") or {}
                    source = str(
                        evidence.get("method")
                        or acoustic_item.get("method")
                        or "mapped_acoustic_classifier"
                    )
                else:
                    profile_family = str(profile.get("tts_voice_family") or "unknown").lower()
                    profile_source = str(profile.get("tts_voice_family_source") or "")
                    try:
                        profile_confidence = float(profile.get("acoustic_confidence", 0.0))
                    except (TypeError, ValueError):
                        profile_confidence = 0.0
                    profile_is_authoritative = profile_source in {
                        "authoritative_identity_gender",
                        "authoritative_speaker_registry",
                    }
                    if (
                        profile_family in {"masculine", "feminine"}
                        and (
                            profile_is_authoritative
                            or profile_confidence >= options.min_voice_family_confidence
                        )
                    ):
                        family = profile_family
                        confidence = 1.0 if profile_is_authoritative else profile_confidence
                        source = profile_source or "speaker_profile"

            assignment: dict[str, Any] | None = None
            if family in {"masculine", "feminine"}:
                assignment = {
                    "speaker_id": speaker_id,
                    "selected_voice": (
                        selected_known_voice
                        or (
                            options.female_voice
                            if family == "feminine"
                            else options.male_voice
                        )
                    ),
                    "voice_family": family,
                    "voice_family_confidence": round(confidence, 6),
                    "voice_family_source": source,
                    "raw_speaker_ids": raw_ids,
                    "base_prosody": {
                        "rate_percent": 0,
                        "pitch_percent": 0,
                        "volume_percent": 0,
                    },
                    "source_features": collect_source_features(profile, acoustic_item),
                    "resolution_status": "mapped_voice_family",
                }
                if known_match_record is not None:
                    assignment.update(
                        {
                            "resolution_status": "known_speaker_signature",
                            "known_speaker": known_match_record,
                        }
                    )

            if speaker_id in overrides:
                value = overrides[speaker_id]
                assignment = {
                    **(assignment or {}),
                    "speaker_id": speaker_id,
                    "selected_voice": value["voice"],
                    "base_prosody": {
                        "rate_percent": int(value.get("rate_percent", 0)),
                        "pitch_percent": int(value.get("pitch_percent", 0)),
                        "volume_percent": int(value.get("volume_percent", 0)),
                    },
                    "resolution_status": "explicit_voice_map",
                    "voice_family_source": "explicit_voice_map",
                    "raw_speaker_ids": raw_ids,
                    "source_features": collect_source_features(profile, acoustic_item),
                }

            if assignment is None:
                unresolved.append(speaker_id)
                resolution_records.append(
                    {
                        "speaker_id": speaker_id,
                        "status": "unresolved",
                        "raw_speaker_ids": raw_ids,
                        "minimum_confidence": options.min_voice_family_confidence,
                        "mapped_acoustic_evidence": dict(acoustic_item),
                        "speaker_profile": dict(profile),
                        "known_speaker_match": known_match_record,
                        "known_speaker_error": known_match_error,
                        "voice_analysis_refresh_error": voice_analysis_refresh_error,
                    }
                )
                continue

            result[speaker_id] = assignment
            resolution_records.append(
                {
                    "speaker_id": speaker_id,
                    "status": "resolved",
                    **assignment,
                    "mapped_acoustic_evidence": dict(acoustic_item),
                    "known_speaker_match": known_match_record,
                    "known_speaker_error": known_match_error,
                }
            )

        result = apply_direct_azure_personas(result)
        for record in resolution_records:
            speaker_id = str(record.get("speaker_id") or "")
            if record.get("status") == "resolved" and speaker_id in result:
                record.update(result[speaker_id])

        artifact = {
            "schema_version": "mathula-direct-voice-resolution-v3",
            "persona_version": DIRECT_AZURE_PERSONA_VERSION,
            "job_id": job.job_id,
            "speaker_id_mapping": mapping,
            "minimum_voice_family_confidence": options.min_voice_family_confidence,
            "legacy_azure_assignments": {
                "path": str(legacy_assignment_path),
                "exists": legacy_assignment_path.is_file(),
                "ignored": True,
                "reason": "may contain historical unknown-to-masculine fallback",
            },
            "sources": {
                "transcript": str(transcript_path),
                "diarization": str(diarization_path),
                "acoustic_analysis": str(acoustic_path),
                "speaker_profiles": str(profiles_path),
                "known_speaker_registry": str(known_registry_path),
                "explicit_voice_map": str(voice_map_path) if voice_map_path else None,
                "voice_analysis_refresh_error": voice_analysis_refresh_error,
            },
            "speakers": resolution_records,
            "unresolved_speakers": unresolved,
            "unknown_defaults_to_masculine": False,
        }
        atomic_write_json(output_path, artifact)

        if unresolved:
            details = ", ".join(
                f"{speaker_id} (raw: {mapping.get('canonical_to_raw', {}).get(speaker_id, [])})"
                for speaker_id in unresolved
            )
            raise DirectDubError(
                "Voice family is unresolved for required speaker(s): "
                f"{details}. Unknown speakers no longer default to Themba. "
                f"Inspect {output_path} or provide --voice-map."
            )

        return result

    def _synthesize_block(
        self,
        block: SpeechBlock,
        artifacts: DirectDubArtifacts,
        *,
        voice: str,
        base_rate_percent: int,
        pitch_percent: int,
        volume_percent: int,
        max_azure_rate_percent: int,
        max_time_stretch_ratio: float,
        force: bool,
    ) -> dict[str, Any]:
        if base_rate_percent > max_azure_rate_percent:
            raise DirectDubError(
                f"Speaker base rate {base_rate_percent:+d}% exceeds the direct Azure limit "
                f"{max_azure_rate_percent:+d}%"
            )
        request = AzureTTSRequest(
            turn_id=block.block_id,
            speaker_id=block.speaker_id,
            text=block.tts_text,
            preferred_duration_ms=block.duration_ms,
            maximum_duration_ms=block.duration_ms,
            voice=voice,
            language="zu-ZA",
            rate_percent=base_rate_percent,
            pitch_percent=pitch_percent,
            volume_percent=volume_percent,
        )

        attempts: list[AzureTTSResult] = []
        initial = self.backend.synthesize(
            request,
            artifacts.candidate(request),
            force=force,
        )
        attempts.append(initial)

        if initial.duration_ms > block.duration_ms:
            estimated = estimate_required_azure_rate(
                measured_duration_ms=initial.duration_ms,
                target_duration_ms=block.duration_ms,
                measured_rate_percent=base_rate_percent,
            )
            rates = []
            estimated = min(max_azure_rate_percent, max(base_rate_percent + 1, estimated))
            if estimated > base_rate_percent:
                rates.append(estimated)
            if max_azure_rate_percent > base_rate_percent and max_azure_rate_percent not in rates:
                rates.append(max_azure_rate_percent)
            for rate in rates:
                candidate = self.backend.synthesize(
                    request.with_rate(rate),
                    artifacts.candidate(request.with_rate(rate)),
                    force=force,
                )
                attempts.append(candidate)
                if candidate.duration_ms <= block.duration_ms:
                    break

        fitting = [item for item in attempts if item.duration_ms <= block.duration_ms]
        selected = min(fitting, key=lambda item: item.rate_percent) if fitting else min(
            attempts,
            key=lambda item: item.duration_ms,
        )
        final_path = artifacts.final_block(block.block_id)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        stretch_ratio = 1.0
        if selected.duration_ms > block.duration_ms:
            stretch_ratio = selected.duration_ms / block.duration_ms
            if stretch_ratio > max_time_stretch_ratio:
                raise TimingOverflowError(
                    f"{block.block_id} still needs {stretch_ratio:.3f}x time compression after "
                    f"Azure {selected.rate_percent:+d}%. Increase the same-speaker grouping window "
                    "or revise the approved segment boundary; the renderer will not rewrite text.",
                    measured_duration_ms=selected.duration_ms,
                    target_duration_ms=block.duration_ms,
                    stretch_ratio=stretch_ratio,
                )
            _time_stretch_to_window(
                Path(selected.output_path),
                final_path,
                target_duration_ms=block.duration_ms,
                ratio=stretch_ratio,
                sample_rate=self.backend.sample_rate,
                runner=self.runner,
            )
        else:
            _atomic_copy(Path(selected.output_path), final_path)

        final_duration_ms = round(probe(final_path)["duration"] * 1000)
        if final_duration_ms > block.duration_ms + 25:
            raise DirectDubError(
                f"{block.block_id} exceeds its speech block after fitting: "
                f"{final_duration_ms}ms > {block.duration_ms}ms"
            )
        return {
            **block.to_dict(),
            "voice": voice,
            "base_prosody": {
                "rate_percent": base_rate_percent,
                "pitch_percent": pitch_percent,
                "volume_percent": volume_percent,
            },
            "azure_attempts": [
                {
                    "rate_percent": item.rate_percent,
                    "duration_ms": item.duration_ms,
                    "output_path": item.output_path,
                    "idempotent_reuse": item.idempotent_reuse,
                }
                for item in attempts
            ],
            "selected_azure_rate_percent": selected.rate_percent,
            "post_synthesis_time_stretch_ratio": round(stretch_ratio, 6),
            "output_path": str(final_path),
            "output_sha256": checksum(final_path),
            "duration_ms": final_duration_ms,
        }

    def _request_fingerprint(
        self,
        job: JobManifest,
        translation_path: Path,
        blocks: Sequence[SpeechBlock],
        assignments: Mapping[str, Mapping[str, Any]],
        options: DirectDubOptions,
        clean_background_stem: Path | None,
        seo_path: Path,
    ) -> str:
        payload = {
            "schema_version": DIRECT_DUB_SCHEMA_VERSION,
            "job_id": job.job_id,
            "source_sha256": checksum(Path(job.local_source_path)),
            "translation_sha256": checksum(translation_path),
            "seo_sha256": checksum(seo_path),
            "blocks": [block.to_dict() for block in blocks],
            "assignments": assignments,
            "options": asdict(options),
            "clean_background_sha256": (
                checksum(clean_background_stem)
                if clean_background_stem and clean_background_stem.is_file()
                else None
            ),
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _reuse_completed(
        self,
        artifacts: DirectDubArtifacts,
        request_fingerprint: str,
    ) -> dict[str, Any] | None:
        if not artifacts.report.is_file() or not artifacts.final_video.is_file():
            return None
        report = read_json(artifacts.report)
        if report.get("request_fingerprint") != request_fingerprint:
            return None
        outputs = report.get("outputs")
        if not isinstance(outputs, Mapping):
            return None
        final_video = Path(str(outputs.get("final_video") or ""))
        if not final_video.is_file() or final_video.stat().st_size <= 0:
            return None
        tiktok_upload_video = Path(str(outputs.get("tiktok_upload_video") or ""))
        if (
            not tiktok_upload_video.is_file()
            or tiktok_upload_video.stat().st_size <= 0
        ):
            return None
        seo_files = outputs.get("seo_files")
        if not isinstance(seo_files, Mapping) or not seo_files:
            return None
        for value in seo_files.values():
            path = Path(str(value or ""))
            if not path.is_file():
                return None
        output = dict(report)
        output["idempotent_reuse"] = True
        return output


def borrow_safe_silence_window(
    blocks: Sequence[SpeechBlock],
    block_index: int,
) -> tuple[SpeechBlock, dict[str, Any]]:
    """Expand one block only into silence bounded by neighbouring speech.

    No text or speaker assignment changes. The neighbouring block boundaries
    are hard collision limits, so the returned window cannot overlap another
    speaker (or another utterance from the same speaker).
    """
    if block_index < 0 or block_index >= len(blocks):
        raise IndexError(f"Invalid block index: {block_index}")

    block = blocks[block_index]
    safe_start_ms = (
        blocks[block_index - 1].end_ms if block_index > 0 else block.start_ms
    )
    safe_end_ms = (
        blocks[block_index + 1].start_ms
        if block_index + 1 < len(blocks)
        else block.end_ms
    )
    safe_start_ms = min(block.start_ms, max(0, safe_start_ms))
    safe_end_ms = max(block.end_ms, safe_end_ms)

    expanded = replace(
        block,
        block_id=f"{block.block_id}_adaptive",
        start_ms=safe_start_ms,
        end_ms=safe_end_ms,
    )
    return expanded, {
        "method": "borrow_safe_silence_between_speech",
        "original_block_id": block.block_id,
        "adaptive_block_id": expanded.block_id,
        "original_start_ms": block.start_ms,
        "original_end_ms": block.end_ms,
        "adjusted_start_ms": expanded.start_ms,
        "adjusted_end_ms": expanded.end_ms,
        "borrowed_before_ms": block.start_ms - expanded.start_ms,
        "borrowed_after_ms": expanded.end_ms - block.end_ms,
        "text_immutable": True,
        "cross_speaker_overlap": False,
    }


def rebalance_cross_speaker_handoff(
    blocks: Sequence[SpeechBlock],
    block_index: int,
    *,
    required_window_ms: int,
    max_shift_ms: int = 500,
) -> tuple[SpeechBlock, SpeechBlock | None, dict[str, Any]]:
    """Move one handoff boundary while preserving speaker order and all text.

    This is a last-resort repair after same-speaker grouping and safe silence
    borrowing. It never overlaps speakers: the next block starts exactly where
    the expanded block ends. The approved segment artifacts remain untouched.
    """
    if block_index < 0 or block_index >= len(blocks):
        raise IndexError(f"Invalid block index: {block_index}")
    block = blocks[block_index]
    required_shift_ms = max(0, required_window_ms - block.duration_ms)
    next_block = blocks[block_index + 1] if block_index + 1 < len(blocks) else None
    repair = {
        "method": "bounded_cross_speaker_handoff_rebalance",
        "original_block_id": block.block_id,
        "original_start_ms": block.start_ms,
        "original_end_ms": block.end_ms,
        "required_window_ms": required_window_ms,
        "boundary_shift_ms": required_shift_ms,
        "maximum_boundary_shift_ms": max_shift_ms,
        "text_immutable": True,
        "speaker_order_preserved": True,
        "cross_speaker_overlap": False,
    }
    if (
        required_shift_ms <= 0
        or required_shift_ms > max_shift_ms
        or next_block is None
        or next_block.speaker_id == block.speaker_id
        or next_block.start_ms != block.end_ms
        or next_block.duration_ms - required_shift_ms < 1000
    ):
        repair["applied"] = False
        return block, None, repair

    boundary_ms = block.end_ms + required_shift_ms
    expanded = replace(
        block,
        block_id=f"{block.block_id}_handoff",
        end_ms=boundary_ms,
    )
    shifted_next = replace(next_block, start_ms=boundary_ms)
    repair.update(
        {
            "applied": True,
            "adaptive_block_id": expanded.block_id,
            "adjusted_end_ms": boundary_ms,
            "next_block_id": next_block.block_id,
            "next_original_start_ms": next_block.start_ms,
            "next_adjusted_start_ms": shifted_next.start_ms,
        }
    )
    return expanded, shifted_next, repair


def genuine_speaker_collision_message(
    blocks: Sequence[SpeechBlock],
    block_index: int,
    overflow: TimingOverflowError,
    *,
    required_window_ms: int,
) -> str:
    block = blocks[block_index]
    previous = blocks[block_index - 1] if block_index > 0 else None
    following = blocks[block_index + 1] if block_index + 1 < len(blocks) else None
    return (
        f"Genuine cross-speaker timeline collision for {block.block_id} "
        f"({block.speaker_id}, {block.start_ms}-{block.end_ms}ms): Azure audio is "
        f"{overflow.measured_duration_ms}ms and needs at least {required_window_ms}ms "
        f"at the bounded stretch limit. Previous="
        f"{previous.speaker_id if previous else 'none'}; next="
        f"{following.speaker_id if following else 'none'}. Approved text was not changed."
    )



def normalize_dialogue_loudness(
    dialogue_path: Path,
    *,
    target_lufs: float = -16.0,
    true_peak_dbfs: float = -1.5,
    loudness_range_lu: float = 11.0,
    sample_rate: int = 24000,
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Normalize the completed dialogue bus with FFmpeg's two-pass loudnorm.

    Speaker identity remains pitch/rate only.  Applying one normalization pass
    after timeline assembly gives every voice a common delivery level without
    changing the approved text, timing, or per-speaker waveform independently.
    """

    if not dialogue_path.is_file() or dialogue_path.stat().st_size <= 0:
        raise FileNotFoundError(dialogue_path)

    input_sample_count, input_sample_rate, input_channels = _wav_timeline(dialogue_path)
    if input_sample_count <= 0:
        raise DirectDubError("Dialogue bus has no audio samples")
    if input_sample_rate != sample_rate:
        raise DirectDubError(
            "Dialogue bus sample rate does not match the Azure renderer: "
            f"{input_sample_rate}Hz != {sample_rate}Hz"
        )
    input_duration_ms = round(input_sample_count * 1000 / input_sample_rate)

    target = _format_loudness_value(target_lufs)
    peak = _format_loudness_value(true_peak_dbfs)
    lra = _format_loudness_value(loudness_range_lu)
    measurement_filter = (
        f"loudnorm=I={target}:TP={peak}:LRA={lra}:print_format=json"
    )
    measurement = _run_loudnorm_command(
        runner,
        [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-i",
            str(dialogue_path),
            "-af",
            measurement_filter,
            "-f",
            "null",
            "-",
        ],
        stage="measurement",
    )
    measured = _extract_loudnorm_json(measurement.stderr)
    required = (
        "input_i",
        "input_tp",
        "input_lra",
        "input_thresh",
        "target_offset",
    )
    missing = [key for key in required if key not in measured]
    if missing:
        raise DirectDubError(
            "FFmpeg loudnorm measurement omitted required values: "
            + ", ".join(missing)
        )

    apply_filter = (
        f"loudnorm=I={target}:TP={peak}:LRA={lra}:"
        f"measured_I={measured['input_i']}:"
        f"measured_TP={measured['input_tp']}:"
        f"measured_LRA={measured['input_lra']}:"
        f"measured_thresh={measured['input_thresh']}:"
        f"offset={measured['target_offset']}:"
        "linear=true:print_format=json,"
        f"aresample={sample_rate},"
        "asetpts=N/SR/TB,"
        "apad,"
        f"atrim=end_sample={input_sample_count}"
    )
    temporary = dialogue_path.with_name(
        f".{dialogue_path.stem}.loudnorm.partial{dialogue_path.suffix}"
    )
    temporary.unlink(missing_ok=True)
    try:
        applied = _run_loudnorm_command(
            runner,
            [
                "ffmpeg",
                "-hide_banner",
                "-nostats",
                "-y",
                "-i",
                str(dialogue_path),
                "-af",
                apply_filter,
                "-ar",
                str(sample_rate),
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                str(temporary),
            ],
            stage="application",
        )
        if not temporary.is_file() or temporary.stat().st_size <= 0:
            raise DirectDubError(
                "FFmpeg loudnorm completed without producing normalized dialogue"
            )
        output_metrics = _extract_loudnorm_json(applied.stderr)
        output_sample_count, output_sample_rate, output_channels = _wav_timeline(
            temporary
        )
        if output_sample_rate != input_sample_rate:
            raise DirectDubError(
                "Dialogue loudness normalization changed sample rate: "
                f"{input_sample_rate}Hz -> {output_sample_rate}Hz"
            )
        if output_channels != input_channels:
            raise DirectDubError(
                "Dialogue loudness normalization changed channel count: "
                f"{input_channels} -> {output_channels}"
            )
        if output_sample_count != input_sample_count:
            raise DirectDubError(
                "Dialogue loudness normalization changed timeline sample count: "
                f"{input_sample_count} -> {output_sample_count}"
            )
        output_duration_ms = round(
            output_sample_count * 1000 / output_sample_rate
        )
        os.replace(temporary, dialogue_path)
    finally:
        temporary.unlink(missing_ok=True)

    return {
        "schema_version": "mathula-dialogue-loudness-v2-exact-samples",
        "method": "ffmpeg_loudnorm_two_pass_linear",
        "scope": "completed_dialogue_bus",
        "per_speaker_volume_percent": 0,
        "target": {
            "integrated_lufs": float(target_lufs),
            "true_peak_dbfs": float(true_peak_dbfs),
            "loudness_range_lu": float(loudness_range_lu),
        },
        "measured_input": measured,
        "measured_output": output_metrics,
        "input_sample_count": input_sample_count,
        "output_sample_count": output_sample_count,
        "sample_rate": input_sample_rate,
        "channels": input_channels,
        "input_duration_ms": input_duration_ms,
        "output_duration_ms": output_duration_ms,
        "timeline_preserved": output_sample_count == input_sample_count,
        "output_path": str(dialogue_path),
        "output_sha256": checksum(dialogue_path),
    }


def _wav_timeline(path: Path) -> tuple[int, int, int]:
    """Return exact PCM frame count, sample rate, and channel count."""

    try:
        with wave.open(str(path), "rb") as audio:
            return (
                int(audio.getnframes()),
                int(audio.getframerate()),
                int(audio.getnchannels()),
            )
    except (wave.Error, EOFError) as exc:
        raise DirectDubError(f"Invalid PCM WAV dialogue bus: {path}") from exc


def _run_loudnorm_command(
    runner: Callable[..., Any],
    command: list[str],
    *,
    stage: str,
) -> Any:
    try:
        result = runner(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = str(exc.stderr or "").strip()
        tail = stderr[-2000:] if stderr else "no FFmpeg diagnostics"
        raise DirectDubError(
            f"Dialogue loudness {stage} failed: {tail}"
        ) from exc
    if result is None or not hasattr(result, "stderr"):
        raise DirectDubError(
            f"Dialogue loudness {stage} runner returned no diagnostics"
        )
    return result


def _extract_loudnorm_json(stderr: str | bytes | None) -> dict[str, Any]:
    text = (
        stderr.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes)
        else str(stderr or "")
    )
    decoder = json.JSONDecoder()
    candidates: list[dict[str, Any]] = []
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            candidates.append(value)
    for value in reversed(candidates):
        if any(key in value for key in ("input_i", "output_i", "target_offset")):
            return value
    raise DirectDubError("FFmpeg loudnorm did not return a JSON measurement")


def _format_loudness_value(value: float) -> str:
    return f"{float(value):.2f}".rstrip("0").rstrip(".")

def load_approved_segments(
    translation: Mapping[str, Any],
    *,
    source_duration_ms: int,
) -> list[ApprovedSegment]:
    if translation.get("production_authorized") is False:
        raise DirectDubError("Translation is not authorized for production")
    raw_segments = translation.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise DirectDubError("Approved translation must contain a non-empty segments array")

    segments: list[ApprovedSegment] = []
    for index, item in enumerate(raw_segments, start=1):
        if not isinstance(item, Mapping):
            raise DirectDubError(f"Translation segment {index} is not an object")
        segment_id = str(item.get("segment_id") or f"seg-{index:05d}")
        speaker_id = str(item.get("speaker_id") or item.get("speaker") or "").strip()
        text = str(item.get("tts_text") or item.get("translated_text") or "").strip()
        translated_text = str(item.get("translated_text") or text).strip()
        if not speaker_id:
            raise DirectDubError(f"{segment_id} has no speaker ID")
        if not text:
            raise DirectDubError(f"{segment_id} has no TTS text")
        try:
            start_ms = round(float(item["start"]) * 1000)
            end_ms = round(float(item["end"]) * 1000)
        except (KeyError, TypeError, ValueError) as exc:
            raise DirectDubError(f"{segment_id} has invalid timing") from exc
        if start_ms < 0 or end_ms <= start_ms:
            raise DirectDubError(f"{segment_id} has an invalid timing window")
        if end_ms > source_duration_ms + 250:
            raise DirectDubError(f"{segment_id} ends beyond the source video")
        segments.append(
            ApprovedSegment(
                segment_id=segment_id,
                speaker_id=speaker_id,
                start_ms=start_ms,
                end_ms=end_ms,
                source_text=str(item.get("source_text") or "").strip(),
                translated_text=translated_text,
                tts_text=text,
            )
        )

    ordered = sorted(segments, key=lambda item: (item.start_ms, item.end_ms, item.segment_id))
    if ordered != segments:
        raise DirectDubError("Approved translation segments must be in timeline order")
    return segments


def coalesce_same_speaker_segments(
    segments: Sequence[ApprovedSegment],
    *,
    max_gap_ms: int = 1200,
) -> list[SpeechBlock]:
    """Join only adjacent same-speaker segments while preserving all text in order."""
    if max_gap_ms < 0:
        raise ValueError("max_gap_ms must be non-negative")
    blocks: list[SpeechBlock] = []
    pending: list[ApprovedSegment] = []

    def flush() -> None:
        if not pending:
            return
        blocks.append(
            SpeechBlock(
                block_id=f"block_{len(blocks) + 1:04d}",
                speaker_id=pending[0].speaker_id,
                start_ms=pending[0].start_ms,
                end_ms=max(item.end_ms for item in pending),
                segment_ids=tuple(item.segment_id for item in pending),
                source_text=_join_text(item.source_text for item in pending),
                translated_text=_join_text(item.translated_text for item in pending),
                tts_text=_join_text(item.tts_text for item in pending),
            )
        )
        pending.clear()

    for segment in segments:
        if not pending:
            pending.append(segment)
            continue
        previous = pending[-1]
        gap_ms = segment.start_ms - previous.end_ms
        if segment.speaker_id == previous.speaker_id and gap_ms <= max_gap_ms:
            pending.append(segment)
        else:
            flush()
            pending.append(segment)
    flush()

    original_words = _normalised_words(_join_text(item.tts_text for item in segments))
    block_words = _normalised_words(_join_text(item.tts_text for item in blocks))
    if original_words != block_words:
        raise AssertionError("Same-speaker coalescing changed the approved TTS word sequence")
    return blocks


def estimate_required_azure_rate(
    *,
    measured_duration_ms: int,
    target_duration_ms: int,
    measured_rate_percent: int,
) -> int:
    if measured_duration_ms <= 0 or target_duration_ms <= 0:
        raise ValueError("Durations must be positive")
    speed = 1.0 + measured_rate_percent / 100.0
    required = (measured_duration_ms / target_duration_ms) * speed
    return math.ceil((required - 1.0) * 100.0)


def _iter_voice_overrides(value: Mapping[str, Any]) -> Iterable[tuple[str, dict[str, Any]]]:
    raw: Any = value.get("speakers", value)
    if not isinstance(raw, Mapping):
        raise ValueError("Voice map must be an object keyed by speaker ID")
    for speaker_id, item in raw.items():
        if isinstance(item, str):
            yield str(speaker_id), {"voice": item}
        elif isinstance(item, Mapping):
            voice = str(item.get("voice") or item.get("selected_voice") or "").strip()
            if not voice:
                raise ValueError(f"Voice override for {speaker_id} is missing voice")
            yield str(speaker_id), {
                "voice": voice,
                "rate_percent": item.get("rate_percent", 0),
                "pitch_percent": item.get("pitch_percent", 0),
                "volume_percent": item.get("volume_percent", 0),
            }
        else:
            raise ValueError(f"Voice override for {speaker_id} is invalid")


def _time_stretch_to_window(
    source: Path,
    output: Path,
    *,
    target_duration_ms: int,
    ratio: float,
    sample_rate: int,
    runner: Callable[..., Any],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.partial{output.suffix}")
    filters = _atempo_filters(ratio)
    filters.extend(["apad", f"atrim=duration={target_duration_ms / 1000.0:.6f}"])
    runner(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-filter:a",
            ",".join(filters),
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(temporary),
        ],
        check=True,
    )
    temporary.replace(output)


def _atempo_filters(ratio: float) -> list[str]:
    if ratio <= 0:
        raise ValueError("Time-stretch ratio must be positive")
    factors: list[float] = []
    remaining = ratio
    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5
    factors.append(remaining)
    return [f"atempo={factor:.8f}" for factor in factors]


def _join_text(values: Iterable[str]) -> str:
    return " ".join(str(value).strip() for value in values if str(value).strip())


def _normalised_words(value: str) -> list[str]:
    return ["".join(character.lower() for character in token if character.isalnum()) for token in value.split() if any(character.isalnum() for character in token)]


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.partial")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


__all__ = [
    "ApprovedSegment",
    "DirectAzureDubRenderer",
    "DirectDubArtifacts",
    "DirectDubError",
    "DirectDubOptions",
    "SpeechBlock",
    "coalesce_same_speaker_segments",
    "create_direct_azure_backend",
    "estimate_required_azure_rate",
    "load_approved_segments",
]
