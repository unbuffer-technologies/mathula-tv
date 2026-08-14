"""Direct Azure dubbing from an approved segment-level translation.

The approved translation artifact remains authoritative and immutable. Adjacent
fragments from one speaker may be coalesced. Timing overflow is resolved only by
Azure-measured selection across the previous, current, and following prepared
translation variants, followed by a complete collision-safe timeline preflight.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
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
from .cadence import (
    aggregate_cadence_qc,
    assess_perceptual_cadence,
    plan_phrase_cadence,
)
from .config import Settings
from .job_store import JobStore
from .media import checksum, prepare_source_derivatives, probe
from .multivariant_translation import VARIANT_IDS, language_suffix
from .mixing import mix_final_buses, render_dialogue_tracks
from .models import JobManifest, utcnow
from .output_naming import (
    load_seo_output_context,
    publish_dubbed_master_outputs,
)
from .professional_dub import (
    ProfessionalDubPolicy,
    auto_approve_low_risk_cadence_revision,
    build_translation_revision_request,
)
from .pronunciation import (
    PronunciationDictionary,
    build_initialism_ssml_parts,
    with_default_organisation_initialisms,
)
from .rendering import render_review_mp4
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


DIRECT_DUB_SCHEMA_VERSION = "mathula-direct-azure-dub-v10-multivariant-manual-first"
VOICE_RESOLUTION_SCHEMA_VERSION = "mathula-direct-voice-resolution-v4-cached"
LEGACY_VOICE_RESOLUTION_SCHEMA_VERSION = "mathula-direct-voice-resolution-v3"


class DirectDubError(RuntimeError):
    """Raised when the direct renderer cannot safely produce a complete dub."""


ProgressCallback = Callable[[Mapping[str, Any]], None]


class _DirectDubProgress:
    """Emit stable, machine-readable progress without coupling rendering to stdout."""

    def __init__(self, callback: ProgressCallback | None) -> None:
        self.callback = callback
        self.started_at = time.monotonic()
        self.stage = ""
        self.stage_started_at = self.started_at

    def emit(
        self,
        stage: str,
        message: str,
        *,
        status: str = "running",
        current: int | None = None,
        total: int | None = None,
        **details: Any,
    ) -> None:
        if self.callback is None:
            return
        now = time.monotonic()
        if stage != self.stage:
            self.stage = stage
            self.stage_started_at = now
        event: dict[str, Any] = {
            "schema_version": "mathula-direct-dub-progress-v1",
            "stage": stage,
            "status": status,
            "message": message,
            "elapsed_seconds": round(now - self.started_at, 1),
        }
        if current is not None:
            event["current"] = current
        if total is not None:
            event["total"] = total
        if current is not None and total and total > 0:
            event["percent"] = round((current / total) * 100, 1)
            stage_elapsed = max(0.0, now - self.stage_started_at)
            if 0 < current < total and stage_elapsed > 0:
                event["eta_seconds"] = round(
                    (stage_elapsed / current) * (total - current),
                    1,
                )
        event.update(
            {key: value for key, value in details.items() if value is not None}
        )
        try:
            self.callback(event)
        except Exception:
            # Progress output must never break a paid dubbing operation.
            return


class PreparedVariantsExhausted(DirectDubError):
    """Raised after every approved translation variant exceeds its window."""

    def __init__(
        self,
        block: "SpeechBlock",
        overflow: "TimingOverflowError",
        attempts: Sequence[Mapping[str, Any]],
    ) -> None:
        super().__init__(
            f"All prepared translation variants overflowed for {block.block_id}"
        )
        self.block = block
        self.overflow = overflow
        self.attempts = [dict(value) for value in attempts]


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
class PreparedTranslationVariant:
    variant_id: str
    spoken_text: str
    tts_text: str
    target_duration_ms: int | None = None
    estimated_duration_ms: int | None = None
    meaning_preserved: bool = True
    omitted_optional_details: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "omitted_optional_details": list(self.omitted_optional_details),
        }


@dataclass(frozen=True)
class ApprovedSegment:
    segment_id: str
    speaker_id: str
    start_ms: int
    end_ms: int
    source_text: str
    translated_text: str
    tts_text: str
    protected_entities: tuple[str, ...] = ()
    variants: tuple[PreparedTranslationVariant, ...] = ()
    selected_variant_id: str = "natural"

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
    protected_entities: tuple[str, ...] = ()
    variants: tuple[PreparedTranslationVariant, ...] = ()
    selected_variant_id: str = "natural"

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "segment_ids": list(self.segment_ids),
            "protected_entities": list(self.protected_entities),
            "variants": [variant.to_dict() for variant in self.variants],
            "selected_variant_id": self.selected_variant_id,
            "duration_ms": self.duration_ms,
        }

    def with_variant(self, variant: PreparedTranslationVariant) -> "SpeechBlock":
        return replace(
            self,
            block_id=f"{self.block_id}_variant_{variant.variant_id}",
            translated_text=variant.spoken_text,
            tts_text=variant.tts_text,
            selected_variant_id=variant.variant_id,
        )


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
            "parts": [
                {"type": type(part).__name__, **asdict(part)}
                for part in request.parts
            ],
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
    def dubbed_master(self) -> Path:
        return self.root / "dubbed_master.mp4"

    @property
    def legacy_final_video(self) -> Path:
        return self.root / "final_dubbed.mp4"

    @property
    def render_manifest(self) -> Path:
        return self.root / "render_manifest.json"

    @property
    def report(self) -> Path:
        return self.root / "report.json"

    @property
    def voice_resolution(self) -> Path:
        return self.root / "voice_resolution.json"

    @property
    def translation_revision_required(self) -> Path:
        return self.root / "translation_revision_required.json"

    @property
    def auto_approved_revisions(self) -> Path:
        return self.root / "auto_approved_translation_revisions.json"




    @property
    def selected_translation_variants(self) -> Path:
        return self.root / "selected_translation_variants.json"

    @property
    def translation_variant_failures(self) -> Path:
        return self.root / "translation_variant_failures.json"


@dataclass(frozen=True)
class DirectDubOptions:
    same_speaker_gap_ms: int = 1200
    max_azure_rate_percent: int = 10
    max_time_stretch_ratio: float = 1.03
    max_time_expansion_ratio: float = 1.40
    collision_max_time_stretch_ratio: float = 1.03
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
        if not 1.0 <= self.max_time_expansion_ratio <= 1.5:
            raise ValueError("max time-expansion ratio must be between 1.0 and 1.5")
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


def choose_three_block_variant_balance(
    previous_options: Sequence[Mapping[str, Any]],
    following_options: Sequence[Mapping[str, Any]],
    *,
    base_capacity_ms: int,
    required_capacity_ms: int,
) -> tuple[Mapping[str, Any], Mapping[str, Any]] | None:
    """Choose the least aggressive measured neighbour-variant combination.

    Every option must describe Azure-measured capacity released relative to the
    currently selected variant.  The score deliberately prefers spreading a
    modest reduction across both neighbours over forcing one sentence to a much
    more aggressive compression level.
    """

    sufficient: list[
        tuple[tuple[int, int, int, int, str, str], Mapping[str, Any], Mapping[str, Any]]
    ] = []
    for previous in previous_options:
        for following in following_options:
            released_ms = int(previous.get("capacity_release_ms", 0)) + int(
                following.get("capacity_release_ms", 0)
            )
            total_capacity_ms = base_capacity_ms + released_ms
            if total_capacity_ms < required_capacity_ms:
                continue
            previous_steps = int(previous.get("compression_steps", 0))
            following_steps = int(following.get("compression_steps", 0))
            changed_count = int(previous_steps > 0) + int(following_steps > 0)
            score = (
                max(previous_steps, following_steps),
                previous_steps + following_steps,
                changed_count,
                total_capacity_ms - required_capacity_ms,
                str(previous.get("variant_id") or ""),
                str(following.get("variant_id") or ""),
            )
            sufficient.append((score, previous, following))
    if not sufficient:
        return None
    _, previous, following = min(sufficient, key=lambda item: item[0])
    return previous, following





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
        refresh_voice_analysis: bool = False,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        progress = _DirectDubProgress(progress_callback)
        progress.emit(
            "prepare",
            "Preparing approved translation and source media",
            status="started",
            job_id=job.job_id,
        )
        options = options or DirectDubOptions(
            max_azure_rate_percent=min(25, self.settings.azure_rate_max_percent)
        )
        options.validate(self.backend)

        artifacts = DirectDubArtifacts(self.jobs.job_dir(job.job_id))
        artifacts.root.mkdir(parents=True, exist_ok=True)
        source_video = Path(job.local_source_path)
        if not source_video.is_file():
            raise FileNotFoundError(source_video)
        # mathula-exact-final-block-boundary-v13.11.1
        # Probe once and use the same authoritative boundary for translation
        # loading, finalized audio certification, and dialogue mixing.
        source_duration_ms = round(probe(source_video)["duration"] * 1000)
        if source_duration_ms <= 0:
            raise DirectDubError("Source video has no positive duration")

        seo_context = load_seo_output_context(
            self.jobs.job_dir(job.job_id),
            job.target_language,
        )

        translation_path = translation_path or (
            self.jobs.job_dir(job.job_id)
            / "translation"
            / f"transcript_{language_suffix(job.target_language)}.json"
        )
        translation = read_json(translation_path)
        dictionary_path = (
            self.jobs.job_dir(job.job_id) / "dubbing" / "pronunciation_dictionary.json"
        )
        if dictionary_path.is_file():
            pronunciation_dictionary = PronunciationDictionary.from_dict(
                read_json(dictionary_path)
            )
        else:
            pronunciation_dictionary = PronunciationDictionary(
                dictionary_version="v1",
                language=job.target_language,
                job_id=job.job_id,
            )
        pronunciation_dictionary = with_default_organisation_initialisms(
            pronunciation_dictionary
        )
        atomic_write_json(dictionary_path, pronunciation_dictionary.to_dict())
        segments = load_approved_segments(
            translation,
            source_duration_ms=source_duration_ms,
            pronunciation_dictionary=pronunciation_dictionary,
        )
        blocks = coalesce_same_speaker_segments(
            segments,
            max_gap_ms=options.same_speaker_gap_ms,
        )
        if not blocks:
            raise DirectDubError("Approved translation contains no speakable segments")

        required_speaker_ids = sorted({block.speaker_id for block in blocks})
        progress.emit(
            "prepare",
            f"Prepared {len(blocks)} speech blocks for {len(required_speaker_ids)} speakers",
            status="completed",
            current=len(blocks),
            total=len(blocks),
            block_count=len(blocks),
            speaker_count=len(required_speaker_ids),
        )
        progress.emit(
            "voice_resolution",
            "Checking the persisted speaker and Azure voice assignments",
            status="started",
            speaker_count=len(required_speaker_ids),
        )
        assignments = None
        if not refresh_voice_analysis:
            assignments = self._load_cached_voice_assignments(
                job,
                voice_map_path,
                options,
                required_speaker_ids=required_speaker_ids,
                output_path=artifacts.voice_resolution,
            )

        # A persisted voice resolution is authoritative for this job until the
        # caller explicitly asks to refresh speaker analysis.  This keeps
        # ordinary retries and even forced audio/render rebuilds from loading
        # SpeechBrain or re-embedding the same speakers.
        if assignments is not None:
            progress.emit(
                "voice_resolution",
                "Reusing cached speaker identities and Azure voice assignments",
                status="completed",
                current=len(assignments),
                total=len(required_speaker_ids),
                cache_reused=True,
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
                    progress.emit(
                        "complete",
                        "Unchanged job reused without SpeechBrain, Azure TTS, Claude, mixing, or rendering",
                        status="completed",
                        idempotent_reuse=True,
                    )
                    return reused
        else:
            progress.emit(
                "voice_resolution",
                "Running speaker analysis and resolving Azure voices",
                status="running",
                speechbrain_required=True,
                refresh_requested=refresh_voice_analysis,
            )
            assignments = self._voice_assignments(
                job,
                voice_map_path,
                options,
                required_speaker_ids=required_speaker_ids,
                output_path=artifacts.voice_resolution,
                refresh_voice_analysis=refresh_voice_analysis,
            )
            progress.emit(
                "voice_resolution",
                "Speaker analysis and Azure voice resolution completed",
                status="completed",
                current=len(assignments),
                total=len(required_speaker_ids),
                cache_reused=False,
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
                    progress.emit(
                        "complete",
                        "Unchanged job reused after voice resolution",
                        status="completed",
                        idempotent_reuse=True,
                    )
                    return reused

        progress.emit(
            "azure_baseline",
            f"Synthesizing {len(blocks)} baseline blocks with Azure TTS",
            status="started",
            current=0,
            total=len(blocks),
        )
        render_blocks = list(blocks)
        baseline_blocks = list(blocks)
        block_results: list[dict[str, Any]] = []
        auto_approved_revisions: list[dict[str, Any]] = []
        clips: list[dict[str, Any]] = []
        synthesis_profiles: dict[int, tuple[str, int, int, int]] = {}
        baseline_results: dict[int, dict[str, Any]] = {}
        baseline_overflows: dict[int, TimingOverflowError] = {}
        prepared_variant_audits: dict[int, list[dict[str, Any]]] = {}

        # Measure prepared natural/concise/compact translations before any
        # optional AI repair.  The original translator has full-clip context;
        # dub-azure only selects and verifies those approved alternatives.
        for block_index in range(len(baseline_blocks)):
            block = baseline_blocks[block_index]
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
            synthesis_profiles[block_index] = (
                voice,
                base_rate,
                pitch,
                volume,
            )
            try:
                selected_block, fitted, variant_audit = (
                    self._synthesize_prepared_variants(
                        block,
                        artifacts,
                        voice=voice,
                        base_rate_percent=base_rate,
                        pitch_percent=pitch,
                        volume_percent=volume,
                        options=options,
                        force=force,
                        progress=progress,
                        block_number=block_index + 1,
                        block_total=len(baseline_blocks),
                    )
                )
            except PreparedVariantsExhausted as exc:
                baseline_blocks[block_index] = exc.block
                render_blocks[block_index] = exc.block
                baseline_overflows[block_index] = exc.overflow
                prepared_variant_audits[block_index] = exc.attempts
                progress.emit(
                    "azure_baseline",
                    (
                        f"{block.block_id}: all prepared variants overflowed "
                        f"({exc.overflow.measured_duration_ms}ms > "
                        f"{exc.overflow.target_duration_ms}ms)"
                    ),
                    current=block_index + 1,
                    total=len(baseline_blocks),
                    block_id=block.block_id,
                    speaker_id=block.speaker_id,
                    outcome="prepared_variants_exhausted",
                    measured_duration_ms=exc.overflow.measured_duration_ms,
                    target_duration_ms=exc.overflow.target_duration_ms,
                    variant_attempt_count=len(exc.attempts),
                )
            except DirectDubError:
                raise
            else:
                baseline_blocks[block_index] = selected_block
                render_blocks[block_index] = selected_block
                baseline_results[block_index] = fitted
                prepared_variant_audits[block_index] = variant_audit
                selected_variant = (fitted.get("translation_variant") or {}).get(
                    "selected_variant_id"
                )
                progress.emit(
                    "azure_baseline",
                    f"{block.block_id} fitted with {selected_variant or 'natural'} translation",
                    current=block_index + 1,
                    total=len(baseline_blocks),
                    block_id=block.block_id,
                    rendered_block_id=selected_block.block_id,
                    speaker_id=block.speaker_id,
                    outcome="fitted",
                    selected_variant_id=selected_variant,
                    duration_ms=fitted.get("duration_ms"),
                    selected_azure_rate_percent=fitted.get(
                        "selected_azure_rate_percent"
                    ),
                )

        progress.emit(
            "azure_baseline",
            (
                f"Baseline synthesis completed: {len(baseline_results)} fitted, "
                f"{len(baseline_overflows)} require timing repair"
            ),
            status="completed",
            current=len(baseline_blocks),
            total=len(baseline_blocks),
            fitted_count=len(baseline_results),
            overflow_count=len(baseline_overflows),
        )
        if baseline_overflows:
            progress.emit(
                "timing_preflight",
                (
                    f"Prepared variants exhausted for {len(baseline_overflows)} "
                    "block(s); evaluating Azure-measured three-sentence timing "
                    "windows"
                ),
                status="started",
                current=0,
                total=len(baseline_overflows),
                backend="azure_measured_three_sentence_variants",
            )
        else:
            progress.emit(
                "timing_preflight",
                "All blocks fit a prepared translation variant",
                status="completed",
                current=0,
                total=0,
                backend="azure_measured_three_sentence_variants",
            )

        three_block_variant_balances: list[dict[str, Any]] = []
        if baseline_overflows:
            initial_required_windows_ms: dict[int, int] = {}
            for measured_index in range(len(render_blocks)):
                measured_duration_ms: int | None = None
                if measured_index in baseline_results:
                    measured_duration_ms = _selected_azure_measurement_ms(
                        baseline_results[measured_index]
                    )
                elif measured_index in baseline_overflows:
                    measured_duration_ms = baseline_overflows[
                        measured_index
                    ].measured_duration_ms
                if measured_duration_ms is not None:
                    initial_required_windows_ms[measured_index] = math.ceil(
                        measured_duration_ms
                        / options.collision_max_time_stretch_ratio
                    )

            attempted_three_block_balances = 0
            shortened_neighbour_count = 0
            for overflow_index in sorted(baseline_overflows):
                if overflow_index <= 0 or overflow_index + 1 >= len(render_blocks):
                    continue
                previous_index = overflow_index - 1
                following_index = overflow_index + 1
                previous = render_blocks[previous_index]
                current = render_blocks[overflow_index]
                following = render_blocks[following_index]
                if not (
                    previous.speaker_id == following.speaker_id
                    and previous.speaker_id != current.speaker_id
                    and current.duration_ms
                    <= GLOBAL_SANDWICHED_INTERJECTION_MAX_SOURCE_WINDOW_MS
                ):
                    continue
                if (
                    previous_index in baseline_overflows
                    or following_index in baseline_overflows
                ):
                    continue

                required_current_ms = initial_required_windows_ms.get(
                    overflow_index, current.duration_ms
                )
                deficit_ms = required_current_ms - current.duration_ms
                if deficit_ms <= 0:
                    continue
                required_previous_ms = initial_required_windows_ms.get(
                    previous_index, previous.duration_ms
                )
                required_following_ms = initial_required_windows_ms.get(
                    following_index, following.duration_ms
                )
                left_gap_ms = max(0, current.start_ms - previous.end_ms)
                right_gap_ms = max(0, following.start_ms - current.end_ms)
                previous_surplus_ms = max(
                    0, previous.duration_ms - required_previous_ms
                )
                following_surplus_ms = max(
                    0, following.duration_ms - required_following_ms
                )
                base_capacity_ms = (
                    left_gap_ms
                    + right_gap_ms
                    + previous_surplus_ms
                    + following_surplus_ms
                )
                if base_capacity_ms >= deficit_ms:
                    continue

                attempted_three_block_balances += 1
                previous_voice, previous_rate, previous_pitch, previous_volume = (
                    synthesis_profiles[previous_index]
                )
                following_voice, following_rate, following_pitch, following_volume = (
                    synthesis_profiles[following_index]
                )
                previous_candidates, previous_attempts = (
                    self._measure_shorter_neighbour_variants_for_timeline_balance(
                        previous,
                        blocks[previous_index],
                        artifacts,
                        role="previous",
                        voice=previous_voice,
                        base_rate_percent=previous_rate,
                        pitch_percent=previous_pitch,
                        volume_percent=previous_volume,
                        options=options,
                        current_required_window_ms=required_previous_ms,
                    )
                )
                following_candidates, following_attempts = (
                    self._measure_shorter_neighbour_variants_for_timeline_balance(
                        following,
                        blocks[following_index],
                        artifacts,
                        role="following",
                        voice=following_voice,
                        base_rate_percent=following_rate,
                        pitch_percent=following_pitch,
                        volume_percent=following_volume,
                        options=options,
                        current_required_window_ms=required_following_ms,
                    )
                )
                previous_options: list[dict[str, Any]] = [
                    {
                        "role": "previous",
                        "variant_id": previous.selected_variant_id,
                        "compression_steps": 0,
                        "capacity_release_ms": 0,
                        "required_window_ms": required_previous_ms,
                        "block": previous,
                        "fitted": baseline_results.get(previous_index),
                    },
                    *previous_candidates,
                ]
                following_options: list[dict[str, Any]] = [
                    {
                        "role": "following",
                        "variant_id": following.selected_variant_id,
                        "compression_steps": 0,
                        "capacity_release_ms": 0,
                        "required_window_ms": required_following_ms,
                        "block": following,
                        "fitted": baseline_results.get(following_index),
                    },
                    *following_candidates,
                ]
                selected_balance = choose_three_block_variant_balance(
                    previous_options,
                    following_options,
                    base_capacity_ms=base_capacity_ms,
                    required_capacity_ms=deficit_ms,
                )
                if selected_balance is None:
                    three_block_variant_balances.append(
                        {
                            "method": "three_block_variant_timeline_balance",
                            "applied": False,
                            "overflow_block_index": overflow_index,
                            "overflow_block_id": current.block_id,
                            "previous_block_index": previous_index,
                            "previous_block_id": previous.block_id,
                            "following_block_index": following_index,
                            "following_block_id": following.block_id,
                            "deficit_ms": deficit_ms,
                            "base_capacity_ms": base_capacity_ms,
                            "previous_attempts": previous_attempts,
                            "following_attempts": following_attempts,
                            "reason": (
                                "no_azure_measured_three_sentence_variant_"
                                "combination_created_enough_capacity"
                            ),
                        }
                    )
                    continue

                selected_previous, selected_following = selected_balance
                neighbour_changes: list[dict[str, Any]] = []
                for neighbour_index, original, selected_option, attempts in (
                    (
                        previous_index,
                        previous,
                        selected_previous,
                        previous_attempts,
                    ),
                    (
                        following_index,
                        following,
                        selected_following,
                        following_attempts,
                    ),
                ):
                    if int(selected_option.get("compression_steps", 0)) <= 0:
                        continue
                    balanced_block = selected_option["block"]
                    balanced_fitted = selected_option["fitted"]
                    balanced_required_ms = int(
                        selected_option["required_window_ms"]
                    )
                    render_blocks[neighbour_index] = balanced_block
                    baseline_blocks[neighbour_index] = balanced_block
                    baseline_results[neighbour_index] = balanced_fitted
                    initial_required_windows_ms[neighbour_index] = (
                        balanced_required_ms
                    )
                    prepared_variant_audits.setdefault(
                        neighbour_index, []
                    ).extend(attempts)
                    shortened_neighbour_count += 1
                    neighbour_changes.append(
                        {
                            "role": selected_option["role"],
                            "block_index": neighbour_index,
                            "source_block_id": original.block_id,
                            "rendered_block_id": balanced_block.block_id,
                            "original_variant_id": original.selected_variant_id,
                            "selected_variant_id": (
                                balanced_block.selected_variant_id
                            ),
                            "compression_steps": int(
                                selected_option["compression_steps"]
                            ),
                            "capacity_released_ms": int(
                                selected_option["capacity_release_ms"]
                            ),
                            "required_window_before_ms": (
                                required_previous_ms
                                if neighbour_index == previous_index
                                else required_following_ms
                            ),
                            "required_window_after_ms": balanced_required_ms,
                        }
                    )

                total_released_ms = sum(
                    int(item.get("capacity_release_ms", 0))
                    for item in (selected_previous, selected_following)
                )
                balance_audit = {
                    "method": "three_block_variant_timeline_balance",
                    "applied": True,
                    "overflow_block_index": overflow_index,
                    "overflow_block_id": current.block_id,
                    "deficit_ms": deficit_ms,
                    "base_capacity_ms": base_capacity_ms,
                    "variant_capacity_released_ms": total_released_ms,
                    "total_capacity_ms": base_capacity_ms + total_released_ms,
                    "previous_selected_variant_id": selected_previous[
                        "variant_id"
                    ],
                    "following_selected_variant_id": selected_following[
                        "variant_id"
                    ],
                    "neighbour_changes": neighbour_changes,
                    "azure_measured": True,
                    "meaning_preserved": True,
                    "selection_policy": (
                        "minimise_max_compression_then_total_compression"
                    ),
                }
                for neighbour_index, selected_option in (
                    (previous_index, selected_previous),
                    (following_index, selected_following),
                ):
                    fitted = selected_option.get("fitted")
                    if isinstance(fitted, dict):
                        fitted["three_block_variant_balance"] = balance_audit
                three_block_variant_balances.append(balance_audit)

            if attempted_three_block_balances:
                applied_three_block_balances = sum(
                    1
                    for item in three_block_variant_balances
                    if item.get("applied")
                )
                progress.emit(
                    "variant_balance",
                    (
                        "Measured three-sentence variant balance completed: "
                        f"{applied_three_block_balances} shared window(s) "
                        f"adjusted; {shortened_neighbour_count} neighbouring "
                        "sentence(s) shortened"
                    ),
                    status="completed",
                    current=applied_three_block_balances,
                    total=attempted_three_block_balances,
                    applied_count=applied_three_block_balances,
                    attempted_count=attempted_three_block_balances,
                    shortened_neighbour_count=shortened_neighbour_count,
                    backend="azure_measured_three_block_variants",
                )

        global_timeline_adjustments: list[dict[str, Any]] = []
        global_timeline_adjustments_by_index: dict[int, dict[str, Any]] = {}
        if baseline_overflows:
            measured_required_windows_ms: dict[int, int] = {}
            for measured_index in range(len(render_blocks)):
                measured_duration_ms: int | None = None
                if measured_index in baseline_results:
                    measured_duration_ms = _selected_azure_measurement_ms(
                        baseline_results[measured_index]
                    )
                elif measured_index in baseline_overflows:
                    measured_duration_ms = baseline_overflows[
                        measured_index
                    ].measured_duration_ms
                if measured_duration_ms is not None:
                    measured_required_windows_ms[measured_index] = math.ceil(
                        measured_duration_ms
                        / options.collision_max_time_stretch_ratio
                    )

            render_blocks, global_timeline_adjustments = (
                rebalance_sandwiched_interjection_timeline(
                    render_blocks,
                    measured_required_windows_ms,
                )
            )
            for adjustment in global_timeline_adjustments:
                middle_index = int(adjustment.get("block_index", -1))
                if middle_index >= 0:
                    global_timeline_adjustments_by_index[middle_index] = {
                        **adjustment,
                        "affected_block_index": middle_index,
                    }
                if not adjustment.get("applied"):
                    continue
                for affected_index in adjustment.get(
                    "affected_block_indices", []
                ):
                    global_timeline_adjustments_by_index[int(affected_index)] = {
                        **adjustment,
                        "affected_block_index": int(affected_index),
                    }
            applied_global_adjustments = sum(
                1
                for adjustment in global_timeline_adjustments
                if adjustment.get("applied")
            )
            if global_timeline_adjustments:
                progress.emit(
                    "timeline_fit",
                    (
                        "Measured global timeline fit completed: "
                        f"{applied_global_adjustments} sandwiched interjection(s) "
                        "rebalanced before block finalization"
                    ),
                    status="completed",
                    current=applied_global_adjustments,
                    total=len(global_timeline_adjustments),
                    applied_count=applied_global_adjustments,
                    attempted_count=len(global_timeline_adjustments),
                    backend="measured_global_timeline",
                )

            unresolved_global_adjustments = [
                adjustment
                for adjustment in global_timeline_adjustments
                if not adjustment.get("applied")
                and int(adjustment.get("block_index", -1))
                in baseline_overflows
            ]
            if unresolved_global_adjustments:
                unresolved_ids = ", ".join(
                    str(item.get("block_id") or "unknown")
                    for item in unresolved_global_adjustments
                )
                raise DirectDubError(
                    "Measured timeline preflight could not fit the remaining "
                    f"cross-speaker block(s): {unresolved_ids}. The overflowing "
                    "block plus shorter variants of both neighbouring sentences "
                    "were Azure-measured as one three-sentence timing window before "
                    "this failure; finalization was not started."
                )

        progress.emit(
            "finalize_blocks",
            f"Finalizing {len(render_blocks)} timeline blocks",
            status="started",
            current=0,
            total=len(render_blocks),
        )
        for block_index in range(len(render_blocks)):
            block = render_blocks[block_index]
            voice, base_rate, pitch, volume = synthesis_profiles[block_index]
            fitted: dict[str, Any] | None = None
            latest_overflow: TimingOverflowError | None = None
            repair_audit: list[dict[str, Any]] = []

            # A prior cross-speaker handoff can shorten this block after the
            # discovery pass. Re-measure that changed window rather than trusting
            # a stale baseline or repair result.
            if block != baseline_blocks[block_index]:
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
                        max_time_expansion_ratio=options.max_time_expansion_ratio,
                        force=False,
                    )
                except TimingOverflowError as exc:
                    latest_overflow = exc
                    repair_audit = list(
                        prepared_variant_audits.get(block_index) or []
                    )
            elif block_index in baseline_results:
                fitted = baseline_results[block_index]
            else:
                latest_overflow = baseline_overflows.get(block_index)
                repair_audit = list(
                    prepared_variant_audits.get(block_index) or []
                )

            if fitted is None:
                if latest_overflow is None:
                    raise DirectDubError(
                        f"No synthesis result or timing failure recorded for {block.block_id}"
                    )
                expanded, timing_adjustment = borrow_safe_silence_window(
                    render_blocks,
                    block_index,
                )
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
                            max_time_expansion_ratio=options.max_time_expansion_ratio,
                            force=False,
                        )
                    except TimingOverflowError as expanded_exc:
                        latest_overflow = expanded_exc
                    else:
                        block = expanded
                        render_blocks[block_index] = expanded
                        fitted["timing_adjustment"] = timing_adjustment
                        if repair_audit:
                            fitted["timing_repair_attempts"] = repair_audit

            if fitted is None:
                assert latest_overflow is not None
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
                    failure_artifact = {
                        "schema_version": "mathula-translation-variant-failures-v1",
                        "job_id": job.job_id,
                        "translation_path": str(translation_path),
                        "translation_sha256": checksum(translation_path),
                        "timing_strategy": "azure_measured_three_sentence_variants",
                        "block_id": blocks[block_index].block_id,
                        "rendered_block_id": block.block_id,
                        "speaker_id": block.speaker_id,
                        "source_segment_ids": list(block.segment_ids),
                        "available_duration_ms": block.duration_ms,
                        "required_window_ms": required_window_ms,
                        "latest_measured_duration_ms": latest_overflow.measured_duration_ms,
                        "prepared_variant_attempts": list(
                            prepared_variant_audits.get(block_index) or repair_audit
                        ),
                        "manual_override_required": True,
                        "next_action": (
                            "Import a revised mathula.translation.multivariant.v1 artifact "
                            "for the measured three-sentence window"
                        ),
                    }
                    atomic_write_json(
                        artifacts.translation_variant_failures,
                        failure_artifact,
                    )
                    message = genuine_speaker_collision_message(
                        render_blocks,
                        block_index,
                        latest_overflow,
                        required_window_ms=required_window_ms,
                    )
                    if repair_audit:
                        message += f" Timing repair attempts={len(repair_audit)}."
                    message += (
                        " All prepared variants across the measured three-sentence "
                        "window were exhausted; import a shorter multivariant "
                        "manual override."
                    )
                    raise DirectDubError(message) from latest_overflow
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
                    max_time_expansion_ratio=options.max_time_expansion_ratio,
                    force=False,
                )
                fitted["timing_adjustment"] = timing_adjustment
                if repair_audit:
                    fitted["timing_repair_attempts"] = repair_audit
            selected_rate = int(fitted.get("selected_azure_rate_percent") or 0)
            if selected_rate - base_rate > 5:
                expanded, silence_adjustment = borrow_safe_silence_window(
                    render_blocks,
                    block_index,
                )
                if expanded.duration_ms > block.duration_ms:
                    natural_fitted = self._synthesize_block(
                        expanded,
                        artifacts,
                        voice=voice,
                        base_rate_percent=base_rate,
                        pitch_percent=pitch,
                        volume_percent=volume,
                        max_azure_rate_percent=options.max_azure_rate_percent,
                        max_time_stretch_ratio=options.max_time_stretch_ratio,
                        max_time_expansion_ratio=options.max_time_expansion_ratio,
                        force=False,
                    )
                    natural_rate = int(
                        natural_fitted.get("selected_azure_rate_percent") or 0
                    )
                    if natural_rate < selected_rate and (
                        natural_fitted.get("cadence_qc") or {}
                    ).get("passed", False):
                        block = expanded
                        render_blocks[block_index] = expanded
                        natural_fitted["timing_adjustment"] = {
                            **silence_adjustment,
                            "reason": "prefer_natural_rate_over_acceleration",
                        }
                        fitted = natural_fitted
            if not (fitted.get("cadence_qc") or {}).get("passed", False):
                decision = auto_approve_low_risk_cadence_revision(
                    block_id=block.block_id,
                    text=block.translated_text,
                    target_language=job.target_language,
                )
                if decision is not None:
                    revised_text = decision["revision"]["after"]
                    revised_block = replace(
                        block,
                        translated_text=revised_text,
                        tts_text=revised_text,
                    )
                    try:
                        revised_fitted = self._synthesize_block(
                            revised_block,
                            artifacts,
                            voice=voice,
                            base_rate_percent=base_rate,
                            pitch_percent=pitch,
                            volume_percent=volume,
                            max_azure_rate_percent=options.max_azure_rate_percent,
                            max_time_stretch_ratio=options.collision_max_time_stretch_ratio,
                            max_time_expansion_ratio=options.max_time_expansion_ratio,
                            force=False,
                        )
                    except TimingOverflowError:
                        pass
                    else:
                        if not (revised_fitted.get("cadence_qc") or {}).get(
                            "passed", False
                        ):
                            selected_duration_ms = max(
                                (
                                    int(attempt["duration_ms"])
                                    for attempt in revised_fitted.get(
                                        "azure_attempts", []
                                    )
                                    if attempt.get("rate_percent")
                                    == revised_fitted.get(
                                        "selected_azure_rate_percent"
                                    )
                                ),
                                default=0,
                            )
                            natural_window_ms = math.ceil(
                                selected_duration_ms
                                / ProfessionalDubPolicy().maximum_whole_voice_change
                            )
                            expanded_revision, shifted_next, handoff = (
                                rebalance_cross_speaker_handoff(
                                    render_blocks,
                                    block_index,
                                    required_window_ms=natural_window_ms,
                                    max_shift_ms=options.max_cross_speaker_boundary_shift_ms,
                                )
                            )
                            if shifted_next is not None:
                                expanded_revision = replace(
                                    expanded_revision,
                                    translated_text=revised_text,
                                    tts_text=revised_text,
                                )
                                retry = self._synthesize_block(
                                    expanded_revision,
                                    artifacts,
                                    voice=voice,
                                    base_rate_percent=base_rate,
                                    pitch_percent=pitch,
                                    volume_percent=volume,
                                    max_azure_rate_percent=options.max_azure_rate_percent,
                                    max_time_stretch_ratio=ProfessionalDubPolicy().maximum_whole_voice_change,
                                    max_time_expansion_ratio=options.max_time_expansion_ratio,
                                    force=False,
                                )
                                retry["timing_adjustment"] = handoff
                                if (retry.get("cadence_qc") or {}).get(
                                    "passed", False
                                ):
                                    render_blocks[block_index] = expanded_revision
                                    render_blocks[block_index + 1] = shifted_next
                                    revised_block = expanded_revision
                                    revised_fitted = retry
                        if (revised_fitted.get("cadence_qc") or {}).get("passed", False):
                            decision["rendered_block_id"] = revised_block.block_id
                            decision["output_sha256"] = revised_fitted["output_sha256"]
                            revised_fitted["auto_approved_revision"] = decision
                            if (
                                fitted.get("timing_adjustment")
                                and not revised_fitted.get("timing_adjustment")
                            ):
                                revised_fitted["timing_adjustment"] = fitted[
                                    "timing_adjustment"
                                ]
                            block = revised_block
                            render_blocks[block_index] = revised_block
                            fitted = revised_fitted
                            auto_approved_revisions.append(decision)
            if not fitted.get("translation_variant"):
                selected_variant = next(
                    (
                        value
                        for value in block.variants
                        if value.variant_id == block.selected_variant_id
                    ),
                    None,
                )
                if selected_variant is not None:
                    fitted["translation_variant"] = {
                        "schema_version": "mathula-selected-translation-variant-v1",
                        "source_block_id": blocks[block_index].block_id,
                        "selected_variant_id": selected_variant.variant_id,
                        "spoken_text": selected_variant.spoken_text,
                        "target_duration_ms": selected_variant.target_duration_ms,
                        "estimated_duration_ms": selected_variant.estimated_duration_ms,
                        "measured_duration_ms": fitted.get("duration_ms"),
                        "omitted_optional_details": list(
                            selected_variant.omitted_optional_details
                        ),
                        "meaning_preserved": selected_variant.meaning_preserved,
                        "selection_reason": "prepared_variant_with_timeline_adjustment",
                        "human_review_required": selected_variant.variant_id != "natural",
                    }
            fitted.setdefault(
                "translation_variant_attempts",
                list(prepared_variant_audits.get(block_index) or []),
            )
            global_adjustment = global_timeline_adjustments_by_index.get(
                block_index
            )
            if global_adjustment is not None:
                fitted["global_timeline_adjustment"] = global_adjustment
                fitted.setdefault("timing_adjustment", global_adjustment)
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
            progress.emit(
                "finalize_blocks",
                f"{block.block_id} finalized",
                current=block_index + 1,
                total=len(render_blocks),
                block_id=block.block_id,
                speaker_id=block.speaker_id,
                duration_ms=fitted.get("duration_ms"),
                timing_repaired=bool(fitted.get("timing_repair")),
                cadence_passed=(fitted.get("cadence_qc") or {}).get("passed"),
            )

        timeline_overflows: list[dict[str, Any]] = []
        for clip in clips:
            clip_start_ms = int(clip["start_ms"])
            clip_duration_ms = int(clip["duration_ms"])
            clip_end_ms = clip_start_ms + clip_duration_ms
            if clip_end_ms > source_duration_ms:
                timeline_overflows.append(
                    {
                        "unit_id": str(clip.get("unit_id") or "unknown"),
                        "start_ms": clip_start_ms,
                        "duration_ms": clip_duration_ms,
                        "end_ms": clip_end_ms,
                        "overflow_ms": clip_end_ms - source_duration_ms,
                    }
                )
        if timeline_overflows:
            first = timeline_overflows[0]
            raise DirectDubError(
                "Finalized dialogue timeline exceeds the source after exact "
                "boundary reconciliation: "
                f"{first['unit_id']} ends at {first['end_ms']}ms, source ends at "
                f"{source_duration_ms}ms (overflow {first['overflow_ms']}ms)"
            )

        progress.emit(
            "finalize_blocks",
            "All timeline blocks finalized inside the source boundary",
            status="completed",
            current=len(render_blocks),
            total=len(render_blocks),
            source_duration_ms=source_duration_ms,
            source_timeline_certified=True,
        )
        progress.emit(
            "audio_render",
            "Rendering per-speaker dialogue tracks",
            status="started",
        )
        dialogue_manifest = render_dialogue_tracks(
            clips,
            artifacts.dialogue_tracks,
            artifacts.dialogue,
            total_duration_ms=source_duration_ms,
            sample_rate=self.backend.sample_rate,
            runner=self.runner,
        )
        progress.emit(
            "audio_render",
            "Dialogue tracks rendered; normalizing dialogue loudness",
            status="running",
            dialogue_track_count=len(dialogue_manifest.get("tracks") or []),
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
        progress.emit(
            "audio_render",
            "Dialogue loudness normalization completed",
            status="completed",
            output_path=str(artifacts.dialogue),
        )
        progress.emit(
            "mix",
            (
                "Preparing background and final audio mix"
                if options.include_background
                else "Preparing dialogue-only final audio mix"
            ),
            status="started",
            background_included=options.include_background,
        )
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

        progress.emit(
            "mix",
            f"Final audio mix completed in {audio_mode} mode",
            status="completed",
            output_path=str(artifacts.final_mix),
            audio_mode=audio_mode,
        )
        progress.emit(
            "video_render",
            "Rendering the clean dubbed master MP4",
            status="started",
        )
        render_manifest = render_review_mp4(
            source_video,
            artifacts.final_mix,
            artifacts.dubbed_master,
            artifacts.render_manifest,
            runner=self.runner,
        )
        artifacts.legacy_final_video.unlink(missing_ok=True)
        progress.emit(
            "video_render",
            "Clean dubbed master rendered; publishing named outputs",
            status="running",
            output_path=str(artifacts.dubbed_master),
        )
        named_outputs = publish_dubbed_master_outputs(
            self.jobs.job_dir(job.job_id),
            artifacts.dubbed_master,
            context=seo_context,
        )
        named_seo_files = {
            key: str(path) for key, path in sorted(named_outputs.seo_files.items())
        }
        # A newly rendered master invalidates any older panel-bearing publication
        # file. Remove the stale final before the TikTok editor atomically creates
        # its replacement; the existing selection manifest may still be reused.
        publication_final = (
            self.jobs.job_dir(job.job_id)
            / "output"
            / f"final_dubbed_{named_outputs.filename_stem}.mp4"
        )
        publication_final.unlink(missing_ok=True)
        progress.emit(
            "video_render",
            "Dubbed master and SEO-named outputs published",
            status="completed",
            output_path=str(named_outputs.final_video),
            publication_panel_pending=True,
        )
        progress.emit(
            "report",
            "Writing dubbing report and review artifacts",
            status="started",
        )
        cadence_summary = aggregate_cadence_qc(block_results)
        auto_revision_artifact = {
            "schema_version": "mathula-auto-approved-cadence-revisions-v1",
            "job_id": job.job_id,
            "revision_count": len(auto_approved_revisions),
            "revisions": auto_approved_revisions,
        }
        if auto_approved_revisions:
            atomic_write_json(
                artifacts.auto_approved_revisions,
                auto_revision_artifact,
            )
        else:
            artifacts.auto_approved_revisions.unlink(missing_ok=True)
        professional_policy = ProfessionalDubPolicy()
        revision_request = build_translation_revision_request(
            job_id=job.job_id,
            translation_path=str(translation_path),
            translation_sha256=checksum(translation_path),
            blocks=block_results,
            policy=professional_policy,
        )
        if revision_request["request_count"]:
            atomic_write_json(
                artifacts.translation_revision_required,
                revision_request,
            )
        else:
            artifacts.translation_revision_required.unlink(missing_ok=True)
        selected_variant_records = [
            {
                "block_id": blocks[index].block_id,
                "rendered_block_id": render_blocks[index].block_id,
                "speaker_id": render_blocks[index].speaker_id,
                "source_segment_ids": list(render_blocks[index].segment_ids),
                **dict(result.get("translation_variant") or {}),
                "attempts": list(result.get("translation_variant_attempts") or []),
            }
            for index, result in enumerate(block_results)
        ]
        selected_variant_artifact = {
            "schema_version": "mathula-selected-translation-variants-v1",
            "job_id": job.job_id,
            "translation_path": str(translation_path),
            "translation_sha256": checksum(translation_path),
            "timing_strategy": "azure_measured_three_sentence_variants",
            "block_count": len(selected_variant_records),
            "non_natural_selection_count": sum(
                1
                for item in selected_variant_records
                if item.get("selected_variant_id") not in {None, "natural"}
            ),
            "blocks": selected_variant_records,
            "human_review_required": any(
                bool(item.get("human_review_required"))
                for item in selected_variant_records
            ),
        }
        atomic_write_json(
            artifacts.selected_translation_variants,
            selected_variant_artifact,
        )
        artifacts.translation_variant_failures.unlink(missing_ok=True)
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
                "multivariant": any(len(segment.variants) >= 3 for segment in segments),
                "selected_variants": str(artifacts.selected_translation_variants),
                "selected_variants_sha256": checksum(
                    artifacts.selected_translation_variants
                ),
                "non_natural_selection_count": selected_variant_artifact[
                    "non_natural_selection_count"
                ],
                "translated_words_preserved": not auto_approved_revisions,
                "translation_repair_used": bool(auto_approved_revisions),
                "auto_approved_revision_count": len(auto_approved_revisions),
                "auto_approved_revisions": (
                    str(artifacts.auto_approved_revisions)
                    if auto_approved_revisions
                    else None
                ),
                "ai_timing_repair_count": 0,
                "ai_timing_repairs": None,
                "wording_revision_allowed": False,
                "approved_translation_artifact_immutable": True,
                "renderer_rewrites_text": False,
                "hard_invariants": [
                    "facts",
                    "numbers",
                    "protected_entities",
                    "attribution",
                    "uncertainty",
                    "editorial_intent",
                ],
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
                "max_time_expansion_ratio": options.max_time_expansion_ratio,
                "prepared_translation_variants": True,
                "variant_selection_mode": "predict_then_measure_natural_concise_compact",
                "timing_repair_mode": (
                    "azure_measured_three_sentence_variant_balance"
                ),
                "timing_repair_backend": "none",
                "timing_repair_batch_size": 0,
                "timing_repair_batch_count": 0,
                "audio_mode": audio_mode,
                "background_included": options.include_background,
                "speaker_volume_percent": 0,
                "dialogue_target_lufs": options.dialogue_target_lufs,
                "dialogue_true_peak_dbfs": options.dialogue_true_peak_dbfs,
                "dialogue_loudness_range_lu": options.dialogue_loudness_range_lu,
            },
            "blocks": block_results,
            "cadence_qc": cadence_summary,
            "professional_dub": {
                "policy": professional_policy.to_dict(),
                "production_ready": cadence_summary["passed"],
                "ai_timing_repair_human_review_required": False,
                "translation_variant_human_review_required": selected_variant_artifact[
                    "human_review_required"
                ],
                "translation_revision_request": (
                    str(artifacts.translation_revision_required)
                    if revision_request["request_count"]
                    else None
                ),
                "translation_revision_request_sha256": (
                    revision_request["artifact_sha256"]
                    if revision_request["request_count"]
                    else None
                ),
            },
            "dialogue": dialogue_manifest,
            "background": background_manifest,
            "mix": mix_manifest,
            "render": render_manifest,
            "outputs": {
                "dialogue": str(artifacts.dialogue),
                "background": (
                    str(artifacts.background) if options.include_background else None
                ),
                "final_mix": str(artifacts.final_mix),
                "canonical_dubbed_master": str(artifacts.dubbed_master),
                "dubbed_master": str(named_outputs.final_video),
                "publication_final_video": None,
                "publication_panel_required": True,
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
        job.media.pop("direct_dub_canonical_video", None)
        job.media["direct_dub_canonical_master"] = str(artifacts.dubbed_master)
        job.media["direct_dub_master"] = str(named_outputs.final_video)
        job.media["direct_dub_master_sha256"] = checksum(named_outputs.final_video)
        job.media.pop("direct_dub_video", None)
        job.media.pop("direct_dub_video_sha256", None)
        job.media.pop("tiktok_upload_video", None)
        job.media.pop("tiktok_upload_video_sha256", None)
        job.media["direct_dub_seo_files"] = named_seo_files
        job.media["direct_dub_seo_file_sha256"] = {
            key: checksum(path)
            for key, path in sorted(named_outputs.seo_files.items())
        }
        job.media["direct_dub_seo_title"] = named_outputs.title
        job.media["direct_dub_report"] = str(artifacts.report)
        job.media["professional_dub_ready"] = cadence_summary["passed"]
        if revision_request["request_count"]:
            job.media["translation_revision_required"] = str(
                artifacts.translation_revision_required
            )
            job.review_readiness = "translation_revision_required"
        else:
            job.media.pop("translation_revision_required", None)
            job.review_readiness = "review_ready"
        self.jobs.save(job)
        progress.emit(
            "report",
            "Dubbing report and job manifest saved",
            status="completed",
            report_path=str(artifacts.report),
        )
        progress.emit(
            "complete",
            "Azure dubbing master completed; publication panel render is next",
            status="completed",
            dubbed_master=str(named_outputs.final_video),
            timing_repair_count=0,
        )
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

    def _load_cached_voice_assignments(
        self,
        job: JobManifest,
        voice_map_path: Path | None,
        options: DirectDubOptions,
        *,
        required_speaker_ids: Sequence[str],
        output_path: Path,
    ) -> dict[str, Mapping[str, Any]] | None:
        """Load stable job-local voice assignments without SpeechBrain.

        Voice identity is intentionally sticky for a job.  Changes to the
        global known-speaker registry do not silently rewrite an existing dub;
        callers opt into that work with ``--refresh-voice-analysis``.  Cheap
        structural checks still invalidate a cache when the speaker mapping,
        explicit voice map, persona version, or resolution options change.
        """
        if not output_path.is_file() or output_path.stat().st_size <= 0:
            return None
        try:
            artifact = read_json(output_path)
        except Exception:
            return None
        if artifact.get("schema_version") not in {
            VOICE_RESOLUTION_SCHEMA_VERSION,
            LEGACY_VOICE_RESOLUTION_SCHEMA_VERSION,
        }:
            return None
        if str(artifact.get("job_id") or "") != job.job_id:
            return None
        if artifact.get("persona_version") != DIRECT_AZURE_PERSONA_VERSION:
            return None
        try:
            cached_confidence = float(
                artifact.get("minimum_voice_family_confidence")
            )
        except (TypeError, ValueError):
            return None
        if abs(cached_confidence - options.min_voice_family_confidence) > 1e-9:
            return None
        if artifact.get("unresolved_speakers"):
            return None

        job_root = self.jobs.job_dir(job.job_id)
        analysis_root = job_root / "analysis"
        transcript_path = analysis_root / "transcript_en.json"
        diarization_path = analysis_root / "azure_diarization.json"
        transcript = (
            read_json(transcript_path)
            if transcript_path.is_file()
            else {"segments": []}
        )
        diarization = (
            read_json(diarization_path)
            if diarization_path.is_file()
            else {"turns": []}
        )
        current_mapping = build_speaker_id_mapping(transcript, diarization)
        if artifact.get("speaker_id_mapping") != current_mapping:
            return None

        sources = artifact.get("sources")
        if not isinstance(sources, Mapping):
            return None
        cached_voice_map = str(sources.get("explicit_voice_map") or "")
        inputs = artifact.get("inputs")
        if not isinstance(inputs, Mapping):
            inputs = {}
        if voice_map_path is None:
            if cached_voice_map:
                return None
        else:
            if not voice_map_path.is_file():
                raise FileNotFoundError(voice_map_path)
            cached_voice_map_sha = str(
                inputs.get("explicit_voice_map_sha256") or ""
            )
            # Legacy artifacts did not persist a voice-map checksum, so they
            # cannot safely satisfy an explicit override request.
            if not cached_voice_map_sha:
                return None
            if cached_voice_map_sha != checksum(voice_map_path):
                return None

        cached_male_voice = str(inputs.get("male_voice") or options.male_voice)
        cached_female_voice = str(
            inputs.get("female_voice") or options.female_voice
        )
        if (
            cached_male_voice != options.male_voice
            or cached_female_voice != options.female_voice
        ):
            return None

        raw_assignments = artifact.get("assignments")
        if isinstance(raw_assignments, Mapping):
            assignments = {
                str(speaker_id): dict(value)
                for speaker_id, value in raw_assignments.items()
                if isinstance(value, Mapping)
            }
        else:
            assignments = {}
            allowed_keys = {
                "speaker_id",
                "selected_voice",
                "voice_family",
                "voice_family_confidence",
                "voice_family_source",
                "raw_speaker_ids",
                "base_prosody",
                "source_features",
                "resolution_status",
                "known_speaker",
            }
            for record in artifact.get("speakers") or []:
                if not isinstance(record, Mapping):
                    continue
                if record.get("status") != "resolved":
                    continue
                speaker_id = str(record.get("speaker_id") or "")
                if not speaker_id:
                    continue
                assignments[speaker_id] = {
                    key: value
                    for key, value in record.items()
                    if key in allowed_keys
                }

        required = {str(value) for value in required_speaker_ids}
        if not required or not required <= set(assignments):
            return None
        selected: dict[str, Mapping[str, Any]] = {}
        for speaker_id in sorted(required):
            assignment = assignments.get(speaker_id)
            if not isinstance(assignment, Mapping):
                return None
            if not str(assignment.get("selected_voice") or ""):
                return None
            if not isinstance(assignment.get("base_prosody"), Mapping):
                return None
            selected[speaker_id] = dict(assignment)
        return selected

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
        overrides: dict[str, dict[str, Any]] = {}
        if voice_map_path is not None:
            override_artifact = read_json(voice_map_path)
            overrides = {
                speaker_id: value
                for speaker_id, value in _iter_voice_overrides(override_artifact)
            }
        speakers_requiring_resolution = [
            speaker_id
            for speaker_id in required_speaker_ids
            if speaker_id not in overrides
        ]
        voice_analysis_refresh_error: str | None = None
        known_matches: dict[str, Any] = {}
        known_match_errors: dict[str, str] = {}

        known_registry_path = (
            self.settings.work_dir / "known_speakers" / "registry.json"
        )
        if known_registry_path.is_file() and speakers_requiring_resolution:
            try:
                analysis_audio_path = self._ensure_analysis_audio(job)
                reels = ensure_canonical_speaker_reels(
                    analysis_audio_path=analysis_audio_path,
                    diarization=diarization,
                    speaker_mapping=mapping,
                    required_speaker_ids=speakers_requiring_resolution,
                    speakers_root=job_root / "audio" / "speakers",
                    force=refresh_voice_analysis,
                )
                for speaker_id, reel in reels.items():
                    if str(reel.get("status") or "ready") != "ready":
                        known_match_errors[speaker_id] = str(
                            reel.get("error")
                            or "Canonical speaker reel is unavailable for identity matching"
                        )
                        continue
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
                for speaker_id in speakers_requiring_resolution:
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
            for speaker_id in speakers_requiring_resolution
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
            "schema_version": VOICE_RESOLUTION_SCHEMA_VERSION,
            "persona_version": DIRECT_AZURE_PERSONA_VERSION,
            "job_id": job.job_id,
            "speaker_id_mapping": mapping,
            "minimum_voice_family_confidence": options.min_voice_family_confidence,
            "inputs": {
                "required_speaker_ids": list(required_speaker_ids),
                "male_voice": options.male_voice,
                "female_voice": options.female_voice,
                "explicit_voice_map_sha256": (
                    checksum(voice_map_path)
                    if voice_map_path is not None and voice_map_path.is_file()
                    else None
                ),
                "refresh_policy": "explicit_refresh_only",
            },
            "assignments": result,
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
        max_time_expansion_ratio: float,
        force: bool,
    ) -> dict[str, Any]:
        effective_max_azure_rate_percent = min(
            max_azure_rate_percent,
            base_rate_percent
            + ProfessionalDubPolicy().maximum_rate_delta_percent,
        )
        if base_rate_percent > effective_max_azure_rate_percent:
            raise DirectDubError(
                f"Speaker base rate {base_rate_percent:+d}% exceeds the direct Azure limit "
                f"{effective_max_azure_rate_percent:+d}%"
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
            parts=build_initialism_ssml_parts(
                block.tts_text,
                language="zu-ZA",
            ),
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
            estimated = min(
                effective_max_azure_rate_percent,
                max(base_rate_percent + 1, estimated),
            )
            if estimated > base_rate_percent:
                rates.append(estimated)
            if (
                effective_max_azure_rate_percent > base_rate_percent
                and effective_max_azure_rate_percent not in rates
            ):
                rates.append(effective_max_azure_rate_percent)
            for rate in rates:
                candidate = self.backend.synthesize(
                    request.with_rate(rate),
                    artifacts.candidate(request.with_rate(rate)),
                    force=force,
                )
                attempts.append(candidate)
                if candidate.duration_ms <= block.duration_ms:
                    break
        elif block.duration_ms - initial.duration_ms > 750:
            estimated = estimate_required_azure_rate(
                measured_duration_ms=initial.duration_ms,
                target_duration_ms=max(
                    1,
                    block.duration_ms
                    - ProfessionalDubPolicy().preferred_transition_room_ms,
                ),
                measured_rate_percent=base_rate_percent,
            )
            slower_rate = max(
                self.backend.ssml_bounds.rate_min_percent,
                min(base_rate_percent - 1, estimated),
            )
            if slower_rate < base_rate_percent:
                slower_request = request.with_rate(slower_rate)
                attempts.append(
                    self.backend.synthesize(
                        slower_request,
                        artifacts.candidate(slower_request),
                        force=force,
                    )
                )

        fitting = [item for item in attempts if item.duration_ms <= block.duration_ms]
        selected = (
            min(
                fitting,
                key=lambda item: (
                    block.duration_ms - item.duration_ms,
                    abs(item.rate_percent - base_rate_percent),
                ),
            )
            if fitting
            else min(attempts, key=lambda item: item.duration_ms)
        )
        final_path = artifacts.final_block(block.block_id)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        stretch_ratio = 1.0
        timing_action = "unchanged"
        pause_fill: dict[str, Any] = {}
        if selected.duration_ms > block.duration_ms:
            stretch_ratio = selected.duration_ms / block.duration_ms
            if stretch_ratio > max_time_stretch_ratio:
                raise TimingOverflowError(
                    f"{block.block_id} still needs {stretch_ratio:.3f}x time compression after "
                    f"Azure {selected.rate_percent:+d}%. A meaning-preserving shorter wording "
                    "revision is required; the renderer will not force unnatural audio.",
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
            timing_action = "compressed_to_window"
        else:
            policy = ProfessionalDubPolicy()
            maximum_expansion = min(
                max_time_expansion_ratio,
                policy.maximum_cadence_repair_voice_change,
            )
            cadence_repair = plan_underfill_cadence_repair(
                measured_duration_ms=selected.duration_ms,
                window_duration_ms=block.duration_ms,
                preferred_transition_room_ms=policy.preferred_transition_room_ms,
                maximum_expansion_ratio=maximum_expansion,
            )
            if cadence_repair["applied"]:
                cadence_target_ms = int(cadence_repair["target_duration_ms"])
                stretch_ratio = selected.duration_ms / cadence_target_ms
                _time_stretch_to_window(
                    Path(selected.output_path),
                    final_path,
                    target_duration_ms=cadence_target_ms,
                    ratio=stretch_ratio,
                    sample_rate=self.backend.sample_rate,
                    runner=self.runner,
                )
                timing_action = "source_cadence_aligned_expansion"
            else:
                _atomic_copy(Path(selected.output_path), final_path)
                timing_action = "natural_speech_with_trailing_room"

        exact_boundary = _enforce_exact_audio_window(
            final_path,
            block_id=block.block_id,
            target_duration_ms=block.duration_ms,
            existing_stretch_ratio=stretch_ratio,
            maximum_time_stretch_ratio=max_time_stretch_ratio,
            sample_rate=self.backend.sample_rate,
            runner=self.runner,
        )
        final_duration_ms = int(exact_boundary["duration_ms"])
        cadence_plan = plan_phrase_cadence(
            block_id=block.block_id,
            text=block.tts_text,
            start_ms=block.start_ms,
            end_ms=block.end_ms,
        )
        cadence_repaired = timing_action == "source_cadence_aligned_expansion"
        policy = ProfessionalDubPolicy()
        cadence_qc = assess_perceptual_cadence(
            selected_rate_percent=selected.rate_percent,
            base_rate_percent=base_rate_percent,
            whole_voice_stretch_ratio=stretch_ratio,
            timeline_fill_action=timing_action,
            pause_fill=pause_fill,
            trailing_room_ms=max(0, block.duration_ms - final_duration_ms),
            maximum_added_pause_ms=0,
            maximum_rate_delta_percent=(
                policy.maximum_cadence_repair_rate_delta_percent
                if cadence_repaired
                else policy.maximum_rate_delta_percent
            ),
            maximum_whole_voice_change=(
                policy.maximum_cadence_repair_voice_change
                if cadence_repaired
                else policy.maximum_whole_voice_change
            ),
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
            "exact_audio_boundary": exact_boundary,
            "timeline_fill_action": timing_action,
            "speech_rate_preserved": timing_action
            in {
                "unchanged",
                "natural_speech_with_trailing_room",
                "source_cadence_aligned_expansion",
            },
            "cadence_plan": cadence_plan.to_dict(),
            "cadence_qc": cadence_qc,
            "output_path": str(final_path),
            "output_sha256": checksum(final_path),
            "duration_ms": final_duration_ms,
        }

    @staticmethod
    def _variant_candidates(block: SpeechBlock) -> list[PreparedTranslationVariant]:
        candidates = [value for value in block.variants if value.meaning_preserved]
        if not candidates:
            candidates = [
                PreparedTranslationVariant(
                    variant_id=block.selected_variant_id or "natural",
                    spoken_text=block.translated_text,
                    tts_text=block.tts_text,
                    target_duration_ms=block.duration_ms,
                    estimated_duration_ms=None,
                    meaning_preserved=True,
                )
            ]
        semantic_order = {value: index for index, value in enumerate(VARIANT_IDS)}
        predicted_fit = [
            value
            for value in candidates
            if value.estimated_duration_ms is not None
            and value.estimated_duration_ms <= round(block.duration_ms * 0.97)
        ]
        if predicted_fit:
            first = min(
                predicted_fit,
                key=lambda value: semantic_order.get(value.variant_id, 99),
            )
            first_rank = semantic_order.get(first.variant_id, 99)
            # If the richest predicted fit is disproved by Azure, try more
            # compressed prepared variants before wasting a call on a richer
            # option that was already predicted not to fit.
            shorter = [
                value
                for value in candidates
                if value is not first
                and semantic_order.get(value.variant_id, 99) > first_rank
            ]
            shorter.sort(key=lambda value: semantic_order.get(value.variant_id, 99))
            richer = [
                value
                for value in candidates
                if value is not first and value not in shorter
            ]
            richer.sort(
                key=lambda value: (
                    value.estimated_duration_ms or 10**12,
                    semantic_order.get(value.variant_id, 99),
                )
            )
            return [first, *shorter, *richer]
        if all(value.estimated_duration_ms is not None for value in candidates):
            return sorted(
                candidates,
                key=lambda value: (
                    value.estimated_duration_ms or 10**12,
                    -semantic_order.get(value.variant_id, 99),
                ),
            )
        return sorted(
            candidates,
            key=lambda value: semantic_order.get(value.variant_id, 99),
        )

    def _measure_shorter_neighbour_variants_for_timeline_balance(
        self,
        selected: SpeechBlock,
        source: SpeechBlock,
        artifacts: DirectDubArtifacts,
        *,
        role: str,
        voice: str,
        base_rate_percent: int,
        pitch_percent: int,
        volume_percent: int,
        options: DirectDubOptions,
        current_required_window_ms: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Azure-measure every approved variant shorter than the current one."""

        semantic_order = {value: index for index, value in enumerate(VARIANT_IDS)}
        selected_rank = semantic_order.get(selected.selected_variant_id, -1)
        candidates = [
            value
            for value in selected.variants
            if value.meaning_preserved
            and semantic_order.get(value.variant_id, 99) > selected_rank
        ]
        candidates.sort(key=lambda value: semantic_order.get(value.variant_id, 99))
        options_found: list[dict[str, Any]] = []
        attempts: list[dict[str, Any]] = []
        for candidate in candidates:
            candidate_block = replace(
                source,
                start_ms=selected.start_ms,
                end_ms=selected.end_ms,
            ).with_variant(candidate)
            compression_steps = max(
                1,
                semantic_order.get(candidate.variant_id, 99) - selected_rank,
            )
            try:
                fitted = self._synthesize_block(
                    candidate_block,
                    artifacts,
                    voice=voice,
                    base_rate_percent=base_rate_percent,
                    pitch_percent=pitch_percent,
                    volume_percent=volume_percent,
                    max_azure_rate_percent=options.max_azure_rate_percent,
                    max_time_stretch_ratio=options.collision_max_time_stretch_ratio,
                    max_time_expansion_ratio=options.max_time_expansion_ratio,
                    force=False,
                )
            except TimingOverflowError as exc:
                attempts.append(
                    {
                        "role": role,
                        "variant_id": candidate.variant_id,
                        "status": "neighbour_variant_overflow",
                        "measured_duration_ms": exc.measured_duration_ms,
                        "available_duration_ms": candidate_block.duration_ms,
                        "compression_steps": compression_steps,
                    }
                )
                continue

            measured_duration_ms = _selected_azure_measurement_ms(fitted)
            if measured_duration_ms is None:
                attempts.append(
                    {
                        "role": role,
                        "variant_id": candidate.variant_id,
                        "status": "missing_azure_measurement",
                        "compression_steps": compression_steps,
                    }
                )
                continue
            required_window_ms = math.ceil(
                measured_duration_ms / options.collision_max_time_stretch_ratio
            )
            capacity_release_ms = max(
                0, current_required_window_ms - required_window_ms
            )
            attempt = {
                "role": role,
                "variant_id": candidate.variant_id,
                "status": "measured",
                "measured_duration_ms": measured_duration_ms,
                "required_window_ms": required_window_ms,
                "current_required_window_ms": current_required_window_ms,
                "capacity_release_ms": capacity_release_ms,
                "compression_steps": compression_steps,
            }
            attempts.append(attempt)
            if capacity_release_ms <= 0:
                continue

            fitted["translation_variant"] = {
                "schema_version": "mathula-selected-translation-variant-v1",
                "source_block_id": source.block_id,
                "selected_variant_id": candidate.variant_id,
                "spoken_text": candidate.spoken_text,
                "target_duration_ms": candidate.target_duration_ms,
                "estimated_duration_ms": candidate.estimated_duration_ms,
                "measured_duration_ms": fitted.get("duration_ms"),
                "omitted_optional_details": list(
                    candidate.omitted_optional_details
                ),
                "meaning_preserved": candidate.meaning_preserved,
                "selection_reason": (
                    "three_block_timeline_balance_azure_verified"
                ),
                "human_review_required": candidate.variant_id != "natural",
            }
            fitted["translation_variant_attempts"] = list(attempts)
            options_found.append(
                {
                    "role": role,
                    "variant_id": candidate.variant_id,
                    "compression_steps": compression_steps,
                    "capacity_release_ms": capacity_release_ms,
                    "required_window_ms": required_window_ms,
                    "block": candidate_block,
                    "fitted": fitted,
                }
            )
        return options_found, attempts

    def _shorten_following_block_for_timeline_balance(
        self,
        following: SpeechBlock,
        source_following: SpeechBlock,
        artifacts: DirectDubArtifacts,
        *,
        voice: str,
        base_rate_percent: int,
        pitch_percent: int,
        volume_percent: int,
        options: DirectDubOptions,
        fixed_capacity_ms: int,
        required_capacity_ms: int,
    ) -> tuple[SpeechBlock, dict[str, Any], list[dict[str, Any]]] | None:
        """Use a shorter approved following variant to free a shared boundary.

        This is the second half of measured A-B-A balancing. The overflowing
        middle turn has already exhausted its own natural/concise/compact
        alternatives. We therefore measure only semantically approved variants
        *shorter than the following block's current selection*. The richest
        Azure-verified candidate that creates enough capacity wins. No duration
        estimate is trusted for the boundary decision.
        """

        semantic_order = {value: index for index, value in enumerate(VARIANT_IDS)}
        selected_rank = semantic_order.get(following.selected_variant_id, -1)
        candidates = [
            value
            for value in following.variants
            if value.meaning_preserved
            and semantic_order.get(value.variant_id, 99) > selected_rank
        ]
        candidates.sort(key=lambda value: semantic_order.get(value.variant_id, 99))
        attempts: list[dict[str, Any]] = []
        for candidate in candidates:
            candidate_block = replace(
                source_following,
                start_ms=following.start_ms,
                end_ms=following.end_ms,
            ).with_variant(candidate)
            try:
                fitted = self._synthesize_block(
                    candidate_block,
                    artifacts,
                    voice=voice,
                    base_rate_percent=base_rate_percent,
                    pitch_percent=pitch_percent,
                    volume_percent=volume_percent,
                    max_azure_rate_percent=options.max_azure_rate_percent,
                    max_time_stretch_ratio=options.collision_max_time_stretch_ratio,
                    max_time_expansion_ratio=options.max_time_expansion_ratio,
                    force=False,
                )
            except TimingOverflowError as exc:
                attempts.append(
                    {
                        "variant_id": candidate.variant_id,
                        "status": "following_variant_overflow",
                        "measured_duration_ms": exc.measured_duration_ms,
                        "available_duration_ms": candidate_block.duration_ms,
                    }
                )
                continue

            measured_duration_ms = _selected_azure_measurement_ms(fitted)
            if measured_duration_ms is None:
                attempts.append(
                    {
                        "variant_id": candidate.variant_id,
                        "status": "missing_azure_measurement",
                    }
                )
                continue
            required_window_ms = math.ceil(
                measured_duration_ms / options.collision_max_time_stretch_ratio
            )
            following_capacity_ms = max(
                0, candidate_block.duration_ms - required_window_ms
            )
            total_capacity_ms = fixed_capacity_ms + following_capacity_ms
            attempt = {
                "variant_id": candidate.variant_id,
                "status": "measured",
                "measured_duration_ms": measured_duration_ms,
                "required_window_ms": required_window_ms,
                "following_capacity_ms": following_capacity_ms,
                "fixed_capacity_ms": fixed_capacity_ms,
                "total_capacity_ms": total_capacity_ms,
                "required_capacity_ms": required_capacity_ms,
            }
            attempts.append(attempt)
            if total_capacity_ms < required_capacity_ms:
                continue

            fitted["translation_variant"] = {
                "schema_version": "mathula-selected-translation-variant-v1",
                "source_block_id": source_following.block_id,
                "selected_variant_id": candidate.variant_id,
                "spoken_text": candidate.spoken_text,
                "target_duration_ms": candidate.target_duration_ms,
                "estimated_duration_ms": candidate.estimated_duration_ms,
                "measured_duration_ms": fitted.get("duration_ms"),
                "omitted_optional_details": list(
                    candidate.omitted_optional_details
                ),
                "meaning_preserved": candidate.meaning_preserved,
                "selection_reason": (
                    "following_variant_timeline_balance_azure_verified"
                ),
                "human_review_required": candidate.variant_id != "natural",
            }
            fitted["translation_variant_attempts"] = attempts
            return candidate_block, fitted, attempts
        return None

    def _synthesize_prepared_variants(
        self,
        block: SpeechBlock,
        artifacts: DirectDubArtifacts,
        *,
        voice: str,
        base_rate_percent: int,
        pitch_percent: int,
        volume_percent: int,
        options: DirectDubOptions,
        force: bool,
        progress: _DirectDubProgress,
        block_number: int,
        block_total: int,
    ) -> tuple[SpeechBlock, dict[str, Any], list[dict[str, Any]]]:
        attempts: list[dict[str, Any]] = []
        last_block = block
        last_overflow: TimingOverflowError | None = None
        candidates = self._variant_candidates(block)
        for candidate_index, variant in enumerate(candidates, start=1):
            candidate_block = block.with_variant(variant)
            last_block = candidate_block
            progress.emit(
                "azure_baseline",
                (
                    f"{block.block_id}: testing prepared {variant.variant_id} "
                    f"translation ({candidate_index}/{len(candidates)})"
                ),
                current=block_number,
                total=block_total,
                block_id=block.block_id,
                speaker_id=block.speaker_id,
                variant_id=variant.variant_id,
                candidate_index=candidate_index,
                candidate_count=len(candidates),
                estimated_duration_ms=variant.estimated_duration_ms,
                target_duration_ms=variant.target_duration_ms,
            )
            try:
                fitted = self._synthesize_block(
                    candidate_block,
                    artifacts,
                    voice=voice,
                    base_rate_percent=base_rate_percent,
                    pitch_percent=pitch_percent,
                    volume_percent=volume_percent,
                    max_azure_rate_percent=options.max_azure_rate_percent,
                    max_time_stretch_ratio=options.max_time_stretch_ratio,
                    max_time_expansion_ratio=options.max_time_expansion_ratio,
                    force=force,
                )
            except TimingOverflowError as exc:
                last_overflow = exc
                attempts.append(
                    {
                        "variant_id": variant.variant_id,
                        "status": "overflow",
                        "spoken_text": variant.spoken_text,
                        "estimated_duration_ms": variant.estimated_duration_ms,
                        "target_duration_ms": variant.target_duration_ms,
                        "measured_duration_ms": exc.measured_duration_ms,
                        "available_duration_ms": candidate_block.duration_ms,
                        "required_stretch_ratio": exc.stretch_ratio,
                    }
                )
                continue
            fitted["translation_variant"] = {
                "schema_version": "mathula-selected-translation-variant-v1",
                "source_block_id": block.block_id,
                "selected_variant_id": variant.variant_id,
                "candidate_index": candidate_index,
                "candidate_count": len(candidates),
                "spoken_text": variant.spoken_text,
                "target_duration_ms": variant.target_duration_ms,
                "estimated_duration_ms": variant.estimated_duration_ms,
                "measured_duration_ms": fitted.get("duration_ms"),
                "omitted_optional_details": list(variant.omitted_optional_details),
                "meaning_preserved": variant.meaning_preserved,
                "selection_reason": (
                    "predicted_fit_then_azure_verified"
                    if variant.estimated_duration_ms is not None
                    else "ordered_azure_measurement"
                ),
                "human_review_required": variant.variant_id != "natural",
            }
            attempts.append(
                {
                    "variant_id": variant.variant_id,
                    "status": "selected",
                    "spoken_text": variant.spoken_text,
                    "estimated_duration_ms": variant.estimated_duration_ms,
                    "target_duration_ms": variant.target_duration_ms,
                    "measured_duration_ms": fitted.get("duration_ms"),
                    "available_duration_ms": candidate_block.duration_ms,
                }
            )
            fitted["translation_variant_attempts"] = attempts
            return candidate_block, fitted, attempts
        assert last_overflow is not None
        raise PreparedVariantsExhausted(last_block, last_overflow, attempts)

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
        if not artifacts.report.is_file() or not artifacts.dubbed_master.is_file():
            return None
        report = read_json(artifacts.report)
        if report.get("request_fingerprint") != request_fingerprint:
            return None
        outputs = report.get("outputs")
        if not isinstance(outputs, Mapping):
            return None
        dubbed_master = Path(str(outputs.get("dubbed_master") or ""))
        if not dubbed_master.is_file() or dubbed_master.stat().st_size <= 0:
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


SANDWICHED_INTERJECTION_MAX_SHIFT_MS = 1000
SANDWICHED_INTERJECTION_MAX_SOURCE_WINDOW_MS = 1200
MINIMUM_SHIFTED_BLOCK_WINDOW_MS = 1000


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

    A narrowly scoped exception allows up to one second for a very short
    interjection sandwiched between two turns from the same surrounding speaker.
    This common broadcast pattern is safer than a generic large handoff shift:
    the interrupted speaker simply resumes slightly later, while speaker order,
    approved wording, and non-overlap remain unchanged.
    """
    if block_index < 0 or block_index >= len(blocks):
        raise IndexError(f"Invalid block index: {block_index}")
    block = blocks[block_index]
    previous_block = blocks[block_index - 1] if block_index > 0 else None
    next_block = blocks[block_index + 1] if block_index + 1 < len(blocks) else None
    required_shift_ms = max(0, required_window_ms - block.duration_ms)
    sandwiched_interjection = bool(
        previous_block is not None
        and next_block is not None
        and previous_block.speaker_id == next_block.speaker_id
        and previous_block.speaker_id != block.speaker_id
        and block.duration_ms <= SANDWICHED_INTERJECTION_MAX_SOURCE_WINDOW_MS
    )
    effective_max_shift_ms = (
        max(max_shift_ms, SANDWICHED_INTERJECTION_MAX_SHIFT_MS)
        if sandwiched_interjection
        else max_shift_ms
    )
    repair = {
        "method": "bounded_cross_speaker_handoff_rebalance",
        "original_block_id": block.block_id,
        "original_start_ms": block.start_ms,
        "original_end_ms": block.end_ms,
        "required_window_ms": required_window_ms,
        "boundary_shift_ms": required_shift_ms,
        "maximum_boundary_shift_ms": max_shift_ms,
        "effective_maximum_boundary_shift_ms": effective_max_shift_ms,
        "sandwiched_interjection": sandwiched_interjection,
        "surrounding_speaker_id": (
            previous_block.speaker_id if sandwiched_interjection else None
        ),
        "text_immutable": True,
        "speaker_order_preserved": True,
        "cross_speaker_overlap": False,
    }
    if (
        required_shift_ms <= 0
        or required_shift_ms > effective_max_shift_ms
        or next_block is None
        or next_block.speaker_id == block.speaker_id
        or next_block.start_ms != block.end_ms
        or next_block.duration_ms - required_shift_ms
        < MINIMUM_SHIFTED_BLOCK_WINDOW_MS
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
            "next_remaining_window_ms": shifted_next.duration_ms,
        }
    )
    return expanded, shifted_next, repair



GLOBAL_SANDWICHED_INTERJECTION_MAX_SOURCE_WINDOW_MS = 4000


def _balanced_capacity_take(
    amount_ms: int,
    left_capacity_ms: int,
    right_capacity_ms: int,
) -> tuple[int, int]:
    """Split a bounded timeline borrow without preferring either neighbour."""

    amount_ms = max(0, int(amount_ms))
    left_capacity_ms = max(0, int(left_capacity_ms))
    right_capacity_ms = max(0, int(right_capacity_ms))
    if amount_ms <= 0:
        return 0, 0
    if amount_ms > left_capacity_ms + right_capacity_ms:
        raise ValueError("timeline borrow exceeds available capacity")

    left_take_ms = min(left_capacity_ms, amount_ms // 2)
    right_take_ms = min(right_capacity_ms, amount_ms - left_take_ms)
    remainder_ms = amount_ms - left_take_ms - right_take_ms
    if remainder_ms:
        extra_left_ms = min(left_capacity_ms - left_take_ms, remainder_ms)
        left_take_ms += extra_left_ms
        remainder_ms -= extra_left_ms
    if remainder_ms:
        extra_right_ms = min(right_capacity_ms - right_take_ms, remainder_ms)
        right_take_ms += extra_right_ms
        remainder_ms -= extra_right_ms
    if remainder_ms:
        raise ValueError("timeline borrow allocation was incomplete")
    return left_take_ms, right_take_ms


def rebalance_sandwiched_interjection_timeline(
    blocks: Sequence[SpeechBlock],
    required_windows_ms: Mapping[int, int],
) -> tuple[list[SpeechBlock], list[dict[str, Any]]]:
    """Fit short A-B-A speech triplets as one measured timeline system.

    ``required_windows_ms`` contains the collision-safe window required by the
    selected Azure audio for each block.  Earlier code expanded only the middle
    interjection and assumed both neighbours already met their own measured
    minima.  When a neighbour also overflowed, that assumption made the final
    capacity assertion impossible even when the complete three-block span had
    enough room.

    This solver keeps the outer triplet boundaries fixed and places all three
    blocks inside that span simultaneously.  Speaker order and approved text are
    immutable.  If the three measured minima cannot fit, the triplet is left
    unchanged and an audit record is returned; this optimisation never aborts
    the renderer with an internal contract error.
    """

    solver_version = "measured_triplet_span_solver_v2"
    adjusted = list(blocks)
    audits: list[dict[str, Any]] = []

    for block_index in range(1, len(adjusted) - 1):
        previous = adjusted[block_index - 1]
        block = adjusted[block_index]
        following = adjusted[block_index + 1]
        if not (
            previous.speaker_id == following.speaker_id
            and previous.speaker_id != block.speaker_id
            and block.duration_ms
            <= GLOBAL_SANDWICHED_INTERJECTION_MAX_SOURCE_WINDOW_MS
        ):
            continue

        required_previous_ms = max(
            1,
            int(required_windows_ms.get(block_index - 1, previous.duration_ms)),
        )
        required_block_ms = max(
            1,
            int(required_windows_ms.get(block_index, block.duration_ms)),
        )
        required_following_ms = max(
            1,
            int(required_windows_ms.get(block_index + 1, following.duration_ms)),
        )

        deficits_ms = {
            previous.block_id: max(0, required_previous_ms - previous.duration_ms),
            block.block_id: max(0, required_block_ms - block.duration_ms),
            following.block_id: max(0, required_following_ms - following.duration_ms),
        }
        if deficits_ms[block.block_id] <= 0:
            continue

        cluster_start_ms = previous.start_ms
        cluster_end_ms = following.end_ms
        cluster_span_ms = cluster_end_ms - cluster_start_ms
        required_total_ms = (
            required_previous_ms + required_block_ms + required_following_ms
        )
        safe_slack_ms = cluster_span_ms - required_total_ms

        base_audit: dict[str, Any] = {
            "method": "global_sandwiched_interjection_timeline_fit",
            "solver_version": solver_version,
            "block_index": block_index,
            "block_id": block.block_id,
            "speaker_id": block.speaker_id,
            "surrounding_speaker_id": previous.speaker_id,
            "affected_block_indices": [
                block_index - 1,
                block_index,
                block_index + 1,
            ],
            "required_windows_ms": {
                previous.block_id: required_previous_ms,
                block.block_id: required_block_ms,
                following.block_id: required_following_ms,
            },
            "original_windows_ms": {
                previous.block_id: previous.duration_ms,
                block.block_id: block.duration_ms,
                following.block_id: following.duration_ms,
            },
            "measured_deficits_ms": deficits_ms,
            "cluster_start_ms": cluster_start_ms,
            "cluster_end_ms": cluster_end_ms,
            "cluster_span_ms": cluster_span_ms,
            "required_total_ms": required_total_ms,
            "safe_slack_ms": max(0, safe_slack_ms),
            "text_immutable": True,
            "speaker_order_preserved": True,
            "cross_speaker_overlap": False,
        }

        if safe_slack_ms < 0:
            audits.append(
                {
                    **base_audit,
                    "applied": False,
                    "reason": "insufficient_measured_triplet_span",
                    "shortfall_ms": -safe_slack_ms,
                }
            )
            continue

        # The middle block must start late enough to leave the previous block's
        # measured minimum and early enough to leave both its own and the
        # following block's measured minima.
        earliest_middle_start_ms = cluster_start_ms + required_previous_ms
        latest_middle_start_ms = (
            cluster_end_ms - required_following_ms - required_block_ms
        )
        desired_middle_start_ms = block.start_ms
        middle_start_ms = min(
            max(desired_middle_start_ms, earliest_middle_start_ms),
            latest_middle_start_ms,
        )
        middle_end_ms = middle_start_ms + required_block_ms

        # Preserve each neighbour's original boundary when it is already safe;
        # otherwise expand it only as far as its measured minimum requires.
        previous_minimum_end_ms = previous.start_ms + required_previous_ms
        previous_end_ms = min(
            middle_start_ms,
            max(previous.end_ms, previous_minimum_end_ms),
        )
        following_latest_start_ms = following.end_ms - required_following_ms
        following_start_ms = max(
            middle_end_ms,
            min(following.start_ms, following_latest_start_ms),
        )

        candidate_previous = replace(previous, end_ms=previous_end_ms)
        candidate_block = replace(
            block,
            start_ms=middle_start_ms,
            end_ms=middle_end_ms,
        )
        candidate_following = replace(following, start_ms=following_start_ms)

        invariant_failures: list[str] = []
        if candidate_previous.duration_ms < required_previous_ms:
            invariant_failures.append("previous_below_required_window")
        if candidate_block.duration_ms < required_block_ms:
            invariant_failures.append("middle_below_required_window")
        if candidate_following.duration_ms < required_following_ms:
            invariant_failures.append("following_below_required_window")
        if candidate_previous.end_ms > candidate_block.start_ms:
            invariant_failures.append("previous_middle_overlap")
        if candidate_block.end_ms > candidate_following.start_ms:
            invariant_failures.append("middle_following_overlap")
        if candidate_previous.start_ms != cluster_start_ms:
            invariant_failures.append("cluster_start_changed")
        if candidate_following.end_ms != cluster_end_ms:
            invariant_failures.append("cluster_end_changed")

        if invariant_failures:
            # This is a bounded optimisation.  Never destroy an otherwise
            # recoverable dub because the optional preflight could not place a
            # triplet.  The later unresolved-window gate will report a normal,
            # operator-facing timing failure when the span genuinely cannot fit.
            audits.append(
                {
                    **base_audit,
                    "applied": False,
                    "reason": "triplet_span_candidate_rejected",
                    "invariant_failures": invariant_failures,
                }
            )
            continue

        adjusted[block_index - 1] = candidate_previous
        adjusted[block_index] = candidate_block
        adjusted[block_index + 1] = candidate_following
        audits.append(
            {
                **base_audit,
                "applied": True,
                "adjusted_windows_ms": {
                    candidate_previous.block_id: candidate_previous.duration_ms,
                    candidate_block.block_id: candidate_block.duration_ms,
                    candidate_following.block_id: candidate_following.duration_ms,
                },
                "adjusted_boundaries_ms": {
                    candidate_previous.block_id: [
                        candidate_previous.start_ms,
                        candidate_previous.end_ms,
                    ],
                    candidate_block.block_id: [
                        candidate_block.start_ms,
                        candidate_block.end_ms,
                    ],
                    candidate_following.block_id: [
                        candidate_following.start_ms,
                        candidate_following.end_ms,
                    ],
                },
                "middle_shift_ms": middle_start_ms - block.start_ms,
                "previous_boundary_shift_ms": previous_end_ms - previous.end_ms,
                "following_boundary_shift_ms": (
                    following_start_ms - following.start_ms
                ),
                "deficit_resolved_ms": deficits_ms[block.block_id],
                "all_triplet_measured_minima_satisfied": True,
                "downstream_overflow_prevented": True,
            }
        )

    return adjusted, audits


def _selected_azure_measurement_ms(fitted: Mapping[str, Any]) -> int | None:
    """Return the raw selected Azure duration before any local time fitting."""

    selected_rate = fitted.get("selected_azure_rate_percent")
    measurements = [
        int(attempt.get("duration_ms") or 0)
        for attempt in fitted.get("azure_attempts", [])
        if attempt.get("rate_percent") == selected_rate
        and int(attempt.get("duration_ms") or 0) > 0
    ]
    if measurements:
        return min(measurements)
    duration_ms = int(fitted.get("duration_ms") or 0)
    return duration_ms if duration_ms > 0 else None

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



def _segment_protected_entities(item: Mapping[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    for key in (
        "protected_entities_found",
        "protected_entities_preserved",
        "protected_entities",
        "protected_names",
    ):
        raw = item.get(key)
        if isinstance(raw, list):
            values.extend(str(value).strip() for value in raw if str(value).strip())
    features = item.get("language_features")
    if isinstance(features, list):
        for feature in features:
            if not isinstance(feature, Mapping):
                continue
            if str(feature.get("policy") or "") not in {
                "preserve_verbatim",
                "spell_out",
                "preserve_and_flag",
            }:
                continue
            value = str(
                feature.get("display_text")
                or feature.get("source_text")
                or ""
            ).strip()
            if value:
                values.append(value)
    return tuple(dict.fromkeys(values))



























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

def _translation_items(translation: Mapping[str, Any]) -> tuple[str, list[Any]]:
    units = translation.get("units")
    if isinstance(units, list) and units:
        return "units", units
    segments = translation.get("segments")
    if isinstance(segments, list) and segments:
        return "segments", segments
    raise DirectDubError(
        "Approved translation must contain a non-empty units or segments array"
    )


def _translation_timing_ms(item: Mapping[str, Any], *, start: bool) -> int:
    if start:
        candidates = (("start_ms", False), ("window_start_ms", False), ("start", True))
    else:
        candidates = (
            ("end_ms", False),
            ("hard_end_ms", False),
            ("window_end_ms", False),
            ("preferred_end_ms", False),
            ("end", True),
        )
    for key, seconds in candidates:
        if key not in item or item.get(key) is None:
            continue
        try:
            value = float(item[key])
        except (TypeError, ValueError) as exc:
            raise DirectDubError(f"Invalid {key}") from exc
        if not math.isfinite(value):
            raise DirectDubError(f"Invalid {key}")
        return round(value * 1000 if seconds else value)

    if not start:
        start_ms = _translation_timing_ms(item, start=True)
        for key in ("maximum_duration_ms", "preferred_duration_ms", "duration_ms"):
            if key not in item or item.get(key) is None:
                continue
            try:
                duration_ms = float(item[key])
            except (TypeError, ValueError) as exc:
                raise DirectDubError(f"Invalid {key}") from exc
            if not math.isfinite(duration_ms):
                raise DirectDubError(f"Invalid {key}")
            return start_ms + round(duration_ms)
        if "duration" in item and item.get("duration") is not None:
            try:
                duration_seconds = float(item["duration"])
            except (TypeError, ValueError) as exc:
                raise DirectDubError("Invalid duration") from exc
            if not math.isfinite(duration_seconds):
                raise DirectDubError("Invalid duration")
            return start_ms + round(duration_seconds * 1000)

    boundary = "start" if start else "end"
    raise DirectDubError(
        f"Translation unit is missing a valid {boundary} timing boundary"
    )


def _prepared_variants(
    item: Mapping[str, Any],
    *,
    pronunciation_dictionary: PronunciationDictionary | None,
) -> tuple[PreparedTranslationVariant, ...]:
    raw_variants = item.get("variants")
    if not isinstance(raw_variants, list):
        return ()
    if [str(value.get("variant_id") or "") for value in raw_variants] != list(VARIANT_IDS):
        raise DirectDubError(
            f"{item.get('unit_id') or item.get('segment_id')} variants must be "
            "natural, concise, compact in that order"
        )
    prepared: list[PreparedTranslationVariant] = []
    for raw in raw_variants:
        if not isinstance(raw, Mapping):
            raise DirectDubError("Translation variant is not an object")
        variant_id = str(raw.get("variant_id") or "").strip()
        spoken = str(raw.get("spoken_text") or "").strip()
        if not spoken:
            raise DirectDubError(f"Translation variant {variant_id} has no spoken_text")
        if raw.get("meaning_preserved") is False:
            raise DirectDubError(
                f"Translation variant {variant_id} is not approved as meaning-preserving"
            )
        tts = (
            pronunciation_dictionary.apply(spoken).tts_text
            if pronunciation_dictionary is not None
            else spoken
        )
        prepared.append(
            PreparedTranslationVariant(
                variant_id=variant_id,
                spoken_text=spoken,
                tts_text=normalize_tts_boundary_artifacts(tts),
                target_duration_ms=(
                    int(raw["target_duration_ms"])
                    if raw.get("target_duration_ms") is not None
                    else None
                ),
                estimated_duration_ms=(
                    int(raw["estimated_duration_ms"])
                    if raw.get("estimated_duration_ms") is not None
                    else None
                ),
                meaning_preserved=bool(raw.get("meaning_preserved", True)),
                omitted_optional_details=tuple(
                    str(value)
                    for value in raw.get("omitted_optional_details", [])
                    if str(value).strip()
                ),
            )
        )
    return tuple(prepared)


def load_approved_segments(
    translation: Mapping[str, Any],
    *,
    source_duration_ms: int,
    pronunciation_dictionary: PronunciationDictionary | None = None,
) -> list[ApprovedSegment]:
    if translation.get("production_authorized") is False:
        raise DirectDubError("Translation is not authorized for production")
    item_kind, raw_segments = _translation_items(translation)

    segments: list[ApprovedSegment] = []
    for index, item in enumerate(raw_segments, start=1):
        if not isinstance(item, Mapping):
            raise DirectDubError(f"Translation {item_kind[:-1]} {index} is not an object")
        segment_id = str(
            item.get("unit_id")
            or item.get("segment_id")
            or f"seg-{index:05d}"
        )
        speaker_id = str(item.get("speaker_id") or item.get("speaker") or "").strip()
        variants = _prepared_variants(
            item,
            pronunciation_dictionary=pronunciation_dictionary,
        )
        if variants:
            selected = variants[0]
            translated_text = selected.spoken_text
            text = selected.tts_text
        else:
            text = str(item.get("tts_text") or item.get("translated_text") or "").strip()
            translated_text = str(item.get("translated_text") or text).strip()
            if pronunciation_dictionary is not None:
                text = pronunciation_dictionary.apply(text).tts_text
            text = normalize_tts_boundary_artifacts(text)
            if translated_text:
                variants = (
                    PreparedTranslationVariant(
                        variant_id="natural",
                        spoken_text=translated_text,
                        tts_text=text,
                        target_duration_ms=None,
                        estimated_duration_ms=None,
                        meaning_preserved=True,
                    ),
                )
        if not speaker_id:
            raise DirectDubError(f"{segment_id} has no speaker ID")
        if not text or not translated_text:
            raise DirectDubError(f"{segment_id} has no approved spoken/TTS text")
        start_ms = _translation_timing_ms(item, start=True)
        end_ms = _translation_timing_ms(item, start=False)
        if start_ms < 0 or end_ms <= start_ms:
            raise DirectDubError(f"{segment_id} has an invalid timing window")
        if start_ms >= source_duration_ms:
            raise DirectDubError(
                f"{segment_id} starts at or beyond the source video boundary"
            )
        if end_ms > source_duration_ms + 250:
            raise DirectDubError(f"{segment_id} ends beyond the source video")
        # Azure/FFmpeg/container timing can differ by a few milliseconds. The
        # loader historically accepted a 250ms tolerance, but the mixer uses a
        # strict source boundary. Normalize accepted tail drift here so the two
        # contracts cannot disagree later.
        end_ms = min(end_ms, source_duration_ms)
        if end_ms <= start_ms:
            raise DirectDubError(
                f"{segment_id} has no usable speech window inside the source video"
            )
        protected_entities = tuple(
            dict.fromkeys(
                (*_segment_protected_entities(item), *(
                    str(value)
                    for value in item.get("protected_spans", [])
                    if str(value).strip()
                ))
            )
        )
        segments.append(
            ApprovedSegment(
                segment_id=segment_id,
                speaker_id=speaker_id,
                start_ms=start_ms,
                end_ms=end_ms,
                source_text=str(item.get("source_text") or "").strip(),
                translated_text=translated_text,
                tts_text=text,
                protected_entities=protected_entities,
                variants=variants,
                selected_variant_id="natural",
            )
        )

    ordered = sorted(segments, key=lambda item: (item.start_ms, item.end_ms, item.segment_id))
    if ordered != segments:
        raise DirectDubError("Approved translation segments must be in timeline order")
    return segments


def _coalesced_variants(pending: Sequence[ApprovedSegment]) -> tuple[PreparedTranslationVariant, ...]:
    available_ids = [
        variant_id
        for variant_id in VARIANT_IDS
        if all(any(value.variant_id == variant_id for value in item.variants) for item in pending)
    ]
    if not available_ids:
        return ()
    output: list[PreparedTranslationVariant] = []
    for variant_id in available_ids:
        values = [
            next(value for value in item.variants if value.variant_id == variant_id)
            for item in pending
        ]
        output.append(
            PreparedTranslationVariant(
                variant_id=variant_id,
                spoken_text=_join_text(value.spoken_text for value in values),
                tts_text=_join_text(value.tts_text for value in values),
                target_duration_ms=sum(
                    value.target_duration_ms or item.duration_ms
                    for value, item in zip(values, pending, strict=True)
                ),
                estimated_duration_ms=(
                    sum(value.estimated_duration_ms or 0 for value in values)
                    if all(value.estimated_duration_ms is not None for value in values)
                    else None
                ),
                meaning_preserved=all(value.meaning_preserved for value in values),
                omitted_optional_details=tuple(
                    dict.fromkeys(
                        detail
                        for value in values
                        for detail in value.omitted_optional_details
                    )
                ),
            )
        )
    return tuple(output)


def coalesce_same_speaker_segments(
    segments: Sequence[ApprovedSegment],
    *,
    max_gap_ms: int = 1200,
) -> list[SpeechBlock]:
    """Join adjacent same-speaker units and preserve aligned variant families."""
    if max_gap_ms < 0:
        raise ValueError("max_gap_ms must be non-negative")
    blocks: list[SpeechBlock] = []
    pending: list[ApprovedSegment] = []

    def flush() -> None:
        if not pending:
            return
        variants = _coalesced_variants(pending)
        natural = next(
            (value for value in variants if value.variant_id == "natural"),
            None,
        )
        blocks.append(
            SpeechBlock(
                block_id=f"block_{len(blocks) + 1:04d}",
                speaker_id=pending[0].speaker_id,
                start_ms=pending[0].start_ms,
                end_ms=max(item.end_ms for item in pending),
                segment_ids=tuple(item.segment_id for item in pending),
                source_text=_join_text(item.source_text for item in pending),
                translated_text=(
                    natural.spoken_text
                    if natural is not None
                    else _join_text(item.translated_text for item in pending)
                ),
                tts_text=(
                    natural.tts_text
                    if natural is not None
                    else _join_text(item.tts_text for item in pending)
                ),
                protected_entities=tuple(
                    dict.fromkeys(
                        value
                        for item in pending
                        for value in item.protected_entities
                        if value
                    )
                ),
                variants=variants,
                selected_variant_id="natural",
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

    # Legacy single-text artifacts retain the old immutable word-sequence check.
    if all(len(item.variants) <= 1 for item in segments):
        original_words = _normalised_words(_join_text(item.tts_text for item in segments))
        block_words = _normalised_words(_join_text(item.tts_text for item in blocks))
        if original_words != block_words:
            raise AssertionError("Same-speaker coalescing changed the approved TTS word sequence")
    return blocks


def borrow_trailing_transition_silence(
    blocks: Sequence[SpeechBlock],
    *,
    maximum_borrow_ms: int = 5000,
) -> list[SpeechBlock]:
    """Give a turn nearby safe silence before increasing its speaking rate."""

    if maximum_borrow_ms < 0:
        raise ValueError("maximum_borrow_ms must be non-negative")
    adjusted: list[SpeechBlock] = []
    for index, block in enumerate(blocks):
        next_start = (
            blocks[index + 1].start_ms
            if index + 1 < len(blocks)
            else block.end_ms
        )
        available = max(0, next_start - block.end_ms)
        adjusted.append(
            replace(block, end_ms=block.end_ms + min(available, maximum_borrow_ms))
        )
    return adjusted


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


def plan_underfill_cadence_repair(
    *,
    measured_duration_ms: int,
    window_duration_ms: int,
    preferred_transition_room_ms: int = 500,
    maximum_expansion_ratio: float = 1.06,
    maximum_unexplained_trailing_room_ms: int = 750,
) -> dict[str, Any]:
    """Plan a bounded full-turn expansion instead of dumping underfill at EOF."""

    if measured_duration_ms <= 0 or window_duration_ms <= 0:
        raise ValueError("Cadence repair durations must be positive")
    trailing_room_ms = max(0, window_duration_ms - measured_duration_ms)
    target_duration_ms = max(
        measured_duration_ms,
        window_duration_ms - max(0, preferred_transition_room_ms),
    )
    required_expansion_ratio = target_duration_ms / measured_duration_ms
    applied = (
        trailing_room_ms > maximum_unexplained_trailing_room_ms
        and required_expansion_ratio <= maximum_expansion_ratio
    )
    return {
        "applied": applied,
        "measured_duration_ms": measured_duration_ms,
        "window_duration_ms": window_duration_ms,
        "trailing_room_before_ms": trailing_room_ms,
        "target_duration_ms": target_duration_ms,
        "transition_room_after_ms": window_duration_ms - target_duration_ms,
        "required_expansion_ratio": round(required_expansion_ratio, 6),
        "maximum_expansion_ratio": maximum_expansion_ratio,
    }


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



def _enforce_exact_audio_window(
    audio_path: Path,
    *,
    block_id: str,
    target_duration_ms: int,
    existing_stretch_ratio: float,
    maximum_time_stretch_ratio: float,
    sample_rate: int,
    runner: Callable[..., Any],
) -> dict[str, Any]:
    """Reconcile measured WAV duration with the exact block boundary.

    Azure candidate metadata and a WAV probe can differ by a few milliseconds.
    Dialogue mixing is intentionally strict and must never truncate speech, so
    any measured overrun is corrected in the block WAV before it reaches the
    mixer. The combined correction remains inside the configured naturalness
    limit.
    """

    if target_duration_ms <= 0:
        raise ValueError("Exact audio target duration must be positive")
    measured_duration_ms = round(probe(audio_path)["duration"] * 1000)
    if measured_duration_ms <= target_duration_ms:
        return {
            "schema_version": "mathula-exact-audio-boundary-v1",
            "corrected": False,
            "measured_before_ms": measured_duration_ms,
            "duration_ms": measured_duration_ms,
            "target_duration_ms": target_duration_ms,
            "correction_ratio": 1.0,
            "combined_stretch_ratio": round(existing_stretch_ratio, 9),
        }

    correction_ratio = measured_duration_ms / target_duration_ms
    # This is a technical WAV/container boundary correction, not another
    # semantic timing strategy. Limit it to one percent; larger differences are
    # real timing failures and must return to normal variant/timeline handling.
    maximum_boundary_correction_ratio = min(1.01, maximum_time_stretch_ratio)
    if correction_ratio > maximum_boundary_correction_ratio + 1e-9:
        raise DirectDubError(
            f"{block_id} needs {correction_ratio:.6f}x exact-boundary correction; "
            f"technical correction maximum is "
            f"{maximum_boundary_correction_ratio:.6f}x"
        )
    combined_stretch_ratio = existing_stretch_ratio * correction_ratio

    _time_stretch_to_window(
        audio_path,
        audio_path,
        target_duration_ms=target_duration_ms,
        ratio=correction_ratio,
        sample_rate=sample_rate,
        runner=runner,
    )
    corrected_duration_ms = round(probe(audio_path)["duration"] * 1000)
    if corrected_duration_ms > target_duration_ms:
        raise DirectDubError(
            f"{block_id} still exceeds its exact speech block after boundary "
            f"reconciliation: {corrected_duration_ms}ms > {target_duration_ms}ms"
        )

    return {
        "schema_version": "mathula-exact-audio-boundary-v1",
        "corrected": True,
        "measured_before_ms": measured_duration_ms,
        "duration_ms": corrected_duration_ms,
        "target_duration_ms": target_duration_ms,
        "correction_ratio": round(correction_ratio, 9),
        "combined_stretch_ratio": round(combined_stretch_ratio, 9),
    }


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


def normalize_tts_boundary_artifacts(value: str) -> str:
    """Turn transcript-fragment ellipses into one natural spoken boundary.

    Display/translated text remains untouched. Azure receives a comma instead
    of literal repeated dots, avoiding audible hesitation or word restarts when
    adjacent STT fragments end and begin with ellipses.
    """

    return re.sub(r"(?:\s*\.{3,}\s*)+", ", ", value).strip(" ,")


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
    "ProgressCallback",
    "SpeechBlock",
    "coalesce_same_speaker_segments",
    "create_direct_azure_backend",
    "estimate_required_azure_rate",
    "load_approved_segments",
]
