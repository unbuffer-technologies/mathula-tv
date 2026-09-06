"""Direct Azure dubbing from an approved segment-level translation.

The approved translation artifact remains authoritative and immutable. The
balanced mode retains prepared variants and bounded timeline fitting. The
opt-in semantic-fit mode instead locks speaker tempo and searches all unresolved
blocks in shared Azure-measured, independently reviewed feedback rounds. The
performance-plan mode goes further: Azure word-timed speech islands define
speaker-lane regions, semantic beats may cross STT storage boundaries, and both
the complete region and every target island must pass locked-tempo measurement.
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
import unicodedata
import wave
from statistics import median
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .ai_provider import AIProvider, TURN_TIMING_REPAIR_OPTIONS_SCHEMA
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
from .contextual_speaker_identity import (
    CONTEXTUAL_SPEAKER_POLICY_VERSION,
    resolve_contextual_speaker_identities,
)
from .job_store import JobStore
from .media import checksum, prepare_source_derivatives, probe
from .multivariant_translation import (
    VARIANT_IDS,
    identity_requirements_for_unit,
    language_suffix,
)
from .mixing import mix_final_buses, render_dialogue_tracks
from .models import JobManifest, utcnow
from .timeline_collision_panel import (
    load_timeline_collision_resolutions,
    resolution_map as timeline_collision_resolution_map,
    upsert_timeline_collision_case,
)
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
    with_web_researched_organisation_pronunciations,
)
from .rendering import render_review_mp4
from .tts_ssml import BreakPart, SSMLBounds, SSMLPart, TextPart
from .transcript_lipsync import (
    TRANSCRIPT_LIPSYNC_POLICY_VERSION,
    TRANSCRIPT_LIPSYNC_SCHEMA_VERSION,
    ISLAND_GAP_MIN_MS,
    build_transcript_lipsync_plan,
)
from .voice_prosody import voice_and_base_prosody
from .speaker_mapping import build_speaker_id_mapping, canonicalize_voice_family_evidence
from .direct_voice_analysis import (
    ensure_canonical_speaker_reels,
    ensure_canonical_voice_family_analysis,
)
from .known_speakers import identify_known_speaker
from .errors import AIRateLimit, ClaudeRateLimit
from .direct_azure_persona import (
    DIRECT_AZURE_PERSONA_VERSION,
    apply_direct_azure_personas,
    collect_source_features,
)


DIRECT_DUB_SCHEMA_VERSION = (
    "mathula-direct-azure-dub-v57-coordinated-cluster-repair-v13.19.10"
)
DUB_MASTERING_OVERRIDE_SCHEMA_VERSION = "mathula-dub-mastering-overrides-v1"
FINALIZATION_CONDUCTOR_SCHEMA_VERSION = (
    "mathula-finalization-conductor-v4-controlled-overlap-v13.18.28"
)
LEGACY_FINALIZATION_CONDUCTOR_SCHEMA_VERSIONS = {
    "mathula-finalization-conductor-v1-v13.18.13",
    "mathula-finalization-conductor-v2-five-block-v13.18.18",
    "mathula-finalization-conductor-v3-declared-compression-v13.18.27",
}
FIVE_BLOCK_TIMELINE_SOLVER_VERSION = (
    "mathula-five-block-timeline-conductor-v2-word-anchored-v13.18.19"
)
FINAL_PRODUCT_CONDUCTOR_VERSION = (
    "mathula-final-product-conductor-v4-performance-first-v13.19.0"
)
SOURCE_WORD_ALIGNMENT_VERSION = (
    "mathula-source-word-speaker-alignment-v1-v13.18.19"
)
SOURCE_PAUSE_ALIGNMENT_VERSION = (
    "mathula-source-pause-alignment-v9-terminal-punctuation-cap-v13.19.12"
)
MINIMUM_SOURCE_PAUSE_MS = 350
SOURCE_PAUSE_BASELINE_MS = 100
MINIMUM_INSERTED_SOURCE_PAUSE_MS = 250
MAXIMUM_INSERTED_SOURCE_PAUSE_MS = 3000
# Azure adds its own substantial sentence-final breath. Keep explicit source
# silence below one second at terminal punctuation so the two pauses do not
# combine into a production-blocking stop.
MAXIMUM_TERMINAL_INSERTED_SOURCE_PAUSE_MS = 900
SOURCE_PAUSE_TRANSITION_ROOM_MS = 100
SEMANTIC_FIT_SCHEMA_VERSION = "mathula-semantic-fit-v2-v13.18.34"
SEMANTIC_FIT_VALIDATOR_VERSION = (
    "mathula-semantic-fit-validator-v6-natural-transition-fit-v13.18.40"
)
SEMANTIC_FIT_BATCH_STRATEGY_VERSION = (
    "mathula-semantic-fit-global-batch-v29-coordinated-cluster-v13.19.10"
)
SEMANTIC_PHRASE_GROUP_VERSION = "mathula-semantic-phrase-group-v1-v13.18.47"
SEMANTIC_PHRASE_GROUP_MAX_GAP_MS = 5000
PERFORMANCE_PLAN_VERSION = (
    "mathula-performance-plan-v15-breath-group-feedback-v13.19.2"
)
WORD_ANCHORED_ISLAND_CONTRACT_VERSION = (
    "mathula-word-anchored-island-contract-v10-coordinated-cluster-v13.19.10"
)
MAXIMUM_INTERNAL_ISLAND_DRIFT_MS = 500
SOURCE_TEMPO_CALIBRATION_VERSION = (
    "mathula-source-tempo-calibration-v1-v13.18.63"
)
SOURCE_TEMPO_CALIBRATION_MAX_SAMPLES = 3
SOURCE_TEMPO_CALIBRATION_MIN_ACTIVE_MS = 1200
SOURCE_TEMPO_CALIBRATION_MIN_SOURCE_SYLLABLES = 8
SOURCE_TEMPO_CALIBRATION_MIN_TARGET_SYLLABLES = 8
SOURCE_TEMPO_CALIBRATION_MIN_RATE_PERCENT = -20
SOURCE_TEMPO_CALIBRATION_MAX_RATE_PERCENT = 10
SOURCE_TEMPO_CALIBRATION_MIN_SHIFT_PERCENT = 2
# Production dialogue adaptation is deliberately bounded to two rounds:
# three broad candidates followed, only when needed, by three candidates that
# use the real Azure-measured breath-group drift as feedback.  Source tempo
# remains locked; the controller never enters an unbounded portfolio loop.
DIALOGUE_ADAPTOR_PREFLIGHT_VERSION = (
    "mathula-dialogue-adaptor-preflight-v4-coordinated-cluster-v13.19.10"
)
DIALOGUE_ADAPTOR_PORTFOLIO_SIZE = 3
DIALOGUE_ADAPTOR_TTS_SHORTLIST = 3
DIALOGUE_ADAPTOR_LIPSYNC_WEIGHT = 0.80
PERFORMANCE_PLAN_PROVIDER_CANDIDATES_PER_SHARD = 3
PERFORMANCE_REGION_MAX_GAP_MS = 5000
PERFORMANCE_REGION_MAX_DURATION_MS = 30000
PERFORMANCE_REGION_MAX_ISLANDS = 12
# A translated phrase may be longer than the source mouth-active span even at
# the correct speaker tempo.  The next source speech onset is the stronger
# cadence/lip-sync anchor: allow the phrase tail to use intervening source
# silence while preserving a small audible rest before the next onset.
SOURCE_ANCHORED_MIN_REST_MS = 150
SOURCE_ANCHORED_FINAL_MIN_REST_MS = 100
SOURCE_ANCHORED_MAX_PAUSE_BORROW_MS = 1500
# Keep phrase-only GPT calls bounded.  Large all-region repair calls were slow
# and made one difficult phrase hold every unresolved region hostage.
LOCALIZED_REPAIR_MAX_UNITS_PER_CALL = 12
LOCALIZED_REPAIR_MAX_GROUPS_PER_CALL = 36
SEMANTIC_FIT_MAX_SAME_SPEAKER_TRANSITION_BORROW_MS = 1500
SEMANTIC_FIT_PAUSE_CALIBRATION_GUARD_MS = 25
SEMANTIC_FIT_PAUSE_CALIBRATION_ATTEMPTS = 2
DIRECT_DUB_TOTAL_SOURCE_PAUSE_MAX_MS = 10_000
TIMING_MODE_BALANCED = "balanced"
TIMING_MODE_SEMANTIC_FIT = "semantic-fit"
TIMING_MODE_PERFORMANCE_PLAN = "performance-plan"
TIMING_MODES = (
    TIMING_MODE_BALANCED,
    TIMING_MODE_SEMANTIC_FIT,
    TIMING_MODE_PERFORMANCE_PLAN,
)
VOICE_RESOLUTION_SCHEMA_VERSION = "mathula-direct-voice-resolution-v4-cached"
LEGACY_VOICE_RESOLUTION_SCHEMA_VERSION = "mathula-direct-voice-resolution-v3"
APPROVED_SEGMENT_BOUNDARY_POLICY_VERSION = (
    "final-speakable-tail-drift-v1-v13.18.11"
)
ORDINARY_TRANSLATION_TAIL_DRIFT_MS = 250
FINAL_SPEAKABLE_TAIL_DRIFT_MS = 500
FINAL_SPEAKABLE_MAX_TRIM_RATIO = 0.10


class DirectDubError(RuntimeError):
    """Raised when the direct renderer cannot safely produce a complete dub."""


class TimelineCollisionReviewRequired(DirectDubError):
    """Raised after a collision review package has been written successfully."""

    def __init__(self, review: Mapping[str, Any]) -> None:
        self.review = dict(review)
        super().__init__(
            "Timeline collision review is required; open "
            f"{self.review.get('panel_path') or self.review.get('review_path')}"
        )

    def to_result(self) -> dict[str, Any]:
        return {
            "status": "timeline_collision_review_required",
            "job_id": self.review.get("job_id"),
            "review_path": self.review.get("review_path"),
            "panel_path": self.review.get("panel_path"),
            "resolutions_path": self.review.get("resolutions_path"),
            "open_case_count": self.review.get("open_case_count"),
            "resume_command": self.review.get("resume_command"),
        }


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
        plain_duration_ms: int | None = None,
        effective_inserted_pause_ms: int | None = None,
    ) -> None:
        super().__init__(message)
        self.measured_duration_ms = measured_duration_ms
        self.target_duration_ms = target_duration_ms
        self.stretch_ratio = stretch_ratio
        self.plain_duration_ms = plain_duration_ms
        self.effective_inserted_pause_ms = effective_inserted_pause_ms


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
    required_facts: tuple[str, ...] = ()
    optional_details: tuple[str, ...] = ()
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
    required_facts: tuple[str, ...] = ()
    optional_details: tuple[str, ...] = ()
    variants: tuple[PreparedTranslationVariant, ...] = ()
    selected_variant_id: str = "natural"
    # A mastering choice is a measured instruction, not a hint.  Keep it
    # separate from selected_variant_id so the ordinary predicted-fit sorter
    # cannot silently choose the pre-mastering variant again.
    mastering_variant_id: str | None = None
    # Audited gaps between source words. These survive renderer-local variant
    # selection and let Azure reproduce slow delivery without changing text.
    source_pause_anchors: tuple[Mapping[str, Any], ...] = ()
    source_speech_start_ms: int | None = None
    source_speech_end_ms: int | None = None
    # Additive transcript-first evidence. Defaults preserve construction and
    # deserialization performed by older callers and tests.
    source_speech_islands: tuple[Mapping[str, Any], ...] = ()
    source_interactions: tuple[Mapping[str, Any], ...] = ()
    source_timing_evidence: str = "block_boundary_fallback"
    # Semantic-fit may combine adjacent same-speaker blocks when a sentence
    # connector crosses their internal boundary. Empty values preserve every
    # legacy construction and balanced-mode behavior.
    semantic_group_member_ids: tuple[str, ...] = ()
    semantic_target_islands: tuple[str, ...] = ()
    performance_region_id: str | None = None
    performance_region_member_ids: tuple[str, ...] = ()

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "segment_ids": list(self.segment_ids),
            "protected_entities": list(self.protected_entities),
            "required_facts": list(self.required_facts),
            "optional_details": list(self.optional_details),
            "variants": [variant.to_dict() for variant in self.variants],
            "source_pause_anchors": [
                dict(anchor) for anchor in self.source_pause_anchors
            ],
            "source_speech_islands": [
                dict(island) for island in self.source_speech_islands
            ],
            "source_interactions": [
                dict(interaction) for interaction in self.source_interactions
            ],
            "semantic_group_member_ids": list(self.semantic_group_member_ids),
            "semantic_target_islands": list(self.semantic_target_islands),
            "performance_region_id": self.performance_region_id,
            "performance_region_member_ids": list(
                self.performance_region_member_ids
            ),
            "selected_variant_id": self.selected_variant_id,
            "duration_ms": self.duration_ms,
        }

    def with_variant(self, variant: PreparedTranslationVariant) -> "SpeechBlock":
        return replace(
            self,
            block_id=f"{_source_block_id(self.block_id)}_variant_{variant.variant_id}",
            translated_text=variant.spoken_text,
            tts_text=variant.tts_text,
            selected_variant_id=variant.variant_id,
        )


def _estimated_english_syllables(text: str) -> int:
    """Conservative orthographic syllable estimate for source-tempo calibration.

    This is not a linguistic transcript normalizer.  It is used only over
    several source turns to estimate the speaker's articulation tempo before
    Azure synthesis.  Numeric-heavy turns are excluded by the caller because
    written digits do not reveal how the source speaker verbalized them.
    """

    total = 0
    for raw_word in re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)?", str(text)):
        word = re.sub(r"[^a-z]", "", raw_word.casefold())
        if not word:
            continue
        groups = re.findall(r"[aeiouy]+", word)
        count = max(1, len(groups))
        if (
            count > 1
            and len(word) > 3
            and word.endswith("e")
            and not word.endswith(("le", "ye"))
        ):
            count -= 1
        total += max(1, count)
    return total


def _estimated_zulu_syllables(text: str) -> int:
    """Estimate audible isiZulu syllables from the actual TTS text.

    IsiZulu orthography is sufficiently vowel-transparent for vowel count to
    be a useful tempo proxy.  Non-vocalic tokens are given a one-syllable floor
    so names/initialisms do not disappear from the estimate.
    """

    total = 0
    for raw_word in re.findall(r"[A-Za-zÀ-ÿ0-9]+(?:[-’'][A-Za-zÀ-ÿ0-9]+)*", str(text)):
        letters = raw_word.casefold()
        vowels = sum(1 for character in letters if character in "aeiou")
        total += max(1, vowels)
    return total


def _dialogue_adaptor_ms_per_syllable(
    evidence: Sequence[Mapping[str, Any]],
    *,
    fallback_text: str,
    fallback_plain_duration_ms: int,
) -> float:
    """Estimate Azure milliseconds per spoken isiZulu syllable at locked tempo.

    The estimate is deliberately local to one speaker/voice/rate contract.  It
    is a cheap ranking heuristic only: Azure measurement remains authoritative.
    Using prior measured candidates makes the preflight self-calibrating without
    asking the LLM to predict TTS duration accurately.
    """

    observations: list[float] = []
    for value in evidence:
        if not isinstance(value, Mapping):
            continue
        text = str(value.get("tts_text") or value.get("spoken_text") or "").strip()
        try:
            plain_ms = int(
                value.get("plain_speech_duration_ms")
                or value.get("measured_duration_ms")
                or 0
            )
        except (TypeError, ValueError):
            plain_ms = 0
        syllables = _estimated_zulu_syllables(text) if text else 0
        if plain_ms > 0 and syllables >= 3:
            observations.append(plain_ms / syllables)
    fallback_syllables = _estimated_zulu_syllables(fallback_text)
    if fallback_plain_duration_ms > 0 and fallback_syllables >= 3:
        observations.append(fallback_plain_duration_ms / fallback_syllables)
    if observations:
        return max(25.0, min(1000.0, float(median(observations))))
    # Last-resort neutral proxy. It is used only for ordering alternatives in
    # the same portfolio; it can never approve timing without Azure synthesis.
    return 180.0


def _dialogue_adaptor_island_drift_ms(
    island_texts: Sequence[str],
    source_islands: Sequence[Mapping[str, Any]],
) -> int:
    """Predict cumulative lip-sync drift from target syllable distribution.

    This is intentionally a distribution score rather than isolated-island TTS.
    Isolated synthesis adds phrase-edge overhead and was responsible for noisy
    hard-gate failures. The score asks a cheaper question before synthesis:
    does the candidate place roughly the right amount of speech before each
    source mouth-active boundary?
    """

    if len(island_texts) < 2 or len(island_texts) != len(source_islands):
        return 0
    source_durations = [
        max(
            1,
            int(value.get("duration_ms") or 0)
            or int(value.get("end_ms") or 0) - int(value.get("start_ms") or 0),
        )
        for value in source_islands
    ]
    source_total = sum(source_durations)
    target_weights = [max(1, _estimated_zulu_syllables(text)) for text in island_texts]
    target_total = sum(target_weights)
    if source_total <= 0 or target_total <= 0:
        return 0
    source_cumulative = 0.0
    target_cumulative = 0.0
    maximum = 0.0
    for source_ms, weight in zip(
        source_durations[:-1],
        target_weights[:-1],
        strict=True,
    ):
        source_cumulative += source_ms
        target_cumulative += source_total * weight / target_total
        maximum = max(maximum, abs(target_cumulative - source_cumulative))
    return int(round(maximum))



def _partition_approved_text_across_source_islands(
    text: str,
    source_islands: Sequence[Mapping[str, Any]],
    *,
    protected_phrases: Sequence[str] = (),
) -> tuple[str, ...]:
    """Partition immutable approved wording across source breath groups.

    This is a semantic-safe production fallback: it may move *boundaries* but
    never changes, deletes, or invents a lexical token.  Dynamic programming
    balances estimated isiZulu syllable mass against the source mouth-active
    durations while preferring punctuation boundaries and refusing to split a
    protected multi-word entity (for example ``Madlanga Commission``).
    """

    cleaned = " ".join(str(text).split()).strip()
    islands = [dict(value) for value in source_islands if isinstance(value, Mapping)]
    if not cleaned or not islands:
        return ()
    if len(islands) == 1:
        return (cleaned,)

    tokens = cleaned.split()
    island_count = len(islands)
    if len(tokens) < island_count:
        return ()

    def lexical_token(value: str) -> str:
        return re.sub(r"[^0-9A-Za-zÀ-ÖØ-öø-ÿ'-]+", "", value).casefold()

    lexical = [lexical_token(value) for value in tokens]
    forbidden_splits: set[int] = set()
    for phrase in protected_phrases:
        phrase_tokens = [
            lexical_token(value)
            for value in str(phrase).split()
            if lexical_token(value)
        ]
        if len(phrase_tokens) < 2:
            continue
        width = len(phrase_tokens)
        def protected_token_matches(actual: str, expected: str) -> bool:
            # isiZulu commonly attaches a short grammatical prefix to an
            # English proper noun with a hyphen (e.g. e-Madlanga, u-Adams).
            # The lexical entity is still the protected English name.
            return actual == expected or actual.endswith("-" + expected)

        for start in range(0, len(lexical) - width + 1):
            if not all(
                protected_token_matches(actual, expected)
                for actual, expected in zip(
                    lexical[start : start + width],
                    phrase_tokens,
                    strict=True,
                )
            ):
                continue
            # A split position is the token index on the right-hand side of
            # the boundary.  Forbid every internal boundary in the entity.
            forbidden_splits.update(range(start + 1, start + width))

    token_syllables = [max(1, _estimated_zulu_syllables(value)) for value in tokens]
    cumulative = [0]
    for value in token_syllables:
        cumulative.append(cumulative[-1] + value)

    source_durations = [
        max(
            1,
            int(value.get("duration_ms") or 0)
            or int(value.get("end_ms") or 0) - int(value.get("start_ms") or 0),
        )
        for value in islands
    ]
    source_total = max(1, sum(source_durations))
    syllable_total = max(1, cumulative[-1])
    target_syllables = [
        syllable_total * duration / source_total for duration in source_durations
    ]

    token_count = len(tokens)
    infinity = float("inf")
    cost = [
        [infinity] * (token_count + 1)
        for _ in range(island_count + 1)
    ]
    previous: list[list[int | None]] = [
        [None] * (token_count + 1)
        for _ in range(island_count + 1)
    ]
    cost[0][0] = 0.0

    for island_index in range(1, island_count + 1):
        minimum_end = island_index
        maximum_end = token_count - (island_count - island_index)
        target = max(1.0, target_syllables[island_index - 1])
        for end in range(minimum_end, maximum_end + 1):
            for start in range(island_index - 1, end):
                if cost[island_index - 1][start] == infinity:
                    continue
                if island_index > 1 and start in forbidden_splits:
                    continue
                segment_syllables = cumulative[end] - cumulative[start]
                relative_error = (segment_syllables - target) / target
                candidate_cost = (
                    cost[island_index - 1][start]
                    + relative_error * relative_error
                )
                if island_index < island_count:
                    if end in forbidden_splits:
                        continue
                    # Natural punctuation is preferable but never required;
                    # this remains a boundary allocator, not a text editor.
                    if not re.search(r"[.!?;:,…]$", tokens[end - 1]):
                        candidate_cost += 0.06
                if candidate_cost < cost[island_index][end]:
                    cost[island_index][end] = candidate_cost
                    previous[island_index][end] = start

    end = token_count
    ranges: list[tuple[int, int]] = []
    for island_index in range(island_count, 0, -1):
        start = previous[island_index][end]
        if start is None:
            return ()
        ranges.append((start, end))
        end = start
    ranges.reverse()
    parts = tuple(" ".join(tokens[start:end]) for start, end in ranges)
    if " ".join(" ".join(parts).split()) != cleaned:
        raise AssertionError("approved breath-group partition changed lexical text")
    return parts


def _dialogue_adaptor_preflight_score(
    *,
    predicted_plain_ms: int,
    preferred_plain_ms: int,
    minimum_plain_ms: int,
    maximum_plain_ms: int,
    predicted_island_drift_ms: int,
) -> float:
    """Rank linguistic alternatives without changing the locked speaker tempo."""

    duration_error = abs(predicted_plain_ms - preferred_plain_ms)
    outside = max(
        0,
        minimum_plain_ms - predicted_plain_ms,
        predicted_plain_ms - maximum_plain_ms,
    )
    return float(
        duration_error
        + outside * 2
        + predicted_island_drift_ms * DIALOGUE_ADAPTOR_LIPSYNC_WEIGHT
    )


def _source_active_speech_metrics(block: SpeechBlock) -> dict[str, float | int] | None:
    """Return word-timed source articulation evidence excluding long islands gaps."""

    if block.source_timing_evidence != "azure_word_timestamps":
        return None
    timed_islands = [
        value
        for value in block.source_speech_islands
        if isinstance(value, Mapping)
        and value.get("word_ids")
        and int(value.get("end_ms") or 0) > int(value.get("start_ms") or 0)
    ]
    if not timed_islands:
        return None
    active_ms = sum(
        int(value.get("end_ms") or 0) - int(value.get("start_ms") or 0)
        for value in timed_islands
    )
    source_text = " ".join(
        str(value.get("text") or "").strip() for value in timed_islands
    ).strip()
    source_word_count = sum(
        len(value.get("word_ids") or ()) for value in timed_islands
    )
    if active_ms < SOURCE_TEMPO_CALIBRATION_MIN_ACTIVE_MS:
        return None
    # Digits are intentionally excluded from the syllable estimate because the
    # written form does not tell us whether the source speaker said e.g.
    # "thirty-one", "three one", or an acronym-like reading.
    if re.search(r"\d", source_text):
        return None
    source_syllables = _estimated_english_syllables(source_text)
    if source_syllables < SOURCE_TEMPO_CALIBRATION_MIN_SOURCE_SYLLABLES:
        return None
    return {
        "active_ms": active_ms,
        "source_word_count": source_word_count,
        "source_syllables": source_syllables,
        "source_syllables_per_second": round(
            source_syllables / max(0.001, active_ms / 1000.0),
            6,
        ),
        "source_words_per_second": round(
            source_word_count / max(0.001, active_ms / 1000.0),
            6,
        ),
    }


def _source_block_id(block_id: str) -> str:
    return str(block_id).split("_variant_", 1)[0]


def _ordered_unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value) for value in values if str(value)))


def _semantic_phrase_group_connector(text: str) -> str | None:
    """Return a dangling English discourse connector, if one is present."""

    match = re.search(
        r"\b(but|and|because|so|although|though|however|that)\s*[,.!?…]*\s*$",
        str(text),
        flags=re.IGNORECASE,
    )
    return match.group(1).casefold() if match else None


def _semantic_group_pause_anchors(
    islands: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Build exact pause anchors between absolute word-timed speech islands."""

    ordered = sorted(
        (dict(value) for value in islands),
        key=lambda value: (int(value.get("start_ms") or 0), int(value.get("end_ms") or 0)),
    )
    word_counts = [max(1, len(str(value.get("text") or "").split())) for value in ordered]
    total_words = max(1, sum(word_counts))
    consumed_words = 0
    anchors: list[dict[str, Any]] = []
    for left, right, left_words in zip(
        ordered,
        ordered[1:],
        word_counts,
        strict=False,
    ):
        consumed_words += left_words
        raw_gap_ms = max(
            0,
            int(right.get("start_ms") or 0) - int(left.get("end_ms") or 0),
        )
        requested_pause_ms = min(
            MAXIMUM_INSERTED_SOURCE_PAUSE_MS,
            max(0, raw_gap_ms - SOURCE_PAUSE_BASELINE_MS),
        )
        if requested_pause_ms < MINIMUM_INSERTED_SOURCE_PAUSE_MS:
            continue
        anchors.append(
            {
                "source_progress": consumed_words / total_words,
                "requested_pause_ms": requested_pause_ms,
                "source_gap_ms": raw_gap_ms,
                "pause_kind": "island_boundary",
                "left_island_id": left.get("island_id"),
                "right_island_id": right.get("island_id"),
            }
        )
    return tuple(anchors)


def build_semantic_phrase_groups(
    blocks: Sequence[SpeechBlock],
) -> tuple[list[SpeechBlock], list[dict[str, Any]]]:
    """Join only safe same-speaker sentence continuations for semantic-fit.

    This changes the translation overlay, never the approved translation. A
    group stops at speaker changes, interactions, non-word-timed blocks and
    gaps larger than the bounded source pause window.
    """

    grouped: list[SpeechBlock] = []
    audits: list[dict[str, Any]] = []
    index = 0
    while index < len(blocks):
        first = blocks[index]
        if index + 1 >= len(blocks):
            grouped.append(first)
            break
        second = blocks[index + 1]
        connector = _semantic_phrase_group_connector(first.source_text)
        first_end = int(first.source_speech_end_ms or first.end_ms)
        second_start = int(second.source_speech_start_ms or second.start_ms)
        gap_ms = max(0, second_start - first_end)
        safe = bool(
            connector
            and first.speaker_id == second.speaker_id
            and first.source_timing_evidence == "azure_word_timestamps"
            and second.source_timing_evidence == "azure_word_timestamps"
            and not first.source_interactions
            and not second.source_interactions
            and 0 <= gap_ms <= SEMANTIC_PHRASE_GROUP_MAX_GAP_MS
            and first.source_speech_islands
            and second.source_speech_islands
        )
        if not safe:
            grouped.append(first)
            index += 1
            continue

        member_ids = (
            *(
                first.semantic_group_member_ids
                or (_source_block_id(first.block_id),)
            ),
            *(
                second.semantic_group_member_ids
                or (_source_block_id(second.block_id),)
            ),
        )
        islands = tuple(
            sorted(
                (
                    *(dict(value) for value in first.source_speech_islands),
                    *(dict(value) for value in second.source_speech_islands),
                ),
                key=lambda value: int(value.get("start_ms") or 0),
            )
        )
        variants: list[PreparedTranslationVariant] = []
        first_variants = {value.variant_id: value for value in first.variants}
        second_variants = {value.variant_id: value for value in second.variants}
        for variant_id in first_variants:
            if variant_id not in second_variants:
                continue
            left = first_variants[variant_id]
            right = second_variants[variant_id]
            variants.append(
                PreparedTranslationVariant(
                    variant_id=variant_id,
                    spoken_text=" ".join((left.spoken_text, right.spoken_text)).strip(),
                    tts_text=" ".join((left.tts_text, right.tts_text)).strip(),
                    target_duration_ms=(
                        (left.target_duration_ms or first.duration_ms)
                        + (right.target_duration_ms or second.duration_ms)
                    ),
                    estimated_duration_ms=(
                        (left.estimated_duration_ms or 0)
                        + (right.estimated_duration_ms or 0)
                    ) or None,
                    meaning_preserved=(left.meaning_preserved and right.meaning_preserved),
                    omitted_optional_details=_ordered_unique(
                        (*left.omitted_optional_details, *right.omitted_optional_details)
                    ),
                )
            )
        group_id = "phrase_group_" + "_".join(member_ids)
        combined = SpeechBlock(
            block_id=group_id,
            speaker_id=first.speaker_id,
            start_ms=first.start_ms,
            end_ms=second.end_ms,
            segment_ids=_ordered_unique((*first.segment_ids, *second.segment_ids)),
            source_text=" ".join((first.source_text, second.source_text)).strip(),
            translated_text=" ".join(
                (first.translated_text, second.translated_text)
            ).strip(),
            tts_text=" ".join((first.tts_text, second.tts_text)).strip(),
            protected_entities=_ordered_unique(
                (*first.protected_entities, *second.protected_entities)
            ),
            required_facts=_ordered_unique(
                (*first.required_facts, *second.required_facts)
            ),
            optional_details=_ordered_unique(
                (*first.optional_details, *second.optional_details)
            ),
            variants=tuple(variants),
            selected_variant_id="natural",
            source_pause_anchors=_semantic_group_pause_anchors(islands),
            source_speech_start_ms=int(first.source_speech_start_ms or first.start_ms),
            source_speech_end_ms=int(second.source_speech_end_ms or second.end_ms),
            source_speech_islands=islands,
            source_interactions=(),
            source_timing_evidence="azure_word_timestamps",
            semantic_group_member_ids=tuple(member_ids),
        )
        grouped.append(combined)
        audits.append(
            {
                "schema_version": SEMANTIC_PHRASE_GROUP_VERSION,
                "group_id": group_id,
                "member_block_ids": list(member_ids),
                "connector": connector,
                "speaker_id": first.speaker_id,
                "inter_block_gap_ms": gap_ms,
                "source_island_count": len(islands),
                "source_pause_count": len(combined.source_pause_anchors),
                "cross_speaker_boundary": False,
                "approved_translation_changed": False,
            }
        )
        index += 2
    return grouped, audits


def _combine_performance_region(
    members: Sequence[SpeechBlock],
    region_number: int,
) -> tuple[SpeechBlock, dict[str, Any]]:
    first = members[0]
    last = members[-1]
    member_ids = tuple(_source_block_id(value.block_id) for value in members)
    islands = tuple(
        sorted(
            (
                dict(island)
                for member in members
                for island in member.source_speech_islands
            ),
            key=lambda value: int(value.get("start_ms") or 0),
        )
    )
    region_id = f"performance_region_{region_number:04d}"
    region = SpeechBlock(
        block_id=region_id,
        speaker_id=first.speaker_id,
        start_ms=first.start_ms,
        end_ms=last.end_ms,
        segment_ids=_ordered_unique(
            segment_id for member in members for segment_id in member.segment_ids
        ),
        source_text=" ".join(value.source_text for value in members).strip(),
        translated_text=" ".join(
            value.translated_text for value in members
        ).strip(),
        tts_text=" ".join(value.tts_text for value in members).strip(),
        protected_entities=_ordered_unique(
            value for member in members for value in member.protected_entities
        ),
        required_facts=_ordered_unique(
            value for member in members for value in member.required_facts
        ),
        optional_details=_ordered_unique(
            value for member in members for value in member.optional_details
        ),
        variants=(),
        selected_variant_id="natural",
        source_pause_anchors=_semantic_group_pause_anchors(islands),
        source_speech_start_ms=int(first.source_speech_start_ms or first.start_ms),
        source_speech_end_ms=int(last.source_speech_end_ms or last.end_ms),
        source_speech_islands=islands,
        source_interactions=(),
        source_timing_evidence="azure_word_timestamps",
        performance_region_id=region_id,
        performance_region_member_ids=member_ids,
    )
    lane = {
        "speaker_id": first.speaker_id,
        "start_ms": region.source_speech_start_ms,
        "end_ms": region.source_speech_end_ms,
        "interaction_safe": True,
    }
    return region, {
        "schema_version": PERFORMANCE_PLAN_VERSION,
        "region_id": region_id,
        "member_block_ids": list(member_ids),
        "speaker_id": first.speaker_id,
        "start_ms": region.start_ms,
        "end_ms": region.end_ms,
        "source_speech_start_ms": region.source_speech_start_ms,
        "source_speech_end_ms": region.source_speech_end_ms,
        "source_island_count": len(islands),
        "source_pause_count": len(region.source_pause_anchors),
        "lane": lane,
        "hard_boundaries": {
            "speaker_changes": True,
            "overlapping_interactions": True,
            "long_silence_ms": PERFORMANCE_REGION_MAX_GAP_MS,
        },
        "stt_blocks_are_storage_members_only": True,
        "approved_translation_changed": False,
    }


def build_performance_regions(
    blocks: Sequence[SpeechBlock],
) -> tuple[list[SpeechBlock], list[dict[str, Any]]]:
    """Build word-timed semantic regions; STT blocks are not boundaries."""

    regions: list[SpeechBlock] = []
    audits: list[dict[str, Any]] = []
    index = 0
    region_number = 1
    while index < len(blocks):
        first = blocks[index]
        eligible_first = bool(
            first.source_timing_evidence == "azure_word_timestamps"
            and first.source_speech_islands
            and not first.source_interactions
        )
        if not eligible_first:
            regions.append(first)
            audits.append(
                {
                    "schema_version": PERFORMANCE_PLAN_VERSION,
                    "region_id": None,
                    "member_block_ids": [_source_block_id(first.block_id)],
                    "speaker_id": first.speaker_id,
                    "mode": "compatibility_block",
                    "reason": "word_timing_or_interaction_boundary",
                    "approved_translation_changed": False,
                }
            )
            index += 1
            continue

        members = [first]
        island_count = len(first.source_speech_islands)
        cursor = index + 1
        while cursor < len(blocks):
            candidate = blocks[cursor]
            previous = members[-1]
            previous_end = int(previous.source_speech_end_ms or previous.end_ms)
            candidate_start = int(
                candidate.source_speech_start_ms or candidate.start_ms
            )
            gap_ms = candidate_start - previous_end
            combined_duration_ms = candidate.end_ms - first.start_ms
            candidate_islands = len(candidate.source_speech_islands)
            safe = bool(
                candidate.speaker_id == first.speaker_id
                and candidate.source_timing_evidence == "azure_word_timestamps"
                and candidate.source_speech_islands
                and not candidate.source_interactions
                and 0 <= gap_ms <= PERFORMANCE_REGION_MAX_GAP_MS
                and combined_duration_ms <= PERFORMANCE_REGION_MAX_DURATION_MS
                and island_count + candidate_islands
                <= PERFORMANCE_REGION_MAX_ISLANDS
            )
            if not safe:
                break
            members.append(candidate)
            island_count += candidate_islands
            cursor += 1

        region, audit = _combine_performance_region(members, region_number)
        regions.append(region)
        audits.append(audit)
        region_number += 1
        index = cursor
    return regions, audits


def attach_word_anchored_island_contracts(
    blocks: Sequence[SpeechBlock],
) -> tuple[list[SpeechBlock], list[dict[str, Any]]]:
    """Require semantic ownership for every long-pause source island.

    Semantic-fit historically transferred pauses by proportional target-word
    progress.  A faithful contraction could therefore put words belonging to
    source island two after the pause for island three.  Whole-block endpoints
    still passed while the dialogue visibly drifted.  Multi-island, word-timed
    blocks now receive the same semantic-beat contract as performance-plan,
    without merging STT blocks or changing single-island/legacy behavior.
    """

    adjusted: list[SpeechBlock] = []
    audits: list[dict[str, Any]] = []
    contract_number = 1
    for block in blocks:
        eligible = bool(
            block.source_timing_evidence == "azure_word_timestamps"
            and len(block.source_speech_islands) >= 2
            and not block.source_interactions
        )
        if not eligible:
            adjusted.append(block)
            continue
        contract_id = f"word_anchor_region_{contract_number:04d}"
        member_ids = (
            block.semantic_group_member_ids
            or (_source_block_id(block.block_id),)
        )
        adjusted.append(
            replace(
                block,
                performance_region_id=contract_id,
                performance_region_member_ids=tuple(member_ids),
            )
        )
        audits.append(
            {
                "schema_version": WORD_ANCHORED_ISLAND_CONTRACT_VERSION,
                "region_id": contract_id,
                "mode": "semantic_fit_word_anchored_islands",
                "member_block_ids": list(member_ids),
                "speaker_id": block.speaker_id,
                "start_ms": block.start_ms,
                "end_ms": block.end_ms,
                "source_speech_start_ms": block.source_speech_start_ms,
                "source_speech_end_ms": block.source_speech_end_ms,
                "source_island_count": len(block.source_speech_islands),
                "source_pause_count": len(block.source_pause_anchors),
                "stt_blocks_are_storage_members_only": False,
                "approved_translation_changed": False,
                "speaker_tempo_locked": True,
                "maximum_internal_drift_ms": MAXIMUM_INTERNAL_ISLAND_DRIFT_MS,
            }
        )
        contract_number += 1
    return adjusted, audits


def _prepared_variant(block: SpeechBlock, variant_id: str) -> PreparedTranslationVariant:
    for variant in block.variants:
        if variant.variant_id == variant_id:
            return variant
    raise DirectDubError(
        f"Timeline collision resolution requested unavailable {variant_id} "
        f"variant for {_source_block_id(block.block_id)}"
    )


def apply_timeline_collision_resolutions(
    blocks: Sequence[SpeechBlock],
    resolutions: Mapping[str, Mapping[str, Any]],
) -> tuple[list[SpeechBlock], list[dict[str, Any]]]:
    """Apply reviewed direct-dub overlays without rewriting translation JSON."""

    adjusted = list(blocks)
    audits: list[dict[str, Any]] = []
    index_by_id = {
        _source_block_id(block.block_id): index for index, block in enumerate(adjusted)
    }
    for raw_collision_id, raw_resolution in resolutions.items():
        collision_id = _source_block_id(raw_collision_id)
        block_index = index_by_id.get(collision_id)
        if block_index is None:
            continue
        resolution = dict(raw_resolution)
        strategy = str(resolution.get("strategy") or "")
        affected: list[dict[str, Any]] = []

        if strategy == "manual_text":
            text = " ".join(str(resolution.get("manual_text") or "").split()).strip()
            if not text:
                raise DirectDubError(
                    f"Timeline collision manual text is blank for {collision_id}"
                )
            block = adjusted[block_index]
            manual = PreparedTranslationVariant(
                variant_id="manual",
                spoken_text=text,
                tts_text=text,
                target_duration_ms=block.duration_ms,
                estimated_duration_ms=None,
                meaning_preserved=True,
            )
            adjusted[block_index] = replace(
                block,
                translated_text=text,
                tts_text=text,
                variants=(manual,),
                selected_variant_id="manual",
            )
            affected.append(
                {
                    "role": "current",
                    "block_id": collision_id,
                    "selected_variant_id": "manual",
                    "spoken_text": text,
                }
            )
        elif strategy == "select_variants":
            variants = resolution.get("variants")
            variants = dict(variants) if isinstance(variants, Mapping) else {}
            for role, offset in (("previous", -1), ("current", 0), ("following", 1)):
                variant_id = str(variants.get(role) or "").strip()
                target_index = block_index + offset
                if not variant_id or target_index < 0 or target_index >= len(adjusted):
                    continue
                block = adjusted[target_index]
                variant = _prepared_variant(block, variant_id)
                adjusted[target_index] = replace(
                    block,
                    translated_text=variant.spoken_text,
                    tts_text=variant.tts_text,
                    variants=(variant,),
                    selected_variant_id=variant.variant_id,
                )
                affected.append(
                    {
                        "role": role,
                        "block_id": _source_block_id(block.block_id),
                        "selected_variant_id": variant.variant_id,
                        "spoken_text": variant.spoken_text,
                    }
                )
        elif strategy == "allow_overlap":
            affected.append(
                {
                    "role": "triplet",
                    "block_id": collision_id,
                    "overlap_before_ms": int(
                        resolution.get("overlap_before_ms") or 0
                    ),
                    "overlap_after_ms": int(
                        resolution.get("overlap_after_ms") or 0
                    ),
                }
            )
        else:
            raise DirectDubError(
                f"Unknown timeline collision resolution strategy {strategy!r} "
                f"for {collision_id}"
            )

        audits.append(
            {
                "schema_version": "mathula.timeline-collision-resolution-audit.v1",
                "collision_id": collision_id,
                "strategy": strategy,
                "reviewed_by": resolution.get("reviewed_by"),
                "reviewed_at": resolution.get("reviewed_at"),
                "authoritative_translation_changed": False,
                "affected": affected,
            }
        )
    return adjusted, audits


def load_dub_mastering_overrides(path: Path) -> dict[str, Any]:
    """Load application-generated mastering choices without trusting free text."""

    if not path.is_file():
        return {
            "schema_version": DUB_MASTERING_OVERRIDE_SCHEMA_VERSION,
            "revision": 0,
            "overrides": [],
        }
    payload = read_json(path)
    if not isinstance(payload, Mapping):
        raise DirectDubError("Dub mastering override artifact is not an object")
    if payload.get("schema_version") != DUB_MASTERING_OVERRIDE_SCHEMA_VERSION:
        raise DirectDubError("Unsupported dub mastering override schema")
    if not isinstance(payload.get("overrides"), list):
        raise DirectDubError("Dub mastering overrides must be a list")
    return dict(payload)


def apply_dub_mastering_overrides(
    blocks: Sequence[SpeechBlock],
    payload: Mapping[str, Any],
) -> tuple[list[SpeechBlock], list[dict[str, Any]]]:
    """Apply STT-audited prepared-variant choices before paid synthesis.

    The mastering loop may select only an already approved, meaning-preserving
    variant. It cannot inject wording, change speaker identity, move source
    boundaries, or modify the authoritative translation artifact.
    """

    adjusted = list(blocks)
    index_by_id = {
        _source_block_id(block.block_id): index for index, block in enumerate(adjusted)
    }
    audits: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in payload.get("overrides") or []:
        if not isinstance(raw, Mapping):
            raise DirectDubError("Dub mastering override is not an object")
        block_id = _source_block_id(str(raw.get("block_id") or "").strip())
        if not block_id or block_id in seen:
            raise DirectDubError("Dub mastering overrides require unique block IDs")
        seen.add(block_id)
        block_index = index_by_id.get(block_id)
        if block_index is None:
            # A new translation may legitimately remove an old mastered block.
            continue
        strategy = str(raw.get("strategy") or "")
        if strategy != "select_variant":
            raise DirectDubError(
                f"Unsupported dub mastering strategy {strategy!r} for {block_id}"
            )
        variant_id = str(raw.get("variant_id") or "").strip()
        block = adjusted[block_index]
        variant = _prepared_variant(block, variant_id)
        if not variant.meaning_preserved:
            raise DirectDubError(
                f"Dub mastering requested unapproved variant {variant_id} for {block_id}"
            )
        # Retain the complete approved variant set. A later measured round must
        # be able to reverse a regression or move one step in the other
        # direction without asking the translation model for new wording.
        adjusted[block_index] = replace(
            block,
            mastering_variant_id=variant.variant_id,
        ).with_variant(variant)
        audits.append(
            {
                "schema_version": "mathula-dub-mastering-override-audit-v1",
                "block_id": block_id,
                "strategy": strategy,
                "selected_variant_id": variant_id,
                "reason_codes": [
                    str(value)
                    for value in raw.get("reason_codes") or []
                    if str(value)
                ],
                "source_round": int(raw.get("source_round") or 0),
                "authoritative_translation_changed": False,
                "speaker_changed": False,
                "timeline_changed": False,
            }
        )
    return adjusted, audits


def _collision_triplet(
    blocks: Sequence[SpeechBlock],
    block_index: int,
    prepared_variant_audits: Mapping[int, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for role, index in (
        ("previous", block_index - 1),
        ("current", block_index),
        ("following", block_index + 1),
    ):
        if index < 0 or index >= len(blocks):
            continue
        values.append(
            {
                "role": role,
                "block_index": index,
                "block": blocks[index].to_dict(),
                "prepared_variant_attempts": [
                    dict(item)
                    for item in prepared_variant_audits.get(index, ())
                    if isinstance(item, Mapping)
                ],
            }
        )
    return values


def _collision_five_block_window(
    blocks: Sequence[SpeechBlock],
    block_index: int,
    prepared_variant_audits: Mapping[int, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    """Return the two preceding and two following turns for conductor audit."""

    roles = (
        "previous_2",
        "previous_1",
        "current",
        "following_1",
        "following_2",
    )
    values: list[dict[str, Any]] = []
    for role, index in zip(roles, range(block_index - 2, block_index + 3)):
        if index < 0 or index >= len(blocks):
            continue
        values.append(
            {
                "role": role,
                "block_index": index,
                "block": blocks[index].to_dict(),
                "prepared_variant_attempts": [
                    dict(item)
                    for item in prepared_variant_audits.get(index, ())
                    if isinstance(item, Mapping)
                ],
            }
        )
    return values


def _collision_allowed_strategies(
    blocks: Sequence[SpeechBlock],
    block_index: int,
) -> list[str]:
    """Expose only review strategies the renderer can apply at this boundary."""

    strategies = ["manual_text", "select_variants"]
    if 0 < block_index < len(blocks) - 1:
        strategies.insert(0, "allow_overlap")
    return strategies


def _bounded_boundary_time_stretch_plan(
    blocks: Sequence[SpeechBlock],
    block_index: int,
    overflow: TimingOverflowError,
    *,
    maximum_ratio: float,
) -> dict[str, Any] | None:
    """Approve a small measured fit only at a source-timeline boundary."""

    if not blocks or block_index < 0 or block_index >= len(blocks):
        return None
    if block_index not in {0, len(blocks) - 1}:
        return None
    required_ratio = float(overflow.stretch_ratio)
    if not 1.0 < required_ratio <= maximum_ratio:
        return None
    if len(blocks) == 1:
        boundary = "only_block"
    elif block_index == 0:
        boundary = "start"
    else:
        boundary = "end"
    return {
        "method": "bounded_boundary_time_stretch",
        "solver_version": "bounded-boundary-time-stretch-v1",
        "boundary": boundary,
        "measured_duration_ms": overflow.measured_duration_ms,
        "target_duration_ms": overflow.target_duration_ms,
        "required_time_stretch_ratio": round(required_ratio, 6),
        "maximum_time_stretch_ratio": maximum_ratio,
        "authoritative_translation_changed": False,
        "speech_truncated": False,
        "speaker_overlap_added": False,
        "human_review_required": False,
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

    @property
    def timeline_collision_review(self) -> Path:
        return self.root / "timeline_collision_review.json"

    @property
    def timeline_collision_resolutions(self) -> Path:
        return self.root / "timeline_collision_resolutions.json"

    @property
    def timeline_collision_panel(self) -> Path:
        return self.root / "timeline_collision_panel.html"

    @property
    def mastering_overrides(self) -> Path:
        return self.root / "mastering_overrides.json"

    @property
    def finalization_conductor(self) -> Path:
        return self.root / "finalization_conductor.json"

    @property
    def source_word_alignment(self) -> Path:
        return self.root / "source_word_alignment.json"

    @property
    def transcript_lipsync(self) -> Path:
        return self.root / "transcript_lipsync.json"

    @property
    def semantic_fit(self) -> Path:
        return self.root / "semantic_fit.json"

    @property
    def source_tempo_calibration(self) -> Path:
        return self.root / "source_tempo_calibration.json"

    @property
    def performance_plan(self) -> Path:
        return self.root / "performance_plan.json"


@dataclass(frozen=True)
class DirectDubOptions:
    timing_mode: str = TIMING_MODE_BALANCED
    semantic_fit_max_attempts: int = 12
    semantic_fit_tolerance_ms: int = 250
    same_speaker_gap_ms: int = 1200
    max_azure_rate_percent: int = 10
    max_time_stretch_ratio: float = 1.03
    max_time_expansion_ratio: float = 1.40
    collision_max_time_stretch_ratio: float = 1.03
    boundary_max_time_stretch_ratio: float = 1.03
    final_product_max_time_stretch_ratio: float = 1.03
    max_cross_speaker_boundary_shift_ms: int = 500
    source_word_alignment_tolerance_ms: int = 200
    finalization_ai_rounds: int = 2
    max_interjection_overlap_ms: int = 2000
    max_interjection_boundary_overlap_ms: int = 1000
    max_interjection_source_gap_ms: int = 250
    male_voice: str = "zu-ZA-ThembaNeural"
    female_voice: str = "zu-ZA-ThandoNeural"
    min_voice_family_confidence: float = 0.70
    include_background: bool = False
    dialogue_target_lufs: float = -16.0
    dialogue_true_peak_dbfs: float = -1.5
    dialogue_loudness_range_lu: float = 11.0
    clean_master_video_mode: str = "smooth_cfr"

    def validate(self, backend: AzureTTSBackend) -> None:
        if self.timing_mode not in TIMING_MODES:
            raise ValueError(
                f"timing mode must be one of {', '.join(TIMING_MODES)}"
            )
        if not 1 <= self.semantic_fit_max_attempts <= 30:
            raise ValueError("semantic-fit attempts must be between 1 and 30")
        if not 0 <= self.semantic_fit_tolerance_ms <= 1000:
            raise ValueError(
                "semantic-fit tolerance must be between 0 and 1000ms"
            )
        if self.same_speaker_gap_ms < 0:
            raise ValueError("same-speaker gap must be non-negative")
        if self.clean_master_video_mode not in {"smooth_cfr", "stream_copy"}:
            raise ValueError(
                "clean-master video mode must be smooth_cfr or stream_copy"
            )
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
        if not self.max_time_stretch_ratio <= self.boundary_max_time_stretch_ratio <= 1.15:
            raise ValueError(
                "boundary time-stretch ratio must be between the normal limit and 1.15"
            )
        if not self.boundary_max_time_stretch_ratio <= self.final_product_max_time_stretch_ratio <= 4.0:
            raise ValueError(
                "final-product time-stretch ratio must be between the boundary "
                "limit and 4.0"
            )
        if not 0 <= self.max_cross_speaker_boundary_shift_ms <= 1000:
            raise ValueError(
                "cross-speaker boundary shift must be between 0 and 1000ms"
            )
        if not 0 <= self.source_word_alignment_tolerance_ms <= 1000:
            raise ValueError(
                "source-word alignment tolerance must be between 0 and 1000ms"
            )
        if not 0 <= self.finalization_ai_rounds <= 3:
            raise ValueError("finalization AI rounds must be between 0 and 3")
        if not 0 <= self.max_interjection_overlap_ms <= 2000:
            raise ValueError(
                "interjection overlap must be between 0 and 2000ms"
            )
        if not 0 <= self.max_interjection_boundary_overlap_ms <= 1000:
            raise ValueError(
                "interjection boundary overlap must be between 0 and 1000ms"
            )
        if (
            self.max_interjection_boundary_overlap_ms
            > self.max_interjection_overlap_ms
        ):
            raise ValueError(
                "interjection boundary overlap cannot exceed total overlap"
            )
        if not 0 <= self.max_interjection_source_gap_ms <= 1000:
            raise ValueError(
                "interjection source gap must be between 0 and 1000ms"
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
        total_pause_max_ms=DIRECT_DUB_TOTAL_SOURCE_PAUSE_MAX_MS,
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


def rebalance_five_block_timeline(
    blocks: Sequence[SpeechBlock],
    required_windows_ms: Mapping[int, int],
    overflow_indices: Iterable[int],
) -> tuple[list[SpeechBlock], list[dict[str, Any]]]:
    """Resolve an overflow by redistributing measured slack across five turns.

    The old triplet planner could only borrow from the immediately adjacent
    turns.  A compact neighbour with no spare time therefore hid usable slack
    in the second preceding/following turn.  This solver keeps the outer
    five-turn boundaries fixed, preserves speech order and every selected text,
    consumes existing source gaps first, and then borrows only Azure-measured
    surplus above each neighbour's required window.
    """

    adjusted = list(blocks)
    audits: list[dict[str, Any]] = []
    for block_index in sorted({int(value) for value in overflow_indices}):
        if block_index < 2 or block_index + 2 >= len(adjusted):
            audits.append(
                {
                    "method": "five_block_measured_timeline_fit",
                    "solver_version": FIVE_BLOCK_TIMELINE_SOLVER_VERSION,
                    "block_index": block_index,
                    "applied": False,
                    "reason": "five_block_window_unavailable_at_timeline_edge",
                }
            )
            continue

        indices = list(range(block_index - 2, block_index + 3))
        window = [adjusted[index] for index in indices]
        current_offset = 2
        required = [
            max(1, int(required_windows_ms.get(index, block.duration_ms)))
            for index, block in zip(indices, window)
        ]
        deficit_ms = required[current_offset] - window[current_offset].duration_ms
        base_audit: dict[str, Any] = {
            "method": "five_block_measured_timeline_fit",
            "solver_version": FIVE_BLOCK_TIMELINE_SOLVER_VERSION,
            "block_index": block_index,
            "block_id": window[current_offset].block_id,
            "affected_block_indices": indices,
            "window_block_ids": [block.block_id for block in window],
            "outer_start_ms": window[0].start_ms,
            "outer_end_ms": window[-1].end_ms,
            "required_windows_ms": {
                block.block_id: value for block, value in zip(window, required)
            },
            "original_windows_ms": {
                block.block_id: block.duration_ms for block in window
            },
            "deficit_ms": max(0, deficit_ms),
            "azure_measured": True,
            "text_immutable": True,
            "speaker_order_preserved": True,
        }
        if deficit_ms <= 0:
            audits.append({**base_audit, "applied": False, "reason": "already_fits"})
            continue

        original_durations = [block.duration_ms for block in window]
        durations = list(original_durations)
        gaps = [
            max(0, window[offset + 1].start_ms - window[offset].end_ms)
            for offset in range(4)
        ]
        original_gaps = list(gaps)
        # Resolve adjacent measured overflows in the same allocation. Growing
        # only the current turn made the solver reject its own candidate when
        # another failed turn occupied one of the four neighbour slots.
        deficits_by_offset = [
            max(0, required_ms - duration_ms)
            for required_ms, duration_ms in zip(required, original_durations)
        ]
        for offset, required_ms in enumerate(required):
            durations[offset] = max(durations[offset], required_ms)
        total_deficit_ms = sum(deficits_by_offset)
        remaining_ms = total_deficit_ms
        reductions: list[dict[str, Any]] = []

        # Silence is lossless capacity. Prefer gaps nearest the failed turn.
        for gap_offset in (1, 2, 0, 3):
            take_ms = min(remaining_ms, gaps[gap_offset])
            if take_ms:
                gaps[gap_offset] -= take_ms
                remaining_ms -= take_ms
                reductions.append(
                    {
                        "kind": "source_gap",
                        "between": [
                            window[gap_offset].block_id,
                            window[gap_offset + 1].block_id,
                        ],
                        "released_ms": take_ms,
                    }
                )
            if remaining_ms <= 0:
                break

        # Then use measured delivery slack, one turn away before two turns away.
        if remaining_ms > 0:
            for offset in (1, 3, 0, 4):
                surplus_ms = max(0, durations[offset] - required[offset])
                take_ms = min(remaining_ms, surplus_ms)
                if take_ms:
                    durations[offset] -= take_ms
                    remaining_ms -= take_ms
                    reductions.append(
                        {
                            "kind": "azure_measured_neighbour_surplus",
                            "block_id": window[offset].block_id,
                            "released_ms": take_ms,
                        }
                    )
                if remaining_ms <= 0:
                    break

        if remaining_ms > 0:
            audits.append(
                {
                    **base_audit,
                    "applied": False,
                    "reason": "insufficient_five_block_measured_capacity",
                    "capacity_released_ms": total_deficit_ms - remaining_ms,
                    "remaining_shortfall_ms": remaining_ms,
                    "neighbouring_deficits_ms": {
                        window[offset].block_id: value
                        for offset, value in enumerate(deficits_by_offset)
                        if value
                    },
                    "reductions_considered": reductions,
                }
            )
            continue

        cursor_ms = window[0].start_ms
        candidates: list[SpeechBlock] = []
        for offset, block in enumerate(window):
            start_ms = cursor_ms
            end_ms = start_ms + durations[offset]
            candidates.append(replace(block, start_ms=start_ms, end_ms=end_ms))
            cursor_ms = end_ms
            if offset < len(gaps):
                cursor_ms += gaps[offset]

        invariant_failures: list[str] = []
        if cursor_ms != window[-1].end_ms:
            invariant_failures.append("outer_five_block_boundary_changed")
        for offset, candidate in enumerate(candidates):
            if candidate.duration_ms < required[offset]:
                invariant_failures.append(
                    f"{candidate.block_id}_below_measured_required_window"
                )
            if offset and candidates[offset - 1].end_ms > candidate.start_ms:
                invariant_failures.append("speech_order_overlap")
        if invariant_failures:
            audits.append(
                {
                    **base_audit,
                    "applied": False,
                    "reason": "five_block_candidate_rejected",
                    "invariant_failures": invariant_failures,
                }
            )
            continue

        for index, candidate in zip(indices, candidates):
            adjusted[index] = candidate
        audits.append(
            {
                **base_audit,
                "applied": True,
                "reason": "five_block_measured_capacity_resolved_overflow",
                "capacity_released_ms": total_deficit_ms,
                "neighbouring_deficits_ms": {
                    window[offset].block_id: value
                    for offset, value in enumerate(deficits_by_offset)
                    if value
                },
                "source_silence_consumed_ms": sum(original_gaps) - sum(gaps),
                "reductions": reductions,
                "adjusted_windows_ms": {
                    candidate.block_id: candidate.duration_ms
                    for candidate in candidates
                },
                "adjusted_boundaries_ms": {
                    candidate.block_id: [candidate.start_ms, candidate.end_ms]
                    for candidate in candidates
                },
                "human_review_required": False,
                "downstream_overflow_prevented": True,
            }
        )

    return adjusted, audits


def _source_word_time_ms(
    word: Mapping[str, Any],
    field: str,
) -> int | None:
    """Read either millisecond or second word timing without guessing units."""

    millisecond_field = f"{field}_ms"
    if word.get(millisecond_field) is not None:
        try:
            return int(round(float(word[millisecond_field])))
        except (TypeError, ValueError):
            return None
    if word.get(field) is not None:
        try:
            return int(round(float(word[field]) * 1000.0))
        except (TypeError, ValueError):
            return None
    return None


def _canonical_source_words(
    transcript: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return timed source words under the transcript's canonical speakers."""

    canonical_speaker_by_word_id: dict[str, str] = {}
    for segment in transcript.get("segments") or []:
        if not isinstance(segment, Mapping):
            continue
        speaker_id = str(
            segment.get("speaker_id") or segment.get("speaker") or ""
        ).strip()
        if not speaker_id:
            continue
        for raw_word_id in segment.get("source_word_ids") or []:
            word_id = str(raw_word_id or "").strip()
            if word_id:
                canonical_speaker_by_word_id[word_id] = speaker_id

    words: list[dict[str, Any]] = []
    for raw_word in transcript.get("words") or []:
        if not isinstance(raw_word, Mapping):
            continue
        word_id = str(raw_word.get("word_id") or "").strip()
        speaker_id = canonical_speaker_by_word_id.get(word_id)
        if not speaker_id:
            raw_speaker = str(
                raw_word.get("speaker_id") or raw_word.get("speaker") or ""
            ).strip()
            if raw_speaker.startswith("SPEAKER_"):
                speaker_id = raw_speaker
        start_ms = _source_word_time_ms(raw_word, "start")
        end_ms = _source_word_time_ms(raw_word, "end")
        if (
            not speaker_id
            or start_ms is None
            or end_ms is None
            or end_ms < start_ms
        ):
            continue
        words.append(
            {
                "word_id": word_id or None,
                "speaker_id": speaker_id,
                "text": str(raw_word.get("text") or "").strip(),
                "start_ms": start_ms,
                "end_ms": end_ms,
            }
        )
    return sorted(
        words,
        key=lambda value: (
            int(value["start_ms"]),
            int(value["end_ms"]),
            str(value.get("word_id") or ""),
        ),
    )


def attach_source_pause_anchors(
    blocks: Sequence[SpeechBlock],
    transcript: Mapping[str, Any],
    *,
    minimum_source_pause_ms: int = MINIMUM_SOURCE_PAUSE_MS,
) -> tuple[list[SpeechBlock], list[dict[str, Any]]]:
    """Attach significant within-turn source pauses to canonical dub blocks.

    A pause is evidence only when it lies between two timed words attributed to
    the same canonical speaker inside the block. The original text and timing
    remain immutable; this stage records where bounded SSML silence may later
    replace otherwise trailing room.
    """

    minimum_source_pause_ms = max(1, int(minimum_source_pause_ms))
    words = _canonical_source_words(transcript)
    adjusted: list[SpeechBlock] = []
    artifact: list[dict[str, Any]] = []
    for block in blocks:
        block_words = [
            word
            for word in words
            if word["speaker_id"] == block.speaker_id
            and (int(word["start_ms"]) + int(word["end_ms"])) / 2.0
            >= block.start_ms
            and (int(word["start_ms"]) + int(word["end_ms"])) / 2.0
            <= block.end_ms
        ]
        anchors: list[dict[str, Any]] = []
        word_count = len(block_words)
        for word_index, (left, right) in enumerate(
            zip(block_words, block_words[1:]),
            start=1,
        ):
            gap_ms = int(right["start_ms"]) - int(left["end_ms"])
            if gap_ms < minimum_source_pause_ms:
                continue
            requested_pause_ms = min(
                MAXIMUM_INSERTED_SOURCE_PAUSE_MS,
                max(0, gap_ms - SOURCE_PAUSE_BASELINE_MS),
            )
            if requested_pause_ms < MINIMUM_INSERTED_SOURCE_PAUSE_MS:
                continue
            anchor = {
                "alignment_version": SOURCE_PAUSE_ALIGNMENT_VERSION,
                "source_gap_ms": gap_ms,
                "requested_pause_ms": requested_pause_ms,
                "pause_kind": (
                    "island_boundary"
                    if gap_ms >= ISLAND_GAP_MIN_MS
                    else "phrase_pause"
                ),
                "source_progress": round(word_index / word_count, 9),
                "left_word": dict(left),
                "right_word": dict(right),
            }
            anchors.append(anchor)
        adjusted.append(
            replace(
                block,
                source_pause_anchors=tuple(anchors),
                source_speech_start_ms=(
                    max(block.start_ms, int(block_words[0]["start_ms"]))
                    if block_words else None
                ),
                source_speech_end_ms=(
                    min(block.end_ms, int(block_words[-1]["end_ms"]))
                    if block_words else None
                ),
            )
        )
        if anchors:
            artifact.append(
                {
                    "block_id": block.block_id,
                    "speaker_id": block.speaker_id,
                    "block_start_ms": block.start_ms,
                    "block_end_ms": block.end_ms,
                    "source_word_count": word_count,
                    "source_speech_start_ms": max(
                        block.start_ms, int(block_words[0]["start_ms"])
                    ),
                    "source_speech_end_ms": min(
                        block.end_ms, int(block_words[-1]["end_ms"])
                    ),
                    "anchors": anchors,
                }
            )
    return adjusted, artifact


def attach_transcript_lipsync_plan(
    blocks: Sequence[SpeechBlock],
    transcript: Mapping[str, Any],
) -> tuple[list[SpeechBlock], dict[str, Any]]:
    """Attach transcript-derived islands/interactions without changing blocks.

    This adapter is intentionally additive. Old transcripts with no usable
    Azure words receive block-boundary fallback islands and otherwise retain
    the exact legacy timing, text, and speaker behavior.
    """

    plan = build_transcript_lipsync_plan(
        blocks,
        _canonical_source_words(transcript),
    )
    plan_blocks = {
        str(value.get("block_id") or ""): value
        for value in plan.get("blocks") or []
        if isinstance(value, Mapping)
    }
    adjusted: list[SpeechBlock] = []
    for block in blocks:
        evidence = plan_blocks.get(_source_block_id(block.block_id), {})
        adjusted.append(
            replace(
                block,
                source_speech_islands=tuple(
                    {
                        **dict(value),
                        "start_ms": max(
                            block.start_ms, int(value.get("start_ms") or block.start_ms)
                        ),
                        "end_ms": min(
                            block.end_ms, int(value.get("end_ms") or block.end_ms)
                        ),
                        "timing_clamped_to_block_boundary": bool(
                            int(value.get("start_ms") or block.start_ms) < block.start_ms
                            or int(value.get("end_ms") or block.end_ms) > block.end_ms
                        ),
                    }
                    for value in evidence.get("speech_islands") or []
                    if isinstance(value, Mapping)
                    and min(
                        block.end_ms, int(value.get("end_ms") or block.end_ms)
                    )
                    > max(
                        block.start_ms, int(value.get("start_ms") or block.start_ms)
                    )
                ),
                source_interactions=tuple(
                    dict(value)
                    for value in evidence.get("interaction_roles") or []
                    if isinstance(value, Mapping)
                ),
                source_timing_evidence=str(
                    evidence.get("timing_evidence")
                    or "block_boundary_fallback"
                ),
            )
        )
    return adjusted, plan


def _protected_entity_boundary_indices(
    text: str,
    protected_entities: Sequence[str],
) -> set[int]:
    """Return word boundaries that would split a protected entity surface."""

    word_matches = list(re.finditer(r"\S+", str(text), re.UNICODE))
    text_tokens = [_normalised_words(match.group(0)) for match in word_matches]
    flattened = [tokens[0] if len(tokens) == 1 else "".join(tokens) for tokens in text_tokens]
    forbidden: set[int] = set()
    for entity in protected_entities:
        entity_tokens = _normalised_words(str(entity))
        if len(entity_tokens) < 2:
            continue
        for start in range(len(flattened) - len(entity_tokens) + 1):
            matched = True
            for offset, expected in enumerate(entity_tokens):
                actual = flattened[start + offset]
                if actual == expected:
                    continue
                if (
                    offset == 0
                    and actual.endswith(expected)
                    and len(actual) - len(expected) <= 4
                ):
                    continue
                matched = False
                break
            if matched:
                forbidden.update(
                    range(start + 1, start + len(entity_tokens))
                )
    return forbidden


def build_source_pause_ssml_parts(
    text: str,
    source_pause_anchors: Sequence[Mapping[str, Any]],
    *,
    pause_budget_ms: int,
    language: str = "zu-ZA",
    fill_pause_budget: bool = False,
    pause_max_ms: int = 2000,
    prioritize_island_boundaries: bool = False,
    protected_entities: Sequence[str] = (),
) -> tuple[tuple[SSMLPart, ...], dict[str, Any]]:
    """Insert bounded source-aligned breaks without changing approved text."""

    value = str(text)
    word_matches = list(re.finditer(r"\S+", value, re.UNICODE))
    pause_budget_ms = max(0, int(pause_budget_ms))
    pause_max_ms = max(1, int(pause_max_ms))
    empty_audit = {
        "alignment_version": SOURCE_PAUSE_ALIGNMENT_VERSION,
        "allocation_policy": (
            "island_first_capacity_aware"
            if prioritize_island_boundaries
            else "legacy_proportional"
        ),
        "applied": False,
        "reason": "no_eligible_source_pause",
        "pause_budget_ms": pause_budget_ms,
        "total_added_pause_ms": 0,
        "maximum_added_pause_ms": 0,
        "insertions": [],
    }
    if len(word_matches) < 2 or pause_budget_ms < MINIMUM_INSERTED_SOURCE_PAUSE_MS:
        return (), empty_audit
    protected_boundaries = _protected_entity_boundary_indices(
        value,
        protected_entities,
    )

    candidates: list[dict[str, Any]] = []
    for raw_anchor in source_pause_anchors:
        try:
            progress = float(raw_anchor.get("source_progress"))
            requested_ms = int(raw_anchor.get("requested_pause_ms") or 0)
        except (TypeError, ValueError):
            continue
        if not 0.0 < progress < 1.0 or requested_ms <= 0:
            continue
        boundary_word_index = max(
            1,
            min(len(word_matches) - 1, round(progress * len(word_matches))),
        )
        punctuation_boundaries = [
            index
            for index in range(1, len(word_matches))
            if word_matches[index - 1].group(0).endswith(
                (",", ";", ":", ".", "!", "?", "…")
            )
        ]
        nearby_punctuation = [
            index
            for index in punctuation_boundaries
            if abs(index - boundary_word_index) <= 1
        ]
        if nearby_punctuation:
            boundary_word_index = min(
                nearby_punctuation,
                key=lambda index: (abs(index - boundary_word_index), index),
            )
        original_boundary_word_index = boundary_word_index
        if boundary_word_index in protected_boundaries:
            safe_boundaries = [
                index
                for index in range(1, len(word_matches))
                if index not in protected_boundaries
            ]
            if not safe_boundaries:
                continue
            boundary_word_index = min(
                safe_boundaries,
                key=lambda index: (
                    abs(index - original_boundary_word_index),
                    0 if index in punctuation_boundaries else 1,
                    index,
                ),
            )
        candidates.append(
            {
                "target_boundary_word_index": boundary_word_index,
                "original_target_boundary_word_index": (
                    original_boundary_word_index
                ),
                "protected_entity_boundary_adjusted": (
                    boundary_word_index != original_boundary_word_index
                ),
                "requested_pause_ms": min(
                    MAXIMUM_INSERTED_SOURCE_PAUSE_MS,
                    requested_ms,
                ),
                "source_gap_ms": int(raw_anchor.get("source_gap_ms") or 0),
                "source_progress": progress,
                "pause_kind": str(
                    raw_anchor.get("pause_kind") or "phrase_pause"
                ),
                "left_word": raw_anchor.get("left_word"),
                "right_word": raw_anchor.get("right_word"),
            }
        )
    if not candidates:
        return (), empty_audit

    # Several source gaps can land at the same translated word boundary. Merge
    # those before budgeting so we never emit adjacent duplicate break tags.
    merged: dict[int, dict[str, Any]] = {}
    for candidate in candidates:
        boundary = int(candidate["target_boundary_word_index"])
        item = merged.setdefault(
            boundary,
            {
                "target_boundary_word_index": boundary,
                "requested_pause_ms": 0,
                "pause_kind": "phrase_pause",
                "source_gaps": [],
            },
        )
        item["requested_pause_ms"] = min(
            MAXIMUM_INSERTED_SOURCE_PAUSE_MS,
            int(item["requested_pause_ms"])
            + int(candidate["requested_pause_ms"]),
        )
        if candidate.get("pause_kind") == "island_boundary":
            item["pause_kind"] = "island_boundary"
        item["source_gaps"].append(candidate)

    maximum_slot_count = pause_budget_ms // MINIMUM_INSERTED_SOURCE_PAUSE_MS
    eligible = sorted(
        merged.values(),
        key=lambda item: (
            0
            if prioritize_island_boundaries
            and item.get("pause_kind") == "island_boundary"
            else 1,
            -int(item["requested_pause_ms"]),
            int(item["target_boundary_word_index"]),
        ),
    )[:maximum_slot_count]

    def allocate_group(
        items: Sequence[dict[str, Any]],
        budget_ms: int,
        *,
        allow_expansion: bool,
    ) -> dict[int, int]:
        """Allocate one pause tier exactly, dropping imperceptible slots."""

        active = list(items)
        while active and budget_ms >= MINIMUM_INSERTED_SOURCE_PAUSE_MS:
            requested_total = sum(
                int(item["requested_pause_ms"]) for item in active
            )
            capacity = sum(MAXIMUM_INSERTED_SOURCE_PAUSE_MS for _ in active)
            target = min(
                budget_ms,
                capacity,
                capacity if allow_expansion else requested_total,
            )
            if target < MINIMUM_INSERTED_SOURCE_PAUSE_MS:
                return {}
            exact = [
                int(item["requested_pause_ms"]) * target / requested_total
                for item in active
            ]
            planned = [
                min(MAXIMUM_INSERTED_SOURCE_PAUSE_MS, int(value))
                for value in exact
            ]
            retained = [
                item
                for item, duration_ms in zip(active, planned, strict=True)
                if duration_ms >= MINIMUM_INSERTED_SOURCE_PAUSE_MS
            ]
            if len(retained) != len(active):
                active = retained
                continue
            remainder = target - sum(planned)
            while remainder > 0:
                progressed = False
                for index in sorted(
                    range(len(active)),
                    key=lambda value: (
                        exact[value] - planned[value],
                        -int(active[value]["target_boundary_word_index"]),
                    ),
                    reverse=True,
                ):
                    if planned[index] >= MAXIMUM_INSERTED_SOURCE_PAUSE_MS:
                        continue
                    planned[index] += 1
                    remainder -= 1
                    progressed = True
                    if remainder <= 0:
                        break
                if not progressed:
                    break
            return {
                int(item["target_boundary_word_index"]): duration_ms
                for item, duration_ms in zip(active, planned, strict=True)
            }
        return {}

    planned_by_boundary: dict[int, int] = {}
    requested_total = sum(int(item["requested_pause_ms"]) for item in eligible)
    if not prioritize_island_boundaries:
        # Preserve the legacy proportional planner byte-for-byte for jobs that
        # do not have authoritative word timing.
        while eligible:
            requested_total = sum(
                int(item["requested_pause_ms"]) for item in eligible
            )
            scale = (
                pause_budget_ms / requested_total
                if fill_pause_budget
                else min(1.0, pause_budget_ms / requested_total)
            )
            planned = [
                min(
                    MAXIMUM_INSERTED_SOURCE_PAUSE_MS,
                    int(round(int(item["requested_pause_ms"]) * scale)),
                )
                for item in eligible
            ]
            retained = [
                (item, duration_ms)
                for item, duration_ms in zip(eligible, planned, strict=True)
                if duration_ms >= MINIMUM_INSERTED_SOURCE_PAUSE_MS
            ]
            if len(retained) == len(eligible):
                planned_by_boundary = {
                    int(item["target_boundary_word_index"]): duration_ms
                    for item, duration_ms in retained
                }
                break
            eligible = [item for item, _duration_ms in retained]
    elif pause_budget_ms < requested_total:
        island_items = [
            item
            for item in eligible
            if item.get("pause_kind") == "island_boundary"
        ]
        phrase_items = [item for item in eligible if item not in island_items]
        island_budget_ms = min(
            pause_budget_ms,
            sum(int(item["requested_pause_ms"]) for item in island_items),
        )
        planned_by_boundary.update(
            allocate_group(
                island_items,
                island_budget_ms,
                allow_expansion=False,
            )
        )
        used_ms = sum(planned_by_boundary.values())
        planned_by_boundary.update(
            allocate_group(
                phrase_items,
                max(0, pause_budget_ms - used_ms),
                allow_expansion=False,
            )
        )
    else:
        planned_by_boundary = allocate_group(
            eligible,
            pause_budget_ms,
            allow_expansion=fill_pause_budget,
        )
    eligible = [
        item
        for item in eligible
        if int(item["target_boundary_word_index"]) in planned_by_boundary
    ]
    if not eligible:
        return (), {
            **empty_audit,
            "reason": "source_pauses_exceed_available_underfill",
        }

    insertions: list[dict[str, Any]] = []
    for item in eligible:
        duration_ms = int(
            planned_by_boundary[int(item["target_boundary_word_index"])]
        )
        if duration_ms < MINIMUM_INSERTED_SOURCE_PAUSE_MS:
            continue
        boundary = int(item["target_boundary_word_index"])
        target_left_text = word_matches[boundary - 1].group(0)
        terminal_boundary = bool(
            re.search(r"[.!?…][\"'”’\)\]}]*$", target_left_text)
        )
        uncapped_duration_ms = duration_ms
        if terminal_boundary and not prioritize_island_boundaries:
            duration_ms = min(
                duration_ms,
                MAXIMUM_TERMINAL_INSERTED_SOURCE_PAUSE_MS,
            )
        insertions.append(
            {
                **item,
                "duration_ms": duration_ms,
                "uncapped_duration_ms": uncapped_duration_ms,
                "terminal_punctuation_cap_applied": (
                    duration_ms != uncapped_duration_ms
                ),
                "target_left_text": target_left_text,
                "target_right_text": word_matches[boundary].group(0),
                "target_character_offset": word_matches[boundary].start(),
            }
        )
    if not insertions:
        return (), empty_audit
    excess_ms = sum(int(item["duration_ms"]) for item in insertions) - pause_budget_ms
    for insertion in sorted(
        insertions,
        key=lambda item: int(item["duration_ms"]),
        reverse=True,
    ):
        if excess_ms <= 0:
            break
        reducible_ms = max(
            0,
            int(insertion["duration_ms"]) - MINIMUM_INSERTED_SOURCE_PAUSE_MS,
        )
        reduction_ms = min(excess_ms, reducible_ms)
        insertion["duration_ms"] = int(insertion["duration_ms"]) - reduction_ms
        excess_ms -= reduction_ms

    parts: list[SSMLPart] = []
    cursor = 0
    for insertion in sorted(
        insertions,
        key=lambda item: int(item["target_character_offset"]),
    ):
        split_at = int(insertion["target_character_offset"])
        chunk = value[cursor:split_at]
        if chunk:
            initialism_parts = build_initialism_ssml_parts(
                chunk,
                language=language,
            )
            parts.extend(initialism_parts or (TextPart(chunk),))
        remaining_pause_ms = int(insertion["duration_ms"])
        break_durations_ms: list[int] = []
        while remaining_pause_ms:
            duration_ms = min(pause_max_ms, remaining_pause_ms)
            parts.append(BreakPart(duration_ms))
            break_durations_ms.append(duration_ms)
            remaining_pause_ms -= duration_ms
        insertion["ssml_break_durations_ms"] = break_durations_ms
        cursor = split_at
    tail = value[cursor:]
    if tail:
        initialism_parts = build_initialism_ssml_parts(tail, language=language)
        parts.extend(initialism_parts or (TextPart(tail),))

    total_added_pause_ms = sum(int(item["duration_ms"]) for item in insertions)
    return tuple(parts), {
        "alignment_version": SOURCE_PAUSE_ALIGNMENT_VERSION,
        "allocation_policy": (
            "island_first_capacity_aware"
            if prioritize_island_boundaries
            else "legacy_proportional"
        ),
        "applied": True,
        "reason": "source_word_gaps_transferred_to_ssml",
        "pause_budget_ms": pause_budget_ms,
        "total_added_pause_ms": total_added_pause_ms,
        "requested_pause_total_ms": sum(
            int(item["requested_pause_ms"]) for item in insertions
        ),
        "island_pause_total_ms": sum(
            int(item["duration_ms"])
            for item in insertions
            if item.get("pause_kind") == "island_boundary"
        ),
        "phrase_pause_total_ms": sum(
            int(item["duration_ms"])
            for item in insertions
            if item.get("pause_kind") != "island_boundary"
        ),
        "maximum_added_pause_ms": max(
            int(item["duration_ms"]) for item in insertions
        ),
        "maximum_ssml_break_ms": max(
            (
                int(duration_ms)
                for item in insertions
                for duration_ms in item.get("ssml_break_durations_ms") or []
            ),
            default=0,
        ),
        "insertions": insertions,
        "approved_text_changed": False,
        "filled_available_pause_budget": fill_pause_budget,
    }


def build_semantic_island_ssml_parts(
    island_texts: Sequence[str],
    source_pause_anchors: Sequence[Mapping[str, Any]],
    *,
    pause_budget_ms: int,
    language: str = "zu-ZA",
    pause_max_ms: int = 2000,
) -> tuple[tuple[SSMLPart, ...], dict[str, Any]]:
    """Place reviewed target phrases on exact source-speech islands."""

    phrases = [str(value).strip() for value in island_texts if str(value).strip()]
    all_anchors = [dict(value) for value in source_pause_anchors]
    # Source pause evidence also contains short rhetorical/phrase pauses.
    # Those live inside a speech island and must never be mistaken for the
    # boundaries between the GPT-owned semantic deliveries.  Missing kinds
    # remain compatible with performance plans produced before pause kinds
    # were persisted.
    anchors = [
        value
        for value in all_anchors
        if str(value.get("pause_kind") or "island_boundary")
        == "island_boundary"
    ]
    empty = {
        "alignment_version": SOURCE_PAUSE_ALIGNMENT_VERSION,
        "allocation_policy": "semantic_island_exact",
        "applied": False,
        "reason": "semantic_island_contract_incomplete",
        "pause_budget_ms": max(0, int(pause_budget_ms)),
        "total_added_pause_ms": 0,
        "insertions": [],
        "source_pause_anchor_count": len(all_anchors),
        "source_island_boundary_count": len(anchors),
        "ignored_intra_island_pause_count": len(all_anchors) - len(anchors),
    }
    if len(phrases) < 2 or len(anchors) != len(phrases) - 1:
        return (), empty
    requested = [
        min(
            MAXIMUM_INSERTED_SOURCE_PAUSE_MS,
            max(0, int(value.get("requested_pause_ms") or 0)),
        )
        for value in anchors
    ]
    if any(value < MINIMUM_INSERTED_SOURCE_PAUSE_MS for value in requested):
        return (), empty
    pause_budget_ms = max(0, int(pause_budget_ms))
    requested_total = sum(requested)
    if pause_budget_ms < requested_total:
        return (), {**empty, "reason": "semantic_island_pause_budget_too_small"}

    parts: list[SSMLPart] = []
    insertions: list[dict[str, Any]] = []
    for index, phrase in enumerate(phrases):
        phrase_parts = build_initialism_ssml_parts(phrase, language=language)
        parts.extend(phrase_parts or (TextPart(phrase),))
        if index >= len(requested):
            continue
        remaining_ms = requested[index]
        split_breaks: list[int] = []
        while remaining_ms:
            duration_ms = min(max(1, int(pause_max_ms)), remaining_ms)
            parts.append(BreakPart(duration_ms))
            split_breaks.append(duration_ms)
            remaining_ms -= duration_ms
        insertions.append(
            {
                **anchors[index],
                "duration_ms": requested[index],
                "target_island_index": index,
                "target_left_text": phrase,
                "target_right_text": phrases[index + 1],
                "ssml_break_durations_ms": split_breaks,
            }
        )
    return tuple(parts), {
        "alignment_version": SOURCE_PAUSE_ALIGNMENT_VERSION,
        "allocation_policy": "semantic_island_exact",
        "applied": True,
        "reason": "reviewed_target_phrases_mapped_to_source_islands",
        "pause_budget_ms": pause_budget_ms,
        "total_added_pause_ms": requested_total,
        "requested_pause_total_ms": requested_total,
        "island_pause_total_ms": requested_total,
        "phrase_pause_total_ms": 0,
        "maximum_added_pause_ms": max(requested),
        "maximum_ssml_break_ms": max(
            duration for item in insertions for duration in item["ssml_break_durations_ms"]
        ),
        "insertions": insertions,
        "approved_text_changed": False,
        "filled_available_pause_budget": False,
        "target_island_count": len(phrases),
        "source_pause_anchor_count": len(all_anchors),
        "source_island_boundary_count": len(anchors),
        "ignored_intra_island_pause_count": len(all_anchors) - len(anchors),
    }


def calibrate_source_pause_ssml_parts(
    parts: Sequence[SSMLPart],
    *,
    reduction_ms: int,
    minimum_break_durations_ms: Sequence[int] | None = None,
) -> tuple[tuple[SSMLPart, ...], tuple[int, ...]]:
    """Reduce pauses after measurement without crossing protected floors."""

    original = [
        int(part.duration_ms) for part in parts if isinstance(part, BreakPart)
    ]
    if minimum_break_durations_ms is None:
        minimums = [0 for _value in original]
    else:
        minimums = [max(0, int(value)) for value in minimum_break_durations_ms]
        if len(minimums) != len(original):
            raise ValueError("pause calibration floors must match SSML breaks")
        minimums = [
            min(duration_ms, minimum_ms)
            for duration_ms, minimum_ms in zip(original, minimums, strict=True)
        ]
    reducible = [
        duration_ms - minimum_ms
        for duration_ms, minimum_ms in zip(original, minimums, strict=True)
    ]
    reducible_total = sum(reducible)
    reduction_ms = min(reducible_total, max(0, int(reduction_ms)))
    if not original or reduction_ms <= 0:
        return tuple(parts), tuple(original)
    retained_reducible_total = reducible_total - reduction_ms
    exact = [
        value * retained_reducible_total / reducible_total
        for value in reducible
    ]
    retained = [int(value) for value in exact]
    remainder = retained_reducible_total - sum(retained)
    for index in sorted(
        range(len(exact)),
        key=lambda item: (exact[item] - retained[item], -item),
        reverse=True,
    )[:remainder]:
        retained[index] += 1
    calibrated = [
        minimum_ms + retained_ms
        for minimum_ms, retained_ms in zip(minimums, retained, strict=True)
    ]

    rebuilt: list[SSMLPart] = []
    break_index = 0
    for part in parts:
        if not isinstance(part, BreakPart):
            rebuilt.append(part)
            continue
        duration_ms = calibrated[break_index]
        break_index += 1
        if duration_ms:
            rebuilt.append(BreakPart(duration_ms))
    return tuple(rebuilt), tuple(calibrated)


def source_pause_calibration_floors(
    audit: Mapping[str, Any],
) -> tuple[int, ...]:
    """Map protected island durations to the flattened SSML break sequence."""

    floors: list[int] = []
    insertions = sorted(
        (dict(item) for item in audit.get("insertions") or []),
        key=lambda item: int(item.get("target_character_offset") or 0),
    )
    for insertion in insertions:
        chunks = [
            max(0, int(value))
            for value in insertion.get("ssml_break_durations_ms") or []
        ]
        protected_total_ms = 0
        if insertion.get("pause_kind") == "island_boundary":
            protected_total_ms = min(
                sum(chunks),
                max(0, int(insertion.get("requested_pause_ms") or 0)),
            )
        elif insertion.get("kind") == "source_leading_pause":
            protected_total_ms = sum(chunks)
        remaining_ms = protected_total_ms
        for duration_ms in chunks:
            floor_ms = min(duration_ms, remaining_ms)
            floors.append(floor_ms)
            remaining_ms -= floor_ms
    return tuple(floors)


def calibrated_source_pause_audit(
    audit: Mapping[str, Any],
    calibrated_durations: Sequence[int],
    *,
    reduction_ms: int,
    calibration_round: int,
) -> dict[str, Any]:
    """Keep the source-pause audit synchronized with calibrated SSML parts."""

    insertions = [dict(item) for item in audit.get("insertions") or []]
    ordered_indices = sorted(
        range(len(insertions)),
        key=lambda index: (
            int(insertions[index].get("target_character_offset") or 0),
            index,
        ),
    )
    duration_cursor = 0
    for index in ordered_indices:
        insertion = insertions[index]
        original_chunks = insertion.get("ssml_break_durations_ms") or [
            insertion.get("duration_ms") or 0
        ]
        chunk_count = len(original_chunks)
        chunk_durations = [
            int(value)
            for value in calibrated_durations[
                duration_cursor : duration_cursor + chunk_count
            ]
            if int(value) > 0
        ]
        duration_cursor += chunk_count
        insertion["ssml_break_durations_ms"] = chunk_durations
        insertion["duration_ms"] = sum(chunk_durations)
    total_added_pause_ms = sum(int(value) for value in calibrated_durations)
    island_pause_total_ms = sum(
        int(value.get("duration_ms") or 0)
        for value in insertions
        if value.get("pause_kind") == "island_boundary"
    )
    phrase_pause_total_ms = sum(
        int(value.get("duration_ms") or 0)
        for value in insertions
        if value.get("pause_kind") != "island_boundary"
        and value.get("kind") != "source_leading_pause"
    )
    return {
        **dict(audit),
        "insertions": insertions,
        "total_added_pause_ms": total_added_pause_ms,
        "island_pause_total_ms": island_pause_total_ms,
        "phrase_pause_total_ms": phrase_pause_total_ms,
        "maximum_added_pause_ms": max(
            (int(value.get("duration_ms") or 0) for value in insertions),
            default=0,
        ),
        "maximum_ssml_break_ms": max(calibrated_durations, default=0),
        "azure_measured_pause_calibration": True,
        "original_total_added_pause_ms": int(
            audit.get("original_total_added_pause_ms")
            or audit.get("total_added_pause_ms")
            or 0
        ),
        "calibration_reduction_ms": int(
            audit.get("calibration_reduction_ms") or 0
        )
        + int(reduction_ms),
        "calibration_round": int(calibration_round),
    }


def build_source_word_speaker_anchors(
    blocks: Sequence[SpeechBlock],
    transcript: Mapping[str, Any],
    *,
    tolerance_ms: int = 250,
) -> list[dict[str, Any]]:
    """Build audited word-level envelopes for cross-speaker handoffs.

    Azure words retain raw diarization labels, while transcript segments carry
    the canonical speaker IDs used by the dub.  The segment word-ID lists are
    therefore the authority used to rebind each word before constructing an
    anchor.  Same-speaker boundaries are intentionally omitted: the timing
    conductor may redistribute those freely without creating visible lip-sync
    attribution errors.
    """

    words = _canonical_source_words(transcript)

    anchors: list[dict[str, Any]] = []
    tolerance_ms = max(0, int(tolerance_ms))
    for boundary_index in range(len(blocks) - 1):
        left = blocks[boundary_index]
        right = blocks[boundary_index + 1]
        if left.speaker_id == right.speaker_id:
            continue

        left_words = [
            word
            for word in words
            if word["speaker_id"] == left.speaker_id
            and (word["start_ms"] + word["end_ms"]) / 2.0 >= left.start_ms
            and (word["start_ms"] + word["end_ms"]) / 2.0 <= left.end_ms
        ]
        right_words = [
            word
            for word in words
            if word["speaker_id"] == right.speaker_id
            and (word["start_ms"] + word["end_ms"]) / 2.0 >= right.start_ms
            and (word["start_ms"] + word["end_ms"]) / 2.0 <= right.end_ms
        ]
        last_left_word = (
            max(left_words, key=lambda value: (value["end_ms"], value["start_ms"]))
            if left_words
            else None
        )
        first_right_word = (
            min(right_words, key=lambda value: (value["start_ms"], value["end_ms"]))
            if right_words
            else None
        )
        left_word_end_ms = int(
            last_left_word["end_ms"] if last_left_word else left.end_ms
        )
        right_word_start_ms = int(
            first_right_word["start_ms"] if first_right_word else right.start_ms
        )
        source_overlap_ms = max(0, left_word_end_ms - right_word_start_ms)
        # A close serial handoff may share one boundary. A real word-timestamp
        # overlap must retain two independent boundaries or the interjection is
        # silently flattened into sequential speech.
        compact_handoff = bool(
            source_overlap_ms == 0
            and abs(right_word_start_ms - left_word_end_ms) <= tolerance_ms * 2
        )
        shared_handoff_min_ms = max(
            left_word_end_ms - tolerance_ms,
            right_word_start_ms - tolerance_ms,
        )
        shared_handoff_max_ms = min(
            left_word_end_ms + tolerance_ms,
            right_word_start_ms + tolerance_ms,
        )
        anchors.append(
            {
                "boundary_index": boundary_index,
                "left_block_id": left.block_id,
                "right_block_id": right.block_id,
                "left_speaker_id": left.speaker_id,
                "right_speaker_id": right.speaker_id,
                "left_word_end_ms": left_word_end_ms,
                "right_word_start_ms": right_word_start_ms,
                "source_overlap_ms": source_overlap_ms,
                "source_authorized_overlap": source_overlap_ms > 0,
                "left_end_min_ms": max(left.start_ms + 1, left_word_end_ms - tolerance_ms),
                "left_end_max_ms": max(left.start_ms + 1, left_word_end_ms + tolerance_ms),
                "right_start_min_ms": min(right.end_ms - 1, right_word_start_ms - tolerance_ms),
                "right_start_max_ms": min(right.end_ms - 1, right_word_start_ms + tolerance_ms),
                "compact_handoff": compact_handoff,
                "handoff_min_ms": shared_handoff_min_ms,
                "handoff_max_ms": shared_handoff_max_ms,
                "left_word": dict(last_left_word) if last_left_word else None,
                "right_word": dict(first_right_word) if first_right_word else None,
                "left_boundary_fallback_used": last_left_word is None,
                "right_boundary_fallback_used": first_right_word is None,
                "tolerance_ms": tolerance_ms,
            }
        )
    return anchors


def _bounded_integer(value: int, lower: int, upper: int) -> int:
    if lower > upper:
        lower, upper = upper, lower
    return max(lower, min(upper, value))


def enforce_source_word_speaker_alignment(
    blocks: Sequence[SpeechBlock],
    source_blocks: Sequence[SpeechBlock],
    anchors: Sequence[Mapping[str, Any]],
    *,
    required_windows_ms: Mapping[int, int] | None = None,
    maximum_capacity_preserving_shift_ms: int = 2000,
    maximum_capacity_preserving_overlap_ms: int = 1000,
) -> tuple[list[SpeechBlock], list[dict[str, Any]]]:
    """Clamp repaired cross-speaker boundaries to source-word activity.

    This is a final geometric guard after broad five-block balancing.  It does
    not rewrite speech.  When it takes timing capacity back from an overflowing
    block, the normal finalization conductor must solve that deficit through a
    shorter validated delivery or the bounded final-product cadence fit.
    """

    adjusted = list(blocks)
    corrections: list[dict[str, Any]] = []
    for raw_anchor in anchors:
        boundary_index = int(raw_anchor.get("boundary_index", -1))
        right_index = boundary_index + 1
        if (
            boundary_index < 0
            or right_index >= len(adjusted)
            or right_index >= len(source_blocks)
        ):
            continue
        left = adjusted[boundary_index]
        right = adjusted[right_index]
        if left.speaker_id == right.speaker_id:
            continue
        before = {
            "left_end_ms": left.end_ms,
            "right_start_ms": right.start_ms,
        }
        source_left = source_blocks[boundary_index]
        source_right = source_blocks[right_index]
        required_left_ms = max(
            1,
            int((required_windows_ms or {}).get(boundary_index, left.duration_ms)),
        )
        required_right_ms = max(
            1,
            int((required_windows_ms or {}).get(right_index, right.duration_ms)),
        )
        boundary_overlap_ms = max(0, left.end_ms - right.start_ms)
        source_authorized_overlap_ms = max(
            0,
            int(raw_anchor.get("source_overlap_ms") or 0),
        )
        effective_capacity_preserving_overlap_ms = max(
            maximum_capacity_preserving_overlap_ms,
            source_authorized_overlap_ms,
        )
        measured_capacity_preserved = bool(
            required_windows_ms is not None
            and left.duration_ms >= required_left_ms
            and right.duration_ms >= required_right_ms
            and boundary_overlap_ms
            <= effective_capacity_preserving_overlap_ms
            and abs(left.end_ms - source_left.end_ms)
            <= maximum_capacity_preserving_shift_ms
            and abs(right.start_ms - source_right.start_ms)
            <= maximum_capacity_preserving_shift_ms
        )
        if measured_capacity_preserved:
            corrections.append(
                {
                    "method": "source_word_alignment_capacity_preserved",
                    "solver_version": SOURCE_WORD_ALIGNMENT_VERSION,
                    "applied": False,
                    "reason": "measured_capacity_fit_takes_precedence",
                    "boundary_index": boundary_index,
                    "block_index": boundary_index,
                    "affected_block_indices": [boundary_index, right_index],
                    "left_block_id": left.block_id,
                    "right_block_id": right.block_id,
                    "before": before,
                    "after": before,
                    "required_left_window_ms": required_left_ms,
                    "required_right_window_ms": required_right_ms,
                    "maximum_capacity_preserving_shift_ms": (
                        maximum_capacity_preserving_shift_ms
                    ),
                    "boundary_overlap_ms": boundary_overlap_ms,
                    "maximum_capacity_preserving_overlap_ms": (
                        effective_capacity_preserving_overlap_ms
                    ),
                    "source_authorized_overlap_ms": (
                        source_authorized_overlap_ms
                    ),
                    "text_immutable": True,
                    "speaker_sequence_preserved": True,
                    "source_overlap_preserved": boundary_overlap_ms > 0,
                    "human_review_required": False,
                }
            )
            continue
        if bool(raw_anchor.get("compact_handoff")):
            proposed_handoff = int(round((left.end_ms + right.start_ms) / 2.0))
            lower = max(
                left.start_ms + 1,
                int(raw_anchor.get("handoff_min_ms", proposed_handoff)),
            )
            upper = min(
                right.end_ms - 1,
                int(raw_anchor.get("handoff_max_ms", proposed_handoff)),
            )
            if lower <= upper:
                handoff_ms = _bounded_integer(proposed_handoff, lower, upper)
                left_end_ms = handoff_ms
                right_start_ms = handoff_ms
            else:
                # Degenerate sub-millisecond or corrupted windows fall back to
                # the immutable source-block boundary instead of inventing an
                # overlap or reversing speaker order.
                handoff_ms = int(round((source_left.end_ms + source_right.start_ms) / 2.0))
                left_end_ms = max(left.start_ms + 1, handoff_ms)
                right_start_ms = min(right.end_ms - 1, handoff_ms)
        else:
            left_end_ms = _bounded_integer(
                left.end_ms,
                max(left.start_ms + 1, int(raw_anchor["left_end_min_ms"])),
                max(left.start_ms + 1, int(raw_anchor["left_end_max_ms"])),
            )
            right_start_ms = _bounded_integer(
                right.start_ms,
                min(right.end_ms - 1, int(raw_anchor["right_start_min_ms"])),
                min(right.end_ms - 1, int(raw_anchor["right_start_max_ms"])),
            )

        adjusted[boundary_index] = replace(left, end_ms=left_end_ms)
        adjusted[right_index] = replace(right, start_ms=right_start_ms)
        after = {
            "left_end_ms": left_end_ms,
            "right_start_ms": right_start_ms,
        }
        if before == after:
            continue
        corrections.append(
            {
                "method": "source_word_speaker_alignment_clamp",
                "solver_version": SOURCE_WORD_ALIGNMENT_VERSION,
                "applied": True,
                "boundary_index": boundary_index,
                "block_index": boundary_index,
                "affected_block_indices": [boundary_index, right_index],
                "left_block_id": left.block_id,
                "right_block_id": right.block_id,
                "left_speaker_id": left.speaker_id,
                "right_speaker_id": right.speaker_id,
                "before": before,
                "after": after,
                "left_word": raw_anchor.get("left_word"),
                "right_word": raw_anchor.get("right_word"),
                "tolerance_ms": int(raw_anchor.get("tolerance_ms", 0)),
                "source_authorized_overlap_ms": (
                    source_authorized_overlap_ms
                ),
                "text_immutable": True,
                "speaker_sequence_preserved": True,
                "source_overlap_preserved": left_end_ms > right_start_ms,
                "human_review_required": False,
            }
        )
    return adjusted, corrections





class DirectAzureDubRenderer:
    """Render an Azure dub directly from the approved translation artifact."""

    def __init__(
        self,
        settings: Settings,
        backend: AzureTTSBackend,
        *,
        runner: Callable[..., Any] = subprocess.run,
        popen_factory: Callable[..., Any] | None = None,
        timing_repair_provider_factory: Callable[[], AIProvider] | None = None,
    ) -> None:
        self.settings = settings
        self.backend = backend
        self.jobs = JobStore(settings.work_dir)
        self.runner = runner
        self.popen_factory = (
            subprocess.Popen
            if popen_factory is None and runner is subprocess.run
            else popen_factory
        )
        self.timing_repair_provider_factory = timing_repair_provider_factory
        self._timing_repair_provider_instance: AIProvider | None = None

    def _timing_repair_provider(self) -> AIProvider | None:
        if self.timing_repair_provider_factory is None:
            return None
        if self._timing_repair_provider_instance is None:
            self._timing_repair_provider_instance = (
                self.timing_repair_provider_factory()
            )
        return self._timing_repair_provider_instance

    @staticmethod
    def _finalization_conductor_payload(
        block: SpeechBlock,
        *,
        maximum_duration_ms: int,
        preferred_duration_ms: int,
        measured_duration_ms: int,
        duration_evidence: Sequence[Mapping[str, Any]],
        voice: str,
        base_rate_percent: int,
        round_number: int,
        current_spoken_text: str | None = None,
    ) -> dict[str, Any]:
        unit_id = _source_block_id(block.block_id)
        natural = next(
            (
                value
                for value in block.variants
                if value.variant_id == "natural"
            ),
            None,
        )
        faithful = (
            natural.spoken_text
            if natural is not None
            else block.translated_text
        )
        current_spoken = current_spoken_text or block.translated_text
        protected = list(block.protected_entities)
        current_translation = {
            "unit_id": unit_id,
            "faithful_translation": faithful,
            "spoken_text": current_spoken,
            "tts_text": current_spoken,
            "language_features": [],
            "human_review_flags": [],
            "protected_entities_found": protected,
            "protected_entities_preserved": protected,
            "numbers_preserved": True,
            "dates_preserved": True,
            "negation_preserved": True,
            "quote_attribution": None,
            "timing_strategy": "finalization_conductor",
            "delivery_hints": [],
            "pronunciation_substitutions": [],
            "omitted_or_compressed_detail": [],
            "sensitive_claim_flags": [],
        }
        return {
            "schema_version": "turn-timing-repair-request-v1",
            "unit_id": unit_id,
            "source_unit": {
                "unit_id": unit_id,
                "source_text": block.source_text,
                "speaker_id": block.speaker_id,
                "start_ms": block.start_ms,
                "end_ms": block.end_ms,
                "maximum_duration_ms": maximum_duration_ms,
                "preferred_duration_ms": preferred_duration_ms,
                "required_facts": list(block.required_facts),
                "optional_details": list(block.optional_details),
            },
            "current_translation": current_translation,
            "neighbouring_translations": [],
            "timing_failure": {
                "kind": "azure_measured_finalization_overflow",
                "round": round_number,
                "measured_duration_ms": measured_duration_ms,
                "maximum_duration_ms": maximum_duration_ms,
                "shortfall_ms": max(
                    0, measured_duration_ms - maximum_duration_ms
                ),
            },
            "duration_evidence": [dict(value) for value in duration_evidence],
            "synthesis_profile": {
                "provider": "azure_tts",
                "voice": voice,
                "base_rate_percent": base_rate_percent,
            },
            "constraints": {
                "faithful_translation_must_not_change": True,
                "protected_entities_must_be_preserved": protected,
                "protected_verbatim_phrases_must_be_preserved": protected,
                "critical_required_facts_must_be_preserved": list(
                    block.required_facts
                ),
                "only_optional_details_may_be_omitted": list(
                    block.optional_details
                ),
                "preferred_delivery_duration_ms": preferred_duration_ms,
                "maximum_duration_ms": maximum_duration_ms,
                "human_review_required": False,
            },
            "content_trust": "untrusted_data",
        }

    @staticmethod
    def _validate_finalization_candidate(
        block: SpeechBlock,
        candidate: Mapping[str, Any],
    ) -> tuple[str, str, tuple[str, ...]]:
        unit = candidate.get("unit")
        if not isinstance(unit, Mapping):
            raise ValueError("finalization candidate has no translation unit")
        if str(unit.get("unit_id") or "") != _source_block_id(block.block_id):
            raise ValueError("finalization candidate changed the block ID")
        spoken = " ".join(str(unit.get("spoken_text") or "").split())
        tts = " ".join(str(unit.get("tts_text") or spoken).split())
        if not spoken or not tts:
            raise ValueError("finalization candidate contains blank speech")

        # Do not let timing repair turn an approved native isiZulu monetary
        # amount into a compact English-written token merely because the latter
        # synthesizes faster.  Job d724a4c9 exposed this with the approved
        # ``izigidi ezingu-31 zamaRandi`` being replaced by ``u-R31 million``.
        # Besides changing the delivery language, the compact token is not
        # reliably pronounced by the zu-ZA voice.  The hidden pronunciation
        # layer can normalize genuine source code-switches; a repair candidate
        # must not introduce one when the authoritative translation is native.
        native_money = bool(
            re.search(
                r"\b(?:amaRandi|zamaRandi|lamaRandi|ngamaRandi)\b",
                block.translated_text,
                re.IGNORECASE | re.UNICODE,
            )
        )
        introduced_compact_money = bool(
            re.search(
                r"(?:^|[^\w])(?:[A-Za-z]+-)?R\s*\d[\d.,]*\s*"
                r"(?:million|billion|thousand|m|bn)\b",
                f"{spoken} {tts}",
                re.IGNORECASE | re.UNICODE,
            )
            or re.search(
                r"\b\d[\d.,]*\s+(?:million|billion|thousand)\s+"
                r"(?:rand|rands)\b",
                f"{spoken} {tts}",
                re.IGNORECASE | re.UNICODE,
            )
        )
        if native_money and introduced_compact_money:
            raise ValueError(
                "finalization candidate regressed a native isiZulu monetary "
                "amount to an English-written currency code switch"
            )
        if not all(
            bool(unit.get(field))
            for field in (
                "numbers_preserved",
                "dates_preserved",
                "negation_preserved",
            )
        ):
            raise ValueError(
                "finalization candidate did not certify numbers, dates, and negation"
            )
        combined = f"{spoken} {tts}"
        identity_contract = identity_requirements_for_unit(
            {
                "source_text": block.source_text,
                "protected_spans": list(block.protected_entities),
            },
            target_locale="zu-ZA",
        )
        requirements_by_span = {
            str(value.get("required_span") or "").casefold(): value
            for value in identity_contract
            if str(value.get("required_span") or "").strip()
        }

        def contains_form(form: str) -> bool:
            # Match the same words despite harmless punctuation/whitespace
            # differences introduced by a delivery-only rewrite.
            expected = _normalised_words(form)
            actual = _normalised_words(combined)
            if not expected:
                return False
            width = len(expected)
            if any(
                actual[index : index + width] == expected
                for index in range(len(actual) - width + 1)
            ):
                return True
            if len(expected[0]) < 5:
                return False
            # IsiZulu noun, possessive and locative constructions attach
            # prefixes to the first word of an identity: Madlanga Commission
            # -> e-Madlanga Commission and Fadiel Adams -> ngoFadiel Adams.
            # Earlier code allowed this only for one-word identities, causing
            # every correctly inflected multiword candidate to be rejected.
            expected_first = expected[0]
            expected_stem = expected_first[1:]

            def first_word_matches(word: str) -> bool:
                return word.endswith(expected_first) or (
                    len(word) >= 5
                    and len(expected_stem) >= 4
                    and word[1:] == expected_stem
                )

            return any(
                first_word_matches(actual[index])
                and actual[index + 1 : index + width] == expected[1:]
                for index in range(len(actual) - width + 1)
            )

        missing: list[str] = []
        accepted_identity_forms: dict[str, list[str]] = {}
        for protected in block.protected_entities:
            requirement = requirements_by_span.get(protected.casefold(), {})
            accepted = [
                str(value).strip()
                for value in requirement.get("accepted_forms") or [protected]
                if str(value).strip()
            ]
            accepted_identity_forms[protected] = accepted
            if not any(contains_form(value) for value in accepted):
                missing.append(protected)
        if missing:
            raise ValueError(
                "finalization candidate lost protected entities or their "
                f"reviewed isiZulu aliases: {missing}; accepted="
                f"{accepted_identity_forms}"
            )
        omissions = tuple(
            str(value).strip()
            for value in unit.get("omitted_or_compressed_detail", [])
            if str(value).strip()
        )
        # The timing-repair schema uses one field for both true omissions and
        # meaning-preserving compression notes. Requiring those free-form notes
        # to exactly equal the translation unit's optional-detail labels caused
        # safe candidates such as a merged vocative or redundant pronoun to be
        # rejected before Azure measurement. Hard facts remain guarded above:
        # protected identities, numbers, dates and negation may never be lost.
        # Keep every declared compression in the audit instead of discarding it.
        return spoken, tts, omissions

    @staticmethod
    def _semantic_fit_required_facts(
        block: SpeechBlock,
    ) -> list[dict[str, str]]:
        # Upstream units can place a bare complementizer in required_facts when
        # a source sentence was split at "that". It is grammar, not an
        # independent proposition. Keeping it as fact_003 made the reviewer
        # reject faithful isiZulu clause fusion in block_0006 indefinitely.
        non_factual_connectors = {"ukuthi"}
        return [
            {"fact_id": f"fact_{index:03d}", "text": str(value)}
            for index, value in enumerate(
                (
                    str(value).strip()
                    for value in block.required_facts
                    if str(value).strip().casefold()
                    not in non_factual_connectors
                ),
                start=1,
            )
        ]

    @staticmethod
    def _apply_candidate_pronunciation(
        spoken_text: str,
        proposed_tts_text: str,
        pronunciation_dictionary: PronunciationDictionary | None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Make the application dictionary authoritative after every rewrite."""

        if pronunciation_dictionary is None:
            return proposed_tts_text, []
        result = pronunciation_dictionary.apply(spoken_text)
        return (
            result.tts_text,
            [value.to_dict() for value in result.substitutions],
        )

    @staticmethod
    def _compact_semantic_fit_evidence(
        duration_evidence: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """Keep the useful timing frontier instead of resending every round."""

        observations = [
            dict(value)
            for value in duration_evidence
            if isinstance(value, Mapping)
        ]
        if not observations:
            return []

        selected_indices = {len(observations) - 1}
        measured = [
            (index, int(value.get("measured_duration_ms") or 0))
            for index, value in enumerate(observations)
            if value.get("measured_duration_ms") is not None
        ]
        overflows = [
            item
            for item in measured
            if observations[item[0]].get("status") == "overflow"
        ]
        underfills = [
            item
            for item in measured
            if observations[item[0]].get("status") == "underfill"
        ]
        semantic_rejections = [
            index
            for index, value in enumerate(observations)
            if value.get("status") == "semantic_review_rejection"
        ]
        if overflows:
            selected_indices.add(min(overflows, key=lambda item: item[1])[0])
        if underfills:
            selected_indices.add(max(underfills, key=lambda item: item[1])[0])
        if semantic_rejections:
            selected_indices.add(semantic_rejections[-1])

        compact: list[dict[str, Any]] = []
        allowed = (
            "attempt",
            "strategy_attempt",
            "strategy_version",
            "batch_round",
            "status",
            "spoken_text",
            "tts_text",
            "measured_duration_ms",
            "plain_speech_duration_ms",
            "effective_inserted_pause_ms",
            "estimated_plain_speech_ms",
            "plain_duration_estimation_error_ms",
            "target_minimum_ms",
            "target_preferred_ms",
            "target_maximum_ms",
            "omitted_optional_details",
            "performance_island_measurements",
            "target_island_texts",
            "performance_beats",
        )
        for index in sorted(selected_indices):
            value = observations[index]
            item = {key: value[key] for key in allowed if key in value}
            review = value.get("semantic_review")
            if isinstance(review, Mapping):
                item["semantic_review"] = {
                    "accepted": bool(review.get("accepted")),
                    "semantic_fidelity_score": review.get(
                        "semantic_fidelity_score"
                    ),
                    "naturalness_score": review.get("naturalness_score"),
                    "reason_codes": list(review.get("reason_codes") or []),
                }
            compact.append(item)
        return compact

    @classmethod
    def _semantic_fit_payload(
        cls,
        block: SpeechBlock,
        *,
        voice: str,
        base_rate_percent: int,
        pitch_percent: int,
        volume_percent: int,
        attempt_number: int,
        target_minimum_ms: int,
        target_preferred_ms: int,
        target_maximum_ms: int,
        duration_evidence: Sequence[Mapping[str, Any]],
        rejected_spoken_texts: Sequence[str],
        current_spoken_text: str,
        neighbouring_blocks: Sequence[SpeechBlock] = (),
    ) -> dict[str, Any]:
        compact_evidence = cls._compact_semantic_fit_evidence(
            duration_evidence
        )
        latest_measured = (
            (compact_evidence[-1] if compact_evidence else {}).get(
                "measured_duration_ms"
            )
        )
        try:
            measured_duration_ms = int(latest_measured or block.duration_ms)
        except (TypeError, ValueError):
            measured_duration_ms = block.duration_ms
        payload = cls._finalization_conductor_payload(
            block,
            maximum_duration_ms=target_maximum_ms,
            preferred_duration_ms=target_preferred_ms,
            measured_duration_ms=measured_duration_ms,
            duration_evidence=compact_evidence,
            voice=voice,
            base_rate_percent=base_rate_percent,
            round_number=attempt_number,
            current_spoken_text=current_spoken_text,
        )
        payload.update(
            {
                "schema_version": SEMANTIC_FIT_SCHEMA_VERSION,
                "operation": "semantic_fit_candidate",
                "attempt_number": attempt_number,
                "target_duration_band_ms": {
                    "minimum": target_minimum_ms,
                    "preferred": target_preferred_ms,
                    "maximum": target_maximum_ms,
                },
                "rejected_spoken_texts": list(
                    dict.fromkeys(str(value) for value in rejected_spoken_texts)
                )[-6:],
                "evidence_compaction": {
                    "input_observation_count": len(duration_evidence),
                    "transmitted_observation_count": len(compact_evidence),
                    "maximum_rejected_texts_transmitted": 6,
                },
                "source_pause_anchors": [
                    dict(value) for value in block.source_pause_anchors
                ],
                "source_speech_islands": [
                    dict(value) for value in block.source_speech_islands
                ],
                "source_interactions": [
                    dict(value) for value in block.source_interactions
                ],
                "source_timing_evidence": block.source_timing_evidence,
                "semantic_phrase_group": (
                    {
                        "schema_version": SEMANTIC_PHRASE_GROUP_VERSION,
                        "member_block_ids": list(block.semantic_group_member_ids),
                        "target_island_count": len(block.source_speech_islands),
                        "delivery_hints_contract": (
                            "Return one exact non-empty spoken phrase per source "
                            "speech island, in order. Joining delivery_hints with "
                            "single spaces must equal spoken_text exactly."
                        ),
                        "meaning_may_cross_internal_member_boundaries": True,
                        "cross_speaker_movement_allowed": False,
                    }
                    if block.semantic_group_member_ids
                    else None
                ),
                "performance_plan": (
                    {
                        "schema_version": PERFORMANCE_PLAN_VERSION,
                        "region_id": block.performance_region_id,
                        "member_block_ids": list(
                            block.performance_region_member_ids
                        ),
                        "source_islands": [
                            {
                                "island_id": value.get("island_id"),
                                "start_ms": value.get("start_ms"),
                                "end_ms": value.get("end_ms"),
                                "duration_ms": (
                                    int(value.get("duration_ms"))
                                    if value.get("duration_ms") is not None
                                    else max(
                                        1,
                                        int(value.get("end_ms") or 0)
                                        - int(value.get("start_ms") or 0),
                                    )
                                ),
                                "source_text": value.get("text"),
                            }
                            for value in block.source_speech_islands
                        ],
                        "contract": (
                            "Return semantic performance_beats. Every source "
                            "island_id is an authoritative mouth-active breath "
                            "group and must appear exactly once in ordered "
                            "island_deliveries. Do not merge delivery across "
                            "the source silence between adjacent islands. Beats "
                            "may own one or more adjacent islands, but each "
                            "island delivery must independently respect its "
                            "duration_ms at the locked source tempo."
                        ),
                        "stt_member_boundaries_are_not_linguistic_boundaries": True,
                        "speaker_lane_is_immutable": True,
                    }
                    if block.performance_region_id
                    else None
                ),
                "source_phrase_cadence": (
                    {
                        "schema_version": (
                            "mathula-source-breath-group-cadence-v1-v13.19.2"
                        ),
                        "authority": "azure_word_timestamps",
                        "block_id": block.block_id,
                        "window_start_ms": block.start_ms,
                        "window_end_ms": (
                            int(block.source_speech_end_ms)
                            if block.source_speech_end_ms is not None
                            else block.end_ms
                        ),
                        "phrase_count": len(block.source_speech_islands),
                        "phrases": [
                            {
                                "phrase_id": str(
                                    value.get("island_id") or ""
                                ),
                                "source_text": str(value.get("text") or ""),
                                "start_ms": int(value.get("start_ms") or 0),
                                "end_ms": int(value.get("end_ms") or 0),
                                "duration_ms": int(
                                    value.get("duration_ms")
                                    if value.get("duration_ms") is not None
                                    else max(
                                        1,
                                        int(value.get("end_ms") or 0)
                                        - int(value.get("start_ms") or 0),
                                    )
                                ),
                            }
                            for value in block.source_speech_islands
                        ],
                    }
                    if (
                        block.source_timing_evidence == "azure_word_timestamps"
                        and block.source_speech_islands
                    )
                    else plan_phrase_cadence(
                        block_id=block.block_id,
                        text=block.source_text,
                        start_ms=block.start_ms,
                        end_ms=block.end_ms,
                    ).to_dict()
                ),
                "required_facts": cls._semantic_fit_required_facts(block),
                "neighbouring_translations": [
                    {
                        "block_id": _source_block_id(value.block_id),
                        "speaker_id": value.speaker_id,
                        "start_ms": value.start_ms,
                        "end_ms": value.end_ms,
                        "source_text": value.source_text,
                        "approved_translation": value.translated_text,
                    }
                    for value in neighbouring_blocks
                ],
            }
        )
        required_fact_texts = [
            item["text"] for item in payload["required_facts"]
        ]
        source_unit = dict(payload.get("source_unit") or {})
        source_unit["required_facts"] = required_fact_texts
        payload["source_unit"] = source_unit
        mandatory_island_pause_ms = sum(
            max(0, int(value.get("requested_pause_ms") or 0))
            for value in block.source_pause_anchors
            if value.get("pause_kind") == "island_boundary"
        )
        payload["acoustic_timing_budget"] = {
            "mandatory_island_pause_ms": mandatory_island_pause_ms,
            "preferred_plain_speech_ms": max(
                1,
                target_preferred_ms - mandatory_island_pause_ms,
            ),
            "maximum_plain_speech_ms": max(
                1,
                target_maximum_ms - mandatory_island_pause_ms,
            ),
            "inserted_phrase_pauses_remain_flexible": True,
            "island_pause_floor_may_not_be_reduced": True,
        }
        maximum_plain_speech_ms = payload["acoustic_timing_budget"][
            "maximum_plain_speech_ms"
        ]
        # Use the best measured acoustic frontier, not merely the most recent
        # candidate. Portfolio rounds can contain several alternatives and GPT
        # sometimes oscillates between an underfill and a much worse overflow.
        # Feeding the last item back made the controller walk away from the
        # closest known solution. Semantic-review failures are also excluded as
        # controller seeds so a fluent but meaning-damaged contraction cannot
        # become the parent of the next rewrite.
        measured_candidates = [
            value
            for value in duration_evidence
            if isinstance(value, Mapping)
            and int(value.get("measured_duration_ms") or 0) > 0
            and str(value.get("status") or "")
            not in {
                "local_semantic_rejection",
                "duplicate_rejection",
                "semantic_review_rejection",
            }
        ]

        def frontier_score(value: Mapping[str, Any]) -> tuple[int, int, int]:
            measured_ms = int(value.get("measured_duration_ms") or 0)
            if measured_ms < target_minimum_ms:
                timing_distance_ms = target_minimum_ms - measured_ms
            elif measured_ms > target_maximum_ms:
                timing_distance_ms = measured_ms - target_maximum_ms
            else:
                timing_distance_ms = 0
            drift_penalty_ms = 0
            island_measurements = value.get("performance_island_measurements")
            if isinstance(island_measurements, list) and island_measurements:
                alignment = cls._performance_island_alignment_summary(
                    island_measurements
                )
                drift_limit_ms = min(
                    MAXIMUM_INTERNAL_ISLAND_DRIFT_MS,
                    max(1, int(target_maximum_ms - target_minimum_ms)),
                )
                configured_limit = alignment.get(
                    "maximum_allowed_cumulative_drift_ms"
                )
                if configured_limit is not None:
                    drift_limit_ms = max(1, int(configured_limit))
                drift_penalty_ms = max(
                    0,
                    int(
                        alignment.get(
                            "maximum_absolute_cumulative_drift_ms"
                        )
                        or 0
                    )
                    - drift_limit_ms,
                )
            return (
                timing_distance_ms + drift_penalty_ms,
                abs(measured_ms - target_preferred_ms),
                -int(value.get("attempt") or 0),
            )

        measured_frontier = None
        if measured_candidates:
            latest_measured = measured_candidates[-1]
            latest_batch_round = latest_measured.get("batch_round")
            if latest_batch_round is None:
                # Sequential feedback remains causal: the newest measured
                # candidate owns the next correction direction. This preserves
                # the underfill/overflow feedback loop used by single-candidate
                # performance-plan rounds.
                measured_frontier = latest_measured
            else:
                same_round = [
                    value
                    for value in measured_candidates
                    if value.get("batch_round") == latest_batch_round
                ]
                # Portfolio alternatives share one stale input measurement, so
                # only within that same round is it correct to choose the best
                # Azure-measured frontier rather than the last JSON item.
                measured_frontier = min(same_round, key=frontier_score)
        if measured_frontier is not None:
            frontier_plain_ms = int(
                measured_frontier.get("plain_speech_duration_ms")
                or measured_frontier.get("measured_duration_ms")
                or 0
            )
            frontier_total_ms = int(
                measured_frontier.get("measured_duration_ms")
                or frontier_plain_ms
            )
            frontier_text = str(
                measured_frontier.get("spoken_text") or ""
            ).strip()
            # Keep current_translation aligned with the actual acoustic
            # frontier. The caller's state may point at the last portfolio item,
            # which is not necessarily the best measured candidate.
            if frontier_text:
                current_translation = dict(
                    payload.get("current_translation") or {}
                )
                current_translation["spoken_text"] = frontier_text
                current_translation["tts_text"] = frontier_text
                payload["current_translation"] = current_translation

            frontier_characters = len(frontier_text)
            required_reduction_ms = max(
                0,
                frontier_plain_ms - maximum_plain_speech_ms,
            )
            preferred_plain_speech_ms = max(
                1,
                int(
                    payload["acoustic_timing_budget"][
                        "preferred_plain_speech_ms"
                    ]
                ),
            )
            minimum_plain_speech_ms = max(
                1,
                target_minimum_ms - mandatory_island_pause_ms,
            )
            advisory_character_ceiling = None
            advisory_character_target = None
            if frontier_characters and frontier_plain_ms > 0:
                preferred_chars = max(
                    1,
                    round(
                        frontier_characters
                        * preferred_plain_speech_ms
                        / frontier_plain_ms
                    ),
                )
                minimum_chars = max(
                    1,
                    math.floor(
                        frontier_characters
                        * minimum_plain_speech_ms
                        / frontier_plain_ms
                        * 0.96
                    ),
                )
                maximum_chars = max(
                    minimum_chars,
                    math.ceil(
                        frontier_characters
                        * maximum_plain_speech_ms
                        / frontier_plain_ms
                        * 1.02
                    ),
                )
                advisory_character_target = {
                    "latest_spoken_characters": frontier_characters,
                    "minimum_spoken_characters": minimum_chars,
                    "preferred_spoken_characters": preferred_chars,
                    "maximum_spoken_characters": maximum_chars,
                    "basis": "latest_azure_plain_speech_ms",
                    "binding": False,
                }
                if required_reduction_ms:
                    advisory_character_ceiling = max(
                        1,
                        min(
                            maximum_chars,
                            math.floor(
                                frontier_characters
                                * maximum_plain_speech_ms
                                / frontier_plain_ms
                                * 0.96
                            ),
                        ),
                    )
            estimated_plain_ms = int(
                measured_frontier.get("estimated_plain_speech_ms") or 0
            )
            payload["measured_acoustic_frontier"] = {
                "attempt": measured_frontier.get("attempt"),
                "status": measured_frontier.get("status"),
                "selection_policy": "closest_valid_azure_frontier",
                "spoken_text": frontier_text,
                "azure_plain_speech_ms": frontier_plain_ms,
                "azure_total_duration_ms": frontier_total_ms,
                "maximum_plain_speech_ms": maximum_plain_speech_ms,
                "required_additional_reduction_ms": required_reduction_ms,
                "must_be_strictly_shorter_when_overflowing": bool(
                    required_reduction_ms
                ),
                "advisory_maximum_spoken_characters": (
                    advisory_character_ceiling
                ),
                "advisory_character_target": advisory_character_target,
                "provider_estimated_plain_speech_ms": (
                    estimated_plain_ms or None
                ),
                "provider_estimate_understatement_ms": (
                    max(0, frontier_plain_ms - estimated_plain_ms)
                    if estimated_plain_ms
                    else None
                ),
                "measurement_is_authoritative": True,
            }
            if frontier_total_ms > target_maximum_ms:
                correction_direction = "contract"
                required_change_ms = frontier_total_ms - target_maximum_ms
            elif frontier_total_ms < target_minimum_ms:
                correction_direction = "expand"
                required_change_ms = target_minimum_ms - frontier_total_ms
            else:
                correction_direction = "maintain"
                required_change_ms = 0
            payload["acoustic_correction"] = {
                "direction": correction_direction,
                "frontier_attempt": measured_frontier.get("attempt"),
                "latest_status": str(
                    measured_frontier.get("status") or "measured"
                ),
                "latest_spoken_text": frontier_text,
                "azure_measured_duration_ms": frontier_total_ms,
                "target_minimum_ms": target_minimum_ms,
                "target_preferred_ms": target_preferred_ms,
                "target_maximum_ms": target_maximum_ms,
                "required_change_to_enter_band_ms": required_change_ms,
                "preferred_change_ms": (
                    target_preferred_ms - frontier_total_ms
                ),
                "target_duration_ratio": round(
                    target_preferred_ms / max(1, frontier_total_ms),
                    4,
                ),
                "advisory_character_target": advisory_character_target,
                "measurement_is_authoritative": True,
                "tempo_change_allowed": False,
                "instruction": {
                    "contract": (
                        "Use a materially shorter faithful grammatical "
                        "construction; preserve every required fact."
                    ),
                    "expand": (
                        "Use a fuller natural construction by restoring only "
                        "source-grounded detail or explicit relations; add no "
                        "new fact."
                    ),
                    "maintain": (
                        "Keep the whole-region length inside the binding band; "
                        "only redistribute words across performance islands."
                    ),
                }[correction_direction],
            }
            island_measurements = measured_frontier.get(
                "performance_island_measurements"
            )
            if isinstance(island_measurements, list) and island_measurements:
                alignment = cls._performance_island_alignment_summary(
                    island_measurements
                )
                target_island_texts = [
                    str(value).strip()
                    for value in (
                        measured_frontier.get("target_island_texts") or []
                    )
                ]
                if len(target_island_texts) == len(island_measurements):
                    current_target_islands = []
                    for text, measurement, normalized in zip(
                        target_island_texts,
                        island_measurements,
                        alignment.get("islands") or [],
                        strict=True,
                    ):
                        target_ms = max(
                            1,
                            int(measurement.get("target_duration_ms") or 0),
                        )
                        normalized_ms = max(
                            1,
                            int(
                                normalized.get(
                                    "normalized_measured_duration_ms"
                                )
                                or target_ms
                            ),
                        )
                        characters = len(text)
                        target_characters = (
                            max(
                                1,
                                round(characters * target_ms / normalized_ms),
                            )
                            if characters
                            else None
                        )
                        current_target_islands.append(
                            {
                                "island_id": str(
                                    measurement.get("island_id") or ""
                                ),
                                "spoken_text": text,
                                "target_duration_ms": target_ms,
                                "normalized_measured_duration_ms": (
                                    normalized_ms
                                ),
                                "normalized_delta_ms": int(
                                    normalized.get("normalized_delta_ms") or 0
                                ),
                                "normalized_cumulative_drift_ms": int(
                                    normalized.get(
                                        "normalized_cumulative_drift_ms"
                                    )
                                    or 0
                                ),
                                "advisory_target_characters": (
                                    target_characters
                                ),
                            }
                        )
                    alignment["current_target_islands"] = (
                        current_target_islands
                    )
                    alignment["current_island_text_is_authoritative_feedback"] = (
                        True
                    )
                payload["performance_island_alignment"] = alignment
        payload["synthesis_profile"] = {
            "provider": "azure_tts",
            "voice": voice,
            "rate_percent": base_rate_percent,
            "pitch_percent": pitch_percent,
            "volume_percent": volume_percent,
            "tempo_locked": True,
            "time_stretch_allowed": False,
        }
        constraints = dict(payload.get("constraints") or {})
        constraints.update(
            {
                "faithful_translation_must_not_change": True,
                "speaker_tempo_locked": True,
                "per_block_rate_change_allowed": False,
                "post_synthesis_time_stretch_allowed": False,
                "cross_speaker_boundary_borrowing_allowed": False,
                "candidate_must_be_new": True,
                "target_duration_band_ms": payload["target_duration_band_ms"],
                "critical_required_facts_must_be_preserved": (
                    required_fact_texts
                ),
                "mandatory_island_pause_ms": mandatory_island_pause_ms,
                "maximum_plain_speech_ms": payload[
                    "acoustic_timing_budget"
                ]["maximum_plain_speech_ms"],
                "semantic_island_delivery_hints_required": bool(
                    block.semantic_group_member_ids
                ),
                "performance_beats_required": bool(
                    block.performance_region_id
                ),
            }
        )
        payload["constraints"] = constraints
        return payload

    @classmethod
    def _semantic_fit_review_payload(
        cls,
        block: SpeechBlock,
        *,
        spoken_text: str,
        tts_text: str,
        omitted_details: Sequence[str],
        measured_duration_ms: int,
    ) -> dict[str, Any]:
        natural = next(
            (
                value.spoken_text
                for value in block.variants
                if value.variant_id == "natural"
            ),
            block.translated_text,
        )
        return {
            "schema_version": SEMANTIC_FIT_SCHEMA_VERSION,
            "operation": "semantic_fit_review",
            "unit_id": _source_block_id(block.block_id),
            "source_text": block.source_text,
            "approved_faithful_translation": natural,
            "candidate": {
                "spoken_text": spoken_text,
                "tts_text": tts_text,
                "omitted_or_compressed_detail": list(omitted_details),
                "measured_duration_ms": measured_duration_ms,
            },
            "required_facts": cls._semantic_fit_required_facts(block),
            "optional_details": list(block.optional_details),
            "protected_entities": list(block.protected_entities),
            "content_trust": "untrusted_data",
            "semantic_group_member_ids": list(block.semantic_group_member_ids),
        }

    @staticmethod
    def _semantic_target_islands(
        block: SpeechBlock,
        candidate_data: Mapping[str, Any],
        spoken_text: str,
    ) -> tuple[str, ...]:
        if block.performance_region_id:
            unit = candidate_data.get("unit")
            if not isinstance(unit, Mapping):
                raise ValueError("performance-plan candidate has no unit")
            raw_beats = unit.get("performance_beats")
            if not isinstance(raw_beats, list) or not raw_beats:
                raise ValueError(
                    "performance-plan candidate requires performance_beats"
                )
            expected_ids = [
                str(value.get("island_id") or "")
                for value in block.source_speech_islands
            ]
            deliveries: dict[str, str] = {}
            beat_ids: set[str] = set()
            for raw_beat in raw_beats:
                if not isinstance(raw_beat, Mapping):
                    raise ValueError("performance beat must be an object")
                beat_id = str(raw_beat.get("beat_id") or "").strip()
                if not beat_id or beat_id in beat_ids:
                    raise ValueError("performance beat IDs must be unique")
                beat_ids.add(beat_id)
                raw_deliveries = raw_beat.get("island_deliveries")
                if not isinstance(raw_deliveries, list) or not raw_deliveries:
                    raise ValueError(
                        "each performance beat requires island_deliveries"
                    )
                for raw_delivery in raw_deliveries:
                    if not isinstance(raw_delivery, Mapping):
                        raise ValueError("island delivery must be an object")
                    island_id = str(
                        raw_delivery.get("island_id") or ""
                    ).strip()
                    text = str(
                        raw_delivery.get("spoken_text") or ""
                    ).strip()
                    if not island_id or not text or island_id in deliveries:
                        raise ValueError(
                            "every performance source island must have one "
                            "non-empty delivery"
                        )
                    deliveries[island_id] = text
            if list(deliveries) != expected_ids:
                raise ValueError(
                    "performance beat island deliveries must cover every "
                    "source island exactly once and in source order"
                )
            islands = tuple(deliveries[value] for value in expected_ids)
            # GPT may move a comma or terminal full stop between beat JSON
            # fields while preserving the exact words. Punctuation is not an
            # acoustic token for this ownership check; lexical changes remain
            # a hard rejection.
            if _normalised_words(" ".join(islands)) != _normalised_words(
                spoken_text
            ):
                raise ValueError(
                    "performance beat island deliveries must concatenate to "
                    "spoken_text"
                )
            return islands
        if not block.semantic_group_member_ids:
            return ()
        unit = candidate_data.get("unit")
        if not isinstance(unit, Mapping):
            raise ValueError("semantic phrase group candidate has no unit")
        islands = tuple(
            str(value).strip()
            for value in unit.get("delivery_hints") or []
            if str(value).strip()
        )
        expected = len(block.source_speech_islands)
        if len(islands) != expected:
            raise ValueError(
                "semantic phrase group requires exactly "
                f"{expected} target island phrases, received {len(islands)}"
            )
        joined = " ".join(islands)
        if " ".join(joined.split()) != " ".join(spoken_text.split()):
            raise ValueError(
                "semantic phrase group delivery_hints must concatenate to spoken_text"
            )
        return islands

    @staticmethod
    def _semantic_review_passed(review: Mapping[str, Any]) -> bool:
        return (
            bool(review.get("accepted"))
            and not review.get("reason_codes")
            and all(
                review.get(field) is True
                for field in (
                    "attribution_preserved",
                    "uncertainty_preserved",
                    "negation_preserved",
                    "no_new_facts",
                    "natural_spoken_target_language",
                )
            )
            and float(review.get("semantic_fidelity_score") or 0) >= 95
            and float(review.get("naturalness_score") or 0) >= 85
        )

    def _calibrate_source_tempo_profiles(
        self,
        blocks: Sequence[SpeechBlock],
        artifacts: DirectDubArtifacts,
        *,
        synthesis_profiles: Mapping[int, tuple[str, int, int, int]],
        assignments: Mapping[str, Mapping[str, Any]],
        options: DirectDubOptions,
        force: bool,
        progress: _DirectDubProgress,
        pronunciation_dictionary: PronunciationDictionary | None,
    ) -> tuple[dict[int, tuple[str, int, int, int]], dict[str, Any]]:
        """Calibrate Azure's immutable speaker rate to source articulation tempo.

        Persona rate is only a relative identity cue.  It cannot represent an
        absolute slow/fast source speaker when there is a single speaker (the
        family median collapses to zero).  This calibration uses Azure word
        timestamps to measure source syllables/second, then synthesizes up to
        three representative target-language samples at the persona rate.  The
        resulting Azure duration tells us the conservative SSML rate required
        for the target voice to articulate at the source speaker's tempo.

        Long source pauses are excluded here because they are transferred by
        the existing pause/island lip-sync path.  No audio stretching is used.
        """

        calibrated = dict(synthesis_profiles)
        by_speaker: dict[str, list[int]] = {}
        for index, block in enumerate(blocks):
            by_speaker.setdefault(block.speaker_id, []).append(index)

        report: dict[str, Any] = {
            "schema_version": SOURCE_TEMPO_CALIBRATION_VERSION,
            "priority_order": [
                "source_speaker_tempo",
                "source_word_lipsync",
                "semantic_rephrase_to_fit",
            ],
            "speaker_count": len(by_speaker),
            "speakers": {},
        }
        lower_bound = max(
            int(self.backend.ssml_bounds.rate_min_percent),
            SOURCE_TEMPO_CALIBRATION_MIN_RATE_PERCENT,
        )
        upper_bound = min(
            int(self.backend.ssml_bounds.rate_max_percent),
            int(options.max_azure_rate_percent),
            SOURCE_TEMPO_CALIBRATION_MAX_RATE_PERCENT,
        )

        for speaker_id, indices in by_speaker.items():
            first_profile = calibrated[indices[0]]
            voice, persona_rate, pitch, volume = first_profile
            assignment = assignments.get(speaker_id) or {}
            if assignment.get("resolution_status") == "explicit_voice_map":
                report["speakers"][speaker_id] = {
                    "status": "explicit_rate_preserved",
                    "voice": voice,
                    "persona_rate_percent": persona_rate,
                    "calibrated_rate_percent": persona_rate,
                }
                continue

            source_rows: list[tuple[int, dict[str, float | int]]] = []
            total_source_syllables = 0
            total_source_words = 0
            total_active_ms = 0
            for index in indices:
                metrics = _source_active_speech_metrics(blocks[index])
                if metrics is None:
                    continue
                source_rows.append((index, metrics))
                total_source_syllables += int(metrics["source_syllables"])
                total_source_words += int(metrics["source_word_count"])
                total_active_ms += int(metrics["active_ms"])

            if (
                not source_rows
                or total_active_ms < SOURCE_TEMPO_CALIBRATION_MIN_ACTIVE_MS * 2
                or total_source_syllables
                < SOURCE_TEMPO_CALIBRATION_MIN_SOURCE_SYLLABLES * 2
            ):
                report["speakers"][speaker_id] = {
                    "status": "insufficient_word_timing",
                    "voice": voice,
                    "persona_rate_percent": persona_rate,
                    "calibrated_rate_percent": persona_rate,
                    "timed_source_block_count": len(source_rows),
                }
                continue

            source_syllables_per_second = (
                total_source_syllables / (total_active_ms / 1000.0)
            )
            source_words_per_second = (
                total_source_words / (total_active_ms / 1000.0)
            )

            sample_candidates: list[tuple[int, str, int]] = []
            for index, _metrics in source_rows:
                block = blocks[index]
                tts_text = block.tts_text
                if pronunciation_dictionary is not None:
                    tts_text = pronunciation_dictionary.apply(
                        block.translated_text
                    ).tts_text
                target_syllables = _estimated_zulu_syllables(tts_text)
                if target_syllables < SOURCE_TEMPO_CALIBRATION_MIN_TARGET_SYLLABLES:
                    continue
                sample_candidates.append((index, tts_text, target_syllables))

            # Prefer medium-length samples.  They are long enough to average out
            # phrase-edge TTS effects but cheap enough not to add noticeable
            # latency to a short-form dub.
            sample_candidates.sort(
                key=lambda value: (abs(value[2] - 20), value[0])
            )
            sample_candidates = sample_candidates[
                :SOURCE_TEMPO_CALIBRATION_MAX_SAMPLES
            ]
            measured_samples: list[dict[str, Any]] = []
            required_rates: list[int] = []
            for index, tts_text, target_syllables in sample_candidates:
                block = blocks[index]
                desired_plain_ms = max(
                    1,
                    round(
                        target_syllables
                        / max(0.001, source_syllables_per_second)
                        * 1000.0
                    ),
                )
                calibration_request = AzureTTSRequest(
                    turn_id=f"{block.block_id}__source_tempo_calibration",
                    speaker_id=block.speaker_id,
                    text=tts_text,
                    preferred_duration_ms=desired_plain_ms,
                    maximum_duration_ms=max(desired_plain_ms, block.duration_ms),
                    voice=voice,
                    language="zu-ZA",
                    rate_percent=persona_rate,
                    pitch_percent=pitch,
                    volume_percent=volume,
                    parts=build_initialism_ssml_parts(
                        tts_text,
                        language="zu-ZA",
                    ),
                )
                result = self.backend.synthesize(
                    calibration_request,
                    artifacts.candidate(calibration_request),
                    force=force,
                )
                current_speed_factor = max(
                    0.1, 1.0 + persona_rate / 100.0
                )
                required_speed_factor = (
                    current_speed_factor
                    * result.duration_ms
                    / max(1, desired_plain_ms)
                )
                raw_required_rate = round(
                    (required_speed_factor - 1.0) * 100.0
                )
                required_rates.append(raw_required_rate)
                measured_samples.append(
                    {
                        "block_id": block.block_id,
                        "source_active_ms": int(
                            (_source_active_speech_metrics(block) or {}).get(
                                "active_ms", 0
                            )
                        ),
                        "target_syllables": target_syllables,
                        "source_syllables_per_second": round(
                            source_syllables_per_second, 6
                        ),
                        "desired_target_plain_ms": desired_plain_ms,
                        "azure_persona_rate_percent": persona_rate,
                        "azure_measured_plain_ms": result.duration_ms,
                        "raw_required_rate_percent": raw_required_rate,
                        "cache_reused": bool(result.idempotent_reuse),
                    }
                )

            if not required_rates:
                report["speakers"][speaker_id] = {
                    "status": "no_representative_target_sample",
                    "voice": voice,
                    "persona_rate_percent": persona_rate,
                    "calibrated_rate_percent": persona_rate,
                    "source_syllables_per_second": round(
                        source_syllables_per_second, 6
                    ),
                    "source_words_per_second": round(
                        source_words_per_second, 6
                    ),
                }
                continue

            robust_rate = int(round(float(median(required_rates))))
            robust_rate = max(lower_bound, min(upper_bound, robust_rate))
            if abs(robust_rate - persona_rate) < SOURCE_TEMPO_CALIBRATION_MIN_SHIFT_PERCENT:
                robust_rate = persona_rate

            for index in indices:
                profile_voice, _old_rate, profile_pitch, profile_volume = calibrated[index]
                calibrated[index] = (
                    profile_voice,
                    robust_rate,
                    profile_pitch,
                    profile_volume,
                )

            speaker_report = {
                "status": "calibrated",
                "voice": voice,
                "persona_rate_percent": persona_rate,
                "calibrated_rate_percent": robust_rate,
                "rate_shift_percent": robust_rate - persona_rate,
                "rate_bounds_percent": [lower_bound, upper_bound],
                "source_active_ms": total_active_ms,
                "source_word_count": total_source_words,
                "source_syllable_count": total_source_syllables,
                "source_words_per_second": round(source_words_per_second, 6),
                "source_syllables_per_second": round(
                    source_syllables_per_second, 6
                ),
                "sample_count": len(measured_samples),
                "samples": measured_samples,
                "tempo_is_primary_quality_constraint": True,
                "long_pauses_excluded_from_rate_calibration": True,
                "long_pauses_transferred_by_lipsync_path": True,
            }
            report["speakers"][speaker_id] = speaker_report
            progress.emit(
                "source_tempo",
                (
                    f"{speaker_id}: source cadence calibrated "
                    f"{persona_rate:+d}% -> {robust_rate:+d}% "
                    f"({source_words_per_second:.2f} source words/s)"
                ),
                status="completed",
                speaker_id=speaker_id,
                persona_rate_percent=persona_rate,
                calibrated_rate_percent=robust_rate,
                source_words_per_second=round(source_words_per_second, 6),
                source_syllables_per_second=round(
                    source_syllables_per_second, 6
                ),
            )

        atomic_write_json(artifacts.source_tempo_calibration, report)
        return calibrated, report


    def _synthesize_locked_tempo_block(
        self,
        block: SpeechBlock,
        artifacts: DirectDubArtifacts,
        *,
        voice: str,
        base_rate_percent: int,
        pitch_percent: int,
        volume_percent: int,
        force: bool,
        timing_deadline_ms: int | None = None,
        timing_preferred_ms: int | None = None,
    ) -> dict[str, Any]:
        """Synthesize at one immutable rate; never slow, accelerate or stretch."""

        allowed_duration_ms = min(
            block.duration_ms,
            max(1, int(timing_deadline_ms or block.duration_ms)),
        )
        preferred_duration_ms = min(
            allowed_duration_ms,
            max(1, int(timing_preferred_ms or allowed_duration_ms)),
        )
        request = AzureTTSRequest(
            turn_id=block.block_id,
            speaker_id=block.speaker_id,
            text=block.tts_text,
            preferred_duration_ms=allowed_duration_ms,
            maximum_duration_ms=allowed_duration_ms,
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
        pause_alignment_by_path: dict[str, dict[str, Any]] = {}
        plain = self.backend.synthesize(
            request,
            artifacts.candidate(request),
            force=force,
        )
        attempts.append(plain)
        selected = plain
        word_timed_lipsync = (
            block.source_timing_evidence == "azure_word_timestamps"
        )
        if (
            block.source_pause_anchors
            or block.source_speech_start_ms is not None
        ) and (word_timed_lipsync or plain.duration_ms < allowed_duration_ms):
            available_pause_ms = max(
                0,
                (
                    preferred_duration_ms
                    if word_timed_lipsync
                    else allowed_duration_ms
                )
                - plain.duration_ms
                - (
                    0
                    if word_timed_lipsync
                    else SOURCE_PAUSE_TRANSITION_ROOM_MS
                ),
            )
            requested_leading_pause_ms = max(
                0,
                int(block.source_speech_start_ms or block.start_ms)
                - block.start_ms,
            )
            leading_pause_ms = min(
                MAXIMUM_INSERTED_SOURCE_PAUSE_MS,
                requested_leading_pause_ms,
                available_pause_ms,
            )
            if leading_pause_ms < MINIMUM_INSERTED_SOURCE_PAUSE_MS:
                leading_pause_ms = 0
            requested_internal_pause_ms = sum(
                max(0, int(value.get("requested_pause_ms") or 0))
                for value in block.source_pause_anchors
            )
            requested_island_pause_ms = sum(
                max(0, int(value.get("requested_pause_ms") or 0))
                for value in block.source_pause_anchors
                if value.get("pause_kind") == "island_boundary"
            )
            pause_budget_ms = min(
                self.backend.ssml_bounds.total_pause_max_ms,
                max(
                    0,
                    available_pause_ms - leading_pause_ms,
                    (
                        requested_island_pause_ms
                        if word_timed_lipsync
                        else 0
                    ),
                ),
            )
            pause_max_ms = int(
                getattr(self.backend.ssml_bounds, "pause_max_ms", 2000)
            )
            if block.semantic_target_islands:
                pause_parts, pause_alignment = build_semantic_island_ssml_parts(
                    block.semantic_target_islands,
                    block.source_pause_anchors,
                    pause_budget_ms=pause_budget_ms,
                    language="zu-ZA",
                    pause_max_ms=pause_max_ms,
                )
            else:
                pause_parts, pause_alignment = build_source_pause_ssml_parts(
                    block.tts_text,
                    block.source_pause_anchors,
                    pause_budget_ms=pause_budget_ms,
                    language="zu-ZA",
                    fill_pause_budget=True,
                    pause_max_ms=pause_max_ms,
                    prioritize_island_boundaries=word_timed_lipsync,
                    protected_entities=block.protected_entities,
                )
            pause_alignment = {
                **pause_alignment,
                "plain_locked_tempo_duration_ms": plain.duration_ms,
                "preferred_duration_ms": preferred_duration_ms,
                "maximum_duration_ms": allowed_duration_ms,
                "requested_internal_pause_ms": requested_internal_pause_ms,
                "requested_island_pause_ms": requested_island_pause_ms,
                "island_pause_preservation_required": word_timed_lipsync,
            }
            if leading_pause_ms:
                if not pause_parts:
                    natural_parts = build_initialism_ssml_parts(
                        block.tts_text,
                        language="zu-ZA",
                    )
                    pause_parts = natural_parts or (TextPart(block.tts_text),)
                maximum_break_ms = max(
                    1,
                    int(
                        getattr(
                            self.backend.ssml_bounds,
                            "pause_max_ms",
                            2000,
                        )
                    ),
                )
                leading_breaks: list[BreakPart] = []
                remaining_leading_pause_ms = leading_pause_ms
                while remaining_leading_pause_ms:
                    duration_ms = min(
                        maximum_break_ms,
                        remaining_leading_pause_ms,
                    )
                    leading_breaks.append(BreakPart(duration_ms))
                    remaining_leading_pause_ms -= duration_ms
                pause_parts = (*leading_breaks, *pause_parts)
                insertions = list(pause_alignment.get("insertions") or [])
                insertions.insert(
                    0,
                    {
                        "kind": "source_leading_pause",
                        "duration_ms": leading_pause_ms,
                        "ssml_break_durations_ms": [
                            value.duration_ms for value in leading_breaks
                        ],
                        "source_gap_ms": requested_leading_pause_ms,
                        "target_left_text": None,
                        "target_right_text": (
                            block.tts_text.split(maxsplit=1)[0]
                            if block.tts_text.split()
                            else None
                        ),
                        "target_character_offset": 0,
                    },
                )
                pause_alignment = {
                    **pause_alignment,
                    "applied": True,
                    "reason": "source_word_gaps_transferred_to_ssml",
                    "pause_budget_ms": available_pause_ms,
                    "requested_internal_pause_ms": requested_internal_pause_ms,
                    "requested_island_pause_ms": requested_island_pause_ms,
                    "total_added_pause_ms": leading_pause_ms
                    + int(pause_alignment.get("total_added_pause_ms") or 0),
                    "maximum_added_pause_ms": max(
                        leading_pause_ms,
                        int(pause_alignment.get("maximum_added_pause_ms") or 0),
                    ),
                    "maximum_ssml_break_ms": max(
                        int(
                            pause_alignment.get("maximum_ssml_break_ms") or 0
                        ),
                        *(value.duration_ms for value in leading_breaks),
                    ),
                    "insertions": insertions,
                    "approved_text_changed": False,
                }
            if pause_alignment.get("applied") and pause_parts:
                pause_request = replace(request, parts=pause_parts)
                pause_result = self.backend.synthesize(
                    pause_request,
                    artifacts.candidate(pause_request),
                    force=force,
                )
                attempts.append(pause_result)
                pause_alignment_by_path[pause_result.output_path] = pause_alignment
                # In semantic-fit mode the measured source pause is part of the
                # timing contract. If Azure's pause-bearing candidate exceeds
                # the window, reject the wording and search again; never fall
                # back to continuous speech merely because it fits.
                selected = pause_result
                current_parts = pause_parts
                current_alignment = pause_alignment
                for calibration_round in range(
                    1,
                    SEMANTIC_FIT_PAUSE_CALIBRATION_ATTEMPTS + 1,
                ):
                    if selected.duration_ms <= allowed_duration_ms:
                        break
                    pause_total_ms = sum(
                        part.duration_ms
                        for part in current_parts
                        if isinstance(part, BreakPart)
                    )
                    if pause_total_ms <= 0:
                        break
                    calibration_floors = (
                        source_pause_calibration_floors(current_alignment)
                        if word_timed_lipsync
                        else None
                    )
                    protected_pause_ms = sum(calibration_floors or ())
                    reducible_pause_ms = max(
                        0,
                        pause_total_ms - protected_pause_ms,
                    )
                    if reducible_pause_ms <= 0:
                        # Every remaining millisecond belongs to a protected
                        # source-speech island. A shorter faithful delivery is
                        # required; never erase the mouth-activity boundary.
                        break
                    overflow_ms = selected.duration_ms - allowed_duration_ms
                    reduction_ms = min(
                        reducible_pause_ms,
                        overflow_ms + SEMANTIC_FIT_PAUSE_CALIBRATION_GUARD_MS,
                    )
                    calibrated_parts, calibrated_durations = (
                        calibrate_source_pause_ssml_parts(
                            current_parts,
                            reduction_ms=reduction_ms,
                            minimum_break_durations_ms=calibration_floors,
                        )
                    )
                    if calibrated_parts == current_parts:
                        break
                    current_alignment = calibrated_source_pause_audit(
                        current_alignment,
                        calibrated_durations,
                        reduction_ms=reduction_ms,
                        calibration_round=calibration_round,
                    )
                    calibrated_request = replace(request, parts=calibrated_parts)
                    selected = self.backend.synthesize(
                        calibrated_request,
                        artifacts.candidate(calibrated_request),
                        force=force,
                    )
                    attempts.append(selected)
                    pause_alignment_by_path[selected.output_path] = current_alignment
                    current_parts = calibrated_parts

        if selected.duration_ms > allowed_duration_ms:
            raise TimingOverflowError(
                f"{block.block_id} exceeds its immutable semantic-fit window at "
                "the locked speaker tempo",
                measured_duration_ms=selected.duration_ms,
                target_duration_ms=allowed_duration_ms,
                stretch_ratio=selected.duration_ms / max(1, allowed_duration_ms),
                plain_duration_ms=plain.duration_ms,
                effective_inserted_pause_ms=max(
                    0,
                    selected.duration_ms - plain.duration_ms,
                ),
            )

        final_path = artifacts.final_block(block.block_id)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_copy(Path(selected.output_path), final_path)
        pause_fill = dict(
            pause_alignment_by_path.get(selected.output_path) or {}
        )
        trailing_room_ms = max(0, allowed_duration_ms - selected.duration_ms)
        policy = ProfessionalDubPolicy()
        cadence_qc = assess_perceptual_cadence(
            selected_rate_percent=base_rate_percent,
            base_rate_percent=base_rate_percent,
            whole_voice_stretch_ratio=1.0,
            timeline_fill_action=(
                "semantic_fit_source_pause_aligned"
                if pause_fill
                else "semantic_fit_locked_tempo"
            ),
            pause_fill=pause_fill,
            trailing_room_ms=trailing_room_ms,
            maximum_added_pause_ms=policy.maximum_added_pause_ms,
            maximum_rate_delta_percent=0,
            maximum_whole_voice_change=1.0,
        )
        rendered_speech_end_ms = block.start_ms + selected.duration_ms
        source_endpoint_error_ms = (
            rendered_speech_end_ms - int(block.source_speech_end_ms)
            if block.source_speech_end_ms is not None
            else None
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
                    "source_pause_alignment": pause_alignment_by_path.get(
                        item.output_path
                    ),
                }
                for item in attempts
            ],
            "selected_azure_rate_percent": base_rate_percent,
            "post_synthesis_time_stretch_ratio": 1.0,
            "exact_audio_boundary": {
                "schema_version": "mathula-exact-audio-boundary-v1",
                "corrected": False,
                "measured_before_ms": selected.duration_ms,
                "duration_ms": selected.duration_ms,
                "target_duration_ms": block.duration_ms,
                "semantic_timing_deadline_ms": allowed_duration_ms,
                "correction_ratio": 1.0,
                "combined_stretch_ratio": 1.0,
            },
            "timeline_fill_action": (
                "semantic_fit_source_pause_aligned"
                if pause_fill
                else "semantic_fit_locked_tempo"
            ),
            "speech_rate_preserved": True,
            "speaker_tempo_locked": True,
            "cadence_plan": plan_phrase_cadence(
                block_id=block.block_id,
                text=block.tts_text,
                start_ms=block.start_ms,
                end_ms=(
                    int(block.source_speech_end_ms)
                    if block.source_timing_evidence
                    == "azure_word_timestamps"
                    and block.source_speech_end_ms is not None
                    else block.end_ms
                ),
            ).to_dict(),
            "cadence_qc": cadence_qc,
            "source_pause_alignment": pause_fill or None,
            "output_path": str(final_path),
            "output_sha256": checksum(final_path),
            "duration_ms": selected.duration_ms,
            "semantic_timing_deadline_ms": allowed_duration_ms,
            "source_lipsync_endpoint": {
                "timing_evidence": block.source_timing_evidence,
                "source_speech_end_ms": block.source_speech_end_ms,
                "rendered_speech_end_ms": rendered_speech_end_ms,
                "endpoint_error_ms": source_endpoint_error_ms,
                "within_deadline": selected.duration_ms <= allowed_duration_ms,
                "same_speaker_transition_borrowed": False,
            },
        }

    @staticmethod
    def _semantic_fit_target_band(
        block: SpeechBlock,
        tolerance_ms: int,
    ) -> tuple[int, int, int]:
        tolerance_ms = max(0, int(tolerance_ms))
        source_speech_end_ms = block.source_speech_end_ms
        source_speech_start_ms = block.source_speech_start_ms
        preferred = (
            max(
                1,
                min(
                    block.duration_ms,
                    source_speech_end_ms
                    - int(source_speech_start_ms or block.start_ms),
                ),
            )
            if source_speech_end_ms is not None
            else max(1, block.duration_ms - SOURCE_PAUSE_TRANSITION_ROOM_MS)
        )
        word_timed = bool(
            block.source_timing_evidence == "azure_word_timestamps"
            and source_speech_end_ms is not None
        )
        # Word-timed dialogue is onset anchored.  A two-sided centring budget
        # allowed a duration miss of twice the configured tolerance and then
        # moved the entire waveform around the visible mouth envelope.  That
        # passed endpoint QC while dialogue inside the turn lagged.  Keep one
        # duration tolerance and leave natural trailing room for underfill.
        duration_tolerance_ms = tolerance_ms
        playable_from_source_start_ms = max(
            1,
            block.end_ms - int(source_speech_start_ms or block.start_ms),
        )
        maximum = (
            min(
                playable_from_source_start_ms,
                preferred + duration_tolerance_ms,
            )
            if word_timed
            else block.duration_ms
        )
        return (
            max(1, preferred - duration_tolerance_ms),
            preferred,
            max(1, maximum),
        )

    @staticmethod
    def _center_word_timed_delivery(
        block: SpeechBlock,
        measured_duration_ms: int,
        tolerance_ms: int,
    ) -> tuple[SpeechBlock, dict[str, Any]]:
        """Hard-anchor audio to the first source word and audit the endpoint."""

        source_start_ms = int(block.source_speech_start_ms or block.start_ms)
        raw_source_end_ms = int(block.source_speech_end_ms or block.end_ms)
        # Azure may report the last word a few milliseconds beyond the decoded
        # media boundary. Centre against the playable endpoint, otherwise a
        # final block that fits can still be positioned past the video tail.
        source_end_ms = min(raw_source_end_ms, block.end_ms)
        tolerance_ms = max(0, int(tolerance_ms))
        selected_start_ms = max(0, source_start_ms)
        selected_end_ms = selected_start_ms + int(measured_duration_ms)
        start_error_ms = selected_start_ms - source_start_ms
        end_error_ms = selected_end_ms - source_end_ms
        return (
            replace(
                block,
                start_ms=selected_start_ms,
                end_ms=selected_end_ms,
            ),
            {
                "timing_evidence": block.source_timing_evidence,
                "source_speech_start_ms": source_start_ms,
                "source_speech_end_ms": source_end_ms,
                "unclamped_source_speech_end_ms": raw_source_end_ms,
                "rendered_speech_start_ms": selected_start_ms,
                "rendered_speech_end_ms": selected_end_ms,
                "start_error_ms": start_error_ms,
                "endpoint_error_ms": end_error_ms,
                "tolerance_ms": tolerance_ms,
                "within_tolerance": (
                    abs(start_error_ms) <= tolerance_ms
                    and abs(end_error_ms) <= tolerance_ms
                ),
                "same_speaker_transition_borrowed": False,
                "placement_policy": "source_word_onset_hard_anchor",
            },
        )

    @staticmethod
    def _same_speaker_transition_borrow_limit(
        blocks: Sequence[SpeechBlock],
        index: int,
    ) -> int:
        """Return safe silence available without consuming most of the pause."""

        if index + 1 >= len(blocks):
            return 0
        block = blocks[index]
        following = blocks[index + 1]
        if block.source_timing_evidence == "azure_word_timestamps":
            # Timed source words are a visible mouth-activity envelope. Do not
            # spend the following silence to make a long dub fit: that was the
            # measured cause of 1-2 second lip lag in job d724a4c9.
            return 0
        if following.speaker_id != block.speaker_id:
            return 0
        gap_ms = max(0, following.start_ms - block.end_ms)
        return min(
            SEMANTIC_FIT_MAX_SAME_SPEAKER_TRANSITION_BORROW_MS,
            gap_ms // 2,
            max(0, gap_ms - SOURCE_PAUSE_TRANSITION_ROOM_MS),
        )

    def _synthesize_semantic_fit_batch(
        self,
        blocks: Sequence[SpeechBlock],
        artifacts: DirectDubArtifacts,
        *,
        synthesis_profiles: Mapping[int, tuple[str, int, int, int]],
        options: DirectDubOptions,
        force: bool,
        progress: _DirectDubProgress,
        pronunciation_dictionary: PronunciationDictionary | None = None,
    ) -> dict[
        int,
        tuple[SpeechBlock, dict[str, Any], list[dict[str, Any]]]
        | PreparedVariantsExhausted,
    ]:
        """Fit every unresolved block in shared GPT feedback rounds.

        Production semantic-fit never forces a semantically safe delivery into
        an audibly rushed window.  If strict fitting cannot meet the natural
        cadence ceiling, the selected text and measured overflow are handed to
        the normal timeline finalizer, which can borrow real silence or adjust
        a handoff before considering bounded compression.
        """

        provider = self._timing_repair_provider()
        if provider is None or not all(
            callable(getattr(provider, method, None))
            for method in ("fit_turns_semantically", "review_semantic_fits")
        ):
            raise DirectDubError(
                "Semantic-fit batch mode requires batch candidate generation "
                "and batch semantic review"
            )

        provider_view = getattr(provider, "with_request_options", None)
        if callable(provider_view):
            timeout_seconds = float(
                os.getenv("MATHULA_TV_SEMANTIC_FIT_TIMEOUT_SECONDS", "180")
            )
            max_retries = int(
                os.getenv("MATHULA_TV_SEMANTIC_FIT_MAX_RETRIES", "1")
            )

            def report_provider_event(event: dict[str, Any]) -> None:
                kind = str(event.get("event") or "")
                if kind not in {
                    "request_attempt",
                    "retry_backoff",
                    "response_usage",
                    "output_budget_adjusted",
                }:
                    return
                if kind == "request_attempt":
                    message = (
                        "GPT batch request "
                        f"{event.get('attempt')}/{event.get('max_retries')}"
                    )
                elif kind == "retry_backoff":
                    message = (
                        "GPT batch retry in "
                        f"{event.get('delay_seconds')}s ({event.get('reason')})"
                    )
                elif kind == "response_usage":
                    reasoning_tokens = int(event.get("thinking_tokens") or 0)
                    message = (
                        "GPT batch response: "
                        f"{int(event.get('output_tokens') or 0):,} output tokens"
                    )
                    if reasoning_tokens:
                        message += f" ({reasoning_tokens:,} reasoning)"
                else:
                    message = (
                        "GPT output budget raised from "
                        f"{int(event.get('requested_max_output_tokens') or 0):,} "
                        "to "
                        f"{int(event.get('effective_max_output_tokens') or 0):,} tokens"
                    )
                progress.emit(
                    "semantic_fit_provider",
                    message,
                    status=("warning" if kind == "retry_backoff" else "running"),
                )

            provider = provider_view(
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                event_callback=report_provider_event,
            )

        rate_limit_retries = max(
            0,
            int(
                os.getenv(
                    "MATHULA_TV_SEMANTIC_FIT_RATE_LIMIT_RETRIES",
                    "5",
                )
            ),
        )
        rate_limit_wait_budget = max(
            0.0,
            float(
                os.getenv(
                    "MATHULA_TV_SEMANTIC_FIT_RATE_LIMIT_WAIT_BUDGET_SECONDS",
                    "360",
                )
            ),
        )
        rate_limit_sleep = getattr(provider, "sleep", time.sleep)
        if not callable(rate_limit_sleep):
            rate_limit_sleep = time.sleep

        def call_with_rate_limit_resume(
            operation: str,
            call: Callable[[Mapping[str, Any]], Any],
            payload: Mapping[str, Any],
        ) -> Any:
            total_wait = 0.0
            for retry_number in range(rate_limit_retries + 1):
                try:
                    return call(payload)
                except (AIRateLimit, ClaudeRateLimit) as exc:
                    if retry_number >= rate_limit_retries:
                        raise
                    raw_hint = (exc.details or {}).get("retry_after_seconds")
                    try:
                        hinted_wait = float(raw_hint)
                    except (TypeError, ValueError):
                        hinted_wait = 0.0
                    fallback_wait = min(15.0 * (2**retry_number), 90.0)
                    delay = max(0.1, hinted_wait or fallback_wait)
                    if total_wait + delay > rate_limit_wait_budget:
                        raise
                    total_wait += delay
                    progress.emit(
                        "semantic_fit_provider",
                        (
                            f"GPT {operation} rate limited; completed progress "
                            f"is checkpointed, retrying in {delay:g}s "
                            f"({retry_number + 1}/{rate_limit_retries})"
                        ),
                        status="warning",
                        rate_limit_retry=retry_number + 1,
                        delay_seconds=delay,
                        accumulated_wait_seconds=total_wait,
                    )
                    rate_limit_sleep(delay)
            raise AssertionError("unreachable semantic-fit rate-limit loop")

        artifacts.semantic_fit.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = (
            read_json(artifacts.semantic_fit)
            if artifacts.semantic_fit.is_file()
            else {
                "schema_version": SEMANTIC_FIT_SCHEMA_VERSION,
                "job_id": artifacts.root.parent.name,
                "cases": {},
            }
        )
        if checkpoint.get("schema_version") != SEMANTIC_FIT_SCHEMA_VERSION:
            checkpoint = {
                "schema_version": SEMANTIC_FIT_SCHEMA_VERSION,
                "job_id": artifacts.root.parent.name,
                "cases": {},
            }
        checkpoint["strategy_version"] = SEMANTIC_FIT_BATCH_STRATEGY_VERSION
        checkpoint["validator_version"] = SEMANTIC_FIT_VALIDATOR_VERSION
        cases = checkpoint.setdefault("cases", {})
        batch_rounds = checkpoint.setdefault("batch_rounds", [])
        results: dict[
            int, tuple[SpeechBlock, dict[str, Any], list[dict[str, Any]]]
        ] = {}
        states: dict[int, dict[str, Any]] = {}

        def select_natural(
            index: int,
            block: SpeechBlock,
            fitted: dict[str, Any],
            audits: list[dict[str, Any]],
            band: tuple[int, int, int],
            *,
            original_block: SpeechBlock | None = None,
            transition_borrow_limit_ms: int = 0,
        ) -> None:
            minimum, preferred, maximum = band
            original = original_block or block
            measured_duration_ms = _selected_azure_measurement_ms(fitted) or 0
            word_timed = bool(
                original.source_timing_evidence == "azure_word_timestamps"
                and original.source_speech_end_ms is not None
            )
            if word_timed:
                selected_block, boundary_alignment = (
                    self._center_word_timed_delivery(
                        block,
                        measured_duration_ms,
                        min(
                            options.semantic_fit_tolerance_ms,
                            options.source_word_alignment_tolerance_ms,
                        ),
                    )
                )
                transition_borrow_used_ms = 0
            else:
                boundary_alignment = None
                transition_borrow_used_ms = min(
                    max(0, int(transition_borrow_limit_ms)),
                    max(
                        0,
                        original.start_ms
                        + measured_duration_ms
                        - original.end_ms,
                    ),
                )
                selected_block = replace(
                    block,
                    end_ms=original.end_ms + transition_borrow_used_ms,
                )
            fitted["translation_variant"] = {
                "schema_version": "mathula-selected-translation-variant-v1",
                "source_block_id": _source_block_id(block.block_id),
                "selected_variant_id": "approved_natural",
                "spoken_text": block.translated_text,
                "measured_duration_ms": measured_duration_ms,
                "selection_reason": (
                    "semantic_fit_approved_text_fits_same_speaker_transition"
                    if transition_borrow_used_ms
                    else "semantic_fit_approved_text_already_fits"
                ),
                "human_review_required": False,
            }
            fitted["semantic_fit"] = {
                "schema_version": SEMANTIC_FIT_SCHEMA_VERSION,
                "strategy_version": SEMANTIC_FIT_BATCH_STRATEGY_VERSION,
                "selected_attempt": 0,
                "speaker_tempo_locked": True,
                "time_stretch_used": False,
                "same_speaker_transition_borrow_ms": transition_borrow_used_ms,
                "same_speaker_transition_borrow_limit_ms": (
                    transition_borrow_limit_ms
                ),
                "original_block_end_ms": original.end_ms,
                "search_block_end_ms": block.end_ms,
                "selected_block_end_ms": selected_block.end_ms,
                "target_duration_band_ms": {
                    "minimum": minimum,
                    "preferred": preferred,
                    "maximum": maximum,
                },
            }
            fitted["start_ms"] = selected_block.start_ms
            fitted["end_ms"] = selected_block.end_ms
            if boundary_alignment is not None:
                fitted["source_lipsync_endpoint"] = boundary_alignment
                fitted["semantic_fit"]["word_boundary_alignment"] = (
                    boundary_alignment
                )
            fitted["cadence_plan"] = plan_phrase_cadence(
                block_id=selected_block.block_id,
                text=selected_block.tts_text,
                start_ms=selected_block.start_ms,
                end_ms=selected_block.end_ms,
            ).to_dict()
            boundary = fitted.get("exact_audio_boundary")
            if isinstance(boundary, dict):
                boundary["target_duration_ms"] = selected_block.duration_ms
            fitted["translation_variant_attempts"] = audits
            results[index] = (selected_block, fitted, audits)
            existing_case = cases.get(_source_block_id(block.block_id))
            if isinstance(existing_case, dict):
                existing_case["status"] = "selected"
                existing_case["selected_attempt"] = 0
                existing_case["selected_variant_id"] = "approved_natural"
                existing_case["selection_reason"] = fitted[
                    "translation_variant"
                ]["selection_reason"]
            atomic_write_json(artifacts.semantic_fit, checkpoint)

        def select_candidate(
            index: int,
            candidate_block: SpeechBlock,
            fitted: dict[str, Any],
            audit: dict[str, Any],
            review: Mapping[str, Any],
        ) -> None:
            state = states[index]
            attempt = int(audit["attempt"])
            original_block = state["original_block"]
            measured_duration_ms = int(audit["measured_duration_ms"])
            word_timed = bool(
                original_block.source_timing_evidence
                == "azure_word_timestamps"
                and original_block.source_speech_end_ms is not None
            )
            if word_timed:
                selected_block, boundary_alignment = (
                    self._center_word_timed_delivery(
                        candidate_block,
                        measured_duration_ms,
                        state["alignment_tolerance_ms"],
                    )
                )
                transition_borrow_used_ms = 0
            else:
                boundary_alignment = None
                transition_borrow_used_ms = min(
                    state["same_speaker_transition_borrow_limit_ms"],
                    max(
                        0,
                        original_block.start_ms
                        + measured_duration_ms
                        - original_block.end_ms,
                    ),
                )
                selected_block = replace(
                    candidate_block,
                    end_ms=original_block.end_ms + transition_borrow_used_ms,
                )
            audit["status"] = "selected"
            audit["same_speaker_transition_borrow_used_ms"] = (
                transition_borrow_used_ms
            )
            audit["semantic_review"] = dict(review)
            cache = audit.pop("_cache")
            cache.update(audit)
            case = state["case"]
            case["status"] = "selected"
            case["selected_attempt"] = attempt
            fitted["translation_variant"] = {
                "schema_version": "mathula-selected-translation-variant-v1",
                "source_block_id": _source_block_id(candidate_block.block_id),
                "selected_variant_id": f"semantic_fit_{attempt:02d}",
                "spoken_text": candidate_block.translated_text,
                "measured_duration_ms": audit["measured_duration_ms"],
                "omitted_optional_details": audit["omitted_optional_details"],
                "meaning_preserved": True,
                "selection_reason": (
                    "semantic_fit_global_batch_azure_measured_and_ai_reviewed"
                ),
                "human_review_required": False,
            }
            fitted["semantic_fit"] = {
                "schema_version": SEMANTIC_FIT_SCHEMA_VERSION,
                "strategy_version": SEMANTIC_FIT_BATCH_STRATEGY_VERSION,
                "selected_attempt": attempt,
                "speaker_tempo_locked": True,
                "time_stretch_used": False,
                "same_speaker_transition_borrow_ms": transition_borrow_used_ms,
                "same_speaker_transition_borrow_limit_ms": state[
                    "same_speaker_transition_borrow_limit_ms"
                ],
                "original_block_end_ms": state["original_block"].end_ms,
                "search_block_end_ms": state["block"].end_ms,
                "selected_block_end_ms": selected_block.end_ms,
                "semantic_review": dict(review),
                "target_duration_band_ms": {
                    "minimum": state["minimum"],
                    "preferred": state["preferred"],
                    "maximum": state["maximum"],
                },
            }
            if original_block.performance_region_id:
                fitted["performance_plan"] = {
                    "schema_version": PERFORMANCE_PLAN_VERSION,
                    "region_id": original_block.performance_region_id,
                    "member_block_ids": list(
                        original_block.performance_region_member_ids
                    ),
                    "selected_attempt": attempt,
                    "performance_beats": list(
                        audit.get("performance_beats") or []
                    ),
                    "target_island_texts": list(
                        candidate_block.semantic_target_islands
                    ),
                    "source_island_ids": [
                        str(value.get("island_id") or "")
                        for value in original_block.source_speech_islands
                    ],
                    "stt_member_boundaries_used_as_linguistic_boundaries": False,
                    "speaker_tempo_locked": True,
                    "time_stretch_used": False,
                }
            fitted["start_ms"] = selected_block.start_ms
            fitted["end_ms"] = selected_block.end_ms
            if boundary_alignment is not None:
                fitted["source_lipsync_endpoint"] = boundary_alignment
                fitted["semantic_fit"]["word_boundary_alignment"] = (
                    boundary_alignment
                )
            fitted["cadence_plan"] = plan_phrase_cadence(
                block_id=selected_block.block_id,
                text=selected_block.tts_text,
                start_ms=selected_block.start_ms,
                end_ms=selected_block.end_ms,
            ).to_dict()
            boundary = fitted.get("exact_audio_boundary")
            if isinstance(boundary, dict):
                boundary["target_duration_ms"] = selected_block.duration_ms
            fitted["translation_variant_attempts"] = state["audits"]
            results[index] = (
                selected_block,
                fitted,
                state["audits"],
            )
            atomic_write_json(artifacts.semantic_fit, checkpoint)
            progress.emit(
                "semantic_fit",
                (
                    f"{state['block'].block_id}: selected batched attempt "
                    f"{attempt} at locked rate {state['base_rate']:+d}%"
                ),
                status="completed",
                current=len(results),
                total=len(blocks),
                block_id=state["block"].block_id,
                attempt=attempt,
                measured_duration_ms=audit["measured_duration_ms"],
            )

        def evaluate_candidate(
            index: int,
            candidate_data: Mapping[str, Any],
            attempt: int,
            cache: dict[str, Any],
            *,
            ai_called: bool,
        ) -> dict[str, Any] | None:
            state = states[index]
            block = state["block"]
            try:
                spoken, tts, omissions = self._validate_finalization_candidate(
                    block, candidate_data
                )
            except ValueError as exc:
                audit = {
                    "attempt": attempt,
                    "status": "local_semantic_rejection",
                    "reason": str(exc),
                    "ai_called": ai_called,
                }
                state["audits"].append(audit)
                state["evidence"].append(audit)
                cache.update(audit)
                return None
            tts, pronunciation_substitutions = (
                self._apply_candidate_pronunciation(
                    spoken,
                    tts,
                    pronunciation_dictionary,
                )
            )
            try:
                spoken_islands = self._semantic_target_islands(
                    block,
                    candidate_data,
                    spoken,
                )
            except ValueError as exc:
                audit = {
                    "attempt": attempt,
                    "status": "local_semantic_rejection",
                    "reason": str(exc),
                    "ai_called": ai_called,
                }
                state["audits"].append(audit)
                state["evidence"].append(audit)
                cache.update(audit)
                return None
            tts_islands = tuple(
                self._apply_candidate_pronunciation(
                    value,
                    value,
                    pronunciation_dictionary,
                )[0]
                for value in spoken_islands
            )
            normalized = " ".join(spoken.casefold().split())
            if normalized in state["rejected_normalized"]:
                audit = {
                    "attempt": attempt,
                    "status": "duplicate_rejection",
                    "spoken_text": spoken,
                    "ai_called": ai_called,
                }
                state["audits"].append(audit)
                state["evidence"].append(audit)
                cache.update(audit)
                return None
            variant = PreparedTranslationVariant(
                variant_id=f"semantic_fit_{attempt:02d}",
                spoken_text=spoken,
                tts_text=tts,
                target_duration_ms=state["preferred"],
                estimated_duration_ms=None,
                meaning_preserved=True,
                omitted_optional_details=omissions,
            )
            candidate_block = replace(
                block,
                block_id=(
                    f"{_source_block_id(block.block_id)}_variant_"
                    f"{variant.variant_id}"
                ),
                translated_text=spoken,
                tts_text=tts,
                variants=(*block.variants, variant),
                selected_variant_id=variant.variant_id,
                semantic_target_islands=tts_islands,
            )
            try:
                fitted = self._synthesize_locked_tempo_block(
                    candidate_block,
                    artifacts,
                    voice=state["voice"],
                    base_rate_percent=state["base_rate"],
                    pitch_percent=state["pitch"],
                    volume_percent=state["volume"],
                    force=False,
                    timing_deadline_ms=state["maximum"],
                    timing_preferred_ms=state["preferred"],
                )
                measured = _selected_azure_measurement_ms(fitted) or 0
                pause_alignment = fitted.get("source_pause_alignment") or {}
                plain_measured = int(
                    pause_alignment.get("plain_locked_tempo_duration_ms")
                    or measured
                )
                effective_pause_ms = max(0, measured - plain_measured)
                status = (
                    "timing_fit"
                    if state["minimum"] <= measured <= state["maximum"]
                    else "underfill"
                )
            except TimingOverflowError as exc:
                fitted = None
                measured = exc.measured_duration_ms
                plain_measured = int(exc.plain_duration_ms or measured)
                effective_pause_ms = int(
                    exc.effective_inserted_pause_ms
                    if exc.effective_inserted_pause_ms is not None
                    else max(0, measured - plain_measured)
                )
                status = "overflow"
            performance_island_measurements: list[dict[str, Any]] = []
            performance_island_alignment_fit = True
            if (
                status == "timing_fit"
                and fitted is not None
                and block.performance_region_id
            ):
                (
                    performance_island_alignment_fit,
                    performance_island_measurements,
                ) = (
                    self._measure_performance_islands(
                        candidate_block,
                        artifacts,
                        island_texts=tts_islands,
                        voice=state["voice"],
                        base_rate_percent=state["base_rate"],
                        pitch_percent=state["pitch"],
                        volume_percent=state["volume"],
                        tolerance_ms=state["alignment_tolerance_ms"],
                    )
                )
                if not performance_island_alignment_fit:
                    status = "internal_island_drift"
            estimated_plain_ms = int(
                cache.get("estimated_plain_speech_ms") or 0
            )
            audit = {
                "attempt": attempt,
                "strategy_attempt": cache.get("strategy_attempt"),
                "strategy_version": cache.get("strategy_version"),
                "batch_round": cache.get("batch_round"),
                "status": status,
                "spoken_text": spoken,
                "tts_text": tts,
                "pronunciation_substitutions": pronunciation_substitutions,
                "omitted_optional_details": list(omissions),
                "measured_duration_ms": measured,
                "plain_speech_duration_ms": plain_measured,
                "effective_inserted_pause_ms": effective_pause_ms,
                "estimated_plain_speech_ms": estimated_plain_ms or None,
                "plain_duration_estimation_error_ms": (
                    plain_measured - estimated_plain_ms
                    if estimated_plain_ms
                    else None
                ),
                "target_minimum_ms": state["minimum"],
                "target_preferred_ms": state["preferred"],
                "target_maximum_ms": state["maximum"],
                "speaker_rate_percent": state["base_rate"],
                "time_stretch_ratio": 1.0,
                "same_speaker_transition_borrow_ms": state[
                    "same_speaker_transition_borrow_limit_ms"
                ],
                "ai_called": ai_called,
                "_cache": cache,
            }
            if block.performance_region_id:
                unit = candidate_data.get("unit")
                audit["performance_beats"] = [
                    dict(value)
                    for value in (
                        unit.get("performance_beats")
                        if isinstance(unit, Mapping)
                        else []
                    )
                    if isinstance(value, Mapping)
                ]
                audit["target_island_texts"] = list(spoken_islands)
                audit["performance_island_measurements"] = (
                    performance_island_measurements
                )
                if performance_island_measurements:
                    audit["performance_island_alignment"] = (
                        self._performance_island_alignment_summary(
                            performance_island_measurements
                        )
                    )
                    audit["performance_island_alignment"].update(
                        {
                            "hard_gate_passed": (
                                performance_island_alignment_fit
                            ),
                            "maximum_allowed_cumulative_drift_ms": min(
                                MAXIMUM_INTERNAL_ISLAND_DRIFT_MS,
                                max(
                                    1,
                                    int(options.semantic_fit_tolerance_ms),
                                ),
                            ),
                        }
                    )
            state["audits"].append(audit)
            state["evidence"].append(
                {key: value for key, value in audit.items() if key != "_cache"}
            )
            state["current_text"] = spoken
            state["rejected"].append(spoken)
            state["rejected_normalized"].add(normalized)
            cache.update({key: value for key, value in audit.items() if key != "_cache"})
            if status != "timing_fit" or fitted is None:
                audit.pop("_cache", None)
                return None
            return {
                "index": index,
                "candidate_block": candidate_block,
                "fitted": fitted,
                "audit": audit,
                "review_payload": self._semantic_fit_review_payload(
                    block,
                    spoken_text=spoken,
                    tts_text=tts,
                    omitted_details=omissions,
                    measured_duration_ms=measured,
                ),
            }

        # Azure measurement is cheap compared with provider turns. Measure all
        # approved blocks first so one GPT request can serve every miss.
        for index, block in enumerate(blocks):
            voice, base_rate, pitch, volume = synthesis_profiles[index]
            alignment_tolerance_ms = (
                min(
                    options.semantic_fit_tolerance_ms,
                    options.source_word_alignment_tolerance_ms,
                )
                if block.source_timing_evidence == "azure_word_timestamps"
                else options.semantic_fit_tolerance_ms
            )
            minimum, preferred, maximum = self._semantic_fit_target_band(
                block, alignment_tolerance_ms
            )
            transition_borrow_ms = self._same_speaker_transition_borrow_limit(
                blocks,
                index,
            )
            search_block = (
                replace(block, end_ms=block.end_ms + transition_borrow_ms)
                if transition_borrow_ms
                else block
            )
            search_maximum = maximum + transition_borrow_ms
            audits: list[dict[str, Any]] = []
            try:
                fitted = self._synthesize_locked_tempo_block(
                    block,
                    artifacts,
                    voice=voice,
                    base_rate_percent=base_rate,
                    pitch_percent=pitch,
                    volume_percent=volume,
                    force=force,
                    timing_deadline_ms=maximum,
                    timing_preferred_ms=preferred,
                )
                measured = _selected_azure_measurement_ms(fitted) or 0
                initial_pause_alignment = fitted.get("source_pause_alignment") or {}
                initial_plain_measured = int(
                    initial_pause_alignment.get("plain_locked_tempo_duration_ms")
                    or measured
                )
                status = (
                    "selected" if minimum <= measured <= maximum else "underfill"
                )
            except TimingOverflowError as exc:
                fitted = None
                measured = exc.measured_duration_ms
                initial_plain_measured = int(exc.plain_duration_ms or measured)
                status = "overflow"
            initial = {
                "attempt": 0,
                "candidate_id": "approved_natural",
                "status": status,
                "spoken_text": block.translated_text,
                "tts_text": block.tts_text,
                "measured_duration_ms": measured,
                "plain_speech_duration_ms": initial_plain_measured,
                "target_minimum_ms": minimum,
                "target_preferred_ms": preferred,
                "target_maximum_ms": maximum,
                "speaker_rate_percent": base_rate,
                "time_stretch_ratio": 1.0,
                "semantic_review": "approved_translation",
            }
            performance_plan_required = bool(block.performance_region_id)
            if performance_plan_required and status == "selected":
                initial["status"] = "performance_plan_required"
            audits.append(initial)
            if (
                status == "selected"
                and fitted is not None
                and not performance_plan_required
            ):
                select_natural(index, block, fitted, audits, (minimum, preferred, maximum))
                continue
            if transition_borrow_ms and not performance_plan_required:
                try:
                    transition_fitted = self._synthesize_locked_tempo_block(
                        search_block,
                        artifacts,
                        voice=voice,
                        base_rate_percent=base_rate,
                        pitch_percent=pitch,
                        volume_percent=volume,
                        force=False,
                        timing_deadline_ms=search_maximum,
                        timing_preferred_ms=preferred,
                    )
                    transition_measured = (
                        _selected_azure_measurement_ms(transition_fitted) or 0
                    )
                    transition_status = (
                        "selected"
                        if minimum <= transition_measured <= search_maximum
                        else "underfill"
                    )
                except TimingOverflowError as exc:
                    transition_fitted = None
                    transition_measured = exc.measured_duration_ms
                    transition_status = "overflow"
                transition_audit = {
                    "attempt": 0,
                    "candidate_id": "approved_natural_transition_window",
                    "status": transition_status,
                    "spoken_text": block.translated_text,
                    "measured_duration_ms": transition_measured,
                    "target_minimum_ms": minimum,
                    "target_preferred_ms": preferred,
                    "target_maximum_ms": search_maximum,
                    "same_speaker_transition_borrow_limit_ms": (
                        transition_borrow_ms
                    ),
                    "speaker_rate_percent": base_rate,
                    "time_stretch_ratio": 1.0,
                    "semantic_review": "approved_translation",
                }
                audits.append(transition_audit)
                if transition_status == "selected" and transition_fitted is not None:
                    select_natural(
                        index,
                        search_block,
                        transition_fitted,
                        audits,
                        (minimum, preferred, search_maximum),
                        original_block=block,
                        transition_borrow_limit_ms=transition_borrow_ms,
                    )
                    continue
            initial["same_speaker_transition_borrow_ms"] = transition_borrow_ms
            initial["original_block_end_ms"] = block.end_ms
            initial["search_block_end_ms"] = search_block.end_ms
            block_id = _source_block_id(block.block_id)
            case = cases.setdefault(
                block_id,
                {
                    "speaker_id": block.speaker_id,
                    "voice": voice,
                    "speaker_rate_percent": base_rate,
                    "attempts": [],
                },
            )
            prior_case_rate_percent = int(
                case.get("speaker_rate_percent")
                if case.get("speaker_rate_percent") is not None
                else base_rate
            )
            # Speaker tempo is part of the acoustic contract.  A cached case
            # may have been generated before source-tempo calibration (often
            # at Azure's neutral 0% rate); keep its text for audit but never
            # present that old rate as the active contract.
            case["speaker_id"] = block.speaker_id
            case["voice"] = voice
            case["speaker_rate_percent"] = base_rate
            if case.get("strategy_version") != SEMANTIC_FIT_BATCH_STRATEGY_VERSION:
                case["strategy_attempt_base"] = max(
                    (
                        int(value.get("attempt") or 0)
                        for value in case.get("attempts") or []
                        if isinstance(value, Mapping)
                    ),
                    default=0,
                )
                case["strategy_version"] = SEMANTIC_FIT_BATCH_STRATEGY_VERSION
            states[index] = {
                "block": search_block,
                "original_block": block,
                "voice": voice,
                "base_rate": base_rate,
                "pitch": pitch,
                "volume": volume,
                "minimum": minimum,
                "preferred": preferred,
                "maximum": search_maximum,
                "alignment_tolerance_ms": alignment_tolerance_ms,
                "same_speaker_transition_borrow_limit_ms": transition_borrow_ms,
                "audits": audits,
                "evidence": [initial],
                "rejected": (
                    []
                    if performance_plan_required and status == "selected"
                    else [block.translated_text]
                ),
                "rejected_normalized": (
                    set()
                    if performance_plan_required and status == "selected"
                    else {" ".join(block.translated_text.casefold().split())}
                ),
                "current_text": block.translated_text,
                "case": case,
                "prior_case_rate_percent": prior_case_rate_percent,
                "strategy_attempt_base": int(
                    case.get("strategy_attempt_base") or 0
                ),
            }

        def review_pending(pending: list[dict[str, Any]]) -> None:
            if not pending:
                return
            request = {
                "schema_version": SEMANTIC_FIT_SCHEMA_VERSION,
                "operation": "semantic_fit_review_batch",
                "reviews": [item["review_payload"] for item in pending],
            }
            response = call_with_rate_limit_resume(
                "review",
                provider.review_semantic_fits,  # type: ignore[attr-defined]
                request,
            )
            data = response.data if hasattr(response, "data") else response
            returned = data.get("reviews") if isinstance(data, Mapping) else None
            if not isinstance(returned, list):
                raise DirectDubError("Semantic-fit batch review returned no reviews")
            by_id = {
                str(item.get("unit_id") or ""): item.get("review")
                for item in returned
                if isinstance(item, Mapping)
            }
            for item in pending:
                index = item["index"]
                unit_id = _source_block_id(states[index]["block"].block_id)
                review = by_id.get(unit_id)
                if not isinstance(review, Mapping):
                    raise DirectDubError(
                        f"Semantic-fit batch review omitted {unit_id}"
                    )
                audit = item["audit"]
                cache = audit["_cache"]
                cache["review"] = dict(review)
                cache["review_response_batch"] = True
                if self._semantic_review_passed(review):
                    select_candidate(
                        index,
                        item["candidate_block"],
                        item["fitted"],
                        audit,
                        review,
                    )
                else:
                    audit["status"] = "semantic_review_rejection"
                    audit["semantic_review"] = dict(review)
                    audit.pop("_cache", None)
                    cache.update(audit)

        # First consume paid candidates already stored by the earlier serial
        # implementation. This makes the upgrade resumable in the middle of
        # block_0006 instead of throwing those calls away.
        cached_by_index: dict[int, list[dict[str, Any]]] = {}
        for index, state in states.items():
            cached_by_index[index] = sorted(
                (
                    value
                    for value in state["case"].get("attempts") or []
                    if isinstance(value, dict)
                    and isinstance(value.get("candidate"), Mapping)
                    and int(
                        value.get("speaker_rate_percent")
                        if value.get("speaker_rate_percent") is not None
                        else state["prior_case_rate_percent"]
                    )
                    == int(state["base_rate"])
                ),
                key=lambda value: int(value.get("attempt") or 0),
            )
        cached_positions = {index: 0 for index in cached_by_index}
        while True:
            cached_pending: list[dict[str, Any]] = []
            progressed = False
            for index, attempts in cached_by_index.items():
                if index in results:
                    continue
                while cached_positions[index] < len(attempts):
                    cache = attempts[cached_positions[index]]
                    cached_positions[index] += 1
                    progressed = True
                    attempt = int(cache.get("attempt") or 0)
                    if attempt < 1:
                        continue
                    pending = evaluate_candidate(
                        index,
                        cache["candidate"],
                        attempt,
                        cache,
                        ai_called=False,
                    )
                    if pending is None:
                        continue
                    review = cache.get("review")
                    if isinstance(review, Mapping):
                        if self._semantic_review_passed(review):
                            select_candidate(
                                index,
                                pending["candidate_block"],
                                pending["fitted"],
                                pending["audit"],
                                review,
                            )
                            break
                        audit = pending["audit"]
                        audit["status"] = "semantic_review_rejection"
                        audit["semantic_review"] = dict(review)
                        audit.pop("_cache", None)
                        cache.update(audit)
                        continue
                    cached_pending.append(pending)
                    break
            review_pending(
                [item for item in cached_pending if item["index"] not in results]
            )
            atomic_write_json(artifacts.semantic_fit, checkpoint)
            if not progressed:
                break

        portfolio_method = getattr(provider, "fit_semantic_portfolios", None)
        portfolio_supported = callable(portfolio_method)
        localized_phrase_method = getattr(
            provider, "repair_localized_breath_groups", None
        )
        localized_phrase_supported = callable(localized_phrase_method)
        round_number = len(batch_rounds)

        def strategy_attempt_limit(state: Mapping[str, Any]) -> int:
            # Production semantic-fit is latency bounded. One portfolio round
            # gets three structurally different alternatives per unresolved
            # region; after that, the reviewed best frontier is handed to the
            # existing final-product cadence conductor. Repeated LLM portfolio
            # rounds are expensive (large performance-beat JSON) and showed
            # sharply diminishing returns on short-form jobs.
            #
            # Explicit performance-plan mode remains the strict research/QA
            # path and continues to honour the caller's full measured-feedback
            # budget.
            if options.timing_mode == TIMING_MODE_SEMANTIC_FIT:
                # One LLM dialogue-adaptor portfolio only. Three linguistic
                # alternatives provide enough structural diversity for a hard
                # line. The cheap preflight records predicted duration and
                # mouth-active distribution, then Azure measures all three at
                # the locked source-speaker tempo. No repeated provider loop.
                # Two bounded dialogue-adaptor rounds maximum: the first
                # generates three structurally different deliveries; if none
                # passes Azure timing + breath-group alignment, the second
                # gets the measured frontier and generates three targeted
                # redistributions.  This preserves closed-loop quality without
                # returning to the unbounded portfolio loops used by older
                # semantic-fit strategies.
                return min(
                    options.semantic_fit_max_attempts,
                    DIALOGUE_ADAPTOR_PORTFOLIO_SIZE * 2,
                )
            if options.timing_mode == TIMING_MODE_PERFORMANCE_PLAN:
                return min(options.semantic_fit_max_attempts, 4)
            return options.semantic_fit_max_attempts

        def strategy_attempts_used(state: Mapping[str, Any]) -> int:
            """Count only attempts made by the active strategy version.

            Absolute ``attempt`` numbers are intentionally monotonic across
            upgrades so cached acoustic evidence remains auditable. They are
            not a retry budget. A new strategy must receive a fresh budget,
            while an interrupted rerun of the same strategy must resume from
            the exact ``strategy_attempt`` already persisted.
            """
            return max(
                (
                    int(value.get("strategy_attempt") or 0)
                    for value in state["case"].get("attempts") or []
                    if isinstance(value, Mapping)
                    and value.get("strategy_version")
                    == SEMANTIC_FIT_BATCH_STRATEGY_VERSION
                ),
                default=0,
            )

        def dialogue_adaptor_preflight(
            index: int,
            candidate_data: Mapping[str, Any],
        ) -> dict[str, Any]:
            """Rank one GPT rephrase before paying for Azure synthesis.

            This layer never accepts a candidate. It predicts duration and
            mouth-active distribution at the immutable source-calibrated
            speaker rate. Semantic validation and Azure timing remain
            authoritative after the shortlist.
            """

            state = states[index]
            block = state["block"]
            try:
                spoken, tts, _omissions = self._validate_finalization_candidate(
                    block, candidate_data
                )
                tts, _pronunciation_substitutions = (
                    self._apply_candidate_pronunciation(
                        spoken,
                        tts,
                        pronunciation_dictionary,
                    )
                )
                spoken_islands = self._semantic_target_islands(
                    block,
                    candidate_data,
                    spoken,
                )
            except ValueError as exc:
                return {
                    "version": DIALOGUE_ADAPTOR_PREFLIGHT_VERSION,
                    "valid": False,
                    "reason": str(exc),
                    "score": float("inf"),
                }

            normalized = " ".join(spoken.casefold().split())
            if normalized in state["rejected_normalized"]:
                return {
                    "version": DIALOGUE_ADAPTOR_PREFLIGHT_VERSION,
                    "valid": False,
                    "reason": "duplicate_rejection",
                    "score": float("inf"),
                    "spoken_text": spoken,
                }

            tts_islands = tuple(
                self._apply_candidate_pronunciation(
                    value,
                    value,
                    pronunciation_dictionary,
                )[0]
                for value in spoken_islands
            )
            initial = state["evidence"][0] if state["evidence"] else {}
            try:
                fallback_plain_ms = int(
                    initial.get("plain_speech_duration_ms")
                    or initial.get("measured_duration_ms")
                    or block.duration_ms
                )
            except (TypeError, ValueError):
                fallback_plain_ms = block.duration_ms
            ms_per_syllable = _dialogue_adaptor_ms_per_syllable(
                state["evidence"],
                fallback_text=block.tts_text,
                fallback_plain_duration_ms=fallback_plain_ms,
            )
            target_syllables = max(1, _estimated_zulu_syllables(tts))
            predicted_plain_ms = max(1, round(target_syllables * ms_per_syllable))
            mandatory_island_pause_ms = sum(
                max(0, int(value.get("requested_pause_ms") or 0))
                for value in block.source_pause_anchors
                if value.get("pause_kind") == "island_boundary"
            )
            minimum_plain_ms = max(
                1, int(state["minimum"]) - mandatory_island_pause_ms
            )
            preferred_plain_ms = max(
                1, int(state["preferred"]) - mandatory_island_pause_ms
            )
            maximum_plain_ms = max(
                1, int(state["maximum"]) - mandatory_island_pause_ms
            )
            predicted_island_drift_ms = (
                _dialogue_adaptor_island_drift_ms(
                    tts_islands,
                    block.source_speech_islands,
                )
                if block.performance_region_id
                else 0
            )
            score = _dialogue_adaptor_preflight_score(
                predicted_plain_ms=predicted_plain_ms,
                preferred_plain_ms=preferred_plain_ms,
                minimum_plain_ms=minimum_plain_ms,
                maximum_plain_ms=maximum_plain_ms,
                predicted_island_drift_ms=predicted_island_drift_ms,
            )
            return {
                "version": DIALOGUE_ADAPTOR_PREFLIGHT_VERSION,
                "valid": True,
                "score": round(score, 3),
                "spoken_text": spoken,
                "target_syllables": target_syllables,
                "locked_rate_ms_per_syllable": round(ms_per_syllable, 3),
                "predicted_plain_speech_ms": predicted_plain_ms,
                "target_plain_speech_band_ms": {
                    "minimum": minimum_plain_ms,
                    "preferred": preferred_plain_ms,
                    "maximum": maximum_plain_ms,
                },
                "predicted_island_cumulative_drift_ms": (
                    predicted_island_drift_ms
                ),
                "source_tempo_locked": True,
                "azure_measurement_authoritative": True,
            }

        while portfolio_supported:
            eligible: list[tuple[int, int]] = []
            for index, state in states.items():
                if index in results:
                    continue
                prior_attempt = max(
                    (
                        int(value.get("attempt") or 0)
                        for value in state["case"].get("attempts") or []
                        if isinstance(value, Mapping)
                    ),
                    default=0,
                )
                strategy_used = strategy_attempts_used(state)
                remaining = strategy_attempt_limit(state) - strategy_used
                if remaining > 0:
                    eligible.append((index, remaining))
            if options.timing_mode == TIMING_MODE_PERFORMANCE_PLAN:
                # Performance-beat JSON is substantially larger than a plain
                # block rewrite. Schedule at most three provider candidates
                # per request, prioritising regions that have not yet received
                # their first holistic plan. The existing outer loop writes a
                # checkpoint after every request, so a later provider failure
                # cannot discard completed paid shards.
                def performance_schedule_key(
                    value: tuple[int, int],
                ) -> tuple[int, int]:
                    candidate_index = value[0]
                    candidate_state = states[candidate_index]
                    prior = max(
                        (
                            int(item.get("attempt") or 0)
                            for item in candidate_state["case"].get(
                                "attempts"
                            )
                            or []
                            if isinstance(item, Mapping)
                        ),
                        default=0,
                    )
                    used = strategy_attempts_used(candidate_state)
                    return (used, candidate_index)

                scheduled: list[tuple[int, int]] = []
                remaining_candidate_slots = (
                    PERFORMANCE_PLAN_PROVIDER_CANDIDATES_PER_SHARD
                )
                for candidate in sorted(
                    eligible,
                    key=performance_schedule_key,
                ):
                    _candidate_index, _candidate_remaining = candidate
                    # One measured revision per region. Parallel alternatives
                    # generated from the same stale duration are expensive
                    # guesses and cannot exploit the Azure result.
                    planned_count = 1
                    if scheduled and planned_count > remaining_candidate_slots:
                        continue
                    scheduled.append(candidate)
                    remaining_candidate_slots -= min(
                        remaining_candidate_slots,
                        planned_count,
                    )
                    if remaining_candidate_slots <= 0:
                        break
                eligible = scheduled
            active: list[int] = []
            payloads: list[dict[str, Any]] = []
            candidate_count_by_index: dict[int, int] = {}
            first_attempt_by_index: dict[int, int] = {}
            for index, remaining in eligible:
                state = states[index]
                prior_attempt = max(
                    (
                        int(value.get("attempt") or 0)
                        for value in state["case"].get("attempts") or []
                        if isinstance(value, Mapping)
                    ),
                    default=0,
                )
                strategy_used = strategy_attempts_used(state)
                performance_region = bool(
                    state["block"].performance_region_id
                )
                # Performance-plan is a closed-loop controller: synthesize one
                # revision, feed the measurement back, then revise again.
                # Word-anchored semantic-fit regions use two alternatives per
                # measured frontier. That halves provider round-trips while
                # still keeping every pair grounded in the same Azure feedback.
                # Legacy non-island semantic-fit retains its portfolio sizes.
                per_round_limit = (
                    1
                    if options.timing_mode == TIMING_MODE_PERFORMANCE_PLAN
                    else DIALOGUE_ADAPTOR_PORTFOLIO_SIZE
                )
                candidate_count = min(per_round_limit, remaining)
                first_attempt = prior_attempt + 1
                block_index = index
                neighbouring = (
                    *blocks[max(0, block_index - 2):block_index],
                    *blocks[block_index + 1:block_index + 3],
                )
                payload = self._semantic_fit_payload(
                    state["block"],
                    voice=state["voice"],
                    base_rate_percent=state["base_rate"],
                    pitch_percent=state["pitch"],
                    volume_percent=state["volume"],
                    attempt_number=strategy_used + 1,
                    target_minimum_ms=state["minimum"],
                    target_preferred_ms=state["preferred"],
                    target_maximum_ms=state["maximum"],
                    duration_evidence=state["evidence"],
                    rejected_spoken_texts=state["rejected"],
                    current_spoken_text=state["current_text"],
                    neighbouring_blocks=neighbouring,
                )
                payload["candidate_count"] = candidate_count
                payload["portfolio_contract"] = {
                    "candidate_order": (
                        "single_measured_revision"
                        if performance_region
                        else "fullest_likely_fit_to_shortest_faithful"
                    ),
                    "distinct_grammatical_contractions_required": (
                        not performance_region
                    ),
                    "every_candidate_must_preserve_every_required_fact": True,
                    "estimated_duration_excludes_inserted_ssml_pauses": True,
                    "azure_measured_frontier_is_authoritative": True,
                    "feedback_rounds_are_reserved_for_hard_final_blocks": True,
                    "performance_plan_candidate_count": (
                        candidate_count if performance_region else None
                    ),
                    "performance_beats_required": performance_region,
                    "stt_member_boundaries_are_advisory": performance_region,
                    "source_tempo_is_primary_and_immutable": True,
                    "dialogue_adaptor_preflight": {
                        "version": DIALOGUE_ADAPTOR_PREFLIGHT_VERSION,
                        "azure_tts_shortlist_count": min(
                            DIALOGUE_ADAPTOR_TTS_SHORTLIST,
                            candidate_count,
                        ),
                        "ranking_priorities": [
                            "locked_source_tempo_duration_fit",
                            "mouth_active_island_distribution",
                            "creative_semantic_rephrase",
                        ],
                        "candidate_roles": (
                            [
                                "measured_frontier_minimal_island_redistribution",
                                "measured_frontier_compact_island_preserving",
                                "measured_frontier_alternative_grammar_same_island_budget",
                            ]
                            if strategy_used > 0
                            else [
                                "natural_clause_fusion",
                                "morphology_and_recoverable_agreement",
                                "semantic_beat_redistribution_or_shortest_faithful_form",
                            ]
                        )[:candidate_count],
                    },
                }
                active.append(index)
                payloads.append(payload)
                candidate_count_by_index[index] = candidate_count
                first_attempt_by_index[index] = first_attempt
            if not active:
                break

            round_number += 1
            total_candidates = sum(candidate_count_by_index.values())
            unresolved_total = sum(
                1 for index in states if index not in results
            )
            progress.emit(
                "semantic_fit",
                (
                    f"portfolio round {round_number}: generating "
                    f"{total_candidates} candidates for {len(active)} "
                    f"scheduled region(s); {unresolved_total} total unresolved"
                ),
                status="running",
                current=len(results),
                total=len(blocks),
                round=round_number,
                unresolved_count=unresolved_total,
                scheduled_count=len(active),
                candidate_count=total_candidates,
                provider_candidate_shard_limit=(
                    PERFORMANCE_PLAN_PROVIDER_CANDIDATES_PER_SHARD
                    if options.timing_mode == TIMING_MODE_PERFORMANCE_PLAN
                    else None
                ),
            )
            batch_payload = {
                "schema_version": SEMANTIC_FIT_SCHEMA_VERSION,
                "operation": "semantic_fit_portfolio_batch",
                "round_number": round_number,
                "repairs": payloads,
            }
            fingerprint = hashlib.sha256(
                json.dumps(
                    batch_payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            response = call_with_rate_limit_resume(
                "candidate portfolio generation",
                portfolio_method,
                batch_payload,
            )
            data = response.data if hasattr(response, "data") else response
            portfolios = (
                data.get("portfolios") if isinstance(data, Mapping) else None
            )
            if not isinstance(portfolios, list):
                raise DirectDubError(
                    "Semantic-fit portfolio batch returned no portfolios"
                )
            by_id = {
                str(item.get("unit_id") or ""): item.get("candidates")
                for item in portfolios
                if isinstance(item, Mapping)
            }
            batch_rounds.append(
                {
                    "round": round_number,
                    "mode": "candidate_portfolio",
                    "request_fingerprint": fingerprint,
                    "unit_ids": [
                        _source_block_id(states[index]["block"].block_id)
                        for index in active
                    ],
                    "candidate_counts": {
                        _source_block_id(states[index]["block"].block_id): (
                            candidate_count_by_index[index]
                        )
                        for index in active
                    },
                    "response": (
                        response.to_artifact()
                        if hasattr(response, "to_artifact")
                        else None
                    ),
                }
            )
            pending_by_index: dict[int, list[dict[str, Any]]] = {}
            for index in active:
                state = states[index]
                unit_id = _source_block_id(state["block"].block_id)
                candidates = by_id.get(unit_id)
                expected_count = candidate_count_by_index[index]
                if not isinstance(candidates, list) or len(candidates) != expected_count:
                    raise DirectDubError(
                        f"Semantic-fit portfolio count differs for {unit_id}"
                    )
                pending_by_index[index] = []
                strategy_used_before = strategy_attempts_used(state)
                ranked_candidates: list[tuple[float, int, Mapping[str, Any], dict[str, Any]]] = []
                for offset, portfolio_candidate in enumerate(candidates):
                    if not isinstance(portfolio_candidate, Mapping):
                        raise DirectDubError(
                            f"Semantic-fit portfolio candidate invalid for {unit_id}"
                        )
                    candidate = portfolio_candidate.get("candidate")
                    if not isinstance(candidate, Mapping):
                        raise DirectDubError(
                            f"Semantic-fit portfolio omitted candidate for {unit_id}"
                        )
                    attempt = first_attempt_by_index[index] + offset
                    strategy_attempt = strategy_used_before + offset + 1
                    entry = {
                        "attempt": attempt,
                        "strategy_attempt": strategy_attempt,
                        "strategy_version": SEMANTIC_FIT_BATCH_STRATEGY_VERSION,
                        "batch_round": round_number,
                        "request_fingerprint": fingerprint,
                        "portfolio_candidate_id": str(
                            portfolio_candidate.get("candidate_id") or ""
                        ),
                        "estimated_plain_speech_ms": int(
                            portfolio_candidate.get(
                                "estimated_plain_speech_ms"
                            )
                            or 0
                        ),
                        "compression_strategy": str(
                            portfolio_candidate.get("compression_strategy") or ""
                        ),
                        "candidate": dict(candidate),
                    }
                    state["case"].setdefault("attempts", []).append(entry)
                    if options.timing_mode == TIMING_MODE_SEMANTIC_FIT:
                        preflight = dialogue_adaptor_preflight(index, candidate)
                        entry["dialogue_adaptor_preflight"] = preflight
                        ranked_candidates.append(
                            (
                                float(preflight.get("score", float("inf"))),
                                offset,
                                candidate,
                                entry,
                            )
                        )
                    else:
                        ranked_candidates.append(
                            (0.0, offset, candidate, entry)
                        )

                if options.timing_mode == TIMING_MODE_SEMANTIC_FIT:
                    ranked_candidates.sort(key=lambda value: (value[0], value[1]))
                    shortlist = ranked_candidates[: min(
                        DIALOGUE_ADAPTOR_TTS_SHORTLIST,
                        len(ranked_candidates),
                    )]
                    # The preflight chooses which alternatives deserve Azure
                    # measurement; among that shortlist preserve GPT's fuller-
                    # to-shorter portfolio order. If multiple candidates fit,
                    # the fuller natural construction is reviewed first.
                    shortlist.sort(key=lambda value: value[1])
                    shortlisted_entry_ids = {id(value[3]) for value in shortlist}
                    for _score, _offset, _candidate, entry in ranked_candidates:
                        selected_for_tts = id(entry) in shortlisted_entry_ids
                        entry["dialogue_adaptor_selected_for_azure"] = selected_for_tts
                        if not selected_for_tts:
                            entry["status"] = "dialogue_adaptor_preflight_pruned"
                            state["audits"].append(
                                {
                                    "attempt": entry["attempt"],
                                    "strategy_attempt": entry["strategy_attempt"],
                                    "strategy_version": entry["strategy_version"],
                                    "status": "dialogue_adaptor_preflight_pruned",
                                    "dialogue_adaptor_preflight": dict(
                                        entry.get("dialogue_adaptor_preflight") or {}
                                    ),
                                    "ai_called": True,
                                    "azure_tts_called": False,
                                }
                            )
                    progress.emit(
                        "semantic_fit",
                        (
                            f"{state['block'].block_id}: dialogue adaptor "
                            f"shortlisted {len(shortlist)}/{len(ranked_candidates)} "
                            "candidate(s) for Azure at locked source tempo"
                        ),
                        status="running",
                        current=len(results),
                        total=len(blocks),
                        block_id=state["block"].block_id,
                        dialogue_adaptor_preflight=True,
                        generated_candidates=len(ranked_candidates),
                        azure_shortlist=len(shortlist),
                    )
                else:
                    shortlist = ranked_candidates

                for _score, _offset, candidate, entry in shortlist:
                    preflight = entry.get("dialogue_adaptor_preflight")
                    if isinstance(preflight, Mapping) and not bool(
                        preflight.get("valid", True)
                    ):
                        entry["status"] = "local_semantic_rejection"
                        entry["reason"] = str(preflight.get("reason") or "preflight")
                        continue
                    pending = evaluate_candidate(
                        index,
                        candidate,
                        int(entry["attempt"]),
                        entry,
                        ai_called=True,
                    )
                    if pending is not None:
                        pending_by_index[index].append(pending)
            atomic_write_json(artifacts.semantic_fit, checkpoint)

            # Review one timing-fit candidate per block at a time. If the
            # fullest fit fails fidelity, consume the next already-generated
            # candidate without paying for another generation request.
            while True:
                review_batch = [
                    pending_by_index[index].pop(0)
                    for index in active
                    if index not in results and pending_by_index[index]
                ]
                if not review_batch:
                    break
                review_pending(review_batch)
                atomic_write_json(artifacts.semantic_fit, checkpoint)

        while not portfolio_supported:
            active: list[int] = []
            payloads: list[dict[str, Any]] = []
            attempt_by_index: dict[int, int] = {}
            strategy_attempt_by_index: dict[int, int] = {}
            for index, state in states.items():
                if index in results:
                    continue
                prior = [
                    int(value.get("attempt") or 0)
                    for value in state["case"].get("attempts") or []
                    if isinstance(value, Mapping)
                ]
                attempt = max(prior, default=0) + 1
                strategy_attempt = strategy_attempts_used(state) + 1
                if strategy_attempt > strategy_attempt_limit(state):
                    continue
                block_index = index
                neighbouring = (
                    *blocks[max(0, block_index - 2):block_index],
                    *blocks[block_index + 1:block_index + 3],
                )
                payload = self._semantic_fit_payload(
                    state["block"],
                    voice=state["voice"],
                    base_rate_percent=state["base_rate"],
                    pitch_percent=state["pitch"],
                    volume_percent=state["volume"],
                    attempt_number=strategy_attempt,
                    target_minimum_ms=state["minimum"],
                    target_preferred_ms=state["preferred"],
                    target_maximum_ms=state["maximum"],
                    duration_evidence=state["evidence"],
                    rejected_spoken_texts=state["rejected"],
                    current_spoken_text=state["current_text"],
                    neighbouring_blocks=neighbouring,
                )
                active.append(index)
                payloads.append(payload)
                attempt_by_index[index] = attempt
                strategy_attempt_by_index[index] = strategy_attempt
            if not active:
                break
            round_number += 1
            progress.emit(
                "semantic_fit",
                (
                    f"batch round {round_number}: generating candidates for "
                    f"{len(active)} unresolved block(s)"
                ),
                status="running",
                current=len(results),
                total=len(blocks),
                round=round_number,
                unresolved_count=len(active),
            )
            batch_payload = {
                "schema_version": SEMANTIC_FIT_SCHEMA_VERSION,
                "operation": "semantic_fit_batch",
                "round_number": round_number,
                "repairs": payloads,
            }
            fingerprint = hashlib.sha256(
                json.dumps(
                    batch_payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            response = call_with_rate_limit_resume(
                "candidate generation",
                provider.fit_turns_semantically,  # type: ignore[attr-defined]
                batch_payload,
            )
            data = response.data if hasattr(response, "data") else response
            repairs = data.get("repairs") if isinstance(data, Mapping) else None
            if not isinstance(repairs, list):
                raise DirectDubError("Semantic-fit batch returned no repairs")
            by_id = {
                str(item.get("unit_id") or ""): item.get("candidate")
                for item in repairs
                if isinstance(item, Mapping)
            }
            batch_rounds.append(
                {
                    "round": round_number,
                    "request_fingerprint": fingerprint,
                    "unit_ids": [
                        _source_block_id(states[index]["block"].block_id)
                        for index in active
                    ],
                    "response": (
                        response.to_artifact()
                        if hasattr(response, "to_artifact")
                        else None
                    ),
                }
            )
            new_entries: dict[int, dict[str, Any]] = {}
            for index in active:
                state = states[index]
                unit_id = _source_block_id(state["block"].block_id)
                candidate = by_id.get(unit_id)
                if not isinstance(candidate, Mapping):
                    raise DirectDubError(f"Semantic-fit batch omitted {unit_id}")
                entry = {
                    "attempt": attempt_by_index[index],
                    "strategy_attempt": strategy_attempt_by_index[index],
                    "strategy_version": SEMANTIC_FIT_BATCH_STRATEGY_VERSION,
                    "batch_round": round_number,
                    "request_fingerprint": fingerprint,
                    "candidate": dict(candidate),
                }
                state["case"].setdefault("attempts", []).append(entry)
                new_entries[index] = entry
            atomic_write_json(artifacts.semantic_fit, checkpoint)
            pending_reviews: list[dict[str, Any]] = []
            for index in active:
                pending = evaluate_candidate(
                    index,
                    new_entries[index]["candidate"],
                    attempt_by_index[index],
                    new_entries[index],
                    ai_called=True,
                )
                if pending is not None:
                    pending_reviews.append(pending)
            atomic_write_json(artifacts.semantic_fit, checkpoint)
            review_pending(pending_reviews)
            atomic_write_json(artifacts.semantic_fit, checkpoint)

        unresolved_indices = [
            index for index in states if index not in results
        ]

        # Production semantic-fit is strict first and completion-safe second.
        # After the configured locked-tempo search is genuinely exhausted, do
        # one independent semantic review of the best acoustically measured
        # frontier for each unresolved block. A reviewed delivery is preferred;
        # otherwise the immutable approved translation is used. The existing
        # bounded Azure-rate/time-stretch final-product conductor then fits that
        # semantically safe text to the source window. This is deliberately not
        # enabled for performance-plan mode, whose contract is explicitly a
        # hard locked-tempo research/QA mode.
        if (
            unresolved_indices
            and options.timing_mode == TIMING_MODE_SEMANTIC_FIT
        ):
            # Keep a short, acoustically ranked ladder per unresolved region.
            # Earlier production rescue reviewed only the single best frontier
            # candidate.  A semantic rejection of that one candidate therefore
            # discarded all other measured alternatives and could abort an
            # otherwise recoverable dub.  Two candidates per word-timed region
            # gives semantic review genuine choice while keeping the review
            # batch bounded and much cheaper than another dialogue-adaptor round.
            rescue_candidate_ladders: dict[int, list[dict[str, Any]]] = {}
            rescue_review_payloads: list[dict[str, Any]] = []

            def rescue_entry_score(
                index: int,
                entry: Mapping[str, Any],
            ) -> tuple[int, int, int, int, int, int]:
                state = states[index]
                original = state["original_block"]
                measured_ms = int(entry.get("measured_duration_ms") or 0)
                if measured_ms < state["minimum"]:
                    timing_distance_ms = state["minimum"] - measured_ms
                elif measured_ms > state["maximum"]:
                    timing_distance_ms = measured_ms - state["maximum"]
                else:
                    timing_distance_ms = 0

                word_timed_region = bool(
                    original.performance_region_id
                    and original.source_timing_evidence
                    == "azure_word_timestamps"
                )
                measurements = [
                    value
                    for value in (
                        entry.get("performance_island_measurements") or []
                    )
                    if isinstance(value, Mapping)
                ]
                if word_timed_region and measurements:
                    miss_count = sum(
                        1
                        for value in measurements
                        if not bool(value.get("raw_isolated_duration_fit"))
                    )
                    max_island_error_ms = max(
                        (
                            abs(int(value.get("delta_ms") or 0))
                            for value in measurements
                        ),
                        default=10**9,
                    )
                    mean_island_error_ms = round(
                        sum(
                            abs(int(value.get("delta_ms") or 0))
                            for value in measurements
                        )
                        / max(1, len(measurements))
                    )
                    prior_review = entry.get("review")
                    review_rank = (
                        0
                        if isinstance(prior_review, Mapping)
                        and self._semantic_review_passed(prior_review)
                        else 1
                    )
                    return (
                        review_rank,
                        miss_count,
                        max_island_error_ms,
                        mean_island_error_ms,
                        timing_distance_ms,
                        -int(entry.get("attempt") or 0),
                    )

                alignment = entry.get("performance_island_alignment")
                drift_ms = (
                    int(
                        alignment.get(
                            "maximum_absolute_cumulative_drift_ms"
                        )
                        or 0
                    )
                    if isinstance(alignment, Mapping)
                    else 0
                )
                unsafe_drift = int(
                    bool(
                        original.performance_region_id
                        and drift_ms > MAXIMUM_INTERNAL_ISLAND_DRIFT_MS
                    )
                )
                drift_penalty_ms = max(
                    0,
                    drift_ms - int(options.semantic_fit_tolerance_ms),
                )
                return (
                    1,
                    unsafe_drift,
                    timing_distance_ms + drift_penalty_ms,
                    abs(measured_ms - state["preferred"]),
                    0,
                    -int(entry.get("attempt") or 0),
                )

            for index in unresolved_indices:
                state = states[index]
                original = state["original_block"]
                word_timed_region = bool(
                    original.performance_region_id
                    and original.source_timing_evidence
                    == "azure_word_timestamps"
                )
                option_limit = 2 if word_timed_region else 1
                ranked_entries = sorted(
                    (
                        entry
                        for entry in state["case"].get("attempts") or []
                        if isinstance(entry, Mapping)
                        and isinstance(entry.get("candidate"), Mapping)
                        and int(entry.get("measured_duration_ms") or 0) > 0
                        and str(entry.get("status") or "")
                        not in {
                            "local_semantic_rejection",
                            "duplicate_rejection",
                            "semantic_review_rejection",
                        }
                    ),
                    key=lambda entry: rescue_entry_score(index, entry),
                )
                ladder: list[dict[str, Any]] = []
                for entry in ranked_entries:
                    score = rescue_entry_score(index, entry)
                    if not word_timed_region and score[1]:
                        continue
                    candidate_data = entry.get("candidate")
                    assert isinstance(candidate_data, Mapping)
                    try:
                        spoken, tts, omissions = (
                            self._validate_finalization_candidate(
                                original,
                                candidate_data,
                            )
                        )
                        tts, pronunciation_substitutions = (
                            self._apply_candidate_pronunciation(
                                spoken,
                                tts,
                                pronunciation_dictionary,
                            )
                        )
                        spoken_islands = self._semantic_target_islands(
                            original,
                            candidate_data,
                            spoken,
                        )
                        tts_islands = tuple(
                            self._apply_candidate_pronunciation(
                                value,
                                value,
                                pronunciation_dictionary,
                            )[0]
                            for value in spoken_islands
                        )
                    except ValueError:
                        # Old checkpoints may contain island ownership created
                        # before the current 600 ms breath-group policy.  Those
                        # measurements stay auditable but are not valid rescue
                        # candidates for the new source-island contract.
                        continue
                    attempt = int(entry.get("attempt") or 0)
                    variant = PreparedTranslationVariant(
                        variant_id=f"semantic_fit_rescue_{attempt:02d}",
                        spoken_text=spoken,
                        tts_text=tts,
                        target_duration_ms=state["preferred"],
                        estimated_duration_ms=None,
                        meaning_preserved=True,
                        omitted_optional_details=tuple(omissions),
                    )
                    candidate_block = replace(
                        original,
                        block_id=(
                            f"{_source_block_id(original.block_id)}"
                            f"_variant_{variant.variant_id}"
                        ),
                        translated_text=spoken,
                        tts_text=tts,
                        variants=(*original.variants, variant),
                        selected_variant_id=variant.variant_id,
                        semantic_target_islands=tts_islands,
                    )
                    option: dict[str, Any] = {
                        "entry": entry,
                        "candidate_block": candidate_block,
                        "spoken_text": spoken,
                        "tts_text": tts,
                        "spoken_islands": tuple(spoken_islands),
                        "omissions": tuple(omissions),
                        "pronunciation_substitutions": pronunciation_substitutions,
                    }
                    prior_review = entry.get("review")
                    if (
                        isinstance(prior_review, Mapping)
                        and self._semantic_review_passed(prior_review)
                    ):
                        option["review"] = dict(prior_review)
                    else:
                        review_key = (
                            f"{_source_block_id(original.block_id)}"
                            f"::production-rescue::{attempt}"
                        )
                        review_payload = self._semantic_fit_review_payload(
                            original,
                            spoken_text=spoken,
                            tts_text=tts,
                            omitted_details=omissions,
                            measured_duration_ms=int(
                                entry.get("measured_duration_ms") or 0
                            ),
                        )
                        review_payload["unit_id"] = review_key
                        option["review_key"] = review_key
                        # Word-timed talking-head regions first exhaust the
                        # already-approved natural/concise/compact wording at
                        # source-anchored breath-group onsets.  Do not spend a
                        # large semantic-review call on an old unreviewed
                        # frontier before we know that approved wording cannot
                        # fit.  If it cannot fit, the localized breath-group
                        # repair stage below rewrites only the failing phrases
                        # and reviews that final candidate once.
                        if not word_timed_region:
                            rescue_review_payloads.append(review_payload)
                        else:
                            # A word-timed frontier is worth reviewing before
                            # rescue only when its existing Azure measurements
                            # already satisfy every current breath-group window.
                            # Otherwise use it later as a repair seed and review
                            # the final merged delivery once, after localized
                            # phrase repair.
                            frontier_measurements = [
                                value
                                for value in (
                                    entry.get("performance_island_measurements")
                                    or []
                                )
                                if isinstance(value, Mapping)
                            ]
                            if (
                                len(frontier_measurements)
                                == len(original.source_speech_islands)
                                and frontier_measurements
                                and all(
                                    bool(value.get("raw_isolated_duration_fit"))
                                    or (
                                        value.get("raw_isolated_duration_fit") is None
                                        and int(value.get("minimum_duration_ms") or 0)
                                        <= int(value.get("measured_duration_ms") or 0)
                                        <= int(value.get("maximum_duration_ms") or 0)
                                    )
                                    for value in frontier_measurements
                                )
                            ):
                                rescue_review_payloads.append(review_payload)
                    ladder.append(option)
                    if len(ladder) >= option_limit:
                        break
                if ladder:
                    rescue_candidate_ladders[index] = ladder

            rescue_reviews: dict[str, Mapping[str, Any]] = {}
            rescue_review_error: str | None = None
            if rescue_review_payloads:
                try:
                    response = call_with_rate_limit_resume(
                        "production rescue semantic review",
                        provider.review_semantic_fits,  # type: ignore[attr-defined]
                        {
                            "schema_version": SEMANTIC_FIT_SCHEMA_VERSION,
                            "operation": "semantic_fit_review_batch",
                            "reviews": rescue_review_payloads,
                        },
                    )
                    data = (
                        response.data
                        if hasattr(response, "data")
                        else response
                    )
                    returned = (
                        data.get("reviews")
                        if isinstance(data, Mapping)
                        else None
                    )
                    if not isinstance(returned, list):
                        raise DirectDubError(
                            "Semantic-fit production rescue returned no reviews"
                        )
                    rescue_reviews = {
                        str(item.get("unit_id") or ""): item.get("review")
                        for item in returned
                        if isinstance(item, Mapping)
                        and isinstance(item.get("review"), Mapping)
                    }
                except Exception as exc:  # optional rescue review; approved text remains safe
                    rescue_review_error = (
                        f"{type(exc).__name__}: {exc}"
                    )

            localized_repair_seeds: dict[int, dict[str, Any]] = {}

            for index in unresolved_indices:
                state = states[index]
                original_block = state["original_block"]
                unit_id = _source_block_id(original_block.block_id)

                word_timed_region = bool(
                    original_block.performance_region_id
                    and original_block.source_timing_evidence
                    == "azure_word_timestamps"
                )

                rescue: dict[str, Any] | None = None
                review: Mapping[str, Any] | None = None
                for option in rescue_candidate_ladders.get(index, []):
                    option_review = option.get("review")
                    if not isinstance(option_review, Mapping):
                        option_review = rescue_reviews.get(
                            str(option.get("review_key") or "")
                        )
                    if (
                        isinstance(option_review, Mapping)
                        and self._semantic_review_passed(option_review)
                    ):
                        rescue = option
                        review = option_review
                        break

                reviewed_candidate = rescue is not None and review is not None
                if reviewed_candidate:
                    selected_block = rescue["candidate_block"]
                    selection_reason = (
                        "semantic_fit_exhaustion_reviewed_frontier_rescue"
                    )
                    selected_attempt: int | str = int(
                        rescue["entry"].get("attempt") or 0
                    )
                    semantic_authority: Mapping[str, Any] | str = dict(review)
                else:
                    selected_block = original_block
                    selection_reason = (
                        "semantic_fit_exhaustion_approved_translation_rescue"
                    )
                    selected_attempt = "approved_translation"
                    semantic_authority = "approved_translation_artifact"

                voice = state["voice"]
                base_rate = state["base_rate"]
                pitch = state["pitch"]
                volume = state["volume"]

                if word_timed_region:
                    # Production rescue is text-choice exhaustive, not
                    # attempt-number exhaustive.  Try a semantically reviewed
                    # dialogue-adaptor frontier first.  If no reviewed frontier
                    # survives, the translation artifact already gives us
                    # three semantically approved variants.  Partition those
                    # immutable words across the current source breath groups
                    # and measure them directly at the locked speaker tempo.
                    # This removes the old false failure where one rejected
                    # rescue candidate caused the whole talking-head region to
                    # abort even though safe approved wording remained.
                    anchored_options: list[dict[str, Any]] = []
                    seen_spoken: set[str] = set()
                    if reviewed_candidate:
                        normalized_spoken = " ".join(
                            selected_block.translated_text.casefold().split()
                        )
                        seen_spoken.add(normalized_spoken)
                        anchored_options.append(
                            {
                                "block": selected_block,
                                "attempt": selected_attempt,
                                "selection_reason": selection_reason,
                                "semantic_authority": semantic_authority,
                                "reviewed_frontier": True,
                                "spoken_islands": tuple(
                                    rescue.get("spoken_islands")
                                    or selected_block.semantic_target_islands
                                ),
                                "preflight_score": (-1, 0, 0, 0),
                            }
                        )

                    ms_per_syllable = _dialogue_adaptor_ms_per_syllable(
                        state["evidence"],
                        fallback_text=original_block.translated_text,
                        fallback_plain_duration_ms=0,
                    )
                    variant_naturalness_rank = {
                        "natural": 0,
                        "concise": 1,
                        "compact": 2,
                    }
                    for variant in original_block.variants:
                        if not variant.meaning_preserved:
                            continue
                        spoken_text = " ".join(variant.spoken_text.split()).strip()
                        if not spoken_text:
                            continue
                        normalized_spoken = " ".join(
                            spoken_text.casefold().split()
                        )
                        if normalized_spoken in seen_spoken:
                            continue
                        spoken_islands = (
                            _partition_approved_text_across_source_islands(
                                spoken_text,
                                original_block.source_speech_islands,
                                protected_phrases=original_block.protected_entities,
                            )
                        )
                        if len(spoken_islands) != len(
                            original_block.source_speech_islands
                        ):
                            continue
                        tts_text, pronunciation_substitutions = (
                            self._apply_candidate_pronunciation(
                                spoken_text,
                                variant.tts_text,
                                pronunciation_dictionary,
                            )
                        )
                        tts_islands = tuple(
                            self._apply_candidate_pronunciation(
                                value,
                                value,
                                pronunciation_dictionary,
                            )[0]
                            for value in spoken_islands
                        )

                        predicted_misses = 0
                        predicted_max_error_ms = 0
                        predicted_total_error_ms = 0
                        for island_text, source_island in zip(
                            tts_islands,
                            original_block.source_speech_islands,
                            strict=True,
                        ):
                            target_ms = max(
                                1,
                                int(source_island.get("duration_ms") or 0)
                                or int(source_island.get("end_ms") or 0)
                                - int(source_island.get("start_ms") or 0),
                            )
                            predicted_ms = max(
                                1,
                                round(
                                    _estimated_zulu_syllables(island_text)
                                    * ms_per_syllable
                                ),
                            )
                            island_tolerance_ms = min(
                                500,
                                max(
                                    int(state["alignment_tolerance_ms"]),
                                    min(250, target_ms // 8),
                                ),
                            )
                            error_ms = abs(predicted_ms - target_ms)
                            predicted_max_error_ms = max(
                                predicted_max_error_ms,
                                error_ms,
                            )
                            predicted_total_error_ms += error_ms
                            if not (
                                target_ms - island_tolerance_ms
                                <= predicted_ms
                                <= target_ms + island_tolerance_ms
                            ):
                                predicted_misses += 1

                        approved_block = replace(
                            original_block,
                            block_id=(
                                f"{_source_block_id(original_block.block_id)}"
                                f"_variant_approved_anchor_{variant.variant_id}"
                            ),
                            translated_text=spoken_text,
                            tts_text=tts_text,
                            selected_variant_id=variant.variant_id,
                            semantic_target_islands=tts_islands,
                        )
                        anchored_options.append(
                            {
                                "block": approved_block,
                                "attempt": f"approved_{variant.variant_id}",
                                "selection_reason": (
                                    f"approved_{variant.variant_id}_source_anchored_partition"
                                ),
                                "semantic_authority": (
                                    f"approved_translation_variant:{variant.variant_id}"
                                ),
                                "reviewed_frontier": False,
                                "spoken_islands": tuple(spoken_islands),
                                "pronunciation_substitutions": pronunciation_substitutions,
                                "preflight_score": (
                                    predicted_misses,
                                    predicted_max_error_ms,
                                    predicted_total_error_ms,
                                    variant_naturalness_rank.get(
                                        variant.variant_id,
                                        9,
                                    ),
                                ),
                            }
                        )
                        seen_spoken.add(normalized_spoken)

                    # Performance-region aggregation can legitimately arrive
                    # without per-region natural/concise/compact variants. The
                    # authoritative approved translation itself is still a
                    # semantically safe rescue seed, so always make it
                    # available when no approved variant survived partitioning.
                    if not any(
                        not value["reviewed_frontier"]
                        for value in anchored_options
                    ):
                        approved_spoken = " ".join(
                            original_block.translated_text.split()
                        ).strip()
                        if approved_spoken:
                            approved_spoken_islands = (
                                _partition_approved_text_across_source_islands(
                                    approved_spoken,
                                    original_block.source_speech_islands,
                                    protected_phrases=(
                                        original_block.protected_entities
                                    ),
                                )
                            )
                            if len(approved_spoken_islands) == len(
                                original_block.source_speech_islands
                            ):
                                approved_tts, approved_substitutions = (
                                    self._apply_candidate_pronunciation(
                                        approved_spoken,
                                        original_block.tts_text,
                                        pronunciation_dictionary,
                                    )
                                )
                                approved_tts_islands = tuple(
                                    self._apply_candidate_pronunciation(
                                        value,
                                        value,
                                        pronunciation_dictionary,
                                    )[0]
                                    for value in approved_spoken_islands
                                )
                                approved_block = replace(
                                    original_block,
                                    block_id=(
                                        f"{_source_block_id(original_block.block_id)}"
                                        "_variant_approved_anchor_authoritative"
                                    ),
                                    translated_text=approved_spoken,
                                    tts_text=approved_tts,
                                    selected_variant_id="approved_authoritative",
                                    semantic_target_islands=approved_tts_islands,
                                )
                                anchored_options.append(
                                    {
                                        "block": approved_block,
                                        "attempt": "approved_authoritative",
                                        "selection_reason": (
                                            "approved_authoritative_source_anchored_partition"
                                        ),
                                        "semantic_authority": (
                                            "approved_translation_artifact"
                                        ),
                                        "reviewed_frontier": False,
                                        "spoken_islands": tuple(
                                            approved_spoken_islands
                                        ),
                                        "pronunciation_substitutions": (
                                            approved_substitutions
                                        ),
                                        # Variant-specific preflight prediction
                                        # is unavailable here. Azure remains the
                                        # timing authority; try this safe seed
                                        # after any ranked approved variants.
                                        "preflight_score": (999, 999999, 999999, 0),
                                    }
                                )

                    # The reviewed LLM frontier stays first when available. For
                    # approved fallbacks, text-only prediction is used only to
                    # reduce unnecessary Azure calls; Azure remains the timing
                    # authority and natural wording wins ties.
                    if reviewed_candidate:
                        reviewed_option = anchored_options[:1]
                        approved_options = sorted(
                            anchored_options[1:],
                            key=lambda value: value["preflight_score"],
                        )
                        anchored_options = [*reviewed_option, *approved_options]
                    else:
                        anchored_options.sort(
                            key=lambda value: value["preflight_score"]
                        )

                    fitted = None
                    selected_delivery: dict[str, Any] | None = None
                    delivery_failures: list[dict[str, Any]] = []
                    best_localized_seed: dict[str, Any] | None = None
                    best_localized_seed_score: tuple[int, int, int, int] | None = None

                    # A measured dialogue-adaptor frontier is a valid *repair
                    # seed* even when it has not yet passed semantic review.
                    # It can never be selected directly in this state: the
                    # phrase-only repair path below merges only failing groups
                    # and independently reviews the complete final delivery.
                    # This matters for agglutinative isiZulu, where an approved
                    # sentence can contain fewer whitespace tokens than the
                    # number of 600 ms source breath groups and therefore cannot
                    # be safely partitioned without linguistic adaptation.
                    for seed_option in rescue_candidate_ladders.get(index, []):
                        seed_entry = seed_option.get("entry")
                        if not isinstance(seed_entry, Mapping):
                            continue
                        seed_measurements = [
                            dict(value)
                            for value in (
                                seed_entry.get("performance_island_measurements")
                                or []
                            )
                            if isinstance(value, Mapping)
                        ]
                        if len(seed_measurements) != len(
                            original_block.source_speech_islands
                        ):
                            continue
                        for value in seed_measurements:
                            if value.get("raw_isolated_duration_fit") is None:
                                measured = int(value.get("measured_duration_ms") or 0)
                                minimum = int(value.get("minimum_duration_ms") or 0)
                                maximum = int(value.get("maximum_duration_ms") or 0)
                                value["raw_isolated_duration_fit"] = (
                                    minimum <= measured <= maximum
                                )
                        seed_misses = [
                            value
                            for value in seed_measurements
                            if not bool(value.get("raw_isolated_duration_fit"))
                        ]
                        if not seed_misses:
                            continue
                        seed_score = (
                            len(seed_misses),
                            max(
                                abs(int(value.get("delta_ms") or 0))
                                for value in seed_misses
                            ),
                            sum(
                                abs(int(value.get("delta_ms") or 0))
                                for value in seed_misses
                            ),
                            sum(
                                max(0, int(value.get("delta_ms") or 0))
                                for value in seed_misses
                            ),
                        )
                        if (
                            best_localized_seed_score is None
                            or seed_score < best_localized_seed_score
                        ):
                            best_localized_seed_score = seed_score
                            best_localized_seed = {
                                "block": seed_option["candidate_block"],
                                "attempt": (
                                    f"unreviewed_frontier_{int(seed_entry.get('attempt') or 0)}"
                                ),
                                "selection_reason": (
                                    "measured_frontier_seed_pending_final_semantic_review"
                                ),
                                "semantic_authority": "pending_final_semantic_review",
                                "reviewed_frontier": False,
                                "spoken_islands": tuple(
                                    seed_option.get("spoken_islands") or ()
                                ),
                                "preflight_score": (-1, 0, 0, 0),
                                "measurements": seed_measurements,
                                "misses": [dict(value) for value in seed_misses],
                            }

                    for delivery in anchored_options:
                        candidate_block = delivery["block"]
                        try:
                            _alignment_fit, measurements = (
                                self._measure_performance_islands(
                                    candidate_block,
                                    artifacts,
                                    island_texts=(
                                        candidate_block.semantic_target_islands
                                    ),
                                    voice=voice,
                                    base_rate_percent=base_rate,
                                    pitch_percent=pitch,
                                    volume_percent=volume,
                                    tolerance_ms=state["alignment_tolerance_ms"],
                                )
                            )
                        except (ValueError, DirectDubError) as exc:
                            delivery_failures.append(
                                {
                                    "attempt": delivery["attempt"],
                                    "selection_reason": delivery["selection_reason"],
                                    "error_type": type(exc).__name__,
                                    "reason": str(exc),
                                }
                            )
                            continue

                        misses = [
                            value
                            for value in measurements
                            if not bool(value.get("raw_isolated_duration_fit"))
                        ]
                        if misses:
                            max_error_ms = max(
                                abs(int(value.get("delta_ms") or 0))
                                for value in misses
                            )
                            total_error_ms = sum(
                                abs(int(value.get("delta_ms") or 0))
                                for value in misses
                            )
                            overflow_ms = sum(
                                max(0, int(value.get("delta_ms") or 0))
                                for value in misses
                            )
                            seed_score = (
                                len(misses),
                                max_error_ms,
                                total_error_ms,
                                overflow_ms,
                            )
                            failure = {
                                "attempt": delivery["attempt"],
                                "selection_reason": delivery["selection_reason"],
                                "error_type": "BreathGroupTimingMiss",
                                "reason": (
                                    f"{len(misses)} source-anchored breath group(s) "
                                    "miss their mouth-active windows at the locked tempo"
                                ),
                                "measurements": [dict(value) for value in measurements],
                                "misses": [dict(value) for value in misses],
                            }
                            delivery_failures.append(failure)
                            if (
                                best_localized_seed_score is None
                                or seed_score < best_localized_seed_score
                            ):
                                best_localized_seed_score = seed_score
                                best_localized_seed = {
                                    **delivery,
                                    "measurements": [
                                        dict(value) for value in measurements
                                    ],
                                    "misses": [dict(value) for value in misses],
                                }
                            continue

                        try:
                            candidate_fitted = (
                                self._synthesize_source_anchored_breath_groups(
                                    candidate_block,
                                    artifacts,
                                    island_texts=(
                                        candidate_block.semantic_target_islands
                                    ),
                                    voice=voice,
                                    base_rate_percent=base_rate,
                                    pitch_percent=pitch,
                                    volume_percent=volume,
                                    tolerance_ms=state["alignment_tolerance_ms"],
                                )
                            )
                        except (
                            TimingOverflowError,
                            ValueError,
                            DirectDubError,
                        ) as exc:
                            delivery_failures.append(
                                {
                                    "attempt": delivery["attempt"],
                                    "selection_reason": delivery["selection_reason"],
                                    "error_type": type(exc).__name__,
                                    "reason": str(exc),
                                    "measured_duration_ms": getattr(
                                        exc, "measured_duration_ms", None
                                    ),
                                    "target_duration_ms": getattr(
                                        exc, "target_duration_ms", None
                                    ),
                                }
                            )
                            continue
                        selected_delivery = delivery
                        fitted = candidate_fitted
                        break

                    if selected_delivery is None or fitted is None:
                        if best_localized_seed is not None and (
                            localized_phrase_supported or portfolio_supported
                        ):
                            localized_repair_seeds[index] = {
                                "delivery": best_localized_seed,
                                "delivery_failures": delivery_failures,
                            }
                            state["case"]["status"] = (
                                "localized_breath_group_repair_pending"
                            )
                            state["case"]["production_rescue"] = {
                                "status": "localized_breath_group_repair_pending",
                                "speaker_tempo_locked": True,
                                "source_lipsync_locked": True,
                                "seed_attempt": best_localized_seed["attempt"],
                                "repair_island_ids": [
                                    str(value.get("island_id") or "")
                                    for value in best_localized_seed["misses"]
                                ],
                                "delivery_failures": delivery_failures,
                            }
                            atomic_write_json(artifacts.semantic_fit, checkpoint)
                            continue

                        state["case"]["status"] = (
                            "breath_group_delivery_exhausted"
                        )
                        state["case"]["attempt_limit"] = (
                            strategy_attempt_limit(state)
                        )
                        state["case"]["production_rescue"] = {
                            "status": "blocked_quality_first",
                            "speaker_tempo_locked": True,
                            "source_lipsync_locked": True,
                            "reviewed_frontier_candidates_considered": len(
                                rescue_candidate_ladders.get(index, [])
                            ),
                            "approved_variants_considered": len(
                                [
                                    value
                                    for value in anchored_options
                                    if not value["reviewed_frontier"]
                                ]
                            ),
                            "delivery_failures": delivery_failures,
                            "reason": (
                                "every semantically safe source-anchored text "
                                "choice has at least one breath group outside "
                                "its mouth-active window at the locked source tempo"
                            ),
                        }
                        atomic_write_json(artifacts.semantic_fit, checkpoint)
                        continue

                    selected_block = selected_delivery["block"]
                    selected_attempt = selected_delivery["attempt"]
                    selection_reason = selected_delivery["selection_reason"]
                    semantic_authority = selected_delivery[
                        "semantic_authority"
                    ]
                    reviewed_candidate = bool(
                        selected_delivery["reviewed_frontier"]
                    )
                    rescue_audit = {
                        "attempt": selected_attempt,
                        "status": "selected_source_anchored_breath_groups",
                        "selection_reason": selection_reason,
                        "semantic_authority": semantic_authority,
                        "review_error": rescue_review_error,
                        "speaker_tempo_locked": True,
                        "source_lipsync_locked": True,
                        "post_synthesis_time_stretch_ratio": 1.0,
                        "selected_azure_rate_percent": base_rate,
                        "selected_from_reviewed_frontier": reviewed_candidate,
                        "human_review_required": False,
                    }
                    state["audits"].append(rescue_audit)
                    state["case"]["status"] = (
                        "selected_source_anchored_breath_groups"
                    )
                    state["case"]["attempt_limit"] = (
                        strategy_attempt_limit(state)
                    )
                    state["case"]["selected_attempt"] = selected_attempt
                    state["case"]["production_rescue"] = dict(rescue_audit)
                    fitted["translation_variant"] = {
                        "schema_version": "mathula-selected-translation-variant-v1",
                        "source_block_id": unit_id,
                        "selected_variant_id": selected_block.selected_variant_id,
                        "spoken_text": selected_block.translated_text,
                        "measured_duration_ms": fitted.get("duration_ms"),
                        "meaning_preserved": True,
                        "selection_reason": (
                            "source_anchored_breath_group_delivery"
                        ),
                        "human_review_required": False,
                    }
                    fitted["semantic_fit"] = {
                        "schema_version": SEMANTIC_FIT_SCHEMA_VERSION,
                        "strategy_version": SEMANTIC_FIT_BATCH_STRATEGY_VERSION,
                        "selected_attempt": selected_attempt,
                        "strict_attempt_limit": strategy_attempt_limit(state),
                        "production_rescue": True,
                        "rescue_mode": "source_anchored_breath_groups",
                        "speaker_tempo_locked": True,
                        "source_lipsync_locked": True,
                        "time_stretch_used": False,
                        "semantic_authority": semantic_authority,
                    }
                    fitted["performance_plan"] = {
                        "schema_version": PERFORMANCE_PLAN_VERSION,
                        "region_id": original_block.performance_region_id,
                        "member_block_ids": list(
                            original_block.performance_region_member_ids
                        ),
                        "selected_attempt": selected_attempt,
                        "target_island_texts": list(
                            selected_block.semantic_target_islands
                        ),
                        "source_island_ids": [
                            str(value.get("island_id") or "")
                            for value in original_block.source_speech_islands
                        ],
                        "continuous_waveform_gate_exhausted": True,
                        "production_rescue": True,
                        "rescue_mode": "source_anchored_breath_groups",
                        "speaker_tempo_locked": True,
                        "source_onsets_locked": True,
                        "time_stretch_used": False,
                    }
                    fitted["translation_variant_attempts"] = state["audits"]
                    results[index] = (
                        selected_block,
                        fitted,
                        state["audits"],
                    )
                    progress.emit(
                        "semantic_fit",
                        (
                            f"{state['block'].block_id}: strict whole-region fit "
                            "exhausted; completed with source-anchored breath groups"
                        ),
                        status="completed",
                        current=len(results),
                        total=len(blocks),
                        block_id=state["block"].block_id,
                        production_rescue=True,
                        source_anchored_breath_groups=True,
                        speaker_tempo_locked=True,
                    )
                    atomic_write_json(artifacts.semantic_fit, checkpoint)
                    continue

                base_rate = state["base_rate"]
                pitch = state["pitch"]
                volume = state["volume"]
                try:
                    fitted = self._synthesize_block(
                        selected_block,
                        artifacts,
                        voice=voice,
                        base_rate_percent=base_rate,
                        pitch_percent=pitch,
                        volume_percent=volume,
                        max_azure_rate_percent=(
                            options.max_azure_rate_percent
                        ),
                        max_time_stretch_ratio=(
                            options.max_time_stretch_ratio
                        ),
                        max_time_expansion_ratio=(
                            options.max_time_expansion_ratio
                        ),
                        cadence_max_time_stretch_ratio=(
                            options.max_time_stretch_ratio
                        ),
                        force=False,
                    )
                    selected_rate = int(
                        fitted.get("selected_azure_rate_percent") or 0
                    )
                    selected_azure_ms = next(
                        (
                            int(value.get("duration_ms") or 0)
                            for value in reversed(
                                list(fitted.get("azure_attempts") or [])
                            )
                            if isinstance(value, Mapping)
                            and int(value.get("rate_percent") or 0)
                            == selected_rate
                        ),
                        int(fitted.get("duration_ms") or 0),
                    )
                    adjustment = {
                        "method": "semantic_fit_natural_cadence_rescue",
                        "solver_version": FINAL_PRODUCT_CONDUCTOR_VERSION,
                        "semantic_fit_rescue": True,
                        "strict_semantic_fit_attempts_exhausted": True,
                        "selected_from_reviewed_frontier": reviewed_candidate,
                        "selected_attempt": selected_attempt,
                        "measured_duration_ms": selected_azure_ms,
                        "available_duration_ms": selected_block.duration_ms,
                        "selected_azure_rate_percent": fitted.get(
                            "selected_azure_rate_percent"
                        ),
                        "applied_time_stretch_ratio": fitted.get(
                            "post_synthesis_time_stretch_ratio"
                        ),
                        "configured_quality_ceiling": (
                            options.final_product_max_time_stretch_ratio
                        ),
                        "quality_ceiling_exceeded": False,
                        "authoritative_translation_changed": False,
                        "delivery_overlay_rewrite_used": reviewed_candidate,
                        "speaker_order_preserved": True,
                        "timeline_boundaries_preserved": True,
                        "human_review_required": False,
                        "quality_warning": (
                            "Strict locked-tempo semantic fit exhausted; the "
                            "semantically safe delivery was completed with "
                            "bounded final-product cadence repair."
                        ),
                    }
                    fitted["timing_adjustment"] = adjustment
                    fitted["finalization_conductor"] = adjustment
                except TimingOverflowError as overflow:
                    # Word-timed talking-head speech is quality critical, including single-island blocks.
                    # Once the bounded dialogue-adaptor feedback rounds are
                    # exhausted, never pass them into the generic timeline
                    # finalizer: that path can borrow source silence or stretch
                    # the whole voice and thereby destroy the very mouth timing
                    # semantic-fit was asked to protect.  Fail closed so the
                    # caller gets an explicit quality error instead of a
                    # superficially complete but visibly late dub.
                    if (
                        original_block.source_timing_evidence
                        == "azure_word_timestamps"
                        and original_block.source_speech_islands
                    ):
                        state["case"]["status"] = (
                            "breath_group_alignment_exhausted"
                        )
                        state["case"]["attempt_limit"] = (
                            strategy_attempt_limit(state)
                        )
                        state["case"]["production_rescue"] = {
                            "status": "blocked_quality_first",
                            "speaker_tempo_locked": True,
                            "source_lipsync_locked": True,
                            "measured_duration_ms": overflow.measured_duration_ms,
                            "available_duration_ms": overflow.target_duration_ms,
                            "required_time_stretch_ratio": overflow.stretch_ratio,
                            "reason": (
                                "word-timed performance region may not fall back "
                                "to generic timeline stretching or silence borrowing"
                            ),
                        }
                        atomic_write_json(artifacts.semantic_fit, checkpoint)
                        continue

                    # Non-word-timed material may still use the legacy natural
                    # cadence/timeline rescue because no authoritative mouth
                    # activity evidence exists for that region.
                    rescue_audit = {
                        "attempt": selected_attempt,
                        "status": "selected_for_timeline_fit",
                        "selection_reason": selection_reason,
                        "semantic_authority": semantic_authority,
                        "review_error": rescue_review_error,
                        "speaker_tempo_locked": True,
                        "measured_duration_ms": overflow.measured_duration_ms,
                        "available_duration_ms": overflow.target_duration_ms,
                        "required_time_stretch_ratio": overflow.stretch_ratio,
                        "selected_from_reviewed_frontier": reviewed_candidate,
                        "human_review_required": False,
                    }
                    state["audits"].append(rescue_audit)
                    state["case"]["status"] = "selected_for_timeline_fit"
                    state["case"]["attempt_limit"] = strategy_attempt_limit(state)
                    state["case"]["selected_attempt"] = selected_attempt
                    state["case"]["production_rescue"] = dict(rescue_audit)
                    results[index] = PreparedVariantsExhausted(
                        selected_block,
                        overflow,
                        state["audits"],
                    )
                    progress.emit(
                        "semantic_fit",
                        (
                            f"{state['block'].block_id}: strict semantic-fit "
                            "exhausted; preserving natural cadence for timeline fit"
                        ),
                        status="completed",
                        current=len(results),
                        total=len(blocks),
                        block_id=state["block"].block_id,
                        production_rescue=True,
                        routed_to_timeline_fit=True,
                        selected_from_reviewed_frontier=reviewed_candidate,
                    )
                    atomic_write_json(artifacts.semantic_fit, checkpoint)
                    continue

                rescue_audit = {
                    "attempt": selected_attempt,
                    "status": "selected_production_rescue",
                    "selection_reason": selection_reason,
                    "semantic_authority": semantic_authority,
                    "review_error": rescue_review_error,
                    "speaker_tempo_locked": False,
                    "selected_azure_rate_percent": fitted.get(
                        "selected_azure_rate_percent"
                    ),
                    "post_synthesis_time_stretch_ratio": fitted.get(
                        "post_synthesis_time_stretch_ratio"
                    ),
                    "human_review_required": False,
                }
                state["audits"].append(rescue_audit)
                state["case"]["status"] = "selected_production_rescue"
                state["case"]["attempt_limit"] = strategy_attempt_limit(
                    state
                )
                state["case"]["selected_attempt"] = selected_attempt
                state["case"]["production_rescue"] = dict(rescue_audit)
                fitted["translation_variant"] = {
                    "schema_version": "mathula-selected-translation-variant-v1",
                    "source_block_id": unit_id,
                    "selected_variant_id": (
                        selected_block.selected_variant_id
                        if reviewed_candidate
                        else "approved_natural"
                    ),
                    "spoken_text": selected_block.translated_text,
                    "measured_duration_ms": fitted.get("duration_ms"),
                    "meaning_preserved": True,
                    "selection_reason": selection_reason,
                    "human_review_required": False,
                }
                fitted["semantic_fit"] = {
                    "schema_version": SEMANTIC_FIT_SCHEMA_VERSION,
                    "strategy_version": SEMANTIC_FIT_BATCH_STRATEGY_VERSION,
                    "selected_attempt": selected_attempt,
                    "strict_attempt_limit": strategy_attempt_limit(state),
                    "production_rescue": True,
                    "speaker_tempo_locked": False,
                    "time_stretch_used": (
                        float(
                            fitted.get("post_synthesis_time_stretch_ratio")
                            or 1.0
                        )
                        != 1.0
                    ),
                    "semantic_authority": semantic_authority,
                    "target_duration_band_ms": {
                        "minimum": state["minimum"],
                        "preferred": state["preferred"],
                        "maximum": state["maximum"],
                    },
                }
                if original_block.performance_region_id:
                    fitted["performance_plan"] = {
                        "schema_version": PERFORMANCE_PLAN_VERSION,
                        "region_id": original_block.performance_region_id,
                        "member_block_ids": list(
                            original_block.performance_region_member_ids
                        ),
                        "selected_attempt": selected_attempt,
                        "target_island_texts": list(
                            selected_block.semantic_target_islands
                        ),
                        "source_island_ids": [
                            str(value.get("island_id") or "")
                            for value in original_block.source_speech_islands
                        ],
                        "strict_internal_island_gate_exhausted": True,
                        "production_rescue": True,
                        "speaker_tempo_locked": False,
                        "time_stretch_used": (
                            float(
                                fitted.get(
                                    "post_synthesis_time_stretch_ratio"
                                )
                                or 1.0
                            )
                            != 1.0
                        ),
                    }
                fitted["translation_variant_attempts"] = state["audits"]
                results[index] = (
                    selected_block,
                    fitted,
                    state["audits"],
                )
                progress.emit(
                    "semantic_fit",
                    (
                        f"{state['block'].block_id}: strict semantic-fit "
                        "exhausted; completed with production cadence rescue"
                    ),
                    status="completed",
                    current=len(results),
                    total=len(blocks),
                    block_id=state["block"].block_id,
                    production_rescue=True,
                    selected_from_reviewed_frontier=reviewed_candidate,
                )
                atomic_write_json(artifacts.semantic_fit, checkpoint)

            # Phrase-only professional dialogue adaptation.  Whole-region
            # structured output is the wrong contract for a localized repair:
            # it lets the model reflow already-good neighbouring phrases and
            # then the server has to reject those changes.  The dedicated
            # provider method receives ONLY failing breath groups and returns
            # ONLY phrase replacements.  Frozen phrases never cross the model
            # boundary, so they are immutable by construction.
            if localized_phrase_supported and unresolved_indices:
                localized_work: dict[int, dict[str, Any]] = {}
                for index, seed_info in localized_repair_seeds.items():
                    if index in results:
                        continue
                    state = states[index]
                    original_block = state["original_block"]
                    delivery = seed_info["delivery"]
                    seed_block = delivery["block"]
                    measurements = [
                        dict(value) for value in delivery.get("measurements") or []
                    ]
                    spoken_islands = list(
                        str(value).strip()
                        for value in (
                            delivery.get("spoken_islands")
                            or seed_block.semantic_target_islands
                        )
                    )
                    tts_islands = list(
                        str(value).strip()
                        for value in seed_block.semantic_target_islands
                    )
                    if (
                        not measurements
                        or len(spoken_islands) != len(original_block.source_speech_islands)
                        or len(tts_islands) != len(original_block.source_speech_islands)
                        or len(measurements) != len(original_block.source_speech_islands)
                    ):
                        state["case"]["localized_breath_group_repair"] = {
                            "schema_version": (
                                "mathula-coordinated-breath-group-repair-v3-v13.19.10"
                            ),
                            "status": "invalid_seed_shape",
                            "source_island_count": len(original_block.source_speech_islands),
                            "spoken_island_count": len(spoken_islands),
                            "tts_island_count": len(tts_islands),
                            "measurement_count": len(measurements),
                        }
                        continue
                    # Re-evaluate cached/seed measurements against the current
                    # onset-anchored window.  This lets a v13.19.8 checkpoint
                    # resume without carrying forward the old hard mouth-active
                    # endpoint as an artificial failure.
                    for position, (source_island, measurement) in enumerate(
                        zip(
                            original_block.source_speech_islands,
                            measurements,
                            strict=True,
                        ),
                        start=1,
                    ):
                        window = self._source_anchored_breath_group_window(
                            original_block,
                            source_island=source_island,
                            island_index=position,
                            tolerance_ms=state["alignment_tolerance_ms"],
                        )
                        measured_ms = int(
                            measurement.get("measured_duration_ms") or 0
                        )
                        measurement.update(window)
                        measurement["source_start_ms"] = int(
                            source_island.get("start_ms") or 0
                        )
                        measurement["source_end_ms"] = int(
                            source_island.get("end_ms")
                            or measurement["source_start_ms"]
                        )
                        if measured_ms > 0:
                            measurement["measured_duration_ms"] = measured_ms
                            measurement["delta_ms"] = (
                                measured_ms
                                - int(measurement["target_duration_ms"])
                            )
                            fits = (
                                int(measurement["minimum_duration_ms"])
                                <= measured_ms
                                <= int(measurement["maximum_duration_ms"])
                            )
                            measurement["raw_isolated_duration_fit"] = fits
                            measurement["pause_borrow_used_ms"] = max(
                                0,
                                measured_ms
                                - int(measurement["target_duration_ms"]),
                            )
                            measurement["status"] = (
                                "fit_with_pause_borrow"
                                if fits
                                and measured_ms
                                > int(
                                    measurement[
                                        "nominal_maximum_duration_ms"
                                    ]
                                )
                                else "fit" if fits else "timing_miss"
                            )
                    pending_positions = {
                        position
                        for position, measurement in enumerate(measurements)
                        if not bool(measurement.get("raw_isolated_duration_fit"))
                    }
                    if not pending_positions:
                        continue
                    localized_work[index] = {
                        "seed_delivery": delivery,
                        "spoken_islands": spoken_islands,
                        "tts_islands": tts_islands,
                        "measurements": measurements,
                        "pending_positions": pending_positions,
                        "rounds": [],
                        "bootstrap_from_source": False,
                    }

                # A word-timed talking-head region must not depend on a prior
                # target-language candidate merely to enter professional
                # dialogue adaptation.  Historical checkpoints and strongly
                # agglutinative translations can legitimately have no usable
                # island-owned seed.  Bootstrap directly from each source
                # breath group in that case; the model sees the whole approved
                # translation for semantic context, Azure owns timing, and the
                # merged result still passes the same independent full-block
                # semantic review before selection.
                for index in unresolved_indices:
                    if index in results or index in localized_work:
                        continue
                    state = states[index]
                    original_block = state["original_block"]
                    if not (
                        original_block.source_timing_evidence
                        == "azure_word_timestamps"
                        and original_block.source_speech_islands
                        and not original_block.source_interactions
                    ):
                        continue
                    bootstrap_measurements: list[dict[str, Any]] = []
                    for position, source_island in enumerate(
                        original_block.source_speech_islands,
                        start=1,
                    ):
                        start_ms = int(source_island.get("start_ms") or 0)
                        end_ms = int(source_island.get("end_ms") or start_ms)
                        window = self._source_anchored_breath_group_window(
                            original_block,
                            source_island=source_island,
                            island_index=position,
                            tolerance_ms=state["alignment_tolerance_ms"],
                        )
                        bootstrap_measurements.append(
                            {
                                "island_id": str(
                                    source_island.get("island_id") or ""
                                ),
                                "source_start_ms": start_ms,
                                "source_end_ms": end_ms,
                                **window,
                                "measured_duration_ms": 0,
                                "delta_ms": 0,
                                "status": "unmeasured_bootstrap",
                                "raw_isolated_duration_fit": False,
                                "speaker_rate_percent": state["base_rate"],
                            }
                        )
                    island_count = len(original_block.source_speech_islands)
                    localized_work[index] = {
                        "seed_delivery": None,
                        "spoken_islands": [""] * island_count,
                        "tts_islands": [""] * island_count,
                        "measurements": bootstrap_measurements,
                        "pending_positions": set(range(island_count)),
                        "rounds": [],
                        "bootstrap_from_source": True,
                    }
                    state["case"]["localized_breath_group_repair"] = {
                        "schema_version": (
                            "mathula-coordinated-breath-group-repair-v3-v13.19.10"
                        ),
                        "status": "bootstrap_from_source_breath_groups",
                        "speaker_tempo_locked": True,
                        "source_onsets_locked": True,
                        "source_island_count": island_count,
                    }

                candidate_role_rank = {
                    "natural_fit": 0,
                    "alternate_grammar": 1,
                    "aggressive_fit": 2,
                }

                def localized_repair_positions(
                    work: Mapping[str, Any],
                ) -> list[int]:
                    """Include one fitted neighbour around each failing cluster.

                    A short isiZulu phrase can be impossible only because the
                    previous repair assigned part of its clause to the wrong
                    source breath group.  Adjacent fitted phrases are therefore
                    elastic support, not immutable linguistic walls.  We expose
                    one neighbour on each side, while the final Azure gate and
                    whole-region semantic review remain authoritative.
                    """

                    pending = {
                        int(value) for value in work["pending_positions"]
                    }
                    repair = set(pending)
                    island_count = len(work["measurements"])
                    for position in tuple(pending):
                        for neighbour in (position - 1, position + 1):
                            if not 0 <= neighbour < island_count:
                                continue
                            if neighbour in pending:
                                continue
                            current_text = str(
                                work["spoken_islands"][neighbour] or ""
                            ).strip()
                            measurement = work["measurements"][neighbour]
                            if (
                                current_text
                                and bool(
                                    measurement.get(
                                        "raw_isolated_duration_fit"
                                    )
                                )
                            ):
                                repair.add(neighbour)
                    return sorted(repair)

                def localized_cluster_map(
                    positions: Sequence[int],
                ) -> dict[int, str]:
                    mapping: dict[int, str] = {}
                    cluster_number = 0
                    previous_position: int | None = None
                    for position in positions:
                        if (
                            previous_position is None
                            or position != previous_position + 1
                        ):
                            cluster_number += 1
                        mapping[position] = f"cluster_{cluster_number:02d}"
                        previous_position = position
                    return mapping

                for localized_round in range(1, 3):
                    request_repairs: list[dict[str, Any]] = []
                    request_indices: list[int] = []
                    for index, work in localized_work.items():
                        if index in results or not work["pending_positions"]:
                            continue
                        state = states[index]
                        original_block = state["original_block"]
                        groups: list[dict[str, Any]] = []
                        pending_set = {
                            int(value) for value in work["pending_positions"]
                        }
                        repair_positions = localized_repair_positions(work)
                        repair_cluster_by_position = localized_cluster_map(
                            repair_positions
                        )
                        cluster_sizes: dict[str, int] = {}
                        for cluster_id in repair_cluster_by_position.values():
                            cluster_sizes[cluster_id] = (
                                cluster_sizes.get(cluster_id, 0) + 1
                            )
                        cluster_offsets: dict[str, int] = {}
                        for position in repair_positions:
                            source_island = original_block.source_speech_islands[position]
                            measurement = work["measurements"][position]
                            delta_ms = int(measurement.get("delta_ms") or 0)
                            cluster_id = repair_cluster_by_position[position]
                            cluster_offset = cluster_offsets.get(cluster_id, 0) + 1
                            cluster_offsets[cluster_id] = cluster_offset
                            groups.append(
                                {
                                    "island_id": str(source_island.get("island_id") or ""),
                                    "repair_cluster_id": cluster_id,
                                    "repair_cluster_position": cluster_offset,
                                    "repair_cluster_size": cluster_sizes[cluster_id],
                                    "source_text": str(source_island.get("text") or "").strip(),
                                    "current_spoken_text": work["spoken_islands"][position],
                                    "left_context_spoken_text": (
                                        work["spoken_islands"][position - 1]
                                        if position > 0 else ""
                                    ),
                                    "right_context_spoken_text": (
                                        work["spoken_islands"][position + 1]
                                        if position + 1 < len(work["spoken_islands"])
                                        else ""
                                    ),
                                    "target_duration_ms": int(
                                        measurement.get("target_duration_ms") or 0
                                    ),
                                    "minimum_duration_ms": int(
                                        measurement.get("minimum_duration_ms") or 0
                                    ),
                                    "maximum_duration_ms": int(
                                        measurement.get("maximum_duration_ms") or 0
                                    ),
                                    "measured_duration_ms": int(
                                        measurement.get("measured_duration_ms") or 0
                                    ),
                                    "delta_ms": delta_ms,
                                    "repair_role": (
                                        "failing"
                                        if position in pending_set
                                        else "elastic_support"
                                    ),
                                    "action": (
                                        "translate_and_fit"
                                        if not work["spoken_islands"][position]
                                        else "support"
                                        if position not in pending_set
                                        else "contract" if delta_ms > 0
                                        else "expand" if delta_ms < 0
                                        else "hold"
                                    ),
                                }
                            )
                        if not groups:
                            continue
                        request_indices.append(index)
                        request_repairs.append(
                            {
                                "unit_id": _source_block_id(original_block.block_id),
                                "speaker_id": original_block.speaker_id,
                                "source_text": original_block.source_text,
                                "approved_translation": original_block.translated_text,
                                "required_facts": list(original_block.required_facts),
                                "protected_entities": list(original_block.protected_entities),
                                "locked_speaker_rate_percent": state["base_rate"],
                                "round": localized_round,
                                "breath_groups": groups,
                            }
                        )
                    if not request_repairs:
                        break

                    progress.emit(
                        "semantic_fit",
                        (
                            "coordinated breath-group repair round "
                            f"{localized_round}: repairing "
                            f"{sum(len(value['breath_groups']) for value in request_repairs)} "
                            f"phrase(s) across {len(request_repairs)} region(s)"
                        ),
                        status="running",
                        current=len(results),
                        total=len(blocks),
                        localized_breath_group_repair=True,
                        localized_round=localized_round,
                        scheduled_count=len(request_repairs),
                    )
                    repair_batches: list[list[dict[str, Any]]] = []
                    current_batch: list[dict[str, Any]] = []
                    current_group_count = 0
                    for repair in request_repairs:
                        repair_group_count = len(repair.get("breath_groups") or [])
                        if current_batch and (
                            len(current_batch) >= LOCALIZED_REPAIR_MAX_UNITS_PER_CALL
                            or current_group_count + repair_group_count
                            > LOCALIZED_REPAIR_MAX_GROUPS_PER_CALL
                        ):
                            repair_batches.append(current_batch)
                            current_batch = []
                            current_group_count = 0
                        current_batch.append(repair)
                        current_group_count += repair_group_count
                    if current_batch:
                        repair_batches.append(current_batch)

                    returned_by_id: dict[str, Mapping[str, Any]] = {}
                    response_artifact_by_id: dict[str, Any] = {}
                    for batch_number, batch_repairs in enumerate(
                        repair_batches, start=1
                    ):
                        repair_request = {
                            "schema_version": (
                                "mathula-localized-breath-group-repair-request-v4-coordinated-cluster-v13.19.10"
                            ),
                            "batch_number": batch_number,
                            "batch_count": len(repair_batches),
                            "repairs": batch_repairs,
                        }
                        progress.emit(
                            "semantic_fit",
                            (
                                "coordinated breath-group repair round "
                                f"{localized_round} batch {batch_number}/"
                                f"{len(repair_batches)}"
                            ),
                            status="running",
                            current=len(results),
                            total=len(blocks),
                            localized_breath_group_repair=True,
                            localized_round=localized_round,
                            batch_number=batch_number,
                            batch_count=len(repair_batches),
                            batch_region_count=len(batch_repairs),
                            batch_phrase_count=sum(
                                len(value.get("breath_groups") or [])
                                for value in batch_repairs
                            ),
                        )
                        response = call_with_rate_limit_resume(
                            (
                                "coordinated breath-group repair "
                                f"{batch_number}/{len(repair_batches)}"
                            ),
                            localized_phrase_method,
                            repair_request,
                        )
                        data = (
                            response.data if hasattr(response, "data") else response
                        )
                        returned = (
                            data.get("repairs")
                            if isinstance(data, Mapping)
                            else None
                        )
                        if not isinstance(returned, list):
                            raise DirectDubError(
                                "Coordinated breath-group repair returned no repairs"
                            )
                        response_artifact = (
                            response.to_artifact()
                            if hasattr(response, "to_artifact")
                            else None
                        )
                        for item in returned:
                            if not isinstance(item, Mapping):
                                continue
                            returned_unit_id = str(item.get("unit_id") or "")
                            if not returned_unit_id:
                                continue
                            returned_by_id[returned_unit_id] = item
                            response_artifact_by_id[returned_unit_id] = {
                                "batch_number": batch_number,
                                "batch_count": len(repair_batches),
                                "provider_response": response_artifact,
                            }

                    for index in request_indices:
                        state = states[index]
                        original_block = state["original_block"]
                        unit_id = _source_block_id(original_block.block_id)
                        work = localized_work[index]
                        returned_unit = returned_by_id.get(unit_id)
                        if not isinstance(returned_unit, Mapping):
                            continue
                        group_results = {
                            str(item.get("island_id") or ""): item
                            for item in (returned_unit.get("breath_groups") or [])
                            if isinstance(item, Mapping)
                        }
                        round_audit: dict[str, Any] = {
                            "round": localized_round,
                            "response": response_artifact_by_id.get(unit_id),
                            "groups": [],
                        }
                        next_pending: set[int] = set()
                        original_pending = {
                            int(value) for value in work["pending_positions"]
                        }
                        repair_positions = localized_repair_positions(work)
                        cluster_by_position = localized_cluster_map(
                            repair_positions
                        )
                        clusters: dict[str, list[int]] = {}
                        for position in repair_positions:
                            clusters.setdefault(
                                cluster_by_position[position], []
                            ).append(position)

                        measured_by_position: dict[
                            int, dict[str, dict[str, Any]]
                        ] = {}
                        candidate_audit_by_position: dict[
                            int, list[dict[str, Any]]
                        ] = {}
                        for position in repair_positions:
                            source_island = original_block.source_speech_islands[position]
                            island_id = str(source_island.get("island_id") or "")
                            group_result = group_results.get(island_id)
                            if not isinstance(group_result, Mapping):
                                next_pending.add(position)
                                round_audit["groups"].append(
                                    {
                                        "island_id": island_id,
                                        "repair_cluster_id": cluster_by_position[position],
                                        "repair_role": (
                                            "failing"
                                            if position in original_pending
                                            else "elastic_support"
                                        ),
                                        "status": "provider_group_missing",
                                    }
                                )
                                continue
                            candidates = [
                                value
                                for value in (group_result.get("candidates") or [])
                                if isinstance(value, Mapping)
                            ]
                            role_candidates: dict[str, dict[str, Any]] = {}
                            candidate_audits: list[dict[str, Any]] = []
                            for candidate in candidates:
                                spoken_candidate = " ".join(
                                    str(candidate.get("spoken_text") or "").split()
                                ).strip()
                                raw_tts_candidate = " ".join(
                                    str(candidate.get("tts_text") or spoken_candidate).split()
                                ).strip()
                                if not spoken_candidate:
                                    continue
                                candidate_id = str(candidate.get("candidate_id") or "")
                                tts_candidate, pronunciation_substitutions = (
                                    self._apply_candidate_pronunciation(
                                        spoken_candidate,
                                        raw_tts_candidate,
                                        pronunciation_dictionary,
                                    )
                                )
                                measurement = self._measure_single_source_breath_group_candidate(
                                    original_block,
                                    artifacts,
                                    source_island=source_island,
                                    island_index=position + 1,
                                    text=tts_candidate,
                                    voice=state["voice"],
                                    base_rate_percent=state["base_rate"],
                                    pitch_percent=state["pitch"],
                                    volume_percent=state["volume"],
                                    tolerance_ms=state["alignment_tolerance_ms"],
                                )
                                measured = {
                                    "candidate_id": candidate_id,
                                    "spoken_text": spoken_candidate,
                                    "tts_text": tts_candidate,
                                    "pronunciation_substitutions": (
                                        pronunciation_substitutions
                                    ),
                                    "measurement": measurement,
                                }
                                role_candidates[candidate_id] = measured
                                candidate_audits.append(
                                    {
                                        "candidate_id": candidate_id,
                                        "spoken_text": spoken_candidate,
                                        "measurement": dict(measurement),
                                    }
                                )
                            if not role_candidates:
                                next_pending.add(position)
                                round_audit["groups"].append(
                                    {
                                        "island_id": island_id,
                                        "repair_cluster_id": cluster_by_position[position],
                                        "repair_role": (
                                            "failing"
                                            if position in original_pending
                                            else "elastic_support"
                                        ),
                                        "status": "no_measurable_candidates",
                                    }
                                )
                                continue
                            measured_by_position[position] = role_candidates
                            candidate_audit_by_position[position] = candidate_audits

                        round_audit["clusters"] = []
                        role_order = (
                            "natural_fit",
                            "alternate_grammar",
                            "aggressive_fit",
                        )
                        for cluster_id, cluster_positions in clusters.items():
                            if any(
                                position not in measured_by_position
                                for position in cluster_positions
                            ):
                                next_pending.update(cluster_positions)
                                round_audit["clusters"].append(
                                    {
                                        "repair_cluster_id": cluster_id,
                                        "status": "incomplete_provider_cluster",
                                        "positions": list(cluster_positions),
                                    }
                                )
                                continue

                            available_roles = [
                                role
                                for role in role_order
                                if all(
                                    role in measured_by_position[position]
                                    for position in cluster_positions
                                )
                            ]
                            if not available_roles:
                                next_pending.update(cluster_positions)
                                round_audit["clusters"].append(
                                    {
                                        "repair_cluster_id": cluster_id,
                                        "status": "no_coherent_candidate_role",
                                        "positions": list(cluster_positions),
                                    }
                                )
                                continue

                            def cluster_role_score(role: str) -> tuple[int, int, int, int, int]:
                                outside_count = 0
                                maximum_window_error_ms = 0
                                total_window_error_ms = 0
                                total_target_delta_ms = 0
                                for cluster_position in cluster_positions:
                                    measurement = measured_by_position[
                                        cluster_position
                                    ][role]["measurement"]
                                    measured_ms = int(
                                        measurement.get("measured_duration_ms") or 0
                                    )
                                    minimum_ms = int(
                                        measurement.get("minimum_duration_ms") or 0
                                    )
                                    maximum_ms = int(
                                        measurement.get("maximum_duration_ms") or 0
                                    )
                                    if measured_ms < minimum_ms:
                                        window_error_ms = minimum_ms - measured_ms
                                        outside_count += 1
                                    elif measured_ms > maximum_ms:
                                        window_error_ms = measured_ms - maximum_ms
                                        outside_count += 1
                                    else:
                                        window_error_ms = 0
                                    maximum_window_error_ms = max(
                                        maximum_window_error_ms,
                                        window_error_ms,
                                    )
                                    total_window_error_ms += window_error_ms
                                    total_target_delta_ms += abs(
                                        int(measurement.get("delta_ms") or 0)
                                    )
                                return (
                                    outside_count,
                                    maximum_window_error_ms,
                                    total_window_error_ms,
                                    total_target_delta_ms,
                                    candidate_role_rank.get(role, 99),
                                )

                            selected_role = min(
                                available_roles,
                                key=cluster_role_score,
                            )
                            selected_score = cluster_role_score(selected_role)
                            cluster_fits = selected_score[0] == 0
                            round_audit["clusters"].append(
                                {
                                    "repair_cluster_id": cluster_id,
                                    "status": (
                                        "fit" if cluster_fits
                                        else "still_outside_window"
                                    ),
                                    "selected_candidate_id": selected_role,
                                    "positions": list(cluster_positions),
                                    "score": {
                                        "outside_count": selected_score[0],
                                        "maximum_window_error_ms": selected_score[1],
                                        "total_window_error_ms": selected_score[2],
                                        "total_target_delta_ms": selected_score[3],
                                    },
                                }
                            )

                            for position in cluster_positions:
                                source_island = original_block.source_speech_islands[position]
                                island_id = str(source_island.get("island_id") or "")
                                selected = measured_by_position[position][selected_role]
                                measurement = selected["measurement"]
                                work["spoken_islands"][position] = selected["spoken_text"]
                                work["tts_islands"][position] = selected["tts_text"]
                                work["measurements"][position] = dict(measurement)
                                fits = bool(
                                    measurement.get("raw_isolated_duration_fit")
                                )
                                if not fits:
                                    next_pending.add(position)
                                round_audit["groups"].append(
                                    {
                                        "island_id": island_id,
                                        "repair_cluster_id": cluster_id,
                                        "repair_role": (
                                            "failing"
                                            if position in original_pending
                                            else "elastic_support"
                                        ),
                                        "status": (
                                            "fit" if fits
                                            else "still_outside_window"
                                        ),
                                        "selected_candidate_id": selected_role,
                                        "selected_spoken_text": selected["spoken_text"],
                                        "measurement": dict(measurement),
                                        "candidate_measurements": candidate_audit_by_position.get(
                                            position, []
                                        ),
                                    }
                                )
                        work["pending_positions"] = next_pending
                        work["rounds"].append(round_audit)
                        state["case"]["localized_breath_group_repair"] = {
                            "schema_version": (
                                "mathula-coordinated-breath-group-repair-v3-v13.19.10"
                            ),
                            "status": (
                                "timing_fit_pending_semantic_review"
                                if not next_pending
                                else "timing_repair_pending"
                            ),
                            "rounds": list(work["rounds"]),
                            "remaining_island_ids": [
                                str(original_block.source_speech_islands[position].get("island_id") or "")
                                for position in sorted(next_pending)
                            ],
                            "speaker_tempo_locked": True,
                            "source_onsets_locked": True,
                        }
                    atomic_write_json(artifacts.semantic_fit, checkpoint)

                localized_prepared_v2: dict[int, dict[str, Any]] = {}
                localized_review_payloads_v2: list[dict[str, Any]] = []
                for index, work in localized_work.items():
                    if index in results or work["pending_positions"]:
                        state = states[index]
                        if work["pending_positions"]:
                            state["case"]["status"] = (
                                "localized_breath_group_timing_exhausted"
                            )
                        continue
                    state = states[index]
                    original_block = state["original_block"]
                    spoken = " ".join(work["spoken_islands"]).strip()
                    tts = " ".join(work["tts_islands"]).strip()
                    variant = PreparedTranslationVariant(
                        variant_id="semantic_fit_phrase_only_breath_group_repair",
                        spoken_text=spoken,
                        tts_text=tts,
                        target_duration_ms=state["preferred"],
                        estimated_duration_ms=None,
                        meaning_preserved=True,
                        omitted_optional_details=(),
                    )
                    candidate_block = replace(
                        original_block,
                        block_id=(
                            f"{_source_block_id(original_block.block_id)}"
                            "_variant_semantic_fit_phrase_only_breath_group_repair"
                        ),
                        translated_text=spoken,
                        tts_text=tts,
                        variants=(*original_block.variants, variant),
                        selected_variant_id=variant.variant_id,
                        semantic_target_islands=tuple(work["tts_islands"]),
                    )
                    review_key = (
                        f"{_source_block_id(original_block.block_id)}"
                        "::coordinated-breath-group-repair-v13.19.10"
                    )
                    review_payload = self._semantic_fit_review_payload(
                        original_block,
                        spoken_text=spoken,
                        tts_text=tts,
                        omitted_details=(),
                        measured_duration_ms=sum(
                            int(value.get("measured_duration_ms") or 0)
                            for value in work["measurements"]
                        ),
                    )
                    review_payload["unit_id"] = review_key
                    localized_review_payloads_v2.append(review_payload)
                    localized_prepared_v2[index] = {
                        "block": candidate_block,
                        "review_key": review_key,
                        "work": work,
                    }

                localized_reviews_v2: dict[str, Mapping[str, Any]] = {}
                if localized_review_payloads_v2:
                    review_method = getattr(provider, "review_semantic_fits", None)
                    if callable(review_method):
                        response = call_with_rate_limit_resume(
                            "coordinated breath-group semantic review",
                            review_method,
                            {
                                "schema_version": SEMANTIC_FIT_SCHEMA_VERSION,
                                "operation": "semantic_fit_review_batch",
                                "reviews": localized_review_payloads_v2,
                            },
                        )
                        data = response.data if hasattr(response, "data") else response
                        returned = data.get("reviews") if isinstance(data, Mapping) else None
                        if isinstance(returned, list):
                            localized_reviews_v2 = {
                                str(item.get("unit_id") or ""): item.get("review")
                                for item in returned
                                if isinstance(item, Mapping)
                                and isinstance(item.get("review"), Mapping)
                            }

                for index, prepared in localized_prepared_v2.items():
                    if index in results:
                        continue
                    state = states[index]
                    original_block = state["original_block"]
                    review = localized_reviews_v2.get(prepared["review_key"])
                    cache = state["case"].setdefault(
                        "localized_breath_group_repair", {}
                    )
                    if not (
                        isinstance(review, Mapping)
                        and self._semantic_review_passed(review)
                    ):
                        if isinstance(cache, dict):
                            cache["status"] = "semantic_review_rejection"
                            if isinstance(review, Mapping):
                                cache["review"] = dict(review)
                        state["case"]["status"] = (
                            "localized_breath_group_semantic_rejection"
                        )
                        continue
                    candidate_block = prepared["block"]
                    try:
                        fitted = self._synthesize_source_anchored_breath_groups(
                            candidate_block,
                            artifacts,
                            island_texts=candidate_block.semantic_target_islands,
                            voice=state["voice"],
                            base_rate_percent=state["base_rate"],
                            pitch_percent=state["pitch"],
                            volume_percent=state["volume"],
                            tolerance_ms=state["alignment_tolerance_ms"],
                        )
                    except (TimingOverflowError, ValueError, DirectDubError) as exc:
                        if isinstance(cache, dict):
                            cache["status"] = "final_timing_rejection"
                            cache["reason"] = str(exc)
                        state["case"]["status"] = (
                            "localized_breath_group_final_timing_rejection"
                        )
                        continue

                    audit = {
                        "attempt": "coordinated_breath_group_repair_v13.19.10",
                        "status": "selected_phrase_only_breath_group_repair",
                        "selection_reason": (
                            "repair_adjacent_breath_group_clusters_with_azure_closed_loop"
                        ),
                        "semantic_authority": dict(review),
                        "speaker_tempo_locked": True,
                        "source_lipsync_locked": True,
                        "post_synthesis_time_stretch_ratio": 1.0,
                        "selected_azure_rate_percent": state["base_rate"],
                        "human_review_required": False,
                    }
                    state["audits"].append(audit)
                    state["case"]["status"] = (
                        "selected_phrase_only_breath_group_repair"
                    )
                    state["case"]["selected_attempt"] = (
                        "coordinated_breath_group_repair_v13.19.10"
                    )
                    state["case"]["production_rescue"] = dict(audit)
                    if isinstance(cache, dict):
                        cache["status"] = "selected"
                        cache["review"] = dict(review)
                    fitted["translation_variant"] = {
                        "schema_version": "mathula-selected-translation-variant-v1",
                        "source_block_id": _source_block_id(original_block.block_id),
                        "selected_variant_id": candidate_block.selected_variant_id,
                        "spoken_text": candidate_block.translated_text,
                        "measured_duration_ms": fitted.get("duration_ms"),
                        "meaning_preserved": True,
                        "selection_reason": (
                            "phrase_only_breath_group_repair_at_source_onsets"
                        ),
                        "human_review_required": False,
                    }
                    fitted["semantic_fit"] = {
                        "schema_version": SEMANTIC_FIT_SCHEMA_VERSION,
                        "strategy_version": SEMANTIC_FIT_BATCH_STRATEGY_VERSION,
                        "selected_attempt": (
                            "coordinated_breath_group_repair_v13.19.10"
                        ),
                        "production_rescue": True,
                        "rescue_mode": (
                            "phrase_only_source_anchored_breath_groups"
                        ),
                        "speaker_tempo_locked": True,
                        "source_lipsync_locked": True,
                        "time_stretch_used": False,
                        "semantic_authority": dict(review),
                    }
                    fitted["translation_variant_attempts"] = state["audits"]
                    results[index] = (
                        candidate_block,
                        fitted,
                        state["audits"],
                    )
                    progress.emit(
                        "semantic_fit",
                        (
                            f"{state['block'].block_id}: repaired only failing "
                            "breath groups with coordinated Azure-closed-loop fit"
                        ),
                        status="completed",
                        current=len(results),
                        total=len(blocks),
                        block_id=state["block"].block_id,
                        localized_breath_group_repair=True,
                        phrase_only_repair=True,
                        speaker_tempo_locked=True,
                    )
                    atomic_write_json(artifacts.semantic_fit, checkpoint)

            # Final professional-dubbing repair: if every semantically safe
            # whole-region/approved variant still has one or more breath groups
            # outside the mouth-active windows, do not regenerate the whole
            # sentence and do not alter speaker tempo.  Freeze every breath
            # group that already fits and ask the dialogue adaptor to rewrite
            # only the failing groups.  All pending blocks share one provider
            # call, so production latency scales with one localized batch rather
            # than one GPT request per phrase/block.
            if (
                localized_repair_seeds
                and portfolio_supported
                and not localized_phrase_supported
            ):
                localized_payloads: list[dict[str, Any]] = []
                localized_active: list[int] = []
                localized_candidate_by_index: dict[int, Mapping[str, Any]] = {}
                localized_fingerprint_by_index: dict[int, str] = {}

                def _repair_text_key(value: str) -> str:
                    return " ".join(str(value).casefold().split())

                for index, seed_info in localized_repair_seeds.items():
                    if index in results:
                        continue
                    state = states[index]
                    original_block = state["original_block"]
                    delivery = seed_info["delivery"]
                    seed_block = delivery["block"]
                    measurements = [
                        dict(value) for value in delivery["measurements"]
                    ]
                    seed_spoken_islands = tuple(
                        str(value).strip()
                        for value in (
                            delivery.get("spoken_islands")
                            or seed_block.semantic_target_islands
                        )
                    )
                    if len(seed_spoken_islands) != len(measurements):
                        state["case"]["status"] = (
                            "localized_breath_group_repair_invalid_seed"
                        )
                        continue

                    repair_island_ids = {
                        str(value.get("island_id") or "")
                        for value in measurements
                        if not bool(value.get("raw_isolated_duration_fit"))
                    }
                    if not repair_island_ids:
                        continue
                    frozen_island_ids = [
                        str(value.get("island_id") or "")
                        for value in measurements
                        if bool(value.get("raw_isolated_duration_fit"))
                    ]
                    synthetic_evidence = {
                        "attempt": "localized_seed",
                        "status": "localized_breath_group_miss",
                        "spoken_text": seed_block.translated_text,
                        "tts_text": seed_block.tts_text,
                        "measured_duration_ms": sum(
                            int(value.get("measured_duration_ms") or 0)
                            for value in measurements
                        ),
                        "plain_speech_duration_ms": sum(
                            int(value.get("measured_duration_ms") or 0)
                            for value in measurements
                        ),
                        "speaker_rate_percent": state["base_rate"],
                        "target_island_texts": list(seed_spoken_islands),
                        "performance_island_measurements": measurements,
                    }
                    payload = self._semantic_fit_payload(
                        seed_block,
                        voice=state["voice"],
                        base_rate_percent=state["base_rate"],
                        pitch_percent=state["pitch"],
                        volume_percent=state["volume"],
                        attempt_number=1,
                        target_minimum_ms=state["minimum"],
                        target_preferred_ms=state["preferred"],
                        target_maximum_ms=state["maximum"],
                        duration_evidence=[synthetic_evidence],
                        rejected_spoken_texts=(),
                        current_spoken_text=seed_block.translated_text,
                        neighbouring_blocks=(
                            *blocks[max(0, index - 2):index],
                            *blocks[index + 1:index + 3],
                        ),
                    )
                    repair_groups: list[dict[str, Any]] = []
                    for island_text, measurement in zip(
                        seed_spoken_islands,
                        measurements,
                        strict=True,
                    ):
                        delta_ms = int(measurement.get("delta_ms") or 0)
                        island_id = str(measurement.get("island_id") or "")
                        repair_groups.append(
                            {
                                "island_id": island_id,
                                "current_spoken_text": island_text,
                                "target_duration_ms": int(
                                    measurement.get("target_duration_ms") or 0
                                ),
                                "azure_measured_duration_ms": int(
                                    measurement.get("measured_duration_ms") or 0
                                ),
                                "delta_ms": delta_ms,
                                "action": (
                                    "contract"
                                    if delta_ms > 0
                                    else "expand"
                                    if delta_ms < 0
                                    else "hold"
                                ),
                                "locked": island_id not in repair_island_ids,
                            }
                        )
                    payload["candidate_count"] = 1
                    payload["localized_breath_group_repair"] = {
                        "schema_version": (
                            "mathula-localized-breath-group-repair-v1-v13.19.5"
                        ),
                        "seed_variant": str(delivery["attempt"]),
                        "source_tempo_locked": True,
                        "source_onsets_locked": True,
                        "repair_island_ids": sorted(repair_island_ids),
                        "frozen_island_ids": frozen_island_ids,
                        "breath_groups": repair_groups,
                        "contract": (
                            "Rewrite ONLY island_deliveries whose island_id is in "
                            "repair_island_ids. Every frozen island must preserve "
                            "current_spoken_text exactly except whitespace. For an "
                            "overflowing island, find a shorter idiomatic isiZulu "
                            "construction; for an underfilled island, restore only "
                            "source-grounded wording. Preserve every required fact, "
                            "protected entity, number and relationship. Do not move "
                            "words into a frozen neighbouring island, do not change "
                            "speaker tempo, and do not merge across source pauses."
                        ),
                    }
                    payload["portfolio_contract"] = {
                        "candidate_order": "localized_breath_group_repair",
                        "candidate_roles": ["localized_breath_group_repair"],
                        "distinct_grammatical_contractions_required": False,
                        "every_candidate_must_preserve_every_required_fact": True,
                        "azure_measured_frontier_is_authoritative": True,
                        "source_tempo_is_primary_and_immutable": True,
                        "performance_beats_required": True,
                        "localized_breath_group_repair": True,
                    }
                    fingerprint = hashlib.sha256(
                        json.dumps(
                            payload,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest()
                    localized_fingerprint_by_index[index] = fingerprint
                    cached_localized = state["case"].get(
                        "localized_breath_group_repair"
                    )
                    if (
                        isinstance(cached_localized, Mapping)
                        and cached_localized.get("request_fingerprint")
                        == fingerprint
                        and isinstance(cached_localized.get("candidate"), Mapping)
                    ):
                        localized_candidate_by_index[index] = cached_localized[
                            "candidate"
                        ]
                        continue
                    localized_active.append(index)
                    localized_payloads.append(payload)

                if localized_payloads:
                    progress.emit(
                        "semantic_fit",
                        (
                            "localized breath-group repair: rewriting only "
                            f"overflowing/underfilled phrases for {len(localized_payloads)} "
                            "region(s)"
                        ),
                        status="running",
                        current=len(results),
                        total=len(blocks),
                        localized_breath_group_repair=True,
                        scheduled_count=len(localized_payloads),
                    )
                    batch_payload = {
                        "schema_version": SEMANTIC_FIT_SCHEMA_VERSION,
                        "operation": "semantic_fit_localized_breath_group_repair",
                        "repairs": localized_payloads,
                    }
                    response = call_with_rate_limit_resume(
                        "localized breath-group repair",
                        portfolio_method,
                        batch_payload,
                    )
                    data = response.data if hasattr(response, "data") else response
                    portfolios = (
                        data.get("portfolios") if isinstance(data, Mapping) else None
                    )
                    if not isinstance(portfolios, list):
                        raise DirectDubError(
                            "Localized breath-group repair returned no portfolios"
                        )
                    returned_by_id = {
                        str(item.get("unit_id") or ""): item.get("candidates")
                        for item in portfolios
                        if isinstance(item, Mapping)
                    }
                    for index in localized_active:
                        state = states[index]
                        unit_id = _source_block_id(
                            state["original_block"].block_id
                        )
                        candidates = returned_by_id.get(unit_id)
                        if not isinstance(candidates, list) or len(candidates) != 1:
                            state["case"]["localized_breath_group_repair"] = {
                                "request_fingerprint": localized_fingerprint_by_index[index],
                                "status": "provider_candidate_count_error",
                            }
                            continue
                        raw_candidate = candidates[0]
                        candidate = (
                            raw_candidate.get("candidate")
                            if isinstance(raw_candidate, Mapping)
                            else None
                        )
                        if not isinstance(candidate, Mapping):
                            state["case"]["localized_breath_group_repair"] = {
                                "request_fingerprint": localized_fingerprint_by_index[index],
                                "status": "provider_candidate_missing",
                            }
                            continue
                        localized_candidate_by_index[index] = candidate
                        state["case"]["localized_breath_group_repair"] = {
                            "schema_version": (
                                "mathula-localized-breath-group-repair-cache-v1-v13.19.5"
                            ),
                            "request_fingerprint": localized_fingerprint_by_index[index],
                            "status": "candidate_received",
                            "candidate": dict(candidate),
                            "response": (
                                response.to_artifact()
                                if hasattr(response, "to_artifact")
                                else None
                            ),
                        }
                    atomic_write_json(artifacts.semantic_fit, checkpoint)

                localized_prepared: dict[int, dict[str, Any]] = {}
                localized_review_payloads: list[dict[str, Any]] = []
                for index, candidate in localized_candidate_by_index.items():
                    if index in results or index not in localized_repair_seeds:
                        continue
                    state = states[index]
                    original_block = state["original_block"]
                    seed_delivery = localized_repair_seeds[index]["delivery"]
                    seed_block = seed_delivery["block"]
                    seed_spoken_islands = tuple(
                        str(value).strip()
                        for value in (
                            seed_delivery.get("spoken_islands")
                            or seed_block.semantic_target_islands
                        )
                    )
                    measurements = [
                        dict(value) for value in seed_delivery["measurements"]
                    ]
                    repair_island_ids = {
                        str(value.get("island_id") or "")
                        for value in measurements
                        if not bool(value.get("raw_isolated_duration_fit"))
                    }
                    try:
                        spoken, tts, omissions = self._validate_finalization_candidate(
                            original_block,
                            candidate,
                        )
                        candidate_spoken_islands = self._semantic_target_islands(
                            original_block,
                            candidate,
                            spoken,
                        )
                        if len(candidate_spoken_islands) != len(seed_spoken_islands):
                            raise ValueError(
                                "localized breath-group repair changed island count"
                            )
                        expected_ids = [
                            str(value.get("island_id") or "")
                            for value in original_block.source_speech_islands
                        ]
                        for island_id, before, after in zip(
                            expected_ids,
                            seed_spoken_islands,
                            candidate_spoken_islands,
                            strict=True,
                        ):
                            if (
                                island_id not in repair_island_ids
                                and _repair_text_key(before)
                                != _repair_text_key(after)
                            ):
                                raise ValueError(
                                    f"localized repair changed frozen breath group {island_id}"
                                )
                        tts, pronunciation_substitutions = (
                            self._apply_candidate_pronunciation(
                                spoken,
                                tts,
                                pronunciation_dictionary,
                            )
                        )
                        candidate_tts_islands = tuple(
                            self._apply_candidate_pronunciation(
                                value,
                                value,
                                pronunciation_dictionary,
                            )[0]
                            for value in candidate_spoken_islands
                        )
                    except ValueError as exc:
                        cache = state["case"].setdefault(
                            "localized_breath_group_repair", {}
                        )
                        if isinstance(cache, dict):
                            cache["status"] = "local_validation_rejection"
                            cache["reason"] = str(exc)
                        continue

                    variant = PreparedTranslationVariant(
                        variant_id="semantic_fit_localized_breath_group_repair",
                        spoken_text=spoken,
                        tts_text=tts,
                        target_duration_ms=state["preferred"],
                        estimated_duration_ms=None,
                        meaning_preserved=True,
                        omitted_optional_details=tuple(omissions),
                    )
                    candidate_block = replace(
                        original_block,
                        block_id=(
                            f"{_source_block_id(original_block.block_id)}"
                            "_variant_semantic_fit_localized_breath_group_repair"
                        ),
                        translated_text=spoken,
                        tts_text=tts,
                        variants=(*original_block.variants, variant),
                        selected_variant_id=variant.variant_id,
                        semantic_target_islands=candidate_tts_islands,
                    )
                    review_key = (
                        f"{_source_block_id(original_block.block_id)}"
                        "::localized-breath-group-repair"
                    )
                    cache = state["case"].get("localized_breath_group_repair")
                    cached_review = (
                        cache.get("review")
                        if isinstance(cache, Mapping)
                        else None
                    )
                    localized_prepared[index] = {
                        "block": candidate_block,
                        "spoken_islands": tuple(candidate_spoken_islands),
                        "omissions": tuple(omissions),
                        "pronunciation_substitutions": pronunciation_substitutions,
                        "review_key": review_key,
                        "cached_review": cached_review,
                    }
                    if not isinstance(cached_review, Mapping):
                        review_payload = self._semantic_fit_review_payload(
                            original_block,
                            spoken_text=spoken,
                            tts_text=tts,
                            omitted_details=omissions,
                            measured_duration_ms=sum(
                                int(value.get("measured_duration_ms") or 0)
                                for value in measurements
                            ),
                        )
                        review_payload["unit_id"] = review_key
                        localized_review_payloads.append(review_payload)

                localized_reviews: dict[str, Mapping[str, Any]] = {}
                if localized_review_payloads:
                    review_method = getattr(provider, "review_semantic_fits", None)
                    if callable(review_method):
                        response = call_with_rate_limit_resume(
                            "localized breath-group semantic review",
                            review_method,
                            {
                                "schema_version": SEMANTIC_FIT_SCHEMA_VERSION,
                                "operation": "semantic_fit_review_batch",
                                "reviews": localized_review_payloads,
                            },
                        )
                        data = response.data if hasattr(response, "data") else response
                        returned = (
                            data.get("reviews") if isinstance(data, Mapping) else None
                        )
                        if isinstance(returned, list):
                            localized_reviews = {
                                str(item.get("unit_id") or ""): item.get("review")
                                for item in returned
                                if isinstance(item, Mapping)
                                and isinstance(item.get("review"), Mapping)
                            }

                for index, prepared in localized_prepared.items():
                    if index in results:
                        continue
                    state = states[index]
                    original_block = state["original_block"]
                    review = prepared.get("cached_review")
                    if not isinstance(review, Mapping):
                        review = localized_reviews.get(prepared["review_key"])
                    cache = state["case"].setdefault(
                        "localized_breath_group_repair", {}
                    )
                    if not (
                        isinstance(review, Mapping)
                        and self._semantic_review_passed(review)
                    ):
                        if isinstance(cache, dict):
                            cache["status"] = "semantic_review_rejection"
                            if isinstance(review, Mapping):
                                cache["review"] = dict(review)
                        continue
                    if isinstance(cache, dict):
                        cache["review"] = dict(review)
                    candidate_block = prepared["block"]
                    try:
                        fitted = self._synthesize_source_anchored_breath_groups(
                            candidate_block,
                            artifacts,
                            island_texts=candidate_block.semantic_target_islands,
                            voice=state["voice"],
                            base_rate_percent=state["base_rate"],
                            pitch_percent=state["pitch"],
                            volume_percent=state["volume"],
                            tolerance_ms=state["alignment_tolerance_ms"],
                        )
                    except (TimingOverflowError, ValueError, DirectDubError) as exc:
                        if isinstance(cache, dict):
                            cache["status"] = "timing_rejection"
                            cache["reason"] = str(exc)
                            cache["measured_duration_ms"] = getattr(
                                exc, "measured_duration_ms", None
                            )
                            cache["target_duration_ms"] = getattr(
                                exc, "target_duration_ms", None
                            )
                        continue

                    audit = {
                        "attempt": "localized_breath_group_repair",
                        "status": "selected_localized_breath_group_repair",
                        "selection_reason": (
                            "rewrite_only_failing_source_anchored_breath_groups"
                        ),
                        "semantic_authority": dict(review),
                        "speaker_tempo_locked": True,
                        "source_lipsync_locked": True,
                        "post_synthesis_time_stretch_ratio": 1.0,
                        "selected_azure_rate_percent": state["base_rate"],
                        "human_review_required": False,
                    }
                    state["audits"].append(audit)
                    state["case"]["status"] = (
                        "selected_localized_breath_group_repair"
                    )
                    state["case"]["selected_attempt"] = (
                        "localized_breath_group_repair"
                    )
                    state["case"]["production_rescue"] = dict(audit)
                    if isinstance(cache, dict):
                        cache["status"] = "selected"
                    fitted["translation_variant"] = {
                        "schema_version": "mathula-selected-translation-variant-v1",
                        "source_block_id": _source_block_id(original_block.block_id),
                        "selected_variant_id": candidate_block.selected_variant_id,
                        "spoken_text": candidate_block.translated_text,
                        "measured_duration_ms": fitted.get("duration_ms"),
                        "meaning_preserved": True,
                        "selection_reason": (
                            "localized_breath_group_repair_at_source_onsets"
                        ),
                        "human_review_required": False,
                    }
                    fitted["semantic_fit"] = {
                        "schema_version": SEMANTIC_FIT_SCHEMA_VERSION,
                        "strategy_version": SEMANTIC_FIT_BATCH_STRATEGY_VERSION,
                        "selected_attempt": "localized_breath_group_repair",
                        "production_rescue": True,
                        "rescue_mode": "localized_source_anchored_breath_groups",
                        "speaker_tempo_locked": True,
                        "source_lipsync_locked": True,
                        "time_stretch_used": False,
                        "semantic_authority": dict(review),
                    }
                    fitted["translation_variant_attempts"] = state["audits"]
                    results[index] = (
                        candidate_block,
                        fitted,
                        state["audits"],
                    )
                    progress.emit(
                        "semantic_fit",
                        (
                            f"{state['block'].block_id}: repaired only failing "
                            "breath groups at locked source tempo"
                        ),
                        status="completed",
                        current=len(results),
                        total=len(blocks),
                        block_id=state["block"].block_id,
                        localized_breath_group_repair=True,
                        speaker_tempo_locked=True,
                    )
                    atomic_write_json(artifacts.semantic_fit, checkpoint)

        unresolved = [
            states[index]["block"].block_id
            for index in states
            if index not in results
        ]
        if unresolved:
            for index, state in states.items():
                if index not in results:
                    state["case"]["status"] = "exhausted"
                    state["case"]["attempt_limit"] = strategy_attempt_limit(
                        state
                    )
            atomic_write_json(artifacts.semantic_fit, checkpoint)
            raise DirectDubError(
                "Semantic-fit batch exhausted the attempt budget for: "
                + ", ".join(unresolved)
                + f". Inspect {artifacts.semantic_fit}."
            )
        return results

    def _synthesize_semantic_fit(
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
        neighbouring_blocks: Sequence[SpeechBlock] = (),
        pronunciation_dictionary: PronunciationDictionary | None = None,
    ) -> tuple[SpeechBlock, dict[str, Any], list[dict[str, Any]]]:
        """Search until one locked-tempo delivery passes timing and meaning."""

        (
            target_minimum_ms,
            target_preferred_ms,
            target_maximum_ms,
        ) = self._semantic_fit_target_band(
            block,
            options.semantic_fit_tolerance_ms,
        )
        audits: list[dict[str, Any]] = []
        rejected_texts: list[str] = []
        evidence: list[dict[str, Any]] = []
        current_text = block.translated_text

        try:
            initial_fitted = self._synthesize_locked_tempo_block(
                block,
                artifacts,
                voice=voice,
                base_rate_percent=base_rate_percent,
                pitch_percent=pitch_percent,
                volume_percent=volume_percent,
                force=force,
                timing_deadline_ms=target_maximum_ms,
                timing_preferred_ms=target_preferred_ms,
            )
            initial_ms = _selected_azure_measurement_ms(initial_fitted) or 0
            initial_status = (
                "selected"
                if target_minimum_ms <= initial_ms <= target_maximum_ms
                else "underfill"
            )
            initial_attempt = {
                "attempt": 0,
                "candidate_id": "approved_natural",
                "status": initial_status,
                "spoken_text": block.translated_text,
                "measured_duration_ms": initial_ms,
                "target_minimum_ms": target_minimum_ms,
                "target_preferred_ms": target_preferred_ms,
                "target_maximum_ms": target_maximum_ms,
                "speaker_rate_percent": base_rate_percent,
                "time_stretch_ratio": 1.0,
                "semantic_review": "approved_translation",
            }
            audits.append(initial_attempt)
            evidence.append(initial_attempt)
            if initial_status == "selected":
                initial_fitted["translation_variant"] = {
                    "schema_version": "mathula-selected-translation-variant-v1",
                    "source_block_id": _source_block_id(block.block_id),
                    "selected_variant_id": "approved_natural",
                    "spoken_text": block.translated_text,
                    "measured_duration_ms": initial_ms,
                    "selection_reason": "semantic_fit_approved_text_already_fits",
                    "human_review_required": False,
                }
                initial_fitted["semantic_fit"] = {
                    "schema_version": SEMANTIC_FIT_SCHEMA_VERSION,
                    "selected_attempt": 0,
                    "speaker_tempo_locked": True,
                    "time_stretch_used": False,
                    "target_duration_band_ms": {
                        "minimum": target_minimum_ms,
                        "preferred": target_preferred_ms,
                        "maximum": target_maximum_ms,
                    },
                }
                initial_fitted["translation_variant_attempts"] = audits
                return block, initial_fitted, audits
            rejected_texts.append(block.translated_text)
        except TimingOverflowError as exc:
            initial_attempt = {
                "attempt": 0,
                "candidate_id": "approved_natural",
                "status": "overflow",
                "spoken_text": block.translated_text,
                "measured_duration_ms": exc.measured_duration_ms,
                "target_minimum_ms": target_minimum_ms,
                "target_preferred_ms": target_preferred_ms,
                "target_maximum_ms": target_maximum_ms,
                "speaker_rate_percent": base_rate_percent,
                "time_stretch_ratio": 1.0,
                "semantic_review": "approved_translation",
            }
            audits.append(initial_attempt)
            evidence.append(initial_attempt)
            rejected_texts.append(block.translated_text)

        try:
            provider = self._timing_repair_provider()
        except Exception as exc:
            raise DirectDubError(
                f"Semantic-fit provider is unavailable for {block.block_id}: {exc}"
            ) from exc
        if provider is None or not all(
            callable(getattr(provider, method, None))
            for method in ("fit_turn_semantically", "review_semantic_fit")
        ):
            raise DirectDubError(
                "Semantic-fit mode requires an AI provider with candidate "
                "generation and independent semantic review"
            )

        checkpoint = (
            read_json(artifacts.semantic_fit)
            if artifacts.semantic_fit.is_file()
            else {
                "schema_version": SEMANTIC_FIT_SCHEMA_VERSION,
                "job_id": artifacts.root.parent.name,
                "cases": {},
            }
        )
        if checkpoint.get("schema_version") != SEMANTIC_FIT_SCHEMA_VERSION:
            checkpoint = {
                "schema_version": SEMANTIC_FIT_SCHEMA_VERSION,
                "job_id": artifacts.root.parent.name,
                "cases": {},
            }
        cases = checkpoint.setdefault("cases", {})
        case = cases.setdefault(
            _source_block_id(block.block_id),
            {
                "speaker_id": block.speaker_id,
                "voice": voice,
                "speaker_rate_percent": base_rate_percent,
                "attempts": [],
            },
        )

        for attempt_number in range(1, options.semantic_fit_max_attempts + 1):
            payload = self._semantic_fit_payload(
                block,
                voice=voice,
                base_rate_percent=base_rate_percent,
                pitch_percent=pitch_percent,
                volume_percent=volume_percent,
                attempt_number=attempt_number,
                target_minimum_ms=target_minimum_ms,
                target_preferred_ms=target_preferred_ms,
                target_maximum_ms=target_maximum_ms,
                duration_evidence=evidence,
                rejected_spoken_texts=rejected_texts,
                current_spoken_text=current_text,
                neighbouring_blocks=neighbouring_blocks,
            )
            fingerprint = hashlib.sha256(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            cached = next(
                (
                    value
                    for value in case.get("attempts") or []
                    if value.get("request_fingerprint") == fingerprint
                    and isinstance(value.get("candidate"), Mapping)
                ),
                None,
            )
            if cached is not None:
                candidate_data = cached["candidate"]
                ai_called = False
            else:
                progress.emit(
                    "semantic_fit",
                    (
                        f"{block.block_id}: semantic-fit attempt "
                        f"{attempt_number}/{options.semantic_fit_max_attempts}"
                    ),
                    status="running",
                    current=block_number,
                    total=block_total,
                    block_id=block.block_id,
                    attempt=attempt_number,
                    measured_duration_ms=evidence[-1].get(
                        "measured_duration_ms"
                    ),
                    target_preferred_ms=target_preferred_ms,
                )
                response = provider.fit_turn_semantically(payload)  # type: ignore[attr-defined]
                candidate_data = (
                    response.data if hasattr(response, "data") else response
                )
                ai_called = True
                cached = {
                    "attempt": attempt_number,
                    "request_fingerprint": fingerprint,
                    "candidate": dict(candidate_data),
                    "response": (
                        response.to_artifact()
                        if hasattr(response, "to_artifact")
                        else None
                    ),
                }
                case.setdefault("attempts", []).append(cached)
                atomic_write_json(artifacts.semantic_fit, checkpoint)

            try:
                spoken, tts, omissions = self._validate_finalization_candidate(
                    block, candidate_data
                )
            except ValueError as exc:
                audit = {
                    "attempt": attempt_number,
                    "status": "local_semantic_rejection",
                    "reason": str(exc),
                    "ai_called": ai_called,
                }
                audits.append(audit)
                evidence.append(audit)
                cached.update(audit)
                atomic_write_json(artifacts.semantic_fit, checkpoint)
                continue
            tts, pronunciation_substitutions = (
                self._apply_candidate_pronunciation(
                    spoken,
                    tts,
                    pronunciation_dictionary,
                )
            )
            normalized_spoken = " ".join(spoken.casefold().split())
            if any(
                normalized_spoken == " ".join(value.casefold().split())
                for value in rejected_texts
            ):
                audit = {
                    "attempt": attempt_number,
                    "status": "duplicate_rejection",
                    "spoken_text": spoken,
                    "measured_duration_ms": evidence[-1].get(
                        "measured_duration_ms"
                    ),
                    "ai_called": ai_called,
                }
                audits.append(audit)
                evidence.append(audit)
                cached.update(audit)
                atomic_write_json(artifacts.semantic_fit, checkpoint)
                continue

            variant = PreparedTranslationVariant(
                variant_id=f"semantic_fit_{attempt_number:02d}",
                spoken_text=spoken,
                tts_text=tts,
                target_duration_ms=target_preferred_ms,
                estimated_duration_ms=None,
                meaning_preserved=True,
                omitted_optional_details=omissions,
            )
            candidate_block = replace(
                block,
                block_id=(
                    f"{_source_block_id(block.block_id)}_variant_"
                    f"{variant.variant_id}"
                ),
                translated_text=spoken,
                tts_text=tts,
                variants=(*block.variants, variant),
                selected_variant_id=variant.variant_id,
            )
            try:
                fitted = self._synthesize_locked_tempo_block(
                    candidate_block,
                    artifacts,
                    voice=voice,
                    base_rate_percent=base_rate_percent,
                    pitch_percent=pitch_percent,
                    volume_percent=volume_percent,
                    force=False,
                    timing_deadline_ms=target_maximum_ms,
                    timing_preferred_ms=target_preferred_ms,
                )
                measured_ms = _selected_azure_measurement_ms(fitted) or 0
                timing_status = (
                    "timing_fit"
                    if target_minimum_ms <= measured_ms <= target_maximum_ms
                    else "underfill"
                )
            except TimingOverflowError as exc:
                fitted = None
                measured_ms = exc.measured_duration_ms
                timing_status = "overflow"

            audit = {
                "attempt": attempt_number,
                "status": timing_status,
                "spoken_text": spoken,
                "tts_text": tts,
                "pronunciation_substitutions": pronunciation_substitutions,
                "omitted_optional_details": list(omissions),
                "measured_duration_ms": measured_ms,
                "target_minimum_ms": target_minimum_ms,
                "target_preferred_ms": target_preferred_ms,
                "target_maximum_ms": target_maximum_ms,
                "speaker_rate_percent": base_rate_percent,
                "time_stretch_ratio": 1.0,
                "ai_called": ai_called,
            }
            cached.update(audit)
            audits.append(audit)
            evidence.append(audit)
            current_text = spoken
            rejected_texts.append(spoken)
            atomic_write_json(artifacts.semantic_fit, checkpoint)
            if timing_status != "timing_fit" or fitted is None:
                continue

            review_payload = self._semantic_fit_review_payload(
                block,
                spoken_text=spoken,
                tts_text=tts,
                omitted_details=omissions,
                measured_duration_ms=measured_ms,
            )
            review_fingerprint = hashlib.sha256(
                json.dumps(
                    review_payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            if (
                cached.get("review_fingerprint") == review_fingerprint
                and isinstance(cached.get("review"), Mapping)
            ):
                review = cached["review"]
                review_called = False
            else:
                response = provider.review_semantic_fit(review_payload)  # type: ignore[attr-defined]
                review = response.data if hasattr(response, "data") else response
                review_called = True
                cached["review_fingerprint"] = review_fingerprint
                cached["review"] = dict(review)
                cached["review_response"] = (
                    response.to_artifact()
                    if hasattr(response, "to_artifact")
                    else None
                )
                atomic_write_json(artifacts.semantic_fit, checkpoint)
            audit["semantic_review"] = dict(review)
            audit["semantic_review_ai_called"] = review_called
            if not self._semantic_review_passed(review):
                audit["status"] = "semantic_review_rejection"
                cached.update(audit)
                atomic_write_json(artifacts.semantic_fit, checkpoint)
                continue

            audit["status"] = "selected"
            cached.update(audit)
            case["status"] = "selected"
            case["selected_attempt"] = attempt_number
            atomic_write_json(artifacts.semantic_fit, checkpoint)
            fitted["translation_variant"] = {
                "schema_version": "mathula-selected-translation-variant-v1",
                "source_block_id": _source_block_id(block.block_id),
                "selected_variant_id": variant.variant_id,
                "spoken_text": spoken,
                "measured_duration_ms": measured_ms,
                "omitted_optional_details": list(omissions),
                "meaning_preserved": True,
                "selection_reason": (
                    "semantic_fit_locked_tempo_azure_measured_and_ai_reviewed"
                ),
                "human_review_required": False,
            }
            fitted["semantic_fit"] = {
                "schema_version": SEMANTIC_FIT_SCHEMA_VERSION,
                "selected_attempt": attempt_number,
                "speaker_tempo_locked": True,
                "time_stretch_used": False,
                "semantic_review": dict(review),
                "target_duration_band_ms": {
                    "minimum": target_minimum_ms,
                    "preferred": target_preferred_ms,
                    "maximum": target_maximum_ms,
                },
            }
            fitted["translation_variant_attempts"] = audits
            progress.emit(
                "semantic_fit",
                (
                    f"{block.block_id}: selected attempt {attempt_number} at "
                    f"locked rate {base_rate_percent:+d}%"
                ),
                status="completed",
                current=block_number,
                total=block_total,
                block_id=block.block_id,
                attempt=attempt_number,
                measured_duration_ms=measured_ms,
                rate_percent=base_rate_percent,
            )
            return candidate_block, fitted, audits

        case["status"] = "exhausted"
        case["attempt_limit"] = options.semantic_fit_max_attempts
        atomic_write_json(artifacts.semantic_fit, checkpoint)
        raise DirectDubError(
            f"Semantic-fit exhausted {options.semantic_fit_max_attempts} "
            f"attempt(s) for {block.block_id} without a faithful locked-tempo "
            f"delivery in {target_minimum_ms}-{target_maximum_ms}ms. Inspect "
            f"{artifacts.semantic_fit}."
        )

    def _attempt_finalization_conductor(
        self,
        *,
        block: SpeechBlock,
        block_index: int,
        block_count: int,
        artifacts: DirectDubArtifacts,
        voice: str,
        base_rate_percent: int,
        pitch_percent: int,
        volume_percent: int,
        options: DirectDubOptions,
        latest_overflow: TimingOverflowError,
        duration_evidence: Sequence[Mapping[str, Any]],
        progress: _DirectDubProgress,
        pronunciation_dictionary: PronunciationDictionary | None = None,
    ) -> tuple[SpeechBlock, dict[str, Any], list[dict[str, Any]]] | None:
        if options.finalization_ai_rounds <= 0:
            return None
        try:
            provider = self._timing_repair_provider()
        except Exception as exc:
            progress.emit(
                "finalization-ai",
                (
                    f"{block.block_id}: final conductor provider is unavailable; "
                    "continuing through remaining safe fit policies"
                ),
                status="warning",
                block_id=block.block_id,
                error_type=type(exc).__name__,
            )
            return None
        if provider is None:
            return None

        allowed_ratio = (
            options.boundary_max_time_stretch_ratio
            if block_index in {0, block_count - 1}
            else options.collision_max_time_stretch_ratio
        )
        maximum_duration_ms = max(1, math.floor(block.duration_ms * allowed_ratio))
        preferred_duration_ms = max(1, math.floor(maximum_duration_ms * 0.96))
        observations = [dict(value) for value in duration_evidence]
        current_text = block.translated_text
        measured_duration_ms = latest_overflow.measured_duration_ms
        audits: list[dict[str, Any]] = []
        cache = (
            read_json(artifacts.finalization_conductor)
            if artifacts.finalization_conductor.is_file()
            else {
                "schema_version": FINALIZATION_CONDUCTOR_SCHEMA_VERSION,
                "cases": {},
            }
        )
        if cache.get("schema_version") not in {
            FINALIZATION_CONDUCTOR_SCHEMA_VERSION,
            *LEGACY_FINALIZATION_CONDUCTOR_SCHEMA_VERSIONS,
        }:
            cache = {
                "schema_version": FINALIZATION_CONDUCTOR_SCHEMA_VERSION,
                "cases": {},
            }
        elif cache.get("schema_version") != FINALIZATION_CONDUCTOR_SCHEMA_VERSION:
            # The request/result contract is unchanged. Preserve paid v1
            # candidates and upgrade only the local checkpoint envelope.
            cache = dict(cache)
            cache["schema_version"] = FINALIZATION_CONDUCTOR_SCHEMA_VERSION
        cases = cache.setdefault("cases", {})
        if not isinstance(cases, dict):
            cases = {}
            cache["cases"] = cases

        for round_number in range(1, options.finalization_ai_rounds + 1):
            payload = self._finalization_conductor_payload(
                block,
                maximum_duration_ms=maximum_duration_ms,
                preferred_duration_ms=preferred_duration_ms,
                measured_duration_ms=measured_duration_ms,
                duration_evidence=observations,
                voice=voice,
                base_rate_percent=base_rate_percent,
                round_number=round_number,
                current_spoken_text=current_text,
            )
            fingerprint = hashlib.sha256(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            cached_case = cases.get(fingerprint)
            response_data: Mapping[str, Any]
            ai_called = False
            try:
                if isinstance(cached_case, Mapping) and isinstance(
                    cached_case.get("result"), Mapping
                ):
                    response_data = cached_case["result"]
                else:
                    progress.emit(
                        "finalization-ai",
                        (
                            f"{block.block_id}: final conductor round "
                            f"{round_number}/{options.finalization_ai_rounds}; "
                            "requesting three duration-aware local rewrites"
                        ),
                        status="running",
                        block_id=block.block_id,
                        round=round_number,
                        maximum_duration_ms=maximum_duration_ms,
                        measured_duration_ms=measured_duration_ms,
                    )
                    response = provider.repair_turn(  # type: ignore[attr-defined]
                        payload,
                        output_schema=TURN_TIMING_REPAIR_OPTIONS_SCHEMA,
                    )
                    ai_called = True
                    response_data = (
                        response.data
                        if hasattr(response, "data")
                        else response
                    )
                    cases[fingerprint] = {
                        "block_id": _source_block_id(block.block_id),
                        "round": round_number,
                        "status": "received",
                        "result": dict(response_data),
                        "response": (
                            response.to_artifact()
                            if hasattr(response, "to_artifact")
                            else None
                        ),
                    }
                    atomic_write_json(artifacts.finalization_conductor, cache)
            except Exception as exc:
                cases[fingerprint] = {
                    "block_id": _source_block_id(block.block_id),
                    "round": round_number,
                    "status": "provider_failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                atomic_write_json(artifacts.finalization_conductor, cache)
                progress.emit(
                    "finalization-ai",
                    (
                        f"{block.block_id}: final conductor AI round failed; "
                        "continuing through remaining safe fit policies"
                    ),
                    status="warning",
                    block_id=block.block_id,
                    round=round_number,
                    error_type=type(exc).__name__,
                )
                return None

            raw_candidates = response_data.get("candidates")
            if not isinstance(raw_candidates, list):
                return None
            rank = {"balanced": 0, "safe": 1, "maximum_compression": 2}
            round_measurements: list[dict[str, Any]] = []
            shortest: tuple[int, str] | None = None
            for raw_candidate in sorted(
                (value for value in raw_candidates if isinstance(value, Mapping)),
                key=lambda value: rank.get(str(value.get("candidate_id")), 99),
            ):
                candidate_id = str(raw_candidate.get("candidate_id") or "")
                attempt: dict[str, Any] = {
                    "round": round_number,
                    "candidate_id": candidate_id,
                    "ai_called": ai_called,
                }
                try:
                    spoken, tts, omissions = self._validate_finalization_candidate(
                        block, raw_candidate
                    )
                except ValueError as exc:
                    attempt.update(status="rejected", reason=str(exc))
                    audits.append(attempt)
                    round_measurements.append(attempt)
                    continue
                tts, pronunciation_substitutions = (
                    self._apply_candidate_pronunciation(
                        spoken,
                        tts,
                        pronunciation_dictionary,
                    )
                )
                attempt["pronunciation_substitutions"] = (
                    pronunciation_substitutions
                )
                variant = PreparedTranslationVariant(
                    variant_id=f"ai_{candidate_id}_r{round_number}",
                    spoken_text=spoken,
                    tts_text=tts,
                    target_duration_ms=block.duration_ms,
                    estimated_duration_ms=int(
                        raw_candidate.get("estimated_duration_ms") or 0
                    ) or None,
                    meaning_preserved=True,
                    omitted_optional_details=omissions,
                )
                candidate_block = replace(
                    block,
                    block_id=(
                        f"{_source_block_id(block.block_id)}_variant_"
                        f"{variant.variant_id}"
                    ),
                    translated_text=spoken,
                    tts_text=tts,
                    variants=(*block.variants, variant),
                    selected_variant_id=variant.variant_id,
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
                        max_time_stretch_ratio=allowed_ratio,
                        max_time_expansion_ratio=options.max_time_expansion_ratio,
                        cadence_max_time_stretch_ratio=allowed_ratio,
                        force=False,
                    )
                except TimingOverflowError as exc:
                    attempt.update(
                        status="overflow",
                        spoken_text=spoken,
                        measured_duration_ms=exc.measured_duration_ms,
                        maximum_duration_ms=maximum_duration_ms,
                    )
                    audits.append(attempt)
                    round_measurements.append(attempt)
                    if shortest is None or exc.measured_duration_ms < shortest[0]:
                        shortest = (exc.measured_duration_ms, spoken)
                    continue

                measured = _selected_azure_measurement_ms(fitted)
                if not (fitted.get("cadence_qc") or {}).get("passed", False):
                    attempt.update(
                        status="rejected",
                        reason="azure_measured_candidate_failed_cadence_qc",
                        spoken_text=spoken,
                        measured_duration_ms=measured,
                    )
                    audits.append(attempt)
                    round_measurements.append(attempt)
                    if measured is not None and (
                        shortest is None or measured < shortest[0]
                    ):
                        shortest = (measured, spoken)
                    continue
                attempt.update(
                    status="selected",
                    spoken_text=spoken,
                    measured_duration_ms=measured,
                    maximum_duration_ms=maximum_duration_ms,
                )
                audits.append(attempt)
                fitted["translation_variant"] = {
                    "schema_version": "mathula-selected-translation-variant-v1",
                    "source_block_id": _source_block_id(block.block_id),
                    "selected_variant_id": variant.variant_id,
                    "spoken_text": spoken,
                    "measured_duration_ms": measured,
                    "selection_reason": (
                        "finalization_conductor_ai_rewrite_azure_verified"
                    ),
                    "human_review_required": False,
                }
                fitted["timing_adjustment"] = {
                    "method": "finalization_ai_conductor",
                    "solver_version": FINALIZATION_CONDUCTOR_SCHEMA_VERSION,
                    "round": round_number,
                    "candidate_id": candidate_id,
                    "authoritative_translation_changed": False,
                    "delivery_overlay_changed": True,
                    "protected_entities_preserved": True,
                    "required_facts_supplied_to_validator": list(
                        block.required_facts
                    ),
                    "human_review_required": False,
                }
                fitted["finalization_conductor"] = dict(
                    fitted["timing_adjustment"]
                )
                fitted["timing_repair_attempts"] = audits
                cases[fingerprint]["status"] = "selected"
                cases[fingerprint]["selected_candidate_id"] = candidate_id
                cases[fingerprint]["measurements"] = round_measurements + [attempt]
                atomic_write_json(artifacts.finalization_conductor, cache)
                progress.emit(
                    "finalization-ai",
                    (
                        f"{block.block_id}: final conductor selected "
                        f"{candidate_id} after Azure measurement"
                    ),
                    status="completed",
                    block_id=block.block_id,
                    round=round_number,
                    candidate_id=candidate_id,
                    measured_duration_ms=measured,
                    maximum_duration_ms=maximum_duration_ms,
                    checkpoint_reused=not ai_called,
                )
                return candidate_block, fitted, audits

            cases[fingerprint]["status"] = "measured_overflow"
            cases[fingerprint]["measurements"] = round_measurements
            atomic_write_json(artifacts.finalization_conductor, cache)
            observations.extend(round_measurements)
            if shortest is None:
                return None
            measured_duration_ms, current_text = shortest
        return None

    def _force_final_product_fit(
        self,
        *,
        block: SpeechBlock,
        artifacts: DirectDubArtifacts,
        voice: str,
        base_rate_percent: int,
        pitch_percent: int,
        volume_percent: int,
        options: DirectDubOptions,
        overflow: TimingOverflowError,
        five_block_window: Sequence[Mapping[str, Any]],
        repair_audit: Sequence[Mapping[str, Any]],
        progress: _DirectDubProgress,
    ) -> dict[str, Any]:
        """Produce a bounded, measurable final block after quality paths fail.

        This is deliberately the last policy. It never changes words, speaker,
        order, or timeline boundaries. The configured whole-voice compression
        ceiling is a hard production-quality limit: the renderer must never
        silently exceed it merely to force an unattended job to complete.
        """

        required_ratio = max(
            1.0,
            overflow.measured_duration_ms / max(1, block.duration_ms),
        )
        applied_limit = options.final_product_max_time_stretch_ratio
        if required_ratio > applied_limit:
            raise TimingOverflowError(
                (
                    f"{block.block_id} requires {required_ratio:.3f}x whole-voice "
                    f"compression, above the production cadence ceiling "
                    f"{applied_limit:.3f}x. Natural cadence is protected; a "
                    "shorter semantically equivalent delivery or more timeline "
                    "room is required."
                ),
                measured_duration_ms=overflow.measured_duration_ms,
                target_duration_ms=block.duration_ms,
                stretch_ratio=required_ratio,
            )
        fitted = self._synthesize_block(
            block,
            artifacts,
            voice=voice,
            base_rate_percent=base_rate_percent,
            pitch_percent=pitch_percent,
            volume_percent=volume_percent,
            max_azure_rate_percent=base_rate_percent,
            max_time_stretch_ratio=applied_limit,
            max_time_expansion_ratio=options.max_time_expansion_ratio,
            cadence_max_time_stretch_ratio=applied_limit,
            force=False,
        )
        adjustment = {
            "method": "final_product_emergency_fit",
            "solver_version": FINAL_PRODUCT_CONDUCTOR_VERSION,
            "measured_duration_ms": overflow.measured_duration_ms,
            "available_duration_ms": block.duration_ms,
            "required_time_stretch_ratio": round(required_ratio, 6),
            "applied_time_stretch_ratio": fitted.get(
                "post_synthesis_time_stretch_ratio"
            ),
            "configured_quality_ceiling": (
                options.final_product_max_time_stretch_ratio
            ),
            "quality_ceiling_exceeded": (
                required_ratio > options.final_product_max_time_stretch_ratio
            ),
            "five_block_window": [dict(value) for value in five_block_window],
            "authoritative_translation_changed": False,
            "speaker_order_preserved": True,
            "timeline_boundaries_preserved": True,
            "human_review_required": False,
            "quality_warning": (
                "Cadence was compressed beyond the preferred quality policy "
                "to guarantee an unattended final product."
            ),
        }
        fitted["timing_adjustment"] = adjustment
        fitted["finalization_conductor"] = adjustment
        fitted["timing_repair_attempts"] = [
            dict(value) for value in repair_audit
        ]
        progress.emit(
            "final_conductor",
            (
                f"{block.block_id}: all semantic and five-block fit paths were "
                "exhausted; final product completed with measured emergency cadence"
            ),
            status="completed",
            block_id=block.block_id,
            measured_duration_ms=overflow.measured_duration_ms,
            available_duration_ms=block.duration_ms,
            required_time_stretch_ratio=round(required_ratio, 6),
            final_product_guaranteed=True,
            manual_review_required=False,
        )
        return fitted

    @staticmethod
    def _source_anchored_breath_group_window(
        block: SpeechBlock,
        *,
        source_island: Mapping[str, Any],
        island_index: int,
        tolerance_ms: int,
    ) -> dict[str, int]:
        """Return the locked-tempo timing window for one source breath group.

        The source island endpoint is useful mouth-activity evidence, but it is
        not a hard linguistic duration limit for translation.  The next source
        speech onset is the stronger cadence anchor.  We therefore permit a
        phrase tail to consume otherwise-silent room, while keeping a short
        audible rest and never crossing the next onset.  No Azure rate change
        or waveform stretch is introduced.
        """

        start_ms = int(source_island.get("start_ms") or 0)
        end_ms = int(source_island.get("end_ms") or start_ms)
        target_ms = max(1, end_ms - start_ms)
        island_tolerance_ms = min(
            500,
            max(int(tolerance_ms), min(250, target_ms // 8)),
        )
        minimum_ms = max(1, target_ms - island_tolerance_ms)
        nominal_maximum_ms = target_ms + island_tolerance_ms

        position = max(0, int(island_index) - 1)
        next_onset_ms: int | None = None
        if position + 1 < len(block.source_speech_islands):
            next_onset_ms = int(
                block.source_speech_islands[position + 1].get("start_ms")
                or end_ms
            )
            hard_boundary_ms = max(end_ms, next_onset_ms)
            source_gap_ms = max(0, hard_boundary_ms - end_ms)
            preserved_rest_ms = min(source_gap_ms, SOURCE_ANCHORED_MIN_REST_MS)
        else:
            hard_boundary_ms = max(end_ms, int(block.end_ms))
            source_gap_ms = max(0, hard_boundary_ms - end_ms)
            preserved_rest_ms = min(
                source_gap_ms, SOURCE_ANCHORED_FINAL_MIN_REST_MS
            )

        anchor_deadline_ms = max(
            end_ms, hard_boundary_ms - preserved_rest_ms
        )
        pause_borrow_capacity_ms = max(0, anchor_deadline_ms - end_ms)
        pause_borrow_budget_ms = min(
            SOURCE_ANCHORED_MAX_PAUSE_BORROW_MS,
            pause_borrow_capacity_ms,
        )
        anchor_maximum_ms = target_ms + pause_borrow_budget_ms
        maximum_ms = max(
            target_ms,
            nominal_maximum_ms,
            anchor_maximum_ms,
        )
        # The computed deadline is always the final authority.  This guard is
        # mostly defensive for malformed source timing with a tiny/negative gap.
        maximum_ms = min(
            maximum_ms,
            max(target_ms, anchor_deadline_ms - start_ms),
        )

        return {
            "target_duration_ms": target_ms,
            "minimum_duration_ms": minimum_ms,
            "nominal_maximum_duration_ms": nominal_maximum_ms,
            "maximum_duration_ms": maximum_ms,
            "next_source_onset_ms": int(next_onset_ms or 0),
            "source_pause_after_ms": source_gap_ms,
            "preserved_source_pause_ms": preserved_rest_ms,
            "pause_borrow_budget_ms": max(0, maximum_ms - target_ms),
            "anchor_deadline_ms": start_ms + maximum_ms,
        }

    def _measure_single_source_breath_group_candidate(
        self,
        block: SpeechBlock,
        artifacts: DirectDubArtifacts,
        *,
        source_island: Mapping[str, Any],
        island_index: int,
        text: str,
        voice: str,
        base_rate_percent: int,
        pitch_percent: int,
        volume_percent: int,
        tolerance_ms: int,
    ) -> dict[str, Any]:
        """Measure one candidate phrase at the immutable source-speaker tempo.

        ``turn_id`` intentionally matches the canonical source-island request
        used by ``_measure_performance_islands``.  The artifact key also
        includes text/prosody, so alternate wordings remain distinct while the
        selected wording is idempotently reused during final rendering.
        """

        start_ms = int(source_island.get("start_ms") or 0)
        end_ms = int(source_island.get("end_ms") or start_ms)
        window = self._source_anchored_breath_group_window(
            block,
            source_island=source_island,
            island_index=island_index,
            tolerance_ms=tolerance_ms,
        )
        target_ms = int(window["target_duration_ms"])
        minimum_ms = int(window["minimum_duration_ms"])
        maximum_ms = int(window["maximum_duration_ms"])
        request = AzureTTSRequest(
            turn_id=f"{block.block_id}_island_{island_index:03d}",
            speaker_id=block.speaker_id,
            text=str(text),
            preferred_duration_ms=target_ms,
            maximum_duration_ms=maximum_ms,
            voice=voice,
            language="zu-ZA",
            rate_percent=base_rate_percent,
            pitch_percent=pitch_percent,
            volume_percent=volume_percent,
            parts=build_initialism_ssml_parts(
                str(text),
                language="zu-ZA",
            ),
        )
        result = self.backend.synthesize(
            request,
            artifacts.candidate(request),
            force=False,
        )
        fits = minimum_ms <= result.duration_ms <= maximum_ms
        return {
            "island_id": str(source_island.get("island_id") or ""),
            "source_start_ms": start_ms,
            "source_end_ms": end_ms,
            "target_duration_ms": target_ms,
            "minimum_duration_ms": minimum_ms,
            "maximum_duration_ms": maximum_ms,
            "nominal_maximum_duration_ms": int(
                window["nominal_maximum_duration_ms"]
            ),
            "next_source_onset_ms": int(window["next_source_onset_ms"]),
            "source_pause_after_ms": int(window["source_pause_after_ms"]),
            "preserved_source_pause_ms": int(
                window["preserved_source_pause_ms"]
            ),
            "pause_borrow_budget_ms": int(window["pause_borrow_budget_ms"]),
            "pause_borrow_used_ms": max(0, result.duration_ms - target_ms),
            "anchor_deadline_ms": int(window["anchor_deadline_ms"]),
            "measured_duration_ms": result.duration_ms,
            "delta_ms": result.duration_ms - target_ms,
            "status": (
                "fit_with_pause_borrow"
                if fits and result.duration_ms > int(
                    window["nominal_maximum_duration_ms"]
                )
                else "fit" if fits else "timing_miss"
            ),
            "raw_isolated_duration_fit": fits,
            "speaker_rate_percent": base_rate_percent,
            "output_path": result.output_path,
            "output_sha256": result.sha256,
            "idempotent_reuse": bool(result.idempotent_reuse),
        }

    def _measure_performance_islands(
        self,
        block: SpeechBlock,
        artifacts: DirectDubArtifacts,
        *,
        island_texts: Sequence[str],
        voice: str,
        base_rate_percent: int,
        pitch_percent: int,
        volume_percent: int,
        tolerance_ms: int,
    ) -> tuple[bool, list[dict[str, Any]]]:
        """Measure and hard-gate cumulative drift between source islands.

        Isolated synthesis has phrase-boundary overhead, so raw island duration
        bands remain diagnostic.  After normalizing that scale error, however,
        cumulative boundary drift is a reliable signal and must veto a region
        that would visibly move later speech behind the mouth.
        """

        if len(island_texts) != len(block.source_speech_islands):
            raise ValueError(
                "performance-plan target/source island counts differ"
            )
        measurements: list[dict[str, Any]] = []
        for index, (source_island, text) in enumerate(
            zip(block.source_speech_islands, island_texts, strict=True),
            start=1,
        ):
            start_ms = int(source_island.get("start_ms") or 0)
            end_ms = int(source_island.get("end_ms") or start_ms)
            window = self._source_anchored_breath_group_window(
                block,
                source_island=source_island,
                island_index=index,
                tolerance_ms=tolerance_ms,
            )
            target_ms = int(window["target_duration_ms"])
            minimum_ms = int(window["minimum_duration_ms"])
            maximum_ms = int(window["maximum_duration_ms"])
            request = AzureTTSRequest(
                turn_id=f"{block.block_id}_island_{index:03d}",
                speaker_id=block.speaker_id,
                text=str(text),
                preferred_duration_ms=target_ms,
                maximum_duration_ms=maximum_ms,
                voice=voice,
                language="zu-ZA",
                rate_percent=base_rate_percent,
                pitch_percent=pitch_percent,
                volume_percent=volume_percent,
                parts=build_initialism_ssml_parts(
                    str(text),
                    language="zu-ZA",
                ),
            )
            result = self.backend.synthesize(
                request,
                artifacts.candidate(request),
                force=False,
            )
            fits = minimum_ms <= result.duration_ms <= maximum_ms
            measurements.append(
                {
                    "island_id": str(source_island.get("island_id") or ""),
                    "source_start_ms": start_ms,
                    "source_end_ms": end_ms,
                    "target_duration_ms": target_ms,
                    "minimum_duration_ms": minimum_ms,
                    "maximum_duration_ms": maximum_ms,
                    "nominal_maximum_duration_ms": int(
                        window["nominal_maximum_duration_ms"]
                    ),
                    "next_source_onset_ms": int(window["next_source_onset_ms"]),
                    "source_pause_after_ms": int(window["source_pause_after_ms"]),
                    "preserved_source_pause_ms": int(
                        window["preserved_source_pause_ms"]
                    ),
                    "pause_borrow_budget_ms": int(
                        window["pause_borrow_budget_ms"]
                    ),
                    "pause_borrow_used_ms": max(
                        0, result.duration_ms - target_ms
                    ),
                    "anchor_deadline_ms": int(window["anchor_deadline_ms"]),
                    "measured_duration_ms": result.duration_ms,
                    "delta_ms": result.duration_ms - target_ms,
                    "status": (
                        "fit_with_pause_borrow"
                        if fits and result.duration_ms > int(
                            window["nominal_maximum_duration_ms"]
                        )
                        else "fit" if fits else "timing_miss"
                    ),
                    "speaker_rate_percent": base_rate_percent,
                    # Persist the deterministic Azure artifact.  Production
                    # breath-group rescue can therefore reuse paid/cacheable
                    # island synthesis instead of asking GPT or Azure to repeat
                    # work after a whole-region alignment miss.
                    "output_path": result.output_path,
                    "output_sha256": result.sha256,
                    "idempotent_reuse": bool(result.idempotent_reuse),
                }
            )
        summary = self._performance_island_alignment_summary(measurements)
        drift_limit_ms = min(
            MAXIMUM_INTERNAL_ISLAND_DRIFT_MS,
            max(1, int(tolerance_ms)),
        )
        cumulative_drift_fit = (
            int(summary["maximum_absolute_cumulative_drift_ms"])
            <= drift_limit_ms
        )
        for measurement, normalized in zip(
            measurements,
            summary["islands"],
            strict=True,
        ):
            measurement.update(normalized)
            measurement["raw_isolated_duration_fit"] = measurement["status"] in {"fit", "fit_with_pause_borrow"}
            measurement["cumulative_drift_limit_ms"] = drift_limit_ms
        # Raw isolated fits are intentionally not binding because phrase-edge
        # prosody can move every isolated duration in the same direction. The
        # normalized internal boundaries are binding; the continuous waveform
        # remains the separate whole-region gate.
        return cumulative_drift_fit, measurements

    def _synthesize_source_anchored_breath_groups(
        self,
        block: SpeechBlock,
        artifacts: DirectDubArtifacts,
        *,
        island_texts: Sequence[str],
        voice: str,
        base_rate_percent: int,
        pitch_percent: int,
        volume_percent: int,
        tolerance_ms: int,
    ) -> dict[str, Any]:
        """Render each word-timed breath group at its source onset.

        A 600+ ms source pause is a real performance boundary, not spare
        timeline capacity.  Once target wording has been assigned to those
        breath groups, independent locked-tempo synthesis removes cumulative
        drift by construction: every target phrase starts where the source
        speaker starts that phrase.  No Azure rate change, whole-voice stretch,
        silence borrowing, or cross-boundary speech is permitted.
        """

        if (
            block.source_timing_evidence != "azure_word_timestamps"
            or not block.source_speech_islands
            or block.source_interactions
        ):
            raise ValueError(
                "source-anchored breath-group synthesis requires non-overlapping "
                "word-timed speech-island evidence"
            )
        if len(island_texts) != len(block.source_speech_islands):
            raise ValueError(
                "source-anchored breath-group target/source island counts differ"
            )

        _alignment_fit, measurements = self._measure_performance_islands(
            block,
            artifacts,
            island_texts=island_texts,
            voice=voice,
            base_rate_percent=base_rate_percent,
            pitch_percent=pitch_percent,
            volume_percent=volume_percent,
            tolerance_ms=tolerance_ms,
        )
        misses = [
            value
            for value in measurements
            if not bool(value.get("raw_isolated_duration_fit"))
        ]
        if misses:
            worst = max(
                misses,
                key=lambda value: abs(int(value.get("delta_ms") or 0)),
            )
            raise TimingOverflowError(
                f"{block.block_id} has a breath group outside its source mouth-active window",
                measured_duration_ms=int(worst.get("measured_duration_ms") or 0),
                target_duration_ms=int(worst.get("target_duration_ms") or 0),
                stretch_ratio=(
                    int(worst.get("measured_duration_ms") or 0)
                    / max(1, int(worst.get("target_duration_ms") or 0))
                ),
            )

        final_path = artifacts.final_block(block.block_id)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        sample_rate = int(self.backend.sample_rate)
        frame_count = max(1, round(block.duration_ms * sample_rate / 1000))
        output_frames = bytearray(frame_count * 2)
        rendered_islands: list[dict[str, Any]] = []
        rendered_speech_end_ms = block.start_ms

        for source_island, text, measurement in zip(
            block.source_speech_islands,
            island_texts,
            measurements,
            strict=True,
        ):
            source_start_ms = int(source_island.get("start_ms") or block.start_ms)
            source_end_ms = int(source_island.get("end_ms") or source_start_ms)
            relative_start_ms = max(0, source_start_ms - block.start_ms)
            start_frame = round(relative_start_ms * sample_rate / 1000)
            source_path = Path(str(measurement.get("output_path") or ""))
            if not source_path.exists():
                raise DirectDubError(
                    f"Breath-group Azure artifact is missing for {block.block_id}: "
                    f"{source_path}"
                )
            with wave.open(str(source_path), "rb") as handle:
                if (
                    handle.getnchannels() != 1
                    or handle.getsampwidth() != 2
                    or handle.getframerate() != sample_rate
                ):
                    raise DirectDubError(
                        f"Breath-group Azure artifact has unexpected PCM format: {source_path}"
                    )
                raw = handle.readframes(handle.getnframes())

            available_bytes = max(0, len(output_frames) - start_frame * 2)
            if len(raw) > available_bytes:
                raise TimingOverflowError(
                    f"{block.block_id} breath-group audio exceeds the immutable region endpoint",
                    measured_duration_ms=int(measurement.get("measured_duration_ms") or 0),
                    target_duration_ms=max(1, block.end_ms - source_start_ms),
                    stretch_ratio=(
                        int(measurement.get("measured_duration_ms") or 0)
                        / max(1, block.end_ms - source_start_ms)
                    ),
                )
            output_frames[start_frame * 2 : start_frame * 2 + len(raw)] = raw
            rendered_end_ms = source_start_ms + int(
                measurement.get("measured_duration_ms") or 0
            )
            rendered_speech_end_ms = max(rendered_speech_end_ms, rendered_end_ms)
            rendered_islands.append(
                {
                    **dict(measurement),
                    "spoken_text": str(text),
                    "render_start_ms": source_start_ms,
                    "render_end_ms": rendered_end_ms,
                    "source_endpoint_error_ms": rendered_end_ms - source_end_ms,
                    "placement": "source_breath_group_onset",
                }
            )

        temporary = final_path.with_suffix(final_path.suffix + ".tmp")
        with wave.open(str(temporary), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(output_frames)
        os.replace(temporary, final_path)

        policy = ProfessionalDubPolicy()
        cadence_qc = assess_perceptual_cadence(
            selected_rate_percent=base_rate_percent,
            base_rate_percent=base_rate_percent,
            whole_voice_stretch_ratio=1.0,
            timeline_fill_action="source_anchored_breath_groups",
            pause_fill={
                "applied": True,
                "reason": "source_breath_group_onsets_are_authoritative",
                "approved_text_changed": False,
            },
            trailing_room_ms=max(
                0,
                int(block.source_speech_end_ms or block.end_ms)
                - rendered_speech_end_ms,
            ),
            maximum_added_pause_ms=policy.maximum_added_pause_ms,
            maximum_rate_delta_percent=0,
            maximum_whole_voice_change=1.0,
        )
        endpoint_error_ms = (
            rendered_speech_end_ms - int(block.source_speech_end_ms)
            if block.source_speech_end_ms is not None
            else None
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
                    "rate_percent": base_rate_percent,
                    "duration_ms": int(value.get("measured_duration_ms") or 0),
                    "output_path": value.get("output_path"),
                    "idempotent_reuse": value.get("idempotent_reuse"),
                    "breath_group_id": value.get("island_id"),
                }
                for value in rendered_islands
            ],
            "selected_azure_rate_percent": base_rate_percent,
            "post_synthesis_time_stretch_ratio": 1.0,
            "timeline_fill_action": "source_anchored_breath_groups",
            "speech_rate_preserved": True,
            "speaker_tempo_locked": True,
            "source_pause_alignment": {
                "applied": True,
                "reason": "source_breath_group_onsets_are_authoritative",
                "breath_group_count": len(rendered_islands),
                "islands": rendered_islands,
                "approved_text_changed": False,
            },
            "performance_island_alignment": {
                "authority": "direct_source_onset_placement",
                "cumulative_drift_eliminated_by_construction": True,
                "source_tempo_locked": True,
                "whole_voice_time_stretch_used": False,
                "islands": rendered_islands,
            },
            "cadence_qc": cadence_qc,
            "output_path": str(final_path),
            # The block-level WAV intentionally includes the original silence
            # between breath groups, so its exact duration is the immutable
            # source region duration.
            "duration_ms": block.duration_ms,
            "semantic_timing_deadline_ms": block.duration_ms,
            "exact_audio_boundary": {
                "schema_version": "mathula-exact-audio-boundary-v1",
                "corrected": False,
                "measured_before_ms": block.duration_ms,
                "duration_ms": block.duration_ms,
                "target_duration_ms": block.duration_ms,
                "semantic_timing_deadline_ms": block.duration_ms,
                "correction_ratio": 1.0,
                "combined_stretch_ratio": 1.0,
            },
            "source_lipsync_endpoint": {
                "timing_evidence": block.source_timing_evidence,
                "source_speech_end_ms": block.source_speech_end_ms,
                "rendered_speech_end_ms": rendered_speech_end_ms,
                "endpoint_error_ms": endpoint_error_ms,
                "within_deadline": rendered_speech_end_ms <= block.end_ms,
            },
            "breath_group_delivery": {
                "schema_version": "mathula-source-anchored-breath-groups-v2-semantic-safe-rescue-v13.19.4",
                "source_pause_threshold_ms": ISLAND_GAP_MIN_MS,
                "source_tempo_locked": True,
                "source_onsets_locked": True,
                "island_count": len(rendered_islands),
                "islands": rendered_islands,
            },
            "output_sha256": checksum(final_path),
        }

    @staticmethod
    def _performance_island_alignment_summary(
        measurements: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Remove isolated-TTS scale error and expose internal boundary drift."""

        source_total_ms = sum(
            max(1, int(value.get("target_duration_ms") or 0))
            for value in measurements
        )
        isolated_total_ms = sum(
            max(1, int(value.get("measured_duration_ms") or 0))
            for value in measurements
        )
        scale = source_total_ms / max(1, isolated_total_ms)
        source_elapsed_ms = 0
        normalized_elapsed_ms = 0
        islands: list[dict[str, Any]] = []
        absolute_drifts: list[int] = []
        for index, value in enumerate(measurements):
            source_ms = max(1, int(value.get("target_duration_ms") or 0))
            measured_ms = max(1, int(value.get("measured_duration_ms") or 0))
            normalized_ms = max(1, round(measured_ms * scale))
            source_elapsed_ms += source_ms
            normalized_elapsed_ms += normalized_ms
            cumulative_drift_ms = (
                0
                if index == len(measurements) - 1
                else normalized_elapsed_ms - source_elapsed_ms
            )
            absolute_drifts.append(abs(cumulative_drift_ms))
            islands.append(
                {
                    "normalized_measured_duration_ms": normalized_ms,
                    "normalized_delta_ms": normalized_ms - source_ms,
                    "normalized_cumulative_drift_ms": cumulative_drift_ms,
                }
            )
        configured_limits = [
            int(value.get("cumulative_drift_limit_ms") or 0)
            for value in measurements
            if int(value.get("cumulative_drift_limit_ms") or 0) > 0
        ]
        maximum_allowed_cumulative_drift_ms = (
            min(configured_limits) if configured_limits else None
        )
        worst_index = max(
            range(len(islands)),
            key=lambda index: abs(
                int(islands[index]["normalized_cumulative_drift_ms"])
            ),
            default=0,
        )
        worst_drift_ms = (
            int(islands[worst_index]["normalized_cumulative_drift_ms"])
            if islands
            else 0
        )
        worst_source = measurements[worst_index] if measurements else {}
        return {
            "authority": "normalized_internal_alignment_gate",
            "whole_region_timing_remains_hard_gate": True,
            "internal_cumulative_drift_is_hard_gate": True,
            "isolated_phrase_boundary_overhead_removed": True,
            "normalization_scale": round(scale, 6),
            "source_total_ms": source_total_ms,
            "isolated_tts_total_ms": isolated_total_ms,
            "maximum_allowed_cumulative_drift_ms": (
                maximum_allowed_cumulative_drift_ms
            ),
            "maximum_absolute_cumulative_drift_ms": max(
                absolute_drifts,
                default=0,
            ),
            "mean_absolute_cumulative_drift_ms": round(
                sum(absolute_drifts) / max(1, len(absolute_drifts))
            ),
            "worst_boundary": {
                "after_island_index": worst_index + 1 if islands else None,
                "after_island_id": (
                    str(worst_source.get("island_id") or "") or None
                ),
                "cumulative_drift_ms": worst_drift_ms,
                "correction": (
                    "move_or_compress_wording_toward_later_islands"
                    if worst_drift_ms > 0
                    else (
                        "move_or_expand_wording_toward_earlier_islands"
                        if worst_drift_ms < 0
                        else "maintain_boundary_distribution"
                    )
                ),
            },
            "islands": islands,
        }

    def _pause_for_timeline_collision_review(
        self,
        *,
        job: JobManifest,
        artifacts: DirectDubArtifacts,
        case: Mapping[str, Any],
        progress: _DirectDubProgress,
    ) -> None:
        review = upsert_timeline_collision_case(
            review_path=artifacts.timeline_collision_review,
            panel_path=artifacts.timeline_collision_panel,
            resolutions_path=artifacts.timeline_collision_resolutions,
            job_id=job.job_id,
            case=case,
        )
        job.media["timeline_collision_review"] = str(
            artifacts.timeline_collision_review
        )
        job.media["timeline_collision_panel"] = str(
            artifacts.timeline_collision_panel
        )
        job.media["timeline_collision_resolutions"] = str(
            artifacts.timeline_collision_resolutions
        )
        job.review_readiness = "timeline_collision_review_required"
        self.jobs.save(job)
        progress.emit(
            "timeline_review",
            (
                "Timeline collision review package is ready; save a panel "
                "decision; the current dub-azure process will resume automatically"
            ),
            status="review_required",
            review_path=str(artifacts.timeline_collision_review),
            panel_path=str(artifacts.timeline_collision_panel),
            collision_id=case.get("collision_id"),
        )
        raise TimelineCollisionReviewRequired(review)

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
        _, translation_source_items = _translation_items(translation)
        skipped_non_speech_unit_ids = [
            str(
                item.get("unit_id")
                or item.get("segment_id")
                or f"item_{index:05d}"
            )
            for index, item in enumerate(translation_source_items, start=1)
            if isinstance(item, Mapping)
            and _punctuation_only_source_unit(item)
        ]
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
        name_research_path = (
            self.jobs.job_dir(job.job_id)
            / "analysis"
            / "autocorrection_name_research.json"
        )
        pronunciation_dictionary = with_web_researched_organisation_pronunciations(
            pronunciation_dictionary,
            read_json(name_research_path) if name_research_path.is_file() else None,
        )
        atomic_write_json(dictionary_path, pronunciation_dictionary.to_dict())
        boundary_normalizations: list[dict[str, Any]] = []
        segments = load_approved_segments(
            translation,
            source_duration_ms=source_duration_ms,
            pronunciation_dictionary=pronunciation_dictionary,
            boundary_normalizations=boundary_normalizations,
        )
        for normalization in boundary_normalizations:
            progress.emit(
                "prepare",
                (
                    f"Unit {normalization['segment_id']} exceeds the media "
                    f"boundary by {normalization['overrun_ms']}ms; clamped to "
                    f"{normalization['normalized_end_ms']}ms without changing "
                    "approved text"
                ),
                status="warning",
                **normalization,
            )
        blocks = coalesce_same_speaker_segments(
            segments,
            max_gap_ms=options.same_speaker_gap_ms,
            preserve_opening_segment=True,
        )
        if not blocks:
            raise DirectDubError("Approved translation contains no speakable segments")
        blocks, final_source_tail_adjustment = extend_final_speech_block_into_source_tail(
            blocks,
            source_duration_ms=source_duration_ms,
        )
        if final_source_tail_adjustment["applied"]:
            progress.emit(
                "prepare",
                (
                    "Reserved "
                    f"{final_source_tail_adjustment['borrowed_tail_ms']}ms of "
                    "unused source tail for collision-free final speech"
                ),
                status="completed",
                **final_source_tail_adjustment,
            )

        mastering_override_payload = load_dub_mastering_overrides(
            artifacts.mastering_overrides
        )
        blocks, mastering_override_audits = apply_dub_mastering_overrides(
            blocks,
            mastering_override_payload,
        )

        collision_resolution_payload = load_timeline_collision_resolutions(
            artifacts.timeline_collision_resolutions
        )
        collision_resolutions = timeline_collision_resolution_map(
            artifacts.timeline_collision_resolutions
        )
        blocks, timeline_collision_resolution_audits = (
            apply_timeline_collision_resolutions(blocks, collision_resolutions)
        )

        source_transcript_path = artifacts.job_root / "analysis" / "transcript_en.json"
        source_transcript = (
            read_json(source_transcript_path)
            if source_transcript_path.is_file()
            else {"segments": [], "words": []}
        )
        blocks, source_pause_alignment = attach_source_pause_anchors(
            blocks,
            source_transcript,
        )
        blocks, transcript_lipsync_plan = attach_transcript_lipsync_plan(
            blocks,
            source_transcript,
        )
        transcript_lipsync_plan = {
            **transcript_lipsync_plan,
            "job_id": job.job_id,
            "created_at": utcnow(),
            "source_transcript_path": (
                str(source_transcript_path)
                if source_transcript_path.is_file()
                else None
            ),
            "source_transcript_sha256": (
                checksum(source_transcript_path)
                if source_transcript_path.is_file()
                else None
            ),
        }
        atomic_write_json(
            artifacts.transcript_lipsync,
            transcript_lipsync_plan,
        )
        semantic_phrase_group_audits: list[dict[str, Any]] = []
        performance_region_audits: list[dict[str, Any]] = []
        if options.timing_mode == TIMING_MODE_SEMANTIC_FIT:
            blocks, semantic_phrase_group_audits = build_semantic_phrase_groups(
                blocks
            )
            if semantic_phrase_group_audits:
                progress.emit(
                    "prepare",
                    (
                        f"Built {len(semantic_phrase_group_audits)} same-speaker "
                        "semantic phrase group(s) across sentence boundaries"
                    ),
                    status="completed",
                    semantic_phrase_group_count=len(
                        semantic_phrase_group_audits
                    ),
                    semantic_phrase_groups=semantic_phrase_group_audits,
                    speaker_tempo_locked=True,
                    source_islands_preserved=True,
                )
            blocks, performance_region_audits = (
                attach_word_anchored_island_contracts(blocks)
            )
            if performance_region_audits:
                progress.emit(
                    "prepare",
                    (
                        f"Locked {len(performance_region_audits)} multi-island "
                        "semantic-fit region(s) to source-word onsets"
                    ),
                    status="completed",
                    performance_region_count=len(performance_region_audits),
                    speaker_tempo_locked=True,
                    source_islands_preserved=True,
                    internal_island_drift_is_hard_gate=True,
                )
        elif options.timing_mode == TIMING_MODE_PERFORMANCE_PLAN:
            blocks, performance_region_audits = build_performance_regions(
                blocks
            )
        if performance_region_audits:
            performance_plan_artifact = {
                "schema_version": PERFORMANCE_PLAN_VERSION,
                "job_id": job.job_id,
                "created_at": utcnow(),
                "source_timing": "azure_word_timestamps",
                "region_count": sum(
                    1 for value in performance_region_audits if value.get("region_id")
                ),
                "compatibility_block_count": sum(
                    1 for value in performance_region_audits if not value.get("region_id")
                ),
                "regions": performance_region_audits,
                "policy": {
                    "stt_blocks_are_linguistic_boundaries": False,
                    "semantic_beats_are_translation_structure": True,
                    "word_islands_are_timing_evidence": True,
                    "every_target_island_is_acoustically_measured": True,
                    "region_and_island_timing_must_both_fit": True,
                    "internal_island_drift_is_hard_gate": True,
                    "maximum_internal_island_drift_ms": (
                        min(
                            MAXIMUM_INTERNAL_ISLAND_DRIFT_MS,
                            max(
                                1,
                                int(options.semantic_fit_tolerance_ms),
                            ),
                        )
                    ),
                    "source_word_onset_is_hard_anchor": True,
                    "speaker_lanes_are_immutable": True,
                    "overlapping_speakers_remain_parallel": True,
                    "speaker_tempo_locked": True,
                    "visual_mouth_analysis_required": False,
                },
            }
            atomic_write_json(
                artifacts.performance_plan,
                performance_plan_artifact,
            )
            progress.emit(
                "prepare",
                (
                    "Built "
                    f"{performance_plan_artifact['region_count']} word-timed "
                    "performance region(s); STT blocks retained only as audit members"
                ),
                status="completed",
                performance_region_count=performance_plan_artifact[
                    "region_count"
                ],
                compatibility_block_count=performance_plan_artifact[
                    "compatibility_block_count"
                ],
                performance_plan_path=str(artifacts.performance_plan),
                speaker_tempo_locked=True,
                source_islands_preserved=True,
            )
        source_word_anchors = build_source_word_speaker_anchors(
            blocks,
            source_transcript,
            tolerance_ms=options.source_word_alignment_tolerance_ms,
        )
        source_word_anchored_boundary_indices = {
            int(anchor["boundary_index"]) for anchor in source_word_anchors
        }
        source_word_anchor_by_boundary = {
            int(anchor["boundary_index"]): anchor for anchor in source_word_anchors
        }
        source_word_alignment_corrections: list[dict[str, Any]] = []

        required_speaker_ids = sorted({block.speaker_id for block in blocks})
        progress.emit(
            "prepare",
            (
                f"Prepared {len(blocks)} speech blocks for "
                f"{len(required_speaker_ids)} speakers"
                + (
                    f"; skipped {len(skipped_non_speech_unit_ids)} "
                    "punctuation-only non-speech unit(s)"
                    if skipped_non_speech_unit_ids
                    else ""
                )
            ),
            status="completed",
            current=len(blocks),
            total=len(blocks),
            block_count=len(blocks),
            speaker_count=len(required_speaker_ids),
            skipped_non_speech_unit_count=len(skipped_non_speech_unit_ids),
            skipped_non_speech_unit_ids=skipped_non_speech_unit_ids,
            transcript_lipsync_word_timed_blocks=int(
                transcript_lipsync_plan.get("word_timed_block_count") or 0
            ),
            transcript_lipsync_interactions=int(
                transcript_lipsync_plan.get("interaction_count") or 0
            ),
            semantic_phrase_group_count=len(semantic_phrase_group_audits),
            performance_region_count=sum(
                1 for value in performance_region_audits if value.get("region_id")
            ),
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
            contextual_names = sorted(
                {
                    str(value.get("identified_name"))
                    for value in assignments.values()
                    if value.get("resolution_status")
                    == "contextual_speaker_identity"
                    and value.get("identified_name")
                },
                key=str.casefold,
            )
            progress.emit(
                "voice_resolution",
                (
                    "Reusing cached speaker identities and Azure voice assignments"
                    + (
                        "; contextual identity: " + ", ".join(contextual_names)
                        if contextual_names
                        else ""
                    )
                ),
                status="completed",
                current=len(assignments),
                total=len(required_speaker_ids),
                cache_reused=True,
                contextual_identity_count=len(contextual_names),
                contextual_identity_names=contextual_names,
            )
            request_fingerprint = self._request_fingerprint(
                job,
                translation_path,
                blocks,
                assignments,
                options,
                pronunciation_dictionary,
                clean_background_stem,
                seo_context.seo_path,
                collision_resolution_payload,
                mastering_override_payload,
            )
            if not force:
                reused = self._reuse_completed(artifacts, request_fingerprint)
                if reused is not None:
                    progress.emit(
                        "complete",
                        "Unchanged job reused without SpeechBrain, Azure TTS, GPT, mixing, or rendering",
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
            contextual_names = sorted(
                {
                    str(value.get("identified_name"))
                    for value in assignments.values()
                    if value.get("resolution_status")
                    == "contextual_speaker_identity"
                    and value.get("identified_name")
                },
                key=str.casefold,
            )
            progress.emit(
                "voice_resolution",
                (
                    "Speaker analysis and Azure voice resolution completed"
                    + (
                        "; context identified " + ", ".join(contextual_names)
                        if contextual_names
                        else ""
                    )
                ),
                status="completed",
                current=len(assignments),
                total=len(required_speaker_ids),
                cache_reused=False,
                contextual_identity_count=len(contextual_names),
                contextual_identity_names=contextual_names,
            )
            request_fingerprint = self._request_fingerprint(
                job,
                translation_path,
                blocks,
                assignments,
                options,
                pronunciation_dictionary,
                clean_background_stem,
                seo_context.seo_path,
                collision_resolution_payload,
                mastering_override_payload,
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

        for block_index, block in enumerate(baseline_blocks):
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

        source_tempo_calibration_report: dict[str, Any] | None = None
        if options.timing_mode in {
            TIMING_MODE_SEMANTIC_FIT,
            TIMING_MODE_PERFORMANCE_PLAN,
        }:
            synthesis_profiles, source_tempo_calibration_report = (
                self._calibrate_source_tempo_profiles(
                    baseline_blocks,
                    artifacts,
                    synthesis_profiles=synthesis_profiles,
                    assignments=assignments,
                    options=options,
                    force=force,
                    progress=progress,
                    pronunciation_dictionary=pronunciation_dictionary,
                )
            )

        semantic_batch_results = (
            self._synthesize_semantic_fit_batch(
                baseline_blocks,
                artifacts,
                synthesis_profiles=synthesis_profiles,
                options=options,
                force=force,
                progress=progress,
                pronunciation_dictionary=pronunciation_dictionary,
            )
            if options.timing_mode in {
                TIMING_MODE_SEMANTIC_FIT,
                TIMING_MODE_PERFORMANCE_PLAN,
            }
            else None
        )

        # Balanced mode measures the prepared three-variant set. Semantic-fit
        # ignores that fixed menu and shares measured feedback rounds across all
        # unresolved blocks at one immutable speaker rate.
        for block_index in range(len(baseline_blocks)):
            block = baseline_blocks[block_index]
            voice, base_rate, pitch, volume = synthesis_profiles[block_index]
            try:
                if options.timing_mode in {
                    TIMING_MODE_SEMANTIC_FIT,
                    TIMING_MODE_PERFORMANCE_PLAN,
                }:
                    assert semantic_batch_results is not None
                    semantic_outcome = semantic_batch_results[block_index]
                    if isinstance(semantic_outcome, PreparedVariantsExhausted):
                        raise semantic_outcome
                    selected_block, fitted, variant_audit = semantic_outcome
                else:
                    selected_block, fitted, variant_audit = self._synthesize_prepared_variants(
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
                    allowed_ratio = options.collision_max_time_stretch_ratio
                    if measured_index in {0, len(render_blocks) - 1}:
                        # Source-timeline edges have no speech on their outer
                        # side and are already governed by the separately
                        # bounded edge cadence policy. Plan the shared-boundary
                        # move against that same 1.08 ceiling so the global
                        # conductor does not reject a fit that final synthesis
                        # is explicitly allowed to render.
                        allowed_ratio = options.boundary_max_time_stretch_ratio
                    measured_required_windows_ms[measured_index] = math.ceil(
                        measured_duration_ms
                        / allowed_ratio
                    )

            render_blocks, five_block_adjustments = rebalance_five_block_timeline(
                render_blocks,
                measured_required_windows_ms,
                baseline_overflows,
            )
            applied_five_block_adjustments = [
                item for item in five_block_adjustments if item.get("applied")
            ]
            global_timeline_adjustments.extend(five_block_adjustments)
            if five_block_adjustments:
                progress.emit(
                    "timeline_fit",
                    (
                        "Measured five-block conductor completed: "
                        f"{len(applied_five_block_adjustments)}/"
                        f"{len(five_block_adjustments)} overflow window(s) "
                        "resolved without manual review"
                    ),
                    status=(
                        "completed"
                        if len(applied_five_block_adjustments)
                        == len(five_block_adjustments)
                        else "running"
                    ),
                    current=len(applied_five_block_adjustments),
                    total=len(five_block_adjustments),
                    backend="azure_measured_five_block_conductor",
                    authoritative_translation_changed=False,
                    ai_called=False,
                    manual_review_required=False,
                )

            render_blocks, edge_surplus_adjustments = (
                rebalance_edge_timeline_with_neighbour_surplus(
                    render_blocks,
                    measured_required_windows_ms,
                    baseline_overflows,
                    max_boundary_shift_ms=(
                        options.max_cross_speaker_boundary_shift_ms
                    ),
                )
            )
            applied_edge_surplus = [
                item for item in edge_surplus_adjustments if item.get("applied")
            ]
            global_timeline_adjustments.extend(applied_edge_surplus)
            if edge_surplus_adjustments:
                progress.emit(
                    "timeline_fit",
                    (
                        "Measured edge-window fit completed: "
                        f"{len(applied_edge_surplus)}/"
                        f"{len(edge_surplus_adjustments)} boundary overflow(s) "
                        "resolved from neighbouring Azure-measured surplus"
                    ),
                    status=(
                        "completed"
                        if len(applied_edge_surplus)
                        == len(edge_surplus_adjustments)
                        else "running"
                    ),
                    current=len(applied_edge_surplus),
                    total=len(edge_surplus_adjustments),
                    backend="azure_measured_edge_neighbour_surplus",
                    authoritative_translation_changed=False,
                    ai_called=False,
                )

            unresolved_after_non_overlapping_fit = {
                index
                for index in baseline_overflows
                if render_blocks[index].duration_ms
                < measured_required_windows_ms.get(
                    index,
                    render_blocks[index].duration_ms,
                )
            }
            render_blocks, sandwiched_timeline_adjustments = (
                rebalance_sandwiched_interjection_timeline(
                    render_blocks,
                    measured_required_windows_ms,
                    target_indices=unresolved_after_non_overlapping_fit,
                    max_total_overlap_ms=options.max_interjection_overlap_ms,
                    max_boundary_overlap_ms=(
                        options.max_interjection_boundary_overlap_ms
                    ),
                    max_source_gap_ms=options.max_interjection_source_gap_ms,
                    manual_overlap_resolutions=collision_resolutions,
                )
            )
            global_timeline_adjustments.extend(
                sandwiched_timeline_adjustments
            )
            unresolved_for_reviewed_fit = [
                adjustment
                for adjustment in global_timeline_adjustments
                if not adjustment.get("applied")
                and int(adjustment.get("block_index", -1)) in baseline_overflows
                and _source_block_id(
                    render_blocks[int(adjustment.get("block_index", -1))].block_id
                ) in collision_resolutions
            ]
            if unresolved_for_reviewed_fit:
                render_blocks, reviewed_fit_outcomes = (
                    apply_reviewed_triplet_silence_expansion(
                        render_blocks,
                        measured_required_windows_ms,
                        unresolved_for_reviewed_fit,
                        collision_resolutions,
                    )
                )
                reviewed_by_index = {
                    int(item.get("block_index", -1)): item
                    for item in reviewed_fit_outcomes
                    if int(item.get("block_index", -1)) >= 0
                }
                global_timeline_adjustments = [
                    reviewed_by_index.get(
                        int(adjustment.get("block_index", -1)), adjustment
                    )
                    for adjustment in global_timeline_adjustments
                ]
                applied_reviewed_fits = sum(
                    1 for item in reviewed_fit_outcomes if item.get("applied")
                )
                progress.emit(
                    "timeline_resolution",
                    (
                        "Applied reviewed collision decision(s): "
                        f"{applied_reviewed_fits}/{len(reviewed_fit_outcomes)} "
                        "triplet window(s) fitted with adjacent silence"
                    ),
                    status=(
                        "completed"
                        if applied_reviewed_fits == len(reviewed_fit_outcomes)
                        else "running"
                    ),
                    current=applied_reviewed_fits,
                    total=len(reviewed_fit_outcomes),
                    backend="human_reviewed_triplet_fit",
                )

            render_blocks, source_word_alignment_corrections = (
                enforce_source_word_speaker_alignment(
                    render_blocks,
                    blocks,
                    source_word_anchors,
                    required_windows_ms=measured_required_windows_ms,
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
            for correction in source_word_alignment_corrections:
                for affected_index in correction.get(
                    "affected_block_indices", []
                ):
                    index = int(affected_index)
                    prior = global_timeline_adjustments_by_index.get(index, {})
                    prior_alignments = list(
                        prior.get("source_word_alignments") or []
                    )
                    global_timeline_adjustments_by_index[index] = {
                        **prior,
                        "source_word_alignment": correction,
                        "source_word_alignments": [
                            *prior_alignments,
                            correction,
                        ],
                        "affected_block_index": index,
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
                        f"{applied_global_adjustments} edge/triplet window(s) "
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
                # A failed geometric preflight is evidence for the finalizer,
                # not a reason to stop the paid render. The block loop below
                # remeasures the exact selected candidate, tries the local AI
                # conductor, and creates a review case only if all validated
                # automatic completion paths are exhausted.
                progress.emit(
                    "timing_preflight",
                    (
                        f"{len(unresolved_global_adjustments)} measured window(s) "
                        "need final-conductor fitting; continuing automatically"
                    ),
                    status="running",
                    current=0,
                    total=len(unresolved_global_adjustments),
                    backend="finalization_conductor",
                    review_deferred=True,
                )

        source_word_alignment_artifact = {
            "schema_version": SOURCE_WORD_ALIGNMENT_VERSION,
            "job_id": job.job_id,
            "created_at": utcnow(),
            "source_transcript_path": (
                str(source_transcript_path)
                if source_transcript_path.is_file()
                else None
            ),
            "source_transcript_sha256": (
                checksum(source_transcript_path)
                if source_transcript_path.is_file()
                else None
            ),
            "tolerance_ms": options.source_word_alignment_tolerance_ms,
            "policy": {
                "cross_speaker_boundaries_are_word_anchored": True,
                "source_word_overlap_is_authoritative": True,
                "source_overlap_is_not_collapsed_to_serial_handoff": True,
                "same_speaker_boundaries_remain_flexible": True,
                "source_silence_remains_flexible": True,
                "alignment_takes_precedence_over_geometric_capacity": True,
                "remaining_deficit_routes_to_final_conductor": True,
                "manual_review_required": False,
            },
            "anchor_count": len(source_word_anchors),
            "correction_count": len(source_word_alignment_corrections),
            "anchors": source_word_anchors,
            "corrections": source_word_alignment_corrections,
        }
        atomic_write_json(
            artifacts.source_word_alignment,
            source_word_alignment_artifact,
        )
        progress.emit(
            "word_alignment",
            (
                f"Protected {len(source_word_anchors)} cross-speaker handoff(s) "
                "with canonical source-word timing; corrected "
                f"{len(source_word_alignment_corrections)} post-fit boundary(s)"
            ),
            status="completed",
            current=len(source_word_anchors),
            total=len(source_word_anchors),
            anchor_count=len(source_word_anchors),
            correction_count=len(source_word_alignment_corrections),
            tolerance_ms=options.source_word_alignment_tolerance_ms,
            same_speaker_boundaries_remain_flexible=True,
            text_changed=False,
            manual_review_required=False,
            artifact_path=str(artifacts.source_word_alignment),
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
                boundary_candidate = (
                    expanded if expanded.duration_ms > block.duration_ms else block
                )
                boundary_plan = _bounded_boundary_time_stretch_plan(
                    render_blocks,
                    block_index,
                    latest_overflow,
                    maximum_ratio=options.boundary_max_time_stretch_ratio,
                )
                if boundary_plan is not None:
                    try:
                        boundary_fitted = self._synthesize_block(
                            boundary_candidate,
                            artifacts,
                            voice=voice,
                            base_rate_percent=base_rate,
                            pitch_percent=pitch,
                            volume_percent=volume,
                            max_azure_rate_percent=options.max_azure_rate_percent,
                            max_time_stretch_ratio=(
                                options.boundary_max_time_stretch_ratio
                            ),
                            max_time_expansion_ratio=(
                                options.max_time_expansion_ratio
                            ),
                            cadence_max_time_stretch_ratio=(
                                options.boundary_max_time_stretch_ratio
                            ),
                            force=False,
                        )
                    except TimingOverflowError as boundary_exc:
                        latest_overflow = boundary_exc
                    else:
                        block = boundary_candidate
                        render_blocks[block_index] = boundary_candidate
                        boundary_plan["applied_time_stretch_ratio"] = (
                            boundary_fitted.get("post_synthesis_time_stretch_ratio")
                        )
                        if boundary_candidate.duration_ms > blocks[block_index].duration_ms:
                            boundary_plan["safe_silence_adjustment"] = dict(
                                timing_adjustment
                            )
                        boundary_fitted["timing_adjustment"] = boundary_plan
                        fitted = boundary_fitted
                        progress.emit(
                            "timeline_fit",
                            (
                                f"{boundary_candidate.block_id} boundary overflow "
                                "auto-fitted with bounded measured time compression"
                            ),
                            status="completed",
                            block_id=boundary_candidate.block_id,
                            boundary=boundary_plan["boundary"],
                            required_time_stretch_ratio=boundary_plan[
                                "required_time_stretch_ratio"
                            ],
                            maximum_time_stretch_ratio=boundary_plan[
                                "maximum_time_stretch_ratio"
                            ],
                            authoritative_translation_changed=False,
                        )

            if fitted is None:
                assert latest_overflow is not None
                required_window_ms = math.ceil(
                    latest_overflow.measured_duration_ms
                    / options.collision_max_time_stretch_ratio
                )
                handoff_anchor = source_word_anchor_by_boundary.get(block_index)
                maximum_word_aligned_shift_ms = (
                    max(
                        0,
                        int(
                            handoff_anchor.get(
                                "handoff_max_ms"
                                if handoff_anchor.get("compact_handoff")
                                else "left_end_max_ms",
                                block.end_ms,
                            )
                        )
                        - block.end_ms,
                    )
                    if handoff_anchor is not None
                    else options.max_cross_speaker_boundary_shift_ms
                )
                expanded, shifted_next, timing_adjustment = (
                    rebalance_cross_speaker_handoff(
                        render_blocks,
                        block_index,
                        required_window_ms=required_window_ms,
                        max_shift_ms=min(
                            options.max_cross_speaker_boundary_shift_ms,
                            options.source_word_alignment_tolerance_ms,
                            maximum_word_aligned_shift_ms,
                        ),
                        allow_sandwiched_interjection_expansion=(
                            block_index
                            not in source_word_anchored_boundary_indices
                        ),
                    )
                )
                if shifted_next is None:
                    reviewed_resolution = dict(
                        collision_resolutions.get(
                            _source_block_id(block.block_id), {}
                        )
                    )
                    if (
                        str(reviewed_resolution.get("strategy") or "")
                        == "allow_overlap"
                        and block_index > 0
                        and block_index + 1 < len(render_blocks)
                    ):
                        overlap_before_ms = int(
                            reviewed_resolution.get("overlap_before_ms") or 0
                        )
                        overlap_after_ms = int(
                            reviewed_resolution.get("overlap_after_ms") or 0
                        )
                        previous_block = render_blocks[block_index - 1]
                        following_block = render_blocks[block_index + 1]
                        reviewed_expanded = replace(
                            block,
                            start_ms=max(
                                previous_block.start_ms,
                                block.start_ms - overlap_before_ms,
                            ),
                            end_ms=min(
                                following_block.end_ms,
                                block.end_ms + overlap_after_ms,
                            ),
                        )
                        if reviewed_expanded.duration_ms >= required_window_ms:
                            expanded = reviewed_expanded
                            shifted_next = following_block
                            timing_adjustment = {
                                "method": "human_reviewed_collision_overlap",
                                "collision_id": _source_block_id(block.block_id),
                                "reviewed_by": reviewed_resolution.get("reviewed_by"),
                                "reviewed_at": reviewed_resolution.get("reviewed_at"),
                                "overlap_before_ms": max(
                                    0, previous_block.end_ms - reviewed_expanded.start_ms
                                ),
                                "overlap_after_ms": max(
                                    0, reviewed_expanded.end_ms - following_block.start_ms
                                ),
                                "authoritative_translation_changed": False,
                                "human_review_required": True,
                            }
                if shifted_next is None:
                    repair_block = boundary_candidate
                    conductor_result = self._attempt_finalization_conductor(
                        block=repair_block,
                        block_index=block_index,
                        block_count=len(render_blocks),
                        artifacts=artifacts,
                        voice=voice,
                        base_rate_percent=base_rate,
                        pitch_percent=pitch,
                        volume_percent=volume,
                        options=options,
                        latest_overflow=latest_overflow,
                        duration_evidence=(
                            prepared_variant_audits.get(block_index)
                            or repair_audit
                        ),
                        progress=progress,
                        pronunciation_dictionary=pronunciation_dictionary,
                    )
                    if conductor_result is not None:
                        block, fitted, conductor_audit = conductor_result
                        render_blocks[block_index] = block
                        repair_audit.extend(conductor_audit)
                if shifted_next is None and fitted is None:
                    # Keep any collision-free silence expansion discovered
                    # above. Falling back to the original smaller window made
                    # the emergency fit compress speech that already had safe
                    # neighbouring room available.
                    block = boundary_candidate
                    render_blocks[block_index] = boundary_candidate
                    five_block_window = _collision_five_block_window(
                        render_blocks,
                        block_index,
                        prepared_variant_audits,
                    )
                    combined_repair_audit = list(
                        prepared_variant_audits.get(block_index) or ()
                    )
                    combined_repair_audit.extend(repair_audit)
                    fitted = self._force_final_product_fit(
                        block=block,
                        artifacts=artifacts,
                        voice=voice,
                        base_rate_percent=base_rate,
                        pitch_percent=pitch,
                        volume_percent=volume,
                        options=options,
                        overflow=latest_overflow,
                        five_block_window=five_block_window,
                        repair_audit=combined_repair_audit,
                        progress=progress,
                    )
                if fitted is None:
                    assert shifted_next is not None
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
            if (
                not fitted.get("finalization_conductor")
                and selected_rate - base_rate > 5
            ):
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
            if (
                not fitted.get("finalization_conductor")
                and not (fitted.get("cadence_qc") or {}).get("passed", False)
            ):
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
                                    max_shift_ms=min(
                                        options.max_cross_speaker_boundary_shift_ms,
                                        options.source_word_alignment_tolerance_ms,
                                        max(
                                            0,
                                            int(
                                                source_word_anchor_by_boundary.get(
                                                    block_index,
                                                    {},
                                                ).get(
                                                    "handoff_max_ms"
                                                    if source_word_anchor_by_boundary.get(
                                                        block_index,
                                                        {},
                                                    ).get("compact_handoff")
                                                    else "left_end_max_ms",
                                                    block.end_ms
                                                    + options.source_word_alignment_tolerance_ms,
                                                )
                                            )
                                            - block.end_ms,
                                        ),
                                    ),
                                    allow_sandwiched_interjection_expansion=(
                                        block_index
                                        not in source_word_anchored_boundary_indices
                                    ),
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
                    "source_timing_evidence": block.source_timing_evidence,
                    "source_speech_islands": [
                        dict(value) for value in block.source_speech_islands
                    ],
                    "source_interactions": [
                        dict(value) for value in block.source_interactions
                    ],
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
        clean_master_stage = (
            "video_remux"
            if options.clean_master_video_mode == "stream_copy"
            else "video_render"
        )
        progress.emit(
            clean_master_stage,
            (
                "Remuxing the clean dubbed master with copied source video"
                if options.clean_master_video_mode == "stream_copy"
                else "Rendering the clean dubbed master MP4"
            ),
            status="started",
            video_encoding_required=(
                options.clean_master_video_mode != "stream_copy"
            ),
        )

        def render_progress(event: dict[str, Any]) -> None:
            payload = dict(event)
            progress.emit(
                str(payload.pop("stage", clean_master_stage)),
                str(
                    payload.pop(
                        "message",
                        (
                            "FFmpeg clean-master remux"
                            if options.clean_master_video_mode == "stream_copy"
                            else "FFmpeg clean-master encoding"
                        ),
                    )
                ),
                status=str(payload.pop("status", "running")),
                current=payload.pop("current", None),
                total=payload.pop("total", None),
                **payload,
            )

        render_manifest = render_review_mp4(
            source_video,
            artifacts.final_mix,
            artifacts.dubbed_master,
            artifacts.render_manifest,
            runner=self.runner,
            progress_callback=(
                render_progress
                if progress_callback is not None and self.popen_factory is not None
                else None
            ),
            popen_factory=self.popen_factory,
            video_mode=options.clean_master_video_mode,
            progress_stage=clean_master_stage,
        )
        artifacts.legacy_final_video.unlink(missing_ok=True)
        progress.emit(
            clean_master_stage,
            "Clean dubbed master created; publishing named outputs",
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
            clean_master_stage,
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
            "timing_strategy": "azure_measured_five_block_conductor",
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
        ai_timing_repairs = [
            {
                "block_id": render_blocks[index].block_id,
                "source_block_id": blocks[index].block_id,
                "timing_adjustment": dict(
                    result.get("finalization_conductor")
                    or result.get("timing_adjustment")
                    or {}
                ),
                "translation_variant": dict(result.get("translation_variant") or {}),
                "attempts": list(result.get("timing_repair_attempts") or []),
            }
            for index, result in enumerate(block_results)
            if (
                result.get("finalization_conductor")
                or result.get("timing_adjustment")
                or {}
            ).get("method")
            == "finalization_ai_conductor"
        ]
        ai_timing_repair_count = len(ai_timing_repairs)
        final_product_emergency_fits = [
            {
                "block_id": render_blocks[index].block_id,
                "source_block_id": blocks[index].block_id,
                "timing_adjustment": dict(
                    result.get("finalization_conductor")
                    or result.get("timing_adjustment")
                    or {}
                ),
            }
            for index, result in enumerate(block_results)
            if (
                result.get("finalization_conductor")
                or result.get("timing_adjustment")
                or {}
            ).get("method")
            == "final_product_emergency_fit"
        ]
        applied_five_block_fits = [
            dict(value)
            for value in global_timeline_adjustments
            if value.get("method") == "five_block_measured_timeline_fit"
            and value.get("applied")
        ]
        source_pause_aligned_blocks = [
            {
                "block_id": str(result.get("block_id") or ""),
                **dict(result.get("source_pause_alignment") or {}),
            }
            for result in block_results
            if (result.get("source_pause_alignment") or {}).get("applied")
        ]
        source_lipsync_endpoints = [
            {
                "block_id": str(result.get("block_id") or ""),
                **dict(result.get("source_lipsync_endpoint") or {}),
            }
            for result in block_results
            if (result.get("source_lipsync_endpoint") or {}).get(
                "source_speech_end_ms"
            )
            is not None
        ]
        semantic_fit_blocks = [
            {
                "block_id": str(result.get("block_id") or ""),
                **dict(result.get("semantic_fit") or {}),
            }
            for result in block_results
            if result.get("semantic_fit")
        ]
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
                "boundary_policy_version": (
                    APPROVED_SEGMENT_BOUNDARY_POLICY_VERSION
                ),
                "boundary_normalization_count": len(boundary_normalizations),
                "boundary_normalizations": boundary_normalizations,
                "skipped_non_speech_unit_count": len(
                    skipped_non_speech_unit_ids
                ),
                "skipped_non_speech_unit_ids": skipped_non_speech_unit_ids,
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
                "ai_timing_repair_count": ai_timing_repair_count,
                "ai_timing_repairs": ai_timing_repairs or None,
                "semantic_fit_count": len(semantic_fit_blocks),
                "semantic_fit_checkpoint": (
                    str(artifacts.semantic_fit)
                    if artifacts.semantic_fit.is_file()
                    else None
                ),
                "semantic_phrase_group_version": SEMANTIC_PHRASE_GROUP_VERSION,
                "semantic_phrase_group_count": len(
                    semantic_phrase_group_audits
                ),
                "semantic_phrase_groups": semantic_phrase_group_audits,
                "performance_plan_version": PERFORMANCE_PLAN_VERSION,
                "performance_region_count": sum(
                    1
                    for value in performance_region_audits
                    if value.get("region_id")
                ),
                "performance_plan": (
                    str(artifacts.performance_plan)
                    if artifacts.performance_plan.is_file()
                    else None
                ),
                "performance_plan_sha256": (
                    checksum(artifacts.performance_plan)
                    if artifacts.performance_plan.is_file()
                    else None
                ),
                "five_block_timeline_fit_count": len(applied_five_block_fits),
                "final_product_emergency_fit_count": len(
                    final_product_emergency_fits
                ),
                "ai_timing_repair_checkpoint": (
                    str(artifacts.finalization_conductor)
                    if artifacts.finalization_conductor.is_file()
                    else None
                ),
                "wording_revision_allowed": ai_timing_repair_count > 0,
                "approved_translation_artifact_immutable": True,
                "renderer_rewrites_text": ai_timing_repair_count > 0,
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
                "source_tempo_calibration_version": (
                    SOURCE_TEMPO_CALIBRATION_VERSION
                ),
                "source_tempo_calibration": (
                    str(artifacts.source_tempo_calibration)
                    if artifacts.source_tempo_calibration.is_file()
                    else None
                ),
                "source_tempo_calibration_sha256": (
                    checksum(artifacts.source_tempo_calibration)
                    if artifacts.source_tempo_calibration.is_file()
                    else None
                ),
                "source_tempo_speakers": (
                    dict(source_tempo_calibration_report.get("speakers") or {})
                    if source_tempo_calibration_report
                    else {}
                ),
            },
            "strategy": {
                "pipeline": "direct_approved_segments_to_azure",
                "timing_mode": options.timing_mode,
                "orchestrator_used": bool(
                    ai_timing_repair_count
                    or applied_five_block_fits
                    or final_product_emergency_fits
                ),
                "generated_micro_units": False,
                "same_speaker_coalescing_only": True,
                "speech_block_count": len(blocks),
                "same_speaker_gap_ms": options.same_speaker_gap_ms,
                "max_azure_rate_percent": options.max_azure_rate_percent,
                "max_time_stretch_ratio": options.max_time_stretch_ratio,
                "boundary_max_time_stretch_ratio": (
                    options.boundary_max_time_stretch_ratio
                ),
                "final_product_max_time_stretch_ratio": (
                    options.final_product_max_time_stretch_ratio
                ),
                "max_time_expansion_ratio": options.max_time_expansion_ratio,
                "source_word_alignment_version": SOURCE_WORD_ALIGNMENT_VERSION,
                "source_pause_alignment_version": SOURCE_PAUSE_ALIGNMENT_VERSION,
                "transcript_lipsync_schema_version": (
                    TRANSCRIPT_LIPSYNC_SCHEMA_VERSION
                ),
                "transcript_lipsync_policy_version": (
                    TRANSCRIPT_LIPSYNC_POLICY_VERSION
                ),
                "timing_evidence": "azure_transcript_word_timestamps",
                "vad_required": False,
                "visual_mouth_analysis_required": False,
                "gpu_required_for_timing_plan": False,
                "source_word_alignment_tolerance_ms": (
                    options.source_word_alignment_tolerance_ms
                ),
                "quality_priority_order": [
                    "source_speaker_tempo",
                    "source_word_lipsync",
                    "semantic_rephrase_to_fit",
                ],
                "source_tempo_calibration_applied": bool(
                    source_tempo_calibration_report
                    and any(
                        value.get("status") == "calibrated"
                        for value in (
                            source_tempo_calibration_report.get("speakers") or {}
                        ).values()
                        if isinstance(value, Mapping)
                    )
                ),
                "prepared_translation_variants": (
                    options.timing_mode == TIMING_MODE_BALANCED
                ),
                "variant_selection_mode": (
                    "performance_region_beats_locked_tempo"
                    if options.timing_mode == TIMING_MODE_PERFORMANCE_PLAN
                    else (
                        "iterative_semantic_fit_locked_tempo"
                        if options.timing_mode == TIMING_MODE_SEMANTIC_FIT
                        else "predict_then_measure_natural_concise_compact"
                    )
                ),
                "timing_repair_mode": (
                    "semantic_fit_strict_then_final_product_rescue"
                    if (
                        options.timing_mode == TIMING_MODE_SEMANTIC_FIT
                        and final_product_emergency_fits
                    )
                    else (
                        "semantic_fit_no_tempo_or_timeline_repair"
                        if options.timing_mode
                        in {
                            TIMING_MODE_SEMANTIC_FIT,
                            TIMING_MODE_PERFORMANCE_PLAN,
                        }
                        else "five_block_fit_then_validated_ai_then_final_product"
                    )
                ),
                "timing_repair_backend": (
                    "gpt_local_rewrite_plus_azure_measurement"
                    if ai_timing_repair_count
                    else "deterministic_azure_measurement"
                ),
                "timing_repair_batch_size": 1 if ai_timing_repair_count else 0,
                "timing_repair_batch_count": ai_timing_repair_count,
                "semantic_fit_schema_version": SEMANTIC_FIT_SCHEMA_VERSION,
                "semantic_fit_validator_version": (
                    SEMANTIC_FIT_VALIDATOR_VERSION
                ),
                "semantic_fit_max_attempts": options.semantic_fit_max_attempts,
                "semantic_fit_tolerance_ms": options.semantic_fit_tolerance_ms,
                "speaker_tempo_locked": (
                    options.timing_mode
                    in {
                        TIMING_MODE_SEMANTIC_FIT,
                        TIMING_MODE_PERFORMANCE_PLAN,
                    }
                ),
                "per_block_rate_change_allowed": (
                    options.timing_mode
                    not in {
                        TIMING_MODE_SEMANTIC_FIT,
                        TIMING_MODE_PERFORMANCE_PLAN,
                    }
                ),
                "post_synthesis_time_stretch_allowed": (
                    options.timing_mode
                    not in {
                        TIMING_MODE_SEMANTIC_FIT,
                        TIMING_MODE_PERFORMANCE_PLAN,
                    }
                ),
                "manual_timeline_collision_review_blocks_render": False,
                "audio_mode": audio_mode,
                "background_included": options.include_background,
                "speaker_volume_percent": 0,
                "dialogue_target_lufs": options.dialogue_target_lufs,
                "dialogue_true_peak_dbfs": options.dialogue_true_peak_dbfs,
                "dialogue_loudness_range_lu": options.dialogue_loudness_range_lu,
                "clean_master_video_mode": options.clean_master_video_mode,
                "clean_master_video_encoded": (
                    options.clean_master_video_mode != "stream_copy"
                ),
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
            "timeline_collision_resolutions": {
                "applied_count": len(timeline_collision_resolution_audits),
                "audits": timeline_collision_resolution_audits,
                "resolution_path": (
                    str(artifacts.timeline_collision_resolutions)
                    if artifacts.timeline_collision_resolutions.is_file()
                    else None
                ),
            },
            "mastering_overrides": {
                "schema_version": DUB_MASTERING_OVERRIDE_SCHEMA_VERSION,
                "path": (
                    str(artifacts.mastering_overrides)
                    if artifacts.mastering_overrides.is_file()
                    else None
                ),
                "revision": int(mastering_override_payload.get("revision") or 0),
                "applied_count": len(mastering_override_audits),
                "audits": mastering_override_audits,
                "authoritative_translation_changed": False,
            },
            "timeline_conductor": {
                "solver_version": FIVE_BLOCK_TIMELINE_SOLVER_VERSION,
                "final_source_tail": final_source_tail_adjustment,
                "five_block_fit_count": len(applied_five_block_fits),
                "five_block_fits": applied_five_block_fits,
                "final_product_conductor_version": FINAL_PRODUCT_CONDUCTOR_VERSION,
                "emergency_fit_count": len(final_product_emergency_fits),
                "emergency_fits": final_product_emergency_fits,
                "manual_review_blocking_enabled": False,
            },
            "source_word_alignment": {
                "schema_version": SOURCE_WORD_ALIGNMENT_VERSION,
                "path": str(artifacts.source_word_alignment),
                "sha256": checksum(artifacts.source_word_alignment),
                "anchor_count": len(source_word_anchors),
                "correction_count": len(source_word_alignment_corrections),
                "tolerance_ms": options.source_word_alignment_tolerance_ms,
                "cross_speaker_boundaries_protected": True,
                "same_speaker_boundaries_flexible": True,
                "manual_review_required": False,
            },
            "transcript_lipsync": {
                "schema_version": TRANSCRIPT_LIPSYNC_SCHEMA_VERSION,
                "policy_version": TRANSCRIPT_LIPSYNC_POLICY_VERSION,
                "path": str(artifacts.transcript_lipsync),
                "sha256": checksum(artifacts.transcript_lipsync),
                "word_timed_block_count": int(
                    transcript_lipsync_plan.get("word_timed_block_count") or 0
                ),
                "fallback_block_count": int(
                    transcript_lipsync_plan.get("fallback_block_count") or 0
                ),
                "interaction_count": int(
                    transcript_lipsync_plan.get("interaction_count") or 0
                ),
                "compute_profile": dict(
                    transcript_lipsync_plan.get("compute_profile") or {}
                ),
                "word_timed_speech_endpoint_authoritative": True,
                "endpoint_count": len(source_lipsync_endpoints),
                "maximum_absolute_endpoint_error_ms": max(
                    (
                        abs(int(value.get("endpoint_error_ms") or 0))
                        for value in source_lipsync_endpoints
                    ),
                    default=0,
                ),
                "endpoints": source_lipsync_endpoints,
                "approved_translation_changed": False,
            },
            "source_pause_alignment": {
                "schema_version": SOURCE_PAUSE_ALIGNMENT_VERSION,
                "detected_block_count": len(source_pause_alignment),
                "detected_blocks": source_pause_alignment,
                "applied_block_count": len(source_pause_aligned_blocks),
                "applied_blocks": source_pause_aligned_blocks,
                "minimum_source_pause_ms": MINIMUM_SOURCE_PAUSE_MS,
                "maximum_inserted_pause_ms": (
                    professional_policy.maximum_added_pause_ms
                ),
                "approved_translation_changed": False,
            },
            "semantic_fit": {
                "schema_version": SEMANTIC_FIT_SCHEMA_VERSION,
                "validator_version": SEMANTIC_FIT_VALIDATOR_VERSION,
                "strategy_version": SEMANTIC_FIT_BATCH_STRATEGY_VERSION,
                "enabled": options.timing_mode
                in {
                    TIMING_MODE_SEMANTIC_FIT,
                    TIMING_MODE_PERFORMANCE_PLAN,
                },
                "block_count": len(semantic_fit_blocks),
                "blocks": semantic_fit_blocks,
                "checkpoint_path": (
                    str(artifacts.semantic_fit)
                    if artifacts.semantic_fit.is_file()
                    else None
                ),
                "speaker_tempo_locked": (
                    options.timing_mode
                    in {
                        TIMING_MODE_SEMANTIC_FIT,
                        TIMING_MODE_PERFORMANCE_PLAN,
                    }
                ),
                "time_stretch_used": False,
                "approved_translation_artifact_changed": False,
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
        job.media["direct_dub_source_word_alignment"] = str(
            artifacts.source_word_alignment
        )
        job.media.pop("timeline_collision_review", None)
        job.media.pop("timeline_collision_panel", None)
        job.media.pop("timeline_collision_resolutions", None)
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
            timing_repair_count=ai_timing_repair_count,
            ai_timing_repair_count=ai_timing_repair_count,
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
                "identified_name",
                "voice_family",
                "voice_family_confidence",
                "voice_family_source",
                "raw_speaker_ids",
                "base_prosody",
                "source_features",
                "resolution_status",
                "known_speaker",
                "contextual_identity",
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
        name_research_path = analysis_root / "autocorrection_name_research.json"
        contextual_identity_path = (
            analysis_root / "contextual_speaker_identities.json"
        )
        legacy_assignment_path = analysis_root / "azure_voice_assignments.json"

        transcript = read_json(transcript_path) if transcript_path.is_file() else {"segments": []}
        diarization = read_json(diarization_path) if diarization_path.is_file() else {"turns": []}
        mapping = build_speaker_id_mapping(transcript, diarization)
        canonical_speaker_turns: dict[str, list[Mapping[str, Any]]] = {}
        has_contextual_turn_repairs = False
        for segment in transcript.get("segments", []):
            if not isinstance(segment, Mapping):
                continue
            canonical_id = str(
                segment.get("speaker_id") or segment.get("speaker") or ""
            ).strip()
            if canonical_id:
                canonical_speaker_turns.setdefault(canonical_id, []).append(segment)
            if str(segment.get("speaker_assignment_method") or "").startswith(
                "contextual_"
            ):
                has_contextual_turn_repairs = True
        # A repaired transcript can split one faulty raw diarization label into
        # multiple canonical speakers. Any reels or classifier output created
        # before that split are contaminated and must not be reused.
        effective_voice_analysis_refresh = (
            refresh_voice_analysis or has_contextual_turn_repairs
        )
        acoustic_raw = read_json(acoustic_path) if acoustic_path.is_file() else {}
        acoustic = canonicalize_voice_family_evidence(acoustic_raw, mapping)

        def cached_acoustic_is_resolved(speaker_id: str) -> bool:
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

        name_research = (
            read_json(name_research_path)
            if name_research_path.is_file()
            else (
                (transcript.get("autocorrect") or {}).get("name_research")
                if isinstance(transcript.get("autocorrect"), Mapping)
                else {}
            )
        )
        if not isinstance(name_research, Mapping):
            name_research = {}
        contextual_identities = resolve_contextual_speaker_identities(
            job_id=job.job_id,
            transcript=transcript,
            name_research=name_research,
            required_speaker_ids=required_speaker_ids,
            output_path=contextual_identity_path,
            registry_path=(
                self.settings.work_dir
                / "contextual_speaker_identities"
                / "registry.json"
            ),
            voice_inference_provider_factory=(
                self._timing_repair_provider
                if self.timing_repair_provider_factory is not None
                else None
            ),
            voice_inference_speaker_ids=[
                speaker_id
                for speaker_id in required_speaker_ids
                if (
                    has_contextual_turn_repairs
                    or not cached_acoustic_is_resolved(speaker_id)
                )
            ],
        )

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
                    canonical_speaker_turns=canonical_speaker_turns,
                    force=effective_voice_analysis_refresh,
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
            and speaker_id not in contextual_identities
            and (
                effective_voice_analysis_refresh
                or not evidence_is_resolved(speaker_id)
            )
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
                    canonical_speaker_turns=canonical_speaker_turns,
                    confidence_threshold=options.min_voice_family_confidence,
                    force=effective_voice_analysis_refresh,
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
            contextual_identity = contextual_identities.get(speaker_id)

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
            elif family == "unknown" and contextual_identity is not None:
                contextual_family = str(
                    contextual_identity.get("voice_family") or "unknown"
                ).lower()
                if contextual_family in {"masculine", "feminine"}:
                    family = contextual_family
                    confidence = float(
                        contextual_identity.get("voice_family_confidence") or 0.99
                    )
                    source = str(
                        contextual_identity.get("voice_family_source")
                        or "contextual_transcript_identity_gender"
                    )
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
                    "identified_name": (
                        known_match_record.get("display_name")
                        if known_match_record is not None
                        else (
                            contextual_identity.get("identified_name")
                            if contextual_identity is not None
                            else profile.get("identified_name")
                        )
                    ),
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
                if contextual_identity is not None and known_match_record is None:
                    assignment.update(
                        {
                            "resolution_status": (
                                "contextual_speaker_identity"
                                if contextual_identity.get("identified_name")
                                else "contextual_transcript_voice_family"
                            ),
                            "contextual_identity": dict(contextual_identity),
                        }
                    )
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
                        "contextual_identity": (
                            dict(contextual_identity)
                            if contextual_identity is not None
                            else None
                        ),
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
                    "contextual_identity": (
                        dict(contextual_identity)
                        if contextual_identity is not None
                        else None
                    ),
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
            "contextual_identity_policy_version": (
                CONTEXTUAL_SPEAKER_POLICY_VERSION
            ),
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
                "name_research": str(name_research_path),
                "contextual_speaker_identities": str(contextual_identity_path),
                "contextual_identity_registry": str(
                    self.settings.work_dir
                    / "contextual_speaker_identities"
                    / "registry.json"
                ),
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
                "Voice assignment remains unresolved for required speaker(s): "
                f"{details}. Contextual identity and acoustic voice-family "
                "evidence were insufficient; unknown voice families no longer "
                f"default to Themba. Inspect {contextual_identity_path} and "
                f"{output_path}, or provide --voice-map."
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
        cadence_max_time_stretch_ratio: float | None = None,
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
        pause_alignment_by_output_path: dict[str, dict[str, Any]] = {}
        initial = self.backend.synthesize(
            request,
            artifacts.candidate(request),
            force=force,
        )
        attempts.append(initial)
        timing_request = request
        timing_reference = initial

        if (
            block.source_pause_anchors
            and initial.duration_ms < block.duration_ms
        ):
            pause_budget_ms = min(
                self.backend.ssml_bounds.total_pause_max_ms,
                max(
                    0,
                    block.duration_ms
                    - initial.duration_ms
                    - SOURCE_PAUSE_TRANSITION_ROOM_MS,
                ),
            )
            pause_parts, pause_alignment = build_source_pause_ssml_parts(
                block.tts_text,
                block.source_pause_anchors,
                pause_budget_ms=pause_budget_ms,
                language="zu-ZA",
                pause_max_ms=int(
                    getattr(
                        self.backend.ssml_bounds,
                        "pause_max_ms",
                        2000,
                    )
                ),
                protected_entities=block.protected_entities,
            )
            if pause_alignment.get("applied") and pause_parts:
                pause_request = replace(request, parts=pause_parts)
                pause_candidate = self.backend.synthesize(
                    pause_request,
                    artifacts.candidate(pause_request),
                    force=force,
                )
                attempts.append(pause_candidate)
                pause_alignment_by_output_path[pause_candidate.output_path] = (
                    pause_alignment
                )
                if pause_candidate.duration_ms <= block.duration_ms:
                    timing_request = pause_request
                    timing_reference = pause_candidate

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
        elif block.duration_ms - timing_reference.duration_ms > 750:
            estimated = estimate_required_azure_rate(
                measured_duration_ms=timing_reference.duration_ms,
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
                slower_request = timing_request.with_rate(slower_rate)
                slower_candidate = self.backend.synthesize(
                    slower_request,
                    artifacts.candidate(slower_request),
                    force=force,
                )
                attempts.append(slower_candidate)
                if timing_request is not request:
                    pause_alignment_by_output_path[slower_candidate.output_path] = (
                        pause_alignment_by_output_path[
                            timing_reference.output_path
                        ]
                    )

        selected = _select_azure_timing_attempt(
            attempts,
            window_duration_ms=block.duration_ms,
            base_rate_percent=base_rate_percent,
            maximum_underfill_slowdown_percent=(
                ProfessionalDubPolicy().maximum_rate_delta_percent
            ),
        )
        final_path = artifacts.final_block(block.block_id)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        stretch_ratio = 1.0
        timing_action = "unchanged"
        pause_fill = dict(
            pause_alignment_by_output_path.get(selected.output_path) or {}
        )
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
            timing_action = (
                "source_pause_aligned_compression"
                if pause_fill
                else "compressed_to_window"
            )
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
                timing_action = (
                    "source_pause_aligned_expansion"
                    if pause_fill
                    else "source_cadence_aligned_expansion"
                )
            else:
                _atomic_copy(Path(selected.output_path), final_path)
                timing_action = (
                    "source_pause_aligned"
                    if pause_fill
                    else "natural_speech_with_trailing_room"
                )

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
        cadence_repaired = timing_action in {
            "source_cadence_aligned_expansion",
            "source_pause_aligned_expansion",
        }
        policy = ProfessionalDubPolicy()
        cadence_qc = assess_perceptual_cadence(
            selected_rate_percent=selected.rate_percent,
            base_rate_percent=base_rate_percent,
            whole_voice_stretch_ratio=stretch_ratio,
            timeline_fill_action=timing_action,
            pause_fill=pause_fill,
            trailing_room_ms=max(0, block.duration_ms - final_duration_ms),
            maximum_added_pause_ms=policy.maximum_added_pause_ms,
            maximum_rate_delta_percent=(
                policy.maximum_cadence_repair_rate_delta_percent
                if cadence_repaired
                else policy.maximum_rate_delta_percent
            ),
            maximum_whole_voice_change=(
                cadence_max_time_stretch_ratio
                if cadence_max_time_stretch_ratio is not None
                else (
                    policy.maximum_cadence_repair_voice_change
                    if cadence_repaired
                    else policy.maximum_whole_voice_change
                )
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
                    "source_pause_alignment": (
                        pause_alignment_by_output_path.get(item.output_path)
                    ),
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
                "source_pause_aligned",
                "source_pause_aligned_expansion",
            },
            "cadence_plan": cadence_plan.to_dict(),
            "cadence_qc": cadence_qc,
            "source_pause_alignment": pause_fill or None,
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
        if block.mastering_variant_id:
            locked = [
                value
                for value in candidates
                if value.variant_id == block.mastering_variant_id
            ]
            if not locked:
                raise DirectDubError(
                    f"Mastering locked unavailable variant "
                    f"{block.mastering_variant_id!r} for "
                    f"{_source_block_id(block.block_id)}"
                )
            return locked
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
        exhausted_candidates: list[
            tuple[SpeechBlock, TimingOverflowError]
        ] = []
        candidates = self._variant_candidates(block)
        for candidate_index, variant in enumerate(candidates, start=1):
            candidate_block = block.with_variant(variant)
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
                exhausted_candidates.append((candidate_block, exc))
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
        assert exhausted_candidates
        # The discovery loop is ordered for semantic quality, not duration.
        # When every candidate overflows, retain the shortest *measured*
        # candidate for the downstream timeline conductor.  Keeping the last
        # attempted candidate made boundary fitting depend on iteration order
        # and could discard a compact version that needed only a few hundred
        # milliseconds of neighbouring surplus.
        best_block, best_overflow = min(
            exhausted_candidates,
            key=lambda item: (
                item[1].measured_duration_ms,
                item[1].stretch_ratio,
                item[0].selected_variant_id,
            ),
        )
        raise PreparedVariantsExhausted(best_block, best_overflow, attempts)

    def _request_fingerprint(
        self,
        job: JobManifest,
        translation_path: Path,
        blocks: Sequence[SpeechBlock],
        assignments: Mapping[str, Mapping[str, Any]],
        options: DirectDubOptions,
        pronunciation_dictionary: PronunciationDictionary,
        clean_background_stem: Path | None,
        seo_path: Path,
        timeline_collision_resolutions: Mapping[str, Any],
        mastering_overrides: Mapping[str, Any],
    ) -> str:
        payload = {
            "schema_version": DIRECT_DUB_SCHEMA_VERSION,
            "job_id": job.job_id,
            "source_sha256": checksum(Path(job.local_source_path)),
            "translation_sha256": checksum(translation_path),
            "pronunciation_dictionary_sha256": pronunciation_dictionary.sha256,
            "seo_sha256": checksum(seo_path),
            "timeline_collision_resolutions": dict(
                timeline_collision_resolutions
            ),
            "mastering_overrides": dict(mastering_overrides),
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


def extend_final_speech_block_into_source_tail(
    blocks: Sequence[SpeechBlock],
    *,
    source_duration_ms: int,
    maximum_tail_borrow_ms: int = 5000,
) -> tuple[list[SpeechBlock], dict[str, Any]]:
    """Expose bounded unused media tail to the final timeline conductor.

    Broadcast clips commonly retain a few seconds of picture after the final
    transcribed words.  Treating the last speech boundary as the media boundary
    forced the previous implementation to overlap the final speakers even when
    enough silent source tail already existed.  This changes timing capacity
    only; approved text, speaker order, and the source-video duration remain
    unchanged.
    """

    values = list(blocks)
    if not values:
        return values, {
            "method": "bounded_final_source_tail_capacity",
            "applied": False,
            "reason": "no_speech_blocks",
            "source_duration_ms": int(source_duration_ms),
            "maximum_tail_borrow_ms": int(maximum_tail_borrow_ms),
            "borrowed_tail_ms": 0,
        }
    if source_duration_ms <= 0:
        raise ValueError("source_duration_ms must be positive")
    if maximum_tail_borrow_ms < 0:
        raise ValueError("maximum_tail_borrow_ms must be non-negative")
    final = values[-1]
    available_tail_ms = max(0, int(source_duration_ms) - final.end_ms)
    borrowed_tail_ms = min(available_tail_ms, int(maximum_tail_borrow_ms))
    adjustment = {
        "method": "bounded_final_source_tail_capacity",
        "block_index": len(values) - 1,
        "block_id": final.block_id,
        "original_end_ms": final.end_ms,
        "source_duration_ms": int(source_duration_ms),
        "available_tail_ms": available_tail_ms,
        "maximum_tail_borrow_ms": int(maximum_tail_borrow_ms),
        "borrowed_tail_ms": borrowed_tail_ms,
        "adjusted_end_ms": final.end_ms + borrowed_tail_ms,
        "text_immutable": True,
        "speaker_order_preserved": True,
        "cross_speaker_overlap": False,
        "source_video_extended": False,
    }
    if borrowed_tail_ms <= 0:
        adjustment.update({"applied": False, "reason": "no_unused_source_tail"})
        return values, adjustment
    values[-1] = replace(final, end_ms=final.end_ms + borrowed_tail_ms)
    adjustment.update(
        {
            "applied": True,
            "reason": "unused_source_tail_exposed_to_timeline_conductor",
        }
    )
    return values, adjustment


SANDWICHED_INTERJECTION_MAX_SHIFT_MS = 1000
SANDWICHED_INTERJECTION_MAX_SOURCE_WINDOW_MS = 1200
MINIMUM_SHIFTED_BLOCK_WINDOW_MS = 1000


def rebalance_cross_speaker_handoff(
    blocks: Sequence[SpeechBlock],
    block_index: int,
    *,
    required_window_ms: int,
    max_shift_ms: int = 500,
    allow_sandwiched_interjection_expansion: bool = True,
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
        if sandwiched_interjection and allow_sandwiched_interjection_expansion
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


def apply_reviewed_triplet_silence_expansion(
    blocks: Sequence[SpeechBlock],
    required_windows_ms: Mapping[int, int],
    unresolved_adjustments: Sequence[Mapping[str, Any]],
    resolutions: Mapping[str, Mapping[str, Any]],
    *,
    max_borrow_per_side_ms: int = 2500,
) -> tuple[list[SpeechBlock], list[dict[str, Any]]]:
    """Apply a reviewed A-B-A fit by borrowing only adjacent silent capacity.

    The ordinary triplet solver keeps the outer triplet boundaries fixed.  That
    is correct for automatic operation, but it cannot solve a collision when
    the two surrounding same-speaker turns alone need more room than the fixed
    triplet span.  Once a human has reviewed the case, this fallback may move
    the outer boundaries into verified gaps immediately before and after the
    triplet.  It never crosses another speech block, never changes speaker
    order or approved text, and uses only the amount needed by Azure's measured
    minimum windows.
    """

    if max_borrow_per_side_ms < 0:
        raise ValueError("max_borrow_per_side_ms must be non-negative")
    adjusted = list(blocks)
    outcomes: list[dict[str, Any]] = []

    for raw_adjustment in unresolved_adjustments:
        adjustment = dict(raw_adjustment)
        block_index = int(adjustment.get("block_index", -1))
        if block_index <= 0 or block_index >= len(adjusted) - 1:
            continue
        middle = adjusted[block_index]
        collision_id = _source_block_id(middle.block_id)
        resolution = dict(resolutions.get(collision_id) or {})
        if not resolution:
            continue

        previous = adjusted[block_index - 1]
        following = adjusted[block_index + 1]
        if not (
            previous.speaker_id == following.speaker_id
            and previous.speaker_id != middle.speaker_id
        ):
            outcomes.append(
                {
                    **adjustment,
                    "applied": False,
                    "reason": "reviewed_fit_requires_a_b_a_triplet",
                    "resolution_strategy": resolution.get("strategy"),
                }
            )
            continue

        required_previous_ms = max(
            1, int(required_windows_ms.get(block_index - 1, previous.duration_ms))
        )
        required_middle_ms = max(
            1, int(required_windows_ms.get(block_index, middle.duration_ms))
        )
        required_following_ms = max(
            1, int(required_windows_ms.get(block_index + 1, following.duration_ms))
        )
        original_cluster_start_ms = previous.start_ms
        original_cluster_end_ms = following.end_ms
        original_span_ms = original_cluster_end_ms - original_cluster_start_ms

        left_neighbour_end_ms = (
            adjusted[block_index - 2].end_ms if block_index >= 2 else previous.start_ms
        )
        right_neighbour_start_ms = (
            adjusted[block_index + 2].start_ms
            if block_index + 2 < len(adjusted)
            else following.end_ms
        )
        left_capacity_ms = min(
            max_borrow_per_side_ms,
            max(0, previous.start_ms - left_neighbour_end_ms),
        )
        right_capacity_ms = min(
            max_borrow_per_side_ms,
            max(0, right_neighbour_start_ms - following.end_ms),
        )

        strategy = str(resolution.get("strategy") or "")
        approved_before_ms = (
            max(0, int(resolution.get("overlap_before_ms") or 0))
            if strategy == "allow_overlap"
            else 0
        )
        approved_after_ms = (
            max(0, int(resolution.get("overlap_after_ms") or 0))
            if strategy == "allow_overlap"
            else 0
        )
        approved_total_overlap_ms = approved_before_ms + approved_after_ms
        required_total_ms = (
            required_previous_ms + required_middle_ms + required_following_ms
        )
        # Both conditions must hold: the surrounding same-speaker turns may not
        # overlap each other, and the complete triplet must fit after any
        # explicitly approved cross-speaker overlap.
        required_expansion_ms = max(
            0,
            required_previous_ms + required_following_ms - original_span_ms,
            required_total_ms - approved_total_overlap_ms - original_span_ms,
        )
        total_capacity_ms = left_capacity_ms + right_capacity_ms
        if required_expansion_ms > total_capacity_ms:
            outcomes.append(
                {
                    **adjustment,
                    "applied": False,
                    "reason": "reviewed_adjacent_silence_insufficient",
                    "resolution_strategy": strategy,
                    "required_expansion_ms": required_expansion_ms,
                    "left_silence_capacity_ms": left_capacity_ms,
                    "right_silence_capacity_ms": right_capacity_ms,
                    "total_silence_capacity_ms": total_capacity_ms,
                    "approved_overlap_before_ms": approved_before_ms,
                    "approved_overlap_after_ms": approved_after_ms,
                }
            )
            continue

        left_borrow_ms = min(left_capacity_ms, (required_expansion_ms + 1) // 2)
        right_borrow_ms = required_expansion_ms - left_borrow_ms
        if right_borrow_ms > right_capacity_ms:
            shift = right_borrow_ms - right_capacity_ms
            right_borrow_ms -= shift
            left_borrow_ms += shift
        if left_borrow_ms > left_capacity_ms:
            shift = left_borrow_ms - left_capacity_ms
            left_borrow_ms -= shift
            right_borrow_ms += shift
        if left_borrow_ms > left_capacity_ms or right_borrow_ms > right_capacity_ms:
            raise DirectDubError("Reviewed silence-borrow allocation exceeded capacity")

        cluster_start_ms = original_cluster_start_ms - left_borrow_ms
        cluster_end_ms = original_cluster_end_ms + right_borrow_ms
        cluster_span_ms = cluster_end_ms - cluster_start_ms
        previous_end_ms = cluster_start_ms + required_previous_ms
        following_start_ms = cluster_end_ms - required_following_ms
        if previous_end_ms > following_start_ms:
            outcomes.append(
                {
                    **adjustment,
                    "applied": False,
                    "reason": "reviewed_surrounding_turns_still_conflict",
                    "resolution_strategy": strategy,
                    "required_expansion_ms": required_expansion_ms,
                }
            )
            continue

        safe_slack_ms = cluster_span_ms - required_total_ms
        if safe_slack_ms >= 0:
            earliest_middle_start_ms = previous_end_ms
            latest_middle_start_ms = following_start_ms - required_middle_ms
        else:
            required_overlap_ms = -safe_slack_ms
            if (
                required_overlap_ms > approved_total_overlap_ms
                or required_overlap_ms > approved_before_ms + approved_after_ms
            ):
                outcomes.append(
                    {
                        **adjustment,
                        "applied": False,
                        "reason": "reviewed_overlap_allowance_insufficient",
                        "resolution_strategy": strategy,
                        "required_overlap_ms": required_overlap_ms,
                        "approved_overlap_before_ms": approved_before_ms,
                        "approved_overlap_after_ms": approved_after_ms,
                    }
                )
                continue
            earliest_middle_start_ms = max(
                previous_end_ms - approved_before_ms,
                following_start_ms - required_middle_ms,
            )
            latest_middle_start_ms = min(
                previous_end_ms,
                following_start_ms + approved_after_ms - required_middle_ms,
            )

        if earliest_middle_start_ms > latest_middle_start_ms:
            outcomes.append(
                {
                    **adjustment,
                    "applied": False,
                    "reason": "reviewed_triplet_placement_unavailable",
                    "resolution_strategy": strategy,
                }
            )
            continue

        middle_start_ms = min(
            max(middle.start_ms, earliest_middle_start_ms),
            latest_middle_start_ms,
        )
        middle_end_ms = middle_start_ms + required_middle_ms
        overlap_before_ms = max(0, previous_end_ms - middle_start_ms)
        overlap_after_ms = max(0, middle_end_ms - following_start_ms)

        candidate_previous = replace(
            previous, start_ms=cluster_start_ms, end_ms=previous_end_ms
        )
        candidate_middle = replace(
            middle, start_ms=middle_start_ms, end_ms=middle_end_ms
        )
        candidate_following = replace(
            following, start_ms=following_start_ms, end_ms=cluster_end_ms
        )
        if block_index >= 2 and candidate_previous.start_ms < left_neighbour_end_ms:
            raise DirectDubError("Reviewed triplet crossed previous speech")
        if (
            block_index + 2 < len(adjusted)
            and candidate_following.end_ms > right_neighbour_start_ms
        ):
            raise DirectDubError("Reviewed triplet crossed following speech")

        adjusted[block_index - 1] = candidate_previous
        adjusted[block_index] = candidate_middle
        adjusted[block_index + 1] = candidate_following
        outcomes.append(
            {
                **adjustment,
                "applied": True,
                "reason": "human_reviewed_triplet_fit",
                "solver_version": "human_reviewed_triplet_fit_v1_adjacent_silence",
                "resolution_strategy": strategy,
                "resolution_case_token": resolution.get("case_token"),
                "affected_block_indices": [
                    block_index - 1,
                    block_index,
                    block_index + 1,
                ],
                "required_windows_ms": {
                    candidate_previous.block_id: required_previous_ms,
                    candidate_middle.block_id: required_middle_ms,
                    candidate_following.block_id: required_following_ms,
                },
                "adjusted_windows_ms": {
                    candidate_previous.block_id: candidate_previous.duration_ms,
                    candidate_middle.block_id: candidate_middle.duration_ms,
                    candidate_following.block_id: candidate_following.duration_ms,
                },
                "left_silence_borrow_ms": left_borrow_ms,
                "right_silence_borrow_ms": right_borrow_ms,
                "overlap_before_ms": overlap_before_ms,
                "overlap_after_ms": overlap_after_ms,
                "cross_speaker_overlap": bool(overlap_before_ms + overlap_after_ms),
                "text_immutable": True,
                "speaker_order_preserved": True,
                "adjacent_speech_boundaries_preserved": True,
            }
        )

    return adjusted, outcomes


def rebalance_edge_timeline_with_neighbour_surplus(
    blocks: Sequence[SpeechBlock],
    required_windows_ms: Mapping[int, int],
    overflow_indices: Iterable[int],
    *,
    max_boundary_shift_ms: int = 500,
) -> tuple[list[SpeechBlock], list[dict[str, Any]]]:
    """Give a first/last overflow measured spare time from its one neighbour.

    The three-block solver deliberately ignores timeline edges because it needs
    both neighbours.  A final handoff can still be resolved safely when the
    previous Azure-measured sentence finishes well inside its assigned window
    (and symmetrically for the first handoff).  Consume source silence first,
    then move only the shared boundary by the remaining bounded amount.  No
    text, speaker order, clip start, or clip end is changed.
    """

    if max_boundary_shift_ms < 0:
        raise ValueError("edge boundary shift must be non-negative")
    adjusted = list(blocks)
    audits: list[dict[str, Any]] = []
    for block_index in sorted({int(value) for value in overflow_indices}):
        if block_index not in {0, len(adjusted) - 1}:
            continue
        block = adjusted[block_index]
        required_block_ms = max(
            1,
            int(required_windows_ms.get(block_index, block.duration_ms)),
        )
        deficit_ms = max(0, required_block_ms - block.duration_ms)
        base_audit: dict[str, Any] = {
            "method": "measured_edge_neighbour_surplus_fit",
            "solver_version": "measured-edge-neighbour-surplus-v1",
            "block_index": block_index,
            "block_id": block.block_id,
            "required_window_ms": required_block_ms,
            "original_window_ms": block.duration_ms,
            "deficit_ms": deficit_ms,
            "maximum_boundary_shift_ms": max_boundary_shift_ms,
            "text_immutable": True,
            "speaker_order_preserved": True,
            "cross_speaker_overlap": False,
            "authoritative_translation_changed": False,
            "ai_called": False,
        }
        if deficit_ms <= 0:
            audits.append({**base_audit, "applied": False, "reason": "no_deficit"})
            continue
        if len(adjusted) < 2:
            audits.append(
                {**base_audit, "applied": False, "reason": "missing_neighbour"}
            )
            continue

        if block_index == len(adjusted) - 1:
            neighbour_index = block_index - 1
            neighbour = adjusted[neighbour_index]
            source_gap_ms = max(0, block.start_ms - neighbour.end_ms)
            desired_start_ms = block.end_ms - required_block_ms
            adjusted_neighbour = replace(
                neighbour,
                end_ms=min(neighbour.end_ms, desired_start_ms),
            )
            adjusted_block = replace(block, start_ms=desired_start_ms)
            boundary_shift_ms = max(
                0, neighbour.end_ms - adjusted_neighbour.end_ms
            )
            edge = "end"
        else:
            neighbour_index = 1
            neighbour = adjusted[neighbour_index]
            source_gap_ms = max(0, neighbour.start_ms - block.end_ms)
            desired_end_ms = block.start_ms + required_block_ms
            adjusted_block = replace(block, end_ms=desired_end_ms)
            adjusted_neighbour = replace(
                neighbour,
                start_ms=max(neighbour.start_ms, desired_end_ms),
            )
            boundary_shift_ms = max(
                0, adjusted_neighbour.start_ms - neighbour.start_ms
            )
            edge = "start"

        required_neighbour_ms = max(
            1,
            int(
                required_windows_ms.get(
                    neighbour_index,
                    neighbour.duration_ms,
                )
            ),
        )
        neighbour_surplus_ms = max(
            0, neighbour.duration_ms - required_neighbour_ms
        )
        invariant_failures: list[str] = []
        if boundary_shift_ms > max_boundary_shift_ms:
            invariant_failures.append("boundary_shift_exceeds_limit")
        if adjusted_block.duration_ms < required_block_ms:
            invariant_failures.append("edge_block_below_required_window")
        if adjusted_neighbour.duration_ms < required_neighbour_ms:
            invariant_failures.append("neighbour_below_required_window")
        if block_index == len(adjusted) - 1:
            if adjusted_neighbour.end_ms > adjusted_block.start_ms:
                invariant_failures.append("previous_edge_overlap")
            if adjusted_block.end_ms != block.end_ms:
                invariant_failures.append("clip_end_changed")
        else:
            if adjusted_block.end_ms > adjusted_neighbour.start_ms:
                invariant_failures.append("following_edge_overlap")
            if adjusted_block.start_ms != block.start_ms:
                invariant_failures.append("clip_start_changed")

        audit = {
            **base_audit,
            "edge": edge,
            "neighbour_index": neighbour_index,
            "neighbour_block_id": neighbour.block_id,
            "required_neighbour_window_ms": required_neighbour_ms,
            "neighbour_original_window_ms": neighbour.duration_ms,
            "neighbour_measured_surplus_ms": neighbour_surplus_ms,
            "source_silence_consumed_ms": min(deficit_ms, source_gap_ms),
            "neighbour_boundary_contribution_ms": boundary_shift_ms,
            "boundary_shift_ms": boundary_shift_ms,
            "affected_block_indices": sorted([block_index, neighbour_index]),
        }
        if invariant_failures:
            audits.append(
                {
                    **audit,
                    "applied": False,
                    "reason": "edge_surplus_candidate_rejected",
                    "invariant_failures": invariant_failures,
                }
            )
            continue

        adjusted[block_index] = adjusted_block
        adjusted[neighbour_index] = adjusted_neighbour
        audits.append(
            {
                **audit,
                "applied": True,
                "reason": "measured_neighbour_surplus_resolved_edge_overflow",
                "adjusted_windows_ms": {
                    adjusted_block.block_id: adjusted_block.duration_ms,
                    adjusted_neighbour.block_id: adjusted_neighbour.duration_ms,
                },
                "deficit_resolved_ms": deficit_ms,
                "human_review_required": False,
                "downstream_overflow_prevented": True,
            }
        )

    return adjusted, audits

def rebalance_sandwiched_interjection_timeline(
    blocks: Sequence[SpeechBlock],
    required_windows_ms: Mapping[int, int],
    *,
    target_indices: Iterable[int] | None = None,
    max_total_overlap_ms: int = 0,
    max_boundary_overlap_ms: int = 0,
    max_source_gap_ms: int = 250,
    manual_overlap_resolutions: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[list[SpeechBlock], list[dict[str, Any]]]:
    """Fit measured A-B-A interjections with bounded source-like overlap.

    Commission and courtroom speech frequently contains a short speaker-B
    interjection while speaker A is still finishing and then immediately
    resumes.  Treating every cross-speaker boundary as strictly serial can make
    an otherwise faithful dub impossible even though the source interaction is
    naturally overlapping.

    The primary path remains collision-free.  When the three Azure-measured
    minima exceed the fixed triplet span, this solver may introduce only the
    exact overlap needed to close that shortfall, and only when the source
    timeline has an A-B-A pattern with close handoffs.  The surrounding A turns
    may never overlap each other, speaker order and text remain immutable, and
    every overlap is bounded and audited.
    """

    if max_total_overlap_ms < 0:
        raise ValueError("max_total_overlap_ms must be non-negative")
    if max_boundary_overlap_ms < 0:
        raise ValueError("max_boundary_overlap_ms must be non-negative")
    if max_source_gap_ms < 0:
        raise ValueError("max_source_gap_ms must be non-negative")

    solver_version = (
        "measured_triplet_span_solver_v3_controlled_overlap"
        if max_total_overlap_ms > 0
        else "measured_triplet_span_solver_v2"
    )
    adjusted = list(blocks)
    audits: list[dict[str, Any]] = []
    targets = (
        {int(value) for value in target_indices}
        if target_indices is not None
        else None
    )

    for block_index in range(1, len(adjusted) - 1):
        if targets is not None and block_index not in targets:
            continue
        source_previous = blocks[block_index - 1]
        source_block = blocks[block_index]
        source_following = blocks[block_index + 1]
        previous = adjusted[block_index - 1]
        block = adjusted[block_index]
        following = adjusted[block_index + 1]
        if not (
            source_previous.speaker_id == source_following.speaker_id
            and source_previous.speaker_id != source_block.speaker_id
            and source_block.duration_ms
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
        source_left_gap_ms = source_block.start_ms - source_previous.end_ms
        source_right_gap_ms = source_following.start_ms - source_block.end_ms
        transcript_timing_available = all(
            value.source_timing_evidence == "azure_word_timestamps"
            for value in (source_previous, source_block, source_following)
        )
        transcript_interjection_roles = [
            dict(value)
            for value in source_block.source_interactions
            if str(value.get("kind") or "")
            == "sandwiched_interjection"
            and str(value.get("role") or "") == "interjector"
            and bool(value.get("source_authorized"))
        ]
        transcript_authorized_overlap_ms = max(
            (
                int(value.get("overlap_ms") or 0)
                for value in transcript_interjection_roles
            ),
            default=0,
        )
        # New timed transcripts may use only overlap the source actually
        # contains. Old transcripts retain the established close-handoff
        # heuristic, which keeps this change backward compatible.
        source_interjection_like = bool(
            transcript_interjection_roles
            if transcript_timing_available
            else (
                source_left_gap_ms <= max_source_gap_ms
                and source_right_gap_ms <= max_source_gap_ms
            )
        )
        manual_resolution = dict(
            (manual_overlap_resolutions or {}).get(
                _source_block_id(source_block.block_id), {}
            )
        )
        manual_overlap = str(manual_resolution.get("strategy") or "") == "allow_overlap"
        local_overlap_before_ms = max_boundary_overlap_ms
        local_overlap_after_ms = max_boundary_overlap_ms
        local_total_overlap_ms = max_total_overlap_ms
        if transcript_timing_available:
            local_total_overlap_ms = min(
                local_total_overlap_ms,
                transcript_authorized_overlap_ms,
            )
            local_overlap_before_ms = min(
                local_overlap_before_ms,
                transcript_authorized_overlap_ms,
            )
            local_overlap_after_ms = min(
                local_overlap_after_ms,
                transcript_authorized_overlap_ms,
            )
        if manual_overlap:
            local_overlap_before_ms = max(
                local_overlap_before_ms,
                int(manual_resolution.get("overlap_before_ms") or 0),
            )
            local_overlap_after_ms = max(
                local_overlap_after_ms,
                int(manual_resolution.get("overlap_after_ms") or 0),
            )
            local_total_overlap_ms = max(
                local_total_overlap_ms,
                int(manual_resolution.get("max_total_overlap_ms") or 0),
                local_overlap_before_ms + local_overlap_after_ms,
            )

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
            "source_left_gap_ms": source_left_gap_ms,
            "source_right_gap_ms": source_right_gap_ms,
            "source_interjection_like": source_interjection_like,
            "transcript_timing_available": transcript_timing_available,
            "transcript_interjection_roles": transcript_interjection_roles,
            "transcript_authorized_overlap_ms": (
                transcript_authorized_overlap_ms
            ),
            "overlap_eligibility": (
                "azure_word_timestamps"
                if transcript_timing_available and source_interjection_like
                else "close_source_handoffs"
                if source_interjection_like
                else "not_source_authorized"
            ),
            "max_total_overlap_ms": local_total_overlap_ms,
            "max_boundary_overlap_before_ms": local_overlap_before_ms,
            "max_boundary_overlap_after_ms": local_overlap_after_ms,
            "manual_resolution": manual_resolution or None,
            "text_immutable": True,
            "speaker_order_preserved": True,
        }

        if safe_slack_ms >= 0:
            earliest_middle_start_ms = cluster_start_ms + required_previous_ms
            latest_middle_start_ms = (
                cluster_end_ms - required_following_ms - required_block_ms
            )
            middle_start_ms = min(
                max(block.start_ms, earliest_middle_start_ms),
                latest_middle_start_ms,
            )
            middle_end_ms = middle_start_ms + required_block_ms
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
            overlap_before_ms = 0
            overlap_after_ms = 0
            overlap_policy = "strict_non_overlap"
        else:
            required_overlap_ms = -safe_slack_ms
            if local_total_overlap_ms <= 0:
                audits.append(
                    {
                        **base_audit,
                        "applied": False,
                        "reason": "insufficient_measured_triplet_span",
                        "shortfall_ms": required_overlap_ms,
                        "cross_speaker_overlap": False,
                    }
                )
                continue
            # A short A-B-A speaker pattern is itself sufficient interjection
            # evidence for commission/courtroom material.  Diarization and VAD
            # often insert artificial positive gaps, so source gap size remains
            # audit evidence but is not a hard eligibility gate.
            if (
                required_overlap_ms > local_total_overlap_ms
                or required_overlap_ms
                > local_overlap_before_ms + local_overlap_after_ms
            ):
                audits.append(
                    {
                        **base_audit,
                        "applied": False,
                        "reason": "interjection_overlap_budget_exceeded",
                        "shortfall_ms": required_overlap_ms,
                        "cross_speaker_overlap": False,
                    }
                )
                continue

            # Use the smallest windows already proven safe by Azure.  This makes
            # the introduced overlap exactly equal to the measured shortfall.
            previous_end_ms = cluster_start_ms + required_previous_ms
            following_start_ms = cluster_end_ms - required_following_ms
            if previous_end_ms > following_start_ms:
                audits.append(
                    {
                        **base_audit,
                        "applied": False,
                        "reason": "surrounding_same_speaker_turns_would_overlap",
                        "shortfall_ms": required_overlap_ms,
                        "cross_speaker_overlap": False,
                    }
                )
                continue

            earliest_middle_start_ms = max(
                previous_end_ms - local_overlap_before_ms,
                following_start_ms - required_block_ms,
            )
            latest_middle_start_ms = min(
                previous_end_ms,
                following_start_ms + local_overlap_after_ms - required_block_ms,
            )
            if earliest_middle_start_ms > latest_middle_start_ms:
                audits.append(
                    {
                        **base_audit,
                        "applied": False,
                        "reason": "bounded_overlap_placement_unavailable",
                        "shortfall_ms": required_overlap_ms,
                        "cross_speaker_overlap": False,
                    }
                )
                continue

            middle_start_ms = min(
                max(block.start_ms, earliest_middle_start_ms),
                latest_middle_start_ms,
            )
            middle_end_ms = middle_start_ms + required_block_ms
            overlap_before_ms = max(0, previous_end_ms - middle_start_ms)
            overlap_after_ms = max(0, middle_end_ms - following_start_ms)
            overlap_policy = (
                "human_approved_commission_interjection_overlap"
                if manual_overlap
                else "source_like_sandwiched_interjection"
            )

        candidate_previous = replace(previous, end_ms=previous_end_ms)
        candidate_block = replace(
            block,
            start_ms=middle_start_ms,
            end_ms=middle_end_ms,
        )
        candidate_following = replace(following, start_ms=following_start_ms)
        total_overlap_ms = overlap_before_ms + overlap_after_ms

        invariant_failures: list[str] = []
        if candidate_previous.duration_ms < required_previous_ms:
            invariant_failures.append("previous_below_required_window")
        if candidate_block.duration_ms < required_block_ms:
            invariant_failures.append("middle_below_required_window")
        if candidate_following.duration_ms < required_following_ms:
            invariant_failures.append("following_below_required_window")
        if candidate_previous.start_ms != cluster_start_ms:
            invariant_failures.append("cluster_start_changed")
        if candidate_following.end_ms != cluster_end_ms:
            invariant_failures.append("cluster_end_changed")
        if candidate_previous.end_ms > candidate_following.start_ms:
            invariant_failures.append("surrounding_same_speaker_overlap")
        if candidate_block.start_ms < candidate_previous.start_ms:
            invariant_failures.append("middle_starts_before_triplet")
        if candidate_block.end_ms > candidate_following.end_ms:
            invariant_failures.append("middle_ends_after_triplet")
        if overlap_before_ms > local_overlap_before_ms:
            invariant_failures.append("previous_middle_overlap_exceeds_budget")
        if overlap_after_ms > local_overlap_after_ms:
            invariant_failures.append("middle_following_overlap_exceeds_budget")
        if total_overlap_ms > local_total_overlap_ms:
            invariant_failures.append("total_overlap_exceeds_budget")
        if safe_slack_ms >= 0 and total_overlap_ms:
            invariant_failures.append("unnecessary_overlap_introduced")
        if safe_slack_ms < 0 and total_overlap_ms != -safe_slack_ms:
            invariant_failures.append("overlap_does_not_equal_measured_shortfall")

        if invariant_failures:
            audits.append(
                {
                    **base_audit,
                    "applied": False,
                    "reason": "triplet_span_candidate_rejected",
                    "invariant_failures": invariant_failures,
                    "cross_speaker_overlap": False,
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
                "cross_speaker_overlap": bool(total_overlap_ms),
                "overlap_policy": overlap_policy,
                "overlap_before_ms": overlap_before_ms,
                "overlap_after_ms": overlap_after_ms,
                "total_overlap_ms": total_overlap_ms,
                "human_review_required": total_overlap_ms > 500,
                "downstream_overflow_prevented": True,
            }
        )

    return adjusted, audits


def _selected_azure_measurement_ms(fitted: Mapping[str, Any]) -> int | None:
    """Return the raw selected Azure duration before any local time fitting."""

    duration_ms = int(fitted.get("duration_ms") or 0)
    if duration_ms > 0:
        return duration_ms
    boundary = fitted.get("exact_audio_boundary")
    if isinstance(boundary, Mapping):
        measured_before_ms = int(boundary.get("measured_before_ms") or 0)
        if measured_before_ms > 0:
            return measured_before_ms
    return None

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


def _punctuation_only_source_unit(item: Mapping[str, Any]) -> bool:
    """Return true for a source-bound unit containing no speakable token."""

    source_text = str(
        item.get("source_text")
        or item.get("text")
        or item.get("display")
        or item.get("lexical")
        or ""
    ).strip()
    return bool(source_text) and all(
        unicodedata.category(character).startswith(("P", "Z"))
        for character in source_text
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
    boundary_normalizations: list[dict[str, Any]] | None = None,
) -> list[ApprovedSegment]:
    if translation.get("production_authorized") is False:
        raise DirectDubError("Translation is not authorized for production")
    item_kind, raw_segments = _translation_items(translation)

    final_speakable_item_index = next(
        (
            index
            for index in range(len(raw_segments) - 1, -1, -1)
            if isinstance(raw_segments[index], Mapping)
            and not _punctuation_only_source_unit(raw_segments[index])
        ),
        None,
    )
    segments: list[ApprovedSegment] = []
    for index, item in enumerate(raw_segments, start=1):
        if not isinstance(item, Mapping):
            raise DirectDubError(f"Translation {item_kind[:-1]} {index} is not an object")
        # Existing installed translations may already contain an Azure
        # punctuation-only pseudo-phrase. Preserve its video/background window
        # and omit only the nonexistent dialogue instead of attempting empty
        # TTS or modifying the approved translation artifact.
        if _punctuation_only_source_unit(item):
            continue
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
        overrun_ms = max(0, end_ms - source_duration_ms)
        original_duration_ms = end_ms - start_ms
        is_final_speakable_item = index - 1 == final_speakable_item_index
        extended_final_recovery = (
            overrun_ms > ORDINARY_TRANSLATION_TAIL_DRIFT_MS
            and is_final_speakable_item
            and overrun_ms <= FINAL_SPEAKABLE_TAIL_DRIFT_MS
            and overrun_ms / original_duration_ms
            <= FINAL_SPEAKABLE_MAX_TRIM_RATIO
        )
        if (
            overrun_ms > ORDINARY_TRANSLATION_TAIL_DRIFT_MS
            and not extended_final_recovery
        ):
            raise DirectDubError(
                f"{segment_id} ends beyond the source video by {overrun_ms}ms; "
                "automatic recovery is limited to bounded drift on the final "
                "speakable unit"
            )
        # Azure/FFmpeg/container timing can differ by a few milliseconds. The
        # loader historically accepted a 250ms tolerance, but Azure word timing
        # can place the final phrase just beyond that fixed cutoff. Recover only
        # a bounded final-speech overrun (at most 500ms and 10% of the phrase)
        # and retain strict rejection for interior or materially truncated
        # units. The mixer continues to receive an exact source boundary.
        if overrun_ms and boundary_normalizations is not None:
            boundary_normalizations.append(
                {
                    "schema_version": (
                        "mathula-approved-segment-boundary-normalization-v1"
                    ),
                    "policy_version": APPROVED_SEGMENT_BOUNDARY_POLICY_VERSION,
                    "segment_id": segment_id,
                    "final_speakable_unit": is_final_speakable_item,
                    "original_start_ms": start_ms,
                    "original_end_ms": end_ms,
                    "normalized_end_ms": source_duration_ms,
                    "source_duration_ms": source_duration_ms,
                    "original_duration_ms": original_duration_ms,
                    "normalized_duration_ms": source_duration_ms - start_ms,
                    "overrun_ms": overrun_ms,
                    "trim_ratio": round(overrun_ms / original_duration_ms, 6),
                    "recovery_mode": (
                        "bounded_final_speakable_tail"
                        if extended_final_recovery
                        else "ordinary_container_tail_drift"
                    ),
                    "approved_text_changed": False,
                    "translation_artifact_changed": False,
                }
            )
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
                required_facts=tuple(
                    str(value).strip()
                    for value in item.get("required_facts", [])
                    if str(value).strip()
                ),
                optional_details=tuple(
                    str(value).strip()
                    for value in item.get("optional_details", [])
                    if str(value).strip()
                ),
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
    preserve_opening_segment: bool = False,
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
                required_facts=tuple(
                    dict.fromkeys(
                        value
                        for item in pending
                        for value in item.required_facts
                        if value
                    )
                ),
                optional_details=tuple(
                    dict.fromkeys(
                        value
                        for item in pending
                        for value in item.optional_details
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
        opening_segment_locked = (
            preserve_opening_segment
            and len(blocks) == 0
            and len(pending) == 1
            and pending[0].start_ms <= 1_000
        )
        if (
            segment.speaker_id == previous.speaker_id
            and gap_ms <= max_gap_ms
            and not opening_segment_locked
        ):
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


def _select_azure_timing_attempt(
    attempts: Sequence[AzureTTSResult],
    *,
    window_duration_ms: int,
    base_rate_percent: int,
    maximum_underfill_slowdown_percent: int = 5,
) -> AzureTTSResult:
    """Choose a measured TTS attempt without dragging speech to fill silence.

    When the base-rate delivery already fits, a large negative Azure rate
    change is not a timing repair: it is audible whole-turn slowdown.  Keep the
    natural delivery and leave legitimate transition room instead.  Faster
    attempts remain eligible when the base delivery overflows its window.
    """

    if not attempts:
        raise ValueError("At least one measured Azure TTS attempt is required")
    fitting = [
        item for item in attempts if item.duration_ms <= window_duration_ms
    ]
    base_fits = any(
        item.rate_percent == base_rate_percent for item in fitting
    )
    if base_fits:
        minimum_safe_rate = (
            base_rate_percent - maximum_underfill_slowdown_percent
        )
        cadence_safe = [
            item for item in fitting if item.rate_percent >= minimum_safe_rate
        ]
        if cadence_safe:
            fitting = cadence_safe
    if fitting:
        return min(
            fitting,
            key=lambda item: (
                window_duration_ms - item.duration_ms,
                abs(item.rate_percent - base_rate_percent),
            ),
        )
    return min(attempts, key=lambda item: item.duration_ms)


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
    "DUB_MASTERING_OVERRIDE_SCHEMA_VERSION",
    "SOURCE_PAUSE_ALIGNMENT_VERSION",
    "TRANSCRIPT_LIPSYNC_SCHEMA_VERSION",
    "TRANSCRIPT_LIPSYNC_POLICY_VERSION",
    "SEMANTIC_FIT_SCHEMA_VERSION",
    "SEMANTIC_FIT_BATCH_STRATEGY_VERSION",
    "SEMANTIC_FIT_VALIDATOR_VERSION",
    "PERFORMANCE_PLAN_VERSION",
    "WORD_ANCHORED_ISLAND_CONTRACT_VERSION",
    "MAXIMUM_INTERNAL_ISLAND_DRIFT_MS",
    "TIMING_MODE_BALANCED",
    "TIMING_MODE_SEMANTIC_FIT",
    "TIMING_MODE_PERFORMANCE_PLAN",
    "TIMING_MODES",
    "TimelineCollisionReviewRequired",
    "apply_timeline_collision_resolutions",
    "apply_dub_mastering_overrides",
    "attach_source_pause_anchors",
    "attach_transcript_lipsync_plan",
    "load_dub_mastering_overrides",
    "build_source_word_speaker_anchors",
    "build_source_pause_ssml_parts",
    "build_semantic_island_ssml_parts",
    "build_semantic_phrase_groups",
    "build_performance_regions",
    "attach_word_anchored_island_contracts",
    "calibrate_source_pause_ssml_parts",
    "DirectDubOptions",
    "enforce_source_word_speaker_alignment",
    "ProgressCallback",
    "SpeechBlock",
    "coalesce_same_speaker_segments",
    "create_direct_azure_backend",
    "estimate_required_azure_rate",
    "extend_final_speech_block_into_source_tail",
    "load_approved_segments",
    "rebalance_five_block_timeline",
    "source_pause_calibration_floors",
]
