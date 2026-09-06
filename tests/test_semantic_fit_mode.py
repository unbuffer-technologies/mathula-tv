from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from mathula_tv.ai_provider import AnthropicClaudeProvider
from mathula_tv.azure_tts import AzureTTSResult
from mathula_tv.cli import parser
from mathula_tv.direct_azure_dub import (
    attach_word_anchored_island_contracts,
    build_performance_regions,
    build_semantic_island_ssml_parts,
    build_semantic_phrase_groups,
    build_source_pause_ssml_parts,
    calibrate_source_pause_ssml_parts,
    DirectAzureDubRenderer,
    DirectDubError,
    DirectDubArtifacts,
    DirectDubOptions,
    SpeechBlock,
    SEMANTIC_FIT_SCHEMA_VERSION,
    TimingOverflowError,
    TIMING_MODE_PERFORMANCE_PLAN,
    TIMING_MODE_SEMANTIC_FIT,
    _DirectDubProgress,
    _selected_azure_measurement_ms,
    create_direct_azure_backend,
    source_pause_calibration_floors,
)
from mathula_tv.errors import ClaudeRateLimit
from mathula_tv.tts_ssml import BreakPart


def test_source_pause_never_splits_a_protected_entity() -> None:
    parts, audit = build_source_pause_ssml_parts(
        "ukhuluma i-The Big Five cartel namuhla",
        (
            {
                "source_progress": 0.5,
                "source_gap_ms": 900,
                "requested_pause_ms": 700,
                "pause_kind": "phrase_pause",
            },
        ),
        pause_budget_ms=700,
        protected_entities=("The Big Five cartel",),
    )

    assert parts
    insertion = audit["insertions"][0]
    assert insertion["target_left_text"] not in {"i-The", "Big", "Five"}
    assert insertion["source_gaps"][0][
        "protected_entity_boundary_adjusted"
    ] is True


class _MeasuredBackend:
    sample_rate = 16_000
    ssml_bounds = SimpleNamespace(
        rate_min_percent=-50,
        rate_max_percent=50,
        total_pause_max_ms=10_000,
    )

    def __init__(self, durations: dict[str, int]) -> None:
        self.durations = durations
        self.rates: list[int] = []

    def synthesize(self, request, output_path: Path, *, force: bool):
        self.rates.append(request.rate_percent)
        duration_ms = self.durations[request.text] + sum(
            part.duration_ms
            for part in request.parts
            if isinstance(part, BreakPart)
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(f"wav:{request.text}".encode())
        digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
        return AzureTTSResult(
            backend="test",
            turn_id=request.turn_id,
            speaker_id=request.speaker_id,
            voice=request.voice or "voice",
            language=request.language,
            input_text_hash=digest,
            ssml_hash=digest,
            output_path=str(output_path),
            sample_rate=self.sample_rate,
            channels=1,
            sample_width=2,
            duration_ms=duration_ms,
            preferred_duration_ms=request.preferred_duration_ms,
            maximum_duration_ms=request.maximum_duration_ms,
            rate_percent=request.rate_percent,
            attempt=1,
            sha256=digest,
            warnings=(),
            request_hash=digest,
            ssml_path=str(output_path.with_suffix(".ssml")),
            manifest_path=str(output_path.with_suffix(".json")),
            ssml_summary="test",
            idempotent_reuse=False,
        )


class _PauseOvershootBackend(_MeasuredBackend):
    def __init__(self, durations: dict[str, int], *, pause_overhead_ms: int) -> None:
        super().__init__(durations)
        self.pause_overhead_ms = pause_overhead_ms

    def synthesize(self, request, output_path: Path, *, force: bool):
        result = super().synthesize(request, output_path, force=force)
        if any(isinstance(part, BreakPart) for part in request.parts):
            return AzureTTSResult(
                **{
                    **result.__dict__,
                    "duration_ms": result.duration_ms + self.pause_overhead_ms,
                }
            )
        return result


class _SemanticProvider:
    def __init__(self) -> None:
        self.fit_calls = 0
        self.review_calls = 0

    def fit_turn_semantically(self, payload):
        self.fit_calls += 1
        unit_id = payload["unit_id"]
        return SimpleNamespace(
            data={
                "schema_version": "claude-turn-repair-v1",
                "unit": {
                    "unit_id": unit_id,
                    "faithful_translation": "Ukuhumusha okugunyaziwe",
                    "spoken_text": "Umusho olingana kahle",
                    "tts_text": "Umusho olingana kahle",
                    "language_features": [],
                    "human_review_flags": [],
                    "numbers_preserved": True,
                    "dates_preserved": True,
                    "negation_preserved": True,
                    "omitted_or_compressed_detail": [],
                },
            }
        )

    def review_semantic_fit(self, payload):
        self.review_calls += 1
        return SimpleNamespace(
            data={
                "schema_version": "claude-semantic-fit-review-v1",
                "accepted": True,
                "required_fact_ids_preserved": [
                    item["fact_id"] for item in payload["required_facts"]
                ],
                "protected_entities_preserved": payload["protected_entities"],
                "attribution_preserved": True,
                "uncertainty_preserved": True,
                "negation_preserved": True,
                "no_new_facts": True,
                "natural_spoken_target_language": True,
                "semantic_fidelity_score": 99,
                "naturalness_score": 95,
                "reason_codes": [],
            }
        )


class _SemanticBatchProvider:
    def __init__(
        self,
        candidates: dict[str, str],
        *,
        review_rate_limits: int = 0,
        retry_after_seconds: float = 0.25,
    ) -> None:
        self.candidates = candidates
        self.fit_batch_calls = 0
        self.review_batch_calls = 0
        self.review_rate_limits = review_rate_limits
        self.retry_after_seconds = retry_after_seconds
        self.sleep_calls: list[float] = []
        self.sleep = self.sleep_calls.append

    def fit_turns_semantically(self, payload):
        self.fit_batch_calls += 1
        repairs = []
        for request in payload["repairs"]:
            unit_id = request["unit_id"]
            spoken = self.candidates[unit_id]
            repairs.append(
                {
                    "unit_id": unit_id,
                    "candidate": {
                        "schema_version": "claude-turn-repair-v1",
                        "unit": {
                            "unit_id": unit_id,
                            "faithful_translation": request["current_translation"][
                                "faithful_translation"
                            ],
                            "spoken_text": spoken,
                            "tts_text": spoken,
                            "language_features": [],
                            "human_review_flags": [],
                            "numbers_preserved": True,
                            "dates_preserved": True,
                            "negation_preserved": True,
                            "omitted_or_compressed_detail": [],
                        },
                    },
                }
            )
        return SimpleNamespace(
            data={
                "schema_version": "claude-semantic-fit-batch-v1",
                "repairs": repairs,
            }
        )

    def review_semantic_fits(self, payload):
        self.review_batch_calls += 1
        if self.review_batch_calls <= self.review_rate_limits:
            raise ClaudeRateLimit(
                "test rate limit",
                details={
                    "status_code": 429,
                    "retry_after_seconds": self.retry_after_seconds,
                },
            )
        reviews = []
        for request in payload["reviews"]:
            reviews.append(
                {
                    "unit_id": request["unit_id"],
                    "review": {
                        "schema_version": "claude-semantic-fit-review-v1",
                        "accepted": True,
                        "required_fact_ids_preserved": [
                            item["fact_id"]
                            for item in request["required_facts"]
                        ],
                        "protected_entities_preserved": request[
                            "protected_entities"
                        ],
                        "attribution_preserved": True,
                        "uncertainty_preserved": True,
                        "negation_preserved": True,
                        "no_new_facts": True,
                        "natural_spoken_target_language": True,
                        "semantic_fidelity_score": 99,
                        "naturalness_score": 95,
                        "reason_codes": [],
                    },
                }
            )
        return SimpleNamespace(
            data={
                "schema_version": "claude-semantic-fit-review-batch-v1",
                "reviews": reviews,
            }
        )


class _SemanticPortfolioProvider(_SemanticBatchProvider):
    def __init__(self, portfolios: dict[str, list[str]]) -> None:
        super().__init__({})
        self.portfolios = portfolios
        self.portfolio_calls = 0
        self.requested_candidate_counts: list[int] = []

    def fit_semantic_portfolios(self, payload):
        self.portfolio_calls += 1
        portfolios = []
        for request in payload["repairs"]:
            unit_id = request["unit_id"]
            requested_count = request["candidate_count"]
            self.requested_candidate_counts.append(requested_count)
            spoken_values = self.portfolios[unit_id]
            assert len(spoken_values) >= requested_count
            candidates = []
            for index, spoken in enumerate(
                spoken_values[:requested_count],
                start=1,
            ):
                candidates.append(
                    {
                        "candidate_id": f"candidate_{index:02d}",
                        "estimated_plain_speech_ms": 1000,
                        "compression_strategy": f"strategy {index}",
                        "candidate": {
                            "schema_version": "claude-turn-repair-v1",
                            "unit": {
                                "unit_id": unit_id,
                                "faithful_translation": request[
                                    "current_translation"
                                ]["faithful_translation"],
                                "spoken_text": spoken,
                                "tts_text": spoken,
                                "language_features": [],
                                "human_review_flags": [],
                                "numbers_preserved": True,
                                "dates_preserved": True,
                                "negation_preserved": True,
                                "omitted_or_compressed_detail": [],
                            },
                        },
                    }
                )
            portfolios.append({"unit_id": unit_id, "candidates": candidates})
        return SimpleNamespace(
            data={
                "schema_version": "claude-semantic-fit-portfolio-batch-v1",
                "portfolios": portfolios,
            }
        )


class _SemanticGroupPortfolioProvider(_SemanticBatchProvider):
    def __init__(self, island_portfolios: list[list[str]]) -> None:
        super().__init__({})
        self.island_portfolios = island_portfolios
        self.portfolio_calls = 0
        self.last_payload = None

    def fit_semantic_portfolios(self, payload):
        self.portfolio_calls += 1
        self.last_payload = payload
        request = payload["repairs"][0]
        candidates = []
        for index, islands in enumerate(
            self.island_portfolios[: request["candidate_count"]],
            start=1,
        ):
            spoken = " ".join(islands)
            candidates.append(
                {
                    "candidate_id": f"candidate_{index:02d}",
                    "estimated_plain_speech_ms": 7000 - (index - 1) * 100,
                    "compression_strategy": f"phrase group strategy {index}",
                    "candidate": {
                        "schema_version": "claude-turn-repair-v1",
                        "unit": {
                            "unit_id": request["unit_id"],
                            "faithful_translation": request[
                                "current_translation"
                            ]["faithful_translation"],
                            "spoken_text": spoken,
                            "tts_text": spoken,
                            "language_features": [],
                            "human_review_flags": [],
                            "numbers_preserved": True,
                            "dates_preserved": True,
                            "negation_preserved": True,
                            "omitted_or_compressed_detail": [],
                            "delivery_hints": islands,
                        },
                    },
                }
            )
        return SimpleNamespace(
            data={
                "schema_version": "claude-semantic-fit-portfolio-batch-v1",
                "portfolios": [
                    {"unit_id": request["unit_id"], "candidates": candidates}
                ],
            }
        )


class _PerformancePlanProvider(_SemanticBatchProvider):
    def __init__(self, island_portfolios: list[list[str]]) -> None:
        super().__init__({})
        self.island_portfolios = island_portfolios
        self.portfolio_calls = 0
        self.requests: list[dict] = []
        self.batch_sizes: list[int] = []

    def fit_semantic_portfolios(self, payload):
        self.portfolio_calls += 1
        repairs = payload["repairs"]
        self.batch_sizes.append(len(repairs))
        self.requests.extend(repairs)
        portfolios = []
        for request in repairs:
            source_islands = request["performance_plan"]["source_islands"]
            candidates = []
            for index, template in enumerate(
                self.island_portfolios[: request["candidate_count"]],
                start=1,
            ):
                islands = (
                    template
                    if len(template) == len(source_islands)
                    else [
                        f"Ukulethwa {source_island['island_id']}"
                        for source_island in source_islands
                    ]
                )
                deliveries = [
                    {
                        "island_id": source_island["island_id"],
                        "spoken_text": spoken,
                        "tts_text": spoken,
                    }
                    for source_island, spoken in zip(
                        source_islands, islands, strict=True
                    )
                ]
                spoken_text = " ".join(islands)
                candidates.append(
                    {
                        "candidate_id": f"candidate_{index:02d}",
                        "estimated_plain_speech_ms": 7000,
                        "compression_strategy": (
                            "semantic beats across speech islands"
                        ),
                        "candidate": {
                            "schema_version": "claude-turn-repair-v1",
                            "unit": {
                                "unit_id": request["unit_id"],
                                "faithful_translation": request[
                                    "current_translation"
                                ]["faithful_translation"],
                                "spoken_text": spoken_text,
                                "tts_text": spoken_text,
                                "language_features": [],
                                "human_review_flags": [],
                                "numbers_preserved": True,
                                "dates_preserved": True,
                                "negation_preserved": True,
                                "omitted_or_compressed_detail": [],
                                "performance_beats": [
                                    {
                                        "beat_id": "beat_001",
                                        "meaning": "complete region meaning",
                                        "island_deliveries": deliveries,
                                    }
                                ],
                            },
                        },
                    }
                )
            portfolios.append(
                {"unit_id": request["unit_id"], "candidates": candidates}
            )
        return SimpleNamespace(
            data={
                "schema_version": "claude-semantic-fit-portfolio-batch-v1",
                "portfolios": portfolios,
            }
        )


class _AdaptivePerformancePlanProvider(_PerformancePlanProvider):
    """Return one new measured revision on each controller round."""

    def fit_semantic_portfolios(self, payload):
        template_index = min(
            self.portfolio_calls,
            len(self.island_portfolios) - 1,
        )
        templates = self.island_portfolios
        self.island_portfolios = [templates[template_index]]
        try:
            return super().fit_semantic_portfolios(payload)
        finally:
            self.island_portfolios = templates

def _block() -> SpeechBlock:
    return SpeechBlock(
        block_id="block_0001_variant_natural",
        speaker_id="SPEAKER_00",
        start_ms=0,
        end_ms=2000,
        segment_ids=("unit_0001",),
        source_text="The source sentence",
        translated_text="Ukuhumusha okugunyaziwe",
        tts_text="Ukuhumusha okugunyaziwe",
        required_facts=("The source sentence remains true",),
    )


def _performance_source_blocks() -> tuple[SpeechBlock, SpeechBlock]:
    first = SpeechBlock(
        block_id="block_0006",
        speaker_id="SPEAKER_00",
        start_ms=0,
        end_ms=6500,
        segment_ids=("unit_0009", "unit_0010"),
        source_text="A complete first statement.",
        translated_text="Isitatimende sokuqala esiphelele.",
        tts_text="Isitatimende sokuqala esiphelele.",
        required_facts=("first statement",),
        source_speech_start_ms=0,
        source_speech_end_ms=6000,
        source_speech_islands=(
            {"island_id": "island-a", "start_ms": 0, "end_ms": 3000, "text": "A complete"},
            {"island_id": "island-b", "start_ms": 4000, "end_ms": 6000, "text": "first statement."},
        ),
        source_timing_evidence="azure_word_timestamps",
    )
    second = SpeechBlock(
        block_id="block_0007",
        speaker_id="SPEAKER_00",
        start_ms=7500,
        end_ms=10000,
        segment_ids=("unit_0011",),
        source_text="A separate second statement.",
        translated_text="Isitatimende sesibili esihlukile.",
        tts_text="Isitatimende sesibili esihlukile.",
        required_facts=("second statement",),
        source_speech_start_ms=8000,
        source_speech_end_ms=9500,
        source_speech_islands=(
            {"island_id": "island-c", "start_ms": 8000, "end_ms": 9500, "text": "A separate second statement."},
        ),
        source_timing_evidence="azure_word_timestamps",
    )
    return first, second


def test_semantic_fit_cli_is_opt_in() -> None:
    balanced = parser().parse_args(["dub-azure", "job-1"])
    semantic = parser().parse_args(
        [
            "dub-azure",
            "job-1",
            "--timing-mode",
            "semantic-fit",
            "--semantic-fit-max-attempts",
            "9",
        ]
    )
    performance = parser().parse_args(
        ["dub-azure", "job-1", "--timing-mode", "performance-plan"]
    )

    assert balanced.timing_mode == "balanced"
    assert semantic.timing_mode == "semantic-fit"
    assert semantic.semantic_fit_max_attempts == 9
    assert performance.timing_mode == "performance-plan"


def test_semantic_fit_options_reject_invalid_limits() -> None:
    backend = _MeasuredBackend({})
    with pytest.raises(ValueError, match="attempts"):
        DirectDubOptions(
            timing_mode=TIMING_MODE_SEMANTIC_FIT,
            semantic_fit_max_attempts=0,
        ).validate(backend)


def test_semantic_fit_payload_forbids_rate_and_stretch_repairs() -> None:
    payload = DirectAzureDubRenderer._semantic_fit_payload(
        _block(),
        voice="zu-ZA-ThandoNeural",
        base_rate_percent=-3,
        pitch_percent=0,
        volume_percent=0,
        attempt_number=2,
        target_minimum_ms=1650,
        target_preferred_ms=1900,
        target_maximum_ms=2000,
        duration_evidence=[{"measured_duration_ms": 2200}],
        rejected_spoken_texts=["Ukuhumusha okugunyaziwe"],
        current_spoken_text="Ukuhumusha okugunyaziwe",
    )

    assert payload["synthesis_profile"]["rate_percent"] == -3
    assert payload["synthesis_profile"]["tempo_locked"] is True
    assert payload["constraints"]["per_block_rate_change_allowed"] is False
    assert payload["constraints"]["post_synthesis_time_stretch_allowed"] is False
    assert payload["target_duration_band_ms"] == {
        "minimum": 1650,
        "preferred": 1900,
        "maximum": 2000,
    }


def test_semantic_fit_measures_one_candidate_at_locked_rate_and_reviews_it(
    tmp_path: Path,
) -> None:
    backend = _MeasuredBackend(
        {
            "Ukuhumusha okugunyaziwe": 1400,
            "Umusho olingana kahle": 1800,
        }
    )
    provider = _SemanticProvider()
    settings = SimpleNamespace(work_dir=tmp_path)
    renderer = DirectAzureDubRenderer(
        settings,
        backend,
        timing_repair_provider_factory=lambda: provider,
    )
    artifacts = DirectDubArtifacts(tmp_path / "jobs" / "job-1")

    selected_block, fitted, attempts = renderer._synthesize_semantic_fit(
        _block(),
        artifacts,
        voice="zu-ZA-ThandoNeural",
        base_rate_percent=-3,
        pitch_percent=0,
        volume_percent=0,
        options=DirectDubOptions(timing_mode=TIMING_MODE_SEMANTIC_FIT),
        force=False,
        progress=_DirectDubProgress(None),
        block_number=1,
        block_total=1,
    )

    assert selected_block.translated_text == "Umusho olingana kahle"
    assert fitted["selected_azure_rate_percent"] == -3
    assert fitted["post_synthesis_time_stretch_ratio"] == 1.0
    assert fitted["speaker_tempo_locked"] is True
    assert fitted["semantic_fit"]["selected_attempt"] == 1
    assert attempts[-1]["status"] == "selected"
    assert backend.rates == [-3, -3]
    assert provider.fit_calls == 1
    assert provider.review_calls == 1
    assert artifacts.semantic_fit.is_file()


def test_semantic_fit_batches_all_unresolved_blocks_in_one_ai_round(
    tmp_path: Path,
) -> None:
    first = _block()
    second = SpeechBlock(
        block_id="block_0002_variant_natural",
        speaker_id="SPEAKER_00",
        start_ms=2000,
        end_ms=4000,
        segment_ids=("unit_0002",),
        source_text="A second source sentence",
        translated_text="Umusho wesibili wemvelo",
        tts_text="Umusho wesibili wemvelo",
        required_facts=("The second sentence remains true",),
    )
    backend = _MeasuredBackend(
        {
            first.translated_text: 2400,
            second.translated_text: 2300,
            "Umusho wokuqala ofushane": 1800,
            "Umusho wesibili ofushane": 1850,
        }
    )
    provider = _SemanticBatchProvider(
        {
            "block_0001": "Umusho wokuqala ofushane",
            "block_0002": "Umusho wesibili ofushane",
        }
    )
    renderer = DirectAzureDubRenderer(
        SimpleNamespace(work_dir=tmp_path),
        backend,
        timing_repair_provider_factory=lambda: provider,
    )
    blocks = (first, second)
    results = renderer._synthesize_semantic_fit_batch(
        blocks,
        DirectDubArtifacts(tmp_path / "jobs" / "job-1"),
        synthesis_profiles={
            0: ("zu-ZA-ThandoNeural", 0, 0, 0),
            1: ("zu-ZA-ThandoNeural", 0, 0, 0),
        },
        options=DirectDubOptions(timing_mode=TIMING_MODE_SEMANTIC_FIT),
        force=False,
        progress=_DirectDubProgress(None),
    )

    assert provider.fit_batch_calls == 1
    assert provider.review_batch_calls == 1
    assert results[0][0].translated_text == "Umusho wokuqala ofushane"
    assert results[1][0].translated_text == "Umusho wesibili ofushane"
    assert all(
        result[1]["post_synthesis_time_stretch_ratio"] == 1.0
        for result in results.values()
    )
    assert backend.rates == [0, 0, 0, 0]


def test_semantic_fit_single_hard_block_uses_three_candidate_feedback_round(
    tmp_path: Path,
) -> None:
    block = _block()
    candidates = [
        "indlela ende kakhulu",
        "indlela ethembekile elinganayo",
        "indlela yesithathu",
        "indlela yesine",
        "indlela yesihlanu",
        "indlela yesithupha",
    ]
    backend = _MeasuredBackend(
        {
            block.translated_text: 2500,
            candidates[0]: 2300,
            candidates[1]: 1850,
            candidates[2]: 1800,
            candidates[3]: 1750,
            candidates[4]: 1700,
            candidates[5]: 1650,
        }
    )
    provider = _SemanticPortfolioProvider({"block_0001": candidates})
    renderer = DirectAzureDubRenderer(
        SimpleNamespace(work_dir=tmp_path),
        backend,
        timing_repair_provider_factory=lambda: provider,
    )
    artifacts = DirectDubArtifacts(tmp_path / "jobs" / "job-1")

    results = renderer._synthesize_semantic_fit_batch(
        (block,),
        artifacts,
        synthesis_profiles={0: ("zu-ZA-ThandoNeural", 0, 0, 0)},
        options=DirectDubOptions(timing_mode=TIMING_MODE_SEMANTIC_FIT),
        force=False,
        progress=_DirectDubProgress(None),
    )

    assert provider.portfolio_calls == 1
    assert provider.fit_batch_calls == 0
    assert provider.review_batch_calls == 1
    assert results[0][0].translated_text == candidates[1]
    checkpoint = json.loads(artifacts.semantic_fit.read_text(encoding="utf-8"))
    attempts = checkpoint["cases"]["block_0001"]["attempts"]
    assert attempts[0]["portfolio_candidate_id"] == "candidate_01"
    assert len(attempts) == 3
    assert attempts[-1]["portfolio_candidate_id"] == "candidate_03"


def test_semantic_fit_required_facts_ignore_bare_complementizer() -> None:
    block = SpeechBlock(
        block_id="block_0006_variant_natural",
        speaker_id="SPEAKER_00",
        start_ms=0,
        end_ms=6000,
        segment_ids=("unit_0009", "unit_0010", "unit_0011"),
        source_text="Patriots tonight, I say that there are many things I do not know. But",
        translated_text="MaPatriots ebusuku, ngithi kuningi engingakwazi. Kodwa",
        tts_text="MaPatriots ebusuku, ngithi kuningi engingakwazi. Kodwa",
        required_facts=(
            "Nina maPatriots namuhla ebusuku",
            "Zikhona izinto eziningi engingazazi",
            "ukuthi",
            "Kodwa",
        ),
        source_pause_anchors=(
            {
                "source_progress": 0.611,
                "requested_pause_ms": 1380,
                "pause_kind": "island_boundary",
            },
        ),
    )

    facts = DirectAzureDubRenderer._semantic_fit_required_facts(block)
    payload = DirectAzureDubRenderer._semantic_fit_payload(
        block,
        voice="zu-ZA-ThembaNeural",
        base_rate_percent=0,
        pitch_percent=0,
        volume_percent=0,
        attempt_number=1,
        target_minimum_ms=5260,
        target_preferred_ms=5760,
        target_maximum_ms=6260,
        duration_evidence=[{"status": "overflow", "measured_duration_ms": 6292}],
        rejected_spoken_texts=[block.translated_text],
        current_spoken_text=block.translated_text,
    )

    assert [item["text"] for item in facts] == [
        "Nina maPatriots namuhla ebusuku",
        "Zikhona izinto eziningi engingazazi",
        "Kodwa",
    ]
    assert [item["fact_id"] for item in facts] == [
        "fact_001",
        "fact_002",
        "fact_003",
    ]
    assert payload["acoustic_timing_budget"] == {
        "mandatory_island_pause_ms": 1380,
        "preferred_plain_speech_ms": 4380,
        "maximum_plain_speech_ms": 4880,
        "inserted_phrase_pauses_remain_flexible": True,
        "island_pause_floor_may_not_be_reduced": True,
    }
    assert "ukuthi" not in payload["constraints"][
        "critical_required_facts_must_be_preserved"
    ]


def test_semantic_fit_payload_uses_measured_plain_speech_frontier() -> None:
    block = SpeechBlock(
        block_id="block_0006_variant_natural",
        speaker_id="SPEAKER_00",
        start_ms=0,
        end_ms=6260,
        segment_ids=("unit_0009",),
        source_text="Patriots tonight, there are many things I do not know. But",
        translated_text=(
            "Nina maPatriots namuhla ebusuku, kuningi engingazazi. Kodwa"
        ),
        tts_text=(
            "Nina maPatriots namuhla ebusuku, kuningi engingazazi. Kodwa"
        ),
        required_facts=("Patriots tonight", "many things I do not know", "But"),
        source_pause_anchors=(
            {
                "source_progress": 0.55,
                "requested_pause_ms": 1380,
                "pause_kind": "island_boundary",
            },
        ),
    )
    frontier_text = "Nina maPatriots namuhla ebusuk', kuningi engingazazi. Kodwa"

    payload = DirectAzureDubRenderer._semantic_fit_payload(
        block,
        voice="zu-ZA-ThembaNeural",
        base_rate_percent=0,
        pitch_percent=0,
        volume_percent=0,
        attempt_number=2,
        target_minimum_ms=5260,
        target_preferred_ms=5760,
        target_maximum_ms=6260,
        duration_evidence=[
            {
                "attempt": 1,
                "status": "overflow",
                "spoken_text": frontier_text,
                "measured_duration_ms": 6630,
                "plain_speech_duration_ms": 5362,
                "effective_inserted_pause_ms": 1268,
                "estimated_plain_speech_ms": 4520,
            }
        ],
        rejected_spoken_texts=[frontier_text],
        current_spoken_text=frontier_text,
    )

    frontier = payload["measured_acoustic_frontier"]
    assert frontier["azure_plain_speech_ms"] == 5362
    assert frontier["maximum_plain_speech_ms"] == 4880
    assert frontier["required_additional_reduction_ms"] == 482
    assert frontier["provider_estimate_understatement_ms"] == 842
    assert frontier["measurement_is_authoritative"] is True
    assert frontier["advisory_maximum_spoken_characters"] < len(frontier_text)


def test_semantic_fit_payload_expands_after_latest_underfill() -> None:
    block = _block()
    payload = DirectAzureDubRenderer._semantic_fit_payload(
        block,
        voice="zu-ZA-ThembaNeural",
        base_rate_percent=0,
        pitch_percent=0,
        volume_percent=0,
        attempt_number=3,
        target_minimum_ms=23380,
        target_preferred_ms=23880,
        target_maximum_ms=24380,
        duration_evidence=[
            {
                "attempt": 1,
                "status": "overflow",
                "spoken_text": "old long delivery",
                "measured_duration_ms": 27000,
            },
            {
                "attempt": 2,
                "status": "underfill",
                "spoken_text": "latest short delivery",
                "measured_duration_ms": 14438,
            },
        ],
        rejected_spoken_texts=["old long delivery", "latest short delivery"],
        current_spoken_text="latest short delivery",
    )

    correction = payload["acoustic_correction"]
    assert correction["direction"] == "expand"
    assert correction["azure_measured_duration_ms"] == 14438
    assert correction["required_change_to_enter_band_ms"] == 8942
    assert correction["preferred_change_ms"] == 9442
    assert correction["target_duration_ratio"] == 1.654
    assert correction["tempo_change_allowed"] is False


def test_semantic_review_contradiction_becomes_rejection_not_job_failure() -> None:
    payload = {
        "schema_version": SEMANTIC_FIT_SCHEMA_VERSION,
        "operation": "semantic_fit_review_batch",
        "reviews": [
            {
                "unit_id": "block_0001",
                "required_facts": [{"fact_id": "fact_0001", "text": "fact"}],
                "protected_entities": ["McKenzie"],
            }
        ],
    }
    raw_review = {
        "schema_version": "claude-semantic-fit-review-batch-v1",
        "reviews": [
            {
                "unit_id": "block_0001",
                "review": {
                    "schema_version": "claude-semantic-fit-review-v1",
                    "accepted": True,
                    "required_fact_ids_preserved": ["fact_0001"],
                    "protected_entities_preserved": ["McKenzie"],
                    "attribution_preserved": True,
                    "uncertainty_preserved": True,
                    "negation_preserved": True,
                    "no_new_facts": True,
                    "natural_spoken_target_language": True,
                    "semantic_fidelity_score": 99,
                    "naturalness_score": 95,
                    "reason_codes": ["minor_wording_concern"],
                },
            }
        ],
    }
    provider = object.__new__(AnthropicClaudeProvider)
    provider.complete_structured = lambda request: request
    structured_request = provider.review_semantic_fits(payload)

    normalised = structured_request.normalizer(raw_review)
    structured_request.validator(normalised)
    review = normalised["reviews"][0]["review"]
    assert review["accepted"] is False
    assert review["reason_codes"] == [
        "minor_wording_concern",
        "provider_returned_rejection_reasons",
    ]
    assert DirectAzureDubRenderer._semantic_review_passed(review) is False

    for field, bad_value, expected_reason in (
        (
            "required_fact_ids_preserved",
            [],
            "required_facts_not_fully_preserved",
        ),
        (
            "protected_entities_preserved",
            [],
            "protected_entities_not_fully_preserved",
        ),
        ("attribution_preserved", False, "failed_attribution_preserved"),
        (
            "semantic_fidelity_score",
            90,
            "semantic_fidelity_below_threshold",
        ),
        ("naturalness_score", "invalid", "naturalness_below_threshold"),
    ):
        variant = json.loads(json.dumps(raw_review))
        raw = variant["reviews"][0]["review"]
        raw["reason_codes"] = []
        raw[field] = bad_value
        rejected = structured_request.normalizer(variant)
        structured_request.validator(rejected)
        rejected_review = rejected["reviews"][0]["review"]
        assert rejected_review["accepted"] is False
        assert expected_reason in rejected_review["reason_codes"]

    recoverable_shape = json.loads(json.dumps(raw_review))
    recoverable = recoverable_shape["reviews"][0]["review"]
    recoverable["reason_codes"] = None
    recoverable["required_fact_ids_preserved"] = "fact_0001"
    recoverable["protected_entities_preserved"] = "McKenzie"
    recoverable["semantic_fidelity_score"] = "99"
    recoverable["naturalness_score"] = "95"
    accepted = structured_request.normalizer(recoverable_shape)
    structured_request.validator(accepted)
    assert accepted["reviews"][0]["review"]["accepted"] is True
    assert accepted["reviews"][0]["review"]["reason_codes"] == []


def test_d724_observed_region_measurements_drive_both_directions() -> None:
    # Measurements captured from job d724a4c947404afeae1d748363b41827.
    cases = (
        (30015, 27940, 28440, 28940, "contract"),
        (27728, 24140, 24640, 25140, "contract"),
        (14438, 23380, 23880, 24380, "expand"),
        (22760, 21570, 22070, 22570, "contract"),
        (10232, 9740, 10240, 10740, "maintain"),
        (4515, 4420, 4920, 5420, "maintain"),
        (27700, 22610, 23110, 23610, "contract"),
        (19340, 20062, 20312, 20562, "expand"),
    )
    block = _block()
    for measured, minimum, preferred, maximum, expected in cases:
        payload = DirectAzureDubRenderer._semantic_fit_payload(
            block,
            voice="zu-ZA-ThembaNeural",
            base_rate_percent=0,
            pitch_percent=0,
            volume_percent=0,
            attempt_number=1,
            target_minimum_ms=minimum,
            target_preferred_ms=preferred,
            target_maximum_ms=maximum,
            duration_evidence=[
                {
                    "status": "measured",
                    "spoken_text": f"delivery {measured}",
                    "measured_duration_ms": measured,
                }
            ],
            rejected_spoken_texts=[],
            current_spoken_text=f"delivery {measured}",
        )
        assert payload["acoustic_correction"]["direction"] == expected

    region_five = [
        {"target_duration_ms": 760, "measured_duration_ms": 1762},
        {"target_duration_ms": 1080, "measured_duration_ms": 1788},
        {"target_duration_ms": 6080, "measured_duration_ms": 5762},
    ]
    alignment = DirectAzureDubRenderer._performance_island_alignment_summary(
        region_five
    )
    assert alignment["whole_region_timing_remains_hard_gate"] is True
    assert alignment["maximum_absolute_cumulative_drift_ms"] == 1180


def test_semantic_phrase_group_joins_dangling_same_speaker_connector() -> None:
    first = SpeechBlock(
        block_id="block_0006",
        speaker_id="SPEAKER_00",
        start_ms=63000,
        end_ms=69530,
        segment_ids=("unit_0009", "unit_0010", "unit_0011"),
        source_text=(
            "They seen Patriots tonight, I want to say to you that there are "
            "many things I do not know. But"
        ),
        translated_text=(
            "Nina maPatriots namuhla ebusuku, kuningi engingakwazi. Kodwa"
        ),
        tts_text=(
            "Nina maPatriots namuhla ebusuku, kuningi engingakwazi. Kodwa"
        ),
        required_facts=("Patriots tonight", "many things I do not know", "But"),
        source_speech_start_ms=63270,
        source_speech_end_ms=69030,
        source_speech_islands=(
            {
                "island_id": "block_0006:island-001",
                "start_ms": 63270,
                "end_ms": 65950,
                "text": "They seen Patriots tonight, I want to say to you that",
            },
            {
                "island_id": "block_0006:island-002",
                "start_ms": 67430,
                "end_ms": 69030,
                "text": "there are many things I do not know. But",
            },
        ),
        source_timing_evidence="azure_word_timestamps",
    )
    second = SpeechBlock(
        block_id="block_0007",
        speaker_id="SPEAKER_00",
        start_ms=72450,
        end_ms=75310,
        segment_ids=("unit_0012",),
        source_text="there is one thing I can tell you for sure.",
        translated_text="Kukhona into eyodwa engiqiniseka ngayo.",
        tts_text="Kukhona into eyodwa engiqiniseka ngayo.",
        required_facts=("one thing I can tell you for sure",),
        source_speech_start_ms=72950,
        source_speech_end_ms=74830,
        source_speech_islands=(
            {
                "island_id": "block_0007:island-001",
                "start_ms": 72950,
                "end_ms": 74830,
                "text": "there is one thing I can tell you for sure.",
            },
        ),
        source_timing_evidence="azure_word_timestamps",
    )

    grouped, audits = build_semantic_phrase_groups((first, second))

    assert len(grouped) == 1
    group = grouped[0]
    assert group.semantic_group_member_ids == ("block_0006", "block_0007")
    assert len(group.source_speech_islands) == 3
    assert [
        item["requested_pause_ms"] for item in group.source_pause_anchors
    ] == [1380, 3000]
    assert group.required_facts == (
        "Patriots tonight",
        "many things I do not know",
        "But",
        "one thing I can tell you for sure",
    )
    assert audits[0]["cross_speaker_boundary"] is False
    assert first.block_id == "block_0006"
    assert second.block_id == "block_0007"


def test_semantic_phrase_group_never_crosses_a_speaker_boundary() -> None:
    first = replace(
        _block(),
        source_text="I know this, but",
        source_timing_evidence="azure_word_timestamps",
        source_speech_start_ms=0,
        source_speech_end_ms=1800,
        source_speech_islands=(
            {"island_id": "a", "start_ms": 0, "end_ms": 1800, "text": "but"},
        ),
    )
    second = replace(
        _block(),
        block_id="block_0002",
        speaker_id="SPEAKER_01",
        start_ms=2000,
        end_ms=4000,
        source_speech_start_ms=2100,
        source_speech_end_ms=3900,
        source_speech_islands=(
            {"island_id": "b", "start_ms": 2100, "end_ms": 3900, "text": "yes"},
        ),
    )

    grouped, audits = build_semantic_phrase_groups((first, second))

    assert grouped == [first, second]
    assert audits == []


def test_performance_regions_ignore_stt_sentence_boundaries() -> None:
    first, second = _performance_source_blocks()

    regions, audits = build_performance_regions((first, second))

    assert len(regions) == 1
    region = regions[0]
    assert region.performance_region_id == "performance_region_0001"
    assert region.performance_region_member_ids == (
        "block_0006",
        "block_0007",
    )
    assert [
        value["island_id"] for value in region.source_speech_islands
    ] == ["island-a", "island-b", "island-c"]
    assert [
        value["requested_pause_ms"] for value in region.source_pause_anchors
    ] == [900, 1900]
    assert audits[0]["stt_blocks_are_storage_members_only"] is True
    assert audits[0]["approved_translation_changed"] is False


def test_semantic_fit_marks_only_multi_island_blocks_for_word_anchoring() -> None:
    first, second = _performance_source_blocks()

    adjusted, audits = attach_word_anchored_island_contracts(
        (second, first)
    )

    assert adjusted[0].performance_region_id is None
    assert adjusted[1].performance_region_id == "word_anchor_region_0001"
    assert adjusted[1].performance_region_member_ids == ("block_0006",)
    assert audits[0]["mode"] == "semantic_fit_word_anchored_islands"
    assert audits[0]["source_island_count"] == 2


def test_performance_regions_stop_at_speaker_and_interaction_boundaries() -> None:
    first, second = _performance_source_blocks()
    other_speaker = replace(second, speaker_id="SPEAKER_01")
    interjection = replace(
        second,
        source_interactions=(
            {
                "interaction_id": "overlap-1",
                "kind": "overlap",
                "other_speaker_id": "SPEAKER_01",
            },
        ),
    )

    speaker_regions, _ = build_performance_regions((first, other_speaker))
    interaction_regions, interaction_audits = build_performance_regions(
        (first, interjection)
    )

    assert len(speaker_regions) == 2
    assert all(len(value.performance_region_member_ids) == 1 for value in speaker_regions)
    assert len(interaction_regions) == 2
    assert interaction_regions[1] == interjection
    assert interaction_audits[1]["mode"] == "compatibility_block"
    assert interaction_audits[1]["reason"] == (
        "word_timing_or_interaction_boundary"
    )


def test_performance_beats_must_cover_islands_once_in_source_order() -> None:
    first, second = _performance_source_blocks()
    region = build_performance_regions((first, second))[0][0]
    valid = {
        "unit": {
            "performance_beats": [
                {
                    "beat_id": "beat-1",
                    "meaning": "first idea",
                    "island_deliveries": [
                        {"island_id": "island-a", "spoken_text": "Owokuqala"},
                        {"island_id": "island-b", "spoken_text": "uphelele"},
                    ],
                },
                {
                    "beat_id": "beat-2",
                    "meaning": "second idea",
                    "island_deliveries": [
                        {"island_id": "island-c", "spoken_text": "Owesibili"},
                    ],
                },
            ]
        }
    }

    assert DirectAzureDubRenderer._semantic_target_islands(
        region, valid, "Owokuqala uphelele Owesibili"
    ) == ("Owokuqala", "uphelele", "Owesibili")

    reversed_islands = json.loads(json.dumps(valid))
    reversed_islands["unit"]["performance_beats"][0][
        "island_deliveries"
    ].reverse()
    with pytest.raises(ValueError, match="source order"):
        DirectAzureDubRenderer._semantic_target_islands(
            region, reversed_islands, "uphelele Owokuqala Owesibili"
        )


def test_semantic_island_ssml_maps_exact_phrases_and_splits_long_pause() -> None:
    parts, audit = build_semantic_island_ssml_parts(
        ("MaPatriots namuhla", "kuningi engingakwazi. Kodwa", "kunento engiyaziyo."),
        (
            {"requested_pause_ms": 1380, "pause_kind": "island_boundary"},
            {"requested_pause_ms": 3000, "pause_kind": "island_boundary"},
        ),
        pause_budget_ms=4380,
        pause_max_ms=2000,
    )

    breaks = [part.duration_ms for part in parts if isinstance(part, BreakPart)]
    assert breaks == [1380, 2000, 1000]
    assert audit["allocation_policy"] == "semantic_island_exact"
    assert audit["target_island_count"] == 3
    assert audit["total_added_pause_ms"] == 4380


def test_semantic_island_ssml_ignores_pauses_inside_source_islands() -> None:
    parts, audit = build_semantic_island_ssml_parts(
        ("Inkulumo yokuqala", "Eyesibili", "Eyesithathu"),
        (
            {"requested_pause_ms": 380, "pause_kind": "phrase_pause"},
            {"requested_pause_ms": 1900, "pause_kind": "island_boundary"},
            {"requested_pause_ms": 420, "pause_kind": "phrase_pause"},
            {"requested_pause_ms": 1180, "pause_kind": "island_boundary"},
        ),
        pause_budget_ms=3080,
        pause_max_ms=2000,
    )

    breaks = [part.duration_ms for part in parts if isinstance(part, BreakPart)]
    assert breaks == [1900, 1180]
    assert audit["applied"] is True
    assert audit["source_pause_anchor_count"] == 4
    assert audit["source_island_boundary_count"] == 2
    assert audit["ignored_intra_island_pause_count"] == 2


def test_semantic_group_candidate_requires_exact_island_phrases() -> None:
    block = replace(
        _block(),
        semantic_group_member_ids=("block_0006", "block_0007"),
        source_speech_islands=(
            {"island_id": "a"},
            {"island_id": "b"},
            {"island_id": "c"},
        ),
    )
    candidate = {
        "unit": {
            "delivery_hints": ["Ingxenye yokuqala", "Kodwa", "eyesithathu"],
        }
    }
    spoken = "Ingxenye yokuqala Kodwa eyesithathu"

    assert DirectAzureDubRenderer._semantic_target_islands(
        block, candidate, spoken
    ) == ("Ingxenye yokuqala", "Kodwa", "eyesithathu")
    with pytest.raises(ValueError, match="exactly 3"):
        DirectAzureDubRenderer._semantic_target_islands(
            block,
            {"unit": {"delivery_hints": ["okukodwa"]}},
            "okukodwa",
        )


def test_semantic_phrase_group_synthesizes_reviewed_islands_at_locked_rate(
    tmp_path: Path,
) -> None:
    natural = (
        "Nina maPatriots namuhla ebusuku, kuningi engingakwazi. Kodwa "
        "kukhona into eyodwa engiqiniseka ngayo."
    )
    island_portfolios = [
        [
            "MaPatriots namuhla",
            "kuningi engingakwazi. Kodwa",
            "kunento eyodwa engiqiniseka ngayo.",
        ],
        [
            "MaPatriots kulobu busuku",
            "kuningi engingakwazi. Kodwa",
            "kunento engiyaziyo impela.",
        ],
        [
            "MaPatriots ebusuku",
            "kuningi engingakwazi. Kodwa",
            "kunento engiqiniseka ngayo.",
        ],
    ]
    spoken_values = [" ".join(value) for value in island_portfolios]
    block = SpeechBlock(
        block_id="phrase_group_block_0006_block_0007",
        speaker_id="SPEAKER_00",
        start_ms=63000,
        end_ms=75310,
        segment_ids=("unit_0009", "unit_0010", "unit_0011", "unit_0012"),
        source_text=(
            "Patriots tonight, there are many things I do not know. But there "
            "is one thing I can tell you for sure."
        ),
        translated_text=natural,
        tts_text=natural,
        required_facts=("the complete source thought remains true",),
        source_pause_anchors=(
            {"requested_pause_ms": 1380, "pause_kind": "island_boundary"},
            {"requested_pause_ms": 3000, "pause_kind": "island_boundary"},
        ),
        source_speech_start_ms=63270,
        source_speech_end_ms=74830,
        source_speech_islands=(
            {"island_id": "one", "start_ms": 63270, "end_ms": 65950},
            {"island_id": "two", "start_ms": 67430, "end_ms": 69030},
            {"island_id": "three", "start_ms": 72950, "end_ms": 74830},
        ),
        source_timing_evidence="azure_word_timestamps",
        semantic_group_member_ids=("block_0006", "block_0007"),
    )
    backend = _MeasuredBackend(
        {
            natural: 9000,
            spoken_values[0]: 7000,
            spoken_values[1]: 6900,
            spoken_values[2]: 6800,
        }
    )
    provider = _SemanticGroupPortfolioProvider(island_portfolios)
    renderer = DirectAzureDubRenderer(
        SimpleNamespace(work_dir=tmp_path),
        backend,
        timing_repair_provider_factory=lambda: provider,
    )

    results = renderer._synthesize_semantic_fit_batch(
        (block,),
        DirectDubArtifacts(tmp_path / "jobs" / "job-1"),
        synthesis_profiles={0: ("zu-ZA-ThembaNeural", 0, 0, 0)},
        options=DirectDubOptions(timing_mode=TIMING_MODE_SEMANTIC_FIT),
        force=False,
        progress=_DirectDubProgress(None),
    )

    selected, fitted, _audits = results[0]
    assert selected.translated_text == spoken_values[0]
    assert selected.semantic_target_islands == tuple(island_portfolios[0])
    assert fitted["speaker_tempo_locked"] is True
    assert fitted["selected_azure_rate_percent"] == 0
    assert fitted["source_pause_alignment"]["allocation_policy"] == (
        "semantic_island_exact"
    )
    assert provider.last_payload["repairs"][0]["semantic_phrase_group"][
        "target_island_count"
    ] == 3
    assert set(backend.rates) == {0}


def test_performance_plan_uses_one_holistic_candidate_then_measures_it(
    tmp_path: Path,
) -> None:
    first, second = _performance_source_blocks()
    region = build_performance_regions((first, second))[0][0]
    target_islands = [
        "Owokuqala",
        "uphelele",
        "Owesibili uhlukile.",
    ]
    planned = " ".join(target_islands)
    backend = _MeasuredBackend(
        {
            region.translated_text: 6700,
            planned: 6700,
            "Owokuqala": 2850,
            "uphelele": 1850,
            "Owesibili uhlukile.": 1350,
        }
    )
    provider = _PerformancePlanProvider([target_islands])
    renderer = DirectAzureDubRenderer(
        SimpleNamespace(work_dir=tmp_path),
        backend,
        timing_repair_provider_factory=lambda: provider,
    )
    artifacts = DirectDubArtifacts(tmp_path / "jobs" / "job-1")

    results = renderer._synthesize_semantic_fit_batch(
        (region,),
        artifacts,
        synthesis_profiles={0: ("zu-ZA-ThembaNeural", 0, 0, 0)},
        options=DirectDubOptions(
            timing_mode=TIMING_MODE_PERFORMANCE_PLAN,
            semantic_fit_max_attempts=12,
        ),
        force=False,
        progress=_DirectDubProgress(None),
    )

    selected, fitted, attempts = results[0]
    assert provider.portfolio_calls == 1
    assert provider.requests[0]["candidate_count"] == 1
    assert provider.requests[0]["performance_plan"][
        "stt_member_boundaries_are_not_linguistic_boundaries"
    ] is True
    assert selected.translated_text == planned
    assert selected.semantic_target_islands == tuple(target_islands)
    assert fitted["selected_azure_rate_percent"] == 0
    assert fitted["post_synthesis_time_stretch_ratio"] == 1.0
    assert fitted["performance_plan"]["region_id"] == (
        "performance_region_0001"
    )
    assert fitted["performance_plan"][
        "stt_member_boundaries_used_as_linguistic_boundaries"
    ] is False
    assert fitted["source_pause_alignment"]["total_added_pause_ms"] == 2800
    assert [
        value["status"]
        for value in fitted["translation_variant_attempts"][-1][
            "performance_island_measurements"
        ]
    ] == ["fit", "fit", "fit"]
    assert attempts[0]["status"] == "performance_plan_required"
    assert attempts[-1]["status"] == "selected"
    assert set(backend.rates) == {0}


def test_performance_plan_rejects_large_internal_island_drift(
    tmp_path: Path,
) -> None:
    first, second = _performance_source_blocks()
    region = build_performance_regions((first, second))[0][0]
    target_islands = ["Owokuqala", "uphelele", "Owesibili uhlukile."]
    planned = " ".join(target_islands)
    backend = _MeasuredBackend(
        {
            region.translated_text: 6700,
            planned: 6700,
            # Isolated synthesis adds/removes phrase-boundary prosody. These
            # deliberately miss all three island bands while the continuous
            # region plus protected pauses is exactly 9.5 seconds.
            "Owokuqala": 5000,
            "uphelele": 500,
            "Owesibili uhlukile.": 1050,
        }
    )
    provider = _PerformancePlanProvider([target_islands])
    renderer = DirectAzureDubRenderer(
        SimpleNamespace(work_dir=tmp_path),
        backend,
        timing_repair_provider_factory=lambda: provider,
    )

    with pytest.raises(DirectDubError, match="exhausted the attempt budget"):
        renderer._synthesize_semantic_fit_batch(
            (region,),
            DirectDubArtifacts(tmp_path / "jobs" / "job-1"),
            synthesis_profiles={0: ("zu-ZA-ThembaNeural", 0, 0, 0)},
            options=DirectDubOptions(timing_mode=TIMING_MODE_PERFORMANCE_PLAN),
            force=False,
            progress=_DirectDubProgress(None),
        )

    checkpoint = json.loads(
        DirectDubArtifacts(tmp_path / "jobs" / "job-1")
        .semantic_fit.read_text(encoding="utf-8")
    )
    attempts = checkpoint["cases"]["performance_region_0001"]["attempts"]
    attempt = next(
        value for value in attempts
        if value["status"] == "internal_island_drift"
    )
    assert attempt["performance_island_alignment"]["hard_gate_passed"] is False


def test_performance_plan_feedback_is_one_measured_revision_per_round(
    tmp_path: Path,
) -> None:
    first, second = _performance_source_blocks()
    region = build_performance_regions((first, second))[0][0]
    short_islands = ["Mfushane", "kakhulu", "lapha."]
    fitted_islands = ["Owokuqala", "uphelele", "Owesibili uhlukile."]
    short = " ".join(short_islands)
    fitted_text = " ".join(fitted_islands)
    backend = _MeasuredBackend(
        {
            region.translated_text: 6700,
            short: 3000,
            fitted_text: 6700,
            **{value: 1000 for value in short_islands},
            "Owokuqala": 2850,
            "uphelele": 1850,
            "Owesibili uhlukile.": 1350,
        }
    )
    provider = _AdaptivePerformancePlanProvider(
        [short_islands, fitted_islands]
    )
    renderer = DirectAzureDubRenderer(
        SimpleNamespace(work_dir=tmp_path),
        backend,
        timing_repair_provider_factory=lambda: provider,
    )

    results = renderer._synthesize_semantic_fit_batch(
        (region,),
        DirectDubArtifacts(tmp_path / "jobs" / "job-1"),
        synthesis_profiles={0: ("zu-ZA-ThembaNeural", 0, 0, 0)},
        options=DirectDubOptions(
            timing_mode=TIMING_MODE_PERFORMANCE_PLAN,
            semantic_fit_max_attempts=12,
        ),
        force=False,
        progress=_DirectDubProgress(None),
    )

    assert results[0][0].translated_text == fitted_text
    assert provider.portfolio_calls == 2
    assert [request["candidate_count"] for request in provider.requests] == [1, 1]
    assert provider.requests[1]["acoustic_correction"]["direction"] == "expand"


def test_performance_plan_caps_measured_feedback_even_when_cli_allows_twelve(
    tmp_path: Path,
) -> None:
    first, second = _performance_source_blocks()
    region = build_performance_regions((first, second))[0][0]
    revisions = [
        [f"Mfushane {index}a", f"{index}b", f"{index}c"]
        for index in range(1, 5)
    ]
    durations = {region.translated_text: 6700}
    for revision in revisions:
        durations[" ".join(revision)] = 2500
        durations.update({value: 800 for value in revision})
    backend = _MeasuredBackend(durations)
    provider = _AdaptivePerformancePlanProvider(revisions)
    renderer = DirectAzureDubRenderer(
        SimpleNamespace(work_dir=tmp_path),
        backend,
        timing_repair_provider_factory=lambda: provider,
    )

    with pytest.raises(DirectDubError, match="exhausted"):
        renderer._synthesize_semantic_fit_batch(
            (region,),
            DirectDubArtifacts(tmp_path / "jobs" / "job-1"),
            synthesis_profiles={0: ("zu-ZA-ThembaNeural", 0, 0, 0)},
            options=DirectDubOptions(
                timing_mode=TIMING_MODE_PERFORMANCE_PLAN,
                semantic_fit_max_attempts=12,
            ),
            force=False,
            progress=_DirectDubProgress(None),
        )

    assert provider.portfolio_calls == 4
    assert [request["candidate_count"] for request in provider.requests] == [
        1,
        1,
        1,
        1,
    ]


def test_performance_plan_island_ownership_allows_punctuation_only() -> None:
    first, second = _performance_source_blocks()
    region = build_performance_regions((first, second))[0][0]
    spoken = "Yebo, ngiyakuzwa. Manje siyaqhubeka."
    candidate = {
        "unit": {
            "performance_beats": [
                {
                    "beat_id": "beat_001",
                    "island_deliveries": [
                        {"island_id": "island-a", "spoken_text": "Yebo"},
                        {"island_id": "island-b", "spoken_text": "ngiyakuzwa"},
                        {
                            "island_id": "island-c",
                            "spoken_text": "Manje siyaqhubeka",
                        },
                    ],
                }
            ]
        }
    }

    assert DirectAzureDubRenderer._semantic_target_islands(
        region,
        candidate,
        spoken,
    ) == ("Yebo", "ngiyakuzwa", "Manje siyaqhubeka")


def test_performance_plan_shards_nine_regions_and_checkpoints_each_batch(
    tmp_path: Path,
) -> None:
    regions = tuple(
        SpeechBlock(
            block_id=f"performance_region_{index + 1:04d}",
            speaker_id="SPEAKER_00",
            start_ms=index * 3000,
            end_ms=index * 3000 + 2000,
            segment_ids=(f"unit_{index + 1:04d}",),
            source_text=f"Source region {index}",
            translated_text=f"Natural region {index}",
            tts_text=f"Natural region {index}",
            required_facts=(f"region fact {index}",),
            source_speech_start_ms=index * 3000,
            source_speech_end_ms=index * 3000 + 2000,
            source_speech_islands=(
                {
                    "island_id": f"region-{index}",
                    "start_ms": index * 3000,
                    "end_ms": index * 3000 + 2000,
                    "text": f"Source region {index}",
                },
            ),
            source_timing_evidence="azure_word_timestamps",
            performance_region_id=f"performance_region_{index + 1:04d}",
            performance_region_member_ids=(f"block_{index + 1:04d}",),
        )
        for index in range(9)
    )
    durations = {
        **{value.translated_text: 1800 for value in regions},
        "single-island template": 1800,
    }
    backend = _MeasuredBackend(durations)
    provider = _PerformancePlanProvider([["single-island template"]])
    renderer = DirectAzureDubRenderer(
        SimpleNamespace(work_dir=tmp_path),
        backend,
        timing_repair_provider_factory=lambda: provider,
    )
    artifacts = DirectDubArtifacts(tmp_path / "jobs" / "job-1")

    results = renderer._synthesize_semantic_fit_batch(
        regions,
        artifacts,
        synthesis_profiles={
            index: ("zu-ZA-ThembaNeural", 0, 0, 0)
            for index in range(9)
        },
        options=DirectDubOptions(
            timing_mode=TIMING_MODE_PERFORMANCE_PLAN,
            semantic_fit_max_attempts=1,
        ),
        force=False,
        progress=_DirectDubProgress(None),
    )

    assert len(results) == 9
    assert provider.portfolio_calls == 3
    assert provider.batch_sizes == [3, 3, 3]
    assert [value["candidate_count"] for value in provider.requests] == [1] * 9
    checkpoint = json.loads(artifacts.semantic_fit.read_text(encoding="utf-8"))
    assert len(checkpoint["batch_rounds"]) == 3
    assert all(
        value["status"] == "selected"
        for value in checkpoint["cases"].values()
    )


def test_semantic_fit_batch_restart_reuses_paid_serial_candidate(
    tmp_path: Path,
) -> None:
    block = _block()
    cached_spoken = "Umusho okhokhelwe osevele ukhona"
    backend = _MeasuredBackend(
        {block.translated_text: 2400, cached_spoken: 1800}
    )
    provider = _SemanticBatchProvider({})
    renderer = DirectAzureDubRenderer(
        SimpleNamespace(work_dir=tmp_path),
        backend,
        timing_repair_provider_factory=lambda: provider,
    )
    artifacts = DirectDubArtifacts(tmp_path / "jobs" / "job-1")
    artifacts.semantic_fit.parent.mkdir(parents=True, exist_ok=True)
    artifacts.semantic_fit.write_text(
        json.dumps(
            {
                "schema_version": SEMANTIC_FIT_SCHEMA_VERSION,
                "job_id": "job-1",
                "cases": {
                    "block_0001": {
                        "status": "exhausted",
                        "speaker_id": "SPEAKER_00",
                        "voice": "zu-ZA-ThandoNeural",
                        "speaker_rate_percent": 0,
                        "attempts": [
                            {
                                "attempt": 1,
                                "request_fingerprint": "old-serial-call",
                                "candidate": {
                                    "schema_version": "claude-turn-repair-v1",
                                    "unit": {
                                        "unit_id": "block_0001",
                                        "faithful_translation": (
                                            block.translated_text
                                        ),
                                        "spoken_text": cached_spoken,
                                        "tts_text": cached_spoken,
                                        "language_features": [],
                                        "human_review_flags": [],
                                        "numbers_preserved": True,
                                        "dates_preserved": True,
                                        "negation_preserved": True,
                                        "omitted_or_compressed_detail": [],
                                    },
                                },
                            }
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    results = renderer._synthesize_semantic_fit_batch(
        (block,),
        artifacts,
        synthesis_profiles={0: ("zu-ZA-ThandoNeural", 0, 0, 0)},
        options=DirectDubOptions(timing_mode=TIMING_MODE_SEMANTIC_FIT),
        force=False,
        progress=_DirectDubProgress(None),
    )

    assert results[0][0].translated_text == cached_spoken
    assert provider.fit_batch_calls == 0
    assert provider.review_batch_calls == 1


def test_hard_final_block_reserves_attempts_for_measured_feedback_rounds(
    tmp_path: Path,
) -> None:
    block = _block()
    cached_overflow = "Umusho omdala omude kakhulu"
    candidates = [
        "Umusho omusha omude",
        "Umusho omusha",
        "Omusha",
    ]
    backend = _MeasuredBackend(
        {
            block.translated_text: 2500,
            cached_overflow: 2300,
            candidates[0]: 2200,
            candidates[1]: 1800,
            candidates[2]: 1600,
        }
    )
    provider = _SemanticPortfolioProvider({"block_0001": candidates})
    renderer = DirectAzureDubRenderer(
        SimpleNamespace(work_dir=tmp_path),
        backend,
        timing_repair_provider_factory=lambda: provider,
    )
    artifacts = DirectDubArtifacts(tmp_path / "jobs" / "job-1")
    artifacts.semantic_fit.parent.mkdir(parents=True, exist_ok=True)
    artifacts.semantic_fit.write_text(
        json.dumps(
            {
                "schema_version": SEMANTIC_FIT_SCHEMA_VERSION,
                "job_id": "job-1",
                "cases": {
                    "block_0001": {
                        "status": "exhausted",
                        "speaker_id": "SPEAKER_00",
                        "voice": "zu-ZA-ThandoNeural",
                        "speaker_rate_percent": 0,
                        "strategy_version": "older-strategy",
                        "attempts": [
                            {
                                "attempt": 1,
                                "candidate": {
                                    "schema_version": "claude-turn-repair-v1",
                                    "unit": {
                                        "unit_id": "block_0001",
                                        "faithful_translation": block.translated_text,
                                        "spoken_text": cached_overflow,
                                        "tts_text": cached_overflow,
                                        "language_features": [],
                                        "human_review_flags": [],
                                        "numbers_preserved": True,
                                        "dates_preserved": True,
                                        "negation_preserved": True,
                                        "omitted_or_compressed_detail": [],
                                    },
                                },
                            }
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    results = renderer._synthesize_semantic_fit_batch(
        (block,),
        artifacts,
        synthesis_profiles={0: ("zu-ZA-ThandoNeural", 0, 0, 0)},
        options=DirectDubOptions(timing_mode=TIMING_MODE_SEMANTIC_FIT),
        force=False,
        progress=_DirectDubProgress(None),
    )

    assert provider.requested_candidate_counts == [3]
    assert results[0][0].translated_text == candidates[1]
    checkpoint = json.loads(artifacts.semantic_fit.read_text(encoding="utf-8"))
    cached = checkpoint["cases"]["block_0001"]["attempts"][0]
    assert cached["plain_speech_duration_ms"] == 2300


def test_new_strategy_gets_fresh_budget_after_old_attempts_are_exhausted(
    tmp_path: Path,
) -> None:
    block = _block()
    old_spoken = "Umusho omdala omude kakhulu"
    new_spoken = "Umusho omusha"
    backend = _MeasuredBackend(
        {
            block.translated_text: 2400,
            old_spoken: 2300,
            new_spoken: 1800,
        }
    )
    provider = _SemanticBatchProvider({"block_0001": new_spoken})
    renderer = DirectAzureDubRenderer(
        SimpleNamespace(work_dir=tmp_path),
        backend,
        timing_repair_provider_factory=lambda: provider,
    )
    artifacts = DirectDubArtifacts(tmp_path / "jobs" / "job-1")
    artifacts.semantic_fit.parent.mkdir(parents=True, exist_ok=True)
    artifacts.semantic_fit.write_text(
        json.dumps(
            {
                "schema_version": SEMANTIC_FIT_SCHEMA_VERSION,
                "job_id": "job-1",
                "strategy_version": "obsolete-strategy",
                "cases": {
                    "block_0001": {
                        "status": "exhausted",
                        "attempts": [
                            {
                                "attempt": 12,
                                "candidate": {
                                    "schema_version": "claude-turn-repair-v1",
                                    "unit": {
                                        "unit_id": "block_0001",
                                        "faithful_translation": block.translated_text,
                                        "spoken_text": old_spoken,
                                        "tts_text": old_spoken,
                                        "language_features": [],
                                        "human_review_flags": [],
                                        "numbers_preserved": True,
                                        "dates_preserved": True,
                                        "negation_preserved": True,
                                        "omitted_or_compressed_detail": [],
                                    },
                                },
                            }
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    results = renderer._synthesize_semantic_fit_batch(
        (block,),
        artifacts,
        synthesis_profiles={0: ("zu-ZA-ThandoNeural", 0, 0, 0)},
        options=DirectDubOptions(
            timing_mode=TIMING_MODE_SEMANTIC_FIT,
            semantic_fit_max_attempts=1,
        ),
        force=False,
        progress=_DirectDubProgress(None),
    )

    assert results[0][0].translated_text == new_spoken
    assert provider.fit_batch_calls == 1
    checkpoint = json.loads(artifacts.semantic_fit.read_text(encoding="utf-8"))
    latest = checkpoint["cases"]["block_0001"]["attempts"][-1]
    assert latest["attempt"] == 13
    assert latest["strategy_attempt"] == 1


def test_semantic_fit_payload_compacts_repeated_history() -> None:
    block = _block()
    evidence = [
        {
            "attempt": attempt,
            "status": "overflow",
            "spoken_text": f"candidate {attempt}",
            "measured_duration_ms": 3000 - attempt * 50,
            "target_maximum_ms": 2000,
        }
        for attempt in range(1, 13)
    ]

    payload = DirectAzureDubRenderer._semantic_fit_payload(
        block,
        voice="zu-ZA-ThandoNeural",
        base_rate_percent=0,
        pitch_percent=0,
        volume_percent=0,
        attempt_number=1,
        target_minimum_ms=1500,
        target_preferred_ms=1750,
        target_maximum_ms=2000,
        duration_evidence=evidence,
        rejected_spoken_texts=[f"candidate {value}" for value in range(12)],
        current_spoken_text="candidate 12",
    )

    assert len(payload["duration_evidence"]) <= 4
    assert len(payload["rejected_spoken_texts"]) == 6
    assert payload["evidence_compaction"] == {
        "input_observation_count": 12,
        "transmitted_observation_count": len(payload["duration_evidence"]),
        "maximum_rejected_texts_transmitted": 6,
    }


def test_semantic_fit_rate_limit_retries_only_pending_review_batch(
    tmp_path: Path,
) -> None:
    block = _block()
    candidate = "Umusho omfushane olingana kahle"
    backend = _MeasuredBackend(
        {block.translated_text: 2400, candidate: 1800}
    )
    provider = _SemanticBatchProvider(
        {"block_0001": candidate},
        review_rate_limits=1,
        retry_after_seconds=0.25,
    )
    renderer = DirectAzureDubRenderer(
        SimpleNamespace(work_dir=tmp_path),
        backend,
        timing_repair_provider_factory=lambda: provider,
    )

    results = renderer._synthesize_semantic_fit_batch(
        (block,),
        DirectDubArtifacts(tmp_path / "jobs" / "job-1"),
        synthesis_profiles={0: ("zu-ZA-ThandoNeural", 0, 0, 0)},
        options=DirectDubOptions(timing_mode=TIMING_MODE_SEMANTIC_FIT),
        force=False,
        progress=_DirectDubProgress(None),
    )

    assert results[0][0].translated_text == candidate
    assert provider.fit_batch_calls == 1
    assert provider.review_batch_calls == 2
    assert provider.sleep_calls == [0.25]


def test_locked_tempo_pause_budget_uses_semantic_deadline_not_full_block(
    tmp_path: Path,
) -> None:
    text = "Umusho omude kodwa wemvelo"
    backend = _MeasuredBackend({text: 9100})
    renderer = DirectAzureDubRenderer(
        SimpleNamespace(work_dir=tmp_path),
        backend,
    )
    block = SpeechBlock(
        block_id="block_0001_variant_natural",
        speaker_id="SPEAKER_00",
        start_ms=0,
        end_ms=11220,
        segment_ids=("unit_0001",),
        source_text="A slow source sentence",
        translated_text=text,
        tts_text=text,
        source_pause_anchors=(
            {
                "source_progress": 0.25,
                "source_gap_ms": 1100,
                "requested_pause_ms": 1000,
            },
            {
                "source_progress": 0.75,
                "source_gap_ms": 1100,
                "requested_pause_ms": 1000,
            },
        ),
        source_speech_start_ms=0,
        source_speech_end_ms=10720,
    )

    fitted = renderer._synthesize_locked_tempo_block(
        block,
        DirectDubArtifacts(tmp_path / "jobs" / "job-1"),
        voice="zu-ZA-ThandoNeural",
        base_rate_percent=0,
        pitch_percent=0,
        volume_percent=0,
        force=False,
        timing_deadline_ms=10970,
    )

    assert fitted["semantic_timing_deadline_ms"] == 10970
    assert fitted["duration_ms"] == 10870
    assert fitted["duration_ms"] <= 10970
    assert fitted["source_pause_alignment"]["total_added_pause_ms"] == 1770
    assert _selected_azure_measurement_ms(fitted) == 10870


def test_selected_measurement_does_not_choose_shortest_same_rate_attempt() -> None:
    fitted = {
        "selected_azure_rate_percent": 0,
        "azure_attempts": [
            {"rate_percent": 0, "duration_ms": 3575},
            {"rate_percent": 0, "duration_ms": 4512},
            {"rate_percent": 0, "duration_ms": 4476},
        ],
        "duration_ms": 4476,
    }

    assert _selected_azure_measurement_ms(fitted) == 4476


def test_semantic_fit_target_band_uses_full_safe_block_window() -> None:
    block = SpeechBlock(
        block_id="block_0001_variant_natural",
        speaker_id="SPEAKER_00",
        start_ms=0,
        end_ms=2380,
        segment_ids=("unit_0001",),
        source_text="There is one thing I can tell you for sure.",
        translated_text="Kukhona into eyodwa engingaqiniseka ngayo.",
        tts_text="Kukhona into eyodwa engingaqiniseka ngayo.",
        source_speech_start_ms=0,
        source_speech_end_ms=1880,
    )

    assert DirectAzureDubRenderer._semantic_fit_target_band(block, 250) == (
        1630,
        1880,
        2380,
    )


def test_word_timed_delivery_is_hard_anchored_to_mouth_onset() -> None:
    block = SpeechBlock(
        block_id="block_0007_variant_semantic_fit_12",
        speaker_id="SPEAKER_00",
        start_ms=72950,
        end_ms=75330,
        segment_ids=("unit_0012",),
        source_text="There is one thing I know.",
        translated_text="kukhon' okunye engikwaz'.",
        tts_text="kukhon' okunye engikwaz'.",
        source_speech_start_ms=72950,
        source_speech_end_ms=74830,
        source_timing_evidence="azure_word_timestamps",
    )

    assert DirectAzureDubRenderer._semantic_fit_target_band(block, 250) == (
        1630,
        1880,
        2130,
    )
    selected, audit = DirectAzureDubRenderer._center_word_timed_delivery(
        block,
        2100,
        250,
    )

    assert (selected.start_ms, selected.end_ms) == (72950, 75050)
    assert audit["start_error_ms"] == 0
    assert audit["endpoint_error_ms"] == 220
    assert audit["placement_policy"] == "source_word_onset_hard_anchor"
    assert audit["within_tolerance"] is True


def test_final_word_timed_delivery_never_centres_past_media_boundary() -> None:
    block = SpeechBlock(
        block_id="block_0021_variant_natural",
        speaker_id="SPEAKER_00",
        start_ms=206510,
        end_ms=227072,
        segment_ids=("unit_0034",),
        source_text="source",
        translated_text="translation",
        tts_text="translation",
        source_speech_start_ms=206510,
        source_speech_end_ms=227110,
        source_timing_evidence="azure_word_timestamps",
    )

    selected, audit = DirectAzureDubRenderer._center_word_timed_delivery(
        block,
        20536,
        250,
    )

    assert selected.start_ms == 206510
    assert selected.end_ms == 227046
    assert selected.end_ms <= block.end_ms
    assert audit["source_speech_end_ms"] == 227072
    assert audit["unclamped_source_speech_end_ms"] == 227110


def test_capacity_aware_islands_fit_exact_failed_block_one(
    tmp_path: Path,
) -> None:
    text = "Bazam' konk' bangiwise. NgoFadile. Uvel' kusasa e-Madlanga Commission. NgoFadiel Adams."
    backend = _MeasuredBackend({text: 8812})
    renderer = DirectAzureDubRenderer(SimpleNamespace(work_dir=tmp_path), backend)
    block = SpeechBlock(
        block_id="block_0001_variant_semantic_fit_09",
        speaker_id="SPEAKER_00",
        start_ms=2390,
        end_ms=13610,
        segment_ids=("unit_0001", "unit_0002"),
        source_text="source",
        translated_text=text,
        tts_text=text,
        source_pause_anchors=(
            {"source_progress": 0.20, "requested_pause_ms": 500, "pause_kind": "phrase_pause"},
            {"source_progress": 0.40, "requested_pause_ms": 620, "pause_kind": "phrase_pause"},
            {"source_progress": 0.60, "requested_pause_ms": 660, "pause_kind": "phrase_pause"},
            {"source_progress": 0.80, "requested_pause_ms": 1180, "pause_kind": "island_boundary"},
        ),
        source_speech_start_ms=2390,
        source_speech_end_ms=13110,
        source_timing_evidence="azure_word_timestamps",
    )

    fitted = renderer._synthesize_locked_tempo_block(
        block,
        DirectDubArtifacts(tmp_path / "jobs" / "job-1"),
        voice="zu-ZA-ThandoNeural",
        base_rate_percent=0,
        pitch_percent=0,
        volume_percent=0,
        force=False,
        timing_deadline_ms=11220,
        timing_preferred_ms=10720,
    )

    pause = fitted["source_pause_alignment"]
    assert fitted["duration_ms"] == 10720
    assert pause["total_added_pause_ms"] == 1908
    assert next(
        item for item in pause["insertions"]
        if item["pause_kind"] == "island_boundary"
    )["duration_ms"] == 1180
    assert all(
        part <= 2000
        for item in pause["insertions"]
        for part in item["ssml_break_durations_ms"]
    )


def test_word_timed_candidate_cannot_erase_an_island_to_fit(
    tmp_path: Path,
) -> None:
    text = "Inkulumo ende egcwalisa iwindi ngaphandle kwekhefu"
    backend = _MeasuredBackend({text: 10662})
    renderer = DirectAzureDubRenderer(SimpleNamespace(work_dir=tmp_path), backend)
    block = SpeechBlock(
        block_id="block_0001_variant_semantic_fit_01",
        speaker_id="SPEAKER_00",
        start_ms=2390,
        end_ms=13610,
        segment_ids=("unit_0001",),
        source_text="source",
        translated_text=text,
        tts_text=text,
        source_pause_anchors=(
            {
                "source_progress": 0.8,
                "requested_pause_ms": 1180,
                "pause_kind": "island_boundary",
            },
        ),
        source_speech_start_ms=2390,
        source_speech_end_ms=13110,
        source_timing_evidence="azure_word_timestamps",
    )

    with pytest.raises(TimingOverflowError):
        renderer._synthesize_locked_tempo_block(
            block,
            DirectDubArtifacts(tmp_path / "jobs" / "job-1"),
            voice="zu-ZA-ThandoNeural",
            base_rate_percent=0,
            pitch_percent=0,
            volume_percent=0,
            force=False,
            timing_deadline_ms=11220,
            timing_preferred_ms=10720,
        )


def test_capacity_aware_pauses_fill_exact_final_block_underflow(
    tmp_path: Path,
) -> None:
    text = "amazwi aqotho avala inkulumo ende ngendlela yemvelo"
    backend = _MeasuredBackend({text: 11675})
    renderer = DirectAzureDubRenderer(SimpleNamespace(work_dir=tmp_path), backend)
    block = SpeechBlock(
        block_id="block_0021_variant_semantic_fit_02",
        speaker_id="SPEAKER_00",
        start_ms=206510,
        end_ms=227072,
        segment_ids=("unit_0034",),
        source_text="source",
        translated_text=text,
        tts_text=text,
        source_pause_anchors=tuple(
            {
                "source_progress": index / 8,
                "requested_pause_ms": duration_ms,
                "pause_kind": (
                    "island_boundary" if duration_ms >= 800 else "phrase_pause"
                ),
            }
            for index, duration_ms in enumerate(
                (620, 900, 1100, 740, 1180, 980, 1460),
                start=1,
            )
        ),
        source_speech_start_ms=206510,
        source_speech_end_ms=227072,
        source_timing_evidence="azure_word_timestamps",
    )

    fitted = renderer._synthesize_locked_tempo_block(
        block,
        DirectDubArtifacts(tmp_path / "jobs" / "job-1"),
        voice="zu-ZA-ThandoNeural",
        base_rate_percent=0,
        pitch_percent=0,
        volume_percent=0,
        force=False,
        timing_deadline_ms=20562,
        timing_preferred_ms=20562,
    )

    assert fitted["duration_ms"] == 20562
    assert fitted["source_pause_alignment"]["total_added_pause_ms"] == 8887


def test_word_timed_calibration_preserves_island_floor_and_reduces_phrase_pause(
    tmp_path: Path,
) -> None:
    text = "lawa ngamazwi amaningi ahlukene okuhlola izikhawu zomthombo ngendlela yemvelo kakhulu manje"
    backend = _PauseOvershootBackend({text: 13662}, pause_overhead_ms=526)
    renderer = DirectAzureDubRenderer(SimpleNamespace(work_dir=tmp_path), backend)
    requested = (
        (620, "phrase_pause"),
        (1100, "island_boundary"),
        (460, "phrase_pause"),
        (1340, "island_boundary"),
        (1140, "island_boundary"),
        (1500, "island_boundary"),
        (760, "island_boundary"),
        (460, "phrase_pause"),
    )
    block = SpeechBlock(
        block_id="block_0021_variant_natural",
        speaker_id="SPEAKER_00",
        start_ms=206510,
        end_ms=227072,
        segment_ids=("unit_0034",),
        source_text="source",
        translated_text=text,
        tts_text=text,
        source_pause_anchors=tuple(
            {
                "source_progress": index / 9,
                "requested_pause_ms": duration_ms,
                "pause_kind": pause_kind,
            }
            for index, (duration_ms, pause_kind) in enumerate(requested, start=1)
        ),
        source_speech_start_ms=206510,
        source_speech_end_ms=227072,
        source_timing_evidence="azure_word_timestamps",
    )

    fitted = renderer._synthesize_locked_tempo_block(
        block,
        DirectDubArtifacts(tmp_path / "jobs" / "job-1"),
        voice="zu-ZA-ThembaNeural",
        base_rate_percent=0,
        pitch_percent=0,
        volume_percent=0,
        force=False,
        timing_deadline_ms=20562,
        timing_preferred_ms=20562,
    )

    alignment = fitted["source_pause_alignment"]
    assert fitted["duration_ms"] <= 20562
    assert alignment["azure_measured_pause_calibration"] is True
    assert alignment["island_pause_total_ms"] == 5840
    assert alignment["phrase_pause_total_ms"] < 1540
    assert all(
        item["duration_ms"] >= item["requested_pause_ms"]
        for item in alignment["insertions"]
        if item["pause_kind"] == "island_boundary"
    )


def test_pause_calibration_floor_matches_chunked_island_breaks() -> None:
    parts = (BreakPart(2000), BreakPart(360), BreakPart(1000))
    audit = {
        "insertions": [
            {
                "target_character_offset": 10,
                "pause_kind": "island_boundary",
                "requested_pause_ms": 1500,
                "duration_ms": 2360,
                "ssml_break_durations_ms": [2000, 360],
            },
            {
                "target_character_offset": 20,
                "pause_kind": "phrase_pause",
                "requested_pause_ms": 1000,
                "duration_ms": 1000,
                "ssml_break_durations_ms": [1000],
            },
        ]
    }
    floors = source_pause_calibration_floors(audit)
    calibrated, durations = calibrate_source_pause_ssml_parts(
        parts,
        reduction_ms=1200,
        minimum_break_durations_ms=floors,
    )

    assert floors == (1500, 0, 0)
    assert sum(durations[:2]) >= 1500
    assert sum(part.duration_ms for part in calibrated) == 2160


def test_locked_tempo_calibrates_measured_pause_overshoot(
    tmp_path: Path,
) -> None:
    text = "Umusho wemvelo onekhefu"
    backend = _PauseOvershootBackend({text: 9000}, pause_overhead_ms=200)
    renderer = DirectAzureDubRenderer(SimpleNamespace(work_dir=tmp_path), backend)
    block = SpeechBlock(
        block_id="block_0001_variant_natural",
        speaker_id="SPEAKER_00",
        start_ms=0,
        end_ms=10000,
        segment_ids=("unit_0001",),
        source_text="A sentence with a measured pause",
        translated_text=text,
        tts_text=text,
        source_pause_anchors=(
            {
                "source_progress": 0.5,
                "source_gap_ms": 1200,
                "requested_pause_ms": 1100,
            },
        ),
        source_speech_start_ms=0,
        source_speech_end_ms=9900,
    )

    fitted = renderer._synthesize_locked_tempo_block(
        block,
        DirectDubArtifacts(tmp_path / "jobs" / "job-1"),
        voice="zu-ZA-ThandoNeural",
        base_rate_percent=0,
        pitch_percent=0,
        volume_percent=0,
        force=False,
        timing_deadline_ms=10000,
    )

    assert fitted["duration_ms"] == 9975
    assert fitted["duration_ms"] <= 10000
    assert fitted["source_pause_alignment"][
        "azure_measured_pause_calibration"
    ] is True
    assert fitted["source_pause_alignment"]["total_added_pause_ms"] == 775
    assert backend.rates == [0, 0, 0]


def test_word_timed_pause_overflow_requires_shorter_semantic_delivery(
    tmp_path: Path,
) -> None:
    text = "Umusho onekhefu elivela emazwini omthombo"
    backend = _PauseOvershootBackend({text: 9000}, pause_overhead_ms=200)
    renderer = DirectAzureDubRenderer(SimpleNamespace(work_dir=tmp_path), backend)
    block = SpeechBlock(
        block_id="block_0001_variant_natural",
        speaker_id="SPEAKER_00",
        start_ms=0,
        end_ms=10000,
        segment_ids=("unit_0001",),
        source_text="A sentence with an authoritative word-timed pause",
        translated_text=text,
        tts_text=text,
        source_pause_anchors=(
            {
                "source_progress": 0.5,
                "source_gap_ms": 1200,
                "requested_pause_ms": 1100,
                "pause_kind": "island_boundary",
            },
        ),
        source_speech_start_ms=0,
        source_speech_end_ms=9900,
        source_timing_evidence="azure_word_timestamps",
    )

    with pytest.raises(TimingOverflowError):
        renderer._synthesize_locked_tempo_block(
            block,
            DirectDubArtifacts(tmp_path / "jobs" / "job-1"),
            voice="zu-ZA-ThandoNeural",
            base_rate_percent=0,
            pitch_percent=0,
            volume_percent=0,
            force=False,
            timing_deadline_ms=10000,
        )

    # Plain speech plus one pause-bearing measurement: the source pause is not
    # shortened by calibration rounds just to make an overlong wording pass.
    assert backend.rates == [0, 0]


def test_semantic_fit_borrows_bounded_same_speaker_transition_room(
    tmp_path: Path,
) -> None:
    first = SpeechBlock(
        block_id="block_0001_variant_natural",
        speaker_id="SPEAKER_00",
        start_ms=0,
        end_ms=1540,
        segment_ids=("unit_0001",),
        source_text="We need to fast.",
        translated_text="Kudingeka sizile ukudla.",
        tts_text="Kudingeka sizile ukudla.",
        required_facts=("The group needs to fast",),
        source_speech_start_ms=0,
        source_speech_end_ms=1040,
    )
    second = SpeechBlock(
        block_id="block_0002_variant_natural",
        speaker_id="SPEAKER_00",
        start_ms=4440,
        end_ms=6000,
        segment_ids=("unit_0002",),
        source_text="A following sentence.",
        translated_text="Umusho olandelayo.",
        tts_text="Umusho olandelayo.",
    )
    candidate = "Kumele sizile."
    backend = _MeasuredBackend(
        {
            first.translated_text: 3500,
            second.translated_text: 1400,
            candidate: 1625,
        }
    )
    provider = _SemanticBatchProvider({"block_0001": candidate})
    renderer = DirectAzureDubRenderer(
        SimpleNamespace(work_dir=tmp_path),
        backend,
        timing_repair_provider_factory=lambda: provider,
    )
    artifacts = DirectDubArtifacts(tmp_path / "jobs" / "job-1")
    artifacts.semantic_fit.parent.mkdir(parents=True, exist_ok=True)
    artifacts.semantic_fit.write_text(
        json.dumps(
            {
                "schema_version": SEMANTIC_FIT_SCHEMA_VERSION,
                "job_id": "job-1",
                "cases": {
                    "block_0001": {
                        "status": "exhausted",
                        "attempt_limit": 12,
                        "attempts": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    results = renderer._synthesize_semantic_fit_batch(
        (first, second),
        artifacts,
        synthesis_profiles={
            0: ("zu-ZA-ThandoNeural", 0, 0, 0),
            1: ("zu-ZA-ThandoNeural", 0, 0, 0),
        },
        options=DirectDubOptions(timing_mode=TIMING_MODE_SEMANTIC_FIT),
        force=False,
        progress=_DirectDubProgress(None),
    )

    selected, fitted, _audits = results[0]
    assert selected.translated_text == candidate
    assert selected.end_ms == 1625
    assert fitted["semantic_fit"]["same_speaker_transition_borrow_ms"] == 85
    assert (
        fitted["semantic_fit"]["same_speaker_transition_borrow_limit_ms"]
        == 1450
    )
    assert fitted["semantic_fit"]["original_block_end_ms"] == 1540
    assert provider.fit_batch_calls == 1
    assert provider.review_batch_calls == 1


def test_semantic_fit_prefers_approved_natural_text_in_safe_transition(
    tmp_path: Path,
) -> None:
    first = SpeechBlock(
        block_id="block_0001_variant_natural",
        speaker_id="SPEAKER_00",
        start_ms=0,
        end_ms=2380,
        segment_ids=("unit_0001",),
        source_text="There is one thing I can tell you for sure.",
        translated_text="Kukhona into eyodwa engingaqiniseka ngayo.",
        tts_text="Kukhona into eyodwa engingaqiniseka ngayo.",
        required_facts=("There is one thing the speaker is sure about",),
        source_speech_start_ms=0,
        source_speech_end_ms=1880,
    )
    second = SpeechBlock(
        block_id="block_0002_variant_natural",
        speaker_id="SPEAKER_00",
        start_ms=4920,
        end_ms=6500,
        segment_ids=("unit_0002",),
        source_text="The following sentence.",
        translated_text="Umusho olandelayo.",
        tts_text="Umusho olandelayo.",
    )
    backend = _MeasuredBackend(
        {
            first.translated_text: 3400,
            second.translated_text: 1400,
        }
    )
    provider = _SemanticBatchProvider({})
    renderer = DirectAzureDubRenderer(
        SimpleNamespace(work_dir=tmp_path),
        backend,
        timing_repair_provider_factory=lambda: provider,
    )
    artifacts = DirectDubArtifacts(tmp_path / "jobs" / "job-1")
    artifacts.semantic_fit.parent.mkdir(parents=True, exist_ok=True)
    artifacts.semantic_fit.write_text(
        json.dumps(
            {
                "schema_version": SEMANTIC_FIT_SCHEMA_VERSION,
                "job_id": "job-1",
                "cases": {
                    "block_0001": {
                        "status": "exhausted",
                        "attempt_limit": 12,
                        "attempts": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    results = renderer._synthesize_semantic_fit_batch(
        (first, second),
        artifacts,
        synthesis_profiles={
            0: ("zu-ZA-ThandoNeural", 0, 0, 0),
            1: ("zu-ZA-ThandoNeural", 0, 0, 0),
        },
        options=DirectDubOptions(timing_mode=TIMING_MODE_SEMANTIC_FIT),
        force=False,
        progress=_DirectDubProgress(None),
    )

    selected, fitted, _audits = results[0]
    assert selected.translated_text == first.translated_text
    assert selected.end_ms == 3400
    assert fitted["semantic_fit"]["same_speaker_transition_borrow_ms"] == 1020
    assert fitted["semantic_fit"]["same_speaker_transition_borrow_limit_ms"] == 1270
    assert fitted["semantic_fit"]["search_block_end_ms"] == 3650
    assert fitted["semantic_fit"]["selected_block_end_ms"] == 3400
    assert provider.fit_batch_calls == 0
    assert provider.review_batch_calls == 0
    checkpoint = json.loads(artifacts.semantic_fit.read_text(encoding="utf-8"))
    assert checkpoint["cases"]["block_0001"]["status"] == "selected"
    assert checkpoint["cases"]["block_0001"]["selected_attempt"] == 0


def test_long_source_pauses_fill_and_calibrate_semantic_window(
    tmp_path: Path,
) -> None:
    text = "a b c d e f g h i j k l m n o p"
    backend = _PauseOvershootBackend({text: 13662}, pause_overhead_ms=525)
    renderer = DirectAzureDubRenderer(SimpleNamespace(work_dir=tmp_path), backend)
    block = SpeechBlock(
        block_id="block_0001_variant_natural",
        speaker_id="SPEAKER_00",
        start_ms=0,
        end_ms=20562,
        segment_ids=("unit_0001",),
        source_text="One slow sentence with seven long pauses.",
        translated_text=text,
        tts_text=text,
        source_pause_anchors=tuple(
            {
                "source_progress": progress,
                "source_gap_ms": 1100,
                "requested_pause_ms": 1000,
            }
            for progress in (0.1, 0.23, 0.36, 0.5, 0.64, 0.77, 0.9)
        ),
        source_speech_start_ms=0,
        source_speech_end_ms=20562,
    )

    fitted = renderer._synthesize_locked_tempo_block(
        block,
        DirectDubArtifacts(tmp_path / "jobs" / "job-1"),
        voice="zu-ZA-ThandoNeural",
        base_rate_percent=0,
        pitch_percent=0,
        volume_percent=0,
        force=False,
        timing_deadline_ms=20562,
    )

    assert fitted["duration_ms"] == 20537
    assert 20312 <= _selected_azure_measurement_ms(fitted) <= 20562
    assert fitted["source_pause_alignment"]["total_added_pause_ms"] == 6350
    assert fitted["source_pause_alignment"]["calibration_reduction_ms"] == 447


def test_direct_backend_uses_ten_second_source_pause_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AZURE_SPEECH_KEY", "test-key")
    settings = SimpleNamespace(
        azure_speech_region="southafricanorth",
        azure_rate_min_percent=-12,
        azure_tts_voices=("zu-ZA-ThandoNeural",),
        azure_tts_default_voice="zu-ZA-ThandoNeural",
        azure_tts_sample_rate=24_000,
        azure_tts_channels=1,
        azure_tts_sample_width=2,
        azure_tts_timeout_seconds=120,
        azure_tts_max_retries=3,
    )

    backend = create_direct_azure_backend(settings, max_rate_percent=15)

    assert backend.ssml_bounds.total_pause_max_ms == 10_000


def test_semantic_fit_accepts_isizulu_prefix_on_multiword_identity() -> None:
    block = SpeechBlock(
        block_id="block_0001_variant_natural",
        speaker_id="SPEAKER_00",
        start_ms=0,
        end_ms=11220,
        segment_ids=("unit_0001",),
        source_text="Fadiel Adams will appear at the Madlanga Commission.",
        translated_text=(
            "UFadiel Adams uzovela e-Madlanga Commission."
        ),
        tts_text="UFadiel Adams uzovela e-Madlanga Commission.",
        protected_entities=("Madlanga Commission", "Fadiel Adams"),
    )
    candidate = {
        "unit": {
            "unit_id": "block_0001",
            "spoken_text": (
                "Uvela kusasa e-Madlanga Commission. "
                "Ake nginitshele ngoFadiel Adams."
            ),
            "tts_text": (
                "Uvela kusasa e-Madlanga Commission. "
                "Ake nginitshele ngoFadiel Adams."
            ),
            "numbers_preserved": True,
            "dates_preserved": True,
            "negation_preserved": True,
            "omitted_or_compressed_detail": [],
        }
    }

    spoken, tts, omissions = (
        DirectAzureDubRenderer._validate_finalization_candidate(
            block,
            candidate,
        )
    )

    assert "e-Madlanga Commission" in spoken
    assert "ngoFadiel Adams" in tts
    assert omissions == ()


def test_semantic_fit_still_rejects_missing_multiword_identity() -> None:
    block = SpeechBlock(
        block_id="block_0001_variant_natural",
        speaker_id="SPEAKER_00",
        start_ms=0,
        end_ms=2000,
        segment_ids=("unit_0001",),
        source_text="Madlanga Commission",
        translated_text="iKhomishini kaMadlanga",
        tts_text="iKhomishini kaMadlanga",
        protected_entities=("Madlanga Commission",),
    )
    candidate = {
        "unit": {
            "unit_id": "block_0001",
            "spoken_text": "Uvela kusasa ekhomishini.",
            "tts_text": "Uvela kusasa ekhomishini.",
            "numbers_preserved": True,
            "dates_preserved": True,
            "negation_preserved": True,
            "omitted_or_compressed_detail": [],
        }
    }

    with pytest.raises(ValueError, match="lost protected entities"):
        DirectAzureDubRenderer._validate_finalization_candidate(
            block,
            candidate,
        )


def test_semantic_fit_restart_measures_cached_candidate_without_new_ai_call(
    tmp_path: Path,
) -> None:
    approved = (
        "Bazozama yonke into ukuze bangiwise. "
        "UFadile uvela kusasa e-Madlanga Commission. "
        "Ake nginitshele ngoFadiel Adams."
    )
    cached_spoken = (
        "Bazozama konke ukungiwisa. "
        "UFadile uvela kusasa e-Madlanga Commission. "
        "Ake nginitshele ngoFadiel Adams."
    )
    block = SpeechBlock(
        block_id="block_0001_variant_natural",
        speaker_id="SPEAKER_00",
        start_ms=0,
        end_ms=11220,
        segment_ids=("unit_0001",),
        source_text=(
            "They will try everything to bring me down. Fadiel Adams will "
            "appear at the Madlanga Commission."
        ),
        translated_text=approved,
        tts_text=approved,
        protected_entities=("Madlanga Commission", "Fadiel Adams"),
        required_facts=("They will try to bring the speaker down",),
        source_speech_start_ms=0,
        source_speech_end_ms=10720,
    )
    backend = _MeasuredBackend({approved: 12500, cached_spoken: 10800})
    provider = _SemanticProvider()
    renderer = DirectAzureDubRenderer(
        SimpleNamespace(work_dir=tmp_path),
        backend,
        timing_repair_provider_factory=lambda: provider,
    )
    artifacts = DirectDubArtifacts(tmp_path / "jobs" / "job-1")
    initial_attempt = {
        "attempt": 0,
        "candidate_id": "approved_natural",
        "status": "overflow",
        "spoken_text": approved,
        "measured_duration_ms": 12500,
        "target_minimum_ms": 10470,
        "target_preferred_ms": 10720,
        "target_maximum_ms": 11220,
        "speaker_rate_percent": 0,
        "time_stretch_ratio": 1.0,
        "semantic_review": "approved_translation",
    }
    payload = renderer._semantic_fit_payload(
        block,
        voice="zu-ZA-ThembaNeural",
        base_rate_percent=0,
        pitch_percent=0,
        volume_percent=0,
        attempt_number=1,
        target_minimum_ms=10470,
        target_preferred_ms=10720,
        target_maximum_ms=11220,
        duration_evidence=[initial_attempt],
        rejected_spoken_texts=[approved],
        current_spoken_text=approved,
    )
    fingerprint = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    cached_candidate = {
        "schema_version": "claude-turn-repair-v1",
        "unit": {
            "unit_id": "block_0001",
            "faithful_translation": approved,
            "spoken_text": cached_spoken,
            "tts_text": cached_spoken,
            "language_features": [],
            "human_review_flags": [],
            "numbers_preserved": True,
            "dates_preserved": True,
            "negation_preserved": True,
            "omitted_or_compressed_detail": [],
        },
    }
    artifacts.semantic_fit.parent.mkdir(parents=True, exist_ok=True)
    artifacts.semantic_fit.write_text(
        json.dumps(
            {
                "schema_version": SEMANTIC_FIT_SCHEMA_VERSION,
                "job_id": "job-1",
                "cases": {
                    "block_0001": {
                        "status": "exhausted",
                        "speaker_id": "SPEAKER_00",
                        "voice": "zu-ZA-ThembaNeural",
                        "speaker_rate_percent": 0,
                        "attempts": [
                            {
                                "attempt": 1,
                                "request_fingerprint": fingerprint,
                                "candidate": cached_candidate,
                            }
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    selected_block, fitted, attempts = renderer._synthesize_semantic_fit(
        block,
        artifacts,
        voice="zu-ZA-ThembaNeural",
        base_rate_percent=0,
        pitch_percent=0,
        volume_percent=0,
        options=DirectDubOptions(timing_mode=TIMING_MODE_SEMANTIC_FIT),
        force=False,
        progress=_DirectDubProgress(None),
        block_number=1,
        block_total=1,
    )

    assert selected_block.translated_text == cached_spoken
    assert fitted["semantic_fit"]["selected_attempt"] == 1
    assert attempts[-1]["ai_called"] is False
    assert provider.fit_calls == 0
    assert provider.review_calls == 1
