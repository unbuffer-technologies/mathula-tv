from __future__ import annotations

from dataclasses import dataclass

from mathula_tv.direct_azure_dub import (
    DirectAzureDubRenderer,
    DirectDubArtifacts,
    DirectDubOptions,
    PreparedTranslationVariant,
    SpeechBlock,
    TimingOverflowError,
)


@dataclass
class Response:
    data: dict

    def to_artifact(self) -> dict:
        return {"schema_version": "test", "result": self.data}


class Provider:
    def __init__(self) -> None:
        self.calls = 0

    def repair_turn(self, payload, *, output_schema):
        self.calls += 1
        unit_id = payload["unit_id"]

        def candidate(candidate_id: str, text: str, estimate: int) -> dict:
            return {
                "candidate_id": candidate_id,
                "estimated_duration_ms": estimate,
                "compression_strategy": candidate_id,
                "unit": {
                    "unit_id": unit_id,
                    "faithful_translation": "Barry Bateman, we-AfriForum Private Prosecution Unit.",
                    "spoken_text": text,
                    "tts_text": text,
                    "language_features": [],
                    "human_review_flags": [],
                    "protected_entities_found": ["Barry Bateman", "AfriForum"],
                    "protected_entities_preserved": ["Barry Bateman", "AfriForum"],
                    "numbers_preserved": True,
                    "dates_preserved": True,
                    "negation_preserved": True,
                    "quote_attribution": None,
                    "timing_strategy": "finalization_conductor",
                    "delivery_hints": [],
                    "pronunciation_substitutions": [],
                    "omitted_or_compressed_detail": [],
                    "sensitive_claim_flags": [],
                },
            }

        return Response(
            {
                "schema_version": "claude-turn-timing-repair-options-v1",
                "candidates": [
                    candidate(
                        "balanced",
                        "Barry Bateman we-AfriForum, siyabonga.",
                        5_000,
                    ),
                    candidate(
                        "safe",
                        "Barry Bateman we-AfriForum, ngiyabonga.",
                        4_700,
                    ),
                    candidate(
                        "maximum_compression",
                        "Barry Bateman, AfriForum. Siyabonga.",
                        4_300,
                    ),
                ],
            }
        )


class Progress:
    def __init__(self) -> None:
        self.events = []

    def emit(self, stage, message, **details):
        self.events.append((stage, message, details))


def test_conductor_rewrites_only_failed_block_and_reuses_checkpoint(tmp_path) -> None:
    provider = Provider()
    renderer = object.__new__(DirectAzureDubRenderer)
    renderer.timing_repair_provider_factory = lambda: provider
    renderer._timing_repair_provider_instance = None

    def synthesize(candidate, *_args, **_kwargs):
        duration = {
            "ai_balanced_r1": 5_100,
            "ai_safe_r1": 4_800,
            "ai_maximum_compression_r1": 4_400,
        }[candidate.selected_variant_id]
        return {
            "output_path": str(tmp_path / "candidate.wav"),
            "start_ms": candidate.start_ms,
            "duration_ms": duration,
            "selected_azure_rate_percent": 0,
            "azure_attempts": [{"rate_percent": 0, "duration_ms": duration}],
            "cadence_qc": {"passed": True},
        }

    renderer._synthesize_block = synthesize
    block = SpeechBlock(
        block_id="block_0015_variant_compact",
        speaker_id="SPEAKER_00",
        start_ms=439_630,
        end_ms=445_057,
        segment_ids=("unit_0074",),
        source_text="Barry Bateman, AfriForum's Private Prosecution Unit. Thank you.",
        translated_text=(
            "Barry Bateman, we-AfriForum Private Prosecution Unit. Siyabonga."
        ),
        tts_text=(
            "Barry Bateman, we-AfriForum Private Prosecution Unit. Siyabonga."
        ),
        protected_entities=("Barry Bateman", "AfriForum"),
        required_facts=("Barry Bateman is thanked for joining",),
        variants=(
            PreparedTranslationVariant(
                "compact",
                "Barry Bateman, we-AfriForum Private Prosecution Unit. Siyabonga.",
                "Barry Bateman, we-AfriForum Private Prosecution Unit. Siyabonga.",
            ),
        ),
        selected_variant_id="compact",
    )
    artifacts = DirectDubArtifacts(tmp_path)
    progress = Progress()
    overflow = TimingOverflowError(
        "overflow",
        measured_duration_ms=6_289,
        target_duration_ms=5_427,
        stretch_ratio=6_289 / 5_427,
    )
    kwargs = dict(
        block=block,
        block_index=14,
        block_count=15,
        artifacts=artifacts,
        voice="zu-ZA-ThembaNeural",
        base_rate_percent=0,
        pitch_percent=0,
        volume_percent=0,
        options=DirectDubOptions(),
        latest_overflow=overflow,
        duration_evidence=[],
        progress=progress,
    )

    selected, fitted, audits = renderer._attempt_finalization_conductor(**kwargs)
    assert selected.selected_variant_id == "ai_balanced_r1"
    assert fitted["timing_adjustment"]["method"] == "finalization_ai_conductor"
    assert audits[0]["status"] == "selected"
    assert provider.calls == 1

    renderer._timing_repair_provider_instance = None
    selected_again, _, _ = renderer._attempt_finalization_conductor(**kwargs)
    assert selected_again.selected_variant_id == "ai_balanced_r1"
    assert provider.calls == 1
