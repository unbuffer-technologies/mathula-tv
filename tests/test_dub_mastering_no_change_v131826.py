from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from mathula_tv.atomic_io import atomic_write_json, read_json
from mathula_tv.direct_azure_dub import (
    DUB_MASTERING_OVERRIDE_SCHEMA_VERSION,
    DirectAzureDubRenderer,
    PreparedTranslationVariant,
    SpeechBlock,
    apply_dub_mastering_overrides,
)
from mathula_tv.dub_mastering import (
    DubMasteringLoop,
    build_mastering_override_plan,
    delivery_word_error_rate,
)


def _speech_block() -> SpeechBlock:
    variants = tuple(
        PreparedTranslationVariant(
            variant_id=variant_id,
            spoken_text=f"{variant_id} spoken text",
            tts_text=f"{variant_id} spoken text",
            estimated_duration_ms=duration,
        )
        for variant_id, duration in (
            ("natural", 2800),
            ("concise", 2200),
            ("compact", 1700),
        )
    )
    return SpeechBlock(
        block_id="block_0042",
        speaker_id="SPEAKER_01",
        start_ms=0,
        end_ms=3000,
        segment_ids=("unit_0042",),
        source_text="source",
        translated_text="natural spoken text",
        tts_text="natural spoken text",
        variants=variants,
    )


def test_mastering_variant_is_locked_through_predicted_fit_selection() -> None:
    payload = {
        "schema_version": DUB_MASTERING_OVERRIDE_SCHEMA_VERSION,
        "revision": 1,
        "overrides": [
            {
                "block_id": "block_0042",
                "strategy": "select_variant",
                "variant_id": "compact",
            }
        ],
    }

    adjusted, _ = apply_dub_mastering_overrides([_speech_block()], payload)
    candidates = DirectAzureDubRenderer._variant_candidates(adjusted[0])

    assert adjusted[0].mastering_variant_id == "compact"
    assert [value.variant_id for value in candidates] == ["compact"]
    assert len(adjusted[0].variants) == 3


def test_repair_plan_skips_audio_equivalent_and_repeated_overrides() -> None:
    block = {
        "block_id": "block_0055",
        "block_index": 55,
        "production_gate_passed": False,
        "production_blockers": ["high_word_error_rate"],
        "selected_variant_id": "compact",
        "available_variant_ids": ["natural", "concise", "compact"],
        "available_variants": [
            {"variant_id": "natural", "audio_equivalent_to_selected": True},
            {"variant_id": "concise", "audio_equivalent_to_selected": True},
            {"variant_id": "compact", "audio_equivalent_to_selected": True},
        ],
        "word_error_rate": 0.6,
    }
    plan = build_mastering_override_plan(
        job_id="job",
        audit={"round": 0, "blocks": [block]},
    )
    assert plan["new_override_count"] == 0

    block["available_variants"] = []
    previous = {
        "revision": 1,
        "overrides": [
            {
                "block_id": "block_0055",
                "strategy": "select_variant",
                "variant_id": "concise",
            }
        ],
    }
    repeated = build_mastering_override_plan(
        job_id="job",
        audit={"round": 1, "blocks": [block]},
        previous_payload=previous,
    )
    assert repeated["new_override_count"] == 0


def test_delivery_wer_accepts_joined_zulu_date_tokens() -> None:
    assert delivery_word_error_rate(
        "mhla ziyishumi nesishiyagalolunye kuNcwaba Sihlalo",
        "Mhlaziyishumi nesishiyagalolunye kungcwaba sihlalo",
    ) == 0


class _Jobs:
    def __init__(self, root: Path):
        self.root = root

    def job_dir(self, job_id: str) -> Path:
        return self.root / job_id


class _NoChangeRenderer:
    def __init__(self, root: Path):
        self.jobs = _Jobs(root)
        self.render_calls = 0

    def render(self, *args, **kwargs):
        self.render_calls += 1
        return {}


class _BadSTT:
    provider = "fake-azure-stt"
    request_duration_seconds = 0.1

    def __init__(self):
        self.calls = 0

    def transcribe(self, audio: Path, locale: str):
        self.calls += 1
        return {
            "durationMilliseconds": 3000,
            "combinedPhrases": [{"text": "amagama angafani nhlobo lapha"}],
            "phrases": [
                {
                    "text": "amagama angafani nhlobo lapha",
                    "offsetMilliseconds": 0,
                    "durationMilliseconds": 2500,
                    "words": [
                        {
                            "text": value,
                            "offsetMilliseconds": index * 500,
                            "durationMilliseconds": 300,
                        }
                        for index, value in enumerate(
                            ("amagama", "angafani", "nhlobo", "lapha")
                        )
                    ],
                }
            ],
        }


def test_loop_stops_before_retranscribing_an_identical_repair_render(
    tmp_path: Path,
) -> None:
    job_id = "job"
    direct_root = tmp_path / job_id / "direct_dub"
    direct_root.mkdir(parents=True)
    audio = direct_root / "final_mix.wav"
    audio.write_bytes(b"unchanged audio")
    block = {
        "block_id": "block_0001_variant_concise",
        "start_ms": 0,
        "end_ms": 3000,
        "speaker_id": "SPEAKER_01",
        "translated_text": "concise expected delivery here",
        "tts_text": "concise expected delivery here",
        "protected_entities": [],
        "translation_variant": {
            "selected_variant_id": "concise",
            "spoken_text": "concise expected delivery here",
            "tts_text": "concise expected delivery here",
        },
        "variants": [
            {
                "variant_id": "natural",
                "spoken_text": "natural expected delivery here with detail",
                "tts_text": "natural expected delivery here with detail",
                "meaning_preserved": True,
            },
            {
                "variant_id": "concise",
                "spoken_text": "concise expected delivery here",
                "tts_text": "concise expected delivery here",
                "meaning_preserved": True,
            },
        ],
        "base_prosody": {"rate_percent": 0},
        "selected_azure_rate_percent": 0,
        "post_synthesis_time_stretch_ratio": 1.0,
    }
    atomic_write_json(
        direct_root / "report.json",
        {"blocks": [block], "outputs": {"final_mix": str(audio)}},
    )
    backend = _BadSTT()
    renderer = _NoChangeRenderer(tmp_path)
    result = DubMasteringLoop(
        stt_backend=backend,
        renderer=renderer,
    ).run(
        SimpleNamespace(job_id=job_id, target_language="zu-ZA"),
        options=SimpleNamespace(),
        max_rounds=2,
    )

    assert result["stopped_reason"] == "repair_render_produced_identical_audio"
    assert result["round_count"] == 1
    assert result["rounds"][0]["repair_audio_changed"] is False
    assert backend.calls == 1
    assert renderer.render_calls == 2  # attempted repair, then best-payload restore
    assert read_json(direct_root / "mastering_overrides.json")["overrides"] == []
