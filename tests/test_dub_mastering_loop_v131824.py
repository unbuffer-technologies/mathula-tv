from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from mathula_tv.atomic_io import atomic_write_json
from mathula_tv.direct_azure_dub import (
    DUB_MASTERING_OVERRIDE_SCHEMA_VERSION,
    PreparedTranslationVariant,
    SpeechBlock,
    apply_dub_mastering_overrides,
)
from mathula_tv.dub_mastering import (
    DUB_MASTERING_SCHEMA_VERSION,
    DubMasteringLoop,
    DubMasteringThresholds,
    analyze_mastered_dub,
    build_mastering_override_plan,
    word_error_rate,
)


def _word(text: str, start: float, end: float) -> dict:
    return {"text": text, "start": start, "end": end, "confidence": 0.98}


def _block(
    block_id: str,
    *,
    start_ms: int,
    end_ms: int,
    text: str,
    speaker_id: str = "SPEAKER_01",
    rate: int = 0,
    stretch: float = 1.0,
    selected_variant_id: str = "natural",
) -> dict:
    return {
        "block_id": block_id,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "speaker_id": speaker_id,
        "translated_text": text,
        "tts_text": text,
        "protected_entities": [],
        "selected_variant_id": selected_variant_id,
        "translation_variant": {
            "selected_variant_id": selected_variant_id,
            "spoken_text": text,
        },
        "variants": [
            {
                "variant_id": variant_id,
                "spoken_text": text,
                "tts_text": text,
                "meaning_preserved": True,
            }
            for variant_id in ("natural", "concise", "compact")
        ],
        "base_prosody": {"rate_percent": 0},
        "selected_azure_rate_percent": rate,
        "post_synthesis_time_stretch_ratio": stretch,
    }


def test_word_error_rate_is_word_level_and_unicode_safe() -> None:
    assert word_error_rate("Ngakho sikhulume noMnumzane Adams", "Ngakho sikhulume noMnumzane Adams") == 0
    assert word_error_rate("igama elilodwa", "igama") == 0.5


def test_perfect_render_reaches_production_score(tmp_path: Path) -> None:
    audio = tmp_path / "final_mix.wav"
    audio.write_bytes(b"audio")
    report = {
        "blocks": [
            _block(
                "block_0001_variant_natural",
                start_ms=0,
                end_ms=3000,
                text="Ngakho sikhulume noMnumzane Adams",
            )
        ]
    }
    normalized = {
        "provider": "fake-azure",
        "phrases": [{"text": "Ngakho sikhulume noMnumzane Adams"}],
        "words": [
            _word("Ngakho", 0.1, 0.4),
            _word("sikhulume", 0.5, 0.9),
            _word("noMnumzane", 1.0, 1.5),
            _word("Adams", 1.6, 2.0),
        ],
    }

    audit = analyze_mastered_dub(
        job_id="job",
        direct_dub_report=report,
        normalized_stt=normalized,
        audio_path=audio,
        round_number=0,
    )

    assert audit["production_ready"] is True
    assert audit["production_score"] == 100
    assert audit["aggregate_word_error_rate"] == 0


def test_audit_detects_false_pause_and_artificial_same_speaker_tempo_jump(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "final_mix.wav"
    audio.write_bytes(b"audio")
    report = {
        "blocks": [
            _block(
                "block_0001_variant_natural",
                start_ms=0,
                end_ms=3000,
                text="Ngakho sikhulume noMnumzane Adams",
            ),
            _block(
                "block_0002_variant_natural",
                start_ms=3000,
                end_ms=6000,
                text="Sizobonana ngomhlaka nineteen August",
                rate=10,
                stretch=1.10,
            ),
        ]
    }
    normalized = {
        "provider": "fake-azure",
        "phrases": [],
        "words": [
            _word("Ngakho", 0.1, 0.4),
            _word("sikhulume", 0.5, 0.9),
            _word("noMnumzane", 1.8, 2.1),
            _word("Adams", 2.2, 2.5),
            _word("Sizobonana", 3.1, 3.4),
            _word("ngomhlaka", 3.5, 3.8),
            _word("nineteen", 3.9, 4.2),
            _word("August", 4.3, 4.6),
        ],
    }

    audit = analyze_mastered_dub(
        job_id="job",
        direct_dub_report=report,
        normalized_stt=normalized,
        audio_path=audio,
        round_number=0,
    )

    assert "unexpected_internal_pause" in audit["blocks"][0]["reason_codes"]
    assert "same_speaker_tempo_jump" in audit["blocks"][1]["reason_codes"]
    assert "excessive_time_compression" in audit["blocks"][1]["reason_codes"]
    assert audit["production_ready"] is False


def test_repair_plan_selects_one_change_per_five_block_window() -> None:
    audit = {
        "round": 0,
        "blocks": [
            {
                "block_id": "block_0001",
                "block_index": 0,
                "passed": False,
                "reason_codes": ["excessive_time_compression"],
                "selected_variant_id": "natural",
                "available_variant_ids": ["natural", "concise", "compact"],
                "word_error_rate": 0.1,
                "maximum_internal_pause_ms": 0,
                "same_speaker_tempo_jump": 0.2,
                "post_synthesis_time_stretch_ratio": 1.10,
                "cross_speaker_overlap_ms": 0,
                "effective_tempo_factor": 1.1,
            },
            {
                "block_id": "block_0002",
                "block_index": 1,
                "passed": False,
                "reason_codes": ["cross_speaker_timeline_collision"],
                "selected_variant_id": "natural",
                "available_variant_ids": ["natural", "concise", "compact"],
                "word_error_rate": 0.1,
                "maximum_internal_pause_ms": 0,
                "same_speaker_tempo_jump": 0,
                "post_synthesis_time_stretch_ratio": 1.0,
                "cross_speaker_overlap_ms": 900,
                "effective_tempo_factor": 1.0,
            },
        ],
    }

    plan = build_mastering_override_plan(job_id="job", audit=audit)

    assert plan["new_override_count"] == 1
    assert plan["new_overrides"][0]["block_id"] == "block_0002"
    assert plan["new_overrides"][0]["variant_id"] == "concise"
    assert plan["five_block_neighbourhood_control"] is True


def test_direct_renderer_accepts_only_approved_mastering_variant() -> None:
    variants = tuple(
        PreparedTranslationVariant(
            variant_id=variant_id,
            spoken_text=f"{variant_id} text",
            tts_text=f"{variant_id} text",
        )
        for variant_id in ("natural", "concise", "compact")
    )
    block = SpeechBlock(
        block_id="block_0001",
        speaker_id="SPEAKER_01",
        start_ms=0,
        end_ms=3000,
        segment_ids=("unit_0001",),
        source_text="source",
        translated_text="natural text",
        tts_text="natural text",
        variants=variants,
    )
    payload = {
        "schema_version": DUB_MASTERING_OVERRIDE_SCHEMA_VERSION,
        "revision": 1,
        "overrides": [
            {
                "block_id": "block_0001",
                "strategy": "select_variant",
                "variant_id": "concise",
                "reason_codes": ["excessive_time_compression"],
                "source_round": 0,
            }
        ],
    }

    adjusted, audits = apply_dub_mastering_overrides([block], payload)

    assert adjusted[0].translated_text == "concise text"
    assert adjusted[0].selected_variant_id == "concise"
    assert len(adjusted[0].variants) == 3
    assert adjusted[0].block_id == "block_0001_variant_concise"
    assert audits[0]["authoritative_translation_changed"] is False

    payload["overrides"][0]["variant_id"] = "invented"
    with pytest.raises(Exception, match="unavailable invented variant"):
        apply_dub_mastering_overrides([block], payload)


class _FakeJobs:
    def __init__(self, root: Path):
        self.root = root

    def job_dir(self, job_id: str) -> Path:
        return self.root / job_id


class _FakeRenderer:
    def __init__(self, root: Path):
        self.jobs = _FakeJobs(root)
        self.render_calls = 0

    def render(self, *args, **kwargs):
        self.render_calls += 1
        return {}


class _FakeSTT:
    provider = "fake-azure-stt"
    api_version = "test"
    request_duration_seconds = 0.1

    def __init__(self):
        self.calls = 0

    def transcribe(self, audio: Path, locale: str):
        self.calls += 1
        return {
            "durationMilliseconds": 3000,
            "combinedPhrases": [{"text": "Ngakho sikhulume noMnumzane Adams"}],
            "phrases": [
                {
                    "text": "Ngakho sikhulume noMnumzane Adams",
                    "offsetMilliseconds": 0,
                    "durationMilliseconds": 2500,
                    "words": [
                        {"text": "Ngakho", "offsetMilliseconds": 100, "durationMilliseconds": 300},
                        {"text": "sikhulume", "offsetMilliseconds": 500, "durationMilliseconds": 400},
                        {"text": "noMnumzane", "offsetMilliseconds": 1000, "durationMilliseconds": 500},
                        {"text": "Adams", "offsetMilliseconds": 1600, "durationMilliseconds": 400},
                    ],
                }
            ],
        }


def test_audit_only_loop_transcribes_final_output_and_reuses_stt_cache(
    tmp_path: Path,
) -> None:
    job_id = "job"
    direct_root = tmp_path / job_id / "direct_dub"
    direct_root.mkdir(parents=True)
    audio = direct_root / "final_mix.wav"
    audio.write_bytes(b"audio")
    atomic_write_json(
        direct_root / "report.json",
        {
            "blocks": [
                _block(
                    "block_0001_variant_natural",
                    start_ms=0,
                    end_ms=3000,
                    text="Ngakho sikhulume noMnumzane Adams",
                )
            ],
            "outputs": {"final_mix": str(audio)},
        },
    )
    backend = _FakeSTT()
    loop = DubMasteringLoop(
        stt_backend=backend,
        renderer=_FakeRenderer(tmp_path),
    )
    job = SimpleNamespace(job_id=job_id, target_language="zu-ZA")

    first = loop.run(
        job,
        options=SimpleNamespace(),
        max_rounds=0,
        audit_only=True,
    )
    second = loop.run(
        job,
        options=SimpleNamespace(),
        max_rounds=0,
        audit_only=True,
    )

    assert first["schema_version"] == DUB_MASTERING_SCHEMA_VERSION
    assert first["state"] == "production_ready"
    assert second["rounds"][0]["stt_cache_reused"] is True
    assert backend.calls == 1
