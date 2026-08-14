from __future__ import annotations

from pathlib import Path

import mathula_tv.direct_azure_dub as direct_dub
from mathula_tv.direct_azure_dub import (
    DirectAzureDubRenderer,
    DirectDubArtifacts,
    DirectDubOptions,
    SpeechBlock,
    TimingOverflowError,
    rebalance_five_block_timeline,
)


def _block(number: int, start_ms: int, duration_ms: int) -> SpeechBlock:
    return SpeechBlock(
        block_id=f"block_{number:04d}_variant_compact",
        speaker_id=f"SPEAKER_{number % 2:02d}",
        start_ms=start_ms,
        end_ms=start_ms + duration_ms,
        segment_ids=(f"unit_{number:04d}",),
        source_text=f"source {number}",
        translated_text=f"translated {number}",
        tts_text=f"translated {number}",
        selected_variant_id="compact",
    )


def test_exact_job_uses_second_preceding_block_capacity() -> None:
    durations = [57_820, 4_960, 17_840, 33_520, 32_800]
    starts = [463_030, 522_430, 527_390, 545_230, 578_750]
    blocks = [
        _block(number, start, duration)
        for number, start, duration in zip(range(49, 54), starts, durations)
    ]

    required = {
        0: 52_063,  # block_0049: 53,624ms Azure / 1.03
        1: 4_816,   # block_0050: 4,960ms Azure / 1.03
        2: 20_731,  # block_0051: 21,352ms Azure / 1.03
        3: 32_321,  # block_0052: 33,290ms Azure / 1.03
        4: 32_177,  # block_0053: 33,142ms Azure / 1.03
    }

    adjusted, audits = rebalance_five_block_timeline(blocks, required, [2])

    assert audits == [
        {
            **audits[0],
            "applied": True,
            "reason": "five_block_measured_capacity_resolved_overflow",
        }
    ]
    assert adjusted[2].duration_ms == 20_731
    assert adjusted[2].start_ms == 525_666
    assert adjusted[2].end_ms == 546_397
    assert sum(
        item["released_ms"] for item in audits[0]["reductions"]
    ) == 2_891
    assert audits[0]["source_silence_consumed_ms"] == 1_580
    assert adjusted[0].start_ms == blocks[0].start_ms
    assert adjusted[-1].end_ms == blocks[-1].end_ms
    assert all(
        adjusted[index].duration_ms >= required[index]
        for index in range(5)
    )
    assert all(
        adjusted[index].end_ms <= adjusted[index + 1].start_ms
        for index in range(4)
    )
    assert audits[0]["human_review_required"] is False


def test_five_block_solver_fails_closed_when_total_capacity_is_insufficient() -> None:
    blocks = []
    cursor = 0
    for number in range(1, 6):
        blocks.append(_block(number, cursor, 1_000))
        cursor += 1_000

    adjusted, audits = rebalance_five_block_timeline(
        blocks,
        {0: 1_000, 1: 1_000, 2: 2_000, 3: 1_000, 4: 1_000},
        [2],
    )

    assert adjusted == blocks
    assert audits[0]["applied"] is False
    assert audits[0]["reason"] == "insufficient_five_block_measured_capacity"
    assert audits[0]["remaining_shortfall_ms"] == 1_000


def test_finalization_accepts_reviewed_isizulu_identity_alias(monkeypatch) -> None:
    monkeypatch.setattr(
        direct_dub,
        "identity_requirements_for_unit",
        lambda *_args, **_kwargs: [
            {
                "required_span": "Madlanga Commission of Inquiry",
                "policy": "contextual_identity",
                "accepted_forms": [
                    "Madlanga Commission of Inquiry",
                    "iKhomishani",
                ],
            }
        ],
    )
    block = SpeechBlock(
        block_id="block_0051_variant_compact",
        speaker_id="SPEAKER_02",
        start_ms=527_390,
        end_ms=545_230,
        segment_ids=("unit_0108",),
        source_text="I emailed the Commission and was given the 11th.",
        translated_text="Ngithumele i-imeyili eKhomishani. Nganikwa i-11.",
        tts_text="Ngithumele i-imeyili eKhomishani. Nganikwa i-11.",
        protected_entities=("Madlanga Commission of Inquiry",),
    )
    candidate = {
        "unit": {
            "unit_id": "block_0051",
            "spoken_text": "Ngithumele i-imeyili eKhomishani. Nganikwa i-11.",
            "tts_text": "Ngithumele i-imeyili eKhomishani. Nganikwa i-11.",
            "numbers_preserved": True,
            "dates_preserved": True,
            "negation_preserved": True,
            "omitted_or_compressed_detail": [],
        }
    }

    spoken, tts, omissions = DirectAzureDubRenderer._validate_finalization_candidate(
        block, candidate
    )

    assert "eKhomishani" in spoken
    assert tts == spoken
    assert omissions == ()


class _Progress:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict]] = []

    def emit(self, stage: str, message: str, **details) -> None:
        self.events.append((stage, message, details))


def test_final_product_fallback_never_requests_manual_review(tmp_path: Path) -> None:
    renderer = object.__new__(DirectAzureDubRenderer)
    block = _block(51, 0, 17_840)
    captured: dict = {}

    def synthesize(candidate, *_args, **kwargs):
        captured.update(kwargs)
        return {
            "post_synthesis_time_stretch_ratio": 1.196861,
            "cadence_qc": {"passed": True},
        }

    renderer._synthesize_block = synthesize
    progress = _Progress()
    fitted = renderer._force_final_product_fit(
        block=block,
        artifacts=DirectDubArtifacts(tmp_path),
        voice="zu-ZA-ThembaNeural",
        base_rate_percent=3,
        pitch_percent=0,
        volume_percent=0,
        options=DirectDubOptions(),
        overflow=TimingOverflowError(
            "overflow",
            measured_duration_ms=21_352,
            target_duration_ms=17_840,
            stretch_ratio=21_352 / 17_840,
        ),
        five_block_window=[],
        repair_audit=[],
        progress=progress,
    )

    assert captured["max_time_stretch_ratio"] >= 2.0
    assert fitted["timing_adjustment"]["method"] == "final_product_emergency_fit"
    assert fitted["timing_adjustment"]["human_review_required"] is False
    assert progress.events[-1][2]["final_product_guaranteed"] is True


def test_render_path_no_longer_calls_blocking_collision_panel() -> None:
    source = Path(direct_dub.__file__).read_text(encoding="utf-8")
    assert "self._pause_for_timeline_collision_review(" not in source
