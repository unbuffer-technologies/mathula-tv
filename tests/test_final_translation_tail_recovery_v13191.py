from __future__ import annotations

import pytest

from mathula_tv.direct_azure_dub import DirectDubError, load_approved_segments


def _unit(
    unit_id: str,
    *,
    start_ms: int,
    end_ms: int,
    source_text: str = "Source speech",
) -> dict[str, object]:
    return {
        "unit_id": unit_id,
        "speaker_id": "SPEAKER_00",
        "start_ms": start_ms,
        "end_ms": end_ms,
        "source_text": source_text,
        "translated_text": "Inkulumo egunyaziwe",
        "tts_text": "Inkulumo egunyaziwe",
    }


def test_exact_job_final_unit_tail_drift_is_clamped_and_audited() -> None:
    audits: list[dict[str, object]] = []

    segments = load_approved_segments(
        {
            "units": [
                _unit("unit_0073", start_ms=435_913, end_ms=439_630),
                _unit("unit_0074", start_ms=439_630, end_ms=445_320),
            ]
        },
        source_duration_ms=445_057,
        boundary_normalizations=audits,
    )

    assert segments[-1].segment_id == "unit_0074"
    assert segments[-1].end_ms == 445_057
    assert segments[-1].duration_ms == 5_427
    assert audits == [
        {
            "schema_version": (
                "mathula-approved-segment-boundary-normalization-v1"
            ),
            "policy_version": "final-speakable-tail-drift-v1-v13.18.11",
            "segment_id": "unit_0074",
            "final_speakable_unit": True,
            "original_start_ms": 439_630,
            "original_end_ms": 445_320,
            "normalized_end_ms": 445_057,
            "source_duration_ms": 445_057,
            "original_duration_ms": 5_690,
            "normalized_duration_ms": 5_427,
            "overrun_ms": 263,
            "trim_ratio": 0.046221,
            "recovery_mode": "bounded_final_speakable_tail",
            "approved_text_changed": False,
            "translation_artifact_changed": False,
        }
    ]


def test_same_overrun_on_an_interior_unit_still_fails_closed() -> None:
    with pytest.raises(DirectDubError, match="bounded drift on the final"):
        load_approved_segments(
            {
                "units": [
                    _unit("unit_0001", start_ms=8_500, end_ms=10_263),
                    _unit("unit_0002", start_ms=9_000, end_ms=9_900),
                ]
            },
            source_duration_ms=10_000,
        )


def test_large_final_overrun_still_fails_closed() -> None:
    with pytest.raises(DirectDubError, match="ends beyond the source video"):
        load_approved_segments(
            {"units": [_unit("unit_0001", start_ms=8_000, end_ms=10_501)]},
            source_duration_ms=10_000,
        )


def test_final_overrun_cannot_remove_more_than_ten_percent_of_phrase() -> None:
    with pytest.raises(DirectDubError, match="bounded drift on the final"):
        load_approved_segments(
            {"units": [_unit("unit_0001", start_ms=9_700, end_ms=10_263)]},
            source_duration_ms=10_000,
        )


def test_trailing_punctuation_unit_does_not_hide_final_speakable_unit() -> None:
    audits: list[dict[str, object]] = []

    segments = load_approved_segments(
        {
            "units": [
                _unit("unit_0074", start_ms=4_000, end_ms=10_263),
                _unit(
                    "unit_0075",
                    start_ms=10_263,
                    end_ms=10_400,
                    source_text=".......",
                ),
            ]
        },
        source_duration_ms=10_000,
        boundary_normalizations=audits,
    )

    assert [segment.segment_id for segment in segments] == ["unit_0074"]
    assert segments[0].end_ms == 10_000
    assert audits[0]["final_speakable_unit"] is True
    assert audits[0]["recovery_mode"] == "bounded_final_speakable_tail"
