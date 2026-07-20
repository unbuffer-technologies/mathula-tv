import math
import wave
from pathlib import Path

import pytest

from mathula_tv.alignment import align_converted_unit, measured_openvoice_ratios, timing_decision
from mathula_tv.background import (
    prepare_background,
    require_publication_background,
    residual_dialogue_qc,
)
from mathula_tv.consent import build_voice_rights_review, require_generation_rights
from mathula_tv.errors import ResidualEnglishDetected, TimingBoundExceeded
from mathula_tv.mixing import overlap_decisions
from mathula_tv.qc import retranscription_comparison, review_readiness
from mathula_tv.references import select_reference_candidates
from mathula_tv.subtitles import write_subtitles
from mathula_tv.voice_selection import rank_voice_candidates


def wav(path: Path, duration_ms=500, *, rate=24000, channels=1, amplitude=1000):
    frames = round(rate * duration_ms / 1000)
    raw = bytearray()
    for index in range(frames):
        value = round(amplitude * math.sin(index / 8))
        for _ in range(channels):
            raw.extend(value.to_bytes(2, "little", signed=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setparams((channels, 2, rate, 0, "NONE", "not compressed"))
        handle.writeframes(raw)
    return path


def unit(preferred=700, maximum=800):
    return {
        "unit_id": "unit_0001",
        "speaker_id": "AZURE_1",
        "preferred_duration_ms": preferred,
        "maximum_duration_ms": maximum,
    }


def test_short_audio_is_padded_but_never_slowed(tmp_path):
    source = wav(tmp_path / "source.wav", 500)
    result = align_converted_unit(source, tmp_path / "aligned.wav", unit())
    assert result["speed"] == 1.0
    assert result["final_duration_ms"] == 700
    assert result["silence_padding_ms"] == 200


def test_excessive_timing_correction_requests_translation_repair(tmp_path):
    source = wav(tmp_path / "source.wav", 1000)
    with pytest.raises(TimingBoundExceeded, match="requires speed"):
        align_converted_unit(source, tmp_path / "aligned.wav", unit(), max_speed=1.06)
    assert timing_decision(1000, unit(), max_speed=1.06)["repair_required"] is True


def test_openvoice_ratio_prediction_is_per_speaker():
    ratios = measured_openvoice_ratios(
        [
            {"speaker_id": "A", "source_duration_ms": 1000, "converted_duration_ms": 1100},
            {"speaker_id": "A", "source_duration_ms": 2000, "converted_duration_ms": 2100},
            {"speaker_id": "B", "source_duration_ms": 1000, "converted_duration_ms": 900},
        ]
    )
    assert ratios == {"A": pytest.approx(1.075), "B": pytest.approx(0.9)}


class Detector:
    def __init__(self, values):
        self.values = values

    def detect(self, audio):
        return self.values


def test_background_modes_and_residual_english_gate(tmp_path):
    source = wav(tmp_path / "mix.wav", rate=48000, channels=2)
    clean = wav(tmp_path / "clean.wav", rate=48000, channels=2)
    manifest = prepare_background(source, tmp_path / "background.wav", source_dialogue=[], clean_stem=clean)
    assert manifest["mode"] == "clean_background_stem" and manifest["publication_candidate"]
    qc = residual_dialogue_qc(
        tmp_path / "background.wav",
        "Important source allegation",
        Detector([{"start": 0, "end": 1.2, "text": "source allegation", "confidence": 0.9}]),
    )
    assert qc["status"] == "needs_background_review"
    with pytest.raises(ResidualEnglishDetected):
        require_publication_background(manifest, qc)


def test_review_ducking_is_never_publication_background(tmp_path):
    source = wav(tmp_path / "mix.wav", rate=48000, channels=2)
    manifest = prepare_background(
        source,
        tmp_path / "background.wav",
        source_dialogue=[{"start": 0.1, "end": 0.3}],
        allow_review_ducking=True,
    )
    assert manifest["mode"] == "original_mix_ducked_review_only"
    with pytest.raises(ResidualEnglishDetected, match="Review-only"):
        require_publication_background(manifest, {"passed": True})


def test_subtitles_use_spoken_text_not_tts_respelling(tmp_path):
    manifest = write_subtitles(
        [
            {
                "unit_id": "unit_0001",
                "speaker_id": "A",
                "start_ms": 100,
                "spoken_text": "UFeroz Khan ufikile.",
                "tts_text": "UFeh-rohz Kaan ufikile.",
                "has_overlap": False,
            }
        ],
        [{"unit_id": "unit_0001", "final_duration_ms": 900}],
        tmp_path,
    )
    srt = Path(manifest["artifacts"]["srt"]["path"]).read_text()
    assert "UFeroz Khan" in srt and "Feh-rohz" not in srt


def test_reference_selection_rejects_cross_speaker_overlap_and_unknown(tmp_path):
    values = [
        {"segment_id": "wrong", "speaker_id": "B", "start": 0, "end": 30, "path": str(tmp_path / "x"), "audio_source_type": "isolated_dialogue"},
        {"segment_id": "overlap", "speaker_id": "A", "start": 0, "end": 30, "path": str(tmp_path / "x"), "audio_source_type": "isolated_dialogue", "overlap": True},
        {"segment_id": "one", "speaker_id": "A", "start": 0, "end": 12, "path": str(tmp_path / "x"), "audio_source_type": "clean_source_segment", "quality_score": .9},
        {"segment_id": "two", "speaker_id": "A", "start": 12, "end": 24, "path": str(tmp_path / "x"), "audio_source_type": "clean_source_segment", "quality_score": .8},
    ]
    selected = select_reference_candidates("A", values)
    assert [item["segment_id"] for item in selected["selected"]] == ["one", "two"]
    assert selected["publication_reference"] is True
    assert {item["segment_id"] for item in selected["rejected"]} == {"wrong", "overlap"}


def test_voice_selection_records_all_candidates_and_override():
    candidates = [
        {"voice": "Thando", "speaker_similarity": .8, "word_error_approximation": .1, "duration_drift_ratio": 1.01, "audio_artifact_count": 0},
        {"voice": "Themba", "speaker_similarity": .7, "word_error_approximation": .05, "duration_drift_ratio": 1.0, "audio_artifact_count": 0},
    ]
    automatic = rank_voice_candidates(candidates, allowed_voices=["Thando", "Themba"])
    override = rank_voice_candidates(candidates, allowed_voices=["Thando", "Themba"], human_override="Themba")
    assert automatic["selected_voice"] == "Thando"
    assert override["selected_voice"] == "Themba" and override["selection_method"] == "human_override"


def test_retranscription_and_readiness_are_truthful():
    comparison = retranscription_comparison("Akukho 25 ANC", "Kukhona 20 ANC", protected_terms=["ANC"])
    assert comparison["protected_terms_passed"]
    assert not comparison["numbers_preserved"] and not comparison["negation_preserved"]
    assert review_readiness({"compatibility": {"state": "pending"}}) == "compatibility_pending"
    assert review_readiness({"compatibility": {"state": "passed"}, "background_review_required": True}) == "needs_background_review"
    assert review_readiness({"compatibility": {"state": "passed"}}) == "review_ready"


def test_voice_rights_do_not_invent_consent():
    review = build_voice_rights_review([{"speaker_id": "A"}], intended_use="internal review")
    assert not review["generation_allowed"] and not review["publication_approved"]
    with pytest.raises(PermissionError):
        require_generation_rights(review)
    proof = build_voice_rights_review([{"speaker_id": "A"}], intended_use="internal review", proof_of_concept=True)
    assert proof["generation_allowed"] and not proof["publication_approved"]


def test_overlap_is_preserved_across_speaker_tracks():
    decisions = overlap_decisions(
        [
            {"unit_id": "u1", "speaker_id": "A", "start_ms": 0, "duration_ms": 1000},
            {"unit_id": "u2", "speaker_id": "B", "start_ms": 500, "duration_ms": 1000},
        ]
    )
    assert decisions == [
        {
            "left_unit_id": "u1",
            "right_unit_id": "u2",
            "start_ms": 500,
            "end_ms": 1000,
            "duration_ms": 500,
            "placement": "preserved_on_separate_speaker_tracks",
            "gain_management": "limiter",
            "human_review_required": False,
        }
    ]
