from __future__ import annotations

import json
import wave
from dataclasses import replace
from pathlib import Path

import pytest

from mathula_tv.compatibility import (
    REQUIRED_HUMAN_REVIEWS,
    CompatibilityThresholds,
    ReviewPackageUnit,
    UnitCompatibilityObservation,
    build_ab_review_package,
    evaluate_compatibility,
)
from mathula_tv.openvoice import sha256_file


def thresholds(*, calibrated: bool = True) -> CompatibilityThresholds:
    return CompatibilityThresholds(
        version="four-turn-calibration-v1",
        calibrated=calibrated,
        max_openvoice_word_error_rate=0.2,
        max_word_error_rate_degradation=0.05,
        min_protected_name_accuracy=1.0,
        min_number_accuracy=1.0,
        min_date_accuracy=1.0,
        min_negation_accuracy=1.0,
        min_speaker_similarity=0.75,
        max_abs_timing_error_ms=120,
        max_clipping_ratio=0.001,
        min_silence_ratio=0.01,
        max_silence_ratio=0.35,
    )


def approved_reviews() -> dict[str, dict[str, str]]:
    return {
        role: {
            "status": "approved",
            "reviewer": f"reviewer-{role}",
            "reviewed_at": "2026-07-20T09:00:00+00:00",
        }
        for role in REQUIRED_HUMAN_REVIEWS
    }


def observation(**changes) -> UnitCompatibilityObservation:
    values = {
        "unit_id": "unit_0001",
        "speaker_id": "SPEAKER_00",
        "intended_spoken_text": "UNaledi ukhokhe amarandi ayikhulu.",
        "reference_audio_sha256": "1" * 64,
        "azure_audio_sha256": "2" * 64,
        "openvoice_audio_sha256": "3" * 64,
        "azure_duration_ms": 5600,
        "openvoice_duration_ms": 5650,
        "final_timing_error_ms": 50,
        "azure_retranscription": "UNaledi ukhokhe amarandi ayikhulu.",
        "openvoice_retranscription": "UNaledi ukhokhe amarandi ayikhulu.",
        "azure_word_error_rate": 0.05,
        "openvoice_word_error_rate": 0.08,
        "protected_name_accuracy": 1.0,
        "number_accuracy": 1.0,
        "date_accuracy": None,
        "negation_accuracy": None,
        "speaker_similarity": 0.84,
        "azure_speaker_similarity": 0.31,
        "audio_quality_passed": True,
        "clipping_ratio": 0.0,
        "silence_ratio": 0.1,
        "protected_name_count": 1,
        "number_count": 1,
        "human_reviews": approved_reviews(),
    }
    values.update(changes)
    return UnitCompatibilityObservation(**values)


def write_wav(path: Path, value: int = 1000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as audio:
        audio.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
        audio.writeframes(value.to_bytes(2, "little", signed=True) * 2400)


def test_gate_pass_requires_calibrated_thresholds_and_all_human_reviews():
    report = evaluate_compatibility("job-123", [observation()], thresholds())
    assert report["gate_state"] == "passed"
    assert report["production_ready"] is False
    assert report["units"][0]["duration_ratio"] == pytest.approx(5650 / 5600)

    pending_review = approved_reviews()
    pending_review["isizulu_language"] = {
        "status": "pending",
        "reviewer": "language-reviewer",
        "reviewed_at": "2026-07-20T09:00:00+00:00",
    }
    report = evaluate_compatibility(
        "job-123", [observation(human_reviews=pending_review)], thresholds()
    )
    assert report["gate_state"] == "needs_human_review"

    report = evaluate_compatibility(
        "job-123", [observation()], thresholds(calibrated=False)
    )
    assert report["gate_state"] == "needs_human_review"
    assert report["thresholds_provisional"] is True


def test_approval_requires_attributed_timestamped_human_evidence():
    with pytest.raises(ValueError, match="reviewer and timestamp"):
        observation(human_reviews={role: "approved" for role in REQUIRED_HUMAN_REVIEWS})


def test_missing_measurements_or_thresholds_keep_gate_pending():
    report = evaluate_compatibility(
        "job-123", [observation(openvoice_retranscription=None)], thresholds()
    )
    assert report["gate_state"] == "pending"
    assert "openvoice_retranscription" in report["units"][0]["missing_measurements"]

    incomplete = replace(thresholds(), min_speaker_similarity=None)
    report = evaluate_compatibility("job-123", [observation()], incomplete)
    assert report["gate_state"] == "pending"
    assert report["missing_thresholds"] == ["min_speaker_similarity"]

    known_failure = evaluate_compatibility(
        "job-123",
        [observation(audio_quality_passed=False)],
        incomplete,
    )
    assert known_failure["gate_state"] == "failed_audio_quality"


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"protected_name_accuracy": 0.5}, "failed_pronunciation"),
        ({"openvoice_word_error_rate": 0.4}, "failed_intelligibility"),
        ({"speaker_similarity": 0.5}, "failed_identity"),
        ({"final_timing_error_ms": 121}, "failed_timing"),
        ({"clipping_ratio": 0.01}, "failed_audio_quality"),
    ],
)
def test_gate_classifies_automatic_failures(changes, expected):
    report = evaluate_compatibility("job-123", [observation(**changes)], thresholds())
    assert report["gate_state"] == expected
    assert report["summary"]["failed_units"] == 1


def test_known_failure_takes_precedence_over_another_pending_unit():
    failed = observation(protected_name_accuracy=0.5)
    pending = observation(unit_id="unit_0002", speaker_similarity=None)
    report = evaluate_compatibility("job-123", [pending, failed], thresholds())
    assert report["gate_state"] == "failed_pronunciation"


def test_ab_review_package_materialises_all_four_audio_variants(tmp_path):
    source = tmp_path / "source"
    paths = {}
    for index, name in enumerate(("original", "azure", "openvoice", "aligned"), 1):
        path = source / f"{name}.wav"
        write_wav(path, value=index * 1000)
        paths[name] = path
    report = evaluate_compatibility(
        "job-123",
        [
            observation(
                reference_audio_sha256=sha256_file(paths["original"]),
                azure_audio_sha256=sha256_file(paths["azure"]),
                openvoice_audio_sha256=sha256_file(paths["openvoice"]),
                aligned_audio_sha256=sha256_file(paths["aligned"]),
            )
        ],
        thresholds(),
    )
    review = tmp_path / "review"
    manifest = build_ab_review_package(
        review,
        report,
        [
            ReviewPackageUnit(
                unit_id="unit_0001",
                original_reference=paths["original"],
                azure_tts=paths["azure"],
                openvoice=paths["openvoice"],
                aligned=paths["aligned"],
            )
        ],
    )
    assert manifest["gate_state"] == "passed"
    for directory in ("original_reference", "azure_tts", "openvoice", "aligned"):
        copied = review / directory / "unit_0001.wav"
        assert copied.is_file()
        assert sha256_file(copied)
    assert (review / "compatibility_report.json").is_file()
    assert (review / "compatibility_report.md").is_file()
    assert (review / "listening_guide.md").is_file()
    comparison = json.loads(
        (review / "comparisons" / "unit_0001.json").read_text()
    )
    assert comparison["measurements"]["intended_spoken_text"].startswith("UNaledi")
    assert comparison["assets"]["openvoice"]["path"] == "openvoice/unit_0001.wav"


def test_ab_review_package_rejects_missing_or_remote_assets(tmp_path):
    local = tmp_path / "local.wav"
    write_wav(local)
    local_hash = sha256_file(local)
    report = evaluate_compatibility(
        "job-123",
        [
            observation(
                reference_audio_sha256=local_hash,
                azure_audio_sha256=local_hash,
                aligned_audio_sha256=local_hash,
            )
        ],
        thresholds(),
    )
    with pytest.raises(FileNotFoundError):
        build_ab_review_package(
            tmp_path / "review",
            report,
            [
                ReviewPackageUnit(
                    "unit_0001",
                    local,
                    local,
                    tmp_path / "missing.wav",
                    local,
                )
            ],
        )


def test_ab_review_package_rejects_audio_that_does_not_match_report(tmp_path):
    local = tmp_path / "local.wav"
    write_wav(local)
    report = evaluate_compatibility("job-123", [observation()], thresholds())
    with pytest.raises(ValueError, match="hash does not match"):
        build_ab_review_package(
            tmp_path / "review",
            report,
            [ReviewPackageUnit("unit_0001", local, local, local, local)],
        )
