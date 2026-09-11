"""Regression test for a real production crash traced to job 33cd7b46d55647e39cc1e8d26ab962ed.

Pass 1's single whole-transcript autocorrection call produced a corrected_text for one
segment that began with the ENTIRE preceding segment's own already-corrected content
verbatim, before that segment's real content -- a repetition artifact, not a real
correction (the raw ASR source for that segment was clean). Left unrepaired, downstream
sentence-splitting proportionally allocated a near-zero-duration sliver of the real
segment's time budget to the phantom duplicate sentences (confirmed real numbers: 17ms
and 8ms slivers), which crashed native-dub's hard-sync speed fit when it tried to
accelerate audio by >24,000% to fit.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from mathula_tv.atomic_io import atomic_write_json
from mathula_tv.native_dub import _strip_leading_cross_segment_duplication, run_native_pass1


class _FakePass1Provider:
    provider = "azure-foundry-grok"

    def __init__(self, corrections):
        self.config = SimpleNamespace(deployment="grok-test", max_output_tokens=8192)
        self._corrections = corrections
        self.calls = []

    def complete_json(self, **kwargs):
        self.calls.append(kwargs)
        assert kwargs["operation"] == "native_stt_restoration"
        return SimpleNamespace(
            data={"corrections": self._corrections},
            model="grok-test",
            input_tokens=10,
            output_tokens=5,
            attempts=1,
        )


def _write_transcript(job_root: Path, segments: list[dict]) -> None:
    atomic_write_json(
        job_root / "analysis" / "transcript_en.json",
        {"language": "en-ZA", "segments": segments},
    )


# --- _strip_leading_cross_segment_duplication (pure function) -----------------------

def test_strips_the_confirmed_real_duplication_case():
    previous = (
        "You're not saying they are about to do the same thing they did in the "
        "Western Cape. You say they have done the same thing."
    )
    candidate = previous + " They're interfering in the investigation, in the investigative process."
    cleaned, duplicated = _strip_leading_cross_segment_duplication(candidate, previous)
    assert duplicated is True
    assert cleaned == "They're interfering in the investigation, in the investigative process."


def test_does_not_touch_genuinely_independent_text():
    cleaned, duplicated = _strip_leading_cross_segment_duplication(
        "They're interfering in the investigation.", "You say they have done the same thing."
    )
    assert duplicated is False
    assert cleaned == "They're interfering in the investigation."


def test_reports_duplicated_with_empty_remainder_when_nothing_new_was_added():
    previous = "You say they have done the same thing."
    cleaned, duplicated = _strip_leading_cross_segment_duplication(previous, previous)
    assert duplicated is True
    assert cleaned == ""


def test_handles_missing_previous_or_candidate_text_gracefully():
    assert _strip_leading_cross_segment_duplication("Some text.", "") == ("Some text.", False)
    assert _strip_leading_cross_segment_duplication("", "Some text.") == ("", False)


# --- run_native_pass1 end-to-end -----------------------------------------------------

def test_run_native_pass1_strips_duplicated_prefix_from_restored_text(tmp_path: Path):
    job_root = tmp_path / "job"
    segments = [
        {
            "segment_id": "seg-00064",
            "speaker": "SPEAKER_05",
            "start": 738.51,
            "end": 746.43,
            "source_text": "You're not saying they are about to do the same thing they did in the Western Cape. You say they have done the same thing.",
        },
        {
            "segment_id": "seg-00065",
            "speaker": "SPEAKER_05",
            "start": 748.19,
            "end": 799.15,
            "source_text": "They're interfering in the investigation, in the investigative process.",
        },
    ]
    _write_transcript(job_root, segments)

    seg64_corrected = (
        "You're not saying they are about to do the same thing they did in the "
        "Western Cape. You say they have done the same thing."
    )
    seg65_duplicated = seg64_corrected + " They're interfering in the investigation, in the investigative process. Fixed typo."
    provider = _FakePass1Provider(
        corrections=[
            {"segment_id": "seg-00064", "corrected_text": seg64_corrected, "reason_code": "grammar"},
            {"segment_id": "seg-00065", "corrected_text": seg65_duplicated, "reason_code": "grammar"},
        ]
    )

    result = run_native_pass1(job_root=job_root, provider=provider)

    restored_by_id = {item["segment_id"]: item for item in result["segments"]}
    assert restored_by_id["seg-00064"]["restored_text"] == seg64_corrected
    assert restored_by_id["seg-00064"].get("cross_segment_duplication_repaired") is not True

    seg65 = restored_by_id["seg-00065"]
    assert seg65["cross_segment_duplication_repaired"] is True
    assert seg65["restored_text"] == "They're interfering in the investigation, in the investigative process. Fixed typo."
    # The phantom duplicate content must be gone, not just reordered.
    assert "Western Cape" not in seg65["restored_text"]


def test_run_native_pass1_falls_back_to_source_text_when_duplication_leaves_nothing_new(tmp_path: Path):
    job_root = tmp_path / "job"
    segments = [
        {
            "segment_id": "seg-1",
            "speaker": "SPEAKER_00",
            "start": 0.0,
            "end": 2.0,
            "source_text": "First sentence.",
        },
        {
            "segment_id": "seg-2",
            "speaker": "SPEAKER_00",
            "start": 2.5,
            "end": 5.0,
            "source_text": "Second sentence here.",
        },
    ]
    _write_transcript(job_root, segments)
    provider = _FakePass1Provider(
        corrections=[
            {"segment_id": "seg-1", "corrected_text": "First sentence.", "reason_code": "grammar"},
            # seg-2's "correction" is a pure duplicate of seg-1's -- nothing new at all.
            {"segment_id": "seg-2", "corrected_text": "First sentence.", "reason_code": "grammar"},
        ]
    )

    result = run_native_pass1(job_root=job_root, provider=provider)
    restored_by_id = {item["segment_id"]: item for item in result["segments"]}
    assert restored_by_id["seg-2"]["cross_segment_duplication_repaired"] is True
    # Nothing salvageable from the duplicate -- falls back to this segment's own source text
    # rather than committing an empty restored_text.
    assert restored_by_id["seg-2"]["restored_text"] == "Second sentence here."
