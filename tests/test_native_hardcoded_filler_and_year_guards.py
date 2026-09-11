"""Tests for two deterministic, hardcoded guards added after a manual sentence-by-
sentence review of job 33cd7b46d55647e39cc1e8d26ab962ed found real, confirmed cases
where relying on Phase A's own prompt instructions was unreliable:

1. Confirmed discourse filler ("of course", "so at this point now", "to be honest
   with you", etc.) survived into the isiZulu verbatim even though Phase A's own
   LENGTH CONTRACT already instructs tightening it -- the model only reliably applies
   that instruction under real timing pressure (Phase B), not as a matter of course.
   Fixed by stripping these phrases from what Phase A even reads, rather than hoping
   it complies with a soft instruction.

2. A truncated year ("...or January of 20, but...") -- almost certainly a clipped
   ASR/Pass-1 restoration artifact -- was translated and carried into the isiZulu
   output as a broken number. There's no way to guess the missing digits from text
   alone, so this is a hardcoded DETECTION-and-warn guard, not an auto-fix.
"""
from pathlib import Path
from types import SimpleNamespace

from mathula_tv.atomic_io import atomic_write_json
from mathula_tv.native_dub import (
    _find_suspected_truncated_year,
    _strip_confirmed_filler_phrases,
    _temporal_sentence_semantic_window,
    run_native_pass1,
)


# --- _strip_confirmed_filler_phrases ------------------------------------------------

def test_strips_so_at_this_point_now_at_sentence_start_and_recapitalizes():
    text = "So at this point now just before the adjournment the question around why the urgency?"
    assert _strip_confirmed_filler_phrases(text) == (
        "Just before the adjournment the question around why the urgency?"
    )


def test_strips_comma_bounded_of_course_and_quite_frankly():
    # Confirmed real case: native_phrase_0017.
    text = (
        "We know that, tomorrow, of course, the NPA has now announced that they are "
        "going to unequivocally withdraw those just because the investigation, quite "
        "frankly, didn't follow the right process."
    )
    cleaned = _strip_confirmed_filler_phrases(text)
    assert "of course" not in cleaned.lower()
    assert "quite frankly" not in cleaned.lower()
    assert "NPA has now announced" in cleaned
    assert "didn't follow the right process" in cleaned


def test_strips_to_be_honest_with_you():
    # Confirmed real case: native_phrase_0136.
    text = "I think I preempted that, to be honest with you, Commissioner."
    cleaned = _strip_confirmed_filler_phrases(text)
    assert "to be honest with you" not in cleaned.lower()
    assert "Commissioner" in cleaned
    assert "preempted that" in cleaned


def test_strips_for_lack_of_a_better_term_and_if_you_will():
    assert "for lack of a better term" not in _strip_confirmed_filler_phrases(
        "He was sounding the alarm, for lack of a better term, about the delay."
    ).lower()
    assert "if you will" not in _strip_confirmed_filler_phrases(
        "It was an attempt, if you will, to derail the process."
    ).lower()


def test_of_course_as_a_standalone_answer_is_not_stripped():
    # "Of course" is load-bearing here -- it IS the answer. Only the comma-bounded
    # mid-sentence hedge usage is a safe target, never a standalone clause.
    assert _strip_confirmed_filler_phrases("Of course.") == "Of course."
    assert _strip_confirmed_filler_phrases("Will you help? Of course.") == "Will you help? Of course."


def test_you_know_him_is_not_stripped_only_the_comma_bounded_hedge_is():
    assert _strip_confirmed_filler_phrases("Do you know him?") == "Do you know him?"
    cleaned = _strip_confirmed_filler_phrases("It was, you know, a difficult time.")
    assert "you know" not in cleaned.lower()
    assert "a difficult time" in cleaned


def test_strips_comma_bounded_i_guess_and_leading_i_mean_and_trailing_so_to_speak():
    # Confirmed real cases: native_phrase_0084 and native_phrase_0093.
    cleaned_84 = _strip_confirmed_filler_phrases(
        "And it's filled with all kinds of charges, even dealing with alleged "
        "hijackings, which is what Adams was trying to, I guess, clear him of "
        "yesterday, saying that the record had been expunged."
    )
    assert "i guess" not in cleaned_84.lower()
    assert "trying to clear him of yesterday" in cleaned_84
    # "kind of" inside "all kinds of charges" must survive untouched -- it's embedded
    # in the noun phrase, not a standalone hedge, and was deliberately excluded.
    assert "kinds of charges" in cleaned_84

    cleaned_93 = _strip_confirmed_filler_phrases(
        "I mean, if you don't want to be seen around this area, perhaps don't be "
        "in the area, so to speak."
    )
    assert cleaned_93 == "If you don't want to be seen around this area, perhaps don't be in the area."


def test_i_guess_so_as_a_standalone_answer_is_not_stripped():
    assert _strip_confirmed_filler_phrases("I guess so.") == "I guess so."


def test_i_mean_it_and_what_do_you_mean_are_not_stripped():
    # "I mean" only matches with a TRAILING comma -- these are the real verb phrase,
    # not the discourse marker, and correctly have no comma after "mean".
    assert _strip_confirmed_filler_phrases("I mean it.") == "I mean it."
    assert _strip_confirmed_filler_phrases("What do you mean?") == "What do you mean?"


def test_no_filler_present_leaves_text_unchanged():
    text = "The Commission heard evidence from three witnesses today."
    assert _strip_confirmed_filler_phrases(text) == text


def test_strip_never_touches_the_canonical_group_source_text():
    # The stripped variant is only what Phase A reads for translation -- subtitles,
    # QA's original-English comparison, and pace-ratio timing must see the real,
    # complete, verbatim source text.
    groups = [
        {
            "group_id": "g0",
            "speaker_id": "S",
            "source_text": "So at this point now, we should proceed.",
        },
    ]
    window = _temporal_sentence_semantic_window(groups, mutable_start=0, mutable_count=1, accepted=[])
    assert window["blocks"][0]["english"] == "We should proceed."
    assert groups[0]["source_text"] == "So at this point now, we should proceed."


# --- _find_suspected_truncated_year -------------------------------------------------

def test_detects_the_confirmed_real_truncated_year_case():
    text = (
        "I couldn't tell you if it was November of 2024 or December of 2024 or "
        "January of 20, but I had spoken to him on the phone once."
    )
    assert _find_suspected_truncated_year(text) == "20"


def test_does_not_flag_a_complete_four_digit_year():
    assert _find_suspected_truncated_year("It happened in November of 2024.") is None


def test_does_not_flag_a_day_of_month_reference():
    # "January 5th" / "on May 3" are complete, valid day-of-month references phrased
    # without "of <digits>" the way the confirmed truncated-year case was -- a day
    # reference wouldn't naturally be phrased "of January of 5".
    assert _find_suspected_truncated_year("The hearing was set for January 5th.") is None
    assert _find_suspected_truncated_year("He arrived on May 3.") is None


def test_does_not_flag_plain_text_with_no_month():
    assert _find_suspected_truncated_year("This happened at 8 of them.") is None


# --- run_native_pass1 integration ---------------------------------------------------

class _FakeTruncatedYearProvider:
    provider = "azure-foundry-grok"

    def __init__(self, corrections):
        self.config = SimpleNamespace(deployment="grok-test", max_output_tokens=8192)
        self._corrections = corrections

    def complete_json(self, **kwargs):
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


def test_run_native_pass1_warns_on_a_suspected_truncated_year(tmp_path: Path):
    job_root = tmp_path / "job"
    _write_transcript(job_root, [
        {
            "segment_id": "seg-1", "speaker": "SPEAKER_00", "start": 0.0, "end": 2.0,
            "source_text": "It was in january of 20 apparently.",
        },
    ])
    provider = _FakeTruncatedYearProvider(corrections=[
        {"segment_id": "seg-1", "corrected_text": "It was in January of 20, apparently.", "reason_code": "punctuation"},
    ])
    warnings: list[str] = []
    result = run_native_pass1(job_root=job_root, provider=provider, progress=warnings.append)

    restored_by_id = {item["segment_id"]: item for item in result["segments"]}
    assert restored_by_id["seg-1"].get("suspected_truncated_year") is True
    assert any("truncated year" in message for message in warnings)


def test_run_native_pass1_does_not_warn_on_a_complete_year(tmp_path: Path):
    job_root = tmp_path / "job"
    _write_transcript(job_root, [
        {
            "segment_id": "seg-1", "speaker": "SPEAKER_00", "start": 0.0, "end": 2.0,
            "source_text": "It was in November of 2024.",
        },
    ])
    provider = _FakeTruncatedYearProvider(corrections=[
        {"segment_id": "seg-1", "corrected_text": "It was in November of 2024.", "reason_code": "grammar"},
    ])
    result = run_native_pass1(job_root=job_root, provider=provider)
    restored_by_id = {item["segment_id"]: item for item in result["segments"]}
    assert "suspected_truncated_year" not in restored_by_id["seg-1"]
