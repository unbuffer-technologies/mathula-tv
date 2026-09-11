from mathula_tv.native_dub import (
    _english_absolute_pace_ratio,
    _english_source_pace_ratios,
    _english_syllables_per_second,
    _estimate_english_syllables,
    _temporal_mask_wire,
)


def test_estimate_english_syllables_matches_known_word_counts():
    # Standard readability-formula heuristic: vowel groups, minus silent trailing "e",
    # plus a syllable back for a consonant+"le" ending.
    assert _estimate_english_syllables("a") == 1
    assert _estimate_english_syllables("the") == 1
    assert _estimate_english_syllables("make") == 1  # silent e
    assert _estimate_english_syllables("table") == 2  # ta-ble
    assert _estimate_english_syllables("little") == 2  # lit-tle
    assert _estimate_english_syllables("urgency") == 3  # ur-gen-cy
    assert _estimate_english_syllables("investigation") == 5  # in-ves-ti-ga-tion
    assert _estimate_english_syllables("Mississippi") == 4


def test_estimate_english_syllables_treats_multi_letter_acronyms_as_spelled_out():
    assert _estimate_english_syllables("IDAC") == 4  # one syllable per letter
    assert _estimate_english_syllables("PISA") == 4


def test_syllables_per_second_basic():
    # "one two three four" is four monosyllabic words.
    assert _english_syllables_per_second("one two three four", 2000) == 2.0


def test_syllables_per_second_guards_against_zero_duration():
    # Must not divide by zero; a near-zero window still returns a finite (large) rate.
    assert _english_syllables_per_second("one two", 0) > 0


def test_pace_ratios_flag_a_block_faster_than_this_jobs_own_median():
    groups = [
        {"group_id": "slow", "source_text": " ".join(["word"] * 10), "source_span_ms": 10_000},  # 1.0 syll/s
        {"group_id": "typical", "source_text": " ".join(["word"] * 10), "source_span_ms": 10_000},  # 1.0 syll/s
        {"group_id": "fast", "source_text": " ".join(["word"] * 30), "source_span_ms": 10_000},  # 3.0 syll/s
    ]
    ratios = _english_source_pace_ratios(groups)
    assert ratios["slow"] == 1.0
    assert ratios["typical"] == 1.0
    assert ratios["fast"] == 3.0  # 3x the job's own 1.0 syll/s median


def test_pace_ratios_handle_empty_source_text_without_crashing():
    # _estimate_english_syllables floors at 1 syllable (mirroring _estimate_zulu_syllables'
    # same convention), so empty text is not literally zero-paced, just the slowest block.
    groups = [
        {"group_id": "g1", "source_text": "", "source_span_ms": 5000},
        {"group_id": "g2", "source_text": "one two three", "source_span_ms": 5000},
    ]
    ratios = _english_source_pace_ratios(groups)
    assert ratios["g1"] < ratios["g2"]
    assert ratios["g1"] > 0


def test_temporal_mask_wire_carries_the_pace_signal_and_threshold_flag():
    groups = [{"group_id": "g1", "speaker_id": "S0", "segment_ids": ["seg-1"], "source_text": "English text.", "source_span_ms": 1000}]
    geometry = {
        "preferred_start_ms": 0,
        "preferred_end_ms": 1000,
        "source_equivalent_syllables": 20,
        "target_syllables": 21,
        "min_syllables": 20,
        "max_syllables": 22,
    }

    typical = _temporal_mask_wire(groups, 0, accepted_previous=[], geometry=geometry, candidate_count=1, source_pace_ratio=1.0)
    assert typical["temporal_mask"]["source_pace_ratio_vs_this_jobs_typical"] == 1.0
    assert typical["temporal_mask"]["source_faster_than_this_jobs_typical"] is False
    # "English text." = "Eng-lish" (2) + "text" (1) = 3 syllables / 1.0s = 3.0 syll/s,
    # below the 3.9 syll/s benchmark.
    assert typical["temporal_mask"]["source_pace_ratio_vs_typical_broadcast_english"] < 1.0
    assert typical["temporal_mask"]["source_faster_than_typical_broadcast_english"] is False

    fast = _temporal_mask_wire(groups, 0, accepted_previous=[], geometry=geometry, candidate_count=1, source_pace_ratio=1.5)
    assert fast["temporal_mask"]["source_pace_ratio_vs_this_jobs_typical"] == 1.5
    assert fast["temporal_mask"]["source_faster_than_this_jobs_typical"] is True


def test_absolute_pace_ratio_matches_confirmed_production_case():
    """Regression test for a real gap: job-relative pacing missed this exact block.

    native_phrase_0001 (the block that triggered a +40.7% native-dub rush) came out at
    1.0 on the job-relative signal -- exactly typical -- because this entire job runs
    fast throughout. The absolute benchmark, now syllable-based rather than a flat word
    count, is what actually catches it.
    """

    english = (
        "My colleague Ayanda Nyati from the Bar and a Justice College and he joins us now "
        "to give us a more comprehensive wrap of what the Commission has heard thus far. "
        "A very good morning to you Ayanda. So at this point now just before the T "
        "adjournment the question around why the urgency? Why was it so urgent to not even "
        "allow the process to maybe start being prepared if you will or the investigation "
        "to get underway? He was somebody that less than 24 hours after laying or opening "
        "those complaints He lays the criminal charges and then less than 24 hours you "
        "e-mail the minister indicating that in fact those dockets have now been "
        "intercepted. We are still stuck on why the urgency."
    )
    ratio = _english_absolute_pace_ratio(english, 38_640)
    assert ratio > 1.15  # crosses the "faster than typical" threshold
    assert ratio == 1.221


def test_absolute_pace_ratio_is_independent_of_job_relative_ratio():
    # A job that is uniformly fast throughout gets flagged by the absolute check even
    # though every block looks "typical" (ratio 1.0) relative to each other.
    groups = [
        {"group_id": "g1", "source_text": " ".join(["word"] * 50), "source_span_ms": 10_000},  # 5.0 syll/s
        {"group_id": "g2", "source_text": " ".join(["word"] * 50), "source_span_ms": 10_000},  # 5.0 syll/s
    ]
    relative = _english_source_pace_ratios(groups)
    assert relative["g1"] == 1.0 and relative["g2"] == 1.0  # no relative outlier

    absolute = _english_absolute_pace_ratio(groups[0]["source_text"], groups[0]["source_span_ms"])
    assert absolute > 1.15  # but the whole job is objectively fast
