from mathula_tv import omnivoice_duration as od

REF_TEXT = "So people that are within that same standpoint as ourselves, that's the angle that we're taking."


def test_weights_follow_the_rule_estimator():
    assert od.weight("abc") == 3.0
    assert od.weight("a b") == 2.2                     # a space counts 0.2
    assert od.weight("a,") == 1.5                      # punctuation 0.5
    assert od.weight("20") == 7.0                      # a digit 3.5
    assert od.weight("é") == 1.0                 # a combining mark is silent


def test_a_unit_spoken_at_the_reference_pace_takes_the_reference_time():
    pace = od.seconds_per_weight(REF_TEXT, 5.0)
    long_text = (REF_TEXT + " ") * 4
    assert abs(od.span_seconds(long_text, pace) - (4 * 5.0 + od.EDGE_PAD_S)) < 0.4    # long text: no short-output lift, 200 ms of padding on top


def test_speed_slows_and_speeds_the_estimate_and_a_join_is_the_planned_silence_or_the_voices_own_gap_if_longer():
    base = od.estimate_seconds([REF_TEXT * 3], [], REF_TEXT, 5.0, 1.0)
    assert od.estimate_seconds([REF_TEXT * 3], [], REF_TEXT, 5.0, 0.5) > base * 1.9
    two = ["a" * 60, "b" * 60]
    held = od.estimate_seconds(two, [1100], REF_TEXT, 5.0) - od.estimate_seconds(two, [0], REF_TEXT, 5.0)
    assert abs(held - ((od.EDGE_PAD_S + 1.1) - od.NATURAL_GAP_S)) < 1e-9             # a held pause (the 200 ms of padding and 1.1 s) against the voice's own gap, which the plain join gets
    ran_on = od.estimate_seconds(two, [-200], REF_TEXT, 5.0)
    assert abs(ran_on - od.estimate_seconds(two, [0], REF_TEXT, 5.0) - (od.NATURAL_GAP_S - od.NATURAL_GAP_S)) < 1e-9      # the speaker ran straight on: the join is still the voice's own gap, never less
    assert abs(od.estimate_seconds(two, [-200, 500], REF_TEXT, 5.0) - ran_on) < 1e-9      # a gap with no span after it is not a join


def test_the_estimate_is_the_fitted_share_of_the_speech_plus_the_joins_and_the_units_own_padding():
    pace = od.seconds_per_weight(REF_TEXT, 5.0)
    spans = [REF_TEXT * 2, REF_TEXT * 3]
    speech = sum(od.span_seconds(s, pace) - od.EDGE_PAD_S for s in spans)
    assert abs(od.estimate_seconds(spans, [700], REF_TEXT, 5.0) - (od.SPEECH_FACTOR * speech + (od.EDGE_PAD_S + 0.7) + od.EDGE_PAD_S)) < 1e-9
    assert 0.9 < od.SPEECH_FACTOR < 1.0 and 0.2 <= od.NATURAL_GAP_S <= 0.4


def test_a_very_short_text_is_lifted_not_shrunk_to_nothing():
    pace = od.seconds_per_weight(REF_TEXT, 5.0)
    one_word = od.span_seconds("Sawubona", pace)
    assert one_word > od.EDGE_PAD_S + 0.5              # the lift keeps it above the raw estimate


def test_speed_to_fit_lands_the_unit_in_the_window():
    spans = ["Ngiyabonga kakhulu nina nonke ababukeli bethu abahloniphekile."] * 3
    speed = od.speed_to_fit(spans, [200, 200], REF_TEXT, 5.0, 20.0)
    assert abs(od.estimate_seconds(spans, [200, 200], REF_TEXT, 5.0, speed) - 20.0) < 0.05
    assert od.speed_to_fit(spans, [], REF_TEXT, 5.0, 0.1) == 2.0                      # clamped: nothing fits a window that short
