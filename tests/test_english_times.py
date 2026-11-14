from mathula_tv.english_times import normalise


def test_a_time_written_with_a_point_is_written_with_a_colon_when_the_text_shows_it_is_a_time():
    assert normalise("when he withdrew at around 8am, we knew that 9.30 would be the opportunity") == "when he withdrew at around 8am, we knew that 9:30 would be the opportunity"
    assert normalise("he was due at 9.30 on Monday") == "he was due at 9:30 on Monday"                      # a preposition of time right before
    assert normalise("around 5:00 o'clock he called, and again at 14.45") == "around 5:00 o'clock he called, and again at 14:45"
    assert normalise("It starts at about 10.05 tonight.") == "It starts at about 10:05 tonight."


def test_decimals_amounts_percentages_versions_and_long_decimals_are_left_alone():
    for text in ("the rate rose to 9.30 percent", "inflation was 9.30%", "R9.30 per litre", "a loss of 9.30 million", "version 3.1.30 was released", "pi is about 3.14159",
                 "he scored 2.30 on the scale", "the price is $9.30", "he paid 9.305 for it"):
        assert normalise(text) == text, text


def test_text_without_times_and_times_already_in_colon_form_are_unchanged_and_nothing_breaks_on_empty():
    assert normalise("We met at 9:30 in the morning.") == "We met at 9:30 in the morning."
    assert normalise("") == "" and normalise(None) == ""
    assert normalise("Section 204 of the Act, 2017.") == "Section 204 of the Act, 2017."
