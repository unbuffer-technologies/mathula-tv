from mathula_tv.zulu_word_notes import WEEKDAYS, weekday_notes


def test_a_day_name_glued_to_a_zulu_prefix_is_flagged_with_the_zulu_day():
    notes = weekday_notes("Nizoshintsha kanjani ngoMonday, ngokwesibonelo?")
    assert len(notes) == 1 and notes[0].startswith("ngoMonday:") and "ngoMsombuluko" in notes[0] and "uMsombuluko" in notes[0]


def test_the_slip_from_the_blind_read_and_a_bare_day_are_found():
    assert weekday_notes("Ingabe uma sekunguMonday?")[0].startswith("sekunguMonday")
    assert weekday_notes("Kuzoba ngu Friday.")[0].startswith("Friday")        # a bare word, any case


def test_every_weekday_has_a_zulu_name_and_each_is_flagged_once_per_unit():
    for day, (name, glued) in WEEKDAYS.items():
        assert name[0] in "ui" and glued.startswith(("ngo", "nge"))
        assert len(weekday_notes(f"Okwenzekile ngo{day.capitalize()}, bese ngo{day.capitalize()} futhi.")) == 1


def test_zulu_days_titles_and_other_words_are_left_alone():
    for text in ("Sizohlangana ngoMsombuluko nangoLwesine.", "Funda iSunday Times namuhla.", "Friday Night Live iyaqala.", "Ngeke ngikubone phakathi nesonto.",
                 "Mondaylike akukho", "~pause800 Kulungile.", ""):
        assert weekday_notes(text) == [], text


def test_a_pause_marker_is_not_read_as_text():
    assert weekday_notes("~pause500 ngoTuesday ~pause300")[0].startswith("ngoTuesday")
