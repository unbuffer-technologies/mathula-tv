from mathula_tv.pronunciation import PronunciationDictionary, with_default_organisation_initialisms


def read(text: str) -> str:
    return with_default_organisation_initialisms(PronunciationDictionary("v1", language="zu-ZA", job_id="x")).apply(text).tts_text


def test_a_legal_number_after_a_possessive_is_read_one_digit_at_a_time():
    assert read("ngesitatimende sika-204, manje") == "ngesitatimende sika-tuu-ow-four, manje"          # the voice said "two hundred and four" for the raw digits
    assert read("kulesi sitatimende sika-204.") == "kulesi sitatimende sika-tuu-ow-four."


def test_a_three_or_four_digit_number_right_after_a_section_word_is_read_digit_by_digit():
    assert read("ngokwesigaba 204") == "ngokwesigaba tuu-ow-four"
    assert read("i-Section 204 statement").endswith("Section tuu-ow-four statement")
    assert read("Act 108 yango").startswith("Act wan-ow-eyt")


def test_two_digit_sections_quantities_and_amounts_keep_their_own_rules():
    assert read("eSection 79") == "eSection seventy-nayn"                                  # two digits: the figure rule, as before
    assert read("bafika abantu 500") == "bafika abantu amakhulu amahlanu"                  # a quantity with no legal word before it
    assert read("Idokethi 123") == "Idokethi wan tuu triiy"                                # an identifier keeps its own reading
    assert "ow-" not in read("amaRandi aka-500")                                        # a possessive before an amount, no statement or section word: not legal
    assert "tuu-ow" not in read("sigaba 2024")                                         # a year-shaped number is not a section


def test_a_phone_number_in_groups_is_one_number_said_digit_by_digit_with_a_breath_between_the_groups():
    assert read("shayela 082 123 4567") == "shayela zeeroh eyt tuu, wan tuu triiy, four fayiv seeks seven"          # the middle group was read as "one hundred and twenty-three"
    assert read("inombolo yocingo 011-123-4567.") == "inombolo yocingo zeeroh wan wan, wan tuu triiy, four fayiv seeks seven."
    assert read("shayela +27 82 123 4567") == "shayela plus tuu seven, eyt tuu, wan tuu triiy, four fayiv seeks seven"
    assert read("0800 123 456") == "zeeroh eyt zeeroh zeeroh, wan tuu triiy, four fayiv seeks"
    assert read("shayela ifoni 0821234567") == "shayela ifoni zeeroh eyt tuu wan tuu triiy four fayiv seeks seven"          # unspaced: as before


def test_dates_amounts_and_counts_that_look_a_little_like_a_phone_number_are_left_alone():
    assert "zeeroh" not in read("ngomhla ka-02-03-2026")
    assert "zeeroh" not in read("abantu abangu-4 500 kuphela")
    assert read("abantu 123 456") == read("abantu 123 456").strip() and "wan tuu triiy" not in read("abantu 123 456")          # no opening 0 or +: not a phone number


def test_other_things_named_by_a_three_digit_number_are_read_digit_by_digit_but_a_quantity_is_not():
    assert read("Bill 204") == "Bill tuu-ow-four"
    assert read("Room 101") == "Room wan-ow-wan"
    assert read("Flight 123") == "Flight wan-tuu-triiy"
    assert read("abantu 204") == "abantu amakhulu amabili nane"                                          # a quantity
