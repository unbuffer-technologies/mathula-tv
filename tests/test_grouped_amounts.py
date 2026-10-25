from mathula_tv.pronunciation import PronunciationDictionary, _native_amount_words, with_default_organisation_initialisms


def read(text: str) -> str:
    return with_default_organisation_initialisms(PronunciationDictionary("v1", language="zu-ZA", job_id="x")).apply(text).tts_text


def test_a_comma_grouped_amount_is_one_quantity_not_a_digit_then_spelt_zeros():
    out = read("bekhokhelwa u-4,500 kuphela ngenyanga")
    assert "zeeroh" not in out and "4," not in out
    assert out == "bekhokhelwa izinkulungwane ezine namakhulu amahlanu zamaRandi kuphela ngenyanga"       # native: the u- copula is part of the figure


def test_a_rand_amount_is_read_as_native_isizulu_numerals():
    assert read("emholweni ka-R4,500") == "emholweni ka-izinkulungwane ezine namakhulu amahlanu zamaRandi"
    assert read("ukhokhelwa u-R19,500 ngenyanga") == "ukhokhelwa izinkulungwane eziyishumi nesishiyagalolunye namakhulu amahlanu zamaRandi ngenyanga"
    assert read("R4 500") == "izinkulungwane ezine namakhulu amahlanu zamaRandi"
    assert read("bakhokhelwa u-4,000") == "bakhokhelwa izinkulungwane ezine zamaRandi"


def test_native_numerals_follow_the_class_concords():
    assert _native_amount_words(1, 0) == "inkulungwane"
    assert _native_amount_words(1, 500) == "inkulungwane namakhulu amahlanu"
    assert _native_amount_words(2, 0) == "izinkulungwane ezimbili"
    assert _native_amount_words(12, 350) == "izinkulungwane eziyishumi nambili namakhulu amathathu namashumi amahlanu"
    assert _native_amount_words(25, 100) == "izinkulungwane ezingamashumi amabili nesihlanu nekhulu"
    assert _native_amount_words(4, 5) == "izinkulungwane ezine nesihlanu"
    assert _native_amount_words(4, 10) == "izinkulungwane ezine neshumi"
    assert _native_amount_words(4, 250) == "izinkulungwane ezine namakhulu amabili namashumi amahlanu"
    assert _native_amount_words(150, 0) is None                       # six-figure amounts keep the English reading


def test_a_grouped_number_that_is_not_money_is_native_too_and_a_count_keeps_its_concord():
    assert read("ka-4,500") == "ka-izinkulungwane ezine namakhulu amahlanu"                       # no unit noun: the number alone
    assert read("kwafika abantu abangu-4,500") == "kwafika abantu abayizinkulungwane ezine namakhulu amahlanu"
    assert read("izinto ezingu-12 000") == "izinto eziyizinkulungwane eziyishumi nambili"
    assert read("amaRandi ayi-10 000") == "amaRandi ayizinkulungwane eziyishumi"                  # was "10 thousand" (code-switched)
    assert "zamaRandi" not in read("kwafika abantu abangu-4,500")                                 # a head count is not rand
    assert read("u-R150,000 ngenyanga") == "u-one hundred and fifty thousand rand ngenyanga"      # six figures: English, as before


def test_a_plain_three_digit_quantity_is_native_but_an_identifier_stays_digit_by_digit():
    assert read("bafika abantu 500") == "bafika abantu amakhulu amahlanu"
    assert read("bafika abantu 150") == "bafika abantu ikhulu namashumi amahlanu"
    assert read("bafika abantu 101") == "bafika abantu ikhulu nanye"
    assert read("Idokethi 123") == "Idokethi wan tuu triiy"                      # an identifier is read digit by digit, as before
    assert read("inombolo 500") == "inombolo fayiv zeeroh zeeroh"
    assert read("Idokethi 2015") == "Idokethi tuu zeeroh wan fayiv"             # four digits: unchanged


def test_dates_are_still_code_switched_while_numbers_around_them_are_native():
    assert read("ngo-5 Novemba") == "ngo-fayiv November"
    assert read("ngo-19 Agasti") == "ngo-naynteen August"
    assert read("ngowesi-10 kuMbasa") == "ngowesi-ten kuMbasa"
    assert read("Ngo-5 Novemba bafika abantu abangu-4,500") == "Ngo-fayiv November bafika abantu abayizinkulungwane ezine namakhulu amahlanu"


def test_other_numbers_keep_their_existing_readings():
    assert "thousand" not in read("u-4500") and "amaRandi" not in read("u-4500")      # an ungrouped number is not touched by this rule
    assert "thousand" not in read("1,234,567")                              # millions are left alone
