from mathula_tv.pronunciation import PronunciationDictionary, with_default_organisation_initialisms


def read(text: str) -> str:
    return with_default_organisation_initialisms(PronunciationDictionary("v1", language="zu-ZA", job_id="x")).apply(text).tts_text


def test_a_comma_grouped_amount_is_one_english_quantity_not_a_digit_then_spelt_zeros():
    assert read("bekhokhelwa u-4,500 kuphela ngenyanga") == "bekhokhelwa u-four thousand fayiv hundred kuphela ngenyanga"
    assert read("ka-4,500") == "ka-four thousand fayiv hundred"
    assert read("kusukela ku-19,500") == "kusukela ku-naynteen thousand fayiv hundred"


def test_the_rand_sign_becomes_the_word_and_hundreds_and_tens_are_spoken():
    assert read("u-R4,500 ngenyanga") == "u-four thousand fayiv hundred rand ngenyanga"
    assert read("R4 500") == "four thousand fayiv hundred rand"
    assert read("ama-R4,550") == "ama-four thousand fayiv hundred and fifty rand"
    assert read("u-4,005") == "u-four thousand and fayiv"
    assert read("u-4,000") == "u-four thousand"


def test_other_numbers_keep_their_existing_readings():
    assert read("amaRandi ayi-10 000") == "amaRandi ayi-10 thousand"        # the earlier space-grouped thousand rule
    assert read("Idokethi 2015") == read("Idokethi 2015")                 # a reference number is still read digit by digit
    assert "thousand" not in read("u-4500")                               # an ungrouped number is not touched by this rule
    assert "thousand" not in read("1,234,567")                            # millions are left alone
