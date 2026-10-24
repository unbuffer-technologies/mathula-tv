from mathula_tv.number_check import digit_by_digit, quantities
from mathula_tv.pronunciation import PronunciationDictionary, with_default_organisation_initialisms


def spoken(text: str) -> str:
    return with_default_organisation_initialisms(PronunciationDictionary("v1", language="zu-ZA", job_id="x")).apply(text).tts_text


def test_quantity_shaped_numbers_are_found_and_plain_numbers_are_not():
    assert quantities("found guards being paid 4,500. We gave them 19,500 and R4 500, also 12,000,000") == ["4500", "19500", "4500", "12000000"]
    assert quantities("on the 5th of November, in Ward 11, case 2015, 500 people") == []


def test_the_bug_from_job_13eacc6f_would_have_been_flagged():
    sent_then = "Safica onogada bekhokhelwa u-4,fayiv zeeroh zeeroh kuphela ngenyanga"
    assert digit_by_digit("being paid 4,500 a month", sent_then) == [
        {"number": "4500", "problem": "sent to the voice digit by digit, not as an amount"}]


def test_the_corrected_reading_is_clean():
    sent_now = spoken("Safica onogada bekhokhelwa u-4,500 kuphela ngenyanga")
    assert "thousand" in sent_now and digit_by_digit("being paid 4,500 a month", sent_now) == []


def test_a_reference_number_read_digit_by_digit_is_not_a_problem():
    assert digit_by_digit("docket 123456 was filed", spoken("Idokethi 123456")) == []
