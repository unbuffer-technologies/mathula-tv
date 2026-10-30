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


def test_the_agent_can_ask_for_the_shorter_english_reading_of_a_figure_with_the_marker():
    assert read("bekhokhelwa u-4,500 kuphela") == "bekhokhelwa izinkulungwane ezine namakhulu amahlanu zamaRandi kuphela"        # default: native
    assert read("bekhokhelwa u-4,500~en kuphela") == "bekhokhelwa u-four thousand fayiv hundred kuphela"                          # asked for
    assert read("bekhokhelwa u-R19,500~en ngenyanga") == "bekhokhelwa u-naynteen thousand fayiv hundred rand ngenyanga"
    assert read("abantu abangu-4,500~en bafika") == "abantu abangu-four thousand fayiv hundred bafika"
    assert read("bafika abantu 500~en") == "bafika abantu fayiv hundred" and read("bafika abantu 150~en") == "bafika abantu wan hundred and fifty"
    assert read("Idokethi 123~en") == "Idokethi wan tuu triiy"                                      # an identifier is never turned into a quantity


def test_the_marker_is_never_spoken_and_never_breaks_the_digits_check():
    from mathula_tv.native_dub import _missing_required_literals

    assert "~en" not in read("u-11~en kuphela") and read("u-11~en kuphela") == "u-11 kuphela"       # a marker no rule used is dropped
    assert "~en" not in read("bekhokhelwa u-4,500 ~en kuphela")
    assert _missing_required_literals(source_text="paid 4,500 a month", candidate_text="bekhokhelwa u-4,500~en kuphela") == []


def test_the_agent_is_told_it_may_use_the_marker_for_figures_that_do_not_fit():
    from mathula_tv.dub_agent import SYSTEM_PROMPT

    assert "~en" in SYSTEM_PROMPT and "native is the default" in SYSTEM_PROMPT


def test_vocabulary_is_chosen_for_the_widest_understanding():
    from mathula_tv.dub_agent import SYSTEM_PROMPT
    from mathula_tv.native_dub import ZULU_GLOSSARY_PROMPT_VERSION, ZULU_GLOSSARY_SYSTEM_PROMPT

    assert "igranti" in SYSTEM_PROMPT and "Widest understanding beats formality" in SYSTEM_PROMPT
    assert "simplicity and maximum reach" in SYSTEM_PROMPT and "simplicity and maximum reach" in ZULU_GLOSSARY_SYSTEM_PROMPT
    assert "WIDEST UNDERSTANDING" in ZULU_GLOSSARY_SYSTEM_PROMPT and '"igranti"' in ZULU_GLOSSARY_SYSTEM_PROMPT
    assert "widest-understanding" in ZULU_GLOSSARY_PROMPT_VERSION        # so every cached glossary is regenerated under the new rule


def test_the_glossary_rule_keeps_the_number_of_an_english_term_and_forbids_a_prefix_built_on_a_changed_number():
    from mathula_tv.native_dub import ZULU_GLOSSARY_PROMPT_VERSION, ZULU_GLOSSARY_SYSTEM_PROMPT
    assert "number-kept" in ZULU_GLOSSARY_PROMPT_VERSION and "widest-understanding" in ZULU_GLOSSARY_PROMPT_VERSION
    assert "keeps the number of the English" in ZULU_GLOSSARY_SYSTEM_PROMPT and "plural -s included" in ZULU_GLOSSARY_SYSTEM_PROMPT
    assert "Zulu noun prefix glued on" in ZULU_GLOSSARY_SYSTEM_PROMPT and "must not become" in ZULU_GLOSSARY_SYSTEM_PROMPT

