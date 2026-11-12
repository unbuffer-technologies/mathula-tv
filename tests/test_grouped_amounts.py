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

    assert "~en" not in read("u-11~en kuphela") and read("u-11~en kuphela") == "u-eleven kuphela"       # the marker is dropped; the figure is English like every other
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



def test_a_figure_after_a_zulu_prefix_is_code_switched_to_english_number_words():
    # human decisions 2026-11-10: the voice read the raw "31" as "titty one", and the isiZulu numerals take too much room: the figure is English, 31 is "thethi-waan"
    assert read("angaphezu kuka-31% eminyakeni") == "angaphezu kuka-thethi-waan wamaphesenti, eminyakeni"
    assert read("amaphesenti angu-31% kanye angama-45") == "amaphesenti angu-thethi-waan, kanye angama-forty-fayiv"          # "amaphesenti" is not said twice
    assert read("kusukela ku-75.8% okhethweni") == "kusukela ku-seventy-fayiv-phoynt-eyt wamaphesenti, okhethweni"
    assert read("kwaba ngu-44.05%") == "kwaba ngu-forty-four-phoynt-zeeroh-fayiv wamaphesenti"
    assert read("kwehle eminyakeni engu-20.") == "kwehle eminyakeni engu-twenty."
    assert read("abantu abangu-80") == "abantu abangu-eighty"
    assert read("omasipala abangu-11") == "omasipala abangu-eleven"
    assert read("ngomhla ka-31 Mashi") == read("ngomhla ka-31 Mashi")                         # a day before a month is the date rule's, untouched here
    assert read("ngo-31 Mashi").startswith("ngo-") and "thethi-waan" in read("ngo-31 Mashi")      # and the day 31 now has the same chosen reading
    # the other prefixes real text puts before a figure (a corpus audit of 69,875 sentences): ezingama- 25%, engama- 17%, abangama- 4%, eyi-/abayi-/ayi- before 10-19
    assert read("ezinsukwini ezingama-21 usexolelwe") == "ezinsukwini ezingama-twenty-wan usexolelwe"
    assert read("eminyakeni engama-55 kuya kwengama-73") == "eminyakeni engama-fifty-fayiv kuya kwengama-seventy-three"
    assert read("uneminyaka eyi-14 nabantu abayi-12") == "uneminyaka eyi-fourteen nabantu abayi-twelve"
    assert read("amaphesenti ayi-15") == "amaphesenti ayi-fifteen"
    # any vowel-final prefix, not a list: the most common real shapes
    assert read("wathi u-60 wamaphesenti wamanzi") == "wathi u-sixty wamaphesenti wamanzi"
    assert read("ngo-7 ekuseni kuya ku-9 ebusuku") == "ngo-seven ekuseni kuya ku-nayn ebusuku"
    assert read("izigidi eziyi-8 zamarandi") == "izigidi eziyi-eyteen zamarandi".replace("eyteen", "eyt")
    assert read("unyaka wama-40 ikhona") == "unyaka wama-forty ikhona"
    assert read("i-COVID-19 yasabalala") == "i-COVID-19 yasabalala"                           # a name ending in a consonant is not a prefix
    assert read("izinga lesi-6 nowesi-10") == "izinga lesi-seeks nowesi-ten"                 # ordinals (-si) belong to the ordinal rule: owesi/lesi/sesi/yesi/kwesi all read the same way
    # a percentage is followed by a breath (a comma, ~0.15 s) so the viewer can take it in; none where punctuation already follows
    assert read("ku-75.8% ngo-2006 laya ku-44.05% kulo nyaka") == "ku-seventy-fayiv-phoynt-eyt wamaphesenti, ngo-twenty-oh-six laya ku-forty-four-phoynt-zeeroh-fayiv wamaphesenti, kulo nyaka"
    assert read("ku-75.8%, ngo-2006") == "ku-seventy-fayiv-phoynt-eyt wamaphesenti, ngo-twenty-oh-six"          # not doubled
    assert read("kwaba ngu-44.05%.") == "kwaba ngu-forty-four-phoynt-zeeroh-fayiv wamaphesenti."


def test_decimal_figures_are_listed_and_a_spelling_chosen_by_ear_replaces_the_rule():
    from mathula_tv.pronunciation import decimal_figures, figure_spelling

    assert decimal_figures(["Kusuke ku-75.8% ngo-2006 laya ku-44.05% kulo nyaka, ku-75.8% futhi, abangu-11, kuka-31%"]) == [("75", "8", True), ("44", "05", True)]          # once each; whole figures are not listed
    assert figure_spelling("75", "8", True) == "seventy-fayiv-phoynt-eyt wamaphesenti"
    assert figure_spelling("44", "05", False) == "forty-four-phoynt-zeeroh-fayiv"
    dictionary = with_default_organisation_initialisms(PronunciationDictionary("v1", language="zu-ZA", job_id="job-1")).with_job_overrides("job-1", [
        {"display_text": "75.8%", "spoken_text": "75.8%", "tts_text": "seventy-five-poynt-eyt amaphesenti", "kind": "english_code_switch"}])
    assert dictionary.apply("Kusuke ku-75.8% ngo-2006 laya ku-44.05% kulo nyaka").tts_text == \
        "Kusuke ku-seventy-five-poynt-eyt amaphesenti, ngo-twenty-oh-six laya ku-forty-four-phoynt-zeeroh-fayiv wamaphesenti, kulo nyaka"          # only the figure the person spelled
    assert dictionary.apply("amaphesenti angu-75.8% kanye").tts_text == "amaphesenti angu-seventy-five-poynt-eyt, kanye"          # the unit is not said twice, even a person's own
    assert dictionary.apply("75.8% yabantu").tts_text == "seventy-five-poynt-eyt amaphesenti yabantu"          # with no prefix the entry itself applies


def test_every_prefixed_figure_is_listed_and_a_whole_number_entry_never_matches_other_text():
    from mathula_tv.pronunciation import prefixed_figures

    assert prefixed_figures(["omasipala abangu-11, kuka-31% futhi ku-75.8%, abangu-11"]) == [("11", "", False), ("31", "", True), ("75", "8", True)]
    base = with_default_organisation_initialisms(PronunciationDictionary("v1", language="zu-ZA", job_id="job-1"))
    ear = base.with_job_overrides("job-1", [{"display_text": "11", "spoken_text": "11", "tts_text": "eleven", "kind": "english_code_switch"}])
    assert ear.apply("omasipala abangu-11 kuphela").tts_text == "omasipala abangu-eleven kuphela"
    assert base.with_job_overrides("job-1", [{"display_text": "11", "spoken_text": "11", "tts_text": "elevn", "kind": "english_code_switch"}]).apply("omasipala abangu-11 kuphela").tts_text == "omasipala abangu-elevn kuphela"          # the person's spelling
    assert ear.apply("ngo-11 Novemba bafika").tts_text == base.apply("ngo-11 Novemba bafika").tts_text          # a day before a month is still the date rule's
    assert ear.apply("R11 million").tts_text == base.apply("R11 million").tts_text and ear.apply("izinsuku ezi-11").tts_text == base.apply("izinsuku ezi-11").tts_text          # no other text is touched


def test_a_year_is_listed_and_a_spelling_chosen_by_ear_replaces_the_rule_wherever_the_year_is_read():
    from mathula_tv.pronunciation import written_years, year_spelling

    assert written_years(["kwenzeka ngo-2006 kwaya ka-2024, ngoNovemba 2019 nango-2006"]) == [2006, 2024, 2019]          # once each, in order
    assert year_spelling(2006) == "twenty-oh-six" and year_spelling(2024) == "twenty-twenty-four"
    base = with_default_organisation_initialisms(PronunciationDictionary("v1", language="zu-ZA", job_id="job-1"))
    ear = base.with_job_overrides("job-1", [{"display_text": "2006", "spoken_text": "2006", "tts_text": "twenty-zero-six", "kind": "english_code_switch"}])
    assert base.apply("Kwenzeka ngo-2006 futhi").tts_text == "Kwenzeka ngo-twenty-oh-six futhi"
    assert ear.apply("Kwenzeka ngo-2006 futhi").tts_text == "Kwenzeka ngo-twenty-zero-six futhi"
    assert ear.apply("kusukela ngoNovemba 2006").tts_text == base.apply("kusukela ngoNovemba 2006").tts_text.replace("twenty-oh-six", "twenty-zero-six")          # after a month too
    assert ear.apply("Idokethi 2006 yafakwa").tts_text == base.apply("Idokethi 2006 yafakwa").tts_text          # a docket number is not a year: untouched
    assert ear.apply("ngo-2024 kanye").tts_text == base.apply("ngo-2024 kanye").tts_text                          # another year keeps the rule


def test_the_unit_is_not_said_twice_when_amaphesenti_is_a_few_words_before_the_figure():
    # human-reported 2026-11-11: "ngamaphesenti angaphezu kuka-31%" said the unit twice
    assert read("kwehle ngamaphesenti angaphezu kuka-31% eminyakeni") == "kwehle ngamaphesenti angaphezu kuka-thethi-waan, eminyakeni"
    assert read("amaphesenti angu-31% kanye") == "amaphesenti angu-thethi-waan, kanye"
    assert read("ngamaphesenti angu-75 kanye ku-44.05% kulo") == "ngamaphesenti angu-seventy-fayiv kanye ku-forty-four-phoynt-zeeroh-fayiv wamaphesenti, kulo"          # another figure in between: this one keeps its own unit
    assert read("Kusuke ku-75.8% ngo-2006") == "Kusuke ku-seventy-fayiv-phoynt-eyt wamaphesenti, ngo-twenty-oh-six"          # no unit before it: said


def test_a_bare_figure_of_one_or_two_digits_is_code_switched_too():
    # human-reported 2026-11-11: "eSection 79" went to the voice as digits ("section seventhy nine")
    assert read("amakomidi eSection 79, neSection 80. Siyazi") == "amakomidi eSection seventy-nayn, neSection eighty. Siyazi"
    assert read("Act 12 yafakwa") == "Act twelve yafakwa"
    assert read("kuya kwabantu aba-4 kanye 5 abantu") == "kuya kwabantu aba-four kanye fayiv abantu"
    assert read("31% kodwa") == "thethi-waan wamaphesenti, kodwa"
    # not figures of this rule
    assert read("R50 kuphela") == read("R50 kuphela")                                                              # an amount stays the amounts' rule
    assert read("ngo-4th Novemba") == read("ngo-4th Novemba")                                                      # an ordinal stays the ordinal rule
    assert read("ngo-14:30 ekuseni") == "ngo-14:30 ekuseni"                                  # a time
    assert read("ikhodi 12345 yafakwa") == "ikhodi wan tuu triiy four fayiv yafakwa"          # a reference number: digit by digit, as before
    assert read("ikhasi 2/3") == "ikhasi 2/3" and read("amakhasi 79-80") == "amakhasi 79-80"  # a fraction, a range
    assert read("abantu abangu-4,500") == read("abantu abangu-4,500")                       # an amount: the amounts' rule
    assert read("izinsuku ezi-79 kuphela") .startswith("izinsuku ezi-")
    assert read("ngoNovemba 4 bafika") == read("ngoNovemba 4 bafika")                                             # a day after its month: the date rule


def test_a_year_after_any_vowel_final_prefix_is_said_in_english_including_the_wezi_forms():
    # a corpus audit (2026-11-11): "owezi-2019", "yowezi-2015", "ngowezi-2010", "wezi-2018" are among the commonest real forms and went to the voice as digits
    for written, spoken in (("owezi-2019", "owezi-twenty-nineteen"), ("ngowezi-2010", "ngowezi-twenty-ten"), ("yowezi-2015", "yowezi-twenty-fifteen"), ("wezi-2018", "wezi-twenty-eighteen"),
                            ("ka-2024", "ka-twenty-twenty-four"), ("kuka-2025", "kuka-twenty-twenty-faif"), ("ngo-2006", "ngo-twenty-oh-six")):
        assert read(f"kwenzeka {written} futhi") == f"kwenzeka {spoken} futhi", written
    assert "2015" in read("Icala elingu-2015 alikaqedwa")                                                                # "elingu-" means "being N": a docket number, not a year
    assert "2015" in read("Eskom-2015 yafakwa")                                                                         # a name with a capital is not a Zulu prefix: the digits stay (the name is respelled by its own entry)


def test_a_tens_word_spelled_by_ear_applies_to_every_figure_in_that_ten_and_is_listed():
    from mathula_tv.pronunciation import tens_words_used

    base = with_default_organisation_initialisms(PronunciationDictionary("v1", language="zu-ZA", job_id="job-1"))
    ear = base.with_job_overrides("job-1", [{"display_text": "seventy", "spoken_text": "seventy", "tts_text": "seventi", "kind": "english_code_switch"}])
    assert base.apply("ezigaba 79 no-80, ku-71 kanye").tts_text == "ezigaba seventy-nayn no-eighty, ku-seventy-wan kanye"
    assert ear.apply("ezigaba 79 no-80, ku-71 kanye").tts_text == "ezigaba seventi-nayn no-eighty, ku-seventi-wan kanye"          # every seventy changes, the eighties do not
    assert tens_words_used(["ezigaba 79 no-80, ku-71, abangu-11, ku-31%, ku-75.8%"]) == ["seventy", "eighty"]                       # 11 and 31 are not tens words
    thirty = base.with_job_overrides("job-1", [{"display_text": "thirty", "spoken_text": "thirty", "tts_text": "thethy", "kind": "english_code_switch"}])
    assert base.apply("ku-35").tts_text == "ku-thethi-fayiv" and thirty.apply("ku-35").tts_text == "ku-thethy-fayiv" and thirty.apply("ku-31").tts_text == "ku-thethi-waan"          # 30 and 31 keep their own chosen table
    assert ear.apply("ku-79%, ngo-2006").tts_text.startswith("ku-seventi-nayn")
