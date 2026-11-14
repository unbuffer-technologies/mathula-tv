from mathula_tv.pronunciation import PronunciationDictionary, with_default_organisation_initialisms


def read(text: str) -> str:
    return with_default_organisation_initialisms(PronunciationDictionary("v1", language="zu-ZA", job_id="x")).apply(text).tts_text


def test_a_time_after_an_at_prefix_is_said_as_a_time_not_a_decimal():
    assert read("sasazi ukuthi ngo-9.30 kwakuzoba") == "sasazi ukuthi ngo-nayn thethi kwakuzoba"          # the figure rule said "nine point three zero"
    assert read("ngo-9:45 ekuseni") == "ngo-nayn forty-fayiv ekuseni"
    assert read("ngo-9.05.") == "ngo-nayn ow-fayiv."
    assert read("ngo-9.00 ekuseni") == "ngo-nayn ekuseni"


def test_a_24_hour_time_is_said_on_the_clock_face():
    assert read("ngo-14:30") == "ngo-tuu thethi"


def test_a_decimal_without_time_evidence_or_with_a_unit_stays_a_decimal():
    assert "phoynt" in read("ngo-9.30%")
    assert "phoynt" in read("izinga lika-9.30 million")
    assert "phoynt" in read("inani eli-9.30 uma kunjalo")          # prefix of neither "at" nor "about", no time word near it


def test_an_other_prefix_with_a_time_word_nearby_is_a_time():
    assert read("lapho eyi-9.30 ntambama").startswith("lapho eyi-nayn thethi")


def test_a_colon_time_after_any_prefix_is_a_time():
    assert read("wamfonela ngabo-5:00.") == "wamfonela ngabo-fayiv."


def test_a_time_is_listed_as_a_time_not_as_a_decimal_figure():
    from mathula_tv.pronunciation import prefixed_figures, written_times

    texts = ["sasazi ukuthi ngo-9.30 kwakuzoba, ngo-8 ekuseni", "wamfonela ngabo-5:00 ntambama", "ngo-14:30"]
    assert [k for k, _ in written_times(texts)] == ["9.30", "5.00", "14.30"]
    assert dict(written_times(texts))["9.30"] == "nayn thethi" and dict(written_times(texts))["14.30"] == "tuu thethi"
    assert [n + ("." + d if d else "") for n, d, _p in prefixed_figures(texts)] == ["8"]          # 9.30 is no decimal figure


def test_a_colon_time_in_the_zulu_satisfies_the_required_literal_gate_for_the_english_point_time():
    from mathula_tv.native_dub import _missing_required_literals

    english = "So when he withdrew at around 8am, we knew that 9.30 would be the opportunity for Andrea Johnson."
    assert _missing_required_literals(source_text=english, candidate_text="sasazi ukuthi ngo-9:30 kwakuzoba yithuba lika-Andrea Johnson ngo-8am") == []
    assert _missing_required_literals(source_text=english, candidate_text="sasazi ukuthi ngo-9.30 kwakuzoba yithuba lika-Andrea Johnson ngo-8am") == []
    missing = _missing_required_literals(source_text=english, candidate_text="sasazi ukuthi ngo-10:30 kwakuzoba yithuba lika-Andrea Johnson ngo-8am")
    assert [m["literal"] for m in missing] == ["9.30"]                                          # a different time is still caught
