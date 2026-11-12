from mathula_tv import comma_check as cc


def _words(*spec):
    """(text, start_s, end_s) -> the recogniser's word rows."""
    return [{"text": t, "start": s, "end": e} for t, s, e in spec]


# "Indeed, Ayanda, hopefully we will ..." as the source speaker said it: no pause on either side of the name; a real pause after "will get answers."
HEARD = _words(("Indeed,", 0.0, 0.4), ("Ayanda,", 0.4, 0.9), ("hopefully", 0.9, 1.4), ("we", 1.4, 1.5), ("will", 1.5, 1.7), ("get", 1.7, 1.9), ("answers.", 1.9, 2.4),
               ("Nokho", 2.9, 3.3), ("it", 3.3, 3.4), ("was", 3.4, 3.5), ("interesting", 3.5, 4.2))


def test_a_comma_beside_a_name_the_speaker_ran_into_the_words_round_it_is_reported():
    found = cc.comma_check("Impela, Ayanda, sethemba sizozithola izimpendulo.", HEARD, 0, 4300)
    assert found and found["commas"] == 2
    said = " ".join(found["commas_where_the_speaker_did_not_pause"])
    assert "Ayanda" in said and "no pause before" in said and "no pause after" in said and "0.00 s" in said


def test_a_name_inside_a_clause_is_not_a_name_of_address_so_its_comma_is_not_reported():
    heard = _words(("so", 0.0, 0.3), ("General", 0.3, 0.7), ("Mkhwanazi", 0.7, 1.2), ("who", 1.2, 1.4))
    assert cc.comma_check("ababesekela uJenene Mkhwanazi, okwakumele nabo babandakanywe", heard, 0, 1400) is None          # a clause comma, not a vocative


def test_a_comma_where_the_speaker_really_paused_is_left_alone_and_a_unit_without_commas_has_nothing_to_report():
    heard = _words(("Mfundo", 0.0, 0.5), ("it", 1.2, 1.3), ("seemed", 1.3, 1.8))              # a real 0.7 s pause after the name
    assert cc.comma_check("Mfundo, kubonakale", heard, 0, 1800) is None
    assert cc.comma_check("Impela Ayanda sethemba", HEARD, 0, 4300) is None


def test_a_stretch_with_far_more_commas_than_the_speaker_had_pauses_is_flagged_as_over_punctuated():
    heard = _words(*[(f"w{i}", i * 0.3, i * 0.3 + 0.28) for i in range(40)])                    # 40 words spoken straight through: no pause
    text = "a, b, c, d, e, f, g, h, i."
    found = cc.comma_check(text, heard, 0, 12000)
    assert found and found["commas"] == 8 and "over_punctuated" in found and "0 pauses" in found["over_punctuated"]


def test_the_prefix_before_a_name_is_grammar_and_a_name_the_recogniser_spelt_its_own_way_still_matches():
    heard = _words(("so", 0.0, 0.3), ("Kenny", 0.3, 0.7), ("Mpanga", 0.7, 1.2), ("has", 1.2, 1.5))
    found = cc.comma_check("Kulungile, noKenny Maphanga, siyabonga", heard, 0, 1500)
    assert found and any("Kenny" in x or "Maphanga" in x for x in found["commas_where_the_speaker_did_not_pause"])


def test_the_english_is_shown_with_the_pauses_the_speaker_really_took():
    english = "Indeed, Ayanda, hopefully we will get answers. Nokho it was interesting"
    assert cc.phrased(english, HEARD, 0, 4300) == "Indeed, Ayanda, hopefully we will get answers. |0.5s| Nokho it was interesting"          # 0.5 s after "answers."; none round Ayanda
    short = _words(("a", 0.0, 0.2), ("b", 0.5, 0.7), ("c", 0.75, 0.9))
    assert cc.phrased("a b c", short, 0, 900) == "a |0.3s| b c"                                  # 0.3 s: a pause worth showing; 0.05 s: none


def test_a_draft_is_checked_while_it_is_a_draft_by_finding_where_its_english_was_said():
    clip = HEARD + _words(("Next", 6.0, 6.4), ("topic", 6.4, 6.9), ("please", 6.9, 7.3))
    assert cc.locate("Indeed Ayanda hopefully we will get answers", clip) == (0, 2400)
    assert cc.locate("something entirely different was said here", clip) is None
    said = cc.draft_check("Impela, Ayanda, sethemba sizozithola izimpendulo.", "Indeed, Ayanda, hopefully we will get answers.", clip)
    assert said and "Ayanda" in said[0]
    assert cc.draft_check("Impela Ayanda sethemba sizozithola izimpendulo.", "Indeed, Ayanda, hopefully we will get answers.", clip) == []


def test_a_clip_of_several_speakers_is_a_conversation_and_one_speaker_is_not():
    talk = cc.conversation([{"speaker_id": "SPEAKER_00"}, {"speaker_id": "SPEAKER_01"}, {"speaker_id": "SPEAKER_02"}, {"speaker_id": "SPEAKER_00"}])
    assert talk["speakers"] == 3 and talk["turn_order"] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02", "SPEAKER_00"] and "conversation between 3 people" in talk["note"]
    assert cc.conversation([{"speaker_id": "SPEAKER_00"}, {"speaker_id": "SPEAKER_00"}]) is None


def _held(text, before, after_gap, after_text="Masemola."):
    """One English sentence said straight through, a pause of ``after_gap`` s, then its last word."""
    spec, t = [], 0.0
    for word in before.split():
        spec.append((word, t, t + 0.3))
        t += 0.3
    t += after_gap
    spec.append((after_text, t, t + 0.5))
    return cc.comma_check(text, _words(*spec), 0, int((t + 0.5) * 1000) + 50)


ENGLISH_BEFORE = "we know Dumisani Khumalo and Nhlanhla Mkhwanazi who was the big catch as well as"
ZULU_NO_PAUSE = "Ubale uDumisani Khumalo noNhlanhla Mkhwanazi, okuthiwa nguyena ababefuna kakhulu, kanye noFannie Masemola."


def test_a_pause_the_speaker_held_inside_a_sentence_is_reported_when_the_zulu_runs_straight_on():
    found = _held(ZULU_NO_PAUSE, ENGLISH_BEFORE, 1.4)
    assert found and "pauses_the_speaker_held_that_the_dub_does_not" in found
    said = " ".join(found["pauses_the_speaker_held_that_the_dub_does_not"])
    assert "as well as" in said and "1.4 s" in said and "~pause1400" in said


def test_a_pause_marker_of_that_length_at_the_place_satisfies_the_check_and_a_short_marker_does_not():
    assert _held("Ubale uDumisani Khumalo noNhlanhla Mkhwanazi, okuthiwa nguyena ababefuna kakhulu, kanye ~pause1400 noFannie Masemola.", ENGLISH_BEFORE, 1.4) is None
    short = _held("Ubale uDumisani Khumalo noNhlanhla Mkhwanazi, okuthiwa nguyena ababefuna kakhulu, kanye ~pause400 noFannie Masemola.", ENGLISH_BEFORE, 1.4)
    assert short and "pauses_the_speaker_held_that_the_dub_does_not" in short


def test_a_pause_shorter_than_eight_tenths_of_a_second_is_not_held_against_the_zulu():
    assert _held(ZULU_NO_PAUSE, ENGLISH_BEFORE, 0.6) is None


def test_a_pause_at_a_sentence_end_is_covered_by_the_full_stop_up_to_about_a_second_and_needs_a_marker_beyond():
    spec = [("It", 0.0, 0.3), ("ended.", 0.3, 0.8), ("Then", 2.1, 2.4), ("more", 2.4, 2.8)]              # a 1.3 s pause after the full stop
    words = _words(*spec)
    assert cc.comma_check("Kwaphela. Bese kuqhubeka", words, 0, 2850) is None                          # the stop's 0.92 s and the renderer's move of up to 0.9 s cover 1.3 s
    longer = _words(("It", 0.0, 0.3), ("ended.", 0.3, 0.8), ("Then", 3.0, 3.3), ("more", 3.3, 3.7))    # 2.2 s
    found = cc.comma_check("Kwaphela. Bese kuqhubeka", longer, 0, 3750)
    assert found and "~pause2200" in " ".join(found["pauses_the_speaker_held_that_the_dub_does_not"])
    assert cc.comma_check("Kwaphela. ~pause2200 Bese kuqhubeka", longer, 0, 3750) is None


def test_the_full_stop_that_ends_the_sentence_does_not_stand_in_for_a_pause_the_speaker_held_before_its_last_words():
    found = _held(ZULU_NO_PAUSE, ENGLISH_BEFORE, 1.2)                      # a 1.2 s pause before "Masemola"; the Zulu's stop comes after it
    assert found and "1.2 s" in " ".join(found["pauses_the_speaker_held_that_the_dub_does_not"])
