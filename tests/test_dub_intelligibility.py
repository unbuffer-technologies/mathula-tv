from mathula_tv.dub_intelligibility import dropped_words


def heard(*items):
    return [{"text": t, "start": s, "end": e} for t, s, e in items]


SPOKEN = "Abanye bona bazothukuthela lapho bethola ukuthi sebesusiwe kulolo hlu. Kukhona abazokwenqatshwa yimiphakathi."


def test_a_word_the_voice_swallowed_is_reported_where_the_audio_has_a_gap():
    words = heard(("Abanye", 0.0, 0.4), ("bona", 0.4, 0.7), ("bazothukuthela", 0.7, 1.6), ("lapho", 1.6, 1.9), ("bethola", 1.9, 2.4),
                  ("ukuthi", 2.4, 2.8), ("sebesusiwe", 2.8, 3.6), ("kulolo", 3.6, 4.1), ("Kukhona", 5.2, 5.7),
                  ("abazokwenqatshwa", 5.7, 6.6), ("yimiphakathi", 6.6, 7.4))
    found = dropped_words(SPOKEN, words)
    assert [f["words"] for f in found] == [["hlu."]] and found[0]["after"] == "kulolo" and found[0]["before"] == "Kukhona"
    assert 4.1 < found[0]["at_s"] < 5.2


def test_a_missing_word_with_no_gap_is_the_recogniser_merging_not_the_voice_dropping():
    words = heard(("Abanye", 0.0, 0.4), ("bona", 0.4, 0.7), ("bazothukuthela", 0.7, 1.6), ("lapho", 1.6, 1.9), ("bethola", 1.9, 2.4),
                  ("ukuthi", 2.4, 2.8), ("sebesusiwe", 2.8, 3.6), ("kulolo", 3.6, 4.1), ("Kukhona", 4.25, 4.7),
                  ("abazokwenqatshwa", 4.7, 5.6), ("yimiphakathi", 5.6, 6.4))
    assert dropped_words(SPOKEN, words) == []


def test_a_misheard_word_is_not_a_dropped_word():
    words = heard(("Abanye", 0.0, 0.4), ("bona", 0.4, 0.7), ("bazothukuthela", 0.7, 1.6), ("lapho", 1.6, 1.9), ("bethola", 1.9, 2.4),
                  ("ukuthi", 2.4, 2.8), ("sebesusiwe", 2.8, 3.6), ("kulolo", 3.6, 4.1), ("hlo", 4.1, 4.5), ("Kukhona", 5.6, 6.1),
                  ("abazokwenqatshwa", 6.1, 7.0), ("yimiphakathi", 7.0, 7.8))
    assert dropped_words(SPOKEN, words) == []


def test_nothing_is_reported_at_the_edges_of_a_unit_where_there_is_no_gap_to_measure():
    assert dropped_words("Sawubona bantu", heard(("bantu", 2.0, 2.5))) == []
