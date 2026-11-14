from mathula_tv import naturalness


def corpora():
    spoken = ["ngiyabonga kakhulu baba"] * 30 + ["ngicela usizo lwakho"] * 30
    news = ["umphakathi uthi ngiyabonga kakhulu"] * 30 + ["umbuso uzophendula umphakathi ngokushesha"] * 30 + ["umbuso uthi ngiyabonga baba"] * 5
    return {"spoken": spoken, "news": news, "gov": ["umbuso uzophendula umphakathi ngokushesha"] * 10}


def test_sentences_split_at_stops_pauses_and_keep_hyphenated_forms_and_drop_digits():
    assert naturalness.sentences("Ngiyabonga, baba. ~pause500 Ama-details ngo-2017 lapha?") == [["ngiyabonga", "baba"], ["ama-details", "ngo", "lapha"]]


def test_counts_and_pairs_come_from_the_right_source(tmp_path):
    index = naturalness.NgramIndex.build(corpora())
    assert index.word("ngiyabonga") == 65 and index.word("ngiyabonga", "spoken") == 30
    assert index.pair("ngiyabonga", "kakhulu") == 60 and index.pair("ngiyabonga", "kakhulu", "spoken") == 30 and index.pair("kakhulu", "ngiyabonga") == 0
    naturalness.NgramIndex.build(corpora()).save(tmp_path)
    again = naturalness.NgramIndex.load(tmp_path)
    assert again.pair("ngiyabonga", "baba") == index.pair("ngiyabonga", "baba") and again.tokens == index.tokens
    assert naturalness.NgramIndex.load(tmp_path / "nowhere") is None


def test_an_unseen_pair_of_common_words_is_reported_only_when_chance_alone_would_have_met_them():
    index = naturalness.NgramIndex.build(corpora())
    found = naturalness.check(index, "kakhulu ngiyabonga baba.", min_word=20, expected_min=0.5)
    assert "kakhulu ngiyabonga" in found["unseen_pairs"] and "ngiyabonga baba" not in found["unseen_pairs"]          # the second pair is attested
    assert naturalness.check(index, "kakhulu ngiyabonga baba.", min_word=20, expected_min=1000)["unseen_pairs"] == []          # absence means nothing when chance would not have met them either
    assert naturalness.check(index, "xyzzy qwerty.", min_word=20)["unseen_pairs"] == []                                      # rare words are not evidence


def test_a_word_the_written_corpora_use_often_and_speakers_never_do_is_bookish():
    spoken = ["ngiyabonga baba"] * 400
    news = ["ngiyabonga baba"] * 50 + ["kuphawulwe ukuthi umbuso"] * 400
    index = naturalness.NgramIndex.build({"spoken": spoken, "news": news, "gov": []})
    assert "kuphawulwe" in naturalness.check(index, "kuphawulwe.")["bookish_words"]
    assert naturalness.check(index, "ngiyabonga baba.")["bookish_words"] == []


def test_cadence_flags_only_a_unit_chopped_into_sentences_shorter_than_almost_any_real_speaker_uses():
    from mathula_tv.cadence import cadence_line, cadence_lines

    chopped = {"item_id": "u5", "source_text": "We went to the office yesterday. It was closed all day long.",
               "zulu_text": "Sahamba. Safika. Kwakuvaliwe. Sabuya. Sathula. Sahamba futhi."}
    line = cadence_line(chopped)
    assert line and "6 Zulu sentences for 2 English" in line and "0.9 s of silence" in line
    natural = {"item_id": "u6", "source_text": "We went to the office yesterday. It was closed all day long.",
               "zulu_text": "Sihambe sayofika ehhovisi izolo kodwa safica livaliwe usuku lonke. Sabuya sinesizungu esikhulu kakhulu ngalokho."}
    assert cadence_line(natural) is None
    long_english = {"item_id": "u7", "source_text": "One long sentence.", "zulu_text": "Sahamba. Safika. Kwakuvaliwe."}          # three short sentences but not 1.5 times the English count (3 vs 1 is, so flagged)
    assert cadence_line(long_english) is not None
    equal = {"item_id": "u8", "source_text": "A. B. C. D.", "zulu_text": "Sahamba. Safika. Kwakuvaliwe. Sabuya."}
    assert cadence_line(equal) is None                                                  # as many sentences as the English: the speaker chopped it, not the dub
    assert [x["item_id"] for x in cadence_lines([chopped, natural, equal])] == ["u5"]


def test_the_workbench_shows_cadence_without_an_index_and_pairs_with_one(tmp_path):
    from pathlib import Path

    from mathula_tv.dub_workbench import Workbench

    class Store:
        root = tmp_path / "work" / "experience"

    class Bench:
        store = Store()
        _naturalness = Workbench._naturalness

    chopped = {"item_id": "u5", "source_text": "We went to the office yesterday. It was closed all day long.", "zulu_text": "Sahamba. Safika. Kwakuvaliwe. Sabuya. Sathula."}
    found = Bench()._naturalness([chopped, {"item_id": "u1", "source_text": "Hello.", "zulu_text": "Sawubona."}])
    assert [f["item_id"] for f in found] == ["u5"] and "cadence" in found[0] and "pairs_that_never_meet_in_real_zulu" not in found[0]          # no index built: cadence only
    index = naturalness.NgramIndex.build({"spoken": ["ngiyabonga baba"] * 60, "news": ["ngiyabonga kakhulu"] * 400 + ["umbuso uthi"] * 400, "gov": []})
    index.save(Path(tmp_path / "work" / "resources" / "naturalness"))
    bench = Bench()
    found = bench._naturalness([{"item_id": "u2", "source_text": "x.", "zulu_text": "Umbuso baba ngiyabonga."}])
    assert found and "umbuso baba" in found[0]["pairs_that_never_meet_in_real_zulu"]


def test_the_cadence_point_reaches_the_wrapup_and_the_submit_result():
    from mathula_tv.dub_agent import open_listener_points, slim_submit

    result = {"accepted": True, "headline": {}, "naturalness": [{"item_id": "u5", "cadence": "6 Zulu sentences for 2 English, 3.8 words each: ..."}, {"item_id": "u9", "pairs_that_never_meet_in_real_zulu": ["a b"]}]}
    assert open_listener_points(result) == ["u5: 6 Zulu sentences for 2 English, 3.8 words each: ..."]
    slim = slim_submit(result)
    assert slim["naturalness"] == result["naturalness"] and "never occur together" in slim["naturalness_note"]


def _register_index():
    gov = ["umqondisi uthe isinyathelo esilandelayo siyabandakanya ukuhlolwa kwezidingo zomphakathi ngokuhambisana nomgomo"] * 400
    native = ["ngiyabonga kakhulu baba sihambe sihambe kusasa sibuye"] * 300 + ["umphakathi uthe uyabonga kakhulu ngoba izinto zihamba kahle namuhla"] * 300
    return naturalness.NgramIndex.build({"spoken": native[:300], "news": native[300:], "gov": gov})


def test_a_sentence_of_words_that_belong_to_translated_government_text_scores_high_and_native_wording_scores_low():
    index = _register_index()
    stilted = ["umqondisi", "uthe", "isinyathelo", "esilandelayo", "siyabandakanya", "ukuhlolwa", "kwezidingo"]
    native = ["ngiyabonga", "kakhulu", "baba", "sihambe", "sihambe", "kusasa", "sibuye"]
    assert naturalness.sentence_register(index, stilted) > naturalness.REGISTER_CUT_SINGLE
    assert naturalness.sentence_register(index, native) < 0
    assert naturalness.sentence_register(index, ["ngiyabonga", "baba"]) is None                  # too short to score
    assert naturalness.word_register(index, "zzz") is None                                         # a word too rare to say anything


def test_a_unit_is_flagged_for_translationese_by_two_sentences_in_a_row_or_one_clear_one_and_not_by_native_speech():
    index = _register_index()
    one = "umqondisi uthe isinyathelo esilandelayo siyabandakanya ukuhlolwa kwezidingo."
    flagged = naturalness.register_check(index, one + " " + one)
    assert flagged and flagged["above"] == 2 and flagged["sentences"] == 2 and flagged["pulling"]
    assert naturalness.register_check(index, one)["above"] == 1                                     # a single, clearly stilted sentence
    assert naturalness.register_check(index, "ngiyabonga kakhulu baba sihambe sihambe kusasa sibuye. umphakathi uthe uyabonga kakhulu ngoba izinto zihamba kahle.") is None
    assert naturalness.register_check(index, "") is None


def test_the_workbench_names_a_unit_that_reads_like_translated_government_zulu(tmp_path):
    from pathlib import Path

    from mathula_tv.dub_workbench import Workbench

    class Store:
        root = tmp_path / "work" / "experience"

    class Bench:
        store = Store()
        _naturalness = Workbench._naturalness

    _register_index().save(Path(tmp_path / "work" / "resources" / "naturalness"))
    one = "umqondisi uthe isinyathelo esilandelayo siyabandakanya ukuhlolwa kwezidingo."
    found = Bench()._naturalness([{"item_id": "u3", "source_text": "x. y.", "zulu_text": one + " " + one}, {"item_id": "u4", "source_text": "x.", "zulu_text": "Ngiyabonga kakhulu baba sihambe kusasa sibuye."}])
    assert [f["item_id"] for f in found] == ["u3"] and "score like translated government text" in found[0]["reads_like_translated_government_zulu"]
