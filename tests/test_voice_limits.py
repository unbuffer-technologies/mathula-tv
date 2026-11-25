from mathula_tv import dub_agent, voice_limits


def test_every_form_of_a_word_the_voice_stumbles_on_is_found_and_ordinary_words_are_not():
    for text in ("akudingeki angene emininingwaneni", "usenikeza imininingwane", "nemininingwane eminingi", "ngemininingwane"):
        lines = voice_limits.find(text)
        assert lines and "ama-details" in lines[0], text
    assert voice_limits.find("usenikeza ulwazi oluthile, nomuntu omningi emnyango") == []


def test_the_units_with_such_a_word_are_listed_and_reach_the_wrapup_points():
    found = voice_limits.for_items([{"item_id": "unit-003", "zulu_text": "usenikeza imininingwane yokuthi ubani"}, {"item_id": "unit-004", "zulu_text": "Sawubona"}])
    assert [f["item_id"] for f in found] == ["unit-003"]
    points = dub_agent.open_listener_points({"voice_limit_words": found})
    assert points and points[0].startswith("unit-003: ") and "ama-details" in points[0]


def test_the_agent_prompt_leaves_the_word_and_its_replacement_to_the_check_not_to_a_rule_for_one_word():
    assert "known to stumble on" in dub_agent.SYSTEM_PROMPT and "check_text" in dub_agent.SYSTEM_PROMPT
    assert "imininingwane" not in dub_agent.SYSTEM_PROMPT                     # the check names the word and what to write instead (the points above): the prompt keeps the principle, not the instance


def test_an_english_word_kept_after_any_plural_or_locative_concord_is_found_for_the_pronunciation_list():
    from mathula_tv.job_pronunciations import core

    for token in ("ama-details", "kuma-details", "nama-details", "ngama-details", "kwama-details", "ngoma-details"):
        assert core(token) == "details", token
    assert core("multi-party") == "multi-party"          # an English compound is left alone: the concord rule only adds pieces ending in -ma


def test_a_hummed_backchannel_is_flagged_but_zulu_words_are_not():
    from mathula_tv import voice_limits

    assert voice_limits.find("Ngoba izintshisekelo zingadlula ilungelo lokuphepha. Mm-hmm.")
    assert voice_limits.find("Hmm, cha.") and voice_limits.find("Mhm.") and voice_limits.find("mmm")
    assert not voice_limits.find("Umhlaba umhlambi mhlophe, uMhlanga, imimoya, ummeli, abantu.")          # real words that hold m and h


def test_a_murmur_in_the_english_is_found_so_the_clip_open_can_name_it():
    from mathula_tv.voice_limits import MURMUR_NOTE, murmurs

    assert murmurs("…the right of Matlala to safety. Mm-hmm.") == ["Mm-hmm"]
    assert murmurs("Mhm. There's quite a lot. Uh-huh, mhm") == ["Mhm", "Uh-huh"]
    assert murmurs("Hmm, I think the Mhlanga committee met in Umhlanga.") == []          # hmm is flagged in the Zulu text, not as a murmur; real words with m and h are not
    assert "Leave it out" in MURMUR_NOTE
