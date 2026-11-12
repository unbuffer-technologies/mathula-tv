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


def test_the_agent_prompt_names_the_word_and_what_to_write_instead():
    assert "imininingwane" in dub_agent.SYSTEM_PROMPT and "ama-details" in dub_agent.SYSTEM_PROMPT


def test_an_english_word_kept_after_any_plural_or_locative_concord_is_found_for_the_pronunciation_list():
    from mathula_tv.job_pronunciations import core

    for token in ("ama-details", "kuma-details", "nama-details", "ngama-details", "kwama-details", "ngoma-details"):
        assert core(token) == "details", token
    assert core("multi-party") == "multi-party"          # an English compound is left alone: the concord rule only adds pieces ending in -ma
