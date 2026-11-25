from mathula_tv.dub_agent import SYSTEM_PROMPT, TOOL_SPECS
from mathula_tv.pronunciation import PronunciationDictionary, with_default_organisation_initialisms
from mathula_tv.say_it import say_it


def _dictionary():
    entries = [{"display_text": w, "spoken_text": w, "tts_text": t, "kind": "english_code_switch"} for w, t in (("Section", "seekshin"), ("Arthur", "aatha"))]
    return with_default_organisation_initialisms(PronunciationDictionary("v1", language="zu-ZA", job_id="job-1")).with_job_overrides("job-1", entries)


def test_the_agent_sees_the_spoken_form_change_by_change():
    out = say_it(_dictionary(), "Sinakho sinoDkt Arthur Shopola. Amakomidi eSection 79, ngo-2006 ~pause500 ku-75.8% kuphela.")
    assert out["spoken"] == "Sinakho sinoDokotela aatha Shopola. Amakomidi eseekshin seventy-nayn, ngo-twenty-oh-six ku-seventy-fayiv-phoynt-eyt wamaphesenti, kuphela."
    kinds = {c["was"]: c["now"] for c in out["changes"]}
    assert kinds["sinoDkt"] == "sinoDokotela" and kinds["79"] == "seventy-nayn" and kinds["ngo-2006"] == "ngo-twenty-oh-six"
    assert out["still_as_written"] == []                                                      # every digit and title was converted
    assert "~pause" not in out["spoken"]                                                        # a pause marker is not speech


def test_double_quote_marks_are_not_sent_to_the_voice_because_it_adds_a_sound_after_the_word_inside_them():
    out = say_it(_dictionary(), "Kenny uVusimuzi “Cat” Matlala ufakaza, uthe \"yebo\" kodwa sasingaphumeleli, i-Simaya’s «phezulu».")
    assert "“" not in out["spoken"] and "”" not in out["spoken"] and '"' not in out["spoken"] and "«" not in out["spoken"]
    assert out["spoken"].startswith("Kenny uVusimuzi Cat Matlala ufakaza, uthe yebo kodwa") and "Simaya’s phezulu." in out["spoken"]          # the word stays, the apostrophe of a word stays


def test_what_no_rule_converted_is_listed_so_the_agent_can_write_it_another_way():
    out = say_it(_dictionary(), "Ikhodi 12 yafakwa ngoMnu. Adams, Th. Smith, eSABC kanye ne-IEC ngo-14:30 kanye 2/3 ne-ABC123")
    assert not any("14:30" in line for line in out["still_as_written"])                           # a time is read by the time rule now (2026-11-12)
    assert any("2/3" in line for line in out["still_as_written"])
    assert not any("Dkt" in line for line in out["still_as_written"])


def test_the_agent_has_the_tool_and_is_told_to_use_it():
    spec = next(s for s in TOOL_SPECS if s["function"]["name"] == "say_it")
    assert "voice will receive" in spec["function"]["description"] and "still_as_written" in spec["function"]["description"]
    assert "See what the voice gets" in SYSTEM_PROMPT and "say_it" in SYSTEM_PROMPT


def test_the_submit_result_shows_the_units_the_voice_would_guess_at_without_a_call():
    from mathula_tv.dub_agent import slim_submit
    from mathula_tv.dub_workbench import Workbench

    class Bench:
        say_it = lambda self, text, speaker_id="SPEAKER_00": say_it(_dictionary(), text)          # noqa: E731
        _spoken_form_gaps = Workbench._spoken_form_gaps

    found = Bench()._spoken_form_gaps([{"item_id": "u1", "zulu_text": "Ngo-5 abantu eSection 79.", "speaker_id": "SPEAKER_00"},
                                        {"item_id": "u2", "zulu_text": "Kuqala 2/3 kusasa.", "speaker_id": "SPEAKER_00"}])
    assert [g["item_id"] for g in found] == ["u2"]                                              # u1 is fully converted; u2 keeps a fraction for the voice
    slim = slim_submit({"accepted": True, "headline": {}, "spoken_form_gaps": found})
    assert slim["spoken_form_gaps"] == found and "Soft evidence" in slim["spoken_form_gaps_note"]


def test_the_submit_result_shows_each_number_with_the_words_the_voice_is_given():
    from mathula_tv.dub_agent import slim_submit
    from mathula_tv.dub_workbench import Workbench

    class Bench:
        say_it = lambda self, text, speaker_id="SPEAKER_00": say_it(_dictionary(), text)          # noqa: E731
        _figures_as_spoken = Workbench._figures_as_spoken

    found = Bench()._figures_as_spoken([{"item_id": "u1", "zulu_text": "Sasazi ukuthi ngo-9.30 kwakuzoba, ngo-8 ekuseni.", "speaker_id": "SPEAKER_00"},
                                         {"item_id": "u2", "zulu_text": "Akukho zinombolo lapha.", "speaker_id": "SPEAKER_00"}])
    assert [g["item_id"] for g in found] == ["u1"]
    assert "ngo-9.30 -> ngo-nayn thethi" in found[0]["spoken"]                                # a time is shown as a time, so a decimal reading would be seen
    slim = slim_submit({"accepted": True, "headline": {}, "figures_as_spoken": found})
    assert slim["figures_as_spoken"] == found and "Soft evidence" in slim["figures_as_spoken_note"]
