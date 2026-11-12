from mathula_tv.dub_agent import SYSTEM_PROMPT
from mathula_tv.listening_load import figures_run_together, long_runs

PROD_76_OPENING = ("Ukwesekwa kwe-ANC okhethweni lohulumeni basekhaya eNorth West kwehle ngaphezu kwamaphesenti angu-31% eminyakeni engu-20. "
                   "Kusuke ku-75.8% ngo-2006 kwaya ku-44.05% kulo nyaka, yalahlekelwa omasipala abangu-11.")


def test_two_figures_with_nothing_between_them_are_flagged():
    found = figures_run_together("kwehle ngo-2006 kwaya ku-44.05% kulo nyaka")
    assert len(found) == 1 and "2006" in found[0] and "44.05%" in found[0]


def test_a_percentage_followed_by_words_counts_as_a_breath_because_the_renderer_adds_a_comma():
    assert figures_run_together("Kusuke ku-75.8% ngo-2006") == []                      # "%" + word: the renderer's comma separates them
    assert figures_run_together("ku-75.8%, ngo-2006") == []                            # the agent's own comma
    assert figures_run_together("ngo-2006. Kwaya ku-44.05%") == []                     # a sentence end
    assert figures_run_together("ngo-2006 ~pause500 ku-44.05%") == []                  # a marked pause is a breath, and its digits are not a figure


def test_figures_far_apart_are_not_run_together():
    assert figures_run_together("Ngo-2006 iqembu lalinabantu abaningi kakhulu futhi lalinamandla amakhulu ezweni lonke, kwaya ku-44.05%") == []


def test_the_real_opening_of_a_dub_is_checked():
    flags = figures_run_together(PROD_76_OPENING)
    assert flags == [] or all("nothing separates" in f for f in flags)
    assert len(figures_run_together("Kusuke ku-75.8% okhethweni lomasipala ngo-2006 laya ku-44.05% kulo nyaka")) == 1          # 2006 runs into 44.05%


def test_a_long_run_without_a_breath_is_flagged_only_past_about_six_seconds():
    long = "Abantu bafike ekhaya namuhla bese beyaphuma ngokushesha ukuze baye edolobheni elikhulu lapho bezothenga khona konke abakudingayo emaphakathini"
    assert len(long_runs(long)) == 1 and "s with no breath" in long_runs(long)[0]
    assert long_runs("Abantu bafike ekhaya namuhla, bese beyaphuma ngokushesha, ukuze baye edolobheni elikhulu, lapho bezothenga khona konke.") == []
    assert long_runs("Abantu bafike ekhaya.") == []


def test_the_agent_prompt_teaches_the_listener_not_only_the_marks():
    assert "Listening load" in SYSTEM_PROMPT and "cannot re-read" in SYSTEM_PROMPT
    assert "figures_run_together" in SYSTEM_PROMPT and "long_run_without_breath" in SYSTEM_PROMPT and "breaths_inside" in SYSTEM_PROMPT


def test_the_submit_result_carries_the_load_flags_whether_or_not_check_text_was_called():
    from mathula_tv.dub_agent import slim_submit
    from mathula_tv.dub_workbench import Workbench

    found = Workbench._listening_load([
        {"item_id": "u1", "zulu_text": "Kwehle ngo-2006 kwaya ku-44.05% kulo nyaka."},
        {"item_id": "u2", "zulu_text": "Sawubona, mhlali wami."},
        {"item_id": "u3", "zulu_text": "Abantu bafike ekhaya namuhla bese beyaphuma ngokushesha ukuze baye edolobheni elikhulu lapho bezothenga khona konke abakudingayo emaphakathini"}])
    assert [f["item_id"] for f in found] == ["u1", "u3"]
    assert "figures_run_together" in found[0] and "long_run_without_breath" in found[1]
    slim = slim_submit({"accepted": True, "headline": {}, "listening_load": found})
    assert slim["listening_load"] == found and "Soft evidence" in slim["listening_load_note"]
    assert long_runs("Abantu bafike ekhaya namuhla bese beyaphuma ngokushesha ukuze baye edolobheni elikhulu lapho") == []          # ~7 s: under the 8 s line
