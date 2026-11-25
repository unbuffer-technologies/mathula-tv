from pathlib import Path

from mathula_tv import attestation, dub_agent


def _news(tmp_path: Path, text: str) -> Path:
    web = tmp_path / "resources" / "attestation" / "web"
    web.mkdir(parents=True)
    (web / "web_news.txt").write_text(text, encoding="utf-8")
    return tmp_path


def test_examples_show_real_sentences_where_the_word_follows_a_comma_and_where_it_does_not(tmp_path):
    work = _news(tmp_path, "\n".join([
        "Umnumzane uthe ukhetho luzoba khona ngoMsombuluko, kodwa abaningi abakaqinisekisi ukuthi bazovota.",
        "Abantu bebesemgwaqweni bethi bayazi kodwa abakakwazi ukuchaza kahle ukuthi kwenzekeni lapho.",
        "Uhulumeni uthe uzokwakha izikole ezintsha kodwa inani lezimali alikaziwa namhlanje ekuseni.",
        "Imvula inele kakhulu eGoli, kodwa izindawo eziningi zisaphila ngaphandle kwamanzi alo nyaka.",
    ]))
    answer = attestation.punctuation_examples(work, "kodwa", limit=2)
    assert answer["phrase"] == "kodwa"
    assert all(", kodwa" in s for s in answer["after_a_comma"]) and len(answer["after_a_comma"]) == 2
    assert all(", kodwa" not in s and "kodwa" in s for s in answer["not_after_a_comma"]) and len(answer["not_after_a_comma"]) == 2
    assert "uthe ukhetho luzoba khona ngoMsombuluko, kodwa" in answer["after_a_comma"][0]              # a whole real sentence, with its own punctuation and case
    assert "rule" in answer["note"] or "must" in answer["note"]                                        # it says these are examples, not a rule


def test_a_phrase_that_is_not_there_and_a_bad_phrase_are_answered_without_failing(tmp_path):
    work = _news(tmp_path, "Sawubona baba bonke abantu abakhona lapha namhlanje ekuseni kakhulu.")
    assert attestation.punctuation_examples(work, "kanti")["after_a_comma"] == []
    assert "error" in attestation.punctuation_examples(work, "") and "error" in attestation.punctuation_examples(work, "a b c d")
    assert "error" in attestation.punctuation_examples(tmp_path / "nowhere", "kanti")


def test_the_agent_has_the_tool_and_is_told_the_speakers_window_comes_first_their_sentence_ends_are_a_best_effort_and_a_comma_is_not_a_breath():
    names = {spec["function"]["name"] for spec in dub_agent.TOOL_SPECS}
    assert "punctuation_examples" in names
    prompt = " ".join(dub_agent.SYSTEM_PROMPT.split())
    assert "starts and stops with the English speaker" in prompt and "never a requirement" in prompt and "Fluent Zulu comes first" in prompt           # the window first, the speaker's sentence ends a best effort
    assert "It is never there to give the voice a breath" in prompt and "punctuation_examples(word)" in prompt
    assert "a comma where a speaker would take a breath" not in prompt and "breath group" not in prompt            # the rules that turned commas into breaths are gone


def test_every_tool_the_agent_can_call_is_wired_in_the_dispatch():
    from types import SimpleNamespace

    workbench = SimpleNamespace(**{spec["function"]["name"]: (lambda *a, **k: None) for spec in dub_agent.TOOL_SPECS})
    dispatch = dub_agent.build_dispatch(workbench, clip_id="c", run_id="r", tts_budget=5)
    assert "punctuation_examples" in dispatch
