import pytest

from mathula_tv import dub_agent, zulu_grammar


def test_the_reference_holds_the_six_modules_in_order_under_one_header():
    ref = zulu_grammar.reference()
    titles = ["ISIZULU NOUN CLASSES AND AGREEMENT", "ISIZULU TENSE, ASPECT, MOOD AND NEGATION", "ISIZULU RELATIVE CLAUSES", "ISIZULU VERB EXTENSIONS",
              "ISIZULU OBJECT CONCORDS AND THE REFLEXIVE", "ISIZULU NUMBERS"]
    positions = [ref.index(t) for t in titles]
    assert positions == sorted(positions) and ref.startswith("ISIZULU GRAMMAR REFERENCE")
    assert [name for name, _ in zulu_grammar.MODULES] == ["agreement", "tense", "relative", "extensions", "object", "numerals"]


def test_the_reference_stays_within_a_budget_so_that_it_does_not_crowd_out_the_rules():
    words = len(zulu_grammar.reference().split())
    assert 2500 < words < 4200, words                   # about 3,500 words today; a module that doubles in size should be noticed


def test_the_grammar_is_off_by_default_until_the_comparison_has_been_read(monkeypatch):
    monkeypatch.delenv("MATHULA_AGENT_GRAMMAR", raising=False)
    assert zulu_grammar.enabled() is False
    prompt = dub_agent.build_system_prompt()
    assert "ISIZULU GRAMMAR REFERENCE" not in prompt
    assert dub_agent._grammar_version_in(prompt) is None


def test_the_agent_prompt_carries_the_reference_between_the_rules_and_the_playbook_when_asked():
    prompt = dub_agent.build_system_prompt(grammar=True)
    assert prompt.index("Length by grammar") < prompt.index("ISIZULU GRAMMAR REFERENCE") < prompt.index("ISIZULU NUMBERS")
    assert "ISIZULU GRAMMATICAL FOUNDATION" not in prompt and "ntambama" not in prompt


@pytest.mark.parametrize("grammar", [True, False])
def test_the_prompt_never_leaves_a_template_placeholder(grammar):
    prompt = dub_agent.build_system_prompt(grammar=grammar)
    assert "{max_dead_air}" not in prompt and "{grammar_pointer}" not in prompt


def test_the_rules_point_at_the_reference_only_when_it_follows():
    on = dub_agent.build_system_prompt(grammar=True)
    off = dub_agent.build_system_prompt(grammar=False)
    assert "The grammar reference below" in on and "ISIZULU GRAMMAR REFERENCE" in on
    assert "grammar reference" not in off.lower() and "ISIZULU GRAMMAR REFERENCE" not in off      # no sentence pointing at nothing
    assert "To shorten without losing a fact" in off                                               # the rest of the paragraph stays
    assert "grammar reference" not in dub_agent.SYSTEM_PROMPT.lower() and "{grammar_pointer}" not in dub_agent.SYSTEM_PROMPT


def test_the_switch_works_both_ways_and_an_argument_wins_over_the_environment(monkeypatch):
    monkeypatch.setenv("MATHULA_AGENT_GRAMMAR", "1")
    assert zulu_grammar.enabled() is True and "ISIZULU GRAMMAR REFERENCE" in dub_agent.build_system_prompt()
    assert "ISIZULU GRAMMAR REFERENCE" not in dub_agent.build_system_prompt(grammar=False)
    monkeypatch.setenv("MATHULA_AGENT_GRAMMAR", "0")
    assert zulu_grammar.enabled() is False and "ISIZULU GRAMMAR REFERENCE" in dub_agent.build_system_prompt(grammar=True)
    for value in ("on", "true", "yes", "ON"):
        monkeypatch.setenv("MATHULA_AGENT_GRAMMAR", value)
        assert zulu_grammar.enabled() is True


def test_the_run_summary_can_tell_which_grammar_the_agent_saw():
    assert dub_agent._grammar_version_in(dub_agent.build_system_prompt(grammar=True)) == zulu_grammar.VERSION
    assert dub_agent._grammar_version_in(dub_agent.build_system_prompt(grammar=False)) is None
    assert dub_agent._grammar_version_in(dub_agent.SYSTEM_PROMPT) is None


@pytest.mark.parametrize("module", [m for _, m in zulu_grammar.MODULES])
def test_every_module_renders_text_without_template_leftovers(module):
    text = module()
    assert len(text.split()) > 300 and "{" not in text and "}" not in text and "None" not in text
