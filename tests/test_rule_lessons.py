"""The listener's decisions replayed against the rules: what a change of an open choice would fix and what it would break."""
from __future__ import annotations

from mathula_tv import rule_lessons as rl

ENTRIES = {
    "staff": {"word": "staff", "kind": "custom", "spelling": {"*": "staf"}, "jobs": ["j1"], "rule_spelling": "staaf"},
    "africa": {"word": "Africa", "kind": "custom", "spelling": {"*": "afrrika"}, "jobs": ["j1"], "rule_spelling": "aafika"},
    "manifesto": {"word": "manifesto", "kind": "approved", "spelling": {"*": "manifeesto"}, "jobs": ["j1"], "rule_spelling": "manifeesto"},
    "court": {"word": "Court", "kind": "approved", "spelling": {"*": "khoot"}, "jobs": ["j2"], "rule_spelling": "khoot"},
    "4th": {"word": "4th", "kind": "approved", "spelling": {"*": "fooo"}, "jobs": ["j1"], "rule_spelling": "fooo"},          # a day is spelled by the number rule: not replayed
    "mk": {"word": "MK", "kind": "custom", "spelling": {"zu-ZA-ThembaNeural": "Em Kheeyi"}, "jobs": [], "rule_spelling": "em-kheeyi"},          # only one voice: not compared
}


def _rules(defaults):
    """A stand-in for the rules: short_stress=0 writes a stressed short vowel once (staf), which also unwrites manifeesto; join=' ' changes nothing here."""
    once = defaults.get("short_stress") == 0
    return {"staff": "staf" if once else "staaf", "Africa": "aafika", "manifesto": "manifesto" if once else "manifeesto", "Court": "khoot"}


def _run(defaults):
    return rl.judge(rl.decisions(ENTRIES), lambda word, jobs: _rules(defaults).get(word))


def test_decisions_are_the_words_with_a_spelling_for_the_voice_custom_ones_first():
    rows = rl.decisions(ENTRIES)
    assert [(r["word"], r["kind"]) for r in rows] == [("Africa", "custom"), ("staff", "custom"), ("Court", "approved"), ("manifesto", "approved")]


def test_judge_compares_the_rules_now_with_what_was_decided_and_a_word_the_rules_cannot_read_never_agrees():
    judged = {r["word"]: r for r in rl.judge(rl.decisions(ENTRIES), lambda word, jobs: {"staff": "STAF", "Court": "khoot"}.get(word))}
    assert judged["staff"]["agrees"] and judged["Court"]["agrees"] and not judged["Africa"]["agrees"] and judged["Africa"]["rules_now"] is None
    assert rl.tally(list(judged.values())) == {"approved": 2, "approved_kept": 1, "custom": 2, "custom_written": 1}


def test_sweep_lists_each_change_with_the_words_it_fixes_and_the_approved_ones_it_breaks():
    labels = {"short_stress": ("A short vowel with the main stress", {1: "written twice", 0: "written once"}), "join": ("Between the letters", {"-": "a hyphen", " ": "a space"})}
    found = rl.sweep(_run, labels)
    assert [(c["rule"], c["option"]) for c in found] == [("short_stress", 0)]                          # join changes nothing here, so it is not listed
    change = found[0]
    assert change["fixes"] == ["staff"] and change["breaks"] == ["manifesto"] and change["net"] == 0 and change["changes"] == 2
    assert change["examples"]["staff"] == "staaf → staf (you: staf)"


def test_sweep_starts_from_the_defaults_the_listener_has_picked_and_ranks_by_net_gain():
    labels = {"short_stress": ("A short vowel with the main stress", {1: "written twice", 0: "written once"}), "length": ("A stressed long vowel", {1: "twice", 0: "once"})}

    def run(defaults):
        rows = _run(defaults)
        if defaults.get("length") == 0:                                                                  # a change that fixes Africa and breaks nothing
            return [{**r, "rules_now": "afrrika", "agrees": True} if r["word"] == "Africa" else r for r in rows]
        return rows

    found = rl.sweep(run, labels)
    assert [(c["rule"], c["option"], c["net"]) for c in found] == [("length", 0, 1), ("short_stress", 0, 0)]
    again = rl.sweep(run, labels, current={"short_stress": 0})                                           # already picked: the other option is the one tried
    assert [(c["rule"], c["option"]) for c in again if c["rule"] == "short_stress"] == [("short_stress", 1)]
