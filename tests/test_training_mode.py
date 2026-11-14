import json
from pathlib import Path

from mathula_tv import thinking_report
from mathula_tv.dub_workbench import Workbench

ISU = {"isu": ["i[NPrePre][5]li[BPre][5]su.5-6[NStem]"], "amasu": ["a[NPrePre][6]ma[BPre][6]su.5-6[NStem]"]}


class Bench:
    _log_thinking = Workbench._log_thinking
    _stem_analyses = staticmethod(Workbench._stem_analyses)


def read(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def test_choice_points_are_logged_with_a_check_that_the_form_is_in_the_unit_and_live_mode_logs_nothing(tmp_path, monkeypatch):
    from mathula_tv import zulu_stems

    monkeypatch.setattr(zulu_stems, "analyse", lambda words: {w: ISU[w] for w in words if w in ISU})
    items = [{"item_id": "u16", "zulu_text": "Akafuni ukwembulela umbuso amasu akhe okuzivikela."}, {"item_id": "u17", "zulu_text": "Ngiyathokoza."}]
    Bench()._log_thinking(None, None, items, tmp_path, 1)
    assert not (tmp_path / "thinking.jsonl").exists()                                       # live mode: nothing sent, nothing written
    Bench()._log_thinking(
        [{"unit": "u16", "english": "defence strategy", "zulu": "isu", "alternatives": ["icebo (plan, too general)", "indlela (way)"], "why": "isu is the established word for a strategy",
          "evidence": "glossary + corpus count"},
         {"unit": "u17", "english": "thank you", "zulu": "ngiyabonga", "why": "more common", "evidence": "own sense of Zulu"},
         {"unit": "u99", "english": "ghost", "zulu": "isu", "why": "x"}],
        [{"unit": "u16", "strategy": "retold, not word for word, to fit the window"}], items, tmp_path, 2)
    rows = read(tmp_path / "thinking.jsonl")
    choices = [r for r in rows if r["kind"] == "choice"]
    assert [c["chosen_in_text"] for c in choices] == [True, False, False]                  # amasu holds the stem of isu; ngiyabonga is not in u17; u99 does not exist
    assert [c["unit_known"] for c in choices] == [True, True, False]
    assert choices[0]["alternatives"][0].startswith("icebo") and choices[0]["submission"] == 2
    assert [r for r in rows if r["kind"] == "strategy"][0]["strategy"].startswith("retold")


def test_the_report_reads_the_latest_record_per_choice_the_evidence_tally_the_drift_and_the_reasoning(tmp_path):
    lines = [{"submission": 1, "kind": "choice", "unit": "u16", "english": "strategy", "zulu": "icebo", "alternatives": ["isu"], "why": "first thought", "evidence": "own sense", "chosen_in_text": True, "unit_known": True},
             {"submission": 2, "kind": "choice", "unit": "u16", "english": "Strategy", "zulu": "isu", "alternatives": ["icebo"], "why": "glossary has it", "evidence": "glossary", "chosen_in_text": True, "unit_known": True},
             {"submission": 2, "kind": "choice", "unit": "u3", "english": "in camera", "zulu": "ngasese", "alternatives": [], "why": "lesson said so", "evidence": "set aside: the loanword flag", "chosen_in_text": False, "unit_known": True},
             {"submission": 2, "kind": "strategy", "unit": "u16", "strategy": "retold to fit"}]
    (tmp_path / "thinking.jsonl").write_text("\n".join(json.dumps(x) for x in lines), encoding="utf-8")
    trace = [{"kind": "model", "turn": 3, "reasoning": "I weighed isu against icebo.", "tool_calls": [{"name": "submit_dub"}]}, {"kind": "model", "turn": 4, "tool_calls": []}]
    (tmp_path / "trace.jsonl").write_text("\n".join(json.dumps(x) for x in trace), encoding="utf-8")
    (tmp_path / "agent_summary.json").write_text(json.dumps({"run_id": "r1", "model": "m", "turns": 4, "stop_reason": "agent_finished", "tokens": {"in": 10, "out": 2}, "agent_mode": "training"}), encoding="utf-8")
    data = thinking_report.load(tmp_path)
    assert len(data["choices"]) == 2 and {c["zulu"] for c in data["choices"]} == {"isu", "ngasese"}        # the later 'Strategy' record replaced the earlier one
    report = thinking_report.write_report(tmp_path).read_text(encoding="utf-8")
    assert "**2 choice points** in 2 units" in report and "**1** name a form that is not in the final unit text" in report
    assert "| 1 | glossary |" in report and "Choices that say they set evidence aside: **1**" in report
    assert report.index("### u3") < report.index("### u16")                                                  # units in order
    assert "Strategy: retold to fit" in report and "**Turn 3** (submit_dub)" in report and "I weighed isu against icebo." in report


def test_a_live_run_reports_that_nothing_was_recorded(tmp_path):
    (tmp_path / "agent_summary.json").write_text(json.dumps({"run_id": "r2", "turns": 9}), encoding="utf-8")
    assert "No choice points were recorded" in thinking_report.build_report(tmp_path)


def test_what_the_agent_says_it_wanted_and_was_not_given_is_logged_and_is_the_first_section_of_the_report(tmp_path):
    Bench()._log_thinking(None, None, [{"item_id": "u1", "zulu_text": "x"}], tmp_path, 1,
                          [{"unit": "u1", "wanted": "how often speakers say ukuhlaziya in an interview", "why": "to choose between two verbs"}, {"wanted": "the speaker's emphasis on 'actually'", "why": "to match it"}])
    rows = read(tmp_path / "thinking.jsonl")
    assert [r["kind"] for r in rows] == ["missing_evidence", "missing_evidence"] and rows[0]["unit"] == "u1"
    report = thinking_report.build_report(tmp_path)
    assert "## What the agent said it wanted and was not given" in report and "how often speakers say ukuhlaziya" in report and "whole clip" in report
    assert report.index("## What the agent said it wanted") < report.index("No choice points were recorded")


def test_the_training_tool_spec_and_prompt_ask_for_missing_evidence():
    from mathula_tv.dub_agent import TRAINING_PROMPT, tool_specs

    props = next(s for s in tool_specs("training") if s["function"]["name"] == "submit_dub")["function"]["parameters"]["properties"]
    assert "missing_evidence" in props and "missing_evidence" in TRAINING_PROMPT
