import json

from mathula_tv.dub_judge import UnitJudge, foreign_words, judge_run, judge_units, summarise, units_from_scorecard


class FakeClient:
    """Answers the back-translate prompt and the compare prompt from canned tables; records every call."""

    def __init__(self, back, scores):
        self.back, self.scores, self.calls = back, scores, []

    def complete(self, messages, *, json_mode=False, max_output_tokens=0, tools=None):
        system, payload = messages[0]["content"], json.loads(messages[1]["content"])
        self.calls.append((system[:12], payload))
        if "translate isiZulu" in system:
            items = [{"id": p["id"], "english": self.back[p["id"]]} for p in payload]
        elif system.startswith("A reviewer flagged"):         # the confirmation pass: answer from the "confirm" entry
            items = [{"id": p["id"], **(self.scores[p["id"]].get("confirm") or {"meaning_preserved": 2, "issues": []})}
                     for p in payload]
        else:
            items = [{"id": p["id"], **self.scores[p["id"]]} for p in payload if p["id"] in self.scores]
        return {"choices": [{"message": {"content": json.dumps({"items": items})}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


UNITS = [{"id": "a", "source": "He was arrested on Monday.", "dub": "Waboshwa ngoMsombuluko."},
         {"id": "b", "source": "She denied the claim.", "dub": "Wavuma isicelo."}]


def test_the_back_translation_is_blind_to_the_source():
    client = FakeClient({"a": "He was arrested on Monday.", "b": "She accepted the request."},
                        {"a": {"meaning_preserved": 3, "issues": []},
                         "b": {"meaning_preserved": 0, "issues": [{"kind": "distortion", "detail": "denial became acceptance"}],
                               "confirm": {"meaning_preserved": 0, "issues": [
                                   {"kind": "distortion", "source_quote": "denied the claim", "dub_quote": "accepted the request"}]}}})
    out = judge_units(client, UNITS)
    blind = client.calls[0][1]
    assert all(set(p) == {"id", "zulu"} for p in blind)                   # no English source reaches pass 1
    assert [v["meaning_preserved"] for v in out["verdicts"]] == [3, 0]
    assert out["summary"]["units_with_material_problems"] == ["b"] and out["usage"]["prompt_tokens"] == 30


def test_missing_or_malformed_verdicts_are_unjudged_not_zero_or_full():
    client = FakeClient({"a": "x", "b": "y"}, {"a": {"meaning_preserved": 9, "issues": "bad"}})   # a out of range, b absent
    out = judge_units(client, UNITS)
    assert [v["meaning_preserved"] for v in out["verdicts"]] == [None, None] and out["summary"]["unjudged"] == 2
    assert out["summary"]["mean_score"] is None


def test_units_are_judged_in_chunks():
    units = [{"id": str(i), "source": "s", "dub": "d"} for i in range(25)]
    client = FakeClient({str(i): "e" for i in range(25)}, {str(i): {"meaning_preserved": 2, "issues": []} for i in range(25)})
    judge_units(client, units, chunk=10)
    assert [len(c[1]) for c in client.calls] == [10, 10, 10, 10, 5, 5]


def test_the_summary_weights_long_units_more():
    verdicts = [{"id": "long", "source": "a b c d e f g h i j", "meaning_preserved": 1, "issues": []},
                {"id": "short", "source": "a b", "meaning_preserved": 3, "issues": []}]
    s = summarise(verdicts)
    assert s["mean_score"] == 2.0 and s["word_weighted_mean"] == 1.33 and s["pct_units_fully_faithful"] == 50.0


def test_judge_run_reads_the_scorecard_and_writes_judge_json(tmp_path):
    groups = [{"group_id": "u1", "source_text": UNITS[0]["source"], "spoken_text": UNITS[0]["dub"]},
              {"group_id": "empty", "source_text": "", "spoken_text": ""}]
    (tmp_path / "scorecard.json").write_text(json.dumps({"groups": groups}), encoding="utf-8")
    assert [u["id"] for u in units_from_scorecard(tmp_path)] == ["u1"]
    client = FakeClient({"u1": "He was arrested on Monday."}, {"u1": {"meaning_preserved": 3, "issues": []}})
    judge_run(client, tmp_path)
    assert json.loads((tmp_path / "judge.json").read_text(encoding="utf-8"))["summary"]["mean_score"] == 3.0


# ---- meaning as part of the bar -----------------------------------------------------------------------------------

from mathula_tv.dub_agent import pushback_message, slim_submit, unmet_targets
from mathula_tv.dub_improve import batch_stats, gate
from mathula_tv.dub_judge import UnitJudge


def test_a_unit_keeps_its_verdict_until_its_text_changes():
    client = FakeClient({"a": "x", "b": "y"}, {})
    # the fake scores by id, but UnitJudge labels units by text hash: answer whatever ids arrive
    original = client.complete

    def answer(messages, **kw):
        payload = json.loads(messages[1]["content"])
        if "translate isiZulu" in messages[0]["content"]:
            items = [{"id": p["id"], "english": "back"} for p in payload]
        elif messages[0]["content"].startswith("A reviewer flagged"):
            items = [{"id": p["id"], "meaning_preserved": 1, "issues": [
                {"kind": "omission", "source_quote": "denied the claim", "dub_quote": ""}]} for p in payload]
        else:
            items = [{"id": p["id"], "meaning_preserved": 1 if "Wavuma" in p["dub"] else 3, "issues": []} for p in payload]
        client.calls.append((messages[0]["content"][:12], payload))
        return {"choices": [{"message": {"content": json.dumps({"items": items})}}], "usage": {"prompt_tokens": 7, "completion_tokens": 3}}

    client.complete = answer
    judge = UnitJudge(client)
    first = judge.judge(UNITS)
    calls_after_first = len(client.calls)
    again = judge.judge(UNITS)                                  # unchanged text -> no model call at all
    assert len(client.calls) == calls_after_first and [v["meaning_preserved"] for v in again] == [3, 1]
    assert [v["id"] for v in again] == ["a", "b"] and first[1]["meaning_preserved"] == 1
    judge.judge([UNITS[0], {"id": "b", "source": UNITS[1]["source"], "dub": "Walile isicelo."}])
    assert len(client.calls) == calls_after_first + 2           # only the edited unit was judged again


def _result(loss):
    return {"accepted": True, "headline": {"meaning_loss_units": len(loss), "residual_dead_air_pct": 0.5},
            "meaning": {"loss": loss, "unjudged": 0}}


LOSS = [{"item_id": "u7", "issues": ["this morning became tomorrow"], "reads_back_as": "tomorrow too"}]


def test_material_meaning_loss_is_below_the_bar_and_shown_to_the_agent():
    assert unmet_targets(_result([]), max_dead_air_pct=2.0) == []
    unmet = unmet_targets(_result(LOSS), max_dead_air_pct=2.0)
    assert len(unmet) == 1 and "lose material meaning" in unmet[0]
    assert slim_submit(_result(LOSS))["meaning_loss"] == LOSS
    assert "tomorrow too" in pushback_message(_result(LOSS), unmet, tts_left=10)


def test_a_judge_outage_adds_no_bar_and_no_loss():
    result = {"accepted": True, "headline": {"residual_dead_air_pct": 0.5}, "meaning": {"error": "boom"}}
    assert unmet_targets(result, max_dead_air_pct=2.0) == [] and "meaning_loss" not in slim_submit(result)


def test_the_adoption_gate_rejects_a_candidate_that_loses_more_meaning():
    base = {"n": 3, "met_bar": 3, "meaning_loss_mean": 0.3}
    ok, reasons = gate(base, {**base, "meaning_loss_mean": 1.0})
    assert not ok and any("lose meaning" in r for r in reasons)
    assert gate(base, dict(base))[0]
    rows = [{"met_bar": True, "processed_tokens": 1, "headline": {"meaning_loss_units": 2}},
            {"met_bar": True, "processed_tokens": 1, "headline": {}}]
    assert batch_stats(rows)["meaning_loss_mean"] == 1.0


def test_the_workbench_counts_loss_in_the_headline_and_survives_a_judge_outage(tmp_path):
    from mathula_tv.dub_workbench import Workbench

    class Judge:
        usage = {"prompt_tokens": 1}

        def judge(self, units):
            return [{**u, "meaning_preserved": 1 if u["id"] == "bad" else 3, "back_translation": "tomorrow too",
                     "issues": [{"kind": "distortion", "detail": "time changed"}] if u["id"] == "bad" else []} for u in units]

    def card():
        return {"headline": {}, "groups": [{"group_id": "ok", "source_text": "a", "spoken_text": "b"},
                                            {"group_id": "bad", "source_text": "c", "spoken_text": "d"}]}

    wb = Workbench(store=None, tts_factory=lambda: None, judge=Judge())
    c = card()
    meaning = wb._judge_meaning(c, tmp_path)
    assert c["headline"]["meaning_loss_units"] == 1 and meaning["loss"][0]["item_id"] == "bad"
    assert (tmp_path / "judge.json").is_file()

    class Down:
        usage = {}

        def judge(self, units):
            raise RuntimeError("503")

    wb.judge, c = Down(), card()
    assert "error" in wb._judge_meaning(c, tmp_path) and "meaning_loss_units" not in c["headline"]
    assert Workbench(store=None, tts_factory=lambda: None)._judge_meaning(card(), tmp_path) is None


def test_a_flag_with_quotes_that_are_not_in_the_text_is_downgraded_to_a_doubt():
    client = FakeClient({"b": "She denied the claim."},
                        {"b": {"meaning_preserved": 0, "issues": [{"kind": "addition", "detail": "adds everything"}],
                               "confirm": {"meaning_preserved": 0, "issues": [
                                   {"kind": "omission", "source_quote": "the minister resigned", "dub_quote": ""}]}}})
    v = judge_units(client, [UNITS[1]])["verdicts"][0]
    assert v["meaning_preserved"] == 2 and v["unconfirmed"] is True            # the quote is not in the source


def test_a_flag_the_confirmer_no_longer_stands_behind_is_downgraded():
    client = FakeClient({"b": "She denied the claim."},
                        {"b": {"meaning_preserved": 1, "issues": [], "confirm": {"meaning_preserved": 3, "issues": []}}})
    assert judge_units(client, [UNITS[1]])["verdicts"][0]["meaning_preserved"] == 2


def test_each_issue_kind_is_checked_against_the_right_text():
    from mathula_tv.dub_judge import _grounded

    src, back = "He was arrested on Monday, police said.", "He was arrested on Tuesday."
    assert _grounded({"kind": "omission", "source_quote": "police said"}, src, back)
    assert not _grounded({"kind": "omission", "source_quote": "the court heard"}, src, back)
    assert _grounded({"kind": "distortion", "source_quote": "arrested on Monday", "dub_quote": "arrested on Tuesday"}, src, back)
    assert not _grounded({"kind": "distortion", "source_quote": "arrested on Monday"}, src, back)     # needs both quotes
    assert _grounded({"kind": "addition", "dub_quote": "on Tuesday"}, src, back) and not _grounded({"kind": "unclear"}, src, back)


class ForeignClient:
    """Answers the foreign-word prompt from a table keyed by the Zulu text; counts calls."""

    def __init__(self, table):
        self.table, self.calls = table, 0

    def complete(self, messages, *, json_mode=False, max_output_tokens=0, tools=None):
        self.calls += 1
        payload = json.loads(messages[1]["content"])
        items = [{"id": p["id"], "words": self.table.get(p["zulu"], [])} for p in payload]
        return {"choices": [{"message": {"content": json.dumps({"items": items})}}], "usage": {"prompt_tokens": 7, "completion_tokens": 3}}


SCIENCE = {"written": "iScience", "english": "science"}


def test_foreign_words_are_read_from_the_zulu_text_alone():
    client = ForeignClient({"Phela iScience iyisifundo.": [SCIENCE, {"written": "", "english": "x"}, "junk"]})
    usage = {}
    assert foreign_words(client, ["Phela iScience iyisifundo.", "Sawubona."], usage) == [{**SCIENCE, "kind": "word", "zulu": ""}]
    assert usage == {"prompt_tokens": 7, "completion_tokens": 3}


def test_pauses_are_not_sent_to_the_model_and_a_text_keeps_its_answer():
    client = ForeignClient({"Phela iScience iyisifundo. Kunjalo.": [SCIENCE]})
    judge = UnitJudge(client)
    text = "Phela iScience iyisifundo. ~pause600 Kunjalo."
    assert judge.foreign_words([text]) == [{**SCIENCE, "kind": "word", "zulu": ""}]
    assert judge.foreign_words([text, text]) == [{**SCIENCE, "kind": "word", "zulu": ""}]
    assert client.calls == 1                                 # asked once, remembered


def test_the_model_can_mark_a_word_as_a_name():
    client = ForeignClient({"Lapha eNkomazi.": [{"written": "eNkomazi", "english": "Nkomazi", "kind": "name"}, {"written": "x", "english": "y", "kind": "bogus"}]})
    found = foreign_words(client, ["Lapha eNkomazi."], {})
    assert [(w["english"], w["kind"]) for w in found] == [("Nkomazi", "name"), ("y", "word")]


def test_the_reading_of_the_zulu_gives_the_everyday_zulu_word_for_a_kept_english_word():
    text = "Kungani weScience weTechnology Malema medical"
    client = ForeignClient({text: [
        {"written": "weScience", "english": "science", "kind": "word", "zulu": "isayensi"},
        {"written": "Malema", "english": "Malema", "kind": "name", "zulu": "isayensi"},
        {"written": "medical", "english": "medical", "kind": "word"}]})
    found = {w["english"]: w for w in foreign_words(client, [text], {})}
    assert found["science"]["zulu"] == "isayensi" and found["medical"]["zulu"] == ""        # no everyday word: empty
    assert found["Malema"]["zulu"] == ""                                                       # a name never has one, whatever the model says


def test_the_foreign_word_prompt_treats_titles_numbered_labels_and_dates_as_names_not_loanword_candidates():
    from mathula_tv.dub_judge import FOREIGN_PROMPT

    assert "Fighters" in FOREIGN_PROMPT and "Ward 11" in FOREIGN_PROMPT and "November" in FOREIGN_PROMPT and "dates stay English by design" in FOREIGN_PROMPT
