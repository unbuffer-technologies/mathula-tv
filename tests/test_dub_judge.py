import json

from mathula_tv.dub_judge import judge_run, judge_units, summarise, units_from_scorecard


class FakeClient:
    """Answers the back-translate prompt and the compare prompt from canned tables; records every call."""

    def __init__(self, back, scores):
        self.back, self.scores, self.calls = back, scores, []

    def complete(self, messages, *, json_mode=False, max_output_tokens=0, tools=None):
        system, payload = messages[0]["content"], json.loads(messages[1]["content"])
        self.calls.append((system[:12], payload))
        if "translate isiZulu" in system:
            items = [{"id": p["id"], "english": self.back[p["id"]]} for p in payload]
        else:
            items = [{"id": p["id"], **self.scores[p["id"]]} for p in payload if p["id"] in self.scores]
        return {"choices": [{"message": {"content": json.dumps({"items": items})}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


UNITS = [{"id": "a", "source": "He was arrested on Monday.", "dub": "Waboshwa ngoMsombuluko."},
         {"id": "b", "source": "She denied the claim.", "dub": "Wavuma isicelo."}]


def test_the_back_translation_is_blind_to_the_source():
    client = FakeClient({"a": "He was arrested on Monday.", "b": "She accepted the request."},
                        {"a": {"meaning_preserved": 3, "issues": []},
                         "b": {"meaning_preserved": 0, "issues": [{"kind": "distortion", "detail": "denial became acceptance"}]}})
    out = judge_units(client, UNITS)
    blind = client.calls[0][1]
    assert all(set(p) == {"id", "zulu"} for p in blind)                   # no English source reaches pass 1
    assert [v["meaning_preserved"] for v in out["verdicts"]] == [3, 0]
    assert out["summary"]["units_with_material_problems"] == ["b"] and out["usage"]["prompt_tokens"] == 20


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
