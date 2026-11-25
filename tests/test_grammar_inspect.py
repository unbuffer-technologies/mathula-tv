import json

from mathula_tv import grammar_inspect


class _Client:
    def __init__(self):
        self.payloads = []

    def complete(self, messages, **_):
        payload = json.loads(messages[1]["content"])
        self.payloads.append(payload)
        items = [{"id": x["id"], "issues": [{"phrase": "u-Ayanda Nyathi nentatheli", "kind": "fused", "problem": "runs together", "fix": "u-Ayanda Nyathi, nentatheli"}] if x["id"] == "unit-01" else []} for x in payload]
        return {"choices": [{"message": {"content": json.dumps({"items": items})}}], "usage": {"prompt_tokens": 3, "completion_tokens": 2}}


def test_the_final_text_of_every_unit_is_inspected_with_its_english():
    items = [{"item_id": "unit-01", "source_segment_ids": ["s1", "s2"], "zulu_text": "u-Ayanda Nyathi nentatheli"}, {"item_id": "unit-02", "source_segment_ids": ["s3"], "zulu_text": "Sawubona."}]
    english = {"s1": "Ayanda Nyathi", "s2": "and the journalist", "s3": "Hello."}
    client = _Client()
    report = grammar_inspect.inspect(items, english, client)
    sent = {x["id"]: x for payload in client.payloads for x in payload}
    assert sent["unit-01"]["english"] == "Ayanda Nyathi and the journalist" and "rushed" not in sent["unit-01"]
    assert report["counts"] == {"units": 2, "units_with_issues": 1, "issues": 1, "by_kind": {"fused": 1}} and report["units"][1]["issues"] == []


def test_the_english_of_a_job_is_read_from_its_corrected_transcript_first(tmp_path):
    (tmp_path / "analysis").mkdir()
    (tmp_path / "analysis" / "transcript_en.json").write_text(json.dumps({"segments": [{"segment_id": "s1", "source_text": "wrong"}]}), encoding="utf-8")
    (tmp_path / "analysis" / "corrected_transcript.json").write_text(json.dumps({"segments": [{"segment_id": "s1", "source_text": "wrong", "restored_text": "right"}]}), encoding="utf-8")
    assert grammar_inspect.source_english(tmp_path) == {"s1": "right"}
    assert grammar_inspect.source_english(tmp_path / "nowhere") == {}
