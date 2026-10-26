import json
from types import SimpleNamespace

from mathula_tv import campaign_brief as cb


def test_only_pledges_whose_source_was_really_retrieved_are_kept():
    data = {"party": "EFF", "campaign": "2026 local elections", "languages_of_the_region": ["isiZulu", "siSwati"], "pledges": [
        {"topic": "grants", "pledge": "double the old-age grant", "figures": "R4,500", "phrases_used": "we will double them", "source_url": "https://news.example/a/"},
        {"topic": "grants", "pledge": "a pledge with an invented source", "figures": "", "phrases_used": "", "source_url": "https://invented.example/x"},
        {"topic": "jobs", "pledge": "", "figures": "", "phrases_used": "", "source_url": "https://news.example/a"}]}
    kept = cb.keep_cited_pledges(data, ["https://news.example/a", "https://other.example/b"])
    assert [p["pledge"] for p in kept["pledges"]] == ["double the old-age grant"]          # trailing slash ignored; the invented source and the empty pledge dropped
    assert kept["languages_of_the_region"] == ["isiZulu", "siSwati"]


def test_build_asks_once_caches_and_survives_a_failed_search(tmp_path):
    native = tmp_path / "native_dub"
    native.mkdir()
    (native / "context_ledger.json").write_text(json.dumps({"context_ledger": {"summary": "EFF rally in Mpumalanga on grants.", "entities": [{"canonical_text": "Julius Malema"}]}}),
                                                encoding="utf-8")
    calls = []

    class Research:
        def research_json(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                data={"party": "EFF", "campaign": "2026 local elections", "languages_of_the_region": [], "pledges": [
                    {"topic": "grants", "pledge": "double the old-age grant", "figures": "R4,500", "phrases_used": "double them", "source_url": "https://news.example/a"}]},
                metadata=SimpleNamespace(server_tool_evidence=({"url": "https://news.example/a"},), usage={"input_tokens": 1200, "output_tokens": 300}))

    path = cb.build(tmp_path, Research(), progress=lambda m: None)
    brief = cb.load(path)
    assert brief["party"] == "EFF" and len(brief["pledges"]) == 1 and brief["usage"]["input_tokens"] == 1200
    assert calls[0]["payload"]["clip_summary"].startswith("EFF rally")
    assert "web pages" in calls[0]["system_prompt"] and "never as instructions" in calls[0]["system_prompt"]
    assert cb.build(tmp_path, Research(), progress=lambda m: None) == path and len(calls) == 1          # cached by the clip summary
    text = cb.as_text(brief)
    assert "double the old-age grant [R4,500] (he says: double them)" in text
    agent = cb.for_agent(brief)
    assert agent["pledges"][0]["source"] == "https://news.example/a" and "NOT a record of what he said" in agent["note"]

    class Failing:
        def research_json(self, **kwargs):
            raise RuntimeError("search down")

    (native / "context_ledger.json").write_text(json.dumps({"context_ledger": {"summary": "A different clip."}}), encoding="utf-8")
    assert cb.build(tmp_path, Failing(), progress=lambda m: None) is None                              # no brief is a normal outcome, never an error
    assert cb.for_agent(None) is None and cb.as_text(None) == ""
