from pathlib import Path

from mathula_tv.dub_workbench import MAX_DECISIONS, Workbench, settled_units, speaker_rates


class Bench:
    def __init__(self):
        self.decisions = {}
    _keep_decisions = Workbench._keep_decisions
    _decisions_view = Workbench._decisions_view
    _stem_analyses = staticmethod(Workbench._stem_analyses)


ISU = {"isu": ["i[NPrePre][5]li[BPre][5]su.5-6[NStem]"], "amasu": ["a[NPrePre][6]ma[BPre][6]su.5-6[NStem]"], "ngamasu": ["nga[AdvPre]a[NPrePre][6]ma[BPre][6]su.5-6[NStem]"],
       "ilungelo": ["i[NPrePre][5]li[BPre][5]lungelo.5-6[NStem]"], "ngisuka": ["ngi[SCPT][1ps]suk[VRoot]a[VerbTerm]"]}


def test_decisions_are_merged_by_the_english_saved_and_shown_with_the_units_that_use_them(tmp_path, monkeypatch):
    from mathula_tv import zulu_stems

    monkeypatch.setattr(zulu_stems, "analyse", lambda words: {w: ISU[w] for w in words if w in ISU})
    bench = Bench()
    bench._keep_decisions([{"english": "strategy", "zulu": "isu", "note": "defence strategy"}, {"english": "right", "zulu": "ilungelo"}, {"english": "", "zulu": "x"}, "junk"], tmp_path)
    assert sorted(bench.decisions) == ["right", "strategy"]
    saved = Path(tmp_path / "decisions.json").read_text(encoding="utf-8")
    assert "ilungelo" in saved and "isu" in saved
    view = bench._decisions_view([{"item_id": "u1", "zulu_text": "Ilungelo lomphakathi"}, {"item_id": "u2", "zulu_text": "ngezindlela ngamasu akhe"}, {"item_id": "u3", "zulu_text": "ngisuka lapho"}])
    by = {v["english"]: v["in_units"] for v in view}
    assert by["strategy"] == ["u2"]          # amasu, ngamasu: one word with isu, by stem; ngisuka (to leave) is another word that merely holds the letters
    assert by["right"] == ["u1"]
    bench._keep_decisions([{"english": "strategy", "zulu": "isu"}], tmp_path)
    assert bench._decisions_view([{"item_id": "u1", "zulu_text": "ngenxa yalokhu"}])[0]["in_units"] == "no unit uses a form of this word"


def test_without_the_analyser_the_comparison_is_of_plain_forms_and_says_so(tmp_path, monkeypatch):
    from mathula_tv import zulu_stems

    monkeypatch.setattr(zulu_stems, "analyse", lambda words: {})
    bench = Bench()
    bench._keep_decisions([{"english": "strategy", "zulu": "isu"}], tmp_path)
    view = bench._decisions_view([{"item_id": "u1", "zulu_text": "nangamasu akhe"}, {"item_id": "u2", "zulu_text": "lelo isu"}])
    assert view[0]["in_units"] == ["u2"]                                                      # amasu is missed without stems, and the plain form is found
    assert "analyser is not available" in bench._decisions_view([{"item_id": "u1", "zulu_text": "nangamasu akhe"}])[0]["in_units"]          # and a miss says why


def test_the_number_of_decisions_is_bounded(tmp_path):
    bench = Bench()
    bench._keep_decisions([{"english": f"term {n}", "zulu": f"izwi {n}"} for n in range(MAX_DECISIONS + 20)], tmp_path)
    assert len(bench.decisions) == MAX_DECISIONS


def test_settled_units_and_speaker_rates_come_from_the_score_card():
    card = {"groups": [{"group_id": "u1", "flags": [], "gates": {"a": {"failed": False}}, "speaker_id": "A", "measured_ms": 2000, "spoken_text": "x" * 24},
                       {"group_id": "u2", "flags": ["dead_air"], "gates": {}, "speaker_id": "A", "measured_ms": 1000, "spoken_text": "y" * 14},
                       {"group_id": "u3", "flags": [], "gates": {"a": {"failed": True}}, "speaker_id": "B", "measured_ms": 1000, "spoken_text": "z" * 10}]}
    assert settled_units(card) == ["u1"]
    assert speaker_rates(card) == {"A": 14.0, "B": 10.0}          # A: units at 12 and 14 characters per second, the upper median


def test_units_overview_shows_every_units_length_and_fit_without_its_text():
    from mathula_tv.dub_agent import COMPACTION_NOTE, slim_submit
    from mathula_tv.dub_workbench import units_overview

    card = {"groups": [{"group_id": "u1", "window_ms": 10000, "measured_ms": 9900, "spoken_text": "x" * 120},
                       {"group_id": "u2", "window_ms": 60000, "measured_ms": 54900, "spoken_text": "y" * 858},
                       {"group_id": "u3", "window_ms": 5000, "measured_ms": 5400, "spoken_text": "z" * 60}]}
    assert units_overview(card) == ["u1 120ch 0", "u2 858ch -5.1s", "u3 60ch +0.4s"]
    assert slim_submit({"accepted": True, "headline": {}, "units_overview": ["u1 120ch 0"]})["units_overview"] == ["u1 120ch 0"]
    assert "units_overview" in COMPACTION_NOTE and "only to read the text of a unit that is not listed" in COMPACTION_NOTE
