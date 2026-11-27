import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mathula_tv import dub_fix

ITEMS = [{"item_id": "u-01", "start_ms": 0, "end_ms": 8000, "zulu_text": "Sawubona baba."},
         {"item_id": "u-02", "start_ms": 9000, "end_ms": 20000, "zulu_text": "Kwaba ngo-1994 kuphela."},
         {"item_id": "u-03", "start_ms": 21000, "end_ms": 30000, "zulu_text": "Siyabonga."}]


def test_a_unit_is_found_by_number_or_item_id_and_only_its_text_changes():
    assert dub_fix.find_unit(ITEMS, 1) == 1 and dub_fix.find_unit(ITEMS, "2") == 2 and dub_fix.find_unit(ITEMS, "u-01") == 0
    with pytest.raises(ValueError):
        dub_fix.find_unit(ITEMS, 7)
    with pytest.raises(ValueError):
        dub_fix.find_unit(ITEMS, "u-09")
    changed, index, before, after = dub_fix.edit_unit(ITEMS, "u-02", replace="ngo-1994=>ngo-1995")
    assert index == 1 and before == "Kwaba ngo-1994 kuphela." and after == "Kwaba ngo-1995 kuphela."
    assert [i["zulu_text"] for i in changed if i is not changed[1]] == ["Sawubona baba.", "Siyabonga."] and ITEMS[1]["zulu_text"] == before          # the others, and the original, untouched
    assert dub_fix.edit_unit(ITEMS, 0, text="Yebo.")[3] == "Yebo."
    with pytest.raises(ValueError):
        dub_fix.edit_unit(ITEMS, 1, replace="nothing here=>x")
    with pytest.raises(ValueError):
        dub_fix.edit_unit(ITEMS, 1, replace="no arrow")


def test_the_fit_is_the_voices_own_reading_against_the_room_until_the_next_unit_starts():
    room = dub_fix.window(ITEMS, 1)
    assert room == {"source_s": 11.0, "room_s": 12.0}                                              # from its start to where the next unit starts
    assert dub_fix.window(ITEMS, 2) == {"source_s": 9.0, "room_s": 9.0}                            # the last: its own length
    assert dub_fix.fit(10.5, room)["fits"] is True and dub_fix.fit(10.5, room)["spare_s"] == 1.5
    long = dub_fix.fit(13.2, room)
    assert long["fits"] is False and long["over_s"] == 1.2 and long["spare_s"] == 0.0


def test_a_fix_edits_one_unit_keeps_the_old_text_and_measures_only_that_unit(tmp_path, monkeypatch):
    run_dir = tmp_path / "experience" / "runs" / "prod-1" / "abcd1234" / "r1"
    run_dir.mkdir(parents=True)
    (run_dir / "submission.json").write_text(json.dumps({"run_id": "prod-1", "items": ITEMS}), encoding="utf-8")

    class Store:
        def __init__(self, work_dir):
            pass

        def repeat_dir(self, run, clip, repeat):
            return run_dir

    monkeypatch.setattr("mathula_tv.experience_store.ExperienceStore", Store)
    asked = []

    class Console:
        def _draw_unit(self, job, run, unit):
            asked.append(unit)
            return None, {}, ["s"], [], None

        def _run_voice(self, job, run):
            return {}

        def _omni_with(self, voice):
            import numpy as np

            return SimpleNamespace(speak=lambda group, spans, gaps, corpus: (np.zeros(24000 * 13), 24000, {}, {"picked": [0]}, None))

    lines = []
    out = dub_fix.fix(SimpleNamespace(work_dir=str(tmp_path)), job_id="a" * 8 + "0" * 24, run_id="prod-1", unit="u-02", replace="1994=>1995", console=Console(), progress=lines.append)
    saved = json.loads((run_dir / "submission.json").read_text(encoding="utf-8"))["items"]
    assert [i["zulu_text"] for i in saved] == ["Sawubona baba.", "Kwaba ngo-1995 kuphela.", "Siyabonga."]
    assert json.loads((run_dir / "submission.before_fix_1.json").read_text(encoding="utf-8"))["items"][1]["zulu_text"] == "Kwaba ngo-1994 kuphela."
    assert asked == [1] and out["fit"]["fits"] is False and out["fit"]["over_s"] == 1.0 and out["edited"]["backup"] == "submission.before_fix_1.json"
    again = dub_fix.fix(SimpleNamespace(work_dir=str(tmp_path)), job_id="a" * 8 + "0" * 24, run_id="prod-1", unit=1, check=True, console=Console(), progress=lines.append)
    assert "edited" not in again and not (run_dir / "submission.before_fix_2.json").exists()          # measuring edits nothing
