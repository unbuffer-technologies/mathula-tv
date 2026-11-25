from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from mathula_tv import warm_start as ws
from mathula_tv.dub_agent import _unit_english
from mathula_tv.experience_store import ExperienceStore
from mathula_tv.submission_rank import BestSubmission

CLIP = "ada0e557"


def headline(gates=0, cross=0, collisions=0, rushed=0, dead=5.0) -> dict:
    return {"gate_failures": gates, "cross_speaker_collisions": cross, "collisions": collisions, "groups_rushed": rushed, "residual_dead_air_pct": dead}


def make_run(root: Path, run_id: str, *, voice: str | None, model: str = "gpt", final: dict | None = None, best: dict | None = None, texts: dict | None = None) -> Path:
    directory = root / "experience" / "runs" / run_id / CLIP / "r1"
    directory.mkdir(parents=True)
    (directory / "scorecard.json").write_text(json.dumps({"headline": final or headline(), "groups": [{"group_id": k, "source_text": v} for k, v in (texts or {"u1": "Hello there"}).items()]}), encoding="utf-8")
    (directory / "submission.json").write_text(json.dumps({"items": [{"item_id": "u1", "zulu_text": f"final of {run_id}"}]}), encoding="utf-8")
    summary = {"model": model, **({"voice_provider": voice} if voice else {})}
    (directory / "agent_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    if best:
        (directory / "best_submission.json").write_text(json.dumps({"headline": best, "items": [{"item_id": "u1", "zulu_text": f"best of {run_id}"}], "unit_texts": {"u1": "Hello best"}}), encoding="utf-8")
    return directory


def test_same_voice_comes_before_a_better_scoring_other_voice(tmp_path):
    make_run(tmp_path, "azure-1", voice="azure", final=headline(dead=0.5))
    make_run(tmp_path, "omni-1", voice="omnivoice", final=headline(collisions=2, dead=7.0))
    chosen = ws.choose(ExperienceStore(tmp_path), CLIP, voice_provider="omnivoice")
    assert chosen.run_id == "omni-1" and chosen.same_voice


def test_within_a_voice_the_scorers_own_order_decides(tmp_path):
    make_run(tmp_path, "a", voice="omnivoice", final=headline(collisions=1, dead=1.0))
    make_run(tmp_path, "b", voice="omnivoice", final=headline(collisions=0, dead=16.0))      # fewer collisions beats less dead air
    make_run(tmp_path, "c", voice="omnivoice", final=headline(gates=1, collisions=0, dead=0.0))
    assert [c.run_id for c in ws.candidates(ExperienceStore(tmp_path), CLIP, "omnivoice")] == ["b", "a", "c"]


def test_a_runs_best_submission_is_used_when_it_ended_below_it(tmp_path):
    make_run(tmp_path, "r", voice="omnivoice", final=headline(collisions=3), best=headline(collisions=0))
    chosen = ws.choose(ExperienceStore(tmp_path), CLIP, voice_provider="omnivoice")
    assert chosen.from_best and chosen.key[2] == 0
    assert chosen.load_items()[0]["zulu_text"] == "best of r"
    assert chosen.unit_texts() == {"u1": "Hello best"}


def test_a_final_submission_reads_its_unit_english_from_the_scorecard(tmp_path):
    make_run(tmp_path, "r", voice="omnivoice", texts={"u1": "One", "u2": "Two"})
    chosen = ws.choose(ExperienceStore(tmp_path), CLIP, voice_provider="omnivoice")
    assert not chosen.from_best and chosen.unit_texts() == {"u1": "One", "u2": "Two"}


def test_runs_without_a_tag_are_read_by_their_agent_model(tmp_path):
    assert ws.infer_voice_provider({"model": "gemini-3.8-flash"}) == "omnivoice"
    assert ws.infer_voice_provider({"model": "gpt-5.6-sol-1"}) == "azure"
    assert ws.infer_voice_provider({"model": "gemini-3.8-flash", "voice_provider": "azure"}) == "azure"
    make_run(tmp_path, "old", voice=None, model="gemini-3.8-flash")
    assert ws.choose(ExperienceStore(tmp_path), CLIP, voice_provider="omnivoice").voice_provider == "omnivoice"


def test_backfill_writes_the_tag_once(tmp_path):
    directory = make_run(tmp_path, "old", voice=None, model="gpt-5.6-sol-1")
    store = ExperienceStore(tmp_path)
    assert ws.backfill(store) == [(f"old/{CLIP}/r1", "azure")]
    written = json.loads((directory / "agent_summary.json").read_text(encoding="utf-8"))
    assert written["voice_provider"] == "azure" and written["voice_provider_inferred"] is True
    assert ws.backfill(store) == []


def test_the_run_being_made_is_never_its_own_start_and_a_new_job_has_none(tmp_path):
    store = ExperienceStore(tmp_path)
    assert ws.choose(store, CLIP, voice_provider="omnivoice") is None
    make_run(tmp_path, "now", voice="omnivoice")
    assert ws.choose(store, CLIP, voice_provider="omnivoice", exclude=("now",)) is None
    assert ws.from_run(store, "now", CLIP, voice_provider="omnivoice").run_id == "now"


def test_a_run_without_a_scorecard_is_not_a_candidate(tmp_path):
    directory = make_run(tmp_path, "half", voice="omnivoice")
    (directory / "scorecard.json").unlink()
    assert ws.choose(ExperienceStore(tmp_path), CLIP, voice_provider="omnivoice") is None


def test_best_submission_keeps_the_english_it_was_written_against():
    workbench = SimpleNamespace(segments=[{"segment_id": "s1", "source_text": "Good", "restored_text": "Good evening"}, {"segment_id": "s2", "source_text": "everyone"}])
    items = [{"item_id": "u1", "source_segment_ids": ["s1", "s2"]}]
    texts = _unit_english(workbench, items)
    assert texts == {"u1": "Good evening everyone"}
    best = BestSubmission()
    best.observe(3, {"accepted": True, "headline": headline()}, items, texts)
    assert best.best["unit_texts"] == texts
