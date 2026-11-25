from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from mathula_tv import dub_finish as module


def test_the_studio_is_waited_for_and_its_audio_file_is_the_track():
    answers = [{"pending": True, "made": 0, "total": 1}, {"ready": True, "seconds": 64.4, "url": "/audio/abc123.wav"}]
    seen = []

    class FakeConsole:
        audio_dir = Path("/audio-dir")

        def studio_run(self, job, run):
            seen.append((job, run))
            return answers.pop(0)

    said = []
    track = module.studio_track(FakeConsole(), "job-1", "prod-1", progress=said.append, poll_seconds=0.0)
    assert track == Path("/audio-dir") / "abc123.wav" and seen == [("job-1", "prod-1")] * 2
    assert "0 of 1 units made" in said[0] and "ready" in said[1]


def test_a_finish_keeps_the_whole_video_by_default_and_the_saved_hook_unless_a_new_one_is_asked_for(monkeypatch, tmp_path):
    from mathula_tv import dub_publish, dub_render, native_dub as nd, orchestrator

    calls: dict = {}
    paths = SimpleNamespace(video=tmp_path / "native_dubbed.mp4", pass2=tmp_path / "pass2.json", pass1_sentences=tmp_path / "s.json", publication_selection=tmp_path / "sel.json")
    paths.publication_selection.write_text(json.dumps({"selection": {"hook_text": "I-ANC ithi izobiza isithangami", "hook_spelling_fixes": [{"hook": "isithangazi", "dub": "isithangami"}]}}), encoding="utf-8")
    monkeypatch.setattr(dub_render, "prepare_render_root", lambda *a, **k: (tmp_path, {"job_id": "job-1"}, tmp_path / "voices.json"))
    monkeypatch.setattr(orchestrator, "Orchestrator", lambda settings: SimpleNamespace(jobs=SimpleNamespace(load=lambda job_id: SimpleNamespace(job_id=job_id))))
    monkeypatch.setattr(nd, "native_dub_paths", lambda root: paths)
    monkeypatch.setattr(nd, "resolve_job_source_video", lambda job, root: tmp_path / "source.mp4")
    monkeypatch.setattr(nd, "read_json", lambda path: {"temporal_mask_mode": False, "translations": [{"segment_id": "s1", "spoken_text": "Sawubona"}], "records": [{"segment_id": "s1"}]})
    monkeypatch.setattr(nd, "render_native_hook_publication", lambda **k: calls.setdefault("publication", k) and {"output": {"path": str(tmp_path / "hook.mp4")}})
    monkeypatch.setattr(dub_publish, "publish_handoff", lambda work, **k: calls.setdefault("handoff", k) and {"final_video": "out/final_dubbed_job-1.mp4", "files": {"tiktok_caption": "out/tiktok_caption_zu_job-1.txt"}})
    monkeypatch.setattr(module, "mux_master", lambda source, track, output: calls.setdefault("mux", (source, track, output)) and output)
    (tmp_path / "native_dubbed.mp4").write_bytes(b"video")

    class FakeConsole:
        audio_dir = tmp_path

        def studio_run(self, job, run):
            return {"ready": True, "seconds": 64.0, "url": "/audio/track.wav"}

    result = module.finish(SimpleNamespace(work_dir=tmp_path), job_id="job-1", run_id="prod-1", console=FakeConsole(), provider=object(), progress=lambda m: None)
    publication = calls["publication"]
    assert publication["apply_opening_cut"] is False and publication["force"] is False          # the whole video, and the saved editorial call is kept
    assert calls["mux"][1] == tmp_path / "track.wav" and calls["mux"][2] == paths.video          # the studio's audio becomes the master the publication step reads
    assert result["hook"] == "I-ANC ithi izobiza isithangami" and result["hook_spelling_fixes"][0]["dub"] == "isithangami"
    assert result["handoff"]["files"]["tiktok_caption"].endswith("tiktok_caption_zu_job-1.txt")
    calls.clear()
    module.finish(SimpleNamespace(work_dir=tmp_path), job_id="job-1", run_id="prod-1", opening_cut=True, new_hook=True, console=FakeConsole(), provider=object(), progress=lambda m: None)
    assert calls["publication"]["apply_opening_cut"] is True and calls["publication"]["force"] is True
