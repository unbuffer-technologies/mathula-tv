"""Pass 1 learns from the listener: earlier corrections are shown to the model, and a job's own answers are applied to its restored text, on a fresh run and on a reused one."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from mathula_tv import language_questions as lq
from mathula_tv.atomic_io import atomic_write_json
from mathula_tv.native_dub import run_native_pass1


class Provider:
    provider = "azure-foundry-grok"

    def __init__(self):
        self.config = SimpleNamespace(deployment="grok-test", max_output_tokens=8192)
        self.calls = []

    def complete_json(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(data={"corrections": []}, model="grok-test", input_tokens=10, output_tokens=5, attempts=1)


def _job(tmp_path: Path) -> Path:
    job = tmp_path / "work" / "jobs" / ("cd" * 16)
    atomic_write_json(job / "analysis" / "transcript_en.json", {"language": "en-ZA", "segments": [
        {"segment_id": "seg-1", "speaker": "SPEAKER_00", "start": 0.0, "end": 3.0, "source_text": "Good evening to you all."},
        {"segment_id": "seg-2", "speaker": "SPEAKER_00", "start": 4.0, "end": 8.0, "source_text": "Nally a bunch of anna see yay in new sir"}]})
    return job


def test_the_listeners_earlier_corrections_are_shown_to_the_model_but_do_not_redo_a_finished_pass_1(tmp_path):
    job, work = _job(tmp_path), tmp_path / "work"
    first = Provider()
    run_native_pass1(job_root=job, provider=first)
    assert "listener_examples" not in first.calls[0]["payload"]
    assert "LISTENER EXAMPLES" in first.calls[0]["system_prompt"]
    lq._write(lq.lessons_path(work), {"words": {}, "stretches": [{"job": "other", "segment_id": "seg-9", "track": "zu-orig", "transcript": "Nally a bunch", "model": "Nally a bunch",
                                                                   "caption": "nale yabantwana", "choice": "typed", "final": "Nale yabantwana"}]})
    again = Provider()
    run_native_pass1(job_root=job, provider=again)
    assert again.calls == []                                                                  # a lesson from another video does not make this video's finished pass 1 run again
    run_native_pass1(job_root=job, provider=again, force=True)
    example = again.calls[0]["payload"]["listener_examples"][0]
    assert example == {"transcript": "Nally a bunch", "caption": "nale yabantwana", "caption_track": "zu-orig", "model_wrote": "Nally a bunch", "listener_decided": "Nale yabantwana",
                       "winner": "the listener's own words"}


def test_a_job_s_own_answers_are_applied_to_its_restored_text_whether_pass_1_runs_or_is_reused(tmp_path):
    job, work = _job(tmp_path), tmp_path / "work"
    provider = Provider()
    run_native_pass1(job_root=job, provider=provider)
    lq.answer_stretch(job, work, "seg-2", "typed", "Nale yabantwana siyayinyusa")
    reused = run_native_pass1(job_root=job, provider=provider)
    assert len(provider.calls) == 1                                                           # no second model call: the answer is applied to the checkpoint
    second = {s["segment_id"]: s for s in reused["segments"]}["seg-2"]
    assert second["restored_text"] == "Nale yabantwana siyayinyusa" and second["model_text"].startswith("Nally") and second["reason_code"] == "listener_confirmed"
    on_disk = json.loads((job / "native_dub" / "pass1_restored_transcript.json").read_text())["segments"][1]
    assert on_disk["restored_text"] == "Nale yabantwana siyayinyusa"
    redone = run_native_pass1(job_root=job, provider=provider, force=True)                      # a fresh run keeps the answer too: the model does not outrank the listener
    assert len(provider.calls) == 2 and {s["segment_id"]: s for s in redone["segments"]}["seg-2"]["restored_text"] == "Nale yabantwana siyayinyusa"
