from pathlib import Path
from types import SimpleNamespace

import mathula_tv.cli as cli


def test_translate_command_dispatches_shared_orchestrator(monkeypatch,tmp_path,capsys):
    job=SimpleNamespace(job_id="job",state="analysis_ready")
    class Jobs:
        def load(self,job_id): return job
    class App:
        def __init__(self,settings): self.jobs=Jobs()
        def translate_and_package(self,loaded,backend,force=False):
            assert loaded is job and backend=="backend"
            return {"job_id":"job","state":"synthesis_queued","translated_segments":1,"target_language":"zu-ZA","title":"Isihloko","translation_request_duration":1,"seo_request_duration":1}
    settings=SimpleNamespace(work_dir=tmp_path)
    monkeypatch.setattr(cli,"load_settings",lambda:settings); monkeypatch.setattr(cli,"Orchestrator",App)
    monkeypatch.setattr(cli,"create_translation_backend",lambda:"backend")
    assert cli.main(["translate","job"])==0
    assert '"state": "synthesis_queued"' in capsys.readouterr().out
