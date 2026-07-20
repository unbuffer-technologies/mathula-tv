from mathula_tv.transcript_merge import reconcile
from mathula_tv.diarization import select_authoritative


def test_overlap_unknown_and_pause_boundaries():
    words = [{"text":"hello","start":0,"end":.5,"confidence":.9},{"text":"both","start":1,"end":1.2,"confidence":.8},{"text":"lost","start":3,"end":3.2,"confidence":.7}]
    turns = [{"speaker":"A","start":0,"end":1.2},{"speaker":"B","start":.9,"end":1.4}]
    result = reconcile(words, turns, tolerance=.1)
    assert result["words"][1]["overlap"] is True
    assert result["words"][2]["speaker"] == "UNKNOWN"
    assert len(result["segments"]) == 3


def test_pyannote_fallback_is_truthful():
    turns, source, warnings = select_authoritative(None, {"turns":[{"speaker":"AZURE_1","start":0,"end":1}]}, "GPU disconnected")
    assert source == "azure" and "GPU disconnected" in warnings[0]

