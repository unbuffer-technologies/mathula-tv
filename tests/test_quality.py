import wave
from mathula_tv.logging_utils import redact
from mathula_tv.quality import inspect_wav


def test_audio_quality_and_secret_redaction(tmp_path):
    path=tmp_path/"x.wav"
    with wave.open(str(path),"wb") as w: w.setparams((1,2,16000,0,"NONE","none")); w.writeframes((1000).to_bytes(2,"little",signed=True)*16000)
    assert inspect_wav(path)["valid"]
    cleaned=redact({"api_key":"abc","url":"https://x.test/a?X-Goog-Signature=secret&ok=1"})
    assert cleaned["api_key"]=="[REDACTED]" and "secret" not in cleaned["url"]

