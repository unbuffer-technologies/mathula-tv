from pathlib import Path

from mathula_tv.omnivoice_adapter import SynthesisResult
from mathula_tv.omnivoice_worker import build_calibration_manifest


def test_deterministic_calibration_manifest(tmp_path):
    outputs={}
    for name in ("current","space_matched"):
        path=tmp_path/f"{name}.wav"; path.write_bytes(name.encode())
        outputs[name]=SynthesisResult("omnivoice","model",str(path),24000,1.0,2.0,2.0,.5,[],"zu",{"num_step":32,"speed":None,"duration":None})
    kwargs={"model_id":"model","package_versions":{"omnivoice":"0.2.1"},"language_id":"zu","reference":{"duration":8,"transcript":"exact reference words"},"sentence":"IsiZulu sentence","results":outputs}
    assert build_calibration_manifest(**kwargs)==build_calibration_manifest(**kwargs)
    assert build_calibration_manifest(**kwargs)["profiles"]["space_matched"]["inference_steps"]==32
