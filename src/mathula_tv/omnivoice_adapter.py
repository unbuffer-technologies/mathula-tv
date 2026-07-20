"""DEPRECATED migration reference: not registered, not selectable, not production."""

from __future__ import annotations

import math
import time
import wave
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class SynthesisResult:
    provider: str
    model_id: str
    output_path: str
    sample_rate: int
    duration: float
    generation_time: float
    real_time_factor: float
    peak: float
    warnings: list[str]
    language_id: str | None = None
    generation_parameters: dict | None = None
    def to_dict(self): return asdict(self)


class OmniVoiceAdapter(ABC):
    @abstractmethod
    def load(self) -> dict: ...
    @abstractmethod
    def synthesize(self, text: str, reference_audio: Path, reference_text: str, output: Path, *, style: dict | None = None) -> SynthesisResult: ...


class RealOmniVoiceAdapter(OmniVoiceAdapter):
    LOCALE_TO_LANGUAGE_ID={"zu-ZA":"zu"}
    SPACE_SETTINGS={"num_step":32,"guidance_scale":2.0,"denoise":True,"preprocess_prompt":True,"postprocess_output":True,"speed":None,"duration":None,"normalize_text":False}
    CURRENT_SETTINGS={"num_step":32,"guidance_scale":2.0,"denoise":True,"preprocess_prompt":True,"postprocess_output":True,"speed":None,"duration":None,"normalize_text":False}
    def __init__(self, model_id="k2-fsa/OmniVoice", device_map="cuda:0", dtype: Any = None, target_locale="zu-ZA", *, model_class: Any = None, generation_config_class: Any = None, torch_module: Any = None, soundfile_module: Any = None):
        self.model_id,self.device_map,self.dtype=model_id,device_map,dtype
        self.model_class,self.generation_config_class,self.torch,self.sf=model_class,generation_config_class,torch_module,soundfile_module
        self.target_locale=target_locale; self.language_id=self.resolve_language_id(target_locale)
        self.model=None; self.load_duration=0.0

    def load(self) -> dict:
        if self.model is not None: return {"provider":"omnivoice","model_id":self.model_id,"device":self.device_map,"load_duration":self.load_duration,"loaded_once":True}
        try:
            if self.torch is None: import torch; self.torch=torch
            if self.model_class is None:
                from omnivoice import OmniVoice,OmniVoiceGenerationConfig
                self.model_class=OmniVoice; self.generation_config_class=OmniVoiceGenerationConfig
            if self.sf is None: import soundfile as sf; self.sf=sf
        except ImportError as exc: raise RuntimeError("Install the official omnivoice package and soundfile in a fresh synthesis runtime") from exc
        if self.device_map.startswith("cuda") and not self.torch.cuda.is_available(): raise RuntimeError("CUDA is required for the OmniVoice synthesis worker")
        dtype=self.dtype if self.dtype is not None else self.torch.float16
        started=time.perf_counter()
        try: self.model=self.model_class.from_pretrained(self.model_id,device_map=self.device_map,dtype=dtype,load_asr=True)
        except Exception as exc:
            if "out of memory" in str(exc).lower(): raise RuntimeError("CUDA out of memory while loading OmniVoice") from exc
            raise RuntimeError(f"Failed to load OmniVoice model: {type(exc).__name__}") from exc
        self.load_duration=time.perf_counter()-started
        return {"provider":"omnivoice","model_id":self.model_id,"device":self.device_map,"load_duration":self.load_duration,"loaded_once":False}

    @classmethod
    def resolve_language_id(cls,locale:str)->str:
        try: return cls.LOCALE_TO_LANGUAGE_ID[locale]
        except KeyError as exc: raise ValueError(f"Unsupported OmniVoice target language: {locale}") from exc

    @staticmethod
    def _mono_array(value: Any):
        import numpy as np
        if hasattr(value,"detach"): value=value.detach().float().cpu().numpy()
        array=np.asarray(value,dtype=np.float32)
        if array.ndim==0: raise ValueError("OmniVoice returned scalar audio")
        if array.ndim>1:
            axis=0 if array.shape[0] <= 8 else 1
            array=array.mean(axis=axis)
        return array.reshape(-1)

    def synthesize(self, text: str, reference_audio: Path, reference_text: str, output: Path, *, style: dict | None = None, generation_profile: str="space_matched") -> SynthesisResult:
        if not text.strip(): raise ValueError("Target text is empty")
        if not reference_audio.is_file() or not reference_text.strip(): raise ValueError("Reference audio and exact reference text are required")
        self.load(); started=time.perf_counter()
        settings=dict(self.SPACE_SETTINGS if generation_profile=="space_matched" else self.CURRENT_SETTINGS)
        try:
            if generation_profile=="space_matched":
                config=self.generation_config_class(**{k:settings[k] for k in ("num_step","guidance_scale","denoise","preprocess_prompt","postprocess_output")})
                prompt=self.model.create_voice_clone_prompt(ref_audio=str(reference_audio),ref_text=reference_text)
                kwargs={"text":text,"language":self.language_id,"generation_config":config,"voice_clone_prompt":prompt,"normalize_text":settings["normalize_text"]}
            elif generation_profile=="current": kwargs={"text":text,"ref_audio":str(reference_audio),"ref_text":reference_text}
            else: raise ValueError(f"Unknown OmniVoice generation profile: {generation_profile}")
            if settings["speed"] is not None: kwargs["speed"]=settings["speed"]
            if settings["duration"] is not None: kwargs["duration"]=settings["duration"]
            generated=self.model.generate(**kwargs)
        except Exception as exc:
            if "out of memory" in str(exc).lower(): raise RuntimeError("CUDA out of memory during OmniVoice generation") from exc
            raise RuntimeError(f"OmniVoice generation failed: {type(exc).__name__}") from exc
        elapsed=time.perf_counter()-started
        if not isinstance(generated,(list,tuple)) or not generated: raise ValueError("OmniVoice returned no audio arrays")
        samples=self._mono_array(generated[0])
        import numpy as np
        if not len(samples) or not np.isfinite(samples).all(): raise ValueError("OmniVoice returned empty or non-finite audio")
        peak=float(np.max(np.abs(samples))); rms=float(np.sqrt(np.mean(samples*samples)))
        if rms < 1e-4: raise ValueError("OmniVoice output is effectively silent")
        if peak > 1.0: raise ValueError("OmniVoice output is clipping")
        if peak > .98: samples=samples*(.98/peak); peak=.98
        output.parent.mkdir(parents=True,exist_ok=True); self.sf.write(str(output),samples,24000,subtype="PCM_16")
        duration=len(samples)/24000
        return SynthesisResult("omnivoice",self.model_id,str(output),24000,duration,elapsed,elapsed/duration,peak,[],self.language_id if generation_profile=="space_matched" else None,settings)


class MockOmniVoiceAdapter(OmniVoiceAdapter):
    """Test-only tone generator. Output is never represented as OmniVoice output."""
    def load(self) -> dict: return {"provider":"mock","test_only":True}
    def synthesize(self,text,reference_audio,reference_text,output,*,style=None):
        rate,duration=24000,max(.2,min(2.,len(text)/40)); output.parent.mkdir(parents=True,exist_ok=True)
        with wave.open(str(output),"wb") as wav:
            wav.setparams((1,2,rate,0,"NONE","not compressed")); wav.writeframes(b"".join(int(1000*math.sin(2*math.pi*220*i/rate)).to_bytes(2,"little",signed=True) for i in range(int(rate*duration))))
        return SynthesisResult("mock","mock",str(output),rate,duration,0,0,1000/32768,["test-only mock output"])
